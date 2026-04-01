#!/usr/bin/env python3
"""
Train an interpretable ML predictor on the 51K labeled dataset.

Uses decision tree (interpretable, gives optimal thresholds) and
logistic regression (gives feature importance weights).

The decision tree IS the physics: it discovers the optimal criteria
for predicting periodicity from tensor properties.
"""

from __future__ import annotations

import logging
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


def main():
    from sklearn.tree import DecisionTreeClassifier, export_text
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.model_selection import cross_val_score, StratifiedKFold

    log.info("=" * 70)
    log.info("ML Predictor: Learning Optimal Periodicity Criteria")
    log.info("=" * 70)

    d = np.load("results/prediction_large.npz", allow_pickle=True)

    # Build feature matrix
    ranks = d["ranks"]
    inner_ranks = d["inner_ranks"]
    gammas = d["gammas"]
    freq_ratios = d["freq_ratios"]
    n_clusters = d["n_clusters"]
    energies = d["energies"]
    r2_r1 = d["r2_r1"]
    is_separable = d["is_separable"].astype(float)

    # Encode scenario as numeric
    scenarios = d["scenarios"]
    scenario_map = {"equal": 0, "binary_light": 1, "hierarchical": 2}
    scenario_num = np.array([scenario_map.get(str(s), 0) for s in scenarios])

    periodic = d["periodic"]
    collision = d["collision"]
    valid = ~collision

    # Filter to valid
    feature_names = [
        "rank",
        "inner_rank",
        "gamma",
        "log_freq_ratio",
        "n_clusters",
        "energy",
        "log_r2_r1",
        "is_separable",
        "scenario",
    ]

    X = np.column_stack(
        [
            ranks[valid],
            inner_ranks[valid],
            gammas[valid],
            np.log10(freq_ratios[valid] + 1),
            n_clusters[valid],
            energies[valid],
            np.log10(r2_r1[valid] + 1),
            is_separable[valid],
            scenario_num[valid],
        ]
    )
    y = periodic[valid].astype(int)

    log.info("Dataset: %d samples, %d periodic (%.1f%%)", len(y), y.sum(), 100 * y.mean())
    log.info("Features: %s", feature_names)

    # Handle infinities and NaN
    X = np.nan_to_num(X, nan=0, posinf=100, neginf=-100)

    # Cross-validated evaluation
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # 1. Decision Tree (interpretable)
    log.info("\n--- Decision Tree (max_depth=5) ---")
    dt = DecisionTreeClassifier(
        max_depth=5, min_samples_leaf=50, class_weight="balanced", random_state=42
    )
    dt_scores = cross_val_score(dt, X, y, cv=cv, scoring="f1")
    log.info("  CV F1: %.3f ± %.3f", dt_scores.mean(), dt_scores.std())

    # Fit on full data and print tree
    dt.fit(X, y)
    tree_text = export_text(dt, feature_names=feature_names, max_depth=5)
    log.info("\n  Decision tree rules:\n%s", tree_text)

    # Feature importance
    importances = dt.feature_importances_
    for name, imp in sorted(zip(feature_names, importances), key=lambda x: -x[1]):
        if imp > 0.01:
            log.info("  Feature importance: %-20s %.3f", name, imp)

    # 2. Decision Tree (deeper, for best performance)
    log.info("\n--- Decision Tree (max_depth=8) ---")
    dt8 = DecisionTreeClassifier(
        max_depth=8, min_samples_leaf=20, class_weight="balanced", random_state=42
    )
    dt8_scores = cross_val_score(dt8, X, y, cv=cv, scoring="f1")
    log.info("  CV F1: %.3f ± %.3f", dt8_scores.mean(), dt8_scores.std())

    # 3. Logistic Regression (feature weights)
    log.info("\n--- Logistic Regression ---")
    lr = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)
    lr_scores = cross_val_score(lr, X, y, cv=cv, scoring="f1")
    log.info("  CV F1: %.3f ± %.3f", lr_scores.mean(), lr_scores.std())

    lr.fit(X, y)
    log.info("  Coefficients:")
    for name, coef in sorted(zip(feature_names, lr.coef_[0]), key=lambda x: -abs(x[1])):
        log.info("    %-20s %+.4f", name, coef)

    # 4. Gradient Boosting (best performance ceiling)
    log.info("\n--- Gradient Boosting (100 trees) ---")
    gb = GradientBoostingClassifier(
        n_estimators=100, max_depth=4, min_samples_leaf=20, random_state=42
    )
    gb_scores = cross_val_score(gb, X, y, cv=cv, scoring="f1")
    log.info("  CV F1: %.3f ± %.3f", gb_scores.mean(), gb_scores.std())

    gb.fit(X, y)
    log.info("  Feature importance:")
    for name, imp in sorted(zip(feature_names, gb.feature_importances_), key=lambda x: -x[1]):
        if imp > 0.01:
            log.info("    %-20s %.3f", name, imp)

    # 5. Per-scenario decision trees
    log.info("\n" + "=" * 70)
    log.info("PER-SCENARIO TREES")
    log.info("=" * 70)

    for scenario_name, scenario_id in scenario_map.items():
        s_mask = scenario_num[valid] == scenario_id
        X_s = X[s_mask]
        y_s = y[s_mask]
        if y_s.sum() < 10:
            continue

        dt_s = DecisionTreeClassifier(
            max_depth=4, min_samples_leaf=30, class_weight="balanced", random_state=42
        )
        scores_s = cross_val_score(dt_s, X_s, y_s, cv=cv, scoring="f1")
        dt_s.fit(X_s, y_s)

        # Remove scenario feature for per-scenario trees
        feat_no_scenario = [f for f in feature_names if f != "scenario"]

        log.info(
            "\n%s (n=%d, periodic=%.1f%%, CV F1=%.3f):",
            scenario_name,
            len(y_s),
            100 * y_s.mean(),
            scores_s.mean(),
        )
        tree_s = export_text(dt_s, feature_names=feature_names, max_depth=4)
        log.info("%s", tree_s)

    # Summary comparison
    log.info("\n" + "=" * 70)
    log.info("SUMMARY")
    log.info("=" * 70)
    log.info("%-25s  CV F1 (mean ± std)", "Model")
    log.info("%-25s  %.3f ± %.3f", "Decision Tree (depth=5)", dt_scores.mean(), dt_scores.std())
    log.info("%-25s  %.3f ± %.3f", "Decision Tree (depth=8)", dt8_scores.mean(), dt8_scores.std())
    log.info("%-25s  %.3f ± %.3f", "Logistic Regression", lr_scores.mean(), lr_scores.std())
    log.info("%-25s  %.3f ± %.3f", "Gradient Boosting", gb_scores.mean(), gb_scores.std())
    log.info("\nHand-crafted best (clusters>=2): F1 ≈ 0.29 (from previous run)")
    log.info("The ML improvement over hand-crafted rules shows how much")
    log.info("structure remains to be discovered in the tensor features.")


if __name__ == "__main__":
    main()
