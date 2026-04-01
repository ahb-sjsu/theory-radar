#!/usr/bin/env python3
"""Real-world high-sigma significance testing on UCI datasets."""

import numpy as np
import sys
import logging
import time

sys.path.insert(0, "src")
from sklearn.datasets import load_breast_cancer, load_wine, fetch_openml
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from scipy import stats
from symbolic_search.radar import TheoryRadar

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger()


def run_cv(name, X, y, features, n_repeats=10, n_folds=5):
    cv = RepeatedStratifiedKFold(n_splits=n_folds, n_repeats=n_repeats, random_state=42)

    astar_f1s, gb_f1s, rf_f1s, lr_f1s = [], [], [], []
    formulas = []

    for fold_i, (tr, te) in enumerate(cv.split(X, y)):
        X_tr, X_te, y_tr, y_te = X[tr], X[te], y[tr], y[te]

        radar = TheoryRadar(X_tr, y_tr.astype(int), features)
        r = radar.search(
            mode="fast",
            f1_target=0,
            max_depth=3,
            max_expansions=15000,
            auroc_threshold=0.52,
            verbose=False,
        )
        astar_f1s.append(r.f1)
        formulas.append(r.formula)

        for Model, store in [
            (
                lambda: GradientBoostingClassifier(
                    n_estimators=100, max_depth=4, random_state=42 + fold_i
                ),
                gb_f1s,
            ),
            (
                lambda: RandomForestClassifier(n_estimators=100, random_state=42 + fold_i),
                rf_f1s,
            ),
            (lambda: LogisticRegression(max_iter=1000, random_state=42), lr_f1s),
        ]:
            m = Model()
            m.fit(X_tr, y_tr)
            store.append(f1_score(y_te, m.predict(X_te)))

        if (fold_i + 1) % 10 == 0:
            log.info(
                "  %s fold %d/%d A*=%.3f GB=%.3f",
                name,
                fold_i + 1,
                n_repeats * n_folds,
                np.mean(astar_f1s[-10:]),
                np.mean(gb_f1s[-10:]),
            )

    astar_f1s = np.array(astar_f1s)
    results = {}
    for bname, bf1s in [
        ("GB", np.array(gb_f1s)),
        ("RF", np.array(rf_f1s)),
        ("LR", np.array(lr_f1s)),
    ]:
        diffs = astar_f1s - bf1s
        t_stat, p_value = stats.ttest_1samp(diffs, 0)
        results[bname] = {
            "mean": bf1s.mean(),
            "std": bf1s.std(),
            "diff": diffs.mean(),
            "sigma": abs(t_stat),
            "p": p_value,
            "dir": "A*>" if diffs.mean() > 0 else bname + ">",
        }

    from collections import Counter

    fc = Counter(formulas)
    top = fc.most_common(1)[0]
    return {
        "astar_mean": astar_f1s.mean(),
        "astar_std": astar_f1s.std(),
        "baselines": results,
        "formula": top[0],
        "stability": top[1] / len(formulas),
    }


def main():
    log.info("=" * 70)
    log.info("REAL-WORLD HIGH-SIGMA: 10x5 CV, 3 baselines")
    log.info("=" * 70)

    datasets = []

    bc = load_breast_cancer()
    X = StandardScaler().fit_transform(bc.data[:, :10])
    datasets.append(("Breast Cancer 10D", X, bc.target, [f"f{i}" for i in range(10)]))

    wine = load_wine()
    X = StandardScaler().fit_transform(wine.data)
    y = (wine.target == 0).astype(int)
    datasets.append(("Wine 13D", X, y, [f"w{i}" for i in range(13)]))

    try:
        pima = fetch_openml("diabetes", version=1, as_frame=False, parser="auto")
        X = StandardScaler().fit_transform(pima.data.astype(float))
        y = (pima.target == "tested_positive").astype(int)
        datasets.append(("Diabetes 8D", X, y, [f"d{i}" for i in range(X.shape[1])]))
    except Exception as e:
        log.info("Diabetes: %s", e)

    try:
        bn = fetch_openml("banknote-authentication", version=1, as_frame=False, parser="auto")
        X = StandardScaler().fit_transform(bn.data.astype(float))
        y = bn.target.astype(int)
        datasets.append(("Banknote 4D", X, y, [f"b{i}" for i in range(X.shape[1])]))
    except Exception as e:
        log.info("Banknote: %s", e)

    try:
        ion = fetch_openml("ionosphere", version=1, as_frame=False, parser="auto")
        X = StandardScaler().fit_transform(ion.data.astype(float))
        y = (ion.target == "g").astype(int)
        datasets.append(("Ionosphere 34D", X, y, [f"i{i}" for i in range(X.shape[1])]))
    except Exception as e:
        log.info("Ionosphere: %s", e)

    for name, X, y, features in datasets:
        log.info(
            "\n%s (N=%d, d=%d, prev=%.0f%%)",
            name,
            X.shape[0],
            X.shape[1],
            100 * y.mean(),
        )
        t0 = time.time()
        r = run_cv(name, X, y, features)
        elapsed = time.time() - t0

        log.info("  Formula: %s (%.0f%% stable)", r["formula"][:50], 100 * r["stability"])
        log.info("  A*: %.3f±%.3f", r["astar_mean"], r["astar_std"])
        for bname in ["GB", "RF", "LR"]:
            b = r["baselines"][bname]
            log.info(
                "  vs %-3s: %.3f±%.3f  diff=%+.4f  %.1fσ  p=%.2e  %s",
                bname,
                b["mean"],
                b["std"],
                b["diff"],
                b["sigma"],
                b["p"],
                b["dir"],
            )
        log.info("  Time: %.0fs", elapsed)


if __name__ == "__main__":
    main()
