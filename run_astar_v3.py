#!/usr/bin/env python3
"""
A* formula search v3: Tight admissible bound via ROC-optimal F1.

Key insight: Instead of using the loose AUROC-to-F1 bound, compute
the EXACT maximum F1 achievable by ANY threshold on the given values.
This is O(N log N) — sort values, sweep all N possible thresholds,
track the best F1. This is EXACT and ADMISSIBLE by construction.

But wait — if we're computing the exact optimal F1 for each formula,
that's already what the exhaustive search does. The pruning comes from
computing a CHEAPER proxy first:

Strategy:
1. For each operation b(f_i, f_j), compute AUROC in O(N log N)
2. If AUROC < threshold_low, skip entirely (very cheap prune)
3. For remaining formulas, compute exact F1 via full sort-and-sweep
   in O(N log N) — this is the same as exhaustive but with a fast
   first-pass filter

The real speedup comes from step 2: many formulas have near-random
AUROC (≈0.5) and can be skipped immediately. We only do the full
F1 computation for formulas with discriminative power.

Additionally, we test a new theoretical result:

THEOREM (Optimal F1 from Sorted Values):
For N samples with P positives, the optimal thresholded F1 can be
computed in O(N log N) by:
1. Sort values descending
2. Walk through sorted values, maintaining running TP and FP counts
3. At each step, compute F1 = 2*TP / (2*TP + FP + FN)
4. Return the maximum

This is EXACT (no percentile approximation) and faster than the
11-percentile sweep for N > ~200.
"""

from __future__ import annotations

import logging
import time

import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.datasets import make_circles, make_moons, load_breast_cancer, load_iris
from sklearn.preprocessing import StandardScaler

import sys
sys.path.insert(0, "src")
from symbolic_search._search import _f1_threshold_sweep
from symbolic_search._ops import BINARY_OPS

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


def exact_optimal_f1(values: np.ndarray, actual: np.ndarray) -> float:
    """Compute the EXACT optimal thresholded F1 in O(N log N).

    This sweeps ALL possible thresholds (between consecutive sorted
    values), not just percentiles. It's guaranteed to find the global
    optimum.
    """
    values = np.nan_to_num(values, nan=0, posinf=1e10, neginf=-1e10)
    actual = actual.astype(bool)
    P = actual.sum()
    N_total = len(actual)
    Neg = N_total - P

    if P == 0 or Neg == 0:
        return 0.0

    best_f1 = 0.0

    # Try "above threshold" direction
    order = np.argsort(-values)  # descending
    sorted_actual = actual[order]

    tp = 0
    fp = 0
    for i in range(N_total):
        if sorted_actual[i]:
            tp += 1
        else:
            fp += 1
        fn = P - tp
        if tp + fp > 0:
            precision = tp / (tp + fp)
            recall = tp / P
            if precision + recall > 0:
                f1 = 2 * precision * recall / (precision + recall)
                best_f1 = max(best_f1, f1)

    # Also try "below threshold" direction
    order_asc = np.argsort(values)
    sorted_actual_asc = actual[order_asc]

    tp = 0
    fp = 0
    for i in range(N_total):
        if sorted_actual_asc[i]:
            tp += 1
        else:
            fp += 1
        fn = P - tp
        if tp + fp > 0:
            precision = tp / (tp + fp)
            recall = tp / P
            if precision + recall > 0:
                f1 = 2 * precision * recall / (precision + recall)
                best_f1 = max(best_f1, f1)

    return best_f1


def auroc_quick(values: np.ndarray, actual: np.ndarray) -> float:
    """Quick AUROC computation."""
    try:
        vals = np.nan_to_num(values, nan=0, posinf=1e10, neginf=-1e10)
        if len(np.unique(actual)) < 2:
            return 0.5
        auc = roc_auc_score(actual, vals)
        return max(auc, 1 - auc)
    except Exception:
        return 0.5


def main():
    log.info("=" * 70)
    log.info("A* FORMULA SEARCH v3: Exact F1 + AUROC Pre-filter")
    log.info("=" * 70)

    # ============================================================
    # First: verify that exact_optimal_f1 matches percentile sweep
    # ============================================================
    log.info("\n--- Verifying exact vs percentile F1 ---")
    rng = np.random.RandomState(42)
    n_match = 0
    n_exact_better = 0
    for trial in range(100):
        values = rng.randn(500)
        actual = (rng.rand(500) < 0.1).astype(bool)  # 10% prevalence
        f1_pct, _, _ = _f1_threshold_sweep(values, actual)
        f1_exact = exact_optimal_f1(values, actual)
        if abs(f1_exact - f1_pct) < 0.001:
            n_match += 1
        elif f1_exact > f1_pct:
            n_exact_better += 1

    log.info("  100 random trials: %d exact match, %d exact better, %d percentile better",
             n_match, n_exact_better, 100 - n_match - n_exact_better)

    # ============================================================
    # Benchmark: exhaustive vs AUROC-filtered exact search
    # ============================================================
    datasets = {}
    X, y = make_circles(n_samples=2000, noise=0.1, factor=0.5, random_state=42)
    datasets["Circles"] = (X, y, ["x1", "x2"])
    X, y = make_moons(n_samples=2000, noise=0.2, random_state=42)
    datasets["Moons"] = (X, y, ["x1", "x2"])
    bc = load_breast_cancer()
    X = StandardScaler().fit_transform(bc.data[:, :15])
    datasets["Breast Cancer (15D)"] = (X, bc.target, [f"f{i}" for i in range(15)])
    rng = np.random.RandomState(42)
    X = rng.randn(2000, 30)
    y = (X[:, 0] ** 2 + X[:, 1] ** 2 > 2).astype(int)
    datasets["Synthetic (30D)"] = (X, y, [f"x{i}" for i in range(30)])

    log.info("\n" + "=" * 70)
    log.info("BENCHMARK: Exhaustive vs AUROC-filtered search")
    log.info("=" * 70)

    auroc_threshold = 0.55  # Only evaluate formulas with AUROC > 0.55

    for name, (X, y, features) in datasets.items():
        N, d = X.shape
        actual = y.astype(bool)

        # Strategy 1: Exhaustive with exact F1
        t0 = time.time()
        best_f1_ex = 0
        best_formula_ex = ""
        n_evaluated_ex = 0

        for i in range(d):
            f1 = exact_optimal_f1(X[:, i], actual)
            n_evaluated_ex += 1
            if f1 > best_f1_ex:
                best_f1_ex = f1
                best_formula_ex = features[i]

        for i in range(d):
            for j in range(d):
                if i == j:
                    continue
                for op_name, op_fn in BINARY_OPS.items():
                    try:
                        vals = op_fn(X[:, i], X[:, j])
                        f1 = exact_optimal_f1(vals, actual)
                        n_evaluated_ex += 1
                        if f1 > best_f1_ex:
                            best_f1_ex = f1
                            best_formula_ex = f"{features[i]} {op_name} {features[j]}"
                    except Exception:
                        n_evaluated_ex += 1
        time_ex = time.time() - t0

        # Strategy 2: AUROC pre-filter + exact F1
        t0 = time.time()
        best_f1_filt = 0
        best_formula_filt = ""
        n_evaluated_filt = 0
        n_pruned = 0

        for i in range(d):
            f1 = exact_optimal_f1(X[:, i], actual)
            n_evaluated_filt += 1
            if f1 > best_f1_filt:
                best_f1_filt = f1
                best_formula_filt = features[i]

        for i in range(d):
            for j in range(d):
                if i == j:
                    continue
                for op_name, op_fn in BINARY_OPS.items():
                    try:
                        vals = op_fn(X[:, i], X[:, j])
                        vals = np.nan_to_num(vals, nan=0, posinf=1e10, neginf=-1e10)

                        # AUROC pre-filter (cheap)
                        auroc = auroc_quick(vals, actual)
                        if auroc < auroc_threshold:
                            n_pruned += 1
                            continue

                        # Exact F1 (expensive)
                        f1 = exact_optimal_f1(vals, actual)
                        n_evaluated_filt += 1
                        if f1 > best_f1_filt:
                            best_f1_filt = f1
                            best_formula_filt = f"{features[i]} {op_name} {features[j]}"
                    except Exception:
                        n_evaluated_filt += 1
        time_filt = time.time() - t0

        f1_match = abs(best_f1_ex - best_f1_filt) < 0.001
        speedup = time_ex / max(time_filt, 0.001)
        prune_rate = n_pruned / max(n_pruned + n_evaluated_filt, 1)

        log.info("\n  %s (d=%d):", name, d)
        log.info("    Exhaustive:     F1=%.4f, evals=%d, time=%.2fs, %s",
                 best_f1_ex, n_evaluated_ex, time_ex, best_formula_ex)
        log.info("    AUROC-filtered: F1=%.4f, evals=%d, pruned=%d (%.0f%%), time=%.2fs",
                 best_f1_filt, n_evaluated_filt, n_pruned, 100 * prune_rate, time_filt)
        log.info("    Speedup: %.1fx, F1 match: %s, Formula: %s",
                 speedup, "YES" if f1_match else "*** NO ***",
                 best_formula_filt)

        if not f1_match:
            log.info("    *** ADMISSIBILITY VIOLATION: pruned the best formula ***")

    # ============================================================
    # Sweep AUROC threshold to find optimal pruning level
    # ============================================================
    log.info("\n" + "=" * 70)
    log.info("AUROC THRESHOLD SWEEP: Finding optimal pruning level")
    log.info("=" * 70)

    # Use Breast Cancer as test case
    X_bc, y_bc, features_bc = datasets["Breast Cancer (15D)"]
    actual_bc = y_bc.astype(bool)

    log.info("%-10s  %8s  %8s  %8s  %8s  %5s",
             "Threshold", "F1", "Evals", "Pruned", "Time", "Match")

    # First get exhaustive baseline
    best_f1_base = 0
    for i in range(X_bc.shape[1]):
        for j in range(X_bc.shape[1]):
            if i == j:
                continue
            for op_fn in BINARY_OPS.values():
                try:
                    vals = op_fn(X_bc[:, i], X_bc[:, j])
                    f1 = exact_optimal_f1(vals, actual_bc)
                    best_f1_base = max(best_f1_base, f1)
                except Exception:
                    pass

    for auroc_thresh in [0.50, 0.52, 0.55, 0.58, 0.60, 0.65, 0.70, 0.75, 0.80]:
        t0 = time.time()
        best_f1 = 0
        n_eval = 0
        n_prune = 0

        for i in range(X_bc.shape[1]):
            f1 = exact_optimal_f1(X_bc[:, i], actual_bc)
            n_eval += 1
            best_f1 = max(best_f1, f1)

        for i in range(X_bc.shape[1]):
            for j in range(X_bc.shape[1]):
                if i == j:
                    continue
                for op_fn in BINARY_OPS.values():
                    try:
                        vals = op_fn(X_bc[:, i], X_bc[:, j])
                        vals = np.nan_to_num(vals, nan=0, posinf=1e10, neginf=-1e10)
                        auroc = auroc_quick(vals, actual_bc)
                        if auroc < auroc_thresh:
                            n_prune += 1
                            continue
                        f1 = exact_optimal_f1(vals, actual_bc)
                        n_eval += 1
                        best_f1 = max(best_f1, f1)
                    except Exception:
                        n_eval += 1
        elapsed = time.time() - t0
        match = abs(best_f1 - best_f1_base) < 0.001

        log.info("%-10.2f  %8.4f  %8d  %8d  %8.2fs  %s",
                 auroc_thresh, best_f1, n_eval, n_prune, elapsed,
                 "YES" if match else "NO ***")


if __name__ == "__main__":
    main()
