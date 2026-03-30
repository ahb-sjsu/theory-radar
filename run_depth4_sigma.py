#!/usr/bin/env python3
"""Depth-4 beam search with fair evaluation.

Tests whether 4-feature formulas close the gap on datasets where
depth-3 formulas lose to baselines (BreastCancer, Ionosphere).
"""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import cupy as cp
import numpy as np
import logging
import time
import json
from collections import Counter
from scipy import stats
from joblib import Parallel, delayed

from sklearn.datasets import load_breast_cancer, fetch_openml
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger()

# Import everything from v2
exec(open("/home/claude/tensor-3body/run_gpu_sigma_v2.py").read().split("def main")[0])


def run_dataset_d4(name, X, y, features, n_repeats=200, n_folds=5):
    total = n_repeats * n_folds
    log.info("=" * 70)
    log.info("%s DEPTH 4 (N=%d, d=%d) — %d folds [FAIR EVAL]",
             name, X.shape[0], X.shape[1], total)
    log.info("=" * 70)

    cv = RepeatedStratifiedKFold(n_splits=n_folds, n_repeats=n_repeats, random_state=42)
    splits = list(cv.split(X, y))

    X_gpu = cp.asarray(X, dtype=cp.float64)
    y_gpu = cp.asarray(y, dtype=cp.float64)

    astar_f1s = []
    astar_train_f1s = []
    formulas = []
    t0 = time.time()

    for fold_i, (tr, te) in enumerate(splits):
        tr_gpu = cp.asarray(tr)
        train_f1, formula_name, trace = gpu_beam_search_traced(
            X_gpu[tr_gpu], y_gpu[tr_gpu], features,
            max_depth=4, beam_width=100)  # DEPTH 4
        astar_train_f1s.append(train_f1)
        formulas.append(formula_name)

        vals_train = trace.evaluate(X[tr])
        vals_test = trace.evaluate(X[te])
        threshold, direction, _ = find_optimal_threshold(vals_train, y[tr])
        preds_test = apply_threshold(vals_test, threshold, direction)
        test_f1 = f1_score(y[te], preds_test)
        astar_f1s.append(test_f1)

        if (fold_i + 1) % 100 == 0:
            elapsed = time.time() - t0
            rate = (fold_i + 1) / elapsed
            log.info("  D4 fold %d/%d  train=%.3f  test=%.3f  %.1f/s",
                     fold_i+1, total, np.mean(astar_train_f1s[-100:]),
                     np.mean(astar_f1s[-100:]), rate)

    gpu_time = time.time() - t0
    log.info("  GPU done: %.0fs (%.1f folds/s)", gpu_time, total/gpu_time)

    # CPU baselines
    t1 = time.time()
    n_jobs = min(os.cpu_count() or 1, 20)
    baseline_results = Parallel(n_jobs=n_jobs, backend="loky", verbose=0)(
        delayed(run_sklearn_fold)(X, y, tr, te, fi)
        for fi, (tr, te) in enumerate(splits)
    )
    cpu_time = time.time() - t1

    astar_f1s = np.array(astar_f1s)
    astar_train_f1s = np.array(astar_train_f1s)
    results = {}
    for bname in ["GB", "RF", "LR"]:
        bf1s = np.array([r[bname] for r in baseline_results])
        diffs = astar_f1s - bf1s
        t_stat, p_value = stats.ttest_1samp(diffs, 0)
        results[bname] = {
            "mean": float(bf1s.mean()), "std": float(bf1s.std()),
            "diff": float(diffs.mean()), "sigma": float(abs(t_stat)),
            "p": float(p_value),
            "dir": "A*>" if diffs.mean() > 0 else f"{bname}>",
        }

    fc = Counter(formulas)
    top = fc.most_common(1)[0]

    log.info("  FORMULA: %s (%.0f%% stable)", top[0][:60], 100*top[1]/len(formulas))
    log.info("  TRAIN: %.4f  TEST: %.4f  gap=%.4f",
             astar_train_f1s.mean(), astar_f1s.mean(),
             astar_train_f1s.mean() - astar_f1s.mean())
    for bname in ["GB", "RF", "LR"]:
        b = results[bname]
        log.info("  vs %-3s: %.4f  diff=%+.4f  %.1f%s  %s",
                 bname, b["mean"], b["diff"], b["sigma"],
                 "σ", b["dir"])

    return {
        "name": name, "depth": 4,
        "test_f1": float(astar_f1s.mean()),
        "train_f1": float(astar_train_f1s.mean()),
        "formula": top[0],
        "baselines": results,
    }


def main():
    log.info("DEPTH-4 BEAM SEARCH — FAIR EVALUATION")

    props = cp.cuda.runtime.getDeviceProperties(0)
    log.info("GPU: %s (%.1f GB free)", props["name"].decode(),
             cp.cuda.Device(0).mem_info[0] / 1e9)

    _ = cp.sort(cp.random.rand(10000, 100), axis=0)
    cp.cuda.Stream.null.synchronize()

    # Only the datasets where depth-3 lost
    datasets = []

    bc = load_breast_cancer()
    X = StandardScaler().fit_transform(bc.data)
    datasets.append(("BreastCancer-D4", X, bc.target,
                     [f"f{i}" for i in range(X.shape[1])]))

    try:
        ion = fetch_openml("ionosphere", version=1, as_frame=False, parser="auto")
        X = StandardScaler().fit_transform(ion.data.astype(float))
        y = (ion.target == "g").astype(int)
        datasets.append(("Ionosphere-D4", X, y,
                         [f"i{i}" for i in range(X.shape[1])]))
    except Exception as e:
        log.warning("Ionosphere: %s", e)

    results = []
    for name, X, y, features in datasets:
        r = run_dataset_d4(name, X, y, features, n_repeats=200, n_folds=5)
        results.append(r)

    with open("depth4_fair_results.json", "w") as f:
        json.dump(results, f, indent=2)
    log.info("Saved depth4_fair_results.json")


if __name__ == "__main__":
    main()
