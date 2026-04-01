#!/usr/bin/env python3
"""GPU-accelerated Theory Radar significance testing — FAIR evaluation.

KEY FIX: Formula is discovered on TRAIN data, then evaluated on TEST data
with threshold tuned on TRAIN. This matches how sklearn baselines are evaluated,
eliminating the train/test asymmetry.

The beam search now returns the formula's OPERATION TRACE (not just a name),
so it can be re-applied to test features. The optimal threshold is found on
train data and applied to test predictions.

Uses CuPy on GPU 0 | joblib (20 workers) for sklearn baselines.
200x5 = 1000 folds per dataset.
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

from sklearn.datasets import load_breast_cancer, load_wine, fetch_openml
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger()


# ─── Vectorized GPU Ops ───────────────────────────────────────────

VBINARY = [
    ("+", lambda B, F: B + F),
    ("-", lambda B, F: B - F),
    ("*", lambda B, F: B * F),
    ("/", lambda B, F: B / (F + 1e-30)),
    ("max", lambda B, F: cp.maximum(B, F)),
    ("min", lambda B, F: cp.minimum(B, F)),
    ("hypot", lambda B, F: cp.sqrt(B**2 + F**2)),
    ("diff_sq", lambda B, F: (B - F) ** 2),
    ("harmonic", lambda B, F: 2 * B * F / (B + F + 1e-30)),
    ("geometric", lambda B, F: cp.sign(B * F) * cp.sqrt(cp.abs(B * F))),
]

VUNARY = [
    ("log", lambda V: cp.log(cp.abs(V) + 1e-30)),
    ("sqrt", lambda V: cp.sqrt(cp.abs(V))),
    ("sq", lambda V: V**2),
    ("abs", lambda V: cp.abs(V)),
    ("sigmoid", lambda V: 1.0 / (1.0 + cp.exp(-cp.clip(V, -500, 500)))),
    ("tanh", lambda V: cp.tanh(cp.clip(V, -500, 500))),
]

# Scalar versions for re-applying formulas to numpy test data
SCALAR_BINARY = {
    "+": lambda a, b: a + b,
    "-": lambda a, b: a - b,
    "*": lambda a, b: a * b,
    "/": lambda a, b: a / (b + 1e-30),
    "max": lambda a, b: np.maximum(a, b),
    "min": lambda a, b: np.minimum(a, b),
    "hypot": lambda a, b: np.sqrt(a**2 + b**2),
    "diff_sq": lambda a, b: (a - b) ** 2,
    "harmonic": lambda a, b: 2 * a * b / (a + b + 1e-30),
    "geometric": lambda a, b: np.sign(a * b) * np.sqrt(np.abs(a * b)),
}

SCALAR_UNARY = {
    "log": lambda x: np.log(np.abs(x) + 1e-30),
    "sqrt": lambda x: np.sqrt(np.abs(x)),
    "sq": lambda x: x**2,
    "abs": lambda x: np.abs(x),
    "sigmoid": lambda x: 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500))),
    "tanh": lambda x: np.tanh(np.clip(x, -500, 500)),
}


# ─── GPU Batched F1 (with constant-value fix) ─────────────────────


def gpu_batch_f1(vals, labels):
    N, K = vals.shape
    pos = labels.sum()
    if float(pos) == 0 or float(pos) == N:
        return cp.zeros(K, dtype=cp.float64)
    return cp.maximum(_sweep_f1(vals, labels, pos), _sweep_f1(-vals, labels, pos))


def _sweep_f1(vals, labels, pos):
    N, K = vals.shape
    idx = cp.argsort(-vals, axis=0)
    sv = cp.take_along_axis(vals, idx, axis=0)
    sl = labels[idx]
    tp = cp.cumsum(sl, axis=0, dtype=cp.float64)
    fp = cp.cumsum(1.0 - sl, axis=0, dtype=cp.float64)
    denom = tp + pos + fp
    f1 = cp.where(denom > 0, 2.0 * tp / denom, 0.0)
    valid = cp.ones((N, K), dtype=cp.bool_)
    valid[:-1, :] = (sv[:-1, :] - sv[1:, :]) > 1e-15
    return cp.max(cp.where(valid, f1, 0.0), axis=0)


# ─── Formula Trace: trackable operations for test-set replay ──────


class FormulaTrace:
    """Records the operations used to build a formula so it can be
    replayed on new data (e.g., test set)."""

    def __init__(self, feature_idx):
        """Leaf node: just a raw feature."""
        self.ops = [("leaf", feature_idx)]

    def binary(self, op_name, feature_idx):
        """Apply binary op with a raw feature."""
        new = FormulaTrace.__new__(FormulaTrace)
        new.ops = self.ops + [("binary", op_name, feature_idx)]
        return new

    def unary(self, op_name):
        """Apply unary op."""
        new = FormulaTrace.__new__(FormulaTrace)
        new.ops = self.ops + [("unary", op_name)]
        return new

    def evaluate(self, X):
        """Replay the formula on numpy array X (N, d). Returns (N,) array."""
        vals = None
        for step in self.ops:
            if step[0] == "leaf":
                vals = X[:, step[1]].copy()
            elif step[0] == "binary":
                op_name, feat_idx = step[1], step[2]
                leaf = X[:, feat_idx]
                vals = SCALAR_BINARY[op_name](vals, leaf)
                vals = np.nan_to_num(vals, nan=0.0, posinf=1e10, neginf=-1e10)
            elif step[0] == "unary":
                op_name = step[1]
                vals = SCALAR_UNARY[op_name](vals)
                vals = np.nan_to_num(vals, nan=0.0, posinf=1e10, neginf=-1e10)
        return vals


def find_optimal_threshold(vals, y):
    """Find the threshold and direction that maximize F1 on the given data.
    Returns (threshold, direction, f1) where direction is +1 or -1."""
    best_f1, best_thresh, best_dir = 0.0, 0.0, 1
    y_bool = y.astype(bool)
    P = int(y_bool.sum())
    N = len(y)
    if P == 0 or P == N:
        return 0.0, 1, 0.0

    for direction in [1, -1]:
        order = np.argsort(direction * vals)[::-1]
        sv = (direction * vals)[order]
        sa = y_bool[order]
        tp, fp = 0, 0
        for i in range(N):
            if sa[i]:
                tp += 1
            else:
                fp += 1
            if tp + fp > 0:
                p = tp / (tp + fp)
                r = tp / P
                if p + r > 0:
                    f1 = 2 * p * r / (p + r)
                    if f1 > best_f1:
                        best_f1 = f1
                        # Threshold is the value at this position
                        best_thresh = sv[i]
                        best_dir = direction
    return best_thresh, best_dir, best_f1


def apply_threshold(vals, threshold, direction):
    """Apply a threshold to get binary predictions."""
    return (direction * vals >= threshold).astype(int)


# ─── GPU Beam Search with Formula Traces ──────────────────────────


def gpu_beam_search_traced(X_gpu, y_gpu, feat_names, max_depth=3, beam_width=100):
    """Beam search returning the best formula's TRACE (replayable on test data)."""
    N, d = X_gpu.shape
    y_f = y_gpu.astype(cp.float64)

    # Depth 1
    f1s = gpu_batch_f1(X_gpu, y_f)
    order = cp.argsort(-f1s).get()

    B = min(beam_width, d)
    beam_vals = cp.empty((N, B), dtype=cp.float64)
    beam_names = []
    beam_traces = []
    for rank, i in enumerate(order[:B]):
        beam_vals[:, rank] = X_gpu[:, i]
        beam_names.append(feat_names[i])
        beam_traces.append(FormulaTrace(i))

    best_f1 = float(f1s[order[0]])
    best_name = beam_names[0]
    best_trace = beam_traces[0]

    for depth in range(2, max_depth + 1):
        B_cur = beam_vals.shape[1]
        bv3 = beam_vals[:, :, None]
        fv3 = X_gpu[:, None, :]

        chunks = []
        name_chunks = []
        trace_chunks = []

        for opname, opfn in VBINARY:
            try:
                out = opfn(bv3, fv3)
                out = cp.nan_to_num(out, nan=0.0, posinf=1e10, neginf=-1e10)
                out = out.reshape(N, B_cur * d)
                chunks.append(out)

                names = []
                traces = []
                for bi in range(B_cur):
                    for fi in range(d):
                        names.append(f"({beam_names[bi]} {opname} {feat_names[fi]})")
                        traces.append(beam_traces[bi].binary(opname, fi))
                name_chunks.append(names)
                trace_chunks.append(traces)
            except Exception:
                pass

        for opname, opfn in VUNARY:
            try:
                out = opfn(beam_vals)
                out = cp.nan_to_num(out, nan=0.0, posinf=1e10, neginf=-1e10)
                chunks.append(out)

                names = [f"{opname}({beam_names[bi]})" for bi in range(B_cur)]
                traces = [beam_traces[bi].unary(opname) for bi in range(B_cur)]
                name_chunks.append(names)
                trace_chunks.append(traces)
            except Exception:
                pass

        if not chunks:
            break

        all_vals = cp.concatenate(chunks, axis=1)
        all_names = []
        all_traces = []
        for nc in name_chunks:
            all_names.extend(nc)
        for tc in trace_chunks:
            all_traces.extend(tc)
        total_K = all_vals.shape[1]

        del chunks, bv3, fv3

        # Kill constant formulas
        var = cp.var(all_vals, axis=0)
        constant_mask = var < 1e-20

        f1s = gpu_batch_f1(all_vals, y_f)
        f1s = cp.where(constant_mask, 0.0, f1s)

        top_k = min(beam_width, total_K)
        top_idx = cp.argsort(-f1s)[:top_k]
        top_idx_cpu = top_idx.get()
        top_f1_cpu = f1s[top_idx].get()

        beam_vals = cp.empty((N, top_k), dtype=cp.float64)
        beam_names = []
        beam_traces = []
        for rank, i in enumerate(top_idx_cpu):
            beam_vals[:, rank] = all_vals[:, i]
            beam_names.append(all_names[i])
            beam_traces.append(all_traces[i])
            f1_val = float(top_f1_cpu[rank])
            if f1_val > best_f1:
                best_f1 = f1_val
                best_name = all_names[i]
                best_trace = all_traces[i]

        del all_vals, f1s
        cp.get_default_memory_pool().free_all_blocks()

    return best_f1, best_name, best_trace


# ─── CPU Baselines ─────────────────────────────────────────────────


def run_sklearn_fold(X, y, tr, te, fold_i):
    X_tr, X_te, y_tr, y_te = X[tr], X[te], y[tr], y[te]
    out = {}
    for name, clf in [
        ("GB", GradientBoostingClassifier(n_estimators=100, max_depth=4, random_state=42 + fold_i)),
        ("RF", RandomForestClassifier(n_estimators=100, random_state=42 + fold_i)),
        ("LR", LogisticRegression(max_iter=1000, random_state=42)),
    ]:
        clf.fit(X_tr, y_tr)
        out[name] = f1_score(y_te, clf.predict(X_te))
    return out


# ─── Main Pipeline ─────────────────────────────────────────────────


def run_dataset(name, X, y, features, n_repeats=200, n_folds=5):
    total = n_repeats * n_folds
    log.info("=" * 70)
    log.info(
        "%s (N=%d, d=%d, prev=%.0f%%) — %dx%d = %d folds [FAIR EVAL]",
        name,
        X.shape[0],
        X.shape[1],
        100 * y.mean(),
        n_repeats,
        n_folds,
        total,
    )
    log.info("=" * 70)

    cv = RepeatedStratifiedKFold(n_splits=n_folds, n_repeats=n_repeats, random_state=42)
    splits = list(cv.split(X, y))

    # Phase 1: GPU formula search + FAIR test evaluation
    X_gpu = cp.asarray(X, dtype=cp.float64)
    y_gpu = cp.asarray(y, dtype=cp.float64)

    astar_f1s = []  # F1 on TEST data (fair)
    astar_train_f1s = []  # F1 on TRAIN data (for comparison)
    formulas = []
    t0 = time.time()

    for fold_i, (tr, te) in enumerate(splits):
        tr_gpu = cp.asarray(tr)
        X_tr_gpu = X_gpu[tr_gpu]
        y_tr_gpu = y_gpu[tr_gpu]

        # Search on training data
        train_f1, formula_name, trace = gpu_beam_search_traced(
            X_tr_gpu, y_tr_gpu, features, max_depth=3, beam_width=100
        )
        astar_train_f1s.append(train_f1)
        formulas.append(formula_name)

        # FAIR EVALUATION: replay formula on test data
        X_tr_np = X[tr]
        X_te_np = X[te]
        y_tr_np = y[tr]
        y_te_np = y[te]

        # Get formula values on train and test
        vals_train = trace.evaluate(X_tr_np)
        vals_test = trace.evaluate(X_te_np)

        # Find optimal threshold on TRAIN
        threshold, direction, _ = find_optimal_threshold(vals_train, y_tr_np)

        # Apply threshold to TEST
        preds_test = apply_threshold(vals_test, threshold, direction)
        test_f1 = f1_score(y_te_np, preds_test)
        astar_f1s.append(test_f1)

        if (fold_i + 1) % 100 == 0:
            elapsed = time.time() - t0
            rate = (fold_i + 1) / elapsed
            log.info(
                "  GPU fold %d/%d  train_F1=%.3f  test_F1=%.3f  %.1f folds/s",
                fold_i + 1,
                total,
                np.mean(astar_train_f1s[-100:]),
                np.mean(astar_f1s[-100:]),
                rate,
            )

    gpu_time = time.time() - t0
    log.info("  GPU done: %d folds in %.0fs (%.1f folds/s)", total, gpu_time, total / gpu_time)

    del X_gpu, y_gpu
    cp.get_default_memory_pool().free_all_blocks()

    # Phase 2: CPU baselines (parallel)
    t1 = time.time()
    n_jobs = min(os.cpu_count() or 1, 20)
    log.info("  CPU baselines: %d folds x %d workers...", total, n_jobs)

    baseline_results = Parallel(n_jobs=n_jobs, backend="loky", verbose=0)(
        delayed(run_sklearn_fold)(X, y, tr, te, fi) for fi, (tr, te) in enumerate(splits)
    )
    cpu_time = time.time() - t1
    log.info("  CPU done: %.0fs (%.1f folds/s)", cpu_time, total / cpu_time)

    # Phase 3: Stats (NOW COMPARING TEST vs TEST — FAIR)
    astar_f1s = np.array(astar_f1s)
    astar_train_f1s = np.array(astar_train_f1s)
    results = {}
    for bname in ["GB", "RF", "LR"]:
        bf1s = np.array([r[bname] for r in baseline_results])
        diffs = astar_f1s - bf1s
        t_stat, p_value = stats.ttest_1samp(diffs, 0)
        results[bname] = {
            "mean": float(bf1s.mean()),
            "std": float(bf1s.std()),
            "diff": float(diffs.mean()),
            "sigma": float(abs(t_stat)),
            "p": float(p_value),
            "dir": "A*>" if diffs.mean() > 0 else f"{bname}>",
        }

    fc = Counter(formulas)
    top = fc.most_common(1)[0]

    summary = {
        "name": name,
        "N": int(X.shape[0]),
        "d": int(X.shape[1]),
        "n_folds": total,
        "astar_test_mean": float(astar_f1s.mean()),
        "astar_test_std": float(astar_f1s.std()),
        "astar_train_mean": float(astar_train_f1s.mean()),
        "astar_train_std": float(astar_train_f1s.std()),
        "train_test_gap": float(astar_train_f1s.mean() - astar_f1s.mean()),
        "baselines": results,
        "formula": top[0],
        "stability": float(top[1] / len(formulas)),
        "gpu_time_s": float(gpu_time),
        "cpu_time_s": float(cpu_time),
    }

    log.info("  FORMULA: %s (%.0f%% stable)", top[0][:60], 100 * summary["stability"])
    log.info("  A* TRAIN: %.4f +/- %.4f", summary["astar_train_mean"], summary["astar_train_std"])
    log.info(
        "  A* TEST:  %.4f +/- %.4f  (gap=%.4f)",
        summary["astar_test_mean"],
        summary["astar_test_std"],
        summary["train_test_gap"],
    )
    for bname in ["GB", "RF", "LR"]:
        b = results[bname]
        log.info(
            "  vs %-3s: %.4f +/- %.4f  diff=%+.4f  %.1fσ  p=%.2e  %s",
            bname,
            b["mean"],
            b["std"],
            b["diff"],
            b["sigma"],
            b["p"],
            b["dir"],
        )
    return summary


def main():
    log.info("=" * 70)
    log.info("GPU THEORY RADAR — FAIR EVALUATION (test-set F1)")
    log.info("Formula: search on TRAIN, threshold on TRAIN, evaluate on TEST")
    log.info("Baselines: train on TRAIN, evaluate on TEST")
    log.info("Both sides evaluated on TEST — no methodological asymmetry")
    log.info("=" * 70)

    props = cp.cuda.runtime.getDeviceProperties(0)
    log.info("GPU: %s (%.1f GB free)", props["name"].decode(), cp.cuda.Device(0).mem_info[0] / 1e9)

    _ = cp.sort(cp.random.rand(10000, 100), axis=0)
    cp.cuda.Stream.null.synchronize()

    datasets = []

    bc = load_breast_cancer()
    X = StandardScaler().fit_transform(bc.data)
    datasets.append(("BreastCancer", X, bc.target, [f"f{i}" for i in range(X.shape[1])]))

    wine = load_wine()
    X = StandardScaler().fit_transform(wine.data)
    y = (wine.target == 0).astype(int)
    datasets.append(("Wine", X, y, [f"w{i}" for i in range(13)]))

    try:
        pima = fetch_openml("diabetes", version=1, as_frame=False, parser="auto")
        X = StandardScaler().fit_transform(pima.data.astype(float))
        y = (pima.target == "tested_positive").astype(int)
        datasets.append(("Diabetes", X, y, [f"d{i}" for i in range(X.shape[1])]))
    except Exception as e:
        log.warning("Diabetes: %s", e)

    try:
        bn = fetch_openml("banknote-authentication", version=1, as_frame=False, parser="auto")
        X = StandardScaler().fit_transform(bn.data.astype(float))
        y_raw = bn.target.astype(int)
        y = (y_raw == y_raw.max()).astype(int)
        datasets.append(("Banknote", X, y, [f"b{i}" for i in range(X.shape[1])]))
    except Exception as e:
        log.warning("Banknote: %s", e)

    try:
        ion = fetch_openml("ionosphere", version=1, as_frame=False, parser="auto")
        X = StandardScaler().fit_transform(ion.data.astype(float))
        y = (ion.target == "g").astype(int)
        datasets.append(("Ionosphere", X, y, [f"i{i}" for i in range(X.shape[1])]))
    except Exception as e:
        log.warning("Ionosphere: %s", e)

    log.info("Loaded %d datasets", len(datasets))

    all_results = []
    for name, X, y, features in datasets:
        result = run_dataset(name, X, y, features, n_repeats=200, n_folds=5)
        all_results.append(result)

    with open("gpu_sigma_fair_results.json", "w") as f:
        json.dump(all_results, f, indent=2)
    log.info("Saved gpu_sigma_fair_results.json")

    log.info("")
    log.info("=" * 100)
    log.info("FINAL SUMMARY — FAIR EVALUATION (200x5=1000 folds, test-set F1)")
    log.info(
        "%-15s  %-8s  %-8s  %-6s  %-25s  %-8s  %-8s  %-8s",
        "Dataset",
        "TrainF1",
        "TestF1",
        "Gap",
        "Formula",
        "vs GB",
        "vs RF",
        "vs LR",
    )
    log.info("-" * 100)
    for r in all_results:
        log.info(
            "%-15s  %.4f   %.4f   %.4f  %-25s  %6.1f   %6.1f   %6.1f",
            r["name"],
            r["astar_train_mean"],
            r["astar_test_mean"],
            r["train_test_gap"],
            r["formula"][:25],
            r["baselines"]["GB"]["sigma"],
            r["baselines"]["RF"]["sigma"],
            r["baselines"]["LR"]["sigma"],
        )
    log.info("=" * 100)


if __name__ == "__main__":
    main()
