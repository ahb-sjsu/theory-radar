#!/usr/bin/env python3
"""Fuzzed feature-subspace symbolic search with fair evaluation.

Instead of searching all d features at depth 3, randomly sample
feature subsets of size k and search each to depth D. This:
- Enables deeper search (depth 4-5 with k=6 is fast)
- Explores more feature combinations (like random subspace method)
- Finds formulas that beam search misses due to beam domination

For each CV fold: run N_TRIALS random subspace searches, keep the
best formula, evaluate on test data with train-tuned threshold.
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

# ─── Import core machinery from v2 ────────────────────────────────
# (GPU batch F1, FormulaTrace, etc.)

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


class FormulaTrace:
    def __init__(self, feature_idx):
        self.ops = [("leaf", feature_idx)]

    def binary(self, op_name, feature_idx):
        new = FormulaTrace.__new__(FormulaTrace)
        new.ops = self.ops + [("binary", op_name, feature_idx)]
        return new

    def unary(self, op_name):
        new = FormulaTrace.__new__(FormulaTrace)
        new.ops = self.ops + [("unary", op_name)]
        return new

    def evaluate(self, X):
        vals = None
        for step in self.ops:
            if step[0] == "leaf":
                vals = X[:, step[1]].copy()
            elif step[0] == "binary":
                vals = SCALAR_BINARY[step[1]](vals, X[:, step[2]])
                vals = np.nan_to_num(vals, nan=0.0, posinf=1e10, neginf=-1e10)
            elif step[0] == "unary":
                vals = SCALAR_UNARY[step[1]](vals)
                vals = np.nan_to_num(vals, nan=0.0, posinf=1e10, neginf=-1e10)
        return vals


def find_optimal_threshold(vals, y):
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
                        best_thresh = sv[i]
                        best_dir = direction
    return best_thresh, best_dir, best_f1


# ─── Fuzzed Feature-Subspace Beam Search ──────────────────────────


def gpu_beam_search_subspace(X_gpu, y_gpu, feat_names, feat_idx, max_depth=4, beam_width=50):
    """Beam search on a SUBSET of features (given by feat_idx).

    feat_idx maps local index → global feature index.
    Names and traces use GLOBAL indices for correct test-set replay.
    """
    N = X_gpu.shape[0]
    d = len(feat_idx)
    X_sub = X_gpu[:, feat_idx]  # (N, d_sub)
    y_f = y_gpu.astype(cp.float64)

    # Map local names
    local_names = [feat_names[i] for i in feat_idx]

    # Depth 1
    f1s = gpu_batch_f1(X_sub, y_f)
    order = cp.argsort(-f1s).get()

    B = min(beam_width, d)
    beam_vals = cp.empty((N, B), dtype=cp.float64)
    beam_names = []
    beam_traces = []
    for rank, li in enumerate(order[:B]):
        gi = feat_idx[li]  # global index
        beam_vals[:, rank] = X_sub[:, li]
        beam_names.append(local_names[li])
        beam_traces.append(FormulaTrace(gi))

    best_f1 = float(f1s[order[0]])
    best_name = beam_names[0]
    best_trace = beam_traces[0]

    for depth in range(2, max_depth + 1):
        B_cur = beam_vals.shape[1]
        bv3 = beam_vals[:, :, None]
        fv3 = X_sub[:, None, :]

        chunks, name_chunks, trace_chunks = [], [], []

        for opname, opfn in VBINARY:
            try:
                out = opfn(bv3, fv3)
                out = cp.nan_to_num(out, nan=0.0, posinf=1e10, neginf=-1e10)
                out = out.reshape(N, B_cur * d)
                chunks.append(out)
                names, traces = [], []
                for bi in range(B_cur):
                    for li in range(d):
                        gi = feat_idx[li]
                        names.append(f"({beam_names[bi]} {opname} {local_names[li]})")
                        traces.append(beam_traces[bi].binary(opname, gi))
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
        all_names, all_traces = [], []
        for nc in name_chunks:
            all_names.extend(nc)
        for tc in trace_chunks:
            all_traces.extend(tc)

        var = cp.var(all_vals, axis=0)
        f1s = gpu_batch_f1(all_vals, y_f)
        f1s = cp.where(var < 1e-20, 0.0, f1s)

        top_k = min(beam_width, all_vals.shape[1])
        top_idx = cp.argsort(-f1s)[:top_k].get()
        top_f1 = f1s[cp.asarray(top_idx)].get()

        beam_vals = cp.empty((N, top_k), dtype=cp.float64)
        beam_names, beam_traces = [], []
        for rank, i in enumerate(top_idx):
            beam_vals[:, rank] = all_vals[:, i]
            beam_names.append(all_names[i])
            beam_traces.append(all_traces[i])
            if float(top_f1[rank]) > best_f1:
                best_f1 = float(top_f1[rank])
                best_name = all_names[i]
                best_trace = all_traces[i]

        del all_vals, chunks
        cp.get_default_memory_pool().free_all_blocks()

    return best_f1, best_name, best_trace


def fuzzed_search(
    X_gpu, y_gpu, feat_names, n_trials=50, subspace_k=6, max_depth=4, beam_width=50, rng=None
):
    """Run n_trials random-subspace searches, return the best formula."""
    if rng is None:
        rng = np.random.RandomState(42)

    d = X_gpu.shape[1]
    k = min(subspace_k, d)

    best_f1, best_name, best_trace = 0.0, "", None

    for trial in range(n_trials):
        feat_idx = sorted(rng.choice(d, k, replace=False).tolist())
        f1, name, trace = gpu_beam_search_subspace(
            X_gpu, y_gpu, feat_names, feat_idx, max_depth=max_depth, beam_width=beam_width
        )
        if f1 > best_f1:
            best_f1 = f1
            best_name = name
            best_trace = trace

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


def run_dataset(
    name, X, y, features, n_repeats=200, n_folds=5, n_trials=50, subspace_k=6, max_depth=4
):
    total = n_repeats * n_folds
    log.info("=" * 70)
    log.info(
        "%s (N=%d, d=%d) — %d folds, %d trials/fold, k=%d, depth=%d",
        name,
        X.shape[0],
        X.shape[1],
        total,
        n_trials,
        subspace_k,
        max_depth,
    )
    log.info("=" * 70)

    cv = RepeatedStratifiedKFold(n_splits=n_folds, n_repeats=n_repeats, random_state=42)
    splits = list(cv.split(X, y))

    X_gpu = cp.asarray(X, dtype=cp.float64)
    y_gpu = cp.asarray(y, dtype=cp.float64)

    astar_f1s, astar_train_f1s, formulas = [], [], []
    t0 = time.time()

    for fold_i, (tr, te) in enumerate(splits):
        tr_gpu = cp.asarray(tr)
        rng = np.random.RandomState(42 + fold_i)

        train_f1, formula_name, trace = fuzzed_search(
            X_gpu[tr_gpu],
            y_gpu[tr_gpu],
            features,
            n_trials=n_trials,
            subspace_k=subspace_k,
            max_depth=max_depth,
            beam_width=50,
            rng=rng,
        )

        astar_train_f1s.append(train_f1)
        formulas.append(formula_name)

        # Fair test evaluation
        vals_train = trace.evaluate(X[tr])
        vals_test = trace.evaluate(X[te])
        threshold, direction, _ = find_optimal_threshold(vals_train, y[tr])
        preds_test = (direction * vals_test >= threshold).astype(int)
        test_f1 = f1_score(y[te], preds_test)
        astar_f1s.append(test_f1)

        if (fold_i + 1) % 50 == 0:
            elapsed = time.time() - t0
            rate = (fold_i + 1) / elapsed
            log.info(
                "  Fuzz %d/%d  train=%.3f test=%.3f  %.2f folds/s  ETA %.0fs",
                fold_i + 1,
                total,
                np.mean(astar_train_f1s[-50:]),
                np.mean(astar_f1s[-50:]),
                rate,
                (total - fold_i - 1) / rate,
            )

    gpu_time = time.time() - t0
    log.info("  GPU done: %.0fs (%.2f folds/s)", gpu_time, total / gpu_time)

    # CPU baselines
    t1 = time.time()
    n_jobs = min(os.cpu_count() or 1, 20)
    baseline_results = Parallel(n_jobs=n_jobs, backend="loky", verbose=0)(
        delayed(run_sklearn_fold)(X, y, tr, te, fi) for fi, (tr, te) in enumerate(splits)
    )
    cpu_time = time.time() - t1
    log.info("  CPU baselines: %.0fs", cpu_time)

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

    log.info("  FORMULA: %s (%.0f%% stable)", top[0][:60], 100 * top[1] / len(formulas))
    log.info(
        "  TRAIN: %.4f  TEST: %.4f  gap=%.4f",
        astar_train_f1s.mean(),
        astar_f1s.mean(),
        astar_train_f1s.mean() - astar_f1s.mean(),
    )
    for bname in ["GB", "RF", "LR"]:
        b = results[bname]
        log.info(
            "  vs %-3s: %.4f  diff=%+.4f  %.1f%s  %s",
            bname,
            b["mean"],
            b["diff"],
            b["sigma"],
            "s",
            b["dir"],
        )

    return {
        "name": name,
        "n_trials": n_trials,
        "subspace_k": subspace_k,
        "max_depth": max_depth,
        "test_f1": float(astar_f1s.mean()),
        "train_f1": float(astar_train_f1s.mean()),
        "formula": top[0],
        "stability": float(top[1] / len(formulas)),
        "baselines": results,
    }


def main():
    log.info("=" * 70)
    log.info("FUZZED FEATURE-SUBSPACE SYMBOLIC SEARCH")
    log.info("50 random subspaces x depth 4 x k=6 features per fold")
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
        datasets.append(("Diabetes", X, y, [f"d{i}" for i in range(8)]))
    except Exception as e:
        log.warning("Diabetes: %s", e)

    try:
        ion = fetch_openml("ionosphere", version=1, as_frame=False, parser="auto")
        X = StandardScaler().fit_transform(ion.data.astype(float))
        y = (ion.target == "g").astype(int)
        datasets.append(("Ionosphere", X, y, [f"i{i}" for i in range(X.shape[1])]))
    except Exception as e:
        log.warning("Ionosphere: %s", e)

    all_results = []
    for name, X, y, features in datasets:
        # More trials for high-d datasets
        n_trials = 100 if X.shape[1] > 15 else 50
        r = run_dataset(
            name,
            X,
            y,
            features,
            n_repeats=200,
            n_folds=5,
            n_trials=n_trials,
            subspace_k=6,
            max_depth=4,
        )
        all_results.append(r)

    with open("fuzz_sigma_results.json", "w") as f:
        json.dump(all_results, f, indent=2)
    log.info("Saved fuzz_sigma_results.json")

    log.info("")
    log.info("=" * 100)
    log.info("FUZZ vs BASELINES (depth-4, k=6 subspace, fair test eval)")
    for r in all_results:
        log.info(
            "%-15s test=%.4f  formula=%s  vs GB %.1f%s  vs RF %.1f%s  vs LR %.1f%s",
            r["name"],
            r["test_f1"],
            r["formula"][:30],
            r["baselines"]["GB"]["sigma"],
            r["baselines"]["GB"]["dir"],
            r["baselines"]["RF"]["sigma"],
            r["baselines"]["RF"]["dir"],
            r["baselines"]["LR"]["sigma"],
            r["baselines"]["LR"]["dir"],
        )


if __name__ == "__main__":
    main()
