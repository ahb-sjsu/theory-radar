#!/usr/bin/env python3
"""Theory Radar on LARGE datasets — GPU-saturating.

These datasets have N=5K-245K, making GPU sort operations meaningful.
Uses batch-probe to find optimal candidate batch size per dataset.
Meta-learned pruning inside each fold. Fair test evaluation.
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

from sklearn.datasets import fetch_openml, load_breast_cancer
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger()

# ─── Core GPU ops (same as before) ────────────────────────────────

def gpu_batch_f1(vals, labels):
    N, K = vals.shape
    pos = labels.sum()
    if float(pos) == 0 or float(pos) == N:
        return cp.zeros(K, dtype=cp.float64)
    return cp.maximum(_sweep(vals, labels, pos), _sweep(-vals, labels, pos))

def _sweep(vals, labels, pos):
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

VBINARY = [
    ("+", lambda B,F: B+F), ("-", lambda B,F: B-F),
    ("*", lambda B,F: B*F), ("/", lambda B,F: B/(F+1e-30)),
    ("max", lambda B,F: cp.maximum(B,F)), ("min", lambda B,F: cp.minimum(B,F)),
    ("hypot", lambda B,F: cp.sqrt(B**2+F**2)),
    ("diff_sq", lambda B,F: (B-F)**2),
    ("harmonic", lambda B,F: 2*B*F/(B+F+1e-30)),
    ("geometric", lambda B,F: cp.sign(B*F)*cp.sqrt(cp.abs(B*F))),
]
VUNARY = [
    ("log", lambda V: cp.log(cp.abs(V)+1e-30)),
    ("sqrt", lambda V: cp.sqrt(cp.abs(V))),
    ("sq", lambda V: V**2), ("abs", lambda V: cp.abs(V)),
    ("sigmoid", lambda V: 1.0/(1.0+cp.exp(-cp.clip(V,-500,500)))),
    ("tanh", lambda V: cp.tanh(cp.clip(V,-500,500))),
]
SCALAR_BIN = {
    "+": lambda a,b: a+b, "-": lambda a,b: a-b,
    "*": lambda a,b: a*b, "/": lambda a,b: a/(b+1e-30),
    "max": lambda a,b: np.maximum(a,b), "min": lambda a,b: np.minimum(a,b),
    "hypot": lambda a,b: np.sqrt(a**2+b**2), "diff_sq": lambda a,b: (a-b)**2,
    "harmonic": lambda a,b: 2*a*b/(a+b+1e-30),
    "geometric": lambda a,b: np.sign(a*b)*np.sqrt(np.abs(a*b)),
}
SCALAR_UN = {
    "log": lambda x: np.log(np.abs(x)+1e-30),
    "sqrt": lambda x: np.sqrt(np.abs(x)),
    "sq": lambda x: x**2, "abs": lambda x: np.abs(x),
    "sigmoid": lambda x: 1.0/(1.0+np.exp(-np.clip(x,-500,500))),
    "tanh": lambda x: np.tanh(np.clip(x,-500,500)),
}

class FormulaTrace:
    def __init__(self, fi):
        self.ops = [("leaf", fi)]
    def binary(self, op, fi):
        t = FormulaTrace.__new__(FormulaTrace); t.ops = self.ops + [("binary", op, fi)]; return t
    def unary(self, op):
        t = FormulaTrace.__new__(FormulaTrace); t.ops = self.ops + [("unary", op)]; return t
    def evaluate(self, X):
        v = None
        for s in self.ops:
            if s[0]=="leaf": v = X[:,s[1]].copy()
            elif s[0]=="binary": v = SCALAR_BIN[s[1]](v, X[:,s[2]]); v = np.nan_to_num(v,nan=0.,posinf=1e10,neginf=-1e10)
            elif s[0]=="unary": v = SCALAR_UN[s[1]](v); v = np.nan_to_num(v,nan=0.,posinf=1e10,neginf=-1e10)
        return v

def find_optimal_threshold(vals, y):
    best_f1, best_t, best_d = 0., 0., 1
    P = int(y.astype(bool).sum()); N = len(y)
    if P==0 or P==N: return 0., 1, 0.
    for d in [1,-1]:
        o = np.argsort(d*vals)[::-1]; sv=(d*vals)[o]; sa=y.astype(bool)[o]
        tp=fp=0
        for i in range(N):
            if sa[i]: tp+=1
            else: fp+=1
            if tp+fp>0:
                p=tp/(tp+fp); r=tp/P
                if p+r>0:
                    f=2*p*r/(p+r)
                    if f>best_f1: best_f1,best_t,best_d=f,sv[i],d
    return best_t, best_d, best_f1


# ─── GPU Beam Search (vectorized, for large N) ────────────────────

def gpu_beam_search_large(X_gpu, y_gpu, feat_names, feat_idx=None,
                           max_depth=3, beam_width=100):
    """Fully vectorized beam search. For large N, the GPU sort dominates."""
    N, d_full = X_gpu.shape
    if feat_idx is not None:
        X_sub = X_gpu[:, feat_idx]
        local_names = [feat_names[i] for i in feat_idx]
    else:
        X_sub = X_gpu
        local_names = feat_names
        feat_idx = list(range(d_full))

    d = X_sub.shape[1]
    y_f = y_gpu.astype(cp.float64)

    # Depth 1
    f1s = gpu_batch_f1(X_sub, y_f)
    order = cp.argsort(-f1s).get()
    B = min(beam_width, d)

    beam_vals = cp.empty((N, B), dtype=cp.float64)
    beam_names = []
    beam_traces = []
    for rank, li in enumerate(order[:B]):
        gi = feat_idx[li]
        beam_vals[:, rank] = X_sub[:, li]
        beam_names.append(local_names[li])
        beam_traces.append(FormulaTrace(gi))

    best_f1 = float(f1s[order[0]])
    best_name = beam_names[0]
    best_trace = beam_traces[0]

    for depth in range(2, max_depth + 1):
        B_cur = beam_vals.shape[1]
        bv3 = beam_vals[:, :, None]  # (N, B, 1)
        fv3 = X_sub[:, None, :]      # (N, 1, d)

        chunks = []
        name_chunks = []
        trace_chunks = []

        for opname, opfn in VBINARY:
            try:
                out = opfn(bv3, fv3)  # (N, B, d)
                out = cp.nan_to_num(out, nan=0., posinf=1e10, neginf=-1e10)
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
            except: pass

        for opname, opfn in VUNARY:
            try:
                out = opfn(beam_vals)
                out = cp.nan_to_num(out, nan=0., posinf=1e10, neginf=-1e10)
                chunks.append(out)
                names = [f"{opname}({beam_names[bi]})" for bi in range(B_cur)]
                traces = [beam_traces[bi].unary(opname) for bi in range(B_cur)]
                name_chunks.append(names)
                trace_chunks.append(traces)
            except: pass

        if not chunks: break

        all_vals = cp.concatenate(chunks, axis=1)
        all_names = [n for nc in name_chunks for n in nc]
        all_traces = [t for tc in trace_chunks for t in tc]

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


# ─── CPU Baselines ─────────────────────────────────────────────────

def run_sklearn_fold(X, y, tr, te, fi):
    out = {}
    for nm, clf in [
        ("GB", GradientBoostingClassifier(n_estimators=100, max_depth=4, random_state=42+fi)),
        ("RF", RandomForestClassifier(n_estimators=100, random_state=42+fi)),
        ("LR", LogisticRegression(max_iter=1000, random_state=42)),
    ]:
        clf.fit(X[tr], y[tr])
        out[nm] = f1_score(y[te], clf.predict(X[te]))
    return out


# ─── Main ──────────────────────────────────────────────────────────

def run_dataset(name, X, y, features, n_repeats=50, n_folds=5,
                max_depth=3, beam_width=100, n_subspaces=10, subspace_k=8):
    total = n_repeats * n_folds
    log.info("=" * 70)
    log.info("%s (N=%d, d=%d, prev=%.2f) — %d folds, depth=%d, beam=%d",
             name, X.shape[0], X.shape[1], y.mean(), total, max_depth, beam_width)
    log.info("=" * 70)

    cv = RepeatedStratifiedKFold(n_splits=n_folds, n_repeats=n_repeats, random_state=42)
    splits = list(cv.split(X, y))

    X_gpu = cp.asarray(X, dtype=cp.float64)
    y_gpu = cp.asarray(y, dtype=cp.float64)

    d = X.shape[1]
    astar_f1s, astar_train_f1s, formulas = [], [], []
    t0 = time.time()

    for fold_i, (tr, te) in enumerate(splits):
        tr_gpu = cp.asarray(tr)
        rng = np.random.RandomState(42 + fold_i)

        # Subspace fuzzing
        best_f1, best_name, best_trace = 0., "", None
        for trial in range(n_subspaces):
            k = min(subspace_k, d)
            feat_idx = list(range(d)) if d <= k else sorted(rng.choice(d, k, replace=False).tolist())

            f1, nm, trace = gpu_beam_search_large(
                X_gpu[tr_gpu], y_gpu[tr_gpu], features,
                feat_idx=feat_idx, max_depth=max_depth, beam_width=beam_width)

            if f1 > best_f1:
                best_f1, best_name, best_trace = f1, nm, trace

        astar_train_f1s.append(best_f1)
        formulas.append(best_name)

        # Fair test eval
        vals_tr = best_trace.evaluate(X[tr])
        vals_te = best_trace.evaluate(X[te])
        thresh, dirn, _ = find_optimal_threshold(vals_tr, y[tr])
        preds = (dirn * vals_te >= thresh).astype(int)
        astar_f1s.append(f1_score(y[te], preds))

        if (fold_i + 1) % 25 == 0:
            elapsed = time.time() - t0
            rate = (fold_i + 1) / elapsed
            log.info("  %d/%d  train=%.3f test=%.3f  %.2f/s  ETA %.0fs",
                     fold_i+1, total, np.mean(astar_train_f1s[-25:]),
                     np.mean(astar_f1s[-25:]), rate, (total-fold_i-1)/rate)

    gpu_time = time.time() - t0
    log.info("  GPU: %.0fs (%.2f folds/s)", gpu_time, total/gpu_time)

    del X_gpu, y_gpu; cp.get_default_memory_pool().free_all_blocks()

    # CPU baselines
    t1 = time.time()
    n_jobs = min(os.cpu_count() or 1, 20)
    baselines = Parallel(n_jobs=n_jobs, backend="loky", verbose=0)(
        delayed(run_sklearn_fold)(X, y, tr, te, fi) for fi, (tr, te) in enumerate(splits))
    cpu_time = time.time() - t1

    astar_f1s = np.array(astar_f1s)
    astar_train_f1s = np.array(astar_train_f1s)
    results = {}
    for bn in ["GB", "RF", "LR"]:
        bf = np.array([r[bn] for r in baselines])
        diff = astar_f1s - bf
        t_stat, p_val = stats.ttest_1samp(diff, 0)
        results[bn] = {"mean": float(bf.mean()), "diff": float(diff.mean()),
                        "sigma": float(abs(t_stat)), "dir": "A*>" if diff.mean() > 0 else f"{bn}>"}

    fc = Counter(formulas); top = fc.most_common(1)[0]
    log.info("  FORMULA: %s (%.0f%%)", top[0][:60], 100*top[1]/len(formulas))
    log.info("  TRAIN: %.4f  TEST: %.4f  gap=%.4f",
             astar_train_f1s.mean(), astar_f1s.mean(),
             astar_train_f1s.mean() - astar_f1s.mean())
    for bn in ["GB", "RF", "LR"]:
        b = results[bn]
        log.info("  vs %-3s: %.4f  diff=%+.4f  %.1fs  %s", bn, b["mean"], b["diff"], b["sigma"], b["dir"])

    return {"name": name, "N": X.shape[0], "d": X.shape[1],
            "test_f1": float(astar_f1s.mean()), "train_f1": float(astar_train_f1s.mean()),
            "formula": top[0], "baselines": results,
            "gpu_time": gpu_time, "cpu_time": cpu_time}


def main():
    log.info("THEORY RADAR ON LARGE DATASETS — GPU-SATURATING")
    props = cp.cuda.runtime.getDeviceProperties(0)
    log.info("GPU: %s (%.1f GB free)", props["name"].decode(),
             cp.cuda.Device(0).mem_info[0] / 1e9)
    _ = cp.sort(cp.random.rand(10000, 100), axis=0); cp.cuda.Stream.null.synchronize()

    datasets = []

    # Large numerical datasets from OpenML
    OPENML = [
        ("eeg-eye-state",  1, "eeg",       14),
        ("MagicTelescope",  1, "magic",     10),
        ("electricity",     1, "elec",       8),
        ("MiniBooNE",       1, "miniboone", 50),
    ]

    for oml_name, ver, short, expected_d in OPENML:
        try:
            ds = fetch_openml(oml_name, version=ver, as_frame=False, parser="auto")
            X = ds.data.astype(float)
            X = StandardScaler().fit_transform(X)
            # Encode target to binary
            le = LabelEncoder()
            y = le.fit_transform(ds.target)
            if len(np.unique(y)) != 2:
                y = (y == y.max()).astype(int)
            datasets.append((short, X, y, [f"v{i}" for i in range(X.shape[1])]))
            log.info("Loaded %s: N=%d d=%d prev=%.2f", short, X.shape[0], X.shape[1], y.mean())
        except Exception as e:
            log.warning("%s: %s", oml_name, e)

    all_results = []
    for name, X, y, feats in datasets:
        d = X.shape[1]
        # Scale parameters by dataset size
        if X.shape[0] > 50000:
            nr, bw, ns, sk = 20, 50, 5, 8   # fewer folds for huge data
        elif X.shape[0] > 10000:
            nr, bw, ns, sk = 50, 100, 10, 8
        else:
            nr, bw, ns, sk = 100, 100, 15, min(8, d)

        r = run_dataset(name, X, y, feats, n_repeats=nr, n_folds=5,
                        max_depth=3, beam_width=bw, n_subspaces=ns, subspace_k=sk)
        all_results.append(r)

    with open("large_dataset_results.json", "w") as f:
        json.dump(all_results, f, indent=2)

    log.info("\n" + "=" * 100)
    log.info("LARGE DATASETS — Theory Radar (depth 3, fair eval)")
    for r in all_results:
        log.info("%-12s N=%-6d d=%-2d  test=%.4f  %s  GB:%.1f%s RF:%.1f%s LR:%.1f%s  gpu:%.0fs",
                 r["name"], r["N"], r["d"], r["test_f1"], r["formula"][:20],
                 r["baselines"]["GB"]["sigma"], r["baselines"]["GB"]["dir"],
                 r["baselines"]["RF"]["sigma"], r["baselines"]["RF"]["dir"],
                 r["baselines"]["LR"]["sigma"], r["baselines"]["LR"]["dir"],
                 r["gpu_time"])


if __name__ == "__main__":
    main()
