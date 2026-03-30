#!/usr/bin/env python3
"""Deep PLS formula search: depth 4-5 on datasets where depth 3 lost.

Tests whether increased compositional depth with PLS projections
can close the gap with ensembles on Ionosphere and Sonar.
"""
import os, sys
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

import cupy as cp
import numpy as np
import logging
import time
import json
from collections import Counter
from scipy import stats
from joblib import Parallel, delayed

from sklearn.datasets import fetch_openml
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger()

# ─── GPU core ──────────────────────────────────────────────────────

def gpu_batch_f1(vals, labels):
    N, K = vals.shape
    pos = labels.sum()
    if float(pos) == 0 or float(pos) == N:
        return cp.zeros(K, dtype=cp.float64)
    return cp.maximum(_sw(vals, labels, pos), _sw(-vals, labels, pos))

def _sw(vals, labels, pos):
    N, K = vals.shape
    idx = cp.argsort(-vals, axis=0)
    sv = cp.take_along_axis(vals, idx, axis=0)
    sl = labels[idx]
    tp = cp.cumsum(sl, axis=0, dtype=cp.float64)
    fp = cp.cumsum(1.0 - sl, axis=0, dtype=cp.float64)
    denom = tp + pos + fp
    f1 = cp.where(denom > 0, 2.0 * tp / denom, 0.0)
    v = cp.ones((N, K), dtype=cp.bool_)
    v[:-1, :] = (sv[:-1, :] - sv[1:, :]) > 1e-15
    return cp.max(cp.where(v, f1, 0.0), axis=0)

VBIN = [
    ("+", lambda B, F: B + F), ("-", lambda B, F: B - F),
    ("*", lambda B, F: B * F), ("/", lambda B, F: B / (F + 1e-30)),
    ("max", lambda B, F: cp.maximum(B, F)), ("min", lambda B, F: cp.minimum(B, F)),
    ("hypot", lambda B, F: cp.sqrt(B**2 + F**2)),
    ("diff_sq", lambda B, F: (B - F) ** 2),
]
VUN = [
    ("sq", lambda V: V**2), ("abs", lambda V: cp.abs(V)),
    ("sqrt", lambda V: cp.sqrt(cp.abs(V))),
    ("tanh", lambda V: cp.tanh(cp.clip(V, -500, 500))),
]
SB = {
    "+": lambda a, b: a + b, "-": lambda a, b: a - b,
    "*": lambda a, b: a * b, "/": lambda a, b: a / (b + 1e-30),
    "max": lambda a, b: np.maximum(a, b), "min": lambda a, b: np.minimum(a, b),
    "hypot": lambda a, b: np.sqrt(a**2 + b**2), "diff_sq": lambda a, b: (a - b) ** 2,
}
SU = {
    "sq": lambda x: x**2, "abs": lambda x: np.abs(x),
    "sqrt": lambda x: np.sqrt(np.abs(x)),
    "tanh": lambda x: np.tanh(np.clip(x, -500, 500)),
}

class FT:
    def __init__(s, fi):
        s.ops = [("l", fi)]
    def b(s, op, fi):
        t = FT.__new__(FT); t.ops = s.ops + [("b", op, fi)]; return t
    def u(s, op):
        t = FT.__new__(FT); t.ops = s.ops + [("u", op)]; return t
    def ev(s, X):
        v = None
        for o in s.ops:
            if o[0] == "l": v = X[:, o[1]].copy()
            elif o[0] == "b":
                v = SB[o[1]](v, X[:, o[2]])
                v = np.nan_to_num(v, nan=0., posinf=1e10, neginf=-1e10)
            elif o[0] == "u":
                v = SU[o[1]](v)
                v = np.nan_to_num(v, nan=0., posinf=1e10, neginf=-1e10)
        return v

def opt_thresh(vals, y):
    bf, bt, bd = 0., 0., 1
    P = int(y.astype(bool).sum()); N = len(y)
    if P == 0 or P == N: return 0., 1, 0.
    for d in [1, -1]:
        o = np.argsort(d * vals)[::-1]; sv = (d * vals)[o]; sa = y.astype(bool)[o]
        tp = fp = 0
        for i in range(N):
            if sa[i]: tp += 1
            else: fp += 1
            if tp + fp > 0:
                p = tp / (tp + fp); r = tp / P
                if p + r > 0:
                    f = 2 * p * r / (p + r)
                    if f > bf: bf, bt, bd = f, sv[i], d
    return bt, bd, bf

def beam_search(X_gpu, y_gpu, feat_names, max_depth=3, beam_width=50):
    N, d = X_gpu.shape; yf = y_gpu.astype(cp.float64)
    f1s = gpu_batch_f1(X_gpu, yf)
    order = cp.argsort(-f1s).get()
    B = min(beam_width, d)
    bv = cp.empty((N, B), dtype=cp.float64); bn, bt = [], []
    for rank, i in enumerate(order[:B]):
        bv[:, rank] = X_gpu[:, i]; bn.append(feat_names[i]); bt.append(FT(i))
    best_f1, best_nm, best_tr = float(f1s[order[0]]), bn[0], bt[0]

    for depth in range(2, max_depth + 1):
        Bc = bv.shape[1]; bv3 = bv[:, :, None]; fv3 = X_gpu[:, None, :]
        ch, nc, tc = [], [], []
        for on, of in VBIN:
            try:
                out = of(bv3, fv3).reshape(N, Bc * d)
                out = cp.nan_to_num(out, nan=0., posinf=1e10, neginf=-1e10)
                ch.append(out)
                for bi in range(Bc):
                    for fi in range(d):
                        nc.append(f"({bn[bi]} {on} {feat_names[fi]})")
                        tc.append(bt[bi].b(on, fi))
            except Exception: pass
        for on, of in VUN:
            try:
                out = of(bv); out = cp.nan_to_num(out, nan=0., posinf=1e10, neginf=-1e10)
                ch.append(out)
                for bi in range(Bc):
                    nc.append(f"{on}({bn[bi]})"); tc.append(bt[bi].u(on))
            except Exception: pass
        if not ch: break
        av = cp.concatenate(ch, axis=1)
        var = cp.var(av, axis=0); f1s = gpu_batch_f1(av, yf)
        f1s = cp.where(var < 1e-20, 0.0, f1s)
        tk = min(beam_width, av.shape[1])
        ti = cp.argsort(-f1s)[:tk].get(); tf = f1s[cp.asarray(ti)].get()
        bv = cp.empty((N, tk), dtype=cp.float64); bn, bt = [], []
        for rank, i in enumerate(ti):
            bv[:, rank] = av[:, i]; bn.append(nc[i]); bt.append(tc[i])
            if float(tf[rank]) > best_f1:
                best_f1, best_nm, best_tr = float(tf[rank]), nc[i], tc[i]
        del av, ch
        try: del bv3, fv3
        except Exception: pass
        cp.get_default_memory_pool().free_all_blocks()
        log.info("    depth %d: best=%.4f %s", depth, best_f1, best_nm[:40])
    return best_f1, best_nm, best_tr

def run_bl(X, y, tr, te, fi):
    o = {}
    for nm, clf in [
        ("GB", GradientBoostingClassifier(n_estimators=100, max_depth=4, random_state=42+fi)),
        ("RF", RandomForestClassifier(n_estimators=100, random_state=42+fi)),
        ("LR", LogisticRegression(max_iter=1000, random_state=42)),
    ]:
        clf.fit(X[tr], y[tr]); o[nm] = f1_score(y[te], clf.predict(X[te]))
    return o

# ─── Main ──────────────────────────────────────────────────────────

def run_experiment(name, X, y, feat_names, max_depth, n_pls=8,
                   n_repeats=100, n_folds=5, beam_width=50):
    total = n_repeats * n_folds
    log.info("=" * 70)
    log.info("%s DEPTH %d (N=%d, d=%d) — %d folds, %d PLS, beam=%d",
             name, max_depth, X.shape[0], X.shape[1], total, n_pls, beam_width)
    log.info("=" * 70)

    cv = RepeatedStratifiedKFold(n_splits=n_folds, n_repeats=n_repeats, random_state=42)
    splits = list(cv.split(X, y))

    astar_f1s, astar_train_f1s, formulas = [], [], []
    t0 = time.time()

    for fold_i, (tr, te) in enumerate(splits):
        # PLS projection
        k = min(n_pls, X.shape[1], len(tr) - 1)
        pls = PLSRegression(n_components=k)
        pls.fit(X[tr], y[tr])
        pls_tr = pls.transform(X[tr])
        pls_te = pls.transform(X[te])

        X_aug_tr = np.hstack([X[tr], pls_tr])
        X_aug_te = np.hstack([X[te], pls_te])
        aug_names = feat_names + [f"pls{i}" for i in range(k)]

        # GPU beam search at specified depth
        X_gpu = cp.asarray(X_aug_tr, dtype=cp.float64)
        y_gpu = cp.asarray(y[tr], dtype=cp.float64)

        train_f1, formula, trace = beam_search(
            X_gpu, y_gpu, aug_names,
            max_depth=max_depth, beam_width=beam_width)

        astar_train_f1s.append(train_f1)
        formulas.append(formula)

        # Fair test eval
        vtr = trace.ev(X_aug_tr)
        vte = trace.ev(X_aug_te)
        th, dr, _ = opt_thresh(vtr, y[tr])
        preds = (dr * vte >= th).astype(int)
        astar_f1s.append(f1_score(y[te], preds))

        del X_gpu, y_gpu
        cp.get_default_memory_pool().free_all_blocks()

        if (fold_i + 1) % 50 == 0:
            elapsed = time.time() - t0
            rate = (fold_i + 1) / elapsed
            log.info("  %d/%d train=%.3f test=%.3f %.2f/s ETA %.0fs",
                     fold_i + 1, total, np.mean(astar_train_f1s[-50:]),
                     np.mean(astar_f1s[-50:]), rate, (total - fold_i - 1) / rate)

    gpu_time = time.time() - t0

    # Baselines
    nj = min(os.cpu_count() or 1, 16)
    bls = Parallel(n_jobs=nj, backend="loky", verbose=0)(
        delayed(run_bl)(X, y, tr, te, fi) for fi, (tr, te) in enumerate(splits))

    af = np.array(astar_f1s); at = np.array(astar_train_f1s)
    res = {}
    for bn in ["GB", "RF", "LR"]:
        bf = np.array([r[bn] for r in bls])
        diff = af - bf
        ts, pv = stats.ttest_1samp(diff, 0)
        res[bn] = {"mean": float(bf.mean()), "diff": float(diff.mean()),
                    "sigma": float(abs(ts)), "dir": "A*>" if diff.mean() > 0 else f"{bn}>"}

    fc = Counter(formulas); top = fc.most_common(1)[0]
    log.info("  FORMULA: %s (%.0f%%)", top[0][:60], 100 * top[1] / len(formulas))
    log.info("  TRAIN: %.4f  TEST: %.4f  gap=%.4f", at.mean(), af.mean(), at.mean() - af.mean())
    for bn in ["GB", "RF", "LR"]:
        b = res[bn]
        log.info("  vs %-3s: %.4f  diff=%+.4f  %.1fs  %s", bn, b["mean"], b["diff"], b["sigma"], b["dir"])

    return {"name": name, "depth": max_depth, "test_f1": float(af.mean()),
            "train_f1": float(at.mean()), "formula": top[0], "baselines": res}


def main():
    log.info("DEEP PLS FORMULA SEARCH — Depth 3 vs 4 vs 5")

    props = cp.cuda.runtime.getDeviceProperties(0)
    log.info("GPU: %s (%.1f GB free)", props["name"].decode(), cp.cuda.Device(0).mem_info[0] / 1e9)
    _ = cp.sort(cp.random.rand(10000, 100), axis=0)
    cp.cuda.Stream.null.synchronize()

    datasets = []

    # Datasets where depth-3 LOST to GB
    try:
        ion = fetch_openml("ionosphere", version=1, as_frame=False, parser="auto")
        X = StandardScaler().fit_transform(ion.data.astype(float))
        y = (ion.target == "g").astype(int)
        datasets.append(("Ionosphere", X, y, [f"i{i}" for i in range(X.shape[1])]))
    except Exception as e:
        log.warning("Ionosphere: %s", e)

    try:
        sonar = fetch_openml("sonar", version=1, as_frame=False, parser="auto")
        X = StandardScaler().fit_transform(sonar.data.astype(float))
        le = LabelEncoder(); y = le.fit_transform(sonar.target)
        datasets.append(("Sonar", X, y, [f"s{i}" for i in range(X.shape[1])]))
    except Exception as e:
        log.warning("Sonar: %s", e)

    all_results = []

    for name, X, y, feats in datasets:
        for depth in [3, 4, 5]:
            # Smaller beam for deeper search to manage memory
            bw = {3: 50, 4: 30, 5: 20}[depth]
            r = run_experiment(name, X, y, feats, max_depth=depth,
                              n_pls=8, n_repeats=100, n_folds=5, beam_width=bw)
            all_results.append(r)

    with open("deep_pls_results.json", "w") as f:
        json.dump(all_results, f, indent=2)

    log.info("\n" + "=" * 90)
    log.info("DEPTH COMPARISON (PLS + fair eval)")
    log.info("%-15s %5s  %-8s %-8s %-5s  %-10s %-10s %-10s",
             "Dataset", "Depth", "Test", "Train", "Gap", "vs GB", "vs RF", "vs LR")
    log.info("-" * 90)
    for r in all_results:
        bl = r["baselines"]
        log.info("%-15s %5d  %.4f  %.4f  %.3f  %5.1f %s  %5.1f %s  %5.1f %s",
                 r["name"], r["depth"], r["test_f1"], r["train_f1"],
                 r["train_f1"] - r["test_f1"],
                 bl["GB"]["sigma"], bl["GB"]["dir"],
                 bl["RF"]["sigma"], bl["RF"]["dir"],
                 bl["LR"]["sigma"], bl["LR"]["dir"])


if __name__ == "__main__":
    main()
