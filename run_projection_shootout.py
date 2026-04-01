#!/usr/bin/env python3
"""Projection Shootout: find the best projection strategy before
committing to the 17-dataset benchmark.

Tests on 3 key datasets:
- BreastCancer (d=30, where projections matter most)
- Ionosphere (d=34, complex boundary)
- Diabetes (d=8, where raw already wins)

Projection modes:
1. raw (no projection)
2. pca-8
3. pls-8 (supervised: maximizes covariance with y)
4. sparse-pca-8
5. pca+pls combined
6. pls-4 (fewer components, less overfitting)
7. pca-auto (tuned component count by explained variance > 95%)

100x5 CV with fair eval for speed. Winner goes into the 17-dataset run.
"""

import os
import sys

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
sys.path.insert(0, "src")

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
from sklearn.decomposition import PCA, SparsePCA
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score


logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger()

# ─── GPU Batch F1 ─────────────────────────────────────────────────


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
VUN = [
    ("log", lambda V: cp.log(cp.abs(V) + 1e-30)),
    ("sqrt", lambda V: cp.sqrt(cp.abs(V))),
    ("sq", lambda V: V**2),
    ("abs", lambda V: cp.abs(V)),
    ("sigmoid", lambda V: 1.0 / (1.0 + cp.exp(-cp.clip(V, -500, 500)))),
    ("tanh", lambda V: cp.tanh(cp.clip(V, -500, 500))),
]
SB = {
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
SU = {
    "log": lambda x: np.log(np.abs(x) + 1e-30),
    "sqrt": lambda x: np.sqrt(np.abs(x)),
    "sq": lambda x: x**2,
    "abs": lambda x: np.abs(x),
    "sigmoid": lambda x: 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500))),
    "tanh": lambda x: np.tanh(np.clip(x, -500, 500)),
}


class FT:
    def __init__(s, fi):
        s.ops = [("l", fi)]

    def b(s, op, fi):
        t = FT.__new__(FT)
        t.ops = s.ops + [("b", op, fi)]
        return t

    def u(s, op):
        t = FT.__new__(FT)
        t.ops = s.ops + [("u", op)]
        return t

    def ev(s, X):
        v = None
        for o in s.ops:
            if o[0] == "l":
                v = X[:, o[1]].copy()
            elif o[0] == "b":
                v = SB[o[1]](v, X[:, o[2]])
                v = np.nan_to_num(v, nan=0.0, posinf=1e10, neginf=-1e10)
            elif o[0] == "u":
                v = SU[o[1]](v)
                v = np.nan_to_num(v, nan=0.0, posinf=1e10, neginf=-1e10)
        return v


def opt_thresh(vals, y):
    bf, bt, bd = 0.0, 0.0, 1
    P = int(y.astype(bool).sum())
    N = len(y)
    if P == 0 or P == N:
        return 0.0, 1, 0.0
    for d in [1, -1]:
        o = np.argsort(d * vals)[::-1]
        sv = (d * vals)[o]
        sa = y.astype(bool)[o]
        tp = fp = 0
        for i in range(N):
            if sa[i]:
                tp += 1
            else:
                fp += 1
            if tp + fp > 0:
                p = tp / (tp + fp)
                r = tp / P
                if p + r > 0:
                    f = 2 * p * r / (p + r)
                    if f > bf:
                        bf, bt, bd = f, sv[i], d
    return bt, bd, bf


def beam_search(X_gpu, y_gpu, feat_names, max_depth=3, beam_width=100):
    N, d = X_gpu.shape
    yf = y_gpu.astype(cp.float64)
    f1s = gpu_batch_f1(X_gpu, yf)
    order = cp.argsort(-f1s).get()
    B = min(beam_width, d)
    bv = cp.empty((N, B), dtype=cp.float64)
    bn, bt = [], []
    for rank, i in enumerate(order[:B]):
        bv[:, rank] = X_gpu[:, i]
        bn.append(feat_names[i])
        bt.append(FT(i))
    best_f1, best_nm, best_tr = float(f1s[order[0]]), bn[0], bt[0]
    for depth in range(2, max_depth + 1):
        Bc = bv.shape[1]
        bv3 = bv[:, :, None]
        fv3 = X_gpu[:, None, :]
        ch, nc, tc = [], [], []
        for on, of in VBIN:
            try:
                out = of(bv3, fv3).reshape(N, Bc * d)
                out = cp.nan_to_num(out, nan=0.0, posinf=1e10, neginf=-1e10)
                ch.append(out)
                for bi in range(Bc):
                    for fi in range(d):
                        nc.append(f"({bn[bi]} {on} {feat_names[fi]})")
                        tc.append(bt[bi].b(on, fi))
            except:
                pass
        for on, of in VUN:
            try:
                out = of(bv)
                out = cp.nan_to_num(out, nan=0.0, posinf=1e10, neginf=-1e10)
                ch.append(out)
                for bi in range(Bc):
                    nc.append(f"{on}({bn[bi]})")
                    tc.append(bt[bi].u(on))
            except:
                pass
        if not ch:
            break
        av = cp.concatenate(ch, axis=1)
        var = cp.var(av, axis=0)
        f1s = gpu_batch_f1(av, yf)
        f1s = cp.where(var < 1e-20, 0.0, f1s)
        tk = min(beam_width, av.shape[1])
        ti = cp.argsort(-f1s)[:tk].get()
        tf = f1s[cp.asarray(ti)].get()
        bv = cp.empty((N, tk), dtype=cp.float64)
        bn, bt = [], []
        for rank, i in enumerate(ti):
            bv[:, rank] = av[:, i]
            bn.append(nc[i])
            bt.append(tc[i])
            if float(tf[rank]) > best_f1:
                best_f1, best_nm, best_tr = float(tf[rank]), nc[i], tc[i]
        del av, ch
        cp.get_default_memory_pool().free_all_blocks()
    return best_f1, best_nm, best_tr


# ─── Projection strategies ────────────────────────────────────────


def make_projection(name, X_tr, y_tr, X_te):
    """Fit projection on train, return (X_tr_aug, X_te_aug, feat_names)."""
    d = X_tr.shape[1]
    raw_names = [f"f{i}" for i in range(d)]

    if name == "raw":
        return X_tr, X_te, raw_names

    elif name == "pca-8":
        k = min(8, d, X_tr.shape[0] - 1)
        pca = PCA(n_components=k)
        p_tr = pca.fit_transform(X_tr)
        p_te = pca.transform(X_te)
        return (
            np.hstack([X_tr, p_tr]),
            np.hstack([X_te, p_te]),
            raw_names + [f"pc{i}" for i in range(k)],
        )

    elif name == "pls-8":
        k = min(8, d, X_tr.shape[0] - 1)
        pls = PLSRegression(n_components=k)
        pls.fit(X_tr, y_tr)
        p_tr = pls.transform(X_tr)
        p_te = pls.transform(X_te)
        return (
            np.hstack([X_tr, p_tr]),
            np.hstack([X_te, p_te]),
            raw_names + [f"pls{i}" for i in range(k)],
        )

    elif name == "pls-4":
        k = min(4, d, X_tr.shape[0] - 1)
        pls = PLSRegression(n_components=k)
        pls.fit(X_tr, y_tr)
        p_tr = pls.transform(X_tr)
        p_te = pls.transform(X_te)
        return (
            np.hstack([X_tr, p_tr]),
            np.hstack([X_te, p_te]),
            raw_names + [f"pls{i}" for i in range(k)],
        )

    elif name == "sparse-pca-8":
        k = min(8, d)
        spca = SparsePCA(n_components=k, random_state=42, max_iter=20)
        p_tr = spca.fit_transform(X_tr)
        p_te = spca.transform(X_te)
        return (
            np.hstack([X_tr, p_tr]),
            np.hstack([X_te, p_te]),
            raw_names + [f"sp{i}" for i in range(k)],
        )

    elif name == "pca+pls":
        k = min(4, d, X_tr.shape[0] - 1)
        pca = PCA(n_components=k)
        pls = PLSRegression(n_components=k)
        pca_tr = pca.fit_transform(X_tr)
        pca_te = pca.transform(X_te)
        pls.fit(X_tr, y_tr)
        pls_tr = pls.transform(X_tr)
        pls_te = pls.transform(X_te)
        return (
            np.hstack([X_tr, pca_tr, pls_tr]),
            np.hstack([X_te, pca_te, pls_te]),
            raw_names + [f"pc{i}" for i in range(k)] + [f"pls{i}" for i in range(k)],
        )

    elif name == "pca-auto":
        pca = PCA(n_components=0.95, svd_solver="full")  # 95% variance
        p_tr = pca.fit_transform(X_tr)
        p_te = pca.transform(X_te)
        k = p_tr.shape[1]
        return (
            np.hstack([X_tr, p_tr]),
            np.hstack([X_te, p_te]),
            raw_names + [f"pc{i}" for i in range(k)],
        )

    else:
        raise ValueError(f"Unknown projection: {name}")


def run_bl(X, y, tr, te, fi):
    o = {}
    for nm, clf in [
        ("GB", GradientBoostingClassifier(n_estimators=100, max_depth=4, random_state=42 + fi)),
        ("RF", RandomForestClassifier(n_estimators=100, random_state=42 + fi)),
        ("LR", LogisticRegression(max_iter=1000, random_state=42)),
    ]:
        clf.fit(X[tr], y[tr])
        o[nm] = f1_score(y[te], clf.predict(X[te]))
    return o


# ─── Main shootout ─────────────────────────────────────────────────


def run_shootout(name, X, y, n_repeats=100, n_folds=5):
    total = n_repeats * n_folds
    d = X.shape[1]
    log.info("=" * 70)
    log.info("SHOOTOUT: %s (N=%d, d=%d) — %d folds", name, X.shape[0], d, total)
    log.info("=" * 70)

    cv = RepeatedStratifiedKFold(n_splits=n_folds, n_repeats=n_repeats, random_state=42)
    splits = list(cv.split(X, y))

    PROJECTIONS = ["raw", "pca-8", "pls-8", "pls-4", "pca+pls", "pca-auto"]
    # Skip sparse PCA — too slow for shootout
    if d <= 15:
        PROJECTIONS.append("sparse-pca-8")

    all_results = {}

    for proj_name in PROJECTIONS:
        f1s_test = []
        f1s_train = []
        formulas = []
        t0 = time.time()

        for fold_i, (tr, te) in enumerate(splits):
            try:
                X_tr_aug, X_te_aug, aug_names = make_projection(proj_name, X[tr], y[tr], X[te])
            except Exception as e:
                log.warning("  %s fold %d failed: %s", proj_name, fold_i, e)
                f1s_test.append(0.0)
                f1s_train.append(0.0)
                formulas.append("")
                continue

            X_gpu = cp.asarray(X_tr_aug, dtype=cp.float64)
            y_gpu = cp.asarray(y[tr], dtype=cp.float64)

            train_f1, formula, trace = beam_search(
                X_gpu, y_gpu, aug_names, max_depth=3, beam_width=100
            )

            f1s_train.append(train_f1)
            formulas.append(formula)

            # Fair test eval
            vtr = trace.ev(X_tr_aug)
            vte = trace.ev(X_te_aug)
            th, dr, _ = opt_thresh(vtr, y[tr])
            preds = (dr * vte >= th).astype(int)
            f1s_test.append(f1_score(y[te], preds))

            del X_gpu, y_gpu
            cp.get_default_memory_pool().free_all_blocks()

        elapsed = time.time() - t0
        f1s_test = np.array(f1s_test)
        f1s_train = np.array(f1s_train)

        fc = Counter(formulas)
        top = fc.most_common(1)[0] if formulas else ("", 0)

        all_results[proj_name] = {
            "test_mean": float(f1s_test.mean()),
            "test_std": float(f1s_test.std()),
            "train_mean": float(f1s_train.mean()),
            "gap": float(f1s_train.mean() - f1s_test.mean()),
            "formula": top[0],
            "stability": float(top[1] / max(len(formulas), 1)),
            "time": elapsed,
            "f1s_test": f1s_test,
        }

        log.info(
            "  %-12s test=%.4f+/-%.4f  train=%.4f  gap=%.3f  %s (%.0f%%)  %.0fs",
            proj_name,
            f1s_test.mean(),
            f1s_test.std(),
            f1s_train.mean(),
            f1s_train.mean() - f1s_test.mean(),
            top[0][:30],
            100 * all_results[proj_name]["stability"],
            elapsed,
        )

    # Baselines (compute once)
    t1 = time.time()
    nj = min(os.cpu_count() or 1, 16)
    bls = Parallel(n_jobs=nj, backend="loky", verbose=0)(
        delayed(run_bl)(X, y, tr, te, fi) for fi, (tr, te) in enumerate(splits)
    )

    # Compare each projection vs baselines
    log.info("")
    log.info(
        "  %-12s %-8s %-8s %-5s  %-30s  %-10s %-10s %-10s",
        "Projection",
        "Test F1",
        "Stab%",
        "Gap",
        "Formula",
        "vs GB",
        "vs RF",
        "vs LR",
    )
    log.info("  " + "-" * 110)

    for proj_name, mr in all_results.items():
        bl_str = {}
        for bn in ["GB", "RF", "LR"]:
            bf = np.array([r[bn] for r in bls])
            diff = mr["f1s_test"] - bf
            ts, _ = stats.ttest_1samp(diff, 0)
            d_str = "A*>" if diff.mean() > 0 else f"{bn}>"
            bl_str[bn] = f"{abs(ts):.1f} {d_str}"

        log.info(
            "  %-12s %.4f   %4.0f%%   %.3f  %-30s  %-10s %-10s %-10s",
            proj_name,
            mr["test_mean"],
            100 * mr["stability"],
            mr["gap"],
            mr["formula"][:30],
            bl_str["GB"],
            bl_str["RF"],
            bl_str["LR"],
        )

    # Delete f1s arrays before JSON
    for v in all_results.values():
        del v["f1s_test"]

    return all_results


def main():
    log.info("PROJECTION SHOOTOUT — Finding the best strategy")

    props = cp.cuda.runtime.getDeviceProperties(0)
    log.info("GPU: %s (%.1f GB free)", props["name"].decode(), cp.cuda.Device(0).mem_info[0] / 1e9)
    _ = cp.sort(cp.random.rand(10000, 100), axis=0)
    cp.cuda.Stream.null.synchronize()

    datasets = []

    bc = load_breast_cancer()
    X = StandardScaler().fit_transform(bc.data)
    datasets.append(("BreastCancer", X, bc.target))

    try:
        ion = fetch_openml("ionosphere", version=1, as_frame=False, parser="auto")
        X = StandardScaler().fit_transform(ion.data.astype(float))
        y = (ion.target == "g").astype(int)
        datasets.append(("Ionosphere", X, y))
    except Exception as e:
        log.warning("Ionosphere: %s", e)

    try:
        pima = fetch_openml("diabetes", version=1, as_frame=False, parser="auto")
        X = StandardScaler().fit_transform(pima.data.astype(float))
        y = (pima.target == "tested_positive").astype(int)
        datasets.append(("Diabetes", X, y))
    except Exception as e:
        log.warning("Diabetes: %s", e)

    all_results = {}
    for name, X, y in datasets:
        r = run_shootout(name, X, y, n_repeats=100, n_folds=5)
        all_results[name] = r

    with open("projection_shootout.json", "w") as f:
        json.dump(all_results, f, indent=2)

    # Final recommendation
    log.info("\n" + "=" * 70)
    log.info("RECOMMENDATION")
    log.info("=" * 70)
    for ds_name, ds_results in all_results.items():
        best_proj = max(ds_results.items(), key=lambda x: x[1]["test_mean"])
        log.info("  %s: best = %s (test=%.4f)", ds_name, best_proj[0], best_proj[1]["test_mean"])


if __name__ == "__main__":
    main()
