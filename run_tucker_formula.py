#!/usr/bin/env python3
"""Tucker-Decomposed Formula Search.

Constructs feature interaction tensors, decomposes via Tucker,
and searches for interpretable formulas over the Tucker factor scores.

Three feature modes compared:
  1. raw: original features only
  2. pca: PCA projections (linear)
  3. tucker: Tucker factor scores (nonlinear interactions)
  4. raw+tucker: both raw and Tucker features

The Tucker decomposition captures how features INTERACT, not just
individual variance. A formula over Tucker factors implicitly uses
all original features through multilinear projections.
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


# ─── Tucker Decomposition ─────────────────────────────────────────


def tucker_features(X, rank=8):
    """Extract Tucker factor scores from feature interaction tensor.

    1. Construct pairwise interaction matrix per sample:
       For each sample x (d,), compute the outer product x ⊗ x (d, d)
       This gives a (N, d, d) tensor of feature interactions.

    2. Unfold along mode-0 (samples) and compute Tucker via HOSVD:
       - Mode-1 unfolding: (N, d*d) → SVD gives sample factors
       - Mode-2,3: factor matrices from the (d,d) interaction structure

    Simplified approach: compute the (d, d) covariance-like interaction
    matrix, eigendecompose it, and project each sample's interaction
    vector onto the top eigenvectors.

    Returns: (N, rank) Tucker factor scores
    """
    N, d = X.shape

    # Method: Higher-Order SVD (HOSVD) on the interaction tensor
    # For efficiency, we use the n-mode unfolding approach:

    # Step 1: Compute feature interaction for each sample
    # X_interact[i] = vec(x_i ⊗ x_i) = x_i ⊗ x_i flattened
    # This is (N, d*d) — each row is the vectorized outer product

    # For memory efficiency, compute the Gram matrix of interactions
    # and extract eigenvectors

    # Efficient: X_interact = (X * X[:, :, None]).reshape(N, d*d)
    # But d*d can be large. Use batched approach:

    if d <= 50:
        # Direct: compute all pairwise products
        # X_interact[i, j*d+k] = X[i,j] * X[i,k]
        X_interact = np.zeros((N, d * d), dtype=np.float64)
        for j in range(d):
            X_interact[:, j * d : (j + 1) * d] = X[:, j : j + 1] * X  # (N, d)

        # SVD of the interaction matrix to get top-rank factors
        # Center first
        X_interact -= X_interact.mean(axis=0, keepdims=True)

        # Truncated SVD for efficiency
        from sklearn.decomposition import TruncatedSVD

        k = min(rank, d * d - 1, N - 1)
        svd = TruncatedSVD(n_components=k, random_state=42)
        tucker_scores = svd.fit_transform(X_interact)

        return tucker_scores, svd

    else:
        # For high-d, use random projection of interactions
        rng = np.random.RandomState(42)

        # Project pairwise interactions to lower dimension
        n_proj = min(rank * 10, 200)
        proj_pairs = rng.choice(d * d, n_proj, replace=False)

        X_proj = np.zeros((N, n_proj), dtype=np.float64)
        for idx, pair in enumerate(proj_pairs):
            j, k = divmod(pair, d)
            X_proj[:, idx] = X[:, j] * X[:, k]

        X_proj -= X_proj.mean(axis=0, keepdims=True)

        from sklearn.decomposition import TruncatedSVD

        k = min(rank, n_proj - 1, N - 1)
        svd = TruncatedSVD(n_components=k, random_state=42)
        tucker_scores = svd.fit_transform(X_proj)

        return tucker_scores, (svd, proj_pairs, d)


def tucker_transform(X, model):
    """Apply learned Tucker decomposition to new data."""
    N, d = X.shape

    if isinstance(model, tuple):
        # High-d case with random projection
        svd, proj_pairs, orig_d = model
        n_proj = len(proj_pairs)
        X_proj = np.zeros((N, n_proj), dtype=np.float64)
        for idx, pair in enumerate(proj_pairs):
            j, k = divmod(pair, orig_d)
            X_proj[:, idx] = X[:, j] * X[:, k]
        X_proj -= X_proj.mean(axis=0, keepdims=True)
        return svd.transform(X_proj)
    else:
        # Low-d case, direct
        svd = model
        d_sq = d * d
        X_interact = np.zeros((N, d_sq), dtype=np.float64)
        for j in range(d):
            X_interact[:, j * d : (j + 1) * d] = X[:, j : j + 1] * X
        X_interact -= X_interact.mean(axis=0, keepdims=True)
        return svd.transform(X_interact)


# ─── Beam Search ───────────────────────────────────────────────────


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


# ─── Main Pipeline ─────────────────────────────────────────────────


def run_dataset(
    name, X, y, feat_names, n_repeats=200, n_folds=5, tucker_rank=8, max_depth=3, beam_width=100
):
    total = n_repeats * n_folds
    d = X.shape[1]
    log.info("=" * 70)
    log.info(
        "TUCKER FORMULA: %s (N=%d, d=%d) — %d folds, rank=%d",
        name,
        X.shape[0],
        d,
        total,
        tucker_rank,
    )
    log.info("=" * 70)

    cv = RepeatedStratifiedKFold(n_splits=n_folds, n_repeats=n_repeats, random_state=42)
    splits = list(cv.split(X, y))

    modes = {
        "raw": False,
        "tucker": True,
        "raw+tucker": True,
    }

    all_mode_results = {}

    for mode_name, use_tucker in modes.items():
        astar_f1s, astar_train_f1s, formulas = [], [], []
        t0 = time.time()

        for fold_i, (tr, te) in enumerate(splits):
            if use_tucker:
                # Fit Tucker on training data
                tuck_tr, tuck_model = tucker_features(X[tr], rank=tucker_rank)
                tuck_te = tucker_transform(X[te], tuck_model)
                tuck_names = [f"T{i}" for i in range(tuck_tr.shape[1])]

                if mode_name == "raw+tucker":
                    X_aug_tr = np.hstack([X[tr], tuck_tr])
                    X_aug_te = np.hstack([X[te], tuck_te])
                    aug_names = feat_names + tuck_names
                else:
                    X_aug_tr = tuck_tr
                    X_aug_te = tuck_te
                    aug_names = tuck_names
            else:
                X_aug_tr = X[tr]
                X_aug_te = X[te]
                aug_names = feat_names

            X_gpu = cp.asarray(X_aug_tr, dtype=cp.float64)
            y_gpu = cp.asarray(y[tr], dtype=cp.float64)

            train_f1, formula_name, trace = beam_search(
                X_gpu, y_gpu, aug_names, max_depth=max_depth, beam_width=beam_width
            )

            astar_train_f1s.append(train_f1)
            formulas.append(formula_name)

            # Fair test eval
            vtr = trace.ev(X_aug_tr)
            vte = trace.ev(X_aug_te)
            th, dr, _ = opt_thresh(vtr, y[tr])
            preds = (dr * vte >= th).astype(int)
            astar_f1s.append(f1_score(y[te], preds))

            del X_gpu, y_gpu
            cp.get_default_memory_pool().free_all_blocks()

            if (fold_i + 1) % 100 == 0:
                elapsed = time.time() - t0
                rate = (fold_i + 1) / elapsed
                log.info(
                    "  [%s] %d/%d  train=%.3f test=%.3f  %.1f/s",
                    mode_name,
                    fold_i + 1,
                    total,
                    np.mean(astar_train_f1s[-100:]),
                    np.mean(astar_f1s[-100:]),
                    rate,
                )

        astar_f1s = np.array(astar_f1s)
        astar_train_f1s = np.array(astar_train_f1s)
        fc = Counter(formulas)
        top = fc.most_common(1)[0]

        log.info(
            "  [%s] TRAIN=%.4f TEST=%.4f gap=%.4f formula=%s (%.0f%%)",
            mode_name,
            astar_train_f1s.mean(),
            astar_f1s.mean(),
            astar_train_f1s.mean() - astar_f1s.mean(),
            top[0][:40],
            100 * top[1] / len(formulas),
        )

        all_mode_results[mode_name] = {
            "test_f1s": astar_f1s,
            "train_mean": float(astar_train_f1s.mean()),
            "test_mean": float(astar_f1s.mean()),
            "formula": top[0],
            "stability": float(top[1] / len(formulas)),
        }

    # CPU baselines
    t1 = time.time()
    nj = min(os.cpu_count() or 1, 16)
    bls = Parallel(n_jobs=nj, backend="loky", verbose=0)(
        delayed(run_bl)(X, y, tr, te, fi) for fi, (tr, te) in enumerate(splits)
    )
    cpu_time = time.time() - t1

    # Compare
    log.info("")
    log.info(
        "  %-12s %-8s %-8s %-35s %-12s %-12s %-12s",
        "Mode",
        "Train",
        "Test",
        "Formula",
        "vs GB",
        "vs RF",
        "vs LR",
    )
    log.info("  " + "-" * 105)

    results = {}
    for mode_name, mr in all_mode_results.items():
        mode_res = {}
        for bn in ["GB", "RF", "LR"]:
            bf = np.array([r[bn] for r in bls])
            diff = mr["test_f1s"] - bf
            ts, pv = stats.ttest_1samp(diff, 0)
            mode_res[bn] = {
                "mean": float(bf.mean()),
                "diff": float(diff.mean()),
                "sigma": float(abs(ts)),
                "dir": "A*>" if diff.mean() > 0 else f"{bn}>",
            }

        log.info(
            "  %-12s %.4f  %.4f  %-35s %5.1f %s  %5.1f %s  %5.1f %s",
            mode_name,
            mr["train_mean"],
            mr["test_mean"],
            mr["formula"][:35],
            mode_res["GB"]["sigma"],
            mode_res["GB"]["dir"],
            mode_res["RF"]["sigma"],
            mode_res["RF"]["dir"],
            mode_res["LR"]["sigma"],
            mode_res["LR"]["dir"],
        )

        results[mode_name] = {
            "test_f1": mr["test_mean"],
            "formula": mr["formula"],
            "stability": mr["stability"],
            "baselines": mode_res,
        }

    return results


def main():
    log.info("TUCKER-DECOMPOSED FORMULA SEARCH")
    log.info("Feature interactions via Tucker → symbolic formulas")

    props = cp.cuda.runtime.getDeviceProperties(0)
    log.info("GPU: %s (%.1f GB free)", props["name"].decode(), cp.cuda.Device(0).mem_info[0] / 1e9)
    _ = cp.sort(cp.random.rand(10000, 100), axis=0)
    cp.cuda.Stream.null.synchronize()

    datasets = []

    # Datasets where raw formulas LOST to baselines
    bc = load_breast_cancer()
    X = StandardScaler().fit_transform(bc.data)
    datasets.append(("BreastCancer", X, bc.target, [f"f{i}" for i in range(X.shape[1])]))

    try:
        ion = fetch_openml("ionosphere", version=1, as_frame=False, parser="auto")
        X = StandardScaler().fit_transform(ion.data.astype(float))
        y = (ion.target == "g").astype(int)
        datasets.append(("Ionosphere", X, y, [f"i{i}" for i in range(X.shape[1])]))
    except Exception as e:
        log.warning("Ionosphere: %s", e)

    # Also test on a winning dataset (should still win or improve)
    try:
        pima = fetch_openml("diabetes", version=1, as_frame=False, parser="auto")
        X = StandardScaler().fit_transform(pima.data.astype(float))
        y = (pima.target == "tested_positive").astype(int)
        datasets.append(("Diabetes", X, y, [f"d{i}" for i in range(8)]))
    except Exception as e:
        log.warning("Diabetes: %s", e)

    all_results = {}
    for name, X, y, feats in datasets:
        r = run_dataset(
            name, X, y, feats, n_repeats=200, n_folds=5, tucker_rank=8, max_depth=3, beam_width=100
        )
        all_results[name] = r

    with open("tucker_results.json", "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    log.info("\nSaved tucker_results.json")


if __name__ == "__main__":
    main()
