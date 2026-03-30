#!/usr/bin/env python3
"""Full Theory Radar Pipeline:
  PCA projections + A* beam search + meta-learned pruning + fair eval.

1. PCA: project all d features into k components (access ALL features)
2. Meta-search: learn pruning criterion inside each training fold
3. A* beam: priority=depth+h, h=0 goals, h=1-F1 alive, h=inf dead
4. Fuzzed subspaces over raw+PCA features
5. Fair test eval via FormulaTrace
"""

import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import cupy as cp
import numpy as np
import logging
import time
import json
from collections import Counter, defaultdict
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


# ─── Quick meta-search (inside fold) ──────────────────────────────


def quick_meta(X_tr, y_tr, feat_names):
    """Learn pruning criterion from exhaustive depth-2 micro-search on top-6 features."""
    Xg = cp.asarray(X_tr, dtype=cp.float64)
    yg = cp.asarray(y_tr, dtype=cp.float64)
    d = Xg.shape[1]
    if d > 6:
        f1s = gpu_batch_f1(Xg, yg)
        top = cp.argsort(-f1s)[:6].get().tolist()
        Xs = Xg[:, top]
    else:
        Xs = Xg
        top = list(range(d))
    ds = Xs.shape[1]
    yf = yg

    # Depth 1
    f1d1 = gpu_batch_f1(Xs, yf).get()
    cat = [{"f1": float(f1d1[i]), "d": 1, "nf": 1, "vals": Xs[:, i].copy()} for i in range(ds)]

    # Depth 2 (vectorized)
    d2_vals = []
    d2_meta = []
    for pi in range(ds):
        pv = cat[pi]["vals"][:, None]
        for opn, opf in VBIN:
            try:
                out = opf(pv, Xs)
                out = cp.nan_to_num(out, nan=0.0, posinf=1e10, neginf=-1e10)
                for j in range(ds):
                    if float(cp.var(out[:, j])) < 1e-20:
                        continue
                    d2_vals.append(out[:, j])
                    d2_meta.append({"d": 2, "nf": 2, "parent": pi})
            except:
                pass

    if d2_vals:
        vm = cp.stack(d2_vals, axis=1)
        f1s2 = gpu_batch_f1(vm, yf).get()
        for i in range(len(d2_vals)):
            cat.append(
                {
                    "f1": float(f1s2[i]),
                    "d": 2,
                    "nf": d2_meta[i]["nf"],
                    "parent": d2_meta[i]["parent"],
                }
            )
        del vm

    del Xg, yg
    cp.get_default_memory_pool().free_all_blocks()
    for c in cat:
        if "vals" in c:
            del c["vals"]

    # Viability
    gb = max(c["f1"] for c in cat)
    for c in cat:
        c["alive"] = 1 if c["f1"] >= gb - 0.005 else 0
    # Propagate
    children = defaultdict(list)
    for i, c in enumerate(cat):
        if "parent" in c:
            children[c["parent"]].append(i)
    for i, c in enumerate(cat):
        if c["d"] == 1 and any(cat[ci]["alive"] == 1 for ci in children.get(i, [])):
            c["alive"] = 1

    shallow = [c for c in cat if c["d"] == 1]
    alive = [c for c in shallow if c["alive"] == 1]
    dead = [c for c in shallow if c["alive"] == 0]
    if not alive or not dead:
        return None, 0.0

    CRIT = [
        ("f1", lambda c: c["f1"]),
        ("f1/d", lambda c: c["f1"] / max(c["d"], 1)),
    ]
    best_name, best_th, best_r = "f1/d", 0.0, 0.0
    for cn, cf in CRIT:
        av = [cf(c) for c in alive]
        dv = [cf(c) for c in dead]
        ma = min(av)
        pr = sum(1 for x in dv if x < ma) / len(dv)
        if pr > best_r:
            best_r = pr
            best_th = ma
            best_name = cn

    # Return a function of (f1, auroc, depth, nf) — matching caller signature
    CRIT_FNS = {
        "f1": lambda f, a, d, k: f,
        "f1/d": lambda f, a, d, k: f / max(d, 1),
    }
    return CRIT_FNS[best_name], best_th


# ─── A* Beam Search with PCA + meta pruning ───────────────────────


def astar_pca_beam(
    X_aug_gpu, y_gpu, feat_names, crit_fn, crit_th, feat_idx=None, max_depth=3, beam_width=100
):
    """A* beam search on augmented (raw+PCA) features with meta pruning."""
    N, d_full = X_aug_gpu.shape
    if feat_idx is not None:
        Xs = X_aug_gpu[:, feat_idx]
        ln = [feat_names[i] for i in feat_idx]
    else:
        Xs = X_aug_gpu
        ln = feat_names
        feat_idx = list(range(d_full))
    d = Xs.shape[1]
    yf = y_gpu.astype(cp.float64)

    # Depth 1
    f1s = gpu_batch_f1(Xs, yf)
    order = cp.argsort(-f1s).get()
    B = min(beam_width, d)

    beam = []
    # Always keep the best raw feature as fallback (never pruned)
    gi_best = feat_idx[order[0]]
    best_f1, best_nm, best_tr = float(f1s[order[0]]), ln[order[0]], FT(gi_best)
    for rank, li in enumerate(order[:B]):
        gi = feat_idx[li]
        fv = float(f1s[li])
        if crit_fn and crit_fn(fv, 0.5, 1, 1) < crit_th:
            continue
        h = 0.0 if fv >= 0.99 else 1.0 - fv
        beam.append((1.0 + h, fv, ln[li], Xs[:, li].copy(), FT(gi), 1))
        if fv > best_f1:
            best_f1, best_nm, best_tr = fv, ln[li], FT(gi)
    beam.sort(key=lambda x: x[0])
    beam = beam[:beam_width]

    for depth in range(2, max_depth + 1):
        if not beam:
            break
        Bc = len(beam)
        bv = cp.stack([b[3] for b in beam], axis=1)
        bv3 = bv[:, :, None]
        fv3 = Xs[:, None, :]

        chunks, meta = [], []
        for on, of in VBIN:
            try:
                out = of(bv3, fv3).reshape(N, Bc * d)
                out = cp.nan_to_num(out, nan=0.0, posinf=1e10, neginf=-1e10)
                chunks.append(out)
                for bi in range(Bc):
                    for li in range(d):
                        gi = feat_idx[li]
                        meta.append(
                            (
                                f"({beam[bi][2]} {on} {ln[li]})",
                                beam[bi][4].b(on, gi),
                                beam[bi][5] + (1 if ln[li] not in beam[bi][2] else 0),
                            )
                        )
            except:
                pass
        for on, of in VUN:
            try:
                out = of(bv)
                out = cp.nan_to_num(out, nan=0.0, posinf=1e10, neginf=-1e10)
                chunks.append(out)
                for bi in range(Bc):
                    meta.append((f"{on}({beam[bi][2]})", beam[bi][4].u(on), beam[bi][5]))
            except:
                pass

        if not chunks:
            break
        av = cp.concatenate(chunks, axis=1)
        K = av.shape[1]
        var = cp.var(av, axis=0)
        f1s = gpu_batch_f1(av, yf)
        f1s = cp.where(var < 1e-20, 0.0, f1s)
        f1c = f1s.get()

        nb = []
        for i in range(K):
            fv = float(f1c[i])
            nm, tr, nf = meta[i]
            if crit_fn and crit_fn(fv, 0.5, depth, nf) < crit_th:
                continue
            h = 0.0 if fv >= 0.99 else 1.0 - fv
            nb.append((float(depth) + h, fv, nm, av[:, i].copy(), tr, nf))
            if fv > best_f1:
                best_f1, best_nm, best_tr = fv, nm, tr
        nb.sort(key=lambda x: x[0])
        beam = nb[:beam_width]
        del av, chunks
        cp.get_default_memory_pool().free_all_blocks()

    return best_f1, best_nm, best_tr


# ─── CPU baselines ─────────────────────────────────────────────────


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


# ─── Main pipeline ─────────────────────────────────────────────────


def run_dataset(
    name,
    X,
    y,
    feat_names,
    n_repeats=200,
    n_folds=5,
    n_proj=8,
    max_depth=3,
    beam_width=100,
    n_subspaces=15,
    subspace_k=10,
):
    total = n_repeats * n_folds
    d = X.shape[1]
    log.info("=" * 70)
    log.info(
        "FULL PIPELINE: %s (N=%d, d=%d) — %d folds, %d PCs, depth=%d, %d subspaces",
        name,
        X.shape[0],
        d,
        total,
        n_pca,
        max_depth,
        n_subspaces,
    )
    log.info("=" * 70)

    cv = RepeatedStratifiedKFold(n_splits=n_folds, n_repeats=n_repeats, random_state=42)
    splits = list(cv.split(X, y))

    # Sequential fold loop — GPU beam search is already vectorized.
    # CPU parallelism handled by joblib for baselines at the end.
    astar_f1s, astar_train_f1s, formulas = [], [], []
    meta_cache = {}
    t0 = time.time()

    for fold_i, (tr, te) in enumerate(splits):
        k = min(n_pca, d, X[tr].shape[0] - 1)
        from sklearn.cross_decomposition import PLSRegression

        pls = PLSRegression(n_components=k)
        pls.fit(X[tr], y[tr])
        pls_tr = pls.transform(X[tr])
        pls_te = pls.transform(X[te])

        X_aug_tr = np.hstack([X[tr], pls_tr])
        X_aug_te = np.hstack([X[te], pls_te])
        aug_names = feat_names + [f"pls{i}" for i in range(k)]
        d_aug = len(aug_names)

        # Meta-search (cached by fold hash)
        ck = hash(tr.tobytes())
        if ck not in meta_cache:
            cf, ct = quick_meta(X_aug_tr, y[tr], aug_names)
            meta_cache[ck] = (cf, ct)
        cf, ct = meta_cache[ck]

        # GPU beam search
        X_gpu = cp.asarray(X_aug_tr, dtype=cp.float64)
        y_gpu = cp.asarray(y[tr], dtype=cp.float64)

        rng = np.random.RandomState(42 + fold_i)
        best_f1, best_nm, best_tr_obj = 0.0, "", None

        for trial in range(n_subspaces):
            sk = min(subspace_k, d_aug)
            fi = (
                list(range(d_aug))
                if d_aug <= sk
                else sorted(rng.choice(d_aug, sk, replace=False).tolist())
            )
            f1, nm, tr_obj = astar_pca_beam(
                X_gpu,
                y_gpu,
                aug_names,
                cf,
                ct,
                feat_idx=fi,
                max_depth=max_depth,
                beam_width=beam_width,
            )
            if f1 > best_f1:
                best_f1, best_nm, best_tr_obj = f1, nm, tr_obj

        del X_gpu, y_gpu
        cp.get_default_memory_pool().free_all_blocks()

        astar_train_f1s.append(best_f1)
        formulas.append(best_nm)

        # Fair test eval
        vtr = best_tr_obj.ev(X_aug_tr)
        vte = best_tr_obj.ev(X_aug_te)
        th, dr, _ = opt_thresh(vtr, y[tr])
        preds = (dr * vte >= th).astype(int)
        astar_f1s.append(f1_score(y[te], preds))

        if (fold_i + 1) % 50 == 0:
            elapsed = time.time() - t0
            rate = (fold_i + 1) / elapsed
            log.info(
                "  %d/%d  train=%.3f test=%.3f  %.2f/s  ETA %.0fs",
                fold_i + 1,
                total,
                np.mean(astar_train_f1s[-50:]),
                np.mean(astar_f1s[-50:]),
                rate,
                (total - fold_i - 1) / rate,
            )

    gpu_time = time.time() - t0
    log.info("  Search: %.0fs (%.2f folds/s)", gpu_time, total / gpu_time)

    # CPU baselines (on raw features, not augmented)
    t1 = time.time()
    nj = min(os.cpu_count() or 1, 20)
    bls = Parallel(n_jobs=nj, backend="loky", verbose=0)(
        delayed(run_bl)(X, y, tr, te, fi) for fi, (tr, te) in enumerate(splits)
    )
    cpu_time = time.time() - t1

    af = np.array(astar_f1s)
    at = np.array(astar_train_f1s)
    res = {}
    for bn in ["GB", "RF", "LR"]:
        bf = np.array([r[bn] for r in bls])
        diff = af - bf
        ts, pv = stats.ttest_1samp(diff, 0)
        res[bn] = {
            "mean": float(bf.mean()),
            "diff": float(diff.mean()),
            "sigma": float(abs(ts)),
            "dir": "A*>" if diff.mean() > 0 else f"{bn}>",
        }

    fc = Counter(formulas)
    top = fc.most_common(1)[0]
    log.info("  FORMULA: %s (%.0f%%)", top[0][:60], 100 * top[1] / len(formulas))
    log.info("  TRAIN: %.4f  TEST: %.4f  gap=%.4f", at.mean(), af.mean(), at.mean() - af.mean())
    for bn in ["GB", "RF", "LR"]:
        b = res[bn]
        log.info(
            "  vs %-3s: %.4f  diff=%+.4f  %.1fs  %s", bn, b["mean"], b["diff"], b["sigma"], b["dir"]
        )

    return {
        "name": name,
        "test_f1": float(af.mean()),
        "train_f1": float(at.mean()),
        "formula": top[0],
        "stability": float(top[1] / len(formulas)),
        "baselines": res,
    }


def main():
    log.info("FULL THEORY RADAR: PCA + A* Beam + Meta Pruning + Fuzz")

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
        bn = fetch_openml("banknote-authentication", version=1, as_frame=False, parser="auto")
        X = StandardScaler().fit_transform(bn.data.astype(float))
        yr = bn.target.astype(int)
        y = (yr == yr.max()).astype(int)
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

    all_res = []
    for name, X, y, feats in datasets:
        d = X.shape[1]
        ns = 20 if d > 15 else 10
        sk = min(12, d + 8)  # raw+pca features
        r = run_dataset(
            name,
            X,
            y,
            feats,
            n_repeats=200,
            n_folds=5,
            n_proj=8,
            max_depth=3,
            beam_width=100,
            n_subspaces=ns,
            subspace_k=sk,
        )
        all_res.append(r)

    with open("full_pipeline_results.json", "w") as f:
        json.dump(all_res, f, indent=2)

    log.info("\n" + "=" * 100)
    log.info("FULL PIPELINE: PCA + A* Beam + Meta Pruning (fair eval)")
    log.info(
        "%-15s %-8s %-8s %-30s %-10s %-10s %-10s",
        "Dataset",
        "Train",
        "Test",
        "Formula",
        "vs GB",
        "vs RF",
        "vs LR",
    )
    log.info("-" * 100)
    for r in all_res:
        log.info(
            "%-15s %.4f  %.4f  %-30s %6.1f%s %s  %6.1f%s %s  %6.1f%s %s",
            r["name"],
            r["train_f1"],
            r["test_f1"],
            r["formula"][:30],
            r["baselines"]["GB"]["sigma"],
            "s",
            r["baselines"]["GB"]["dir"],
            r["baselines"]["RF"]["sigma"],
            "s",
            r["baselines"]["RF"]["dir"],
            r["baselines"]["LR"]["sigma"],
            "s",
            r["baselines"]["LR"]["dir"],
        )


if __name__ == "__main__":
    main()
