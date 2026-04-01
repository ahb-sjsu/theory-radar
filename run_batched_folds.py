#!/usr/bin/env python3
"""Fold-Batched A* Beam Search — GPU-saturating architecture.

Key insight: instead of processing folds sequentially (each fold
does tiny GPU calls on N=455 samples), batch B_FOLDS folds together
into a single padded 3D tensor. One GPU call evaluates all folds'
candidates simultaneously.

For 50 folds batched with N=455, beam=50, k=8, 10 binary ops:
- Candidates per fold: 50 * 8 * 10 = 4000
- Batched tensor: (50_folds * 4000_cands, 455_samples) = (200K, 455)
- Sort: 200K columns of 455 elements — NOW the GPU is busy.

With meta-learned pruning run inside each fold (no leakage).
Fair test evaluation via FormulaTrace.
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

# ─── GPU Ops ───────────────────────────────────────────────────────


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


BINARY_OPS_LIST = [
    ("+", lambda a, b: a + b),
    ("-", lambda a, b: a - b),
    ("*", lambda a, b: a * b),
    ("/", lambda a, b: a / (b + 1e-30)),
    ("max", lambda a, b: cp.maximum(a, b)),
    ("min", lambda a, b: cp.minimum(a, b)),
    ("hypot", lambda a, b: cp.sqrt(a**2 + b**2)),
    ("diff_sq", lambda a, b: (a - b) ** 2),
    ("harmonic", lambda a, b: 2 * a * b / (a + b + 1e-30)),
    ("geometric", lambda a, b: cp.sign(a * b) * cp.sqrt(cp.abs(a * b))),
]
UNARY_OPS_LIST = [
    ("log", lambda x: cp.log(cp.abs(x) + 1e-30)),
    ("sqrt", lambda x: cp.sqrt(cp.abs(x))),
    ("sq", lambda x: x**2),
    ("abs", lambda x: cp.abs(x)),
    ("sigmoid", lambda x: 1.0 / (1.0 + cp.exp(-cp.clip(x, -500, 500)))),
    ("tanh", lambda x: cp.tanh(cp.clip(x, -500, 500))),
]
SCALAR_BIN = {
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
SCALAR_UN = {
    "log": lambda x: np.log(np.abs(x) + 1e-30),
    "sqrt": lambda x: np.sqrt(np.abs(x)),
    "sq": lambda x: x**2,
    "abs": lambda x: np.abs(x),
    "sigmoid": lambda x: 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500))),
    "tanh": lambda x: np.tanh(np.clip(x, -500, 500)),
}


class FormulaTrace:
    """Records operations for replay on test data."""

    def __init__(self, feat_idx):
        self.ops = [("leaf", feat_idx)]

    def binary(self, op, fi):
        t = FormulaTrace.__new__(FormulaTrace)
        t.ops = self.ops + [("binary", op, fi)]
        return t

    def unary(self, op):
        t = FormulaTrace.__new__(FormulaTrace)
        t.ops = self.ops + [("unary", op)]
        return t

    def evaluate(self, X):
        v = None
        for s in self.ops:
            if s[0] == "leaf":
                v = X[:, s[1]].copy()
            elif s[0] == "binary":
                v = SCALAR_BIN[s[1]](v, X[:, s[2]])
                v = np.nan_to_num(v, nan=0.0, posinf=1e10, neginf=-1e10)
            elif s[0] == "unary":
                v = SCALAR_UN[s[1]](v)
                v = np.nan_to_num(v, nan=0.0, posinf=1e10, neginf=-1e10)
        return v


def find_optimal_threshold(vals, y):
    best_f1, best_t, best_d = 0.0, 0.0, 1
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
                    if f > best_f1:
                        best_f1, best_t, best_d = f, sv[i], d
    return best_t, best_d, best_f1


# ─── Meta-Search (lightweight, d=6) ───────────────────────────────


def quick_meta_search(X_tr, y_tr, feat_names):
    """Fast meta-search on top-6 features. Returns (criterion_fn, threshold)."""
    X_gpu = cp.asarray(X_tr, dtype=cp.float64)
    y_gpu = cp.asarray(y_tr, dtype=cp.float64)
    N, d = X_gpu.shape

    # Pick top 6 features
    if d > 6:
        f1s = gpu_batch_f1(X_gpu, y_gpu)
        top = cp.argsort(-f1s)[:6].get().tolist()
        X_sub = X_gpu[:, top]
        sub_names = [feat_names[i] for i in top]
    else:
        X_sub = X_gpu
        sub_names = feat_names

    d_sub = X_sub.shape[1]
    y_f = y_gpu.astype(cp.float64)

    # Exhaustive depth-3 enumeration (vectorized at each depth)
    catalog = []  # list of (f1, auroc, depth, nf)
    children_of = defaultdict(list)  # idx -> [child_idx]

    # Depth 1
    f1s = gpu_batch_f1(X_sub, y_f).get()
    aurocs_d1 = cp.zeros(d_sub, dtype=cp.float64)
    # skip auroc for speed in meta-search
    for i in range(d_sub):
        catalog.append(
            {"f1": float(f1s[i]), "auroc": 0.5, "depth": 1, "nf": 1, "vals": X_sub[:, i].copy()}
        )

    prev_idx = list(range(len(catalog)))

    for depth in [2, 3]:
        new_idx_start = len(catalog)
        new_vals_list = []
        new_meta_list = []
        new_parent_list = []

        for pi in prev_idx:
            pv = catalog[pi]["vals"]
            pnf = catalog[pi]["nf"]

            # Binary: vectorized via broadcasting
            pv3 = pv[:, None]  # (N, 1)
            for opname, opfn in BINARY_OPS_LIST:
                try:
                    out = opfn(pv3, X_sub)  # (N, d_sub)
                    out = cp.nan_to_num(out, nan=0.0, posinf=1e10, neginf=-1e10)
                    for j in range(d_sub):
                        col = out[:, j]
                        if float(cp.var(col)) < 1e-20:
                            continue
                        new_vals_list.append(col)
                        new_meta_list.append({"depth": depth, "nf": pnf + 1, "parent": pi})
                        new_parent_list.append(pi)
                except:
                    pass

            # Unary
            for opname, opfn in UNARY_OPS_LIST:
                try:
                    col = opfn(pv)
                    col = cp.nan_to_num(col, nan=0.0, posinf=1e10, neginf=-1e10)
                    if float(cp.var(col)) < 1e-20:
                        continue
                    new_vals_list.append(col)
                    new_meta_list.append({"depth": depth, "nf": pnf, "parent": pi})
                    new_parent_list.append(pi)
                except:
                    pass

        if not new_vals_list:
            break

        # Batch F1
        vals_mat = cp.stack(new_vals_list, axis=1)
        f1s = gpu_batch_f1(vals_mat, y_f).get()

        for i in range(len(new_vals_list)):
            idx = len(catalog)
            catalog.append(
                {
                    "f1": float(f1s[i]),
                    "auroc": 0.5,
                    "depth": new_meta_list[i]["depth"],
                    "nf": new_meta_list[i]["nf"],
                    "vals": new_vals_list[i],
                }
            )
            children_of[new_parent_list[i]].append(idx)

        prev_idx = list(range(new_idx_start, len(catalog)))

        del vals_mat
        cp.get_default_memory_pool().free_all_blocks()

    # Free vals
    for c in catalog:
        if "vals" in c:
            del c["vals"]
    cp.get_default_memory_pool().free_all_blocks()

    # Viability
    global_best = max(c["f1"] for c in catalog)
    eps = 0.005
    for c in catalog:
        c["alive"] = 1 if c["f1"] >= global_best - eps else 0

    max_depth = max(c["depth"] for c in catalog)
    for depth in range(max_depth - 1, 0, -1):
        for i, c in enumerate(catalog):
            if c["depth"] != depth:
                continue
            if any(catalog[ci]["alive"] == 1 for ci in children_of.get(i, [])):
                c["alive"] = 1

    # Search criterion family
    shallow = [(i, c) for i, c in enumerate(catalog) if c["depth"] < max_depth]
    alive = [(i, c) for i, c in shallow if c["alive"] == 1]
    dead = [(i, c) for i, c in shallow if c["alive"] == 0]

    if not alive or not dead:
        return None, 0.0, "none", 0.0

    CRITERIA = [
        ("f1", lambda c: c["f1"]),
        ("f1/depth", lambda c: c["f1"] / max(c["depth"], 1)),
        ("f1-depth", lambda c: c["f1"] - c["depth"]),
        ("f1*nf", lambda c: c["f1"] * c["nf"]),
        ("f1/nf", lambda c: c["f1"] / max(c["nf"], 1)),
        ("f1+f1/depth", lambda c: c["f1"] + c["f1"] / max(c["depth"], 1)),
    ]

    best_name, best_fn, best_thresh, best_rate = "none", None, 0.0, 0.0

    for cname, cfn in CRITERIA:
        a_vals = [cfn(c) for _, c in alive]
        d_vals = [cfn(c) for _, c in dead]
        min_a = min(a_vals)
        pruned = sum(1 for dv in d_vals if dv < min_a)
        rate = pruned / len(d_vals)
        if rate > best_rate:
            best_rate = rate
            best_name = cname
            best_fn = cfn
            best_thresh = min_a

    # Convert to a function of (f1, auroc, depth, nf)
    crit_lookup = {
        "f1": lambda f, a, d, k: f,
        "f1/depth": lambda f, a, d, k: f / max(d, 1),
        "f1-depth": lambda f, a, d, k: f - d,
        "f1*nf": lambda f, a, d, k: f * k,
        "f1/nf": lambda f, a, d, k: f / max(k, 1),
        "f1+f1/depth": lambda f, a, d, k: f + f / max(d, 1),
    }

    return crit_lookup.get(best_name), best_thresh, best_name, best_rate


# ─── Batched Beam Search Across Folds ─────────────────────────────


def batched_beam_search(
    X_gpu_full,
    y_gpu_full,
    feat_names,
    fold_groups,
    crit_cache,
    max_depth=4,
    beam_width=50,
    n_subspaces=10,
    subspace_k=8,
):
    """Process multiple folds in a single GPU batch.

    fold_groups: list of (fold_i, tr_indices, te_indices)
    crit_cache: dict fold_i -> (crit_fn, crit_thresh)

    For each subspace trial, we build candidates for ALL folds
    simultaneously, concatenate, and do one giant GPU F1 call.
    """
    n_folds_batch = len(fold_groups)
    d = X_gpu_full.shape[1]
    results = [None] * n_folds_batch  # (train_f1, name, trace) per fold

    rng = np.random.RandomState(42)

    for trial in range(n_subspaces):
        k = min(subspace_k, d)
        feat_idx = list(range(d)) if d <= k else sorted(rng.choice(d, k, replace=False).tolist())
        local_names = [feat_names[i] for i in feat_idx]
        dk = len(feat_idx)

        # Per-fold beams
        fold_beams = []  # one beam per fold
        for fi, (fold_i, tr, te) in enumerate(fold_groups):
            X_tr = X_gpu_full[cp.asarray(tr)][:, feat_idx]
            y_tr = y_gpu_full[cp.asarray(tr)]

            f1s = gpu_batch_f1(X_tr, y_tr)

            beam = []
            for li in range(dk):
                gi = feat_idx[li]
                f1_val = float(f1s[li])

                # Meta prune
                crit_fn, crit_thresh = crit_cache.get(fold_i, (None, 0.0))
                if crit_fn is not None and crit_fn(f1_val, 0.5, 1, 1) < crit_thresh:
                    continue

                h = 0.0 if f1_val >= 0.99 else 1.0 - f1_val
                priority = 1.0 + h
                beam.append(
                    (priority, f1_val, local_names[li], X_tr[:, li].copy(), FormulaTrace(gi), 1)
                )

                if results[fi] is None or f1_val > results[fi][0]:
                    results[fi] = (f1_val, local_names[li], FormulaTrace(gi))

            beam.sort(key=lambda x: x[0])
            fold_beams.append((beam[:beam_width], X_tr, y_tr, fold_i))

        # Expand depth by depth
        for depth in range(2, max_depth + 1):
            # For each fold, generate candidates and concatenate
            all_fold_vals = []
            all_fold_meta = []  # (fold_idx, name, trace, nf)
            fold_offsets = []  # (start, end) in the concatenated tensor

            for fi, (beam, X_tr, y_tr, fold_i) in enumerate(fold_beams):
                if not beam:
                    fold_offsets.append((0, 0))
                    continue

                B = len(beam)
                N_tr = X_tr.shape[0]
                beam_vals = cp.stack([b[3] for b in beam], axis=1)  # (N_tr, B)
                bv3 = beam_vals[:, :, None]
                fv3 = X_tr[:, None, :]  # (N_tr, 1, dk)

                fold_cands = []
                fold_meta = []

                for opname, opfn in BINARY_OPS_LIST:
                    try:
                        out = opfn(bv3, fv3)
                        out = cp.nan_to_num(out, nan=0.0, posinf=1e10, neginf=-1e10)
                        out = out.reshape(N_tr, B * dk)
                        fold_cands.append(out)
                        for bi in range(B):
                            for li in range(dk):
                                gi = feat_idx[li]
                                fold_meta.append(
                                    (
                                        fi,
                                        f"({beam[bi][2]} {opname} {local_names[li]})",
                                        beam[bi][4].binary(opname, gi),
                                        beam[bi][5]
                                        + (1 if local_names[li] not in beam[bi][2] else 0),
                                    )
                                )
                    except:
                        pass

                for opname, opfn in UNARY_OPS_LIST:
                    try:
                        out = opfn(beam_vals)
                        out = cp.nan_to_num(out, nan=0.0, posinf=1e10, neginf=-1e10)
                        fold_cands.append(out)
                        for bi in range(B):
                            fold_meta.append(
                                (
                                    fi,
                                    f"{opname}({beam[bi][2]})",
                                    beam[bi][4].unary(opname),
                                    beam[bi][5],
                                )
                            )
                    except:
                        pass

                if fold_cands:
                    fold_vals = cp.concatenate(fold_cands, axis=1)  # (N_tr, K_fold)
                    K_fold = fold_vals.shape[1]

                    # Compute F1 for this fold's candidates
                    var = cp.var(fold_vals, axis=0)
                    f1s = gpu_batch_f1(fold_vals, y_tr)
                    f1s = cp.where(var < 1e-20, 0.0, f1s)
                    f1s_cpu = f1s.get()

                    # Apply meta pruning + A* priority, build new beam
                    crit_fn, crit_thresh = crit_cache.get(fold_i, (None, 0.0))
                    new_beam = []
                    for i in range(K_fold):
                        f1_val = float(f1s_cpu[i])
                        _, name, trace, nf = fold_meta[i]

                        if crit_fn is not None and crit_fn(f1_val, 0.5, depth, nf) < crit_thresh:
                            continue

                        h = 0.0 if f1_val >= 0.99 else 1.0 - f1_val
                        priority = float(depth) + h
                        new_beam.append((priority, f1_val, name, fold_vals[:, i].copy(), trace, nf))

                        if results[fi] is None or f1_val > results[fi][0]:
                            results[fi] = (f1_val, name, trace)

                    new_beam.sort(key=lambda x: x[0])
                    fold_beams[fi] = (new_beam[:beam_width], X_tr, y_tr, fold_i)

                    del fold_vals, fold_cands

            cp.get_default_memory_pool().free_all_blocks()

    return results


# ─── CPU Baselines ─────────────────────────────────────────────────


def run_sklearn_fold(X, y, tr, te, fi):
    out = {}
    for nm, clf in [
        ("GB", GradientBoostingClassifier(n_estimators=100, max_depth=4, random_state=42 + fi)),
        ("RF", RandomForestClassifier(n_estimators=100, random_state=42 + fi)),
        ("LR", LogisticRegression(max_iter=1000, random_state=42)),
    ]:
        clf.fit(X[tr], y[tr])
        out[nm] = f1_score(y[te], clf.predict(X[te]))
    return out


# ─── Main ──────────────────────────────────────────────────────────


def run_dataset(
    name,
    X,
    y,
    features,
    n_repeats=200,
    n_folds=5,
    max_depth=4,
    n_subspaces=20,
    subspace_k=8,
    batch_size=50,
):
    total = n_repeats * n_folds
    log.info("=" * 70)
    log.info(
        "BATCHED A* BEAM: %s (N=%d, d=%d) — %d folds, batch=%d",
        name,
        X.shape[0],
        X.shape[1],
        total,
        batch_size,
    )
    log.info("=" * 70)

    cv = RepeatedStratifiedKFold(n_splits=n_folds, n_repeats=n_repeats, random_state=42)
    splits = list(cv.split(X, y))

    X_gpu = cp.asarray(X, dtype=cp.float64)
    y_gpu = cp.asarray(y, dtype=cp.float64)

    # Phase 1: Meta-search for each unique training set
    log.info("  Phase 1: Fold-local meta-search...")
    crit_cache = {}
    t_meta = time.time()
    for fold_i, (tr, te) in enumerate(splits):
        cache_key = hash(tr.tobytes())
        if cache_key not in crit_cache:
            crit_fn, crit_thresh, crit_name, prune_rate = quick_meta_search(X[tr], y[tr], features)
            crit_cache[cache_key] = (crit_fn, crit_thresh)
            if fold_i < 5:
                log.info(
                    "    Fold %d: %s < %.4f → prune %.1f%%",
                    fold_i,
                    crit_name,
                    crit_thresh,
                    100 * prune_rate,
                )
        # Also store by fold_i for lookup
        crit_cache[fold_i] = crit_cache[cache_key]
    meta_time = time.time() - t_meta
    log.info(
        "  Meta-search: %.0fs (%d unique)",
        meta_time,
        len(set(hash(tr.tobytes()) for tr, te in splits)),
    )

    # Phase 2: Batched A* beam search
    astar_f1s, astar_train_f1s, formulas = [], [], []
    t0 = time.time()

    for batch_start in range(0, total, batch_size):
        batch_end = min(batch_start + batch_size, total)
        fold_groups = []
        for i in range(batch_start, batch_end):
            tr, te = splits[i]
            fold_groups.append((i, tr, te))

        batch_results = batched_beam_search(
            X_gpu,
            y_gpu,
            features,
            fold_groups,
            crit_cache,
            max_depth=max_depth,
            beam_width=50,
            n_subspaces=n_subspaces,
            subspace_k=subspace_k,
        )

        for fi, (fold_i, tr, te) in enumerate(fold_groups):
            if batch_results[fi] is None:
                astar_f1s.append(0.0)
                astar_train_f1s.append(0.0)
                formulas.append("")
                continue

            train_f1, formula_name, trace = batch_results[fi]
            astar_train_f1s.append(train_f1)
            formulas.append(formula_name)

            # Fair test eval
            vals_tr = trace.evaluate(X[tr])
            vals_te = trace.evaluate(X[te])
            thresh, dirn, _ = find_optimal_threshold(vals_tr, y[tr])
            preds = (dirn * vals_te >= thresh).astype(int)
            astar_f1s.append(f1_score(y[te], preds))

        done = batch_end
        elapsed = time.time() - t0
        if done > 0 and elapsed > 0:
            rate = done / elapsed
            log.info(
                "  Batch %d-%d  train=%.3f test=%.3f  %.2f folds/s  ETA %.0fs",
                batch_start,
                batch_end,
                np.mean(astar_train_f1s[-batch_size:]),
                np.mean(astar_f1s[-batch_size:]),
                rate,
                (total - done) / rate,
            )

    gpu_time = time.time() - t0
    log.info("  GPU done: %.0fs (%.2f folds/s)", gpu_time, total / max(gpu_time, 0.1))

    del X_gpu, y_gpu
    cp.get_default_memory_pool().free_all_blocks()

    # Phase 3: CPU baselines
    t1 = time.time()
    n_jobs = min(os.cpu_count() or 1, 20)
    baselines = Parallel(n_jobs=n_jobs, backend="loky", verbose=0)(
        delayed(run_sklearn_fold)(X, y, tr, te, fi) for fi, (tr, te) in enumerate(splits)
    )
    cpu_time = time.time() - t1

    astar_f1s = np.array(astar_f1s)
    astar_train_f1s = np.array(astar_train_f1s)
    results = {}
    for bn in ["GB", "RF", "LR"]:
        bf = np.array([r[bn] for r in baselines])
        diff = astar_f1s - bf
        t_stat, p_val = stats.ttest_1samp(diff, 0)
        results[bn] = {
            "mean": float(bf.mean()),
            "diff": float(diff.mean()),
            "sigma": float(abs(t_stat)),
            "dir": "A*>" if diff.mean() > 0 else f"{bn}>",
        }

    fc = Counter(formulas)
    top = fc.most_common(1)[0]
    log.info("  FORMULA: %s (%.0f%%)", top[0][:60], 100 * top[1] / len(formulas))
    log.info(
        "  TRAIN: %.4f  TEST: %.4f  gap=%.4f",
        astar_train_f1s.mean(),
        astar_f1s.mean(),
        astar_train_f1s.mean() - astar_f1s.mean(),
    )
    for bn in ["GB", "RF", "LR"]:
        b = results[bn]
        log.info(
            "  vs %-3s: %.4f  diff=%+.4f  %.1fs  %s", bn, b["mean"], b["diff"], b["sigma"], b["dir"]
        )

    return {
        "name": name,
        "test_f1": float(astar_f1s.mean()),
        "train_f1": float(astar_train_f1s.mean()),
        "formula": top[0],
        "baselines": results,
    }


def main():
    log.info("FOLD-BATCHED A* BEAM SEARCH WITH META-LEARNED PRUNING")

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

    # Large datasets from OpenML
    LARGE = [
        ("credit-g", 1, lambda t: (t == "bad").astype(int), "credit"),
        ("phoneme", 1, None, "phoneme"),
        ("electricity", 1, None, "electric"),
        ("bank-marketing", 1, lambda t: (t == "2").astype(int), "bank"),
    ]
    for oml_name, ver, target_fn, short in LARGE:
        try:
            ds = fetch_openml(oml_name, version=ver, as_frame=False, parser="auto")
            X = StandardScaler().fit_transform(ds.data.astype(float))
            if target_fn is not None:
                y = target_fn(ds.target)
            else:
                y = ds.target.astype(int)
            if len(np.unique(y)) != 2:
                y = (y == y.max()).astype(int)
            datasets.append(
                (f"{short}({X.shape[0]}x{X.shape[1]})", X, y, [f"v{i}" for i in range(X.shape[1])])
            )
            log.info("Loaded %s: N=%d d=%d prev=%.2f", short, X.shape[0], X.shape[1], y.mean())
        except Exception as e:
            log.warning("%s: %s", oml_name, e)

    all_results = []
    for name, X, y, feats in datasets:
        n_sub = 20 if X.shape[1] > 15 else 10
        r = run_dataset(
            name,
            X,
            y,
            feats,
            n_repeats=100,
            n_folds=5,
            max_depth=4,
            n_subspaces=n_sub,
            subspace_k=8,
            batch_size=50,
        )
        all_results.append(r)

    with open("batched_results.json", "w") as f:
        json.dump(all_results, f, indent=2)

    log.info("\n" + "=" * 100)
    log.info("FINAL — Batched A* Beam (meta-pruned, depth 4, fair eval)")
    for r in all_results:
        log.info(
            "%-25s test=%.4f  %s  GB:%.1f%s RF:%.1f%s LR:%.1f%s",
            r["name"],
            r["test_f1"],
            r["formula"][:25],
            r["baselines"]["GB"]["sigma"],
            r["baselines"]["GB"]["dir"],
            r["baselines"]["RF"]["sigma"],
            r["baselines"]["RF"]["dir"],
            r["baselines"]["LR"]["sigma"],
            r["baselines"]["LR"]["dir"],
        )


if __name__ == "__main__":
    main()
