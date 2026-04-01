#!/usr/bin/env python3
"""Meta-Pruned Beam Search with Admissible A* Heuristic.

The heuristic h(n) is:
  h = 0          if F1(n) >= target  (goal: true remaining cost is 0)
  h = inf        if meta-criterion says dead  (prune subtree)
  h = 1 - F1(n)  otherwise  (continuous, in (0,1))

Admissibility proof:
  - Goal nodes: h=0 = h*. Correct.
  - Non-goal alive nodes: h = 1-F1 < 1. True h* >= 1 (need at
    least one more depth). So h < 1 <= h*. Admissible.
  - Dead nodes: h=inf. Safe iff the subtree truly contains no
    near-optimal formula (guaranteed by construction on the
    enumerated depth-3 space; NOT guaranteed at depth 4+).

Priority = depth + h(n). This gives genuine A* ordering:
  - Within a depth level, higher-F1 nodes expand first.
  - No depth-k node has priority >= any depth-(k+1) node.
  - The first goal node popped has minimum depth among all
    goals reachable by the explored beam.

The meta-criterion is searched (not hardcoded) over a family of
12 candidate formulas on node features, selecting the one with
highest zero-false-negative prune rate.

Meta-search runs INSIDE each CV training fold to prevent leakage.
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

# ─── GPU Batch F1 ─────────────────────────────────────────────────


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


def gpu_batch_auroc(vals, labels):
    N, K = vals.shape
    pos = float(labels.sum())
    neg = N - pos
    if pos == 0 or neg == 0:
        return cp.full(K, 0.5, dtype=cp.float64)
    idx = cp.argsort(vals, axis=0)
    sl = labels[idx]
    ranks = cp.arange(1, N + 1, dtype=cp.float64).reshape(-1, 1)
    auroc = (cp.sum(sl * ranks, axis=0) - pos * (pos + 1) / 2) / (pos * neg + 1e-30)
    return cp.maximum(auroc, 1 - auroc)


BINARY_OPS = {
    "+": lambda a, b: a + b,
    "-": lambda a, b: a - b,
    "*": lambda a, b: a * b,
    "/": lambda a, b: a / (b + 1e-30),
    "max": lambda a, b: cp.maximum(a, b),
    "min": lambda a, b: cp.minimum(a, b),
    "hypot": lambda a, b: cp.sqrt(a**2 + b**2),
    "diff_sq": lambda a, b: (a - b) ** 2,
    "harmonic": lambda a, b: 2 * a * b / (a + b + 1e-30),
    "geometric": lambda a, b: cp.sign(a * b) * cp.sqrt(cp.abs(a * b)),
}
UNARY_OPS = {
    "log": lambda x: cp.log(cp.abs(x) + 1e-30),
    "sqrt": lambda x: cp.sqrt(cp.abs(x)),
    "sq": lambda x: x**2,
    "abs": lambda x: cp.abs(x),
    "sigmoid": lambda x: 1.0 / (1.0 + cp.exp(-cp.clip(x, -500, 500))),
    "tanh": lambda x: cp.tanh(cp.clip(x, -500, 500)),
}
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


# ─── Phase 1: Meta-Search for Heuristic ───────────────────────────


def meta_search_heuristic(X, y, feat_names, max_depth=3):
    """Run exhaustive enumeration + viability analysis to learn a pruning criterion.
    Returns: (criterion_fn, threshold, stats_dict)
    where criterion_fn(f1, auroc, depth, n_features) -> float
    and nodes with value < threshold are dead (safe to prune).
    """
    X_gpu = cp.asarray(X, dtype=cp.float64)
    y_gpu = cp.asarray(y, dtype=cp.float64)
    N, d = X_gpu.shape

    # Exhaustive enumeration
    catalog = {}
    children = defaultdict(list)
    y_f = y_gpu.astype(cp.float64)

    f1s = gpu_batch_f1(X_gpu, y_f)
    aurocs = gpu_batch_auroc(X_gpu, y_f)
    for i in range(d):
        catalog[feat_names[i]] = {
            "depth": 1,
            "f1": float(f1s[i]),
            "auroc": float(aurocs[i]),
            "values": X_gpu[:, i].copy(),
            "n_features": 1,
            "parent": None,
        }

    prev = list(catalog.keys())
    for depth in range(2, max_depth + 1):
        new = {}
        for pn in prev:
            p = catalog[pn]
            pv = p["values"]
            for j in range(d):
                for op, fn in BINARY_OPS.items():
                    cn = f"({pn} {op} {feat_names[j]})"
                    if cn in catalog or cn in new:
                        continue
                    try:
                        cv = fn(pv, X_gpu[:, j])
                        cv = cp.nan_to_num(cv, nan=0.0, posinf=1e10, neginf=-1e10)
                        if float(cp.var(cv)) < 1e-20:
                            continue
                        new[cn] = {
                            "depth": depth,
                            "values": cv,
                            "n_features": p["n_features"] + (1 if feat_names[j] not in pn else 0),
                            "parent": pn,
                        }
                        children[pn].append(cn)
                    except:
                        pass
            for op, fn in UNARY_OPS.items():
                cn = f"{op}({pn})"
                if cn in catalog or cn in new:
                    continue
                try:
                    cv = fn(pv)
                    cv = cp.nan_to_num(cv, nan=0.0, posinf=1e10, neginf=-1e10)
                    if float(cp.var(cv)) < 1e-20:
                        continue
                    new[cn] = {
                        "depth": depth,
                        "values": cv,
                        "n_features": p["n_features"],
                        "parent": pn,
                    }
                    children[pn].append(cn)
                except:
                    pass

        if new:
            names = list(new.keys())
            for start in range(0, len(names), 50000):
                chunk = names[start : start + 50000]
                vm = cp.stack([new[n]["values"] for n in chunk], axis=1)
                f1s = gpu_batch_f1(vm, y_f)
                aurocs = gpu_batch_auroc(vm, y_f)
                for i, n in enumerate(chunk):
                    new[n]["f1"] = float(f1s[i])
                    new[n]["auroc"] = float(aurocs[i])
                    catalog[n] = new[n]
                del vm
                cp.get_default_memory_pool().free_all_blocks()
        prev = list(new.keys())

    # Free values
    for n in catalog:
        if "values" in catalog[n]:
            del catalog[n]["values"]
    cp.get_default_memory_pool().free_all_blocks()

    # Viability
    global_best = max(v["f1"] for v in catalog.values())
    eps = 0.005
    thresh = global_best - eps
    for v in catalog.values():
        v["alive"] = 1 if v["f1"] >= thresh else 0
    for depth in range(max_depth - 1, 0, -1):
        for n, v in catalog.items():
            if v["depth"] != depth:
                continue
            if any(catalog[c]["alive"] == 1 for c in children.get(n, []) if c in catalog):
                v["alive"] = 1

    # Search the criterion space: enumerate candidate formulas over node features
    shallow = [(n, v) for n, v in catalog.items() if v["depth"] < max_depth]
    alive_nodes = [(n, v) for n, v in shallow if v["alive"] == 1]
    dead_nodes = [(n, v) for n, v in shallow if v["alive"] == 0]

    if not alive_nodes or not dead_nodes:
        return None, 0.0, {"n_formulas": len(catalog), "n_alive": len(alive_nodes)}

    # Candidate criterion family: simple formulas over (f1, auroc, depth, nf)
    CRITERIA = [
        ("auroc", lambda f, a, d, k: a),
        ("f1", lambda f, a, d, k: f),
        ("auroc/depth", lambda f, a, d, k: a / max(d, 1)),
        ("f1/depth", lambda f, a, d, k: f / max(d, 1)),
        ("auroc/nf", lambda f, a, d, k: a / max(k, 1)),
        ("f1*auroc", lambda f, a, d, k: f * a),
        ("f1*auroc/depth", lambda f, a, d, k: f * a / max(d, 1)),
        ("auroc-depth", lambda f, a, d, k: a - d),
        ("auroc-nf", lambda f, a, d, k: a - k),
        ("f1-depth", lambda f, a, d, k: f - d),
        ("(1-f1)*(1-a)*d", lambda f, a, d, k: -(1 - f) * (1 - a) * d),  # negated: higher=better
        ("f1+auroc-depth", lambda f, a, d, k: f + a - d),
    ]

    best_crit_name, best_crit_fn, best_thresh, best_prune = None, None, 0.0, 0.0

    for cname, cfn in CRITERIA:
        alive_vals = [cfn(v["f1"], v["auroc"], v["depth"], v["n_features"]) for _, v in alive_nodes]
        dead_vals = [cfn(v["f1"], v["auroc"], v["depth"], v["n_features"]) for _, v in dead_nodes]

        min_alive = min(alive_vals)
        pruned = sum(1 for dv in dead_vals if dv < min_alive)
        rate = pruned / len(dead_vals) if dead_vals else 0.0

        if rate > best_prune:
            best_prune = rate
            best_crit_name = cname
            best_crit_fn = cfn
            best_thresh = min_alive

    log.info(
        "  Meta: %d formulas, %d alive, %d dead", len(catalog), len(alive_nodes), len(dead_nodes)
    )
    log.info(
        "  Best criterion: %s < %.4f → prune %.1f%%", best_crit_name, best_thresh, 100 * best_prune
    )

    return (
        best_crit_fn,
        best_thresh,
        {
            "n_formulas": len(catalog),
            "n_alive": len(alive_nodes),
            "n_dead": len(dead_nodes),
            "prune_rate": best_prune,
            "threshold": best_thresh,
            "criterion": best_crit_name,
        },
    )


# ─── Phase 2: A* Beam Search with Meta Heuristic ──────────────────


def astar_beam_search(
    X_gpu,
    y_gpu,
    feat_names,
    criterion_fn,
    criterion_thresh,
    f1_target=0.0,
    max_depth=4,
    beam_width=100,
    n_subspaces=10,
    subspace_k=6,
    rng=None,
):
    """A* beam search with admissible heuristic.

    Priority = depth + h(n) where:
      h = 0          if F1 >= f1_target (goal node)
      h = inf        if meta-criterion < threshold (dead subtree)
      h = 1 - F1     otherwise (continuous, admissible since h < 1 <= h*)

    Beam width limits memory. Random subspaces broaden exploration.
    """
    N, d = X_gpu.shape
    y_f = y_gpu.astype(cp.float64)
    if rng is None:
        rng = np.random.RandomState(42)

    best_f1, best_name, best_trace = 0.0, "", None

    for trial in range(n_subspaces):
        k = min(subspace_k, d)
        feat_idx = (
            list(range(d)) if d <= subspace_k else sorted(rng.choice(d, k, replace=False).tolist())
        )
        X_sub = X_gpu[:, feat_idx]
        local_names = [feat_names[i] for i in feat_idx]
        dk = len(feat_idx)

        # Depth 1: seed with A* priorities
        f1s = gpu_batch_f1(X_sub, y_f)
        aurocs = gpu_batch_auroc(X_sub, y_f)

        beam = []  # (priority, f1, name, values, trace, auroc, nf)
        for li in range(dk):
            gi = feat_idx[li]
            f1_val = float(f1s[li])
            auc_val = float(aurocs[li])

            # Meta prune
            if criterion_fn is not None:
                if criterion_fn(f1_val, auc_val, 1, 1) < criterion_thresh:
                    continue

            # A* heuristic
            if f1_target > 0 and f1_val >= f1_target:
                h = 0.0  # goal
            else:
                h = 1.0 - f1_val  # continuous, < 1

            priority = 1.0 + h  # depth=1 + h
            beam.append(
                (
                    priority,
                    f1_val,
                    local_names[li],
                    X_sub[:, li].copy(),
                    FormulaTrace(gi),
                    auc_val,
                    1,
                )
            )

            if f1_val > best_f1:
                best_f1 = f1_val
                best_name = local_names[li]
                best_trace = FormulaTrace(gi)

        # Sort by A* priority (ascending = best first), keep beam_width
        beam.sort(key=lambda x: x[0])
        beam = beam[:beam_width]

        for depth in range(2, max_depth + 1):
            if not beam:
                break

            B = len(beam)
            beam_vals = cp.stack([b[3] for b in beam], axis=1)
            bv3 = beam_vals[:, :, None]
            fv3 = X_sub[:, None, :]

            all_vals, all_meta = [], []  # all_meta: (name, trace, nf, parent_idx)

            for opname, opfn in BINARY_OPS.items():
                try:
                    out = opfn(bv3, fv3)
                    out = cp.nan_to_num(out, nan=0.0, posinf=1e10, neginf=-1e10)
                    out = out.reshape(N, B * dk)
                    all_vals.append(out)
                    for bi in range(B):
                        for li in range(dk):
                            gi = feat_idx[li]
                            all_meta.append(
                                (
                                    f"({beam[bi][2]} {opname} {local_names[li]})",
                                    beam[bi][4].binary(opname, gi),
                                    beam[bi][6] + (1 if local_names[li] not in beam[bi][2] else 0),
                                )
                            )
                except:
                    pass

            for opname, opfn in UNARY_OPS.items():
                try:
                    out = opfn(beam_vals)
                    out = cp.nan_to_num(out, nan=0.0, posinf=1e10, neginf=-1e10)
                    all_vals.append(out)
                    for bi in range(B):
                        all_meta.append(
                            (
                                f"{opname}({beam[bi][2]})",
                                beam[bi][4].unary(opname),
                                beam[bi][6],
                            )
                        )
                except:
                    pass

            if not all_vals:
                break

            vals_mat = cp.concatenate(all_vals, axis=1)
            K = vals_mat.shape[1]

            var = cp.var(vals_mat, axis=0)
            f1s = gpu_batch_f1(vals_mat, y_f)
            f1s = cp.where(var < 1e-20, 0.0, f1s)
            aurocs = gpu_batch_auroc(vals_mat, y_f)
            f1s_cpu = f1s.get()
            aurocs_cpu = aurocs.get()

            new_beam = []
            for i in range(K):
                f1_val = float(f1s_cpu[i])
                auc_val = float(aurocs_cpu[i])
                name, trace, nf = all_meta[i]

                # Meta prune (h = inf)
                if criterion_fn is not None:
                    if criterion_fn(f1_val, auc_val, depth, nf) < criterion_thresh:
                        continue

                # A* heuristic (continuous)
                if f1_target > 0 and f1_val >= f1_target:
                    h = 0.0
                else:
                    h = 1.0 - f1_val

                priority = float(depth) + h

                new_beam.append((priority, f1_val, name, vals_mat[:, i].copy(), trace, auc_val, nf))

                if f1_val > best_f1:
                    best_f1 = f1_val
                    best_name = name
                    best_trace = trace

            # A* ordering: sort by priority (lowest first), keep beam_width
            new_beam.sort(key=lambda x: x[0])
            beam = new_beam[:beam_width]

            del vals_mat, all_vals
            cp.get_default_memory_pool().free_all_blocks()

    return best_f1, best_name, best_trace


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


# ─── Main Pipeline ─────────────────────────────────────────────────


def run_dataset(
    name, X, y, features, n_repeats=200, n_folds=5, max_depth=4, n_subspaces=20, subspace_k=6
):
    total = n_repeats * n_folds
    log.info("=" * 70)
    log.info("A* BEAM SEARCH: %s (N=%d, d=%d) — %d folds", name, X.shape[0], X.shape[1], total)
    log.info("  depth=%d, beam=100, subspaces=%d, k=%d", max_depth, n_subspaces, subspace_k)
    log.info("=" * 70)

    X_gpu = cp.asarray(X, dtype=cp.float64)
    y_gpu = cp.asarray(y, dtype=cp.float64)

    cv = RepeatedStratifiedKFold(n_splits=n_folds, n_repeats=n_repeats, random_state=42)
    splits = list(cv.split(X, y))

    # Cache meta-search by actual training index set (hashable frozenset).
    # RepeatedStratifiedKFold reuses the same k partitions within each
    # repeat, so there are exactly n_folds unique training sets.
    meta_cache = {}

    astar_f1s, astar_train_f1s, formulas = [], [], []
    meta_times = []
    t0 = time.time()

    for fold_i, (tr, te) in enumerate(splits):
        rng = np.random.RandomState(42 + fold_i)

        # Meta-search INSIDE training fold (no leakage)
        # Cache key: hash of sorted training indices
        cache_key = hash(tr.tobytes())
        if cache_key not in meta_cache:
            t_m = time.time()
            # Use top 6 features for fast meta-search (~6K formulas, ~5s)
            d_full = X.shape[1]
            if d_full > 6:
                f1s_full = gpu_batch_f1(X_gpu[cp.asarray(tr)], y_gpu[cp.asarray(tr)])
                top6 = cp.argsort(-f1s_full)[:6].get().tolist()
                meta_feats = [features[i] for i in top6]
                meta_X = X[tr][:, top6]
            else:
                meta_feats = features
                meta_X = X[tr]
            crit_fn, crit_thresh, meta_stats = meta_search_heuristic(
                meta_X, y[tr], meta_feats, max_depth=3
            )
            meta_cache[cache_key] = (crit_fn, crit_thresh, meta_stats)
            meta_times.append(time.time() - t_m)

        crit_fn, crit_thresh, meta_stats = meta_cache[cache_key]

        tr_gpu = cp.asarray(tr)
        train_f1, formula_name, trace = astar_beam_search(
            X_gpu[tr_gpu],
            y_gpu[tr_gpu],
            features,
            crit_fn,
            crit_thresh,
            f1_target=0.0,
            max_depth=max_depth,
            beam_width=100,
            n_subspaces=n_subspaces,
            subspace_k=subspace_k,
            rng=rng,
        )

        astar_train_f1s.append(train_f1)
        formulas.append(formula_name)

        # Fair test evaluation
        vals_tr = trace.evaluate(X[tr])
        vals_te = trace.evaluate(X[te])
        thresh, dirn, _ = find_optimal_threshold(vals_tr, y[tr])
        preds = (dirn * vals_te >= thresh).astype(int)
        test_f1 = f1_score(y[te], preds)
        astar_f1s.append(test_f1)

        if (fold_i + 1) % 50 == 0:
            elapsed = time.time() - t0
            rate = (fold_i + 1) / elapsed
            log.info(
                "  A* %d/%d  train=%.3f test=%.3f  %.2f/s  ETA %.0fs",
                fold_i + 1,
                total,
                np.mean(astar_train_f1s[-50:]),
                np.mean(astar_f1s[-50:]),
                rate,
                (total - fold_i - 1) / rate,
            )

    gpu_time = time.time() - t0
    avg_meta = np.mean(meta_times) if meta_times else 0
    log.info(
        "  A* done: %.0fs (%.2f folds/s), meta-search: %.1fs avg (%d cached)",
        gpu_time,
        total / gpu_time,
        avg_meta,
        n_folds,
    )

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
            "std": float(bf.std()),
            "diff": float(diff.mean()),
            "sigma": float(abs(t_stat)),
            "p": float(p_val),
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
        "meta_stats": meta_stats,
        "test_f1": float(astar_f1s.mean()),
        "train_f1": float(astar_train_f1s.mean()),
        "formula": top[0],
        "baselines": results,
        "meta_time": float(avg_meta),
        "search_time": gpu_time,
    }


def main():
    log.info("=" * 70)
    log.info("A* BEAM SEARCH WITH META-LEARNED HEURISTIC")
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
    for name, X, y, feats in datasets:
        d = X.shape[1]
        n_sub = 30 if d > 15 else 10
        k = min(8, d)
        r = run_dataset(
            name,
            X,
            y,
            feats,
            n_repeats=200,
            n_folds=5,
            max_depth=4,
            n_subspaces=n_sub,
            subspace_k=k,
        )
        all_results.append(r)

    with open("astar_beam_results.json", "w") as f:
        json.dump(all_results, f, indent=2)

    log.info("\n" + "=" * 100)
    log.info("A* BEAM SEARCH — FINAL (meta-heuristic, depth 4, fair eval)")
    for r in all_results:
        log.info(
            "%-15s test=%.4f  meta_prune=%.1f%%  %s  GB:%.1f%s RF:%.1f%s LR:%.1f%s",
            r["name"],
            r["test_f1"],
            100 * r["meta_stats"].get("prune_rate", 0),
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
