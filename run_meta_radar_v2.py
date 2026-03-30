#!/usr/bin/env python3
"""Meta Theory Radar v2: discover the optimal pruning criterion.

Reframing: instead of h*(n) = depth-to-go, ask:
  "Does this node's subtree contain a near-optimal formula?"

If NO → safe to prune (admissible).
If YES → must expand.

This is a BINARY CLASSIFICATION problem on node features.
Theory Radar can search for the optimal classifier — discovering its own
pruning criterion. A pruning criterion is admissible iff it has zero
false negatives (never prunes a subtree containing the optimum).

The meta-radar finds: what function of (F1, AUROC, n_features, ...)
best predicts "this subtree is dead" with zero false negatives?
"""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import cupy as cp
import numpy as np
import logging
import time
import json
from collections import defaultdict

from sklearn.datasets import load_breast_cancer, load_wine, fetch_openml
from sklearn.preprocessing import StandardScaler

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger()


# ─── GPU Batch F1 (with constant-formula fix) ─────────────────────

def gpu_batch_f1(vals, labels):
    N, K = vals.shape
    pos = labels.sum()
    if float(pos) == 0 or float(pos) == N:
        return cp.zeros(K, dtype=cp.float64)
    best = _sweep_f1(vals, labels, pos)
    best2 = _sweep_f1(-vals, labels, pos)
    return cp.maximum(best, best2)


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
    f1 = cp.where(valid, f1, 0.0)
    return cp.max(f1, axis=0)


def gpu_batch_auroc(vals, labels):
    N, K = vals.shape
    pos = float(labels.sum())
    neg = N - pos
    if pos == 0 or neg == 0:
        return cp.full(K, 0.5, dtype=cp.float64)
    idx = cp.argsort(vals, axis=0)
    sl = labels[idx]
    ranks = cp.arange(1, N + 1, dtype=cp.float64).reshape(-1, 1)
    pos_rank_sum = cp.sum(sl * ranks, axis=0)
    auroc = (pos_rank_sum - pos * (pos + 1) / 2) / (pos * neg + 1e-30)
    return cp.maximum(auroc, 1 - auroc)


BINARY_OPS = {
    "+": lambda a, b: a + b, "-": lambda a, b: a - b,
    "*": lambda a, b: a * b, "/": lambda a, b: a / (b + 1e-30),
    "max": lambda a, b: cp.maximum(a, b), "min": lambda a, b: cp.minimum(a, b),
    "hypot": lambda a, b: cp.sqrt(a**2 + b**2),
    "diff_sq": lambda a, b: (a - b) ** 2,
    "harmonic": lambda a, b: 2 * a * b / (a + b + 1e-30),
    "geometric": lambda a, b: cp.sign(a * b) * cp.sqrt(cp.abs(a * b)),
}

UNARY_OPS = {
    "log": lambda x: cp.log(cp.abs(x) + 1e-30),
    "sqrt": lambda x: cp.sqrt(cp.abs(x)),
    "sq": lambda x: x ** 2, "abs": lambda x: cp.abs(x),
    "sigmoid": lambda x: 1.0 / (1.0 + cp.exp(-cp.clip(x, -500, 500))),
    "tanh": lambda x: cp.tanh(cp.clip(x, -500, 500)),
}


# ─── Phase 1: Exhaustive enumeration with parent tracking ─────────

def exhaustive_with_lineage(X_gpu, y_gpu, feat_names, max_depth=3):
    """Enumerate all formulas, track parent-child relationships."""
    N, d = X_gpu.shape
    y_f = y_gpu.astype(cp.float64)

    catalog = {}  # name → info dict
    children = defaultdict(list)  # parent_name → [child_names]

    # Depth 1
    f1s = gpu_batch_f1(X_gpu, y_f)
    aurocs = gpu_batch_auroc(X_gpu, y_f)
    for i in range(d):
        name = feat_names[i]
        catalog[name] = {
            "depth": 1, "f1": float(f1s[i]), "auroc": float(aurocs[i]),
            "values": X_gpu[:, i].copy(), "n_features": 1, "parent": None,
        }

    log.info("  Depth 1: %d formulas", d)
    prev_names = list(catalog.keys())

    for depth in range(2, max_depth + 1):
        new_entries = {}

        for pname in prev_names:
            parent = catalog[pname]
            pv = parent["values"]

            for j in range(d):
                leaf = X_gpu[:, j]
                for opname, opfn in BINARY_OPS.items():
                    cname = f"({pname} {opname} {feat_names[j]})"
                    if cname in catalog or cname in new_entries:
                        continue
                    try:
                        cv = opfn(pv, leaf)
                        cv = cp.nan_to_num(cv, nan=0.0, posinf=1e10, neginf=-1e10)
                        if float(cp.var(cv)) < 1e-20:
                            continue
                        new_entries[cname] = {
                            "depth": depth, "values": cv,
                            "n_features": parent["n_features"] + (1 if feat_names[j] not in pname else 0),
                            "parent": pname,
                        }
                        children[pname].append(cname)
                    except Exception:
                        pass

            for opname, opfn in UNARY_OPS.items():
                cname = f"{opname}({pname})"
                if cname in catalog or cname in new_entries:
                    continue
                try:
                    cv = opfn(pv)
                    cv = cp.nan_to_num(cv, nan=0.0, posinf=1e10, neginf=-1e10)
                    if float(cp.var(cv)) < 1e-20:
                        continue
                    new_entries[cname] = {
                        "depth": depth, "values": cv,
                        "n_features": parent["n_features"], "parent": pname,
                    }
                    children[pname].append(cname)
                except Exception:
                    pass

        # Batch F1 + AUROC
        if new_entries:
            names = list(new_entries.keys())
            vals_mat = cp.stack([new_entries[n]["values"] for n in names], axis=1)
            f1s = gpu_batch_f1(vals_mat, y_f)
            aurocs = gpu_batch_auroc(vals_mat, y_f)
            for i, n in enumerate(names):
                new_entries[n]["f1"] = float(f1s[i])
                new_entries[n]["auroc"] = float(aurocs[i])
                catalog[n] = new_entries[n]
            del vals_mat

        best = max(v["f1"] for v in catalog.values())
        log.info("  Depth %d: %d new (%d total), best F1=%.4f", depth, len(new_entries), len(catalog), best)
        prev_names = list(new_entries.keys())

    cp.get_default_memory_pool().free_all_blocks()
    return catalog, children


# ─── Phase 2: Subtree viability (alive/dead classification) ───────

def compute_subtree_viability(catalog, children, epsilon=0.005):
    """For each node, determine if its subtree contains a near-optimal formula.

    alive=1: subtree contains formula with F1 >= global_best - epsilon
    alive=0: subtree is dead (safe to prune without losing optimality)
    """
    global_best = max(v["f1"] for v in catalog.values())
    threshold = global_best - epsilon

    # For leaf nodes (depth 3), alive iff f1 >= threshold
    for name, info in catalog.items():
        info["alive"] = 1 if info["f1"] >= threshold else 0

    # Propagate upward: a parent is alive if ANY child is alive
    # Process from deepest to shallowest
    for depth in [2, 1]:
        for name, info in catalog.items():
            if info["depth"] != depth:
                continue
            # Check if any child is alive
            child_alive = any(
                catalog[c]["alive"] == 1
                for c in children.get(name, [])
                if c in catalog
            )
            # Node is alive if IT reaches threshold OR any descendant does
            info["alive"] = 1 if (info["f1"] >= threshold or child_alive) else 0

    # Stats
    for depth in [1, 2, 3]:
        nodes = [(n, v) for n, v in catalog.items() if v["depth"] == depth]
        alive = sum(1 for _, v in nodes if v["alive"] == 1)
        dead = len(nodes) - alive
        log.info("  Depth %d: %d alive, %d dead (%.1f%% prunable)",
                 depth, alive, dead, 100 * dead / max(len(nodes), 1))

    return catalog


# ─── Phase 3: Theory Radar on node features ───────────────────────

def meta_radar_search(catalog, prevalence):
    """Use Theory Radar to find the best pruning criterion.

    Search for a formula over node features that classifies dead subtrees
    with ZERO false negatives (never prunes an alive subtree).
    Maximize true negatives (prune as many dead subtrees as possible).
    """
    # Build dataset: only depth 1 and 2 nodes (depth 3 are terminal)
    feat_names = ["f1", "auroc", "n_features", "f1_gap", "auroc_gap", "prev"]
    rows = []
    for name, info in catalog.items():
        if info["depth"] >= 3:
            continue
        rows.append({
            "name": name,
            "depth": info["depth"],
            "alive": info["alive"],
            "f1": info["f1"],
            "auroc": info["auroc"],
            "n_features": info["n_features"],
            "f1_gap": 1.0 - info["f1"],
            "auroc_gap": 1.0 - info["auroc"],
            "prev": prevalence,
        })

    if not rows:
        return []

    X = np.array([[r[f] for f in feat_names] for r in rows], dtype=np.float64)
    alive = np.array([r["alive"] for r in rows], dtype=np.float64)
    N, d = X.shape

    n_alive = int(alive.sum())
    n_dead = N - n_alive
    log.info("  Meta-search: %d shallow nodes (%d alive, %d dead)", N, n_alive, n_dead)

    if n_dead == 0:
        log.info("  No dead subtrees — no pruning criterion to learn")
        return []

    X_gpu = cp.asarray(X)
    alive_gpu = cp.asarray(alive)

    # Search for formulas where: threshold the formula to classify alive/dead
    # Requirement: ALL alive nodes must be above the threshold (zero false negatives)
    # Maximize: dead nodes below the threshold (true negatives = pruned)

    results = []

    def evaluate_criterion(vals_gpu, crit_name):
        """Find best threshold for this formula that has zero false negatives."""
        vals_cpu = vals_gpu.get()
        alive_cpu = alive

        # For zero false negatives: threshold must be ≤ min(vals[alive])
        alive_vals = vals_cpu[alive_cpu == 1]
        dead_vals = vals_cpu[alive_cpu == 0]

        if len(alive_vals) == 0 or len(dead_vals) == 0:
            return

        min_alive = alive_vals.min()

        # Threshold at min_alive: predict dead if val < min_alive
        pruned = (dead_vals < min_alive).sum()
        total_dead = len(dead_vals)
        prune_rate = pruned / total_dead if total_dead > 0 else 0

        # Also try: threshold such that ALL alive are above AND maximize pruning
        # Sort dead values, find how many are below min_alive
        results.append({
            "criterion": crit_name,
            "threshold": float(min_alive),
            "pruned": int(pruned),
            "total_dead": int(total_dead),
            "prune_rate": float(prune_rate),
            "false_negatives": 0,  # guaranteed by construction
            "total_alive": int(n_alive),
        })

    # Depth 1: raw features
    for i in range(d):
        evaluate_criterion(X_gpu[:, i], feat_names[i])
        evaluate_criterion(-X_gpu[:, i], f"-{feat_names[i]}")

    # Depth 2: pairwise combinations
    for i in range(d):
        for j in range(d):
            if i == j:
                continue
            for opname, opfn in [
                ("+", lambda a, b: a + b),
                ("-", lambda a, b: a - b),
                ("*", lambda a, b: a * b),
                ("/", lambda a, b: a / (b + 1e-30)),
                ("max", lambda a, b: cp.maximum(a, b)),
                ("min", lambda a, b: cp.minimum(a, b)),
            ]:
                try:
                    vals = opfn(X_gpu[:, i], X_gpu[:, j])
                    vals = cp.nan_to_num(vals, nan=0.0, posinf=1e10, neginf=-1e10)
                    cname = f"({feat_names[i]} {opname} {feat_names[j]})"
                    evaluate_criterion(vals, cname)
                except Exception:
                    pass

    # Depth 2: unary on single features
    unary_ops = {
        "log": lambda x: cp.log(cp.abs(x) + 1e-30),
        "sqrt": lambda x: cp.sqrt(cp.abs(x)),
        "sq": lambda x: x ** 2,
        "inv": lambda x: 1.0 / (x + 1e-30),
    }
    for i in range(d):
        for uname, ufn in unary_ops.items():
            try:
                vals = ufn(X_gpu[:, i])
                vals = cp.nan_to_num(vals, nan=0.0, posinf=1e10, neginf=-1e10)
                cname = f"{uname}({feat_names[i]})"
                evaluate_criterion(vals, cname)
            except Exception:
                pass

    # Sort by prune rate (most dead subtrees pruned)
    results.sort(key=lambda x: -x["prune_rate"])
    return results


# ─── Main ──────────────────────────────────────────────────────────

def run_on_dataset(name, X, y, feat_names, max_d=10):
    log.info("=" * 70)
    log.info("META THEORY RADAR v2: %s", name)
    log.info("=" * 70)

    d = X.shape[1]
    if d > max_d:
        X_gpu = cp.asarray(X, dtype=cp.float64)
        y_gpu = cp.asarray(y, dtype=cp.float64)
        f1s = gpu_batch_f1(X_gpu, y_gpu)
        top_idx = cp.argsort(-f1s)[:max_d].get()
        X = X[:, top_idx]
        feat_names = [feat_names[i] for i in top_idx]
        del X_gpu, y_gpu
        cp.get_default_memory_pool().free_all_blocks()

    X_gpu = cp.asarray(X, dtype=cp.float64)
    y_gpu = cp.asarray(y, dtype=cp.float64)
    prevalence = float(y.mean())
    log.info("  N=%d, d=%d, prev=%.2f", X.shape[0], X.shape[1], prevalence)

    # Phase 1: Exhaustive enumeration with lineage tracking
    t0 = time.time()
    catalog, children = exhaustive_with_lineage(X_gpu, y_gpu, feat_names, max_depth=3)
    log.info("  %d formulas in %.0fs", len(catalog), time.time() - t0)

    # Free values
    for n in catalog:
        if "values" in catalog[n]:
            del catalog[n]["values"]
    cp.get_default_memory_pool().free_all_blocks()

    # Phase 2: Subtree viability
    log.info("  Computing subtree viability (epsilon=0.005)...")
    catalog = compute_subtree_viability(catalog, children, epsilon=0.005)

    # Phase 3: Meta-radar search for pruning criterion
    log.info("  Searching for optimal pruning criterion...")
    criteria = meta_radar_search(catalog, prevalence)

    log.info("")
    log.info("  TOP 15 PRUNING CRITERIA (0 false negatives guaranteed):")
    log.info("  %-45s  %-10s  %-10s  %-10s", "Criterion", "Threshold", "Pruned", "Prune%")
    log.info("  " + "-" * 80)
    for c in criteria[:15]:
        log.info("  %-45s  %8.4f    %4d/%4d   %5.1f%%",
                 c["criterion"][:45], c["threshold"],
                 c["pruned"], c["total_dead"], 100 * c["prune_rate"])

    # Analysis: what features separate alive from dead?
    log.info("")
    log.info("  ALIVE vs DEAD node statistics:")
    feat_names_meta = ["f1", "auroc", "n_features"]
    for fn in feat_names_meta:
        alive_vals = [v[fn] for v in catalog.values() if v["depth"] < 3 and v["alive"] == 1]
        dead_vals = [v[fn] for v in catalog.values() if v["depth"] < 3 and v["alive"] == 0]
        if alive_vals and dead_vals:
            log.info("    %s: alive=%.4f±%.4f  dead=%.4f±%.4f",
                     fn, np.mean(alive_vals), np.std(alive_vals),
                     np.mean(dead_vals), np.std(dead_vals))

    return {
        "dataset": name, "n_formulas": len(catalog),
        "top_criteria": criteria[:20],
    }


def main():
    log.info("=" * 70)
    log.info("META THEORY RADAR v2 — Discovering Optimal Pruning Criteria")
    log.info("Theory Radar searching for its own admissible pruning rule")
    log.info("=" * 70)

    props = cp.cuda.runtime.getDeviceProperties(0)
    log.info("GPU: %s (%.1f GB free)", props["name"].decode(),
             cp.cuda.Device(0).mem_info[0] / 1e9)

    datasets = []

    bc = load_breast_cancer()
    X = StandardScaler().fit_transform(bc.data[:, :10])
    datasets.append(("BreastCancer-10D", X, bc.target, [f"f{i}" for i in range(10)]))

    wine = load_wine()
    X = StandardScaler().fit_transform(wine.data)
    y = (wine.target == 0).astype(int)
    datasets.append(("Wine-13D", X, y, [f"w{i}" for i in range(13)]))

    try:
        pima = fetch_openml("diabetes", version=1, as_frame=False, parser="auto")
        X = StandardScaler().fit_transform(pima.data.astype(float))
        y = (pima.target == "tested_positive").astype(int)
        datasets.append(("Diabetes-8D", X, y, [f"d{i}" for i in range(8)]))
    except Exception as e:
        log.warning("Diabetes: %s", e)

    all_results = []
    for name, X, y, feats in datasets:
        result = run_on_dataset(name, X, y, feats)
        all_results.append(result)

    with open("meta_radar_v2_results.json", "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    log.info("Saved meta_radar_v2_results.json")


if __name__ == "__main__":
    main()
