#!/usr/bin/env python3
"""Meta Theory Radar: use Theory Radar to search for its own optimal heuristic.

Phase 1: Exhaustive search on small problems → ground-truth h*(n) for every node
Phase 2: Extract features from each node → (features, h*) dataset
Phase 3: Search the space of heuristic formulas using Theory Radar itself
Phase 4: Validate admissibility and measure speedup

The heuristic IS a formula over node features, so we can search for it
with the same beam search machinery we use for classification formulas.
"""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import cupy as cp
import numpy as np
import logging
import time
import json
from itertools import product

from sklearn.datasets import load_breast_cancer, load_wine, fetch_openml
from sklearn.preprocessing import StandardScaler

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger()


# ─── GPU Batch F1 (with constant-formula fix) ─────────────────────

def gpu_batch_f1(vals, labels):
    """Optimal thresholded F1 for each column."""
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
    """Batch AUROC via Mann-Whitney U for K candidates."""
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


# ─── GPU Binary/Unary Ops ─────────────────────────────────────────

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
    "sq": lambda x: x ** 2,
    "abs": lambda x: cp.abs(x),
    "sigmoid": lambda x: 1.0 / (1.0 + cp.exp(-cp.clip(x, -500, 500))),
    "tanh": lambda x: cp.tanh(cp.clip(x, -500, 500)),
}


# ─── Phase 1: Exhaustive enumeration → ground truth ───────────────

def exhaustive_search(X_gpu, y_gpu, feat_names, max_depth=3):
    """Enumerate ALL formulas up to max_depth. Return dict: formula → (depth, f1, auroc, values)."""
    N, d = X_gpu.shape
    y_f = y_gpu.astype(cp.float64)

    catalog = {}  # name → (depth, f1, auroc, values, n_features)

    # Depth 1: raw features
    f1s = gpu_batch_f1(X_gpu, y_f)
    aurocs = gpu_batch_auroc(X_gpu, y_f)
    for i in range(d):
        name = feat_names[i]
        catalog[name] = {
            "depth": 1, "f1": float(f1s[i]), "auroc": float(aurocs[i]),
            "values": X_gpu[:, i].copy(), "n_features": 1,
        }

    log.info("  Depth 1: %d formulas, best F1=%.4f", len(catalog),
             max(v["f1"] for v in catalog.values()))

    prev_depth_formulas = list(catalog.keys())

    for depth in range(2, max_depth + 1):
        new_formulas = {}

        for parent_name in prev_depth_formulas:
            parent = catalog[parent_name]
            pv = parent["values"]

            # Binary ops with all features
            for j in range(d):
                leaf = X_gpu[:, j]
                for opname, opfn in BINARY_OPS.items():
                    cname = f"({parent_name} {opname} {feat_names[j]})"
                    if cname in catalog or cname in new_formulas:
                        continue
                    try:
                        cv = opfn(pv, leaf)
                        cv = cp.nan_to_num(cv, nan=0.0, posinf=1e10, neginf=-1e10)
                        if float(cp.var(cv)) < 1e-20:
                            continue
                        new_formulas[cname] = {
                            "depth": depth, "values": cv,
                            "n_features": parent["n_features"] + (1 if feat_names[j] not in parent_name else 0),
                            "parent": parent_name,
                        }
                    except Exception:
                        pass

            # Unary ops
            for opname, opfn in UNARY_OPS.items():
                cname = f"{opname}({parent_name})"
                if cname in catalog or cname in new_formulas:
                    continue
                try:
                    cv = opfn(pv)
                    cv = cp.nan_to_num(cv, nan=0.0, posinf=1e10, neginf=-1e10)
                    if float(cp.var(cv)) < 1e-20:
                        continue
                    new_formulas[cname] = {
                        "depth": depth, "values": cv,
                        "n_features": parent["n_features"],
                        "parent": parent_name,
                    }
                except Exception:
                    pass

        # Batch-evaluate F1 and AUROC for all new formulas
        if new_formulas:
            names = list(new_formulas.keys())
            vals_matrix = cp.stack([new_formulas[n]["values"] for n in names], axis=1)
            f1s = gpu_batch_f1(vals_matrix, y_f)
            aurocs = gpu_batch_auroc(vals_matrix, y_f)

            for i, name in enumerate(names):
                entry = new_formulas[name]
                entry["f1"] = float(f1s[i])
                entry["auroc"] = float(aurocs[i])
                catalog[name] = entry

            del vals_matrix
            cp.get_default_memory_pool().free_all_blocks()

        best_f1 = max(v["f1"] for v in catalog.values())
        log.info("  Depth %d: %d new formulas (%d total), best F1=%.4f",
                 depth, len(new_formulas), len(catalog), best_f1)

        prev_depth_formulas = list(new_formulas.keys())

    return catalog


# ─── Phase 2: Compute h*(n) for every node ────────────────────────

def compute_hstar(catalog, max_depth):
    """For each formula, compute h* = min additional depth to reach the global best F1."""
    best_f1_at_depth = {}
    for name, info in catalog.items():
        d = info["depth"]
        if d not in best_f1_at_depth or info["f1"] > best_f1_at_depth[d]:
            best_f1_at_depth[d] = info["f1"]

    global_best = max(info["f1"] for info in catalog.values())
    log.info("  Best F1 per depth: %s", {d: f"{f:.4f}" for d, f in sorted(best_f1_at_depth.items())})
    log.info("  Global best F1: %.4f", global_best)

    # h*(n) = minimum additional depth for ANY descendant of n to reach global_best
    # For simplicity: h*(n) = 0 if f1 >= global_best, else min(depth_of_best - n.depth, max_depth - n.depth)
    # More precisely: track which formulas are ancestors of the optimal formula

    # Find all optimal formulas (within 0.001 of global best)
    optimal_names = {n for n, info in catalog.items() if info["f1"] >= global_best - 0.001}
    optimal_depths = {catalog[n]["depth"] for n in optimal_names}
    min_optimal_depth = min(optimal_depths)

    # For each node, h* = max(0, min_optimal_depth - node.depth) if node could reach optimal
    # Conservative: h*(n) = max(0, min_depth_to_beat_current - n.depth)
    # But true h* requires knowing the search tree structure

    # Simpler: for each node, does there exist a descendant (at deeper depth) with f1 >= global_best?
    # If yes: h* = min(descendant.depth - node.depth)
    # If no: h* = infinity (this subtree is dead)

    # For now, use the conservative estimate:
    # h*(n) = 0 if n.f1 >= global_best - 0.001
    # h*(n) = min over all optimal formulas o: o.depth - n.depth (if positive)
    # This is a LOWER BOUND on true h*, so using it to evaluate heuristics is conservative

    for name, info in catalog.items():
        if info["f1"] >= global_best - 0.001:
            info["hstar"] = 0
        else:
            info["hstar"] = max(0, min_optimal_depth - info["depth"])

    return catalog


# ─── Phase 3: Extract node features → build training set ──────────

def extract_features(catalog, prevalence):
    """Extract heuristic-relevant features from each node."""
    rows = []
    for name, info in catalog.items():
        if "f1" not in info:
            continue
        rows.append({
            "name": name,
            "depth": info["depth"],
            "f1": info["f1"],
            "auroc": info["auroc"],
            "n_features": info["n_features"],
            "hstar": info["hstar"],
            # Derived features a heuristic could use:
            "f1_gap": 1.0 - info["f1"],  # gap to perfect
            "auroc_gap": 1.0 - info["auroc"],
            "prevalence": prevalence,
        })
    return rows


# ─── Phase 4: Search for optimal admissible heuristic ─────────────

def search_heuristic(training_data):
    """Use beam search to find a formula over node features that is:
    1. Admissible: h(n) <= h*(n) for all n
    2. Tight: sum(h*(n) - h(n)) is minimized (most informative)

    Node features available: f1, auroc, n_features, f1_gap, auroc_gap, prevalence
    Heuristic output: integer (0, 1, 2, ...) = estimated remaining depth
    """
    # Convert training data to arrays
    feat_names = ["f1", "auroc", "n_features", "f1_gap", "auroc_gap", "prevalence"]
    X = np.array([[r[f] for f in feat_names] for r in training_data], dtype=np.float64)
    hstar = np.array([r["hstar"] for r in training_data], dtype=np.float64)

    X_gpu = cp.asarray(X)
    hstar_gpu = cp.asarray(hstar)
    N, d = X_gpu.shape

    log.info("  Heuristic search: %d nodes, %d features, h* range [%d, %d]",
             N, d, int(hstar.min()), int(hstar.max()))

    # For the heuristic, we want formulas that:
    # - Output values correlated with h* (higher when h* is higher)
    # - Never exceed h* (admissibility)
    # Strategy: find formula f(node_features), then threshold:
    #   h(n) = floor(f(n)) if f(n) <= h*(n) for all n, else scale down

    # Beam search over formulas of node features
    beam_width = 50
    best_formulas = []

    # Depth 1: raw features
    for i in range(d):
        vals = X_gpu[:, i]
        # Check: can we scale this to be admissible?
        # h(n) = floor(alpha * feat_i) where alpha = min(h*(n) / feat_i) for feat_i > 0
        positive_mask = vals > 1e-10
        if float(positive_mask.sum()) == 0:
            continue

        # Find max alpha such that alpha * vals <= hstar for all nodes
        ratios = cp.where(positive_mask, hstar_gpu / (vals + 1e-30), cp.inf)
        alpha = float(cp.min(ratios))
        if alpha <= 0 or cp.isinf(cp.asarray(alpha)):
            continue

        h_vals = cp.floor(alpha * vals)
        h_vals = cp.clip(h_vals, 0, 10)

        # Check admissibility
        violations = int((h_vals > hstar_gpu + 0.01).sum())
        if violations > 0:
            continue

        # Tightness: sum of (h* - h) — lower is better
        slack = float((hstar_gpu - h_vals).sum())
        tightness = float((h_vals > 0).sum())  # how many nodes get non-zero h

        best_formulas.append({
            "formula": f"floor({alpha:.4f} * {feat_names[i]})",
            "alpha": alpha,
            "feature": feat_names[i],
            "violations": violations,
            "tightness": tightness,
            "slack": slack,
            "nonzero_frac": tightness / N,
        })

    # Also try composite features
    for i in range(d):
        for j in range(d):
            if i == j:
                continue
            for opname, opfn in [
                ("+", lambda a, b: a + b),
                ("*", lambda a, b: a * b),
                ("-", lambda a, b: a - b),
                ("max", lambda a, b: cp.maximum(a, b)),
            ]:
                vals = opfn(X_gpu[:, i], X_gpu[:, j])
                vals = cp.nan_to_num(vals, nan=0.0, posinf=1e10, neginf=-1e10)

                positive_mask = vals > 1e-10
                if float(positive_mask.sum()) == 0:
                    continue

                ratios = cp.where(positive_mask, hstar_gpu / (vals + 1e-30), cp.inf)
                alpha = float(cp.min(ratios))
                if alpha <= 0 or cp.isinf(cp.asarray(alpha)):
                    continue

                h_vals = cp.floor(alpha * vals)
                h_vals = cp.clip(h_vals, 0, 10)

                violations = int((h_vals > hstar_gpu + 0.01).sum())
                if violations > 0:
                    continue

                slack = float((hstar_gpu - h_vals).sum())
                tightness = float((h_vals > 0).sum())

                best_formulas.append({
                    "formula": f"floor({alpha:.4f} * ({feat_names[i]} {opname} {feat_names[j]}))",
                    "alpha": alpha,
                    "violations": violations,
                    "tightness": tightness,
                    "slack": slack,
                    "nonzero_frac": tightness / N,
                })

    # Sort by tightness (most non-zero predictions)
    best_formulas.sort(key=lambda x: -x["tightness"])
    return best_formulas


# ─── Main ─────���────────────────────────────────────────────────────

def run_on_dataset(name, X, y, feat_names, max_d_features=10):
    """Run the full meta-radar pipeline on one dataset."""
    log.info("=" * 70)
    log.info("META THEORY RADAR: %s", name)
    log.info("=" * 70)

    # Use subset of features if too many
    d = X.shape[1]
    if d > max_d_features:
        # Pick top features by individual F1
        X_gpu = cp.asarray(X, dtype=cp.float64)
        y_gpu = cp.asarray(y, dtype=cp.float64)
        f1s = gpu_batch_f1(X_gpu, y_gpu)
        top_idx = cp.argsort(-f1s)[:max_d_features].get()
        X = X[:, top_idx]
        feat_names = [feat_names[i] for i in top_idx]
        log.info("  Reduced to top %d features by F1", max_d_features)

    X_gpu = cp.asarray(X, dtype=cp.float64)
    y_gpu = cp.asarray(y, dtype=cp.float64)
    prevalence = float(y.mean())

    log.info("  N=%d, d=%d, prev=%.2f", X.shape[0], X.shape[1], prevalence)

    # Phase 1: Exhaustive search
    t0 = time.time()
    log.info("  Phase 1: Exhaustive enumeration...")
    catalog = exhaustive_search(X_gpu, y_gpu, feat_names, max_depth=3)
    t1 = time.time()
    log.info("  Enumerated %d formulas in %.1fs", len(catalog), t1 - t0)

    # Free GPU values we don't need for phases 2-4
    for name_k in catalog:
        if "values" in catalog[name_k]:
            del catalog[name_k]["values"]
    cp.get_default_memory_pool().free_all_blocks()

    # Phase 2: Compute h*
    log.info("  Phase 2: Computing ground-truth h*...")
    catalog = compute_hstar(catalog, max_depth=3)

    # Phase 3: Extract features
    log.info("  Phase 3: Extracting node features...")
    training_data = extract_features(catalog, prevalence)
    log.info("  %d training examples", len(training_data))

    # Distribution of h*
    hstar_counts = {}
    for r in training_data:
        h = r["hstar"]
        hstar_counts[h] = hstar_counts.get(h, 0) + 1
    log.info("  h* distribution: %s", dict(sorted(hstar_counts.items())))

    # Phase 4: Search for optimal heuristic
    log.info("  Phase 4: Searching heuristic space...")
    heuristics = search_heuristic(training_data)

    log.info("")
    log.info("  TOP 10 ADMISSIBLE HEURISTICS:")
    log.info("  %-60s  %-8s  %-8s  %-10s", "Formula", "NonZero%", "Slack", "Violations")
    log.info("  " + "-" * 90)
    for h in heuristics[:10]:
        log.info("  %-60s  %5.1f%%    %8.0f  %d",
                 h["formula"][:60], 100 * h["nonzero_frac"], h["slack"], h["violations"])

    return {
        "dataset": name,
        "n_formulas": len(catalog),
        "n_training": len(training_data),
        "hstar_dist": hstar_counts,
        "top_heuristics": heuristics[:20],
    }


def main():
    log.info("=" * 70)
    log.info("META THEORY RADAR")
    log.info("Theory Radar searching for its own optimal heuristic")
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

    with open("meta_radar_results.json", "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    log.info("Saved meta_radar_results.json")


if __name__ == "__main__":
    main()
