#!/usr/bin/env python3
"""Meta Theory Radar v3: Deep search for optimal pruning criteria.

Pushes the meta-search to its limits:
1. Depth-3 heuristic formulas over node features (not just depth 1-2)
2. Multiple epsilon values to find the Pareto frontier (pruning vs safety)
3. Cross-dataset generalization: train on one dataset, test on others
4. Ablation: which node features matter most?
5. Larger formula space: 10 binary ops × 6 unary ops × 8 node features
6. Exhaustive depth-4 enumeration on small datasets (d=6) for deeper ground truth
"""

import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import cupy as cp
import numpy as np
import logging
import time
import json
from collections import defaultdict

from sklearn.datasets import load_breast_cancer, load_wine, load_iris, fetch_openml
from sklearn.preprocessing import StandardScaler

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger()


# ─── GPU Batch F1 ─────────────────────────────────────────────────


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


# ─── Formula Search Ops ───────────────────────────────────────────

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


# ─── Exhaustive Search with Lineage ───────────────────────────────


def exhaustive_with_lineage(X_gpu, y_gpu, feat_names, max_depth=3):
    N, d = X_gpu.shape
    y_f = y_gpu.astype(cp.float64)
    catalog = {}
    children = defaultdict(list)

    f1s = gpu_batch_f1(X_gpu, y_f)
    aurocs = gpu_batch_auroc(X_gpu, y_f)
    for i in range(d):
        name = feat_names[i]
        catalog[name] = {
            "depth": 1,
            "f1": float(f1s[i]),
            "auroc": float(aurocs[i]),
            "values": X_gpu[:, i].copy(),
            "n_features": 1,
            "parent": None,
        }

    log.info("    Depth 1: %d formulas", d)
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
                            "depth": depth,
                            "values": cv,
                            "n_features": parent["n_features"]
                            + (1 if feat_names[j] not in pname else 0),
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
                        "depth": depth,
                        "values": cv,
                        "n_features": parent["n_features"],
                        "parent": pname,
                    }
                    children[pname].append(cname)
                except Exception:
                    pass

        if new_entries:
            names = list(new_entries.keys())
            # Batch in chunks to avoid OOM
            chunk_size = 50000
            for start in range(0, len(names), chunk_size):
                chunk_names = names[start : start + chunk_size]
                vals_mat = cp.stack([new_entries[n]["values"] for n in chunk_names], axis=1)
                f1s = gpu_batch_f1(vals_mat, y_f)
                aurocs = gpu_batch_auroc(vals_mat, y_f)
                for i, n in enumerate(chunk_names):
                    new_entries[n]["f1"] = float(f1s[i])
                    new_entries[n]["auroc"] = float(aurocs[i])
                    catalog[n] = new_entries[n]
                del vals_mat
                cp.get_default_memory_pool().free_all_blocks()

        best = max(v["f1"] for v in catalog.values())
        log.info(
            "    Depth %d: %d new (%d total), best F1=%.4f",
            depth,
            len(new_entries),
            len(catalog),
            best,
        )
        prev_names = list(new_entries.keys())

    cp.get_default_memory_pool().free_all_blocks()
    return catalog, children


# ─── Subtree Viability ─────────────────────────────────────────────


def compute_viability(catalog, children, epsilon=0.005):
    global_best = max(v["f1"] for v in catalog.values())
    threshold = global_best - epsilon

    for name, info in catalog.items():
        info["alive"] = 1 if info["f1"] >= threshold else 0

    max_depth = max(v["depth"] for v in catalog.values())
    for depth in range(max_depth - 1, 0, -1):
        for name, info in catalog.items():
            if info["depth"] != depth:
                continue
            child_alive = any(
                catalog[c]["alive"] == 1 for c in children.get(name, []) if c in catalog
            )
            info["alive"] = 1 if (info["f1"] >= threshold or child_alive) else 0

    return catalog


# ─── Deep Meta-Search (Depth-3 Heuristic Formulas) ────────────────


def deep_meta_search(catalog, prevalence, max_depth=3):
    """Search for depth-3 heuristic formulas over node features.

    Node features: f1, auroc, n_features, f1_gap, auroc_gap, prev,
                   f1*auroc, auroc/n_features, f1_gap*auroc_gap
    """
    # Build dataset
    META_FEATS = [
        ("f1", lambda r: r["f1"]),
        ("auroc", lambda r: r["auroc"]),
        ("nf", lambda r: float(r["n_features"])),
        ("f1_gap", lambda r: 1.0 - r["f1"]),
        ("auc_gap", lambda r: 1.0 - r["auroc"]),
        ("prev", lambda r: prevalence),
        ("f1xa", lambda r: r["f1"] * r["auroc"]),
        ("a_nf", lambda r: r["auroc"] / max(r["n_features"], 0.01)),
        ("fg_ag", lambda r: (1.0 - r["f1"]) * (1.0 - r["auroc"])),
        ("depth", lambda r: float(r["depth"])),
    ]

    rows = []
    for name, info in catalog.items():
        if info["depth"] >= max_depth:  # only shallow nodes
            continue
        if "f1" not in info:
            continue
        row = {fn: fn_fn(info) for fn, fn_fn in META_FEATS}
        row["alive"] = info["alive"]
        row["name"] = name
        rows.append(row)

    if not rows:
        return []

    feat_names = [fn for fn, _ in META_FEATS]
    d = len(feat_names)
    X = np.array([[r[fn] for fn in feat_names] for r in rows], dtype=np.float64)
    alive = np.array([r["alive"] for r in rows], dtype=np.float64)
    N = len(rows)

    n_alive = int(alive.sum())
    n_dead = N - n_alive
    log.info(
        "    Meta-search depth %d: %d nodes (%d alive, %d dead), %d features",
        max_depth,
        N,
        n_alive,
        n_dead,
        d,
    )

    if n_dead == 0:
        return []

    X_gpu = cp.asarray(X)
    alive_gpu = cp.asarray(alive)

    results = []

    def eval_criterion(vals_gpu, crit_name):
        vals_cpu = vals_gpu.get()
        alive_vals = vals_cpu[alive == 1]
        dead_vals = vals_cpu[alive == 0]
        if len(alive_vals) == 0 or len(dead_vals) == 0:
            return

        # Threshold = min of alive values (ensures zero false negatives)
        min_alive = alive_vals.min()
        pruned = int((dead_vals < min_alive).sum())
        prune_rate = pruned / len(dead_vals)

        # Also try the other direction
        max_alive = alive_vals.max()
        pruned_rev = int((dead_vals > max_alive).sum())
        prune_rate_rev = pruned_rev / len(dead_vals)

        best_dir = ">" if prune_rate >= prune_rate_rev else "<"
        best_pruned = max(pruned, pruned_rev)
        best_rate = max(prune_rate, prune_rate_rev)
        best_thresh = min_alive if best_dir == ">" else max_alive

        if best_rate > 0.01:  # only report if it prunes something
            results.append(
                {
                    "criterion": crit_name,
                    "direction": best_dir,
                    "threshold": float(best_thresh),
                    "pruned": int(best_pruned),
                    "total_dead": int(n_dead),
                    "prune_rate": float(best_rate),
                    "false_negatives": 0,
                    "total_alive": int(n_alive),
                }
            )

    # ── Depth 1: raw meta-features ──
    for i in range(d):
        eval_criterion(X_gpu[:, i], feat_names[i])

    # ── Depth 2: all pairwise ──
    for i in range(d):
        for j in range(d):
            if i == j:
                continue
            for opname, opfn in BINARY_OPS.items():
                try:
                    vals = opfn(X_gpu[:, i], X_gpu[:, j])
                    vals = cp.nan_to_num(vals, nan=0.0, posinf=1e10, neginf=-1e10)
                    if float(cp.var(vals)) < 1e-20:
                        continue
                    eval_criterion(vals, f"({feat_names[i]} {opname} {feat_names[j]})")
                except Exception:
                    pass

        # Unary on raw features
        for uname, ufn in UNARY_OPS.items():
            try:
                vals = ufn(X_gpu[:, i])
                vals = cp.nan_to_num(vals, nan=0.0, posinf=1e10, neginf=-1e10)
                if float(cp.var(vals)) < 1e-20:
                    continue
                eval_criterion(vals, f"{uname}({feat_names[i]})")
            except Exception:
                pass

    # ── Depth 3: beam over depth-2 results, expand one more level ──
    # Take top 50 depth-2 criteria, expand each with binary/unary ops
    results.sort(key=lambda x: -x["prune_rate"])
    depth2_top = results[:50]

    log.info(
        "    Depth-2 meta-search found %d criteria, expanding top 50 to depth 3...", len(results)
    )

    # Reconstruct values for top depth-2 criteria
    depth2_vals = {}
    for res in depth2_top:
        cname = res["criterion"]
        # Re-evaluate this formula
        try:
            vals = _eval_meta_formula(cname, X_gpu, feat_names)
            if vals is not None:
                depth2_vals[cname] = vals
        except Exception:
            pass

    # Expand depth-2 with binary ops against raw features and unary ops
    for parent_name, parent_vals in depth2_vals.items():
        for j in range(d):
            for opname, opfn in BINARY_OPS.items():
                try:
                    vals = opfn(parent_vals, X_gpu[:, j])
                    vals = cp.nan_to_num(vals, nan=0.0, posinf=1e10, neginf=-1e10)
                    if float(cp.var(vals)) < 1e-20:
                        continue
                    cname = f"({parent_name} {opname} {feat_names[j]})"
                    eval_criterion(vals, cname)
                except Exception:
                    pass

        for uname, ufn in UNARY_OPS.items():
            try:
                vals = ufn(parent_vals)
                vals = cp.nan_to_num(vals, nan=0.0, posinf=1e10, neginf=-1e10)
                if float(cp.var(vals)) < 1e-20:
                    continue
                cname = f"{uname}({parent_name})"
                eval_criterion(vals, cname)
            except Exception:
                pass

    results.sort(key=lambda x: -x["prune_rate"])
    return results


def _eval_meta_formula(name, X_gpu, feat_names):
    """Re-evaluate a named meta-formula. Simple parser for depth-1 and depth-2."""
    d = len(feat_names)
    feat_idx = {fn: i for i, fn in enumerate(feat_names)}

    # Depth 1: raw feature
    if name in feat_idx:
        return X_gpu[:, feat_idx[name]]

    # Unary: uname(feat)
    for uname, ufn in UNARY_OPS.items():
        if name.startswith(f"{uname}(") and name.endswith(")"):
            inner = name[len(uname) + 1 : -1]
            if inner in feat_idx:
                return ufn(X_gpu[:, feat_idx[inner]])

    # Binary: (feat1 op feat2)
    if name.startswith("(") and name.endswith(")"):
        inner = name[1:-1]
        for opname, opfn in BINARY_OPS.items():
            parts = inner.split(f" {opname} ", 1)
            if len(parts) == 2:
                left, right = parts
                if left in feat_idx and right in feat_idx:
                    return opfn(X_gpu[:, feat_idx[left]], X_gpu[:, feat_idx[right]])

    return None


# ─── Pareto Frontier: epsilon sweep ───────────────────────────────


def pareto_sweep(catalog, children, prevalence):
    """Sweep epsilon values to find the Pareto frontier of pruning vs safety."""
    epsilons = [0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.10]
    log.info("    Pareto sweep: %d epsilon values", len(epsilons))

    pareto = []
    for eps in epsilons:
        cat_copy = {n: dict(info) for n, info in catalog.items()}
        cat_copy = compute_viability(cat_copy, children, epsilon=eps)

        global_best = max(v["f1"] for v in cat_copy.values())
        max_depth = max(v["depth"] for v in cat_copy.values())

        # Count alive/dead at shallow depths
        shallow = [v for v in cat_copy.values() if v["depth"] < max_depth]
        n_alive = sum(1 for v in shallow if v["alive"] == 1)
        n_dead = sum(1 for v in shallow if v["alive"] == 0)

        # Run meta-search
        criteria = deep_meta_search(cat_copy, prevalence, max_depth=max_depth)
        best_rate = criteria[0]["prune_rate"] if criteria else 0.0
        best_crit = criteria[0]["criterion"] if criteria else "none"

        pareto.append(
            {
                "epsilon": eps,
                "global_best": float(global_best),
                "n_alive": n_alive,
                "n_dead": n_dead,
                "best_prune_rate": float(best_rate),
                "best_criterion": best_crit,
            }
        )

        log.info(
            "      eps=%.3f: %d alive, %d dead, best prune=%.1f%% (%s)",
            eps,
            n_alive,
            n_dead,
            100 * best_rate,
            best_crit[:40],
        )

    return pareto


# ─── Cross-Dataset Generalization ─────────────────────────────────


def cross_validate_criterion(datasets_info, criterion_name, direction, threshold):
    """Test a pruning criterion discovered on one dataset against others."""
    results = []
    for ds_name, catalog in datasets_info:
        max_depth = max(v["depth"] for v in catalog.values())
        shallow = {n: v for n, v in catalog.items() if v["depth"] < max_depth}

        # Compute criterion value for each shallow node
        fn = 0  # false negatives
        tn = 0  # true negatives (correctly pruned dead)
        fp = 0  # false positives (wrongly kept alive dead)
        tp = 0  # true positives (correctly kept alive)

        for name, info in shallow.items():
            # Compute criterion
            val = _compute_criterion_value(info, criterion_name)
            if val is None:
                continue

            if direction == ">":
                predicted_dead = val < threshold
            else:
                predicted_dead = val > threshold

            if info["alive"] == 1:
                if predicted_dead:
                    fn += 1  # INADMISSIBLE: pruned alive subtree
                else:
                    tp += 1
            else:
                if predicted_dead:
                    tn += 1
                else:
                    fp += 1

        total_dead = tn + fp
        prune_rate = tn / total_dead if total_dead > 0 else 0

        results.append(
            {
                "dataset": ds_name,
                "false_negatives": fn,
                "true_negatives": tn,
                "prune_rate": float(prune_rate),
                "admissible": fn == 0,
            }
        )

    return results


def _compute_criterion_value(info, criterion_name):
    """Compute a meta-criterion value from node info."""
    f1 = info.get("f1", 0)
    auroc = info.get("auroc", 0.5)
    nf = info.get("n_features", 1)

    lookup = {
        "f1": f1,
        "auroc": auroc,
        "nf": float(nf),
        "f1_gap": 1.0 - f1,
        "auc_gap": 1.0 - auroc,
        "f1xa": f1 * auroc,
        "a_nf": auroc / max(nf, 0.01),
        "fg_ag": (1.0 - f1) * (1.0 - auroc),
        "depth": float(info.get("depth", 1)),
    }

    # Try raw feature
    if criterion_name in lookup:
        return lookup[criterion_name]

    # Try binary: (feat1 op feat2)
    if criterion_name.startswith("(") and criterion_name.endswith(")"):
        inner = criterion_name[1:-1]
        ops = {
            "+": lambda a, b: a + b,
            "-": lambda a, b: a - b,
            "*": lambda a, b: a * b,
            "/": lambda a, b: a / (b + 1e-30),
            "max": lambda a, b: max(a, b),
            "min": lambda a, b: min(a, b),
        }
        for opname, opfn in ops.items():
            parts = inner.split(f" {opname} ", 1)
            if len(parts) == 2 and parts[0] in lookup and parts[1] in lookup:
                return opfn(lookup[parts[0]], lookup[parts[1]])

    return None


# ─── Feature Ablation ─────────────────────────────────────────────


def ablation_study(catalog, prevalence):
    """Test each meta-feature individually and in combinations."""
    max_depth = max(v["depth"] for v in catalog.values())
    shallow = [(n, v) for n, v in catalog.items() if v["depth"] < max_depth]

    meta_feats = {
        "f1": [v["f1"] for _, v in shallow],
        "auroc": [v["auroc"] for _, v in shallow],
        "n_features": [float(v["n_features"]) for _, v in shallow],
    }
    alive = np.array([v["alive"] for _, v in shallow])

    log.info("    Feature ablation on %d shallow nodes:", len(shallow))

    for fname, vals in meta_feats.items():
        vals = np.array(vals)
        alive_vals = vals[alive == 1]
        dead_vals = vals[alive == 0]

        if len(alive_vals) == 0 or len(dead_vals) == 0:
            continue

        # Compute separation: min(alive) - max(dead) > 0 means perfectly separable
        gap = alive_vals.min() - dead_vals.min()
        overlap = max(0, dead_vals.max() - alive_vals.min())

        # Single-feature prune rate
        thresh = alive_vals.min()
        pruned = (dead_vals < thresh).sum()
        rate = pruned / len(dead_vals)

        log.info(
            "      %s: alive=[%.3f,%.3f] dead=[%.3f,%.3f] overlap=%.3f prune=%.1f%%",
            fname,
            alive_vals.min(),
            alive_vals.max(),
            dead_vals.min(),
            dead_vals.max(),
            overlap,
            100 * rate,
        )


# ─��─ Main Pipeline ─────────────────────────────────────────────────


def process_dataset(name, X, y, feat_names, max_d=10, max_depth=3):
    log.info("  " + "=" * 66)
    log.info(
        "  %s (N=%d, d=%d, prev=%.2f) — depth %d exhaustive",
        name,
        X.shape[0],
        X.shape[1],
        y.mean(),
        max_depth,
    )
    log.info("  " + "=" * 66)

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
        log.info("    Reduced to top %d features", max_d)

    X_gpu = cp.asarray(X, dtype=cp.float64)
    y_gpu = cp.asarray(y, dtype=cp.float64)
    prevalence = float(y.mean())

    t0 = time.time()
    catalog, children = exhaustive_with_lineage(X_gpu, y_gpu, feat_names, max_depth=max_depth)
    enum_time = time.time() - t0
    log.info("    %d formulas in %.0fs", len(catalog), enum_time)

    # Free values
    for n in catalog:
        if "values" in catalog[n]:
            del catalog[n]["values"]
    cp.get_default_memory_pool().free_all_blocks()

    # Viability
    catalog = compute_viability(catalog, children, epsilon=0.005)
    max_d_found = max(v["depth"] for v in catalog.values())
    for depth in range(1, max_d_found + 1):
        nodes = [(n, v) for n, v in catalog.items() if v["depth"] == depth]
        alive = sum(1 for _, v in nodes if v["alive"] == 1)
        log.info(
            "    Depth %d: %d alive / %d total (%.1f%% prunable)",
            depth,
            alive,
            len(nodes),
            100 * (1 - alive / max(len(nodes), 1)),
        )

    # Ablation
    ablation_study(catalog, prevalence)

    # Deep meta-search
    log.info("    Deep meta-search (depth-3 heuristic formulas)...")
    t1 = time.time()
    criteria = deep_meta_search(catalog, prevalence, max_depth=max_d_found)
    search_time = time.time() - t1
    log.info("    Found %d criteria in %.1fs", len(criteria), search_time)

    # Top results
    log.info("")
    log.info("    TOP 20 PRUNING CRITERIA:")
    log.info("    %-55s  %s  %-10s  %-8s", "Criterion", "Dir", "Pruned", "Rate")
    log.info("    " + "-" * 85)
    for c in criteria[:20]:
        log.info(
            "    %-55s   %s  %4d/%4d   %5.1f%%",
            c["criterion"][:55],
            c["direction"],
            c["pruned"],
            c["total_dead"],
            100 * c["prune_rate"],
        )

    # Pareto frontier
    log.info("")
    log.info("    Pareto frontier (epsilon sweep):")
    pareto = pareto_sweep(catalog, children, prevalence)

    return {
        "name": name,
        "catalog_size": len(catalog),
        "criteria": criteria[:50],
        "pareto": pareto,
        "catalog": catalog,  # for cross-validation
    }


def main():
    log.info("=" * 70)
    log.info("META THEORY RADAR v3 — DEEP SEARCH")
    log.info("Depth-3 heuristic formulas | Pareto sweep | Cross-dataset")
    log.info("=" * 70)

    props = cp.cuda.runtime.getDeviceProperties(0)
    log.info("GPU: %s (%.1f GB free)", props["name"].decode(), cp.cuda.Device(0).mem_info[0] / 1e9)

    datasets = []

    bc = load_breast_cancer()
    X = StandardScaler().fit_transform(bc.data[:, :10])
    datasets.append(("BreastCancer-10D", X, bc.target, [f"f{i}" for i in range(10)]))

    wine = load_wine()
    X = StandardScaler().fit_transform(wine.data)
    y = (wine.target == 0).astype(int)
    datasets.append(("Wine-10D", X, y, [f"w{i}" for i in range(X.shape[1])]))

    try:
        pima = fetch_openml("diabetes", version=1, as_frame=False, parser="auto")
        X = StandardScaler().fit_transform(pima.data.astype(float))
        y = (pima.target == "tested_positive").astype(int)
        datasets.append(("Diabetes-8D", X, y, [f"d{i}" for i in range(8)]))
    except Exception as e:
        log.warning("Diabetes: %s", e)

    # Also try Iris (easy, d=4) for depth-4 exhaustive
    iris = load_iris()
    X = StandardScaler().fit_transform(iris.data)
    y = (iris.target == 2).astype(int)  # virginica vs rest
    datasets.append(("Iris-4D", X, y, [f"i{i}" for i in range(4)]))

    # Process each dataset
    all_results = []
    for name, X, y, feats in datasets:
        # Use depth 4 for small datasets (d<=6)
        md = 4 if X.shape[1] <= 6 else 3
        result = process_dataset(name, X, y, feats, max_d=10, max_depth=md)
        all_results.append(result)

    # ── Cross-Dataset Generalization ──
    log.info("")
    log.info("=" * 70)
    log.info("CROSS-DATASET GENERALIZATION")
    log.info("=" * 70)

    # Collect top criteria from each dataset
    all_criteria = {}
    for r in all_results:
        for c in r["criteria"][:5]:
            key = c["criterion"]
            if key not in all_criteria:
                all_criteria[key] = c

    # Test each criterion on all datasets
    datasets_info = [(r["name"], r["catalog"]) for r in all_results]

    log.info("Testing %d criteria across %d datasets:", len(all_criteria), len(datasets_info))
    log.info("")

    for crit_name, crit_info in list(all_criteria.items())[:15]:
        xval = cross_validate_criterion(
            datasets_info, crit_name, crit_info["direction"], crit_info["threshold"]
        )
        admissible_all = all(r["admissible"] for r in xval)
        avg_prune = np.mean([r["prune_rate"] for r in xval])

        log.info(
            "  %-50s  avg_prune=%.1f%%  universal=%s",
            crit_name[:50],
            100 * avg_prune,
            "YES"
            if admissible_all
            else "NO (FN: "
            + ",".join(
                f"{r['dataset']}:{r['false_negatives']}" for r in xval if not r["admissible"]
            )
            + ")",
        )

        for r in xval:
            log.info(
                "    %s: prune=%.1f%% FN=%d %s",
                r["dataset"],
                100 * r["prune_rate"],
                r["false_negatives"],
                "✓" if r["admissible"] else "✗",
            )

    # Save (without catalog to keep file size reasonable)
    save_results = []
    for r in all_results:
        save_r = {k: v for k, v in r.items() if k != "catalog"}
        save_results.append(save_r)

    with open("meta_radar_v3_results.json", "w") as f:
        json.dump(save_results, f, indent=2, default=str)
    log.info("\nSaved meta_radar_v3_results.json")


if __name__ == "__main__":
    main()
