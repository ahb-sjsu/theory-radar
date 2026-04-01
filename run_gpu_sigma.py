#!/usr/bin/env python3
"""GPU-accelerated Theory Radar significance testing.

Fully vectorized: candidate generation AND F1 evaluation on GPU via broadcasting.
No Python loops over individual formulas — everything is batched tensor ops.

Uses CuPy on GPU 0 for batched beam search.
Uses joblib for parallel sklearn baselines on all CPU cores.
200x5 = 1000 folds for high-sigma results on real-world datasets.
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

from sklearn.datasets import load_breast_cancer, load_wine, fetch_openml
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger()


# ─── Vectorized GPU Binary Ops (broadcasting: (N,B,1) op (N,1,d) → (N,B,d)) ──


def _vbin_add(B, F):
    return B + F


def _vbin_sub(B, F):
    return B - F


def _vbin_mul(B, F):
    return B * F


def _vbin_div(B, F):
    return B / (F + 1e-30)


def _vbin_max(B, F):
    return cp.maximum(B, F)


def _vbin_min(B, F):
    return cp.minimum(B, F)


def _vbin_hypot(B, F):
    return cp.sqrt(B**2 + F**2)


def _vbin_diffsq(B, F):
    return (B - F) ** 2


def _vbin_harmonic(B, F):
    return 2 * B * F / (B + F + 1e-30)


def _vbin_geometric(B, F):
    return cp.sign(B * F) * cp.sqrt(cp.abs(B * F))


VBINARY = [
    ("+", _vbin_add),
    ("-", _vbin_sub),
    ("*", _vbin_mul),
    ("/", _vbin_div),
    ("max", _vbin_max),
    ("min", _vbin_min),
    ("hypot", _vbin_hypot),
    ("diff_sq", _vbin_diffsq),
    ("harmonic", _vbin_harmonic),
    ("geometric", _vbin_geometric),
]


def _vun_log(V):
    return cp.log(cp.abs(V) + 1e-30)


def _vun_sqrt(V):
    return cp.sqrt(cp.abs(V))


def _vun_sq(V):
    return V**2


def _vun_abs(V):
    return cp.abs(V)


def _vun_sigmoid(V):
    return 1.0 / (1.0 + cp.exp(-cp.clip(V, -500, 500)))


def _vun_tanh(V):
    return cp.tanh(cp.clip(V, -500, 500))


VUNARY = [
    ("log", _vun_log),
    ("sqrt", _vun_sqrt),
    ("sq", _vun_sq),
    ("abs", _vun_abs),
    ("sigmoid", _vun_sigmoid),
    ("tanh", _vun_tanh),
]


# ─── GPU Batched F1 (fully vectorized, no Python loops) ───────────


def gpu_batch_f1(vals, labels):
    """Optimal thresholded F1 for each column. vals: (N,K), labels: (N,) float."""
    N, K = vals.shape
    pos = labels.sum()
    if float(pos) == 0 or float(pos) == N:
        return cp.zeros(K, dtype=cp.float64)

    best = _sweep_f1(vals, labels, pos)
    best2 = _sweep_f1(-vals, labels, pos)
    return cp.maximum(best, best2)


def _sweep_f1(vals, labels, pos):
    """Descending sort-and-sweep F1 for all K columns.

    Only considers thresholds at positions where the sorted value changes,
    preventing spurious F1=1.0 from constant-valued formulas.
    """
    N, K = vals.shape
    idx = cp.argsort(-vals, axis=0)

    # Sorted values and labels
    sv = cp.take_along_axis(vals, idx, axis=0)  # (N, K) sorted vals descending
    sl = labels[idx]  # (N, K) sorted labels

    tp = cp.cumsum(sl, axis=0, dtype=cp.float64)
    fp = cp.cumsum(1.0 - sl, axis=0, dtype=cp.float64)
    denom = tp + pos + fp
    f1 = cp.where(denom > 0, 2.0 * tp / denom, 0.0)

    # Mask: only valid thresholds where value changes between positions
    # A threshold at position i means "predict positive for positions 0..i"
    # This is only meaningful if sv[i] > sv[i+1] (a real boundary exists)
    # For the last position (predict ALL positive), always valid
    valid = cp.ones((N, K), dtype=cp.bool_)
    valid[:-1, :] = (sv[:-1, :] - sv[1:, :]) > 1e-15  # value must change

    f1 = cp.where(valid, f1, 0.0)
    return cp.max(f1, axis=0)


# ─── Fully Vectorized GPU Beam Search ─────────────────────────────


def gpu_beam_search(X_gpu, y_gpu, feat_names, max_depth=3, beam_width=100):
    """Beam search with fully vectorized candidate generation + F1 eval.

    At each depth:
    1. Binary ops: (N,B,1) op (N,1,d) → (N,B,d) per op → reshape (N, B*d*10)
    2. Unary ops: (N,B) per op → stack (N, B*6)
    3. Concatenate → (N, total_K), batch F1 in one call
    4. Keep top beam_width by F1

    Returns: (best_f1, best_formula_name)
    """
    N, d = X_gpu.shape
    y_f = y_gpu.astype(cp.float64)

    # Depth 1: raw features
    f1s = gpu_batch_f1(X_gpu, y_f)
    order = cp.argsort(-f1s).get()

    B = min(beam_width, d)
    beam_vals = cp.empty((N, B), dtype=cp.float64)
    beam_names = []
    for rank, i in enumerate(order[:B]):
        beam_vals[:, rank] = X_gpu[:, i]
        beam_names.append(feat_names[i])

    best_f1 = float(f1s[order[0]])
    best_name = beam_names[0]

    for depth in range(2, max_depth + 1):
        B_cur = beam_vals.shape[1]

        # Broadcast shapes: beams (N, B, 1) and features (N, 1, d)
        bv3 = beam_vals[:, :, None]  # (N, B, 1)
        fv3 = X_gpu[:, None, :]  # (N, 1, d)

        chunks = []  # list of (N, K_chunk) arrays
        name_chunks = []  # parallel list of name lists

        # Binary ops — each produces (N, B, d), reshape to (N, B*d)
        for opname, opfn in VBINARY:
            try:
                out = opfn(bv3, fv3)  # (N, B, d)
                out = cp.nan_to_num(out, nan=0.0, posinf=1e10, neginf=-1e10)
                out = out.reshape(N, B_cur * d)  # (N, B*d)
                chunks.append(out)

                names = []
                for bi in range(B_cur):
                    for fi in range(d):
                        names.append(f"({beam_names[bi]} {opname} {feat_names[fi]})")
                name_chunks.append(names)
            except Exception:
                pass

        # Unary ops — each produces (N, B)
        for opname, opfn in VUNARY:
            try:
                out = opfn(beam_vals)  # (N, B)
                out = cp.nan_to_num(out, nan=0.0, posinf=1e10, neginf=-1e10)
                chunks.append(out)

                names = [f"{opname}({beam_names[bi]})" for bi in range(B_cur)]
                name_chunks.append(names)
            except Exception:
                pass

        if not chunks:
            break

        # Concatenate all candidates → (N, total_K)
        all_vals = cp.concatenate(chunks, axis=1)
        all_names = []
        for nc in name_chunks:
            all_names.extend(nc)
        total_K = all_vals.shape[1]

        del chunks, bv3, fv3
        cp.get_default_memory_pool().free_all_blocks()

        # Zero out constant-valued candidates (no discriminative power)
        var = cp.var(all_vals, axis=0)  # (total_K,)
        constant_mask = var < 1e-20

        # Batch F1 — single GPU call for ALL candidates
        f1s = gpu_batch_f1(all_vals, y_f)  # (total_K,)
        f1s = cp.where(constant_mask, 0.0, f1s)  # kill constant formulas

        # Keep top beam_width
        top_k = min(beam_width, total_K)
        top_idx = cp.argsort(-f1s)[:top_k]
        top_idx_cpu = top_idx.get()
        top_f1_cpu = f1s[top_idx].get()

        beam_vals = cp.empty((N, top_k), dtype=cp.float64)
        beam_names = []
        for rank, i in enumerate(top_idx_cpu):
            beam_vals[:, rank] = all_vals[:, i]
            beam_names.append(all_names[i])
            f1_val = float(top_f1_cpu[rank])
            if f1_val > best_f1:
                best_f1 = f1_val
                best_name = all_names[i]

        del all_vals, f1s
        cp.get_default_memory_pool().free_all_blocks()

    return best_f1, best_name


# ─── CPU Baselines (parallel with joblib) ─────────────────────────


def run_sklearn_fold(X, y, tr, te, fold_i):
    """3 sklearn baselines for one fold."""
    X_tr, X_te, y_tr, y_te = X[tr], X[te], y[tr], y[te]
    out = {}
    for name, clf in [
        ("GB", GradientBoostingClassifier(n_estimators=100, max_depth=4, random_state=42 + fold_i)),
        ("RF", RandomForestClassifier(n_estimators=100, random_state=42 + fold_i)),
        ("LR", LogisticRegression(max_iter=1000, random_state=42)),
    ]:
        clf.fit(X_tr, y_tr)
        out[name] = f1_score(y_te, clf.predict(X_te))
    return out


# ─── Main Pipeline ─────────────────────────────────────────────────


def run_dataset(name, X, y, features, n_repeats=200, n_folds=5):
    total = n_repeats * n_folds
    log.info("=" * 70)
    log.info(
        "%s (N=%d, d=%d, prev=%.0f%%) — %dx%d = %d folds",
        name,
        X.shape[0],
        X.shape[1],
        100 * y.mean(),
        n_repeats,
        n_folds,
        total,
    )
    log.info("=" * 70)

    cv = RepeatedStratifiedKFold(n_splits=n_folds, n_repeats=n_repeats, random_state=42)
    splits = list(cv.split(X, y))

    # Phase 1: GPU beam search
    X_gpu = cp.asarray(X, dtype=cp.float64)
    y_gpu = cp.asarray(y, dtype=cp.float64)

    astar_f1s = []
    formulas = []
    t0 = time.time()

    for fold_i, (tr, te) in enumerate(splits):
        tr_gpu = cp.asarray(tr)
        f1, formula = gpu_beam_search(
            X_gpu[tr_gpu], y_gpu[tr_gpu], features, max_depth=3, beam_width=100
        )
        astar_f1s.append(f1)
        formulas.append(formula)

        if (fold_i + 1) % 100 == 0:
            elapsed = time.time() - t0
            rate = (fold_i + 1) / elapsed
            log.info(
                "  GPU %d/%d  A*=%.3f  %.1f folds/s  ETA %.0fs",
                fold_i + 1,
                total,
                np.mean(astar_f1s[-100:]),
                rate,
                (total - fold_i - 1) / rate,
            )

    gpu_time = time.time() - t0
    log.info("  GPU done: %d folds in %.0fs (%.1f folds/s)", total, gpu_time, total / gpu_time)

    del X_gpu, y_gpu
    cp.get_default_memory_pool().free_all_blocks()

    # Phase 2: CPU baselines (parallel)
    t1 = time.time()
    n_jobs = min(os.cpu_count() or 1, 20)  # cap at 20 to keep CPU temps under control
    log.info("  CPU baselines: %d folds x %d workers...", total, n_jobs)

    baseline_results = Parallel(n_jobs=n_jobs, backend="loky", verbose=0)(
        delayed(run_sklearn_fold)(X, y, tr, te, fi) for fi, (tr, te) in enumerate(splits)
    )
    cpu_time = time.time() - t1
    log.info("  CPU done: %.0fs (%.1f folds/s)", cpu_time, total / cpu_time)

    # Phase 3: Stats
    astar_f1s = np.array(astar_f1s)
    results = {}
    for bname in ["GB", "RF", "LR"]:
        bf1s = np.array([r[bname] for r in baseline_results])
        diffs = astar_f1s - bf1s
        t_stat, p_value = stats.ttest_1samp(diffs, 0)
        results[bname] = {
            "mean": float(bf1s.mean()),
            "std": float(bf1s.std()),
            "diff": float(diffs.mean()),
            "sigma": float(abs(t_stat)),
            "p": float(p_value),
            "dir": "A*>" if diffs.mean() > 0 else f"{bname}>",
        }

    fc = Counter(formulas)
    top = fc.most_common(1)[0]

    summary = {
        "name": name,
        "N": int(X.shape[0]),
        "d": int(X.shape[1]),
        "n_folds": total,
        "astar_mean": float(astar_f1s.mean()),
        "astar_std": float(astar_f1s.std()),
        "baselines": results,
        "formula": top[0],
        "stability": float(top[1] / len(formulas)),
        "gpu_time_s": float(gpu_time),
        "cpu_time_s": float(cpu_time),
    }

    log.info("  FORMULA: %s (%.0f%% stable)", top[0][:60], 100 * summary["stability"])
    log.info("  A*:  %.4f +/- %.4f", summary["astar_mean"], summary["astar_std"])
    for bname in ["GB", "RF", "LR"]:
        b = results[bname]
        log.info(
            "  vs %-3s: %.4f +/- %.4f  diff=%+.4f  %.1fσ  p=%.2e  %s",
            bname,
            b["mean"],
            b["std"],
            b["diff"],
            b["sigma"],
            b["p"],
            b["dir"],
        )
    return summary


def main():
    log.info("=" * 70)
    log.info("GPU THEORY RADAR — FULLY VECTORIZED BEAM SEARCH")
    log.info("GPU 0: beam search | %d CPUs: sklearn baselines", os.cpu_count())
    log.info("=" * 70)

    props = cp.cuda.runtime.getDeviceProperties(0)
    log.info("GPU: %s (%.1f GB free)", props["name"].decode(), cp.cuda.Device(0).mem_info[0] / 1e9)

    # Warmup
    _ = cp.sort(cp.random.rand(10000, 100), axis=0)
    cp.cuda.Stream.null.synchronize()
    log.info("GPU warmup done")

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
        datasets.append(("Diabetes", X, y, [f"d{i}" for i in range(X.shape[1])]))
    except Exception as e:
        log.warning("Diabetes: %s", e)

    try:
        bn = fetch_openml("banknote-authentication", version=1, as_frame=False, parser="auto")
        X = StandardScaler().fit_transform(bn.data.astype(float))
        y_raw = bn.target.astype(int)
        y = (y_raw == y_raw.max()).astype(int)  # ensure binary 0/1
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

    log.info("Loaded %d datasets", len(datasets))

    all_results = []
    for name, X, y, features in datasets:
        result = run_dataset(name, X, y, features, n_repeats=200, n_folds=5)
        all_results.append(result)

    with open("gpu_sigma_results.json", "w") as f:
        json.dump(all_results, f, indent=2)
    log.info("Saved gpu_sigma_results.json")

    # Final table
    log.info("")
    log.info("=" * 90)
    log.info("FINAL SUMMARY — 200x5=1000 folds per dataset")
    log.info(
        "%-15s  %-8s  %-25s  %-8s  %-8s  %-8s",
        "Dataset",
        "A* F1",
        "Formula",
        "vs GB σ",
        "vs RF σ",
        "vs LR σ",
    )
    log.info("-" * 90)
    for r in all_results:
        log.info(
            "%-15s  %.4f   %-25s  %6.1fσ   %6.1fσ   %6.1fσ",
            r["name"],
            r["astar_mean"],
            r["formula"][:25],
            r["baselines"]["GB"]["sigma"],
            r["baselines"]["RF"]["sigma"],
            r["baselines"]["LR"]["sigma"],
        )
    log.info("=" * 90)


if __name__ == "__main__":
    main()
