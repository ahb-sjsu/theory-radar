#!/usr/bin/env python3
"""
GPU-vectorized formula search with expanded operation space.

Key improvements over v1:
- Features as CuPy arrays (10K x 21) — formula evaluation is GPU-vectorized
- Conditional operations: if(a > b, c, d), max, min, abs
- Piecewise: threshold(x, t) = 1 if x > t else 0
- Deeper search: depth 4, 50K states
- All F1 evaluations on GPU
"""

from __future__ import annotations

import heapq
import logging
import time

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

try:
    import cupy as cp
    HAS_CUPY = True
except ImportError:
    HAS_CUPY = False

from tensor_3body.hamiltonian import hessian_analytical, hamiltonian
from tensor_3body.tensor_ops import effective_rank, singular_values, participation_ratio
from tensor_3body.sampling import config_to_phase_space_circular
from tensor_3body.integrator_gpu import integrate_batch


def compute_features_batch(r1_vals, r2_vals, theta_vals, phi_vals, m1, m2, m3):
    """Compute all features as numpy arrays (N,) — vectorized where possible."""
    N = len(r1_vals)
    feat = {name: np.zeros(N) for name in [
        "rank", "qq_rank", "inner_rank", "max_gap", "qq_max_gap",
        "freq_ratio", "n_clusters", "entropy", "energy",
        "r2_r1", "r1", "r2", "kepler_deviation", "pr",
        "max_eig", "min_nonzero_eig", "qq_gap_over_max",
        "log_r2_r1", "log_freq_ratio", "inv_r1", "r1_cubed_inv",
        "binding_energy", "n_clusters_sq",
    ]}

    for i in range(N):
        z = config_to_phase_space_circular(r1_vals[i], r2_vals[i], theta_vals[i], phi_vals[i], m1, m2, m3)
        H = hessian_analytical(z, m1, m2, m3)
        eigs = np.sort(np.linalg.eigvalsh(H))
        sv = singular_values(H)
        qq = H[0:6, 0:6]
        qq_eigs = np.sort(np.linalg.eigvalsh(qq))
        freqs = np.sort(np.sqrt(np.abs(qq_eigs)))[::-1]
        pos_freqs = freqs[freqs > 1e-10]
        gaps = np.diff(eigs)

        E = hamiltonian(z, m1, m2, m3)

        feat["rank"][i] = effective_rank(H)
        feat["qq_rank"][i] = np.sum(np.abs(qq_eigs) > 1e-6 * max(np.abs(qq_eigs).max(), 1e-30))
        feat["max_gap"][i] = gaps.max()
        feat["qq_max_gap"][i] = np.diff(np.sort(np.abs(qq_eigs))).max() if len(qq_eigs) > 1 else 0
        feat["freq_ratio"][i] = pos_freqs[0] / (pos_freqs[-1] + 1e-30) if len(pos_freqs) >= 2 else 1.0

        n_cl = 1
        for k in range(len(pos_freqs) - 1):
            if pos_freqs[k] / (pos_freqs[k+1] + 1e-30) > 10:
                n_cl += 1
        feat["n_clusters"][i] = n_cl

        sv_pos = sv[sv > 1e-30]
        if len(sv_pos) > 0:
            p = sv_pos / sv_pos.sum()
            feat["entropy"][i] = -np.sum(p * np.log(p + 1e-30))

        feat["energy"][i] = E
        feat["r2_r1"][i] = r2_vals[i] / (r1_vals[i] + 1e-10)
        feat["r1"][i] = r1_vals[i]
        feat["r2"][i] = r2_vals[i]

        nonzero_qq = np.abs(qq_eigs[np.abs(qq_eigs) > 1e-6 * max(np.abs(qq_eigs).max(), 1e-30)])
        if len(nonzero_qq) >= 2:
            sorted_nz = np.sort(nonzero_qq)[::-1]
            feat["kepler_deviation"][i] = abs(sorted_nz[0] / (sorted_nz[1] + 1e-30) - 2.0)
        else:
            feat["kepler_deviation"][i] = 99

        feat["pr"][i] = participation_ratio(H)
        feat["max_eig"][i] = eigs[-1]
        nonzero_abs = np.abs(eigs[np.abs(eigs) > 1e-10])
        feat["min_nonzero_eig"][i] = nonzero_abs.min() if len(nonzero_abs) > 0 else 0
        feat["qq_gap_over_max"][i] = feat["qq_max_gap"][i] / (np.abs(qq_eigs).max() + 1e-30)

        # Derived features
        feat["log_r2_r1"][i] = np.log10(feat["r2_r1"][i] + 1)
        feat["log_freq_ratio"][i] = np.log10(feat["freq_ratio"][i] + 1)
        feat["inv_r1"][i] = 1.0 / (r1_vals[i] + 1e-10)
        feat["r1_cubed_inv"][i] = 1.0 / (r1_vals[i]**3 + 1e-10)
        feat["binding_energy"][i] = -E if E < 0 else 0
        feat["n_clusters_sq"][i] = n_cl * n_cl

        inner = np.zeros((6, 6))
        inner[0:3, 0:3] = H[0:3, 0:3]
        inner[3:6, 3:6] = H[6:9, 6:9]
        feat["inner_rank"][i] = effective_rank(inner)

    return feat


def gpu_f1_sweep(values_gpu, actual_gpu, n_actual, percentiles_gpu):
    """GPU-vectorized F1 computation across multiple thresholds simultaneously."""
    N = len(values_gpu)
    best_f1 = 0.0

    for pct_idx in range(len(percentiles_gpu)):
        thresh = percentiles_gpu[pct_idx]

        # Above threshold
        pred = values_gpu > thresh
        tp = int((pred & actual_gpu).sum())
        fp = int((pred & ~actual_gpu).sum())
        fn = n_actual - tp
        if (tp + fp) > 0 and (tp + fn) > 0:
            prec = tp / (tp + fp)
            rec = tp / (tp + fn)
            if (prec + rec) > 0:
                f1 = 2 * prec * rec / (prec + rec)
                best_f1 = max(best_f1, f1)

        # Below threshold
        pred = values_gpu < thresh
        tp = int((pred & actual_gpu).sum())
        fp = int((pred & ~actual_gpu).sum())
        fn = n_actual - tp
        if (tp + fp) > 0 and (tp + fn) > 0:
            prec = tp / (tp + fp)
            rec = tp / (tp + fn)
            if (prec + rec) > 0:
                f1 = 2 * prec * rec / (prec + rec)
                best_f1 = max(best_f1, f1)

    return best_f1


def main():
    log.info("=" * 70)
    log.info("GPU FORMULA SEARCH: Expanded operations, deeper search")
    log.info("=" * 70)

    m1, m2, m3 = 1.0, 1.0, 1.0
    N = 15000
    rng = np.random.RandomState(7777)

    r1_vals = 10 ** rng.uniform(-0.5, 2.0, N)
    r2_vals = 10 ** rng.uniform(-0.5, 2.0, N)
    theta_vals = rng.uniform(0.01, np.pi, N)
    phi_vals = rng.uniform(0, 2 * np.pi, N)

    log.info("Computing features for %d configs...", N)
    t0 = time.time()
    feat = compute_features_batch(r1_vals, r2_vals, theta_vals, phi_vals, m1, m2, m3)
    log.info("Features: %.1f s", time.time() - t0)

    # Integrate for ground truth
    z0 = np.array([
        config_to_phase_space_circular(r1_vals[i], r2_vals[i], theta_vals[i], phi_vals[i], m1, m2, m3)
        for i in range(N)
    ])
    log.info("Integrating %d orbits on GPU...", N)
    result = integrate_batch(z0, m1, m2, m3, dt=0.005, n_steps=80000, gpu_id=0)

    valid = ~result["collision"]
    actual = result["is_periodic"] & valid
    log.info("Ground truth: %d/%d periodic (%.2f%%)",
             actual.sum(), valid.sum(), 100 * actual.sum() / valid.sum())

    # Build feature matrix on GPU
    feature_names = list(feat.keys())
    n_features = len(feature_names)

    # Filter to valid
    X = np.column_stack([feat[name][valid] for name in feature_names])
    y = actual[valid]
    X = np.nan_to_num(X, nan=0, posinf=1e10, neginf=-1e10)

    N_valid = len(y)
    n_periodic = int(y.sum())
    log.info("Valid: %d, periodic: %d, features: %d", N_valid, n_periodic, n_features)

    # Precompute percentiles for threshold sweep
    percentiles = np.array([20, 30, 40, 50, 60, 70, 75, 80, 85, 90, 95])

    # GPU arrays
    if HAS_CUPY:
        X_gpu = cp.asarray(X, dtype=cp.float64)
        y_gpu = cp.asarray(y, dtype=cp.bool_)
    else:
        X_gpu = X
        y_gpu = y

    # ================================================================
    # EXPANDED SEARCH: evaluate formulas as vectorized array operations
    # ================================================================
    log.info("\nPhase 1: Single features...")
    results = []

    for i, name in enumerate(feature_names):
        vals = X[:, i]
        pcts = np.percentile(vals[np.isfinite(vals)], percentiles) if np.any(np.isfinite(vals)) else percentiles * 0
        f1 = gpu_f1_sweep(
            cp.asarray(vals) if HAS_CUPY else vals,
            y_gpu, n_periodic,
            cp.asarray(pcts) if HAS_CUPY else pcts,
        )
        results.append((f1, name))

    results.sort(key=lambda x: -x[0])
    log.info("Top 5 single features:")
    for f1, name in results[:5]:
        log.info("  F1=%.3f  %s", f1, name)

    # Phase 2: All pairwise combinations with expanded operations
    log.info("\nPhase 2: Pairwise combinations (%d x %d x 10 ops = %d formulas)...",
             n_features, n_features, n_features * n_features * 10)

    ops = {
        "+": lambda a, b: a + b,
        "-": lambda a, b: a - b,
        "*": lambda a, b: a * b,
        "/": lambda a, b: a / (b + 1e-30),
        "max": lambda a, b: np.maximum(a, b),
        "min": lambda a, b: np.minimum(a, b),
        "hypot": lambda a, b: np.sqrt(a**2 + b**2),
        "diff_sq": lambda a, b: (a - b)**2,
        "harmonic": lambda a, b: 2*a*b / (a + b + 1e-30),
        "geometric": lambda a, b: np.sign(a*b) * np.sqrt(np.abs(a*b)),
    }

    t0 = time.time()
    pair_results = []

    for i in range(n_features):
        for j in range(n_features):
            if i == j:
                continue
            a = X[:, i]
            b = X[:, j]
            for op_name, op_fn in ops.items():
                try:
                    vals = op_fn(a, b)
                    vals = np.nan_to_num(vals, nan=0, posinf=1e10, neginf=-1e10)
                    finite = np.isfinite(vals)
                    if finite.sum() < N_valid * 0.5:
                        continue
                    pcts = np.percentile(vals[finite], percentiles)

                    if HAS_CUPY:
                        f1 = gpu_f1_sweep(cp.asarray(vals), y_gpu, n_periodic, cp.asarray(pcts))
                    else:
                        f1 = gpu_f1_sweep(vals, y_gpu, n_periodic, pcts)

                    if f1 > 0.3:
                        pair_results.append((f1, f"{feature_names[i]} {op_name} {feature_names[j]}"))
                except Exception:
                    continue

    elapsed = time.time() - t0
    pair_results.sort(key=lambda x: -x[0])
    log.info("Pairwise search: %.1f s, %d formulas with F1 > 0.3", elapsed, len(pair_results))
    log.info("Top 10 pairwise:")
    for f1, desc in pair_results[:10]:
        log.info("  F1=%.3f  %s", f1, desc)

    # Phase 3: Top pairwise + unary ops
    log.info("\nPhase 3: Unary transforms of top pairwise formulas...")
    unary_ops = {
        "log": lambda x: np.log(np.abs(x) + 1e-30),
        "sqrt": lambda x: np.sqrt(np.abs(x)),
        "sq": lambda x: x**2,
        "inv": lambda x: 1.0 / (x + 1e-30),
        "neg": lambda x: -x,
        "abs": lambda x: np.abs(x),
        "sigmoid": lambda x: 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500))),
        "tanh": lambda x: np.tanh(np.clip(x, -500, 500)),
    }

    # Get the top 50 pairwise formulas and apply unary ops
    top_pairs = pair_results[:50]
    phase3_results = []

    for _, pair_desc in top_pairs:
        # Recompute the values
        parts = pair_desc.split(" ")
        fname_a = parts[0]
        op_name = parts[1]
        fname_b = parts[2]
        ia = feature_names.index(fname_a)
        ib = feature_names.index(fname_b)
        base_vals = ops[op_name](X[:, ia], X[:, ib])
        base_vals = np.nan_to_num(base_vals, nan=0, posinf=1e10, neginf=-1e10)

        for uname, ufn in unary_ops.items():
            try:
                vals = ufn(base_vals)
                vals = np.nan_to_num(vals, nan=0, posinf=1e10, neginf=-1e10)
                finite = np.isfinite(vals)
                if finite.sum() < N_valid * 0.5:
                    continue
                pcts = np.percentile(vals[finite], percentiles)
                if HAS_CUPY:
                    f1 = gpu_f1_sweep(cp.asarray(vals), y_gpu, n_periodic, cp.asarray(pcts))
                else:
                    f1 = gpu_f1_sweep(vals, y_gpu, n_periodic, pcts)

                if f1 > 0.3:
                    phase3_results.append((f1, f"{uname}({pair_desc})"))
            except Exception:
                continue

    phase3_results.sort(key=lambda x: -x[0])
    log.info("Phase 3: %d formulas with F1 > 0.3", len(phase3_results))
    log.info("Top 10 depth-3:")
    for f1, desc in phase3_results[:10]:
        log.info("  F1=%.3f  %s", f1, desc)

    # Phase 4: Triple combinations of top features
    log.info("\nPhase 4: Triple feature combinations...")
    top_single = [name for _, name in results[:8]]  # top 8 features
    triple_results = []

    for i, fa in enumerate(top_single):
        ia = feature_names.index(fa)
        for j, fb in enumerate(top_single):
            if j <= i:
                continue
            ib = feature_names.index(fb)
            for k, fc in enumerate(top_single):
                if k <= j:
                    continue
                ic = feature_names.index(fc)
                a, b, c = X[:, ia], X[:, ib], X[:, ic]

                # Try various 3-feature combinations
                combos = [
                    (f"{fa}+{fb}+{fc}", a + b + c),
                    (f"{fa}*{fb}+{fc}", a * b + c),
                    (f"{fa}+{fb}*{fc}", a + b * c),
                    (f"{fa}*{fb}*{fc}", a * b * c),
                    (f"({fa}+{fb})/{fc}", (a + b) / (c + 1e-30)),
                    (f"{fa}/({fb}+{fc})", a / (b + c + 1e-30)),
                    (f"({fa}-{fb})*{fc}", (a - b) * c),
                    (f"max({fa},{fb})*{fc}", np.maximum(a, b) * c),
                    (f"{fa}*{fb}/{fc}", a * b / (c + 1e-30)),
                ]

                for desc, vals in combos:
                    try:
                        vals = np.nan_to_num(vals, nan=0, posinf=1e10, neginf=-1e10)
                        finite = np.isfinite(vals)
                        if finite.sum() < N_valid * 0.5:
                            continue
                        pcts = np.percentile(vals[finite], percentiles)
                        if HAS_CUPY:
                            f1 = gpu_f1_sweep(cp.asarray(vals), y_gpu, n_periodic, cp.asarray(pcts))
                        else:
                            f1 = gpu_f1_sweep(vals, y_gpu, n_periodic, pcts)
                        if f1 > 0.3:
                            triple_results.append((f1, desc))
                    except Exception:
                        continue

    triple_results.sort(key=lambda x: -x[0])
    log.info("Phase 4: %d formulas with F1 > 0.3", len(triple_results))
    log.info("Top 10 triples:")
    for f1, desc in triple_results[:10]:
        log.info("  F1=%.3f  %s", f1, desc)

    # Grand summary
    all_results = results + pair_results + phase3_results + triple_results
    all_results.sort(key=lambda x: -x[0])

    log.info("\n" + "=" * 70)
    log.info("GRAND TOP 20 (all phases)")
    log.info("=" * 70)
    seen = set()
    count = 0
    for f1, desc in all_results:
        if desc not in seen and count < 20:
            log.info("  F1=%.3f  %s", f1, desc)
            seen.add(desc)
            count += 1

    best_f1, best_desc = all_results[0]
    log.info("\n" + "=" * 70)
    log.info("BEST DISCOVERED FORMULA: %s", best_desc)
    log.info("F1 = %.3f", best_f1)
    log.info("=" * 70)


if __name__ == "__main__":
    main()
