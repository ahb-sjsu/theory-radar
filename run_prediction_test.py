#!/usr/bin/env python3
"""
Prediction test: Can spectral gap γ predict periodicity WITHOUT integration?

Protocol:
1. Sample 10,000 NEW configurations (not from Phase 1 landscape)
2. Compute γ and tensor rank at each (fast, no integration)
3. PREDICT which are periodic based on γ > threshold
4. Integrate ALL on GPU to get ground truth
5. Measure prediction accuracy (precision, recall, F1, ROC-AUC)

This is the falsifiable test that turns correlation into prediction.
"""

from __future__ import annotations

import logging
import time

import numpy as np

from tensor_3body.hamiltonian import hessian_analytical
from tensor_3body.sampling import config_to_phase_space_circular
from tensor_3body.tensor_ops import effective_rank
from tensor_3body.integrator_gpu import integrate_batch

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


def spectral_gap_ratio(H: np.ndarray) -> float:
    sv = np.linalg.svd(H, compute_uv=False)
    if sv[0] < 1e-30:
        return 0.0
    gaps = sv[:-1] - sv[1:]
    return float(gaps.max() / sv[0])


def characteristic_frequencies(H: np.ndarray) -> np.ndarray:
    qq = H[0:6, 0:6]
    eigenvalues = np.linalg.eigvalsh(qq)
    return np.sort(np.sqrt(np.abs(eigenvalues)))[::-1]


def frequency_cluster_count(freqs: np.ndarray, ratio_threshold: float = 10.0) -> int:
    """Count frequency clusters separated by ratio > threshold."""
    pos = freqs[freqs > 1e-10]
    if len(pos) <= 1:
        return len(pos)
    sorted_f = np.sort(pos)[::-1]
    clusters = 1
    for i in range(len(sorted_f) - 1):
        if sorted_f[i] / (sorted_f[i+1] + 1e-30) > ratio_threshold:
            clusters += 1
    return clusters


def main():
    log.info("=" * 70)
    log.info("PREDICTION TEST: Can tensor properties predict periodicity?")
    log.info("=" * 70)

    m1, m2, m3 = 1.0, 1.0, 1.0
    N = 10000
    rng = np.random.RandomState(2026)

    # Step 1: Generate NEW random configurations
    # Uniform in log(r1), log(r2), uniform in angles
    log.info("\nStep 1: Generating %d new random configurations...", N)
    r1_vals = 10 ** rng.uniform(-1, 2, N)       # 0.1 to 100
    r2_vals = 10 ** rng.uniform(-1, 2, N)       # 0.1 to 100
    theta_vals = rng.uniform(0.01, np.pi, N)
    phi_vals = rng.uniform(0, 2 * np.pi, N)

    # Step 2: Compute tensor properties (no integration)
    log.info("Step 2: Computing tensor properties (γ, rank, frequencies)...")
    t0 = time.time()

    gammas = np.zeros(N)
    ranks = np.zeros(N, dtype=int)
    freq_ratios = np.zeros(N)
    n_clusters = np.zeros(N, dtype=int)
    z0_list = []

    for i in range(N):
        z = config_to_phase_space_circular(
            r1_vals[i], r2_vals[i], theta_vals[i], phi_vals[i], m1, m2, m3
        )
        z0_list.append(z)

        H = hessian_analytical(z, m1, m2, m3)
        gammas[i] = spectral_gap_ratio(H)
        ranks[i] = effective_rank(H)
        freqs = characteristic_frequencies(H)
        pos = freqs[freqs > 1e-10]
        freq_ratios[i] = pos[0] / pos[-1] if len(pos) >= 2 else 1.0
        n_clusters[i] = frequency_cluster_count(freqs)

    z0 = np.array(z0_list)
    t_features = time.time() - t0
    log.info("  Computed in %.1f s (%.0f configs/s)", t_features, N / t_features)

    # Feature summary
    log.info("\n  Feature distributions:")
    log.info("  γ: mean=%.4f, std=%.4f, min=%.4f, max=%.4f",
             gammas.mean(), gammas.std(), gammas.min(), gammas.max())
    log.info("  rank: mean=%.1f, min=%d, max=%d",
             ranks.mean(), ranks.min(), ranks.max())
    log.info("  freq_ratio: mean=%.1f, median=%.1f, max=%.1f",
             freq_ratios.mean(), np.median(freq_ratios), freq_ratios.max())
    log.info("  n_clusters: %s", dict(zip(*np.unique(n_clusters, return_counts=True))))

    # Step 3: Make predictions BEFORE integration
    log.info("\nStep 3: Making predictions...")

    # Multiple prediction rules to test
    predictions = {}

    # Rule 1: rank <= 10
    predictions["rank<=10"] = ranks <= 10

    # Rule 2: rank <= 9
    predictions["rank<=9"] = ranks <= 9

    # Rule 3: γ > 0.7
    predictions["gamma>0.7"] = gammas > 0.7

    # Rule 4: γ > 0.6
    predictions["gamma>0.6"] = gammas > 0.6

    # Rule 5: freq_ratio > 100
    predictions["freq_ratio>100"] = freq_ratios > 100

    # Rule 6: freq_ratio > 10
    predictions["freq_ratio>10"] = freq_ratios > 10

    # Rule 7: n_clusters >= 2
    predictions["clusters>=2"] = n_clusters >= 2

    # Rule 8: Combined: rank <= 10 AND γ > 0.6
    predictions["rank<=10_AND_gamma>0.6"] = (ranks <= 10) & (gammas > 0.6)

    # Rule 9: Combined: rank <= 10 AND freq_ratio > 50
    predictions["rank<=10_AND_fratio>50"] = (ranks <= 10) & (freq_ratios > 50)

    for name, pred in predictions.items():
        log.info("  %-30s predicts %d/%d periodic (%.1f%%)",
                 name, pred.sum(), N, 100 * pred.sum() / N)

    # Step 4: Integrate ALL on GPU (ground truth)
    log.info("\nStep 4: Integrating all %d orbits on GPU (ground truth)...", N)
    t0 = time.time()
    result = integrate_batch(z0, m1, m2, m3, dt=0.005, n_steps=100000, gpu_id=0)
    t_integrate = time.time() - t0

    actual = result["is_periodic"] & ~result["collision"]
    n_actual_periodic = int(actual.sum())
    n_collision = int(result["collision"].sum())

    log.info("  Integration: %.1f s (%.0f orbits/s)", t_integrate, N / t_integrate)
    log.info("  Ground truth: %d/%d periodic (%.2f%%), %d collisions",
             n_actual_periodic, N, 100 * n_actual_periodic / N, n_collision)

    # Step 5: Score each prediction rule
    log.info("\n" + "=" * 70)
    log.info("RESULTS: Prediction Accuracy")
    log.info("=" * 70)
    log.info("%-30s  %6s %6s %6s %6s %6s",
             "Rule", "Prec", "Recall", "F1", "Acc", "Predicted")

    valid = ~result["collision"]
    actual_valid = actual[valid]

    best_f1 = 0
    best_rule = ""

    for name, pred in predictions.items():
        pred_valid = pred[valid]

        tp = int((pred_valid & actual_valid).sum())
        fp = int((pred_valid & ~actual_valid).sum())
        fn = int((~pred_valid & actual_valid).sum())
        tn = int((~pred_valid & ~actual_valid).sum())

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        accuracy = (tp + tn) / (tp + fp + fn + tn)

        log.info("%-30s  %5.1f%% %5.1f%% %5.1f%% %5.1f%% %6d",
                 name, 100*precision, 100*recall, 100*f1, 100*accuracy,
                 int(pred_valid.sum()))

        if f1 > best_f1:
            best_f1 = f1
            best_rule = name

    log.info("\nBest rule: %s (F1=%.1f%%)", best_rule, 100*best_f1)

    # ROC-like analysis: sweep γ threshold
    log.info("\n--- γ threshold sweep ---")
    log.info("%-10s  %6s %6s %6s", "γ_thresh", "Prec", "Recall", "F1")
    for threshold in [0.3, 0.4, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.9]:
        pred_t = gammas[valid] > threshold
        tp = int((pred_t & actual_valid).sum())
        fp = int((pred_t & ~actual_valid).sum())
        fn = int((~pred_t & actual_valid).sum())
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
        log.info("%-10.2f  %5.1f%% %5.1f%% %5.1f%%", threshold, 100*prec, 100*rec, 100*f1)

    # Rank sweep
    log.info("\n--- Rank threshold sweep ---")
    log.info("%-10s  %6s %6s %6s", "rank<=", "Prec", "Recall", "F1")
    for max_rank in [7, 8, 9, 10, 11]:
        pred_r = ranks[valid] <= max_rank
        tp = int((pred_r & actual_valid).sum())
        fp = int((pred_r & ~actual_valid).sum())
        fn = int((~pred_r & actual_valid).sum())
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
        log.info("%-10d  %5.1f%% %5.1f%% %5.1f%%", max_rank, 100*prec, 100*rec, 100*f1)

    # Save everything
    np.savez_compressed(
        "results/prediction_test.npz",
        r1=r1_vals, r2=r2_vals, theta=theta_vals, phi=phi_vals,
        gammas=gammas, ranks=ranks, freq_ratios=freq_ratios,
        n_clusters=n_clusters,
        is_periodic=result["is_periodic"],
        collision=result["collision"],
        return_distance=result["return_distance"],
    )
    log.info("\nSaved to results/prediction_test.npz")

    # Timing comparison
    log.info("\n--- Timing ---")
    log.info("  Feature computation (γ, rank, freqs): %.1f s for %d configs", t_features, N)
    log.info("  GPU integration (ground truth):       %.1f s for %d configs", t_integrate, N)
    log.info("  Speedup of prediction over integration: %.0fx", t_integrate / t_features)


if __name__ == "__main__":
    main()
