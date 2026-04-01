#!/usr/bin/env python3
"""
Large-scale prediction test with richer features.

- 50K configs (5x previous)
- Multiple mass ratios (not just equal masses)
- Higher-order features: Tucker mode ranks, block structure, energy
- Recursive rank: compute rank of inner subsystem separately
- Tighter periodicity threshold (2.5% instead of 5%)
"""

from __future__ import annotations

import logging
import time

import numpy as np

from tensor_3body.hamiltonian import hessian_analytical, hamiltonian
from tensor_3body.sampling import config_to_phase_space_circular
from tensor_3body.tensor_ops import effective_rank, block_structure, participation_ratio
from tensor_3body.integrator_gpu import integrate_batch

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


def spectral_gap_ratio(H):
    sv = np.linalg.svd(H, compute_uv=False)
    if sv[0] < 1e-30:
        return 0.0
    gaps = sv[:-1] - sv[1:]
    return float(gaps.max() / sv[0])


def characteristic_frequencies(H):
    qq = H[0:6, 0:6]
    eigenvalues = np.linalg.eigvalsh(qq)
    return np.sort(np.sqrt(np.abs(eigenvalues)))[::-1]


def frequency_cluster_count(freqs, ratio_threshold=10.0):
    pos = freqs[freqs > 1e-10]
    if len(pos) <= 1:
        return len(pos)
    sorted_f = np.sort(pos)[::-1]
    clusters = 1
    for i in range(len(sorted_f) - 1):
        if sorted_f[i] / (sorted_f[i + 1] + 1e-30) > ratio_threshold:
            clusters += 1
    return clusters


def inner_subsystem_rank(H):
    """Rank of just the inner pair (rho1, pi1) 6x6 subblock."""
    inner = np.zeros((6, 6))
    inner[0:3, 0:3] = H[0:3, 0:3]  # rho1-rho1
    inner[3:6, 3:6] = H[6:9, 6:9]  # pi1-pi1
    inner[0:3, 3:6] = H[0:3, 6:9]  # rho1-pi1
    inner[3:6, 0:3] = H[6:9, 0:3]  # pi1-rho1
    return effective_rank(inner)


def compute_features(z, m1, m2, m3, r1, r2):
    """Compute full feature vector for one configuration."""
    H = hessian_analytical(z, m1, m2, m3)
    E = hamiltonian(z, m1, m2, m3)
    freqs = characteristic_frequencies(H)
    pos_freqs = freqs[freqs > 1e-10]
    bs = block_structure(H)

    return {
        "rank": effective_rank(H),
        "gamma": spectral_gap_ratio(H),
        "pr": participation_ratio(H),
        "freq_ratio": pos_freqs[0] / pos_freqs[-1] if len(pos_freqs) >= 2 else 1.0,
        "n_clusters": frequency_cluster_count(freqs),
        "n_clusters_5": frequency_cluster_count(freqs, 5.0),
        "n_clusters_20": frequency_cluster_count(freqs, 20.0),
        "inner_rank": inner_subsystem_rank(H),
        "cross_body_q": bs["cross_body_position"],
        "is_block_diag": bs["is_block_diagonal"],
        "is_separable": bs["is_separable"],
        "energy": E,
        "r2_over_r1": r2 / (r1 + 1e-10),
        "r_total": r1 + r2,
        "r_product": r1 * r2,
    }


def main():
    log.info("=" * 70)
    log.info("LARGE-SCALE PREDICTION TEST: 50K configs, 3 mass ratios")
    log.info("=" * 70)

    rng = np.random.RandomState(42)
    N_per_mass = 17000  # ~50K total

    mass_scenarios = [
        ("equal", 1.0, 1.0, 1.0),
        ("binary_light", 1.0, 1.0, 0.01),
        ("hierarchical", 1.0, 0.1, 0.01),
    ]

    all_z = []
    all_features = []
    all_meta = []

    for scenario_name, m1, m2, m3 in mass_scenarios:
        log.info(
            "\n--- Generating %d configs for %s (%.1f, %.1f, %.1f) ---",
            N_per_mass,
            scenario_name,
            m1,
            m2,
            m3,
        )

        r1_vals = 10 ** rng.uniform(-1, 2, N_per_mass)
        r2_vals = 10 ** rng.uniform(-1, 2, N_per_mass)
        theta_vals = rng.uniform(0.01, np.pi, N_per_mass)
        phi_vals = rng.uniform(0, 2 * np.pi, N_per_mass)

        t0 = time.time()
        for i in range(N_per_mass):
            z = config_to_phase_space_circular(
                r1_vals[i], r2_vals[i], theta_vals[i], phi_vals[i], m1, m2, m3
            )
            all_z.append(z)

            feat = compute_features(z, m1, m2, m3, r1_vals[i], r2_vals[i])
            feat["scenario"] = scenario_name
            all_features.append(feat)

            all_meta.append(
                {
                    "r1": r1_vals[i],
                    "r2": r2_vals[i],
                    "theta": theta_vals[i],
                    "phi": phi_vals[i],
                    "m1": m1,
                    "m2": m2,
                    "m3": m3,
                    "scenario": scenario_name,
                }
            )

        elapsed = time.time() - t0
        log.info("  Features computed in %.1f s (%.0f/s)", elapsed, N_per_mass / elapsed)

        # Feature summary for this scenario
        scenario_feats = all_features[-N_per_mass:]
        ranks = [f["rank"] for f in scenario_feats]
        inner_ranks = [f["inner_rank"] for f in scenario_feats]
        clusters = [f["n_clusters"] for f in scenario_feats]
        log.info(
            "  rank: mean=%.1f, inner_rank: mean=%.1f, clusters: mean=%.1f",
            np.mean(ranks),
            np.mean(inner_ranks),
            np.mean(clusters),
        )

    N = len(all_z)
    z0 = np.array(all_z)
    log.info("\nTotal: %d configurations", N)

    # Build prediction arrays
    ranks = np.array([f["rank"] for f in all_features])
    inner_ranks = np.array([f["inner_rank"] for f in all_features])
    gammas = np.array([f["gamma"] for f in all_features])
    freq_ratios = np.array([f["freq_ratio"] for f in all_features])
    n_clusters = np.array([f["n_clusters"] for f in all_features])
    n_clusters_5 = np.array([f["n_clusters_5"] for f in all_features])
    cross_body = np.array([f["cross_body_q"] for f in all_features])
    energies = np.array([f["energy"] for f in all_features])
    r2_r1 = np.array([f["r2_over_r1"] for f in all_features])
    is_separable = np.array([f["is_separable"] for f in all_features])
    pr = np.array([f["pr"] for f in all_features])
    scenarios = np.array([f["scenario"] for f in all_features])

    # Predictions
    predictions = {}
    predictions["clusters>=2"] = n_clusters >= 2
    predictions["clusters>=2_tight5"] = n_clusters_5 >= 2
    predictions["rank<=10"] = ranks <= 10
    predictions["freq_ratio>10"] = freq_ratios > 10
    predictions["separable"] = is_separable

    # NEW: inner_rank based (addresses false positive problem)
    predictions["clusters>=2_AND_inner>=4"] = (n_clusters >= 2) & (inner_ranks >= 4)
    predictions["clusters>=2_AND_inner>=5"] = (n_clusters >= 2) & (inner_ranks >= 5)

    # Energy window
    predictions["clusters>=2_AND_E>-3"] = (n_clusters >= 2) & (energies > -3) & (energies < -0.3)

    # Goldilocks ratio
    predictions["clusters>=2_AND_ratio_10_100"] = (n_clusters >= 2) & (r2_r1 > 10) & (r2_r1 < 100)

    # Combined best
    predictions["BEST_combo"] = (
        (n_clusters >= 2) & (inner_ranks >= 4) & (energies > -3) & (energies < -0.1)
    )

    log.info("\nPredictions made. Integrating on GPU...")

    # Integrate in batches (50K might exceed GPU memory)
    batch_size = 10000
    all_periodic = np.zeros(N, dtype=bool)
    all_collision = np.zeros(N, dtype=bool)
    all_ret_dist = np.zeros(N)

    t0 = time.time()
    for batch_start in range(0, N, batch_size):
        batch_end = min(batch_start + batch_size, N)
        batch_z = z0[batch_start:batch_end]

        # Get masses for this batch
        batch_m1 = all_meta[batch_start]["m1"]
        batch_m2 = all_meta[batch_start]["m2"]
        batch_m3 = all_meta[batch_start]["m3"]

        # Check if batch crosses scenario boundary
        batch_scenario = scenarios[batch_start]
        same_scenario = all(scenarios[i] == batch_scenario for i in range(batch_start, batch_end))

        if not same_scenario:
            # Split at boundary and integrate separately
            for i in range(batch_start, batch_end):
                if scenarios[i] != batch_scenario:
                    # Integrate first part
                    if i > batch_start:
                        sub_z = z0[batch_start:i]
                        r = integrate_batch(
                            sub_z, batch_m1, batch_m2, batch_m3, dt=0.005, n_steps=80000, gpu_id=0
                        )
                        all_periodic[batch_start:i] = r["is_periodic"]
                        all_collision[batch_start:i] = r["collision"]
                        all_ret_dist[batch_start:i] = r["return_distance"]

                    # Switch scenario
                    batch_start = i
                    batch_m1 = all_meta[i]["m1"]
                    batch_m2 = all_meta[i]["m2"]
                    batch_m3 = all_meta[i]["m3"]
                    batch_scenario = scenarios[i]

            # Integrate remaining
            sub_z = z0[batch_start:batch_end]
            r = integrate_batch(
                sub_z, batch_m1, batch_m2, batch_m3, dt=0.005, n_steps=80000, gpu_id=0
            )
            all_periodic[batch_start:batch_end] = r["is_periodic"]
            all_collision[batch_start:batch_end] = r["collision"]
            all_ret_dist[batch_start:batch_end] = r["return_distance"]
        else:
            r = integrate_batch(
                batch_z, batch_m1, batch_m2, batch_m3, dt=0.005, n_steps=80000, gpu_id=0
            )
            all_periodic[batch_start:batch_end] = r["is_periodic"]
            all_collision[batch_start:batch_end] = r["collision"]
            all_ret_dist[batch_start:batch_end] = r["return_distance"]

        log.info("  Batch %d-%d done", batch_start, batch_end)

    t_integrate = time.time() - t0
    valid = ~all_collision
    actual = all_periodic & valid

    log.info(
        "\nIntegration: %.1f s, %d/%d periodic (%.2f%%), %d collisions",
        t_integrate,
        actual.sum(),
        valid.sum(),
        100 * actual.sum() / valid.sum(),
        all_collision.sum(),
    )

    # Score per scenario
    log.info("\n" + "=" * 70)
    log.info("RESULTS BY SCENARIO")
    log.info("=" * 70)

    for scenario in ["equal", "binary_light", "hierarchical"]:
        s_mask = scenarios == scenario
        s_valid = s_mask & valid
        n_p = (actual & s_mask).sum()
        n_v = s_valid.sum()
        log.info(
            "\n%s: %d/%d periodic (%.2f%%)", scenario, n_p, n_v, 100 * n_p / n_v if n_v > 0 else 0
        )

        for name, pred in predictions.items():
            pred_s = pred & s_valid
            actual_s = actual & s_valid
            tp = int((pred_s & actual_s).sum())
            fp = int((pred_s & ~actual_s).sum())
            fn = int((~pred_s & actual_s).sum())
            tn = int((~pred_s & ~actual_s).sum())
            prec = tp / (tp + fp) if (tp + fp) > 0 else 0
            rec = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
            if f1 > 0.01:
                log.info(
                    "  %-35s prec=%5.1f%% rec=%5.1f%% F1=%5.1f%%",
                    name,
                    100 * prec,
                    100 * rec,
                    100 * f1,
                )

    # Overall
    log.info("\n" + "=" * 70)
    log.info("OVERALL RESULTS")
    log.info("=" * 70)
    log.info("%-35s  %6s %6s %6s", "Rule", "Prec", "Recall", "F1")
    for name, pred in predictions.items():
        pred_v = pred[valid]
        actual_v = actual[valid]
        tp = int((pred_v & actual_v).sum())
        fp = int((pred_v & ~actual_v).sum())
        fn = int((~pred_v & actual_v).sum())
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
        log.info("%-35s  %5.1f%% %5.1f%% %5.1f%%", name, 100 * prec, 100 * rec, 100 * f1)

    # Save
    np.savez_compressed(
        "results/prediction_large.npz",
        ranks=ranks,
        inner_ranks=inner_ranks,
        gammas=gammas,
        freq_ratios=freq_ratios,
        n_clusters=n_clusters,
        energies=energies,
        r2_r1=r2_r1,
        is_separable=is_separable,
        scenarios=scenarios,
        periodic=all_periodic,
        collision=all_collision,
        return_distance=all_ret_dist,
    )
    log.info("\nSaved to results/prediction_large.npz")


if __name__ == "__main__":
    main()
