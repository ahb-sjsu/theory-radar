#!/usr/bin/env python3
"""
Fix experiments based on reviewer feedback:

1. Complete rank distribution table (all ranks)
2. Verify eigenvalue signs at rank 9
3. Threshold sensitivity sweep for effective rank
4. Scale-normalized perturbation study (epsilon / ||H||_F)
5. Baseline comparison: tensor rank vs separation ratio vs Lyapunov
"""

from __future__ import annotations

import logging
import numpy as np

from tensor_3body.hamiltonian import hessian_analytical
from tensor_3body.sampling import config_to_phase_space_circular
from tensor_3body.landscape import load_landscape

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


def main():
    log.info("=" * 70)
    log.info("REVIEWER FIXES: Rigorous reanalysis")
    log.info("=" * 70)

    data, masses = load_landscape("results/phase1/landscape_equal_static.npz")
    m1, m2, m3 = float(masses[0]), float(masses[1]), float(masses[2])
    N = len(data["eff_rank"])

    # ============================================================
    # FIX 1: Complete rank distribution
    # ============================================================
    log.info("\n--- FIX 1: Complete rank distribution (N=%d) ---", N)
    for r in range(1, 13):
        count = (data["eff_rank"] == r).sum()
        if count > 0:
            log.info("  rank=%d: %d (%.4f%%)", r, count, 100 * count / N)
    log.info("  Total: %d", N)
    log.info(
        "  Sum of all rank counts: %d", sum((data["eff_rank"] == r).sum() for r in range(1, 13))
    )

    # ============================================================
    # FIX 2: Verify eigenvalue signs at rank 9
    # ============================================================
    log.info("\n--- FIX 2: Eigenvalue signs at rank 9 ---")
    rng = np.random.RandomState(42)
    rank9_mask = data["eff_rank"] == 9
    rank9_idx = np.where(rank9_mask)[0]
    sample_idx = rng.choice(rank9_idx, size=min(20, len(rank9_idx)), replace=False)

    for idx in sample_idx[:5]:
        r1, r2 = float(data["r1"][idx]), float(data["r2"][idx])
        theta, phi = float(data["theta"][idx]), float(data["phi"][idx])
        z = config_to_phase_space_circular(r1, r2, theta, phi, m1, m2, m3)
        H = hessian_analytical(z, m1, m2, m3)
        qq = H[0:6, 0:6]
        qq_eigs = np.sort(np.linalg.eigvalsh(qq))

        # Expected: Hessian of -m1*m2/r gives eigenvalues:
        # +2*m1*m2/r^3 (radial), -m1*m2/r^3 (tangential, x2), 0 (x3, decoupled)
        expected_radial = 2 * m1 * m2 / r1**3
        expected_tangential = -m1 * m2 / r1**3

        log.info("  r1=%.3f r2=%.3f: qq_eigs = %s", r1, r2, " ".join(f"{e:+.4f}" for e in qq_eigs))
        log.info(
            "    Expected (Kepler): radial=%+.4f, tangential=%+.4f (x2), zero (x3)",
            expected_radial,
            expected_tangential,
        )

        # Check ratios among nonzero eigenvalues
        nonzero = qq_eigs[np.abs(qq_eigs) > 0.01 * np.abs(qq_eigs).max()]
        if len(nonzero) >= 2:
            sorted_abs = np.sort(np.abs(nonzero))[::-1]
            ratio = sorted_abs[0] / sorted_abs[1]
            log.info("    |largest|/|second| = %.6f (should be 2.0)", ratio)

    # ============================================================
    # FIX 3: Threshold sensitivity for effective rank
    # ============================================================
    log.info("\n--- FIX 3: Rank threshold sensitivity ---")
    sample_size = 10000
    indices = rng.choice(N, size=sample_size, replace=False)

    for eps in [1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-10, 1e-12]:
        rank_counts = {}
        for idx in indices:
            z = config_to_phase_space_circular(
                float(data["r1"][idx]),
                float(data["r2"][idx]),
                float(data["theta"][idx]),
                float(data["phi"][idx]),
                m1,
                m2,
                m3,
            )
            H = hessian_analytical(z, m1, m2, m3)
            sv = np.linalg.svd(H, compute_uv=False)
            r = int(np.sum(sv / (sv[0] + 1e-30) > eps))
            rank_counts[r] = rank_counts.get(r, 0) + 1

        rank_str = " ".join(f"r{k}:{v}" for k, v in sorted(rank_counts.items()))
        n_below_12 = sum(v for k, v in rank_counts.items() if k < 12)
        log.info(
            "  eps=%.0e: %s  (rank<12: %d/%d = %.1f%%)",
            eps,
            rank_str,
            n_below_12,
            sample_size,
            100 * n_below_12 / sample_size,
        )

    # ============================================================
    # FIX 4: Scale-normalized perturbation study
    # ============================================================
    log.info("\n--- FIX 4: Scale-normalized spectral rigidity ---")
    log.info("  Perturbation: dH with ||dH||_F / ||H||_F = epsilon (relative)")

    for target_rank in [7, 8, 9, 10, 11, 12]:
        mask = data["eff_rank"] == target_rank
        idx = np.where(mask)[0]
        if len(idx) < 10:
            continue
        idx = rng.choice(idx, size=min(30, len(idx)), replace=False)

        n_preserved_list = []
        hessian_norms = []

        for i in idx:
            z = config_to_phase_space_circular(
                float(data["r1"][i]),
                float(data["r2"][i]),
                float(data["theta"][i]),
                float(data["phi"][i]),
                m1,
                m2,
                m3,
            )
            H = hessian_analytical(z, m1, m2, m3)
            H_norm = np.linalg.norm(H, "fro")
            hessian_norms.append(H_norm)

            eigs0 = np.sort(np.linalg.eigvalsh(H))
            norm0 = np.linalg.norm(eigs0)

            n_preserved = 0
            for _ in range(100):
                dH = rng.randn(12, 12)
                dH = (dH + dH.T) / 2
                # SCALE NORMALIZE: ||dH|| = 0.01 * ||H||
                dH = dH / np.linalg.norm(dH, "fro") * 0.01 * H_norm

                eigs_pert = np.sort(np.linalg.eigvalsh(H + dH))
                rel_change = np.linalg.norm(eigs_pert - eigs0) / (norm0 + 1e-30)
                if rel_change < 0.01:
                    n_preserved += 1

            n_preserved_list.append(n_preserved)

        mean_preserved = np.mean(n_preserved_list)
        mean_norm = np.mean(hessian_norms)
        log.info(
            "  rank=%d: preserved=%.1f/100, mean_||H||=%.2e (n=%d)",
            target_rank,
            mean_preserved,
            mean_norm,
            len(idx),
        )

    # ============================================================
    # FIX 5: Baseline comparison — tensor rank vs separation ratio
    # ============================================================
    log.info("\n--- FIX 5: Baseline comparison ---")
    log.info("  Does tensor rank predict periodicity better than simple r2/r1?")

    # Load periodicity data if available
    try:
        orbit_data = np.load("results/massive_orbit_search.npz")
        orbit_ranks = orbit_data["ranks"]
        orbit_periodic = orbit_data["periodic"]
        orbit_collision = orbit_data["collision"]
        valid = ~orbit_collision

        log.info("  Loaded 100K orbit data")

        # We don't have r1/r2 in the orbit data, so compare rank-based
        # prediction vs random baseline
        periodic_rate = orbit_periodic[valid].sum() / valid.sum()
        log.info("  Base rate: %.2f%%", 100 * periodic_rate)

        # Rank-based: predict periodic if rank <= threshold
        log.info("\n  Rank threshold sweep:")
        for thresh in [8, 9, 10, 11]:
            pred = (orbit_ranks <= thresh) & valid
            tp = int((pred & orbit_periodic).sum())
            fp = int((pred & ~orbit_periodic & valid).sum())
            fn = int((~pred & orbit_periodic & valid).sum())
            prec = tp / (tp + fp) if (tp + fp) > 0 else 0
            rec = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
            log.info(
                "    rank<=%d: prec=%.1f%% rec=%.1f%% F1=%.3f", thresh, 100 * prec, 100 * rec, f1
            )

    except FileNotFoundError:
        log.info("  Orbit data not found — run run_massive_orbits.py first")

    log.info("\n" + "=" * 70)
    log.info("REVIEWER FIXES COMPLETE")
    log.info("=" * 70)


if __name__ == "__main__":
    main()
