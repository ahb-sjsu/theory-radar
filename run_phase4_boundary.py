#!/usr/bin/env python3
"""
Phase 4b: Search the rank 9-10 boundary for periodic orbits.

These are the transition configurations where periodicity is partial
(12-21%) — the interesting zone between fully stable hierarchical
orbits and chaotic generic configurations.
"""

from __future__ import annotations

import logging
import numpy as np

from tensor_3body.landscape import load_landscape
from tensor_3body.sampling import config_to_phase_space_circular
from tensor_3body.integrator_gpu import integrate_batch

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


def main():
    log.info("=" * 60)
    log.info("Phase 4b: Rank 9-10 Boundary Search")
    log.info("=" * 60)

    data, masses = load_landscape("results/phase1/landscape_equal_static.npz")
    m1, m2, m3 = float(masses[0]), float(masses[1]), float(masses[2])

    # Select rank 9-10 configs
    mask = (data["eff_rank"] >= 9) & (data["eff_rank"] <= 10)
    indices = np.where(mask)[0]
    log.info("Rank 9-10 configs: %d", len(indices))

    # Sample 5000
    if len(indices) > 5000:
        indices = np.random.choice(indices, size=5000, replace=False)
    log.info("Sampled %d configs", len(indices))

    # Build initial conditions
    z0_list = []
    ranks = []
    configs = []
    for idx in indices:
        r1 = float(data["r1"][idx])
        r2 = float(data["r2"][idx])
        theta = float(data["theta"][idx])
        phi = float(data["phi"][idx])
        z = config_to_phase_space_circular(r1, r2, theta, phi, m1, m2, m3)
        z0_list.append(z)
        ranks.append(int(data["eff_rank"][idx]))
        configs.append((r1, r2, theta, phi))

    z0 = np.array(z0_list)
    log.info("Built %d initial conditions", len(z0))

    # Integrate with longer time for longer-period orbits
    log.info("Integrating on GPU 0 (dt=0.005, steps=100000, T=500)...")
    result = integrate_batch(z0, m1, m2, m3, dt=0.005, n_steps=100000, gpu_id=0)

    # Results
    valid = ~result["collision"]
    n_valid = int(valid.sum())
    n_periodic = int(result["is_periodic"][valid].sum())
    n_collision = int(result["collision"].sum())

    log.info("=" * 60)
    log.info("RESULTS: %d valid, %d collisions, %d periodic (%.2f%%)",
             n_valid, n_collision, n_periodic,
             100 * n_periodic / n_valid if n_valid > 0 else 0)

    # By rank
    for r in [9, 10]:
        rmask = np.array(ranks) == r
        v = rmask & valid
        n_p = int(result["is_periodic"][v].sum())
        n_t = int(v.sum())
        log.info("  rank=%d: %d/%d periodic (%.1f%%)",
                 r, n_p, n_t, 100 * n_p / n_t if n_t > 0 else 0)

    # Top 20 closest returns
    vi = np.where(valid)[0]
    order = np.argsort(result["return_distance"][vi])
    log.info("\nTop 20 closest returns:")
    for i in order[:20]:
        idx = vi[i]
        r1, r2, theta, phi = configs[idx]
        log.info("  r1=%.3f r2=%.3f th=%.3f phi=%.3f rank=%d  "
                 "ret=%.4f t=%.1f periodic=%s E=%.1e",
                 r1, r2, theta, phi, ranks[idx],
                 result["return_distance"][idx],
                 result["return_time"][idx],
                 result["is_periodic"][idx],
                 result["energy_error"][idx])

    # Find configurations where periodicity breaks — the boundary
    # Group by (r1, r2) bins and find bins with mixed periodic/non-periodic
    log.info("\nBoundary analysis: (r1,r2) bins with mixed periodicity")
    r1_bins = np.logspace(-1, 2, 20)
    r2_bins = np.logspace(-1, 2, 20)

    for i in range(len(r1_bins) - 1):
        for j in range(len(r2_bins) - 1):
            bin_mask = (
                valid &
                (np.array([c[0] for c in configs]) >= r1_bins[i]) &
                (np.array([c[0] for c in configs]) < r1_bins[i+1]) &
                (np.array([c[1] for c in configs]) >= r2_bins[j]) &
                (np.array([c[1] for c in configs]) < r2_bins[j+1])
            )
            n_in_bin = int(bin_mask.sum())
            if n_in_bin < 5:
                continue
            n_per = int(result["is_periodic"][bin_mask].sum())
            frac = n_per / n_in_bin
            # Interesting = mixed (not 0% and not 100%)
            if 0.05 < frac < 0.95:
                log.info("  r1=[%.1f,%.1f] r2=[%.1f,%.1f]: %d/%d periodic (%.0f%%)",
                         r1_bins[i], r1_bins[i+1],
                         r2_bins[j], r2_bins[j+1],
                         n_per, n_in_bin, 100 * frac)

    # Save
    np.savez_compressed(
        "results/phase4_rank910.npz",
        return_distance=result["return_distance"],
        is_periodic=result["is_periodic"],
        collision=result["collision"],
        ranks=np.array(ranks),
        r1=np.array([c[0] for c in configs]),
        r2=np.array([c[1] for c in configs]),
    )
    log.info("Saved to results/phase4_rank910.npz")


if __name__ == "__main__":
    main()
