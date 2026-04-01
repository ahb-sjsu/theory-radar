#!/usr/bin/env python3
"""
Massive orbit search: use batch-probe to maximize GPU utilization,
integrate as many orbits as GPU 0 can handle simultaneously.
"""

from __future__ import annotations

import logging
import numpy as np

from tensor_3body.integrator_gpu import integrate_batch, probe_max_orbits
from tensor_3body.sampling import config_to_phase_space_circular
from tensor_3body.landscape import load_landscape

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


def main():
    log.info("=" * 60)
    log.info("MASSIVE ORBIT SEARCH with batch-probe auto-sizing")
    log.info("=" * 60)

    # Probe max batch size
    max_n = probe_max_orbits(1.0, 1.0, 1.0, gpu_id=0)
    log.info("Max orbits per GPU batch: %d", max_n)

    # Load landscape
    data, masses = load_landscape("results/phase1/landscape_equal_static.npz")
    m1, m2, m3 = float(masses[0]), float(masses[1]), float(masses[2])

    # Sample up to max_n configs
    N = min(max_n, 100000)
    rng = np.random.RandomState(2026)
    indices = rng.choice(len(data["eff_rank"]), size=N, replace=False)
    log.info("Selected %d configs from landscape", N)

    # Build initial conditions
    z0_list = []
    ranks = []
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
        z0_list.append(z)
        ranks.append(int(data["eff_rank"][idx]))

    z0 = np.array(z0_list)
    ranks = np.array(ranks)
    log.info("Built %d initial conditions", len(z0))

    # Integrate
    result = integrate_batch(z0, m1, m2, m3, dt=0.005, n_steps=80000, gpu_id=0, auto_batch=True)

    # Results
    valid = ~result["collision"]
    periodic = result["is_periodic"] & valid
    log.info("=" * 60)
    log.info(
        "RESULTS: %d/%d periodic (%.2f%%), %d collisions",
        periodic.sum(),
        valid.sum(),
        100 * periodic.sum() / valid.sum() if valid.sum() > 0 else 0,
        result["collision"].sum(),
    )

    for r in sorted(set(ranks)):
        rmask = (ranks == r) & valid
        if rmask.sum() < 10:
            continue
        n_p = (periodic & (ranks == r)).sum()
        log.info("  rank=%d: %d/%d periodic (%.2f%%)", r, n_p, rmask.sum(), 100 * n_p / rmask.sum())

    np.savez_compressed(
        "results/massive_orbit_search.npz",
        ranks=ranks,
        periodic=result["is_periodic"],
        collision=result["collision"],
        return_distance=result["return_distance"],
    )
    log.info("Saved to results/massive_orbit_search.npz")


if __name__ == "__main__":
    main()
