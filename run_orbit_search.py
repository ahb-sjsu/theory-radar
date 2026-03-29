#!/usr/bin/env python3
"""
Orbit search: Integrate low-rank configurations to find periodic orbits.

Takes the rank-10 (and lower) configurations from Phase 1 and integrates
them forward in time to check if they lie near periodic orbits.

Usage:
    python run_orbit_search.py --landscape results/phase1/landscape_equal_static.npz
    python run_orbit_search.py --landscape results/phase1/landscape_equal_static.npz --max-rank 10 --workers 44
"""

from __future__ import annotations

import argparse
import logging
import time
from multiprocessing import Pool, cpu_count
from pathlib import Path

import numpy as np

from tensor_3body.integrator import integrate
from tensor_3body.sampling import config_to_phase_space, config_to_phase_space_circular
from tensor_3body.landscape import load_landscape

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


def _integrate_one(args: tuple) -> dict:
    """Worker: integrate one configuration."""
    idx, r1, r2, theta, phi, m1, m2, m3, use_circular, dt, n_steps = args
    try:
        if use_circular:
            z0 = config_to_phase_space_circular(r1, r2, theta, phi, m1, m2, m3)
        else:
            z0 = config_to_phase_space(r1, r2, theta, phi, m1, m2, m3)

        result = integrate(z0, m1, m2, m3, dt=dt, n_steps=n_steps, save_every=100)

        return {
            "idx": idx,
            "r1": r1, "r2": r2, "theta": theta, "phi": phi,
            "return_distance": result["return_distance"],
            "return_time": result["return_time"],
            "is_periodic": result["is_periodic"],
            "collision": result["collision"],
            "energy_error": result["energy_error"],
            "error": None,
        }
    except Exception as e:
        return {"idx": idx, "r1": r1, "r2": r2, "theta": theta, "phi": phi,
                "error": str(e)}


def main():
    parser = argparse.ArgumentParser(description="Orbit search at low-rank configurations")
    parser.add_argument("--landscape", type=str, required=True,
                        help="Path to Phase 1 landscape NPZ file")
    parser.add_argument("--max-rank", type=int, default=10,
                        help="Search configs with rank <= this (default: 10)")
    parser.add_argument("--max-configs", type=int, default=5000,
                        help="Maximum configs to integrate (default: 5000)")
    parser.add_argument("--dt", type=float, default=0.005,
                        help="Integration timestep (default: 0.005)")
    parser.add_argument("--n-steps", type=int, default=50000,
                        help="Number of integration steps (default: 50000)")
    parser.add_argument("--circular", action="store_true",
                        help="Use circular orbit initial momenta")
    parser.add_argument("--workers", type=int, default=None)

    args = parser.parse_args()
    n_workers = args.workers or min(cpu_count(), 44)

    # Load landscape
    data, masses = load_landscape(args.landscape)
    m1, m2, m3 = masses
    log.info("Loaded landscape: %d configs, masses=%s", len(data["eff_rank"]), masses)

    # Select low-rank configurations
    mask = data["eff_rank"] <= args.max_rank
    n_low = mask.sum()
    log.info("Configurations with rank <= %d: %d", args.max_rank, n_low)

    if n_low == 0:
        log.info("No low-rank configurations found. Try increasing --max-rank.")
        return

    # Sample if too many
    if n_low > args.max_configs:
        indices = np.where(mask)[0]
        indices = np.random.choice(indices, size=args.max_configs, replace=False)
        log.info("Sampling %d of %d low-rank configs", args.max_configs, n_low)
    else:
        indices = np.where(mask)[0]

    # Build work items
    work = [
        (i, data["r1"][idx], data["r2"][idx], data["theta"][idx], data["phi"][idx],
         m1, m2, m3, args.circular, args.dt, args.n_steps)
        for i, idx in enumerate(indices)
    ]

    log.info("Integrating %d configurations (dt=%.4f, steps=%d, T=%.1f) with %d workers...",
             len(work), args.dt, args.n_steps, args.dt * args.n_steps, n_workers)

    t0 = time.time()
    if n_workers <= 1:
        results = [_integrate_one(w) for w in work]
    else:
        with Pool(n_workers) as pool:
            results = pool.map(_integrate_one, work, chunksize=max(1, len(work) // (n_workers * 4)))

    elapsed = time.time() - t0
    log.info("Integration complete in %.1f s (%.1f configs/s)", elapsed, len(work) / elapsed)

    # Analyze results
    n_errors = sum(1 for r in results if r.get("error"))
    n_collision = sum(1 for r in results if r.get("collision"))
    n_periodic = sum(1 for r in results if r.get("is_periodic") and not r.get("error"))
    valid = [r for r in results if not r.get("error") and not r.get("collision")]

    log.info("Results: %d valid, %d collisions, %d errors", len(valid), n_collision, n_errors)
    log.info("PERIODIC ORBITS FOUND: %d / %d (%.2f%%)",
             n_periodic, len(valid), 100 * n_periodic / len(valid) if valid else 0)

    if valid:
        return_dists = [r["return_distance"] for r in valid]
        energy_errors = [r["energy_error"] for r in valid]
        log.info("Return distance: min=%.4f, mean=%.4f, median=%.4f",
                 min(return_dists), np.mean(return_dists), np.median(return_dists))
        log.info("Energy conservation: mean error=%.2e, max=%.2e",
                 np.mean(energy_errors), max(energy_errors))

    # Show best candidates (closest returns)
    valid.sort(key=lambda r: r["return_distance"])
    log.info("\nTop 10 closest returns:")
    for r in valid[:10]:
        log.info("  r1=%.3f r2=%.3f th=%.3f phi=%.3f  return_dist=%.4f  t_return=%.1f  periodic=%s",
                 r["r1"], r["r2"], r["theta"], r["phi"],
                 r["return_distance"], r["return_time"], r["is_periodic"])

    # Save results
    out_path = Path(args.landscape).parent / "orbit_search_results.npz"
    np.savez_compressed(
        out_path,
        r1=np.array([r["r1"] for r in valid]),
        r2=np.array([r["r2"] for r in valid]),
        theta=np.array([r["theta"] for r in valid]),
        phi=np.array([r["phi"] for r in valid]),
        return_distance=np.array([r["return_distance"] for r in valid]),
        return_time=np.array([r["return_time"] for r in valid]),
        is_periodic=np.array([r.get("is_periodic", False) for r in valid]),
    )
    log.info("Saved to %s", out_path)


if __name__ == "__main__":
    main()
