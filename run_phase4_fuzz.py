#!/usr/bin/env python3
"""
Phase 4: Fuzzing for New Periodic Orbits

Starting from the periodic orbits found in the GPU orbit search,
perturb initial conditions along low-rank directions in tensor space
to map out orbit families and find bifurcations.

Strategy:
1. Take the rank-7/8 periodic orbits (100% periodic) as seeds
2. Compute the Tucker decomposition at each seed
3. Perturb along the low-variance Tucker modes (directions that
   preserve low rank)
4. Integrate perturbed orbits to check if periodicity persists
5. Binary search for the family boundary (where periodicity breaks)

Usage:
    CUDA_VISIBLE_DEVICES=0 python run_phase4_fuzz.py
"""

from __future__ import annotations

import logging
import time

import numpy as np

from tensor_3body.hamiltonian import hessian_analytical
from tensor_3body.tensor_ops import (
    reshape_to_rank6,
    effective_rank,
    singular_values,
    participation_ratio,
)
from tensor_3body.sampling import config_to_phase_space_circular
from tensor_3body.integrator_gpu import integrate_batch, HAS_CUPY
from tensor_3body.landscape import load_landscape

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


def get_perturbation_directions(z: np.ndarray, m1: float, m2: float, m3: float) -> np.ndarray:
    """Compute perturbation directions from the Hessian eigenvectors.

    Returns eigenvectors sorted by eigenvalue magnitude — the smallest
    eigenvalues correspond to the "softest" directions, which are most
    likely to preserve the orbit family structure.

    Returns: (12, 12) array where rows are perturbation directions.
    """
    H = hessian_analytical(z, m1, m2, m3)
    eigenvalues, eigenvectors = np.linalg.eigh(H)
    # Sort by absolute eigenvalue (smallest = softest)
    order = np.argsort(np.abs(eigenvalues))
    return eigenvectors[:, order].T  # (12, 12), rows are directions


def fuzz_orbit_family(
    seed_z: np.ndarray,
    m1: float, m2: float, m3: float,
    n_perturbations: int = 100,
    max_amplitude: float = 0.5,
    n_directions: int = 6,
    dt: float = 0.005,
    n_steps: int = 50000,
    gpu_id: int = 0,
) -> dict:
    """Fuzz a seed orbit to map its family.

    Perturbs the seed along the softest Hessian eigenvectors at
    geometrically spaced amplitudes. Returns the family extent
    (how far you can perturb before periodicity breaks).
    """
    directions = get_perturbation_directions(seed_z, m1, m2, m3)

    # Generate perturbations: each direction x multiple amplitudes
    amplitudes = np.geomspace(0.001, max_amplitude, n_perturbations // n_directions)

    z0_list = [seed_z.copy()]  # include seed as control
    meta = [{"dir": -1, "amp": 0.0}]

    for d_idx in range(min(n_directions, 12)):
        direction = directions[d_idx]
        direction = direction / (np.linalg.norm(direction) + 1e-30)

        for amp in amplitudes:
            z_pert = seed_z + amp * direction
            z0_list.append(z_pert)
            meta.append({"dir": d_idx, "amp": amp})

            # Also try negative direction
            z_pert_neg = seed_z - amp * direction
            z0_list.append(z_pert_neg)
            meta.append({"dir": d_idx, "amp": -amp})

    z0 = np.array(z0_list)
    log.info("  Fuzzing with %d perturbations (%d directions x %d amplitudes x 2 signs + seed)",
             len(z0), n_directions, len(amplitudes))

    # GPU integration
    result = integrate_batch(z0, m1, m2, m3, dt=dt, n_steps=n_steps, gpu_id=gpu_id)

    # Analyze: for each direction, find the maximum amplitude that stays periodic
    family_extent = {}
    for d_idx in range(n_directions):
        periodic_amps = []
        nonperiodic_amps = []
        for i, m in enumerate(meta):
            if m["dir"] == d_idx:
                if result["is_periodic"][i] and not result["collision"][i]:
                    periodic_amps.append(abs(m["amp"]))
                else:
                    nonperiodic_amps.append(abs(m["amp"]))

        max_periodic = max(periodic_amps) if periodic_amps else 0.0
        min_nonperiodic = min(nonperiodic_amps) if nonperiodic_amps else max_amplitude

        family_extent[d_idx] = {
            "max_periodic_amp": max_periodic,
            "min_nonperiodic_amp": min_nonperiodic,
            "n_periodic": len(periodic_amps),
            "n_total": len(periodic_amps) + len(nonperiodic_amps),
        }

    # Overall stats
    n_periodic = int(result["is_periodic"].sum())
    n_collision = int(result["collision"].sum())

    return {
        "n_total": len(z0),
        "n_periodic": n_periodic,
        "n_collision": n_collision,
        "family_extent": family_extent,
        "seed_periodic": bool(result["is_periodic"][0]),
        "return_distances": result["return_distance"],
        "meta": meta,
    }


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Phase 4: Fuzz periodic orbits")
    parser.add_argument("--landscape", type=str,
                        default="results/phase1/landscape_equal_static.npz")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--n-seeds", type=int, default=20,
                        help="Number of seed orbits to fuzz")
    parser.add_argument("--n-perturbations", type=int, default=200,
                        help="Perturbations per seed")
    parser.add_argument("--max-amplitude", type=float, default=1.0)
    args = parser.parse_args()

    log.info("=" * 70)
    log.info("Phase 4: Fuzzing Periodic Orbit Families")
    log.info("=" * 70)

    # Load landscape and find low-rank periodic seeds
    data, masses = load_landscape(args.landscape)
    m1, m2, m3 = masses

    # Get rank-7 and rank-8 configs (100% periodic from Phase 3)
    low_rank_mask = data["eff_rank"] <= 8
    n_low = low_rank_mask.sum()
    log.info("Found %d rank<=8 configs (seed candidates)", n_low)

    if n_low == 0:
        # Fall back to rank <= 10
        low_rank_mask = data["eff_rank"] <= 10
        n_low = low_rank_mask.sum()
        log.info("Falling back to rank<=10: %d configs", n_low)

    indices = np.where(low_rank_mask)[0]
    if len(indices) > args.n_seeds:
        # Pick diverse seeds (spread across the configuration space)
        step = len(indices) // args.n_seeds
        indices = indices[::step][:args.n_seeds]

    log.info("Selected %d seed orbits", len(indices))

    # Fuzz each seed
    all_results = []
    t0 = time.time()

    for seed_i, idx in enumerate(indices):
        r1 = float(data["r1"][idx])
        r2 = float(data["r2"][idx])
        theta = float(data["theta"][idx])
        phi = float(data["phi"][idx])
        rank = int(data["eff_rank"][idx])

        z_seed = config_to_phase_space_circular(r1, r2, theta, phi, m1, m2, m3)

        log.info("\n[Seed %d/%d] r1=%.3f r2=%.3f th=%.3f phi=%.3f rank=%d",
                 seed_i + 1, len(indices), r1, r2, theta, phi, rank)

        result = fuzz_orbit_family(
            z_seed, m1, m2, m3,
            n_perturbations=args.n_perturbations,
            max_amplitude=args.max_amplitude,
            gpu_id=args.gpu,
        )

        result["seed"] = {"r1": r1, "r2": r2, "theta": theta, "phi": phi, "rank": rank}
        all_results.append(result)

        log.info("  Result: %d/%d periodic (%.1f%%), %d collisions",
                 result["n_periodic"], result["n_total"],
                 100 * result["n_periodic"] / result["n_total"],
                 result["n_collision"])
        log.info("  Seed itself periodic: %s", result["seed_periodic"])

        for d_idx, ext in result["family_extent"].items():
            if ext["n_periodic"] > 0:
                log.info("    dir %d: periodic to amp=%.4f (%d/%d)",
                         d_idx, ext["max_periodic_amp"],
                         ext["n_periodic"], ext["n_total"])

    elapsed = time.time() - t0
    log.info("\n" + "=" * 70)
    log.info("Phase 4 complete in %.1f s", elapsed)
    log.info("=" * 70)

    # Summary
    total_periodic = sum(r["n_periodic"] for r in all_results)
    total_tested = sum(r["n_total"] for r in all_results)
    log.info("Total: %d/%d periodic (%.1f%%) across %d seed families",
             total_periodic, total_tested,
             100 * total_periodic / total_tested if total_tested > 0 else 0,
             len(all_results))

    # Find the widest family
    widest = None
    widest_amp = 0
    for r in all_results:
        for d_idx, ext in r["family_extent"].items():
            if ext["max_periodic_amp"] > widest_amp:
                widest_amp = ext["max_periodic_amp"]
                widest = (r["seed"], d_idx, ext)

    if widest:
        seed, d_idx, ext = widest
        log.info("\nWidest orbit family:")
        log.info("  Seed: r1=%.3f r2=%.3f rank=%d",
                 seed["r1"], seed["r2"], seed["rank"])
        log.info("  Direction: %d, extends to amplitude %.4f", d_idx, ext["max_periodic_amp"])
        log.info("  %d/%d periodic along this direction", ext["n_periodic"], ext["n_total"])

    # Save
    np.savez_compressed(
        "results/phase4_fuzz_results.npz",
        n_seeds=len(all_results),
        total_periodic=total_periodic,
        total_tested=total_tested,
    )
    log.info("Saved results to results/phase4_fuzz_results.npz")


if __name__ == "__main__":
    main()
