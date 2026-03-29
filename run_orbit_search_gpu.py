#!/usr/bin/env python3
"""
GPU-accelerated orbit search at low-rank configurations.

Integrates thousands of orbits simultaneously on GPU 0, leaving GPU 1
for Qwen training. Much faster than the CPU version.

Usage:
    CUDA_VISIBLE_DEVICES=0 python run_orbit_search_gpu.py \
        --landscape results/phase1/landscape_equal_static.npz
"""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

import numpy as np

from tensor_3body.integrator_gpu import integrate_batch
from tensor_3body.sampling import config_to_phase_space_circular, config_to_phase_space
from tensor_3body.landscape import load_landscape

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="GPU orbit search at low-rank configs")
    parser.add_argument("--landscape", type=str, required=True)
    parser.add_argument("--max-rank", type=int, default=10)
    parser.add_argument("--max-configs", type=int, default=10000)
    parser.add_argument("--dt", type=float, default=0.005)
    parser.add_argument("--n-steps", type=int, default=50000)
    parser.add_argument("--circular", action="store_true")
    parser.add_argument("--gpu", type=int, default=0)

    args = parser.parse_args()

    # Load landscape
    data, masses = load_landscape(args.landscape)
    m1, m2, m3 = masses
    log.info("Loaded landscape: %d configs, masses=%s", len(data["eff_rank"]), masses)

    # Select low-rank
    mask = data["eff_rank"] <= args.max_rank
    n_low = mask.sum()
    log.info("Configs with rank <= %d: %d", args.max_rank, n_low)

    if n_low == 0:
        log.info("No low-rank configs found.")
        return

    indices = np.where(mask)[0]
    if len(indices) > args.max_configs:
        indices = np.random.choice(indices, size=args.max_configs, replace=False)
        log.info("Sampling %d configs", args.max_configs)

    # Build initial conditions array (N, 12)
    z0_list = []
    meta = {"r1": [], "r2": [], "theta": [], "phi": [], "rank": []}

    convert_fn = config_to_phase_space_circular if args.circular else config_to_phase_space
    for idx in indices:
        r1 = float(data["r1"][idx])
        r2 = float(data["r2"][idx])
        theta = float(data["theta"][idx])
        phi = float(data["phi"][idx])
        z = convert_fn(r1, r2, theta, phi, m1, m2, m3)
        z0_list.append(z)
        meta["r1"].append(r1)
        meta["r2"].append(r2)
        meta["theta"].append(theta)
        meta["phi"].append(phi)
        meta["rank"].append(int(data["eff_rank"][idx]))

    z0 = np.array(z0_list)
    log.info("Built %d initial conditions", len(z0))

    # GPU integration
    t0 = time.time()
    result = integrate_batch(z0, m1, m2, m3,
                             dt=args.dt, n_steps=args.n_steps,
                             gpu_id=args.gpu)
    elapsed = time.time() - t0

    # Analysis
    valid = ~result["collision"]
    n_valid = valid.sum()
    n_periodic = result["is_periodic"][valid].sum()
    n_collision = result["collision"].sum()

    log.info("=" * 60)
    log.info("RESULTS: %d valid, %d collisions, %d periodic (%.2f%%)",
             n_valid, n_collision, n_periodic,
             100 * n_periodic / n_valid if n_valid > 0 else 0)
    log.info("Integration time: %.1f s (%.0f orbits/s)", elapsed, len(z0) / elapsed)

    if n_valid > 0:
        rd = result["return_distance"][valid]
        ee = result["energy_error"][valid]
        log.info("Return distance: min=%.4f, mean=%.4f, median=%.4f",
                 rd.min(), rd.mean(), np.median(rd))
        log.info("Energy error: mean=%.2e, max=%.2e", ee.mean(), ee.max())

    # Top 20 closest returns
    valid_indices = np.where(valid)[0]
    sort_order = np.argsort(result["return_distance"][valid_indices])
    log.info("\nTop 20 closest returns:")
    for i in sort_order[:20]:
        vi = valid_indices[i]
        log.info("  r1=%.3f r2=%.3f th=%.3f phi=%.3f rank=%d  "
                 "ret_dist=%.4f t_ret=%.1f periodic=%s E_err=%.1e",
                 meta["r1"][vi], meta["r2"][vi],
                 meta["theta"][vi], meta["phi"][vi], meta["rank"][vi],
                 result["return_distance"][vi], result["return_time"][vi],
                 result["is_periodic"][vi], result["energy_error"][vi])

    # Periodic orbits by rank
    if n_periodic > 0:
        log.info("\nPeriodic orbits by tensor rank:")
        for rank_val in range(1, args.max_rank + 1):
            rank_mask = np.array(meta["rank"]) == rank_val
            periodic_at_rank = result["is_periodic"][rank_mask & valid].sum()
            total_at_rank = (rank_mask & valid).sum()
            if total_at_rank > 0:
                log.info("  rank=%d: %d/%d periodic (%.1f%%)",
                         rank_val, periodic_at_rank, total_at_rank,
                         100 * periodic_at_rank / total_at_rank)

    # Save
    out_path = Path(args.landscape).parent / "orbit_search_gpu_results.npz"
    np.savez_compressed(
        out_path,
        r1=np.array(meta["r1"]),
        r2=np.array(meta["r2"]),
        theta=np.array(meta["theta"]),
        phi=np.array(meta["phi"]),
        rank=np.array(meta["rank"]),
        return_distance=result["return_distance"],
        return_time=result["return_time"],
        is_periodic=result["is_periodic"],
        energy_error=result["energy_error"],
        collision=result["collision"],
    )
    log.info("Saved to %s", out_path)


if __name__ == "__main__":
    main()
