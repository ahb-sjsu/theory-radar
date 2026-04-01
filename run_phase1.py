#!/usr/bin/env python3
"""
Phase 1: Tensor Landscape Mapping

Computes the coupling tensor effective rank across the 3-body configuration
space for multiple mass ratios. Produces NPZ data files and summary statistics.

Usage:
    python run_phase1.py                    # coarse grid, quick check
    python run_phase1.py --resolution fine  # full resolution
    python run_phase1.py --resolution fine --circular  # with orbital momenta
"""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

import numpy as np

from tensor_3body.sampling import make_config_grid, make_coarse_grid
from tensor_3body.landscape import compute_landscape, save_landscape

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

OUTPUT_DIR = Path(__file__).resolve().parent / "results" / "phase1"

# Mass ratio scenarios
MASS_SCENARIOS = {
    "equal": (1.0, 1.0, 1.0),
    "sun_jupiter_earth": (1.0, 1e-3, 3e-6),
    "binary_light": (1.0, 1.0, 0.01),
}


def run_scenario(
    name: str,
    masses: tuple[float, float, float],
    configs: np.ndarray,
    use_circular: bool = False,
    n_workers: int | None = None,
):
    """Run landscape computation for one mass scenario."""
    log.info("=" * 60)
    log.info("Scenario: %s, masses=%s, configs=%d", name, masses, len(configs))

    data = compute_landscape(
        configs,
        m1=masses[0],
        m2=masses[1],
        m3=masses[2],
        use_circular=use_circular,
        n_workers=n_workers,
    )

    # Summary statistics
    ranks = data["eff_rank"]
    valid = ranks >= 0
    if valid.sum() > 0:
        r = ranks[valid]
        log.info(
            "Effective rank: min=%d, max=%d, mean=%.1f, median=%d",
            r.min(),
            r.max(),
            r.mean(),
            np.median(r),
        )
        for rank_val in range(1, 13):
            count = np.sum(r == rank_val)
            if count > 0:
                log.info("  rank=%d: %d configs (%.2f%%)", rank_val, count, 100 * count / len(r))

        pr = data["participation_ratio"][valid]
        log.info(
            "Participation ratio: min=%.2f, max=%.2f, mean=%.2f", pr.min(), pr.max(), pr.mean()
        )

        n_sep = data["is_separable"][valid].sum()
        n_bd = data["is_block_diagonal"][valid].sum()
        log.info("Fully separable: %d (%.2f%%)", n_sep, 100 * n_sep / len(r))
        log.info("Block diagonal: %d (%.2f%%)", n_bd, 100 * n_bd / len(r))

        # Find lowest-rank configurations
        min_rank = r.min()
        low_rank_mask = ranks == min_rank
        low_rank_configs = configs[low_rank_mask]
        if len(low_rank_configs) > 0 and len(low_rank_configs) <= 20:
            log.info("Lowest-rank (%d) configurations:", min_rank)
            for cfg in low_rank_configs:
                log.info("  r1=%.3f, r2=%.3f, theta=%.3f, phi=%.3f", cfg[0], cfg[1], cfg[2], cfg[3])

    # Save
    suffix = "_circular" if use_circular else "_static"
    out_path = OUTPUT_DIR / f"landscape_{name}{suffix}.npz"
    save_landscape(data, out_path, masses)

    return data


def main():
    parser = argparse.ArgumentParser(description="Phase 1: Tensor Landscape Mapping")
    parser.add_argument(
        "--resolution",
        choices=["coarse", "medium", "fine"],
        default="coarse",
        help="Grid resolution (default: coarse)",
    )
    parser.add_argument(
        "--circular", action="store_true", help="Use circular orbit momenta instead of zero momenta"
    )
    parser.add_argument(
        "--scenario",
        choices=list(MASS_SCENARIOS.keys()) + ["all"],
        default="all",
        help="Which mass scenario to run (default: all)",
    )
    parser.add_argument(
        "--workers", type=int, default=None, help="Number of parallel workers (default: cpu_count)"
    )

    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Build grid
    if args.resolution == "coarse":
        configs = make_coarse_grid(n_r=20, n_angle=12)
    elif args.resolution == "medium":
        configs = make_config_grid(n_r1=30, n_r2=30, n_theta=18, n_phi=18)
    else:
        configs = make_config_grid(n_r1=50, n_r2=50, n_theta=36, n_phi=36)

    log.info("Grid resolution: %s (%d configurations)", args.resolution, len(configs))

    # Run scenarios
    scenarios = (
        MASS_SCENARIOS if args.scenario == "all" else {args.scenario: MASS_SCENARIOS[args.scenario]}
    )

    t0 = time.time()
    for name, masses in scenarios.items():
        run_scenario(name, masses, configs, use_circular=args.circular, n_workers=args.workers)

    elapsed = time.time() - t0
    log.info("=" * 60)
    log.info("Phase 1 complete in %.1f minutes.", elapsed / 60)


if __name__ == "__main__":
    main()
