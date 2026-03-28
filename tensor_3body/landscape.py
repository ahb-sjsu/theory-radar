"""
Parallel computation of the tensor rank landscape.

Computes the Hessian and its effective rank at each configuration
in a sampling grid, using multiprocessing for parallelism.
"""

from __future__ import annotations

import logging
import time
from multiprocessing import Pool, cpu_count
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from .hamiltonian import hessian_analytical
from .sampling import config_to_phase_space, config_to_phase_space_circular
from .tensor_ops import effective_rank, singular_values, participation_ratio, block_structure

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


def _compute_one(args: tuple) -> dict:
    """Compute Hessian and rank for a single configuration. Worker function."""
    idx, r1, r2, theta, phi, m1, m2, m3, use_circular = args

    try:
        if use_circular:
            z = config_to_phase_space_circular(r1, r2, theta, phi, m1, m2, m3)
        else:
            z = config_to_phase_space(r1, r2, theta, phi, m1, m2, m3)

        H = hessian_analytical(z, m1, m2, m3)
        sv = singular_values(H)
        eff_rank = effective_rank(H)
        pr = participation_ratio(H)
        blocks = block_structure(H)

        return {
            "idx": idx,
            "r1": r1,
            "r2": r2,
            "theta": theta,
            "phi": phi,
            "singvals": sv,
            "eff_rank": eff_rank,
            "participation_ratio": pr,
            "is_block_diagonal": blocks["is_block_diagonal"],
            "is_separable": blocks["is_separable"],
            "cross_body_q": blocks["cross_body_position"],
            "cross_body_p": blocks["cross_body_momentum"],
            "error": None,
        }
    except Exception as e:
        return {
            "idx": idx,
            "r1": r1,
            "r2": r2,
            "theta": theta,
            "phi": phi,
            "singvals": np.zeros(12),
            "eff_rank": -1,
            "participation_ratio": 0.0,
            "is_block_diagonal": False,
            "is_separable": False,
            "cross_body_q": 0.0,
            "cross_body_p": 0.0,
            "error": str(e),
        }


def compute_landscape(
    configs: NDArray,
    m1: float = 1.0,
    m2: float = 1.0,
    m3: float = 1.0,
    use_circular: bool = False,
    n_workers: int | None = None,
) -> dict[str, NDArray]:
    """Compute the tensor rank landscape across a configuration grid.

    Args:
        configs: Array of shape (N, 4) with columns [r1, r2, theta, phi].
        m1, m2, m3: Body masses.
        use_circular: If True, use circular orbit momenta; else zero momenta.
        n_workers: Number of parallel workers (default: cpu_count).

    Returns:
        Dictionary with arrays:
            r1, r2, theta, phi: (N,) configuration parameters
            singvals: (N, 12) singular values at each point
            eff_rank: (N,) effective rank
            participation_ratio: (N,) continuous rank measure
            is_block_diagonal: (N,) bool
            is_separable: (N,) bool
            cross_body_q: (N,) position cross-coupling magnitude
    """
    N = len(configs)
    if n_workers is None:
        n_workers = min(cpu_count(), N)

    log.info(
        "Computing tensor landscape: %d configurations, %d workers, "
        "masses=(%.3g, %.3g, %.3g), circular=%s",
        N, n_workers, m1, m2, m3, use_circular,
    )

    # Build work items
    work = [
        (i, configs[i, 0], configs[i, 1], configs[i, 2], configs[i, 3],
         m1, m2, m3, use_circular)
        for i in range(N)
    ]

    t0 = time.time()
    if n_workers <= 1:
        results = [_compute_one(w) for w in work]
    else:
        with Pool(n_workers) as pool:
            results = pool.map(_compute_one, work, chunksize=max(1, N // (n_workers * 4)))

    elapsed = time.time() - t0
    n_errors = sum(1 for r in results if r["error"] is not None)
    log.info(
        "Landscape computed in %.1f s (%.0f configs/s). Errors: %d/%d",
        elapsed, N / elapsed, n_errors, N,
    )

    # Sort by index to maintain order
    results.sort(key=lambda r: r["idx"])

    # Pack into arrays
    return {
        "r1": np.array([r["r1"] for r in results]),
        "r2": np.array([r["r2"] for r in results]),
        "theta": np.array([r["theta"] for r in results]),
        "phi": np.array([r["phi"] for r in results]),
        "singvals": np.array([r["singvals"] for r in results]),
        "eff_rank": np.array([r["eff_rank"] for r in results]),
        "participation_ratio": np.array([r["participation_ratio"] for r in results]),
        "is_block_diagonal": np.array([r["is_block_diagonal"] for r in results]),
        "is_separable": np.array([r["is_separable"] for r in results]),
        "cross_body_q": np.array([r["cross_body_q"] for r in results]),
    }


def save_landscape(data: dict[str, NDArray], path: str | Path, masses: tuple[float, ...]):
    """Save landscape data to compressed NPZ file.

    Args:
        data: Output of compute_landscape.
        path: Output file path (.npz).
        masses: (m1, m2, m3) tuple for metadata.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        masses=np.array(masses),
        **data,
    )
    size_mb = path.stat().st_size / 1e6
    log.info("Saved landscape to %s (%.1f MB)", path, size_mb)


def load_landscape(path: str | Path) -> tuple[dict[str, NDArray], tuple[float, ...]]:
    """Load landscape data from NPZ file.

    Returns:
        (data_dict, masses_tuple)
    """
    d = np.load(path)
    masses = tuple(d["masses"])
    data = {k: d[k] for k in d.files if k != "masses"}
    return data, masses
