#!/usr/bin/env python3
"""
Phase 3: A* Search for Optimal Tensor Decomposition

Searches through sequences of coordinate transforms to minimize the
coupling tensor's complexity. Uses participation ratio + block structure
as the heuristic (not just effective rank, per Phase 2 findings).

The "cost" of a state is how coupled the tensor is; A* finds the
transform sequence that maximally decouples the dynamics.

Usage:
    python run_phase3.py                          # search at known solutions
    python run_phase3.py --sample 1000            # search at random configs
    python run_phase3.py --sample 1000 --workers 8
"""

from __future__ import annotations

import argparse
import heapq
import logging
import time
from dataclasses import dataclass, field
from multiprocessing import Pool, cpu_count

import numpy as np
from numpy.typing import NDArray

from tensor_3body.hamiltonian import hessian_analytical
from tensor_3body.tensor_ops import (
    effective_rank,
    participation_ratio,
    block_structure,
    singular_values,
)
from tensor_3body.transforms import get_transform_registry
from tensor_3body.known_solutions import get_all_known_solutions
from tensor_3body.sampling import make_coarse_grid, config_to_phase_space

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


# ============================================================
# Complexity score (the heuristic A* minimizes)
# ============================================================

def complexity_score(H: NDArray) -> float:
    """Compute a composite complexity score for the coupling tensor.

    Lower = more decomposable / tractable.

    Components:
        - Participation ratio (continuous rank measure, lower = more concentrated)
        - Cross-body coupling (lower = more separable)
        - Singular value entropy (lower = more structured)

    Weighted sum tuned so fully separable 2-body scores near 0,
    and generic 3-body scores near 1.
    """
    pr = participation_ratio(H)
    bs = block_structure(H)
    sv = singular_values(H)

    # Normalize participation ratio: PR in [1, 12] -> [0, 1]
    pr_norm = (pr - 1) / 11.0

    # Cross-body coupling: already in [0, ~1]
    coupling = bs["cross_body_position"]

    # Singular value entropy (normalized)
    sv_pos = sv[sv > 1e-30]
    if len(sv_pos) > 0:
        sv_prob = sv_pos / sv_pos.sum()
        entropy = -np.sum(sv_prob * np.log(sv_prob + 1e-30))
        max_entropy = np.log(len(sv_pos))
        entropy_norm = entropy / max_entropy if max_entropy > 0 else 0
    else:
        entropy_norm = 0

    # Block diagonal bonus: if already block-diagonal, reduce score
    bd_bonus = -0.2 if bs["is_block_diagonal"] else 0.0
    sep_bonus = -0.3 if bs["is_separable"] else 0.0

    score = (0.4 * pr_norm
             + 0.3 * coupling
             + 0.3 * entropy_norm
             + bd_bonus + sep_bonus)

    return max(score, 0.0)


# ============================================================
# A* search state
# ============================================================

@dataclass(order=True)
class SearchState:
    priority: float
    z: NDArray = field(compare=False)
    path: list[str] = field(compare=False)
    g_cost: float = field(compare=False)  # path cost so far
    h_cost: float = field(compare=False)  # heuristic (complexity score)


def astar_search(
    z0: NDArray,
    m1: float = 1.0, m2: float = 1.0, m3: float = 1.0,
    max_depth: int = 4,
    max_states: int = 500,
) -> dict:
    """A* search for the transform sequence that minimizes tensor complexity.

    Args:
        z0: Initial phase space vector.
        m1, m2, m3: Body masses.
        max_depth: Maximum number of transforms to chain.
        max_states: Maximum states to explore.

    Returns:
        Dict with best_z, best_path, best_score, initial_score, states_explored.
    """
    transforms = get_transform_registry()

    # Initial state
    H0 = hessian_analytical(z0, m1, m2, m3)
    initial_score = complexity_score(H0)

    start = SearchState(
        priority=initial_score,
        z=z0.copy(),
        path=[],
        g_cost=0.0,
        h_cost=initial_score,
    )

    frontier = [start]
    best_state = start
    explored = 0
    visited_hashes = set()

    while frontier and explored < max_states:
        current = heapq.heappop(frontier)
        explored += 1

        # Track best
        if current.h_cost < best_state.h_cost:
            best_state = current

        # Depth limit
        if len(current.path) >= max_depth:
            continue

        # Hash to avoid revisiting similar states
        z_hash = hash(tuple(np.round(current.z, 6)))
        if z_hash in visited_hashes:
            continue
        visited_hashes.add(z_hash)

        # Expand: try each transform
        for t in transforms:
            try:
                z_new = t["fn"](current.z)
            except Exception:
                continue

            # Skip if transform didn't change anything
            if np.allclose(z_new, current.z, atol=1e-10):
                continue

            # Compute new Hessian and score
            H_new = hessian_analytical(z_new, m1, m2, m3)
            h_new = complexity_score(H_new)
            g_new = current.g_cost + t["cost"]
            f_new = g_new + h_new

            new_state = SearchState(
                priority=f_new,
                z=z_new,
                path=current.path + [t["name"]],
                g_cost=g_new,
                h_cost=h_new,
            )

            heapq.heappush(frontier, new_state)

    # Compute detailed analysis of best state
    H_best = hessian_analytical(best_state.z, m1, m2, m3)
    sv_best = singular_values(H_best)
    bs_best = block_structure(H_best)

    return {
        "initial_score": initial_score,
        "best_score": best_state.h_cost,
        "improvement": initial_score - best_state.h_cost,
        "best_path": best_state.path,
        "states_explored": explored,
        "best_rank": effective_rank(H_best),
        "best_pr": participation_ratio(H_best),
        "best_separable": bs_best["is_separable"],
        "best_block_diag": bs_best["is_block_diagonal"],
        "best_cross_body_q": bs_best["cross_body_position"],
        "best_singvals": sv_best,
    }


# ============================================================
# Parallel search across configurations
# ============================================================

def _search_one(args: tuple) -> dict:
    """Worker: A* search at one configuration."""
    idx, z, m1, m2, m3, max_depth, max_states = args
    try:
        result = astar_search(z, m1, m2, m3, max_depth, max_states)
        result["idx"] = idx
        result["error"] = None
        return result
    except Exception as e:
        return {"idx": idx, "error": str(e)}


def search_landscape(
    configs: list[tuple[NDArray, tuple[float, float, float], str]],
    max_depth: int = 4,
    max_states: int = 500,
    n_workers: int | None = None,
) -> list[dict]:
    """Run A* search across multiple configurations.

    Args:
        configs: List of (z, masses, name) tuples.
        max_depth: Max transform chain depth.
        max_states: Max states per search.
        n_workers: Parallel workers.
    """
    if n_workers is None:
        n_workers = min(cpu_count(), len(configs))

    work = [
        (i, z, m[0], m[1], m[2], max_depth, max_states)
        for i, (z, m, _name) in enumerate(configs)
    ]

    if n_workers <= 1:
        results = [_search_one(w) for w in work]
    else:
        with Pool(n_workers) as pool:
            results = pool.map(_search_one, work)

    # Re-attach names
    for r, (_, _, name) in zip(results, configs):
        r["name"] = name

    return results


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Phase 3: A* Tensor Decomposition Search")
    parser.add_argument("--max-depth", type=int, default=4)
    parser.add_argument("--max-states", type=int, default=500)
    parser.add_argument("--sample", type=int, default=0,
                        help="Also search N random configurations")
    parser.add_argument("--workers", type=int, default=None)
    args = parser.parse_args()

    log.info("=" * 70)
    log.info("Phase 3: A* Search for Optimal Tensor Decomposition")
    log.info("  max_depth=%d, max_states=%d", args.max_depth, args.max_states)
    log.info("=" * 70)

    # Part 1: Search at known solutions
    known = get_all_known_solutions()
    configs = [(s["z"], s["masses"], s["name"]) for s in known]

    log.info("\n--- Known solutions (%d) ---", len(configs))
    t0 = time.time()
    results_known = search_landscape(configs, args.max_depth, args.max_states, args.workers)

    for r in results_known:
        if r.get("error"):
            log.info("%-25s  ERROR: %s", r["name"], r["error"])
            continue
        log.info("%-25s  %.3f -> %.3f (delta=%.3f) rank=%d PR=%.2f sep=%s path=%s",
                 r["name"], r["initial_score"], r["best_score"], r["improvement"],
                 r["best_rank"], r["best_pr"], r["best_separable"],
                 " -> ".join(r["best_path"]) if r["best_path"] else "(none)")

    # Part 2: Search at random configurations
    if args.sample > 0:
        log.info("\n--- Random configurations (%d) ---", args.sample)
        grid = make_coarse_grid(n_r=10, n_angle=6)
        indices = np.random.choice(len(grid), size=min(args.sample, len(grid)), replace=False)

        random_configs = []
        for i, idx in enumerate(indices):
            r1, r2, theta, phi = grid[idx]
            z = config_to_phase_space(r1, r2, theta, phi)
            random_configs.append((z, (1.0, 1.0, 1.0), f"random_{i}"))

        results_random = search_landscape(random_configs, args.max_depth, args.max_states, args.workers)

        improvements = [r["improvement"] for r in results_random if not r.get("error")]
        if improvements:
            log.info("Random sample improvements: mean=%.3f, max=%.3f, >0.1: %d/%d",
                     np.mean(improvements), max(improvements),
                     sum(1 for x in improvements if x > 0.1), len(improvements))

            # Show top 5 most improved
            results_random.sort(key=lambda r: r.get("improvement", 0), reverse=True)
            log.info("\nTop 5 most improved:")
            for r in results_random[:5]:
                if r.get("error"):
                    continue
                log.info("  %.3f -> %.3f (delta=%.3f) rank=%d sep=%s path=%s",
                         r["initial_score"], r["best_score"], r["improvement"],
                         r["best_rank"], r["best_separable"],
                         " -> ".join(r["best_path"]))

    elapsed = time.time() - t0
    log.info("\nPhase 3 complete in %.1f seconds.", elapsed)


if __name__ == "__main__":
    main()
