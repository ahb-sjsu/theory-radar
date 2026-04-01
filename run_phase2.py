#!/usr/bin/env python3
"""
Phase 2: Validate Known Solutions

Computes the coupling tensor at known exact/special solutions and
checks whether they correspond to low effective rank. This is the
decision gate: if known solutions don't show lower rank than generic
configurations, the tensor decomposition approach is invalidated.

Usage:
    python run_phase2.py
"""

from __future__ import annotations

import logging

import numpy as np

from tensor_3body.hamiltonian import hessian_analytical
from tensor_3body.tensor_ops import (
    effective_rank,
    singular_values,
    participation_ratio,
    block_structure,
    reshape_to_rank6,
    tucker_decomposition,
    multilinear_rank,
    mode_coupling_analysis,
)
from tensor_3body.known_solutions import get_all_known_solutions

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


def analyze_solution(sol: dict) -> dict:
    """Compute tensor properties at a known solution."""
    z = sol["z"]
    m1, m2, m3 = sol["masses"]

    H = hessian_analytical(z, m1, m2, m3)
    sv = singular_values(H)
    er = effective_rank(H)
    pr = participation_ratio(H)
    bs = block_structure(H)
    T6 = reshape_to_rank6(H)

    # Analyze the rank-6 tensor structure
    # Check which sub-blocks carry the most weight
    qq_block = H[0:6, 0:6]  # position-position
    pp_block = H[6:12, 6:12]  # momentum-momentum
    qq_rank = effective_rank(qq_block)
    pp_rank = effective_rank(pp_block)

    # Symmetry check: how symmetric is the Hessian under body exchange?
    # For equal masses, swapping body indices should leave H invariant
    # In Jacobi coords, this maps rho1 -> -rho1 (and pi1 -> -pi1)
    # which is a sign flip on indices 0:3 and 6:9
    swap = np.eye(12)
    for i in [0, 1, 2, 6, 7, 8]:
        swap[i, i] = -1
    H_swapped = swap @ H @ swap.T
    swap_asymmetry = np.linalg.norm(H - H_swapped) / (np.linalg.norm(H) + 1e-30)

    # Tucker decomposition of rank-6 tensor
    ml_rank = multilinear_rank(T6)
    tucker = tucker_decomposition(T6)
    couplings = mode_coupling_analysis(T6)

    return {
        "name": sol["name"],
        "description": sol["description"],
        "expected": sol["expected"],
        "eff_rank": er,
        "participation_ratio": pr,
        "singular_values": sv,
        "qq_rank": qq_rank,
        "pp_rank": pp_rank,
        "is_block_diagonal": bs["is_block_diagonal"],
        "is_separable": bs["is_separable"],
        "cross_body_q": bs["cross_body_position"],
        "cross_body_p": bs["cross_body_momentum"],
        "swap_asymmetry": swap_asymmetry,
        "multilinear_rank": ml_rank,
        "tucker_mode_ranks": tucker["mode_ranks"],
        "tucker_mode_names": tucker["mode_names"],
        "tucker_mode_variance": tucker["mode_explained_variance"],
        "tucker_recon_error": tucker["reconstruction_error"],
        "mode_couplings": couplings,
    }


def main():
    solutions = get_all_known_solutions()
    results = []

    log.info("=" * 70)
    log.info("Phase 2: Validating %d known solutions", len(solutions))
    log.info("=" * 70)

    # Also compute a "generic" configuration for comparison
    z_generic = np.array([1.5, 0.3, 0.1, -0.8, 2.1, 0.4, 0.1, -0.2, 0.05, -0.15, 0.1, -0.08])
    H_generic = hessian_analytical(z_generic, 1.0, 1.0, 1.0)
    generic_rank = effective_rank(H_generic)
    generic_pr = participation_ratio(H_generic)
    log.info("Generic configuration baseline: rank=%d, PR=%.2f", generic_rank, generic_pr)
    log.info("-" * 70)

    for sol in solutions:
        r = analyze_solution(sol)
        results.append(r)

        rank_vs_generic = (
            "LOWER"
            if r["eff_rank"] < generic_rank
            else ("SAME" if r["eff_rank"] == generic_rank else "HIGHER")
        )

        log.info("")
        log.info(
            "%-25s  rank=%d (%s vs generic=%d)  PR=%.2f",
            r["name"],
            r["eff_rank"],
            rank_vs_generic,
            generic_rank,
            r["participation_ratio"],
        )
        log.info(
            "  qq_rank=%d  pp_rank=%d  block_diag=%s  separable=%s",
            r["qq_rank"],
            r["pp_rank"],
            r["is_block_diagonal"],
            r["is_separable"],
        )
        log.info(
            "  cross_body_q=%.2e  cross_body_p=%.2e  swap_asym=%.2e",
            r["cross_body_q"],
            r["cross_body_p"],
            r["swap_asymmetry"],
        )
        log.info("  singular values: %s", "  ".join(f"{s:.3e}" for s in r["singular_values"]))
        log.info("  multilinear rank: %s", r["multilinear_rank"])
        log.info(
            "  tucker mode ranks (99%% var): %s",
            dict(zip(r["tucker_mode_names"], r["tucker_mode_ranks"])),
        )
        # Show strongest and weakest couplings
        coups = r["mode_couplings"]
        sorted_coups = sorted(coups.items(), key=lambda x: x[1]["concentration"])
        log.info(
            "  weakest coupling:  %s (conc=%.3f, rank=%d/%d)",
            sorted_coups[0][0],
            sorted_coups[0][1]["concentration"],
            sorted_coups[0][1]["eff_rank"],
            sorted_coups[0][1]["max_rank"],
        )
        log.info(
            "  strongest coupling: %s (conc=%.3f, rank=%d/%d)",
            sorted_coups[-1][0],
            sorted_coups[-1][1]["concentration"],
            sorted_coups[-1][1]["eff_rank"],
            sorted_coups[-1][1]["max_rank"],
        )
        log.info("  expected: %s", r["expected"])

    # Summary and decision gate
    log.info("")
    log.info("=" * 70)
    log.info("DECISION GATE: Do known solutions show lower tensor rank?")
    log.info("=" * 70)

    n_lower = sum(1 for r in results if r["eff_rank"] < generic_rank)
    n_same = sum(1 for r in results if r["eff_rank"] == generic_rank)
    n_higher = sum(1 for r in results if r["eff_rank"] > generic_rank)
    n_separable = sum(1 for r in results if r["is_separable"])
    n_block_diag = sum(1 for r in results if r["is_block_diagonal"])

    mean_pr_known = np.mean([r["participation_ratio"] for r in results])
    mean_rank_known = np.mean([r["eff_rank"] for r in results])

    log.info("Generic baseline: rank=%d, PR=%.2f", generic_rank, generic_pr)
    log.info("Known solutions:  mean_rank=%.1f, mean_PR=%.2f", mean_rank_known, mean_pr_known)
    log.info("  Lower rank: %d/%d", n_lower, len(results))
    log.info("  Same rank:  %d/%d", n_same, len(results))
    log.info("  Higher rank: %d/%d", n_higher, len(results))
    log.info("  Separable:  %d/%d", n_separable, len(results))
    log.info("  Block diag: %d/%d", n_block_diag, len(results))

    if n_lower > len(results) // 2:
        log.info("")
        log.info("PASS: Majority of known solutions show lower rank than generic.")
        log.info("      Tensor decomposition approach is VALIDATED. Proceed to Phase 3.")
    elif n_lower > 0:
        log.info("")
        log.info("PARTIAL: Some known solutions show lower rank.")
        log.info("         Approach has merit but selectivity needs investigation.")
    else:
        log.info("")
        log.info("FAIL: No known solutions show lower rank than generic configurations.")
        log.info("      Tensor decomposition approach is NOT validated.")
        log.info("      Consider: different tensor construction, or different rank measure.")


if __name__ == "__main__":
    main()
