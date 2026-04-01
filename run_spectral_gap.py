#!/usr/bin/env python3
"""
Measure the spectral gap ratio γ and characteristic frequencies at
configurations across the rank landscape, testing the Rank-Timescale
Equivalence conjecture directly.

The prediction: γ (spectral gap) should correlate with periodicity,
and the number of frequency clusters should equal (12 - rank).
"""

from __future__ import annotations

import logging
import numpy as np
from multiprocessing import Pool, cpu_count

from tensor_3body.hamiltonian import hessian_analytical
from tensor_3body.sampling import config_to_phase_space_circular
from tensor_3body.landscape import load_landscape

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


def spectral_gap_ratio(H: np.ndarray) -> float:
    """Compute the spectral gap ratio γ = max_k(σ_k - σ_{k+1}) / σ_1."""
    sv = np.linalg.svd(H, compute_uv=False)
    if sv[0] < 1e-30:
        return 0.0
    gaps = sv[:-1] - sv[1:]
    return float(gaps.max() / sv[0])


def characteristic_frequencies(H: np.ndarray) -> np.ndarray:
    """Extract characteristic frequencies from the qq block of the Hessian."""
    qq = H[0:6, 0:6]
    eigenvalues = np.linalg.eigvalsh(qq)
    # Frequencies are sqrt of positive eigenvalues
    freqs = np.sqrt(np.abs(eigenvalues))
    return np.sort(freqs)[::-1]


def frequency_ratio(freqs: np.ndarray) -> float:
    """Max/min frequency ratio (timescale separation)."""
    pos = freqs[freqs > 1e-10]
    if len(pos) < 2:
        return 1.0
    return float(pos[0] / pos[-1])


def _analyze_one(args):
    """Worker: compute spectral properties at one configuration."""
    idx, r1, r2, theta, phi, m1, m2, m3, eff_rank = args
    try:
        z = config_to_phase_space_circular(r1, r2, theta, phi, m1, m2, m3)
        H = hessian_analytical(z, m1, m2, m3)
        gamma = spectral_gap_ratio(H)
        freqs = characteristic_frequencies(H)
        freq_rat = frequency_ratio(freqs)
        return {
            "idx": idx,
            "r1": r1,
            "r2": r2,
            "rank": eff_rank,
            "gamma": gamma,
            "freq_ratio": freq_rat,
            "freqs": freqs,
            "error": None,
        }
    except Exception as e:
        return {"idx": idx, "error": str(e)}


def main():
    log.info("=" * 60)
    log.info("Spectral Gap Analysis: Testing Rank-Timescale Conjecture")
    log.info("=" * 60)

    data, masses = load_landscape("results/phase1/landscape_equal_static.npz")
    m1, m2, m3 = float(masses[0]), float(masses[1]), float(masses[2])

    # Sample across all ranks
    N_per_rank = 500
    work = []
    for target_rank in range(7, 13):
        mask = data["eff_rank"] == target_rank
        indices = np.where(mask)[0]
        if len(indices) > N_per_rank:
            indices = np.random.choice(indices, size=N_per_rank, replace=False)
        for idx in indices:
            work.append(
                (
                    len(work),
                    float(data["r1"][idx]),
                    float(data["r2"][idx]),
                    float(data["theta"][idx]),
                    float(data["phi"][idx]),
                    m1,
                    m2,
                    m3,
                    int(data["eff_rank"][idx]),
                )
            )

    log.info("Analyzing %d configurations across ranks 7-12", len(work))

    n_workers = min(cpu_count(), 16)
    with Pool(n_workers) as pool:
        results = pool.map(_analyze_one, work)

    valid = [r for r in results if r.get("error") is None]
    log.info("Valid: %d/%d", len(valid), len(results))

    # Group by rank and compute statistics
    log.info(
        "\n%-6s  %-8s  %-12s  %-12s  %-8s", "Rank", "N", "γ (mean±std)", "ω_ratio (mean)", "PR"
    )

    for rank in range(7, 13):
        subset = [r for r in valid if r["rank"] == rank]
        if not subset:
            continue
        gammas = [r["gamma"] for r in subset]
        ratios = [r["freq_ratio"] for r in subset]
        log.info(
            "%-6d  %-8d  %.4f±%.4f  %-12.1f  -",
            rank,
            len(subset),
            np.mean(gammas),
            np.std(gammas),
            np.mean(ratios),
        )

    # Correlation: γ vs rank
    all_gammas = np.array([r["gamma"] for r in valid])
    all_ranks = np.array([r["rank"] for r in valid])
    all_ratios = np.array([r["freq_ratio"] for r in valid])

    corr_gamma_rank = np.corrcoef(all_gammas, all_ranks)[0, 1]
    corr_ratio_rank = np.corrcoef(all_ratios, all_ranks)[0, 1]

    log.info("\nCorrelations:")
    log.info("  γ vs rank: r = %.4f", corr_gamma_rank)
    log.info("  ω_ratio vs rank: r = %.4f", corr_ratio_rank)

    # The prediction: lower rank should have HIGHER γ (bigger spectral gap)
    # and HIGHER frequency ratio (more timescale separation)
    if corr_gamma_rank < -0.1:
        log.info("  → CONSISTENT: lower rank ↔ higher spectral gap (γ)")
    elif corr_gamma_rank > 0.1:
        log.info("  → INCONSISTENT: lower rank ↔ lower spectral gap")
    else:
        log.info("  → INCONCLUSIVE: weak correlation")

    if corr_ratio_rank < -0.1:
        log.info("  → CONSISTENT: lower rank ↔ higher frequency ratio")
    elif corr_ratio_rank > 0.1:
        log.info("  → INCONSISTENT: lower rank ↔ lower frequency ratio")
    else:
        log.info("  → INCONCLUSIVE: weak correlation")

    # Frequency spectrum examples
    log.info("\nExample frequency spectra:")
    for rank in [7, 9, 12]:
        subset = [r for r in valid if r["rank"] == rank]
        if subset:
            ex = subset[0]
            log.info(
                "  rank=%d (r1=%.2f, r2=%.2f): ω = %s",
                rank,
                ex["r1"],
                ex["r2"],
                " ".join(f"{f:.3f}" for f in ex["freqs"]),
            )


if __name__ == "__main__":
    main()
