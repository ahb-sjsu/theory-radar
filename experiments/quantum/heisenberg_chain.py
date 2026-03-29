#!/usr/bin/env python3
"""
Tensor rank analysis of the Heisenberg spin chain.

The 1D Heisenberg model is exactly solvable (Bethe ansatz) at certain
parameters and non-integrable at others. The coupling tensor should
show lower rank at integrable points — validating the framework on a
system with known ground truth.

H = J Σᵢ (Sˣᵢ Sˣᵢ₊₁ + Sʸᵢ Sʸᵢ₊₁ + Δ Sᶻᵢ Sᶻᵢ₊₁) + h Σᵢ Sᶻᵢ

Parameters:
- J: exchange coupling (set to 1)
- Δ: anisotropy (Δ=1: isotropic/integrable, Δ=0: XX model/integrable)
- h: external field (h=0: integrable for any Δ)

The coupling tensor has shape (N_sites, 3, N_sites, 3) where 3 = spin
components (x, y, z).
"""

from __future__ import annotations

import logging
import numpy as np
from itertools import product

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

try:
    import tensorly
    from tensorly.decomposition import tucker
    HAS_TENSORLY = True
except ImportError:
    HAS_TENSORLY = False


def heisenberg_coupling_tensor(
    N: int,
    delta: float = 1.0,
    h: float = 0.0,
    J: float = 1.0,
    periodic: bool = True,
) -> np.ndarray:
    """Construct the coupling tensor for the Heisenberg chain.

    The "coupling tensor" here is the interaction matrix reshaped as
    (N, 3, N, 3) where entry [i, a, j, b] gives the coupling strength
    between spin component a of site i and spin component b of site j.

    For nearest-neighbor Heisenberg:
        T[i, x, i+1, x] = J
        T[i, y, i+1, y] = J
        T[i, z, i+1, z] = J * Δ

    External field adds diagonal terms:
        T[i, z, i, z] += h
    """
    T = np.zeros((N, 3, N, 3))

    for i in range(N):
        j = (i + 1) % N if periodic else i + 1
        if j >= N:
            continue

        # XX coupling
        T[i, 0, j, 0] = J  # Sx_i Sx_j
        T[j, 0, i, 0] = J

        # YY coupling
        T[i, 1, j, 1] = J  # Sy_i Sy_j
        T[j, 1, i, 1] = J

        # ZZ coupling (with anisotropy)
        T[i, 2, j, 2] = J * delta  # Sz_i Sz_j
        T[j, 2, i, 2] = J * delta

    # External field
    for i in range(N):
        T[i, 2, i, 2] += h

    return T


def analyze_tensor(T: np.ndarray, name: str = ""):
    """Compute multilinear rank and Tucker decomposition of coupling tensor."""
    # Flatten to matrix for SVD
    N = T.shape[0]
    H = T.reshape(N * 3, N * 3)
    sv = np.linalg.svd(H, compute_uv=False)
    matrix_rank = int(np.sum(sv / (sv[0] + 1e-30) > 1e-6))

    # Multilinear rank
    ml_ranks = []
    for mode in range(4):
        unf = tensorly.unfold(T, mode) if HAS_TENSORLY else T.reshape(T.shape[mode], -1)
        sv_mode = np.linalg.svd(unf, compute_uv=False)
        r = int(np.sum(sv_mode / (sv_mode[0] + 1e-30) > 1e-6))
        ml_ranks.append(r)

    # Tucker mode ranks at 99% variance
    tucker_ranks = []
    for mode in range(4):
        unf = tensorly.unfold(T, mode) if HAS_TENSORLY else T.reshape(T.shape[mode], -1)
        sv_mode = np.linalg.svd(unf, compute_uv=False)
        total = np.sum(sv_mode**2)
        if total > 1e-30:
            cumvar = np.cumsum(sv_mode**2) / total
            r = int(np.searchsorted(cumvar, 0.99)) + 1
        else:
            r = 0
        tucker_ranks.append(r)

    # Participation ratio
    sv_all = np.linalg.svd(H, compute_uv=False)
    sv_pos = sv_all[sv_all > 1e-30]
    pr = np.sum(sv_pos)**2 / np.sum(sv_pos**2) if len(sv_pos) > 0 else 0

    log.info("%-30s matrix_rank=%d  ml_rank=%s  tucker_99=%s  PR=%.2f",
             name, matrix_rank, ml_ranks, tucker_ranks, pr)

    return {
        "name": name,
        "matrix_rank": matrix_rank,
        "multilinear_rank": ml_ranks,
        "tucker_ranks": tucker_ranks,
        "participation_ratio": pr,
    }


def main():
    log.info("=" * 70)
    log.info("Heisenberg Chain: Tensor Rank at Integrable vs Non-Integrable Points")
    log.info("=" * 70)

    N = 12  # chain length

    # Scan anisotropy parameter Δ
    # Integrable points: Δ = 0 (XX), Δ = 1 (XXX/isotropic), |Δ| → ∞ (Ising)
    # Non-integrable: Δ with external field in non-z direction (not implemented here)
    # But: adding staggered field or next-nearest-neighbor breaks integrability

    results = []

    log.info("\n--- Anisotropy scan (h=0, integrable for all Δ) ---")
    for delta in [0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 5.0]:
        T = heisenberg_coupling_tensor(N, delta=delta, h=0.0)
        r = analyze_tensor(T, f"Δ={delta:.2f}, h=0 (integrable)")
        r["delta"] = delta
        r["h"] = 0.0
        r["integrable"] = True
        results.append(r)

    log.info("\n--- External field scan (Δ=1, h≠0, still integrable) ---")
    for h in [0.1, 0.5, 1.0, 2.0, 5.0]:
        T = heisenberg_coupling_tensor(N, delta=1.0, h=h)
        r = analyze_tensor(T, f"Δ=1.00, h={h:.1f} (integrable)")
        r["delta"] = 1.0
        r["h"] = h
        r["integrable"] = True
        results.append(r)

    log.info("\n--- Non-integrable: next-nearest-neighbor coupling ---")
    for J2 in [0.1, 0.3, 0.5, 1.0]:
        T = heisenberg_coupling_tensor(N, delta=1.0, h=0.0)
        # Add NNN coupling (breaks integrability)
        for i in range(N):
            j = (i + 2) % N
            T[i, 0, j, 0] += J2
            T[j, 0, i, 0] += J2
            T[i, 1, j, 1] += J2
            T[j, 1, i, 1] += J2
            T[i, 2, j, 2] += J2
            T[j, 2, i, 2] += J2
        r = analyze_tensor(T, f"Δ=1, J2={J2:.1f} (NON-integrable)")
        r["delta"] = 1.0
        r["h"] = 0.0
        r["J2"] = J2
        r["integrable"] = False
        results.append(r)

    # Summary
    log.info("\n" + "=" * 70)
    log.info("SUMMARY: Does tensor rank distinguish integrable from non-integrable?")
    log.info("=" * 70)

    int_ranks = [r["matrix_rank"] for r in results if r.get("integrable")]
    non_ranks = [r["matrix_rank"] for r in results if not r.get("integrable")]

    int_pr = [r["participation_ratio"] for r in results if r.get("integrable")]
    non_pr = [r["participation_ratio"] for r in results if not r.get("integrable")]

    log.info("Integrable:     mean_rank=%.1f  mean_PR=%.2f  (n=%d)",
             np.mean(int_ranks), np.mean(int_pr), len(int_ranks))
    log.info("Non-integrable: mean_rank=%.1f  mean_PR=%.2f  (n=%d)",
             np.mean(non_ranks), np.mean(non_pr), len(non_ranks))

    int_tucker = [sum(r["tucker_ranks"]) for r in results if r.get("integrable")]
    non_tucker = [sum(r["tucker_ranks"]) for r in results if not r.get("integrable")]
    log.info("Tucker sum — Integrable: %.1f  Non-integrable: %.1f",
             np.mean(int_tucker), np.mean(non_tucker))


if __name__ == "__main__":
    main()
