"""
Tensor operations for the 3-body coupling tensor.

Reshapes the 12x12 Hessian into rank-6 form, computes effective rank,
and provides decomposition utilities.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


# Index mapping for the rank-6 tensor
# The 12D phase space is ordered:
#   [rho1_x, rho1_y, rho1_z, rho2_x, rho2_y, rho2_z,
#    pi1_x,  pi1_y,  pi1_z,  pi2_x,  pi2_y,  pi2_z]
#
# Rank-6 indices: (body, spatial, phase) x 2
#   body:    {0, 1}    -- Jacobi body (0=inner pair, 1=outer)
#   spatial: {0, 1, 2} -- x, y, z
#   phase:   {0, 1}    -- 0=position (rho), 1=momentum (pi)
#
# Flat index = body*3 + spatial + phase*6

def flat_to_rank6(a: int) -> tuple[int, int, int]:
    """Convert flat index (0-11) to (body, spatial, phase) triple."""
    phase = a // 6       # 0=position, 1=momentum
    rem = a % 6
    body = rem // 3      # 0=inner, 1=outer
    spatial = rem % 3    # 0=x, 1=y, 2=z
    return body, spatial, phase


def rank6_to_flat(body: int, spatial: int, phase: int) -> int:
    """Convert (body, spatial, phase) triple to flat index (0-11)."""
    return phase * 6 + body * 3 + spatial


def reshape_to_rank6(H: NDArray) -> NDArray:
    """Reshape 12x12 Hessian to rank-6 tensor (2,3,2,2,3,2).

    Args:
        H: Hessian matrix, shape (12, 12).

    Returns:
        Rank-6 tensor, shape (2, 3, 2, 2, 3, 2).
        T[i, mu, alpha, j, nu, beta] = H[flat(i,mu,alpha), flat(j,nu,beta)]
    """
    T = np.zeros((2, 3, 2, 2, 3, 2))
    for a in range(12):
        i, mu, alpha = flat_to_rank6(a)
        for b in range(12):
            j, nu, beta = flat_to_rank6(b)
            T[i, mu, alpha, j, nu, beta] = H[a, b]
    return T


def reshape_to_matrix(T: NDArray) -> NDArray:
    """Reshape rank-6 tensor (2,3,2,2,3,2) back to 12x12 matrix.

    Args:
        T: Rank-6 tensor, shape (2, 3, 2, 2, 3, 2).

    Returns:
        Matrix, shape (12, 12).
    """
    H = np.zeros((12, 12))
    for a in range(12):
        i, mu, alpha = flat_to_rank6(a)
        for b in range(12):
            j, nu, beta = flat_to_rank6(b)
            H[a, b] = T[i, mu, alpha, j, nu, beta]
    return H


def effective_rank(H: NDArray, epsilon: float = 1e-6) -> int:
    """Compute effective rank of the coupling tensor.

    Args:
        H: Hessian matrix, shape (12, 12), or any 2D array.
        epsilon: Threshold ratio. Singular values below
                 epsilon * sigma_max are treated as zero.

    Returns:
        Number of singular values above the threshold.
    """
    sv = np.linalg.svd(H, compute_uv=False)
    if sv[0] < 1e-30:
        return 0
    return int(np.sum(sv / sv[0] > epsilon))


def singular_values(H: NDArray) -> NDArray:
    """Compute singular values of the Hessian, sorted descending.

    Args:
        H: Matrix, shape (12, 12) or (N, N).

    Returns:
        Sorted singular values, shape (min(N,M),).
    """
    return np.linalg.svd(H, compute_uv=False)


def block_structure(H: NDArray, threshold: float = 1e-6) -> dict:
    """Analyze the block structure of the Hessian.

    Identifies which blocks (position-position, position-momentum,
    momentum-momentum, body-body cross terms) are significant.

    Args:
        H: Hessian matrix, shape (12, 12).
        threshold: Entries below this (relative to max) are "zero."

    Returns:
        Dictionary with block norms and structure description.
    """
    scale = np.abs(H).max()
    if scale < 1e-30:
        scale = 1.0

    blocks = {
        "qq_11": np.linalg.norm(H[0:3, 0:3]),    # rho1-rho1
        "qq_12": np.linalg.norm(H[0:3, 3:6]),    # rho1-rho2
        "qq_22": np.linalg.norm(H[3:6, 3:6]),    # rho2-rho2
        "pp_11": np.linalg.norm(H[6:9, 6:9]),    # pi1-pi1
        "pp_12": np.linalg.norm(H[6:9, 9:12]),   # pi1-pi2
        "pp_22": np.linalg.norm(H[9:12, 9:12]),  # pi2-pi2
        "qp_11": np.linalg.norm(H[0:3, 6:9]),    # rho1-pi1
        "qp_12": np.linalg.norm(H[0:3, 9:12]),   # rho1-pi2
        "qp_21": np.linalg.norm(H[3:6, 6:9]),    # rho2-pi1
        "qp_22": np.linalg.norm(H[3:6, 9:12]),   # rho2-pi2
    }

    # Determine coupling structure
    cross_body_q = blocks["qq_12"] / scale
    cross_body_p = blocks["pp_12"] / scale
    cross_qp = max(blocks["qp_11"], blocks["qp_12"],
                   blocks["qp_21"], blocks["qp_22"]) / scale

    return {
        "block_norms": blocks,
        "cross_body_position": cross_body_q,
        "cross_body_momentum": cross_body_p,
        "position_momentum_coupling": cross_qp,
        "is_block_diagonal": cross_body_q < threshold and cross_body_p < threshold,
        "is_separable": (cross_body_q < threshold and cross_body_p < threshold
                         and cross_qp < threshold),
    }


def tucker_decomposition(T: NDArray, ranks: tuple[int, ...] | None = None) -> dict:
    """Tucker decomposition of the rank-6 coupling tensor.

    Decomposes T (shape 2,3,2,2,3,2) as:
        T ≈ G ×₁ U_body1 ×₂ U_spatial1 ×₃ U_phase1 ×₄ U_body2 ×₅ U_spatial2 ×₆ U_phase2

    where G is the core tensor and U_i are factor matrices per mode.

    Args:
        T: Rank-6 tensor, shape (2, 3, 2, 2, 3, 2).
        ranks: Target ranks per mode. None = full rank (no truncation).

    Returns:
        Dictionary with:
            core: core tensor G
            factors: list of 6 factor matrices [U_body1, U_spatial1, ...]
            mode_ranks: effective rank per mode
            mode_explained_variance: fraction of variance per mode
            reconstruction_error: relative error ||T - T_approx|| / ||T||
    """
    import tensorly
    from tensorly.decomposition import tucker

    if ranks is None:
        ranks = T.shape  # full rank

    core, factors = tucker(T, rank=ranks)

    # Compute reconstruction error
    T_approx = tensorly.tucker_to_tensor((core, factors))
    error = np.linalg.norm(T - T_approx) / (np.linalg.norm(T) + 1e-30)

    # Effective rank per mode via SVD of the unfolding
    mode_names = ["body_1", "spatial_1", "phase_1", "body_2", "spatial_2", "phase_2"]
    mode_ranks = []
    mode_variance = []

    for mode in range(6):
        unfolding = tensorly.unfold(T, mode)
        sv = np.linalg.svd(unfolding, compute_uv=False)
        total_var = np.sum(sv**2)
        # Effective rank: how many singular values needed for 99% variance
        cumvar = np.cumsum(sv**2) / (total_var + 1e-30)
        eff_rank = int(np.searchsorted(cumvar, 0.99)) + 1
        mode_ranks.append(eff_rank)
        # Variance in first component
        mode_variance.append(sv[0]**2 / (total_var + 1e-30) if total_var > 0 else 0)

    return {
        "core": core,
        "factors": factors,
        "mode_names": mode_names,
        "mode_ranks": mode_ranks,
        "mode_explained_variance": mode_variance,
        "reconstruction_error": error,
    }


def multilinear_rank(T: NDArray) -> list[int]:
    """Compute the multilinear rank of a tensor (rank along each mode).

    The multilinear rank (r1, r2, ..., r6) gives the effective dimensionality
    along each mode independently. This is strictly more informative than
    the matrix rank of the flattened tensor.

    Args:
        T: Tensor of arbitrary shape.

    Returns:
        List of ranks, one per mode.
    """
    import tensorly

    ranks = []
    for mode in range(len(T.shape)):
        unfolding = tensorly.unfold(T, mode)
        sv = np.linalg.svd(unfolding, compute_uv=False)
        # Count singular values above threshold
        if sv[0] < 1e-30:
            ranks.append(0)
        else:
            ranks.append(int(np.sum(sv / sv[0] > 1e-6)))
    return ranks


def mode_coupling_analysis(T: NDArray) -> dict:
    """Analyze coupling between modes of the rank-6 tensor.

    Determines which pairs of modes (body-body, spatial-spatial,
    phase-phase, body-spatial, etc.) are coupled vs independent.

    Args:
        T: Rank-6 tensor, shape (2, 3, 2, 2, 3, 2).

    Returns:
        Dictionary with coupling strengths between mode pairs.
    """
    import tensorly

    mode_names = ["body_1", "spatial_1", "phase_1", "body_2", "spatial_2", "phase_2"]
    n_modes = 6

    # Compute coupling via mutual information of mode unfoldings
    # Proxy: correlation between singular value spectra
    spectra = []
    for mode in range(n_modes):
        unfolding = tensorly.unfold(T, mode)
        sv = np.linalg.svd(unfolding, compute_uv=False)
        spectra.append(sv)

    # Pairwise coupling: how much does one mode's spectrum predict another's?
    # Use normalized Frobenius inner product of unfoldings
    couplings = {}
    for i in range(n_modes):
        for j in range(i + 1, n_modes):
            # Contract T over all modes except i and j
            # Simpler proxy: check if tensor is separable in modes i vs j
            # by comparing rank of (i,j)-unfolding to product of mode ranks
            combined = list(range(n_modes))
            # Reshape to group mode i and j together vs everything else
            perm = [i, j] + [k for k in range(n_modes) if k not in (i, j)]
            T_perm = np.transpose(T, perm)
            shape_ij = T.shape[i] * T.shape[j]
            shape_rest = int(np.prod([T.shape[k] for k in range(n_modes) if k not in (i, j)]))
            mat = T_perm.reshape(shape_ij, shape_rest)

            sv = np.linalg.svd(mat, compute_uv=False)
            total = np.sum(sv**2)
            if total > 1e-30:
                # Concentration: how much is in the first singular value
                concentration = sv[0]**2 / total
            else:
                concentration = 0

            key = f"{mode_names[i]}-{mode_names[j]}"
            couplings[key] = {
                "concentration": concentration,
                "eff_rank": int(np.sum(sv / (sv[0] + 1e-30) > 1e-6)),
                "max_rank": min(shape_ij, shape_rest),
            }

    return couplings


def participation_ratio(H: NDArray) -> float:
    """Compute the participation ratio of singular values.

    PR = (sum sigma_i)^2 / (sum sigma_i^2)

    Ranges from 1 (rank-1) to N (full rank, uniform singular values).
    More continuous measure of effective dimensionality than threshold-based rank.

    Args:
        H: Matrix, shape (N, N).

    Returns:
        Participation ratio (float).
    """
    sv = np.linalg.svd(H, compute_uv=False)
    sv = sv[sv > 1e-30]
    if len(sv) == 0:
        return 0.0
    s1 = np.sum(sv)
    s2 = np.sum(sv**2)
    if s2 < 1e-60:
        return 0.0
    return s1**2 / s2
