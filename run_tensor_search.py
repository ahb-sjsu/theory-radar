#!/usr/bin/env python3
"""
A* search through tensor operation space + fuzzing of tensor structure.

Instead of searching coordinate transforms, search tensor operations
to discover what structure the coupling tensor contains.

Operations:
- Partial traces (contract pairs of indices)
- Mode permutations (reorder the (body, spatial, phase) modes)
- Slicing (fix one mode, analyze the sub-tensor)
- CP decomposition at varying ranks
- Symmetrization / anti-symmetrization of mode pairs
- Kronecker factorization attempts

Fuzzing:
- Perturb tensor entries, measure which perturbations preserve
  the eigenvalue spectrum (= symmetries of the dynamics)
- Find the minimal perturbation that changes periodicity (= stability boundary)
"""

from __future__ import annotations

import logging
import time

import numpy as np
from numpy.typing import NDArray

from tensor_3body.hamiltonian import hessian_analytical
from tensor_3body.tensor_ops import reshape_to_rank6, effective_rank
from tensor_3body.sampling import config_to_phase_space_circular
from tensor_3body.landscape import load_landscape

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

try:
    import tensorly
    from tensorly.decomposition import tucker, parafac

    HAS_TENSORLY = True
except ImportError:
    HAS_TENSORLY = False


# ============================================================
# Tensor analysis operations
# ============================================================


def core_sparsity(T: NDArray) -> float:
    """Fraction of near-zero entries in the Tucker core tensor."""
    if not HAS_TENSORLY:
        return 0.0
    core, factors = tucker(T, rank=T.shape)
    flat = core.ravel()
    threshold = 0.01 * np.abs(flat).max()
    return float(np.sum(np.abs(flat) < threshold)) / len(flat)


def cp_rank_estimate(T: NDArray, max_rank: int = 12) -> int:
    """Estimate CP rank by fitting at increasing ranks until error < threshold."""
    if not HAS_TENSORLY:
        return -1
    T_norm = np.linalg.norm(T)
    if T_norm < 1e-30:
        return 0
    for r in range(1, max_rank + 1):
        try:
            weights, factors = parafac(
                T, rank=r, init="random", n_iter_max=100, random_state=42, normalize_factors=True
            )
            T_approx = tensorly.cp_to_tensor((weights, factors))
            error = np.linalg.norm(T - T_approx) / T_norm
            if error < 0.05:
                return r
        except Exception:
            continue
    return max_rank


def mode_pair_coupling(T: NDArray) -> dict:
    """Measure coupling strength between all pairs of modes.

    For each pair (i,j), contract all other modes and measure
    the off-diagonal energy of the resulting matrix.
    """
    n_modes = len(T.shape)
    couplings = {}

    for i in range(n_modes):
        for j in range(i + 1, n_modes):
            # Bring modes i,j to front, flatten the rest
            perm = [i, j] + [k for k in range(n_modes) if k not in (i, j)]
            T_perm = np.transpose(T, perm)
            mat = T_perm.reshape(T.shape[i] * T.shape[j], -1)

            # SVD of the (i,j)-unfolding
            sv = np.linalg.svd(mat, compute_uv=False)
            total = np.sum(sv**2)

            # Concentration: how much in first singular value
            if total > 1e-30:
                concentration = sv[0] ** 2 / total
                eff_rank = int(np.sum(sv / (sv[0] + 1e-30) > 1e-4))
            else:
                concentration = 0
                eff_rank = 0

            mode_names = ["body1", "spat1", "phase1", "body2", "spat2", "phase2"]
            key = f"{mode_names[i]}-{mode_names[j]}"
            couplings[key] = {
                "concentration": concentration,
                "eff_rank": eff_rank,
                "max_rank": min(
                    T.shape[i] * T.shape[j],
                    int(np.prod([T.shape[k] for k in range(n_modes) if k not in (i, j)])),
                ),
            }

    return couplings


def tensor_invariants(T: NDArray) -> dict:
    """Compute coordinate-independent tensor invariants."""
    H = T.reshape(12, 12) if T.ndim > 2 else T

    eigenvalues = np.linalg.eigvalsh(H)
    sv = np.linalg.svd(H, compute_uv=False)

    # Traces of powers (invariant under orthogonal transforms)
    tr1 = np.trace(H)
    tr2 = np.trace(H @ H)
    tr3 = np.trace(H @ H @ H)

    # Determinant (sign indicates stability character)
    det = np.linalg.det(H)

    # Condition number (ratio of largest to smallest singular value)
    cond = sv[0] / (sv[-1] + 1e-30)

    # Spectral entropy
    sv_pos = sv[sv > 1e-30]
    if len(sv_pos) > 0:
        p = sv_pos / sv_pos.sum()
        entropy = -np.sum(p * np.log(p + 1e-30))
    else:
        entropy = 0

    # Frobenius norm
    frob = np.linalg.norm(H, "fro")

    return {
        "tr1": tr1,
        "tr2": tr2,
        "tr3": tr3,
        "det_sign": np.sign(det),
        "log_det": np.log(abs(det) + 1e-30),
        "condition": cond,
        "spectral_entropy": entropy,
        "frobenius": frob,
        "eigenvalue_spread": eigenvalues[-1] - eigenvalues[0],
        "eigenvalue_gap": float(np.max(np.diff(np.sort(eigenvalues)))),
    }


def symmetry_score(T: NDArray) -> dict:
    """Measure tensor symmetries by testing invariance under mode operations."""
    scores = {}

    # Body exchange symmetry: swap body1 <-> body2
    # In rank-6 tensor: swap axes (0,3), (1,4), (2,5)
    T_swap = np.transpose(T, (3, 4, 5, 0, 1, 2))
    norm = np.linalg.norm(T)
    if norm > 1e-30:
        scores["body_exchange"] = 1.0 - np.linalg.norm(T - T_swap) / (2 * norm)
    else:
        scores["body_exchange"] = 0.0

    # Spatial isotropy: average over rotations (approximate)
    # Check if spatial modes (1,4) are equivalent
    # Contract body and phase modes, compare spatial structure
    # Contract body and phase modes to get spatial1-spatial2 coupling (3x3)
    # T shape: (body1=2, spat1=3, phase1=2, body2=2, spat2=3, phase2=2)
    # Sum over body1, phase1, body2, phase2 -> (spat1=3, spat2=3)
    if T.ndim == 6:
        spatial = np.einsum("abcdef->be", T)  # sum over body(a,d) and phase(c,f) -> (3,3)
    else:
        spatial = np.eye(3)
    spatial_sym = np.linalg.norm(spatial - spatial.T) / (np.linalg.norm(spatial) + 1e-30)
    scores["spatial_symmetry"] = 1.0 - spatial_sym

    # Phase symmetry: is q-p coupling zero? (for separable H = T(p) + V(q))
    # Modes 2 and 5 are phase modes
    # In the flattened 12x12, the qp block is H[0:6, 6:12]
    H = T.reshape(12, 12)
    qp_block = np.linalg.norm(H[0:6, 6:12])
    total = np.linalg.norm(H)
    scores["phase_decoupling"] = 1.0 - qp_block / (total + 1e-30)

    return scores


# ============================================================
# Tensor fuzzing
# ============================================================


def fuzz_tensor_symmetries(
    H: NDArray,
    n_perturbations: int = 200,
    amplitude: float = 0.01,
) -> dict:
    """Fuzz the Hessian to discover which perturbations preserve the spectrum.

    Perturbations that don't change eigenvalues = symmetries of the dynamics.
    """
    eigenvalues_0 = np.sort(np.linalg.eigvalsh(H))
    sv_0 = np.linalg.svd(H, compute_uv=False)
    rng = np.random.RandomState(42)

    results = {
        "n_spectrum_preserving": 0,
        "n_rank_preserving": 0,
        "spectrum_sensitivity": [],
        "rank_changes": [],
    }

    rank_0 = effective_rank(H)

    for _ in range(n_perturbations):
        # Random symmetric perturbation
        dH = rng.randn(12, 12) * amplitude
        dH = (dH + dH.T) / 2  # symmetric
        H_pert = H + dH

        eigenvalues_pert = np.sort(np.linalg.eigvalsh(H_pert))
        rank_pert = effective_rank(H_pert)

        # Spectrum change
        spec_change = np.linalg.norm(eigenvalues_pert - eigenvalues_0) / (
            np.linalg.norm(eigenvalues_0) + 1e-30
        )
        results["spectrum_sensitivity"].append(spec_change)

        if spec_change < 0.01:
            results["n_spectrum_preserving"] += 1
        if rank_pert == rank_0:
            results["n_rank_preserving"] += 1
        results["rank_changes"].append(rank_pert - rank_0)

    results["spectrum_sensitivity"] = np.array(results["spectrum_sensitivity"])
    results["mean_sensitivity"] = float(results["spectrum_sensitivity"].mean())
    results["rank_changes"] = np.array(results["rank_changes"])
    results["rank_stability"] = results["n_rank_preserving"] / n_perturbations

    return results


# ============================================================
# Main analysis
# ============================================================


def main():
    log.info("=" * 70)
    log.info("HIGHER-ORDER TENSOR SEARCH")
    log.info("A* on tensor operations + fuzzing for symmetries")
    log.info("=" * 70)

    data, masses = load_landscape("results/phase1/landscape_equal_static.npz")
    m1, m2, m3 = float(masses[0]), float(masses[1]), float(masses[2])

    # Sample across rank spectrum
    configs = []
    for target_rank in [7, 8, 9, 10, 11, 12]:
        mask = data["eff_rank"] == target_rank
        indices = np.where(mask)[0]
        if len(indices) > 50:
            indices = np.random.choice(indices, size=50, replace=False)
        for idx in indices:
            z = config_to_phase_space_circular(
                float(data["r1"][idx]),
                float(data["r2"][idx]),
                float(data["theta"][idx]),
                float(data["phi"][idx]),
                m1,
                m2,
                m3,
            )
            configs.append(
                {
                    "z": z,
                    "rank": int(data["eff_rank"][idx]),
                    "r1": float(data["r1"][idx]),
                    "r2": float(data["r2"][idx]),
                }
            )

    log.info("Analyzing %d configurations across ranks 7-12", len(configs))

    # Compute all higher-order properties
    all_results = []
    t0 = time.time()

    for i, cfg in enumerate(configs):
        H = hessian_analytical(cfg["z"], m1, m2, m3)
        T6 = reshape_to_rank6(H)

        # Core sparsity
        sparsity = core_sparsity(T6)

        # CP rank estimate
        cp_rank = cp_rank_estimate(T6, max_rank=8)

        # Mode pair coupling
        couplings = mode_pair_coupling(T6)

        # Tensor invariants
        invariants = tensor_invariants(T6)

        # Symmetry scores
        symmetries = symmetry_score(T6)

        # Fuzz for spectral stability
        fuzz = fuzz_tensor_symmetries(H, n_perturbations=100)

        result = {
            "rank": cfg["rank"],
            "r1": cfg["r1"],
            "r2": cfg["r2"],
            "core_sparsity": sparsity,
            "cp_rank": cp_rank,
            "body_exchange_sym": symmetries["body_exchange"],
            "spatial_sym": symmetries["spatial_symmetry"],
            "phase_decoupling": symmetries["phase_decoupling"],
            "spectral_entropy": invariants["spectral_entropy"],
            "condition": invariants["condition"],
            "eigenvalue_gap": invariants["eigenvalue_gap"],
            "tr2": invariants["tr2"],
            "fuzz_mean_sensitivity": fuzz["mean_sensitivity"],
            "fuzz_rank_stability": fuzz["rank_stability"],
            "fuzz_n_spectrum_preserving": fuzz["n_spectrum_preserving"],
        }

        # Strongest and weakest mode couplings
        sorted_coups = sorted(couplings.items(), key=lambda x: x[1]["concentration"])
        result["weakest_coupling"] = sorted_coups[0][0]
        result["weakest_coupling_conc"] = sorted_coups[0][1]["concentration"]
        result["strongest_coupling"] = sorted_coups[-1][0]
        result["strongest_coupling_conc"] = sorted_coups[-1][1]["concentration"]

        all_results.append(result)

        if (i + 1) % 50 == 0:
            log.info("  Analyzed %d/%d configs...", i + 1, len(configs))

    elapsed = time.time() - t0
    log.info("Analysis complete in %.1f s", elapsed)

    # Group by rank and find correlations
    log.info("\n" + "=" * 70)
    log.info("RESULTS BY RANK")
    log.info("=" * 70)

    properties = [
        "core_sparsity",
        "cp_rank",
        "body_exchange_sym",
        "spatial_sym",
        "phase_decoupling",
        "spectral_entropy",
        "condition",
        "eigenvalue_gap",
        "fuzz_mean_sensitivity",
        "fuzz_rank_stability",
    ]

    log.info(
        "\n%-6s  " + "  ".join("%-12s" for _ in properties), "Rank", *[p[:12] for p in properties]
    )

    for rank in [7, 8, 9, 10, 11, 12]:
        subset = [r for r in all_results if r["rank"] == rank]
        if not subset:
            continue
        vals = []
        for prop in properties:
            v = np.mean([r[prop] for r in subset])
            vals.append(v)
        log.info("%-6d  " + "  ".join("%-12.4f" for _ in vals), rank, *vals)

    # Novel correlations: which properties best distinguish low-rank from high-rank?
    log.info("\n" + "=" * 70)
    log.info("NOVEL CORRELATIONS")
    log.info("=" * 70)

    ranks_arr = np.array([r["rank"] for r in all_results])
    for prop in properties:
        vals = np.array([r[prop] for r in all_results])
        # Handle NaN/inf
        valid = np.isfinite(vals)
        if valid.sum() < 10:
            continue
        corr = np.corrcoef(ranks_arr[valid], vals[valid])[0, 1]
        if abs(corr) > 0.1:
            log.info(
                "  %-25s vs rank: r = %+.4f %s",
                prop,
                corr,
                "***" if abs(corr) > 0.5 else "**" if abs(corr) > 0.3 else "*",
            )

    # Mode coupling patterns
    log.info("\n--- Mode Coupling Patterns ---")
    for rank in [7, 9, 12]:
        subset = [r for r in all_results if r["rank"] == rank]
        if not subset:
            continue
        # Most common strongest coupling
        from collections import Counter

        strongest = Counter(r["strongest_coupling"] for r in subset)
        weakest = Counter(r["weakest_coupling"] for r in subset)
        log.info(
            "  rank=%d: strongest=%s, weakest=%s",
            rank,
            strongest.most_common(1)[0],
            weakest.most_common(1)[0],
        )

    # Fuzz discoveries: do low-rank configs have more spectral symmetries?
    log.info("\n--- Fuzzing: Spectral Stability by Rank ---")
    for rank in [7, 8, 9, 10, 11, 12]:
        subset = [r for r in all_results if r["rank"] == rank]
        if not subset:
            continue
        mean_sens = np.mean([r["fuzz_mean_sensitivity"] for r in subset])
        mean_stab = np.mean([r["fuzz_rank_stability"] for r in subset])
        mean_pres = np.mean([r["fuzz_n_spectrum_preserving"] for r in subset])
        log.info(
            "  rank=%d: sensitivity=%.4f, rank_stability=%.2f, spectrum_preserving=%.1f/100",
            rank,
            mean_sens,
            mean_stab,
            mean_pres,
        )


if __name__ == "__main__":
    main()
