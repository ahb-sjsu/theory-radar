#!/usr/bin/env python3
"""
Test the KAM persistence prediction:

    KAM stability radius = gamma * Delta_max

where:
    gamma = min Diophantine gap of the orbital frequencies
    Delta_max = largest eigenvalue gap in the coupling tensor

The conjecture: this product predicts periodicity better than
either factor alone. If confirmed, this completes the proof chain:
    Keplerian eigenvalues -> Davis-Kahan rigidity -> KAM persistence -> periodicity
"""

from __future__ import annotations

import logging
import numpy as np

from tensor_3body.hamiltonian import hessian_analytical
from tensor_3body.tensor_ops import effective_rank
from tensor_3body.sampling import config_to_phase_space_circular
from tensor_3body.integrator_gpu import integrate_batch

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


def diophantine_gap(freqs: np.ndarray, max_order: int = 10) -> float:
    """Compute the minimum Diophantine gap for a set of frequencies.

    gamma = min_{k in Z^n, 0<|k|<=max_order} |omega . k|

    This measures how far the frequency ratios are from low-order resonances.
    Larger gamma = more irrational = KAM tori more likely to persist.
    """
    pos = freqs[freqs > 1e-10]
    if len(pos) < 2:
        return 0.0

    # Normalize frequencies
    omega = pos / pos[0]
    n = len(omega)

    min_gap = np.inf
    # Check all integer vectors k with |k| <= max_order
    # For efficiency, only check pairs and triples
    for i in range(n):
        for j in range(i + 1, n):
            ratio = omega[i] / omega[j]
            # Check how close ratio is to p/q for small p, q
            for p in range(1, max_order + 1):
                for q in range(1, max_order + 1):
                    gap = abs(ratio - p / q)
                    if gap > 0:
                        # Weight by order: higher order resonances matter less
                        weighted_gap = gap * (p + q)
                        min_gap = min(min_gap, weighted_gap)

    return float(min_gap) if np.isfinite(min_gap) else 0.0


def eigenvalue_gap(H: np.ndarray) -> float:
    """Largest gap in the eigenvalue spectrum."""
    eigs = np.sort(np.linalg.eigvalsh(H))
    gaps = np.diff(eigs)
    return float(gaps.max()) if len(gaps) > 0 else 0.0


def characteristic_frequencies(H: np.ndarray) -> np.ndarray:
    """Orbital frequencies from qq block."""
    qq = H[0:6, 0:6]
    eigs = np.linalg.eigvalsh(qq)
    return np.sort(np.sqrt(np.abs(eigs)))[::-1]


def main():
    log.info("=" * 70)
    log.info("KAM PERSISTENCE TEST")
    log.info("Testing: KAM_radius = gamma * Delta predicts periodicity")
    log.info("=" * 70)

    m1, m2, m3 = 1.0, 1.0, 1.0
    N = 20000
    rng = np.random.RandomState(12345)

    # Generate fresh configs (not from landscape — independent test)
    r1_vals = 10 ** rng.uniform(-0.5, 2.0, N)
    r2_vals = 10 ** rng.uniform(-0.5, 2.0, N)
    theta_vals = rng.uniform(0.01, np.pi, N)
    phi_vals = rng.uniform(0, 2 * np.pi, N)

    log.info("Computing tensor features for %d configs...", N)

    z0_list = []
    features = {
        "rank": [],
        "delta": [],
        "gamma": [],
        "kam_radius": [],
        "freq_ratio": [],
        "n_clusters": [],
    }

    for i in range(N):
        z = config_to_phase_space_circular(
            r1_vals[i], r2_vals[i], theta_vals[i], phi_vals[i], m1, m2, m3
        )
        z0_list.append(z)

        H = hessian_analytical(z, m1, m2, m3)
        freqs = characteristic_frequencies(H)
        delta = eigenvalue_gap(H)
        gamma = diophantine_gap(freqs)
        kam_rad = gamma * delta

        pos_freqs = freqs[freqs > 1e-10]
        freq_ratio = pos_freqs[0] / pos_freqs[-1] if len(pos_freqs) >= 2 else 1.0

        # Cluster count
        n_cl = 1
        for k in range(len(pos_freqs) - 1):
            if pos_freqs[k] / (pos_freqs[k + 1] + 1e-30) > 10:
                n_cl += 1

        features["rank"].append(effective_rank(H))
        features["delta"].append(delta)
        features["gamma"].append(gamma)
        features["kam_radius"].append(kam_rad)
        features["freq_ratio"].append(freq_ratio)
        features["n_clusters"].append(n_cl)

    z0 = np.array(z0_list)
    for k in features:
        features[k] = np.array(features[k])

    log.info("Features computed. Integrating on GPU...")

    # Integrate
    result = integrate_batch(z0, m1, m2, m3, dt=0.005, n_steps=80000, gpu_id=0)

    valid = ~result["collision"]
    periodic = result["is_periodic"] & valid
    n_periodic = periodic.sum()
    log.info(
        "Ground truth: %d/%d periodic (%.2f%%)",
        n_periodic,
        valid.sum(),
        100 * n_periodic / valid.sum(),
    )

    # Test each predictor
    log.info("\n" + "=" * 70)
    log.info("PREDICTOR COMPARISON")
    log.info("=" * 70)

    predictors = {
        "rank <= 10": features["rank"][valid] <= 10,
        "delta > 100": features["delta"][valid] > 100,
        "gamma > 0.1": features["gamma"][valid] > 0.1,
        "kam_radius > 10": features["kam_radius"][valid] > 10,
        "kam_radius > 50": features["kam_radius"][valid] > 50,
        "kam_radius > 100": features["kam_radius"][valid] > 100,
        "clusters >= 2": features["n_clusters"][valid] >= 2,
        "delta > 50 AND gamma > 0.05": (features["delta"][valid] > 50)
        & (features["gamma"][valid] > 0.05),
    }

    actual = periodic[valid]

    log.info("%-35s  %6s %6s %6s %6s", "Predictor", "Prec", "Recall", "F1", "N_pred")
    for name, pred in predictors.items():
        tp = int((pred & actual).sum())
        fp = int((pred & ~actual).sum())
        fn = int((~pred & actual).sum())
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
        log.info(
            "%-35s  %5.1f%% %5.1f%% %5.1f%% %6d", name, 100 * prec, 100 * rec, 100 * f1, pred.sum()
        )

    # Correlation analysis
    log.info("\n--- Correlations with periodicity ---")
    actual_float = actual.astype(float)
    for name, vals in [
        ("rank", -features["rank"][valid]),  # negative because lower rank = more periodic
        ("delta", features["delta"][valid]),
        ("gamma", features["gamma"][valid]),
        ("kam_radius", features["kam_radius"][valid]),
        ("freq_ratio", features["freq_ratio"][valid]),
    ]:
        finite = np.isfinite(vals)
        if finite.sum() > 100:
            corr = np.corrcoef(vals[finite], actual_float[finite])[0, 1]
            log.info(
                "  %-20s vs periodic: r = %+.4f %s",
                name,
                corr,
                "***"
                if abs(corr) > 0.15
                else "**"
                if abs(corr) > 0.1
                else "*"
                if abs(corr) > 0.05
                else "",
            )

    # THE KEY TEST: does kam_radius predict better than delta or gamma alone?
    log.info("\n" + "=" * 70)
    log.info("KEY TEST: Is kam_radius = gamma * delta a better predictor")
    log.info("          than either gamma or delta alone?")
    log.info("=" * 70)

    # Sweep thresholds for each predictor and find best F1
    best_f1 = {}
    for name, vals in [
        ("delta", features["delta"][valid]),
        ("gamma", features["gamma"][valid]),
        ("kam_radius", features["kam_radius"][valid]),
    ]:
        best = 0
        best_thresh = 0
        finite_vals = vals[np.isfinite(vals)]
        if len(finite_vals) == 0:
            continue
        for pct in [50, 60, 70, 75, 80, 85, 90, 95]:
            thresh = (
                np.percentile(finite_vals[finite_vals > 0], pct)
                if (finite_vals > 0).sum() > 0
                else 0
            )
            if thresh == 0:
                continue
            pred = vals > thresh
            tp = int((pred & actual).sum())
            fp = int((pred & ~actual).sum())
            fn = int((~pred & actual).sum())
            prec = tp / (tp + fp) if (tp + fp) > 0 else 0
            rec = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
            if f1 > best:
                best = f1
                best_thresh = thresh
        best_f1[name] = (best, best_thresh)
        log.info("  %-15s best F1 = %.3f (threshold = %.2f)", name, best, best_thresh)

    if "kam_radius" in best_f1 and "delta" in best_f1 and "gamma" in best_f1:
        kam_f1 = best_f1["kam_radius"][0]
        delta_f1 = best_f1["delta"][0]
        gamma_f1 = best_f1["gamma"][0]
        if kam_f1 > max(delta_f1, gamma_f1):
            log.info(
                "\n  CONFIRMED: kam_radius (F1=%.3f) > delta (%.3f) and gamma (%.3f)",
                kam_f1,
                delta_f1,
                gamma_f1,
            )
            log.info("  The product gamma*delta is a BETTER predictor than either factor alone.")
            log.info("  This supports the proof chain:")
            log.info("    Keplerian eigenvalues -> Davis-Kahan rigidity -> KAM persistence")
        else:
            log.info(
                "\n  NOT CONFIRMED: kam_radius (F1=%.3f) vs delta (%.3f), gamma (%.3f)",
                kam_f1,
                delta_f1,
                gamma_f1,
            )
            log.info("  The product does not outperform individual factors.")

    # Save
    np.savez_compressed(
        "results/kam_test.npz",
        ranks=features["rank"],
        deltas=features["delta"],
        gammas=features["gamma"],
        kam_radii=features["kam_radius"],
        periodic=result["is_periodic"],
        collision=result["collision"],
    )
    log.info("\nSaved to results/kam_test.npz")


if __name__ == "__main__":
    main()
