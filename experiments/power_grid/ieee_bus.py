#!/usr/bin/env python3
"""
Tensor rank analysis of power grid stability.

Constructs the admittance matrix (Ybus) and power flow Jacobian for
standard IEEE test cases and analyzes their tensor structure.

The coupling tensor has shape (N_bus, 2, N_bus, 2) where the 2 modes
are active power (P) and reactive power (Q). Low-rank decomposition
identifies buses that can be controlled independently.

Uses standard IEEE test cases (14-bus, 30-bus, 57-bus, 118-bus).
"""

from __future__ import annotations

import logging
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

try:
    import tensorly
    HAS_TENSORLY = True
except ImportError:
    HAS_TENSORLY = False


# IEEE 14-bus system (standard test case)
# Format: (from_bus, to_bus, resistance, reactance, susceptance)
IEEE_14_BUS_LINES = [
    (1, 2, 0.01938, 0.05917, 0.0528),
    (1, 5, 0.05403, 0.22304, 0.0492),
    (2, 3, 0.04699, 0.19797, 0.0438),
    (2, 4, 0.05811, 0.17632, 0.0340),
    (2, 5, 0.05695, 0.17388, 0.0346),
    (3, 4, 0.06701, 0.17103, 0.0128),
    (4, 5, 0.01335, 0.04211, 0.0),
    (4, 7, 0.0, 0.20912, 0.0),
    (4, 9, 0.0, 0.55618, 0.0),
    (5, 6, 0.0, 0.25202, 0.0),
    (6, 11, 0.09498, 0.19890, 0.0),
    (6, 12, 0.12291, 0.25581, 0.0),
    (6, 13, 0.06615, 0.13027, 0.0),
    (7, 8, 0.0, 0.17615, 0.0),
    (7, 9, 0.11001, 0.20640, 0.0),
    (9, 10, 0.03181, 0.08450, 0.0),
    (9, 14, 0.12711, 0.27038, 0.0),
    (10, 11, 0.08205, 0.19207, 0.0),
    (12, 13, 0.22092, 0.19988, 0.0),
    (13, 14, 0.17093, 0.34802, 0.0),
]

# IEEE 30-bus system (abbreviated — key lines)
IEEE_30_BUS_LINES = [
    (1, 2, 0.0192, 0.0575, 0.0528),
    (1, 3, 0.0452, 0.1852, 0.0408),
    (2, 4, 0.0570, 0.1737, 0.0368),
    (3, 4, 0.0132, 0.0379, 0.0084),
    (2, 5, 0.0472, 0.1983, 0.0418),
    (2, 6, 0.0581, 0.1763, 0.0374),
    (4, 6, 0.0119, 0.0414, 0.0090),
    (5, 7, 0.0460, 0.1160, 0.0204),
    (6, 7, 0.0267, 0.0820, 0.0170),
    (6, 8, 0.0120, 0.0420, 0.0090),
    (6, 9, 0.0, 0.2080, 0.0),
    (6, 10, 0.0, 0.5560, 0.0),
    (9, 11, 0.0, 0.2080, 0.0),
    (9, 10, 0.0, 0.1100, 0.0),
    (4, 12, 0.0, 0.2560, 0.0),
    (12, 13, 0.0, 0.1400, 0.0),
    (12, 14, 0.1231, 0.2559, 0.0),
    (12, 15, 0.0662, 0.1304, 0.0),
    (12, 16, 0.0945, 0.1987, 0.0),
    (14, 15, 0.2210, 0.1997, 0.0),
    (16, 17, 0.0524, 0.1923, 0.0),
    (15, 18, 0.1073, 0.2185, 0.0),
    (18, 19, 0.0639, 0.1292, 0.0),
    (19, 20, 0.0340, 0.0680, 0.0),
    (10, 20, 0.0936, 0.2090, 0.0),
    (10, 17, 0.0324, 0.0845, 0.0),
    (10, 21, 0.0348, 0.0749, 0.0),
    (10, 22, 0.0727, 0.1499, 0.0),
    (21, 22, 0.0116, 0.0236, 0.0),
    (15, 23, 0.1000, 0.2020, 0.0),
    (22, 24, 0.1150, 0.1790, 0.0),
    (23, 24, 0.1320, 0.2700, 0.0),
    (24, 25, 0.1885, 0.3292, 0.0),
    (25, 26, 0.2544, 0.3800, 0.0),
    (25, 27, 0.1093, 0.2087, 0.0),
    (27, 28, 0.0, 0.3960, 0.0),
    (27, 29, 0.2198, 0.4153, 0.0),
    (27, 30, 0.3202, 0.6027, 0.0),
    (29, 30, 0.2399, 0.4533, 0.0),
    (8, 28, 0.0636, 0.2000, 0.0428),
    (6, 28, 0.0169, 0.0599, 0.0130),
]


def build_ybus(lines: list[tuple], n_bus: int) -> np.ndarray:
    """Build the bus admittance matrix from line data.

    Returns complex Ybus matrix, shape (n_bus, n_bus).
    """
    Y = np.zeros((n_bus, n_bus), dtype=complex)

    for from_b, to_b, r, x, b in lines:
        i, j = from_b - 1, to_b - 1  # 0-indexed
        if abs(r) + abs(x) < 1e-20:
            continue
        y = 1.0 / complex(r, x)
        Y[i, j] -= y
        Y[j, i] -= y
        Y[i, i] += y + complex(0, b / 2)
        Y[j, j] += y + complex(0, b / 2)

    return Y


def ybus_to_coupling_tensor(Y: np.ndarray) -> np.ndarray:
    """Convert Ybus to a real-valued coupling tensor.

    Shape: (N, 2, N, 2) where the 2 modes are real(Y) and imag(Y),
    corresponding to resistive (P) and reactive (Q) coupling.
    """
    N = Y.shape[0]
    T = np.zeros((N, 2, N, 2))

    # Real part: resistive coupling (active power)
    T[:, 0, :, 0] = Y.real
    # Imaginary part: reactive coupling (reactive power)
    T[:, 1, :, 1] = Y.imag
    # Cross terms: P-Q coupling
    T[:, 0, :, 1] = -Y.imag
    T[:, 1, :, 0] = Y.imag

    return T


def analyze_grid(T: np.ndarray, name: str) -> dict:
    """Analyze coupling tensor of a power grid."""
    N = T.shape[0]
    H = T.reshape(N * 2, N * 2)

    sv = np.linalg.svd(H, compute_uv=False)
    matrix_rank = int(np.sum(sv / (sv[0] + 1e-30) > 1e-6))

    # Multilinear rank
    ml_ranks = []
    for mode in range(4):
        if HAS_TENSORLY:
            unf = tensorly.unfold(T, mode)
        else:
            unf = T.reshape(T.shape[mode], -1) if mode < 2 else T.transpose(
                list(range(mode, 4)) + list(range(mode))).reshape(T.shape[mode], -1)
        sv_mode = np.linalg.svd(unf, compute_uv=False)
        r = int(np.sum(sv_mode / (sv_mode[0] + 1e-30) > 1e-6))
        ml_ranks.append(r)

    # Block structure: does P decouple from Q?
    PP = np.linalg.norm(T[:, 0, :, 0])
    QQ = np.linalg.norm(T[:, 1, :, 1])
    PQ = np.linalg.norm(T[:, 0, :, 1])
    pq_coupling = PQ / (PP + QQ + 1e-30)

    # Participation ratio
    sv_pos = sv[sv > 1e-30]
    pr = np.sum(sv_pos)**2 / np.sum(sv_pos**2) if len(sv_pos) > 0 else 0

    # Area decomposition: check if grid has natural clusters
    # Use spectral clustering on |Ybus|
    from scipy.sparse.csgraph import connected_components
    adjacency = (np.abs(H) > 1e-6).astype(int)
    n_components, labels = connected_components(adjacency, directed=False)

    log.info("%-20s  N=%d  rank=%d/%d  ml=%s  PR=%.2f  PQ_coupling=%.3f  components=%d",
             name, N, matrix_rank, 2 * N, ml_ranks, pr, pq_coupling, n_components)

    return {
        "name": name,
        "n_bus": N,
        "matrix_rank": matrix_rank,
        "max_rank": 2 * N,
        "multilinear_rank": ml_ranks,
        "participation_ratio": pr,
        "pq_coupling": pq_coupling,
        "n_components": n_components,
    }


def main():
    log.info("=" * 70)
    log.info("Power Grid: Tensor Rank Analysis of IEEE Bus Systems")
    log.info("=" * 70)

    results = []

    # IEEE 14-bus
    Y14 = build_ybus(IEEE_14_BUS_LINES, 14)
    T14 = ybus_to_coupling_tensor(Y14)
    r = analyze_grid(T14, "IEEE 14-bus")
    results.append(r)

    # IEEE 30-bus
    Y30 = build_ybus(IEEE_30_BUS_LINES, 30)
    T30 = ybus_to_coupling_tensor(Y30)
    r = analyze_grid(T30, "IEEE 30-bus")
    results.append(r)

    # Stressed grid: scale line impedances to simulate overload
    log.info("\n--- Stress test: scaling impedances ---")
    for scale in [0.5, 0.75, 1.0, 1.25, 1.5, 2.0]:
        stressed_lines = [(f, t, r * scale, x * scale, b)
                          for f, t, r, x, b in IEEE_14_BUS_LINES]
        Y = build_ybus(stressed_lines, 14)
        T = ybus_to_coupling_tensor(Y)
        r = analyze_grid(T, f"14-bus stress={scale:.2f}")
        r["stress"] = scale
        results.append(r)

    # Line removal: simulate failures
    log.info("\n--- Line failure simulation ---")
    for remove_idx in range(0, len(IEEE_14_BUS_LINES), 4):
        remaining = [l for i, l in enumerate(IEEE_14_BUS_LINES) if i != remove_idx]
        removed = IEEE_14_BUS_LINES[remove_idx]
        Y = build_ybus(remaining, 14)
        T = ybus_to_coupling_tensor(Y)
        r = analyze_grid(T, f"14-bus no line {removed[0]}-{removed[1]}")
        r["removed_line"] = (removed[0], removed[1])
        results.append(r)

    log.info("\n" + "=" * 70)
    log.info("SUMMARY")
    log.info("=" * 70)
    for r in results:
        log.info("%-30s rank=%d/%d  PR=%.2f  PQ=%.3f",
                 r["name"], r["matrix_rank"], r["max_rank"],
                 r["participation_ratio"], r["pq_coupling"])


if __name__ == "__main__":
    main()
