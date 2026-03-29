#!/usr/bin/env python3
"""
A* search through formula space to discover the mechanism linking
tensor properties to periodicity.

Instead of guessing gamma*Delta, search over:
- Unary ops: log, sqrt, inv, square, abs, neg
- Binary ops: +, -, *, /, max, min
- Features: rank, delta, gamma, freq_ratio, energy, r2/r1, inner_rank,
            spectral_entropy, condition, core_sparsity, n_clusters

A* heuristic: F1 score of the formula as a periodicity predictor.
Fuzzing: sweep thresholds to find optimal decision boundary.

The discovered formula IS the theorem we're looking for.
"""

from __future__ import annotations

import heapq
import logging
import time
from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from tensor_3body.hamiltonian import hessian_analytical
from tensor_3body.tensor_ops import effective_rank, singular_values, participation_ratio
from tensor_3body.sampling import config_to_phase_space_circular
from tensor_3body.integrator_gpu import integrate_batch

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


# ============================================================
# Feature computation
# ============================================================

def compute_all_features(z, m1, m2, m3, r1, r2):
    """Compute every tensor feature we have."""
    H = hessian_analytical(z, m1, m2, m3)
    eigs = np.sort(np.linalg.eigvalsh(H))
    sv = singular_values(H)
    qq = H[0:6, 0:6]
    qq_eigs = np.sort(np.linalg.eigvalsh(qq))

    # Frequencies
    freqs = np.sort(np.sqrt(np.abs(qq_eigs)))[::-1]
    pos_freqs = freqs[freqs > 1e-10]

    # Eigenvalue gaps
    gaps = np.diff(eigs)
    max_gap = gaps.max()
    max_gap_pos = np.argmax(gaps)

    # QQ gaps
    qq_gaps = np.diff(np.sort(np.abs(qq_eigs)))
    qq_max_gap = qq_gaps.max() if len(qq_gaps) > 0 else 0

    # Frequency ratio
    freq_ratio = pos_freqs[0] / (pos_freqs[-1] + 1e-30) if len(pos_freqs) >= 2 else 1.0

    # Frequency clusters
    n_clusters = 1
    for k in range(len(pos_freqs) - 1):
        if pos_freqs[k] / (pos_freqs[k+1] + 1e-30) > 10:
            n_clusters += 1

    # Spectral entropy
    sv_pos = sv[sv > 1e-30]
    if len(sv_pos) > 0:
        p = sv_pos / sv_pos.sum()
        entropy = -np.sum(p * np.log(p + 1e-30))
    else:
        entropy = 0

    # QQ nonzero count
    qq_nonzero = np.sum(np.abs(qq_eigs) > 1e-6 * np.abs(qq_eigs).max())

    # Inner subsystem rank
    inner = np.zeros((6, 6))
    inner[0:3, 0:3] = H[0:3, 0:3]
    inner[3:6, 3:6] = H[6:9, 6:9]
    inner_rank = effective_rank(inner)

    # Energy
    from tensor_3body.hamiltonian import hamiltonian
    E = hamiltonian(z, m1, m2, m3)

    # Traces
    tr2 = np.trace(H @ H)

    # Keplerian ratio (should be 2.0 at rank 9)
    nonzero_qq = np.abs(qq_eigs[np.abs(qq_eigs) > 1e-6 * np.abs(qq_eigs).max()])
    if len(nonzero_qq) >= 2:
        sorted_nz = np.sort(nonzero_qq)[::-1]
        kepler_ratio = sorted_nz[0] / (sorted_nz[1] + 1e-30)
        kepler_deviation = abs(kepler_ratio - 2.0)
    else:
        kepler_ratio = 0
        kepler_deviation = 99

    return {
        "rank": effective_rank(H),
        "qq_rank": qq_nonzero,
        "inner_rank": inner_rank,
        "max_gap": max_gap,
        "qq_max_gap": qq_max_gap,
        "gap_position": max_gap_pos,
        "freq_ratio": freq_ratio,
        "n_clusters": n_clusters,
        "entropy": entropy,
        "condition": sv[0] / (sv[-1] + 1e-30),
        "energy": E,
        "r2_r1": r2 / (r1 + 1e-10),
        "r1": r1,
        "r2": r2,
        "tr2": tr2,
        "kepler_ratio": kepler_ratio,
        "kepler_deviation": kepler_deviation,
        "pr": participation_ratio(H),
        "max_eig": eigs[-1],
        "min_nonzero_eig": np.min(np.abs(eigs[np.abs(eigs) > 1e-10])) if np.any(np.abs(eigs) > 1e-10) else 0,
        "qq_gap_over_max": qq_max_gap / (np.abs(qq_eigs).max() + 1e-30),
    }


# ============================================================
# Formula tree for symbolic search
# ============================================================

class Formula:
    """A formula tree node."""
    def evaluate(self, features: dict) -> float:
        raise NotImplementedError

    def complexity(self) -> int:
        raise NotImplementedError

    def __repr__(self) -> str:
        raise NotImplementedError


class Leaf(Formula):
    def __init__(self, name: str):
        self.name = name

    def evaluate(self, f):
        return f.get(self.name, 0.0)

    def complexity(self):
        return 1

    def __repr__(self):
        return self.name


class Unary(Formula):
    def __init__(self, op: str, child: Formula):
        self.op = op
        self.child = child

    def evaluate(self, f):
        x = self.child.evaluate(f)
        if self.op == "log":
            return np.log(abs(x) + 1e-30)
        elif self.op == "sqrt":
            return np.sqrt(abs(x))
        elif self.op == "inv":
            return 1.0 / (x + 1e-30)
        elif self.op == "sq":
            return x * x
        elif self.op == "neg":
            return -x
        return x

    def complexity(self):
        return 1 + self.child.complexity()

    def __repr__(self):
        return f"{self.op}({self.child})"


class Binary(Formula):
    def __init__(self, op: str, left: Formula, right: Formula):
        self.op = op
        self.left = left
        self.right = right

    def evaluate(self, f):
        a = self.left.evaluate(f)
        b = self.right.evaluate(f)
        if self.op == "+":
            return a + b
        elif self.op == "-":
            return a - b
        elif self.op == "*":
            return a * b
        elif self.op == "/":
            return a / (b + 1e-30)
        elif self.op == "max":
            return max(a, b)
        elif self.op == "min":
            return min(a, b)
        return a

    def complexity(self):
        return 1 + self.left.complexity() + self.right.complexity()

    def __repr__(self):
        return f"({self.left} {self.op} {self.right})"


# ============================================================
# A* search
# ============================================================

def evaluate_formula_f1(formula: Formula, feature_dicts: list, actual: np.ndarray) -> float:
    """Evaluate a formula as a periodicity predictor. Sweep threshold, return best F1."""
    values = np.array([formula.evaluate(f) for f in feature_dicts])
    values = np.nan_to_num(values, nan=0, posinf=1e10, neginf=-1e10)

    if np.std(values) < 1e-15:
        return 0.0

    best_f1 = 0.0
    for pct in [30, 40, 50, 60, 70, 80, 85, 90, 95]:
        thresh = np.percentile(values, pct)
        pred = values > thresh
        tp = int((pred & actual).sum())
        fp = int((pred & ~actual).sum())
        fn = int((~pred & actual).sum())
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
        best_f1 = max(best_f1, f1)

        # Also try below threshold
        pred_low = values < thresh
        tp = int((pred_low & actual).sum())
        fp = int((pred_low & ~actual).sum())
        fn = int((~pred_low & actual).sum())
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
        best_f1 = max(best_f1, f1)

    return best_f1


@dataclass(order=True)
class SearchState:
    neg_f1: float  # negative F1 for min-heap (we want to maximize F1)
    formula: Formula = field(compare=False)
    depth: int = field(compare=False)


def search_formulas(
    feature_dicts: list,
    actual: np.ndarray,
    max_depth: int = 3,
    max_states: int = 5000,
    beam_width: int = 200,
) -> list[tuple[float, Formula]]:
    """A* search through formula space."""

    feature_names = [
        "rank", "qq_rank", "inner_rank", "max_gap", "qq_max_gap",
        "gap_position", "freq_ratio", "n_clusters", "entropy",
        "energy", "r2_r1", "kepler_deviation", "pr",
        "qq_gap_over_max", "max_eig", "min_nonzero_eig",
    ]
    unary_ops = ["log", "sqrt", "inv", "sq", "neg"]
    binary_ops = ["+", "-", "*", "/"]

    # Initialize with leaf nodes
    frontier = []
    results = []

    for name in feature_names:
        leaf = Leaf(name)
        f1 = evaluate_formula_f1(leaf, feature_dicts, actual)
        heapq.heappush(frontier, SearchState(-f1, leaf, 1))
        results.append((f1, leaf))

    explored = 0
    seen = set()

    while frontier and explored < max_states:
        state = heapq.heappop(frontier)
        explored += 1
        formula = state.formula
        depth = state.depth

        formula_str = repr(formula)
        if formula_str in seen:
            continue
        seen.add(formula_str)

        if depth >= max_depth:
            continue

        # Keep only top beam_width formulas in frontier
        if len(frontier) > beam_width * 3:
            frontier = heapq.nsmallest(beam_width, frontier)
            heapq.heapify(frontier)

        # Expand: unary operations on this formula
        for op in unary_ops:
            child = Unary(op, formula)
            if child.complexity() <= max_depth * 2 + 1:
                f1 = evaluate_formula_f1(child, feature_dicts, actual)
                heapq.heappush(frontier, SearchState(-f1, child, depth + 1))
                results.append((f1, child))

        # Expand: binary operations with each leaf
        for name in feature_names:
            leaf = Leaf(name)
            for op in binary_ops:
                child1 = Binary(op, formula, leaf)
                child2 = Binary(op, leaf, formula)
                for child in [child1, child2]:
                    if child.complexity() <= max_depth * 2 + 1:
                        f1 = evaluate_formula_f1(child, feature_dicts, actual)
                        heapq.heappush(frontier, SearchState(-f1, child, depth + 1))
                        results.append((f1, child))

        if explored % 500 == 0:
            best = max(results, key=lambda x: x[0])
            log.info("  Explored %d formulas, best F1=%.3f: %s",
                     explored, best[0], best[1])

    results.sort(key=lambda x: -x[0])
    return results[:20]


def main():
    log.info("=" * 70)
    log.info("FORMULA SEARCH: A* through mathematical operation space")
    log.info("Discover the mechanism linking tensor properties to periodicity")
    log.info("=" * 70)

    m1, m2, m3 = 1.0, 1.0, 1.0
    N = 10000
    rng = np.random.RandomState(99)

    r1_vals = 10 ** rng.uniform(-0.5, 2.0, N)
    r2_vals = 10 ** rng.uniform(-0.5, 2.0, N)
    theta_vals = rng.uniform(0.01, np.pi, N)
    phi_vals = rng.uniform(0, 2 * np.pi, N)

    log.info("Computing features for %d configs...", N)
    t0 = time.time()
    feature_dicts = []
    z0_list = []
    for i in range(N):
        z = config_to_phase_space_circular(r1_vals[i], r2_vals[i], theta_vals[i], phi_vals[i], m1, m2, m3)
        z0_list.append(z)
        feat = compute_all_features(z, m1, m2, m3, r1_vals[i], r2_vals[i])
        feature_dicts.append(feat)
    log.info("Features: %.1f s", time.time() - t0)

    # Integrate for ground truth
    z0 = np.array(z0_list)
    log.info("Integrating %d orbits on GPU...", N)
    result = integrate_batch(z0, m1, m2, m3, dt=0.005, n_steps=80000, gpu_id=0)

    valid = ~result["collision"]
    actual = result["is_periodic"] & valid
    log.info("Ground truth: %d/%d periodic (%.2f%%)",
             actual.sum(), valid.sum(), 100 * actual.sum() / valid.sum())

    # Filter to valid configs
    valid_features = [f for f, v in zip(feature_dicts, valid) if v]
    valid_actual = actual[valid]

    # Search
    log.info("\nSearching formula space (max_depth=3, max_states=5000)...")
    t0 = time.time()
    results = search_formulas(valid_features, valid_actual,
                               max_depth=3, max_states=5000)
    elapsed = time.time() - t0

    log.info("\n" + "=" * 70)
    log.info("TOP 20 DISCOVERED FORMULAS")
    log.info("=" * 70)
    for f1, formula in results:
        log.info("  F1=%.3f  %s", f1, formula)

    log.info("\nSearch time: %.1f s, explored %d+ formulas", elapsed, 5000)

    # Analyze the best formula
    best_f1, best_formula = results[0]
    log.info("\n--- Best formula analysis ---")
    log.info("Formula: %s", best_formula)
    log.info("F1: %.3f", best_f1)

    # Compute values and find optimal threshold
    values = np.array([best_formula.evaluate(f) for f in valid_features])
    values = np.nan_to_num(values, nan=0, posinf=1e10, neginf=-1e10)

    best_thresh_f1 = 0
    best_thresh = 0
    best_direction = ">"
    for pct in range(10, 96):
        thresh = np.percentile(values, pct)
        for direction in [">", "<"]:
            pred = values > thresh if direction == ">" else values < thresh
            tp = int((pred & valid_actual).sum())
            fp = int((pred & ~valid_actual).sum())
            fn = int((~pred & valid_actual).sum())
            prec = tp / (tp + fp) if (tp + fp) > 0 else 0
            rec = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
            if f1 > best_thresh_f1:
                best_thresh_f1 = f1
                best_thresh = thresh
                best_direction = direction

    log.info("Optimal threshold: %s %.4f (F1=%.3f)", best_direction, best_thresh, best_thresh_f1)

    # The formula IS the conjecture
    log.info("\n" + "=" * 70)
    log.info("DISCOVERED CONJECTURE:")
    log.info("  Periodicity is predicted by: %s %s %.4f",
             best_formula, best_direction, best_thresh)
    log.info("  F1 = %.3f", best_thresh_f1)
    log.info("=" * 70)


if __name__ == "__main__":
    main()
