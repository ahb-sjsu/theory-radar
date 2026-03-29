"""
DAG of Admissible Heuristics for A* Formula Search.

Combines multiple admissible heuristic functions via max() to produce
a tighter bound while maintaining admissibility. Heuristics are
organized from cheapest to most expensive and evaluated lazily.

Theorem: If h₁, h₂, ..., hₖ are each admissible (never overestimate
the true remaining cost), then h = max(h₁, ..., hₖ) is also admissible.

Proof: For each i, hᵢ(n) ≤ h*(n) (true remaining cost).
Therefore max(h₁, ..., hₖ) ≤ h*(n). QED.

For formula search, "cost" = formula depth, and the heuristics
estimate the minimum additional depth needed to reach F1 ≥ target.
"""

from __future__ import annotations

import heapq
import logging
import time
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray
from sklearn.metrics import roc_auc_score

log = logging.getLogger(__name__)


def exact_optimal_f1(values: NDArray, actual: NDArray) -> float:
    """O(N log N) exact optimal thresholded F1."""
    values = np.nan_to_num(values, nan=0, posinf=1e10, neginf=-1e10)
    actual = actual.astype(bool)
    P = int(actual.sum())
    N = len(actual)
    if P == 0 or P == N:
        return 0.0
    best = 0.0
    for direction in [1, -1]:
        order = np.argsort(direction * values)[::-1]
        sa = actual[order]
        tp, fp = 0, 0
        for i in range(N):
            if sa[i]:
                tp += 1
            else:
                fp += 1
            if tp + fp > 0:
                p = tp / (tp + fp)
                r = tp / P
                if p + r > 0:
                    best = max(best, 2 * p * r / (p + r))
    return best


def auroc_safe(values: NDArray, actual: NDArray) -> float:
    """Direction-invariant AUROC."""
    try:
        v = np.nan_to_num(values, nan=0, posinf=1e10, neginf=-1e10)
        if len(np.unique(actual)) < 2:
            return 0.5
        a = roc_auc_score(actual, v)
        return max(a, 1 - a)
    except Exception:
        return 0.5


def max_f1_for_auroc(auroc: float, prevalence: float) -> float:
    """Proven upper bound on F1 given AUROC (Theorem 3)."""
    if auroc <= 0.5:
        return 2 * prevalence
    J = 2 * auroc - 1
    best = 0.0
    for tpr in np.linspace(max(0.01, J), 1.0, 200):
        fpr = tpr - J
        if fpr < 0 or fpr > 1:
            continue
        prec = tpr * prevalence / (tpr * prevalence + fpr * (1 - prevalence) + 1e-30)
        if prec + tpr > 0:
            f1 = 2 * prec * tpr / (prec + tpr)
            best = max(best, f1)
    return best


# ============================================================
# Individual Admissible Heuristics
# ============================================================

class H1_TargetCheck:
    """h₁: 0 if current F1 ≥ target, 1 otherwise. O(1)."""

    def __init__(self, target: float):
        self.target = target
        self.name = "H1_target"

    def __call__(self, f1: float, **kwargs) -> int:
        return 0 if f1 >= self.target else 1


class H2_AUROCBound:
    """h₂: 0 if AUROC-F1 bound ≥ target, 1 otherwise. O(N log N)."""

    def __init__(self, target: float, prevalence: float):
        self.target = target
        self.prevalence = prevalence
        self.name = "H2_auroc"

    def __call__(self, f1: float, auroc: float = 0.5, **kwargs) -> int:
        bound = max_f1_for_auroc(auroc, self.prevalence)
        return 0 if bound >= self.target else 1


class H3_FeatureCoverage:
    """h₃: If formula uses < k features and best k-feature F1 < target,
    need at least 1 more binary op. Precomputed. O(1) per query."""

    def __init__(self, X: NDArray, y: NDArray, target: float):
        self.target = target
        # Precompute: best single-feature F1
        actual = y.astype(bool)
        self.best_single = max(
            exact_optimal_f1(X[:, i], actual)
            for i in range(X.shape[1])
        )
        self.name = "H3_coverage"

    def __call__(self, f1: float, n_features_used: int = 1, **kwargs) -> int:
        if n_features_used <= 1 and self.best_single < self.target:
            return 1  # single feature can't meet target, need binary op
        return 0


class H4_Lookahead:
    """h₄: Try all 1-step extensions. 0 if any meets target, else 2.
    Expensive but very tight. O(d·k·N log N)."""

    def __init__(self, X: NDArray, y: NDArray, target: float,
                 binary_ops: dict, max_evals: int = 50):
        self.X = X
        self.actual = y.astype(bool)
        self.target = target
        self.binary_ops = binary_ops
        self.max_evals = max_evals
        self.name = "H4_lookahead"

    def __call__(self, f1: float, values: NDArray = None, **kwargs) -> int:
        if values is None or f1 >= self.target:
            return 0

        d = self.X.shape[1]
        evals = 0

        for j in range(d):
            leaf = self.X[:, j]
            for op_fn in self.binary_ops.values():
                try:
                    child = op_fn(values, leaf)
                    child = np.nan_to_num(child, nan=0, posinf=1e10, neginf=-1e10)
                    child_f1 = exact_optimal_f1(child, self.actual)
                    evals += 1
                    if child_f1 >= self.target:
                        return 0  # achievable in 1 step
                    if evals >= self.max_evals:
                        return 1  # budget exceeded, assume 1
                except Exception:
                    pass

        return 2  # no 1-step extension meets target → need ≥ 2


# ============================================================
# The Heuristic DAG
# ============================================================

class HeuristicDAG:
    """Combines multiple admissible heuristics via max().

    Evaluates lazily: cheaper heuristics first, stops early if a
    heuristic returns a high value (no need to compute expensive ones
    if a cheap one already gives a good bound).

    The max of admissible heuristics is admissible (proven).
    """

    def __init__(self, heuristics: list):
        """heuristics: list of callable heuristic objects, ordered
        from cheapest to most expensive."""
        self.heuristics = heuristics
        self.stats = {h.name: {"calls": 0, "total_h": 0} for h in heuristics}

    def __call__(self, **kwargs) -> int:
        """Compute max of all heuristics. Lazy evaluation."""
        h_max = 0

        for h in self.heuristics:
            h_val = h(**kwargs)
            self.stats[h.name]["calls"] += 1
            self.stats[h.name]["total_h"] += h_val
            h_max = max(h_max, h_val)

            # Early termination: if this heuristic gives h=2+,
            # skip more expensive heuristics
            if h_val >= 2:
                break

        return h_max

    def summary(self) -> str:
        lines = ["Heuristic DAG Statistics:"]
        for h in self.heuristics:
            s = self.stats[h.name]
            avg = s["total_h"] / max(s["calls"], 1)
            lines.append(f"  {h.name}: {s['calls']} calls, avg h={avg:.2f}")
        return "\n".join(lines)


# ============================================================
# A* with DAG Heuristic
# ============================================================

@dataclass(order=True)
class AStarNode:
    priority: float
    depth: int = field(compare=False)
    formula: str = field(compare=False)
    values: np.ndarray = field(compare=False, repr=False)
    f1: float = field(compare=False)
    auroc: float = field(compare=False)
    n_features: int = field(compare=False)


def astar_dag(
    X: NDArray,
    y: NDArray,
    feature_names: list[str],
    binary_ops: dict,
    unary_ops: dict,
    f1_target: float,
    max_depth: int = 3,
    max_expansions: int = 50000,
    use_lookahead: bool = False,
    verbose: bool = True,
) -> dict:
    """A* formula search with DAG of admissible heuristics.

    The heuristic h = max(h₁, h₂, h₃, [h₄]) is admissible.
    Combined with g = depth, this gives true A* optimality:
    the first formula meeting the target is guaranteed to be
    at minimum depth.
    """
    N, d = X.shape
    actual = y.astype(bool)
    prevalence = float(actual.mean())

    # Build heuristic DAG
    heuristics = [
        H1_TargetCheck(f1_target),
        H2_AUROCBound(f1_target, prevalence),
        H3_FeatureCoverage(X, y, f1_target),
    ]
    if use_lookahead:
        heuristics.append(H4_Lookahead(X, y, f1_target, binary_ops))

    dag = HeuristicDAG(heuristics)

    # Initialize
    frontier = []
    best_f1 = 0.0
    best_formula = ""
    best_depth = float("inf")
    expansions = 0
    seen = set()

    for i in range(d):
        vals = X[:, i]
        f1 = exact_optimal_f1(vals, actual)
        auc = auroc_safe(vals, actual)
        h = dag(f1=f1, auroc=auc, n_features_used=1, values=vals)
        g = 1

        node = AStarNode(
            priority=g + h,
            depth=1,
            formula=feature_names[i],
            values=vals,
            f1=f1,
            auroc=auc,
            n_features=1,
        )
        heapq.heappush(frontier, node)

        if f1 > best_f1:
            best_f1 = f1
            best_formula = feature_names[i]
            best_depth = 1

    if verbose:
        log.info("DAG-A* init: %d features, best F1=%.4f, target=%.3f",
                 d, best_f1, f1_target)

    # Main loop
    while frontier and expansions < max_expansions:
        node = heapq.heappop(frontier)
        expansions += 1

        # A* termination: if target met and this node's g > best depth,
        # we have found the optimal (shallowest) formula
        if best_f1 >= f1_target and node.depth > best_depth:
            if verbose:
                log.info("DAG-A* OPTIMAL: target met at depth %d after %d expansions",
                         best_depth, expansions)
            break

        if node.depth >= max_depth:
            continue

        if node.formula in seen:
            continue
        seen.add(node.formula)

        # Expand: binary ops with each leaf
        for j in range(d):
            leaf = X[:, j]
            for bname, bfn in binary_ops.items():
                try:
                    child_vals = bfn(node.values, leaf)
                    child_vals = np.nan_to_num(child_vals, nan=0, posinf=1e10, neginf=-1e10)

                    child_desc = f"({node.formula} {bname} {feature_names[j]})"
                    if child_desc in seen:
                        continue

                    child_auc = auroc_safe(child_vals, actual)
                    child_f1 = exact_optimal_f1(child_vals, actual)
                    child_depth = node.depth + 1
                    child_nf = node.n_features + (1 if feature_names[j] not in node.formula else 0)

                    h = dag(f1=child_f1, auroc=child_auc,
                            n_features_used=child_nf, values=child_vals)
                    g = child_depth

                    if child_f1 > best_f1:
                        best_f1 = child_f1
                        best_formula = child_desc
                        best_depth = child_depth
                        if verbose:
                            log.info("  NEW BEST: %s F1=%.4f depth=%d",
                                     child_desc[:50], child_f1, child_depth)

                    heapq.heappush(frontier, AStarNode(
                        priority=g + h,
                        depth=child_depth,
                        formula=child_desc,
                        values=child_vals,
                        f1=child_f1,
                        auroc=child_auc,
                        n_features=child_nf,
                    ))
                except Exception:
                    pass

        # Expand: non-monotone unary ops (monotone ones have same F1,
        # but we still expand them for their DESCENDANTS)
        MONOTONE = {"log", "sqrt", "inv", "neg"}
        for uname, ufn in unary_ops.items():
            try:
                child_vals = ufn(node.values)
                child_vals = np.nan_to_num(child_vals, nan=0, posinf=1e10, neginf=-1e10)

                child_desc = f"{uname}({node.formula})"
                if child_desc in seen:
                    continue

                child_auc = auroc_safe(child_vals, actual)

                if uname in MONOTONE:
                    child_f1 = node.f1  # Theorem 1: same F1
                else:
                    child_f1 = exact_optimal_f1(child_vals, actual)

                child_depth = node.depth + 1

                h = dag(f1=child_f1, auroc=child_auc,
                        n_features_used=node.n_features, values=child_vals)
                g = child_depth

                if child_f1 > best_f1:
                    best_f1 = child_f1
                    best_formula = child_desc
                    best_depth = child_depth

                heapq.heappush(frontier, AStarNode(
                    priority=g + h,
                    depth=child_depth,
                    formula=child_desc,
                    values=child_vals,
                    f1=child_f1,
                    auroc=child_auc,
                    n_features=node.n_features,
                ))
            except Exception:
                pass

        if expansions % 2000 == 0 and verbose:
            log.info("  [%d exp] best=%.4f depth=%d frontier=%d",
                     expansions, best_f1, best_depth, len(frontier))

    elapsed_heuristic = dag.summary()
    if verbose:
        log.info(elapsed_heuristic)

    return {
        "best_f1": best_f1,
        "best_formula": best_formula,
        "best_depth": best_depth,
        "expansions": expansions,
        "frontier_remaining": len(frontier),
        "target_met": best_f1 >= f1_target,
        "heuristic_stats": dag.stats,
    }
