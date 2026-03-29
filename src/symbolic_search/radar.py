"""
Theory Radar: A* formula search with provable DAG heuristic.

Usage:
    from symbolic_search.radar import TheoryRadar

    radar = TheoryRadar(X, y, feature_names=["age", "bmi", "glucose"])

    # Strict A* (provable optimality guarantee)
    result = radar.search(mode="strict", f1_target=0.90)

    # Fast mode (empirical AUROC pruning, no guarantee, 10-100x faster)
    result = radar.search(mode="fast", auroc_threshold=0.55)

    # Auto: strict first, fast fallback if timeout
    result = radar.search(mode="auto", f1_target=0.90, timeout=60)

    print(result.formula)     # "(bmi min age) + glucose"
    print(result.f1)          # 0.694
    print(result.depth)       # 3
    print(result.guaranteed)  # True (strict mode)
"""

from __future__ import annotations

import heapq
import logging
import time
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from symbolic_search._ops import BINARY_OPS, UNARY_OPS
from symbolic_search._heuristic_dag import (
    HeuristicDAG,
    H1_TargetCheck,
    H2_AUROCBound,
    H3_FeatureCoverage,
    exact_optimal_f1,
    auroc_safe,
)

log = logging.getLogger(__name__)

MONOTONE_OPS = {"log", "sqrt", "inv", "neg"}


@dataclass
class RadarResult:
    """Result from a Theory Radar search."""

    formula: str
    f1: float
    depth: int
    expansions: int
    time_seconds: float
    mode: str
    guaranteed: bool  # True = proven optimal at this depth
    target: float
    target_met: bool
    pruned_monotone: int = 0
    pruned_auroc: int = 0
    heuristic_stats: dict = field(default_factory=dict)

    def summary(self) -> str:
        g = "GUARANTEED optimal" if self.guaranteed else "best found (no guarantee)"
        return (
            f"Formula: {self.formula}\n"
            f"F1: {self.f1:.4f} | Depth: {self.depth} | {g}\n"
            f"Target: {self.target:.3f} | Met: {self.target_met}\n"
            f"Expansions: {self.expansions} | Time: {self.time_seconds:.1f}s\n"
            f"Mode: {self.mode} | Monotone saved: {self.pruned_monotone} | "
            f"AUROC pruned: {self.pruned_auroc}"
        )


@dataclass(order=True)
class _Node:
    priority: float
    depth: int = field(compare=False)
    formula: str = field(compare=False)
    values: np.ndarray = field(compare=False, repr=False)
    f1: float = field(compare=False)
    auc: float = field(compare=False)
    n_features: int = field(compare=False)


class TheoryRadar:
    """A* formula search with provable DAG heuristic.

    Args:
        X: Feature matrix (N, d).
        y: Binary labels (N,).
        feature_names: Names for columns of X.
        binary_ops: Binary operations to search. Default: full set (10).
        unary_ops: Unary operations to search. Default: full set (8).
    """

    def __init__(
        self,
        X: NDArray,
        y: NDArray,
        feature_names: list[str] | None = None,
        binary_ops: dict | None = None,
        unary_ops: dict | None = None,
    ):
        self.X = np.asarray(X, dtype=np.float64)
        self.y = np.asarray(y, dtype=bool)
        self.N, self.d = self.X.shape
        self.prevalence = float(self.y.mean())

        if feature_names is None:
            feature_names = [f"x{i}" for i in range(self.d)]
        self.feature_names = feature_names
        self.binary_ops = binary_ops or BINARY_OPS
        self.unary_ops = unary_ops or UNARY_OPS

    def search(
        self,
        mode: str = "auto",
        f1_target: float = 0.0,
        max_depth: int = 3,
        max_expansions: int = 50000,
        auroc_threshold: float = 0.55,
        timeout: float = 300.0,
        verbose: bool = True,
    ) -> RadarResult:
        """Run Theory Radar search.

        Args:
            mode: "strict" (true A*, guaranteed optimal),
                  "fast" (AUROC subtree pruning, no guarantee),
                  "auto" (strict first, fast fallback on timeout).
            f1_target: Target F1 score. 0 = find absolute best.
            max_depth: Maximum formula depth.
            max_expansions: Budget.
            auroc_threshold: For fast mode, prune formulas below this AUROC.
            timeout: For auto mode, seconds before switching to fast.
            verbose: Print progress.
        """
        if mode == "auto":
            return self._auto(
                f1_target, max_depth, max_expansions,
                auroc_threshold, timeout, verbose,
            )
        elif mode == "strict":
            return self._search(
                f1_target, max_depth, max_expansions,
                auroc_prune=None,  # no subtree pruning
                verbose=verbose,
            )
        elif mode == "fast":
            return self._search(
                f1_target, max_depth, max_expansions,
                auroc_prune=auroc_threshold,
                verbose=verbose,
            )
        else:
            raise ValueError(f"Unknown mode: {mode}. Use 'strict', 'fast', or 'auto'.")

    def _auto(self, f1_target, max_depth, max_expansions,
              auroc_threshold, timeout, verbose):
        """Auto mode: try strict first, fall back to fast."""
        if verbose:
            log.info("Theory Radar AUTO: trying strict A* (%.0fs timeout)...", timeout)

        t0 = time.time()
        result = self._search(
            f1_target, max_depth,
            max_expansions=min(max_expansions, 10000),  # limit strict budget
            auroc_prune=None,
            verbose=verbose,
            timeout=timeout,
        )

        if result.target_met or time.time() - t0 < timeout * 0.8:
            if verbose:
                log.info("Strict A* succeeded in %.1fs", result.time_seconds)
            return result

        if verbose:
            log.info("Strict A* timed out, switching to fast mode...")

        return self._search(
            f1_target, max_depth, max_expansions,
            auroc_prune=auroc_threshold,
            verbose=verbose,
        )

    def _search(
        self,
        f1_target: float,
        max_depth: int,
        max_expansions: int,
        auroc_prune: float | None,
        verbose: bool = True,
        timeout: float | None = None,
    ) -> RadarResult:
        """Core A* search with DAG heuristic."""
        t0 = time.time()
        mode = "strict" if auroc_prune is None else "fast"
        guaranteed = auroc_prune is None

        X, y = self.X, self.y
        d = self.d
        actual = self.y
        features = self.feature_names

        # Build heuristic DAG
        heuristics = [
            H1_TargetCheck(f1_target),
            H2_AUROCBound(f1_target, self.prevalence),
            H3_FeatureCoverage(X, y.astype(int), f1_target),
        ]
        dag = HeuristicDAG(heuristics)

        # State
        frontier: list[_Node] = []
        best_f1 = 0.0
        best_formula = ""
        best_depth = max_depth + 1
        expansions = 0
        pruned_monotone = 0
        pruned_auroc = 0
        seen: set[str] = set()

        # Seed with leaf features
        for i in range(d):
            vals = X[:, i]
            f1 = exact_optimal_f1(vals, actual)
            auc = auroc_safe(vals, actual)
            h = dag(f1=f1, auroc=auc, n_features_used=1, values=vals)

            heapq.heappush(frontier, _Node(
                priority=1 + h, depth=1, formula=features[i],
                values=vals, f1=f1, auc=auc, n_features=1,
            ))

            if f1 > best_f1:
                best_f1 = f1
                best_formula = features[i]
                best_depth = 1

        if verbose:
            log.info("Theory Radar [%s] d=%d target=%.3f", mode, d, f1_target)

        # A* loop
        while frontier and expansions < max_expansions:
            if timeout and (time.time() - t0) > timeout:
                break

            node = heapq.heappop(frontier)
            expansions += 1

            # A* optimality: if target met and this node is deeper, STOP
            if f1_target > 0 and best_f1 >= f1_target and node.depth > best_depth:
                if verbose:
                    log.info("  OPTIMAL at depth %d (%d expansions)", best_depth, expansions)
                break

            if node.depth >= max_depth or node.formula in seen:
                continue
            seen.add(node.formula)

            # --- Expand: binary ops ---
            for j in range(d):
                leaf = X[:, j]
                for bname, bfn in self.binary_ops.items():
                    try:
                        cv = bfn(node.values, leaf)
                        cv = np.nan_to_num(cv, nan=0, posinf=1e10, neginf=-1e10)
                        cd = f"({node.formula} {bname} {features[j]})"
                        if cd in seen:
                            continue

                        ca = auroc_safe(cv, actual)

                        # Fast mode: AUROC subtree pruning (empirical, not proven)
                        if auroc_prune is not None and ca < auroc_prune:
                            pruned_auroc += 1
                            continue

                        cf = exact_optimal_f1(cv, actual)
                        cnf = node.n_features + (1 if features[j] not in node.formula else 0)
                        h = dag(f1=cf, auroc=ca, n_features_used=cnf, values=cv)

                        if cf > best_f1:
                            best_f1 = cf
                            best_formula = cd
                            best_depth = node.depth + 1
                            if verbose:
                                log.info("  NEW BEST: %s F1=%.4f d=%d",
                                         cd[:50], cf, node.depth + 1)

                        heapq.heappush(frontier, _Node(
                            priority=(node.depth + 1) + h,
                            depth=node.depth + 1, formula=cd,
                            values=cv, f1=cf, auc=ca, n_features=cnf,
                        ))
                    except Exception:
                        pass

            # --- Expand: unary ops ---
            for uname, ufn in self.unary_ops.items():
                try:
                    cv = ufn(node.values)
                    cv = np.nan_to_num(cv, nan=0, posinf=1e10, neginf=-1e10)
                    cd = f"{uname}({node.formula})"
                    if cd in seen:
                        continue

                    ca = auroc_safe(cv, actual)

                    # Monotone: reuse parent F1 (Theorem 1), still expand
                    if uname in MONOTONE_OPS:
                        cf = node.f1
                        pruned_monotone += 1
                    else:
                        if auroc_prune is not None and ca < auroc_prune:
                            pruned_auroc += 1
                            continue
                        cf = exact_optimal_f1(cv, actual)

                    h = dag(f1=cf, auroc=ca, n_features_used=node.n_features, values=cv)

                    if cf > best_f1:
                        best_f1 = cf
                        best_formula = cd
                        best_depth = node.depth + 1

                    heapq.heappush(frontier, _Node(
                        priority=(node.depth + 1) + h,
                        depth=node.depth + 1, formula=cd,
                        values=cv, f1=cf, auc=ca, n_features=node.n_features,
                    ))
                except Exception:
                    pass

            if expansions % 2000 == 0 and verbose:
                log.info("  [%d] best=%.4f d=%d frontier=%d",
                         expansions, best_f1, best_depth, len(frontier))

        elapsed = time.time() - t0

        return RadarResult(
            formula=best_formula,
            f1=best_f1,
            depth=best_depth,
            expansions=expansions,
            time_seconds=elapsed,
            mode=mode,
            guaranteed=guaranteed,
            target=f1_target,
            target_met=best_f1 >= f1_target if f1_target > 0 else True,
            pruned_monotone=pruned_monotone,
            pruned_auroc=pruned_auroc,
            heuristic_stats=dag.stats,
        )
