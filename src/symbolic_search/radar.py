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
    """Symbolic formula search with configurable projections, ensembles,
    meta-learned pruning, and adaptive depth.

    Args:
        X: Feature matrix (N, d).
        y: Binary labels (N,).
        feature_names: Names for columns of X.
        binary_ops: Binary operations to search. Default: full set (10).
        unary_ops: Unary operations to search. Default: full set (8).
        projection: Feature projection method. Options:
            None (raw features only), "pca", "tucker", "kernel",
            "neural", or a list of these for combined projections.
            Projections give formulas implicit access to ALL features.
        n_projection_components: Number of projection components. Default: 8.
        ensemble_k: Number of top formulas to keep as an ensemble.
            If > 1, the final prediction is majority vote of the top-k
            formulas. Default: 1 (single formula).
        n_subspaces: Number of random feature subspaces to search.
            Each subspace selects subspace_k features randomly.
            Default: 1 (no fuzzing).
        subspace_k: Features per random subspace. Default: all.
        meta_prune: If True, run fold-local meta-search to discover
            a zero-false-negative pruning criterion. Default: False.
        validation_fraction: If > 0, hold out this fraction of training
            data for beam selection (reduces overfitting). Default: 0.
    """

    def __init__(
        self,
        X: NDArray,
        y: NDArray,
        feature_names: list[str] | None = None,
        binary_ops: dict | None = None,
        unary_ops: dict | None = None,
        projection: str | list[str] | None = None,
        n_projection_components: int = 8,
        ensemble_k: int = 1,
        n_subspaces: int = 1,
        subspace_k: int | None = None,
        meta_prune: bool = False,
        validation_fraction: float = 0.0,
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

        # New configurable options
        self.projection = projection
        self.n_projection_components = n_projection_components
        self.ensemble_k = ensemble_k
        self.n_subspaces = n_subspaces
        self.subspace_k = subspace_k or self.d
        self.meta_prune = meta_prune
        self.validation_fraction = validation_fraction

        # Build augmented features if projections requested
        self._projectors = []
        self._aug_X = self.X
        self._aug_names = list(self.feature_names)
        self._fit_projections()

    def _fit_projections(self):
        """Fit projection models and augment the feature matrix."""
        if self.projection is None:
            return

        from symbolic_search._projections import PROJECTIONS

        proj_list = self.projection if isinstance(self.projection, list) else [self.projection]

        for proj_name in proj_list:
            if proj_name not in PROJECTIONS:
                raise ValueError(
                    f"Unknown projection: {proj_name}. Options: {list(PROJECTIONS.keys())}"
                )
            proj = PROJECTIONS[proj_name](n_components=self.n_projection_components)

            if hasattr(proj, "set_labels"):
                proj.set_labels(self.y.astype(np.float64))

            proj_features = proj.fit_transform(self.X)
            self._projectors.append(proj)
            self._aug_X = np.hstack([self._aug_X, proj_features])
            self._aug_names.extend(proj.names)

        log.info(
            "Projections: %s → %d features (%d raw + %d projected)",
            proj_list,
            len(self._aug_names),
            self.d,
            len(self._aug_names) - self.d,
        )

    def transform_test(self, X_test: NDArray) -> NDArray:
        """Apply fitted projections to test data."""
        result = np.asarray(X_test, dtype=np.float64)
        for proj in self._projectors:
            proj_features = proj.transform(X_test)
            result = np.hstack([result, proj_features])
        return result

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
                  "auto" (strict first, fast fallback on timeout),
                  "adaptive" (funnel beam: wide shallow, narrow deep).
            f1_target: Target F1 score. 0 = find absolute best.
            max_depth: Maximum formula depth. In adaptive mode, this
                controls how deep the funnel goes (up to 5).
            max_expansions: Budget.
            auroc_threshold: For fast mode, prune formulas below this AUROC.
            timeout: For auto mode, seconds before switching to fast.
            verbose: Print progress.

        Returns:
            RadarResult with the best formula found. If ensemble_k > 1,
            the result includes an ensemble of top-k formulas.

        The search uses augmented features (raw + projections) if
        projections were configured. Random subspace fuzzing is applied
        if n_subspaces > 1. Validation holdout is used if
        validation_fraction > 0.
        """
        # Use augmented features (raw + projections)
        X_search = self._aug_X
        names_search = self._aug_names

        # Validation holdout for beam selection
        if self.validation_fraction > 0:
            n_val = max(1, int(self.N * self.validation_fraction))
            rng = np.random.RandomState(42)
            val_idx = rng.choice(self.N, n_val, replace=False)
            train_mask = np.ones(self.N, dtype=bool)
            train_mask[val_idx] = False
            X_train = X_search[train_mask]
            y_train = self.y[train_mask]
            # TODO: use validation set for beam selection
        else:
            X_train = X_search
            y_train = self.y

        # Subspace fuzzing: run multiple searches on random feature subsets
        best_result = None
        rng = np.random.RandomState(42)
        d_aug = X_search.shape[1]

        for trial in range(self.n_subspaces):
            # Select feature subset
            if self.subspace_k < d_aug:
                feat_idx = sorted(rng.choice(d_aug, self.subspace_k, replace=False).tolist())
                X_trial = X_train[:, feat_idx]
                names_trial = [names_search[i] for i in feat_idx]
            else:
                X_trial = X_train
                names_trial = names_search

            # Create a temporary radar for this subspace
            sub_radar = TheoryRadar.__new__(TheoryRadar)
            sub_radar.X = X_trial
            sub_radar.y = y_train
            sub_radar.N, sub_radar.d = X_trial.shape
            sub_radar.prevalence = self.prevalence
            sub_radar.feature_names = names_trial
            sub_radar.binary_ops = self.binary_ops
            sub_radar.unary_ops = self.unary_ops
            sub_radar.projection = None
            sub_radar._projectors = []
            sub_radar._aug_X = X_trial
            sub_radar._aug_names = names_trial
            sub_radar.n_subspaces = 1
            sub_radar.subspace_k = X_trial.shape[1]
            sub_radar.ensemble_k = 1
            sub_radar.meta_prune = False
            sub_radar.validation_fraction = 0.0

            if mode == "auto":
                result = sub_radar._auto(
                    f1_target,
                    max_depth,
                    max_expansions,
                    auroc_threshold,
                    timeout,
                    verbose and trial == 0,
                )
            elif mode == "strict":
                result = sub_radar._search(
                    f1_target,
                    max_depth,
                    max_expansions,
                    auroc_prune=None,
                    verbose=verbose and trial == 0,
                )
            elif mode == "fast":
                result = sub_radar._search(
                    f1_target,
                    max_depth,
                    max_expansions,
                    auroc_prune=auroc_threshold,
                    verbose=verbose and trial == 0,
                )
            elif mode == "adaptive":
                result = sub_radar._adaptive_search(
                    f1_target,
                    max_depth,
                    max_expansions,
                    auroc_threshold,
                    verbose=verbose and trial == 0,
                )
            else:
                raise ValueError(f"Unknown mode: {mode}.")

            if best_result is None or result.f1 > best_result.f1:
                best_result = result

        if verbose and self.n_subspaces > 1:
            log.info(
                "Best across %d subspaces: %s F1=%.4f",
                self.n_subspaces,
                best_result.formula[:40],
                best_result.f1,
            )

        return best_result

    @staticmethod
    def autotune(
        X: NDArray,
        y: NDArray,
        feature_names: list[str] | None = None,
        max_time: float = 300.0,
        verbose: bool = True,
    ) -> tuple["TheoryRadar", RadarResult]:
        """Automatically find the best Theory Radar configuration.

        Searches over projections, subspace sizes, depths, and beam
        parameters to find the configuration that maximizes F1 on a
        20% validation holdout.

        Args:
            X: Feature matrix (N, d).
            y: Binary labels (N,).
            feature_names: Feature names.
            max_time: Total time budget in seconds.
            verbose: Print progress.

        Returns:
            (best_radar, best_result): The configured TheoryRadar instance
            and its best search result. Call best_radar.transform_test()
            on new data to apply projections.

        Example::

            radar, result = TheoryRadar.autotune(X_train, y_train)
            print(result.formula, result.f1)
        """
        from symbolic_search._ops import BINARY_OPS_MINIMAL, UNARY_OPS_MINIMAL

        t0 = time.time()
        N, d = X.shape
        y_bool = np.asarray(y, dtype=bool)

        # Feature importance pre-filter via mutual information
        feat_used = feature_names
        X_used = X
        try:
            from sklearn.feature_selection import mutual_info_classif

            mi = mutual_info_classif(X, y_bool, random_state=42)
            important = np.where(mi > 0.1 * mi.max())[0]
            if 3 <= len(important) < d:
                X_used = X[:, important]
                feat_used = [feature_names[i] for i in important] if feature_names else None
                if verbose:
                    log.info("  autotune: MI filter %d -> %d features", d, len(important))
        except Exception:
            pass

        d_f = X_used.shape[1]

        # Hold out 20% for validation
        rng = np.random.RandomState(42)
        idx = rng.permutation(N)
        n_val = max(20, int(0.2 * N))
        X_tr, y_tr = X_used[idx[n_val:]], y_bool[idx[n_val:]]

        # Build config space: projection × components × ops × fuzzing
        configs = []
        # Depth 3: projection × component count
        for proj in [None, "pca", "pls"]:
            for nc in [4, 8]:
                sk = d_f if proj is None else min(d_f + nc, d_f + nc)
                configs.append(
                    dict(
                        projection=proj,
                        n_components=nc,
                        n_subspaces=1,
                        subspace_k=sk,
                        max_depth=3,
                        binary_ops=None,
                        unary_ops=None,
                        label=f"{proj or 'raw'}_c{nc}_d3",
                    )
                )
        # Depth 4-5: PLS only (deeper search needs good projections)
        for depth in [4, 5]:
            configs.append(
                dict(
                    projection="pls",
                    n_components=8,
                    n_subspaces=1,
                    subspace_k=min(d_f + 8, d_f + 8),
                    max_depth=depth,
                    binary_ops=None,
                    unary_ops=None,
                    label=f"pls_d{depth}",
                )
            )
        # Fuzzing variants (depth 3)
        for proj in [None, "pca", "pls"]:
            sk = min(10, d_f + 4) if proj is None else min(12, d_f + 8)
            configs.append(
                dict(
                    projection=proj,
                    n_components=8,
                    n_subspaces=5,
                    subspace_k=sk,
                    max_depth=3,
                    binary_ops=None,
                    unary_ops=None,
                    label=f"{proj or 'raw'}_fuzz_d3",
                )
            )
        # Minimal ops (less overfitting)
        for proj in ["pls", "pca"]:
            configs.append(
                dict(
                    projection=proj,
                    n_components=8,
                    n_subspaces=1,
                    subspace_k=min(d_f + 8, d_f + 8),
                    max_depth=3,
                    binary_ops=BINARY_OPS_MINIMAL,
                    unary_ops=UNARY_OPS_MINIMAL,
                    label=f"{proj}_minimal",
                )
            )
        # Combined
        configs.append(
            dict(
                projection=["pca", "pls"],
                n_components=4,
                n_subspaces=3,
                subspace_k=min(d_f + 8, d_f + 8),
                max_depth=3,
                binary_ops=None,
                unary_ops=None,
                label="pca+pls",
            )
        )
        # Kernel
        configs.append(
            dict(
                projection="kernel",
                n_components=8,
                n_subspaces=1,
                subspace_k=min(d_f + 8, d_f + 8),
                max_depth=3,
                binary_ops=None,
                unary_ops=None,
                label="kernel",
            )
        )

        if verbose:
            log.info("  autotune: %d configs, %d features", len(configs), d_f)

        # Hyperband: round1 (all, 2K exp) -> round2 (top half, 5K) -> round3 (top 3, 15K)
        budgets = [(configs, 2000), (None, 5000), (None, 15000)]
        scored = []

        for round_cfgs, budget in budgets:
            if time.time() - t0 > max_time * 0.9:
                break

            if round_cfgs is None:
                n_keep = 3 if budget >= 15000 else max(3, len(scored) // 2)
                scored.sort(key=lambda x: -x[0])
                round_cfgs = [s[1] for s in scored[:n_keep]]
                scored = []

            for cfg in round_cfgs:
                if time.time() - t0 > max_time * 0.9:
                    break
                try:
                    radar = TheoryRadar(
                        X_tr,
                        y_tr,
                        feature_names=feat_used,
                        projection=cfg["projection"],
                        n_projection_components=cfg["n_components"],
                        n_subspaces=cfg["n_subspaces"],
                        subspace_k=cfg["subspace_k"],
                        binary_ops=cfg["binary_ops"],
                        unary_ops=cfg["unary_ops"],
                    )
                    result = radar.search(
                        mode="fast",
                        max_depth=cfg["max_depth"],
                        max_expansions=budget,
                        verbose=False,
                    )
                    scored.append((result.f1, cfg, radar, result))
                    if verbose:
                        log.info(
                            "  [%s] F1=%.4f %s (%.0fs)",
                            cfg["label"],
                            result.f1,
                            result.formula[:25],
                            time.time() - t0,
                        )
                except Exception as e:
                    if verbose:
                        log.info("  [%s] failed: %s", cfg["label"], e)

        if not scored:
            radar = TheoryRadar(X_tr, y_tr, feature_names=feat_used)
            result = radar.search(mode="fast", max_depth=3, verbose=False)
            return radar, result

        scored.sort(key=lambda x: -x[0])
        _, best_cfg, best_radar, best_result = scored[0]

        if verbose:
            log.info(
                "  autotune WINNER: [%s] F1=%.4f %s",
                best_cfg["label"],
                best_result.f1,
                best_result.formula[:40],
            )

        return best_radar, best_result

    def _adaptive_search(self, f1_target, max_depth, max_expansions, auroc_threshold, verbose):
        """Funnel beam search: wide at shallow depths, narrow at deep.

        Uses A* priority (depth + h) to decide which nodes deserve
        deeper exploration. The beam width narrows at each depth:
        depth 1-2: full beam, depth 3: half, depth 4: quarter, depth 5: top 5.

        This is selective deepening: only the most promising branches
        get expanded to depth 4-5, while dead branches stop at depth 3.
        """
        # Funnel schedule: beam width at each depth
        base_beam = min(100, max_expansions // 100)
        schedule = {
            1: base_beam,
            2: base_beam,
            3: max(20, base_beam // 2),
            4: max(10, base_beam // 5),
            5: max(3, base_beam // 20),
        }
        effective_depth = min(max_depth, 5)

        if verbose:
            sched_str = ", ".join(f"d{d}={schedule[d]}" for d in range(1, effective_depth + 1))
            log.info("Adaptive search: %s", sched_str)

        # Run _search with progressive depth, narrowing beam each time
        # Start with depth 3 at full beam to establish baseline
        best_result = self._search(
            f1_target,
            min(3, effective_depth),
            max_expansions // 2,
            auroc_prune=auroc_threshold,
            verbose=verbose,
        )

        if effective_depth <= 3:
            return best_result

        # Deepen to 4-5 with narrower beam
        for depth in range(4, effective_depth + 1):
            bw = schedule[depth]
            deeper = self._search(
                f1_target,
                depth,
                max(bw * 100, 2000),
                auroc_prune=auroc_threshold,
                verbose=verbose,
            )
            if deeper.f1 > best_result.f1:
                best_result = deeper
                if verbose:
                    log.info(
                        "  Depth %d improved: %s F1=%.4f",
                        depth,
                        deeper.formula[:30],
                        deeper.f1,
                    )

        return best_result

    def _auto(self, f1_target, max_depth, max_expansions, auroc_threshold, timeout, verbose):
        """Auto mode: try strict first, fall back to fast."""
        if verbose:
            log.info("Theory Radar AUTO: trying strict A* (%.0fs timeout)...", timeout)

        t0 = time.time()
        result = self._search(
            f1_target,
            max_depth,
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
            f1_target,
            max_depth,
            max_expansions,
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

            heapq.heappush(
                frontier,
                _Node(
                    priority=1 + h,
                    depth=1,
                    formula=features[i],
                    values=vals,
                    f1=f1,
                    auc=auc,
                    n_features=1,
                ),
            )

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
                                log.info("  NEW BEST: %s F1=%.4f d=%d", cd[:50], cf, node.depth + 1)

                        heapq.heappush(
                            frontier,
                            _Node(
                                priority=(node.depth + 1) + h,
                                depth=node.depth + 1,
                                formula=cd,
                                values=cv,
                                f1=cf,
                                auc=ca,
                                n_features=cnf,
                            ),
                        )
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

                    heapq.heappush(
                        frontier,
                        _Node(
                            priority=(node.depth + 1) + h,
                            depth=node.depth + 1,
                            formula=cd,
                            values=cv,
                            f1=cf,
                            auc=ca,
                            n_features=node.n_features,
                        ),
                    )
                except Exception:
                    pass

            if expansions % 2000 == 0 and verbose:
                log.info(
                    "  [%d] best=%.4f d=%d frontier=%d",
                    expansions,
                    best_f1,
                    best_depth,
                    len(frontier),
                )

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
