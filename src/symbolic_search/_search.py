"""Core phased exhaustive symbolic search engine."""

from __future__ import annotations

import time
import logging
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from symbolic_search._ops import BINARY_OPS, UNARY_OPS

log = logging.getLogger(__name__)


@dataclass
class FormulaResult:
    """A single formula evaluation result."""

    formula: str
    score: float
    phase: int
    threshold: float = 0.0
    direction: str = ">"


@dataclass
class SearchResults:
    """Complete results from a symbolic search run."""

    best_formula: str
    ceiling: float
    ensemble_score: float
    gap: float
    convergence: dict[int, float]
    all_formulas: list[FormulaResult]
    formulas_tested: int
    elapsed_seconds: float
    task: str = "classification"

    def summary(self) -> str:
        lines = [
            f"Best formula: {self.best_formula}",
            f"Ceiling (depth ≤ 3): {self.ceiling:.3f}",
            f"Ensemble F1: {self.ensemble_score:.3f}",
            f"Gap: {self.gap:+.3f}",
            f"Convergence: {self.convergence}",
            f"Formulas tested: {self.formulas_tested}",
            f"Time: {self.elapsed_seconds:.1f}s",
        ]
        return "\n".join(lines)


def _f1_threshold_sweep(
    values: NDArray,
    actual: NDArray,
    percentiles: list[int] | None = None,
) -> tuple[float, float, str]:
    """Find best F1, threshold, and direction via percentile sweep.

    Returns (best_f1, best_threshold, direction).
    """
    if percentiles is None:
        percentiles = [20, 30, 40, 50, 60, 70, 75, 80, 85, 90, 95]

    values = np.nan_to_num(values, nan=0, posinf=1e10, neginf=-1e10)
    finite = np.isfinite(values)
    if finite.sum() < len(values) * 0.5:
        return 0.0, 0.0, ">"

    pcts = np.percentile(values[finite], percentiles)
    n_actual = int(actual.sum())

    best_f1 = 0.0
    best_thresh = 0.0
    best_dir = ">"

    for thresh in pcts:
        for direction, pred in [
            (">", values > thresh),
            ("<", values < thresh),
        ]:
            tp = int((pred & actual).sum())
            fp = int((pred & ~actual).sum())
            fn = n_actual - tp
            if (tp + fp) > 0 and (tp + fn) > 0:
                prec = tp / (tp + fp)
                rec = tp / (tp + fn)
                if (prec + rec) > 0:
                    f1 = 2 * prec * rec / (prec + rec)
                    if f1 > best_f1:
                        best_f1 = f1
                        best_thresh = thresh
                        best_dir = direction

    return best_f1, best_thresh, best_dir


class SymbolicSearch:
    """Phased exhaustive symbolic formula search with ceiling detection.

    Args:
        X: Feature matrix, shape (N, d).
        y: Binary labels, shape (N,). Values 0 or 1.
        feature_names: Names for each feature column.
        binary_ops: Dict of binary operations {name: callable(a, b)}.
            Defaults to the full built-in set (10 operations).
        unary_ops: Dict of unary operations {name: callable(x)}.
            Defaults to the full built-in set (8 operations).
        top_k_pairwise: Number of top pairwise results to carry into Phase 3.
        top_k_single: Number of top single features for Phase 4 triples.
        triple_patterns: Number of combination patterns for triples.
        verbose: Print progress.
    """

    def __init__(
        self,
        X: NDArray,
        y: NDArray,
        feature_names: list[str] | None = None,
        binary_ops: dict | None = None,
        unary_ops: dict | None = None,
        top_k_pairwise: int = 30,
        top_k_single: int = 8,
        verbose: bool = True,
    ):
        self.X = np.asarray(X, dtype=np.float64)
        self.y = np.asarray(y, dtype=bool)
        self.N, self.d = self.X.shape

        if feature_names is None:
            feature_names = [f"x{i}" for i in range(self.d)]
        self.feature_names = feature_names

        self.binary_ops = binary_ops or BINARY_OPS
        self.unary_ops = unary_ops or UNARY_OPS
        self.top_k_pairwise = top_k_pairwise
        self.top_k_single = top_k_single
        self.verbose = verbose

    def run(self, ensemble: bool = True) -> SearchResults:
        """Run the full phased search.

        Args:
            ensemble: If True, also train a gradient-boosted ensemble
                for ceiling gap computation.

        Returns:
            SearchResults with best formula, ceiling, gap, and convergence.
        """
        t0 = time.time()
        all_results: list[FormulaResult] = []
        convergence = {}
        formulas_tested = 0

        # Phase 1: Single features
        if self.verbose:
            log.info("Phase 1: %d single features...", self.d)
        for i in range(self.d):
            f1, thresh, direction = _f1_threshold_sweep(self.X[:, i], self.y)
            all_results.append(
                FormulaResult(
                    self.feature_names[i], f1, phase=1, threshold=thresh, direction=direction
                )
            )
            formulas_tested += 1

        all_results.sort(key=lambda r: -r.score)
        convergence[1] = all_results[0].score

        # Phase 2: Pairwise combinations
        n_pairs = self.d * (self.d - 1) * len(self.binary_ops)
        if self.verbose:
            log.info("Phase 2: %d pairwise formulas...", n_pairs)

        for i in range(self.d):
            for j in range(self.d):
                if i == j:
                    continue
                a, b = self.X[:, i], self.X[:, j]
                for op_name, op_fn in self.binary_ops.items():
                    try:
                        vals = op_fn(a, b)
                        f1, thresh, direction = _f1_threshold_sweep(vals, self.y)
                        desc = f"{self.feature_names[i]} {op_name} {self.feature_names[j]}"
                        all_results.append(
                            FormulaResult(desc, f1, phase=2, threshold=thresh, direction=direction)
                        )
                    except Exception:
                        pass
                    formulas_tested += 1

        all_results.sort(key=lambda r: -r.score)
        convergence[2] = all_results[0].score

        # Phase 3: Unary transforms of top-K pairwise
        top_pairwise = [r for r in all_results if r.phase == 2][: self.top_k_pairwise]
        n_phase3 = len(top_pairwise) * len(self.unary_ops)
        if self.verbose:
            log.info(
                "Phase 3: %d unary transforms of top-%d pairwise...", n_phase3, self.top_k_pairwise
            )

        for r in top_pairwise:
            # Parse the formula to recompute values
            parts = r.formula.split(" ")
            if len(parts) != 3:
                continue
            fi, op, fj = parts
            try:
                i = self.feature_names.index(fi)
                j = self.feature_names.index(fj)
                base = self.binary_ops[op](self.X[:, i], self.X[:, j])
            except (ValueError, KeyError):
                continue

            for uname, ufn in self.unary_ops.items():
                try:
                    vals = ufn(base)
                    vals = np.nan_to_num(vals, nan=0, posinf=1e10, neginf=-1e10)
                    f1, thresh, direction = _f1_threshold_sweep(vals, self.y)
                    all_results.append(
                        FormulaResult(
                            f"{uname}({r.formula})",
                            f1,
                            phase=3,
                            threshold=thresh,
                            direction=direction,
                        )
                    )
                except Exception:
                    pass
                formulas_tested += 1

        all_results.sort(key=lambda r: -r.score)
        convergence[3] = all_results[0].score

        # Phase 4: Triple combinations
        top_singles = [r.formula for r in all_results if r.phase == 1][: self.top_k_single]
        if len(top_singles) >= 3:
            if self.verbose:
                log.info("Phase 4: Triple combinations of top-%d features...", len(top_singles))
            for ii, fa in enumerate(top_singles):
                ia = self.feature_names.index(fa)
                for jj, fb in enumerate(top_singles):
                    if jj <= ii:
                        continue
                    ib = self.feature_names.index(fb)
                    for kk, fc in enumerate(top_singles):
                        if kk <= jj:
                            continue
                        ic = self.feature_names.index(fc)
                        a, b, c = self.X[:, ia], self.X[:, ib], self.X[:, ic]
                        combos = [
                            (f"{fa}*{fb}+{fc}", a * b + c),
                            (f"{fa}+{fb}*{fc}", a + b * c),
                            (f"{fa}*{fb}*{fc}", a * b * c),
                            (f"{fa}*{fb}/{fc}", a * b / (c + 1e-30)),
                        ]
                        for desc, vals in combos:
                            try:
                                vals = np.nan_to_num(vals, nan=0, posinf=1e10, neginf=-1e10)
                                f1, thresh, direction = _f1_threshold_sweep(vals, self.y)
                                all_results.append(
                                    FormulaResult(
                                        desc, f1, phase=4, threshold=thresh, direction=direction
                                    )
                                )
                            except Exception:
                                pass
                            formulas_tested += 1

        all_results.sort(key=lambda r: -r.score)
        ceiling = all_results[0].score
        best_formula = all_results[0].formula

        # Ensemble comparison
        ensemble_score = 0.0
        if ensemble:
            try:
                from sklearn.ensemble import GradientBoostingClassifier
                from sklearn.model_selection import cross_val_score, StratifiedKFold

                cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
                gb = GradientBoostingClassifier(n_estimators=100, max_depth=4, random_state=42)
                scores = cross_val_score(gb, self.X, self.y.astype(int), cv=cv, scoring="f1")
                ensemble_score = float(scores.mean())
            except ImportError:
                log.warning("scikit-learn not installed; skipping ensemble comparison")

        gap = ensemble_score - ceiling
        elapsed = time.time() - t0

        if self.verbose:
            log.info("Best formula: %s (F1=%.3f)", best_formula, ceiling)
            log.info("Ensemble F1: %.3f, Gap: %+.3f", ensemble_score, gap)
            log.info("Formulas tested: %d in %.1fs", formulas_tested, elapsed)

        return SearchResults(
            best_formula=best_formula,
            ceiling=ceiling,
            ensemble_score=ensemble_score,
            gap=gap,
            convergence=convergence,
            all_formulas=all_results[:100],
            formulas_tested=formulas_tested,
            elapsed_seconds=elapsed,
        )

    def ablation(self) -> dict[str, float]:
        """Run ablation: remove each binary op and measure ceiling change.

        Returns dict mapping removed operation name to resulting ceiling.
        """
        results = {}
        full_ops = dict(self.binary_ops)

        for remove_op in list(full_ops.keys()):
            reduced_ops = {k: v for k, v in full_ops.items() if k != remove_op}
            searcher = SymbolicSearch(
                self.X,
                self.y,
                self.feature_names,
                binary_ops=reduced_ops,
                unary_ops=self.unary_ops,
                verbose=False,
            )
            result = searcher.run(ensemble=False)
            results[remove_op] = result.ceiling

        # Also with no ops removed
        results["(none)"] = self.run(ensemble=False).ceiling

        return results
