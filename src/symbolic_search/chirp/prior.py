"""The ProbePrior — what the broadband probe hands to the classical engine."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class ProbePrior:
    """Output of a `chirp(...)` call.

    Used by `TheoryRadar.search(prior=...)` to narrow the classical
    enumeration without mandating it — if verdict is `"no_elementary_fit"`
    the caller can skip symbolic search entirely and report that the
    ensemble is the right tool for this dataset.

    Attributes:
        feature_importance: (D,) array, gradient-norm of output w.r.t. each
            input column averaged over the training set. Higher = more
            load-bearing.
        active_features: Top-k feature indices by importance.
        notable_constants: List of (leaf_index, value, matched_name) for
            leaf constants that converged near natural values (1, pi, e,
            small integers). HINTS only, not facts — a random leaf const
            of 0.97 isn't evidence of "1".
        final_loss: Training loss of the best restart.
        converged: True if `final_loss < 0.5 * baseline_loss` (baseline =
            predict the mean of y). Used by `verdict()`.
        source_tree: The raw trained `EmlTree` (for dashboards / debugging).
            Marked as `repr=False` so dataclass printing stays readable.
        depth: Tree depth used.
        baseline_loss: Loss of the constant-mean predictor (reference point).
    """

    feature_importance: np.ndarray
    active_features: list[int]
    notable_constants: list[tuple[int, float, str]]
    final_loss: float
    converged: bool
    source_tree: Any = field(default=None, repr=False)
    depth: int = 0
    baseline_loss: float = 1.0

    def restrict(self, candidates: list[int]) -> list[int]:
        """Keep only candidate features the probe says matter."""
        active = set(self.active_features)
        return [f for f in candidates if f in active]

    def verdict(self) -> str:
        """Top-level summary for the classical engine.

        - ``"no_elementary_fit"``: probe did not converge. Classical search
          is unlikely to find a better formula than a tuned ensemble. Skip.
        - ``"narrow_prior"``: probe converged and flagged ≤ 3 features as
          active. High-confidence pointer to a small-feature formula.
        - ``"wide_prior"``: probe converged but flagged > 3 features.
          Use the probe ranking, but don't aggressively restrict.
        """
        if not self.converged:
            return "no_elementary_fit"
        if len(self.active_features) <= 3:
            return "narrow_prior"
        return "wide_prior"

    def summary(self) -> str:
        """One-line human-readable summary."""
        n_const = len(self.notable_constants)
        return (
            f"ProbePrior(verdict={self.verdict()!r}, "
            f"depth={self.depth}, "
            f"active={self.active_features}, "
            f"notable_consts={n_const}, "
            f"loss={self.final_loss:.4f} / baseline={self.baseline_loss:.4f})"
        )
