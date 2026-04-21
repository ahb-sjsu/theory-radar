"""Broadband "chirp" reconnaissance for Theory Radar.

Implements the eml-tree primitive from Odrzywolek (2026, arXiv:2603.21852):

    eml(x, y) = exp(x) - log(y)

This operator + the constant 1 is functionally complete over the elementary
functions. A binary tree of depth d with this single operator is the
symbolic-math analogue of an RF chirp: one configurable "waveform" whose
trainable parameters sweep the entire elementary-function space. Trained
with Adam on (X, y), the converged tree is a broadband probe whose gradients
and constants form a `ProbePrior` that narrows the classical Theory Radar
enumeration before it runs.

The probe does NOT replace the interpretable symbolic classifier — it feeds
the classical engine a tighter search space. Interpretability is preserved
because the public output of `radar.search()` is still a human-readable
formula over the active_features the probe surfaced.

Public API:
    from symbolic_search.chirp import chirp, ProbePrior
    prior = chirp(X, y, depth=3, epochs=500)
    print(prior.verdict())          # "narrow_prior" | "wide_prior" | "no_elementary_fit"
    print(prior.active_features)    # top-k features by gradient-norm
"""
from __future__ import annotations

try:
    import torch  # noqa: F401
    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False

__all__ = ["chirp", "ProbePrior", "_HAS_TORCH"]

if _HAS_TORCH:
    from .probe import chirp
    from .prior import ProbePrior
else:
    def chirp(*_args, **_kwargs):  # type: ignore[misc]
        raise ImportError(
            "The chirp module requires PyTorch. Install with: "
            "pip install 'theory-radar[chirp]'"
        )

    class ProbePrior:  # type: ignore[no-redef]
        """Placeholder raised when PyTorch is not installed."""

        def __init__(self, *_args, **_kwargs):
            raise ImportError(
                "The chirp module requires PyTorch. Install with: "
                "pip install 'theory-radar[chirp]'"
            )
