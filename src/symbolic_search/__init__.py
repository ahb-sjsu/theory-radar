"""
Theory Radar: Symbolic formula search with configurable projections,
meta-learned pruning, ensemble formulas, and autotune.

Basic usage:
    from symbolic_search import TheoryRadar
    radar = TheoryRadar(X, y)
    result = radar.search(mode="fast")
    print(result.formula, result.f1)

With PCA projections (access all features):
    radar = TheoryRadar(X, y, projection="pca")
    result = radar.search(mode="fast")

With Tucker decomposition (capture feature interactions):
    radar = TheoryRadar(X, y, projection="tucker")

Combined projections + subspace fuzzing:
    radar = TheoryRadar(X, y,
        projection=["pca", "tucker"],
        n_subspaces=10, subspace_k=12)

Autotune (find best configuration automatically):
    radar, result = TheoryRadar.autotune(X, y, max_time=120)
"""

from symbolic_search.radar import TheoryRadar, RadarResult
from symbolic_search._search import SymbolicSearch, SearchResults
from symbolic_search._projections import (
    PCAProjection,
    PLSProjection,
    TuckerProjection,
    KernelProjection,
    NeuralProjection,
    SparsePCAProjection,
    PROJECTIONS,
)

__all__ = [
    "TheoryRadar",
    "RadarResult",
    "SymbolicSearch",
    "SearchResults",
    "PCAProjection",
    "PLSProjection",
    "TuckerProjection",
    "KernelProjection",
    "NeuralProjection",
    "SparsePCAProjection",
    "PROJECTIONS",
]
__version__ = "0.4.0"
