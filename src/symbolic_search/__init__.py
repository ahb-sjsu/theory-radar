"""
Theory Radar: A* formula search with provable DAG heuristic.

Usage:
    from symbolic_search import TheoryRadar

    radar = TheoryRadar(X, y, feature_names=["age", "bmi", "glucose"])

    # Strict A* (provable optimality guarantee)
    result = radar.search(mode="strict", f1_target=0.90)

    # Fast mode (empirical AUROC pruning, 10-100x faster)
    result = radar.search(mode="fast", auroc_threshold=0.55)

    # Auto: strict first, fast fallback if timeout
    result = radar.search(mode="auto", f1_target=0.90, timeout=60)

    print(result.formula)     # "(bmi min age) + glucose"
    print(result.f1)          # 0.694
    print(result.guaranteed)  # True (strict mode)
"""

from symbolic_search.radar import TheoryRadar, RadarResult
from symbolic_search._search import SymbolicSearch, SearchResults

__all__ = ["TheoryRadar", "RadarResult", "SymbolicSearch", "SearchResults"]
__version__ = "0.3.0"
