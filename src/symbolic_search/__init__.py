"""
Theory Radar: A* formula search with provable pruning.

Usage:
    from symbolic_search import TheoryRadar, SymbolicSearch

    # A* search (recommended)
    from symbolic_search._theory import astar_with_pruning
    result = astar_with_pruning(X, y, feature_names)

    # Phased enumeration (baseline)
    search = SymbolicSearch(X, y, feature_names=["x1", "x2", "x3"])
    results = search.run()

    print(results.best_formula)      # "x1 hypot x2"
    print(results.ceiling)           # 0.996
    print(results.gap)               # -0.007 (formula beats ensemble!)
"""

from symbolic_search._search import SymbolicSearch, SearchResults

__all__ = ["SymbolicSearch", "SearchResults"]
__version__ = "0.2.0"
