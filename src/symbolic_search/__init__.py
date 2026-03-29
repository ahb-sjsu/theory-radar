"""
symbolic-search: Phased exhaustive symbolic formula discovery with ceiling detection.

Usage:
    from symbolic_search import SymbolicSearch

    search = SymbolicSearch(X, y, feature_names=["x1", "x2", "x3"])
    results = search.run()

    print(results.best_formula)      # "x1 hypot x2"
    print(results.ceiling)           # 0.996
    print(results.gap)               # -0.007 (formula beats ensemble!)
    print(results.convergence)       # {1: 0.83, 2: 0.996, 3: 0.996}
"""

from symbolic_search._search import SymbolicSearch, SearchResults

__all__ = ["SymbolicSearch", "SearchResults"]
__version__ = "0.1.0"
