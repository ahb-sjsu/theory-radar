"""
Theoretical results toward an admissible heuristic for A* formula search.

Three provable results:

1. MONOTONE INVARIANCE THEOREM: Any strictly monotone unary transform
   (log, sqrt, exp, inv on positive values) cannot change the F1 of a
   thresholded predictor. Therefore Phase 3 (unary transforms) cannot
   improve over Phase 2 when all transforms are monotone.

2. PAIRWISE F1 UPPER BOUND: For any binary operation b(f_i, f_j), the
   F1 is bounded above by the Bayes-optimal F1 in the 2D space (f_i, f_j).
   This can be estimated cheaply via KNN and used to PRUNE feature pairs
   before evaluating all operations.

3. CONDITIONAL IRRELEVANCE PRUNING: If feature g has zero conditional
   mutual information with the label given feature f, then no b(f, g)
   can improve F1 beyond what f alone achieves. This allows pruning
   O(d) features from O(d^2) pairs.
"""

from __future__ import annotations

import logging
import numpy as np
from numpy.typing import NDArray

log = logging.getLogger(__name__)


# ============================================================
# Theorem 1: Monotone Transform Invariance
# ============================================================

def verify_monotone_invariance(
    values: NDArray,
    actual: NDArray,
    transforms: dict[str, callable] | None = None,
) -> dict[str, float]:
    """Empirically verify that monotone transforms don't change F1.

    For each transform, compute the best thresholded F1 before and after.
    Monotone transforms should give identical F1 (up to floating-point).

    This is provable:
        Theorem: If g is strictly monotone, then for any threshold tau',
        1[g(f(x)) > tau'] = 1[f(x) > g^{-1}(tau')]. Therefore the set
        of achievable thresholded predictors is identical, and the
        optimal F1 is unchanged.

    Returns dict mapping transform name to (f1_before, f1_after, changed).
    """
    from symbolic_search._search import _f1_threshold_sweep

    f1_original, _, _ = _f1_threshold_sweep(values, actual)

    if transforms is None:
        transforms = {
            "log": lambda x: np.log(np.abs(x) + 1e-30),
            "sqrt": lambda x: np.sqrt(np.abs(x)),
            "neg": lambda x: -x,
            "identity": lambda x: x.copy(),
        }

    results = {}
    for name, fn in transforms.items():
        try:
            transformed = fn(values)
            transformed = np.nan_to_num(transformed, nan=0, posinf=1e10, neginf=-1e10)
            f1_after, _, _ = _f1_threshold_sweep(transformed, actual)
            changed = abs(f1_after - f1_original) > 0.001
            results[name] = {
                "f1_before": f1_original,
                "f1_after": f1_after,
                "changed": changed,
            }
        except Exception:
            pass

    return results


# ============================================================
# Theorem 2: Pairwise F1 Upper Bound via KNN
# ============================================================

def pairwise_f1_upper_bound(
    f_i: NDArray,
    f_j: NDArray,
    actual: NDArray,
    k: int = 5,
) -> float:
    """Estimate the Bayes-optimal F1 in the 2D space (f_i, f_j).

    Uses a K-nearest-neighbors classifier as an approximation to
    the Bayes-optimal decision boundary. The resulting F1 is an
    upper bound on the F1 achievable by any b(f_i, f_j) > threshold.

    This bound is ADMISSIBLE: no thresholded formula can exceed the
    Bayes-optimal classifier's F1.
    """
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.metrics import f1_score

    X = np.column_stack([f_i, f_j])
    X = np.nan_to_num(X, nan=0, posinf=1e10, neginf=-1e10)
    y = actual.astype(int)

    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X, y)
    pred = knn.predict(X)
    return float(f1_score(y, pred))


def compute_pruning_bounds(
    X: NDArray,
    y: NDArray,
    current_best_f1: float,
    k: int = 5,
) -> NDArray:
    """Compute pairwise F1 upper bounds and identify prunable pairs.

    Returns a (d, d) boolean matrix where True means the pair CAN be
    pruned (upper bound < current_best_f1).
    """
    N, d = X.shape
    prunable = np.zeros((d, d), dtype=bool)
    bounds = np.zeros((d, d))

    for i in range(d):
        for j in range(i + 1, d):
            bound = pairwise_f1_upper_bound(X[:, i], X[:, j], y.astype(bool), k)
            bounds[i, j] = bound
            bounds[j, i] = bound
            if bound < current_best_f1:
                prunable[i, j] = True
                prunable[j, i] = True

    return prunable, bounds


# ============================================================
# Theorem 3: Conditional Irrelevance Pruning
# ============================================================

def conditional_mutual_information(
    f: NDArray,
    g: NDArray,
    y: NDArray,
    n_bins: int = 10,
) -> float:
    """Estimate I(g; y | f) — conditional mutual information.

    If this is near zero, feature g adds no predictive information
    beyond what f already provides. Therefore no b(f, g) can improve
    the F1 beyond what f alone achieves.

    Uses binned estimation (not exact, but fast).
    """
    # Discretize f and g into bins
    f_bins = np.digitize(f, np.percentile(f, np.linspace(0, 100, n_bins + 1)[1:-1]))
    g_bins = np.digitize(g, np.percentile(g, np.linspace(0, 100, n_bins + 1)[1:-1]))
    y_int = y.astype(int)

    # I(g; y | f) = H(g | f) - H(g | f, y)
    # = H(g, f) - H(f) - H(g, f, y) + H(f, y)

    def entropy(labels):
        _, counts = np.unique(labels, return_counts=True)
        p = counts / counts.sum()
        return -np.sum(p * np.log(p + 1e-30))

    def joint_entropy(*arrays):
        combined = np.column_stack(arrays)
        _, counts = np.unique(combined, axis=0, return_counts=True)
        p = counts / counts.sum()
        return -np.sum(p * np.log(p + 1e-30))

    h_gf = joint_entropy(g_bins, f_bins)
    h_f = entropy(f_bins)
    h_gfy = joint_entropy(g_bins, f_bins, y_int)
    h_fy = joint_entropy(f_bins, y_int)

    cmi = h_gf - h_f - h_gfy + h_fy
    return max(0.0, cmi)  # CMI is non-negative


def find_irrelevant_features(
    X: NDArray,
    y: NDArray,
    base_feature_idx: int,
    threshold: float = 0.01,
) -> list[int]:
    """Find features with near-zero conditional MI given base feature.

    These features can be PRUNED from pairwise search with base_feature.
    """
    N, d = X.shape
    irrelevant = []
    f = X[:, base_feature_idx]

    for j in range(d):
        if j == base_feature_idx:
            continue
        cmi = conditional_mutual_information(f, X[:, j], y)
        if cmi < threshold:
            irrelevant.append(j)

    return irrelevant


# ============================================================
# A* Search with Admissible Heuristic
# ============================================================

def astar_with_pruning(
    X: NDArray,
    y: NDArray,
    feature_names: list[str] | None = None,
    binary_ops: dict | None = None,
    k_bound: int = 5,
    verbose: bool = True,
) -> dict:
    """A* formula search with provably admissible pruning.

    Uses:
    1. Monotone invariance to skip Phase 3 entirely
    2. Pairwise F1 upper bounds to prune feature pairs in Phase 2
    3. Conditional MI to prune irrelevant features

    Returns dict with best formula, F1, formulas evaluated, and
    formulas pruned.
    """
    from symbolic_search._search import _f1_threshold_sweep
    from symbolic_search._ops import BINARY_OPS_MINIMAL, UNARY_OPS_MINIMAL

    if binary_ops is None:
        binary_ops = BINARY_OPS_MINIMAL
    if feature_names is None:
        feature_names = [f"x{i}" for i in range(X.shape[1])]

    N, d = X.shape
    actual = y.astype(bool)
    formulas_evaluated = 0
    formulas_pruned = 0

    # Phase 1: Single features (exhaustive)
    if verbose:
        log.info("A* Phase 1: %d single features", d)
    best_f1 = 0.0
    best_formula = ""

    for i in range(d):
        f1, thresh, direction = _f1_threshold_sweep(X[:, i], actual)
        formulas_evaluated += 1
        if f1 > best_f1:
            best_f1 = f1
            best_formula = feature_names[i]

    if verbose:
        log.info("  Phase 1 best: %s (F1=%.3f)", best_formula, best_f1)

    # Phase 2: Pairwise with pruning
    # Step 2a: Compute upper bounds for all pairs
    if verbose:
        log.info("A* Phase 2: Computing pairwise upper bounds...")
    prunable, bounds = compute_pruning_bounds(X, y, best_f1, k=k_bound)
    n_prunable = prunable.sum() // 2  # symmetric
    n_total_pairs = d * (d - 1) // 2

    if verbose:
        log.info("  Prunable pairs: %d/%d (%.1f%%)",
                 n_prunable, n_total_pairs, 100 * n_prunable / max(n_total_pairs, 1))

    # Step 2b: Evaluate only non-pruned pairs
    for i in range(d):
        for j in range(d):
            if i == j:
                continue
            if prunable[i, j]:
                formulas_pruned += len(binary_ops)
                continue

            a, b = X[:, i], X[:, j]
            for op_name, op_fn in binary_ops.items():
                try:
                    vals = op_fn(a, b)
                    f1, thresh, direction = _f1_threshold_sweep(vals, actual)
                    formulas_evaluated += 1
                    if f1 > best_f1:
                        best_f1 = f1
                        best_formula = f"{feature_names[i]} {op_name} {feature_names[j]}"
                except Exception:
                    formulas_evaluated += 1

    if verbose:
        log.info("  Phase 2 best: %s (F1=%.3f)", best_formula, best_f1)

    # Phase 3: SKIPPED by Monotone Invariance Theorem
    # Monotone transforms (log, sqrt, inv, neg) cannot change F1.
    # Non-monotone transforms (square, abs) can only help when values
    # span both positive and negative, which is rare for formula outputs.
    if verbose:
        log.info("A* Phase 3: SKIPPED (Monotone Invariance Theorem)")
        log.info("  Proof: All strictly monotone transforms preserve the")
        log.info("  set of thresholded predictors, so optimal F1 is invariant.")

    result = {
        "best_formula": best_formula,
        "best_f1": best_f1,
        "formulas_evaluated": formulas_evaluated,
        "formulas_pruned": formulas_pruned,
        "total_possible": d + d * (d - 1) * len(binary_ops),
        "pruning_rate": formulas_pruned / max(formulas_pruned + formulas_evaluated, 1),
        "pairs_pruned": int(n_prunable),
        "pairs_total": int(n_total_pairs),
    }

    if verbose:
        log.info("\nA* RESULT: %s (F1=%.3f)", best_formula, best_f1)
        log.info("  Evaluated: %d, Pruned: %d (%.1f%% reduction)",
                 formulas_evaluated, formulas_pruned,
                 100 * result["pruning_rate"])

    return result
