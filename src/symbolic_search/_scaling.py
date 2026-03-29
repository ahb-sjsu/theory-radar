"""
Scaling improvements for deeper A* search (depth 4+):

1. Dynamic Feature Pruning: eliminate features that never pass AUROC filter
2. Semantic Hashing: detect algebraically equivalent formulas cheaply
3. Beam Search with Diversity: maintain diverse formula population
"""

from __future__ import annotations

import hashlib
import logging

import numpy as np
from numpy.typing import NDArray

log = logging.getLogger(__name__)


# ============================================================
# 1. Dynamic Feature Pruning
# ============================================================

def identify_dead_features(
    X: NDArray,
    y: NDArray,
    binary_ops: dict,
    auroc_threshold: float = 0.55,
    base_feature_idx: int | None = None,
) -> list[int]:
    """Identify features that never produce a useful formula at depth 2.

    A feature j is "dead" if for ALL other features i and ALL binary
    operations b: AUROC(b(x_i, x_j)) < auroc_threshold.

    Dead features can be eliminated from deeper search, reducing d
    and thus the exponential base.
    """
    from sklearn.metrics import roc_auc_score

    N, d = X.shape
    actual = y.astype(bool)
    feature_useful = np.zeros(d, dtype=bool)

    for j in range(d):
        for i in range(d):
            if i == j:
                continue
            for op_fn in binary_ops.values():
                try:
                    vals = op_fn(X[:, i], X[:, j])
                    vals = np.nan_to_num(vals, nan=0, posinf=1e10, neginf=-1e10)
                    auc = roc_auc_score(actual, vals)
                    auc = max(auc, 1 - auc)
                    if auc >= auroc_threshold:
                        feature_useful[j] = True
                        break  # This feature is useful, stop checking
                except Exception:
                    continue
            if feature_useful[j]:
                break

    dead = [j for j in range(d) if not feature_useful[j]]
    return dead


def prune_features(
    X: NDArray,
    feature_names: list[str],
    dead_indices: list[int],
) -> tuple[NDArray, list[str]]:
    """Remove dead features from the feature matrix."""
    alive = [i for i in range(X.shape[1]) if i not in dead_indices]
    return X[:, alive], [feature_names[i] for i in alive]


# ============================================================
# 2. Semantic Hashing
# ============================================================

class SemanticHasher:
    """Detect algebraically equivalent formulas using output hashing.

    Evaluates each formula on a tiny fixed subset of data points and
    hashes the output vector. Formulas with identical hashes are
    algebraically equivalent (with high probability) and can be
    deduplicated before expensive AUROC/F1 computation.
    """

    def __init__(self, X: NDArray, n_probe: int = 7, seed: int = 42):
        """Initialize with a small random subset of data points."""
        rng = np.random.RandomState(seed)
        N = X.shape[0]
        self.probe_indices = rng.choice(N, size=min(n_probe, N), replace=False)
        self.X_probe = X[self.probe_indices]
        self.seen_hashes: set[str] = set()
        self.n_duplicates = 0

    def hash_values(self, values: NDArray) -> str:
        """Compute a hash from the formula's output on probe points."""
        probe_vals = values[self.probe_indices]
        # Round to 6 decimal places to handle floating-point variation
        rounded = np.round(probe_vals, 6)
        return hashlib.md5(rounded.tobytes()).hexdigest()

    def is_duplicate(self, values: NDArray) -> bool:
        """Check if this formula's output has been seen before."""
        h = self.hash_values(values)
        if h in self.seen_hashes:
            self.n_duplicates += 1
            return True
        self.seen_hashes.add(h)
        return False

    def reset(self):
        """Clear the hash table."""
        self.seen_hashes.clear()
        self.n_duplicates = 0


# ============================================================
# 3. Beam Search with Diversity
# ============================================================

def diversity_penalty(
    candidate_values: NDArray,
    beam_values: list[NDArray],
    threshold: float = 0.95,
) -> float:
    """Compute diversity penalty for a candidate formula.

    Returns a value in [0, 1] where 0 = completely novel and
    1 = perfectly correlated with an existing beam member.
    """
    if not beam_values:
        return 0.0

    max_corr = 0.0
    for existing in beam_values:
        try:
            corr = abs(np.corrcoef(candidate_values, existing)[0, 1])
            max_corr = max(max_corr, corr)
        except Exception:
            pass

    return max_corr


def beam_search_diverse(
    X: NDArray,
    y: NDArray,
    feature_names: list[str],
    binary_ops: dict,
    unary_ops: dict,
    beam_width: int = 100,
    max_depth: int = 4,
    diversity_weight: float = 0.3,
    auroc_threshold: float = 0.55,
) -> list[dict]:
    """Beam search with diversity penalties for deep formula search.

    At each depth level, keeps the top beam_width formulas ranked by:
        score = F1 - diversity_weight * max_correlation_with_beam

    This prevents the beam from collapsing into variations of the
    same formula while still prioritizing high-F1 candidates.
    """
    from symbolic_search._search import _f1_threshold_sweep
    from sklearn.metrics import roc_auc_score

    N, d = X.shape
    actual = y.astype(bool)
    hasher = SemanticHasher(X)

    # Initialize beam with single features
    beam = []
    for i in range(d):
        vals = X[:, i]
        f1, thresh, direction = _f1_threshold_sweep(vals, actual)
        beam.append({
            "formula": feature_names[i],
            "values": vals,
            "f1": f1,
            "depth": 1,
        })
        hasher.is_duplicate(vals)  # register hash

    beam.sort(key=lambda x: -x["f1"])
    beam = beam[:beam_width]

    best_overall = max(beam, key=lambda x: x["f1"])
    log.info("Beam depth 1: %d formulas, best F1=%.4f (%s)",
             len(beam), best_overall["f1"], best_overall["formula"])

    # Expand beam depth by depth
    for depth in range(2, max_depth + 1):
        candidates = []
        beam_values = [b["values"] for b in beam]

        for parent in beam:
            # Binary operations with each leaf
            for j in range(d):
                leaf_vals = X[:, j]
                for op_name, op_fn in binary_ops.items():
                    try:
                        vals = op_fn(parent["values"], leaf_vals)
                        vals = np.nan_to_num(vals, nan=0, posinf=1e10, neginf=-1e10)

                        # Semantic dedup
                        if hasher.is_duplicate(vals):
                            continue

                        # AUROC filter
                        try:
                            auc = roc_auc_score(actual, vals)
                            auc = max(auc, 1 - auc)
                        except:
                            continue
                        if auc < auroc_threshold:
                            continue

                        # F1
                        f1, _, _ = _f1_threshold_sweep(vals, actual)

                        # Diversity penalty
                        div = diversity_penalty(vals, beam_values)
                        score = f1 - diversity_weight * div

                        candidates.append({
                            "formula": f"({parent['formula']} {op_name} {feature_names[j]})",
                            "values": vals,
                            "f1": f1,
                            "score": score,
                            "depth": depth,
                        })
                    except Exception:
                        pass

            # Non-monotone unary operations
            for uname, ufn in unary_ops.items():
                if uname in {"log", "sqrt", "inv", "neg"}:  # skip monotone
                    continue
                try:
                    vals = ufn(parent["values"])
                    vals = np.nan_to_num(vals, nan=0, posinf=1e10, neginf=-1e10)

                    if hasher.is_duplicate(vals):
                        continue

                    f1, _, _ = _f1_threshold_sweep(vals, actual)
                    div = diversity_penalty(vals, beam_values)
                    score = f1 - diversity_weight * div

                    candidates.append({
                        "formula": f"{uname}({parent['formula']})",
                        "values": vals,
                        "f1": f1,
                        "score": score,
                        "depth": depth,
                    })
                except Exception:
                    pass

        if not candidates:
            break

        # Select top beam_width by score (F1 - diversity)
        candidates.sort(key=lambda x: -x["score"])
        beam = candidates[:beam_width]

        best_at_depth = max(beam, key=lambda x: x["f1"])
        if best_at_depth["f1"] > best_overall["f1"]:
            best_overall = best_at_depth

        log.info("Beam depth %d: %d candidates, kept %d, best F1=%.4f (%s), "
                 "dedup=%d",
                 depth, len(candidates), len(beam),
                 best_at_depth["f1"], best_at_depth["formula"][:40],
                 hasher.n_duplicates)

    return [{
        "best_formula": best_overall["formula"],
        "best_f1": best_overall["f1"],
        "best_depth": best_overall["depth"],
        "total_dedup": hasher.n_duplicates,
    }]
