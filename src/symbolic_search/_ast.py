"""Typed symbolic expression trees for Theory Radar.

Provides proper AST representation with:
1. Typed nodes (leaf/unary/binary) with operator metadata
2. Algebraic simplification (constant folding, identity elimination, commutativity)
3. Canonical hashing for exact deduplication
4. Formal properties (monotonicity, boundedness) for search optimization
5. Efficient evaluation via compiled numpy operations

Replaces the ad-hoc FormulaTrace tuple lists with a structured,
simplifiable, deduplicatable representation.
"""

from __future__ import annotations

import logging
from enum import Enum, auto
from typing import Callable

import numpy as np
from numpy.typing import NDArray

log = logging.getLogger(__name__)

try:
    from pydantic import BaseModel, model_validator

    _HAS_PYDANTIC = True
except ImportError:
    _HAS_PYDANTIC = False


class Monotonicity(Enum):
    UNKNOWN = auto()
    INCREASING = auto()
    DECREASING = auto()
    NON_MONOTONE = auto()


if _HAS_PYDANTIC:

    class ExprNode(BaseModel):
        """A node in the symbolic expression tree (Pydantic model).

        Validated at construction, serializable to JSON,
        with computed canonical hash for exact deduplication.
        """

        model_config = {"arbitrary_types_allowed": True, "frozen": False}

        kind: str  # "const", "var", "unary", "binary"
        op: str = ""
        var_index: int = -1
        const_value: float = 0.0
        left: ExprNode | None = None
        right: ExprNode | None = None
        depth: int = 0
        n_features: int = 0
        monotonicity: Monotonicity = Monotonicity.UNKNOWN
        is_bounded: bool = False
        bound_lo: float = -1e30
        bound_hi: float = 1e30
        is_commutative: bool = False
        is_idempotent: bool = False
        _hash: str = ""

        @model_validator(mode="after")
        def _validate_structure(self) -> "ExprNode":
            if self.kind == "var" and self.var_index < 0:
                raise ValueError("var node requires var_index >= 0")
            if self.kind in ("unary", "binary") and self.left is None:
                raise ValueError(f"{self.kind} node requires left child")
            if self.kind == "binary" and self.right is None:
                raise ValueError("binary node requires right child")
            return self

        @property
        def canonical_hash(self) -> str:
            if not self._hash:
                self._hash = self._compute_hash()
            return self._hash

        def _compute_hash(self) -> str:
            if self.kind == "const":
                return f"C:{self.const_value!r}"
            if self.kind == "var":
                return f"V:{self.var_index}"
            if self.kind == "unary":
                return f"U:{self.op}({self.left.canonical_hash})"
            if self.kind == "binary":
                lh = self.left.canonical_hash
                rh = self.right.canonical_hash
                if self.is_commutative and lh > rh:
                    lh, rh = rh, lh
                return f"B:{self.op}({lh},{rh})"
            return "?"

        def evaluate(self, X: NDArray) -> NDArray:
            if self.kind == "const":
                return np.full(X.shape[0], self.const_value)
            if self.kind == "var":
                return X[:, self.var_index].copy()
            if self.kind == "unary":
                return _UNARY_EVAL[self.op](self.left.evaluate(X))
            if self.kind == "binary":
                return _BINARY_EVAL[self.op](self.left.evaluate(X), self.right.evaluate(X))
            return np.zeros(X.shape[0])

        def to_string(self, feature_names: list[str] | None = None) -> str:
            if self.kind == "const":
                return f"{self.const_value:.4g}"
            if self.kind == "var":
                if feature_names and self.var_index < len(feature_names):
                    return feature_names[self.var_index]
                return f"x{self.var_index}"
            if self.kind == "unary":
                return f"{self.op}({self.left.to_string(feature_names)})"
            if self.kind == "binary":
                ls = self.left.to_string(feature_names)
                rs = self.right.to_string(feature_names)
                return f"({ls} {self.op} {rs})"
            return "?"

        def __hash__(self):
            return hash(self.canonical_hash)

        def __eq__(self, other):
            if not isinstance(other, ExprNode):
                return False
            return self.canonical_hash == other.canonical_hash

else:
    # Fallback: plain dataclass when pydantic not installed
    from dataclasses import dataclass, field

    @dataclass(frozen=False)
    class ExprNode:
        """A node in the symbolic expression tree (dataclass fallback)."""

        kind: str
        op: str = ""
        var_index: int = -1
        const_value: float = 0.0
        left: ExprNode | None = None
        right: ExprNode | None = None
        depth: int = 0
        n_features: int = 0
        monotonicity: Monotonicity = Monotonicity.UNKNOWN
        is_bounded: bool = False
        bound_lo: float = -1e30
        bound_hi: float = 1e30
        is_commutative: bool = False
        is_idempotent: bool = False
        _hash: str = field(default="", repr=False)

        @property
        def canonical_hash(self) -> str:
            if not self._hash:
                self._hash = self._compute_hash()
            return self._hash

        def _compute_hash(self) -> str:
            if self.kind == "const":
                return f"C:{self.const_value!r}"
            if self.kind == "var":
                return f"V:{self.var_index}"
            if self.kind == "unary":
                return f"U:{self.op}({self.left.canonical_hash})"
            if self.kind == "binary":
                lh = self.left.canonical_hash
                rh = self.right.canonical_hash
                if self.is_commutative and lh > rh:
                    lh, rh = rh, lh
                return f"B:{self.op}({lh},{rh})"
            return "?"

        def evaluate(self, X: NDArray) -> NDArray:
            if self.kind == "const":
                return np.full(X.shape[0], self.const_value)
            if self.kind == "var":
                return X[:, self.var_index].copy()
            if self.kind == "unary":
                return _UNARY_EVAL[self.op](self.left.evaluate(X))
            if self.kind == "binary":
                return _BINARY_EVAL[self.op](self.left.evaluate(X), self.right.evaluate(X))
            return np.zeros(X.shape[0])

        def to_string(self, feature_names: list[str] | None = None) -> str:
            if self.kind == "const":
                return f"{self.const_value:.4g}"
            if self.kind == "var":
                if feature_names and self.var_index < len(feature_names):
                    return feature_names[self.var_index]
                return f"x{self.var_index}"
            if self.kind == "unary":
                return f"{self.op}({self.left.to_string(feature_names)})"
            if self.kind == "binary":
                ls = self.left.to_string(feature_names)
                rs = self.right.to_string(feature_names)
                return f"({ls} {self.op} {rs})"
            return "?"

        def __hash__(self):
            return hash(self.canonical_hash)

        def __eq__(self, other):
            if not isinstance(other, ExprNode):
                return False
            return self.canonical_hash == other.canonical_hash

    @property
    def canonical_hash(self) -> str:
        if not self._hash:
            self._hash = self._compute_hash()
        return self._hash

    def _compute_hash(self) -> str:
        if self.kind == "const":
            # Exact string representation — no float fuzz
            return f"C:{self.const_value!r}"
        if self.kind == "var":
            return f"V:{self.var_index}"
        if self.kind == "unary":
            return f"U:{self.op}({self.left.canonical_hash})"
        if self.kind == "binary":
            lh = self.left.canonical_hash
            rh = self.right.canonical_hash
            # Commutative normalization: sort children by hash
            if self.is_commutative and lh > rh:
                lh, rh = rh, lh
            return f"B:{self.op}({lh},{rh})"
        return "?"

    def evaluate(self, X: NDArray) -> NDArray:
        """Evaluate on data matrix X (N, d). Returns (N,) array."""
        if self.kind == "const":
            return np.full(X.shape[0], self.const_value)
        if self.kind == "var":
            return X[:, self.var_index].copy()
        if self.kind == "unary":
            child_val = self.left.evaluate(X)
            return _UNARY_EVAL[self.op](child_val)
        if self.kind == "binary":
            lv = self.left.evaluate(X)
            rv = self.right.evaluate(X)
            return _BINARY_EVAL[self.op](lv, rv)
        return np.zeros(X.shape[0])

    def to_string(self, feature_names: list[str] | None = None) -> str:
        """Human-readable formula string."""
        if self.kind == "const":
            return f"{self.const_value:.4g}"
        if self.kind == "var":
            if feature_names and self.var_index < len(feature_names):
                return feature_names[self.var_index]
            return f"x{self.var_index}"
        if self.kind == "unary":
            return f"{self.op}({self.left.to_string(feature_names)})"
        if self.kind == "binary":
            ls = self.left.to_string(feature_names)
            rs = self.right.to_string(feature_names)
            return f"({ls} {self.op} {rs})"
        return "?"

    def __hash__(self):
        return hash(self.canonical_hash)

    def __eq__(self, other):
        if not isinstance(other, ExprNode):
            return False
        return self.canonical_hash == other.canonical_hash


# ─── Constructors ──────────────────────────────────────────────────


def var(index: int) -> ExprNode:
    """Create a variable (feature) leaf node."""
    return ExprNode(
        kind="var",
        var_index=index,
        depth=1,
        n_features=1,
        monotonicity=Monotonicity.INCREASING,
    )


def const(value: float) -> ExprNode:
    """Create a constant leaf node."""
    return ExprNode(
        kind="const",
        const_value=value,
        depth=0,
        n_features=0,
        monotonicity=Monotonicity.INCREASING,
        is_bounded=True,
        bound_lo=value,
        bound_hi=value,
    )


def unary(op: str, child: ExprNode) -> ExprNode:
    """Create a unary operation node with derived properties."""
    node = ExprNode(
        kind="unary",
        op=op,
        left=child,
        depth=child.depth + 1,
        n_features=child.n_features,
    )
    _derive_unary_properties(node)
    return node


def binary(op: str, left: ExprNode, right: ExprNode) -> ExprNode:
    """Create a binary operation node with derived properties."""
    # Count unique features
    left_vars = _collect_vars(left)
    right_vars = _collect_vars(right)
    all_vars = left_vars | right_vars

    node = ExprNode(
        kind="binary",
        op=op,
        left=left,
        right=right,
        depth=1 + max(left.depth, right.depth),
        n_features=len(all_vars),
        is_commutative=op in _COMMUTATIVE_OPS,
        is_idempotent=op in _IDEMPOTENT_OPS,
    )
    _derive_binary_properties(node)
    return node


def _collect_vars(node: ExprNode) -> set[int]:
    """Collect all variable indices in the subtree."""
    if node.kind == "var":
        return {node.var_index}
    if node.kind == "const":
        return set()
    result = set()
    if node.left:
        result |= _collect_vars(node.left)
    if node.right:
        result |= _collect_vars(node.right)
    return result


# ─── Property Derivation ──────────────────────────────────────────

_COMMUTATIVE_OPS = {"+", "*", "max", "min", "hypot", "diff_sq", "harmonic", "geometric"}
_IDEMPOTENT_OPS = {"max", "min", "-", "/"}  # x op x simplifies

_MONO_INCREASING_UNARY = {"log", "sqrt", "sigmoid", "tanh"}
_MONO_DECREASING_UNARY = {"inv", "neg"}
_MONO_NON_MONOTONE_UNARY = {"sq", "abs"}

_MONO_INCREASING_BINARY = {"+", "max", "min"}
_MONO_NON_MONOTONE_BINARY = {"*", "/", "hypot", "diff_sq", "harmonic", "geometric"}


def _derive_unary_properties(node: ExprNode):
    op = node.op
    if op in _MONO_INCREASING_UNARY:
        node.monotonicity = Monotonicity.INCREASING
    elif op in _MONO_DECREASING_UNARY:
        node.monotonicity = Monotonicity.DECREASING
    elif op in _MONO_NON_MONOTONE_UNARY:
        node.monotonicity = Monotonicity.NON_MONOTONE
    else:
        node.monotonicity = Monotonicity.UNKNOWN

    # Bounds
    if op == "sigmoid":
        node.is_bounded, node.bound_lo, node.bound_hi = True, 0.0, 1.0
    elif op == "tanh":
        node.is_bounded, node.bound_lo, node.bound_hi = True, -1.0, 1.0
    elif op in ("sq", "abs"):
        node.is_bounded, node.bound_lo = True, 0.0


def _derive_binary_properties(node: ExprNode):
    op = node.op
    if op in _MONO_INCREASING_BINARY:
        node.monotonicity = Monotonicity.INCREASING
    elif op in _MONO_NON_MONOTONE_BINARY:
        node.monotonicity = Monotonicity.NON_MONOTONE
    else:
        node.monotonicity = Monotonicity.UNKNOWN

    if op in ("hypot", "diff_sq"):
        node.is_bounded, node.bound_lo = True, 0.0


# ─── Algebraic Simplification ─────────────────────────────────────


def simplify(node: ExprNode) -> ExprNode:
    """Recursively simplify an expression tree.

    Rules:
    - Constant folding (exact with Python Decimal if needed)
    - Identity elimination: x+0=x, x*1=x, x*0=0
    - Idempotent: x-x=0, x/x=1, max(x,x)=x, min(x,x)=x
    - Double negation: neg(neg(x))=x
    - sqrt(sq(x))=abs(x)
    - Commutativity normalization (sorted canonical order)
    """
    if node.kind in ("const", "var"):
        return node

    # Simplify children first
    if node.left:
        node = ExprNode(**{**node.__dict__, "left": simplify(node.left)})
    if node.right:
        node = ExprNode(**{**node.__dict__, "right": simplify(node.right)})

    # Constant folding
    if node.kind == "binary" and node.left.kind == "const" and node.right.kind == "const":
        try:
            val = _BINARY_EVAL[node.op](
                np.array([node.left.const_value]),
                np.array([node.right.const_value]),
            )[0]
            if np.isfinite(val):
                return const(float(val))
        except Exception:
            pass

    if node.kind == "unary" and node.left.kind == "const":
        try:
            val = _UNARY_EVAL[node.op](np.array([node.left.const_value]))[0]
            if np.isfinite(val):
                return const(float(val))
        except Exception:
            pass

    if node.kind == "binary":
        lft, r = node.left, node.right

        # x + 0 = x, 0 + x = x
        if node.op == "+" and r.kind == "const" and r.const_value == 0:
            return lft
        if node.op == "+" and lft.kind == "const" and lft.const_value == 0:
            return r

        # x * 1 = x, 1 * x = x
        if node.op == "*" and r.kind == "const" and r.const_value == 1:
            return lft
        if node.op == "*" and lft.kind == "const" and lft.const_value == 1:
            return r

        # x * 0 = 0
        if node.op == "*" and (
            (r.kind == "const" and r.const_value == 0)
            or (lft.kind == "const" and lft.const_value == 0)
        ):
            return const(0.0)

        # Idempotent: x - x = 0, x / x = 1, max(x,x) = x
        if lft.canonical_hash == r.canonical_hash:
            if node.op == "-":
                return const(0.0)
            if node.op == "/":
                return const(1.0)
            if node.op in ("max", "min"):
                return lft

    if node.kind == "unary":
        c = node.left
        # neg(neg(x)) = x
        if node.op == "neg" and c.kind == "unary" and c.op == "neg":
            return c.left
        # sqrt(sq(x)) = abs(x)
        if node.op == "sqrt" and c.kind == "unary" and c.op == "sq":
            return unary("abs", c.left)
        # sq(sqrt(x)) = abs(x) (for non-negative)
        if node.op == "sq" and c.kind == "unary" and c.op == "sqrt":
            return unary("abs", c.left)

    # Reset hash after simplification
    node._hash = ""
    return node


# ─── Deduplication Registry ───────────────────────────────────────


class ExprRegistry:
    """Tracks all seen expressions by canonical hash.
    Provides O(1) exact deduplication."""

    def __init__(self):
        self._seen: dict[str, ExprNode] = {}
        self.n_duplicates: int = 0

    def is_duplicate(self, node: ExprNode) -> bool:
        h = node.canonical_hash
        if h in self._seen:
            self.n_duplicates += 1
            return True
        self._seen[h] = node
        return False

    def get_canonical(self, node: ExprNode) -> ExprNode:
        """Return the canonical (first-seen) version of this expression."""
        h = node.canonical_hash
        if h in self._seen:
            return self._seen[h]
        self._seen[h] = node
        return node

    def __len__(self):
        return len(self._seen)

    def clear(self):
        self._seen.clear()
        self.n_duplicates = 0


# ─── Evaluation Functions ─────────────────────────────────────────


def _safe(arr):
    return np.nan_to_num(arr, nan=0.0, posinf=1e10, neginf=-1e10)


_BINARY_EVAL: dict[str, Callable] = {
    "+": lambda a, b: _safe(a + b),
    "-": lambda a, b: _safe(a - b),
    "*": lambda a, b: _safe(a * b),
    "/": lambda a, b: _safe(a / (b + 1e-30)),
    "max": lambda a, b: np.maximum(a, b),
    "min": lambda a, b: np.minimum(a, b),
    "hypot": lambda a, b: _safe(np.sqrt(a**2 + b**2)),
    "diff_sq": lambda a, b: _safe((a - b) ** 2),
    "harmonic": lambda a, b: _safe(2 * a * b / (a + b + 1e-30)),
    "geometric": lambda a, b: _safe(np.sign(a * b) * np.sqrt(np.abs(a * b))),
}

_UNARY_EVAL: dict[str, Callable] = {
    "log": lambda x: _safe(np.log(np.abs(x) + 1e-30)),
    "sqrt": lambda x: _safe(np.sqrt(np.abs(x))),
    "sq": lambda x: _safe(x**2),
    "abs": lambda x: np.abs(x),
    "inv": lambda x: _safe(1.0 / (x + 1e-30)),
    "neg": lambda x: -x,
    "sigmoid": lambda x: _safe(1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))),
    "tanh": lambda x: np.tanh(np.clip(x, -500, 500)),
}
