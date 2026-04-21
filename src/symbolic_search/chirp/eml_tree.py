"""Minimal eml-tree: one operator, uniform grammar, trainable with Adam.

Grammar:
    S    -> Leaf | eml(S, S)
    Leaf  = softmax(w) . [x_1, ..., x_D, c]
    eml(x, y) = exp(clamp(x)) - log(clamp(y))

A depth-d full binary tree has 2**d leaves; each leaf carries (D+1) softmax
logits plus one learnable scalar constant. Soft feature selection keeps the
whole thing differentiable end-to-end.

Numerical guards: exp is clamped at EXP_CLAMP ~= 15 (exp(15) ~= 3.3e6), log
is floored at EPS to avoid log(0), and the operator output is hard-clamped
at OUT_CLAMP. These bounds preserve gradient flow while preventing overflow
during early training when leaf logits are near-uniform.
"""
from __future__ import annotations

import torch
import torch.nn as nn

EPS = 1e-6
EXP_CLAMP = 15.0
OUT_CLAMP = 50.0


def eml(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """eml(x, y) = exp(x) - log(y), numerically guarded.

    The Odrzywolek (2026) operator. With the constant 1 it is functionally
    complete over the elementary functions.
    """
    x_safe = x.clamp(min=-EXP_CLAMP, max=EXP_CLAMP)
    log_upper = torch.exp(torch.tensor(EXP_CLAMP)).item()
    y_safe = y.clamp(min=EPS, max=log_upper)
    out = torch.exp(x_safe) - torch.log(y_safe)
    return out.clamp(min=-OUT_CLAMP, max=OUT_CLAMP)


class EmlLeaf(nn.Module):
    """Soft-selected leaf.

    Output = softmax(logits) . concat(X, const_column).
    Gradient flows to every feature, so the leaf can "slide" toward the
    right feature during training instead of being hard-selected up front.
    """

    def __init__(self, n_features: int):
        super().__init__()
        self.n_features = n_features
        # +1 slot for the learnable constant. Small random init so sibling
        # leaves aren't identical at step 0 (symmetric gradients stall Adam).
        self.logits = nn.Parameter(torch.randn(n_features + 1) * 0.1)
        self.const = nn.Parameter(torch.tensor(1.0))

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        const_col = self.const.expand(X.shape[0], 1)
        X_aug = torch.cat([X, const_col], dim=1)  # (N, D+1)
        return X_aug @ torch.softmax(self.logits, dim=0)

    def dominant_feature(self) -> tuple[int, float]:
        """Return (slot_index, weight). slot == n_features is the const slot."""
        weights = torch.softmax(self.logits.detach(), dim=0)
        idx = int(torch.argmax(weights).item())
        return idx, float(weights[idx].item())


class EmlTree(nn.Module):
    """Full binary eml-tree of configurable depth.

    Has an optional trainable output rescaling (`scale`, `bias`) applied at
    the root: `output = scale * tree_out + bias`. This lets the eml-tree
    focus on discovering function *shape* while the linear readout absorbs
    scale/offset — a standard trick in symbolic regression. Pass
    `output_rescale=False` to disable.
    """

    def __init__(
        self, depth: int, n_features: int, *, output_rescale: bool = True
    ):
        super().__init__()
        if depth < 0:
            raise ValueError("depth must be >= 0")
        self.depth = depth
        self.n_features = n_features
        self.output_rescale = output_rescale

        if depth == 0:
            self.leaf: EmlLeaf | None = EmlLeaf(n_features)
            self.left: EmlTree | None = None
            self.right: EmlTree | None = None
        else:
            self.leaf = None
            self.left = EmlTree(depth - 1, n_features, output_rescale=False)
            self.right = EmlTree(depth - 1, n_features, output_rescale=False)

        if output_rescale:
            # Root-only rescaling: tree shape * scale + bias.
            # scale=1.0 preserves tree magnitude at step 0; Adam learns it down
            # if the tree over-produces.
            self.scale = nn.Parameter(torch.tensor(1.0))
            self.bias = nn.Parameter(torch.tensor(0.0))
        else:
            self.scale = None
            self.bias = None

    def _raw_forward(self, X: torch.Tensor) -> torch.Tensor:
        if self.leaf is not None:
            return self.leaf(X)
        assert self.left is not None and self.right is not None
        return eml(self.left(X), self.right(X))

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        out = self._raw_forward(X)
        if self.output_rescale:
            assert self.scale is not None and self.bias is not None
            out = self.scale * out + self.bias
        return out

    def leaves(self):
        """Pre-order traversal yielding every EmlLeaf in the tree."""
        if self.leaf is not None:
            yield self.leaf
        else:
            assert self.left is not None and self.right is not None
            yield from self.left.leaves()
            yield from self.right.leaves()

    def n_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())
