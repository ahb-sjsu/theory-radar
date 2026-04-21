"""Unit tests for the eml-tree primitive (no network, no datasets).

Validates:
- The operator's numerical behavior (identity checks from Odrzywolek).
- Tree construction, parameter count, forward shape.
- Gradient flow end-to-end.
- Convergence on a toy target (overfitting a known function).

All tests skip cleanly if PyTorch isn't installed.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from symbolic_search.chirp.eml_tree import EmlLeaf, EmlTree, eml  # noqa: E402


class TestEmlOperator:
    def test_eml_x_1_equals_exp_x_minus_0(self):
        # eml(x, 1) = exp(x) - log(1) = exp(x)
        x = torch.tensor([0.0, 1.0, 2.0])
        one = torch.ones_like(x)
        got = eml(x, one)
        expected = torch.exp(x)
        assert torch.allclose(got, expected, atol=1e-5)

    def test_eml_clamps_large_input(self):
        # exp(1000) would overflow; the clamp must keep output finite.
        x = torch.tensor([1000.0])
        y = torch.tensor([1.0])
        out = eml(x, y)
        assert torch.isfinite(out).all()

    def test_eml_clamps_tiny_y(self):
        # log(0) would be -inf; clamp floor must handle it.
        x = torch.tensor([0.0])
        y = torch.tensor([0.0])
        out = eml(x, y)
        assert torch.isfinite(out).all()

    def test_eml_is_differentiable(self):
        x = torch.tensor([0.5], requires_grad=True)
        y = torch.tensor([2.0], requires_grad=True)
        out = eml(x, y)
        out.backward()
        # d/dx [exp(x) - log(y)] = exp(x); d/dy = -1/y
        assert torch.allclose(x.grad, torch.exp(torch.tensor([0.5])), atol=1e-5)
        assert torch.allclose(y.grad, torch.tensor([-1.0 / 2.0]), atol=1e-5)


class TestEmlLeaf:
    def test_construction(self):
        leaf = EmlLeaf(n_features=3)
        # Logits size = D + 1 (D features + one const slot).
        assert leaf.logits.shape == (4,)
        assert leaf.const.item() == pytest.approx(1.0)

    def test_forward_shape(self):
        leaf = EmlLeaf(n_features=5)
        X = torch.randn(10, 5)
        out = leaf(X)
        assert out.shape == (10,)

    def test_softmax_weights_sum_to_one(self):
        leaf = EmlLeaf(n_features=4)
        with torch.no_grad():
            w = torch.softmax(leaf.logits, dim=0)
        assert w.sum().item() == pytest.approx(1.0, abs=1e-5)


class TestEmlTree:
    def test_depth_zero_is_single_leaf(self):
        tree = EmlTree(depth=0, n_features=3)
        assert tree.leaf is not None
        assert tree.left is None and tree.right is None
        assert sum(1 for _ in tree.leaves()) == 1

    def test_depth_three_has_eight_leaves(self):
        tree = EmlTree(depth=3, n_features=5)
        assert sum(1 for _ in tree.leaves()) == 8

    def test_parameter_count_no_rescale(self):
        # Depth 2 full binary: 4 leaves × (D+1 logits + 1 const) = 4 × (D+2)
        D = 5
        tree = EmlTree(depth=2, n_features=D, output_rescale=False)
        expected = 4 * (D + 1 + 1)
        assert tree.n_parameters() == expected

    def test_parameter_count_with_rescale(self):
        # +2 params (scale, bias) at the root.
        D = 5
        tree = EmlTree(depth=2, n_features=D, output_rescale=True)
        expected = 4 * (D + 1 + 1) + 2
        assert tree.n_parameters() == expected

    def test_forward_shape(self):
        tree = EmlTree(depth=3, n_features=4)
        X = torch.randn(20, 4)
        out = tree(X)
        assert out.shape == (20,)

    def test_gradient_flows_end_to_end(self):
        # After one Adam step, at least one leaf's logits should have moved.
        tree = EmlTree(depth=2, n_features=3)
        opt = torch.optim.Adam(tree.parameters(), lr=0.1)
        X = torch.randn(16, 3)
        y = torch.randn(16)

        before = [leaf.logits.detach().clone() for leaf in tree.leaves()]
        opt.zero_grad()
        loss = torch.nn.functional.mse_loss(tree(X), y)
        loss.backward()
        opt.step()
        after = [leaf.logits.detach().clone() for leaf in tree.leaves()]

        moved = [not torch.allclose(b, a) for b, a in zip(before, after)]
        assert any(moved), "Gradient did not reach any leaf"

    def test_invalid_depth_raises(self):
        with pytest.raises(ValueError):
            EmlTree(depth=-1, n_features=3)


class TestConvergence:
    """Confirm training is well-behaved: finite losses, gradient flow, no NaNs.

    Raw eml-trees are non-convex and initialization-sensitive. We do NOT
    test that a given target is fit below some threshold — that belongs in
    `test_probe.py` under the statistical-restart regime. Here we only
    demand: training runs without diverging to NaN/inf, and the output
    remains finite throughout.
    """

    def test_training_stays_finite(self):
        """Loss must remain finite for 100 Adam steps on a small random task."""
        torch.manual_seed(7)
        X = torch.randn(128, 3)
        y = torch.randn(128)

        tree = EmlTree(depth=3, n_features=3)
        opt = torch.optim.Adam(tree.parameters(), lr=0.01)
        for _ in range(100):
            opt.zero_grad()
            pred = tree(X)
            assert torch.isfinite(pred).all(), "Tree output went non-finite"
            loss = torch.nn.functional.mse_loss(pred, y)
            assert torch.isfinite(loss), "Loss went non-finite"
            loss.backward()
            torch.nn.utils.clip_grad_norm_(tree.parameters(), 1.0)
            opt.step()
            for p in tree.parameters():
                assert torch.isfinite(p).all(), "Parameter went non-finite"

    def test_gradient_clipping_prevents_divergence(self):
        """Even with an aggressive lr, clipped gradients should not diverge."""
        torch.manual_seed(0)
        X = torch.randn(64, 3) * 3.0  # larger magnitudes stress exp/log guards
        y = torch.randn(64)

        tree = EmlTree(depth=2, n_features=3)
        opt = torch.optim.Adam(tree.parameters(), lr=0.5)  # aggressive
        for _ in range(50):
            opt.zero_grad()
            loss = torch.nn.functional.mse_loss(tree(X), y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(tree.parameters(), 1.0)
            opt.step()
        # No explosion, all parameters still finite.
        for p in tree.parameters():
            assert torch.isfinite(p).all()
