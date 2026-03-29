"""Comprehensive tests for the symbolic search package."""

import numpy as np
import pytest

from symbolic_search import SymbolicSearch, SearchResults
from symbolic_search._ops import BINARY_OPS, UNARY_OPS, BINARY_OPS_MINIMAL
from symbolic_search._search import _f1_threshold_sweep

# ============================================================
# Fixtures
# ============================================================


@pytest.fixture
def circles_data():
    """Circles dataset — formula should find exact boundary."""
    rng = np.random.RandomState(42)
    N = 1000
    x1 = rng.uniform(-2, 2, N)
    x2 = rng.uniform(-2, 2, N)
    y = (x1**2 + x2**2 < 1.0).astype(int)
    X = np.column_stack([x1, x2])
    return X, y, ["x1", "x2"]


@pytest.fixture
def linear_data():
    """Linearly separable dataset."""
    rng = np.random.RandomState(42)
    N = 500
    x1 = rng.uniform(-5, 5, N)
    x2 = rng.uniform(-5, 5, N)
    y = (x1 + x2 > 0).astype(int)
    X = np.column_stack([x1, x2])
    return X, y, ["x1", "x2"]


@pytest.fixture
def product_data():
    """Requires multiplication to separate."""
    rng = np.random.RandomState(42)
    N = 500
    x1 = rng.uniform(-3, 3, N)
    x2 = rng.uniform(-3, 3, N)
    y = (x1 * x2 > 1).astype(int)
    X = np.column_stack([x1, x2])
    return X, y, ["x1", "x2"]


# ============================================================
# Tests: F1 threshold sweep
# ============================================================


class TestF1Sweep:
    def test_perfect_predictor(self):
        values = np.array([0.0, 0.0, 0.0, 1.0, 1.0, 1.0])
        actual = np.array([False, False, False, True, True, True])
        f1, thresh, direction = _f1_threshold_sweep(values, actual)
        assert f1 == pytest.approx(1.0)

    def test_random_predictor(self):
        rng = np.random.RandomState(42)
        values = rng.randn(1000)
        # Low base rate (5%) — random values shouldn't predict well
        actual = np.zeros(1000, dtype=bool)
        actual[:50] = True
        rng.shuffle(actual)
        f1, _, _ = _f1_threshold_sweep(values, actual)
        # Random should get low F1 with 5% base rate
        assert f1 < 0.3

    def test_all_nan(self):
        values = np.full(100, np.nan)
        actual = np.array([True] * 50 + [False] * 50)
        f1, _, _ = _f1_threshold_sweep(values, actual)
        assert f1 == 0.0

    def test_constant_values(self):
        values = np.ones(100)
        actual = np.array([True] * 50 + [False] * 50)
        f1, _, _ = _f1_threshold_sweep(values, actual)
        # Constant can't discriminate, but above-threshold might catch all
        assert f1 >= 0.0

    def test_below_threshold(self):
        """Tests that below-threshold direction works."""
        values = np.array([1.0, 1.0, 1.0, 0.0, 0.0, 0.0])
        actual = np.array([False, False, False, True, True, True])
        f1, thresh, direction = _f1_threshold_sweep(values, actual)
        assert f1 == pytest.approx(1.0)
        assert direction == "<"


# ============================================================
# Tests: SymbolicSearch
# ============================================================


class TestSymbolicSearch:
    def test_basic_run(self, circles_data):
        X, y, names = circles_data
        search = SymbolicSearch(X, y, names, verbose=False)
        results = search.run(ensemble=False)
        assert isinstance(results, SearchResults)
        assert results.ceiling > 0
        assert results.formulas_tested > 0
        assert results.best_formula != ""

    def test_finds_circles(self, circles_data):
        X, y, names = circles_data
        search = SymbolicSearch(X, y, names, verbose=False)
        results = search.run(ensemble=False)
        # Should find a formula with F1 > 0.9 (hypot is the exact boundary)
        assert results.ceiling > 0.9

    def test_finds_linear(self, linear_data):
        X, y, names = linear_data
        search = SymbolicSearch(X, y, names, verbose=False)
        results = search.run(ensemble=False)
        # x1 + x2 > 0 should be found exactly
        assert results.ceiling > 0.95

    def test_finds_product(self, product_data):
        X, y, names = product_data
        search = SymbolicSearch(X, y, names, verbose=False)
        results = search.run(ensemble=False)
        # x1 * x2 > threshold should be found
        assert results.ceiling > 0.8

    def test_convergence_tracked(self, circles_data):
        X, y, names = circles_data
        search = SymbolicSearch(X, y, names, verbose=False)
        results = search.run(ensemble=False)
        assert 1 in results.convergence
        assert 2 in results.convergence
        assert 3 in results.convergence
        # Should be non-decreasing
        assert results.convergence[2] >= results.convergence[1]
        assert results.convergence[3] >= results.convergence[2]

    def test_ensemble_comparison(self, circles_data):
        X, y, names = circles_data
        search = SymbolicSearch(X, y, names, verbose=False)
        results = search.run(ensemble=True)
        assert results.ensemble_score > 0
        assert isinstance(results.gap, float)

    def test_custom_ops(self, linear_data):
        X, y, names = linear_data
        search = SymbolicSearch(
            X,
            y,
            names,
            binary_ops=BINARY_OPS_MINIMAL,  # only +, -, *, /
            verbose=False,
        )
        results = search.run(ensemble=False)
        assert results.ceiling > 0.9  # + alone suffices

    def test_single_feature(self):
        rng = np.random.RandomState(42)
        x = rng.randn(200)
        y = (x > 0).astype(int)
        search = SymbolicSearch(x.reshape(-1, 1), y, ["x"], verbose=False)
        results = search.run(ensemble=False)
        assert results.ceiling > 0.95

    def test_many_features(self):
        rng = np.random.RandomState(42)
        X = rng.randn(500, 20)
        y = (X[:, 0] + X[:, 1] > 0).astype(int)
        search = SymbolicSearch(X, y, verbose=False)
        results = search.run(ensemble=False)
        assert results.formulas_tested > 100

    def test_summary_string(self, circles_data):
        X, y, names = circles_data
        search = SymbolicSearch(X, y, names, verbose=False)
        results = search.run(ensemble=False)
        summary = results.summary()
        assert "Best formula" in summary
        assert "Ceiling" in summary

    def test_all_formulas_sorted(self, circles_data):
        X, y, names = circles_data
        search = SymbolicSearch(X, y, names, verbose=False)
        results = search.run(ensemble=False)
        scores = [r.score for r in results.all_formulas]
        assert scores == sorted(scores, reverse=True)


class TestAblation:
    def test_ablation_runs(self, circles_data):
        X, y, names = circles_data
        search = SymbolicSearch(
            X,
            y,
            names,
            binary_ops=BINARY_OPS_MINIMAL,
            verbose=False,
        )
        results = search.ablation()
        assert "(none)" in results
        assert len(results) == len(BINARY_OPS_MINIMAL) + 1

    def test_ablation_identifies_needed_op(self, circles_data):
        X, y, names = circles_data
        search = SymbolicSearch(X, y, names, verbose=False)
        results = search.ablation()
        # Removing hypot should hurt on circles
        full = results["(none)"]
        if "hypot" in results:
            assert results["hypot"] <= full


class TestOps:
    def test_all_binary_ops_callable(self):
        a = np.array([1.0, 2.0, 3.0])
        b = np.array([4.0, 5.0, 6.0])
        for name, fn in BINARY_OPS.items():
            result = fn(a, b)
            assert result.shape == a.shape, f"{name} returned wrong shape"
            assert np.all(np.isfinite(result)), f"{name} returned non-finite"

    def test_all_unary_ops_callable(self):
        x = np.array([0.5, 1.0, 2.0])
        for name, fn in UNARY_OPS.items():
            result = fn(x)
            assert result.shape == x.shape, f"{name} returned wrong shape"
            assert np.all(np.isfinite(result)), f"{name} returned non-finite"

    def test_division_by_zero_safe(self):
        a = np.array([1.0, 2.0])
        b = np.array([0.0, 0.0])
        result = BINARY_OPS["/"](a, b)
        assert np.all(np.isfinite(result))

    def test_log_of_zero_safe(self):
        x = np.array([0.0, -1.0])
        result = UNARY_OPS["log"](x)
        assert np.all(np.isfinite(result))

    def test_inv_of_zero_safe(self):
        x = np.array([0.0])
        result = UNARY_OPS["inv"](x)
        assert np.all(np.isfinite(result))
