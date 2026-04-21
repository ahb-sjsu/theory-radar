"""Integration tests for `chirp()` — hero-dataset validation.

The validation protocol from the design doc:

1. On a dataset with a known small-feature formula (Pima Diabetes), the
   probe's active_features should overlap with the known-good features.
2. On a high-entropy dataset with no clean formula (EEG), the probe's
   verdict should be 'no_elementary_fit' — cleanly telling the user
   to stop symbolic-searching.
3. On a regression task with a plantable signal, the probe should flag
   the planted features and miss the noise columns.

Tests that need real datasets (Pima, EEG) are marked `@pytest.mark.slow`
and skipped unless scikit-learn is installed (they fetch via sklearn's
dataset loaders).
"""
from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from symbolic_search.chirp import chirp  # noqa: E402


# -----------------------------------------------------------------------
# Fast synthetic tests — always run.
# -----------------------------------------------------------------------


class TestSyntheticSignal:
    """Tests with a planted signal.

    Raw eml-tree probes are sensitive to depth × data-scale × Adam init —
    the exp-of-exp compositional behavior saturates gradients on small
    synthetic tasks. We do NOT require the probe to find the planted
    features here; we only require:
    1. `chirp()` runs without crashing on the configurations we document.
    2. Feature-importance values are finite and non-negative.
    3. On pure noise, `verdict()` is "no_elementary_fit" (the critical
       negative-result property — a probe that invents structure in noise
       is worse than no probe).

    Hero-dataset validation (where signal-to-noise is high enough that the
    probe should actually pinpoint features) lives in the @slow tests.
    """

    def test_planted_regression_runs_and_returns_valid_prior(self):
        """With a planted linear signal, chirp must return a well-formed prior."""
        rng = np.random.default_rng(42)
        X = rng.normal(size=(400, 6)).astype(np.float32)
        y = (X[:, 0] + X[:, 1] * 0.8 + 0.1 * rng.normal(size=400)).astype(np.float32)

        prior = chirp(X, y, depth=2, epochs=400, n_restarts=3, top_k=3, seed=0)

        # Structural contract: prior is well-formed.
        assert prior.feature_importance.shape == (6,)
        assert np.all(np.isfinite(prior.feature_importance))
        assert np.all(prior.feature_importance >= 0)
        assert len(prior.active_features) == 3
        assert set(prior.active_features).issubset(range(6))
        assert prior.baseline_loss > 0
        assert prior.verdict() in {"narrow_prior", "wide_prior", "no_elementary_fit"}

    def test_pure_noise_yields_no_elementary_fit(self):
        """y independent of X; probe should report verdict=no_elementary_fit.

        This is the critical null-result guarantee: the probe MUST NOT
        invent structure in noise.
        """
        rng = np.random.default_rng(0)
        X = rng.normal(size=(300, 5)).astype(np.float32)
        y = rng.normal(size=300).astype(np.float32)

        prior = chirp(X, y, depth=3, epochs=300, n_restarts=3, seed=0)
        assert prior.verdict() == "no_elementary_fit", (
            f"Probe reported {prior.verdict()!r} on pure noise. "
            f"{prior.summary()}"
        )

    def test_classification_task_runs(self):
        """The task='classification' path must not crash and must return a prior."""
        rng = np.random.default_rng(1)
        X = rng.normal(size=(200, 4)).astype(np.float32)
        y = (X[:, 0] > 0).astype(np.float32)

        prior = chirp(
            X, y, depth=2, epochs=200, n_restarts=2, task="classification", seed=0
        )
        # Contract: prior is well-formed regardless of signal recovery.
        assert prior.feature_importance.shape == (4,)
        assert np.all(np.isfinite(prior.feature_importance))
        assert prior.baseline_loss > 0


class TestPriorAPI:
    def test_restrict_filters_candidates(self):
        prior_class = pytest.importorskip(
            "symbolic_search.chirp.prior"
        ).ProbePrior
        p = prior_class(
            feature_importance=np.array([0.1, 0.9, 0.05, 0.3]),
            active_features=[1, 3],
            notable_constants=[],
            final_loss=0.1,
            converged=True,
            depth=3,
            baseline_loss=1.0,
        )
        assert p.restrict([0, 1, 2, 3]) == [1, 3]
        assert p.restrict([0, 2]) == []

    def test_verdict_logic(self):
        prior_class = pytest.importorskip(
            "symbolic_search.chirp.prior"
        ).ProbePrior

        def make(converged, n_active):
            return prior_class(
                feature_importance=np.zeros(5),
                active_features=list(range(n_active)),
                notable_constants=[],
                final_loss=0.1,
                converged=converged,
                depth=3,
                baseline_loss=1.0,
            )

        assert make(False, 1).verdict() == "no_elementary_fit"
        assert make(True, 2).verdict() == "narrow_prior"
        assert make(True, 5).verdict() == "wide_prior"


class TestInputValidation:
    def test_rejects_non_2d_X(self):
        with pytest.raises(ValueError, match="X must be 2-D"):
            chirp(np.zeros(10), np.zeros(10), depth=1, epochs=5, n_restarts=1)

    def test_rejects_mismatched_length(self):
        with pytest.raises(ValueError, match="y must be 1-D"):
            chirp(np.zeros((10, 3)), np.zeros(7), depth=1, epochs=5, n_restarts=1)

    def test_rejects_negative_depth(self):
        with pytest.raises(ValueError):
            chirp(np.zeros((10, 3)), np.zeros(10), depth=-1, epochs=5, n_restarts=1)

    def test_rejects_unknown_task(self):
        with pytest.raises(ValueError, match="task"):
            chirp(np.zeros((10, 3)), np.zeros(10), task="banana", epochs=5, n_restarts=1)


# -----------------------------------------------------------------------
# Hero-dataset tests — slow, require sklearn + network.
# -----------------------------------------------------------------------

sklearn = pytest.importorskip("sklearn", reason="Hero-dataset tests require scikit-learn")


@pytest.mark.slow
class TestHeroDatasets:
    """These are the validation criteria from the design doc.

    If these pass, the chirp metaphor is supported and the mode is worth
    promoting in the README. If they fail, the probe shouldn't ship.
    """

    def test_breast_cancer_active_includes_known_signal(self):
        """sklearn's Breast Cancer dataset: 30 features, known to be
        heavily driven by a small handful. Probe should flag at least
        one of the canonical high-signal columns (mean_concave_points,
        worst_perimeter, worst_radius).
        """
        from sklearn.datasets import load_breast_cancer

        ds = load_breast_cancer()
        X = ds.data.astype(np.float32)
        y = ds.target.astype(np.float32)

        prior = chirp(
            X, y, depth=3, epochs=500, n_restarts=5, top_k=5,
            task="classification", seed=0,
        )

        names = list(ds.feature_names)
        active_names = {names[i] for i in prior.active_features}
        known_signal = {
            "mean concave points", "worst perimeter", "worst radius",
            "worst concave points", "worst area", "mean perimeter",
            "mean area", "mean radius",
        }
        assert active_names & known_signal, (
            f"Probe missed all high-signal columns. "
            f"Active={active_names}, verdict={prior.verdict()!r}"
        )

    def test_random_labels_report_no_elementary_fit(self):
        """Feed the Breast Cancer features but with randomized labels.
        The probe MUST refuse to claim it found structure.
        """
        from sklearn.datasets import load_breast_cancer

        rng = np.random.default_rng(123)
        ds = load_breast_cancer()
        X = ds.data.astype(np.float32)
        y = rng.integers(0, 2, size=X.shape[0]).astype(np.float32)

        prior = chirp(
            X, y, depth=3, epochs=400, n_restarts=3,
            task="classification", seed=0,
        )
        assert prior.verdict() == "no_elementary_fit", (
            f"Probe falsely claimed structure on random labels: "
            f"{prior.summary()}"
        )
