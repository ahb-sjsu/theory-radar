"""Feature projection methods for Theory Radar.

Provides PCA, Tucker, kernel, and neural projections that allow
formulas to access all original features through learned projections.
"""

from __future__ import annotations

import logging
import numpy as np
from numpy.typing import NDArray

log = logging.getLogger(__name__)


class Projection:
    """Base class for feature projections."""

    def fit_transform(self, X: NDArray) -> NDArray:
        raise NotImplementedError

    def transform(self, X: NDArray) -> NDArray:
        raise NotImplementedError

    @property
    def names(self) -> list[str]:
        raise NotImplementedError


class PCAProjection(Projection):
    """Linear PCA projection. Each component is a linear combination
    of all original features."""

    def __init__(self, n_components: int = 8):
        self.n_components = n_components
        self._pca = None

    def fit_transform(self, X: NDArray) -> NDArray:
        from sklearn.decomposition import PCA
        k = min(self.n_components, X.shape[1], X.shape[0] - 1)
        self._pca = PCA(n_components=k)
        return self._pca.fit_transform(X)

    def transform(self, X: NDArray) -> NDArray:
        return self._pca.transform(X)

    @property
    def names(self) -> list[str]:
        return [f"pc{i}" for i in range(self._pca.n_components_)]


class TuckerProjection(Projection):
    """Tucker decomposition of feature interaction tensor.
    Captures pairwise feature interactions via HOSVD."""

    def __init__(self, n_components: int = 8):
        self.n_components = n_components
        self._model = None

    def fit_transform(self, X: NDArray) -> NDArray:
        from sklearn.decomposition import TruncatedSVD
        N, d = X.shape

        if d <= 50:
            X_interact = np.zeros((N, d * d), dtype=np.float64)
            for j in range(d):
                X_interact[:, j * d:(j + 1) * d] = X[:, j:j + 1] * X
        else:
            rng = np.random.RandomState(42)
            n_proj = min(self.n_components * 20, 500)
            pairs = rng.choice(d * d, n_proj, replace=False)
            X_interact = np.zeros((N, n_proj), dtype=np.float64)
            for idx, pair in enumerate(pairs):
                j, k = divmod(pair, d)
                X_interact[:, idx] = X[:, j] * X[:, k]
            self._pairs = pairs
            self._orig_d = d

        self._mean = X_interact.mean(axis=0)
        X_interact -= self._mean
        k = min(self.n_components, X_interact.shape[1] - 1, N - 1)
        self._svd = TruncatedSVD(n_components=k, random_state=42)
        self._d = d
        self._model = "direct" if d <= 50 else "projected"
        return self._svd.fit_transform(X_interact)

    def transform(self, X: NDArray) -> NDArray:
        N, d = X.shape
        if self._model == "direct":
            X_interact = np.zeros((N, d * d), dtype=np.float64)
            for j in range(d):
                X_interact[:, j * d:(j + 1) * d] = X[:, j:j + 1] * X
        else:
            X_interact = np.zeros((N, len(self._pairs)), dtype=np.float64)
            for idx, pair in enumerate(self._pairs):
                j, k = divmod(pair, self._orig_d)
                X_interact[:, idx] = X[:, j] * X[:, k]
        X_interact -= self._mean
        return self._svd.transform(X_interact)

    @property
    def names(self) -> list[str]:
        return [f"T{i}" for i in range(self._svd.n_components)]


class KernelProjection(Projection):
    """Kernel PCA or Random Fourier Features for nonlinear projections."""

    def __init__(self, n_components: int = 8, method: str = "rff",
                 gamma: float | None = None):
        self.n_components = n_components
        self.method = method
        self.gamma = gamma
        self._model = None

    def fit_transform(self, X: NDArray) -> NDArray:
        if self.method == "kpca":
            from sklearn.decomposition import KernelPCA
            g = self.gamma or 1.0 / X.shape[1]
            self._model = KernelPCA(
                n_components=self.n_components, kernel="rbf",
                gamma=g, random_state=42)
            return self._model.fit_transform(X)
        else:
            from sklearn.kernel_approximation import RBFSampler
            g = self.gamma or 1.0 / X.shape[1]
            self._model = RBFSampler(
                n_components=self.n_components, gamma=g, random_state=42)
            return self._model.fit_transform(X)

    def transform(self, X: NDArray) -> NDArray:
        return self._model.transform(X)

    @property
    def names(self) -> list[str]:
        prefix = "K" if self.method == "kpca" else "R"
        return [f"{prefix}{i}" for i in range(self.n_components)]


class NeuralProjection(Projection):
    """Learned nonlinear projection via a small neural network.
    Trains a 1-hidden-layer net, then uses hidden activations as features."""

    def __init__(self, n_components: int = 8, epochs: int = 50,
                 lr: float = 0.01):
        self.n_components = n_components
        self.epochs = epochs
        self.lr = lr
        self._W1 = None
        self._b1 = None

    def fit_transform(self, X: NDArray) -> NDArray:
        N, d = X.shape
        y_placeholder = np.zeros(N)  # will be set externally
        self._y = y_placeholder

        rng = np.random.RandomState(42)
        self._W1 = rng.randn(d, self.n_components) * np.sqrt(2.0 / d)
        self._b1 = np.zeros(self.n_components)
        W2 = rng.randn(self.n_components, 1) * 0.1
        b2 = np.zeros(1)

        for epoch in range(self.epochs):
            # Forward
            h = np.maximum(0, X @ self._W1 + self._b1)  # ReLU
            logits = (h @ W2 + b2).ravel()
            probs = 1.0 / (1.0 + np.exp(-np.clip(logits, -500, 500)))

            # Binary cross-entropy gradient
            err = probs - self._y
            dW2 = h.T @ err.reshape(-1, 1) / N
            db2 = err.mean()
            dh = err.reshape(-1, 1) @ W2.T
            dh[h <= 0] = 0  # ReLU gradient
            dW1 = X.T @ dh / N
            db1 = dh.mean(axis=0)

            W2 -= self.lr * dW2
            b2 -= self.lr * db2
            self._W1 -= self.lr * dW1
            self._b1 -= self.lr * db1

        return np.maximum(0, X @ self._W1 + self._b1)

    def set_labels(self, y: NDArray):
        """Set training labels (must be called before fit_transform)."""
        self._y = y.astype(np.float64)

    def transform(self, X: NDArray) -> NDArray:
        return np.maximum(0, X @ self._W1 + self._b1)

    @property
    def names(self) -> list[str]:
        return [f"N{i}" for i in range(self.n_components)]


class PLSProjection(Projection):
    """Partial Least Squares projection (supervised).
    Finds directions that maximize covariance with the target,
    not just variance. More discriminative than PCA for classification."""

    def __init__(self, n_components: int = 8):
        self.n_components = n_components
        self._pls = None

    def fit_transform(self, X: NDArray) -> NDArray:
        from sklearn.cross_decomposition import PLSRegression
        k = min(self.n_components, X.shape[1], X.shape[0] - 1)
        self._pls = PLSRegression(n_components=k)
        self._pls.fit(X, self._y)
        return self._pls.transform(X)

    def set_labels(self, y: NDArray):
        self._y = y.astype(np.float64)

    def transform(self, X: NDArray) -> NDArray:
        return self._pls.transform(X)

    @property
    def names(self) -> list[str]:
        return [f"pls{i}" for i in range(self._pls.x_loadings_.shape[1])]


class SparsePCAProjection(Projection):
    """Sparse PCA projection. Components use only a few features each,
    making the projection itself interpretable."""

    def __init__(self, n_components: int = 8):
        self.n_components = n_components
        self._spca = None

    def fit_transform(self, X: NDArray) -> NDArray:
        from sklearn.decomposition import SparsePCA
        k = min(self.n_components, X.shape[1])
        self._spca = SparsePCA(n_components=k, random_state=42, max_iter=20)
        return self._spca.fit_transform(X)

    def transform(self, X: NDArray) -> NDArray:
        return self._spca.transform(X)

    @property
    def names(self) -> list[str]:
        return [f"sp{i}" for i in range(self.n_components)]


PROJECTIONS = {
    "pca": PCAProjection,
    "pls": PLSProjection,
    "tucker": TuckerProjection,
    "kernel": KernelProjection,
    "neural": NeuralProjection,
    "sparse_pca": SparsePCAProjection,
}
