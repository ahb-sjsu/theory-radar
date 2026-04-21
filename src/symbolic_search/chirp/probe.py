"""The `chirp()` entry point — train an eml-tree, extract a ProbePrior.

Workflow (RF radar metaphor):

    (X, y)  --->  eml-tree + Adam (broadband probe)
                        |
                        v
            gradient norm per feature  (spectrum)
            leaf constants             (spectral peaks)
            final loss vs baseline     (SNR check)
                        |
                        v
                   ProbePrior  ---> TheoryRadar.search(prior=...)
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from .eml_tree import EPS, EmlTree
from .prior import ProbePrior

# Natural-constant reference table — order matters (earliest match wins).
_REFERENCES: tuple[tuple[str, float], ...] = (
    ("0", 0.0),
    ("1", 1.0),
    ("-1", -1.0),
    ("1/2", 0.5),
    ("2", 2.0),
    ("-1/2", -0.5),
    ("e", float(np.e)),
    ("π", float(np.pi)),
    ("π/2", float(np.pi / 2)),
    ("ln(2)", float(np.log(2))),
    ("1/e", float(1.0 / np.e)),
)
_CONST_TOL = 0.03
_CONVERGED_RATIO = 0.5


def chirp(
    X: np.ndarray,
    y: np.ndarray,
    *,
    depth: int = 3,
    epochs: int = 500,
    lr: float = 0.01,
    top_k: int = 5,
    n_restarts: int = 3,
    task: str = "regression",
    device: str = "cpu",
    seed: int = 0,
    verbose: bool = False,
) -> ProbePrior:
    """Fit an eml-tree of depth `depth` to (X, y) and extract a ProbePrior.

    Args:
        X: (N, D) feature matrix.
        y: (N,) target. Regression (continuous) or classification (0/1).
        depth: Tree depth. The Odrzywolek paper reports exact recovery of
            many closed-form targets at depth <= 4. Start at 3.
        epochs: Adam steps per restart.
        lr: Adam learning rate.
        top_k: Number of features kept in `active_features`.
        n_restarts: Random inits to try; best by loss wins. Adam is very
            init-sensitive on eml-trees, so >= 3 restarts is recommended.
        task: "regression" (MSE) or "classification" (BCE with sigmoid wrap).
        device: PyTorch device string.
        seed: Base random seed. Restart `k` uses `seed + k`.
        verbose: If True, print per-restart final loss.

    Returns:
        A `ProbePrior` instance. Call `.verdict()` for the top-level summary,
        `.restrict(candidates)` to filter a feature list, or read
        `.feature_importance` / `.active_features` directly.
    """
    if task not in ("regression", "classification"):
        raise ValueError(f"task must be 'regression' or 'classification', got {task!r}")
    if X.ndim != 2:
        raise ValueError(f"X must be 2-D, got shape {X.shape}")
    if y.ndim != 1 or y.shape[0] != X.shape[0]:
        raise ValueError(f"y must be 1-D with length N; got {y.shape} vs N={X.shape[0]}")
    if depth < 0:
        raise ValueError(f"depth must be >= 0, got {depth}")

    X_t = torch.as_tensor(X, dtype=torch.float32, device=device)
    y_t = torch.as_tensor(y, dtype=torch.float32, device=device)

    if task == "regression":
        y_mean = y_t.mean()
        y_std = y_t.std().clamp(min=EPS)
        y_target = (y_t - y_mean) / y_std
        baseline_loss = float(F.mse_loss(torch.zeros_like(y_target), y_target).item())
        loss_fn = lambda pred: F.mse_loss(pred, y_target)  # noqa: E731
    else:  # classification
        y_target = y_t
        # Baseline: predict the majority-class mean (constant probability = mean(y)).
        p_mean = y_target.mean().clamp(min=EPS, max=1.0 - EPS)
        const_logit = torch.log(p_mean / (1.0 - p_mean)).expand_as(y_target)
        baseline_loss = float(F.binary_cross_entropy_with_logits(const_logit, y_target).item())
        loss_fn = lambda pred: F.binary_cross_entropy_with_logits(pred, y_target)  # noqa: E731

    best_tree: EmlTree | None = None
    best_loss = float("inf")

    for restart in range(n_restarts):
        torch.manual_seed(seed + restart)
        tree = EmlTree(depth=depth, n_features=X.shape[1]).to(device)
        opt = torch.optim.Adam(tree.parameters(), lr=lr)

        for step in range(epochs):
            opt.zero_grad()
            pred = tree(X_t)
            loss = loss_fn(pred)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(tree.parameters(), max_norm=1.0)
            opt.step()

        final = float(loss.item())
        if verbose:
            print(f"  restart {restart}: final loss {final:.4f}")
        if final < best_loss:
            best_loss = final
            best_tree = tree

    assert best_tree is not None  # guaranteed by n_restarts >= 1

    # -------------------------------------------------------------------
    # Feature importance: average |gradient| of output w.r.t. each X column.
    # -------------------------------------------------------------------
    X_grad = X_t.clone().detach().requires_grad_(True)
    pred_sum = best_tree(X_grad).sum()
    grads = torch.autograd.grad(pred_sum, X_grad, retain_graph=False)[0]
    importance = grads.abs().mean(dim=0).detach().cpu().numpy()

    order = np.argsort(importance)[::-1]
    k = min(top_k, X.shape[1])
    active = order[:k].tolist()

    # -------------------------------------------------------------------
    # Notable constants: leaf scalars that landed near natural values.
    # -------------------------------------------------------------------
    notable: list[tuple[int, float, str]] = []
    for i, leaf in enumerate(best_tree.leaves()):
        c = float(leaf.const.item())
        for name, target in _REFERENCES:
            if abs(c - target) < _CONST_TOL:
                notable.append((i, c, name))
                break

    converged = best_loss < _CONVERGED_RATIO * baseline_loss

    return ProbePrior(
        feature_importance=importance,
        active_features=active,
        notable_constants=notable,
        final_loss=best_loss,
        converged=converged,
        source_tree=best_tree,
        depth=depth,
        baseline_loss=baseline_loss,
    )
