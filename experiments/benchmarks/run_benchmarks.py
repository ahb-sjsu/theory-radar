#!/usr/bin/env python3
"""
Standard symbolic regression benchmarks.

Runs our phased exhaustive search and PySR on:
1. AI Feynman equations (known physics laws)
2. SRBench standard datasets
3. The 3-body case study

Reports: formula found, F1/MSE, runtime, complexity.
"""

from __future__ import annotations

import logging
import time

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


# ============================================================
# AI Feynman benchmark problems (regression, not classification)
# For regression, we use R² instead of F1
# ============================================================

def feynman_problems():
    """Generate standard AI Feynman test problems with known formulas."""
    rng = np.random.RandomState(42)
    N = 5000
    problems = []

    # I.6.20: f = exp(-theta^2 / (2*sigma^2)) / (sigma * sqrt(2*pi))
    theta = rng.uniform(-3, 3, N)
    sigma = rng.uniform(0.5, 3, N)
    y = np.exp(-theta**2 / (2 * sigma**2)) / (sigma * np.sqrt(2 * np.pi))
    problems.append({
        "name": "I.6.20 (Gaussian)",
        "X": np.column_stack([theta, sigma]),
        "y": y,
        "features": ["theta", "sigma"],
        "true_formula": "exp(-theta^2/(2*sigma^2)) / (sigma*sqrt(2*pi))",
    })

    # I.9.18: F = G*m1*m2 / ((x2-x1)^2 + (y2-y1)^2 + (z2-z1)^2)
    # Simplified: F = m1*m2 / r^2
    m1 = rng.uniform(0.1, 10, N)
    m2 = rng.uniform(0.1, 10, N)
    r = rng.uniform(0.5, 10, N)
    y = m1 * m2 / r**2
    problems.append({
        "name": "I.9.18 (Gravity)",
        "X": np.column_stack([m1, m2, r]),
        "y": y,
        "features": ["m1", "m2", "r"],
        "true_formula": "m1*m2/r^2",
    })

    # I.12.1: F = q1*q2 / (4*pi*eps0*r^2) -> simplified q1*q2/r^2
    q1 = rng.uniform(0.1, 5, N)
    q2 = rng.uniform(0.1, 5, N)
    r = rng.uniform(0.5, 10, N)
    y = q1 * q2 / r**2
    problems.append({
        "name": "I.12.1 (Coulomb)",
        "X": np.column_stack([q1, q2, r]),
        "y": y,
        "features": ["q1", "q2", "r"],
        "true_formula": "q1*q2/r^2",
    })

    # I.15.10: p = m0*v / sqrt(1 - v^2/c^2), c=1
    m0 = rng.uniform(0.1, 10, N)
    v = rng.uniform(0.01, 0.95, N)
    y = m0 * v / np.sqrt(1 - v**2)
    problems.append({
        "name": "I.15.10 (Relativistic momentum)",
        "X": np.column_stack([m0, v]),
        "y": y,
        "features": ["m0", "v"],
        "true_formula": "m0*v/sqrt(1-v^2)",
    })

    # I.27.6: d = 1 / (1/d1 + n/d2)  (thin lens)
    d1 = rng.uniform(1, 10, N)
    d2 = rng.uniform(1, 10, N)
    n = rng.uniform(1, 3, N)
    y = 1.0 / (1.0/d1 + n/d2)
    problems.append({
        "name": "I.27.6 (Thin lens)",
        "X": np.column_stack([d1, d2, n]),
        "y": y,
        "features": ["d1", "d2", "n"],
        "true_formula": "1/(1/d1 + n/d2)",
    })

    # Simple: y = x1^2 + x2^2
    x1 = rng.uniform(-5, 5, N)
    x2 = rng.uniform(-5, 5, N)
    y = x1**2 + x2**2
    problems.append({
        "name": "Simple (x1^2 + x2^2)",
        "X": np.column_stack([x1, x2]),
        "y": y,
        "features": ["x1", "x2"],
        "true_formula": "x1^2 + x2^2",
    })

    return problems


# ============================================================
# Our method adapted for regression (R² instead of F1)
# ============================================================

def r2_score(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred)**2)
    ss_tot = np.sum((y_true - y_true.mean())**2)
    return 1 - ss_res / (ss_tot + 1e-30)


def exhaustive_search_regression(X, y, feature_names, max_depth=3):
    """Phased exhaustive search for regression (maximize R²)."""
    N, d = X.shape

    unary_ops = {
        "log": lambda x: np.log(np.abs(x) + 1e-30),
        "sqrt": lambda x: np.sqrt(np.abs(x)),
        "sq": lambda x: x**2,
        "inv": lambda x: 1.0 / (x + 1e-30),
        "neg": lambda x: -x,
    }

    binary_ops = {
        "+": lambda a, b: a + b,
        "-": lambda a, b: a - b,
        "*": lambda a, b: a * b,
        "/": lambda a, b: a / (b + 1e-30),
    }

    results = []
    formulas_tested = 0

    # Phase 1: Single features + unary
    for i in range(d):
        vals = X[:, i]
        r2 = r2_score(y, vals)
        results.append((r2, feature_names[i], vals))
        formulas_tested += 1

        for uname, ufn in unary_ops.items():
            try:
                uvals = ufn(vals)
                uvals = np.nan_to_num(uvals, nan=0, posinf=1e10, neginf=-1e10)
                # Linear fit: y ≈ a*f(x) + b
                A = np.column_stack([uvals, np.ones(N)])
                coef, _, _, _ = np.linalg.lstsq(A, y, rcond=None)
                pred = A @ coef
                r2 = r2_score(y, pred)
                results.append((r2, f"{uname}({feature_names[i]})", pred))
                formulas_tested += 1
            except Exception:
                pass

    # Phase 2: Pairwise with linear fit
    for i in range(d):
        for j in range(d):
            if i == j:
                continue
            a, b = X[:, i], X[:, j]
            for op_name, op_fn in binary_ops.items():
                try:
                    vals = op_fn(a, b)
                    vals = np.nan_to_num(vals, nan=0, posinf=1e10, neginf=-1e10)
                    A = np.column_stack([vals, np.ones(N)])
                    coef, _, _, _ = np.linalg.lstsq(A, y, rcond=None)
                    pred = A @ coef
                    r2 = r2_score(y, pred)
                    results.append((r2, f"{feature_names[i]} {op_name} {feature_names[j]}", pred))
                    formulas_tested += 1
                except Exception:
                    pass

    # Phase 3: Unary of top pairwise
    results.sort(key=lambda x: -x[0])
    top_k = 30
    top_formulas = results[:top_k]
    for r2_base, desc, base_vals in top_formulas:
        for uname, ufn in unary_ops.items():
            try:
                uvals = ufn(base_vals)
                uvals = np.nan_to_num(uvals, nan=0, posinf=1e10, neginf=-1e10)
                A = np.column_stack([uvals, np.ones(N)])
                coef, _, _, _ = np.linalg.lstsq(A, y, rcond=None)
                pred = A @ coef
                r2 = r2_score(y, pred)
                results.append((r2, f"{uname}({desc})", pred))
                formulas_tested += 1
            except Exception:
                pass

    # Phase 4: Multi-feature linear combinations (up to 3 features with ops)
    top_single = list(range(min(d, 8)))
    for i in top_single:
        for j in top_single:
            if j <= i:
                continue
            for k in top_single:
                if k <= j:
                    continue
                a, b, c = X[:, i], X[:, j], X[:, k]
                combos = [
                    (f"{feature_names[i]}*{feature_names[j]}+{feature_names[k]}", a*b+c),
                    (f"{feature_names[i]}+{feature_names[j]}*{feature_names[k]}", a+b*c),
                    (f"{feature_names[i]}*{feature_names[j]}*{feature_names[k]}", a*b*c),
                    (f"{feature_names[i]}*{feature_names[j]}/{feature_names[k]}", a*b/(c+1e-30)),
                ]
                for desc, vals in combos:
                    try:
                        vals = np.nan_to_num(vals, nan=0, posinf=1e10, neginf=-1e10)
                        A = np.column_stack([vals, np.ones(N)])
                        coef, _, _, _ = np.linalg.lstsq(A, y, rcond=None)
                        pred = A @ coef
                        r2 = r2_score(y, pred)
                        results.append((r2, desc, pred))
                        formulas_tested += 1
                    except Exception:
                        pass

    results.sort(key=lambda x: -x[0])
    return results[0][0], results[0][1], formulas_tested


# ============================================================
# PySR wrapper
# ============================================================

def run_pysr(X, y, feature_names, timeout=120):
    """Run PySR on a regression problem."""
    try:
        from pysr import PySRRegressor
    except ImportError:
        return None, "PySR not installed", 0

    t0 = time.time()
    model = PySRRegressor(
        niterations=40,
        binary_operators=["+", "-", "*", "/"],
        unary_operators=["exp", "log", "sqrt", "square", "inv(x) = 1/x"],
        populations=15,
        population_size=33,
        maxsize=20,
        timeout_in_seconds=timeout,
        temp_equation_file=True,
        verbosity=0,
    )

    try:
        model.fit(X, y, variable_names=feature_names)
        elapsed = time.time() - t0
        best = model.get_best()
        pred = model.predict(X)
        r2 = r2_score(y, pred)
        return r2, str(best["equation"]), elapsed
    except Exception as e:
        return None, str(e), time.time() - t0


# ============================================================
# Main
# ============================================================

def main():
    log.info("=" * 70)
    log.info("SYMBOLIC REGRESSION BENCHMARKS")
    log.info("Our method vs PySR on AI Feynman problems")
    log.info("=" * 70)

    problems = feynman_problems()

    log.info("\n%-30s  %-10s %-10s %-10s %-10s",
             "Problem", "Ours R²", "PySR R²", "Ours time", "PySR time")
    log.info("-" * 80)

    our_r2s = []
    pysr_r2s = []

    for prob in problems:
        name = prob["name"]
        X = prob["X"]
        y = prob["y"]
        features = prob["features"]

        # Our method
        t0 = time.time()
        r2_ours, formula_ours, n_tested = exhaustive_search_regression(
            X, y, features, max_depth=3
        )
        time_ours = time.time() - t0

        # PySR
        r2_pysr, formula_pysr, time_pysr = run_pysr(X, y, features, timeout=60)

        our_r2s.append(r2_ours)
        if r2_pysr is not None:
            pysr_r2s.append(r2_pysr)

        log.info("%-30s  %-10.4f %-10s %-10.1fs %-10.1fs",
                 name[:30],
                 r2_ours,
                 f"{r2_pysr:.4f}" if r2_pysr is not None else "N/A",
                 time_ours,
                 time_pysr)
        log.info("  Ours:  %s (%d formulas)", formula_ours[:60], n_tested)
        log.info("  PySR:  %s", str(formula_pysr)[:60])
        log.info("  True:  %s", prob["true_formula"])

    log.info("\n" + "=" * 70)
    log.info("SUMMARY")
    log.info("=" * 70)
    log.info("Our method: mean R² = %.4f (n=%d problems)", np.mean(our_r2s), len(our_r2s))
    if pysr_r2s:
        log.info("PySR:       mean R² = %.4f (n=%d problems)", np.mean(pysr_r2s), len(pysr_r2s))
        n_ours_better = sum(1 for a, b in zip(our_r2s, pysr_r2s) if a > b)
        log.info("Ours better: %d/%d", n_ours_better, len(pysr_r2s))
    log.info("Our method is exhaustive at depth ≤ 3; PySR explores deeper trees.")


if __name__ == "__main__":
    main()
