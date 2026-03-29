"""Built-in operation registries for symbolic search."""

from __future__ import annotations

import numpy as np

BINARY_OPS = {
    "+": lambda a, b: a + b,
    "-": lambda a, b: a - b,
    "*": lambda a, b: a * b,
    "/": lambda a, b: a / (b + 1e-30),
    "max": lambda a, b: np.maximum(a, b),
    "min": lambda a, b: np.minimum(a, b),
    "hypot": lambda a, b: np.sqrt(a**2 + b**2),
    "diff_sq": lambda a, b: (a - b) ** 2,
    "harmonic": lambda a, b: 2 * a * b / (a + b + 1e-30),
    "geometric": lambda a, b: np.sign(a * b) * np.sqrt(np.abs(a * b)),
}

UNARY_OPS = {
    "log": lambda x: np.log(np.abs(x) + 1e-30),
    "sqrt": lambda x: np.sqrt(np.abs(x)),
    "sq": lambda x: x**2,
    "inv": lambda x: 1.0 / (x + 1e-30),
    "neg": lambda x: -x,
    "abs": lambda x: np.abs(x),
    "sigmoid": lambda x: 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500))),
    "tanh": lambda x: np.tanh(np.clip(x, -500, 500)),
}

BINARY_OPS_MINIMAL = {k: BINARY_OPS[k] for k in ["+", "-", "*", "/"]}
UNARY_OPS_MINIMAL = {k: UNARY_OPS[k] for k in ["log", "sqrt", "sq", "inv", "neg"]}
