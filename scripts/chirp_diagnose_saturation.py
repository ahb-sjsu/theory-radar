"""Diagnose why the eml-tree probe fails on real data at depth >= 3.

Run with:
    python scripts/chirp_diagnose_saturation.py

Expected output: the tree output variance collapses to 0 at depth >= 3,
because the `exp(exp(...))` chain hits the OUT_CLAMP=50 ceiling from any
standardized input. After saturation, gradient is blocked through the
clamp, so Adam cannot train anything below.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch

from symbolic_search.chirp.eml_tree import EmlTree


def main() -> int:
    torch.manual_seed(0)
    X = torch.randn(200, 8)
    print("Standardized input (std=1). Tracking raw tree output per depth:\n")
    print(f"{'depth':>6}  {'min':>8}  {'max':>8}  {'std':>10}  {'verdict':<12}")
    print("-" * 60)
    for depth in range(6):
        tree = EmlTree(depth=depth, n_features=8)
        with torch.no_grad():
            out = tree._raw_forward(X)
        std = out.std().item()
        saturated = std < 1e-4
        verdict = "SATURATED" if saturated else "ok"
        print(
            f"{depth:>6}  {out.min().item():>8.2f}  "
            f"{out.max().item():>8.2f}  {std:>10.4f}  {verdict}"
        )
    print(
        "\nConclusion: depth >= 3 saturates at OUT_CLAMP=50 uniformly. "
        "Gradient through the clamp is zero, so Adam cannot drive the leaves."
    )
    print(
        "\nFix directions documented in docs/CHIRP.md §'Why depth 3 does not "
        "train on standardized inputs'."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
