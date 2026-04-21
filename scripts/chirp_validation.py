"""Manual validation of the chirp probe on Pima and EEG.

Decision criteria:
  PIMA: probe's active_features must include at least one of
        {glucose (index 1), insulin (index 4), age (index 7)}.
        Known-good formula from README is `min(insulin, age) + glucose`.
  EEG:  verdict() must be 'no_elementary_fit' — ensembles win on this
        dataset per the README, and the probe should say so cleanly.

Run with:
    python scripts/chirp_validation.py
"""
from __future__ import annotations

import sys
import time

import numpy as np

# Allow running before install.
from pathlib import Path
_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src"))
sys.path.insert(0, str(_ROOT))  # for dataset_loader.py at repo root

from symbolic_search.chirp import chirp  # noqa: E402


PIMA_FEATURES = [
    "Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
    "Insulin", "BMI", "DiabetesPedigree", "Age",
]
PIMA_KNOWN_SIGNAL = {1, 4, 7}  # Glucose, Insulin, Age


def _print_prior(prior, names):
    print(f"  verdict:        {prior.verdict()!r}")
    print(f"  converged:      {prior.converged}")
    print(f"  final loss:     {prior.final_loss:.4f}")
    print(f"  baseline loss:  {prior.baseline_loss:.4f}")
    print(f"  active_features: {prior.active_features}")
    if names:
        print(f"    named:         {[names[i] for i in prior.active_features]}")
    print(f"  importance:     {prior.feature_importance.round(4).tolist()}")
    if prior.notable_constants:
        print("  notable consts:")
        for i, v, name in prior.notable_constants[:5]:
            print(f"    leaf {i}: {v:.4f}  ~  {name}")


def validate_pima():
    print("\n=" * 1 + "=" * 60)
    print("PIMA DIABETES — should flag glucose/insulin/age")
    print("=" * 60)
    from dataset_loader import load_dataset

    X, y, _ = load_dataset("Diabetes")
    print(f"  dataset: N={X.shape[0]}, D={X.shape[1]}")
    t0 = time.perf_counter()
    prior = chirp(
        X, y.astype(np.float32),
        depth=3, epochs=500, n_restarts=5, top_k=5,
        task="classification", seed=0,
    )
    elapsed = time.perf_counter() - t0
    print(f"  elapsed:        {elapsed:.1f}s")
    _print_prior(prior, PIMA_FEATURES)

    hit = set(prior.active_features) & PIMA_KNOWN_SIGNAL
    verdict_ok = len(hit) > 0
    print()
    print(f"  VERDICT: {'PASS' if verdict_ok else 'FAIL'}  "
          f"(known-signal intersection: {sorted(hit)})")
    return verdict_ok


def validate_eeg():
    print("\n=" * 1 + "=" * 60)
    print("EEG — should report 'no_elementary_fit' (ensembles win)")
    print("=" * 60)
    from dataset_loader import load_dataset

    X, y, _ = load_dataset("EEG")
    print(f"  dataset: N={X.shape[0]}, D={X.shape[1]}")
    t0 = time.perf_counter()
    prior = chirp(
        X, y.astype(np.float32),
        depth=3, epochs=500, n_restarts=3, top_k=5,
        task="classification", seed=0,
    )
    elapsed = time.perf_counter() - t0
    print(f"  elapsed:        {elapsed:.1f}s")
    _print_prior(prior, [f"eeg{i}" for i in range(X.shape[1])])

    verdict_ok = prior.verdict() == "no_elementary_fit"
    print()
    print(f"  VERDICT: {'PASS' if verdict_ok else 'FAIL'}  "
          f"(got {prior.verdict()!r})")
    return verdict_ok


def main():
    results = {}
    try:
        results["pima"] = validate_pima()
    except Exception as e:
        print(f"  PIMA errored: {type(e).__name__}: {e}")
        results["pima"] = False

    try:
        results["eeg"] = validate_eeg()
    except Exception as e:
        print(f"  EEG errored: {type(e).__name__}: {e}")
        results["eeg"] = False

    print()
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for name, passed in results.items():
        print(f"  {name:8s}  {'PASS' if passed else 'FAIL'}")
    if all(results.values()):
        print("\nAll checks passed. Safe to open the integration PR.")
        return 0
    print("\nAt least one check failed. Tune defaults or shelf the probe.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
