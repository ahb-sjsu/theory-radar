# Chirp — broadband probe for Theory Radar

**Status:** prototype (branch `feature/chirp-eml-probe`) — not yet wired into
`TheoryRadar.search()`. Standalone, importable, tested.
**Paper:** Odrzywołek, *All elementary functions from a single binary operator*,
arXiv:2603.21852 (March 2026).

## The metaphor

Theory Radar's classical engine is the CW-radar equivalent — it probes
formula space one narrow template at a time (`f1 + f2 * f3`, then
`min(f1, f2) + f3`, ...). An eml-tree is a **broadband chirp**: one
configurable waveform sweeps the entire elementary-function space, trainable
end-to-end with Adam.

| Radar concept | Chirp module analogue |
|---|---|
| Single VCO + frequency sweep | `eml` operator + tree depth |
| Chirp probes channel broadband | `EmlTree` fit to `(X, y)` |
| Matched filter / pulse compression | Adam minimum → candidate formula structure |
| Spectrum analyzer output | `ProbePrior.feature_importance`, `.notable_constants` |
| Range gate narrowed by pre-survey | `TheoryRadar.search(prior=...)` |

## The operator

Odrzywołek (2026) showed:

```
eml(x, y) = exp(x) − log(y)
```

is functionally complete for elementary functions when combined with the
constant `1`. Specifically:

- `exp(x) = eml(x, 1)`
- `log(y) = -eml(0, y) + 1`
- Addition, multiplication, trig, etc. emerge from composition

Uniform grammar: `S → 1 | eml(S, S)`. Every expression is a binary tree
with one node type, which lets you treat the tree as a differentiable
circuit and train with Adam.

## The probe

See `src/symbolic_search/chirp/`:

- `eml_tree.py` — the primitive. `EmlTree(depth, n_features)` is a full
  binary tree of depth `d` with soft-selected leaves. A depth-3 tree over
  `D` features has `8 × (D + 1 + 1)` trainable parameters.
- `prior.py` — `ProbePrior` dataclass; what the probe hands off.
- `probe.py` — `chirp(X, y, ...)` entry point. Does `n_restarts` Adam runs,
  picks the best by loss, and extracts:
  1. `feature_importance` — gradient-norm of output w.r.t. each column
  2. `active_features` — top-k features by importance
  3. `notable_constants` — leaf constants near `{0, 1, ±1/2, 2, e, π, ...}`
  4. `converged` — did loss drop below `0.5 × baseline`?
  5. `verdict()` — `"narrow_prior" | "wide_prior" | "no_elementary_fit"`

## Intended integration (not yet implemented)

The planned but unshipped integration looks like:

```python
radar = TheoryRadar(X, y, projection="pca")

prior = radar.chirp(depth=3, epochs=500, n_restarts=3)
print(prior.verdict())
# "narrow_prior"  →  radar.search() restricts to prior.active_features
# "wide_prior"    →  radar.search() uses prior ranking as a tie-break
# "no_elementary_fit" → radar.search() short-circuits to "ensemble_wins"

result = radar.search(mode="fast", prior=prior)
```

This integration is deliberately NOT in this branch. Ship the probe
standalone, validate on hero datasets, then wire it into `search()` in a
follow-up PR if validation passes.

## Validation protocol

The chirp mode ships IF all three of these pass on a fresh run:

1. **Known-signal dataset (Pima / Breast Cancer)**: `active_features` must
   include at least one canonical high-signal column (glucose, insulin,
   age for Pima; mean_concave_points, worst_radius, etc. for BreastCancer).
2. **Random-label null test**: randomized labels on the same features must
   produce `verdict == "no_elementary_fit"`. A probe that invents structure
   in noise is worse than no probe at all.
3. **Wall-clock budget**: a depth-3 probe over (200–600 samples, 5–30
   features) must complete in under 30 seconds on CPU. Anything slower
   defeats the "reconnaissance before search" framing.

See `tests/test_symbolic_search/chirp/test_probe.py`. The `@pytest.mark.slow`
tests are the hero-dataset validation.

## Known limits

1. **Numerics**: `exp(clamp(x)) − log(clamp(y))` uses `EXP_CLAMP = 15.0`
   (so `exp` tops out near `3.3e6`) and `EPS = 1e-6` floor on `log`.
   Aggressive clipping prevents overflow but also flattens gradients in
   saturated regions — Adam sometimes stalls. `n_restarts ≥ 3` mitigates.
2. **Non-convexity**: Adam on eml-trees has no optimality guarantee.
   You get a local peak, not *the* peak. The `converged` flag is a weak
   guard (loss < 0.5 × baseline); a stricter comparison to a null
   permutation test is future work.
3. **Depth scaling**: the Odrzywołek paper reports exact recovery at
   depth ≤ 4. Past depth 4 the parameter count balloons and training
   becomes much less reliable. Start at depth 3.
4. **`notable_constants` is hints only**: a leaf const of 0.97 is not
   evidence of `1`. Tolerance is `0.03` currently; tighten if you see
   false positives in practice.
5. **Classification uses BCE + sigmoid wrap**: `task="classification"`
   assumes binary 0/1 labels and wraps tree output in a sigmoid implicitly
   via `binary_cross_entropy_with_logits`. Multiclass is not implemented.

## References

- A. Odrzywołek, *All elementary functions from a single binary operator*,
  arXiv:2603.21852, 2026.
- Code at Zenodo record 19183008 (paper's supplementary).
