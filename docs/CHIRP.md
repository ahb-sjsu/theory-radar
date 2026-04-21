# Chirp — broadband probe for Theory Radar

> **⚠️ Status — 2026-04-20: VALIDATION FAILED.** The probe compiles, runs
> cleanly, and passes its API-contract tests, but **does not extract signal
> from real datasets at depth ≥ 3**. Do not merge this into the public API.
> See §"Validation result" below for the diagnosis.
>
> The repo/branch is preserved for future work; the metaphor is still
> correct and the paper's mathematics is untouched. What fails is the naive
> training setup as currently implemented.

**Branch:** `feature/chirp-eml-probe` — not yet wired into `TheoryRadar.search()`.
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

## Validation result (2026-04-20) — the probe does not train at depth 3

Manual validation on two hero datasets with `scripts/chirp_validation.py`:

| Dataset | Verdict | `feature_importance` |
|---|---|---|
| Pima Diabetes (N=768, D=8) | `no_elementary_fit` | `[0, 0, 0, 0, 0, 0, 0, 0]` |
| EEG Eye State (N=14980, D=14) | `no_elementary_fit` | `[0, 0, …, 0]` |

Both verdicts read as "correct" under `verdict()` but **zero importance
across every feature means the probe is not producing any signal at all** —
the tree output is a constant, insensitive to `X`. This is an architectural
failure, not a convergence failure.

### Why — trace with `scripts/chirp_diagnose_saturation.py`

```
depth=0: raw out  min=  -0.75  max=   1.17  std=0.3137    ok
depth=1: raw out  min=   2.18  max=  14.81  std=5.4074    ok
depth=2: raw out  min=   7.66  max=  50.00  std=18.4485   ok  (hitting ceiling)
depth=3: raw out  min=  50.00  max=  50.00  std=0.0000    SATURATED
depth=4: raw out  min=  50.00  max=  50.00  std=0.0000    SATURATED
```

The eml operator's `exp(x)` branch compounds with depth: `exp(exp(…))`
saturates at `OUT_CLAMP = 50` after 2–3 layers from any standardized input.
Gradient through a hard clamp is zero, so Adam cannot drive the leaves.

Adam still has an escape hatch — the root `scale`/`bias` readout — so it
drives `scale → 0`, outputting the constant `bias ≈ ȳ`. Loss lands at the
baseline (predict mean) and stays there. The probe appears to "converge"
to a constant predictor, and `feature_importance` is identically zero.

The earlier Breast Cancer hero-test "pass" was a coincidence: sklearn's
default Breast Cancer load returns unstandardized data with per-column
scales differing by 3+ orders of magnitude, which happens to keep the
tree output unsaturated. The `argsort` of a zero-importance array
returns indices in descending order, and the test's "hit set" contained
index 27 — pure luck.

### Why depth 3 does not train on standardized inputs

Standardized X has `std ≈ 1`. After one `eml` layer, output `std ≈ 5`.
After two, output range is `[8, 50]`. After three, uniformly 50.
Every subsequent layer is constant. The operator is simply not a
fixed-point-stable composition on the real line; it requires a
**contractive wrapper** to be trainable at depth.

## Fix directions (if someone wants to revive this)

1. **Replace hard clamps with smooth saturating functions.** The natural
   candidates:
   - `soft_clip(x, c) = c * tanh(x / c)` — bounded in [−c, c] with
     O(1/c) gradient in the saturated regime. Still small, but nonzero.
   - `smooth_exp(x) = log1pexp(x)` (softplus) as a replacement for `exp`.
     Grows linearly, not exponentially. Kills the compositional blowup,
     changes the operator's identities — you lose exact recovery of
     `exp(x) = eml(x, 1)`. Now a different operator family.
2. **Per-layer batchnorm.** Normalize tree output between layers. Standard
   deep-learning trick for compositional blowup. Adds ~4d parameters per
   eml-tree of depth d. Also changes the operator's identities.
3. **Initialize leaves so compositional output stays bounded.** This
   requires solving `eml(c, c) = c` for the leaf-constant fixed point;
   approximate answer is `c ≈ -1.31` (LambertW-style). Aggressive
   leaf-init pinning might keep the tree trainable at depth 3, but
   Adam will rapidly drift away from the fixed point.
4. **Shelf the probe** and accept that Theory Radar's classical engine
   doesn't need a broadband precursor on the datasets we care about.

My recommendation is (4) unless someone has a research reason to pursue
(1) or (2). The metaphor is lovely, the paper's math is right, but the
naive operator isn't trainable without structural modifications — and
once you modify the operator you've lost the "eml = single primitive"
property that made it interesting. The probe becomes just another
differentiable symbolic-regression engine, of which there are several
(PySR, KAN, DeepSymReg).

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
