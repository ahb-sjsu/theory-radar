# Theory Radar — Architecture & Design Document

## Overview

Theory Radar is a symbolic formula search engine that discovers interpretable classifiers from labeled data. Given a feature matrix X and binary labels y, it finds a formula like `min(insulin, age) + glucose` that classifies data by thresholding: `y_hat = 1[f(x) > τ]`.

## Core Question

**On this specific dataset, can a simple interpretable formula match an ensemble method?** If yes, the formula is preferable (auditable, regulatable, hand-computable). If no, Theory Radar quantifies exactly how much accuracy interpretability costs.

## Three-Tier Architecture

```
┌─────────────────────────────────────────────────────────┐
│                                                         │
│   Tier 3: REPRESENTATION AUGMENTERS                     │
│   ┌─────┐ ┌────────┐ ┌────────┐ ┌────────┐            │
│   │ PCA │ │ Tucker │ │ Kernel │ │ Neural │             │
│   └──┬──┘ └───┬────┘ └───┬────┘ └───┬────┘            │
│      │        │          │          │                   │
│      └────────┴──────────┴──────────┘                   │
│                    │                                    │
│         augmented features (raw + projected)            │
│                    │                                    │
│ ──────────────────────────────────────────────────────  │
│                                                         │
│   Tier 2: SEARCH ACCELERATORS                           │
│   ┌────────────┐ ┌────────────┐ ┌──────────────────┐   │
│   │ A* Beam    │ │ Subspace   │ │ Meta-Learned     │   │
│   │ Search     │ │ Fuzzing    │ │ Pruning          │   │
│   │            │ │            │ │                  │   │
│   │ priority = │ │ random k   │ │ exhaustive →     │   │
│   │ depth+h(n) │ │ of d feats │ │ viability →      │   │
│   │            │ │ per trial  │ │ criterion search │   │
│   └─────┬──────┘ └─────┬──────┘ └────────┬─────────┘   │
│         │              │                 │              │
│         └──────────────┴─────────────────┘              │
│                        │                                │
│              pruned search candidates                   │
│                        │                                │
│ ──────────────────────────────────────────────────────  │
│                                                         │
│   Tier 1: CORE ENGINE                                   │
│   ┌──────────────────────────────────────────────────┐  │
│   │                                                  │  │
│   │  Phased Enumeration                              │  │
│   │  depth 1 → depth 2 → depth 3 → ...              │  │
│   │  binary ops × features + unary ops               │  │
│   │                                                  │  │
│   │  Exact Optimal F1 (sort-and-sweep, O(N log N))   │  │
│   │  Monotone Invariance Theorem (F1/AUROC reuse)    │  │
│   │                                                  │  │
│   │  FormulaTrace (records ops for test-set replay)  │  │
│   │                                                  │  │
│   │  Fair CV Evaluation                              │  │
│   │  train: discover formula + tune threshold        │  │
│   │  test: replay formula + apply threshold + score  │  │
│   │                                                  │  │
│   └──────────────────────────────────────────────────┘  │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

### Tier 1: Core Engine

The foundation. Everything else is optional; this tier alone produces valid results.

**Phased enumeration.** At each depth level, generate all candidate formulas by applying binary operations (10 ops: +, -, ×, ÷, max, min, hypot, diff_sq, harmonic, geometric) between existing formulas and raw features, plus unary operations (8 ops: log, sqrt, sq, inv, neg, abs, sigmoid, tanh).

**Exact optimal F1.** For each candidate formula, compute the optimal thresholded F1 by sorting all N samples and sweeping all possible thresholds in O(N log N). Both threshold directions are evaluated. Constant-valued candidates get F1 = 0 to prevent sort-order artifacts.

**Monotone Invariance Theorem (proved).** Strictly monotone transforms preserve optimal thresholded F1 and AUROC. The search reuses the parent's F1 and AUROC for monotone unary children (log, sqrt, inv, neg), saving O(N log N) evaluations. Subtrees are still expanded — only the evaluation is skipped, not the node.

**FormulaTrace.** Records each formula's operation sequence (list of leaf/binary/unary steps with feature indices). Enables faithful replay on test data — the exact same computation is applied to unseen features.

**Fair CV evaluation.** For each fold: (1) discover formula on training split, (2) find optimal threshold on training split, (3) replay formula on test split, (4) apply threshold to test predictions, (5) score with F1. This is identical to how sklearn baselines are evaluated. There is no methodological asymmetry.

### Tier 2: Search Accelerators

Optional components that make the search faster or more thorough. The core engine works without them; they improve results.

**A* beam search.** Expands nodes in order of priority = depth + h(n), where h = 0 for goal nodes, h = 1 - F1 for alive non-goals (admissible since h < 1 ≤ h*), and h = ∞ for meta-pruned dead nodes. Beam width B limits memory by keeping only the top B nodes at each depth.

**Subspace fuzzing.** For each of n_subspaces trials, randomly select k features from the augmented feature set and run a full search on that subspace. Analogous to the random subspace method in Random Forests. Ensures coverage of C(d, 3) possible 3-feature combinations that a single beam might miss.

**Meta-learned pruning.** The algorithm's primary novel contribution:
1. Exhaustive enumeration to depth 3 on top-6 features (~6K formulas)
2. Label each shallow node's subtree as alive (contains near-optimal) or dead (safe to prune)
3. Search a family of 12 candidate criteria over node features (F1, AUROC, depth, feature count)
4. Select the criterion with highest prune rate at zero false negatives
5. Threshold = min(criterion value among alive nodes) → zero FN by construction

The zero-FN guarantee holds within the enumerated space (depth 3, specific operators, specific dataset). Application at greater depth is heuristic extrapolation, noted honestly.

### Tier 3: Representation Augmenters

Optional projections that give formulas implicit access to ALL features, not just the 2-3 directly referenced at depth 3. Each projection is fit on training data and applied to test data via `transform_test()`.

| Projection | What it captures | Mechanism |
|-----------|-----------------|-----------|
| **PCA** | Linear variance directions | Standard PCA → top-k components |
| **Tucker** | Pairwise feature interactions | HOSVD of x⊗x interaction tensor |
| **Kernel** | Nonlinear manifold structure | Kernel PCA or Random Fourier Features |
| **Neural** | Learned nonlinear projection | 1-layer ReLU net, hidden activations as features |

A formula `min(pc0, T3) + f5` combines a PCA component (linear combination of all features), a Tucker factor (interaction pattern), and a raw feature. This is interpretable at the projection level — each component can be traced back to which original features it weights.

**Impact:** On BreastCancer (d=30), PCA projections improved test F1 from 0.955 to 0.963, closing the gap with gradient boosting (0.967) from 21σ to 6σ.

## Theoretical Foundation

### Monotone Invariance Theorem (proved)

If g is strictly monotone on the range of f, then the optimal thresholded F1 of g(f(X)) equals that of f(X) when both threshold directions are considered. Proof: increasing g preserves the threshold equivalence; decreasing g swaps the direction, which is already covered by the bidirectional search.

**Corollary (AUROC Invariance):** Since AUROC depends only on rank order, and we define AUROC = max(AUC, 1-AUC), both increasing and decreasing monotone transforms preserve AUROC.

### What Was Abandoned

An earlier version claimed an "admissible DAG heuristic" with:
- **H2 (AUROC-F1 bound):** Proved false. Counterexample: step ROC (0,0)→(0,0.75)→(1,0.75)→(1,1) has AUROC=0.75 but Youden J=0.75 > 2A-1=0.5.
- **H3 (Feature coverage):** Inadmissible. Unary transforms like x² can improve F1 without adding features.
- **Three-component max:** Collapsed to trivial 0/1 goal indicator.

These were removed and acknowledged in the paper's "What Theory Radar Is Not" section. The meta-search framework replaced them.

## Configuration & Autotune

All tier 2 and tier 3 components are configurable:

```python
radar = TheoryRadar(
    X, y,
    projection=["pca", "tucker"],      # Tier 3
    n_projection_components=8,
    n_subspaces=10,                     # Tier 2: fuzzing
    subspace_k=12,
    meta_prune=True,                    # Tier 2: learned pruning
    ensemble_k=3,                       # Tier 2: formula ensemble (planned)
    validation_fraction=0.2,            # Tier 2: holdout for beam selection
)
result = radar.search(mode="fast", max_depth=3)
```

**Autotune** searches over configurations automatically:

```python
radar, result = TheoryRadar.autotune(X, y, max_time=120)
```

Evaluates 10 configurations on a 20% validation holdout and returns the best.

## Experimental Results

### Full Pipeline (Tier 1 + 2 + 3, Fair Evaluation)

| Dataset | N | d | Test F1 | Formula | vs GB | vs RF | vs LR |
|---------|---|---|---------|---------|-------|-------|-------|
| BreastCancer | 569 | 30 | 0.963 | `(f19 - pc0) + f5` | 6.0σ GB> | 10.3σ RF> | 36.9σ LR> |
| Wine | 178 | 13 | 0.953 | `min(w6,w0) + w12` | **16.2σ A*>** | 17.5σ RF> | 22.6σ LR> |
| Diabetes | 768 | 8 | 0.668 | `min(v5,v7) + v1` | **24.7σ A*>** | **26.9σ A*>** | **21.4σ A*>** |
| Banknote | 1372 | 4 | 0.986 | `(v0+v1) + v2` | 25.6σ GB> | 22.1σ RF> | **26.5σ A*>** |
| EEG | 14980 | 14 | 0.655 | `max(v13-v5, v6)` | 245.6σ GB> | 417.0σ RF> | **250.0σ A*>** |

*Additional datasets (Heart, Sonar, Spambase, German, Australian, Adult, HIGGS, Electricity, MiniBooNE, Ionosphere, Magic) running.*

### Key Findings

1. **Interpretability is sometimes free.** On Diabetes, a 3-feature formula beats all baselines at >21σ.
2. **PCA projections close the gap.** On BreastCancer, PCA took the formula from 0.955 to 0.963 (6σ from GB vs 21σ).
3. **Formulas generalize well.** Train-test gaps range from 0.002 to 0.040 across datasets.
4. **When ensembles win, they win big.** On EEG, GB beats the formula at 246σ. The boundary is genuinely high-dimensional.
5. **Theory Radar quantifies the tradeoff.** This is the core value: "on your data, interpretability costs X% accuracy."

## File Structure

```
src/symbolic_search/
├── __init__.py              # Exports, version (0.4.0)
├── radar.py                 # TheoryRadar class (Tier 1 + 2 + autotune)
├── _ops.py                  # Binary/unary operation registries
├── _projections.py          # PCA, Tucker, Kernel, Neural (Tier 3)
├── _heuristic_dag.py        # exact_optimal_f1, auroc_safe, heuristics
├── _search.py               # Phased exhaustive search (Tier 1 baseline)
├── _scaling.py              # Beam search, semantic hashing (Tier 2)
├── _auroc_proof.py          # Historical (theorem is false)
├── _theory.py               # Theoretical utilities
```

## Dependencies

- **Core (Tier 1):** numpy ≥ 1.24
- **Projections (Tier 3):** scikit-learn ≥ 1.0
- **GPU acceleration:** cupy-cuda12x (optional)
- **Thermal management:** batch-probe ≥ 0.4.0 (optional, for ThermalJobManager)

## Paper Status

- **Target:** IEEE Transactions on Artificial Intelligence (TAI)
- **Title:** "Theory Radar: Learning Safe Pruning Rules for Symbolic Formula Search from Exhaustive Micro-Search"
- **Status:** Anonymized, impact statement added, benchmark suite running (targeting 17 datasets)
- **Primary contribution:** Meta-search framework (Tier 2) for discovering zero-FN pruning rules
- **Secondary:** PCA/Tucker projections closing the interpretability gap (Tier 3)
- **Tertiary:** Fair evaluation methodology with FormulaTrace (Tier 1)
