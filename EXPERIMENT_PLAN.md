# Theory Radar — Experiment Plan

## Objective

Produce a 17-dataset benchmark table for the TAI paper submission, with all results from the full pipeline (Tier 1 + 2 + 3) under fair held-out evaluation.

## Dataset Selection

All datasets are established ML benchmarks that reviewers will recognize. Selected for diversity in N, d, domain, and difficulty.

### Small Benchmarks (N ≤ 2K)

| # | Dataset | N | d | Domain | Source | Status |
|---|---------|---|---|--------|--------|--------|
| 1 | Breast Cancer Wisconsin | 569 | 30 | Medical | UCI | **DONE** |
| 2 | Wine (class 0 vs rest) | 178 | 13 | Chemistry | UCI | **DONE** |
| 3 | Pima Diabetes | 768 | 8 | Medical | UCI | **DONE** |
| 4 | Banknote Authentication | 1372 | 4 | Security | UCI | **DONE** |
| 5 | Ionosphere | 351 | 34 | Physics | UCI | **RUNNING** |
| 6 | Heart Disease (Statlog) | 303 | 13 | Medical | UCI/Statlog | QUEUED |
| 7 | Sonar | 208 | 60 | Military | UCI | QUEUED |

### Medium Benchmarks (2K < N ≤ 10K)

| # | Dataset | N | d | Domain | Source | Status |
|---|---------|---|---|--------|--------|--------|
| 8 | Spambase | 4601 | 57 | NLP/Security | UCI | QUEUED |
| 9 | German Credit | 1000 | 24 | Finance | UCI/Statlog | QUEUED |
| 10 | Australian Credit | 690 | 14 | Finance | Statlog | QUEUED |

### Large Benchmarks (N > 10K)

| # | Dataset | N | d | Domain | Source | Status |
|---|---------|---|---|--------|--------|--------|
| 11 | EEG Eye State | 14980 | 14 | Neuro | UCI | **DONE** |
| 12 | MAGIC Gamma Telescope | 19020 | 10 | Physics | UCI | **RUNNING** |
| 13 | Electricity | 45312 | 8 | Energy | OpenML | RETRY (OOM) |
| 14 | Adult/Census Income | 48842 | 14 | Economics | UCI | QUEUED |
| 15 | HIGGS (98K subsample) | 98050 | 28 | Physics | UCI/Nature | QUEUED |
| 16 | MiniBooNE | 130064 | 50 | Physics | UCI | RETRY (OOM) |
| 17 | Covertype (binarized) | 581012 | 54 | Ecology | UCI | FUTURE |

## Experiment Configuration

### Per-Dataset Parameters

| Size class | n_repeats | beam_width | n_subspaces | subspace_k | n_pca |
|-----------|-----------|------------|-------------|------------|-------|
| N ≤ 2K | 200 | 100 | 15 | 12 | 8 |
| 2K < N ≤ 10K | 100 | 100 | 10 | 10 | 8 |
| 10K < N ≤ 50K | 50 | 50 | 5 | 8 | 8 |
| N > 50K | 10-20 | 20-30 | 3 | 6 | 8 |

### Fixed Parameters (all datasets)

- **Search depth:** 3 (formulas use at most 3 features/ops)
- **Projection:** PCA with 8 components (Tier 3)
- **Meta pruning:** Per-fold, 12-criterion family search (Tier 2)
- **CV:** Repeated stratified k-fold, k=5
- **Evaluation:** Fair test-set replay via FormulaTrace
- **Baselines:** Gradient Boosting (100 trees, depth 4), Random Forest (100 trees), Logistic Regression (max_iter=1000)
- **Significance:** Paired t-test, reported as effect size (σ) with caveat about repeated-CV correlation

### Thermal Management

- `ThermalJobManager(target_temp=84, max_concurrent=3, cooldown_margin=4)`
- Each dataset is a separate subprocess via `run_one.py`
- Results saved incrementally as `result_{name}.json`
- Failed jobs automatically reported (exit code)

## Preprocessing Protocol

### Numeric Datasets (most)
1. Drop non-numeric columns (if any)
2. StandardScaler (zero mean, unit variance)
3. NaN → 0.0

### Mixed-Type Datasets (German Credit, Adult)
1. Numeric columns: StandardScaler
2. Categorical columns: OrdinalEncoder
3. Concatenate
4. NaN → 0.0

### Target Encoding
- Binary datasets: use as-is
- Multi-class: binarize (largest class vs rest, or domain-specific)
- String targets: LabelEncoder
- Known targets: dataset-specific (e.g., Diabetes: "tested_positive" = 1)

## Expected Outcomes

### Datasets Where Formulas Should Win
- **Diabetes:** 3-feature formula already beats GB at 25σ
- **Wine:** Already beats GB at 16σ
- **Heart:** Low-d, medical — formula likely captures clinical boundary
- **Australian Credit:** Low-d, likely linear boundary

### Datasets Where Ensembles Should Win
- **EEG:** Already confirmed — 246σ GB advantage
- **HIGGS:** High-d physics with complex interactions
- **Covertype:** 54 features, ecological complexity
- **Spambase:** 57 features, sparse word patterns

### Datasets Where PCA/Tucker May Close the Gap
- **BreastCancer:** PCA already closed from 21σ to 6σ
- **Ionosphere:** PCA showing 0.923 vs 0.912 raw (early results)
- **German Credit:** Mixed features — Tucker interactions may help

## Tucker Decomposition Experiment

Separate from the main benchmark. Tests whether Tucker-decomposed feature interaction tensors can push formula F1 past ensemble methods on BreastCancer and Ionosphere.

- **Modes:** raw, tucker-only, raw+tucker
- **Tucker rank:** 8
- **Hypothesis:** Tucker captures pairwise interactions that PCA misses; raw+tucker should exceed PCA's 0.963 on BreastCancer

## Deliverables

1. **`result_{name}.json`** for each of 17 datasets — machine-readable results
2. **Paper Table III** — 17-row results table with formula, test F1, gap, and σ vs each baseline
3. **Updated `theory_radar_paper.tex`** — full results, honest interpretation
4. **`ARCHITECTURE.md`** — three-tier design document (this document)

## Timeline

- **Completed:** 5 datasets (BreastCancer, Wine, Diabetes, Banknote, EEG)
- **Running:** 2 datasets (Ionosphere, Magic)
- **Queued:** 8 datasets (Heart, Sonar, Spambase, German, Australian, Adult, HIGGS, Electricity, MiniBooNE)
- **Future:** Covertype (after hardware upgrade — needs >32GB GPU for N=581K)
- **Tucker:** Runs after main benchmark completes

Estimated total runtime: ~4-8 hours depending on thermal throttling.

## Risks

1. **GPU OOM on large datasets:** Mitigated by reduced beam width (20-30 for N>50K)
2. **CPU overheating:** Mitigated by ThermalJobManager (target 84°C, max 3 concurrent)
3. **Categorical encoding artifacts:** German Credit and Adult have categorical features; OrdinalEncoder is arbitrary but functional. Alternative: one-hot (increases d significantly)
4. **Repeated-CV correlation:** σ values are effect sizes, not strict Gaussian significance. Noted in paper.
5. **HIGGS download time:** 98K×28 via OpenML may take several minutes
