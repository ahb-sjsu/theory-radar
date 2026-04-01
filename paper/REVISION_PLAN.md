# Theory Radar Paper — Revision Plan

## Reviewer Summary
"Interesting paper, materially stronger than a typical symbolic-search submission, but not yet journal-ready. Three papers at once."

## Issue 1: Contribution Drift (CRITICAL)
**Problem:** Paper sells three-tier system + "rigorous answer on any dataset" but says primary contribution is meta-learned pruning, while benchmark uses unpruned search.

**Fix:** Restructure with benchmark as center of gravity.
- **New title option:** "When Do Simple Formulas Beat Ensembles? A 30-Dataset Benchmark for Interpretable Binary Classification"
- **Abstract/intro:** Lead with the benchmark question, not the system architecture
- **Meta-learned pruning:** Compress to one section, present as search efficiency contribution
- **Three-tier architecture:** Describe as methodology, not as the contribution
- **Conclusion:** Center on the empirical findings (7/30 wins, characteristics of winning datasets)

## Issue 2: Benchmark Comparability (CRITICAL)
**Problem:** 19 datasets use 200x5 CV vs GB; 11 use 20x5 CV vs RF. Pooled counts misleading.

**Fix:** Re-running all 11 new datasets with identical protocol (200x5 CV vs GB/RF/LR). IN PROGRESS on Atlas. When done, all 30 datasets will have:
- 1000 paired measurements
- Same three baselines (GB, RF, LR)
- Same beam search config (depth 3, width 100, PLS projections)
- Same statistical reporting

## Issue 3: Overclaiming (IMPORTANT)
**Problem:** "Rigorously on any tabular dataset" vs limitations saying guarantees are bounded.

**Fix:** Replace all instances of "any tabular dataset" with:
"Within a bounded depth-3 symbolic formula class for binary tabular classification, the method determines whether simple formulas are competitive under the specified evaluation protocol."

Specific edits:
- Abstract: "answers this question rigorously" → "provides empirical evidence"
- Intro: "on any tabular dataset" → "across standard tabular benchmarks"
- Related work: "first rigorous YES/NO" → "first systematic empirical comparison"

## Issue 4: Statistical Language (IMPORTANT)
**Problem:** "22σ win" notation throughout, while admitting CV folds are correlated.

**Fix:**
- Drop "σ significance" everywhere
- Report: mean ΔF1, std, 95% CI, and corrected resampled t-statistic
- Table columns: "Test F1", "GB F1", "ΔF1", "95% CI", "Direction"
- Text: "the formula outperforms GB by ΔF1=0.031, 95% CI [0.028, 0.034]"
- Add Nadeau-Bengio corrected t-test (accounts for CV fold correlation)
- Keep effect size as supplementary but don't call it "σ"

## Issue 5: "A* wins" Residue (EASY FIX)
**Problem:** Table says "A* wins" but paper says A* framing was abandoned.

**Fix:** Global replace:
- "A* wins" → "Formula wins" or "TR wins"
- "A*>" → "Formula>"
- Check all tables, text, ethics section

## Issue 6: PLS Interpretability (IMPORTANT)
**Problem:** PLS formulas aren't transparent without loading vectors.

**Fix:** Add systematic PLS analysis for all 7 winning datasets:
- Table showing top-3 PLS loading weights per component for each winner
- Stability metric: mean pairwise cosine similarity of PLS weights across folds
- Sparsity: how many features contribute >10% of loading weight
- For each winner: one-sentence clinical/domain interpretation
- Note: run_unified.py now saves pls_weights_sample in results

## Issue 7: Ethics Section (MODERATE)
**Problem:** "Ethically suspect" is under-argued for a technical paper.

**Fix:**
- "deploying an opaque ensemble... is ethically suspect" → "our results strengthen the empirical case for preferring transparent models when they match performance in regulated settings"
- Keep Sullins citation but soften framing
- Remove "not merely suboptimal" language
- Keep EU AI Act and FDA references as context, not as arguments

## Issue 8: TurboQuant/Depth Material (MODERATE)
**Problem:** Paper feels overstuffed. TurboQuant is a negative result.

**Fix:**
- Compress depth experiments to one paragraph + one small table
- Move TurboQuant to a "Search Space Saturation" paragraph in Discussion
- Key message in 2-3 sentences: "We verified depth-3 optimality by searching to depth 9 with 5x-wider beams (via 3-bit quantized beam storage). Deeper formulas consistently overfit; wider beams at depth 3 produce statistically identical results. The depth-3 formula space is well-explored by a 100-wide beam."

## Issue 9: Depth-3 Claim (EASY FIX)
**Problem:** "Structural optimum" is too absolute; Ionosphere improves at depth 5.

**Fix:** "Depth 3 is the best default across the benchmark" or "Depth 3 provides the best generalization on 28 of 30 datasets; Ionosphere benefits from depth 5 (N/d=10.3)."

## Issue 10: Reproducibility (IMPORTANT)
**Problem:** Missing implementation details for reimplementation.

**Fix:** Add a "System Details" subsection or appendix with:
- Full operator set: +, -, *, /, max, min, hypot, diff_sq, harmonic, geometric (binary); log, sqrt, sq, abs, sigmoid, tanh (unary)
- Domain handling: log(|x|+1e-30), sqrt(|x|), clip sigmoid/tanh to [-500,500]
- NaN/inf: nan_to_num(nan=0, posinf=1e10, neginf=-1e10)
- Constant detection: skip candidates with std < 1e-12
- Autotune: MI prefiltering (keep features with MI > 0.1*max), Hyperband with budget [1x, 2x, 4x] and halving
- PLS: sklearn PLSRegression, n_components=min(8, d-1, N_train-1)
- Search: beam width 100, depth 3, 15 random subspaces of size min(10, d)
- Evaluation: sort-and-sweep F1, both threshold directions, constant candidates get F1=0
- Code availability statement with GitHub URL

## Execution Order
1. Wait for unified benchmark results (overnight)
2. Fix Issue 5 (A* → Formula, 5 min)
3. Fix Issue 3 (overclaiming, 15 min)
4. Fix Issue 4 (statistical language, 1 hour — requires recomputing CIs)
5. Fix Issue 1 (restructure, 2 hours)
6. Fix Issue 6 (PLS analysis, 1 hour — needs results data)
7. Fix Issues 7, 8, 9 (moderate edits, 30 min each)
8. Fix Issue 10 (reproducibility section, 30 min)
9. Rebuild PDF, full proofread
