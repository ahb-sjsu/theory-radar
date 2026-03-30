# Symbolic Expression Trees for Chaga: A Proposal

**From:** Andrew H. Bond, San Jose State University
**To:** Robert James Bruce, Sonoma State University
**Re:** Extending Chaga with a symbolic abstraction layer

## Motivation

I'm building Theory Radar, a symbolic formula search engine that discovers interpretable classifiers from data (`pip install theory-radar`). The system finds formulas like `min(insulin, age) + glucose` that can match or beat gradient boosting on real-world datasets.

The core bottleneck is **symbolic representation**. Currently, formulas are ad-hoc strings and tuple lists. We need proper expression trees with algebraic properties, and Chaga's design — particularly metadata attributes and arbitrary-precision arithmetic — maps naturally onto this problem.

## What the Code Demonstrates

The attached `symbolic_ast.chaga` implements five capabilities using Chaga's features:

### 1. Typed Expression Trees (ExprNode struct)

Every node carries Chaga-style metadata attributes:
- `.monotonicity` — whether the operation preserves ordering (enables a proven theorem: monotone transforms preserve optimal F1)
- `.is_bounded` — output range (enables pruning: if bound_hi < current best, skip)
- `.is_commutative` — for canonical form normalization
- `.canonical_hash` — exact algebraic equivalence key

### 2. Algebraic Simplification

Rules applied recursively:
- **Constant folding**: `3 + 4` → `7` (exact with BCD!)
- **Identity elimination**: `x + 0` → `x`, `x * 1` → `x`
- **Idempotent reduction**: `x - x` → `0`, `max(x, x)` → `x`
- **Double negation**: `neg(neg(x))` → `x`
- **Algebraic identities**: `sqrt(sq(x))` → `abs(x)`

### 3. Canonical Forms

Commutative operations normalize child order by hash:
- `(x1 + x0)` and `(x0 + x1)` produce identical canonical hashes
- This enables **exact deduplication** of the formula search space
- With BCD arithmetic, constant values are exact — no float-tolerance hacks

### 4. Formal Properties

Derived automatically during tree construction:
- **Monotonicity** (increasing/decreasing/non-monotone per operator)
- **Boundedness** (sigmoid → [0,1], tanh → [-1,1], sq → [0,∞))
- **Commutativity and idempotency** flags per operator

These properties enable search-time optimizations:
- Monotone children reuse parent's F1 score (O(N log N) savings)
- Bounded nodes can be pruned if their range can't improve the best score
- Idempotent pairs (x op x) are simplified before evaluation

### 5. Compiled Evaluation

`eval_node()` recursively evaluates the tree for a given data point. In Chaga, this compiles to native x86-64 code — no interpreter overhead. For batch evaluation over N samples, the outer loop calls this per-sample.

## Why Chaga Is a Good Fit

1. **Metadata attributes are the natural home for node properties.** In Python, we fake this with dicts and dataclasses. Chaga's `.self.type`, `.protect`, `.scope` pattern is exactly what an AST node needs.

2. **Arbitrary-precision BCD makes canonical hashing exact.** In IEEE 754, `0.1 + 0.2 ≠ 0.3`. In Chaga BCD, `0.1 + 0.2 = 0.3`. This means constant folding is exact and canonical forms are guaranteed unique.

3. **Compiled evaluation eliminates interpreter overhead.** Theory Radar evaluates millions of formulas per search. A compiled evaluator (even without GPU) would be dramatically faster than Python's interpreted loops.

4. **The `.protect` attribute enforces immutability.** Once an AST is constructed, setting `.protect = "R"` guarantees no accidental mutation during search — a common bug source.

## What Would Make This Production-Ready in Chaga

1. **Dynamic memory / pointer support for recursive trees.** The current code assumes pointer-based tree construction. Chaga's planned garbage collection would help.

2. **Array support for batch evaluation.** Evaluating a formula on N=10,000 samples needs efficient array iteration.

3. **A standard library for math functions.** `log()`, `sqrt()`, `exp()`, `tanh()` need to be available as built-ins or a math library.

4. **String hashing.** The canonical hash is currently a string. A fast hash function (FNV, xxHash) would make equivalence checking O(1) instead of O(length).

## Potential Collaboration

I'd be interested in:
- Testing this module against the Chaga compiler once pointer/struct support is complete
- Co-authoring a paper on symbolic abstraction in systems languages (Chaga vs. SymPy vs. Julia Symbolics)
- Using this as a case study for your programming languages course at Sonoma State

Theory Radar is open source: https://github.com/ahb-sjsu/theory-radar

Andrew H. Bond
Senior Member, IEEE
Department of Computer Engineering
San Jose State University
andrew.bond@sjsu.edu
