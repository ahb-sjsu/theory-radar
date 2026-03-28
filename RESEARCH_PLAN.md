# Tensor Decomposition Methods for the 3-Body Problem

## Principal Investigator
Andrew H. Bond, Senior Member IEEE
Department of Computer Engineering, San Jose State University

## Research Question
Does the rank-6 coupling tensor of the gravitational 3-body problem admit
exact or near-exact low-rank decompositions at configurations beyond the
classically known special cases, and can A*-guided search through
transformation space systematically discover them?

## Core Idea
Represent the full dynamical coupling of the 3-body system as a rank-6
tensor M_{i,mu,alpha, j,nu,beta} (body x spatial x phase, twice) and treat
the question "is this system tractable?" as a tensor rank problem. Low
effective rank = nearly decomposable dynamics = tractable sub-problems.

This reframes Poincare's impossibility result: he proved no closed-form
solution exists as convergent power series. Tensor decomposability is a
different question -- decomposition into independent sub-problems, not
into a series.

## Connection to Existing Work
- Geometric Reasoning book (Bond, 2026): A* on tool space, manifold search
- AGI-HPC benchmarks: invariance testing methodology (perturbation -> measure
  displacement) applies directly to orbit stability analysis
- ARC Prize solver: A* search, fuzzing, geometric verification -- same
  algorithmic toolkit, different domain

## Phases

### Phase 1: Tensor Landscape Mapping (Weeks 1-3)
Compute the coupling tensor across configuration space and map effective rank.
- Deliverable: Rank atlas of the 3-body problem
- Code: tensor_landscape.py
- Compute: Atlas CPU (48 cores), embarrassingly parallel

### Phase 2: Validate on Known Solutions (Weeks 3-4)
Verify Lagrange, Euler, figure-eight correspond to low tensor rank.
- Deliverable: Tensor characterization of known periodic orbits
- Code: validate_known.py
- Decision gate: if known solutions don't correlate with low rank, stop

### Phase 3: A* Search for New Decompositions (Weeks 5-10)
A* through coordinate-transformation space, minimizing effective rank.
- Deliverable: Optimal representation map; new decomposable configurations
- Code: tensor_astar.py
- Compute: Atlas CPU+GPU

### Phase 4: Fuzzing for Periodic Orbits (Weeks 8-12)
Perturb initial conditions along low-rank directions in tensor space.
- Deliverable: New periodic orbit families with tensor characterization
- Code: tensor_fuzz.py
- Compute: Atlas CPU+GPU

### Phase 5: Theoretical Characterization (Weeks 10-16)
Formalize findings. Either: new integrable regions, or tensor-theoretic
characterization of the obstruction to integrability.
- Deliverable: Paper manuscript

## Publication Targets
- Phases 1-2: Workshop paper (AAS/DDA or SIAM DS)
- Phases 3-4: Journal paper (J. Computational Physics or Celestial Mechanics)
- Phase 5: Chapter in Geometric Reasoning book
- Full arc: "Tensor Decomposition Methods for the N-Body Problem"

## Infrastructure
- Atlas workstation: 48 cores, 128GB RAM (upgradeable to 512GB), 2x GV100 32GB
- Python 3.12, NumPy, SciPy, sympy (symbolic), matplotlib
- Existing code: A* (geometric-reasoning/tmp_astar_v3.py), fuzzing framework

## Risk Register
| Risk | Likelihood | Mitigation |
|------|-----------|------------|
| Known solutions don't show low rank | Medium | Phase 2 is the decision gate |
| No new decompositions exist | Medium | Characterizing the obstruction is itself publishable |
| Tensor computation too expensive at high resolution | Low | Exploit symmetry to reduce sampling; GPU acceleration |
| Overlaps with existing literature | Low | Literature search in Phase 1; novel angle is the A* search |

## Budget
Compute: Atlas (owned), electricity only (~$20/month)
Software: All open-source
Total: ~$0 direct cost
