# Tensor Rank as a Predictor of Dynamical Tractability

**Automatic decomposition of coupled systems via A* search**

Andrew H. Bond, Senior Member IEEE — San Jose State University

---

## Key Finding

The multilinear rank of a system's coupling tensor predicts whether it is dynamically tractable. In the gravitational 3-body problem, configurations at tensor rank 7-8 are **100% periodic** (67/67 tested). A* search through coordinate transformations automatically finds representations that minimize this rank.

## What This Is

A unified framework for answering: *when does a complex coupled system simplify?*

Given any coupled dynamical system (N-body gravity, protein dynamics, power grids, neural networks, quantum spin chains), we:

1. Compute the **coupling tensor** (Hessian of the energy functional, reshaped to expose multi-modal structure)
2. Map its **multilinear rank** across the parameter/state space
3. Use **A* search** through coordinate transformations to find the representation that minimizes rank
4. Use rank as a **predictor** of tractability (periodicity, stability, compressibility)

## Results So Far

### 3-Body Problem (primary validation)

| Tensor Rank | Periodic Orbits | Total Tested | Rate |
|------------|----------------|--------------|------|
| 7 | 43 | 43 | **100%** |
| 8 | 24 | 24 | **100%** |
| 9 | 692 | 5,887 | 11.8% |
| 10 | 581 | 2,783 | 20.9% |

- Phase 1: Rank landscape atlas across 3.24M configurations for 3 mass ratios
- Phase 2: Tucker decomposition validates known solutions (Lagrange, Euler, figure-eight)
- Phase 3: A* search rediscovers optimal Jacobi decomposition from first principles
- GPU orbit search: 10,000 simultaneous integrations on GV100, 100% periodicity at low rank

## Structure

```
tensor_3body/
    hamiltonian.py      # 3-body Hamiltonian in Jacobi coords (analytical Hessian)
    tensor_ops.py       # Tucker decomposition, multilinear rank, mode coupling
    sampling.py         # Configuration space sampling
    landscape.py        # Parallel rank landscape computation
    transforms.py       # Coordinate transforms for A* search
    known_solutions.py  # Lagrange, Euler, figure-eight, hierarchical
    integrator.py       # Symplectic leapfrog (CPU)
    integrator_gpu.py   # Batch GPU integrator (CuPy)
experiments/
    molecular/          # Protein coupling tensor analysis
    power_grid/         # IEEE bus system Jacobian analysis
    neural_net/         # Weight tensor rank analysis
    quantum/            # Heisenberg chain coupling tensor
tests/
    test_hamiltonian.py # 11 tests validating against known physics
run_phase1.py           # Rank landscape mapping
run_phase2.py           # Known solution validation (with Tucker)
run_phase3.py           # A* decomposition search
run_orbit_search_gpu.py # GPU-accelerated periodic orbit search
```

## Quick Start

```bash
pip install numpy scipy tensorly cupy-cuda12x matplotlib h5py
python -m pytest tests/ -v          # 11 tests
python run_phase1.py --resolution coarse --scenario equal
python run_phase2.py                # Tucker analysis of known solutions
python run_phase3.py --sample 200   # A* search
CUDA_VISIBLE_DEVICES=0 python run_orbit_search_gpu.py \
    --landscape results/phase1/landscape_equal_static.npz
```

## Infrastructure

Developed on Atlas (HP Z840): 2x Quadro GV100 32GB, 48 CPU cores, 128GB RAM.
GPU integrator processes 10,000 orbits simultaneously (~100x CPU speedup).

## Publication

Theory paper in preparation: *"Tensor Rank as a Predictor of Dynamical Tractability: Automatic Decomposition of Coupled Systems via A* Search"*

## License

MIT
