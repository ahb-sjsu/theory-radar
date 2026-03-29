# Zenodo Data Package: Tensor Rank and Spectral Rigidity in the 3-Body Problem

Andrew H. Bond, San Jose State University
andrew.bond@sjsu.edu

## Paper
`preprint.pdf` — "Keplerian Eigenvalue Structure and Spectral Rigidity in the Gravitational Three-Body Coupling Tensor"

## Code
Complete source code at: https://github.com/ahb-sjsu/tensor-tractability

## Datasets

### landscape_equal_static.npz
Coupling tensor rank computed at 3,240,000 configurations (equal masses).
Keys: r1, r2, theta, phi, singvals, eff_rank, participation_ratio, is_block_diagonal, is_separable, cross_body_q

### prediction_test.npz (10K configs)
Tensor features + periodicity ground truth from GPU integration.
Keys: r1, r2, theta, phi, gammas, ranks, freq_ratios, n_clusters, is_periodic, collision, return_distance

### prediction_large.npz (51K configs, 3 mass ratios)
Large-scale prediction test across equal, binary+light, and hierarchical mass ratios.
Keys: ranks, inner_ranks, gammas, freq_ratios, n_clusters, energies, r2_r1, is_separable, scenarios, periodic, collision, return_distance

### massive_orbit_search.npz (100K orbits)
GPU-integrated orbits sampled from the full landscape.
Keys: ranks, periodic, collision, return_distance

### phase4_rank910.npz
Boundary search at rank 9-10 configurations.
Keys: return_distance, is_periodic, collision, ranks, r1, r2

### four_body_results.npz
4-body generalization test (3,000 configs).
Keys: ranks, periodic, collision

## Reproduction
```bash
pip install numpy scipy tensorly cupy-cuda12x matplotlib h5py scikit-learn
git clone https://github.com/ahb-sjsu/tensor-tractability.git
cd tensor-tractability
python -m pytest tests/ -v
python run_phase1.py --resolution fine --scenario equal
python run_phase2.py
CUDA_VISIBLE_DEVICES=0 python run_orbit_search_gpu.py --landscape results/phase1/landscape_equal_static.npz
```
