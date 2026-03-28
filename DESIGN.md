# Technical Design: Phase 1 -- Tensor Landscape Mapping

## 1. Mathematical Setup

### 1.1 The 3-Body Hamiltonian

For masses m_1, m_2, m_3 with positions q_i in R^3 and momenta p_i in R^3:

    H = sum_i |p_i|^2 / (2 m_i)  -  G * sum_{i<j} m_i m_j / |q_i - q_j|

Full phase space: z = (q_1, q_2, q_3, p_1, p_2, p_3) in R^18.

### 1.2 Center-of-Mass Reduction

Jacobi coordinates eliminate 6 DOF (COM position + momentum):
    rho_1 = q_2 - q_1                    (relative vector, pair 1-2)
    rho_2 = q_3 - (m_1*q_1 + m_2*q_2)/(m_1+m_2)  (third body relative to COM of pair)

Reduced masses:
    mu_1 = m_1*m_2 / (m_1+m_2)
    mu_2 = m_3*(m_1+m_2) / (m_1+m_2+m_3)

Reduced phase space: (rho_1, rho_2, pi_1, pi_2) in R^12.

### 1.3 The Coupling Tensor

The Hessian of H in the reduced phase space:

    C_{ab} = d^2 H / (d z_a d z_b),   a,b = 1..12

This 12x12 matrix encodes the full linearized dynamics at each configuration.
It can be reshaped as a rank-6 tensor:

    M_{i,mu,alpha, j,nu,beta}

where:
    i,j in {1,2}     -- Jacobi body index
    mu,nu in {x,y,z}  -- spatial component
    alpha,beta in {q,p} -- position or momentum

Shape: (2, 3, 2, 2, 3, 2) = 144 entries (symmetric, so 78 independent).

### 1.4 Effective Rank

Given singular values sigma_1 >= sigma_2 >= ... >= sigma_12 of C_{ab}:

    effective_rank(epsilon) = |{k : sigma_k / sigma_1 > epsilon}|

where epsilon is a threshold (default: 1e-6).

Low effective rank at a configuration means the dynamics are nearly
decomposable into independent sub-problems at that point.

## 2. Sampling Strategy

### 2.1 Configuration Parameterization

After COM elimination, configurations are parameterized by:
    rho_1 = (r_1, theta_1, phi_1)   -- spherical coords for pair separation
    rho_2 = (r_2, theta_2, phi_2)   -- spherical coords for third body

Use angular momentum conservation to fix the orbital plane (phi_1 = 0,
theta_1 = pi/2), reducing to 4 effective parameters:
    (r_1, r_2, theta_2, phi_2)

### 2.2 Sampling Grid

    r_1:     logarithmic, 50 points in [0.1, 100] (units: G*M_total = 1)
    r_2:     logarithmic, 50 points in [0.1, 100]
    theta_2: uniform, 36 points in [0, pi]
    phi_2:   uniform, 36 points in [0, 2*pi]

Total: 50 * 50 * 36 * 36 = 3,240,000 configurations.
At ~0.1ms per Hessian evaluation: ~5.4 minutes on 1 core, ~7 seconds on 48 cores.

### 2.3 Mass Ratios

Run the full grid for:
    1. Equal masses: m_1 = m_2 = m_3 = 1 (most symmetric)
    2. Sun-Jupiter-Earth: m_1 = 1, m_2 = 1e-3, m_3 = 3e-6 (hierarchical)
    3. Binary + light: m_1 = m_2 = 1, m_3 = 0.01 (restricted-like)

## 3. Output

### 3.1 Data Format

HDF5 file per mass ratio:
    /config/r1          -- (N,) float64
    /config/r2          -- (N,) float64
    /config/theta2      -- (N,) float64
    /config/phi2        -- (N,) float64
    /tensor/hessian     -- (N, 12, 12) float64
    /tensor/singvals    -- (N, 12) float64
    /tensor/eff_rank    -- (N,) int32
    /metadata           -- masses, units, epsilon

### 3.2 Visualizations

1. Effective rank heatmap: (r_1, r_2) plane, averaged over angles
2. Effective rank heatmap: (theta_2, phi_2) plane at fixed r_1, r_2
3. Rank-1 regions highlighted (exactly decomposable)
4. Rank profile along known periodic orbits (Phase 2 overlay)

## 4. Implementation

### 4.1 Module Structure

    tensor_3body/
        __init__.py
        hamiltonian.py      -- H, gradient, Hessian in Jacobi coords
        tensor_ops.py       -- reshape to rank-6, effective rank, decomposition
        sampling.py         -- configuration grid generation
        landscape.py        -- parallel Hessian computation + rank mapping
        visualize.py        -- plotting functions
        integrator.py       -- symplectic leapfrog for orbit integration (Phase 2+)
        validate.py         -- check against known solutions (Phase 2)

### 4.2 Dependencies

    numpy, scipy, h5py, matplotlib, multiprocessing
    No GPU required for Phase 1 (pure linear algebra on 12x12 matrices).

### 4.3 Testing

    tests/
        test_hamiltonian.py  -- verify H, grad, Hessian against sympy symbolic
        test_two_body.py     -- 2-body limit: should be exactly rank-6 (fully decomposable)
        test_lagrange.py     -- Lagrange points: known stability eigenvalues
        test_symmetry.py     -- equal mass permutation symmetry of the tensor
