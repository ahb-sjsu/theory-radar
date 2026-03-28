"""
Gravitational 3-body Hamiltonian in Jacobi coordinates.

Provides H, gradient, and Hessian for the reduced (COM-eliminated) system.
The 12D phase space is (rho_1, rho_2, pi_1, pi_2) where rho_i are Jacobi
position vectors and pi_i are conjugate momenta.

Convention: z = [rho_1x, rho_1y, rho_1z, rho_2x, rho_2y, rho_2z,
                 pi_1x,  pi_1y,  pi_1z,  pi_2x,  pi_2y,  pi_2z]

Units: G = 1, lengths and masses in natural units.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def jacobi_masses(m1: float, m2: float, m3: float) -> tuple[float, float]:
    """Compute reduced masses for Jacobi coordinates.

    Returns (mu1, mu2) where:
        mu1 = m1*m2 / (m1+m2)           -- reduced mass of inner pair
        mu2 = m3*(m1+m2) / (m1+m2+m3)   -- reduced mass of outer body
    """
    mu1 = m1 * m2 / (m1 + m2)
    mu2 = m3 * (m1 + m2) / (m1 + m2 + m3)
    return mu1, mu2


def _body_positions_from_jacobi(
    rho1: NDArray, rho2: NDArray,
    m1: float, m2: float, m3: float,
) -> tuple[NDArray, NDArray, NDArray]:
    """Convert Jacobi vectors to body positions (COM frame).

    rho1 = q2 - q1
    rho2 = q3 - (m1*q1 + m2*q2)/(m1+m2)

    Returns (q1, q2, q3) each shape (3,).
    """
    M12 = m1 + m2
    # In COM frame: m1*q1 + m2*q2 + m3*q3 = 0
    # q2 - q1 = rho1
    # q3 - (m1*q1 + m2*q2)/M12 = rho2
    # Solving:
    #   q1 = -m2/M12 * rho1 - m3/M_total * rho2
    #   q2 =  m1/M12 * rho1 - m3/M_total * rho2
    #   q3 = M12/M_total * rho2
    M_total = m1 + m2 + m3
    q1 = -(m2 / M12) * rho1 - (m3 / M_total) * rho2
    q2 = (m1 / M12) * rho1 - (m3 / M_total) * rho2
    q3 = (M12 / M_total) * rho2
    return q1, q2, q3


def hamiltonian(
    z: NDArray,
    m1: float = 1.0, m2: float = 1.0, m3: float = 1.0,
) -> float:
    """Compute the Hamiltonian H(z) in Jacobi coordinates.

    Args:
        z: Phase space vector, shape (12,).
           [rho1(3), rho2(3), pi1(3), pi2(3)]
        m1, m2, m3: Body masses.

    Returns:
        Scalar energy value.
    """
    rho1 = z[0:3]
    rho2 = z[3:6]
    pi1 = z[6:9]
    pi2 = z[9:12]

    mu1, mu2 = jacobi_masses(m1, m2, m3)

    # Kinetic energy
    T = np.dot(pi1, pi1) / (2 * mu1) + np.dot(pi2, pi2) / (2 * mu2)

    # Potential energy: need body positions
    q1, q2, q3 = _body_positions_from_jacobi(rho1, rho2, m1, m2, m3)

    r12 = np.linalg.norm(q2 - q1)
    r13 = np.linalg.norm(q3 - q1)
    r23 = np.linalg.norm(q3 - q2)

    # Guard against singularity
    eps = 1e-30
    V = -(m1 * m2 / max(r12, eps)
          + m1 * m3 / max(r13, eps)
          + m2 * m3 / max(r23, eps))

    return T + V


def gradient(
    z: NDArray,
    m1: float = 1.0, m2: float = 1.0, m3: float = 1.0,
) -> NDArray:
    """Compute dH/dz via central finite differences.

    Args:
        z: Phase space vector, shape (12,).
        m1, m2, m3: Body masses.

    Returns:
        Gradient vector, shape (12,).
    """
    h = 1e-7
    grad = np.zeros(12)
    for k in range(12):
        z_plus = z.copy()
        z_minus = z.copy()
        z_plus[k] += h
        z_minus[k] -= h
        grad[k] = (hamiltonian(z_plus, m1, m2, m3)
                    - hamiltonian(z_minus, m1, m2, m3)) / (2 * h)
    return grad


def hessian(
    z: NDArray,
    m1: float = 1.0, m2: float = 1.0, m3: float = 1.0,
) -> NDArray:
    """Compute d^2 H / (dz_a dz_b) via central finite differences.

    Args:
        z: Phase space vector, shape (12,).
        m1, m2, m3: Body masses.

    Returns:
        Hessian matrix, shape (12, 12). Symmetric.
    """
    h = 1e-5
    H = np.zeros((12, 12))
    H0 = hamiltonian(z, m1, m2, m3)

    for a in range(12):
        for b in range(a, 12):
            z_pp = z.copy()
            z_pm = z.copy()
            z_mp = z.copy()
            z_mm = z.copy()

            z_pp[a] += h; z_pp[b] += h
            z_pm[a] += h; z_pm[b] -= h
            z_mp[a] -= h; z_mp[b] += h
            z_mm[a] -= h; z_mm[b] -= h

            H[a, b] = (hamiltonian(z_pp, m1, m2, m3)
                        - hamiltonian(z_pm, m1, m2, m3)
                        - hamiltonian(z_mp, m1, m2, m3)
                        + hamiltonian(z_mm, m1, m2, m3)) / (4 * h * h)
            H[b, a] = H[a, b]

    return H


def hessian_analytical(
    z: NDArray,
    m1: float = 1.0, m2: float = 1.0, m3: float = 1.0,
) -> NDArray:
    """Compute d^2 H / (dz_a dz_b) analytically.

    The kinetic part is diagonal: d^2 T / dpi_a dpi_b = delta_ab / mu_i.
    The potential part requires derivatives of 1/|q_i - q_j| w.r.t. Jacobi coords.

    Returns:
        Hessian matrix, shape (12, 12).
    """
    rho1 = z[0:3]
    rho2 = z[3:6]

    mu1, mu2 = jacobi_masses(m1, m2, m3)
    M12 = m1 + m2
    M_total = m1 + m2 + m3

    # Body positions from Jacobi
    q1, q2, q3 = _body_positions_from_jacobi(rho1, rho2, m1, m2, m3)

    # Pairwise separation vectors and distances
    r12_vec = q2 - q1  # = rho1
    r13_vec = q3 - q1
    r23_vec = q3 - q2

    r12 = np.linalg.norm(r12_vec)
    r13 = np.linalg.norm(r13_vec)
    r23 = np.linalg.norm(r23_vec)

    # d^2/dx_a dx_b (1/|r|) = (3 r_a r_b - |r|^2 delta_ab) / |r|^5
    def d2_inv_r(rvec: NDArray, rnorm: float) -> NDArray:
        """Second derivative of -1/|r| w.r.t. components of r. Shape (3,3)."""
        r5 = rnorm**5
        I = np.eye(3)
        return (3 * np.outer(rvec, rvec) - rnorm**2 * I) / r5

    # Potential Hessian in body coordinates:
    # V = -m1*m2/r12 - m1*m3/r13 - m2*m3/r23
    # d^2V / d(q_i)_mu d(q_j)_nu requires chain rule through Jacobi transform.

    # Jacobian: d(q_i)/d(rho_k) -- each is a 3x3 block (scalar * I_3)
    # dq1/drho1 = -m2/M12 * I,   dq1/drho2 = -m3/M_total * I
    # dq2/drho1 =  m1/M12 * I,   dq2/drho2 = -m3/M_total * I
    # dq3/drho1 =  0,             dq3/drho2 =  M12/M_total * I

    # Coefficients (scalars multiplying I_3 in the Jacobian blocks)
    J = np.zeros((3, 2))  # J[body, jacobi_vec]
    J[0, 0] = -m2 / M12
    J[0, 1] = -m3 / M_total
    J[1, 0] = m1 / M12
    J[1, 1] = -m3 / M_total
    J[2, 0] = 0.0
    J[2, 1] = M12 / M_total

    # d^2 V / d(rho_k)_mu d(rho_l)_nu
    # = sum_{i<j} (-m_i m_j) * sum over chain rule terms
    # For pair (i,j), separation r_ij = q_j - q_i, so
    # d(r_ij)/d(rho_k) = (J[j,k] - J[i,k]) * I_3
    # d^2 V_{pair} / d(rho_k)_mu d(rho_l)_nu
    #   = -m_i*m_j * (J[j,k]-J[i,k]) * (J[j,l]-J[i,l]) * d2_inv_r(r_ij, |r_ij|)_mu_nu

    pairs = [(0, 1, m1, m2, r12_vec, r12),
             (0, 2, m1, m3, r13_vec, r13),
             (1, 2, m2, m3, r23_vec, r23)]

    # Build the 6x6 position-position block of the Hessian
    # Indices: rho1 = 0:3, rho2 = 3:6
    Hqq = np.zeros((6, 6))
    for i, j, mi, mj, rvec, rnorm in pairs:
        d2 = d2_inv_r(rvec, rnorm)  # (3, 3)
        for k in range(2):  # Jacobi index k
            for l in range(2):  # Jacobi index l
                coeff = -mi * mj * (J[j, k] - J[i, k]) * (J[j, l] - J[i, l])
                Hqq[3*k:3*k+3, 3*l:3*l+3] += coeff * d2

    # Full 12x12 Hessian
    H = np.zeros((12, 12))

    # Position-position block (0:6, 0:6)
    H[0:6, 0:6] = Hqq

    # Momentum-momentum block (6:12, 6:12) -- from kinetic energy
    # d^2 T / dpi1_mu dpi1_nu = delta_mu_nu / mu1
    # d^2 T / dpi2_mu dpi2_nu = delta_mu_nu / mu2
    H[6:9, 6:9] = np.eye(3) / mu1
    H[9:12, 9:12] = np.eye(3) / mu2

    # Position-momentum cross terms are zero (H separates as T(p) + V(q))

    return H
