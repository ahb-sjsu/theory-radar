"""
Configuration space sampling for the 3-body tensor landscape.

Generates grids of Jacobi configurations parameterized by
(r1, r2, theta2, phi2) and converts to full 12D phase space vectors.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def make_config_grid(
    n_r1: int = 50,
    n_r2: int = 50,
    n_theta: int = 36,
    n_phi: int = 36,
    r_min: float = 0.1,
    r_max: float = 100.0,
) -> NDArray:
    """Generate a 4D grid of Jacobi configurations.

    Parameterization:
        rho1 = (r1, 0, 0)           -- inner pair along x-axis
        rho2 = r2 * (sin(theta)*cos(phi), sin(theta)*sin(phi), cos(theta))

    Args:
        n_r1, n_r2: Number of radial points (log-spaced).
        n_theta: Number of polar angle points.
        n_phi: Number of azimuthal angle points.
        r_min, r_max: Radial range.

    Returns:
        Array of shape (N, 4) with columns [r1, r2, theta, phi].
    """
    r1_vals = np.logspace(np.log10(r_min), np.log10(r_max), n_r1)
    r2_vals = np.logspace(np.log10(r_min), np.log10(r_max), n_r2)
    theta_vals = np.linspace(0.01, np.pi - 0.01, n_theta)  # avoid poles
    phi_vals = np.linspace(0, 2 * np.pi, n_phi, endpoint=False)

    # Meshgrid and flatten
    R1, R2, TH, PH = np.meshgrid(r1_vals, r2_vals, theta_vals, phi_vals,
                                   indexing="ij")
    configs = np.column_stack([
        R1.ravel(), R2.ravel(), TH.ravel(), PH.ravel()
    ])
    return configs


def make_coarse_grid(
    n_r: int = 20,
    n_angle: int = 12,
    r_min: float = 0.3,
    r_max: float = 30.0,
) -> NDArray:
    """Generate a coarse grid for quick exploration.

    Returns:
        Array of shape (N, 4) with columns [r1, r2, theta, phi].
    """
    return make_config_grid(n_r, n_r, n_angle, n_angle, r_min, r_max)


def config_to_phase_space(
    r1: float, r2: float, theta: float, phi: float,
    m1: float = 1.0, m2: float = 1.0, m3: float = 1.0,
) -> NDArray:
    """Convert (r1, r2, theta, phi) to a 12D phase space vector.

    Places bodies at the given configuration with zero momenta
    (instantaneously at rest). This samples the potential energy surface.

    Args:
        r1: Inner pair separation.
        r2: Third body distance from inner pair COM.
        theta, phi: Angular position of third body.
        m1, m2, m3: Body masses.

    Returns:
        Phase space vector z, shape (12,).
    """
    rho1 = np.array([r1, 0.0, 0.0])
    rho2 = r2 * np.array([
        np.sin(theta) * np.cos(phi),
        np.sin(theta) * np.sin(phi),
        np.cos(theta),
    ])
    pi1 = np.zeros(3)
    pi2 = np.zeros(3)

    return np.concatenate([rho1, rho2, pi1, pi2])


def config_to_phase_space_circular(
    r1: float, r2: float, theta: float, phi: float,
    m1: float = 1.0, m2: float = 1.0, m3: float = 1.0,
) -> NDArray:
    """Convert to phase space with circular orbit momenta.

    Sets momenta for approximately circular orbits of the inner pair
    and the outer body, providing a dynamically relevant configuration
    rather than zero-momentum snapshots.

    Returns:
        Phase space vector z, shape (12,).
    """
    from .hamiltonian import jacobi_masses

    mu1, mu2 = jacobi_masses(m1, m2, m3)

    rho1 = np.array([r1, 0.0, 0.0])
    rho2 = r2 * np.array([
        np.sin(theta) * np.cos(phi),
        np.sin(theta) * np.sin(phi),
        np.cos(theta),
    ])

    # Circular orbit velocity: v = sqrt(M/r) for the relevant mass combination
    # Inner pair
    v1 = np.sqrt(m1 * m2 / r1) if r1 > 0 else 0.0
    pi1 = mu1 * np.array([0.0, v1, 0.0])  # perpendicular to rho1

    # Outer body -- approximate
    M12 = m1 + m2
    v2 = np.sqrt(M12 * m3 / r2) if r2 > 0 else 0.0
    # Perpendicular to rho2 in the orbital plane
    rho2_hat = rho2 / (np.linalg.norm(rho2) + 1e-30)
    # Use cross product with z-axis to get perpendicular direction
    z_hat = np.array([0.0, 0.0, 1.0])
    perp = np.cross(z_hat, rho2_hat)
    perp_norm = np.linalg.norm(perp)
    if perp_norm > 1e-10:
        perp = perp / perp_norm
    else:
        perp = np.array([0.0, 1.0, 0.0])
    pi2 = mu2 * v2 * perp

    return np.concatenate([rho1, rho2, pi1, pi2])
