"""
Symplectic integrator for the 3-body problem.

Leapfrog (Stormer-Verlet) integrator that preserves the Hamiltonian
structure. Used to check whether low-rank configurations lie near
periodic orbits.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from .hamiltonian import hamiltonian, jacobi_masses, _body_positions_from_jacobi


def _acceleration(
    rho1: NDArray, rho2: NDArray,
    m1: float, m2: float, m3: float,
) -> tuple[NDArray, NDArray]:
    """Compute accelerations in Jacobi coordinates.

    Returns (a1, a2) = (d^2 rho1/dt^2, d^2 rho2/dt^2).
    """
    mu1, mu2 = jacobi_masses(m1, m2, m3)
    q1, q2, q3 = _body_positions_from_jacobi(rho1, rho2, m1, m2, m3)

    M12 = m1 + m2
    M_total = m1 + m2 + m3

    # Forces on bodies (F_ij = -G m_i m_j (q_i - q_j) / |q_i - q_j|^3)
    def grav_accel(qi, qj, mj):
        r = qj - qi
        rnorm = np.linalg.norm(r)
        if rnorm < 1e-20:
            return np.zeros(3)
        return mj * r / rnorm**3

    # Accelerations of bodies
    a_q1 = grav_accel(q1, q2, m2) + grav_accel(q1, q3, m3)
    a_q2 = grav_accel(q2, q1, m1) + grav_accel(q2, q3, m3)
    a_q3 = grav_accel(q3, q1, m1) + grav_accel(q3, q2, m2)

    # Convert to Jacobi accelerations
    # d^2 rho1/dt^2 = a_q2 - a_q1
    # d^2 rho2/dt^2 = a_q3 - (m1*a_q1 + m2*a_q2)/M12
    a_rho1 = a_q2 - a_q1
    a_rho2 = a_q3 - (m1 * a_q1 + m2 * a_q2) / M12

    return a_rho1, a_rho2


def leapfrog_step(
    z: NDArray, dt: float,
    m1: float, m2: float, m3: float,
) -> NDArray:
    """Single leapfrog step.

    Kick-Drift-Kick (KDK) variant for better energy conservation.
    """
    mu1, mu2 = jacobi_masses(m1, m2, m3)
    rho1 = z[0:3].copy()
    rho2 = z[3:6].copy()
    pi1 = z[6:9].copy()
    pi2 = z[9:12].copy()

    # Half kick
    a1, a2 = _acceleration(rho1, rho2, m1, m2, m3)
    pi1 += 0.5 * dt * mu1 * a1
    pi2 += 0.5 * dt * mu2 * a2

    # Full drift
    rho1 += dt * pi1 / mu1
    rho2 += dt * pi2 / mu2

    # Half kick
    a1, a2 = _acceleration(rho1, rho2, m1, m2, m3)
    pi1 += 0.5 * dt * mu1 * a1
    pi2 += 0.5 * dt * mu2 * a2

    return np.concatenate([rho1, rho2, pi1, pi2])


def integrate(
    z0: NDArray,
    m1: float, m2: float, m3: float,
    dt: float = 0.01,
    n_steps: int = 10000,
    save_every: int = 10,
) -> dict:
    """Integrate the 3-body system and return trajectory + diagnostics.

    Returns:
        Dictionary with:
            trajectory: (n_saved, 12) array of phase space snapshots
            times: (n_saved,) time values
            energy: (n_saved,) Hamiltonian values (should be conserved)
            return_distance: minimum distance to initial state in phase space
            return_time: time of closest return
            is_periodic: True if return_distance / initial_extent < threshold
    """
    z = z0.copy()
    H0 = hamiltonian(z, m1, m2, m3)

    n_saved = n_steps // save_every + 1
    trajectory = np.zeros((n_saved, 12))
    times = np.zeros(n_saved)
    energy = np.zeros(n_saved)

    trajectory[0] = z
    times[0] = 0.0
    energy[0] = H0

    # Track closest return to initial state
    initial_extent = np.linalg.norm(z0[:6])  # characteristic scale
    min_return_dist = np.inf
    min_return_time = 0.0
    save_idx = 1

    for step in range(1, n_steps + 1):
        z = leapfrog_step(z, dt, m1, m2, m3)

        if step % save_every == 0 and save_idx < n_saved:
            trajectory[save_idx] = z
            times[save_idx] = step * dt
            energy[save_idx] = hamiltonian(z, m1, m2, m3)
            save_idx += 1

        # Check return distance (skip first 10% to avoid trivial near-returns)
        if step > n_steps // 10:
            dist = np.linalg.norm(z - z0)
            if dist < min_return_dist:
                min_return_dist = dist
                min_return_time = step * dt

    # Check for close collisions (NaN detection)
    if np.any(np.isnan(z)):
        return {
            "trajectory": trajectory[:save_idx],
            "times": times[:save_idx],
            "energy": energy[:save_idx],
            "return_distance": np.inf,
            "return_time": 0.0,
            "is_periodic": False,
            "collision": True,
            "energy_error": np.inf,
        }

    energy_error = abs(energy[save_idx - 1] - H0) / (abs(H0) + 1e-30)
    threshold = 0.05  # 5% of initial extent
    is_periodic = (min_return_dist / (initial_extent + 1e-30)) < threshold

    return {
        "trajectory": trajectory[:save_idx],
        "times": times[:save_idx],
        "energy": energy[:save_idx],
        "return_distance": min_return_dist,
        "return_time": min_return_time,
        "is_periodic": is_periodic,
        "collision": False,
        "energy_error": energy_error,
    }
