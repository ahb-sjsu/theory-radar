"""
GPU-accelerated batch orbit integrator for the 3-body problem.

Integrates thousands of orbits simultaneously on a single GPU using CuPy.
Each orbit is independent — embarrassingly parallel. The entire state
array (N_orbits x 12) is updated in one vectorized leapfrog step.

Requires: cupy (pip install cupy-cuda12x)
"""

from __future__ import annotations

import logging

import numpy as np
from numpy.typing import NDArray

log = logging.getLogger(__name__)

try:
    import cupy as cp
    HAS_CUPY = True
except ImportError:
    HAS_CUPY = False


def _body_positions_batch(
    rho1: "cp.ndarray", rho2: "cp.ndarray",
    m1: float, m2: float, m3: float,
) -> tuple:
    """Batch convert Jacobi -> body positions. All arrays shape (N, 3)."""
    M12 = m1 + m2
    M_total = m1 + m2 + m3
    q1 = -(m2 / M12) * rho1 - (m3 / M_total) * rho2
    q2 = (m1 / M12) * rho1 - (m3 / M_total) * rho2
    q3 = (M12 / M_total) * rho2
    return q1, q2, q3


def _acceleration_batch(
    rho1: "cp.ndarray", rho2: "cp.ndarray",
    m1: float, m2: float, m3: float,
) -> tuple:
    """Batch compute accelerations. All arrays shape (N, 3)."""
    q1, q2, q3 = _body_positions_batch(rho1, rho2, m1, m2, m3)

    def grav(qi, qj, mj):
        r = qj - qi
        rnorm = cp.linalg.norm(r, axis=1, keepdims=True)
        rnorm = cp.maximum(rnorm, 1e-20)
        return mj * r / (rnorm ** 3)

    a_q1 = grav(q1, q2, m2) + grav(q1, q3, m3)
    a_q2 = grav(q2, q1, m1) + grav(q2, q3, m3)
    a_q3 = grav(q3, q1, m1) + grav(q3, q2, m2)

    M12 = m1 + m2
    a_rho1 = a_q2 - a_q1
    a_rho2 = a_q3 - (m1 * a_q1 + m2 * a_q2) / M12

    return a_rho1, a_rho2


def _hamiltonian_batch(
    z: "cp.ndarray",
    m1: float, m2: float, m3: float,
    mu1: float, mu2: float,
) -> "cp.ndarray":
    """Batch Hamiltonian. z shape (N, 12). Returns (N,)."""
    rho1 = z[:, 0:3]
    rho2 = z[:, 3:6]
    pi1 = z[:, 6:9]
    pi2 = z[:, 9:12]

    T = cp.sum(pi1**2, axis=1) / (2 * mu1) + cp.sum(pi2**2, axis=1) / (2 * mu2)

    q1, q2, q3 = _body_positions_batch(rho1, rho2, m1, m2, m3)

    r12 = cp.linalg.norm(q2 - q1, axis=1)
    r13 = cp.linalg.norm(q3 - q1, axis=1)
    r23 = cp.linalg.norm(q3 - q2, axis=1)

    r12 = cp.maximum(r12, 1e-20)
    r13 = cp.maximum(r13, 1e-20)
    r23 = cp.maximum(r23, 1e-20)

    V = -(m1 * m2 / r12 + m1 * m3 / r13 + m2 * m3 / r23)

    return T + V


def integrate_batch(
    z0_np: NDArray,
    m1: float, m2: float, m3: float,
    dt: float = 0.005,
    n_steps: int = 50000,
    check_every: int = 1000,
    gpu_id: int = 0,
) -> dict:
    """Integrate N orbits simultaneously on GPU.

    Args:
        z0_np: Initial conditions, shape (N, 12), numpy array.
        m1, m2, m3: Masses.
        dt: Timestep.
        n_steps: Total integration steps.
        check_every: Check return distance every this many steps.
        gpu_id: Which GPU to use.

    Returns:
        Dictionary with numpy arrays:
            return_distance: (N,) min distance to initial state
            return_time: (N,) time of closest return
            is_periodic: (N,) bool
            energy_error: (N,) relative energy conservation error
            collision: (N,) bool
    """
    if not HAS_CUPY:
        raise ImportError("CuPy not available. Install with: pip install cupy-cuda12x")

    with cp.cuda.Device(gpu_id):
        N = z0_np.shape[0]
        mu1 = m1 * m2 / (m1 + m2)
        mu2 = m3 * (m1 + m2) / (m1 + m2 + m3)

        # Transfer to GPU
        z = cp.asarray(z0_np, dtype=cp.float64)
        z0 = z.copy()

        # Initial energy
        H0 = _hamiltonian_batch(z, m1, m2, m3, mu1, mu2)

        # Tracking arrays
        min_return_dist = cp.full(N, cp.inf)
        min_return_time = cp.zeros(N)
        collision = cp.zeros(N, dtype=cp.bool_)
        initial_extent = cp.linalg.norm(z0[:, :6], axis=1)

        log.info("GPU integrating %d orbits (dt=%.4f, steps=%d, T=%.1f) on GPU %d",
                 N, dt, n_steps, dt * n_steps, gpu_id)

        for step in range(1, n_steps + 1):
            rho1 = z[:, 0:3]
            rho2 = z[:, 3:6]
            pi1 = z[:, 6:9]
            pi2 = z[:, 9:12]

            # Leapfrog: half kick
            a1, a2 = _acceleration_batch(rho1, rho2, m1, m2, m3)
            pi1 = pi1 + 0.5 * dt * mu1 * a1
            pi2 = pi2 + 0.5 * dt * mu2 * a2

            # Full drift
            rho1 = rho1 + dt * pi1 / mu1
            rho2 = rho2 + dt * pi2 / mu2

            # Half kick
            a1, a2 = _acceleration_batch(rho1, rho2, m1, m2, m3)
            pi1 = pi1 + 0.5 * dt * mu1 * a1
            pi2 = pi2 + 0.5 * dt * mu2 * a2

            z[:, 0:3] = rho1
            z[:, 3:6] = rho2
            z[:, 6:9] = pi1
            z[:, 9:12] = pi2

            # Detect NaN (collisions)
            nan_mask = cp.any(cp.isnan(z), axis=1)
            collision |= nan_mask

            # Check return distance periodically (skip first 10%)
            if step % check_every == 0 and step > n_steps // 10:
                dist = cp.linalg.norm(z - z0, axis=1)
                improved = dist < min_return_dist
                min_return_dist = cp.where(improved, dist, min_return_dist)
                min_return_time = cp.where(improved, step * dt, min_return_time)

            if step % 10000 == 0:
                n_col = int(collision.sum())
                log.info("  step %d/%d, collisions: %d", step, n_steps, n_col)

        # Final energy
        H_final = _hamiltonian_batch(z, m1, m2, m3, mu1, mu2)
        energy_error = cp.abs(H_final - H0) / (cp.abs(H0) + 1e-30)

        # Periodicity check
        threshold = 0.05
        is_periodic = (min_return_dist / (initial_extent + 1e-30)) < threshold

        # Transfer back to CPU
        result = {
            "return_distance": cp.asnumpy(min_return_dist),
            "return_time": cp.asnumpy(min_return_time),
            "is_periodic": cp.asnumpy(is_periodic),
            "energy_error": cp.asnumpy(energy_error),
            "collision": cp.asnumpy(collision),
        }

        n_periodic = int(is_periodic.sum())
        n_collision = int(collision.sum())
        log.info("GPU integration complete: %d periodic, %d collisions out of %d",
                 n_periodic, n_collision, N)

        return result
