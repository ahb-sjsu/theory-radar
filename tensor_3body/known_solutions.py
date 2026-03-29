"""
Known solutions to the 3-body problem for Phase 2 validation.

Provides initial conditions for:
- Lagrange equilateral triangle (L4/L5)
- Euler collinear configurations (L1/L2/L3)
- Figure-eight choreography (Chenciner-Montgomery, 2000)
- Broucke-Hadjidemetriou-Henon family (selected members)
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from .hamiltonian import jacobi_masses, _body_positions_from_jacobi


def _bodies_to_jacobi(
    q1: NDArray, q2: NDArray, q3: NDArray,
    p1: NDArray, p2: NDArray, p3: NDArray,
    m1: float, m2: float, m3: float,
) -> NDArray:
    """Convert body positions/momenta to Jacobi phase space vector."""
    M12 = m1 + m2
    M_total = m1 + m2 + m3

    rho1 = q2 - q1
    rho2 = q3 - (m1 * q1 + m2 * q2) / M12

    # Jacobi momenta from body momenta:
    # pi1 = mu1 * d(rho1)/dt = mu1 * (p2/m2 - p1/m1)
    # pi2 = mu2 * d(rho2)/dt = mu2 * (p3/m3 - (p1+p2)/M12)
    mu1, mu2 = jacobi_masses(m1, m2, m3)
    pi1 = mu1 * (p2 / m2 - p1 / m1)
    pi2 = mu2 * (p3 / m3 - (p1 + p2) / M12)

    return np.concatenate([rho1, rho2, pi1, pi2])


def lagrange_equilateral(
    m1: float = 1.0, m2: float = 1.0, m3: float = 1.0,
    R: float = 1.0,
) -> NDArray:
    """Lagrange equilateral triangle configuration with circular orbit velocities.

    All three bodies at vertices of an equilateral triangle of side R,
    rotating about the common center of mass.

    Returns: Jacobi phase space vector, shape (12,).
    """
    M = m1 + m2 + m3

    # Place bodies at equilateral triangle vertices in COM frame
    # Body 1 at angle 0, body 2 at 2pi/3, body 3 at 4pi/3
    angles = np.array([0, 2 * np.pi / 3, 4 * np.pi / 3])
    masses = np.array([m1, m2, m3])

    # Circumradius from side length R: R_circ = R / sqrt(3)
    R_circ = R / np.sqrt(3)

    positions = np.array([
        [R_circ * np.cos(a), R_circ * np.sin(a), 0.0] for a in angles
    ])

    # Shift to COM frame
    com = np.average(positions, axis=0, weights=masses)
    positions -= com

    q1, q2, q3 = positions[0], positions[1], positions[2]

    # Angular velocity for circular orbit: omega^2 * R_circ = G*M / R^2
    # (simplified for equilateral with G=1)
    omega = np.sqrt(M / R**3)

    # Velocities: v_i = omega x r_i (rotation about z-axis)
    def vel(q):
        return omega * np.array([-q[1], q[0], 0.0])

    p1 = m1 * vel(q1)
    p2 = m2 * vel(q2)
    p3 = m3 * vel(q3)

    return _bodies_to_jacobi(q1, q2, q3, p1, p2, p3, m1, m2, m3)


def euler_collinear(
    m1: float = 1.0, m2: float = 1.0, m3: float = 1.0,
    d12: float = 1.0, d23: float = 1.0,
) -> NDArray:
    """Euler collinear configuration (bodies on a line).

    Bodies placed along x-axis: q1 at origin, q2 at d12, q3 at d12+d23.
    Zero momenta (instantaneous configuration, not orbital).

    Returns: Jacobi phase space vector, shape (12,).
    """
    M = m1 + m2 + m3

    q1 = np.array([0.0, 0.0, 0.0])
    q2 = np.array([d12, 0.0, 0.0])
    q3 = np.array([d12 + d23, 0.0, 0.0])

    # Shift to COM
    com = (m1 * q1 + m2 * q2 + m3 * q3) / M
    q1 -= com
    q2 -= com
    q3 -= com

    p1 = np.zeros(3)
    p2 = np.zeros(3)
    p3 = np.zeros(3)

    return _bodies_to_jacobi(q1, q2, q3, p1, p2, p3, m1, m2, m3)


def euler_collinear_rotating(
    m1: float = 1.0, m2: float = 1.0, m3: float = 1.0,
    d12: float = 1.0, d23: float = 1.0,
) -> NDArray:
    """Euler collinear configuration with rigid rotation velocities.

    Returns: Jacobi phase space vector, shape (12,).
    """
    M = m1 + m2 + m3

    q1 = np.array([0.0, 0.0, 0.0])
    q2 = np.array([d12, 0.0, 0.0])
    q3 = np.array([d12 + d23, 0.0, 0.0])

    com = (m1 * q1 + m2 * q2 + m3 * q3) / M
    q1 -= com
    q2 -= com
    q3 -= com

    # Angular velocity for this configuration
    # V = -sum m_i m_j / |r_ij|, centripetal = omega^2 * sum m_i |r_i|^2
    r12 = np.linalg.norm(q2 - q1)
    r13 = np.linalg.norm(q3 - q1)
    r23 = np.linalg.norm(q3 - q2)

    V_mag = m1 * m2 / r12 + m1 * m3 / r13 + m2 * m3 / r23
    I = m1 * np.dot(q1, q1) + m2 * np.dot(q2, q2) + m3 * np.dot(q3, q3)

    omega = np.sqrt(V_mag / I) if I > 1e-30 else 0.0

    def vel(q):
        return omega * np.array([-q[1], q[0], 0.0])

    p1 = m1 * vel(q1)
    p2 = m2 * vel(q2)
    p3 = m3 * vel(q3)

    return _bodies_to_jacobi(q1, q2, q3, p1, p2, p3, m1, m2, m3)


def figure_eight(m: float = 1.0) -> NDArray:
    """Chenciner-Montgomery figure-eight choreography (2000).

    Three equal masses trace a single figure-eight curve.
    Initial conditions from Simó's high-precision values.

    Returns: Jacobi phase space vector, shape (12,).
    """
    # Simó's initial conditions (equal masses, period ~ 6.3259)
    # Body 1 starts at (-1, 0), body 2 at (1, 0), body 3 at (0, 0)
    # with specific velocities
    x3 = 0.0347753
    y3 = 0.5327490

    q1 = np.array([-1.0, 0.0, 0.0])
    q2 = np.array([1.0, 0.0, 0.0])
    q3 = np.array([0.0, 0.0, 0.0])

    p1 = m * np.array([x3, y3, 0.0])
    p2 = m * np.array([x3, y3, 0.0])
    p3 = m * np.array([-2 * x3, -2 * y3, 0.0])

    return _bodies_to_jacobi(q1, q2, q3, p1, p2, p3, m, m, m)


def hierarchical_triple(
    m1: float = 1.0, m2: float = 1.0, m3: float = 0.01,
    a_inner: float = 1.0, a_outer: float = 10.0,
) -> NDArray:
    """Hierarchical triple: tight binary + distant third body.

    Inner pair in circular orbit, outer body in circular orbit
    around the binary COM. This is the regime where perturbation
    theory works well.

    Returns: Jacobi phase space vector, shape (12,).
    """
    M12 = m1 + m2
    M_total = m1 + m2 + m3
    mu1, mu2 = jacobi_masses(m1, m2, m3)

    # Inner pair: circular orbit
    rho1 = np.array([a_inner, 0.0, 0.0])
    v1 = np.sqrt(m1 * m2 / (mu1 * a_inner))
    pi1 = mu1 * np.array([0.0, v1, 0.0])

    # Outer body: circular orbit around inner COM
    rho2 = np.array([0.0, a_outer, 0.0])  # perpendicular to inner
    v2 = np.sqrt(M12 * m3 / (mu2 * a_outer))
    pi2 = mu2 * np.array([v2, 0.0, 0.0])

    return np.concatenate([rho1, rho2, pi1, pi2])


def get_all_known_solutions() -> list[dict]:
    """Return all known solutions with metadata.

    Returns list of dicts with keys:
        name, z, masses, description, expected_properties
    """
    solutions = []

    # Lagrange equilateral -- equal masses
    solutions.append({
        "name": "lagrange_equal_R1",
        "z": lagrange_equilateral(1.0, 1.0, 1.0, R=1.0),
        "masses": (1.0, 1.0, 1.0),
        "description": "Equilateral triangle, equal masses, R=1",
        "expected": "Symmetric — expect rank reduction from 3-fold symmetry",
    })

    solutions.append({
        "name": "lagrange_equal_R5",
        "z": lagrange_equilateral(1.0, 1.0, 1.0, R=5.0),
        "masses": (1.0, 1.0, 1.0),
        "description": "Equilateral triangle, equal masses, R=5",
        "expected": "Same symmetry at larger scale",
    })

    # Lagrange -- unequal masses
    solutions.append({
        "name": "lagrange_unequal",
        "z": lagrange_equilateral(1.0, 0.5, 0.1, R=1.0),
        "masses": (1.0, 0.5, 0.1),
        "description": "Equilateral triangle, unequal masses",
        "expected": "Reduced symmetry — rank may be higher than equal-mass case",
    })

    # Euler collinear -- static
    solutions.append({
        "name": "euler_static",
        "z": euler_collinear(1.0, 1.0, 1.0),
        "masses": (1.0, 1.0, 1.0),
        "description": "Collinear, equal masses, static",
        "expected": "1D constraint — expect rank reduction along collinear axis",
    })

    # Euler collinear -- rotating
    solutions.append({
        "name": "euler_rotating",
        "z": euler_collinear_rotating(1.0, 1.0, 1.0),
        "masses": (1.0, 1.0, 1.0),
        "description": "Collinear, equal masses, rigid rotation",
        "expected": "Rotating equilibrium — should show stability structure",
    })

    # Figure-eight
    solutions.append({
        "name": "figure_eight",
        "z": figure_eight(1.0),
        "masses": (1.0, 1.0, 1.0),
        "description": "Chenciner-Montgomery figure-eight choreography",
        "expected": "High symmetry (cyclic permutation + time-shift) — expect low rank",
    })

    # Hierarchical triple -- well-separated
    solutions.append({
        "name": "hierarchical_a10",
        "z": hierarchical_triple(1.0, 1.0, 0.01, a_inner=1.0, a_outer=10.0),
        "masses": (1.0, 1.0, 0.01),
        "description": "Tight binary + distant light body, a_outer/a_inner=10",
        "expected": "Nearly decoupled — expect low rank (near 2-body + 2-body)",
    })

    solutions.append({
        "name": "hierarchical_a100",
        "z": hierarchical_triple(1.0, 1.0, 0.01, a_inner=1.0, a_outer=100.0),
        "masses": (1.0, 1.0, 0.01),
        "description": "Tight binary + very distant light body",
        "expected": "Strongly decoupled — expect very low rank",
    })

    # Hierarchical with comparable masses
    solutions.append({
        "name": "hierarchical_heavy",
        "z": hierarchical_triple(1.0, 1.0, 1.0, a_inner=1.0, a_outer=10.0),
        "masses": (1.0, 1.0, 1.0),
        "description": "Equal-mass hierarchical, a_outer/a_inner=10",
        "expected": "Weaker decoupling than light third body",
    })

    return solutions
