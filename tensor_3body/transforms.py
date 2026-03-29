"""
Coordinate transformations for the 3-body coupling tensor.

Each transform takes a 12D phase space vector z and returns a transformed z'
in a new coordinate system. The A* search explores sequences of these
transforms to minimize the coupling tensor's effective complexity.

Transforms are organized by type:
- Geometric: rotations, reflections, scalings
- Physical: Jacobi variants, rotating frames, regularization
- Algebraic: applied to the Hessian directly (diagonalization, decomposition)
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from typing import Callable


# Type for a coordinate transform: z -> z'
Transform = Callable[[NDArray], NDArray]


def _rotation_matrix_z(angle: float) -> NDArray:
    """3x3 rotation matrix around z-axis."""
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def _rotation_matrix_x(angle: float) -> NDArray:
    """3x3 rotation matrix around x-axis."""
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def _apply_rotation_to_phase(z: NDArray, R: NDArray) -> NDArray:
    """Apply a 3x3 rotation to all 3-vectors in phase space."""
    z_new = z.copy()
    z_new[0:3] = R @ z[0:3]    # rho1
    z_new[3:6] = R @ z[3:6]    # rho2
    z_new[6:9] = R @ z[6:9]    # pi1
    z_new[9:12] = R @ z[9:12]  # pi2
    return z_new


# ============================================================
# Geometric transforms
# ============================================================

def rotate_to_rho1_x(z: NDArray) -> NDArray:
    """Rotate so rho1 lies along the x-axis.

    This eliminates 2 angular DOF from rho1, potentially revealing
    block structure in the remaining coordinates.
    """
    rho1 = z[0:3]
    r = np.linalg.norm(rho1)
    if r < 1e-30:
        return z.copy()

    # Rotation to align rho1 with x-axis
    rho1_hat = rho1 / r
    x_hat = np.array([1.0, 0.0, 0.0])

    # Rodrigues' rotation formula
    v = np.cross(rho1_hat, x_hat)
    s = np.linalg.norm(v)
    c = np.dot(rho1_hat, x_hat)

    if s < 1e-14:  # Already aligned (or anti-aligned)
        if c > 0:
            return z.copy()
        else:
            R = np.diag([-1.0, 1.0, -1.0])
            return _apply_rotation_to_phase(z, R)

    vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    R = np.eye(3) + vx + vx @ vx * (1 - c) / (s * s)
    return _apply_rotation_to_phase(z, R)


def rotate_to_orbital_plane(z: NDArray) -> NDArray:
    """Rotate so the orbital plane lies in the x-y plane.

    Computes L = rho1 x pi1 + rho2 x pi2 and rotates L to z-axis.
    If the system is planar, this zeros all z-components.
    """
    rho1, rho2 = z[0:3], z[3:6]
    pi1, pi2 = z[6:9], z[9:12]

    L = np.cross(rho1, pi1) + np.cross(rho2, pi2)
    L_norm = np.linalg.norm(L)
    if L_norm < 1e-30:
        return z.copy()

    L_hat = L / L_norm
    z_hat = np.array([0.0, 0.0, 1.0])

    v = np.cross(L_hat, z_hat)
    s = np.linalg.norm(v)
    c = np.dot(L_hat, z_hat)

    if s < 1e-14:
        return z.copy()

    vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    R = np.eye(3) + vx + vx @ vx * (1 - c) / (s * s)
    return _apply_rotation_to_phase(z, R)


def scale_to_unit_energy(z: NDArray, m1: float = 1.0, m2: float = 1.0, m3: float = 1.0) -> NDArray:
    """Scale positions and momenta so |H| = 1.

    This normalizes the energy scale, making tensor magnitudes comparable
    across different configurations.
    """
    from .hamiltonian import hamiltonian
    H = hamiltonian(z, m1, m2, m3)
    if abs(H) < 1e-30:
        return z.copy()

    # Virial scaling: if we scale r -> alpha*r, p -> p/alpha,
    # then T -> T/alpha^2, V -> V/alpha, H -> T/alpha^2 + V/alpha
    # For simplicity, just scale positions
    alpha = abs(H) ** (-1.0 / 3.0)  # Approximate scaling
    z_new = z.copy()
    z_new[0:6] *= alpha       # positions
    z_new[6:12] /= alpha      # momenta
    return z_new


def reflect_rho2_to_upper(z: NDArray) -> NDArray:
    """Reflect so rho2_y >= 0 (canonical half-plane)."""
    z_new = z.copy()
    if z_new[4] < 0:  # rho2_y
        z_new[4] = -z_new[4]
        z_new[10] = -z_new[10]  # pi2_y
    return z_new


# ============================================================
# Physical transforms
# ============================================================

def to_rotating_frame(z: NDArray, omega: float | None = None,
                      m1: float = 1.0, m2: float = 1.0, m3: float = 1.0) -> NDArray:
    """Transform to a frame rotating with angular velocity omega about z-axis.

    If omega is None, compute the instantaneous angular velocity from
    the total angular momentum.
    """
    rho1, rho2 = z[0:3], z[3:6]
    pi1, pi2 = z[6:9], z[9:12]
    mu1, mu2 = (m1 * m2 / (m1 + m2)), (m3 * (m1 + m2) / (m1 + m2 + m3))

    if omega is None:
        L = np.cross(rho1, pi1) + np.cross(rho2, pi2)
        I = mu1 * np.dot(rho1, rho1) + mu2 * np.dot(rho2, rho2)
        omega = L[2] / I if I > 1e-30 else 0.0

    # In rotating frame, momenta get Coriolis correction:
    # pi_rot = pi - mu * omega x rho
    omega_vec = np.array([0, 0, omega])
    pi1_rot = pi1 - mu1 * np.cross(omega_vec, rho1)
    pi2_rot = pi2 - mu2 * np.cross(omega_vec, rho2)

    return np.concatenate([rho1, rho2, pi1_rot, pi2_rot])


def swap_jacobi_vectors(z: NDArray) -> NDArray:
    """Swap which pair is the 'inner' pair.

    Exchanges rho1 <-> rho2 and pi1 <-> pi2. This corresponds to
    choosing a different Jacobi decomposition (pair 1-3 instead of 1-2).
    """
    return np.concatenate([z[3:6], z[0:3], z[9:12], z[6:9]])


def rescale_inner(z: NDArray, factor: float = 2.0) -> NDArray:
    """Rescale inner pair coordinates by a factor.

    rho1 -> factor * rho1, pi1 -> pi1 / factor.
    Preserves Poisson bracket structure.
    """
    z_new = z.copy()
    z_new[0:3] *= factor
    z_new[6:9] /= factor
    return z_new


# ============================================================
# Algebraic transforms (applied to Hessian, not phase space)
# ============================================================

def diagonalize_qq_block(H: NDArray) -> tuple[NDArray, NDArray]:
    """Diagonalize the position-position block of the Hessian.

    Returns (H_rotated, rotation_matrix) where the qq block is diagonal.
    This finds the principal axes of the potential energy curvature.
    """
    qq = H[0:6, 0:6]
    eigenvalues, eigenvectors = np.linalg.eigh(qq)

    # Build full 12x12 rotation: rotate positions, leave momenta
    R = np.eye(12)
    R[0:6, 0:6] = eigenvectors.T

    H_rot = R @ H @ R.T
    return H_rot, R


def diagonalize_full(H: NDArray) -> tuple[NDArray, NDArray]:
    """Full eigendecomposition of the Hessian.

    Returns (eigenvalues, eigenvectors).
    """
    eigenvalues, eigenvectors = np.linalg.eigh(H)
    return eigenvalues, eigenvectors


# ============================================================
# Transform registry for A* search
# ============================================================

def mix_jacobi(z: NDArray, alpha: float = 0.5) -> NDArray:
    """Linear combination of Jacobi vectors: rho1' = alpha*rho1 + (1-alpha)*rho2.

    Explores intermediate decompositions between the two Jacobi trees.
    Preserves symplectic structure via canonical scaling.
    """
    z_new = z.copy()
    z_new[0:3] = alpha * z[0:3] + (1 - alpha) * z[3:6]
    z_new[3:6] = -((1 - alpha) * z[0:3]) + alpha * z[3:6]
    z_new[6:9] = alpha * z[6:9] + (1 - alpha) * z[9:12]
    z_new[9:12] = -((1 - alpha) * z[6:9]) + alpha * z[9:12]
    return z_new


def project_to_plane(z: NDArray) -> NDArray:
    """Project to the orbital plane (zero all z-components).

    Valid when angular momentum is along z-axis. Reduces effective
    dimensionality from 12 to 8.
    """
    z_new = z.copy()
    z_new[2] = 0.0   # rho1_z
    z_new[5] = 0.0   # rho2_z
    z_new[8] = 0.0   # pi1_z
    z_new[11] = 0.0  # pi2_z
    return z_new


def hyperbolic_rotation(z: NDArray, angle: float = 0.3) -> NDArray:
    """Hyperbolic rotation mixing position and momentum of inner pair.

    This is a canonical (symplectic) transform that mixes q and p:
    q' = q*cosh(a) + p*sinh(a)/mu, p' = q*mu*sinh(a) + p*cosh(a)
    Useful for exploring phase-space rotations that diagonalize the flow.
    """
    ch, sh = np.cosh(angle), np.sinh(angle)
    z_new = z.copy()
    z_new[0:3] = ch * z[0:3] + sh * z[6:9]
    z_new[6:9] = sh * z[0:3] + ch * z[6:9]
    return z_new


def center_of_mass_offset(z: NDArray, delta: float = 0.1) -> NDArray:
    """Shift the Jacobi origin slightly toward/away from inner pair COM.

    Explores whether a slightly different decomposition center reduces coupling.
    """
    z_new = z.copy()
    rho1_hat = z[0:3] / (np.linalg.norm(z[0:3]) + 1e-30)
    z_new[3:6] += delta * rho1_hat
    return z_new


def time_reversal(z: NDArray) -> NDArray:
    """Reverse all momenta (time reversal symmetry)."""
    z_new = z.copy()
    z_new[6:12] = -z[6:12]
    return z_new


def rotate_z_45(z: NDArray) -> NDArray:
    """Rotate 45 degrees about z-axis."""
    R = _rotation_matrix_z(np.pi / 4)
    return _apply_rotation_to_phase(z, R)


def rotate_x_90(z: NDArray) -> NDArray:
    """Rotate 90 degrees about x-axis (tilt orbital plane)."""
    R = _rotation_matrix_x(np.pi / 2)
    return _apply_rotation_to_phase(z, R)


def get_transform_registry() -> list[dict]:
    """Return all available transforms with metadata.

    Each entry has:
        name: human-readable name
        fn: callable(z) -> z' (or callable(z, **kwargs) -> z')
        type: 'geometric', 'physical', 'algebraic'
        cost: estimated computational cost (1=cheap, 10=expensive)
    """
    return [
        # Geometric
        {"name": "align_rho1_x", "fn": rotate_to_rho1_x,
         "type": "geometric", "cost": 1},
        {"name": "align_orbital_plane", "fn": rotate_to_orbital_plane,
         "type": "geometric", "cost": 1},
        {"name": "reflect_upper", "fn": reflect_rho2_to_upper,
         "type": "geometric", "cost": 1},
        {"name": "rotate_z_45", "fn": rotate_z_45,
         "type": "geometric", "cost": 1},
        {"name": "rotate_x_90", "fn": rotate_x_90,
         "type": "geometric", "cost": 1},
        {"name": "project_plane", "fn": project_to_plane,
         "type": "geometric", "cost": 1},

        # Physical / canonical
        {"name": "swap_jacobi", "fn": swap_jacobi_vectors,
         "type": "physical", "cost": 1},
        {"name": "rotating_frame", "fn": to_rotating_frame,
         "type": "physical", "cost": 2},
        {"name": "rescale_inner_2x", "fn": lambda z: rescale_inner(z, 2.0),
         "type": "physical", "cost": 1},
        {"name": "rescale_inner_0.5x", "fn": lambda z: rescale_inner(z, 0.5),
         "type": "physical", "cost": 1},
        {"name": "rescale_inner_3x", "fn": lambda z: rescale_inner(z, 3.0),
         "type": "physical", "cost": 1},
        {"name": "rescale_inner_0.33x", "fn": lambda z: rescale_inner(z, 1.0/3.0),
         "type": "physical", "cost": 1},
        {"name": "mix_jacobi_0.3", "fn": lambda z: mix_jacobi(z, 0.3),
         "type": "physical", "cost": 2},
        {"name": "mix_jacobi_0.7", "fn": lambda z: mix_jacobi(z, 0.7),
         "type": "physical", "cost": 2},
        {"name": "hyp_rot_0.3", "fn": lambda z: hyperbolic_rotation(z, 0.3),
         "type": "physical", "cost": 2},
        {"name": "hyp_rot_-0.3", "fn": lambda z: hyperbolic_rotation(z, -0.3),
         "type": "physical", "cost": 2},
        {"name": "com_offset_+", "fn": lambda z: center_of_mass_offset(z, 0.1),
         "type": "physical", "cost": 1},
        {"name": "com_offset_-", "fn": lambda z: center_of_mass_offset(z, -0.1),
         "type": "physical", "cost": 1},
        {"name": "time_reversal", "fn": time_reversal,
         "type": "physical", "cost": 1},
    ]
