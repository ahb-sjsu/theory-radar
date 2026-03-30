"""Tests for the 3-body Hamiltonian and Hessian computation."""

import numpy as np
import pytest

from tensor_3body.hamiltonian import (
    hamiltonian,
    hessian,
    hessian_analytical,
    jacobi_masses,
    _body_positions_from_jacobi,
)


class TestJacobiMasses:
    def test_equal_masses(self):
        mu1, mu2 = jacobi_masses(1.0, 1.0, 1.0)
        assert mu1 == pytest.approx(0.5)
        assert mu2 == pytest.approx(2 / 3)

    def test_sun_jupiter(self):
        mu1, mu2 = jacobi_masses(1.0, 1e-3, 1.0)
        assert mu1 == pytest.approx(1e-3 / 1.001, rel=1e-3)


class TestBodyPositions:
    def test_com_is_zero(self):
        """Center of mass should be at origin."""
        rho1 = np.array([1.0, 0.0, 0.0])
        rho2 = np.array([0.0, 2.0, 0.0])
        m1, m2, m3 = 1.0, 1.0, 1.0
        q1, q2, q3 = _body_positions_from_jacobi(rho1, rho2, m1, m2, m3)
        com = m1 * q1 + m2 * q2 + m3 * q3
        np.testing.assert_allclose(com, 0.0, atol=1e-14)

    def test_rho1_is_separation(self):
        """rho1 should equal q2 - q1."""
        rho1 = np.array([3.0, 0.0, 0.0])
        rho2 = np.array([0.0, 5.0, 0.0])
        q1, q2, q3 = _body_positions_from_jacobi(rho1, rho2, 1.0, 2.0, 3.0)
        np.testing.assert_allclose(q2 - q1, rho1, atol=1e-14)


class TestHamiltonian:
    def test_two_body_limit(self):
        """When m3 -> 0 and rho2 -> inf, H should approach 2-body energy."""
        z = np.zeros(12)
        z[0] = 1.0  # rho1_x = 1 (unit separation)
        z[3] = 1000.0  # rho2_x very far
        H = hamiltonian(z, m1=1.0, m2=1.0, m3=1e-10)
        # Should be close to -m1*m2/r12 = -1.0
        assert H == pytest.approx(-1.0, abs=0.01)

    def test_energy_conservation_symmetry(self):
        """H should be invariant under rotation of the full system."""
        z = np.array([1.0, 0.0, 0.0, 0.5, 0.5, 0.0, 0.1, 0.2, 0.0, -0.1, 0.1, 0.0])
        H1 = hamiltonian(z)

        # Rotate 90 degrees around z-axis
        R = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]], dtype=float)
        z_rot = np.zeros(12)
        z_rot[0:3] = R @ z[0:3]
        z_rot[3:6] = R @ z[3:6]
        z_rot[6:9] = R @ z[6:9]
        z_rot[9:12] = R @ z[9:12]
        H2 = hamiltonian(z_rot)

        assert H1 == pytest.approx(H2, rel=1e-10)


class TestHessian:
    def test_analytical_matches_numerical(self):
        """Analytical and numerical Hessians should agree."""
        z = np.array([2.0, 0.0, 0.0, 0.0, 3.0, 0.0, 0.1, -0.1, 0.0, 0.0, 0.05, 0.0])
        H_num = hessian(z)
        H_ana = hessian_analytical(z)
        np.testing.assert_allclose(H_ana, H_num, atol=1e-4, rtol=1e-3)

    def test_symmetry(self):
        """Hessian should be symmetric."""
        z = np.array([1.5, 0.3, 0.0, -0.5, 2.0, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        H = hessian_analytical(z)
        np.testing.assert_allclose(H, H.T, atol=1e-14)

    def test_momentum_block_diagonal(self):
        """The pp block should be diagonal (kinetic energy is T = p^2/2mu)."""
        z = np.array([2.0, 0.0, 0.0, 0.0, 3.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        H = hessian_analytical(z)
        pp_block = H[6:12, 6:12]
        # Off-diagonal should be zero (pi1 doesn't couple to pi2)
        np.testing.assert_allclose(pp_block[0:3, 3:6], 0.0, atol=1e-14)
        np.testing.assert_allclose(pp_block[3:6, 0:3], 0.0, atol=1e-14)

    def test_qp_cross_terms_zero(self):
        """Position-momentum cross terms should be zero (H = T(p) + V(q))."""
        z = np.array([2.0, 0.0, 0.0, 0.0, 3.0, 0.0, 0.1, 0.0, 0.0, 0.0, 0.05, 0.0])
        H = hessian_analytical(z)
        qp_block = H[0:6, 6:12]
        np.testing.assert_allclose(qp_block, 0.0, atol=1e-14)


class TestTwoBodyDecomposition:
    def test_distant_third_body_decouples(self):
        """When third body is very far, coupling should be near-zero."""
        z = np.zeros(12)
        z[0] = 1.0  # inner pair at unit separation
        z[3] = 1e6  # third body extremely far away
        H = hessian_analytical(z, m1=1.0, m2=1.0, m3=1.0)

        # Cross-body position coupling should be tiny
        qq_12 = np.linalg.norm(H[0:3, 3:6])
        qq_11 = np.linalg.norm(H[0:3, 0:3])
        assert qq_12 / qq_11 < 1e-10
