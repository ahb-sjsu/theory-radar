#!/usr/bin/env python3
"""
4-Body Generalization: Does tensor rank predict periodicity for N=4?

The 4-body coupling tensor has shape (3,3,2,3,3,2) = rank-6 in Jacobi
coordinates (3 Jacobi vectors x 3 spatial x 2 phase). The full phase
space is 18D (after COM elimination).

Tests the same pipeline: compute Hessian, measure rank, integrate,
check correlation.
"""

from __future__ import annotations

import logging
import time

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

try:
    import cupy as cp
    HAS_CUPY = True
except ImportError:
    HAS_CUPY = False


def jacobi_4body(m1, m2, m3, m4):
    """Reduced masses for 4-body Jacobi coordinates.

    Jacobi vectors:
        rho1 = q2 - q1 (inner pair)
        rho2 = q3 - COM(1,2) (third body)
        rho3 = q4 - COM(1,2,3) (fourth body)
    """
    M12 = m1 + m2
    M123 = m1 + m2 + m3
    M1234 = m1 + m2 + m3 + m4

    mu1 = m1 * m2 / M12
    mu2 = m3 * M12 / M123
    mu3 = m4 * M123 / M1234
    return mu1, mu2, mu3


def body_positions_4body(rho1, rho2, rho3, m1, m2, m3, m4):
    """Convert 3 Jacobi vectors to 4 body positions in COM frame."""
    M12 = m1 + m2
    M123 = m1 + m2 + m3
    M1234 = m1 + m2 + m3 + m4

    q1 = -(m2/M12)*rho1 - (m3/M123)*rho2 - (m4/M1234)*rho3
    q2 =  (m1/M12)*rho1 - (m3/M123)*rho2 - (m4/M1234)*rho3
    q3 =  (M12/M123)*rho2 - (m4/M1234)*rho3
    q4 =  (M123/M1234)*rho3
    return q1, q2, q3, q4


def hamiltonian_4body(z, m1, m2, m3, m4):
    """4-body Hamiltonian in Jacobi coordinates. z shape (18,)."""
    rho1, rho2, rho3 = z[0:3], z[3:6], z[6:9]
    pi1, pi2, pi3 = z[9:12], z[12:15], z[15:18]

    mu1, mu2, mu3 = jacobi_4body(m1, m2, m3, m4)

    T = (np.dot(pi1,pi1)/(2*mu1) + np.dot(pi2,pi2)/(2*mu2)
         + np.dot(pi3,pi3)/(2*mu3))

    q1, q2, q3, q4 = body_positions_4body(rho1, rho2, rho3, m1, m2, m3, m4)

    pairs = [(q1,q2,m1,m2), (q1,q3,m1,m3), (q1,q4,m1,m4),
             (q2,q3,m2,m3), (q2,q4,m2,m4), (q3,q4,m3,m4)]
    V = 0.0
    for qi, qj, mi, mj in pairs:
        r = np.linalg.norm(qj - qi)
        V -= mi * mj / max(r, 1e-20)

    return T + V


def hessian_4body(z, m1, m2, m3, m4):
    """Numerical Hessian of 4-body Hamiltonian. Returns (18,18)."""
    h = 1e-5
    N = 18
    H = np.zeros((N, N))
    for a in range(N):
        for b in range(a, N):
            zpp = z.copy(); zpp[a] += h; zpp[b] += h
            zpm = z.copy(); zpm[a] += h; zpm[b] -= h
            zmp = z.copy(); zmp[a] -= h; zmp[b] += h
            zmm = z.copy(); zmm[a] -= h; zmm[b] -= h
            H[a,b] = (hamiltonian_4body(zpp,m1,m2,m3,m4)
                      - hamiltonian_4body(zpm,m1,m2,m3,m4)
                      - hamiltonian_4body(zmp,m1,m2,m3,m4)
                      + hamiltonian_4body(zmm,m1,m2,m3,m4)) / (4*h*h)
            H[b,a] = H[a,b]
    return H


def make_4body_config(r1, r2, r3, m1, m2, m3, m4):
    """Create a 4-body config with circular orbit momenta."""
    mu1, mu2, mu3 = jacobi_4body(m1, m2, m3, m4)
    M12 = m1 + m2
    M123 = m1 + m2 + m3

    rho1 = np.array([r1, 0, 0])
    rho2 = np.array([0, r2, 0])
    rho3 = np.array([0, 0, r3])

    v1 = np.sqrt(m1*m2/(mu1*r1)) if r1 > 0 else 0
    v2 = np.sqrt(M12*m3/(mu2*r2)) if r2 > 0 else 0
    v3 = np.sqrt(M123*m4/(mu3*r3)) if r3 > 0 else 0

    pi1 = mu1 * np.array([0, v1, 0])
    pi2 = mu2 * np.array([v2, 0, 0])
    pi3 = mu3 * np.array([v3, v3/2, 0])

    return np.concatenate([rho1, rho2, rho3, pi1, pi2, pi3])


def integrate_4body_batch_gpu(z0_np, m1, m2, m3, m4, dt=0.005, n_steps=50000, gpu_id=0):
    """GPU batch integration for 4-body."""
    if not HAS_CUPY:
        raise ImportError("CuPy required")

    mu1, mu2, mu3 = jacobi_4body(m1, m2, m3, m4)
    M12 = m1 + m2
    M123 = m1 + m2 + m3
    M1234 = m1 + m2 + m3 + m4

    with cp.cuda.Device(gpu_id):
        N = z0_np.shape[0]
        z = cp.asarray(z0_np, dtype=cp.float64)
        z0 = z.copy()

        min_ret = cp.full(N, cp.inf)
        min_ret_t = cp.zeros(N)
        collision = cp.zeros(N, dtype=cp.bool_)
        extent = cp.linalg.norm(z0[:, :9], axis=1)

        log.info("GPU integrating %d 4-body orbits (T=%.0f)", N, dt*n_steps)

        for step in range(1, n_steps+1):
            r1, r2, r3 = z[:,0:3], z[:,3:6], z[:,6:9]
            p1, p2, p3 = z[:,9:12], z[:,12:15], z[:,15:18]

            # Body positions
            q1 = -(m2/M12)*r1 - (m3/M123)*r2 - (m4/M1234)*r3
            q2 =  (m1/M12)*r1 - (m3/M123)*r2 - (m4/M1234)*r3
            q3 =  (M12/M123)*r2 - (m4/M1234)*r3
            q4 =  (M123/M1234)*r3

            # Gravitational accelerations
            def grav(qi, qj, mj):
                rv = qj - qi
                rn = cp.linalg.norm(rv, axis=1, keepdims=True)
                rn = cp.maximum(rn, 1e-20)
                return mj * rv / rn**3

            aq1 = grav(q1,q2,m2) + grav(q1,q3,m3) + grav(q1,q4,m4)
            aq2 = grav(q2,q1,m1) + grav(q2,q3,m3) + grav(q2,q4,m4)
            aq3 = grav(q3,q1,m1) + grav(q3,q2,m2) + grav(q3,q4,m4)
            aq4 = grav(q4,q1,m1) + grav(q4,q2,m2) + grav(q4,q3,m3)

            # Jacobi accelerations
            a_r1 = aq2 - aq1
            a_r2 = aq3 - (m1*aq1 + m2*aq2)/M12
            a_r3 = aq4 - (m1*aq1 + m2*aq2 + m3*aq3)/M123

            # Leapfrog half kick
            p1 = p1 + 0.5*dt*mu1*a_r1
            p2 = p2 + 0.5*dt*mu2*a_r2
            p3 = p3 + 0.5*dt*mu3*a_r3

            # Drift
            r1 = r1 + dt*p1/mu1
            r2 = r2 + dt*p2/mu2
            r3 = r3 + dt*p3/mu3

            # Recompute accelerations at new positions
            q1 = -(m2/M12)*r1 - (m3/M123)*r2 - (m4/M1234)*r3
            q2 =  (m1/M12)*r1 - (m3/M123)*r2 - (m4/M1234)*r3
            q3 =  (M12/M123)*r2 - (m4/M1234)*r3
            q4 =  (M123/M1234)*r3

            aq1 = grav(q1,q2,m2) + grav(q1,q3,m3) + grav(q1,q4,m4)
            aq2 = grav(q2,q1,m1) + grav(q2,q3,m3) + grav(q2,q4,m4)
            aq3 = grav(q3,q1,m1) + grav(q3,q2,m2) + grav(q3,q4,m4)
            aq4 = grav(q4,q1,m1) + grav(q4,q2,m2) + grav(q4,q3,m3)

            a_r1 = aq2 - aq1
            a_r2 = aq3 - (m1*aq1 + m2*aq2)/M12
            a_r3 = aq4 - (m1*aq1 + m2*aq2 + m3*aq3)/M123

            # Half kick
            p1 = p1 + 0.5*dt*mu1*a_r1
            p2 = p2 + 0.5*dt*mu2*a_r2
            p3 = p3 + 0.5*dt*mu3*a_r3

            z[:,0:3], z[:,3:6], z[:,6:9] = r1, r2, r3
            z[:,9:12], z[:,12:15], z[:,15:18] = p1, p2, p3

            collision |= cp.any(cp.isnan(z), axis=1)

            if step % 1000 == 0 and step > n_steps//10:
                dist = cp.linalg.norm(z - z0, axis=1)
                better = dist < min_ret
                min_ret = cp.where(better, dist, min_ret)
                min_ret_t = cp.where(better, step*dt, min_ret_t)

            if step % 10000 == 0:
                log.info("  step %d/%d, collisions: %d", step, n_steps, int(collision.sum()))

        is_periodic = (min_ret / (extent + 1e-30)) < 0.05

        return {
            "return_distance": cp.asnumpy(min_ret),
            "return_time": cp.asnumpy(min_ret_t),
            "is_periodic": cp.asnumpy(is_periodic),
            "collision": cp.asnumpy(collision),
        }


def effective_rank(H, eps=1e-6):
    sv = np.linalg.svd(H, compute_uv=False)
    return int(np.sum(sv / (sv[0] + 1e-30) > eps))


def main():
    log.info("=" * 70)
    log.info("4-BODY GENERALIZATION TEST")
    log.info("=" * 70)

    m1, m2, m3, m4 = 1.0, 1.0, 1.0, 1.0
    N = 3000
    rng = np.random.RandomState(42)

    # Generate configs with varying separations
    r1_vals = 10 ** rng.uniform(-0.5, 2, N)
    r2_vals = 10 ** rng.uniform(-0.5, 2, N)
    r3_vals = 10 ** rng.uniform(-0.5, 2, N)

    log.info("Computing Hessians and ranks for %d configs...", N)
    t0 = time.time()
    z0_list = []
    ranks = []
    freqs_list = []

    for i in range(N):
        z = make_4body_config(r1_vals[i], r2_vals[i], r3_vals[i], m1, m2, m3, m4)
        z0_list.append(z)

        H = hessian_4body(z, m1, m2, m3, m4)
        ranks.append(effective_rank(H))

        # Frequencies from qq block (9x9 for 4-body)
        qq = H[0:9, 0:9]
        eigs = np.linalg.eigvalsh(qq)
        freqs = np.sort(np.sqrt(np.abs(eigs)))[::-1]
        freqs_list.append(freqs)

    z0 = np.array(z0_list)
    ranks = np.array(ranks)
    elapsed = time.time() - t0
    log.info("Hessians computed in %.1f s", elapsed)

    # Rank distribution
    log.info("\nRank distribution (4-body, max=18):")
    for r in range(1, 19):
        c = (ranks == r).sum()
        if c > 0:
            log.info("  rank=%d: %d (%.1f%%)", r, c, 100*c/N)

    # Integrate on GPU
    log.info("\nIntegrating %d orbits on GPU...", N)
    result = integrate_4body_batch_gpu(z0, m1, m2, m3, m4,
                                       dt=0.003, n_steps=60000, gpu_id=0)

    valid = ~result["collision"]
    periodic = result["is_periodic"] & valid
    n_periodic = periodic.sum()
    log.info("\nRESULTS: %d/%d periodic (%.2f%%), %d collisions",
             n_periodic, valid.sum(), 100*n_periodic/valid.sum(), result["collision"].sum())

    # Rank vs periodicity
    log.info("\n4-Body: Rank vs Periodicity:")
    for r in sorted(set(ranks)):
        rmask = (ranks == r) & valid
        n_at_rank = rmask.sum()
        if n_at_rank < 5:
            continue
        n_per = (periodic & (ranks == r)).sum()
        log.info("  rank=%d: %d/%d periodic (%.1f%%)", r, n_per, n_at_rank,
                 100*n_per/n_at_rank if n_at_rank > 0 else 0)

    # Does the pattern hold? Lower rank = more periodic?
    log.info("\nDoes lower rank predict higher periodicity?")
    low_rank = ranks <= np.median(ranks)
    high_rank = ranks > np.median(ranks)
    p_low = periodic[low_rank & valid].sum() / (low_rank & valid).sum()
    p_high = periodic[high_rank & valid].sum() / (high_rank & valid).sum()
    log.info("  Below-median rank: %.1f%% periodic", 100*p_low)
    log.info("  Above-median rank: %.1f%% periodic", 100*p_high)
    if p_low > p_high:
        log.info("  → YES: lower rank predicts higher periodicity (%.1fx)", p_low/p_high if p_high > 0 else float('inf'))
    else:
        log.info("  → NO: pattern does not hold for 4-body")

    np.savez_compressed("results/four_body_results.npz",
                        ranks=ranks, periodic=result["is_periodic"],
                        collision=result["collision"])
    log.info("Saved to results/four_body_results.npz")


if __name__ == "__main__":
    main()
