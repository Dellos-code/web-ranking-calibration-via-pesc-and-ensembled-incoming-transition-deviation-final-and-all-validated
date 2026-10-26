"""
Unit tests for the Stage 2 additions to ncd_calibration_lib.py.

Run:  python3 tests/test_stage2.py

1. KendallRef reproduces scipy.stats.kendalltau (tau-b) exactly, with ties.
2. alpha_sweep (one Neumann series per r) equals the factorised power iteration
   for every alpha on the grid and several r, to 1e-11 in L1.
3. alpha_sweep keeps exact structural ties (bit-identical scores for pages with
   identical in-neighbourhoods and block), like the power iteration.
4. rapr_reference(a=b=1) (TotalRank) equals the integral of pi(alpha) over
   [0, 1] computed by Gauss-Legendre quadrature of dense solves.
5. h equals ||u^T Z - u^T||_1, mean Sigma_in equals 1, and h_v equals the L1
   norm of the finite-difference derivative of pi(alpha) at alpha = 0.
6. Displacement bound: ||pi(alpha) - v||_1 <= alpha/(1-alpha) * h_v for every alpha.
   (pi - v = sum_k (1-a) a^k (v Z^k - v) and ||v Z^k - v||_1 <= k ||v Z - v||_1.)
"""
import os
import sys

import numpy as np
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import ncd_calibration_lib as L  # noqa: E402
from test_lib import random_graph  # noqa: E402


def dense_pi(ops, alpha, r):
    D = L.dense_P(ops, (1 - r) * alpha, r * alpha)
    n = ops.n
    A = np.eye(n) - D["P"].T
    A[-1, :] = 1.0
    b = np.zeros(n)
    b[-1] = 1.0
    return np.linalg.solve(A, b)


def test_kendall_ref():
    rng = np.random.default_rng(0)
    for n in (5, 50, 500, 3000):
        for ties in (False, True):
            y = rng.integers(0, max(2, n // 7), n).astype(float) if ties else rng.random(n)
            K = L.KendallRef(y)
            for _ in range(5):
                x = rng.integers(0, max(2, n // 5), n).astype(float) if ties else rng.random(n) + 0.3 * y
                a, b = K(x), stats.kendalltau(x, y).statistic
                assert abs(a - b) < 1e-14, (n, ties, a, b)


def test_alpha_sweep_matches_power():
    rng = np.random.default_rng(1)
    for _ in range(4):
        ops = L.build_operators(random_graph(rng, n=60, m=300, N=7, dangling=5))
        rs = [0.0, L.R_CANONICAL, 0.37, 0.9]
        S = L.alpha_sweep(ops, rs)
        for c, r in enumerate(rs):
            for j, a in enumerate(L.ALPHA_GRID):
                pi, _ = L.ncdawarerank(ops, a, r, tol=1e-15, maxit=200000)
                err = np.abs(S[c, j] - pi).sum()
                assert err < 1e-11, (r, a, err)
        # dense check at a few points
        for c, r in enumerate(rs):
            for j in (0, 7, int(np.argmin(np.abs(L.ALPHA_GRID - 0.85))), len(L.ALPHA_GRID) - 1):
                assert np.abs(S[c, j] - dense_pi(ops, L.ALPHA_GRID[j], r)).sum() < 1e-10


def test_alpha_sweep_ties():
    rng = np.random.default_rng(2)
    g = random_graph(rng, n=80, m=320, N=6, dangling=4)
    # make pages 70 and 71 structurally identical: same block, same in-neighbours,
    # no out-links of their own
    src, dst = g.src, g.dst
    keep = ~np.isin(src, [70, 71]) & ~np.isin(dst, [70, 71])
    src, dst = src[keep], dst[keep]
    new_src = np.array([3, 9, 15, 3, 9, 15])
    new_dst = np.array([70, 70, 70, 71, 71, 71])
    block = g.block.copy()
    block[71] = block[70]
    gg = L.Graph(g.n, np.concatenate([src, new_src]), np.concatenate([dst, new_dst]), block)
    ops = L.build_operators(gg)
    S = L.alpha_sweep(ops, [0.0, 0.2, 0.7])
    assert np.all(S[:, :, 70] == S[:, :, 71])
    pi, _ = L.ncdawarerank(ops, 0.85, 0.2, tol=1e-14)
    assert pi[70] == pi[71]


def test_totalrank_quadrature():
    rng = np.random.default_rng(3)
    ops = L.build_operators(random_graph(rng, n=40, m=200, N=5, dangling=3))
    r = L.R_CANONICAL
    t, rem = L.rapr_reference(ops, r, 1.0, 1.0, K=20000)
    assert abs(rem - 1.0 / 20002) < 1e-12 and abs(t.sum() - 1) < 1e-12
    xg, wg = np.polynomial.legendre.leggauss(200)
    al = 0.5 * (xg + 1)
    quad = sum(0.5 * w * dense_pi(ops, a, r) for a, w in zip(al, wg))
    assert np.abs(t - quad).sum() < 1e-4, np.abs(t - quad).sum()
    # Beta(1,1) weights equal 1/((k+1)(k+2))
    k = np.arange(10)
    w = np.exp(L._gammaln(1 + k) + L._gammaln(2) - L._gammaln(k + 3))
    assert np.allclose(w, 1.0 / ((k + 1) * (k + 2)))


def test_h_statistics():
    rng = np.random.default_rng(4)
    for _ in range(4):
        ops = L.build_operators(random_graph(rng))
        D = L.dense_P(ops, 0.75, 0.10)
        for r in (0.0, 0.2, 0.6, 1.0):
            hs = L.h_statistics(ops, r)
            Z = (1 - r) * D["H"] + r * D["M"]
            u = np.full(ops.n, 1.0 / ops.n)
            assert abs(hs["h"] - np.abs(u @ Z - u).sum()) < 1e-12
            assert abs(hs["mean_sigma"] - 1.0) < 1e-12
            assert abs(hs["h_v"] - np.abs(ops.v @ Z - ops.v).sum()) < 1e-12
            if r < 1.0:
                eps = 1e-6
                fd = np.abs((dense_pi(ops, eps, r) - ops.v) / eps).sum()
                assert abs(fd - hs["h_v"]) < 1e-4 * max(1.0, hs["h_v"]), (fd, hs["h_v"])


def test_displacement_bound():
    rng = np.random.default_rng(9)
    for _ in range(10):
        ops = L.build_operators(random_graph(rng, n=60, m=250, N=6, dangling=5))
        for r in (0.0, 0.1, 0.5, 0.9):
            hv = L.h_statistics(ops, r)["h_v"]
            S = L.alpha_sweep(ops, [r])
            for j, a in enumerate(L.ALPHA_GRID):
                assert np.abs(S[0, j] - ops.v).sum() <= a / (1 - a) * hv * (1 + 1e-12)


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print("ok", name)
    print("all Stage 2 tests passed")
