"""
Unit tests for ncd_calibration_lib.py.

Run:  python3 tests/test_lib.py      (or: pytest tests/)

What is checked
---------------
1. The tiny Web graph of [NG13] Figure 1(b): M, R, A reproduce the matrices
   printed in Section 3.1 / 3.2.1 entry by entry.
2. Row-stochasticity of H (with uniform dangling rows), M and P.
3. The factorised power iteration equals the dense solution of pi^T P = pi^T.
4. Theorem 1 of [NG13]: |lambda_2(P)| <= eta + mu.
5. Theorem 2 of [NG13]: theta = pi^T A^T is the stationary vector of the
   coupling matrix C.
6. P_esc^(M)(u) = k_u/(k_u+1) equals the M-mass leaving the home block.
7. Sigma_in^(H), Sigma_in^(M) equal dense column sums and both average to 1.
8. Remark 1 of [NG13]: one block => M = e e^T / n.  n singleton blocks =>
   M is the *lazy* hyperlink walk (D+I)^-1 (Adj + I), not H itself, because
   u always belongs to X_u.  We record the exact statement that holds.
9. The population model's transpose matvec equals the dense expected walk.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import ncd_calibration_lib as L  # noqa: E402


def tiny_graph():
    # pages 1..7 of [NG13] Fig. 1 -> 0..6 ; blocks {1,2,3}, {4,5,6}, {7}
    arcs = [(1, 2), (2, 3), (3, 1), (3, 4), (3, 7), (4, 5), (5, 6), (6, 4)]
    src = np.array([a - 1 for a, _ in arcs])
    dst = np.array([b - 1 for _, b in arcs])
    block = np.array([0, 0, 0, 1, 1, 1, 2])
    return L.Graph(7, src, dst, block)


def random_graph(rng, n=40, m=160, N=6, dangling=4):
    src = rng.integers(0, n, m)
    dst = rng.integers(0, n, m)
    src, dst = L.clean_arcs(n, src, dst)
    keep = ~np.isin(src, np.arange(dangling))      # make pages 0..dangling-1 dangling
    block = rng.integers(0, N, n)
    block[:N] = np.arange(N)
    return L.Graph(n, src[keep], dst[keep], block)


def test_tiny_matrices():
    ops = L.build_operators(tiny_graph())
    D = L.dense_P(ops, 0.75, 0.10)
    t, n9 = 1 / 3, 1 / 9
    M_paper = np.array([
        [t, t, t, 0, 0, 0, 0],
        [t, t, t, 0, 0, 0, 0],
        [n9, n9, n9, n9, n9, n9, t],
        [0, 0, 0, t, t, t, 0],
        [0, 0, 0, t, t, t, 0],
        [0, 0, 0, t, t, t, 0],
        [0, 0, 0, 0, 0, 0, 1]])
    R_paper = np.array([[t, 0, 0], [t, 0, 0], [n9, n9, t], [0, t, 0], [0, t, 0], [0, t, 0], [0, 0, 1]])
    A_paper = np.array([[1, 1, 1, 0, 0, 0, 0], [0, 0, 0, 1, 1, 1, 0], [0, 0, 0, 0, 0, 0, 1]])
    assert np.allclose(D["M"], M_paper), D["M"]
    assert np.allclose(D["R"], R_paper)
    assert np.allclose(D["A"], A_paper)
    assert np.allclose(D["R"] @ D["A"], D["M"])
    p, pbar = L.block_escape(ops)
    assert np.allclose(p, [0, 0, 2 / 3, 0, 0, 0, 0]) and abs(pbar - 2 / 21) < 1e-15


def test_stochastic_and_solver():
    rng = np.random.default_rng(1)
    for _ in range(5):
        ops = L.build_operators(random_graph(rng))
        for alpha, r in [(0.85, 0.0), (0.85, 0.1176), (0.85, 0.6), (0.6, 0.95)]:
            eta, mu = (1 - r) * alpha, r * alpha
            D = L.dense_P(ops, eta, mu)
            for k in ("H", "M", "P"):
                assert np.allclose(D[k].sum(1), 1.0)
            pi, it = L.ncdawarerank(ops, alpha, r, tol=1e-14)
            # dense stationary vector: solve (P^T - I) x = 0 with sum 1
            n = ops.n
            Aeq = np.vstack([D["P"].T - np.eye(n), np.ones(n)])
            b = np.zeros(n + 1)
            b[-1] = 1
            x = np.linalg.lstsq(Aeq, b, rcond=None)[0]
            assert np.abs(pi - x).sum() < 1e-10, (alpha, r, np.abs(pi - x).sum())


def test_theorem1_and_theorem2():
    rng = np.random.default_rng(2)
    for _ in range(5):
        ops = L.build_operators(random_graph(rng))
        for eta, mu in [(0.75, 0.10), (0.85, 0.10), (0.3, 0.6)]:
            D = L.dense_P(ops, eta, mu)
            ev = np.sort(np.abs(np.linalg.eigvals(D["P"])))[::-1]
            assert abs(ev[0] - 1) < 1e-10 and ev[1] <= eta + mu + 1e-10
            # Theorem 2: coupling matrix C_IJ = (pi_I^T / pi_I^T e) P_IJ e
            pi, _ = L.ncdawarerank(ops, eta + mu, mu / (eta + mu), tol=1e-14)
            N = ops.N
            C = np.zeros((N, N))
            for I in range(N):
                rows = ops.block == I
                w = pi[rows] / pi[rows].sum()
                for J in range(N):
                    C[I, J] = w @ D["P"][np.ix_(rows, ops.block == J)].sum(1)
            theta = D["A"] @ pi
            assert np.allclose(theta @ C, theta, atol=1e-12) and abs(theta.sum() - 1) < 1e-12


def test_escape_and_incoming():
    rng = np.random.default_rng(3)
    for _ in range(5):
        ops = L.build_operators(random_graph(rng))
        D = L.dense_P(ops, 0.75, 0.10)
        own = ops.block[:, None] == ops.block[None, :]
        leave = (D["M"] * ~own).sum(1)
        p, _ = L.block_escape(ops)
        assert np.allclose(p, leave)
        sh, sm = L.incoming_sums(ops)
        assert np.allclose(sh, D["H"].sum(0)) and np.allclose(sm, D["M"].sum(0))
        assert abs(sh.mean() - 1) < 1e-12 and abs(sm.mean() - 1) < 1e-12
        for r in (0.0, 0.3, 0.9):
            h, mean_s = L.h_statistic(ops, r)
            assert abs(mean_s - 1) < 1e-12
            Z = (1 - r) * D["H"] + r * D["M"]
            assert abs(h - np.abs(1 - Z.sum(0)).mean()) < 1e-12


def test_remark1():
    rng = np.random.default_rng(4)
    g = random_graph(rng)
    # one block containing every page: M = e e^T / n
    ops = L.build_operators(L.Graph(g.n, g.src, g.dst, np.zeros(g.n, int)))
    assert np.allclose(L.dense_P(ops, 0.5, 0.3)["M"], 1.0 / g.n)
    # n singleton blocks: M = (D+I)^-1 (Adj + I) on linked rows (lazy walk), not H
    ops = L.build_operators(L.Graph(g.n, g.src, g.dst, np.arange(g.n)))
    D = L.dense_P(ops, 0.5, 0.3)
    adj = np.zeros((g.n, g.n))
    adj[g.src, g.dst] = 1
    lazy = (adj + np.eye(g.n)) / (adj.sum(1, keepdims=True) + 1)
    assert np.allclose(D["M"], lazy)
    linked = adj.sum(1) > 0
    assert not np.allclose(D["M"][linked], D["H"][linked])


def test_population_model():
    rng = np.random.default_rng(5)
    params = L.sample_static_params(rng, 300)
    g, pop = L.generate_static_web(params, rng)
    n = g.n
    # dense expected walk from the generator definition
    T = np.zeros((n, n))
    for u in range(n):
        if not pop.linked[u]:
            T[u] = 1.0 / n
            continue
        B = pop.block[u]
        inb = pop.block == B
        if pop.phi_eff[u] > 0:
            w = np.where(inb, pop.q, 0.0)
            w[u] = 0
            T[u] += pop.phi_eff[u] * w / w.sum()
        Nb = len(pop.Qb)
        wg = pop.Qt.copy()
        wg[B] = 0
        wb = (1 - pop.lam_eff[B]) * wg / wg.sum()
        if pop.lam_eff[B] > 0:
            wl = np.where(pop.domain_of_block == pop.domain_of_block[B], pop.Qt, 0.0)
            wl[B] = 0
            wb = wb + pop.lam_eff[B] * wl / wl.sum()
        page_w = (1 - pop.phi_eff[u]) * wb[pop.block] * pop.q / pop.Qb[pop.block]
        T[u] += page_w
    assert np.allclose(T.sum(1), 1.0)
    x = rng.random(n)
    assert np.allclose(T.T @ x, pop.transpose_matvec(x))


def test_perturbations():
    rng = np.random.default_rng(6)
    g, _ = L.generate_static_web(L.sample_static_params(rng, 500), rng)
    ga = L.sparsify_arcs(g, 0.4, rng)
    assert ga.m == round(0.4 * g.m) and ga.n == g.n
    gb = L.newly_added_pages(g, 0.3, rng)
    assert gb.m < g.m and gb.n == g.n
    gc = L.mixed_damage(g, 0.8, 0.3, np.random.default_rng(7))
    assert gc.n == g.n and gc.m <= g.m
    assert gc.meta["mixed_damage_order"] == "B_then_A"
    seed = 71
    direct = L.sparsify_arcs(g, 0.6, np.random.default_rng(seed))
    mixed_a = L.mixed_damage(g, 0.6, 0.0, np.random.default_rng(seed))
    assert np.array_equal(direct.src, mixed_a.src)
    assert np.array_equal(direct.dst, mixed_a.dst)
    direct_b = L.newly_added_pages(g, 0.3, np.random.default_rng(seed))
    mixed_b = L.mixed_damage(g, 1.0, 0.3, np.random.default_rng(seed))
    assert np.array_equal(direct_b.src, mixed_b.src)
    assert np.array_equal(direct_b.dst, mixed_b.dst)
    again = L.mixed_damage(g, 0.8, 0.3, np.random.default_rng(99))
    same = L.mixed_damage(g, 0.8, 0.3, np.random.default_rng(99))
    assert np.array_equal(again.src, same.src) and np.array_equal(again.dst, same.dst)
    levels = (0.9, 0.7, 0.5)
    dz = []
    for level in levels:
        vals = []
        for seed in range(5):
            damaged = L.mixed_damage(g, level, 0.4, np.random.default_rng(seed))
            ops = L.build_operators(damaged)
            feats = L.structural_features(ops)
            vals.append((feats["dangling_frac"], feats["zero_in_frac"]))
        dz.append(np.mean(vals, axis=0))
    assert np.all(np.diff(np.asarray(dz), axis=0) >= -0.05)


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print("ok", name)
    print("all tests passed")
