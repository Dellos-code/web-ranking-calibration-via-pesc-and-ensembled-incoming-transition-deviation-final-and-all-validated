"""
ncd_calibration_lib.py
======================

Core library for the two-stage calibration of NCDawareRank
(A. N. Nikolakopoulos and J. D. Garofalakis, "NCDawareRank: a novel ranking
method that exploits the decomposable structure of the Web", WSDM 2013;
cited below as [NG13]).

Notation follows [NG13] Section 3.1 and the manuscript paper.tex:

    W           set of pages, n = |W|
    G_u         out-neighbours of u,  d_u = |G_u|
    {A_1..A_N}  partition of W into NCD blocks, A_(u) the block of u
    X_u         union of the blocks of u and of its out-neighbours
    N_u         number of distinct blocks in X_u  (= k_u + 1)
    H           row-normalised hyperlink matrix, dangling rows -> e^T/n
    M = R A     inter-level proximity matrix, M_uv = 1/(N_u |A_(v)|), v in X_u
    E = e v^T   teleportation, v block-even: v_u = 1/(N |A_(u)|)
    P = eta H + mu M + (1-eta-mu) E
      = alpha Z + (1-alpha) E,  alpha = eta+mu,  Z = (1-r) H + r M,  r = mu/alpha

Contents
--------
1. Graph container and cleaning ........................ Graph, clean_arcs
2. Operators (H^T, R^T, A as an index map, v) ......... NCDOperators, build_operators
3. Solver (factorised power iteration, [NG13] Alg. 1) . ncdawarerank, sweep_r
4. Dense reference implementation (tests only) ........ dense_P
5. Structural statistics
     Stage 1: block escape probability P_esc^(M) ....... block_escape
     Stage 2: incoming sums Sigma_in^(H), ^(M), h ...... incoming_sums, h_statistic
     competitor features ............................... structural_features
6. Perturbations of [NG13] Section 4.2 ................ sparsify_arcs, newly_added_pages
7. Synthetic generators
     static fitness block web (with population truth) . generate_static_web
     growth / preferential-attachment block web ........ generate_growth_web
8. Rank agreement metrics ............................. kendall_tau_b, spearman_rho
9. Calibration-rule helpers ........................... clip_linear, logistic, interp_curve

All per-iteration work is O(nnz(H) + nnz(R) + n); all structural statistics
are O(|E| + n).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Optional

import numpy as np
import scipy.sparse as sp
from scipy import stats

ALPHA0 = 0.85                       # canonical damping of [NG13]
R_CANONICAL = 0.10 / 0.85           # (eta, mu) = (0.75, 0.10)  =>  r = mu/alpha
R_GRID = np.round(np.arange(0.0, 1.0, 0.01), 2)   # 0.00, 0.01, ..., 0.99
ORACLE_MASK = R_GRID >= 0.01                       # oracle searched on [0.01, 0.99]


# ---------------------------------------------------------------------------
# 1. Graph container
# ---------------------------------------------------------------------------
@dataclass
class Graph:
    """A directed graph with an NCD block partition.

    src, dst : int64 arrays of arcs u -> v (self-loops and duplicates removed)
    block    : int64 array of length n, block id of each page (0..N-1)
    """
    n: int
    src: np.ndarray
    dst: np.ndarray
    block: np.ndarray
    name: str = ""
    meta: dict = field(default_factory=dict)

    @property
    def m(self) -> int:
        return int(len(self.src))

    def subgraph_arcs(self, keep: np.ndarray, name_suffix: str = "") -> "Graph":
        """Same page set and partition, only the arcs selected by `keep`."""
        return Graph(self.n, self.src[keep], self.dst[keep], self.block,
                     self.name + name_suffix, dict(self.meta))


def clean_arcs(n: int, src, dst):
    """Drop self-loops and duplicate arcs; return sorted (src, dst)."""
    src = np.asarray(src, dtype=np.int64)
    dst = np.asarray(dst, dtype=np.int64)
    ok = src != dst
    key = np.unique(src[ok] * np.int64(n) + dst[ok])
    return key // n, key % n


# ---------------------------------------------------------------------------
# 2. Operators
# ---------------------------------------------------------------------------
@dataclass
class NCDOperators:
    """Sparse factors of the NCDawareRank chain.

    HT       (n x n) CSR, HT[v,u] = H_uv = 1/d_u for arcs u->v (non-dangling rows)
    dangling indices of pages with d_u = 0; their H row is e^T/n, applied
             implicitly as a rank-one term
    RT       (N x n) CSR, RT[B,u] = R_uB = 1/(N_u |A_B|) if A_B in X_u
    block    page -> block map; it *is* the matrix A (A[B,v] = 1 iff v in A_B),
             so (pi^T R) A is just (RT @ pi)[block]
    v        block-even teleportation vector
    """
    n: int
    N: int
    m: int
    block: np.ndarray
    block_size: np.ndarray
    d_out: np.ndarray
    dangling: np.ndarray
    HT: sp.csr_matrix
    RT: sp.csr_matrix
    k_ext: np.ndarray
    n_ext_arcs: np.ndarray
    v: np.ndarray


def build_operators(g: Graph) -> NCDOperators:
    """Build H^T, R^T, A (as the page->block index map) and v for graph `g`.

    Memory is O(|E| + n).  The distinct (page, block) pairs that form the non-zeros of
    R are found here with one vectorised np.unique on integer keys, i.e. in
    O(m' log m') time for m' = n + (number of inter-block arcs); the paper's
    Proposition 5 gives an O(|E| + n) marker-array construction of the same pairs
    (a block-indexed array 'last' records the last page that touched each block)."""
    n = int(g.n)
    src, dst = np.asarray(g.src, np.int64), np.asarray(g.dst, np.int64)
    _, block = np.unique(np.asarray(g.block), return_inverse=True)
    block = block.astype(np.int64)
    N = int(block.max()) + 1
    bsize = np.bincount(block, minlength=N).astype(np.int64)

    d_out = np.bincount(src, minlength=n).astype(np.int64)
    dangling = np.flatnonzero(d_out == 0)
    HT = sp.csr_matrix((1.0 / d_out[src], (dst, src)), shape=(n, n))

    # proximal (page, block) pairs: own block for every page, plus the block of
    # every out-neighbour lying in another block
    ext = block[src] != block[dst]
    pu = np.concatenate([np.arange(n, dtype=np.int64), src[ext]])
    pb = np.concatenate([block, block[dst[ext]]])
    key = np.unique(pu * np.int64(N) + pb)
    pu, pb = key // N, key % N
    N_u = np.bincount(pu, minlength=n).astype(np.int64)   # >= 1 (own block)
    RT = sp.csr_matrix((1.0 / (N_u[pu] * bsize[pb]), (pb, pu)), shape=(N, n))

    v = 1.0 / (N * bsize[block].astype(float))
    return NCDOperators(n=n, N=N, m=len(src), block=block, block_size=bsize,
                        d_out=d_out, dangling=dangling, HT=HT, RT=RT,
                        k_ext=N_u - 1,
                        n_ext_arcs=np.bincount(src[ext], minlength=n).astype(np.int64),
                        v=v)


# ---------------------------------------------------------------------------
# 3. Solver
# ---------------------------------------------------------------------------
def ncdawarerank(ops: NCDOperators, alpha: float, r: float,
                 x0: Optional[np.ndarray] = None, tol: float = 1e-10,
                 maxit: int = 20000, teleport: Optional[np.ndarray] = None):
    """Stationary distribution of P = alpha[(1-r)H + rM] + (1-alpha) e v^T.

    Factorised power iteration of [NG13] Algorithm 1:
        pi <- eta (pi^T H) + mu (pi^T R) A + (1-eta-mu) v^T,
    with eta = (1-r) alpha, mu = r alpha, the dangling rows of H applied as
    the rank-one term (sum of pi over dangling pages)/n, and stopping rule
    ||pi^(k+1) - pi^(k)||_1 < tol.  Returns (pi, iterations).
    """
    v = ops.v if teleport is None else teleport
    eta, mu = (1.0 - r) * alpha, r * alpha
    x = v.copy() if x0 is None else np.asarray(x0, float) / np.sum(x0)
    base = (1.0 - alpha) * v
    for it in range(1, maxit + 1):
        y = base.copy()
        if eta > 0.0:
            y += eta * (ops.HT @ x)
            if len(ops.dangling):
                y += eta * x[ops.dangling].sum() / ops.n
        if mu > 0.0:
            y += mu * (ops.RT @ x)[ops.block]
        y /= y.sum()
        err = np.abs(y - x).sum()
        x = y
        if err < tol:
            return x, it
    return x, maxit


def sweep_r(ops: NCDOperators, alpha: float, r_values: Iterable[float],
            tol: float = 1e-10):
    """Yield (r, pi_r, iterations) along r_values, warm-starting each solve."""
    x = None
    for r in r_values:
        x, it = ncdawarerank(ops, alpha, float(r), x0=x, tol=tol)
        yield float(r), x, it


# ---------------------------------------------------------------------------
# 4. Dense reference (unit tests only)
# ---------------------------------------------------------------------------
def dense_P(ops: NCDOperators, eta: float, mu: float):
    """Dense H, M, R, A, E and P = eta H + mu M + (1-eta-mu) E."""
    n, N = ops.n, ops.N
    H = ops.HT.T.toarray()
    H[ops.dangling, :] = 1.0 / n
    R = ops.RT.T.toarray()
    A = np.zeros((N, n))
    A[ops.block, np.arange(n)] = 1.0
    M = R @ A
    E = np.outer(np.ones(n), ops.v)
    return dict(H=H, M=M, R=R, A=A, E=E, P=eta * H + mu * M + (1 - eta - mu) * E)


# ---------------------------------------------------------------------------
# 5. Structural statistics
# ---------------------------------------------------------------------------
def block_escape(ops: NCDOperators):
    """Stage 1 statistic.

    P_esc^(M)(u) = 1 - 1/(k_u+1) = k_u/(k_u+1), with k_u the number of distinct
    external blocks u links to; under M the surfer picks one of the N_u = k_u+1
    proximal blocks uniformly, so this is the probability of leaving A_(u).
    Returns (per-page array, mean).
    """
    p = ops.k_ext / (ops.k_ext + 1.0)
    return p, float(p.mean())


def incoming_sums(ops: NCDOperators):
    """Stage 2 ingredients: column sums of H (with dangling rows e^T/n) and of M.

    Sigma_in^(H)(v) = sum_u H_uv = sum_{u->v} 1/d_u + n_dangling/n
    Sigma_in^(M)(v) = sum_u M_uv = (1/|A_(v)|) sum_{u: A_(v) in X_u} 1/N_u
                    = (row sum of R^T at block(v)), an O(nnz(R) + n) pass.
    Both have mean exactly 1 because H and M are row-stochastic.
    """
    sig_h = np.asarray(ops.HT.sum(axis=1)).ravel() + len(ops.dangling) / ops.n
    sig_m = np.asarray(ops.RT.sum(axis=1)).ravel()[ops.block]
    return sig_h, sig_m


def h_statistic(ops: NCDOperators, r: float, sums=None):
    """h = mean_v |1 - [r Sigma_in^(M)(v) + (1-r) Sigma_in^(H)(v)]|."""
    sig_h, sig_m = incoming_sums(ops) if sums is None else sums
    s = r * sig_m + (1.0 - r) * sig_h
    return float(np.mean(np.abs(1.0 - s))), float(np.mean(s))


def structural_features(ops: NCDOperators) -> dict:
    """Stage 1 statistic and the competitor features used as baselines.

    Every quantity is computable in O(|E| + n) from the observed graph.
    pesc_H_block is the block-escape probability under the logic of H, i.e.
    the H-mass leaving the home block: (external out-arcs)/d_u for linked
    pages and 1 - |A_(u)|/n for dangling pages (uniform row).
    """
    n = ops.n
    p_m, pbar_m = block_escape(ops)
    nd = ops.d_out > 0
    esc_h = np.empty(n)
    esc_h[nd] = ops.n_ext_arcs[nd] / ops.d_out[nd]
    esc_h[~nd] = 1.0 - ops.block_size[ops.block[~nd]] / n
    linked = ops.d_out > 0
    return dict(
        pesc_M=pbar_m,
        pesc_M_linked=float(p_m[linked].mean()) if linked.any() else 0.0,
        mean_k=float(ops.k_ext.mean()),
        pesc_H_block=float(esc_h.mean()),
        inter_arc_frac=float(ops.n_ext_arcs.sum() / max(ops.m, 1)),
        mean_outdeg=float(ops.m / n),
        log_mean_outdeg=float(np.log1p(ops.m / n)),
        dangling_frac=float(len(ops.dangling) / n),
        zero_in_frac=float(np.mean(np.asarray(ops.HT.getnnz(axis=1)) == 0)),
        n=n, N=ops.N, m=ops.m,
        mean_block_size=float(n / ops.N),
        log_n=float(np.log(n)),
    )


# ---------------------------------------------------------------------------
# 6. Perturbations ([NG13] Section 4.2)
# ---------------------------------------------------------------------------
def sparsify_arcs(g: Graph, keep_frac: float, rng: np.random.Generator) -> Graph:
    """Experiment A ('Sparseness'): keep a uniformly random keep_frac of arcs."""
    k = int(round(keep_frac * g.m))
    idx = np.sort(rng.choice(g.m, size=k, replace=False))
    keep = np.zeros(g.m, dtype=bool)
    keep[idx] = True
    return g.subgraph_arcs(keep, f"|A{keep_frac:.2f}")


def newly_added_pages(g: Graph, page_frac: float, rng: np.random.Generator,
                      remove: float = 0.9) -> Graph:
    """Experiment B ('Newly added pages'): choose page_frac of the pages and
    delete each of their incoming arcs independently with probability `remove`
    (90% in [NG13])."""
    chosen = rng.choice(g.n, size=int(round(page_frac * g.n)), replace=False)
    is_new = np.zeros(g.n, dtype=bool)
    is_new[chosen] = True
    drop = is_new[g.dst] & (rng.random(g.m) < remove)
    return g.subgraph_arcs(~drop, f"|B{page_frac:.2f}")


# ---------------------------------------------------------------------------
# 7. Synthetic generators
# ---------------------------------------------------------------------------
def _lognormal_sizes(rng, total, mean, sigma):
    out, s = [], 0
    while s < total:
        x = int(max(1, round(np.exp(rng.normal(np.log(mean) - sigma ** 2 / 2, sigma)))))
        x = min(x, total - s)
        out.append(x)
        s += x
    return np.array(out, dtype=np.int64)


def sample_static_params(rng: np.random.Generator, n: int) -> dict:
    """Random draw from the static-generator parameter space."""
    return dict(
        n=int(n),
        mean_block=float(np.exp(rng.uniform(np.log(4), np.log(100)))),
        block_sigma=float(rng.uniform(0.3, 1.3)),
        domain_size=int(rng.integers(3, 31)),
        locality=float(rng.uniform(0.0, 0.9)),
        fit_sigma=float(rng.uniform(0.8, 2.0)),
        icc=float(rng.uniform(0.0, 0.8)),
        block_beta=float(rng.uniform(0.5, 1.2)),
        block_xi=float(rng.uniform(0.0, 1.0)),
        mean_out=float(np.exp(rng.uniform(np.log(2), np.log(25)))),
        out_sigma=float(rng.uniform(0.5, 1.2)),
        dangling=float(rng.uniform(0.02, 0.30)),
        phi=float(rng.uniform(0.30, 0.97)),
    )


@dataclass
class PopulationModel:
    """Expected (infinite-sample) walk of the static generator.

    For a linked page u the target distribution is
        phi_u * q_v / (Q_B(u) - q_u)                      v in A_(u), v != u
        (1-phi_u) * omega_u(B') * q_v / Q_B'              v in B' != A_(u)
    with omega_u a mixture of domain-local and global block choice
    proportional to block attractiveness.  Dangling pages jump uniformly.
    PageRank of this walk is the latent 'population' importance.
    """
    n: int
    block: np.ndarray
    domain_of_block: np.ndarray
    q: np.ndarray
    Qb: np.ndarray        # sum of q per block
    Qt: np.ndarray        # block attractiveness
    phi_eff: np.ndarray   # per page
    lam_eff: np.ndarray   # per block
    linked: np.ndarray    # bool per page

    def transpose_matvec(self, x):
        n, b = self.n, self.block
        N = len(self.Qb)
        y = np.zeros(n)
        xl = np.where(self.linked, x, 0.0)
        y += (x.sum() - xl.sum()) / n                          # dangling rows
        # intra-block part
        denom = self.Qb[b] - self.q
        a = np.where(self.phi_eff > 0, xl * self.phi_eff / np.where(denom > 0, denom, 1.0), 0.0)
        S = np.bincount(b, weights=a, minlength=N)
        y += self.q * (S[b] - a)
        # inter-block part
        bb = np.bincount(b, weights=xl * (1.0 - self.phi_eff), minlength=N)
        b_loc = self.lam_eff * bb
        b_glob = (1.0 - self.lam_eff) * bb
        Qtot = self.Qt.sum()
        g = b_glob / (Qtot - self.Qt)
        F = self.Qt * (g.sum() - g)
        dom = self.domain_of_block
        Qdom = np.bincount(dom, weights=self.Qt)
        den = Qdom[dom] - self.Qt
        l = np.where(self.lam_eff > 0, b_loc / np.where(den > 0, den, 1.0), 0.0)
        L = np.bincount(dom, weights=l)
        F += self.Qt * (L[dom] - l)
        y += F[b] * self.q / self.Qb[b]
        return y

    def pagerank(self, alpha, v, tol=1e-12, maxit=20000):
        x = v.copy()
        for _ in range(maxit):
            y = alpha * self.transpose_matvec(x) + (1 - alpha) * v
            y /= y.sum()
            if np.abs(y - x).sum() < tol:
                return y
            x = y
        return x


def _sample_within(rng, cum_by_block, start, size, blocks):
    """Sample one page per entry of `blocks`, proportional to q inside the block.
    cum_by_block: cumulative q over pages sorted by block; start/size per block."""
    s, z = start[blocks], size[blocks]
    lo = np.where(s > 0, cum_by_block[np.maximum(s - 1, 0)], 0.0)
    hi = cum_by_block[s + z - 1]
    u = lo + rng.random(len(blocks)) * (hi - lo)
    pos = np.searchsorted(cum_by_block, u, side="right")
    return np.minimum(pos, s + z - 1)


def generate_static_web(params: dict, rng: np.random.Generator):
    """Static fitness block web (power-law-like in/out degrees, heavy-tailed
    block sizes, domain-level locality).  Returns (Graph, PopulationModel)."""
    p = params
    n = p["n"]
    sizes = _lognormal_sizes(rng, n, p["mean_block"], p["block_sigma"])
    N = len(sizes)
    block = np.repeat(np.arange(N), sizes)            # pages sorted by block
    start = np.concatenate([[0], np.cumsum(sizes)[:-1]])
    # domains (second hierarchy level)
    perm = rng.permutation(N)
    dom_of_block = np.empty(N, dtype=np.int64)
    dom_of_block[perm] = np.arange(N) // p["domain_size"]
    # fitness with a block random effect: log q = s_b z_B + s_w z_u
    sb = p["fit_sigma"] * np.sqrt(p["icc"])
    sw = p["fit_sigma"] * np.sqrt(1 - p["icc"])
    q = np.exp(sb * rng.normal(size=N)[block] + sw * rng.normal(size=n))
    Qb = np.bincount(block, weights=q, minlength=N)
    Qt = Qb ** p["block_beta"] * np.exp(p["block_xi"] * rng.normal(size=N) - p["block_xi"] ** 2 / 2)
    # out-degrees
    d = np.maximum(1, np.rint(np.exp(rng.normal(np.log(p["mean_out"]) - p["out_sigma"] ** 2 / 2,
                                                   p["out_sigma"], size=n)))).astype(np.int64)
    linked = rng.random(n) >= p["dangling"]
    d = np.where(linked, np.minimum(d, n - 1), 0)
    phi_eff = np.where(sizes[block] >= 2, p["phi"], 0.0)
    dom_nblocks = np.bincount(dom_of_block)
    lam_eff = np.where(dom_nblocks[dom_of_block] >= 2, p["locality"], 0.0)

    cum_q = np.cumsum(q)
    src_all = np.repeat(np.arange(n), d)
    intra = rng.random(len(src_all)) < phi_eff[src_all]
    # intra-block targets (self-targets resampled)
    s_in = src_all[intra]
    t_in = _sample_within(rng, cum_q, start, sizes, block[s_in])
    for _ in range(20):
        bad = t_in == s_in
        if not bad.any():
            break
        t_in[bad] = _sample_within(rng, cum_q, start, sizes, block[s_in[bad]])
    # inter-block targets: choose block (local or global, never own), then page
    s_ex = src_all[~intra]
    bsrc = block[s_ex]
    use_loc = rng.random(len(s_ex)) < lam_eff[bsrc]
    tb = np.empty(len(s_ex), dtype=np.int64)
    cum_Qt = np.cumsum(Qt)
    order_dom = np.argsort(dom_of_block, kind="stable")
    Qt_dom_sorted = Qt[order_dom]
    cum_dom = np.cumsum(Qt_dom_sorted)
    dom_start = np.searchsorted(dom_of_block[order_dom], np.arange(dom_nblocks.size))
    for _ in range(50):
        need = np.ones(len(s_ex), dtype=bool) if _ == 0 else (tb == bsrc)
        if not need.any():
            break
        gidx = np.flatnonzero(need & ~use_loc)
        if len(gidx):
            u = rng.random(len(gidx)) * cum_Qt[-1]
            tb[gidx] = np.minimum(np.searchsorted(cum_Qt, u, side="right"), N - 1)
        lidx = np.flatnonzero(need & use_loc)
        if len(lidx):
            dd = dom_of_block[bsrc[lidx]]
            s0, z0 = dom_start[dd], dom_nblocks[dd]
            lo = np.where(s0 > 0, cum_dom[np.maximum(s0 - 1, 0)], 0.0)
            hi = cum_dom[s0 + z0 - 1]
            u = lo + rng.random(len(lidx)) * (hi - lo)
            pos = np.minimum(np.searchsorted(cum_dom, u, side="right"), s0 + z0 - 1)
            tb[lidx] = order_dom[pos]
    okx = tb != bsrc
    s_ex, tb = s_ex[okx], tb[okx]
    t_ex = _sample_within(rng, cum_q, start, sizes, tb)
    src = np.concatenate([s_in, s_ex])
    dst = np.concatenate([t_in, t_ex])
    src, dst = clean_arcs(n, src, dst)
    # shuffle page ids so that block membership is not encoded in the id order
    pid = rng.permutation(n)
    inv = np.empty(n, dtype=np.int64)
    inv[pid] = np.arange(n)
    g = Graph(n=n, src=pid[src], dst=pid[dst], block=block[inv], name="static",
              meta={"generator": "static", **p})
    pop = PopulationModel(n=n, block=block[inv], domain_of_block=dom_of_block, q=q[inv],
                          Qb=Qb, Qt=Qt, phi_eff=phi_eff[inv], lam_eff=lam_eff, linked=linked[inv])
    return g, pop


def sample_growth_params(rng: np.random.Generator, n: int) -> dict:
    return dict(
        n=int(n),
        new_block=float(np.exp(rng.uniform(np.log(0.01), np.log(0.25)))),
        mean_out=float(np.exp(rng.uniform(np.log(2), np.log(20)))),
        out_sigma=float(rng.uniform(0.5, 1.1)),
        dangling=float(rng.uniform(0.02, 0.30)),
        phi=float(rng.uniform(0.30, 0.95)),
        pa_offset=float(np.exp(rng.uniform(np.log(0.2), np.log(5.0)))),
        update_frac=float(rng.uniform(0.1, 0.5)),
    )


def generate_growth_web(params: dict, rng: np.random.Generator) -> Graph:
    """Growth block web: pages arrive one at a time, join an existing site
    (probability proportional to its size) or found a new one, and link to
    earlier pages by preferential attachment (in-degree + offset), inside
    their own site with probability phi.  A later 'update' phase lets existing
    pages add arcs to any page, which creates cycles.  Mechanistically
    different from the static generator (age effects, Yule block sizes)."""
    p = params
    n = p["n"]
    a = p["pa_offset"]
    block = np.empty(n, dtype=np.int64)
    members = []                     # pages of each block
    urn_block = []                   # per-block urn of in-arc targets
    urn_all = []                     # global urn of in-arc targets
    src, dst = [], []
    d = np.maximum(1, np.rint(np.exp(rng.normal(np.log(p["mean_out"]) - p["out_sigma"] ** 2 / 2,
                                                   p["out_sigma"], size=n)))).astype(np.int64)
    linked = rng.random(n) >= p["dangling"]
    d = np.where(linked, d, 0)

    def pick_in_block(bk, exclude):
        mem, urn = members[bk], urn_block[bk]
        for _ in range(20):
            if rng.random() < a * len(mem) / (a * len(mem) + len(urn)):
                t = mem[rng.integers(len(mem))]
            else:
                t = urn[rng.integers(len(urn))]
            if t != exclude:
                return t
        return -1

    def pick_global(n_now, bk_excl):
        for _ in range(30):
            if rng.random() < a * n_now / (a * n_now + len(urn_all)):
                t = int(rng.integers(n_now))
            else:
                t = urn_all[rng.integers(len(urn_all))]
            if block[t] != bk_excl:
                return t
        return -1

    def add_arc(u, t):
        src.append(u)
        dst.append(t)
        urn_all.append(t)
        urn_block[block[t]].append(t)

    for t in range(n):
        if t == 0 or rng.random() < p["new_block"]:
            block[t] = len(members)
            members.append([t])
            urn_block.append([])
        else:
            bk = block[rng.integers(t)]          # proportional to block size
            block[t] = bk
            members[bk].append(t)
        bk = block[t]
        for _ in range(d[t] if t > 0 else 0):
            intra_ok = len(members[bk]) > 1
            if intra_ok and rng.random() < p["phi"]:
                tg = pick_in_block(bk, t)
            else:
                tg = pick_global(t, bk) if len(members) > 1 else -1
            if tg >= 0:
                add_arc(t, tg)
    # update phase: arcs from existing linked pages to any page
    n_upd = int(p["update_frac"] * len(src))
    linked_ids = np.flatnonzero(linked)
    for _ in range(n_upd):
        u = int(linked_ids[rng.integers(len(linked_ids))])
        bk = block[u]
        if len(members[bk]) > 1 and rng.random() < p["phi"]:
            tg = pick_in_block(bk, u)
        else:
            tg = pick_global(n, bk)
        if tg >= 0:
            add_arc(u, tg)
    s, dd = clean_arcs(n, np.array(src), np.array(dst))
    pid = rng.permutation(n)
    inv = np.empty(n, dtype=np.int64)
    inv[pid] = np.arange(n)
    return Graph(n=n, src=pid[s], dst=pid[dd], block=block[inv], name="growth",
                 meta={"generator": "growth", **p})


# ---------------------------------------------------------------------------
# 8. Rank agreement
# ---------------------------------------------------------------------------
def kendall_tau_b(x, y) -> float:
    """Kendall's tau-b (ties handled as in [NG13]'s use of tau for rankings)."""
    return float(stats.kendalltau(x, y).statistic)


class SpearmanRef:
    """Spearman's rho against a fixed reference, with the reference ranked once."""

    def __init__(self, ref):
        rr = stats.rankdata(ref)
        self.rc = rr - rr.mean()
        self.nrm = np.sqrt((self.rc ** 2).sum())

    def __call__(self, x) -> float:
        rx = stats.rankdata(x)
        rx -= rx.mean()
        den = self.nrm * np.sqrt((rx ** 2).sum())
        return float((rx * self.rc).sum() / den) if den > 0 else np.nan


def spearman_rho(x, y) -> float:
    return SpearmanRef(y)(x)


# ---------------------------------------------------------------------------
# 9. Calibration-rule helpers
# ---------------------------------------------------------------------------
def clip_linear(x, c0, c1, lo=0.01, hi=0.99):
    """r = clip(c0 + c1 x, lo, hi)."""
    return np.clip(c0 + c1 * np.asarray(x, float), lo, hi)


def logistic(x, a, b, lo=0.01, hi=0.99):
    """r = lo + (hi - lo) / (1 + exp(-(a + b x)))."""
    return lo + (hi - lo) / (1.0 + np.exp(-(a + b * np.asarray(x, float))))


def interp_curve(curves: np.ndarray, r_hat: np.ndarray, grid: np.ndarray = R_GRID):
    """Linear interpolation of per-instance metric curves at r_hat.

    curves: (n_inst, len(grid)); r_hat: (n_inst,).  Returns (n_inst,)."""
    r_hat = np.clip(np.asarray(r_hat, float), grid[0], grid[-1])
    step = grid[1] - grid[0]
    j = np.clip(np.floor((r_hat - grid[0]) / step + 1e-9).astype(int), 0, len(grid) - 2)
    w = (r_hat - grid[j]) / step
    rows = np.arange(len(r_hat))
    return (1 - w) * curves[rows, j] + w * curves[rows, j + 1]


# ---------------------------------------------------------------------------
# 10. Stage 2: damping sweeps, alpha-integrated references, h statistics
# ---------------------------------------------------------------------------
from scipy.special import gammaln as _gammaln            # noqa: E402
from scipy.stats._stats import _kendall_dis               # noqa: E402

# alpha grid: coarse diagnostic points below 0.40, the Stage 2 search grid 0.40..0.99
ALPHA_GRID = np.round(np.concatenate([np.arange(0.05, 0.40, 0.05), np.arange(0.40, 1.0, 0.01)]), 2)
ALPHA_ORACLE_MASK = ALPHA_GRID >= 0.40 - 1e-12
# [NG13] Section 4 ranking experiments used (eta, mu) = (0.85, 0.10): alpha = 0.95, r = 0.10/0.95
ALPHA_NG13 = 0.95
R_NG13 = 0.10 / 0.95


def zt_multi(ops: NCDOperators, X: np.ndarray, r) -> np.ndarray:
    """Z_r^T X for X of shape (n, R) and r of shape (R,), Z_r = (1-r) H + r M
    (dangling rows of H uniform).  Column c uses r[c]."""
    r = np.atleast_1d(np.asarray(r, float))
    Y = ops.HT @ X
    if len(ops.dangling):
        Y += X[ops.dangling].sum(axis=0) / ops.n
    Y *= (1.0 - r)
    Y += r * (ops.RT @ X)[ops.block]
    return Y


def alpha_sweep(ops: NCDOperators, r_values, alphas=ALPHA_GRID, tail: float = 1e-13,
                block: int = 32) -> np.ndarray:
    """NCDawareRank vectors pi(alpha; r) for every alpha in `alphas` and every r in
    `r_values`, from ONE Neumann series per r (batched over r):

        pi(alpha; r)^T = (1 - alpha) * sum_k alpha^k v^T Z_r^k .

    The series for alpha_j is truncated after K_j terms (alpha_j^(K_j+1) <= tail) and
    the remainder is replaced by alpha_j^(K_j+1) v^T Z_r^(K_j+1); the L1 error is at
    most 2*tail.  Iterates are accumulated in blocks with one GEMM per block.  Only
    sparse matvecs and entry-independent scalings are used, so structurally
    equivalent pages keep bit-identical scores and exact ties survive (unlike a
    pivoted direct solve).  Returns S with shape (R, len(alphas), n)."""
    r = np.atleast_1d(np.asarray(r_values, float))
    a = np.asarray(alphas, float)
    assert np.all(np.diff(a) > 0) and a[0] > 0 and a[-1] < 1
    R, nA, n = len(r), len(a), ops.n
    K = np.maximum(np.ceil(np.log(tail) / np.log(a)).astype(np.int64), 1)
    Kmax = int(K.max())
    S = np.zeros((R, nA, n))
    buf = np.empty((R, block, n))
    X = np.repeat(ops.v[:, None], R, axis=1)            # u_0 = v, one column per r
    k0, t = 0, 0
    for k in range(Kmax + 2):
        buf[:, t, :] = X.T
        t += 1
        if t == block or k == Kmax + 1:
            ks = np.arange(k0, k0 + t)
            j0 = int(np.searchsorted(K + 1, k0))          # alphas still contributing
            if j0 < nA:
                aa, KK, kk = a[j0:, None], K[j0:, None], ks[None, :]
                pw = np.exp(kk * np.log(aa))
                W = np.where(kk <= KK, (1.0 - aa) * pw, np.where(kk == KK + 1, pw, 0.0))
                for c in range(R):
                    S[c, j0:, :] += W @ buf[c, :t, :]
            k0, t = k0 + t, 0
        if k <= Kmax:
            X = zt_multi(ops, X, r)
    return S


def rapr_reference(ops: NCDOperators, r: float, a: float = 1.0, b: float = 1.0, K: int = 20000):
    """Random-alpha NCDawareRank: E[pi(A; r)] for A ~ Beta(a, b) (Constantine & Gleich,
    'Random Alpha PageRank').  a = b = 1 gives TotalRank (Boldi 2005), the integral of
    pi(alpha) over alpha in [0, 1], with path weights w_k = 1/((k+1)(k+2)).
    In general w_k = E[(1-A) A^k] = B(a+k, b+1)/B(a, b).  The mass beyond K terms
    (1/(K+2) for TotalRank) is assigned to the (K+1)-th iterate.  Returns (vector, tail)."""
    k = np.arange(K + 1)
    logw = (_gammaln(a + k) + _gammaln(b + 1) - _gammaln(a + b + k + 1)
            - (_gammaln(a) + _gammaln(b) - _gammaln(a + b)))
    w = np.exp(logw)
    rem = max(0.0, 1.0 - float(w.sum()))
    x = ops.v[:, None].copy()
    out = np.zeros(ops.n)
    rr = np.array([r])
    for kk in range(K + 1):
        out += w[kk] * x[:, 0]
        x = zt_multi(ops, x, rr)
    out += rem * x[:, 0]
    return out, rem


def h_statistics(ops: NCDOperators, r: float, sums=None) -> dict:
    """Stage 2 statistics for mixing ratio r.

    h     = mean_v |1 - Sigma_in(v; r)|,  Sigma_in = r Sigma_in^(M) + (1-r) Sigma_in^(H)
          = ||u^T Z_r - u^T||_1 with u the uniform vector (one step of Z_r from uniform)
    mean_sigma = mean_v Sigma_in(v; r)  (equals 1 because Z_r is row-stochastic,
          which requires the uniform dangling rows in Sigma_in^(H))
    h_v   = ||v^T Z_r - v^T||_1 with v the block-even teleportation vector; this is
          the L1 norm of d pi / d alpha at alpha = 0, since pi(alpha) = v + alpha v(Z - I) + O(alpha^2)
    """
    sig_h, sig_m = incoming_sums(ops) if sums is None else sums
    s = r * sig_m + (1.0 - r) * sig_h
    vz = zt_multi(ops, ops.v[:, None], np.array([r]))[:, 0]
    return dict(h=float(np.mean(np.abs(1.0 - s))), mean_sigma=float(np.mean(s)),
                h_v=float(np.abs(vz - ops.v).sum()))


class KendallRef:
    """Kendall's tau-b against a fixed reference y.  Reproduces scipy.stats.kendalltau
    (tau-b) exactly, with the reference's ordering, dense ranks and tie count computed
    once; each call costs one stable argsort of x plus scipy's O(n log n) discordance count."""

    def __init__(self, y):
        y = np.asarray(y)
        self.n = len(y)
        self.perm = np.argsort(y, kind="mergesort")
        ys = y[self.perm]
        self.yd = np.r_[True, ys[1:] != ys[:-1]].cumsum(dtype=np.intp)
        c = np.bincount(self.yd).astype(np.int64)
        c = c[c > 1]
        self.ytie = int((c * (c - 1) // 2).sum())
        self.tot = self.n * (self.n - 1) // 2

    def __call__(self, x) -> float:
        x = np.asarray(x)[self.perm]
        p = np.argsort(x, kind="mergesort")
        xs, yv = x[p], self.yd[p]
        xd = np.r_[True, xs[1:] != xs[:-1]].cumsum(dtype=np.intp)
        dis = int(_kendall_dis(xd, yv))
        obs = np.r_[True, (xd[1:] != xd[:-1]) | (yv[1:] != yv[:-1]), True]
        cnt = np.diff(np.nonzero(obs)[0]).astype(np.int64)
        ntie = int((cnt * (cnt - 1) // 2).sum())
        c = np.bincount(xd).astype(np.int64)
        c = c[c > 1]
        xtie = int((c * (c - 1) // 2).sum())
        if xtie == self.tot or self.ytie == self.tot:
            return float("nan")
        cmd = self.tot - xtie - self.ytie + ntie - 2 * dis
        return float(cmd / np.sqrt(self.tot - xtie) / np.sqrt(self.tot - self.ytie))


def centered_ranks(x):
    rx = stats.rankdata(x)
    rx -= rx.mean()
    return rx, float(np.sqrt((rx ** 2).sum()))


def spearman_from_ranks(rxc, nrm_x, ref: "SpearmanRef") -> float:
    """Spearman's rho from precomputed centred ranks of x (shared across references)."""
    den = nrm_x * ref.nrm
    return float((rxc * ref.rc).sum() / den) if den > 0 else float("nan")
