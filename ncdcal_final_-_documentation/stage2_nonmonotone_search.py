"""
stage2_nonmonotone_search.py
============================
Non-monotone and dangling-deconfounded Stage 2 rules alpha = g(.) against the
locked constant alpha = 0.810 (primary truth NCD, locked Stage 1 rule for r).

    python3 stage2_nonmonotone_search.py              # replay once (cached), then search
    python3 stage2_nonmonotone_search.py --replay     # force the replay of the column sums
    python3 stage2_nonmonotone_search.py --workers 2

Question
--------
Along both perturbation experiments of [NG13] the oracle damping alpha* falls with
severity, while h falls under Experiment A (uniform link removal: the uniform rows
of the new dangling pages add n_dang/n to every column sum) and rises under
Experiment B (newly added pages: starved columns).  Within one graph, alpha* is
therefore an inverted V in h around the graph's unperturbed value h0.  Can a
deployable rule exploit that shape?

  1. one-dimensional non-monotone families g(x), x in {h, h_v} and the
     deconfounded statistics below: asymmetric tent, symmetric tent, concave
     quadratic, Gaussian bump, two- and three-region step functions and a
     two-knot linear spline (plus the monotone clipped-linear rule as a
     reference), each fitted by least squares to alpha* (LS) and by direct
     maximisation of mean Kendall tau (TAU);
  2. dangling-deconfounded deviations and bivariate rules:
       h_raw     mean_v |1 - (Sigma_in(v; r) - (1 - r) delta)|
                 (the uniform dangling contribution removed from every column sum)
       h_rawrel  mean_v |S(v)/mean(S) - 1|,  S(v) = Sigma_in(v; r) - (1 - r) delta
                 (relative dispersion: the level shift removed as well)
       hH_raw    mean_v |1 - sum_{u->v} 1/d_u|
                 (raw hyperlink matrix, r = 0, before the dangling patch)
       delta     n_dang / n, the dangling fraction
       zeta      the share of pages without an in-link (zero inflow)
     delta and zeta rise under both perturbations.  All five enter as one-dimensional
     rules, and delta and zeta also in rules g(h, delta), g(h, zeta): clipped-linear,
     tent(h) + c s, a 3x3 cell rule and boosted trees (also on (h, h_raw, delta));
  3. two non-deployable diagnostics that locate the obstacle: rules on
     dh = h - h0(graph), which need the unperturbed graph's own h0, and the
     per-graph best constant (an in-sample ceiling for any graph-level rule).

Inputs
------
The cached Stage 2 surfaces results/stage2/<panel>/<base>.npz: Kendall tau of
NCDawareRank(alpha, r) against every truth, for all 3,007 instances and all 67
values of alpha.  No ranking is recomputed: tau at a predicted alpha is read off
the cached curve by linear interpolation (stage2_analysis.interp).  The surfaces
do not store per-page column sums, so the perturbed graphs are replayed with the
seeds and the random stream of `run_experiments.py stage2` (graph generation and
O(|E| + n) column sums only, a few minutes, cached in nonmono_features.csv).  The
replay recomputes h, r and the dangling fraction of every instance and asserts
agreement with the stored values.

Protocol (as in stage2_analysis.py)
-----------------------------------
Rules are fitted on calib only.  Exact 8-fold cross-validation grouped by base
graph (the folds of stage2_analysis.grouped_cv), with every rule AND the constant
refitted inside each fold.  Comparisons with the constant are paired on
per-base-graph means (Wilcoxon signed-rank, 20,000-resample bootstrap CI); Holm
correction across the four held-out panels; the pooled held-out comparison uses
the 73 held-out base graphs as clusters.

Decision rule (fixed before the held-out results were computed)
---------------------------------------------------------------
A deployable rule replaces or joins the constant only if
  (a) its calib-CV gain over the (refitted) constant is >= 0.0005 tau with p < 0.05,
  (b) its pooled held-out gain is >= 0.0005 tau with p < 0.05, and
  (c) no held-out panel shows a Holm-significant loss.
Otherwise alpha = 0.810 stays locked and the results are reported in Section 8.2.

Outputs (results/stage2/)
-------------------------
nonmono_features.csv   replayed per-instance statistics
nonmono_rules.csv      one row per (configuration, rule): parameters, calib in-sample,
                       calib CV and held-out comparisons with the constant
nonmono_summary.json   trajectories, directions of movement, h0 dispersion, decision
"""
import argparse
import json
import multiprocessing as mp
import os
import time
import warnings

import numpy as np
import pandas as pd
from scipy import optimize, stats

import ncd_calibration_lib as L
import run_experiments as RE
import stage2_analysis as S2

warnings.filterwarnings("ignore")
RES = S2.RES
FEAT_CSV = os.path.join(RES, "nonmono_features.csv")
RULES_CSV = os.path.join(RES, "nonmono_rules.csv")
SUMMARY_JSON = os.path.join(RES, "nonmono_summary.json")
GRID, OMASK, ALO, AHI = S2.GRID, S2.OMASK, S2.ALO, S2.AHI
GO = GRID[OMASK]
HELD = ["test", "scale", "topo", "real"]
KEY = ["panel", "base", "exp", "level", "rep"]
MIN_GAIN, P_CRIT = 0.0005, 0.05
H0_RANGE = (0.50, 0.70)          # admissible centre of the inverted V for rules on h
FAMS_1D = ("cliplin", "tent", "symtent", "quad", "gauss", "step2", "step3", "spline2")


# ---------------------------------------------------------------------------
# 1. replay of the perturbed graphs (column sums are not stored in the surfaces)
# ---------------------------------------------------------------------------
def replay_base(spec):
    """Same generator calls and the same random stream as run_experiments.run_base_stage2
    (the reference rankings computed there consume no random numbers)."""
    g, pop, params, rng = RE.make_base(spec)
    static = pop is not None
    rows = []
    for exp, lvl in RE.LEVELS:
        reps = 1 if (exp == "A" and lvl == 1.0) else spec["reps"]
        for rep in range(reps):
            if exp == "A":
                go = g if lvl == 1.0 else L.sparsify_arcs(g, lvl, rng)
            else:
                go = L.newly_added_pages(g, lvl, rng)
            oo = L.build_operators(go)
            rv = RE.r_settings(L.block_escape(oo)[1], static)
            sig_h, sig_m = L.incoming_sums(oo)
            delta = len(oo.dangling) / oo.n
            raw = sig_h - delta                       # sum_{u->v} 1/d_u, raw hyperlink matrix
            row = dict(panel=spec["panel"], base=spec["base"], exp=exp, level=lvl, rep=rep,
                       delta=delta, n_dang=int(len(oo.dangling)), n_pages=int(oo.n),
                       zero_in_rp=float(np.mean(np.asarray(oo.HT.getnnz(axis=1)) == 0)))
            for name in ("rule", "ng13", "can"):
                r = float(rv[RE.R_NAMES.index(name)])
                s = r * sig_m + (1.0 - r) * sig_h      # Sigma_in(v; r), exactly as in L.h_statistics
                sr = s - (1.0 - r) * delta            # dangling contribution removed
                row[f"r_{name}_rp"] = r
                row[f"h_{name}_rp"] = float(np.mean(np.abs(1.0 - s)))
                row[f"hraw_{name}"] = float(np.mean(np.abs(1.0 - sr)))
                row[f"hrawrel_{name}"] = float(np.mean(np.abs(sr / sr.mean() - 1.0)))
            row["hHraw"] = float(np.mean(np.abs(1.0 - raw)))
            row["hHrawrel"] = float(np.mean(np.abs(raw / raw.mean() - 1.0)))
            rows.append(row)
    return rows


def replay(workers):
    specs = [s for p in S2.PANELS for s in RE.panel_specs(p)]
    specs.sort(key=lambda s: (s["panel"] != "real", s["panel"] != "scale"))     # heavy graphs first
    t0 = time.time()
    with mp.get_context("spawn").Pool(workers) as pool:
        out = pool.map(replay_base, specs, chunksize=1)
    fe = pd.DataFrame([r for rows in out for r in rows])
    fe.to_csv(FEAT_CSV, index=False)
    print(f"replayed {len(fe)} instances of {len(specs)} base graphs in {time.time() - t0:.0f} s", flush=True)
    return fe


def load_all():
    df, TAU, _, _ = S2.load()
    fe = pd.read_csv(FEAT_CSV)
    m = df[KEY].merge(fe, on=KEY, how="left", validate="one_to_one")
    assert len(m) == len(df) and m["h_rule_rp"].notna().all(), "replay does not cover every instance"
    checks = {}
    for name in ("rule", "ng13", "can"):
        checks[f"max_abs_diff_h_{name}"] = float(np.max(np.abs(m[f"h_{name}_rp"] - df[f"h_{name}"])))
        checks[f"max_abs_diff_r_{name}"] = float(np.max(np.abs(m[f"r_{name}_rp"] - df[f"r_{name}"])))
    checks["max_abs_diff_dangling_frac"] = float(np.max(np.abs(m["delta"] - df["dangling_frac"])))
    checks["max_abs_diff_zero_in_frac"] = float(np.max(np.abs(m["zero_in_rp"] - df["zero_in_frac"])))
    assert max(checks.values()) < 1e-9, checks
    for c in fe.columns:
        if c not in KEY and not c.endswith("_rp"):
            df[c] = m[c].to_numpy()
    # the graph's own unperturbed statistics (non-deployable diagnostic only)
    a10 = df[(df["exp"] == "A") & (df["level"] == 1.0)].drop_duplicates("base").set_index("base")
    for f in ("h_rule", "h_ng13", "hraw_rule", "hraw_ng13"):
        df["d" + f] = df[f].to_numpy() - df["base"].map(a10[f]).to_numpy()
    return df, TAU, checks


# ---------------------------------------------------------------------------
# 2. rule families
# ---------------------------------------------------------------------------
def _c(a):
    return np.clip(a, ALO, AHI)


class Scale:
    """z = (x - q05)/(q95 - q05) on the training data; all fits are done in z."""

    def __init__(self, x):
        self.lo, self.hi = (float(v) for v in np.quantile(np.asarray(x, float), [0.05, 0.95]))
        self.w = max(self.hi - self.lo, 1e-12)

    def __call__(self, x):
        return (np.asarray(x, float) - self.lo) / self.w

    def inv(self, z):
        return self.lo + self.w * np.asarray(z, float)


def _tent_core(p, z, zb):
    z0 = np.clip(p[1], zb[0], zb[1])
    return p[0] - np.abs(p[2]) * np.maximum(0.0, z0 - z) - np.abs(p[3]) * np.maximum(0.0, z - z0)


def _knots(p):
    k1 = float(np.clip(min(p[4], p[5]), 0.02, 0.96))
    k2 = float(np.clip(max(p[4], p[5]), k1 + 0.02, 0.98))
    return k1, k2


def _spline2(p, z, zb):
    k1, k2 = _knots(p)
    return _c(np.interp(z, [0.0, k1, k2, 1.0], np.asarray(p[:4], float)))


FAM = {
    "cliplin": lambda p, z, zb: _c(p[0] + p[1] * z),
    "tent": lambda p, z, zb: _c(_tent_core(p, z, zb)),
    "symtent": lambda p, z, zb: _c(p[0] - np.abs(p[2]) * np.abs(z - np.clip(p[1], zb[0], zb[1]))),
    "quad": lambda p, z, zb: _c(p[0] - np.abs(p[2]) * (z - np.clip(p[1], zb[0], zb[1])) ** 2),
    "gauss": lambda p, z, zb: _c(p[0] - np.abs(p[3]) * (1.0 - np.exp(
        -0.5 * ((z - np.clip(p[1], zb[0], zb[1])) / np.maximum(np.abs(p[2]), 0.02)) ** 2))),
    "spline2": _spline2,
}
NPAR = dict(cliplin=2, tent=4, symtent=3, quad=3, gauss=4, spline2=6, step2=3, step3=5)
STEP = dict(cliplin=[0.01, 0.02], tent=[0.01, 0.05, 0.03, 0.03], symtent=[0.01, 0.05, 0.03],
            quad=[0.01, 0.05, 0.05], gauss=[0.01, 0.05, 0.05, 0.03], spline2=[0.01] * 4 + [0.05, 0.05])


def grid_points(fam, zb, a0):
    peaks = np.round(a0 + np.arange(-0.03, 0.0901, 0.01), 3)
    z0s = np.linspace(zb[0], zb[1], 21)
    if fam == "tent":
        s = (0.0, 0.02, 0.05, 0.1, 0.2, 0.4)
        return np.array([(a, z, l, r) for a in peaks for z in z0s for l in s for r in s], float)
    if fam == "symtent":
        return np.array([(a, z, s) for a in peaks for z in z0s for s in (0, .01, .02, .05, .1, .2, .4)], float)
    if fam == "quad":
        return np.array([(a, z, c) for a in peaks for z in z0s for c in (0, .02, .05, .1, .2, .5, 1.0)], float)
    if fam == "gauss":
        return np.array([(a, z, s, d) for a in peaks for z in z0s for s in (.05, .1, .2, .4)
                         for d in (0, .02, .05, .1, .2)], float)
    return None


def tau_mean_batch(C, A):
    """Mean tau for every row of A (P x N predicted alphas) on the curves C (N x G)."""
    A = np.clip(A, GRID[0], GRID[-1])
    j = np.clip(np.searchsorted(GRID, A, side="right") - 1, 0, len(GRID) - 2)
    w = (A - GRID[j]) / (GRID[j + 1] - GRID[j])
    rows = np.arange(C.shape[0])[None, :]
    return ((1.0 - w) * C[rows, j] + w * C[rows, j + 1]).mean(axis=1)


def _simplex(x0, step):
    x0 = np.asarray(x0, float)
    S = np.tile(x0, (len(x0) + 1, 1))
    for i in range(len(x0)):
        S[i + 1, i] += step[i]
    return S


def minimise(obj, starts, step):
    """Nelder-Mead from every start (explicit initial simplex), then a Powell polish."""
    best = None
    for s0 in starts:
        r = optimize.minimize(obj, np.asarray(s0, float), method="Nelder-Mead",
                              options=dict(initial_simplex=_simplex(s0, step), maxiter=4000, maxfev=4000,
                                           xatol=1e-7, fatol=1e-11))
        if best is None or r.fun < best.fun:
            best = r
    r = optimize.minimize(obj, best.x, method="Powell", options=dict(maxiter=4000, xtol=1e-7, ftol=1e-12))
    if r.fun < best.fun:
        best = r
    return np.asarray(best.x, float), float(best.fun)


def fit_param(fam, z, y, C, fit, zb, a0, extra_starts=()):
    f, k = FAM[fam], NPAR[fam]
    if fit == "TAU":
        obj = lambda p: -float(np.mean(S2.interp(C, f(p, z, zb))))
        batch = lambda P: -tau_mean_batch(C, f([P[:, [i]] for i in range(k)], z[None, :], zb))
    else:
        obj = lambda p: float(np.mean((f(p, z, zb) - y) ** 2))
        batch = lambda P: ((f([P[:, [i]] for i in range(k)], z[None, :], zb) - y[None, :]) ** 2).mean(axis=1)
    starts = [np.asarray(s, float) for s in extra_starts]
    G = grid_points(fam, zb, a0)
    if G is not None:
        vals = np.concatenate([batch(G[i:i + 384]) for i in range(0, len(G), 384)])
        starts = [G[i] for i in np.argsort(vals, kind="stable")[:3]] + starts
    return minimise(obj, starts, STEP[fam])


def fit_step(x, y, C, k, fit, qs=np.arange(0.05, 0.9501, 0.025)):
    """Exact k-region step function (k = 2, 3) with knots on a quantile grid.  For TAU the
    value of each region is the exact maximiser of the region's mean tau (mean tau is
    piecewise linear in a constant alpha, so the maximum lies on a grid node)."""
    o = np.argsort(x, kind="mergesort")
    xs, N = x[o], len(x)
    idx = np.unique(np.clip(np.round(qs * N).astype(int), 1, N - 1))
    if fit == "TAU":
        cs = np.vstack([np.zeros((1, int(OMASK.sum()))), np.cumsum(C[o][:, OMASK], axis=0)])
        seg = lambda a, b: float((cs[b] - cs[a]).max())
        val = lambda a, b: float(GO[int(np.argmax(cs[b] - cs[a]))])
    else:
        ys = y[o]
        c1, c2 = np.r_[0.0, np.cumsum(ys)], np.r_[0.0, np.cumsum(ys ** 2)]
        seg = lambda a, b: -float((c2[b] - c2[a]) - (c1[b] - c1[a]) ** 2 / (b - a))
        val = lambda a, b: float(np.clip((c1[b] - c1[a]) / (b - a), ALO, AHI))
    best, cut = -np.inf, None
    if k == 2:
        for i in idx:
            s = seg(0, i) + seg(i, N)
            if s > best:
                best, cut = s, (i,)
    else:
        for a_i, i in enumerate(idx):
            for j in idx[a_i + 1:]:
                s = seg(0, i) + seg(i, j) + seg(j, N)
                if s > best:
                    best, cut = s, (i, j)
    b = (0,) + cut + (N,)
    knots = [0.5 * (xs[i - 1] + xs[i]) for i in cut]
    vals = [val(b[q], b[q + 1]) for q in range(len(b) - 1)]
    return np.array(knots), np.array(vals)


class Fitted:
    def __init__(self, name, fam, fit, cols, lab, fn, params, formula, deployable=True, npar=0):
        self.name, self.fam, self.fit, self.cols, self.lab, self.fn = name, fam, fit, list(cols), lab, fn
        self.params, self.formula, self.deployable, self.npar = params, formula, deployable, npar

    def __call__(self, D):
        return np.clip(self.fn(*[D[c].to_numpy(float) for c in self.cols]), ALO, AHI)


def describe(fam, p, sc, zb, lab):
    """Parameters and formula in the original units of the feature."""
    p = np.asarray(p, float)
    if fam == "cliplin":
        b = p[1] / sc.w
        a = p[0] - b * sc.lo
        return dict(a=a, b=b), f"clip({a:.3f} {b:+.3f} {lab})"
    z0 = float(np.clip(p[1], zb[0], zb[1])) if fam != "spline2" else None
    if fam == "tent":
        d = dict(alpha_peak=p[0], x0=float(sc.inv(z0)), s_L=abs(p[2]) / sc.w, s_R=abs(p[3]) / sc.w)
        return d, (f"clip({d['alpha_peak']:.3f} - {d['s_L']:.3f} max(0, {d['x0']:.3f} - {lab}) "
                   f"- {d['s_R']:.3f} max(0, {lab} - {d['x0']:.3f}))")
    if fam == "symtent":
        d = dict(alpha_peak=p[0], x0=float(sc.inv(z0)), s=abs(p[2]) / sc.w)
        return d, f"clip({d['alpha_peak']:.3f} - {d['s']:.3f} |{lab} - {d['x0']:.3f}|)"
    if fam == "quad":
        d = dict(alpha_peak=p[0], x0=float(sc.inv(z0)), c1=abs(p[2]) / sc.w ** 2)
        return d, f"clip({d['alpha_peak']:.3f} - {d['c1']:.3f} ({lab} - {d['x0']:.3f})^2)"
    if fam == "gauss":
        d = dict(alpha_peak=p[0], alpha_min=p[0] - abs(p[3]), x0=float(sc.inv(z0)),
                 sigma=max(abs(p[2]), 0.02) * sc.w)
        return d, (f"clip({d['alpha_min']:.3f} + {abs(p[3]):.3f} exp(-({lab} - {d['x0']:.3f})^2 / "
                   f"(2 * {d['sigma']:.4f}^2)))")
    if fam == "spline2":
        k1, k2 = _knots(p)
        xs = sc.inv(np.array([0.0, k1, k2, 1.0]))
        d = dict(nodes=xs.tolist(), values=np.clip(p[:4], ALO, AHI).tolist())
        return d, "linear spline through " + ", ".join(f"({a:.3f}, {b:.3f})" for a, b in zip(d["nodes"], d["values"]))
    raise ValueError(fam)


def _step_rule(name, fam, fit, col, lab, knots, vals, deployable):
    fn = lambda x, kn=knots, v=vals: v[np.searchsorted(kn, x, side="right")]
    form = " | ".join([f"{vals[0]:.2f}"] + [f"{lab}>={k:.3f}: {v:.2f}" for k, v in zip(knots, vals[1:])])
    return Fitted(name, fam, fit, [col], lab, fn, dict(knots=knots.tolist(), values=vals.tolist()), form,
                  deployable, NPAR[fam])


def fit_1d(col, lab, D, C, y, a0, fams=FAMS_1D, fits=("LS", "TAU"), deployable=True):
    x = D[col].to_numpy(float)
    sc = Scale(x)
    z = sc(x)
    zb = tuple(float(v) for v in sc(np.array(H0_RANGE))) if col in ("h_rule", "h_ng13", "h_can") else (0.0, 1.0)
    zb = (max(zb[0], -0.5), min(zb[1], 1.5))
    b1, b0 = np.polyfit(z, y, 1)
    out, sol = {}, {}
    for fit in fits:
        for fam in fams:
            name = f"{fam}_{fit}[{lab}]"
            if fam in ("step2", "step3"):
                kn, vv = fit_step(x, y, C, 2 if fam == "step2" else 3, fit)
                sol[(fam, fit)] = (kn, vv)
                out[name] = _step_rule(name, fam, fit, col, lab, kn, vv, deployable)
                continue
            extra = []
            if fam == "cliplin":
                extra = [(b0, b1), (a0, 0.0)]
            if fam == "spline2":
                extra = [(a0, a0, a0, a0, 1 / 3, 2 / 3)]
                if ("tent", fit) in sol:
                    pt = sol[("tent", fit)]
                    z0 = float(np.clip(pt[1], 0.05, 0.85))
                    nodes = np.array([0.0, z0, min(z0 + 0.1, 0.95), 1.0])
                    extra.append(tuple(FAM["tent"](pt, nodes, zb)) + (nodes[1], nodes[2]))
                if ("step3", fit) in sol:
                    kn, vv = sol[("step3", fit)]
                    kz = np.clip(np.sort(sc(kn)), 0.05, 0.95)
                    extra.append((vv[0], 0.5 * (vv[0] + vv[1]), 0.5 * (vv[1] + vv[2]), vv[2], kz[0], kz[1]))
            p, _ = fit_param(fam, z, y, C, fit, zb, a0, extra)
            sol[(fam, fit)] = p
            prm, form = describe(fam, p, sc, zb, lab)
            fn = lambda xx, p=p, sc=sc, zb=zb, f=FAM[fam]: f(p, sc(xx), zb)
            out[name] = Fitted(name, fam, fit, [col], lab, fn, prm, "alpha = " + form, deployable, NPAR[fam])
    return out, sol, (sc, zb)


def fit_bivariate(D, C, y, a0, hcol, hlab, tent_sol, hscale, second=("delta", "delta"), extra_raw=None):
    """Rules on (h, s) for a second statistic s (delta, or the zero-inflow share zeta)
    [and, for s = delta, one clipped-linear rule and one boosted-tree rule that add h_raw]."""
    from sklearn.ensemble import HistGradientBoostingRegressor
    scol, slab = second
    out = {}
    xh, xd = D[hcol].to_numpy(float), D[scol].to_numpy(float)
    sh, zbh = hscale
    sd = Scale(xd)
    zh, zd = sh(xh), sd(xd)
    tag = f"{hlab},{slab}"

    def bilin(cols, labs, scs, zs, tg):
        f = lambda p, z1, z2: _c(p[0] + p[1] * z1 + p[2] * z2)
        X = np.c_[np.ones_like(zs[0]), zs[0], zs[1]]
        beta = np.linalg.lstsq(X, y, rcond=None)[0]
        pls, _ = minimise(lambda p: float(np.mean((f(p, *zs) - y) ** 2)), [beta], [0.01, 0.02, 0.02])
        ptau, _ = minimise(lambda p: -float(np.mean(S2.interp(C, f(p, *zs)))), [pls, (a0, 0.0, 0.0)],
                           [0.01, 0.02, 0.02])
        for fit, p in (("LS", pls), ("TAU", ptau)):
            b1, b2 = p[1] / scs[0].w, p[2] / scs[1].w
            a = p[0] - b1 * scs[0].lo - b2 * scs[1].lo
            name = f"bilin_{fit}[{tg}]"
            fn = lambda u, v, p=p, s1=scs[0], s2=scs[1]: f(p, s1(u), s2(v))
            out[name] = Fitted(name, "bilin", fit, cols, tg, fn, dict(a=a, b=b1, c=b2),
                               f"alpha = clip({a:.3f} {b1:+.3f} {labs[0]} {b2:+.3f} {labs[1]})", True, 3)

    bilin([hcol, scol], (hlab, slab), (sh, sd), (zh, zd), tag)
    if extra_raw is not None:
        xr = D[extra_raw[0]].to_numpy(float)
        sr = Scale(xr)
        bilin([extra_raw[0], scol], (extra_raw[1], slab), (sr, sd), (sr(xr), zd), f"{extra_raw[1]},{slab}")
    # tent(h) + c * s
    ft = lambda p, z1, z2: _c(_tent_core(p[:4], z1, zbh) + p[4] * (z2 - 0.5))
    starts = [tuple(tent_sol) + (c,) for c in (-0.05, 0.0, 0.05)]
    p, _ = minimise(lambda q: -float(np.mean(S2.interp(C, ft(q, zh, zd)))), starts, [0.01, 0.05, 0.03, 0.03, 0.02])
    z0 = float(np.clip(p[1], *zbh))
    prm = dict(alpha_peak=p[0] - 0.5 * p[4] - p[4] * sd.lo / sd.w, x0=float(sh.inv(z0)), s_L=abs(p[2]) / sh.w,
               s_R=abs(p[3]) / sh.w, c=p[4] / sd.w)
    name = f"tentplus_TAU[{tag}]"
    out[name] = Fitted(name, "tentplus", "TAU", [hcol, scol], tag,
                       lambda u, v, p=p, sd=sd: ft(p, sh(u), sd(v)), prm,
                       (f"alpha = clip({prm['alpha_peak']:.3f} - {prm['s_L']:.3f} max(0, {prm['x0']:.3f} - {hlab}) - "
                        f"{prm['s_R']:.3f} max(0, {hlab} - {prm['x0']:.3f}) {prm['c']:+.3f} {slab})"), True, 5)
    # 3 x 3 cell rule on tertiles of h and s
    eh, ed = np.quantile(xh, [1 / 3, 2 / 3]), np.quantile(xd, [1 / 3, 2 / 3])
    cell = np.searchsorted(eh, xh, side="right") * 3 + np.searchsorted(ed, xd, side="right")
    for fit in ("LS", "TAU"):
        vals = np.full(9, a0)
        for c in range(9):
            mm = cell == c
            if mm.sum() >= 10:
                vals[c] = GO[int(np.argmax(C[mm][:, OMASK].mean(axis=0)))] if fit == "TAU" else float(np.mean(y[mm]))
        name = f"cells_{fit}[{tag}]"
        fn = lambda u, v, eh=eh, ed=ed, vals=vals: vals[np.searchsorted(eh, u, side="right") * 3 +
                                                           np.searchsorted(ed, v, side="right")]
        out[name] = Fitted(name, "cells", fit, [hcol, scol], tag, fn,
                           dict(h_edges=eh.tolist(), s_edges=ed.tolist(), values=vals.tolist()),
                           f"3x3 cells on tertiles of ({hlab}, {slab}): " + ", ".join(f"{v:.2f}" for v in vals),
                           True, 13)
    # boosted trees (least squares to alpha*)
    combos = [([hcol, scol], tag)]
    if extra_raw is not None:
        combos.append(([hcol, extra_raw[0], scol], f"{hlab},{extra_raw[1]},{slab}"))
    for cols, tg in combos:
        mdl = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, max_leaf_nodes=15, min_samples_leaf=20,
                                            random_state=0).fit(D[cols].to_numpy(float), y)
        name = f"gbm_LS[{tg}]"
        out[name] = Fitted(name, "gbm", "LS", cols, tg, lambda *xs, mdl=mdl: mdl.predict(np.column_stack(xs)),
                           dict(features=cols), "boosted trees on " + ", ".join(cols), True, -1)
    return out


def const_rule(C, col):
    a0 = float(GO[int(np.argmax(C[:, OMASK].mean(axis=0)))])
    return Fitted("const", "const", "TAU", [col], "-", lambda x, a=a0: np.full(len(x), a), dict(alpha=a0),
                  f"alpha = {a0:.3f}", True, 1), a0


def fit_all(D, C, y, cfg):
    """Every rule of one configuration, fitted on (D, C, y)."""
    rules = {}
    cr, a0 = const_rule(C, cfg["h"])
    rules["const"] = cr
    tent_sol, hscale = None, None
    for col, lab in cfg["feats"]:
        out, sol, scz = fit_1d(col, lab, D, C, y, a0, fams=cfg["fams"], fits=cfg["fits"])
        rules.update(out)
        if col == cfg["h"]:
            tent_sol, hscale = sol.get(("tent", "TAU")), scz
    for col, lab in cfg["diag"]:
        out, _, _ = fit_1d(col, lab, D, C, y, a0, fams=cfg["diag_fams"], fits=cfg["fits"], deployable=False)
        rules.update(out)
    for second in cfg["second"]:
        rules.update(fit_bivariate(D, C, y, a0, cfg["h"], cfg["h_lab"], tent_sol, hscale, second,
                                   cfg.get("raw") if second[0] == "delta" else None))
    return rules


# ---------------------------------------------------------------------------
# 3. evaluation
# ---------------------------------------------------------------------------
def make_folds(Dc, k=8, seed=0):
    """The folds of stage2_analysis.grouped_cv (base graphs shuffled with seed 0)."""
    bases = np.array(sorted(Dc["base"].unique()))
    rng = np.random.default_rng(seed)
    rng.shuffle(bases)
    return [Dc["base"].isin(f).to_numpy() for f in np.array_split(bases, k)]


_CV = None
# worker processes are SPAWNED, not forked: the parent has already run scikit-learn's
# OpenMP code (boosted trees), and GNU OpenMP deadlocks in a forked child
SPAWN = mp.get_context("spawn")


def _cv_init(payload):
    global _CV
    _CV = payload


def _cv_fold(i):
    Dc, Cc, yc, cfg = _CV
    te = make_folds(Dc)[i]
    tr = ~te
    rules = fit_all(Dc[tr].reset_index(drop=True), Cc[tr], yc[tr], cfg)
    Dte = Dc[te].reset_index(drop=True)
    return np.flatnonzero(te), {n: ru(Dte) for n, ru in rules.items()}


def cluster(diff, groups):
    return S2.cluster(diff, groups)


def holm(ps):
    ps = np.where(np.isfinite(ps), ps, 1.0)
    order = np.argsort(ps)
    adj, run = np.empty(len(ps)), 0.0
    for k, i in enumerate(order):
        run = max(run, min(1.0, (len(ps) - k) * ps[i]))
        adj[i] = run
    return adj


def within_corr(a, y, groups):
    s = pd.DataFrame(dict(a=a, y=y, g=np.asarray(groups)))
    aw = s["a"] - s.groupby("g")["a"].transform("mean")
    yw = s["y"] - s.groupby("g")["y"].transform("mean")
    return float(np.corrcoef(aw, yw)[0, 1]) if aw.std() > 1e-12 else float("nan")


def graph_constant(D, C):
    """Per-graph best constant alpha (in-sample ceiling of any graph-level rule)."""
    a = np.empty(len(D))
    for b, idx in D.groupby("base").indices.items():
        a[idx] = GO[int(np.argmax(C[idx][:, OMASK].mean(axis=0)))]
    return a


def run_config(cfg, df, TAU, workers):
    t0 = time.time()
    Call = S2.curves(TAU, cfg["r"], cfg["truth"])
    m = S2.valid(df, cfg["truth"]) & np.isfinite(Call[:, -1])
    D, C = df[m].reset_index(drop=True), Call[m]
    a_star, t_star = S2.oracle(C)
    cal = (D["panel"] == "calib").to_numpy()
    Dc, Cc, yc = D[cal].reset_index(drop=True), C[cal], a_star[cal]
    rules = fit_all(Dc, Cc, yc, cfg)
    const = rules["const"]
    print(f"[{cfg['name']}] fitted {len(rules)} rules on calib ({time.time() - t0:.0f} s); constant "
          f"{const.params['alpha']:.3f}", flush=True)
    with SPAWN.Pool(workers, initializer=_cv_init, initargs=((Dc, Cc, yc, cfg),)) as pool:
        res = pool.map(_cv_fold, range(8), chunksize=1)
    cv_a = {n: np.full(len(Dc), np.nan) for n in rules}
    for idx, preds in res:
        for n, a in preds.items():
            cv_a[n][idx] = a
    cv_t = {n: S2.interp(Cc, a) for n, a in cv_a.items()}
    print(f"[{cfg['name']}] 8-fold CV done ({time.time() - t0:.0f} s)", flush=True)
    panels = {p: (D["panel"] == p).to_numpy() for p in ["calib"] + HELD}
    tconst = {p: S2.interp(C[pm], const(D[pm])) for p, pm in panels.items()}
    rows = []

    def add(name, fam, fit, lab, deployable, npar, prm, formula, a_by_panel, cv_alpha, note=""):
        row = dict(config=cfg["name"], truth=cfg["truth"], r_setting=cfg["r"], name=name, family=fam, fit=fit,
                   feature=lab, deployable=deployable, n_params=npar, formula=formula,
                   params=json.dumps(prm, default=float), note=note, const_alpha=const.params["alpha"])
        tt = S2.interp(Cc, a_by_panel["calib"])
        st = cluster(tt - tconst["calib"], Dc["base"])
        row.update(calib_tau=float(tt.mean()), calib_d=st["mean"], calib_p=st["p"], calib_wins=st["wins"],
                   calib_within_corr=within_corr(a_by_panel["calib"], yc, Dc["base"]),
                   calib_pooled_corr=float(np.corrcoef(a_by_panel["calib"], yc)[0, 1])
                   if np.std(a_by_panel["calib"]) > 1e-12 else float("nan"),
                   calib_rmse=float(np.sqrt(np.mean((a_by_panel["calib"] - yc) ** 2))),
                   alpha_hat_sd_calib=float(np.std(a_by_panel["calib"])))
        if cv_alpha is not None:
            tcv = S2.interp(Cc, cv_alpha)
            st = cluster(tcv - cv_t["const"], Dc["base"])
            row.update(cv_tau=float(tcv.mean()), cv_d=st["mean"], cv_lo=st["lo"], cv_hi=st["hi"], cv_p=st["p"],
                       cv_wins=st["wins"], cv_n=st["n"],
                       cv_within_corr=within_corr(cv_alpha, yc, Dc["base"]))
        diffs, groups = [], []
        for p in HELD:
            pm = panels[p]
            d = S2.interp(C[pm], a_by_panel[p]) - tconst[p]
            st = cluster(d, D["base"][pm])
            row.update({f"{p}_d": st["mean"], f"{p}_lo": st["lo"], f"{p}_hi": st["hi"], f"{p}_p": st["p"],
                        f"{p}_wins": st["wins"], f"{p}_n": st["n"],
                        f"{p}_tau": float(np.mean(tconst[p] + d))})
            diffs.append(d)
            groups.append(D["base"][pm].to_numpy())
        ph = holm(np.array([row[f"{p}_p"] for p in HELD], float))
        for p, v in zip(HELD, ph):
            row[f"{p}_pholm"] = float(v)
        st = cluster(np.concatenate(diffs), np.concatenate(groups))
        row.update(held_d=st["mean"], held_lo=st["lo"], held_hi=st["hi"], held_p=st["p"], held_wins=st["wins"],
                   held_n=st["n"])
        rows.append(row)

    for n, ru in rules.items():
        add(n, ru.fam, ru.fit, ru.lab, ru.deployable, ru.npar, ru.params, ru.formula,
            {p: ru(D[pm]) for p, pm in panels.items()}, cv_a[n])
    # non-deployable ceilings
    add("graph_const", "ceiling", "TAU", "graph", False, -1, {}, "per-graph best constant (in-sample)",
        {p: graph_constant(D[pm].reset_index(drop=True), C[pm]) for p, pm in panels.items()},
        graph_constant(Dc, Cc), note="in-sample on every panel")
    add("oracle", "ceiling", "TAU", "instance", False, -1, {}, "per-instance oracle alpha*",
        {p: a_star[pm] for p, pm in panels.items()}, yc, note="in-sample on every panel")
    out = pd.DataFrame(rows)
    out["meets_cv"] = (out["cv_d"] >= MIN_GAIN) & (out["cv_p"] < P_CRIT)
    out["meets_held"] = (out["held_d"] >= MIN_GAIN) & (out["held_p"] < P_CRIT)
    out["no_sig_loss"] = ~np.any([(out[f"{p}_d"] < 0) & (out[f"{p}_pholm"] < P_CRIT) for p in HELD], axis=0)
    out["promote"] = out["deployable"] & (out["name"] != "const") & out["meets_cv"] & out["meets_held"] & \
        out["no_sig_loss"]
    print(f"[{cfg['name']}] evaluated ({time.time() - t0:.0f} s)", flush=True)
    return out


# ---------------------------------------------------------------------------
# 4. mechanism diagnostics on the calibration panel
# ---------------------------------------------------------------------------
TRAJ_FEATS = ["h_rule", "hv_rule", "hraw_rule", "hrawrel_rule", "hHraw", "hHrawrel", "delta", "zero_in_frac"]


def mechanism(df, TAU):
    C = S2.curves(TAU, "rule", "NCD")
    a, _ = S2.oracle(C)
    c = df.assign(astar=a)[df["panel"] == "calib"].copy()
    unpert = (c["exp"] == "A") & (c["level"] == 1.0)
    traj = c.groupby(["exp", "level"])[TRAJ_FEATS + ["astar"]].mean()
    out = dict(trajectories={f"{e}{l:.1f}": {k: float(v) for k, v in r.items()} for (e, l), r in traj.iterrows()})
    c["sev"] = np.where(c["exp"] == "A", 1.0 - c["level"], c["level"])
    dirs = {}
    for e, severe in (("A", 0.2), ("B", 0.6)):
        d = c[(c["exp"] == e) | unpert].copy()            # the unperturbed graph is severity 0 of both
        d_pert = c[(c["exp"] == e) & ~unpert]
        lvl = d.groupby(["exp", "level"])[TRAJ_FEATS].mean()
        for f in TRAJ_FEATS:
            dirs.setdefault(f, {})[e] = dict(
                within_corr_severity=within_corr(d[f].to_numpy(), d["sev"].to_numpy(), d["base"]),
                change_at_severest=float(lvl.loc[(e, severe), f] - lvl.loc[("A", 1.0), f]),
                within_corr_astar=within_corr(d_pert[f].to_numpy(), d_pert["astar"].to_numpy(), d_pert["base"]))
    for f in TRAJ_FEATS:
        a_, b_ = dirs[f]["A"]["change_at_severest"], dirs[f]["B"]["change_at_severest"]
        dirs[f]["same_direction"] = bool(np.sign(a_) == np.sign(b_))
    out["directions"] = dirs
    # dispersion of the unperturbed centre h0 across graphs vs the within-graph excursions
    a10 = c[unpert].set_index("base")
    disp = {}
    pert = c[~unpert]
    gmean = pert.groupby("base")["astar"].mean()
    for f in ("h_rule", "hv_rule", "hraw_rule", "hrawrel_rule"):
        ex = {}
        for e, severe in (("A", 0.2), ("B", 0.6)):
            s = c[(c["exp"] == e) & (c["level"] == severe)].groupby("base")[f].mean()
            ex[e] = float((s - a10[f].reindex(s.index)).mean())
        disp[f] = dict(h0_mean=float(a10[f].mean()), h0_sd=float(a10[f].std()),
                       h0_q05=float(a10[f].quantile(0.05)), h0_q95=float(a10[f].quantile(0.95)),
                       mean_excursion_A=ex["A"], mean_excursion_B=ex["B"],
                       corr_h0_graph_mean_astar=float(stats.pearsonr(a10[f].reindex(gmean.index), gmean)[0]))
    out["h0_dispersion"] = disp
    # distance from the graph's own centre vs distance from the best global centre
    hh, h0g, ast = pert["h_rule"].to_numpy(), pert["base"].map(a10["h_rule"]).to_numpy(), pert["astar"].to_numpy()
    cs = np.round(np.arange(0.50, 0.7001, 0.005), 3)
    cg = [float(np.corrcoef(np.abs(hh - c_), ast)[0, 1]) for c_ in cs]
    j = int(np.argmin(cg))
    out["distance_corr"] = dict(own_centre=float(np.corrcoef(np.abs(hh - h0g), ast)[0, 1]),
                                best_global_centre=float(cs[j]), global_centre_corr=cg[j],
                                own_centre_within=within_corr(np.abs(hh - h0g), ast, pert["base"]))
    out["alpha_star_by_level"] = {f"{e}{l:.1f}": float(v) for (e, l), v in
                                  c[~unpert].groupby(["exp", "level"])["astar"].mean().items()}
    return out


# ---------------------------------------------------------------------------
# 5. configurations
# ---------------------------------------------------------------------------
def configs():
    both = [("delta", "delta"), ("zero_in_frac", "zeta")]      # both rise under A and under B
    prim = dict(name="NCD|rule", truth="NCD", r="rule", h="h_rule", h_lab="h",
                feats=[("h_rule", "h"), ("hv_rule", "h_v"), ("hraw_rule", "h_raw"), ("hrawrel_rule", "h_rawrel"),
                       ("hHraw", "hH_raw"), ("delta", "delta"), ("zero_in_frac", "zeta")],
                fams=FAMS_1D, fits=("LS", "TAU"),
                diag=[("dh_rule", "dh"), ("dhraw_rule", "dh_raw")], diag_fams=("tent", "quad", "step3"),
                second=both, raw=("hraw_rule", "h_raw"))
    light = ("tent", "symtent", "quad", "gauss", "step3")
    ncd95 = dict(name="NCD95|ng13", truth="NCD95", r="ng13", h="h_ng13", h_lab="h",
                 feats=[("h_ng13", "h"), ("hraw_ng13", "h_raw"), ("zero_in_frac", "zeta")], fams=light, fits=("TAU",),
                 diag=[("dh_ng13", "dh")], diag_fams=("tent",), second=both, raw=("hraw_ng13", "h_raw"))
    tot = dict(name="TOT|rule", truth="TOT", r="rule", h="h_rule", h_lab="h",
               feats=[("h_rule", "h"), ("hraw_rule", "h_raw"), ("zero_in_frac", "zeta")], fams=light, fits=("TAU",),
               diag=[("dh_rule", "dh")], diag_fams=("tent",), second=both, raw=("hraw_rule", "h_raw"))
    return [prim, ncd95, tot]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--replay", action="store_true")
    ap.add_argument("--workers", type=int, default=max(1, min(8, os.cpu_count() or 1)))
    ap.add_argument("--only", default=None, help="run a single configuration (e.g. 'NCD|rule')")
    a = ap.parse_args()
    if a.replay or not os.path.exists(FEAT_CSV):
        replay(a.workers)
    df, TAU, checks = load_all()
    print("replay checks:", json.dumps(checks), flush=True)
    mech = mechanism(df, TAU)
    tabs = []
    for cfg in configs():
        if a.only and cfg["name"] != a.only:
            continue
        tabs.append(run_config(cfg, df, TAU, a.workers))
    tab = pd.concat(tabs, ignore_index=True)
    tab.to_csv(RULES_CSV, index=False)
    prim = tab[tab["config"] == "NCD|rule"]
    dep = prim[prim["deployable"] & (prim["name"] != "const")]
    decision = dict(
        criteria=dict(min_gain=MIN_GAIN, p_crit=P_CRIT,
                      text="promote iff CV gain >= min_gain with p < p_crit, pooled held-out gain >= min_gain "
                           "with p < p_crit, and no Holm-significant held-out loss"),
        n_deployable_rules=int(len(dep)),
        promoted=dep[dep["promote"]]["name"].tolist(),
        best_cv=dep.sort_values("cv_d", ascending=False).iloc[0][["name", "cv_d", "cv_p", "held_d", "held_p"]].to_dict()
        if len(dep) else {},
        best_held=dep.sort_values("held_d", ascending=False).iloc[0][["name", "cv_d", "cv_p", "held_d", "held_p"]]
        .to_dict() if len(dep) else {},
        best_calib=dep.sort_values("calib_d", ascending=False).iloc[0][["name", "calib_d", "cv_d", "held_d"]].to_dict()
        if len(dep) else {},
        n_meets_cv=int(dep["meets_cv"].sum()), n_meets_held=int(dep["meets_held"].sum()),
        n_positive_cv=int((dep["cv_d"] > 0).sum()), n_positive_held=int((dep["held_d"] > 0).sum()),
        locked_constant=float(prim[prim["name"] == "const"]["const_alpha"].iloc[0]) if len(prim) else None)
    decision["verdict"] = "promote" if decision["promoted"] else "keep_constant"
    json.dump(dict(checks=checks, mechanism=mech, decision=decision), open(SUMMARY_JSON, "w"), indent=2, default=float)
    cols = ["name", "calib_d", "cv_d", "cv_p", "held_d", "held_p", "test_d", "scale_d", "topo_d", "real_d", "promote"]
    with pd.option_context("display.width", 200, "display.max_rows", 300, "display.float_format", "{:+.5f}".format):
        for cname, sub in tab.groupby("config", sort=False):
            print(f"\n=== {cname} (constant {sub['const_alpha'].iloc[0]:.3f}) ===")
            print(sub[cols].to_string(index=False))
    print("\nverdict:", decision["verdict"], json.dumps(decision, default=float, indent=1))


if __name__ == "__main__":
    main()
