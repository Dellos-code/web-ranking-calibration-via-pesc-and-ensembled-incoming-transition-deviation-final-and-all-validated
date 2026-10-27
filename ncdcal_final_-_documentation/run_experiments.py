"""
run_experiments.py
==================
Experiment driver for both calibration stages.

    python3 run_experiments.py stage1 --panel calib  [--workers 2]
    python3 run_experiments.py stage1 --panel all
    python3 run_experiments.py stage2 --panel all    (replays the Stage 1 instances)

Panels (split by BASE GRAPH, never by perturbed instance)
---------------------------------------------------------
calib   static fitness block web, n = 2000, 64 base graphs, 2 replicates
test    same generator, independent seeds, n = 2000, 32 base graphs
scale   same generator, n = 16000, 12 base graphs           (scale transfer)
topo    growth / preferential-attachment block web, n = 3000, 24 bases
        (topology transfer: different generative mechanism)
real    cnr-2000 (host proxy), WikiCS, Cora-full, Cora-ML, CiteSeer

Perturbations ([NG13] Section 4.2)
----------------------------------
A  sparsity: keep p in {1.0, 0.9, ..., 0.2} of the arcs (A1.0 = unperturbed)
B  newly added pages: pick f in {0.1, ..., 0.6} of the pages, delete each of
   their in-arcs with probability 0.9

Ground truths (all computed on the complete base graph, alpha0 = 0.85,
block-even teleportation, uniform dangling rows)
-----------------------------------------------------------------------
NCD   canonical NCDawareRank, (eta, mu) = (0.75, 0.10), r = 0.1176
PR    PageRank, r = 0
SELF  NCDawareRank with the same r as the candidate ([NG13]'s protocol:
      each method against its own ranking on the complete graph)
POP   PageRank of the generator's expected walk (static generator only):
      a latent importance that no ranking method defines

For every perturbed instance the observed graph is ranked with
NCDawareRank(alpha0 = 0.85, r) for r on R_GRID = 0.00..0.99 and Kendall's
tau-b and Spearman's rho against each ground truth are stored, together
with the structural features of the OBSERVED graph.
"""
import argparse
import json
import os
import sys
import time
from multiprocessing import Pool

import numpy as np

import ncd_calibration_lib as L

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results", "stage1")
GTS = ["NCD", "PR", "SELF", "POP"]
LEVELS = [("A", p) for p in (1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2)] + \
         [("B", f) for f in (0.1, 0.2, 0.3, 0.4, 0.5, 0.6)]
REAL = ["cora_ml", "citeseer", "cora_full", "wikics", "cnr2000"]
REAL2 = ["cornell", "texas", "wisconsin", "film", "email-Eu-core",
         "web-NotreDame", "web-Stanford", "ogbn-arxiv"]


def panel_specs(panel):
    if panel == "calib":
        return [dict(panel="calib", base=f"calib_{i:03d}", kind="static", n=2000, seed=1000 + i, reps=2)
                for i in range(64)]
    if panel == "test":
        return [dict(panel="test", base=f"test_{i:03d}", kind="static", n=2000, seed=5000 + i, reps=1)
                for i in range(32)]
    if panel == "scale":
        return [dict(panel="scale", base=f"scale_{i:03d}", kind="static", n=16000, seed=9000 + i, reps=1)
                for i in range(12)]
    if panel == "topo":
        return [dict(panel="topo", base=f"topo_{i:03d}", kind="growth", n=3000, seed=13000 + i, reps=1)
                for i in range(24)]
    if panel == "real":
        return [dict(panel="real", base=name, kind="real", n=None, seed=17000 + j,
                     reps=1 if name == "cnr2000" else 2) for j, name in enumerate(REAL)]
    if panel == "real2":
        return [dict(panel="real2", base=name, kind="real2", n=None, seed=17100 + j,
                     reps=10 if name in {"cornell", "texas", "wisconsin"} else 2)
                for j, name in enumerate(REAL2)]
    raise ValueError(panel)


def make_base(spec):
    rng = np.random.default_rng(spec["seed"])
    pop = None
    if spec["kind"] == "static":
        params = L.sample_static_params(rng, spec["n"])
        g, pop = L.generate_static_web(params, rng)
    elif spec["kind"] == "growth":
        params = L.sample_growth_params(rng, spec["n"])
        g = L.generate_growth_web(params, rng)
    else:
        folder = "real2" if spec["kind"] == "real2" else "real"
        z = np.load(os.path.join(HERE, "data", folder, f"{spec['base']}.npz"))
        g = L.Graph(int(z["n"]), z["src"], z["dst"], z["block"], name=spec["base"])
        params = {"dataset": spec["base"]}
    return g, pop, params, rng


def run_base(spec):
    out = os.path.join(RES, spec["panel"], f"{spec['base']}.npz")
    if os.path.exists(out):
        return spec["base"], "cached", 0.0
    t0 = time.time()
    g, pop, params, rng = make_base(spec)
    ops = L.build_operators(g)
    tol_ref = 1e-12 if g.n < 100000 else 1e-11
    full = [x for _, x, _ in L.sweep_r(ops, L.ALPHA0, L.R_GRID, tol=1e-10)]
    refs = {"NCD": L.ncdawarerank(ops, L.ALPHA0, L.R_CANONICAL, tol=tol_ref)[0],
            "PR": L.ncdawarerank(ops, L.ALPHA0, 0.0, tol=tol_ref)[0]}
    if pop is not None:
        refs["POP"] = pop.pagerank(L.ALPHA0, ops.v)
    sp_refs = {k: L.SpearmanRef(v) for k, v in refs.items()}
    full_feats = L.structural_features(ops)

    G, nr = len(GTS), len(L.R_GRID)
    rows, TAU, RHO, TAUc, RHOc, ITERS = [], [], [], [], [], []
    for exp, lvl in LEVELS:
        reps = 1 if (exp == "A" and lvl == 1.0) else spec["reps"]
        for rep in range(reps):
            if exp == "A":
                go = g if lvl == 1.0 else L.sparsify_arcs(g, lvl, rng)
            else:
                go = L.newly_added_pages(g, lvl, rng)
            oo = L.build_operators(go)
            feats = L.structural_features(oo)
            tau = np.full((G, nr), np.nan, dtype=np.float32)
            rho = np.full((G, nr), np.nan, dtype=np.float32)
            its = []
            for j, (r, x, it) in enumerate(L.sweep_r(oo, L.ALPHA0, L.R_GRID, tol=1e-10)):
                its.append(it)
                for gi, k in enumerate(GTS):
                    if k == "SELF":
                        tau[gi, j] = L.kendall_tau_b(x, full[j])
                        rho[gi, j] = L.spearman_rho(x, full[j])
                    elif k in refs:
                        tau[gi, j] = L.kendall_tau_b(x, refs[k])
                        rho[gi, j] = sp_refs[k](x)
            xc, _ = L.ncdawarerank(oo, L.ALPHA0, L.R_CANONICAL, tol=1e-10)
            fullc = refs["NCD"]
            tc = np.full(G, np.nan, dtype=np.float32)
            rc = np.full(G, np.nan, dtype=np.float32)
            for gi, k in enumerate(GTS):
                ref = fullc if k == "SELF" else refs.get(k)
                if ref is not None:
                    tc[gi] = L.kendall_tau_b(xc, ref)
                    rc[gi] = L.spearman_rho(xc, ref)
            TAU.append(tau); RHO.append(rho); TAUc.append(tc); RHOc.append(rc); ITERS.append(its)
            rows.append(dict(panel=spec["panel"], base=spec["base"], kind=spec["kind"], exp=exp,
                             level=lvl, rep=rep, keep_frac=go.m / g.m, **feats))
    os.makedirs(os.path.dirname(out), exist_ok=True)
    meta = dict(spec=spec, params=params, full_features=full_feats, gts=GTS,
                r_grid=L.R_GRID.tolist(), r_canonical=L.R_CANONICAL, rows=rows,
                seconds=time.time() - t0)
    np.savez_compressed(out, tau=np.array(TAU), rho=np.array(RHO), tau_can=np.array(TAUc),
                        rho_can=np.array(RHOc), iters=np.array(ITERS, dtype=np.int32),
                        meta=json.dumps(meta))
    return spec["base"], "done", time.time() - t0


# ---------------------------------------------------------------------------
# Stage 2
# ---------------------------------------------------------------------------
RES2 = os.path.join(HERE, "results", "stage2")
R_NAMES = ["can", "ng13", "const", "rule", "popLS", "popTAU"]
TRUTHS2 = ["NCD", "NCD95", "TOT", "POP", "Q"]
# which r-settings are evaluated against which truth
PAIRS = {"NCD": ["can", "ng13", "const", "rule"],
         "NCD95": ["can", "ng13", "rule"],
         "TOT": ["can", "ng13", "const", "rule"],
         "POP": ["can", "ng13", "const", "rule", "popLS", "popTAU"],
         "Q": ["can", "ng13", "const", "rule", "popLS", "popTAU"]}
# locked Stage 1 rules (Checkpoint 1)
RULE_NCD = (0.242, -0.143)        # cliplin_TAU fitted on the NCD truth
RULE_POP_LS = (0.420, -0.386)     # cliplin_LS fitted on the POP truth
RULE_POP_TAU = (0.448, -0.547)    # cliplin_TAU fitted on the POP truth


def r_settings(pbar, static):
    """Stage 1 mixing ratios evaluated in Stage 2 (np.nan = not evaluated)."""
    clip = lambda c: float(np.clip(c[0] + c[1] * pbar, 0.01, 0.99))
    vals = {"can": L.R_CANONICAL, "ng13": L.R_NG13, "const": 0.18, "rule": clip(RULE_NCD),
            "popLS": clip(RULE_POP_LS) if static else np.nan,
            "popTAU": clip(RULE_POP_TAU) if static else np.nan}
    return np.array([vals[k] for k in R_NAMES])


def run_base_stage2(spec):
    """Replay the Stage 1 instances of one base graph and run the Stage 2 sweep.

    For every instance and every evaluated r-setting the observed graph is ranked
    by NCDawareRank(alpha, r) for all alpha in ALPHA_GRID (one Neumann series per r)
    and compared with every truth listed in PAIRS (Kendall tau-b, Spearman rho).
    Truths, computed on the complete base graph:
      NCD    canonical NCDawareRank, alpha = 0.85, r = 0.10/0.85      (primary)
      NCD95  [NG13]'s experimental setting, alpha = 0.95, r = 0.10/0.95
      TOT    alpha-integrated (TotalRank = RAPr with Beta(1,1)) NCDawareRank, r = 0.10/0.85
      POP    PageRank (alpha = 0.85) of the generator's expected walk   (static only)
      Q      latent page quality q of the generator, walk-free         (static only)
    """
    out = os.path.join(RES2, spec["panel"], f"{spec['base']}.npz")
    if os.path.exists(out):
        return spec["base"], "cached", 0.0
    t0 = time.time()
    g, pop, params, rng = make_base(spec)
    static = pop is not None
    ops = L.build_operators(g)
    big = g.n >= 100000
    tol_ref = 1e-11 if big else 1e-12
    refs = {"NCD": L.ncdawarerank(ops, L.ALPHA0, L.R_CANONICAL, tol=tol_ref)[0],
            "NCD95": L.ncdawarerank(ops, L.ALPHA_NG13, L.R_NG13, tol=tol_ref, maxit=100000)[0]}
    refs["TOT"], tot_tail = L.rapr_reference(ops, L.R_CANONICAL, 1.0, 1.0, K=10000 if big else 20000)
    if static:
        refs["POP"] = pop.pagerank(L.ALPHA0, ops.v)
        refs["Q"] = pop.q
    kref = {k: L.KendallRef(v) for k, v in refs.items()}
    sref = {k: L.SpearmanRef(v) for k, v in refs.items()}
    nA, C, T = len(L.ALPHA_GRID), len(R_NAMES), len(TRUTHS2)
    rows, TAU, RHO, RV, HH, HV, MS = [], [], [], [], [], [], []
    for exp, lvl in LEVELS:
        reps = 1 if (exp == "A" and lvl == 1.0) else spec["reps"]
        for rep in range(reps):
            if exp == "A":
                go = g if lvl == 1.0 else L.sparsify_arcs(g, lvl, rng)
            else:
                go = L.newly_added_pages(g, lvl, rng)
            oo = L.build_operators(go)
            feats = L.structural_features(oo)
            rv = r_settings(feats["pesc_M"], static)
            active = [c for c in range(C) if np.isfinite(rv[c])]
            sums = L.incoming_sums(oo)
            hh, hv, ms = np.full(C, np.nan), np.full(C, np.nan), np.full(C, np.nan)
            for c in active:
                st = L.h_statistics(oo, rv[c], sums)
                hh[c], hv[c], ms[c] = st["h"], st["h_v"], st["mean_sigma"]
            ext = {f"h_r{x}": L.h_statistics(oo, x, sums) for x in (0.0, 1.0)}
            S = L.alpha_sweep(oo, rv[active], L.ALPHA_GRID, tail=1e-13, block=16 if big else 32)
            tau = np.full((C, T, nA), np.nan, dtype=np.float32)
            rho = np.full((C, T, nA), np.nan, dtype=np.float32)
            for ia, c in enumerate(active):
                name = R_NAMES[c]
                tl = [ti for ti, t in enumerate(TRUTHS2) if t in refs and name in PAIRS[t]]
                for j in range(nA):
                    x = S[ia, j]
                    rxc, nrm = L.centered_ranks(x)
                    for ti in tl:
                        t = TRUTHS2[ti]
                        tau[c, ti, j] = kref[t](x)
                        rho[c, ti, j] = L.spearman_from_ranks(rxc, nrm, sref[t])
            del S
            TAU.append(tau); RHO.append(rho); RV.append(rv); HH.append(hh); HV.append(hv); MS.append(ms)
            rows.append(dict(panel=spec["panel"], base=spec["base"], kind=spec["kind"], exp=exp, level=lvl,
                             rep=rep, keep_frac=go.m / g.m, **feats,
                             h_H=ext["h_r0.0"]["h"], h_M=ext["h_r1.0"]["h"],
                             hv_H=ext["h_r0.0"]["h_v"], hv_M=ext["h_r1.0"]["h_v"]))
    os.makedirs(os.path.dirname(out), exist_ok=True)
    meta = dict(spec=spec, params=params, r_names=R_NAMES, truths=TRUTHS2, pairs=PAIRS,
                alpha_grid=L.ALPHA_GRID.tolist(), rule_ncd=RULE_NCD, rule_pop_ls=RULE_POP_LS,
                rule_pop_tau=RULE_POP_TAU, tot_tail=tot_tail, rows=rows, seconds=time.time() - t0)
    np.savez_compressed(out, tau=np.array(TAU), rho=np.array(RHO), r=np.array(RV), h=np.array(HH),
                        h_v=np.array(HV), mean_sigma=np.array(MS), meta=json.dumps(meta))
    return spec["base"], "done", time.time() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["stage1", "stage2"])
    ap.add_argument("--panel", default="calib")
    ap.add_argument("--workers", type=int, default=2)
    a = ap.parse_args()
    panels = ["calib", "test", "topo", "scale", "real", "real2"] if a.panel == "all" else a.panel.split(",")
    specs = [s for p in panels for s in panel_specs(p)]
    # largest jobs first so the pool stays busy
    real_n = {}
    summ = os.path.join(HERE, "data", "real", "summary.json")
    if os.path.exists(summ):
        real_n = {k: v["n"] for k, v in json.load(open(summ)).items()}
    summ2 = os.path.join(HERE, "data", "real2", "provenance.json")
    if os.path.exists(summ2):
        real_n.update({k: v["n"] for k, v in json.load(open(summ2)).items()})
    specs.sort(key=lambda s: -(s["n"] or real_n.get(s["base"], 1e6)))   # largest graphs first
    t0 = time.time()
    job = run_base if a.cmd == "stage1" else run_base_stage2
    with Pool(a.workers) as pool:
        for i, (base, status, sec) in enumerate(pool.imap_unordered(job, specs), 1):
            print(f"[{i}/{len(specs)}] {base} {status} {sec:.1f}s  (elapsed {time.time()-t0:.0f}s)",
                  flush=True)


if __name__ == "__main__":
    sys.exit(main())
