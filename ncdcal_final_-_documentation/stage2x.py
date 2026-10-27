"""NCD(a) surface driver for the operational anchor extension.

The output contains tau(alpha) surfaces for the locked Stage-1 r rule and the
reference-r sensitivity, evaluated against NCD(a) truths.  Truth construction
is independent of the existing stage2 result files and includes an explicit
0.85/0.95 anchor reproduction check.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from multiprocessing import Pool

import numpy as np

import ncd_calibration_lib as L
import run_experiments as R

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "results", "stage2x")
A_VALUES = np.array([0.50, 0.60, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 0.97])
ALPHA = L.ALPHA_GRID


def _truths(ops):
    refs = []
    for a in A_VALUES:
        refs.append(L.ncdawarerank(ops, float(a), 0.10 / float(a),
                                   tol=1e-11 if ops.n >= 100000 else 1e-12,
                                   maxit=100000)[0])
    # The existing NCD and NCD95 references must be reproduced exactly enough
    # for the stored tau surfaces to be interchangeable.
    old085 = L.ncdawarerank(ops, L.ALPHA0, L.R_CANONICAL,
                            tol=1e-11 if ops.n >= 100000 else 1e-12)[0]
    old095 = L.ncdawarerank(ops, L.ALPHA_NG13, L.R_NG13,
                            tol=1e-11 if ops.n >= 100000 else 1e-12,
                            maxit=100000)[0]
    i85, i95 = int(np.where(np.isclose(A_VALUES, .85))[0][0]), int(np.where(np.isclose(A_VALUES, .95))[0][0])
    e85, e95 = np.max(np.abs(refs[i85] - old085)), np.max(np.abs(refs[i95] - old095))
    if e85 > 2e-10 or e95 > 2e-10:
        raise RuntimeError(f"NCD(a) anchor reproduction failed: {e85:.3g}, {e95:.3g}")
    return refs, {"anchor_max_abs_085": float(e85), "anchor_max_abs_095": float(e95)}


def run_one(spec):
    out = os.path.join(OUT, spec["panel"], f"{spec['base']}.npz")
    if os.path.exists(out):
        return spec["base"], "cached", 0.0
    t0 = time.time()
    g, _pop, params, rng = R.make_base(spec)
    ops = L.build_operators(g)
    refs, check = _truths(ops)
    ref_objs = [L.KendallRef(x) for x in refs]
    rvals = []
    rows = []
    surfaces = []
    for exp, lvl in R.LEVELS:
        reps = 1 if (exp == "A" and lvl == 1.0) else spec["reps"]
        for rep in range(reps):
            go = (g if exp == "A" and lvl == 1.0 else
                  L.sparsify_arcs(g, lvl, rng) if exp == "A" else
                  L.newly_added_pages(g, lvl, rng))
            oo = L.build_operators(go)
            feat = L.structural_features(oo)
            r_rule = float(np.clip(.242 - .143 * feat["pesc_M"], .01, .99))
            r_ref = 0.10 / .85
            S = L.alpha_sweep(oo, [r_rule, r_ref], ALPHA,
                              tail=1e-13, block=16 if g.n >= 100000 else 32)
            tau = np.empty((2, len(A_VALUES), len(ALPHA)), dtype=np.float32)
            for ir in range(2):
                for ia in range(len(ALPHA)):
                    x = S[ir, ia]
                    for it, ref in enumerate(ref_objs):
                        tau[ir, it, ia] = ref(x)
            surfaces.append(tau)
            rvals.append((r_rule, r_ref))
            rows.append(dict(panel=spec["panel"], base=spec["base"], exp=exp, level=lvl,
                             rep=rep, **feat))
    os.makedirs(os.path.dirname(out), exist_ok=True)
    meta = dict(spec=spec, params=params, a_values=A_VALUES.tolist(),
                alpha_grid=ALPHA.tolist(), r_names=["rule", "ref085"],
                rows=rows, anchor_check=check, seconds=time.time() - t0)
    np.savez_compressed(out, tau=np.asarray(surfaces), r=np.asarray(rvals),
                        meta=json.dumps(meta))
    return spec["base"], "done", time.time() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--panel", default="calib,test,topo,real")
    ap.add_argument("--workers", type=int, default=2)
    args = ap.parse_args()
    panels = args.panel.split(",")
    specs = [s for p in panels for s in R.panel_specs(p)]
    specs.sort(key=lambda s: -(s["n"] or 0))
    t0 = time.time()
    with Pool(args.workers) as pool:
        for i, (name, status, sec) in enumerate(pool.imap_unordered(run_one, specs), 1):
            print(f"[{i}/{len(specs)}] {name} {status} {sec:.1f}s (elapsed {time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
