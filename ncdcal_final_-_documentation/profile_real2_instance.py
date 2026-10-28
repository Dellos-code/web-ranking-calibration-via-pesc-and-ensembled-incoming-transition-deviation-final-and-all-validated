"""Profile one large real2 Stage 2 instance without writing a finished result."""
from __future__ import annotations

import json
import os
import time
import tracemalloc

import numpy as np

import ncd_calibration_lib as L
import run_experiments as R

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    spec = next(s for s in R.panel_specs("real2") if s["base"] == "web-Stanford")
    g, pop, _params, rng = R.make_base(spec)
    tracemalloc.start()
    t0 = time.time()
    ops = L.build_operators(g)
    refs = {
        "NCD": L.ncdawarerank(ops, L.ALPHA0, L.R_CANONICAL, tol=1e-11)[0],
        "NCD95": L.ncdawarerank(ops, L.ALPHA_NG13, L.R_NG13, tol=1e-11, maxit=100000)[0],
    }
    refs["TOT"], tail = L.rapr_reference(ops, L.R_CANONICAL, 1.0, 1.0, K=10000)
    kref = {k: L.KendallRef(v) for k, v in refs.items()}
    go = L.sparsify_arcs(g, 0.5, rng)
    oo = L.build_operators(go)
    feats = L.structural_features(oo)
    rv = R.r_settings(feats["pesc_M"], False)
    active = [c for c, value in enumerate(rv) if np.isfinite(value)]
    curves = L.alpha_sweep(oo, rv[active], L.ALPHA_GRID, tail=1e-13, block=16)
    means = [float(np.mean([kref[t](curves[i, j]) for j in range(len(L.ALPHA_GRID))]))
             for i, t in enumerate(R.R_NAMES[:len(active)])] if False else []
    current, peak = tracemalloc.get_traced_memory()
    result = {"graph": spec["base"], "level": 0.5, "rules": len(active),
              "alpha_grid": len(L.ALPHA_GRID), "truths": list(R.TRUTHS2),
              "seconds": time.time() - t0, "peak_memory_bytes": peak,
              "tot_tail": float(tail)}
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
