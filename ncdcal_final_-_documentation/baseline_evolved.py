"""WP4 B2 evolved-baseline stress test with fixed WP3 C1 parameters."""
from __future__ import annotations

import csv
import json
import os
import time

import numpy as np

import ncd_calibration_lib as L
import run_experiments as R

HERE = os.path.dirname(os.path.abspath(__file__))


def induced(g, fraction, rng, ordered):
    if ordered:
        nodes = np.arange(g.n)
    else:
        nodes = rng.permutation(g.n)
    keep = np.sort(nodes[:max(2, int(round((1 - fraction) * g.n)))])
    mask = np.zeros(g.n, dtype=bool); mask[keep] = True
    arcs = mask[g.src] & mask[g.dst]
    remap = np.full(g.n, -1, dtype=np.int64); remap[keep] = np.arange(len(keep))
    src, dst = remap[g.src[arcs]], remap[g.dst[arcs]]
    block = g.block[keep]
    _, block = np.unique(block, return_inverse=True)
    return L.Graph(len(keep), src, dst, block, g.name + "_evolved")


def main():
    out_dir = os.path.join(HERE, "results", "baselines_v2")
    os.makedirs(out_dir, exist_ok=True)
    rows = []
    t0 = time.time()
    for panel in ("calib", "test", "topo"):
        ordered = panel == "topo"
        for spec in R.panel_specs(panel):
            g, _pop, _params, rng = R.make_base(spec)
            full = L.build_operators(g)
            r0 = float(np.clip(.242 - .143 * L.structural_features(full)["pesc_M"], .01, .99))
            ref = L.ncdawarerank(full, .85, L.R_CANONICAL, tol=1e-12)[0]
            kref = L.KendallRef(ref)
            for gamma in (.05, .10, .20, .40):
                for keep in (1.0, .9, .7):
                    base = induced(g, gamma, rng, ordered)
                    base = L.sparsify_arcs(base, keep, rng) if keep < 1 else base
                    bh = L.h_statistic(L.build_operators(base), r0,
                                       L.incoming_sums(L.build_operators(base)))[0]
                    current = L.mixed_damage(g, .6, .3, rng)
                    co = L.build_operators(current)
                    cr = float(np.clip(.242 - .143 * L.structural_features(co)["pesc_M"], .01, .99))
                    ch = L.h_statistic(co, cr, L.incoming_sums(co))[0]
                    alpha = float(np.clip(.81 + 0.25 * (ch - bh), .40, .99))
                    aa = np.array(sorted({.81, alpha}))
                    curve = L.alpha_sweep(co, [cr], aa, tail=1e-13)[0]
                    tau = [kref(curve[i]) for i in range(len(aa))]
                    rows.append({"panel": panel, "base": spec["base"], "gamma": gamma,
                                 "baseline_keep": keep, "baseline_h": bh, "current_h": ch,
                                 "damage_order": "growth_order" if ordered else "random_order",
                                 "delta_h": ch - bh, "alpha": alpha,
                                 "gain_over_constant": float(tau[int(np.argmin(abs(aa - alpha)))] -
                                                               tau[int(np.argmin(abs(aa - .81)))])})
        print(f"{panel} complete", flush=True)
    with open(os.path.join(out_dir, "b2.csv"), "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=sorted(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    with open(os.path.join(out_dir, "b2_summary.json"), "w") as fh:
        json.dump({"rows": len(rows), "runtime_seconds": time.time() - t0,
                   "panels": ["calib", "test", "topo"]}, fh, indent=2)


if __name__ == "__main__":
    main()
