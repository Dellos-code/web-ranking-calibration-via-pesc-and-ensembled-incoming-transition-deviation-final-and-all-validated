"""WP4 B3 descriptive policies on eight damaged snapshots per graph."""
from __future__ import annotations

import csv
import os
import json

import numpy as np

import ncd_calibration_lib as L
import run_experiments as R

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    rows = []
    for panel in ("calib", "test", "topo"):
        for spec in R.panel_specs(panel):
            g, _pop, _params, rng = R.make_base(spec)
            full = L.build_operators(g)
            ref = L.ncdawarerank(full, .85, L.R_CANONICAL, tol=1e-12)[0]
            kref = L.KendallRef(ref)
            hs, curves = [], []
            for step in range(8):
                keep = max(.2, 1.0 - .1 * step)
                pages = .1 if step not in (3, 7) else .0
                go = L.mixed_damage(g, keep, pages, rng)
                oo = L.build_operators(go)
                r = float(np.clip(.242 - .143 * L.structural_features(oo)["pesc_M"], .01, .99))
                h = L.h_statistic(oo, r, L.incoming_sums(oo))[0]
                curves.append((oo, r, h))
                hs.append(h)
            policies = {
                "reset": [hs[max(0, i - 1)] for i in range(8)],
                "best_so_far": [min(hs[:i + 1]) for i in range(8)],
                "median_last_k": [float(np.median(hs[max(0, i - 2):i + 1])) for i in range(8)],
            }
            avg = hs[0]
            policies["exponential"] = []
            for h in hs:
                avg = .5 * avg + .5 * h
                policies["exponential"].append(avg)
            for name, baselines in policies.items():
                gains = []
                for (oo, r, _), bh in zip(curves, baselines):
                    alpha = float(np.clip(.81 + .25 * (hs[len(gains)] - bh), .40, .99))
                    aa = np.array(sorted({.81, alpha}))
                    score = L.alpha_sweep(oo, [r], aa, tail=1e-13)[0]
                    gains.append(float(kref(score[int(np.argmin(abs(aa - alpha)))]) -
                                      kref(score[int(np.argmin(abs(aa - .81)))])))
                rows.append({"panel": panel, "base": spec["base"], "policy": name,
                             "mean_gain": float(np.mean(gains))})
        print(f"{panel} complete", flush=True)
    out = os.path.join(HERE, "results", "baselines_v2")
    with open(os.path.join(out, "b3.csv"), "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=sorted(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    with open(os.path.join(out, "b3_summary.json"), "w") as fh:
        json.dump({"rows": len(rows), "policies": sorted(set(r["policy"] for r in rows))},
                  fh, indent=2)


if __name__ == "__main__":
    main()
