"""WP4 B1 full synthetic baseline-damage evaluation using the fixed C1 fit."""
from __future__ import annotations

import argparse
import csv
import os
import time

import numpy as np

import ncd_calibration_lib as L
import run_experiments as R

HERE = os.path.dirname(os.path.abspath(__file__))


def load_fit():
    rows = []
    with open(os.path.join(HERE, "results", "mixed", "instances.csv")) as fh:
        rows = [r for r in csv.DictReader(fh) if r["panel"] == "calib"]
    x = np.asarray([float(r["dh"]) for r in rows])
    y = np.asarray([float(r["oracle_NCD"]) for r in rows])
    return np.polyfit(x, y, 1)


def run(panel):
    coef = load_fit()
    out = []
    for spec in R.panel_specs(panel):
        g, _pop, _params, rng = R.make_base(spec)
        base_ops = L.build_operators(g)
        base_r = float(np.clip(.242 - .143 * L.structural_features(base_ops)["pesc_M"], .01, .99))
        sums = L.incoming_sums(base_ops)
        base_h = L.h_statistic(base_ops, base_r, sums)[0]
        ref = L.ncdawarerank(base_ops, .85, L.R_CANONICAL, tol=1e-12)[0]
        kref = L.KendallRef(ref)
        for keep in (.9, .8, .6):
            for pages in (0.1, 0.3):
                baseline = L.sparsify_arcs(g, keep, rng)
                bo = L.build_operators(baseline)
                bh = L.h_statistic(bo, base_r, L.incoming_sums(bo))[0]
                current = L.mixed_damage(g, .6, pages, rng)
                co = L.build_operators(current)
                cr = float(np.clip(.242 - .143 * L.structural_features(co)["pesc_M"], .01, .99))
                ch = L.h_statistic(co, cr, L.incoming_sums(co))[0]
                alpha = float(np.clip(coef[1] + coef[0] * (ch - bh), .40, .99))
                aa = np.array(sorted({.81, alpha}))
                curves = L.alpha_sweep(co, [cr], aa, tail=1e-13)[0]
                vals = {float(a): kref(curves[i]) for i, a in enumerate(aa)}
                out.append({"panel": panel, "base": spec["base"], "baseline_keep": keep,
                            "new_pages": pages, "alpha": alpha,
                            "gain_over_constant": float(vals[alpha] - vals[.81]),
                            "baseline_h": bh, "current_h": ch, "base_h": base_h})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--panels", default="calib,test")
    ap.add_argument("--output-dir", default=os.path.join(HERE, "results", "baselines_v2"))
    args = ap.parse_args()
    t0 = time.time()
    os.makedirs(args.output_dir, exist_ok=True)
    all_rows = []
    for panel in args.panels.split(","):
        rows = run(panel)
        all_rows.extend(rows)
        with open(os.path.join(args.output_dir, f"b1_{panel}.csv"), "w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=sorted(rows[0]))
            writer.writeheader(); writer.writerows(rows)
        print(f"{panel}: {len(rows)} rows", flush=True)
    with open(os.path.join(args.output_dir, "summary.json"), "w") as fh:
        import json
        json.dump({"panels": args.panels.split(","), "rows": len(all_rows),
                   "runtime_seconds": time.time() - t0,
                   "mean_gain": float(np.mean([r["gain_over_constant"] for r in all_rows]))},
                  fh, indent=2)


if __name__ == "__main__":
    main()
