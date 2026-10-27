"""Reduced WP4 baseline stress test using the fixed WP3 C1 fit."""
from __future__ import annotations

import csv
import os

import numpy as np

import ncd_calibration_lib as L
import run_experiments as R

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "results", "baselines")


def main():
    rows = []
    with open(os.path.join(HERE, "results", "mixed", "instances.csv")) as fh:
        for r in csv.DictReader(fh):
            if r["panel"] == "calib":
                rows.append(r)
    x = np.asarray([float(r["dh"]) for r in rows])
    y = np.asarray([float(r["oracle_NCD"]) for r in rows])
    coef = np.polyfit(x, y, 1)
    out = []
    for spec in R.panel_specs("calib")[:3]:
        g, _pop, _params, rng = R.make_base(spec)
        base_ops = L.build_operators(g)
        base_r = float(np.clip(.242 - .143 * L.structural_features(base_ops)["pesc_M"], .01, .99))
        sums = L.incoming_sums(base_ops)
        base_h = L.h_statistic(base_ops, base_r, sums)[0]
        ref = L.ncdawarerank(base_ops, .85, L.R_CANONICAL, tol=1e-12)[0]
        kref = L.KendallRef(ref)
        for keep in (.9, .8, .6):
            baseline = L.sparsify_arcs(g, keep, rng)
            bo = L.build_operators(baseline)
            bh = L.h_statistic(bo, base_r, L.incoming_sums(bo))[0]
            current = L.mixed_damage(g, .6, .3, rng)
            co = L.build_operators(current)
            cr = float(np.clip(.242 - .143 * L.structural_features(co)["pesc_M"], .01, .99))
            ch = L.h_statistic(co, cr, L.incoming_sums(co))[0]
            alpha = float(np.clip(coef[1] + coef[0] * (ch - bh), .40, .99))
            aa = np.array(sorted({.81, alpha}))
            curves = L.alpha_sweep(co, [cr], aa, tail=1e-13)[0]
            vals = {float(a): kref(curves[i]) for i, a in enumerate(aa)}
            gain = float(vals[alpha] - vals[.81])
            out.append({"base": spec["base"], "baseline_keep": keep,
                        "alpha": alpha, "gain_over_constant": gain,
                        "baseline_h": bh, "current_h": ch, "base_h": base_h})
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "b1_reduced.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=sorted(out[0]))
        w.writeheader(); w.writerows(out)
    np.savez_compressed(os.path.join(OUT, "fit.npz"), coef=coef)
    print(f"wrote {len(out)} B1 rows; mean gain={np.mean([r['gain_over_constant'] for r in out]):.6f}")


if __name__ == "__main__":
    main()
