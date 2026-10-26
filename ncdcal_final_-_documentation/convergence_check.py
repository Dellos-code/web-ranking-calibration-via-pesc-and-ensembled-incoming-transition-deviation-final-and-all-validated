"""
convergence_check.py
====================
Iteration counts of the factorised power iteration ([NG13] Algorithm 1) before and
after the Stage 2 noise discount, on the replayed Stage 1/2 instances.

Configurations (stopping rule ||pi^(k+1) - pi^(k)||_1 < 1e-6, start v):
  NCD  branch: r = Stage 1 rule,  alpha = 0.85  vs  alpha = 0.81
  NCD95 branch: r = 0.10/0.95,    alpha = 0.95  vs  alpha = 0.93
The a-priori estimate from the subdominant-eigenvalue bound |lambda_2(P)| <= alpha
([NG13] Theorem 1) is k(alpha) ~ ln(tol)/ln(alpha); the measured counts are lower
because the actual contraction is alpha*|lambda_2(Z)| or better.
Writes results/stage2/convergence.csv.

    python3 convergence_check.py [panels...]       (default: test real)
"""
import os
import sys

import numpy as np
import pandas as pd

import ncd_calibration_lib as L
import run_experiments as R

HERE = os.path.dirname(os.path.abspath(__file__))
TOL = 1e-6
CONFIGS = [("rule", 0.85), ("rule", 0.81), ("ng13", 0.95), ("ng13", 0.93)]


def main(panels):
    rows = []
    for panel in panels:
        for spec in R.panel_specs(panel):
            g, pop, params, rng = R.make_base(spec)
            for exp, lvl in R.LEVELS:
                reps = 1 if (exp == "A" and lvl == 1.0) else spec["reps"]
                for rep in range(reps):
                    go = (g if lvl == 1.0 else L.sparsify_arcs(g, lvl, rng)) if exp == "A" else L.newly_added_pages(g, lvl, rng)
                    if exp == "A" and lvl == 1.0:
                        continue
                    oo = L.build_operators(go)
                    pbar = L.block_escape(oo)[1]
                    r_rule = float(np.clip(R.RULE_NCD[0] + R.RULE_NCD[1] * pbar, 0.01, 0.99))
                    row = dict(panel=panel, base=spec["base"], exp=exp, level=lvl, rep=rep, n=oo.n)
                    for rn, a in CONFIGS:
                        r = r_rule if rn == "rule" else L.R_NG13
                        _, it = L.ncdawarerank(oo, a, r, tol=TOL, maxit=100000)
                        row[f"it_{rn}_{a}"] = it
                    rows.append(row)
            print(panel, spec["base"], flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(HERE, "results", "stage2", "convergence.csv"), index=False)
    for panel, d in df.groupby("panel"):
        a, b = d["it_rule_0.85"].mean(), d["it_rule_0.81"].mean()
        c, e = d["it_ng13_0.95"].mean(), d["it_ng13_0.93"].mean()
        print(f"{panel}: NCD branch {a:.1f} -> {b:.1f} ({100 * (1 - b / a):.1f}%), "
              f"NCD95 branch {c:.1f} -> {e:.1f} ({100 * (1 - e / c):.1f}%)")
    for a0, a1 in ((0.85, 0.81), (0.95, 0.93)):
        k0, k1 = np.log(TOL) / np.log(a0), np.log(TOL) / np.log(a1)
        print(f"bound-based estimate {a0}->{a1}: {k0:.1f} -> {k1:.1f} ({100 * (1 - k1 / k0):.1f}%)")


if __name__ == "__main__":
    main(sys.argv[1:] or ["test", "real"])
