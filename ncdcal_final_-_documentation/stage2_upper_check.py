"""
stage2_upper_check.py
=====================
Diagnostic for censoring at the top of the Stage 2 grid under the walk-free truth Q.

Under Q (latent page quality) many oracle alphas sit at the grid bound 0.99.  For
the first 8 calibration graphs this script replays the Stage 1 instances and
evaluates Kendall tau against q at alpha in {0.99, 0.995, 0.999} (r = Stage 1 rule),
to show whether tau keeps increasing beyond the search grid.
Writes results/stage2/upper_check.csv.
"""
import os
import sys

import numpy as np
import pandas as pd

import ncd_calibration_lib as L
import run_experiments as R

HERE = os.path.dirname(os.path.abspath(__file__))
ALPHAS = np.array([0.99, 0.995, 0.999])


def main(n_bases=8):
    rows = []
    for spec in R.panel_specs("calib")[:n_bases]:
        g, pop, params, rng = R.make_base(spec)
        kq = L.KendallRef(pop.q)
        for exp, lvl in R.LEVELS:
            reps = 1 if (exp == "A" and lvl == 1.0) else spec["reps"]
            for rep in range(reps):
                go = (g if lvl == 1.0 else L.sparsify_arcs(g, lvl, rng)) if exp == "A" else L.newly_added_pages(g, lvl, rng)
                oo = L.build_operators(go)
                pbar = L.block_escape(oo)[1]
                r = float(np.clip(R.RULE_NCD[0] + R.RULE_NCD[1] * pbar, 0.01, 0.99))
                S = L.alpha_sweep(oo, [r], ALPHAS, tail=1e-12)
                rows.append(dict(base=spec["base"], exp=exp, level=lvl, rep=rep,
                                 **{f"tau_{a}": kq(S[0, j]) for j, a in enumerate(ALPHAS)}))
        print(spec["base"], flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(HERE, "results", "stage2", "upper_check.csv"), index=False)
    print(df[[c for c in df.columns if c.startswith("tau_")]].mean().round(5).to_string())
    best = df[[c for c in df.columns if c.startswith("tau_")]].to_numpy().argmax(axis=1)
    print("share of instances whose best alpha among", ALPHAS.tolist(), "is:",
          {float(a): float(np.mean(best == j)) for j, a in enumerate(ALPHAS)})


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 8)
