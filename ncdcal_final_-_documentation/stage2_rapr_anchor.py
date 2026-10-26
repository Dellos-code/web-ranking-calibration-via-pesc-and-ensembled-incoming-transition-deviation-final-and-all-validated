"""
stage2_rapr_anchor.py
=====================
Dose-response test of the anchoring explanation.  If the oracle damping alpha* is
set by the reference ranking's own damping, then Random-alpha NCDawareRank
references (Constantine & Gleich) with different Beta priors on alpha should move
alpha* with the prior mean.

Priors: Beta(1,1) (TotalRank, mean 0.5) and Beta(a,b) with a+b = 10 and mean
0.3, 0.5, 0.7 (the range of empirically measured means in Gleich et al., WWW 2010)
plus 0.85.  For the first n calibration graphs the Stage 1 instances are replayed,
ranked with NCDawareRank(alpha, r = Stage 1 rule) on the full alpha grid
(0.05..0.99), and alpha* is the argmax of Kendall tau against each reference.
Writes results/stage2/rapr_anchor.csv.
"""
import os
import sys

import numpy as np
import pandas as pd

import ncd_calibration_lib as L
import run_experiments as R

HERE = os.path.dirname(os.path.abspath(__file__))
PRIORS = {"Beta(1,1)": (1.0, 1.0), "Beta(3,7)": (3.0, 7.0), "Beta(5,5)": (5.0, 5.0),
          "Beta(7,3)": (7.0, 3.0), "Beta(8.5,1.5)": (8.5, 1.5)}


def main(n_bases=16):
    rows = []
    for spec in R.panel_specs("calib")[:n_bases]:
        g, pop, params, rng = R.make_base(spec)
        ops = L.build_operators(g)
        refs = {k: L.KendallRef(L.rapr_reference(ops, L.R_CANONICAL, a, b, K=20000)[0]) for k, (a, b) in PRIORS.items()}
        for exp, lvl in R.LEVELS:
            reps = 1 if (exp == "A" and lvl == 1.0) else spec["reps"]
            for rep in range(reps):
                go = (g if lvl == 1.0 else L.sparsify_arcs(g, lvl, rng)) if exp == "A" else L.newly_added_pages(g, lvl, rng)
                if exp == "A" and lvl == 1.0:
                    continue
                oo = L.build_operators(go)
                pbar = L.block_escape(oo)[1]
                r = float(np.clip(R.RULE_NCD[0] + R.RULE_NCD[1] * pbar, 0.01, 0.99))
                S = L.alpha_sweep(oo, [r])[0]
                for k, kr in refs.items():
                    tau = np.array([kr(S[j]) for j in range(len(L.ALPHA_GRID))])
                    a, b = PRIORS[k]
                    rows.append(dict(base=spec["base"], exp=exp, level=lvl, rep=rep, prior=k, prior_mean=a / (a + b),
                                     alpha_star=float(L.ALPHA_GRID[int(np.argmax(tau))]), tau_star=float(tau.max())))
        print(spec["base"], flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(HERE, "results", "stage2", "rapr_anchor.csv"), index=False)
    print(df.groupby("prior").agg(prior_mean=("prior_mean", "first"), alpha_star_mean=("alpha_star", "mean"),
                                  alpha_star_sd=("alpha_star", "std")).round(3).to_string())


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 16)
