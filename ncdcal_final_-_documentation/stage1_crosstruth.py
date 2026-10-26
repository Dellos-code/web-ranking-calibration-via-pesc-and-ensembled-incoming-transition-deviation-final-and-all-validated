"""
stage1_crosstruth.py
====================
Cross-truth robustness of the Stage 1 rules: a rule fitted under one ground
truth is evaluated under every other truth on every panel, plus a per-graph
breakdown of the real panel.  Answers "what does adopting the rule fitted on
truth X cost if the relevant truth is Y?".

Reads fits_<GT>.json; writes results/stage1/crosstruth.csv and
results/stage1/real_by_graph.csv.
"""
import json
import os
import warnings

import numpy as np
import pandas as pd

import ncd_calibration_lib as L
import stage1_analysis as A

warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results", "stage1")
RULES = ["const", "cliplin_LS", "cliplin_TAU", "logistic_TAU", "isotonic"]


def load_rules(gt):
    out = {}
    for d in json.load(open(os.path.join(RES, f"fits_{gt}.json"))):
        if d.get("feature") != "pesc_M" or d["name"] not in RULES:
            continue
        r = A.Rule(d["name"], d["form"], d.get("params", []), d["feature"], d["fit"])
        if d["form"] == "isotonic":
            from sklearn.isotonic import IsotonicRegression
            xs, ys = np.asarray(d["iso_x"]), np.asarray(d["iso_y"])
            r.iso = IsotonicRegression(increasing=bool(ys[-1] >= ys[0]), out_of_bounds="clip").fit(xs, ys)
        out[d["name"]] = r
    return out


def main():
    df, TAU, RHO, TAUc, RHOc, _ = A.load()
    df = A.add_oracles(df, TAU, RHO, TAUc, RHOc)
    fitted = {gt: load_rules(gt) for gt in ("NCD", "PR", "POP")}
    rows, real_rows = [], []
    for truth in ("NCD", "PR", "POP"):
        gi = A.GTS.index(truth)
        vm = A.valid_mask(df, truth)
        for p in A.PANELS:
            m = vm & (df["panel"] == p).to_numpy()
            if m.sum() == 0:
                continue
            D, TG = df[m].reset_index(drop=True), TAU[m, gi]
            orc, can = D[f"taustar_{truth}"].mean(), D[f"tau_can_{truth}"].mean()
            for fit_gt, rules in fitted.items():
                for name, ru in rules.items():
                    t = L.interp_curve(TG, ru(D["pesc_M"].values))
                    rows.append(dict(truth=truth, panel=p, fitted_on=fit_gt, rule=name, tau=t.mean(),
                                     tau_oracle=orc, tau_canonical=can,
                                     gap_closed=(t.mean() - can) / (orc - can) if orc - can > 1e-12 else np.nan,
                                     r_mean=float(np.mean(ru(D["pesc_M"].values)))))
                    if p == "real":
                        for b, g in D.groupby("base"):
                            idx = g.index.to_numpy()
                            tt = t[idx]
                            o, c = g[f"taustar_{truth}"].mean(), g[f"tau_can_{truth}"].mean()
                            real_rows.append(dict(truth=truth, graph=b, fitted_on=fit_gt, rule=name,
                                                  pesc_M_full=float(g[(g["exp"] == "A") & (g["level"] == 0.9)]["pesc_M"].mean()),
                                                  r_rule=float(np.mean(ru(g["pesc_M"].values))),
                                                  tau=tt.mean(), tau_oracle=o, tau_canonical=c,
                                                  rstar_mean=float(g[f"rstar_{truth}"].mean()),
                                                  gap_closed=(tt.mean() - c) / (o - c) if o - c > 1e-12 else np.nan))
    pd.DataFrame(rows).to_csv(os.path.join(RES, "crosstruth.csv"), index=False)
    pd.DataFrame(real_rows).to_csv(os.path.join(RES, "real_by_graph.csv"), index=False)
    c = pd.DataFrame(rows)
    print(c[c["rule"].isin(["const", "cliplin_TAU"])].pivot_table(
        index=["truth", "panel"], columns=["fitted_on", "rule"], values="gap_closed").round(3).to_string())


if __name__ == "__main__":
    main()
