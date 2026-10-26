"""
stage1_decomposition.py
=======================
What determines the oracle ratio r*?  (calibration panel, static generator)

For each ground truth we report the share of the variance of r* that is
explained (grouped 8-fold CV by base graph where a model is fitted) by

  base           base-graph identity (in-sample ANOVA share)
  level          perturbation experiment and level (in-sample ANOVA share)
  base+level     additive two-way model (in-sample)
  hidden         gradient-boosted trees on the generator's hidden parameters
                 plus the (unobservable) perturbation level     [grouped CV]
  observable     gradient-boosted trees, and separately a ridge regression,
                 on every observable global statistic of the observed graph
                                                                [grouped CV]
  pesc_M         clip-linear least squares on Pbar_esc^(M) alone [grouped CV]

and the correlation of the base-mean r* with each hidden generator parameter.
Writes results/stage1/decomposition.json.
"""
import json
import os
import warnings

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import stage1_analysis as A

warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
PRM = ["prm_mean_block", "prm_block_sigma", "prm_domain_size", "prm_locality", "prm_fit_sigma", "prm_icc",
       "prm_block_beta", "prm_block_xi", "prm_mean_out", "prm_out_sigma", "prm_dangling", "prm_phi"]


def r2(y, p):
    return float(1 - np.mean((y - p) ** 2) / np.var(y))


def main():
    df, TAU, RHO, TAUc, RHOc, _ = A.load(["calib"])
    df = A.add_oracles(df, TAU, RHO, TAUc, RHOc)
    df["severity"] = np.where(df["exp"] == "A", 1 - df["level"], df["level"])
    df["log_mean_block"] = np.log(df["mean_block_size"])
    out = {}
    for gt in ["NCD", "PR", "SELF", "POP"]:
        d = df[A.valid_mask(df, gt)].reset_index(drop=True)
        y = d[f"rstar_{gt}"].to_numpy()
        g = d["base"].to_numpy()
        bm = d.groupby("base")[f"rstar_{gt}"].transform("mean").to_numpy()
        lm = d.groupby(["exp", "level"])[f"rstar_{gt}"].transform("mean").to_numpy()
        gbm = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, random_state=0)
        cv = GroupKFold(8)
        Xh = d[PRM + ["severity"]].assign(expB=(d["exp"] == "B").astype(float)).to_numpy(float)
        Xo = d[A.OBSERVABLE].to_numpy(float)
        ph = cross_val_predict(gbm, Xh, y, groups=g, cv=cv)
        po = cross_val_predict(gbm, Xo, y, groups=g, cv=cv)
        lin = make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-3, 3, 13)))
        pl = np.clip(cross_val_predict(lin, Xo, y, groups=g, cv=cv), 0.01, 0.99)
        # single-feature clip-linear LS on Pbar, grouped CV
        pp = np.empty_like(y)
        for tr, te in cv.split(Xo, y, g):
            ru = A.fit_rules(d["pesc_M"].to_numpy()[tr], y[tr], np.zeros((len(tr), len(A.GRID))),
                             which=("cliplin_LS",), restarts=2)[0]
            pp[te] = ru(d["pesc_M"].to_numpy()[te])
        base_tab = d.groupby("base").agg({f"rstar_{gt}": "mean", **{p: "first" for p in PRM}})
        out[gt] = dict(
            var_rstar=float(np.var(y)), n=int(len(y)), n_bases=int(len(base_tab)),
            r2_base=r2(y, bm), r2_level=r2(y, lm), r2_base_level=r2(y, bm + lm - y.mean()),
            r2_hidden_cv=r2(y, ph), r2_observable_cv=r2(y, po), r2_observable_ridge_cv=r2(y, pl), r2_pesc_cv=r2(y, pp),
            corr_basemean_with_param={p.replace("prm_", ""): float(np.corrcoef(base_tab[p], base_tab[f"rstar_{gt}"])[0, 1])
                                      for p in PRM})
        print(gt, {k: (round(v, 3) if isinstance(v, float) else v) for k, v in out[gt].items()
                   if not isinstance(v, dict)})
    json.dump(out, open(os.path.join(HERE, "results", "stage1", "decomposition.json"), "w"), indent=2)


if __name__ == "__main__":
    main()
