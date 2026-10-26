"""
stage2_decomposition.py
=======================
What determines the oracle damping alpha*?  (Stage 2 calibration panel)

Same decomposition as stage1_decomposition.py, for alpha* of the pairs
(r-setting, truth) used in Stage 2:

  base         base-graph identity (in-sample ANOVA share)
  level        perturbation experiment and level (in-sample ANOVA share)
  base+level   additive two-way model (in-sample)
  hidden       boosted trees on hidden generator parameters + perturbation level [grouped CV]
  observable   ridge regression on every observable statistic incl. h, h_v       [grouped CV]
  h alone      clip-linear least squares on h                                     [grouped CV]

Also reports corr(base-mean alpha*, hidden parameter) and, for every truth, the
anchor: alpha* on the unperturbed graph (A1.0) where that is a recovery problem.
Writes results/stage2/decomposition.json.
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

import stage2_analysis as A

warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
PRM = ["prm_mean_block", "prm_block_sigma", "prm_domain_size", "prm_locality", "prm_fit_sigma", "prm_icc",
       "prm_block_beta", "prm_block_xi", "prm_mean_out", "prm_out_sigma", "prm_dangling", "prm_phi"]
PAIRS = [("rule", "NCD"), ("can", "NCD"), ("ng13", "NCD95"), ("rule", "TOT"), ("popLS", "POP"),
         ("popTAU", "POP"), ("rule", "Q")]


def r2(y, p):
    return float(1 - np.mean((y - p) ** 2) / np.var(y)) if np.var(y) > 0 else float("nan")


def main():
    df, TAU, RHO, _ = A.load(["calib"])
    df["severity"] = np.where(df["exp"] == "A", 1 - df["level"], df["level"])
    out = {}
    for r, t in PAIRS:
        C = A.curves(TAU, r, t)
        a_all, _ = A.oracle(C)
        anchor = df["exp"].eq("A") & df["level"].eq(1.0)
        anc = dict(mean=float(a_all[anchor.to_numpy()].mean()), sd=float(a_all[anchor.to_numpy()].std()))
        m = A.valid(df, t)
        d = df[m].reset_index(drop=True)
        y = a_all[m]
        g = d["base"].to_numpy()
        bm = d.groupby("base")["base"].transform("size") * 0 + d.assign(y=y).groupby("base")["y"].transform("mean")
        lm = d.assign(y=y).groupby(["exp", "level"])["y"].transform("mean").to_numpy()
        cv = GroupKFold(8)
        gbm = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, random_state=0)
        Xh = d[PRM + ["severity"]].assign(expB=(d["exp"] == "B").astype(float)).to_numpy(float)
        Xo = d[A.OBS_FEATS + [f"h_{r}", f"hv_{r}"]].to_numpy(float)
        ph = cross_val_predict(gbm, Xh, y, groups=g, cv=cv)
        lin = make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-3, 3, 13)))
        pl = np.clip(cross_val_predict(lin, Xo, y, groups=g, cv=cv), A.ALO, A.AHI)
        ph_h = np.empty_like(y)
        xh = d[f"h_{r}"].to_numpy()
        for tr, te in cv.split(Xo, y, g):
            ru = A.fit_rules(xh[tr], y[tr], np.zeros((len(tr), len(A.GRID))), f"h_{r}", which=("cliplin_LS",),
                             restarts=2)[0]
            ph_h[te] = ru(xh[te])
        base_tab = d.assign(y=y).groupby("base").agg({"y": "mean", **{p: "first" for p in PRM}})
        out[f"{r}|{t}"] = dict(
            n=int(len(y)), n_bases=int(len(base_tab)), mean=float(y.mean()), sd=float(y.std()),
            anchor_A10=anc, r2_base=r2(y, bm.to_numpy()), r2_level=r2(y, lm), r2_base_level=r2(y, bm.to_numpy() + lm - y.mean()),
            r2_hidden_cv=r2(y, ph), r2_observable_ridge_cv=r2(y, pl), r2_h_cv=r2(y, ph_h),
            corr_basemean_with_param={p.replace("prm_", ""): float(np.corrcoef(base_tab[p], base_tab["y"])[0, 1])
                                      for p in PRM})
        print(f"{r}|{t}", {k: (round(v, 3) if isinstance(v, float) else v) for k, v in out[f"{r}|{t}"].items()
                           if not isinstance(v, dict)}, "anchor", {k: round(v, 3) for k, v in anc.items()})
    json.dump(out, open(os.path.join(HERE, "results", "stage2", "decomposition.json"), "w"), indent=2)


if __name__ == "__main__":
    main()
