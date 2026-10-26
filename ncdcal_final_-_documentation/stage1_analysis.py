"""
stage1_analysis.py
==================
Stage 1 analysis: oracle ratios r*, candidate rules r = f(Pbar_esc^(M)),
out-of-sample evaluation and diagnostics.

    python3 stage1_analysis.py [--gt NCD]

Outputs (results/stage1/)
-------------------------
instances.csv          one row per perturbed instance: observed-graph features,
                       oracle r* and tau* for every ground truth (Kendall- and
                       Spearman-optimal), tau at the canonical r
fits_<GT>.json         fitted parameters of every candidate rule
summary_<GT>.csv/.md   candidate x panel table: mean tau / rho at the rule's r,
                       % of oracle, gap closed vs canonical, cluster tests
diagnostics_<GT>.json  noise ceiling, within/between-base decomposition,
                       competitor features, Kendall-vs-Spearman consistency

Statistical conventions
-----------------------
* Rules are fitted on the calibration panel only.
* The unit of independent replication is the BASE graph: every paired test
  (Wilcoxon signed-rank) and every bootstrap CI is computed over per-base
  means, never over perturbed instances, which share a base graph.
* A1.0 (the unperturbed graph) is excluded for NCD, PR and SELF, where the
  reference is the observed graph itself; it is kept for POP.
"""
import argparse
import json
import os
import warnings

import numpy as np
import pandas as pd
from scipy import optimize, stats
from sklearn.isotonic import IsotonicRegression

import ncd_calibration_lib as L

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results", "stage1")
GTS = ["NCD", "PR", "SELF", "POP"]
GRID = L.R_GRID
OMASK = L.ORACLE_MASK
PANELS = ["calib", "test", "scale", "topo", "real"]
COMPETITORS = ["pesc_H_block", "inter_arc_frac", "mean_k", "log_mean_outdeg",
               "dangling_frac", "zero_in_frac", "pesc_M_linked"]


# ---------------------------------------------------------------------------
# loading
# ---------------------------------------------------------------------------
def load(panels=PANELS):
    rows, T, Rr, Tc, Rc, IT = [], [], [], [], [], []
    for p in panels:
        d = os.path.join(RES, p)
        if not os.path.isdir(d):
            continue
        for f in sorted(os.listdir(d)):
            if not f.endswith(".npz"):
                continue
            z = np.load(os.path.join(d, f))
            meta = json.loads(str(z["meta"]))
            prm = {f"prm_{k}": v for k, v in meta["params"].items() if isinstance(v, (int, float))}
            for i, row in enumerate(meta["rows"]):
                rows.append({**row, **prm})
            T.append(z["tau"]); Rr.append(z["rho"]); Tc.append(z["tau_can"]); Rc.append(z["rho_can"])
            IT.append(z["iters"])
    df = pd.DataFrame(rows)
    return (df, np.concatenate(T).astype(float), np.concatenate(Rr).astype(float),
            np.concatenate(Tc).astype(float), np.concatenate(Rc).astype(float), np.concatenate(IT))


def oracle(curves):
    """argmax over r in [0.01, 0.99]; returns (r*, metric*)."""
    c = np.where(OMASK[None, :], curves, -np.inf)
    j = np.argmax(c, axis=1)
    return GRID[j], curves[np.arange(len(j)), j]


def add_oracles(df, TAU, RHO, TAUc, RHOc):
    for gi, k in enumerate(GTS):
        ok = ~np.isnan(TAU[:, gi, 1])
        rs, ts = np.full(len(df), np.nan), np.full(len(df), np.nan)
        rr, rh = np.full(len(df), np.nan), np.full(len(df), np.nan)
        if ok.any():
            rs[ok], ts[ok] = oracle(TAU[ok, gi])
            rr[ok], rh[ok] = oracle(RHO[ok, gi])
        df[f"rstar_{k}"], df[f"taustar_{k}"] = rs, ts
        df[f"rstar_rho_{k}"], df[f"rhostar_{k}"] = rr, rh
        df[f"tau_can_{k}"], df[f"rho_can_{k}"] = TAUc[:, gi], RHOc[:, gi]
        df[f"tau_r0_{k}"], df[f"rho_r0_{k}"] = TAU[:, gi, 0], RHO[:, gi, 0]
        # flatness: share of the oracle grid within 0.005 of the optimum
        c = TAU[:, gi][:, OMASK]
        df[f"flat_{k}"] = np.mean(c >= (np.nanmax(c, axis=1, keepdims=True) - 0.005), axis=1)
    return df


def valid_mask(df, gt):
    m = df[f"rstar_{gt}"].notna().to_numpy().copy()
    if gt != "POP":
        m &= ~((df["exp"] == "A") & (df["level"] == 1.0)).to_numpy()
    return m


# ---------------------------------------------------------------------------
# candidate rules
# ---------------------------------------------------------------------------
def _nm(fun, x0, restarts=8, seed=0):
    rng = np.random.default_rng(seed)
    best = None
    for k in range(restarts):
        start = np.asarray(x0, float) if k == 0 else np.asarray(x0, float) + rng.normal(0, 0.5, len(x0))
        res = optimize.minimize(fun, start, method="Nelder-Mead",
                                options=dict(maxiter=4000, xatol=1e-7, fatol=1e-10))
        if best is None or res.fun < best.fun:
            best = res
    return best.x


class Rule:
    """A fitted calibration rule r = f(x)."""

    def __init__(self, name, form, params, feature="pesc_M", fit="LS"):
        self.name, self.form, self.params, self.feature, self.fit = name, form, list(map(float, params)), feature, fit
        self.iso = None

    def __call__(self, x):
        x = np.asarray(x, float)
        p = self.params
        if self.form == "const":
            return np.full_like(x, p[0])
        if self.form == "clip_linear":
            return L.clip_linear(x, p[0], p[1])
        if self.form == "clip_linear_free":
            lo, hi = sorted((np.clip(p[2], 0.01, 0.99), np.clip(p[3], 0.01, 0.99)))
            return np.clip(p[0] + p[1] * x, lo, hi)
        if self.form == "logistic":
            return L.logistic(x, p[0], p[1])
        if self.form == "exp":
            return np.clip(np.exp(p[0] + p[1] * x), 0.01, 0.99)
        if self.form == "isotonic":
            return np.clip(self.iso.predict(x), 0.01, 0.99)
        raise ValueError(self.form)

    def formula(self):
        p = self.params
        x = self.feature if self.feature != "pesc_M" else "Pbar"
        if self.form == "const":
            return f"r = {p[0]:.3f}"
        if self.form == "clip_linear":
            return f"r = clip({p[0]:.3f} {p[1]:+.3f} {x}, 0.01, 0.99)"
        if self.form == "clip_linear_free":
            lo, hi = sorted((np.clip(p[2], 0.01, 0.99), np.clip(p[3], 0.01, 0.99)))
            return f"r = clip({p[0]:.3f} {p[1]:+.3f} {x}, {lo:.3f}, {hi:.3f})"
        if self.form == "logistic":
            return f"r = 0.01 + 0.98 / (1 + exp(-({p[0]:.3f} {p[1]:+.3f} {x})))"
        if self.form == "exp":
            return f"r = clip(exp({p[0]:.3f} {p[1]:+.3f} {x}), 0.01, 0.99)"
        return f"isotonic step function of {x}"

    def to_dict(self):
        d = dict(name=self.name, form=self.form, params=self.params, feature=self.feature,
                 fit=self.fit, formula=self.formula())
        if self.iso is not None:
            d["iso_x"] = self.iso.X_thresholds_.tolist()
            d["iso_y"] = self.iso.y_thresholds_.tolist()
        return d


FULL_SET = ("const", "cliplin_LS", "cliplin_free_LS", "logistic_LS", "exp_LS", "isotonic",
            "cliplin_TAU", "cliplin_free_TAU", "logistic_TAU")
LIGHT_SET = ("const", "cliplin_LS", "isotonic", "cliplin_TAU")


def fit_rules(x, y, curves, feature="pesc_M", tag="", which=FULL_SET, restarts=4):
    """Fit candidate rules on calibration data (x, oracle r* = y, tau curves).

    *_LS   least squares on r*  (the classical calibration fit)
    *_TAU  decision-focused: parameters maximise the mean calibration tau
           directly, using the stored tau(r) curves (robust to flat optima and
           to r* censored at the grid bounds)."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    rules = []

    def mean_tau(rule):
        return float(np.mean(L.interp_curve(curves, rule(x))))

    sse = lambda r: float(np.sum((r - y) ** 2))
    if "const" in which:
        taus = curves[:, OMASK].mean(axis=0)
        rules.append(Rule(f"const{tag}", "const", [GRID[OMASK][np.argmax(taus)]], feature, "tau"))
    b1, b0 = np.polyfit(x, y, 1)
    p = _nm(lambda q: sse(L.clip_linear(x, *q)), [b0, b1], restarts=restarts)
    if "cliplin_LS" in which:
        rules.append(Rule(f"cliplin_LS{tag}", "clip_linear", p, feature, "LS"))
    p4 = [p[0], p[1], 0.01, 0.99]
    if "cliplin_free_LS" in which:
        p4 = _nm(lambda q: sse(Rule("", "clip_linear_free", q)(x)), p4, restarts=restarts)
        rules.append(Rule(f"cliplin_free_LS{tag}", "clip_linear_free", p4, feature, "LS"))
    yl = np.log(np.clip(y, 0.01, 0.99) - 0.01 + 1e-3) - np.log(0.99 - np.clip(y, 0.01, 0.99) + 1e-3)
    a1, a0 = np.polyfit(x, yl, 1)
    pl = [a0, a1]
    if "logistic_LS" in which:
        pl = _nm(lambda q: sse(L.logistic(x, *q)), pl, restarts=restarts)
        rules.append(Rule(f"logistic_LS{tag}", "logistic", pl, feature, "LS"))
    if "exp_LS" in which:
        e1, e0 = np.polyfit(x, np.log(np.clip(y, 0.01, 0.99)), 1)
        pe = _nm(lambda q: sse(np.clip(np.exp(q[0] + q[1] * x), 0.01, 0.99)), [e0, e1], restarts=restarts)
        rules.append(Rule(f"exp_LS{tag}", "exp", pe, feature, "LS"))
    if "isotonic" in which:
        iso = Rule(f"isotonic{tag}", "isotonic", [], feature, "LS")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            iso.iso = IsotonicRegression(increasing=bool(stats.spearmanr(x, y)[0] >= 0),
                                         out_of_bounds="clip").fit(x, y)
        rules.append(iso)
    if "cliplin_TAU" in which:
        pt = _nm(lambda q: -mean_tau(Rule("", "clip_linear", q)), p, restarts=restarts)
        rules.append(Rule(f"cliplin_TAU{tag}", "clip_linear", pt, feature, "tau"))
    if "cliplin_free_TAU" in which:
        pt4 = _nm(lambda q: -mean_tau(Rule("", "clip_linear_free", q)), list(p4), restarts=restarts)
        rules.append(Rule(f"cliplin_free_TAU{tag}", "clip_linear_free", pt4, feature, "tau"))
    if "logistic_TAU" in which:
        plt_ = _nm(lambda q: -mean_tau(Rule("", "logistic", q)), pl, restarts=restarts)
        rules.append(Rule(f"logistic_TAU{tag}", "logistic", plt_, feature, "tau"))
    return rules


# ---------------------------------------------------------------------------
# evaluation
# ---------------------------------------------------------------------------
def cluster_stats(diff, groups, n_boot=20000, seed=0):
    """Per-base means of a paired difference; Wilcoxon + bootstrap CI over bases."""
    s = pd.Series(diff).groupby(np.asarray(groups)).mean()
    v = s.values
    if len(v) < 2 or np.allclose(v, 0):
        return dict(mean=float(np.mean(v)) if len(v) else np.nan, lo=np.nan, hi=np.nan, p=np.nan,
                    wins=int((v > 0).sum()), n=len(v))
    rng = np.random.default_rng(seed)
    bs = rng.choice(v, size=(n_boot, len(v)), replace=True).mean(axis=1)
    p = stats.wilcoxon(v).pvalue if len(v) >= 5 else np.nan
    return dict(mean=float(v.mean()), lo=float(np.percentile(bs, 2.5)), hi=float(np.percentile(bs, 97.5)),
                p=float(p), wins=int((v > 0).sum()), n=len(v))


OBSERVABLE = ["pesc_M", "pesc_M_linked", "mean_k", "pesc_H_block", "inter_arc_frac", "log_mean_outdeg",
              "dangling_frac", "zero_in_frac", "log_n", "log_mean_block"]


class MultiRule:
    """Diagnostic rules on several observable features (not part of the method):
    'lin2'  r = clip(b0 + b1 Pbar + b2 inter_arc_frac)    interpretable extension
    'gbm'   gradient-boosted trees on all observable features (ceiling of what
            the observed graph's global statistics can predict)."""

    def __init__(self, name, kind, feats):
        self.name, self.kind, self.feats, self.feature, self.fit = name, kind, feats, "multi", "LS"

    def _X(self, df):
        X = df.copy()
        X["log_mean_block"] = np.log(X["mean_block_size"])
        return X[self.feats].to_numpy(float)

    def fit_xy(self, df, y):
        X = self._X(df)
        if self.kind == "lin2":
            Xa = np.column_stack([np.ones(len(X)), X])
            b = np.linalg.lstsq(Xa, y, rcond=None)[0]
            self.b = _nm(lambda q: float(np.sum((np.clip(Xa @ q, 0.01, 0.99) - y) ** 2)), b)
        else:
            from sklearn.ensemble import HistGradientBoostingRegressor
            self.m = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, max_leaf_nodes=15,
                                                   min_samples_leaf=20, random_state=0).fit(X, y)
        return self

    def predict(self, df):
        X = self._X(df)
        if self.kind == "lin2":
            return np.clip(np.column_stack([np.ones(len(X)), X]) @ self.b, 0.01, 0.99)
        return np.clip(self.m.predict(X), 0.01, 0.99)

    def formula(self):
        if self.kind == "lin2":
            terms = " ".join(f"{c:+.3f} {f}" for c, f in zip(self.b[1:], self.feats))
            return f"r = clip({self.b[0]:.3f} {terms}, 0.01, 0.99)"
        return "gradient-boosted trees on " + ", ".join(self.feats)

    def to_dict(self):
        d = dict(name=self.name, form=self.kind, feature="multi", feats=self.feats, formula=self.formula())
        if self.kind == "lin2":
            d["params"] = list(map(float, self.b))
        return d


def predict(rule, df):
    return rule.predict(df) if isinstance(rule, MultiRule) else rule(df[rule.feature].values)


def evaluate(rule, df, TAUg, RHOg, gt, t_const=None):
    """Mean tau / rho at the rule's r on one panel, oracle-relative scores, and
    cluster-level (per base graph) paired tests against the canonical r and,
    if t_const is given, against the best calibration constant."""
    r_hat = predict(rule, df)
    t = L.interp_curve(TAUg, r_hat)
    rh = L.interp_curve(RHOg, r_hat)
    out = dict(tau=float(t.mean()), rho=float(rh.mean()))
    t_or = df[f"taustar_{gt}"].values
    t_can = df[f"tau_can_{gt}"].values
    out["tau_oracle"] = float(t_or.mean())
    out["tau_canonical"] = float(t_can.mean())
    out["pct_oracle"] = float(100 * t.mean() / t_or.mean())
    den = t_or.mean() - t_can.mean()
    out["gap_closed_vs_can"] = float((t.mean() - t_can.mean()) / den) if den > 1e-12 else np.nan
    st = cluster_stats(t - t_can, df["base"].values)
    out.update({f"vs_can_{k}": v for k, v in st.items()})
    if t_const is not None:
        st = cluster_stats(t - t_const, df["base"].values)
        out.update({f"vs_const_{k}": v for k, v in st.items()})
    out["rho_rank_fx_rstar"] = float(stats.spearmanr(r_hat, df[f"rstar_{gt}"].values)[0]) \
        if np.std(r_hat) > 0 else np.nan
    out["r_hat_mean"], out["r_hat_sd"] = float(np.mean(r_hat)), float(np.std(r_hat))
    y = df[f"rstar_{gt}"].values
    out["rmse_r"] = float(np.sqrt(np.mean((r_hat - y) ** 2)))
    out["pearson_fx_rstar"] = float(stats.pearsonr(r_hat, y)[0]) if np.std(r_hat) > 0 else np.nan
    return out, t


def grouped_cv(df, TAUg, gt, feature="pesc_M", k=8, seed=0, multi=True):
    """K-fold CV over calibration BASES; returns mean held-out tau per rule name
    (and the held-out oracle / canonical means for reference)."""
    bases = np.array(sorted(df["base"].unique()))
    rng = np.random.default_rng(seed)
    rng.shuffle(bases)
    folds = np.array_split(bases, k)
    acc = {}
    for f in folds:
        te = df["base"].isin(f).values
        tr = ~te
        y = df[f"rstar_{gt}"].values[tr]
        rules = fit_rules(df[feature].values[tr], y, TAUg[tr], feature, restarts=2)
        for ru in rules:
            t = L.interp_curve(TAUg[te], ru(df[feature].values[te]))
            acc.setdefault(ru.name, []).append(t)
        if multi:
            for mr in multi_rules():
                mr.fit_xy(df[tr], y)
                acc.setdefault(mr.name, []).append(L.interp_curve(TAUg[te], mr.predict(df[te])))
        acc.setdefault("_oracle", []).append(df[f"taustar_{gt}"].values[te])
        acc.setdefault("_canonical", []).append(df[f"tau_can_{gt}"].values[te])
    return {k_: float(np.mean(np.concatenate(v))) for k_, v in acc.items()}


def multi_rules():
    return [MultiRule("lin2[pesc_M+inter_arc_frac]", "lin2", ["pesc_M", "inter_arc_frac"]),
            MultiRule("gbm[all observable]", "gbm", OBSERVABLE)]


def within_between(df, xcol, ycol):
    g = df.groupby("base")
    xb, yb = g[xcol].transform("mean"), g[ycol].transform("mean")
    w = stats.pearsonr(df[xcol] - xb, df[ycol] - yb)[0]
    m = g[[xcol, ycol]].mean()
    b = stats.pearsonr(m[xcol], m[ycol])[0] if len(m) > 2 else np.nan
    return dict(pooled=float(stats.pearsonr(df[xcol], df[ycol])[0]), within=float(w), between=float(b),
                n_bases=int(len(m)))


def noise_ceiling(df, gt):
    d = df[df["panel"] == "calib"]
    d = d[~((d["exp"] == "A") & (d["level"] == 1.0))] if gt != "POP" else d[~((d["exp"] == "A") & (d["level"] == 1.0))]
    piv = d.pivot_table(index=["base", "exp", "level"], columns="rep", values=f"rstar_{gt}")
    piv = piv.dropna()
    if piv.shape[1] < 2 or len(piv) < 3:
        return {}
    a, b = piv[0].values, piv[1].values
    xr = d.pivot_table(index=["base", "exp", "level"], columns="rep", values="pesc_M").dropna()
    return dict(test_retest_pearson=float(stats.pearsonr(a, b)[0]),
                test_retest_mad=float(np.mean(np.abs(a - b))),
                pesc_test_retest=float(stats.pearsonr(xr[0], xr[1])[0]),
                n_pairs=int(len(piv)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt", default="all")
    a = ap.parse_args()
    df, TAU, RHO, TAUc, RHOc, IT = load()
    df = add_oracles(df, TAU, RHO, TAUc, RHOc)
    df.to_csv(os.path.join(RES, "instances.csv"), index=False)
    gts = GTS if a.gt == "all" else [a.gt]
    for gt in gts:
        gi = GTS.index(gt)
        vm = valid_mask(df, gt)
        D, TG, RG = df[vm].reset_index(drop=True), TAU[vm, gi], RHO[vm, gi]
        cal = (D["panel"] == "calib").values
        if cal.sum() < 20:
            continue
        x, y = D["pesc_M"].values[cal], D[f"rstar_{gt}"].values[cal]
        rules = fit_rules(x, y, TG[cal])
        # competitor single features (clip-linear, both fits) and a non-observable diagnostic
        comp_rules = []
        for fcol in COMPETITORS + ["keep_frac"]:
            comp_rules += fit_rules(D[fcol].values[cal], y, TG[cal], fcol, tag=f"[{fcol}]",
                                    which=("cliplin_LS", "isotonic", "cliplin_TAU"), restarts=2)
        cv = grouped_cv(D[cal].reset_index(drop=True), TG[cal], gt)
        mrules = [mr.fit_xy(D[cal], y) for mr in multi_rules()]
        rows = []
        const_rule = [ru for ru in rules if ru.name == "const"][0]
        t_const = {p: L.interp_curve(TG[(D["panel"] == p).values],
                                     const_rule(D[(D["panel"] == p).values]["pesc_M"].values))
                   for p in PANELS if (D["panel"] == p).any()}
        for ru in rules + comp_rules + mrules:
            for p in PANELS:
                m = (D["panel"] == p).values
                if m.sum() == 0:
                    continue
                ev, _ = evaluate(ru, D[m].reset_index(drop=True), TG[m], RG[m], gt, t_const[p])
                rows.append(dict(gt=gt, rule=ru.name, feature=ru.feature, fit=ru.fit, panel=p,
                                 n_inst=int(m.sum()), n_bases=int(D[m]["base"].nunique()),
                                 cv_tau=cv.get(ru.name, np.nan) if p == "calib" else np.nan, **ev))
        # hindsight-best constant on each panel (an upper bound for any constant)
        for p in PANELS:
            m = (D["panel"] == p).values
            if m.sum() == 0:
                continue
            taus = TG[m][:, OMASK].mean(axis=0)
            j = np.argmax(taus)
            rows.append(dict(gt=gt, rule="const_hindsight", feature="-", fit="test", panel=p,
                             n_inst=int(m.sum()), n_bases=int(D[m]["base"].nunique()),
                             tau=float(taus[j]), rho=float(RG[m][:, OMASK].mean(axis=0)[j]),
                             tau_oracle=float(D[m][f"taustar_{gt}"].mean()),
                             tau_canonical=float(D[m][f"tau_can_{gt}"].mean()),
                             pct_oracle=float(100 * taus[j] / D[m][f"taustar_{gt}"].mean()),
                             r_value=float(GRID[OMASK][j])))
            rows.append(dict(gt=gt, rule="PageRank_r0", feature="-", fit="-", panel=p,
                             n_inst=int(m.sum()), n_bases=int(D[m]["base"].nunique()),
                             tau=float(TG[m][:, 0].mean()), rho=float(RG[m][:, 0].mean()),
                             tau_oracle=float(D[m][f"taustar_{gt}"].mean()),
                             tau_canonical=float(D[m][f"tau_can_{gt}"].mean()),
                             pct_oracle=float(100 * TG[m][:, 0].mean() / D[m][f"taustar_{gt}"].mean())))
        summ = pd.DataFrame(rows)
        summ.to_csv(os.path.join(RES, f"summary_{gt}.csv"), index=False)
        # diagnostics
        Dc = D[cal]
        diag = dict(gt=gt,
                    n_calib_instances=int(cal.sum()), n_calib_bases=int(Dc["base"].nunique()),
                    corr_pesc_rstar=dict(pearson=float(stats.pearsonr(x, y)[0]),
                                         spearman=float(stats.spearmanr(x, y)[0]),
                                         kendall=float(stats.kendalltau(x, y)[0])),
                    within_between=within_between(Dc, "pesc_M", f"rstar_{gt}"),
                    within_between_by_exp={e: within_between(Dc[Dc["exp"] == e], "pesc_M", f"rstar_{gt}")
                                           for e in ("A", "B")},
                    noise_ceiling=noise_ceiling(df[valid_mask(df, gt)], gt),
                    rstar_kendall_vs_spearman=dict(
                        pearson=float(stats.pearsonr(D[f"rstar_{gt}"], D[f"rstar_rho_{gt}"])[0]),
                        mean_abs_diff=float(np.mean(np.abs(D[f"rstar_{gt}"] - D[f"rstar_rho_{gt}"])))),
                    rstar_at_bounds=dict(lower=float(np.mean(y <= 0.01)), upper=float(np.mean(y >= 0.99))),
                    flatness_mean=float(Dc[f"flat_{gt}"].mean()),
                    competitor_corr={f: float(stats.pearsonr(Dc[f], y)[0]) for f in COMPETITORS + ["keep_frac"]},
                    cv_tau=cv)
        json.dump(diag, open(os.path.join(RES, f"diagnostics_{gt}.json"), "w"), indent=2)
        json.dump([ru.to_dict() for ru in rules + comp_rules + mrules],
                  open(os.path.join(RES, f"fits_{gt}.json"), "w"), indent=2)
        print(f"\n==== GT = {gt}: calib n={cal.sum()} corr(Pbar, r*) = {diag['corr_pesc_rstar']}")
        print("within/between:", diag["within_between"])
        print("noise ceiling:", diag["noise_ceiling"])
        show = summ[summ["feature"].isin(["pesc_M", "-", "multi"])]
        print(show.pivot_table(index="rule", columns="panel", values="tau").round(4).to_string())
        print("CV (calib, grouped by base):", {k: round(v, 4) for k, v in cv.items()})


if __name__ == "__main__":
    main()
