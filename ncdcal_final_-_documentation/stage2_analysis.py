"""
stage2_analysis.py
==================
Stage 2 analysis: oracle damping alpha*, candidate rules alpha = g(h), out-of-sample
evaluation against the static [NG13] baselines, and diagnostics.

    python3 stage2_analysis.py

Inputs: results/stage2/<panel>/<base>.npz written by `run_experiments.py stage2`.

r-settings (Stage 1 locked at Checkpoint 1)
------------------------------------------
can     r = 0.10/0.85        [NG13] Table 2 split (eta, mu) = (0.75, 0.10)
ng13    r = 0.10/0.95        [NG13] Section 4 experimental split (0.85, 0.10)
const   r = 0.18             best constant of Stage 1
rule    r = clip(0.242 - 0.143 Pbar, 0.01, 0.99)    locked NCD-branch rule
popLS   r = clip(0.420 - 0.386 Pbar, 0.01, 0.99)    POP branch, as locked at Checkpoint 1
popTAU  r = clip(0.448 - 0.547 Pbar, 0.01, 0.99)    POP branch, tau-optimised counterpart

alpha policies
--------------
a085      alpha = 0.85 (Stage 1 only)        a095   alpha = 0.95 ([NG13] experiments)
const     best constant alpha on the calibration panel (for that r-setting and truth)
g(h)      fitted rule on h (or h_v), several functional forms
oracle    per-instance alpha* = argmax over alpha in [0.40, 0.99]

Conventions as in Stage 1: rules are fitted on the calibration panel only; paired
tests (Wilcoxon signed-rank) and bootstrap CIs are computed over per-base-graph
means; A1.0 (unperturbed) instances are excluded for the truths computed on the
complete graph itself (NCD, NCD95, TOT) and kept for POP and Q.
"""
import json
import os
import warnings

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.isotonic import IsotonicRegression

import ncd_calibration_lib as L
import stage1_analysis as S1

warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results", "stage2")
R_NAMES = ["can", "ng13", "const", "rule", "popLS", "popTAU"]
TRUTHS = ["NCD", "NCD95", "TOT", "POP", "Q"]
PANELS = ["calib", "test", "scale", "topo", "real"]
GRID = L.ALPHA_GRID
OMASK = L.ALPHA_ORACLE_MASK
ALO, AHI = 0.40, 0.99
J085 = int(np.argmin(np.abs(GRID - 0.85)))
J095 = int(np.argmin(np.abs(GRID - 0.95)))
CI = {k: i for i, k in enumerate(R_NAMES)}
TI = {k: i for i, k in enumerate(TRUTHS)}
OBS_FEATS = ["pesc_M", "mean_k", "pesc_H_block", "inter_arc_frac", "log_mean_outdeg", "dangling_frac",
             "zero_in_frac", "log_n", "log_mean_block", "h_H", "h_M", "hv_H", "hv_M"]


# ---------------------------------------------------------------------------
# loading
# ---------------------------------------------------------------------------
def load(panels=PANELS):
    rows, T, Rr, RV, H, HV, MS = [], [], [], [], [], [], []
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
            rows += [{**row, **prm} for row in meta["rows"]]
            T.append(z["tau"]); Rr.append(z["rho"]); RV.append(z["r"]); H.append(z["h"])
            HV.append(z["h_v"]); MS.append(z["mean_sigma"])
    df = pd.DataFrame(rows)
    df["log_mean_block"] = np.log(df["mean_block_size"])
    RV, H, HV, MS = map(np.concatenate, (RV, H, HV, MS))
    for c, name in enumerate(R_NAMES):
        df[f"r_{name}"], df[f"h_{name}"], df[f"hv_{name}"] = RV[:, c], H[:, c], HV[:, c]
    TAU = np.concatenate(T).astype(float)
    RHO = np.concatenate(Rr).astype(float)
    return df, TAU, RHO, MS


def valid(df, truth):
    m = np.ones(len(df), dtype=bool)
    if truth in ("NCD", "NCD95", "TOT"):
        m &= ~((df["exp"] == "A") & (df["level"] == 1.0)).to_numpy()
    return m


def curves(TAU, r, t):
    return TAU[:, CI[r], TI[t], :]


def oracle(C, mask=OMASK):
    c = np.where(mask[None, :], C, -np.inf)
    j = np.argmax(c, axis=1)
    return GRID[j], C[np.arange(len(j)), j]


def interp(C, a_hat):
    """Linear interpolation of tau(alpha) curves on the (non-uniform) alpha grid."""
    a_hat = np.clip(np.asarray(a_hat, float), GRID[0], GRID[-1])
    j = np.clip(np.searchsorted(GRID, a_hat, side="right") - 1, 0, len(GRID) - 2)
    w = (a_hat - GRID[j]) / (GRID[j + 1] - GRID[j])
    rows = np.arange(len(a_hat))
    return (1 - w) * C[rows, j] + w * C[rows, j + 1]


# ---------------------------------------------------------------------------
# rules alpha = g(x)
# ---------------------------------------------------------------------------
class ARule:
    def __init__(self, name, form, params, feature, fit):
        self.name, self.form, self.params, self.feature, self.fit = name, form, list(map(float, params)), feature, fit
        self.iso = None

    def __call__(self, x):
        x = np.asarray(x, float)
        p = self.params
        if self.form == "const":
            return np.full_like(x, p[0])
        if self.form == "clip_linear":
            return np.clip(p[0] + p[1] * x, ALO, AHI)
        if self.form == "clip_linear_free":
            lo, hi = sorted((np.clip(p[2], ALO, AHI), np.clip(p[3], ALO, AHI)))
            return np.clip(p[0] + p[1] * x, lo, hi)
        if self.form == "logistic":
            return ALO + (AHI - ALO) / (1.0 + np.exp(-(p[0] + p[1] * x)))
        if self.form == "isotonic":
            return np.clip(self.iso.predict(x), ALO, AHI)
        if self.form == "budget":
            c = np.exp(p[0])
            return np.clip(c / (c + np.maximum(x, 0.0)), ALO, AHI)
        raise ValueError(self.form)

    def formula(self):
        p, x = self.params, self.feature
        if self.form == "const":
            return f"alpha = {p[0]:.3f}"
        if self.form == "clip_linear":
            return f"alpha = clip({p[0]:.3f} {p[1]:+.3f} {x}, 0.40, 0.99)"
        if self.form == "clip_linear_free":
            lo, hi = sorted((np.clip(p[2], ALO, AHI), np.clip(p[3], ALO, AHI)))
            return f"alpha = clip({p[0]:.3f} {p[1]:+.3f} {x}, {lo:.3f}, {hi:.3f})"
        if self.form == "logistic":
            return f"alpha = 0.40 + 0.59 / (1 + exp(-({p[0]:.3f} {p[1]:+.3f} {x})))"
        if self.form == "budget":
            return f"alpha = clip(c/(c + {x}), 0.40, 0.99), c = {np.exp(p[0]):.4f}"
        return f"isotonic step function of {x}"

    def to_dict(self):
        d = dict(name=self.name, form=self.form, params=self.params, feature=self.feature, fit=self.fit,
                 formula=self.formula())
        if self.iso is not None:
            d["iso_x"], d["iso_y"] = self.iso.X_thresholds_.tolist(), self.iso.y_thresholds_.tolist()
        return d


FULL = ("const", "cliplin_LS", "cliplin_free_LS", "logistic_LS", "isotonic", "cliplin_TAU",
        "cliplin_free_TAU", "logistic_TAU", "budget_LS", "budget_TAU")
LIGHT = ("const", "cliplin_LS", "cliplin_TAU", "isotonic")


def fit_rules(x, y, C, feature, which=FULL, restarts=4):
    x, y = np.asarray(x, float), np.asarray(y, float)
    out = []
    sse = lambda a: float(np.sum((a - y) ** 2))
    mt = lambda ru: float(np.mean(interp(C, ru(x))))
    if "const" in which:
        taus = C[:, OMASK].mean(axis=0)
        out.append(ARule("const", "const", [GRID[OMASK][np.argmax(taus)]], feature, "tau"))
    sx = np.std(x)
    b1, b0 = np.polyfit(x, y, 1) if sx > 0 else (0.0, float(np.mean(y)))
    p = S1._nm(lambda q: sse(np.clip(q[0] + q[1] * x, ALO, AHI)), [b0, b1], restarts=restarts)
    if "cliplin_LS" in which:
        out.append(ARule("cliplin_LS", "clip_linear", p, feature, "LS"))
    p4 = [p[0], p[1], ALO, AHI]
    if "cliplin_free_LS" in which:
        p4 = S1._nm(lambda q: sse(ARule("", "clip_linear_free", q, feature, "")(x)), p4, restarts=restarts)
        out.append(ARule("cliplin_free_LS", "clip_linear_free", p4, feature, "LS"))
    yc = np.clip(y, ALO + 1e-3, AHI - 1e-3)
    yl = np.log((yc - ALO) / (AHI - yc))
    a1, a0 = np.polyfit(x, yl, 1) if sx > 0 else (0.0, float(np.mean(yl)))
    pl = [a0, a1]
    if "logistic_LS" in which:
        pl = S1._nm(lambda q: sse(ARule("", "logistic", q, feature, "")(x)), pl, restarts=restarts)
        out.append(ARule("logistic_LS", "logistic", pl, feature, "LS"))
    if "isotonic" in which:
        ru = ARule("isotonic", "isotonic", [], feature, "LS")
        inc = bool(stats.spearmanr(x, y)[0] >= 0) if sx > 0 else True
        ru.iso = IsotonicRegression(increasing=inc, out_of_bounds="clip").fit(x, y)
        out.append(ru)
    if "cliplin_TAU" in which:
        pt = S1._nm(lambda q: -mt(ARule("", "clip_linear", q, feature, "")), p, restarts=restarts)
        out.append(ARule("cliplin_TAU", "clip_linear", pt, feature, "tau"))
    if "cliplin_free_TAU" in which:
        pt4 = S1._nm(lambda q: -mt(ARule("", "clip_linear_free", q, feature, "")), list(p4), restarts=restarts)
        out.append(ARule("cliplin_free_TAU", "clip_linear_free", pt4, feature, "tau"))
    if "logistic_TAU" in which:
        plt_ = S1._nm(lambda q: -mt(ARule("", "logistic", q, feature, "")), pl, restarts=restarts)
        out.append(ARule("logistic_TAU", "logistic", plt_, feature, "tau"))
    # displacement-budget rule alpha = c/(c + x), motivated by ||pi(alpha) - v||_1 <= alpha/(1-alpha) h_v
    if "budget_LS" in which or "budget_TAU" in which:
        xm = max(float(np.median(x)), 1e-6)
        c0 = [np.log(xm * np.median(y) / max(1 - np.median(y), 1e-3))]
        pb = S1._nm(lambda q: sse(ARule("", "budget", q, feature, "")(x)), c0, restarts=restarts)
        if "budget_LS" in which:
            out.append(ARule("budget_LS", "budget", pb, feature, "LS"))
        if "budget_TAU" in which:
            pbt = S1._nm(lambda q: -mt(ARule("", "budget", q, feature, "")), pb, restarts=restarts)
            out.append(ARule("budget_TAU", "budget", pbt, feature, "tau"))
    return out


class GBMRule:
    """Diagnostic ceiling: boosted trees on every observable statistic + h, h_v."""

    def __init__(self, feats):
        self.name, self.feature, self.fit, self.form, self.feats = "gbm[observables]", "multi", "LS", "gbm", feats

    def fit_xy(self, df, y):
        from sklearn.ensemble import HistGradientBoostingRegressor
        self.m = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, max_leaf_nodes=15,
                                               min_samples_leaf=20, random_state=0).fit(df[self.feats].to_numpy(float), y)
        return self

    def predict(self, df):
        return np.clip(self.m.predict(df[self.feats].to_numpy(float)), ALO, AHI)

    def formula(self):
        return "boosted trees on " + ", ".join(self.feats)

    def to_dict(self):
        return dict(name=self.name, form="gbm", feature="multi", feats=self.feats, formula=self.formula())


def predict(rule, df):
    return rule.predict(df) if isinstance(rule, GBMRule) else rule(df[rule.feature].to_numpy())


# ---------------------------------------------------------------------------
# evaluation helpers
# ---------------------------------------------------------------------------
def cluster(diff, groups):
    return S1.cluster_stats(np.asarray(diff, float), np.asarray(groups))


def grouped_cv(D, C, y, feature, k=8, seed=0, which=("const", "cliplin_LS", "cliplin_TAU", "logistic_TAU",
                                                       "isotonic", "budget_LS", "budget_TAU"), gbm_feats=None):
    bases = np.array(sorted(D["base"].unique()))
    rng = np.random.default_rng(seed)
    rng.shuffle(bases)
    acc = {}
    for f in np.array_split(bases, k):
        te = D["base"].isin(f).to_numpy()
        tr = ~te
        for ru in fit_rules(D[feature].to_numpy()[tr], y[tr], C[tr], feature, which=which, restarts=2):
            acc.setdefault(ru.name, []).append(interp(C[te], ru(D[feature].to_numpy()[te])))
        if gbm_feats:
            g = GBMRule(gbm_feats).fit_xy(D[tr], y[tr])
            acc.setdefault(g.name, []).append(interp(C[te], g.predict(D[te])))
    return {k_: float(np.mean(np.concatenate(v))) for k_, v in acc.items()}


def analyse_config(df, TAU, RHO, r, t, feature, which, with_gbm=False, cv=False):
    """Fit g on (feature -> alpha*) for one (r-setting, truth) and evaluate on all panels."""
    m = valid(df, t) & np.isfinite(curves(TAU, r, t)[:, -1])
    D = df[m].reset_index(drop=True)
    C, Cr = curves(TAU, r, t)[m], RHO[:, CI[r], TI[t], :][m]
    a_star, t_star = oracle(C)
    D["a_star"], D["t_star"] = a_star, t_star
    cal = (D["panel"] == "calib").to_numpy()
    if cal.sum() < 20:
        return None
    which = tuple(which) if "const" in which else ("const",) + tuple(which)   # const is the paired reference
    rules = fit_rules(D[feature].to_numpy()[cal], a_star[cal], C[cal], feature, which=which)
    if with_gbm:
        rules.append(GBMRule(OBS_FEATS + [f"h_{r}", f"hv_{r}"]).fit_xy(D[cal], a_star[cal]))
    const = [ru for ru in rules if ru.name == "const"][0]
    rows = []
    cvres = grouped_cv(D[cal].reset_index(drop=True), C[cal], a_star[cal], feature,
                       gbm_feats=(OBS_FEATS + [f"h_{r}", f"hv_{r}"]) if with_gbm else None) if cv else {}
    for p in PANELS:
        pm = (D["panel"] == p).to_numpy()
        if pm.sum() == 0:
            continue
        Dp, Cp, Crp = D[pm].reset_index(drop=True), C[pm], Cr[pm]
        t085, t095 = Cp[:, J085], Cp[:, J095]
        t_or = Dp["t_star"].to_numpy()
        t_c = interp(Cp, const(Dp[feature].to_numpy()))
        for ru in rules:
            ah = predict(ru, Dp)
            tt, rr = interp(Cp, ah), interp(Crp, ah)
            head = t_or.mean() - t085.mean()
            st085 = cluster(tt - t085, Dp["base"])
            stc = cluster(tt - t_c, Dp["base"])
            rows.append(dict(r_setting=r, truth=t, feature=feature, rule=ru.name, fit=ru.fit, panel=p,
                             n_inst=int(pm.sum()), n_bases=int(Dp["base"].nunique()),
                             tau=float(tt.mean()), rho=float(rr.mean()), tau_085=float(t085.mean()),
                             tau_095=float(t095.mean()), tau_oracle=float(t_or.mean()),
                             gap_closed_vs_085=float((tt.mean() - t085.mean()) / head) if head > 1e-12 else np.nan,
                             pct_oracle=float(100 * tt.mean() / t_or.mean()),
                             d_vs_085=st085["mean"], p_vs_085=st085["p"], wins_vs_085=st085["wins"],
                             lo_vs_085=st085["lo"], hi_vs_085=st085["hi"],
                             d_vs_const=stc["mean"], p_vs_const=stc["p"], wins_vs_const=stc["wins"],
                             n_clusters=st085["n"],
                             corr_ghat_astar=float(stats.pearsonr(ah, Dp["a_star"])[0]) if np.std(ah) > 0 else np.nan,
                             rmse_alpha=float(np.sqrt(np.mean((ah - Dp["a_star"]) ** 2))),
                             alpha_hat_mean=float(np.mean(ah)), alpha_star_mean=float(Dp["a_star"].mean()),
                             cv_tau=cvres.get(ru.name, np.nan) if p == "calib" else np.nan))
    wb = S1.within_between(D[cal], feature, "a_star")
    info = dict(r_setting=r, truth=t, feature=feature, n_calib=int(cal.sum()),
                corr=dict(pearson=float(stats.pearsonr(D[feature][cal], a_star[cal])[0]),
                          spearman=float(stats.spearmanr(D[feature][cal], a_star[cal])[0])),
                within_between=wb,
                alpha_star_calib=dict(mean=float(a_star[cal].mean()), sd=float(a_star[cal].std()),
                                      at_lower=float(np.mean(a_star[cal] <= ALO + 1e-9)),
                                      at_upper=float(np.mean(a_star[cal] >= AHI - 1e-9))),
                cv=cvres, rules=[ru.to_dict() for ru in rules])
    return rows, info, D, C


# ---------------------------------------------------------------------------
# pipelines: two-stage dynamic rule vs the static [NG13] baselines
# ---------------------------------------------------------------------------
def rule_from_dict(d):
    ru = ARule(d["name"], d["form"], d.get("params", []), d["feature"], d["fit"])
    if d["form"] == "isotonic":
        xs, ys = np.asarray(d["iso_x"]), np.asarray(d["iso_y"])
        ru.iso = IsotonicRegression(increasing=bool(ys[-1] >= ys[0]), out_of_bounds="clip").fit(xs, ys)
    return ru


def pipelines(df, TAU, fits, g_names=("cliplin_TAU", "cliplin_LS", "logistic_TAU", "isotonic", "const")):
    """tau of complete (r, alpha) pipelines on every panel and truth.  All rules are
    the ones fitted on the calibration panel under the PRIMARY truth (NCD), i.e. the
    locked pipeline is evaluated unchanged under every truth."""
    pick = lambda key: {d["name"]: rule_from_dict(d) for d in fits[key] if d.get("form") != "gbm"}
    g_rule, g_can, g_const = pick("rule|NCD|h_rule"), pick("can|NCD|h_can"), pick("const|NCD|h_const")
    out = []
    for t in TRUTHS:
        m = valid(df, t)
        for p in PANELS:
            pm = m & (df["panel"] == p).to_numpy()
            if pm.sum() == 0 or not np.isfinite(curves(TAU, "rule", t)[pm][:, -1]).all():
                continue
            D = df[pm]
            pol = {}
            cr, cc, cn = curves(TAU, "rule", t)[pm], curves(TAU, "can", t)[pm], curves(TAU, "ng13", t)[pm]
            pol["NG13 static (r=0.118, a=0.85)"] = cc[:, J085]
            pol["NG13 static (r=0.105, a=0.95)"] = cn[:, J095]
            pol["Stage 1 only (rule r, a=0.85)"] = cr[:, J085]
            if np.isfinite(curves(TAU, "const", t)[pm][:, -1]).all():
                ck = curves(TAU, "const", t)[pm]
                pol["Stage 1 const r=0.18, a=0.85"] = ck[:, J085]
                pol["const r=0.18 + g(h) [Stage 2 only]"] = interp(ck, g_const["cliplin_TAU"](D["h_const"]))
            pol["canonical r + g(h) [Stage 2 only]"] = interp(cc, g_can["cliplin_TAU"](D["h_can"]))
            for gn in g_names:
                pol[f"two-stage: rule r + {gn}(h)"] = interp(cr, g_rule[gn](D["h_rule"]))
            a_or, t_or = oracle(cr)
            pol["oracle alpha* at rule r"] = t_or
            base0 = pol["NG13 static (r=0.118, a=0.85)"]
            base1 = pol["NG13 static (r=0.105, a=0.95)"]
            s1 = pol["Stage 1 only (rule r, a=0.85)"]
            for name, tt in pol.items():
                st0, st1, st2 = cluster(tt - base0, D["base"]), cluster(tt - base1, D["base"]), cluster(tt - s1, D["base"])
                out.append(dict(truth=t, panel=p, policy=name, n_inst=int(pm.sum()), n_bases=int(D["base"].nunique()),
                                tau=float(np.mean(tt)), d_vs_ng13_085=st0["mean"], p_vs_ng13_085=st0["p"],
                                wins_vs_ng13_085=st0["wins"], d_vs_ng13_095=st1["mean"], p_vs_ng13_095=st1["p"],
                                wins_vs_ng13_095=st1["wins"], d_vs_stage1=st2["mean"], p_vs_stage1=st2["p"],
                                wins_vs_stage1=st2["wins"], n_clusters=st0["n"]))
    return pd.DataFrame(out)


def main():
    import sys
    if "--pipelines-only" in sys.argv:
        df, TAU, RHO, MS = load()
        fits = json.load(open(os.path.join(RES, "fits.json")))
        pipelines(df, TAU, fits).to_csv(os.path.join(RES, "pipelines.csv"), index=False)
        print("pipelines written")
        return
    df, TAU, RHO, MS = load()
    os.makedirs(RES, exist_ok=True)
    sig = np.abs(MS[np.isfinite(MS)] - 1.0)
    checks = dict(n_instances=int(len(df)), max_abs_mean_sigma_minus_1=float(sig.max()))
    configs = []
    # primary and its h_v variant
    configs.append(("rule", "NCD", "h_rule", FULL, True, True))
    configs.append(("rule", "NCD", "hv_rule", FULL, False, True))
    # Stage 2 on top of the static r-settings (isolates Stage 1 from Stage 2)
    for r in ("can", "ng13", "const"):
        configs.append((r, "NCD", f"h_{r}", LIGHT, False, False))
    # sensitivity truths
    for r in ("rule", "can", "ng13", "const"):
        configs.append((r, "TOT", f"h_{r}", FULL if r == "rule" else LIGHT, r == "rule", r == "rule"))
    for r in ("rule", "can", "const", "ng13", "popLS", "popTAU"):
        configs.append((r, "POP", f"h_{r}", FULL if r in ("popLS", "popTAU") else LIGHT, False, r in ("popLS", "popTAU")))
        configs.append((r, "Q", f"h_{r}", FULL if r == "rule" else LIGHT, False, r == "rule"))
    for r in ("ng13", "rule", "can"):
        configs.append((r, "NCD95", f"h_{r}", LIGHT, False, False))
    # competitor features for the primary configuration
    comp = ["pesc_M", "dangling_frac", "log_mean_outdeg", "zero_in_frac", "h_H", "hv_H", "keep_frac"]
    for f in comp:
        configs.append(("rule", "NCD", f, ("cliplin_LS", "cliplin_TAU", "isotonic"), False, False))
    rows, infos, fits = [], [], {}
    for r, t, feat, which, gbm, cv in configs:
        res = analyse_config(df, TAU, RHO, r, t, feat, which, with_gbm=gbm, cv=cv)
        if res is None:
            continue
        rr, info, D, C = res
        rows += rr
        infos.append({k: v for k, v in info.items() if k != "rules"})
        fits[f"{r}|{t}|{feat}"] = info["rules"]
        print(f"{r:6s} {t:5s} {feat:16s} corr={info['corr']['pearson']:+.3f} within={info['within_between']['within']:+.3f} "
              f"between={info['within_between']['between']:+.3f} a*={info['alpha_star_calib']['mean']:.3f}"
              f"+-{info['alpha_star_calib']['sd']:.3f} lower={info['alpha_star_calib']['at_lower']:.2f} "
              f"upper={info['alpha_star_calib']['at_upper']:.2f}", flush=True)
    summ = pd.DataFrame(rows)
    summ.to_csv(os.path.join(RES, "summary.csv"), index=False)
    json.dump(fits, open(os.path.join(RES, "fits.json"), "w"), indent=2)
    json.dump(dict(checks=checks, configs=infos), open(os.path.join(RES, "diagnostics.json"), "w"), indent=2)
    # instance table with oracle alphas for every evaluated pair
    inst = df.copy()
    for t in TRUTHS:
        for r in R_NAMES:
            C = curves(TAU, r, t)
            ok = np.isfinite(C[:, -1])
            if not ok.any():
                continue
            a, tt = np.full(len(df), np.nan), np.full(len(df), np.nan)
            a[ok], tt[ok] = oracle(C[ok])
            ae = np.full(len(df), np.nan)
            ae[ok], _ = oracle(C[ok], np.ones_like(OMASK))
            inst[f"astar_{r}_{t}"], inst[f"tstar_{r}_{t}"], inst[f"astar_ext_{r}_{t}"] = a, tt, ae
            inst[f"t085_{r}_{t}"], inst[f"t095_{r}_{t}"] = C[:, J085], C[:, J095]
    inst.to_csv(os.path.join(RES, "instances.csv"), index=False)
    pipe = pipelines(df, TAU, fits)
    pipe.to_csv(os.path.join(RES, "pipelines.csv"), index=False)
    print(json.dumps(checks))


if __name__ == "__main__":
    main()
