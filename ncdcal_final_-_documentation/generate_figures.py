"""
generate_figures.py
===================
Publication figures for Stage 1 (Stage 2 figures are added after Checkpoint 2).

    python3 generate_figures.py stage1

Writes to figures/ (PDF for LaTeX, PNG for previews):
  stage1_calibration_scatter   (Pbar_esc^(M), r*) with the fitted candidate rules,
                               under the protocol-literal truth (canonical
                               NCDawareRank on the complete graph) and the latent
                               population truth; right column joins each base
                               graph's perturbation series (Simpson structure)
  stage1_gt_sensitivity        r* vs Pbar under all four ground truths
  stage1_diagnostics           tau(r) curves, variance decomposition of r*,
                               the latent driver, out-of-sample gap closed

Style: light print surface, three categorical slots validated all-pairs with
the dataviz validator (blue #2a78d6, orange #eb6834, aqua #1baf7a), solid
hairline grid, thin lines, markers with a surface ring, ink-coloured text.
"""
import argparse
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

import ncd_calibration_lib as L  # noqa: E402
import stage1_analysis as A  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(HERE, "figures")
RES = os.path.join(HERE, "results", "stage1")

SURF, INK, INK2, MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#898781"
GRIDC, AXISC = "#e1e0d9", "#c3c2b7"
S1, S2, S3 = "#2a78d6", "#eb6834", "#1baf7a"
BLUE_RAMP = ["#86b6ef", "#6da7ec", "#5598e7", "#3987e5", "#2a78d6", "#256abf", "#1c5cab", "#104281"]
GT_LABEL = {"NCD": "canonical NCDawareRank on the complete graph",
            "PR": "PageRank on the complete graph",
            "SELF": "[NG13] protocol: same r on the complete graph",
            "POP": "latent population importance"}

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 8.5, "axes.titlesize": 9, "axes.labelsize": 8.5,
    "axes.edgecolor": AXISC, "axes.linewidth": 0.8, "axes.labelcolor": INK, "axes.titlecolor": INK,
    "xtick.color": AXISC, "ytick.color": AXISC, "xtick.labelcolor": INK2, "ytick.labelcolor": INK2,
    "axes.grid": True, "grid.color": GRIDC, "grid.linewidth": 0.6, "grid.linestyle": "-",
    "axes.facecolor": SURF, "figure.facecolor": "white", "savefig.facecolor": "white",
    "legend.frameon": False, "legend.fontsize": 7.2, "lines.linewidth": 1.5,
    "axes.spines.top": False, "axes.spines.right": False, "axes.axisbelow": True,
    "text.color": INK,
})


def save(fig, name):
    os.makedirs(FIG, exist_ok=True)
    fig.savefig(os.path.join(FIG, name + ".pdf"), bbox_inches="tight")
    fig.savefig(os.path.join(FIG, name + ".png"), dpi=220, bbox_inches="tight")
    plt.close(fig)


def load_all():
    df, TAU, RHO, TAUc, RHOc, IT = A.load()
    df = A.add_oracles(df, TAU, RHO, TAUc, RHOc)
    return df, TAU, RHO


def rules_from_json(gt):
    out = {}
    for d in json.load(open(os.path.join(RES, f"fits_{gt}.json"))):
        if d.get("feature") == "multi":
            continue
        r = A.Rule(d["name"], d["form"], d.get("params", []), d["feature"], d["fit"])
        if d["form"] == "isotonic":
            from sklearn.isotonic import IsotonicRegression
            xs, ys = np.asarray(d["iso_x"]), np.asarray(d["iso_y"])
            r.iso = IsotonicRegression(increasing=bool(ys[-1] >= ys[0]), out_of_bounds="clip").fit(xs, ys)
        out[d["name"]] = r
    return out


def trajectories(ax, d, gt):
    for (b, e), g in d.groupby(["base", "exp"]):
        g = g.groupby("level")[["pesc_M", f"rstar_{gt}"]].mean().reset_index()
        g = g.sort_values("level", ascending=(e == "B"))
        ax.plot(g["pesc_M"], g[f"rstar_{gt}"], color=AXISC, lw=0.55, zorder=1)


def scatter_ab(ax, d, gt, s=12, alpha=0.7, label=True, short=False):
    labs = (("A", S1, "Exp. A: sparsity"), ("B", S2, "Exp. B: new pages")) if short else (
        ("A", S1, "Exp. A: sparsity (keep 90%-20% of links)"),
        ("B", S2, "Exp. B: new pages (10%-60% lose 90% of in-links)"))
    for e, c, lab in labs:
        g = d[d["exp"] == e]
        ax.scatter(g["pesc_M"], g[f"rstar_{gt}"], s=s, color=c, alpha=alpha, edgecolor=SURF,
                   linewidth=0.4, label=lab if label else None, zorder=3)


def panel_fit(ax, d, gt, rules, xmax):
    xs = np.linspace(0, xmax, 500)
    scatter_ab(ax, d, gt)
    spec = [("cliplin_LS", "-", INK, 1.6, "clip-linear, least squares"),
            ("cliplin_TAU", (0, (5, 2)), INK, 1.6, "clip-linear, tau-optimised"),
            ("isotonic", (0, (1, 1.4)), INK2, 1.6, "isotonic (monotone)"),
            ("const", "-", MUTED, 1.1, "best constant")]
    for name, ls, col, lw, lab in spec:
        ru = rules[name]
        ax.plot(xs, ru(xs), ls=ls, color=col, lw=lw, zorder=4, label=f"{lab}: {ru.formula()}")
    ax.axhline(L.R_CANONICAL, color=S3, lw=1.1, zorder=2,
               label=r"canonical $r=0.10/0.85$ ($\eta=0.75,\ \mu=0.10$)")


def fig_scatter():
    df, TAU, RHO = load_all()
    xmax = 0.85
    fig, axes = plt.subplots(2, 2, figsize=(10.4, 7.6), sharex=True)
    for row, gt in enumerate(["NCD", "POP"]):
        d = df[A.valid_mask(df, gt) & (df["panel"] == "calib").to_numpy()]
        rules = rules_from_json(gt)
        wb = A.within_between(d, "pesc_M", f"rstar_{gt}")
        ax = axes[row, 0]
        panel_fit(ax, d, gt, rules, xmax)
        ax.set_ylabel(rf"oracle ratio $r^*=\mu^*/\alpha_0$")
        ax.set_title(f"({'ac'[row]}) truth: {GT_LABEL[gt]}\n{d['base'].nunique()} calibration graphs, "
                     f"{len(d)} perturbed instances", loc="left", fontsize=8.4)
        ax.legend(loc="upper left", fontsize=6.2, handlelength=2.8, borderaxespad=0.2, ncol=1)
        ax = axes[row, 1]
        trajectories(ax, d, gt)
        scatter_ab(ax, d, gt, s=8, alpha=0.45, label=False)
        bm = d.groupby("base")[["pesc_M", f"rstar_{gt}"]].mean()
        ax.scatter(bm["pesc_M"], bm[f"rstar_{gt}"], s=26, color=INK, edgecolor=SURF, linewidth=0.8, zorder=5,
                   label="mean of each graph")
        ax.text(0.98, 0.97, f"corr(Pbar, r*): pooled {wb['pooled']:+.2f}\nwithin graphs {wb['within']:+.2f}\n"
                            f"between graph means {wb['between']:+.2f}",
                transform=ax.transAxes, ha="right", va="top", fontsize=7.5, color=INK,
                bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="none", alpha=0.85), zorder=6)
        ax.set_title(f"({'bd'[row]}) same points; grey lines join each graph's\nperturbation series (Simpson structure)",
                     loc="left", fontsize=8.4)
        ax.legend(loc="upper left", fontsize=6.8)
        for a in axes[row]:
            a.set_ylim(-0.02, 1.36)
            a.set_yticks(np.arange(0, 1.01, 0.2))
            a.set_xlim(0, xmax)
    for a in axes[1]:
        a.set_xlabel(r"$\bar P^{(M)}_{\mathrm{esc}}$ of the observed graph")
    fig.tight_layout()
    save(fig, "stage1_calibration_scatter")


def fig_gt_sensitivity():
    df, TAU, RHO = load_all()
    fig, axes = plt.subplots(1, 4, figsize=(12.4, 3.3), sharey=True)
    for ax, gt in zip(axes, A.GTS):
        d = df[A.valid_mask(df, gt) & (df["panel"] == "calib").to_numpy()]
        trajectories(ax, d, gt)
        scatter_ab(ax, d, gt, s=7, alpha=0.6, label=(gt == "NCD"), short=True)
        wb = A.within_between(d, "pesc_M", f"rstar_{gt}")
        top = gt != "SELF"
        ax.text(0.98, 0.97 if top else 0.30, f"corr(Pbar, r*)\npooled {wb['pooled']:+.2f}\nwithin graphs {wb['within']:+.2f}\n"
                f"between graphs {wb['between']:+.2f}", transform=ax.transAxes, ha="right", va="top", fontsize=7,
                color=INK, bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="none", alpha=0.85), zorder=6)
        ax.set_title(f"{gt}: {GT_LABEL[gt]}", loc="left", fontsize=7.6)
        ax.set_xlabel(r"$\bar P^{(M)}_{\mathrm{esc}}$")
        ax.set_ylim(-0.02, 1.02)
        ax.set_xlim(0, 0.85)
    axes[0].set_ylabel(r"oracle $r^*$")
    axes[0].legend(loc="upper left", fontsize=6.2)
    fig.tight_layout()
    save(fig, "stage1_gt_sensitivity")


def fig_diagnostics():
    df, TAU, RHO = load_all()
    dec = json.load(open(os.path.join(RES, "decomposition.json")))
    fig, axes = plt.subplots(2, 2, figsize=(10.4, 7.4))

    # (a) tau(r) curves of one calibration graph under the protocol-literal truth
    ax = axes[0, 0]
    gi = A.GTS.index("NCD")
    cal = df[(df["panel"] == "calib") & (df["exp"] == "A") & (df["rep"] == 0) & (df["level"] < 1.0)]
    base = cal.groupby("base")["rstar_NCD"].agg(lambda s: s.max() - s.min()).sort_values().index[-1 * len(cal["base"].unique()) // 2]
    sel = cal[cal["base"] == base].sort_values("level", ascending=False)
    for k, (i, row) in enumerate(sel.iterrows()):
        c = TAU[i, gi]
        col = BLUE_RAMP[min(k, len(BLUE_RAMP) - 1)]
        ax.plot(L.R_GRID, c, color=col, lw=1.3, label=f"keep {row['level']:.0%} of links")
        j = int(np.argmax(np.where(A.OMASK, c, -np.inf)))
        ax.scatter([L.R_GRID[j]], [c[j]], s=22, color=col, edgecolor=SURF, linewidth=0.8, zorder=4)
    ax.axvline(L.R_CANONICAL, color=S3, lw=1.1, label="canonical r")
    ax.legend(loc="upper right", fontsize=6.3, ncol=2)
    ax.set_xlabel(r"mixing ratio $r=\mu/\alpha_0$  ($\alpha_0=0.85$)")
    ax.set_ylabel(r"Kendall $\tau$ vs canonical NCDawareRank (complete graph)")
    ax.set_title(f"(a) tau(r) for one calibration graph, Exp. A; dots mark r*", loc="left")
    ax.set_xlim(0, 1.0)

    # (b) variance decomposition of r*
    ax = axes[0, 1]
    cats = [("r2_base", "graph identity"), ("r2_level", "perturbation level"),
            ("r2_base_level", "graph + level (additive)"),
            ("r2_hidden_cv", "hidden generator params + level*"),
            ("r2_observable_ridge_cv", "all observable statistics (ridge)*"),
            ("r2_pesc_cv", r"$\bar P^{(M)}_{esc}$ alone (clip-linear)*")]
    gts = [("NCD", S1), ("PR", S2), ("POP", S3)]
    y0 = np.arange(len(cats))[::-1]
    h = 0.24
    for k, (gt, col) in enumerate(gts):
        vals = [max(dec[gt][c], 0.0) for c, _ in cats]
        ax.barh(y0 + (1 - k) * h, vals, height=h * 0.86, color=col, label=f"{gt}: {GT_LABEL[gt]}")
        for yy, (c, _), v in zip(y0 + (1 - k) * h, cats, vals):
            raw = dec[gt][c]
            ax.text(v + 0.01, yy, f"{raw:.2f}" if raw > 0 else f"{raw:.2f} (<= 0)", va="center", fontsize=6.3,
                    color=INK2)
    ax.set_yticks(y0)
    ax.set_yticklabels([lab for _, lab in cats], fontsize=7.2)
    ax.set_xlim(0, 1.05)
    ax.set_xlabel(r"share of the variance of $r^*$ explained ($R^2$; * = grouped 8-fold CV by graph)")
    ax.set_title("(b) What determines r*?  (calibration panel)", loc="left")
    ax.legend(loc="lower right", fontsize=6.2)
    ax.grid(axis="y", visible=False)

    # (c) the latent driver: base-mean r* (POP) vs latent fitness ICC, shaded by fitness dispersion
    ax = axes[1, 0]
    d = df[A.valid_mask(df, "POP") & (df["panel"] == "calib").to_numpy()]
    bm = d.groupby("base").agg(r=("rstar_POP", "mean"), icc=("prm_icc", "first"), fs=("prm_fit_sigma", "first"),
                              pesc=("pesc_M", "mean"))
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list("blue", ["#cde2fb", "#3987e5", "#0d366b"])
    sc = ax.scatter(bm["icc"], bm["r"], c=bm["fs"], cmap=cmap, s=30, edgecolor=SURF, linewidth=0.8, zorder=3)
    cb = fig.colorbar(sc, ax=ax, pad=0.01)
    cb.set_label("latent fitness dispersion (sigma of log quality)", fontsize=7)
    cb.ax.tick_params(labelsize=6.5, colors=INK2)
    cb.outline.set_edgecolor(AXISC)
    rr = np.corrcoef(bm["icc"], bm["r"])[0, 1]
    ax.text(0.02, 0.97, f"corr(ICC, mean r*) = {rr:+.2f}\ncorr(dispersion, mean r*) = {np.corrcoef(bm['fs'], bm['r'])[0,1]:+.2f}"
                        f"\ncorr(mean Pbar, mean r*) = {np.corrcoef(bm['pesc'], bm['r'])[0,1]:+.2f}",
            transform=ax.transAxes, va="top", fontsize=7.2, color=INK)
    ax.set_xlabel("latent block share of page quality (ICC, hidden)")
    ax.set_ylabel("mean r* of the graph (population truth)")
    ax.set_title("(c) The driver is latent block-quality structure", loc="left")

    # (d) out-of-sample gap closed (canonical -> oracle)
    ax = axes[1, 1]
    rows = []
    for gt, col in (("NCD", S1), ("POP", S3)):
        s = pd.read_csv(os.path.join(RES, f"summary_{gt}.csv"))
        for p in ["test", "scale", "topo", "real"]:
            for rule, lab in (("cliplin_TAU", "rule"), ("const", "const")):
                q = s[(s["rule"] == rule) & (s["panel"] == p)]
                if len(q):
                    rows.append(dict(gt=gt, panel=p, who=lab, gap=float(q["gap_closed_vs_can"].iloc[0]),
                                     head=float(q["tau_oracle"].iloc[0] - q["tau_canonical"].iloc[0])))
    g = pd.DataFrame(rows)
    order = [(gt, p) for gt in ("NCD", "POP") for p in ["test", "scale", "topo", "real"]
             if ((g["gt"] == gt) & (g["panel"] == p)).any()]
    ys = np.arange(len(order))[::-1]
    for yy, (gt, p) in zip(ys, order):
        q = g[(g["gt"] == gt) & (g["panel"] == p)].set_index("who")
        a_, b_ = q.loc["rule", "gap"], q.loc["const", "gap"]
        ax.plot([b_, a_], [yy, yy], color=AXISC, lw=1.2, zorder=1)
        dy = 0.13 if abs(a_ - b_) < 0.02 else 0.0
        ax.scatter([a_], [yy + dy], s=34, color=S1, edgecolor=SURF, linewidth=0.8, zorder=3,
                   label=r"clip-linear on $\bar P^{(M)}_{esc}$, tau-optimised" if yy == ys[0] else None)
        ax.scatter([b_], [yy - dy], s=34, color=S2, edgecolor=SURF, linewidth=0.8, zorder=3,
                   label="best constant r (calibration)" if yy == ys[0] else None)
        ax.text(1.02, yy, f"headroom {q['head'].iloc[0]:.3f}", fontsize=6.4, color=INK2, va="center",
                transform=ax.get_yaxis_transform())
    ax.axvline(0, color=AXISC, lw=0.8)
    ax.set_yticks(ys)
    ax.set_yticklabels([f"{gt} | {p}" for gt, p in order], fontsize=7.2)
    ax.set_xlabel("share of the canonical-to-oracle tau gap closed (0 = canonical r, 1 = oracle)")
    ax.set_title("(d) Out-of-sample: rule vs constant", loc="left")
    ax.legend(loc="lower left", fontsize=6.4)
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    save(fig, "stage1_diagnostics")


# ---------------------------------------------------------------------------
# Stage 2 figures
# ---------------------------------------------------------------------------
RES2 = os.path.join(HERE, "results", "stage2")
T2_LABEL = {"NCD": "NCD: canonical NCDawareRank, alpha=0.85",
            "NCD95": "NCD95: [NG13] experiments, alpha=0.95",
            "TOT": "TOT: TotalRank = RAPr Beta(1,1)",
            "POP": "POP: population PageRank, alpha=0.85",
            "Q": "Q: latent quality q (walk-free)"}
T2_OWN = {"NCD": 0.85, "NCD95": 0.95, "TOT": 0.50, "POP": 0.85, "Q": None}
T2_R = {"NCD": "rule", "NCD95": "ng13", "TOT": "rule", "POP": "popLS", "Q": "rule"}
LINE5 = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]   # adjacent-validated slots 1-5 (lines/bars)


def _s2():
    import stage2_analysis as A2
    df, TAU, RHO, _ = A2.load()
    fits = json.load(open(os.path.join(RES2, "fits.json")))
    return A2, df, TAU, fits


def _calib_pair(A2, df, TAU, r, t):
    m = A2.valid(df, t) & (df["panel"] == "calib").to_numpy() & np.isfinite(A2.curves(TAU, r, t)[:, -1])
    d = df[m].copy()
    a, _ = A2.oracle(A2.curves(TAU, r, t)[m])
    d["a_star"] = a
    return d


def _traj2(ax, d, x):
    for (b, e), g in d.groupby(["base", "exp"]):
        g = g.groupby("level")[[x, "a_star"]].mean().reset_index().sort_values("level", ascending=(e == "B"))
        ax.plot(g[x], g["a_star"], color=AXISC, lw=0.55, zorder=1)


def _scatter2(ax, d, x, s=12, alpha=0.7, label=True):
    for e, c, lab in (("A", S1, "Exp. A: sparsity"), ("B", S2, "Exp. B: new pages")):
        g = d[d["exp"] == e]
        ax.scatter(g[x], g["a_star"], s=s, color=c, alpha=alpha, edgecolor=SURF, linewidth=0.4,
                   label=lab if label else None, zorder=3)


def _fit_panel2(ax, A2, d, fits, r, t, show_legend=True):
    x = f"h_{r}"
    _scatter2(ax, d, x)
    rules = {dd["name"]: A2.rule_from_dict(dd) for dd in fits[f"{r}|{t}|{x}"] if dd.get("form") != "gbm"}
    xs = np.linspace(d[x].min(), d[x].max(), 400)
    spec = [("cliplin_LS", "-", INK, 1.6, "clip-linear LS"), ("cliplin_TAU", (0, (5, 2)), INK, 1.6, "clip-linear tau-opt."),
            ("isotonic", (0, (1, 1.4)), INK2, 1.6, "isotonic"), ("const", "-", MUTED, 1.1, "best constant")]
    for name, ls, col, lw, lab in spec:
        if name in rules:
            ax.plot(xs, rules[name](xs), ls=ls, color=col, lw=lw, zorder=4, label=f"{lab}: {rules[name].formula()}")
    ax.axhline(0.85, color=S3, lw=1.1, zorder=2, label="alpha0 = 0.85 (Stage 1)")
    ax.axhline(0.95, color=S3, lw=1.1, ls=(0, (3, 2)), zorder=2, label="alpha = 0.95 ([NG13] experiments)")
    ax.set_ylim(0.38, 1.01)
    ax.set_yticks(np.arange(0.4, 1.01, 0.1))
    if show_legend:   # below the axes, so it never covers data
        ax.legend(loc="upper left", bbox_to_anchor=(0.0, -0.17), fontsize=6.3, handlelength=2.6, ncol=1)


def fig2_scatter():
    A2, df, TAU, fits = _s2()
    fig, axes = plt.subplots(2, 2, figsize=(10.6, 12.0), gridspec_kw=dict(hspace=0.95, wspace=0.18))
    d = _calib_pair(A2, df, TAU, "rule", "NCD")
    ax = axes[0, 0]
    _fit_panel2(ax, A2, d, fits, "rule", "NCD")
    ax.set_title(f"(a) truth NCD (primary), r = Stage 1 rule\n{d['base'].nunique()} calibration graphs, "
                 f"{len(d)} instances", loc="left", fontsize=8.4)
    ax.set_ylabel(r"oracle damping $\alpha^*$")
    ax = axes[0, 1]
    _traj2(ax, d, "h_rule")
    _scatter2(ax, d, "h_rule", s=8, alpha=0.45, label=False)
    bm = d.groupby("base")[["h_rule", "a_star"]].mean()
    ax.scatter(bm["h_rule"], bm["a_star"], s=26, color=INK, edgecolor=SURF, linewidth=0.8, zorder=5,
               label="mean of each graph")
    wb = A.within_between(d, "h_rule", "a_star")
    ax.text(0.98, 0.97, f"corr(h, alpha*): pooled {wb['pooled']:+.2f}\nwithin graphs {wb['within']:+.2f}\n"
                        f"between graph means {wb['between']:+.2f}", transform=ax.transAxes, ha="right", va="top",
            fontsize=7.5, bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="none", alpha=0.85), zorder=6)
    ax.set_ylim(0.38, 1.01)
    ax.set_yticks(np.arange(0.4, 1.01, 0.1))
    ax.legend(loc="upper left", bbox_to_anchor=(0.0, -0.17), fontsize=6.3)
    ax.set_title("(b) same points; grey lines join each graph's\nperturbation series", loc="left", fontsize=8.4)
    for ax, t, letter in ((axes[1, 0], "TOT", "c"), (axes[1, 1], "Q", "d")):
        dd = _calib_pair(A2, df, TAU, "rule", t)
        _fit_panel2(ax, A2, dd, fits, "rule", t, show_legend=True)
        ax.set_title(f"({letter}) alpha-free truth {T2_LABEL[t]}\nr = Stage 1 rule", loc="left", fontsize=8.4)
        ax.set_xlabel(r"$h=\frac{1}{n}\sum_v|1-\Sigma_{\mathrm{in}}(v)|$ at the Stage 1 $r$")
    axes[1, 0].set_ylabel(r"oracle damping $\alpha^*$")
    axes[0, 0].set_xlabel(r"$h$ at the Stage 1 $r$")
    axes[0, 1].set_xlabel(r"$h$ at the Stage 1 $r$")
    save(fig, "stage2_calibration_scatter")


def fig2_gt_sensitivity():
    A2, df, TAU, fits = _s2()
    fig, axes = plt.subplots(1, 5, figsize=(14.5, 3.4), sharey=True)
    for ax, t in zip(axes, A2.TRUTHS):
        r = T2_R[t]
        d = _calib_pair(A2, df, TAU, r, t)
        x = f"h_{r}"
        _traj2(ax, d, x)
        _scatter2(ax, d, x, s=7, alpha=0.6, label=(t == "NCD"))
        wb = A.within_between(d, x, "a_star")
        if T2_OWN[t] is not None:
            ax.axhline(T2_OWN[t], color=S3, lw=1.1, label="truth's own alpha" if t == "NCD" else None)
        top = t in ("TOT", "POP")
        ax.text(0.98, 0.97 if top else 0.03, f"corr(h, alpha*)\npooled {wb['pooled']:+.2f}\nwithin {wb['within']:+.2f}\n"
                f"between {wb['between']:+.2f}", transform=ax.transAxes, ha="right", va="top" if top else "bottom",
                fontsize=6.8, bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="none", alpha=0.85), zorder=6)
        ax.set_title(f"{T2_LABEL[t]}\nr = {r}", loc="left", fontsize=7.2)
        ax.set_xlabel(f"h at r = {r}")
        ax.set_ylim(0.38, 1.02)
    axes[0].set_ylabel(r"oracle $\alpha^*$")
    axes[0].legend(loc="lower left", fontsize=6.2, framealpha=0.9)
    fig.tight_layout()
    save(fig, "stage2_gt_sensitivity")


def fig2_diagnostics():
    A2, df, TAU, fits = _s2()
    dec = json.load(open(os.path.join(RES2, "decomposition.json")))
    pipe = pd.read_csv(os.path.join(RES2, "pipelines.csv"))
    fig, axes = plt.subplots(2, 2, figsize=(10.6, 7.6))
    # (a) where each truth's tau(alpha) peaks, one instance
    ax = axes[0, 0]
    cal = df[(df["panel"] == "calib") & (df["exp"] == "A") & (df["level"] == 0.6) & (df["rep"] == 0)]
    i = cal.index[len(cal) // 2]
    for k, t in enumerate(A2.TRUTHS):
        r = T2_R[t]
        c = A2.curves(TAU, r, t)[i]
        dc = c - np.nanmax(c[A2.OMASK])
        ax.plot(A2.GRID, dc, color=LINE5[k], lw=1.5, label=f"{t} (r = {r})")
        j = int(np.argmax(np.where(A2.OMASK, c, -np.inf)))
        ax.scatter([A2.GRID[j]], [0], s=26, color=LINE5[k], edgecolor=SURF, linewidth=0.8, zorder=4)
    ax.axvspan(0, 0.40, color=GRIDC, alpha=0.6, lw=0)
    ax.text(0.2, -0.0015, "outside the\nsearch grid", ha="center", va="top", fontsize=6.5, color=INK2)
    ax.set_ylim(-0.03, 0.004)
    ax.set_xlim(0.05, 1.0)
    ax.set_xlabel(r"damping $\alpha$")
    ax.set_ylabel(r"$\tau(\alpha)-\max_\alpha\tau$")
    ax.set_title("(a) One graph (Exp. A, keep 60%): each truth's\ntau(alpha) peaks at its own alpha; dots = alpha*",
                 loc="left", fontsize=8.4)
    ax.legend(loc="lower center", fontsize=6.3, ncol=2)
    # (b) alpha* distribution per truth (calibration panel)
    ax = axes[0, 1]
    data, labels = [], []
    for t in A2.TRUTHS:
        d = _calib_pair(A2, df, TAU, T2_R[t], t)
        data.append(d["a_star"].to_numpy())
        labels.append(f"{t}\n(r={T2_R[t]})")
    bp = ax.boxplot(data, vert=True, widths=0.5, patch_artist=True, showfliers=False,
                    medianprops=dict(color=INK, lw=1.4), whiskerprops=dict(color=INK2), capprops=dict(color=INK2),
                    boxprops=dict(facecolor="#cde2fb", edgecolor=S1))
    rng_ = np.random.default_rng(0)
    for k, v in enumerate(data, 1):
        ax.scatter(k + rng_.uniform(-0.18, 0.18, len(v)), v, s=3, color=S1, alpha=0.25, zorder=3, lw=0)
        if T2_OWN[A2.TRUTHS[k - 1]] is not None:
            ax.plot([k - 0.32, k + 0.32], [T2_OWN[A2.TRUTHS[k - 1]]] * 2, color=S2, lw=1.6, zorder=5)
    ax.set_xticks(range(1, len(labels) + 1))
    ax.set_xticklabels(labels, fontsize=7)
    ax.set_ylim(0.38, 1.01)
    ax.set_ylabel(r"oracle $\alpha^*$ (calibration panel)")
    ax.set_title("(b) alpha* is set by the choice of truth\n(orange bar = the truth's own alpha; TOT: Beta(1,1) mean 0.5)",
                 loc="left", fontsize=8.4)
    ax.grid(axis="x", visible=False)
    # (c) variance decomposition of alpha*
    ax = axes[1, 0]
    cats = [("r2_base", "graph identity"), ("r2_level", "perturbation level"), ("r2_base_level", "graph + level"),
            ("r2_hidden_cv", "hidden params + level*"), ("r2_observable_ridge_cv", "all observables incl. h*"),
            ("r2_h_cv", "h alone (clip-linear)*")]
    keys = [("rule|NCD", "NCD, r=rule"), ("rule|TOT", "TOT, r=rule"), ("rule|Q", "Q, r=rule"), ("popLS|POP", "POP, r=popLS")]
    y0 = np.arange(len(cats))[::-1]
    hgt = 0.2
    for k, (key, lab) in enumerate(keys):
        if key not in dec:
            continue
        vals = [dec[key][c] for c, _ in cats]
        ax.barh(y0 + (1.5 - k) * hgt, [max(v, 0) for v in vals], height=hgt * 0.86, color=LINE5[k], label=lab)
        for yy, v in zip(y0 + (1.5 - k) * hgt, vals):
            ax.text(max(v, 0) + 0.01, yy, f"{v:.2f}", va="center", fontsize=5.6, color=INK2)
    ax.set_yticks(y0)
    ax.set_yticklabels([c[1] for c in cats], fontsize=7)
    ax.set_xlim(0, 1.08)
    ax.set_xlabel(r"share of the variance of $\alpha^*$ explained ($R^2$; * grouped 8-fold CV)")
    ax.set_title("(c) What determines alpha*?", loc="left", fontsize=8.4)
    ax.legend(loc="lower right", fontsize=6.2)
    ax.grid(axis="y", visible=False)
    # (d) pipelines vs the static NG13 baseline (test panel)
    ax = axes[1, 1]
    pols = ["NG13 static (r=0.105, a=0.95)", "Stage 1 only (rule r, a=0.85)", "Stage 1 const r=0.18, a=0.85",
            "canonical r + g(h) [Stage 2 only]", "two-stage: rule r + cliplin_TAU(h)", "two-stage: rule r + cliplin_LS(h)",
            "oracle alpha* at rule r"]
    shown = [("NCD", S1), ("TOT", S2), ("Q", S3)]
    ys = np.arange(len(pols))[::-1]
    for k, (t, col) in enumerate(shown):
        q = pipe[(pipe["truth"] == t) & (pipe["panel"] == "test")].set_index("policy")
        vals = [q.loc[p, "d_vs_ng13_085"] if p in q.index else np.nan for p in pols]
        ax.scatter(vals, ys + (1 - k) * 0.18, s=26, color=col, edgecolor=SURF, linewidth=0.8, zorder=3,
                   label=f"truth {t}")
    ax.axvline(0, color=AXISC, lw=0.9)
    ax.set_yticks(ys)
    ax.set_yticklabels(pols, fontsize=6.6)
    ax.set_xlabel("mean tau gain over the static [NG13] baseline (r = 0.118, alpha = 0.85), test panel")
    ax.set_title("(d) Pipelines vs the static baseline (held-out test panel)", loc="left", fontsize=8.4)
    ax.legend(loc="lower right", fontsize=6.4)
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    save(fig, "stage2_diagnostics")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["stage1", "stage2"])
    ap.add_argument("--only", default="")
    ap.add_argument("--lang", default="en", choices=["en", "gr"])
    a = ap.parse_args()
    if a.lang == "gr":       # Greek labels: translate every text call, write to figures_gr/
        global FIG
        import fig_lang
        fig_lang.patch()
        FIG = os.path.join(HERE, "figures_gr")
    if a.what == "stage1":
        todo = a.only.split(",") if a.only else ["scatter", "gt", "diag"]
        if "scatter" in todo:
            fig_scatter()
        if "gt" in todo:
            fig_gt_sensitivity()
        if "diag" in todo:
            fig_diagnostics()
    else:
        todo = a.only.split(",") if a.only else ["scatter", "gt", "diag"]
        if "scatter" in todo:
            fig2_scatter()
        if "gt" in todo:
            fig2_gt_sensitivity()
        if "diag" in todo:
            fig2_diagnostics()


if __name__ == "__main__":
    main()
