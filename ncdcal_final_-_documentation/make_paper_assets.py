"""
make_paper_assets.py
====================
Generates every number and table used in paper_en.tex / paper_gr.tex from the saved
result files, so that no number in either manuscript is typed by hand.

Outputs
  paper/generated/macros.tex          \\newcommand macros for numbers quoted in the text
  paper/generated/en/tab_*.tex        tables with English captions and labels
  paper/generated/gr/tab_*.tex        the same tables with Greek captions and labels
  results/paper_numbers.json          the same numbers in machine-readable form

Run after both stages, rapr_anchor, upper_check and convergence_check:
    python3 make_paper_assets.py
"""
import glob
import json
import os
import warnings

import numpy as np
import pandas as pd

import ncd_calibration_lib as L
import stage1_analysis as S1
import stage2_analysis as S2

warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
R1, R2 = os.path.join(HERE, "results", "stage1"), os.path.join(HERE, "results", "stage2")
OUT = os.path.join(HERE, "paper", "generated")
NUM = {}            # macro name -> LaTeX string


# ---------------------------------------------------------------------------
# formatting helpers
# ---------------------------------------------------------------------------
def f3(x):
    return f"{x:.3f}".replace("-", "$-$")


def f4(x):
    return f"{x:.4f}".replace("-", "$-$")


def sgn4(x):
    if abs(x) < 5e-5:
        return "0.0000"
    return f"{x:+.4f}".replace("-", "$-$").replace("+", "$+$")


def sgn3(x):
    if abs(x) < 5e-4:
        return "0.000"
    return f"{x:+.3f}".replace("-", "$-$").replace("+", "$+$")


def pv(p):
    if p is None or (isinstance(p, float) and np.isnan(p)):
        return "--"
    if p < 1e-3:
        m, e = f"{p:.1e}".split("e")
        return f"${m}\\times10^{{{int(e)}}}$"
    return f"{p:.3f}"


def macro(name, value):
    assert name.isalpha(), name
    NUM[name] = value


def mode_agnostic(value):
    """'$-$0.012' -> '\\ensuremath{-0.012}', '$1.3\\times10^{-4}$' -> '\\ensuremath{1.3\\times10^{-4}}'."""
    s = value.replace("$-$", "-").replace("$+$", "+").replace("$", "")
    if any(c in s for c in "-+\\^"):
        return f"\\ensuremath{{{s}}}"
    return s


def write(path, text):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as fh:
        fh.write(text)


def table_env(label, cap_en, cap_gr, body, lang, size="\\small", star=False, place="t"):
    env = "table*" if star else "table"
    cap = cap_en if lang == "en" else cap_gr
    return (f"\\begin{{{env}}}[{place}]\n\\centering\n{size}\n\\caption{{{cap}}}\n\\label{{{label}}}\n"
            f"{body}\n\\end{{{env}}}\n")


# ---------------------------------------------------------------------------
# raw Stage 2 arrays (for confidence intervals on exact grid points)
# ---------------------------------------------------------------------------
df2, TAU2, RHO2, MS2 = S2.load()
GRID = S2.GRID
J = {a: int(np.argmin(np.abs(GRID - a))) for a in (0.81, 0.85, 0.93, 0.95)}


def tau_at(r, t, a, mask):
    return S2.curves(TAU2, r, t)[mask][:, J[a]]


def cmp_stats(t, panel, pol_a, pol_b):
    """cluster stats of tau(pol_a) - tau(pol_b) on one panel/truth; policies are (r, alpha)."""
    m = S2.valid(df2, t) & (df2["panel"] == panel).to_numpy()
    if m.sum() == 0:
        return None
    d = tau_at(*pol_a[:1], t, pol_a[1], m) - tau_at(*pol_b[:1], t, pol_b[1], m)
    st = S1.cluster_stats(d, df2["base"].to_numpy()[m])
    st["inst_mean"] = float(np.mean(d))
    return st


# ---------------------------------------------------------------------------
# 1. data and panels
# ---------------------------------------------------------------------------
def real_dataset_table():
    summ = json.load(open(os.path.join(HERE, "data", "real", "summary.json")))
    names = {"cnr2000": "cnr-2000", "wikics": "WikiCS", "cora_full": "Cora (full)", "cora_ml": "Cora-ML",
             "citeseer": "CiteSeer"}
    part_en = {"cnr2000": "host proxy (contiguous URL-order segments)", "wikics": "10 topic labels",
               "cora_full": "70 topic labels", "cora_ml": "7 topic labels", "citeseer": "6 topic labels"}
    part_gr = {"cnr2000": "προσέγγιση hosts (συνεχή τμήματα σειράς URL)", "wikics": "10 θεματικές ετικέτες",
               "cora_full": "70 θεματικές ετικέτες", "cora_ml": "7 θεματικές ετικέτες", "citeseer": "6 θεματικές ετικέτες"}
    rows_en, rows_gr = [], []
    for k in ["cnr2000", "wikics", "cora_full", "cora_ml", "citeseer"]:
        s = summ[k]
        z = np.load(os.path.join(HERE, "data", "real", f"{k}.npz"))
        ops = L.build_operators(L.Graph(int(z["n"]), z["src"], z["dst"], z["block"]))
        pb = L.block_escape(ops)[1]
        hh = L.h_statistics(ops, L.R_CANONICAL)
        NUM_real[k] = dict(pbar=pb, h=hh["h"], hv=hh["h_v"])
        common = (f"{s['n']:,} & {s['arcs']:,} & {s['blocks']:,} & {100 * s['intra_block_arc_fraction']:.1f} & "
                  f"{100 * s['dangling_fraction']:.1f} & {pb:.3f}").replace(",", "\\,")
        rows_en.append(f"{names[k]} & {part_en[k]} & {common} \\\\")
        rows_gr.append(f"{names[k]} & {part_gr[k]} & {common} \\\\")
    head_en = ("graph & NCD blocks & $n$ & arcs & blocks & intra-block arcs (\\%) & dangling (\\%) & "
               "$\\bar P^{(M)}_{\\mathrm{esc}}$ \\\\")
    head_gr = ("γράφος & μπλοκ NCD & $n$ & τόξα & μπλοκ & εντός μπλοκ (\\%) & κρεμάμενες (\\%) & "
               "$\\bar P^{(M)}_{\\mathrm{esc}}$ \\\\")
    for lang, head, rows in (("en", head_en, rows_en), ("gr", head_gr, rows_gr)):
        body = ("\\begin{tabular}{@{}l>{\\raggedright\\arraybackslash}p{3.6cm}rrrrrr@{}}\n\\toprule\n" + head + "\n\\midrule\n" + "\n".join(rows) +
                "\n\\bottomrule\n\\end{tabular}")
        cap_en = ("Real networks used for transfer. Self-loops and duplicate arcs removed. "
                  "$\\bar P^{(M)}_{\\mathrm{esc}}$ is computed on the complete graph.")
        cap_gr = ("Πραγματικά δίκτυα για τον έλεγχο μεταφοράς. Οι βρόχοι και τα διπλά τόξα έχουν αφαιρεθεί. "
                  "Το $\\bar P^{(M)}_{\\mathrm{esc}}$ υπολογίζεται στον πλήρη γράφο.")
        write(os.path.join(OUT, lang, "tab_real.tex"), table_env("tab:real", cap_en, cap_gr, body, lang, "\\footnotesize", star=True))


NUM_real = {}


def panel_table():
    rows = []
    for p in S2.PANELS:
        d = df2[df2["panel"] == p]
        rows.append((p, d["base"].nunique(), len(d), int(d["n"].min()), int(d["n"].max())))
    gen_en = {"calib": "static fitness block web", "test": "static (independent seeds)",
              "scale": "static, $8\\times$ larger", "topo": "growth / preferential attachment",
              "real": "Table~\\ref{tab:real}"}
    gen_gr = {"calib": "στατικός ιστός με μπλοκ", "test": "στατικός (ανεξάρτητοι σπόροι)",
              "scale": "στατικός, $8\\times$ μεγαλύτερος", "topo": "ανάπτυξη / προνομιακή προσκόλληση",
              "real": "Πίνακας~\\ref{tab:real}"}
    role_en = {"calib": "fitting", "test": "held-out", "scale": "scale transfer", "topo": "topology transfer",
               "real": "real-network transfer"}
    role_gr = {"calib": "προσαρμογή", "test": "εκτός δείγματος", "scale": "μεταφορά κλίμακας",
               "topo": "μεταφορά τοπολογίας", "real": "πραγματικά δίκτυα"}
    for lang, gen, role, head in (("en", gen_en, role_en, "panel & generator & role & graphs & instances & $n$ \\\\"),
                                  ("gr", gen_gr, role_gr, "σύνολο & γεννήτρια & ρόλος & γράφοι & στιγμιότυπα & $n$ \\\\")):
        lines = []
        th = lambda x: f"{x:,}".replace(",", "\\,")          # thin-space thousands separator
        for p, g, i, n0, n1 in rows:
            nn = th(n0) if n0 == n1 else f"{th(n0)}--{th(n1)}"
            lines.append(f"\\texttt{{{p}}} & {gen[p]} & {role[p]} & {g} & {th(i)} & {nn} \\\\")
        body = ("\\begin{tabular}{@{}lllrrr@{}}\n\\toprule\n" + head + "\n\\midrule\n" + "\n".join(lines) +
                "\n\\bottomrule\n\\end{tabular}")
        cap_en = ("Experimental panels. Every instance is one perturbation of one base graph (Experiments A and B, "
                  "Section~\\ref{sec:perturb}); splits are by base graph. The instance counts are those of the "
                  "Stage~2 replay and include the unperturbed graph A1.0.")
        cap_gr = ("Σύνολα πειραμάτων. Κάθε στιγμιότυπο είναι μία διαταραχή ενός γράφου βάσης (Πειράματα Α και Β, "
                  "Ενότητα~\\ref{sec:perturb})· ο διαχωρισμός γίνεται ανά γράφο βάσης. Τα πλήθη στιγμιοτύπων είναι "
                  "αυτά της επανάληψης του Σταδίου~2 και περιλαμβάνουν τον αδιατάρακτο γράφο A1.0.")
        write(os.path.join(OUT, lang, "tab_panels.tex"),
              table_env("tab:panels", cap_en, cap_gr, body, lang, "\\small" if lang == "en" else "\\footnotesize"))
    macro("nInstTotal", f"{len(df2):,}".replace(",", "\\,"))
    macro("nCalibGraphs", str(df2[df2.panel == 'calib']['base'].nunique()))


# ---------------------------------------------------------------------------
# 2. Stage 1
# ---------------------------------------------------------------------------
def stage1():
    s = pd.read_csv(os.path.join(R1, "summary_NCD.csv"))
    fits = {d["name"]: d for d in json.load(open(os.path.join(R1, "fits_NCD.json")))}
    diag = json.load(open(os.path.join(R1, "diagnostics_NCD.json")))
    dec = json.load(open(os.path.join(R1, "decomposition.json")))
    wb, wbe = diag["within_between"], diag["within_between_by_exp"]
    macro("SOnePooled", sgn3(wb["pooled"]))
    macro("SOneWithin", sgn3(wb["within"]))
    macro("SOneBetween", sgn3(wb["between"]))
    macro("SOneWithinA", sgn3(wbe["A"]["within"]))
    macro("SOneWithinB", sgn3(wbe["B"]["within"]))
    macro("SOneRetest", f3(diag["noise_ceiling"]["test_retest_pearson"]))
    macro("SOneSpearmanKendall", f3(diag["rstar_kendall_vs_spearman"]["pearson"]))
    for k, nm in (("NCD", "Ncd"), ("PR", "Pr"), ("POP", "Pop"), ("SELF", "Self")):
        dd = json.load(open(os.path.join(R1, f"diagnostics_{k}.json")))
        macro(f"SOne{nm}WithinT", sgn3(dd["within_between"]["within"]))
        macro(f"SOne{nm}BetweenT", sgn3(dd["within_between"]["between"]))
        macro(f"SOne{nm}Lower", f"{100 * dd['rstar_at_bounds']['lower']:.0f}")
        macro(f"SOne{nm}Upper", f"{100 * dd['rstar_at_bounds']['upper']:.0f}")
    # P vs keep fraction within graphs
    inst1 = pd.read_csv(os.path.join(R1, "instances.csv"))
    c = inst1[(inst1.panel == "calib") & ~((inst1.exp == "A") & (inst1.level == 1.0))]
    for e in ("A", "B"):
        ce = c[c.exp == e]
        g = ce.groupby("base")
        x, y = ce.pesc_M - g.pesc_M.transform("mean"), ce.keep_frac - g.keep_frac.transform("mean")
        macro(f"SOnePkeep{e}", sgn3(np.corrcoef(x, y)[0, 1]))
    # candidate table
    order = [("const", "constant $r$", "σταθερό $r$"),
             ("cliplin_LS", "clip-linear, LS", "αποκομμένος γραμμικός, ΕΤ"),
             ("cliplin_TAU", "clip-linear, $\\tau$-opt.\\ (locked)", "αποκομμένος γραμμικός, βέλτ.\\ $\\tau$ (κλειδ.)"),
             ("logistic_TAU", "logistic, $\\tau$-opt.", "λογιστικός, βέλτ.\\ $\\tau$"),
             ("isotonic", "isotonic", "ισοτονικός")]
    panels = ["test", "scale", "topo", "real"]
    body_rows = {"en": [], "gr": []}
    for name, en, gr in order:
        cal = s[(s.rule == name) & (s.panel == "calib")].iloc[0]
        cells = []
        for p in panels:
            q = s[(s.rule == name) & (s.panel == p)].iloc[0]
            cells.append(f4(q["tau"]))
        q = s[(s.rule == name) & (s.panel == "test")].iloc[0]
        gap = f"{100 * q['gap_closed_vs_can']:.0f}"
        pc = "--" if name == "const" else pv(q.get("vs_const_p", np.nan))
        if name in ("isotonic", "logistic_TAU"):
            form = "--"
        elif name == "const":
            form = f"${fits[name]['params'][0]:.3f}$"
        else:
            pr = fits[name]["params"]
            form = f"$\\mathrm{{clip}}({pr[0]:.3f}{pr[1]:+.3f}\\,\\bar P)$"
        for lang, lab in (("en", en), ("gr", gr)):
            body_rows[lang].append(f"{lab} & {form} & {f4(cal['cv_tau'])} & " + " & ".join(cells) + f" & {gap} & {pc} \\\\")
    can, orc = [], []
    for p in panels:
        q = s[(s.rule == "const") & (s.panel == p)].iloc[0]
        can.append(f4(q["tau_canonical"]))
        orc.append(f4(q["tau_oracle"]))
    head_en = ("rule & formula & CV $\\tau$ & \\texttt{test} & \\texttt{scale} & \\texttt{topo} & \\texttt{real} & "
               "gap \\% & $p$ \\\\")
    head_gr = ("κανόνας & τύπος & CV $\\tau$ & \\texttt{test} & \\texttt{scale} & \\texttt{topo} & \\texttt{real} & "
               "κάλυψη \\% & $p$ \\\\")
    for lang, head in (("en", head_en), ("gr", head_gr)):
        lab_can = "canonical $r=0.118$" if lang == "en" else "κανονικό $r=0.118$"
        lab_orc = "oracle $r^*$" if lang == "en" else "μαντείο $r^*$"
        colspec = "@{}llrrrrrrr@{}" if lang == "en" else "@{}>{\\raggedright\\arraybackslash}p{3.4cm}lrrrrrrr@{}"
        body = ("\\begin{tabular}{" + colspec + "}\n\\toprule\n" + head + "\n\\midrule\n" + "\n".join(body_rows[lang]) +
                "\n\\midrule\n" + f"{lab_can} & & & " + " & ".join(can) + " & 0 & \\\\\n" +
                f"{lab_orc} & & & " + " & ".join(orc) + " & 100 & \\\\" + "\n\\bottomrule\n\\end{tabular}")
        cap_en = ("Stage~1 candidate rules $r=f(\\bar P)$, $\\bar P=\\bar P^{(M)}_{\\mathrm{esc}}$, at $\\alpha_0=0.85$, "
                  "truth NCD. Columns \\texttt{test}--\\texttt{real}: mean Kendall $\\tau$ at the rule's $r$. CV $\\tau$: "
                  "grouped 8-fold cross-validation over calibration graphs. Gap: share of the canonical-to-oracle gap "
                  "closed on \\texttt{test}. $p$: Wilcoxon signed-rank test of the per-graph mean differences against the "
                  "constant rule on \\texttt{test}. Clip-linear rules are clipped to $[0.01,0.99]$; LS: least squares on "
                  "the oracle values; $\\tau$-opt.: fitted by maximising the mean calibration $\\tau$. The logistic and "
                  "isotonic rules are reported for completeness.")
        cap_gr = ("Υποψήφιοι κανόνες του Σταδίου~1, $r=f(\\bar P)$ με $\\bar P=\\bar P^{(M)}_{\\mathrm{esc}}$, για "
                  "$\\alpha_0=0.85$ και αλήθεια NCD. Στήλες \\texttt{test}--\\texttt{real}: μέσο $\\tau$ του Kendall στο "
                  "$r$ του κανόνα. CV $\\tau$: ομαδοποιημένη 8-πλή διασταυρωμένη επικύρωση ανά γράφο βαθμονόμησης. "
                  "Κάλυψη: ποσοστό του χάσματος κανονικού--μαντείου που καλύπτεται στο \\texttt{test}. $p$: έλεγχος "
                  "προσημασμένων τάξεων του Wilcoxon στις μέσες διαφορές ανά γράφο έναντι του σταθερού κανόνα στο "
                  "\\texttt{test}. Οι αποκομμένοι γραμμικοί κανόνες περιορίζονται στο $[0.01,0.99]$· ΕΤ: ελάχιστα τετράγωνα "
                  "στις βέλτιστες τιμές· βέλτ.\\ $\\tau$: προσαρμογή με μεγιστοποίηση του μέσου $\\tau$ βαθμονόμησης. "
                  "Ο λογιστικός και ο ισοτονικός κανόνας δίνονται για πληρότητα.")
        write(os.path.join(OUT, lang, "tab_stage1.tex"), table_env("tab:stage1", cap_en, cap_gr, body, lang, "\\footnotesize", star=True))
    # headline Stage 1 numbers
    get = lambda rule, p, col: s[(s.rule == rule) & (s.panel == p)].iloc[0][col]
    macro("SOneTestCan", f4(get("const", "test", "tau_canonical")))
    macro("SOneTestOracle", f4(get("const", "test", "tau_oracle")))
    macro("SOneTestRule", f4(get("cliplin_TAU", "test", "tau")))
    macro("SOneTestConst", f4(get("const", "test", "tau")))
    macro("SOneTestGapRule", f"{100 * get('cliplin_TAU', 'test', 'gap_closed_vs_can'):.0f}")
    macro("SOneTestGapConst", f"{100 * get('const', 'test', 'gap_closed_vs_can'):.0f}")
    macro("SOneTestPConst", pv(get("cliplin_TAU", "test", "vs_const_p")))
    macro("SOneTopoRule", f4(get("cliplin_TAU", "topo", "tau")))
    macro("SOneTopoConst", f4(get("const", "topo", "tau")))
    macro("SOneTopoCan", f4(get("const", "topo", "tau_canonical")))
    macro("SOneTopoPConst", pv(get("cliplin_TAU", "topo", "vs_const_p")))
    macro("SOneTopoConstVsCanP", pv(get("const", "topo", "vs_can_p")))
    macro("SOneTopoConstVsCan", sgn4(get("const", "topo", "vs_can_mean")))
    macro("SOneTopoRuleVsConst", sgn4(get("cliplin_TAU", "topo", "vs_const_mean")))
    macro("SOneRealRule", f4(get("cliplin_TAU", "real", "tau")))
    macro("SOneRealConst", f4(get("const", "real", "tau")))
    macro("SOneRealCan", f4(get("const", "real", "tau_canonical")))
    macro("SOneRealPConst", pv(get("cliplin_TAU", "real", "vs_const_p")))
    macro("SOneScaleRule", f4(get("cliplin_TAU", "scale", "tau")))
    macro("SOneScaleConst", f4(get("const", "scale", "tau")))
    macro("SOneTestConstVsCan", sgn4(get("const", "test", "vs_can_mean")))
    macro("SOneTestConstVsCanLo", f4(get("const", "test", "vs_can_lo")))
    macro("SOneTestConstVsCanHi", f4(get("const", "test", "vs_can_hi")))
    macro("SOneTestConstVsCanWins", str(int(get("const", "test", "vs_can_wins"))))
    macro("SOneTestConstVsCanP", pv(get("const", "test", "vs_can_p")))
    cvd = diag["cv_tau"]
    macro("SOneCVConst", f4(cvd["const"]))
    macro("SOneCVRule", f4(cvd["cliplin_TAU"]))
    macro("SOneCVLogit", f4(cvd["logistic_TAU"]))
    macro("SOneCVLS", f4(cvd["cliplin_LS"]))
    # decomposition table
    rows = {"en": [], "gr": []}
    lab_t = {"NCD": ("NCD", "NCD"), "PR": ("PR", "PR"), "POP": ("POP", "POP"), "SELF": ("SELF", "SELF")}
    for t in ["NCD", "PR", "POP", "SELF"]:
        d = dec[t]
        cells = " & ".join(f"{d[k]:.2f}" for k in ["r2_base", "r2_level", "r2_base_level", "r2_hidden_cv",
                                                    "r2_observable_ridge_cv", "r2_pesc_cv"])
        for lang in ("en", "gr"):
            rows[lang].append(f"{lab_t[t][0]} & {cells} \\\\")
    head_en = ("truth & graph & level & graph+level & hidden+level$^{*}$ & all observables$^{*}$ & "
               "$\\bar P^{(M)}_{\\mathrm{esc}}$ alone$^{*}$ \\\\")
    head_gr = ("αλήθεια & γράφος & επίπεδο & γράφος+επίπεδο & κρυφές+επίπεδο$^{*}$ & όλα τα παρατηρ.$^{*}$ & "
               "μόνο $\\bar P^{(M)}_{\\mathrm{esc}}$$^{*}$ \\\\")
    for lang, head in (("en", head_en), ("gr", head_gr)):
        body = ("\\begin{tabular}{@{}lrrrrrr@{}}\n\\toprule\n" + head + "\n\\midrule\n" + "\n".join(rows[lang]) +
                "\n\\bottomrule\n\\end{tabular}")
        cap_en = ("Share of the variance of the oracle ratio $r^*$ explained ($R^2$) on the calibration panel. "
                  "Graph identity, perturbation level and their additive combination are in-sample ANOVA shares. "
                  "Starred columns are out of sample (grouped 8-fold CV by graph): boosted trees on the hidden "
                  "generator parameters plus the perturbation level; ridge regression on every observable global "
                  "statistic; and the clip-linear rule on $\\bar P^{(M)}_{\\mathrm{esc}}$. Negative values mean worse "
                  "than predicting the mean.")
        cap_gr = ("Ποσοστό της διασποράς του βέλτιστου λόγου $r^*$ που εξηγείται ($R^2$) στο σύνολο βαθμονόμησης. "
                  "Η ταυτότητα του γράφου, το επίπεδο διαταραχής και ο προσθετικός τους συνδυασμός είναι μερίδια "
                  "ANOVA εντός δείγματος. Οι στήλες με αστερίσκο είναι εκτός δείγματος (ομαδοποιημένη 8-πλή CV ανά "
                  "γράφο): ενισχυμένα δέντρα στις κρυφές παραμέτρους της γεννήτριας μαζί με το επίπεδο διαταραχής· "
                  "παλινδρόμηση ridge σε όλα τα παρατηρήσιμα καθολικά στατιστικά· και ο αποκομμένος γραμμικός κανόνας στο "
                  "$\\bar P^{(M)}_{\\mathrm{esc}}$. Αρνητικές τιμές σημαίνουν χειρότερη πρόβλεψη από τη μέση τιμή.")
        write(os.path.join(OUT, lang, "tab_stage1_dec.tex"), table_env("tab:stage1dec", cap_en, cap_gr, body, lang, "\\footnotesize"))
    d = dec["NCD"]
    macro("SOneDecBase", f"{d['r2_base']:.2f}")
    macro("SOneDecLevel", f"{d['r2_level']:.2f}")
    macro("SOneDecBL", f"{d['r2_base_level']:.2f}")
    macro("SOneDecHidden", f"{d['r2_hidden_cv']:.2f}")
    macro("SOneDecObs", f"{d['r2_observable_ridge_cv']:.2f}")
    macro("SOneDecP", f"{d['r2_pesc_cv']:.2f}")
    cb = d["corr_basemean_with_param"]
    macro("SOneCorrICC", sgn3(cb["icc"]).replace("$", ""))
    macro("SOneCorrFit", sgn3(cb["fit_sigma"]).replace("$", ""))
    cbp = dec["POP"]["corr_basemean_with_param"]
    macro("SOneCorrICCPop", sgn3(cbp["icc"]))
    macro("SOneCorrFitPop", sgn3(cbp["fit_sigma"]))
    # Stage 1 rule constants
    macro("SOneConstR", f3(fits["const"]["params"][0]))
    # real per graph (Stage 1)
    rb = pd.read_csv(os.path.join(R1, "real_by_graph.csv"))
    q = rb[(rb.truth == "NCD") & (rb.rule == "cliplin_TAU") & (rb.fitted_on == "NCD")].set_index("graph")
    for g, nm in (("citeseer", "Citeseer"), ("cnr2000", "Cnr"), ("cora_full", "Corafull"), ("cora_ml", "Coraml"),
                  ("wikics", "Wikics")):
        macro(f"SOneReal{nm}", sgn4(q.loc[g, "tau"] - q.loc[g, "tau_canonical"]))


def stage1_truth_table():
    """Appendix: Stage 1 under the four Stage 1 truths (constant, tau-optimised rule, canonical)."""
    rows = []
    dag = lambda pv_: "$^{\\dagger}$" if (pv_ == pv_ and pv_ < 0.01) else ""
    for t in ["NCD", "PR", "POP", "SELF"]:
        s = pd.read_csv(os.path.join(R1, f"summary_{t}.csv"))
        fits = {d["name"]: d for d in json.load(open(os.path.join(R1, f"fits_{t}.json")))}
        pr = fits["cliplin_TAU"]["params"]
        c_test = s[(s.rule == "const") & (s.panel == "test")].iloc[0]
        cells = [f"${fits['const']['params'][0]:.2f}$", f"$\\mathrm{{clip}}({pr[0]:.3f}{pr[1]:+.3f}\\,\\bar P)$",
                 f4(c_test["tau_canonical"]), sgn4(c_test["vs_can_mean"]) + dag(c_test["vs_can_p"])]
        for pnl in ["test", "scale", "topo", "real"]:
            q = s[(s.rule == "cliplin_TAU") & (s.panel == pnl)]
            cells.append("--" if q.empty else sgn4(q.iloc[0]["vs_const_mean"]) + dag(q.iloc[0]["vs_const_p"]))
        rows.append(f"{t} & " + " & ".join(cells) + " \\\\")
    for lang in ("en", "gr"):
        if lang == "en":
            head = ("& & & \\multicolumn{2}{c}{\\texttt{test}} & \\multicolumn{4}{c}{rule $-$ constant} \\\\\n"
                    "\\cmidrule(lr){4-5}\\cmidrule(l){6-9}\n"
                    "truth & const $r$ & $\\tau$-opt.\\ rule & canonical $\\tau$ & const $-$ can. & \\texttt{test} & "
                    "\\texttt{scale} & \\texttt{topo} & \\texttt{real} \\\\")
            cap = ("Stage~1 under the four Stage~1 truths. Const $r$: the best constant on the calibration panel; "
                   "$\\tau$-opt.\\ rule: the $\\tau$-optimised clip-linear rule on $\\bar P=\\bar P^{(M)}_{\\mathrm{esc}}$. "
                   "Canonical $\\tau$: mean Kendall $\\tau$ of the canonical $r=0.118$ on \\texttt{test}; const $-$ can.: "
                   "gain of the constant over it. The last four columns give the mean $\\tau$ difference between the "
                   "rule and the constant per panel. $^{\\dagger}$: Wilcoxon signed-rank $p<0.01$ over graphs. POP exists "
                   "only for the static generator. Under SELF both the constant and the rule sit at the upper bound "
                   "$r=0.99$.")
        else:
            head = ("& & & \\multicolumn{2}{c}{\\texttt{test}} & \\multicolumn{4}{c}{κανόνας $-$ σταθερά} \\\\\n"
                    "\\cmidrule(lr){4-5}\\cmidrule(l){6-9}\n"
                    "αλήθεια & σταθ.\\ $r$ & κανόνας βέλτ.\\ $\\tau$ & κανονικό $\\tau$ & σταθ.\\ $-$ καν. & \\texttt{test} & "
                    "\\texttt{scale} & \\texttt{topo} & \\texttt{real} \\\\")
            cap = ("Το Στάδιο~1 υπό τις τέσσερις αλήθειες του Σταδίου~1. Σταθ.\\ $r$: η καλύτερη σταθερά στο σύνολο "
                   "βαθμονόμησης· κανόνας βέλτ.\\ $\\tau$: ο αποκομμένος γραμμικός κανόνας στο $\\bar P=\\bar P^{(M)}_{\\mathrm{esc}}$, "
                   "βέλτιστος ως προς $\\tau$. Κανονικό $\\tau$: μέσο $\\tau$ του Kendall του κανονικού $r=0.118$ στο "
                   "\\texttt{test}· σταθ.\\ $-$ καν.: το κέρδος της σταθεράς έναντι αυτού. Οι τέσσερις τελευταίες στήλες "
                   "δίνουν τη μέση διαφορά $\\tau$ μεταξύ κανόνα και σταθεράς ανά σύνολο. $^{\\dagger}$: $p<0.01$ στον "
                   "έλεγχο προσημασμένων τάξεων του Wilcoxon ανά γράφο. Η αλήθεια POP υπάρχει μόνο για τη στατική "
                   "γεννήτρια. Υπό την SELF τόσο η σταθερά όσο και ο κανόνας βρίσκονται στο άνω όριο $r=0.99$.")
        body = ("\\begin{tabular}{@{}lllrrrrrr@{}}\n\\toprule\n" + head + "\n\\midrule\n" + "\n".join(rows) +
                "\n\\bottomrule\n\\end{tabular}")
        write(os.path.join(OUT, lang, "tab_stage1truth.tex"),
              table_env("tab:stage1truth", cap, cap, body, lang, "\\footnotesize\\setlength{\\tabcolsep}{4pt}", place="!htbp"))


def bound_numbers():
    """Numbers of the displacement-bound check (bound_check.py)."""
    b = pd.read_csv(os.path.join(R2, "bound_check.csv"))
    macro("BoundCases", f"{len(b):,}".replace(",", "\\,"))
    macro("BoundGraphs", str(b.graph.nunique()))
    macro("BoundMax", f3(b.ratio.max()))
    lo = b[b.alpha <= 0.051]
    macro("BoundLoMin", f"{lo.ratio.min():.2f}")
    macro("BoundLoMax", f"{lo.ratio.max():.2f}")
    macro("BoundMedEightyFive", f"{b[np.isclose(b.alpha, 0.85)].ratio.median():.2f}")


# ---------------------------------------------------------------------------
# 3. Stage 2
# ---------------------------------------------------------------------------
def stage2():
    s = pd.read_csv(os.path.join(R2, "summary.csv"))
    fits = json.load(open(os.path.join(R2, "fits.json")))
    diag = json.load(open(os.path.join(R2, "diagnostics.json")))
    dec = json.load(open(os.path.join(R2, "decomposition.json")))
    inst = pd.read_csv(os.path.join(R2, "instances.csv"))
    macro("MaxSigmaDev", "3.3\\times10^{-16}" if diag["checks"]["max_abs_mean_sigma_minus_1"] < 4e-16
          else f"{diag['checks']['max_abs_mean_sigma_minus_1']:.1e}")
    cfg = {(c["r_setting"], c["truth"], c["feature"]): c for c in diag["configs"]}
    prim = cfg[("rule", "NCD", "h_rule")]
    wb = prim["within_between"]
    macro("STwoAstarMean", f3(prim["alpha_star_calib"]["mean"]))
    macro("STwoAstarSd", f3(prim["alpha_star_calib"]["sd"]))
    macro("STwoPooled", sgn3(wb["pooled"]))
    macro("STwoWithin", sgn3(wb["within"]))
    macro("STwoBetween", sgn3(wb["between"]))
    wbv = cfg[("rule", "NCD", "hv_rule")]["within_between"]
    macro("STwoWithinHv", sgn3(wbv["within"]))
    macro("STwoBetweenHv", sgn3(wbv["between"]))
    # mechanism by experiment
    c = inst[(inst.panel == "calib") & ~((inst.exp == "A") & (inst.level == 1.0))]

    def within(d, x, y):
        g = d.groupby("base")
        return float(np.corrcoef(d[x] - g[x].transform("mean"), d[y] - g[y].transform("mean"))[0, 1])
    for e in ("A", "B"):
        d = c[c.exp == e]
        macro(f"STwoHkeep{e}", sgn3(within(d, "h_rule", "keep_frac")))
        macro(f"STwoHastar{e}", sgn3(within(d, "h_rule", "astar_rule_NCD")))
        macro(f"STwoKeepAstar{e}", sgn3(within(d, "keep_frac", "astar_rule_NCD")))
        macro(f"STwoHdang{e}", sgn3(within(d, "h_rule", "dangling_frac")))
        macro(f"STwoHzin{e}", sgn3(within(d, "h_rule", "zero_in_frac")))
        hl = d.groupby("level")["h_rule"].mean()
        al = d.groupby("level")["astar_rule_NCD"].mean()
        lo, hi = (0.2, 0.9) if e == "A" else (0.6, 0.1)
        macro(f"STwoHmild{e}", f3(hl[hi]))
        macro(f"STwoHsevere{e}", f3(hl[lo]))
        macro(f"STwoAmild{e}", f3(al[hi]))
        macro(f"STwoAsevere{e}", f3(al[lo]))
    a10 = inst[(inst.panel == "calib") & (inst.exp == "A") & (inst.level == 1.0)]
    macro("STwoHfull", f3(a10["h_rule"].mean()))
    # candidate g table (primary) incl. h_v rows
    names = [("cliplin_TAU", "clip-linear, $\\tau$-opt.\\ (locked)", "αποκομμένος γραμμικός, βέλτ.\\ $\\tau$ (κλειδ.)"),
             ("cliplin_LS", "clip-linear, LS", "αποκομμένος γραμμικός, ΕΤ"),
             ("isotonic", "isotonic", "ισοτονικός"),
             ("budget_TAU", "budget $c/(c+x)$, $\\tau$-opt.", "προϋπολ.\\ $c/(c+x)$, βέλτ.\\ $\\tau$"),
             ("gbm[observables]", "boosted trees, all observables", "ενισχυμένα δέντρα, όλα τα παρατηρ.")]
    panels = ["test", "scale", "topo", "real"]
    rows = {"en": [], "gr": []}
    sub = s[(s.r_setting == "rule") & (s.truth == "NCD") & (s.feature == "h_rule")]
    fdh = {d["name"]: d for d in fits["rule|NCD|h_rule"]}
    # block 1: absolute tau of the static setting, the locked constant and the oracle
    q = {p: sub[(sub.rule == "const") & (sub.panel == p)].iloc[0] for p in panels + ["calib"]}
    top = {"en": [], "gr": []}
    for key, en, gr in (("085", "static $\\alpha=0.85$", "στατικό $\\alpha=0.85$"),
                        ("const", f"constant $\\alpha={fdh['const']['params'][0]:.3f}$ (locked)",
                         f"σταθερό $\\alpha={fdh['const']['params'][0]:.3f}$ (κλειδωμένο)"),
                        ("diff", "\\quad difference", "\\quad διαφορά"),
                        ("oracle", "oracle $\\alpha^*$", "μαντείο $\\alpha^*$")):
        if key == "085":
            cells = [f4(q[p]["tau_085"]) for p in panels]
            cv = "--"
        elif key == "const":
            cells = [f4(q[p]["tau"]) for p in panels]
            cv = f4(q["calib"]["cv_tau"])
        elif key == "diff":
            cells = [sgn4(q[p]["d_vs_085"]) + ("$^{\\dagger}$" if q[p]["p_vs_085"] < 0.01 else "") for p in panels]
            cv = ""
        else:
            cells = [f4(q[p]["tau_oracle"]) for p in panels]
            cv = "--"
        for lang, lab in (("en", en), ("gr", gr)):
            top[lang].append(f"{lab} & & {cv} & " + " & ".join(cells) + " \\\\")
    # block 2: differences of every candidate against the locked constant
    for feat, featlab in (("h_rule", "h"), ("hv_rule", "h_v")):
        fd = {d["name"]: d for d in fits[f"rule|NCD|{feat}"]}
        subf = s[(s.r_setting == "rule") & (s.truth == "NCD") & (s.feature == feat)]
        for lang in ("en", "gr"):
            hdr = (f"\\multicolumn{{7}}{{@{{}}l}}{{\\emph{{rules on ${featlab}$: difference in mean $\\tau$ against the "
                   f"locked constant}}}} \\\\" if lang == "en" else
                   f"\\multicolumn{{7}}{{@{{}}l}}{{\\emph{{κανόνες στο ${featlab}$: διαφορά μέσου $\\tau$ έναντι της "
                   f"κλειδωμένης σταθεράς}}}} \\\\")
            rows[lang].append("\\midrule\n" + hdr)
        for name, en, gr in names:
            if name not in fd or subf[subf.rule == name].empty:
                continue
            if feat == "hv_rule" and name == "gbm[observables]":
                continue
            cal = subf[(subf.rule == name) & (subf.panel == "calib")].iloc[0]
            cells = []
            for p in panels:
                qq = subf[(subf.rule == name) & (subf.panel == p)].iloc[0]
                star = "" if np.isnan(qq["p_vs_const"]) else ("$^{\\dagger}$" if qq["p_vs_const"] < 0.01 else "")
                cells.append(f"{sgn4(qq['d_vs_const'])}{star}")
            if name in ("cliplin_TAU", "cliplin_LS"):
                pr = fd[name]["params"]
                form = f"$\\mathrm{{clip}}({pr[0]:.3f}{pr[1]:+.3f}\\,{featlab})$"
            elif name == "budget_TAU":
                form = f"$c={np.exp(fd[name]['params'][0]):.2f}$"
            else:
                form = "--"
            cv = "--" if np.isnan(cal["cv_tau"]) else f4(cal["cv_tau"])
            for lang, lab in (("en", en), ("gr", gr)):
                rows[lang].append(f"{lab} & {form} & {cv} & " + " & ".join(cells) + " \\\\")
    head_en = "rule & formula & CV $\\tau$ & \\texttt{test} & \\texttt{scale} & \\texttt{topo} & \\texttt{real} \\\\"
    head_gr = "κανόνας & τύπος & CV $\\tau$ & \\texttt{test} & \\texttt{scale} & \\texttt{topo} & \\texttt{real} \\\\"
    for lang, head in (("en", head_en), ("gr", head_gr)):
        sub1 = ("\\multicolumn{7}{@{}l}{\\emph{mean Kendall $\\tau$}} \\\\"
                if lang == "en" else
                "\\multicolumn{7}{@{}l}{\\emph{μέσο $\\tau$ του Kendall}} \\\\")
        body = ("\\begin{tabular}{@{}llrrrrr@{}}\n\\toprule\n" + head + "\n\\midrule\n" + sub1 + "\n" +
                "\n".join(top[lang]) + "\n" + "\n".join(rows[lang]) + "\n\\bottomrule\n\\end{tabular}")
        cap_en = ("Stage~2 candidate rules $\\alpha=g(h)$ and $\\alpha=g(h_v)$ with $r$ locked to the Stage~1 rule, "
                  "truth NCD. Top block: mean Kendall $\\tau$ of the static damping, the locked constant, their difference "
                  "and the oracle. "
                  "Lower blocks: difference in mean $\\tau$ of each fitted rule against the locked constant. CV $\\tau$: "
                  "grouped 8-fold cross-validation on the calibration graphs. $^{\\dagger}$: Wilcoxon signed-rank "
                  "$p<0.01$ over graphs. Every rule is clipped to $[0.40,0.99]$; LS: least squares on the oracle values; "
                  "$\\tau$-opt.: fitted by maximising the mean calibration $\\tau$.")
        cap_gr = ("Υποψήφιοι κανόνες του Σταδίου~2, $\\alpha=g(h)$ και $\\alpha=g(h_v)$, με το $r$ κλειδωμένο στον "
                  "κανόνα του Σταδίου~1 και αλήθεια NCD. Άνω τμήμα: μέσο $\\tau$ του Kendall για τη στατική απόσβεση, "
                  "την κλειδωμένη σταθερά, τη διαφορά τους και το μαντείο. Κάτω τμήματα: διαφορά του μέσου $\\tau$ κάθε προσαρμοσμένου "
                  "κανόνα έναντι της κλειδωμένης σταθεράς. CV $\\tau$: ομαδοποιημένη 8-πλή διασταυρωμένη επικύρωση "
                  "στους γράφους βαθμονόμησης. $^{\\dagger}$: $p<0.01$ στον έλεγχο προσημασμένων τάξεων του Wilcoxon "
                  "ανά γράφο. Όλοι οι κανόνες αποκόπτονται στο $[0.40,0.99]$· ΕΤ: ελάχιστα τετράγωνα στις βέλτιστες "
                  "τιμές· βέλτ.\\ $\\tau$: προσαρμογή με μεγιστοποίηση του μέσου $\\tau$ βαθμονόμησης.")
        write(os.path.join(OUT, lang, "tab_stage2.tex"), table_env("tab:stage2", cap_en, cap_gr, body, lang, "\\footnotesize", star=True))
    get = lambda rule, p, col, feat="h_rule": sub_all(s, feat)[(sub_all(s, feat).rule == rule) & (sub_all(s, feat).panel == p)].iloc[0][col]
    macro("STwoConstAlpha", f3({d["name"]: d for d in fits["rule|NCD|h_rule"]}["const"]["params"][0]))
    lsp = {d["name"]: d for d in fits["rule|NCD|h_rule"]}["cliplin_LS"]["params"]
    macro("STwoLSa", f3(lsp[0]))
    macro("STwoLSb", f"{lsp[1]:.3f}".replace("-", "$-$"))
    lvp = {d["name"]: d for d in fits["rule|NCD|hv_rule"]}["cliplin_LS"]["params"]
    macro("STwoLSva", f3(lvp[0]))
    macro("STwoLSvb", f"{lvp[1]:.3f}".replace("-", "$-$"))
    macro("STwoBudgetC", f"{np.exp({d['name']: d for d in fits['rule|NCD|h_rule']}['budget_TAU']['params'][0]):.2f}")
    for p, nm in (("test", "Test"), ("scale", "Scale"), ("topo", "Topo"), ("real", "Real")):
        macro(f"STwo{nm}Const", f4(get("const", p, "tau")))
        macro(f"STwo{nm}ConstVs", sgn4(get("const", p, "d_vs_085")))
        macro(f"STwo{nm}ConstP", pv(get("const", p, "p_vs_085")))
        macro(f"STwo{nm}ConstGap", f"{100 * get('const', p, 'gap_closed_vs_085'):.0f}")
        macro(f"STwo{nm}LSvsConst", sgn4(get("cliplin_LS", p, "d_vs_const")))
        macro(f"STwo{nm}LSvsConstP", pv(get("cliplin_LS", p, "p_vs_const")))
        macro(f"STwo{nm}IsoVsConst", sgn4(get("isotonic", p, "d_vs_const")))
        macro(f"STwo{nm}IsoVsConstP", pv(get("isotonic", p, "p_vs_const")))
        macro(f"STwo{nm}BudVsConst", sgn4(get("budget_TAU", p, "d_vs_const")))
        macro(f"STwo{nm}BudVsConstP", pv(get("budget_TAU", p, "p_vs_const")))
        macro(f"STwo{nm}GbmVsConst", sgn4(get("gbm[observables]", p, "d_vs_const")))
        macro(f"STwo{nm}GbmVsConstP", pv(get("gbm[observables]", p, "p_vs_const")))
        macro(f"STwo{nm}LSvVsConst", sgn4(get("cliplin_LS", p, "d_vs_const", "hv_rule")))
        macro(f"STwo{nm}LSvVsConstP", pv(get("cliplin_LS", p, "p_vs_const", "hv_rule")))
    # anchoring table
    own = {"NCD": "0.85", "NCD95": "0.95", "TOT": "0.50$^{a}$", "POP": "0.85$^{b}$", "Q": "--"}
    pairs = [("rule", "NCD"), ("ng13", "NCD95"), ("rule", "TOT"), ("popLS", "POP"), ("rule", "Q")]
    rl = {"en": {"rule": "Stage~1", "ng13": "0.105", "popLS": "POP rule", "can": "0.118"},
          "gr": {"rule": "Στάδιο~1", "ng13": "0.105", "popLS": "κανόνας POP", "can": "0.118"}}
    rows = {"en": [], "gr": []}
    for r, t in pairs:
        d = dec[f"{r}|{t}"]
        info = cfg[(r, t, f"h_{r}")]
        constv = {x["name"]: x for x in fits[f"{r}|{t}|h_{r}"]}["const"]["params"][0]
        cells = f"{own[t]} & " + (f"{d['anchor_A10']['mean']:.3f} & {d['mean']:.3f} & {d['mean'] - d['anchor_A10']['mean']:+.3f} & "
                                  f"{constv:.2f} & {info['within_between']['within']:+.2f} & "
                                  f"{info['within_between']['between']:+.2f}").replace("-", "$-$").replace("+", "$+$")
        for lang in ("en", "gr"):
            rows[lang].append(f"{t} & {rl[lang][r]} & {cells} \\\\")
    head_en = ("truth & $r$ & own $\\alpha$ & $\\alpha^*$ (A1.0) & mean $\\alpha^*$ & discount & const $\\alpha$ & "
               "within & between \\\\")
    head_gr = ("αλήθεια & $r$ & δικό $\\alpha$ & $\\alpha^*$ (A1.0) & μέσο $\\alpha^*$ & έκπτωση & σταθ.\\ $\\alpha$ & "
               "εντός & μεταξύ \\\\")
    for lang, head in (("en", head_en), ("gr", head_gr)):
        body = ("\\begin{tabular}{@{}llrrrrrrr@{}}\n\\toprule\n" + head + "\n\\midrule\n" + "\n".join(rows[lang]) +
                "\n\\bottomrule\n\\end{tabular}")
        cap_en = ("Oracle damping under the five truths (calibration panel). Own $\\alpha$: the damping that defines the "
                  "truth ($^{a}$TotalRank: mean of the Beta(1,1) prior; $^{b}$POP is the PageRank of the expected walk "
                  "at $0.85$). $\\alpha^*$ (A1.0): mean oracle on the unperturbed graphs. Mean $\\alpha^*$: mean over the "
                  "perturbed instances; discount: their difference. Const $\\alpha$: the best constant on the "
                  "calibration panel. Within/between: correlation of $h$ with $\\alpha^*$ inside graphs and between "
                  "graph means.")
        cap_gr = ("Βέλτιστη απόσβεση υπό τις πέντε αλήθειες (σύνολο βαθμονόμησης). Δικό $\\alpha$: η απόσβεση που "
                  "ορίζει την αλήθεια ($^{a}$TotalRank: μέση τιμή της εκ των προτέρων Beta(1,1)· $^{b}$το POP είναι το "
                  "PageRank του αναμενόμενου περιπάτου στο $0.85$). $\\alpha^*$ (A1.0): μέσο μαντείο στους αδιατάρακτους "
                  "γράφους. Μέσο $\\alpha^*$: μέση τιμή στα διαταραγμένα στιγμιότυπα· έκπτωση: η διαφορά τους. Σταθ.\\ "
                  "$\\alpha$: η καλύτερη σταθερά στο σύνολο βαθμονόμησης. Εντός/μεταξύ: συσχέτιση του $h$ με το "
                  "$\\alpha^*$ εντός γράφων και μεταξύ των μέσων τιμών των γράφων.")
        write(os.path.join(OUT, lang, "tab_anchor.tex"), table_env("tab:anchor", cap_en, cap_gr, body, lang, "\\footnotesize"))
    # Stage 2 variance decomposition table
    dpairs = [("rule", "NCD"), ("can", "NCD"), ("ng13", "NCD95"), ("rule", "TOT"), ("popLS", "POP"), ("rule", "Q")]
    rows = {"en": [], "gr": []}
    for r, t in dpairs:
        d = dec[f"{r}|{t}"]
        cells = " & ".join(f"{d[k]:.2f}".replace("-", "$-$") for k in
                           ["r2_base", "r2_level", "r2_base_level", "r2_hidden_cv", "r2_observable_ridge_cv", "r2_h_cv"])
        for lang in ("en", "gr"):
            rows[lang].append(f"{t} & {rl[lang][r]} & {cells} \\\\")
    head_en = ("truth & $r$ & graph & level & graph+level & hidden+level$^{*}$ & all observables$^{*}$ & "
               "$h$ alone$^{*}$ \\\\")
    head_gr = ("αλήθεια & $r$ & γράφος & επίπεδο & γράφος+επίπεδο & κρυφές+επίπεδο$^{*}$ & όλα τα παρατηρ.$^{*}$ & "
               "μόνο $h$$^{*}$ \\\\")
    for lang, head in (("en", head_en), ("gr", head_gr)):
        body = ("\\begin{tabular}{@{}llrrrrrr@{}}\n\\toprule\n" + head + "\n\\midrule\n" + "\n".join(rows[lang]) +
                "\n\\bottomrule\n\\end{tabular}")
        cap_en = ("Share of the variance of the oracle damping $\\alpha^*$ explained ($R^2$) on the calibration panel, "
                  "for each truth and $r$-setting. Graph identity, perturbation level and their additive combination "
                  "are in-sample ANOVA shares. Starred columns are out of sample (grouped 8-fold CV by graph): boosted "
                  "trees on the hidden generator parameters plus the perturbation level; ridge regression on every "
                  "observable global statistic, including $h$ and $h_v$; and the clip-linear rule on $h$ alone. "
                  "Negative values mean worse than predicting the mean.")
        cap_gr = ("Ποσοστό της διασποράς της βέλτιστης απόσβεσης $\\alpha^*$ που εξηγείται ($R^2$) στο σύνολο "
                  "βαθμονόμησης, για κάθε αλήθεια και ρύθμιση του $r$. Η ταυτότητα του γράφου, το επίπεδο διαταραχής "
                  "και ο προσθετικός τους συνδυασμός είναι μερίδια ANOVA εντός δείγματος. Οι στήλες με αστερίσκο είναι "
                  "εκτός δείγματος (ομαδοποιημένη 8-πλή CV ανά γράφο): ενισχυμένα δέντρα στις κρυφές παραμέτρους της "
                  "γεννήτριας μαζί με το επίπεδο διαταραχής· παλινδρόμηση ridge σε όλα τα παρατηρήσιμα καθολικά "
                  "στατιστικά, μαζί με τα $h$ και $h_v$· και ο αποκομμένος γραμμικός κανόνας μόνο στο $h$. Αρνητικές τιμές "
                  "σημαίνουν χειρότερη πρόβλεψη από τη μέση τιμή.")
        write(os.path.join(OUT, lang, "tab_stage2_dec.tex"), table_env("tab:stage2dec", cap_en, cap_gr, body, lang, "\\footnotesize"))
    d = dec["rule|NCD"]
    macro("STwoDecBase", f"{d['r2_base']:.2f}")
    macro("STwoDecLevel", f"{d['r2_level']:.2f}")
    macro("STwoDecBL", f"{d['r2_base_level']:.2f}")
    macro("STwoDecHidden", f"{d['r2_hidden_cv']:.2f}")
    macro("STwoDecObs", f"{d['r2_observable_ridge_cv']:.2f}")
    macro("STwoDecH", f"{d['r2_h_cv']:.2f}".replace("-", "$-$"))
    hmax = max(dec[k]["r2_h_cv"] for k in dec)
    macro("STwoDecHmax", f"{hmax:.2f}")
    for (r, t), nm in zip(pairs, ("Ncd", "Ncdnf", "Tot", "Pop", "Q")):
        macro(f"Anchor{nm}", f3(dec[f"{r}|{t}"]["anchor_A10"]["mean"]))
        macro(f"Mean{nm}", f3(dec[f"{r}|{t}"]["mean"]))
    # RAPr dose-response
    ra = pd.read_csv(os.path.join(R2, "rapr_anchor.csv"))
    g = ra.groupby("prior").agg(m=("prior_mean", "first"), a=("alpha_star", "mean"), sd=("alpha_star", "std"))
    g = g.sort_values(["m", "a"])
    rows = []
    for k, rr in g.iterrows():
        rows.append(f"{k} & {rr['m']:.2f} & {rr['a']:.3f} & {rr['sd']:.3f} & {sgn3(rr['a'] - rr['m'])} \\\\")
    nb = ra["base"].nunique()
    macro("RaprGraphs", str(nb))
    slope, icpt = np.polyfit(g["m"], g["a"], 1)
    macro("RaprSlope", f3(slope))
    macro("RaprIcpt", f"{icpt:.3f}".replace("-", "$-$"))
    for k, nm in (("Beta(3,7)", "Lo"), ("Beta(5,5)", "Mid"), ("Beta(1,1)", "Tot"), ("Beta(7,3)", "Hi"), ("Beta(8.5,1.5)", "Top")):
        macro(f"Rapr{nm}", f3(g.loc[k, "a"]))
    for lang in ("en", "gr"):
        head = ("prior & prior mean & mean $\\alpha^*$ & s.d. & $\\alpha^*-$mean \\\\" if lang == "en" else
                "εκ των προτέρων & μέση τιμή & μέσο $\\alpha^*$ & τ.α. & $\\alpha^*-$μέση τιμή \\\\")
        body = ("\\begin{tabular}{@{}lrrrr@{}}\n\\toprule\n" + head + "\n\\midrule\n" + "\n".join(rows) +
                "\n\\bottomrule\n\\end{tabular}")
        cap_en = (f"Dose-response test of anchoring: oracle damping $\\alpha^*$ against Random-$\\alpha$ NCDawareRank "
                  f"references $\\mathbb{{E}}[\\pi(A)]$ with $A\\sim\\mathrm{{Beta}}(a,b)$, on {nb} calibration graphs "
                  "(perturbed instances only, $r$ = Stage~1 rule, grid $0.05$--$0.99$). Beta(1,1) is TotalRank.")
        cap_gr = (f"Έλεγχος δόσης--απόκρισης της αγκύρωσης: βέλτιστη απόσβεση $\\alpha^*$ έναντι αναφορών "
                  f"Random-$\\alpha$ NCDawareRank $\\mathbb{{E}}[\\pi(A)]$ με $A\\sim\\mathrm{{Beta}}(a,b)$, σε {nb} "
                  "γράφους βαθμονόμησης (μόνο διαταραγμένα στιγμιότυπα, $r$ = κανόνας Σταδίου~1, πλέγμα "
                  "$0.05$--$0.99$). Η Beta(1,1) είναι το TotalRank.")
        write(os.path.join(OUT, lang, "tab_rapr.tex"), table_env("tab:rapr", cap_en, cap_gr, body, lang, "\\small"))
    # upper check for Q
    up = pd.read_csv(os.path.join(R2, "upper_check.csv"))
    macro("QupNine", f4(up["tau_0.99"].mean()))
    macro("QupNineFive", f4(up["tau_0.995"].mean()))
    macro("QupNineNine", f4(up["tau_0.999"].mean()))
    macro("QupGraphs", str(up["base"].nunique()))
    # sensitivity truths: h-rule vs constant
    conf = [("rule", "TOT", ("TOT", "TOT")), ("rule", "Q", ("Q", "Q")),
            ("popLS", "POP", ("POP ($r$: LS rule)", "POP ($r$: ΕΤ)")),
            ("popTAU", "POP", ("POP ($r$: $\\tau$-opt.\\ rule)", "POP ($r$: βέλτ.\\ $\\tau$)")),
            ("ng13", "NCD95", ("NCD95", "NCD95"))]
    rows = {"en": [], "gr": []}
    for r, t, labs in conf:
        fd = {d["name"]: d for d in fits[f"{r}|{t}|h_{r}"]}
        sub = s[(s.r_setting == r) & (s.truth == t) & (s.feature == f"h_{r}")]
        pr = fd["cliplin_TAU"]["params"]
        cells = []
        for p in ["test", "scale", "topo", "real"]:
            q = sub[(sub.rule == "cliplin_TAU") & (sub.panel == p)]
            if q.empty:
                cells.append("--")
                continue
            q = q.iloc[0]
            pstr = pv(q["p_vs_const"]) if not np.isnan(q["p_vs_const"]) else "--"
            cells.append(f"{sgn4(q['d_vs_const'])} ({pstr})")
        for lang, lab in zip(("en", "gr"), labs):
            rows[lang].append(f"{lab} & ${fd['const']['params'][0]:.2f}$ & $\\mathrm{{clip}}({pr[0]:.3f}{pr[1]:+.3f}\\,h)$ & " +
                              " & ".join(cells) + " \\\\")
    for lang in ("en", "gr"):
        head = ("truth & const $\\alpha$ & $\\tau$-optimised $g(h)$ & \\texttt{test} & \\texttt{scale} & "
                "\\texttt{topo} & \\texttt{real} \\\\") if lang == "en" else (
            "αλήθεια & σταθ.\\ $\\alpha$ & $g(h)$ βέλτ.\\ $\\tau$ & \\texttt{test} & \\texttt{scale} & "
            "\\texttt{topo} & \\texttt{real} \\\\")
        body = ("\\begin{tabular}{@{}lllllll@{}}\n\\toprule\n" + head + "\n\\midrule\n" + "\n".join(rows[lang]) +
                "\n\\bottomrule\n\\end{tabular}")
        cap_en = ("Sensitivity truths, with $g$ refitted for each truth: mean $\\tau$ difference of the "
                  "$\\tau$-optimised clip-linear rule on $h$ against the truth's best constant $\\alpha$ (Wilcoxon $p$ "
                  "over graphs). A dash means the panel has no such truth, or the rule equals the constant.")
        cap_gr = ("Αλήθειες ευαισθησίας, με το $g$ επαναπροσαρμοσμένο ανά αλήθεια: μέση διαφορά $\\tau$ του αποκομμένου "
                  "γραμμικού κανόνα στο $h$, βέλτιστου ως προς $\\tau$, έναντι του καλύτερου σταθερού $\\alpha$ της "
                  "αλήθειας (έλεγχος Wilcoxon ανά γράφο). Παύλα σημαίνει ότι το σύνολο δεν έχει αυτή την αλήθεια ή ότι "
                  "ο κανόνας συμπίπτει με τη σταθερά.")
        write(os.path.join(OUT, lang, "tab_sens.tex"), table_env("tab:sens", cap_en, cap_gr, body, lang, "\\footnotesize\\setlength{\\tabcolsep}{4pt}", place="!htbp"))
    tot = s[(s.r_setting == "rule") & (s.truth == "TOT") & (s.feature == "h_rule") & (s.rule == "cliplin_TAU")].set_index("panel")
    macro("TotHTest", sgn4(tot.loc["test", "d_vs_const"]))
    macro("TotHTestP", pv(tot.loc["test", "p_vs_const"]))
    macro("TotHTopo", sgn4(tot.loc["topo", "d_vs_const"]))
    macro("TotHTopoP", pv(tot.loc["topo", "p_vs_const"]))
    macro("TotHReal", sgn4(tot.loc["real", "d_vs_const"]))


def sub_all(s, feat):
    return s[(s.r_setting == "rule") & (s.truth == "NCD") & (s.feature == feat)]


# ---------------------------------------------------------------------------
# 3b. Section 8.2: non-monotone and dangling-deconfounded rules (stage2_nonmonotone_search.py)
# ---------------------------------------------------------------------------
def e4(x):
    """A tau difference in units of 1e-4, one decimal."""
    if x is None or not np.isfinite(x):
        return "--"
    v = 1e4 * x
    if abs(v) < 1e-9:
        return "0.0"
    return f"{v:+.1f}".replace("-", "$-$").replace("+", "$+$")


NONMONO_ROWS = [
    ("non-monotone rules on $h$ and $h_v$", "μη μονότονοι κανόνες στα $h$ και $h_v$", [
        ("tent_LS[h]", "asymmetric tent, LS", "ασύμμετρη τέντα, ΕΤ"),
        ("tent_TAU[h]", "asymmetric tent", "ασύμμετρη τέντα"),
        ("symtent_TAU[h]", "symmetric tent", "συμμετρική τέντα"),
        ("quad_TAU[h]", "concave quadratic", "κοίλη τετραγωνική"),
        ("gauss_TAU[h]", "Gaussian bump", "γκαουσιανό εξόγκωμα"),
        ("step3_TAU[h]", "three-region step", "κλιμακωτή 3 περιοχών"),
        ("spline2_TAU[h]", "two-knot linear spline", "γραμμική spline 2 κόμβων"),
        ("tent_TAU[h_v]", "asymmetric tent on $h_v$", "ασύμμετρη τέντα στο $h_v$")]),
    ("dangling-deconfounded and severity statistics", "στατιστικά χωρίς την επίδραση των κρεμάμενων και στατιστικά έντασης", [
        ("tent_TAU[h_raw]", "tent on $h_{\\mathrm{raw}}$", "τέντα στο $h_{\\mathrm{raw}}$"),
        ("tent_TAU[h_rawrel]", "tent on $h_{\\mathrm{raw}}^{\\mathrm{rel}}$", "τέντα στο $h_{\\mathrm{raw}}^{\\mathrm{rel}}$"),
        ("tent_TAU[hH_raw]", "tent on $h_{\\mathrm{raw}}^{H}$", "τέντα στο $h_{\\mathrm{raw}}^{H}$"),
        ("step2_TAU[delta]", "two-region step on $\\delta$", "κλιμακωτή 2 περιοχών στο $\\delta$"),
        ("cliplin_TAU[zeta]", "clipped-linear on $\\zeta$", "αποκομμένος γραμμικός στο $\\zeta$"),
        ("step3_TAU[zeta]", "three-region step on $\\zeta$", "κλιμακωτή 3 περιοχών στο $\\zeta$")]),
    ("bivariate rules", "κανόνες δύο μεταβλητών", [
        ("bilin_TAU[h,delta]", "clipped-linear in $(h,\\delta)$", "αποκομμένος γραμμικός στα $(h,\\delta)$"),
        ("tentplus_TAU[h,delta]", "tent in $h$ $+\\,c\\,\\delta$", "τέντα στο $h$ $+\\,c\\,\\delta$"),
        ("cells_TAU[h,delta]", "$3\\times3$ cells in $(h,\\delta)$", "$3\\times3$ κελιά στα $(h,\\delta)$"),
        ("gbm_LS[h,delta]", "boosted trees on $(h,\\delta)$, LS", "ενισχυμένα δέντρα στα $(h,\\delta)$, ΕΤ"),
        ("bilin_TAU[h,zeta]", "clipped-linear in $(h,\\zeta)$", "αποκομμένος γραμμικός στα $(h,\\zeta)$")]),
    ("not deployable: need the unperturbed graph or the truth", "μη εφαρμόσιμοι: απαιτούν τον αδιατάρακτο γράφο ή την αλήθεια", [
        ("tent_TAU[dh]", "tent in $h-h_0$", "τέντα στο $h-h_0$"),
        ("step3_TAU[dh_raw]", "three-region step in $h_{\\mathrm{raw}}-h_{\\mathrm{raw},0}$",
         "κλιμακωτή 3 περιοχών στο $h_{\\mathrm{raw}}-h_{\\mathrm{raw},0}$"),
        ("graph_const", "best constant per graph", "καλύτερη σταθερά ανά γράφο"),
        ("oracle", "oracle $\\alpha^*$", "μαντείο $\\alpha^*$")]),
]


def nonmono():
    tab = pd.read_csv(os.path.join(R2, "nonmono_rules.csv"))
    sm = json.load(open(os.path.join(R2, "nonmono_summary.json")))
    P = tab[tab.config == "NCD|rule"].set_index("name")
    panels = ["test", "scale", "topo", "real"]
    rows = {"en": [], "gr": []}
    for blk_en, blk_gr, items in NONMONO_ROWS:
        for lang, blk in (("en", blk_en), ("gr", blk_gr)):
            rows[lang].append(f"\\multicolumn{{10}}{{@{{}}l}}{{\\emph{{{blk}}}}} \\\\")
        for name, en, gr in items:
            q = P.loc[name]
            ceiling = q["family"] == "ceiling"
            cells = [e4(q["calib_d"])]
            if ceiling:
                cells += ["--", "--"]
            else:
                cells += [e4(q["cv_d"]), pv(q["cv_p"])]
            cells += [e4(q["held_d"]), pv(q["held_p"]) if not ceiling else "--"]
            for p in panels:
                mark = "$^{\\dagger}$" if (not ceiling and np.isfinite(q[f"{p}_pholm"]) and q[f"{p}_pholm"] < 0.05) else ""
                cells.append(e4(q[f"{p}_d"]) + mark)
            for lang, lab in (("en", en), ("gr", gr)):
                rows[lang].append(f"\\quad {lab} & " + " & ".join(cells) + " \\\\")
    head = {"en": ("rule & calib & CV & $p_{\\mathrm{CV}}$ & held-out & $p$ & \\texttt{test} & \\texttt{scale} & "
                   "\\texttt{topo} & \\texttt{real} \\\\"),
            "gr": ("κανόνας & calib & CV & $p_{\\mathrm{CV}}$ & εκτός δείγμ. & $p$ & \\texttt{test} & \\texttt{scale} & "
                   "\\texttt{topo} & \\texttt{real} \\\\")}
    nd = int(sm["decision"]["n_deployable_rules"])
    cap_en = ("Non-monotone, dangling-deconfounded and bivariate Stage~2 rules against the locked constant "
              "$\\alpha=0.810$ (truth NCD, $r$ = Stage~1 rule), selected from the " + str(nd) + " deployable rules of "
              "\\texttt{stage2\\_nonmonotone\\_search.py}. Entries are differences in mean Kendall $\\tau$ in units of "
              "$10^{-4}$, rule minus constant (the decision threshold $0.0005$ is $5.0$). calib: in sample; CV: grouped "
              "8-fold cross-validation on \\texttt{calib}, with the constant refitted in every fold; held-out: the 73 "
              "graphs of \\texttt{test}, \\texttt{scale}, \\texttt{topo} and \\texttt{real} pooled. $p$: Wilcoxon "
              "signed-rank test over per-graph means; $^{\\dagger}$: Holm-adjusted $p<0.05$ across the four panels. "
              "Rules are $\\tau$-optimised unless marked LS. $h_0$ and $h_{\\mathrm{raw},0}$ are the values on the "
              "graph's own unperturbed version; the two ceilings are in sample.")
    cap_gr = ("Μη μονότονοι κανόνες, κανόνες χωρίς την επίδραση των κρεμάμενων σελίδων και κανόνες δύο μεταβλητών "
              "για το Στάδιο~2, έναντι της κλειδωμένης σταθεράς $\\alpha=0.810$ (αλήθεια NCD, $r$ = κανόνας Σταδίου~1), "
              "επιλεγμένοι από τους " + str(nd) + " εφαρμόσιμους κανόνες του \\texttt{stage2\\_nonmonotone\\_search.py}. "
              "Οι τιμές είναι διαφορές του μέσου $\\tau$ του Kendall σε μονάδες $10^{-4}$, κανόνας μείον σταθερά (το "
              "κατώφλι απόφασης $0.0005$ αντιστοιχεί σε $5.0$). calib: εντός δείγματος· CV: ομαδοποιημένη 8-πλή "
              "διασταυρωμένη επικύρωση στο \\texttt{calib}, με τη σταθερά επαναπροσαρμοσμένη σε κάθε τμήμα· εκτός δείγμ.: οι 73 "
              "γράφοι των \\texttt{test}, \\texttt{scale}, \\texttt{topo} και \\texttt{real} μαζί. $p$: έλεγχος "
              "προσημασμένων τάξεων Wilcoxon στις μέσες τιμές ανά γράφο· $^{\\dagger}$: $p<0.05$ μετά από διόρθωση Holm "
              "στα τέσσερα σύνολα. Οι κανόνες είναι βέλτιστοι ως προς $\\tau$, εκτός όσων σημειώνονται με ΕΤ. Τα $h_0$ "
              "και $h_{\\mathrm{raw},0}$ είναι οι τιμές στην αδιατάρακτη εκδοχή του ίδιου του γράφου· τα δύο ανώτατα "
              "όρια είναι εντός δείγματος.")
    for lang, cap in (("en", cap_en), ("gr", cap_gr)):
        colspec = "@{}lrrrrrrrrr@{}"
        body = ("\\begin{tabular}{" + colspec + "}\n\\toprule\n" + head[lang] + "\n\\midrule\n" +
                "\n".join(rows[lang]) + "\n\\bottomrule\n\\end{tabular}")
        write(os.path.join(OUT, lang, "tab_nonmono.tex"),
              table_env("tab:nonmono", cap_en, cap_gr, body, lang,
                        "\\footnotesize\\setlength{\\tabcolsep}{" + ("4pt" if lang == "en" else "2.6pt") + "}"))
    # ---- numbers quoted in the text
    macro("NmRules", str(nd))
    macro("NmMinGain", "0.0005")
    t = P.loc["tent_TAU[h]"]
    tp = json.loads(t["params"])
    macro("NmTentPeak", f3(tp["alpha_peak"]))
    macro("NmTentHzero", f3(tp["x0"]))
    macro("NmTentSL", f3(tp["s_L"]))
    macro("NmTentSR", f3(tp["s_R"]))
    macro("NmTentWithin", sgn3(t["calib_within_corr"]))
    for key, nm in (("tent_TAU[h]", "Tent"), ("cliplin_TAU[zeta]", "Zeta"), ("bilin_TAU[h,delta]", "Bilin"),
                    ("tent_TAU[dh]", "Dh"), ("step3_TAU[dh_raw]", "Dhraw"), ("step2_TAU[delta]", "Delta"),
                    ("tent_TAU[h_rawrel]", "Rel")):
        q = P.loc[key]
        macro(f"Nm{nm}Calib", sgn4(q["calib_d"]))
        macro(f"Nm{nm}CV", sgn4(q["cv_d"]))
        macro(f"Nm{nm}CVP", pv(q["cv_p"]))
        macro(f"Nm{nm}Held", sgn4(q["held_d"]))
        macro(f"Nm{nm}HeldP", pv(q["held_p"]))
        for p in panels:
            macro(f"Nm{nm}{p.capitalize()}", sgn4(q[f"{p}_d"]))
        macro(f"Nm{nm}RealWins", f"{int(q['real_wins'])}/{int(q['real_n'])}")
    zp = json.loads(P.loc["cliplin_TAU[zeta]", "params"])
    macro("NmZetaA", f3(zp["a"]))
    macro("NmZetaB", f"{zp['b']:+.3f}")
    bz = json.loads(P.loc["bilin_TAU[h,zeta]", "params"])
    macro("NmBilinZetaH", f"{bz['b']:+.3f}")
    bd = json.loads(P.loc["bilin_TAU[h,delta]", "params"])
    macro("NmBilinDeltaA", f3(bd["a"]))
    macro("NmBilinDeltaH", f"{bd['b']:+.3f}")
    macro("NmBilinDeltaD", f"{bd['c']:+.3f}")
    inst = pd.read_csv(os.path.join(R2, "instances.csv"))
    a10 = inst[(inst.exp == "A") & (inst.level == 1.0)]
    zin = a10.groupby("panel")["zero_in_frac"].mean()
    for p, nm in (("calib", "Calib"), ("topo", "Topo"), ("real", "Real")):
        macro(f"NmZinUnpert{nm}", f3(zin[p]))
    cite = a10[a10.base.isin(["citeseer", "cora_full", "cora_ml"])]["zero_in_frac"]
    macro("NmZinCiteMin", f"{cite.min():.2f}")
    macro("NmZinCiteMax", f"{cite.max():.2f}")
    dep = tab[(tab.config == "NCD|rule") & tab.deployable & (tab.name != "const")]
    onedh = dep[dep.feature.isin(["h", "h_v"]) & (dep.fit == "TAU")]
    macro("NmMaxOneDCV", sgn4(onedh["cv_d"].max()))
    macro("NmMaxOneDHeld", sgn4(onedh["held_d"].max()))
    ls_h = dep[(dep.feature == "h") & (dep.fit == "LS")]
    macro("NmLSWorst", sgn4(ls_h["held_d"].min()))
    macro("NmLSBest", sgn4(ls_h["held_d"].max()))
    raw = dep[dep.feature.isin(["h_raw", "h_rawrel", "hH_raw"]) & (dep.fit == "TAU")]
    macro("NmRawMaxCV", sgn4(raw["cv_d"].max()))
    macro("NmRawMaxHeld", sgn4(raw["held_d"].max()))
    macro("NmMeetsCV", str(int(sm["decision"]["n_meets_cv"])))
    orc, gc = P.loc["oracle"], P.loc["graph_const"]
    macro("NmOracleCalib", sgn4(orc["calib_d"]))
    macro("NmOracleHeld", sgn4(orc["held_d"]))
    macro("NmGraphConstCalib", sgn4(gc["calib_d"]))
    macro("NmDhShare", f"{100 * P.loc['tent_TAU[dh]', 'cv_d'] / orc['calib_d']:.0f}")
    # mechanism
    me = sm["mechanism"]
    tr = me["trajectories"]
    for col, nm in (("h_rule", "H"), ("hraw_rule", "Hraw"), ("hrawrel_rule", "Hrel"), ("hHraw", "HH"),
                    ("delta", "Dang"), ("zero_in_frac", "Zin")):
        macro(f"Nm{nm}Full", f3(tr["A1.0"][col]))
        macro(f"Nm{nm}SevA", f3(tr["A0.2"][col]))
        macro(f"Nm{nm}SevB", f3(tr["B0.6"][col]))
    di = me["directions"]
    for col, nm in (("hraw_rule", "Hraw"), ("hrawrel_rule", "Hrel"), ("hHraw", "HH"), ("delta", "Dang"),
                    ("zero_in_frac", "Zin")):
        macro(f"Nm{nm}AstarA", sgn3(di[col]["A"]["within_corr_astar"]))
        macro(f"Nm{nm}AstarB", sgn3(di[col]["B"]["within_corr_astar"]))
    hd = me["h0_dispersion"]
    macro("NmHzeroSd", f3(hd["h_rule"]["h0_sd"]))
    macro("NmExcA", sgn3(hd["h_rule"]["mean_excursion_A"]))
    macro("NmExcB", sgn3(hd["h_rule"]["mean_excursion_B"]))
    macro("NmRelSd", f3(hd["hrawrel_rule"]["h0_sd"]))
    macro("NmRelExcA", sgn3(hd["hrawrel_rule"]["mean_excursion_A"]))
    macro("NmRelExcB", sgn3(hd["hrawrel_rule"]["mean_excursion_B"]))
    macro("NmRelBetween", sgn3(hd["hrawrel_rule"]["corr_h0_graph_mean_astar"]))
    dc = me["distance_corr"]
    macro("NmDistOwn", sgn3(dc["own_centre"]))
    macro("NmDistOwnWithin", sgn3(dc["own_centre_within"]))
    macro("NmDistGlobal", sgn3(dc["global_centre_corr"]))
    macro("NmDistGlobalC", f"{dc['best_global_centre']:.2f}")
    # sensitivity truths
    for cfg, nm in (("NCD95|ng13", "Nf"), ("TOT|rule", "Tot")):
        Q = tab[tab.config == cfg]
        dq = Q[Q.deployable & (Q.name != "const")]
        macro(f"Nm{nm}Rules", str(len(dq)))
        macro(f"Nm{nm}MaxHeld", sgn4(dq["held_d"].max()))
        macro(f"Nm{nm}MaxCV", sgn4(dq["cv_d"].max()))
        macro(f"Nm{nm}Promoted", str(int(dq["promote"].sum())))
        z = Q.set_index("name").loc["tent_TAU[dh]"]
        macro(f"Nm{nm}DhHeld", sgn4(z["held_d"]))
        macro(f"Nm{nm}DhHeldP", pv(z["held_p"]))
        macro(f"Nm{nm}Const", f3(float(Q["const_alpha"].iloc[0])))


# ---------------------------------------------------------------------------
# 4. pipelines, intervals and convergence
# ---------------------------------------------------------------------------
def pipelines():
    p = pd.read_csv(os.path.join(R2, "pipelines.csv"))
    pols = [("NG13 static (r=0.118, a=0.85)", "[NG13] $(0.118,\\,0.85)$", "[NG13] $(0.118,\\,0.85)$"),
            ("NG13 static (r=0.105, a=0.95)", "[NG13] $(0.105,\\,0.95)$", "[NG13] $(0.105,\\,0.95)$"),
            ("Stage 1 only (rule r, a=0.85)", "Stage 1 only", "μόνο Στάδιο 1"),
            ("two-stage: rule r + cliplin_TAU(h)", "two-stage ($\\alpha=0.81$)", "δύο στάδια ($\\alpha=0.81$)"),
            ("oracle alpha* at rule r", "oracle $\\alpha^*$", "μαντείο $\\alpha^*$")]
    panels = ["calib", "test", "scale", "topo", "real"]
    rows = {"en": [], "gr": []}
    for t in ["NCD", "NCD95", "TOT", "POP", "Q"]:
        q = p[p.truth == t]
        if q.empty:
            continue
        for lang in ("en", "gr"):
            rows[lang].append(f"\\multicolumn{{6}}{{@{{}}l}}{{\\textit{{{'truth' if lang == 'en' else 'αλήθεια'} {t}}}}} \\\\")
        for key, en, gr in pols:
            cells = []
            for pn in panels:
                z = q[(q.policy == key) & (q.panel == pn)]
                if z.empty:
                    cells.append("--")
                    continue
                z = z.iloc[0]
                if key.startswith("NG13 static (r=0.118"):
                    cells.append(f4(z["tau"]))
                else:
                    cells.append(f"{f4(z['tau'])} ({sgn4(z['d_vs_ng13_085'])})")
            for lang, lab in (("en", en), ("gr", gr)):
                rows[lang].append(f"\\quad {lab} & " + " & ".join(cells) + " \\\\")
    for lang in ("en", "gr"):
        head = ("policy $(r,\\alpha)$ & \\texttt{calib} & \\texttt{test} & \\texttt{scale} & \\texttt{topo} & "
                "\\texttt{real} \\\\") if lang == "en" else (
            "πολιτική $(r,\\alpha)$ & \\texttt{calib} & \\texttt{test} & \\texttt{scale} & \\texttt{topo} & "
            "\\texttt{real} \\\\")
        body = ("\\begin{tabular}{@{}lccccc@{}}\n\\toprule\n" + head + "\n\\midrule\n" + "\n".join(rows[lang]) +
                "\n\\bottomrule\n\\end{tabular}")
        cap_en = ("Complete pipelines against the static [NG13] settings under all five truths. Cells: mean Kendall "
                  "$\\tau$ over instances; in parentheses, the difference against [NG13] $(0.118,0.85)$ averaged over "
                  "graphs (each graph weighted equally). \\emph{Two-stage}: $r$ from the locked Stage~1 rule and "
                  "$\\alpha=0.81$, both fitted on calibration graphs under NCD and applied unchanged under every truth. "
                  "POP and Q exist only for the static generator.")
        cap_gr = ("Πλήρεις διαδικασίες έναντι των στατικών ρυθμίσεων του [NG13] υπό και τις πέντε αλήθειες. Κελιά: "
                  "μέσο $\\tau$ του Kendall ανά στιγμιότυπο· σε παρένθεση, η διαφορά έναντι του [NG13] "
                  "$(0.118,0.85)$ με μέσο όρο ανά γράφο (κάθε γράφος με ίσο βάρος). \\emph{Δύο στάδια}: $r$ από τον "
                  "κλειδωμένο κανόνα του Σταδίου~1 και $\\alpha=0.81$, αμφότερα προσαρμοσμένα σε γράφους "
                  "βαθμονόμησης υπό NCD και εφαρμοσμένα αμετάβλητα υπό κάθε αλήθεια. Οι αλήθειες POP και Q υπάρχουν "
                  "μόνο για τη στατική γεννήτρια.")
        write(os.path.join(OUT, lang, "tab_pipe.tex"), table_env("tab:pipe", cap_en, cap_gr, body, lang, "\\scriptsize", star=True))
    # headline pipeline macros
    def pq(t, pol, pn, col):
        z = p[(p.truth == t) & (p.policy == pol) & (p.panel == pn)]
        return z.iloc[0][col] if len(z) else np.nan
    two = "two-stage: rule r + cliplin_TAU(h)"
    for t, nm in (("NCD", "Ncd"), ("NCD95", "Ncdnf"), ("TOT", "Tot"), ("POP", "Pop"), ("Q", "Q")):
        macro(f"Pipe{nm}TwoTest", f4(pq(t, two, "test", "tau")))
        macro(f"Pipe{nm}TwoVsLoTest", sgn4(pq(t, two, "test", "d_vs_ng13_085")))
        macro(f"Pipe{nm}TwoVsHiTest", sgn4(pq(t, two, "test", "d_vs_ng13_095")))
        macro(f"Pipe{nm}OracleVsLoTest", sgn4(pq(t, "oracle alpha* at rule r", "test", "d_vs_ng13_085")))
    macro("PipeNcdTwoVsLoScale", sgn4(pq("NCD", two, "scale", "d_vs_ng13_085")))
    macro("PipeNcdTwoVsLoTopo", sgn4(pq("NCD", two, "topo", "d_vs_ng13_085")))
    macro("PipeNcdTwoVsLoReal", sgn4(pq("NCD", two, "real", "d_vs_ng13_085")))
    macro("PipeNcdTwoVsLoRealP", pv(pq("NCD", two, "real", "p_vs_ng13_085")))
    macro("PipeNcdTwoVsLoRealWins", str(int(pq("NCD", two, "real", "wins_vs_ng13_085"))))
    macro("PipeNcdStageOneTest", sgn4(pq("NCD", "Stage 1 only (rule r, a=0.85)", "test", "d_vs_ng13_085")))
    macro("PipeNcdStageOneReal", sgn4(pq("NCD", "Stage 1 only (rule r, a=0.85)", "real", "d_vs_ng13_085")))
    # intervals on exact grid points (NCD truth, cluster bootstrap over graphs)
    for pn, nm in (("test", "Test"), ("scale", "Scale"), ("topo", "Topo"), ("real", "Real")):
        st = cmp_stats("NCD", pn, ("rule", 0.81), ("can", 0.85))
        macro(f"Two{nm}D", sgn4(st["mean"]))
        macro(f"Two{nm}Lo", f4(st["lo"]))
        macro(f"Two{nm}Hi", f4(st["hi"]))
        macro(f"Two{nm}P", pv(st["p"]))
        macro(f"Two{nm}Wins", f"{st['wins']}/{st['n']}")
        st = cmp_stats("NCD", pn, ("rule", 0.81), ("rule", 0.85))
        macro(f"Sth{nm}D", sgn4(st["mean"]))
        macro(f"Sth{nm}P", pv(st["p"]))
        st = cmp_stats("NCD", pn, ("rule", 0.85), ("can", 0.85))
        macro(f"Sto{nm}D", sgn4(st["mean"]))
        macro(f"Sto{nm}P", pv(st["p"]))
        st = cmp_stats("NCD95", pn, ("ng13", 0.93), ("ng13", 0.95))
        macro(f"Nf{nm}D", sgn4(st["mean"]))
        macro(f"Nf{nm}P", pv(st["p"]))
        macro(f"Nf{nm}Wins", f"{st['wins']}/{st['n']}")
    # real-graph per-graph two-stage differences
    m = S2.valid(df2, "NCD") & (df2["panel"] == "real").to_numpy()
    d = tau_at("rule", "NCD", 0.81, m) - tau_at("can", "NCD", 0.85, m)
    per = pd.Series(d).groupby(df2["base"].to_numpy()[m]).mean()
    for g, nm in (("citeseer", "Citeseer"), ("cnr2000", "Cnr"), ("cora_full", "Corafull"), ("cora_ml", "Coraml"),
                  ("wikics", "Wikics")):
        macro(f"TwoReal{nm}", sgn4(per[g]))
    # NCD95 constant
    fits = json.load(open(os.path.join(R2, "fits.json")))
    macro("NfConstAlpha", f3({x["name"]: x for x in fits["ng13|NCD95|h_ng13"]}["const"]["params"][0]))


def convergence():
    path = os.path.join(R2, "convergence.csv")
    if not os.path.exists(path):
        return
    c = pd.read_csv(path)
    rows = {"en": [], "gr": []}
    tol = 1e-6
    for (rn, a0, a1, lab_en, lab_gr) in (("rule", 0.85, 0.81, "NCD branch ($r$ = rule)", "κλάδος NCD ($r$ = κανόνας)"),
                                         ("ng13", 0.95, 0.93, "NCD95 branch ($r=0.105$)", "κλάδος NCD95 ($r=0.105$)")):
        k0, k1 = np.log(tol) / np.log(a0), np.log(tol) / np.log(a1)
        cells = [f"{a0:.2f}$\\to${a1:.2f}", f"{k0:.0f}$\\to${k1:.0f} ($-{100 * (1 - k1 / k0):.1f}\\%$)"]
        for pn in ("test", "real"):
            d = c[c.panel == pn]
            m0, m1 = d[f"it_{rn}_{a0}"].mean(), d[f"it_{rn}_{a1}"].mean()
            cells.append(f"{m0:.1f}$\\to${m1:.1f} ($-{100 * (1 - m1 / m0):.1f}\\%$)")
            macro(f"Conv{'Ncd' if rn == 'rule' else 'Nf'}{'Test' if pn == 'test' else 'Real'}",
                  f"{100 * (1 - m1 / m0):.1f}")
        macro(f"Conv{'Ncd' if rn == 'rule' else 'Nf'}Bound", f"{100 * (1 - k1 / k0):.1f}")
        macro(f"Conv{'Ncd' if rn == 'rule' else 'Nf'}BoundK", f"{k0:.0f}\\to{k1:.0f}")
        for lang, lab in (("en", lab_en), ("gr", lab_gr)):
            rows[lang].append(f"{lab} & " + " & ".join(cells) + " \\\\")
    for lang in ("en", "gr"):
        head = ("configuration & $\\alpha$ & bound $\\ln\\varepsilon/\\ln\\alpha$ & measured, \\texttt{test} & "
                "measured, \\texttt{real} \\\\") if lang == "en" else (
            "διάταξη & $\\alpha$ & φράγμα $\\ln\\varepsilon/\\ln\\alpha$ & μετρημένο, \\texttt{test} & "
            "μετρημένο, \\texttt{real} \\\\")
        body = ("\\begin{tabular}{@{}lllll@{}}\n\\toprule\n" + head + "\n\\midrule\n" + "\n".join(rows[lang]) +
                "\n\\bottomrule\n\\end{tabular}")
        cap_en = ("Power-iteration counts before and after the noise discount (stopping rule "
                  "$\\|\\pi^{(k+1)}-\\pi^{(k)}\\|_1<\\varepsilon=10^{-6}$, start $v$). The bound column is the "
                  "a-priori estimate from $|\\lambda_2(P)|\\le\\alpha$; measured columns are mean counts over all "
                  "perturbed instances of the panel.")
        cap_gr = ("Πλήθος επαναλήψεων της μεθόδου δυνάμεων πριν και μετά την έκπτωση θορύβου (κριτήριο τερματισμού "
                  "$\\|\\pi^{(k+1)}-\\pi^{(k)}\\|_1<\\varepsilon=10^{-6}$, αρχή $v$). Η στήλη του φράγματος είναι η εκ των "
                  "προτέρων εκτίμηση από το $|\\lambda_2(P)|\\le\\alpha$· οι μετρημένες στήλες είναι μέσοι όροι σε όλα "
                  "τα διαταραγμένα στιγμιότυπα του συνόλου.")
        write(os.path.join(OUT, lang, "tab_conv.tex"), table_env("tab:conv", cap_en, cap_gr, body, lang, "\\small", star=True))


def real_detail():
    inst = pd.read_csv(os.path.join(R2, "instances.csv"))
    r = inst[(inst.panel == "real") & ~((inst.exp == "A") & (inst.level == 1.0))]
    m = S2.valid(df2, "NCD") & (df2["panel"] == "real").to_numpy()
    two = pd.Series(tau_at("rule", "NCD", 0.81, m)).groupby(df2["base"].to_numpy()[m]).mean()
    names = {"cnr2000": "cnr-2000", "wikics": "WikiCS", "cora_full": "Cora (full)", "cora_ml": "Cora-ML",
             "citeseer": "CiteSeer"}
    rows = []
    for g in ["cnr2000", "wikics", "cora_full", "cora_ml", "citeseer"]:
        d = r[r.base == g]
        rows.append(f"{names[g]} & {d['r_rule'].mean():.3f} & {d['t085_can_NCD'].mean():.4f} & "
                    f"{d['t085_rule_NCD'].mean():.4f} & {two[g]:.4f} & {d['tstar_rule_NCD'].mean():.4f} & "
                    f"{d['astar_rule_NCD'].mean():.3f} & {d['astar_rule_TOT'].mean():.3f} \\\\")
    for lang in ("en", "gr"):
        head = ("graph & mean $r$ & [NG13] $(0.118,0.85)$ & Stage 1 & two-stage & oracle & $\\alpha^*$ NCD & "
                "$\\alpha^*$ TOT \\\\") if lang == "en" else (
            "γράφος & μέσο $r$ & [NG13] $(0.118,0.85)$ & Στάδιο 1 & δύο στάδια & μαντείο & $\\alpha^*$ NCD & "
            "$\\alpha^*$ TOT \\\\")
        body = ("\\begin{tabular}{@{}lrrrrrrr@{}}\n\\toprule\n" + head + "\n\\midrule\n" + "\n".join(rows) +
                "\n\\bottomrule\n\\end{tabular}")
        cap_en = ("Real networks, truth NCD: mean Kendall $\\tau$ per graph over the perturbed instances, and the mean "
                  "oracle damping under NCD and TOT.")
        cap_gr = ("Πραγματικά δίκτυα, αλήθεια NCD: μέσο $\\tau$ του Kendall ανά γράφο στα διαταραγμένα στιγμιότυπα "
                  "και μέση βέλτιστη απόσβεση υπό NCD και TOT.")
        write(os.path.join(OUT, lang, "tab_realdetail.tex"), table_env("tab:realdetail", cap_en, cap_gr, body, lang, "\\footnotesize", place="!htbp"))


def wp0_tables():
    """Generate the mixture mechanism and anchor-matching outputs used by WP0 text."""
    path = os.path.join(R2, "nonmono_summary.json")
    if not os.path.exists(path):
        return
    me = json.load(open(path))["mechanism"]["trajectories"]
    levels = (("A", "A1.0", "A0.2"), ("B", "B0.1", "B0.6"))
    rows = []
    for exp, first, last in levels:
        a, b = me[first], me[last]
        rows.append({
            "experiment": exp,
            "delta0": float(me["A1.0"]["delta"]),
            "delta1": float(b["delta"]),
            "zeta0": float(me["A1.0"]["zero_in_frac"]),
            "zeta1": float(b["zero_in_frac"]),
            "h0": float(me["A1.0"]["h_rule"]),
            "h1": float(b["h_rule"]),
            "hraw0": float(me["A1.0"]["hraw_rule"]),
            "hraw1": float(b["hraw_rule"]),
        })
    for lang in ("en", "gr"):
        head = ("experiment & $\\delta$ & $\\zeta$ & $h$ & $h_{\\rm raw}$ \\\\"
                if lang == "en" else
                "πείραμα & $\\delta$ & $\\zeta$ & $h$ & $h_{\\rm raw}$ \\\\")
        lines = []
        for r in rows:
            label = ("A: link keep 1.0$\\to$0.2" if r["experiment"] == "A" else
                     "B: page loss 0.1$\\to$0.6")
            if lang == "gr":
                label = ("Α: διατήρηση τόξων 1.0$\\to$0.2" if r["experiment"] == "A" else
                         "Β: απώλεια σελίδων 0.1$\\to$0.6")
            cells = [f"{r[k+'0']:.3f}$\\to${r[k+'1']:.3f}" for k in ("delta", "zeta", "h", "hraw")]
            lines.append(label + " & " + " & ".join(cells) + " \\\\")
        body = "\\begin{tabular}{@{}lrrrr@{}}\\toprule\n" + head + "\n\\midrule\n" + "\n".join(lines) + "\n\\bottomrule\\end{tabular}"
        cap_en = ("Experiments A and B are mixtures of dangling-page and zero-in-link effects. "
                  "Values run from the least damaged to the most severe level.")
        cap_gr = ("Τα Πειράματα Α και Β είναι μείγματα επιδράσεων κρεμάμενων σελίδων και μηδενικών εισερχόμενων "
                  "συνδέσεων. Οι τιμές εκτείνονται από την ηπιότερη έως τη σοβαρότερη στάθμη.")
        write(os.path.join(OUT, lang, "tab_mixture.tex"),
              table_env("tab:mixture", cap_en, cap_gr, body, lang, "\\small"))
    write(os.path.join(HERE, "V2_WP0_mixture_table.md"),
          "| Experiment | $\\delta$ | $\\zeta$ | $h$ | $h_{raw}$ |\n|---|---:|---:|---:|---:|\n" +
          "\n".join(f"| {r['experiment']} | {r['delta0']:.3f} → {r['delta1']:.3f} | "
                    f"{r['zeta0']:.3f} → {r['zeta1']:.3f} | {r['h0']:.3f} → {r['h1']:.3f} | "
                    f"{r['hraw0']:.3f} → {r['hraw1']:.3f} |" for r in rows) + "\n")
    p = pd.read_csv(os.path.join(R2, "pipelines.csv"))
    def value(truth, policy, panel="test", col="d_vs_ng13_085"):
        q = p[(p.truth == truth) & (p.policy == policy) & (p.panel == panel)]
        return float(q.iloc[0][col])
    two = "two-stage: rule r + cliplin_TAU(h)"
    macro("WpZeroNcdVsCanonical", sgn4(value("NCD", two, col="d_vs_ng13_085")))
    macro("WpZeroNcdVsNg", sgn4(value("NCD", two, col="d_vs_ng13_095")))
    matched = cmp_stats("NCD95", "test", ("ng13", 0.93), ("ng13", 0.95))
    macro("WpZeroNfMatched", sgn4(matched["mean"]))
    macro("WpZeroNfMatchedP", pv(matched["p"]))


def main():
    os.makedirs(OUT, exist_ok=True)
    real_dataset_table()
    for k, v in NUM_real.items():
        nm = {"cnr2000": "Cnr", "wikics": "Wikics", "cora_full": "Corafull", "cora_ml": "Coraml", "citeseer": "Citeseer"}[k]
        macro(f"Pbar{nm}", f3(v["pbar"]))
    pbs = [v["pbar"] for v in NUM_real.values()]
    macro("PbarRealMin", f"{min(pbs):.2f}")
    macro("PbarRealMax", f"{max(pbs):.2f}")
    rc = 0.242 - 0.143 * NUM_real["cnr2000"]["pbar"]
    macro("RuleRCnr", f3(rc))
    macro("RuleMuCnr", f3(0.85 * rc))
    panel_table()
    stage1()
    stage1_truth_table()
    stage2()
    nonmono()
    bound_numbers()
    pipelines()
    convergence()
    real_detail()
    wp0_tables()
    lines = ["% generated by make_paper_assets.py -- do not edit",
             "% every value is mode-agnostic: signed numbers and powers of ten are wrapped in",
             "% \\ensuremath, so a macro can be used both in running text and inside $...$"]
    for k in sorted(NUM):
        lines.append(f"\\newcommand{{\\{k}}}{{{mode_agnostic(NUM[k])}}}")
    write(os.path.join(OUT, "macros.tex"), "\n".join(lines) + "\n")
    json.dump(NUM, open(os.path.join(HERE, "results", "paper_numbers.json"), "w"), indent=1, sort_keys=True)
    print(f"{len(NUM)} macros; tables in {OUT}")


if __name__ == "__main__":
    main()
