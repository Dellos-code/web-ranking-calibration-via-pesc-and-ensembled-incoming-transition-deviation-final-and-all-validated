"""
fig_lang.py
===========
Greek versions of the publication figures without duplicating the plotting code.

patch() wraps the matplotlib text entry points used by generate_figures.py
(titles, axis labels, free text, tick labels, legend labels, colour-bar labels) so
that every string passes through translate(), an ordered list of literal and regex
replacements.  Mathematical TeX ($...$) is left untouched.

    python3 generate_figures.py stage1 --lang gr   ->  figures_gr/stage1_*.pdf/.png
    python3 generate_figures.py stage2 --lang gr   ->  figures_gr/stage2_*.pdf/.png
"""
import re

import matplotlib.axes
import matplotlib.colorbar

# longest / most specific phrases first; applied in order
RULES = [
    # ---- Stage 1 scatter / sensitivity
    ("truth: canonical NCDawareRank on the complete graph", "αλήθεια: κανονικό NCDawareRank στον πλήρη γράφο"),
    ("canonical NCDawareRank on the complete graph", "κανονικό NCDawareRank στον πλήρη γράφο"),
    ("truth: latent population importance", "αλήθεια: λανθάνουσα πληθυσμιακή σημαντικότητα"),
    ("latent population importance", "λανθάνουσα πληθυσμιακή σημαντικότητα"),
    ("PageRank on the complete graph", "PageRank στον πλήρη γράφο"),
    ("[NG13] protocol: same r on the complete graph", "πρωτόκολλο [NG13]: ίδιο r στον πλήρη γράφο"),
    (re.compile(r"(\d+) calibration graphs, (\d+) perturbed instances"), r"\1 γράφοι βαθμονόμησης, \2 διαταραγμένα στιγμιότυπα"),
    (re.compile(r"(\d+) calibration graphs, (\d+) instances"), r"\1 γράφοι βαθμονόμησης, \2 στιγμιότυπα"),
    ("same points; grey lines join each graph's\nperturbation series (Simpson structure)",
     "ίδια σημεία· οι γκρι γραμμές ενώνουν τη σειρά\nδιαταραχών κάθε γράφου (δομή Simpson)"),
    ("same points; grey lines join each graph's\nperturbation series",
     "ίδια σημεία· οι γκρι γραμμές ενώνουν τη σειρά\nδιαταραχών κάθε γράφου"),
    ("Exp. A: sparsity (keep 90%-20% of links)", "Πείρ. Α: αραίωση (διατήρηση 90%-20% των συνδέσμων)"),
    ("Exp. B: new pages (10%-60% lose 90% of in-links)", "Πείρ. Β: νέες σελίδες (10%-60% χάνουν το 90% των εισερχόμενων)"),
    ("Exp. A: sparsity", "Πείρ. Α: αραίωση"),
    ("Exp. B: new pages", "Πείρ. Β: νέες σελίδες"),
    ("clip-linear, least squares", "αποκομμένος γραμμικός, ελάχιστα τετράγωνα"),
    ("clip-linear, tau-optimised", "αποκομμένος γραμμικός, βέλτιστος ως προς τ"),
    ("clip-linear tau-opt.", "αποκομμένος γραμμικός, βέλτ. ως προς τ"),
    ("clip-linear LS", "αποκομμένος γραμμικός, ΕΤ"),
    ("isotonic (monotone)", "ισοτονικός (μονότονος)"),
    (re.compile(r"isotonic step function of (\S+)"), r"ισοτονική κλιμακωτή συνάρτηση του \1"),
    ("isotonic:", "ισοτονικός:"),
    ("best constant r (calibration)", "καλύτερο σταθερό r (βαθμονόμηση)"),
    ("best constant", "καλύτερη σταθερά"),
    ("canonical $r=0.10/0.85$", "κανονικό $r=0.10/0.85$"),
    ("mean of each graph", "μέση τιμή κάθε γράφου"),
    ("of the observed graph", "του παρατηρούμενου γράφου"),
    ("oracle ratio", "βέλτιστος λόγος"),
    ("between graph means", "μεταξύ μέσων γράφων"),
    ("within graphs", "εντός γράφων"),
    ("between graphs", "μεταξύ γράφων"),
    ("pooled", "συνολική"),
    (re.compile(r"\bcorr\("), "συσχ.("),
    # ---- Stage 1 diagnostics
    ("tau(r) for one calibration graph, Exp. A; dots mark r*", "τ(r) για έναν γράφο βαθμονόμησης, Πείρ. Α· οι κουκκίδες δείχνουν το r*"),
    (re.compile(r"keep (\d+)% of links"), r"διατήρηση \1%"),
    ("canonical r", "κανονικό r"),
    ("mixing ratio", "λόγος ανάμειξης"),
    ("Kendall $\\tau$ vs canonical NCDawareRank (complete graph)", "$\\tau$ του Kendall έναντι κανονικού NCDawareRank (πλήρης γράφος)"),
    ("What determines r*?  (calibration panel)", "Τι καθορίζει το r*; (σύνολο βαθμονόμησης)"),
    ("graph + level (additive)", "γράφος + επίπεδο (προσθετικά)"),
    ("hidden generator params + level*", "κρυφές παράμετροι γεννήτριας + επίπεδο*"),
    ("all observable statistics (ridge)*", "όλα τα παρατηρήσιμα στατιστικά (ridge)*"),
    ("alone (clip-linear)*", "μόνο του (αποκομμένος γραμμικός)*"),
    ("graph identity", "ταυτότητα γράφου"),
    ("perturbation level", "επίπεδο διαταραχής"),
    ("share of the variance of $r^*$ explained ($R^2$; * = grouped 8-fold CV by graph)",
     "μερίδιο της διασποράς του $r^*$ που εξηγείται ($R^2$· * = ομαδοποιημένη 8-πλή CV ανά γράφο)"),
    ("The driver is latent block-quality structure", "Ο οδηγός είναι η λανθάνουσα δομή ποιότητας των μπλοκ"),
    ("latent block share of page quality (ICC, hidden)", "λανθάνον μερίδιο μπλοκ στην ποιότητα σελίδων (ICC, κρυφό)"),
    ("mean r* of the graph (population truth)", "μέσο r* του γράφου (πληθυσμιακή αλήθεια)"),
    ("latent fitness dispersion (sigma of log quality)", "διασπορά λανθάνουσας ποιότητας (σ του log q)"),
    ("mean Pbar", "μέσο Pbar"),
    ("mean r*", "μέσο r*"),
    ("dispersion", "διασπορά"),
    ("Out-of-sample: rule vs constant", "Εκτός δείγματος: κανόνας έναντι σταθεράς"),
    ("tau-optimised", "βέλτιστος ως προς τ"),
    ("clip-linear on", "αποκομμένος γραμμικός στο"),
    ("headroom", "περιθώριο"),
    ("share of the canonical-to-oracle tau gap closed (0 = canonical r, 1 = oracle)",
     "μερίδιο του χάσματος τ κανονικού–μαντείου που καλύπτεται (0 = κανονικό r, 1 = μαντείο)"),
    ("NCD: κανονικό NCDawareRank στον πλήρη γράφο", "NCD: κανονικό NCDawareRank στον πλήρη γράφο"),
    # ---- Stage 2
    ("truth NCD (primary), r = Stage 1 rule", "αλήθεια NCD (κύρια), r = κανόνας Σταδίου 1"),
    ("alpha-free truth", "αλήθεια ανεξάρτητη από α"),
    ("r = Stage 1 rule", "r = κανόνας Σταδίου 1"),
    ("at the Stage 1 $r$", "στο $r$ του Σταδίου 1"),
    ("oracle damping $\\alpha^*$", "βέλτιστη απόσβεση $\\alpha^*$"),
    ("alpha0 = 0.85 (Stage 1)", "α0 = 0.85 (Στάδιο 1)"),
    ("alpha = 0.95 ([NG13] experiments)", "α = 0.95 (πειράματα [NG13])"),
    ("canonical NCDawareRank, alpha=0.85", "κανονικό NCDawareRank, α=0.85"),
    ("[NG13] experiments, alpha=0.95", "πειράματα [NG13], α=0.95"),
    ("population PageRank, alpha=0.85", "πληθυσμιακό PageRank, α=0.85"),
    ("latent quality q (walk-free)", "λανθάνουσα ποιότητα q (χωρίς περίπατο)"),
    ("truth's own alpha", "το δικό α της αλήθειας"),
    (re.compile(r"\bh at r = "), "h στο r = "),
    ("One graph (Exp. A, keep 60%): each truth's\ntau(alpha) peaks at its own alpha; dots = alpha*",
     "Ένας γράφος (Πείρ. Α, διατήρηση 60%): το τ(α) κάθε\nαλήθειας κορυφώνεται στο δικό της α· κουκκίδες = α*"),
    ("outside the\nsearch grid", "εκτός του\nπλέγματος αναζήτησης"),
    ("damping $\\alpha$", "απόσβεση $\\alpha$"),
    ("alpha* is set by the choice of truth\n(orange bar = the truth's own alpha; TOT: Beta(1,1) mean 0.5)",
     "Το α* καθορίζεται από την επιλογή αλήθειας\n(πορτοκαλί γραμμή = το δικό α της αλήθειας· TOT: μέση τιμή Beta(1,1) 0.5)"),
    ("oracle $\\alpha^*$ (calibration panel)", "βέλτιστο $\\alpha^*$ (σύνολο βαθμονόμησης)"),
    ("oracle $\\alpha^*$", "βέλτιστο $\\alpha^*$"),
    ("oracle $r^*$", "βέλτιστο $r^*$"),
    ("What determines alpha*?", "Τι καθορίζει το α*;"),
    ("graph + level", "γράφος + επίπεδο"),
    ("hidden params + level*", "κρυφές παράμ. + επίπεδο*"),
    ("all observables incl. h*", "όλα τα παρατηρήσιμα μαζί με h*"),
    ("h alone (clip-linear)*", "μόνο h (αποκομμένος γραμμικός)*"),
    ("share of the variance of $\\alpha^*$ explained ($R^2$; * grouped 8-fold CV)",
     "μερίδιο της διασποράς του $\\alpha^*$ που εξηγείται ($R^2$· * ομαδοποιημένη 8-πλή CV)"),
    ("Pipelines vs the static baseline (held-out test panel)", "Διαδικασίες έναντι στατικής βάσης (σύνολο ελέγχου)"),
    ("NG13 static", "NG13 στατικό"),
    ("Stage 1 only (rule r, a=0.85)", "μόνο Στάδιο 1 (r κανόνα, a=0.85)"),
    ("Stage 1 const r=0.18, a=0.85", "Στάδιο 1, σταθερό r=0.18, a=0.85"),
    ("canonical r + g(h) [Stage 2 only]", "κανονικό r + g(h) [μόνο Στάδιο 2]"),
    ("two-stage: rule r +", "δύο στάδια: r κανόνα +"),
    ("oracle alpha* at rule r", "μαντείο α* στο r κανόνα"),
    ("mean tau gain over the static [NG13] baseline (r = 0.118, alpha = 0.85), test panel",
     "μέσο κέρδος τ έναντι στατικής βάσης [NG13] (r = 0.118, α = 0.85), σύνολο ελέγχου"),
    (re.compile(r"\btruth (NCD95|NCD|TOT|POP|Q)\b"), r"αλήθεια \1"),
    (re.compile(r"\balpha\*"), "α*"),
    (re.compile(r"\balpha = "), "α = "),
    (re.compile(r"\balpha="), "α="),
    ("calibration panel", "σύνολο βαθμονόμησης"),
    ("within", "εντός"),
    ("between", "μεταξύ"),
    # panel letters at the start of titles
    (re.compile(r"^\(a\)"), "(α)"), (re.compile(r"^\(b\)"), "(β)"),
    (re.compile(r"^\(c\)"), "(γ)"), (re.compile(r"^\(d\)"), "(δ)"),
]


def translate(s):
    if not isinstance(s, str) or not s:
        return s
    # protect TeX math segments
    parts = re.split(r"(\$[^$]*\$)", s)
    out = []
    for i, part in enumerate(parts):
        if i % 2 == 1:
            out.append(part)
        else:
            out.append(part)
    s2 = "".join(out)
    # literal phrases longest-first (so specific phrases win over their sub-phrases), then regexes
    for pat, rep in LITERALS:
        s2 = s2.replace(pat, rep)
    for pat, rep in REGEXES:
        s2 = pat.sub(rep, s2)
    return s2


LITERALS = sorted([(p, r) for p, r in RULES if isinstance(p, str)], key=lambda pr: -len(pr[0]))
REGEXES = [(p, r) for p, r in RULES if not isinstance(p, str)]


def patch():
    A = matplotlib.axes.Axes

    def wrap(name, argpos=0):
        orig = getattr(A, name)

        def f(self, *args, **kw):
            args = list(args)
            if len(args) > argpos:
                if isinstance(args[argpos], str):
                    args[argpos] = translate(args[argpos])
                elif isinstance(args[argpos], (list, tuple)):
                    args[argpos] = [translate(x) for x in args[argpos]]
            if "label" in kw:
                kw["label"] = translate(kw["label"])
            return orig(self, *args, **kw)
        setattr(A, name, f)

    for nm in ("set_title", "set_xlabel", "set_ylabel", "set_xticklabels", "set_yticklabels"):
        wrap(nm, 0)
    wrap("text", 2)
    orig_legend = A.legend

    def legend(self, *args, **kw):
        if not args:
            h, l = self.get_legend_handles_labels()
            return orig_legend(self, h, [translate(x) for x in l], **kw)
        return orig_legend(self, *args, **kw)
    A.legend = legend
    orig_cb = matplotlib.colorbar.Colorbar.set_label

    def cb_label(self, label, *a, **kw):
        return orig_cb(self, translate(label), *a, **kw)
    matplotlib.colorbar.Colorbar.set_label = cb_label
