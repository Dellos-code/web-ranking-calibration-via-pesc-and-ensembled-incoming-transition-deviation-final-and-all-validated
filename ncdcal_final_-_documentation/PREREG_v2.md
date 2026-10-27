# V2 extension preregistration

This preregistration is committed before any new held-out outcome from the v2 panels is
computed. The replication unit is always the base graph; perturbations from one base graph
are averaged before paired tests, bootstrap intervals, or cross-validation splits.

## Decision rule

A new deployable rule is promoted only if its mean Kendall tau gain over the locked rule is at
least `0.0005` with exact two-sided Wilcoxon `p < 0.05` in grouped 8-fold calibration CV and on
the pooled held-out panels, with no Holm-significant held-out loss on any declared panel. The
locked defaults remain the documented defaults even when an optional rule passes.

All declared comparisons use per-graph means, paired differences, exact two-sided Wilcoxon
signed-rank tests, percentile bootstrap 95% CIs over base graphs, wins/n, and Holm correction.
Kendall tau-b is primary; Spearman rho is secondary. The family size is three for WP1's three
primary comparisons, three for WP2's three comparator families, and one for each WP3 and WP4
pre-registered contrast. Exploratory property associations are not used for promotion.

## WP1: real2

The new panel uses seeds `17100 + j`, new names only, and the locked Stage 1 and Stage 2
pipeline. Candidate datasets are verified for directedness, label provenance, licence, size and
SHA-256 before inclusion. Failed or unavailable candidates are recorded in `V2_REPORT.md`.

Primary hypotheses on new graphs and on old-plus-new graphs:

1. The locked two-stage rule (`r = clip(0.242 - 0.143 Pbar, 0.01, 0.99)`, `alpha = 0.81`)
   beats `(r, alpha) = (0.118, 0.85)` under NCD.
2. The same two-stage rule beats `(0.105, 0.95)` under NCD.
3. `(0.105, 0.93)` beats `(0.105, 0.95)` under NCD95.

The subgroup split is label partitions versus proxy partitions. A generalisation claim is allowed
only for hypothesis 1 if it is Holm-significant on new graphs and no subgroup has a
Holm-significant loss.

## WP2: operational anchor

Truths are `NCD(a)` for `a` in `{0.50, 0.60, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 0.97}`,
with `mu = 0.10` and `r_ref = 0.10/a`. The locked Stage 1 rule is primary; `r = r_ref` is
sensitivity-only. On calibration graphs, compare `d(a) = d0` with
`d(a) = d0 + d1(a - 0.85)` by grouped CV. Held-out comparisons are fitted rule
`a - d_hat(a)` versus no discount `a` and the panel best constant. No result is extrapolated
outside the listed anchors.

## WP3: mixed damage

Experiment C composes B then A in that fixed order on the grid
`keep in {1.0, .8, .6, .4, .2}` by `new_pages in {0, .1, .2, .3, .4, .5, .6}`,
excluding `(1.0, 0)`. The reverse order is a sensitivity run on 12 calibration graphs.
Rules tried are C0 constant, C1 tent in Delta-h, C2 three-step Delta-h_raw, C3 tents/steps
in Delta-h_raw^rel, C4/C5 additive hinges, C6 min(C1,C2), C7 3x3 cell table, C8 gradient
boosting ceiling, and the three single-snapshot counterparts. Every attempted rule is counted,
including failures. The preregistered mixture contrast is restricted to damaged instances with
`|Delta-h| < 0.02` and oracle alpha below `0.80`: C4/C5/C6/C7 must beat C1 in mean gain
over C0, separately for pure A, pure B, and mixed groups.

## WP4: realistic baselines

Rules and parameters are fixed from WP3. B1 varies independent baseline damage
(`keep = .9, .8, .6`, with `new_pages = .1, .3` sensitivity); B2 varies
`gamma = .05, .10, .20, .40` and baseline crawl loss `keep = 1.0, .9, .7`; B3 compares
reset, best-so-far, median-last-k, and exponential-average policies on eight-snapshot sequences.
The report records the largest tolerable baseline damage and gamma for which the baseline rule
still beats the locked constant. If no regime passes, the result is negative.

## WP5 and WP6

WP5 is validated by formula-equality tests, consistency tests, and a 200k-arc timing smoke test.
The CLI supports only alpha references validated by WP2; otherwise it raises a clear error.
Baseline use is optional only if the WP3/WP4 decision rules pass. WP6, if run, is descriptive
and does not promote a rule.

## Compute contingencies

If downloads, licences, or compute prevent a target panel from finishing, the run is labelled
`reduced` or `skipped`, exact commands and wall-clock time are recorded in `RESUME.md`, and no
unobserved graphs or levels are extrapolated.
