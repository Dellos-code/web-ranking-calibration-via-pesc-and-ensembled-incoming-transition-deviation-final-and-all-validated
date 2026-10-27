# MASTER PROMPT — v2 extension of the NCDawareRank calibration project

You are a senior research engineer and statistician extending an existing, fully validated research
package. Read this whole prompt before touching anything. Your job is to run additional experiments,
add the results to the code, the documentation and both manuscripts (English and Greek), and to
report honestly, including negative results. You are NOT here to make the project look better than
the evidence allows.

---------------------------------------------------------------------------------------------------

## 0. Repository, canonical folder, and ground rules

* Repository: `Dellos-code/web-ranking-calibration-via-pesc-and-ensembled-incoming-transition-deviation-final-and-all-validated`.
* **The canonical project is the subfolder `ncdcal_final_-_documentation/`.** The repository root also
  holds copies of some files (`DOCUMENTATION_EN/GR.md|pdf`, `paper_en/gr.tex|pdf`,
  `stage2_nonmonotone_search.py`, the NG13 PDF). Work only in the subfolder. At the very end, copy the
  changed top-level files from the subfolder to the root and verify with `diff` that they are
  byte-identical.
* Work on a new branch `v2-extensions`. One commit per work package (WP). Never force-push, never
  rewrite history. Open one pull request at the end.
* Read first, in this order: `README.md`, `DOCUMENTATION_EN.md` (all of it, especially §3 design
  principles, §8 panels/seeds/perturbations, §9 statistical methodology, §11 reproduction),
  `paper/paper_en.tex` (all of it), `ncd_calibration_lib.py`, `run_experiments.py`,
  `stage2_analysis.py`, `stage2_nonmonotone_search.py`, `make_paper_assets.py`, `tests/`.
* **Additive only.** Do not modify any existing file under `results/`, any existing seed, any existing
  panel definition (`calib`, `test`, `scale`, `topo`, `real`), or any locked rule. New experiments go to
  new directories (`results/stage2x/`, `results/mixed/`, `results/real2/`, …) and new panel names.
* **Locked (do not change):**
  - Stage 1: `r = clip(0.242 − 0.143·P̄, 0.01, 0.99)`.
  - Stage 2: the constant `α = 0.810` under NCD (`0.930` under NCD95).
  - The decision rule of §3 below.
  - The documentation titles ("Two-Stage Calibration of NCDawareRank — Technical Documentation" /
    "Βαθμονόμηση δύο σταδίων του NCDawareRank — Τεχνική τεκμηρίωση"). Do not add or restore the words
    "thesis guide" / "οδηγός διπλωματικής". Leave the README row text as it is.
  - A locked rule may change only if a new rule passes the pre-registered decision rule of §3 on
    held-out data. Even then, keep the old rule as the documented default and present the new one as an
    option with its evidence.
* **Unit of replication = the base graph.** Perturbed instances of one base graph are never treated as
  independent. All splits, cross-validation folds, paired tests and bootstraps are by base graph
  (existing code does this; reuse it).
* **No hand-typed numbers.** Every number in the papers and docs must come from a result file through
  `make_paper_assets.py` (LaTeX macros and generated tables). Extend that script; never type a result
  into a `.tex` or `.md` file.
* **Determinism and restartability.** Every base graph has its own seed; perturbations are drawn from
  the same `numpy.random.Generator` in a fixed order. One compressed `npz` per base graph; the driver
  skips existing files. Use the `spawn` multiprocessing context (the project hit a fork-after-OpenMP
  deadlock before). Use at most as many workers as the machine has cores; log to `logs/`.
* **Time and compute.** You may have a limited session. Work in the priority order below; after every
  WP, commit, and update `RESUME.md` with the exact commands to continue. If a run cannot finish, run a
  clearly labelled reduced version (fewer graphs or levels), say so in the text, and never extrapolate.
  Measure and record wall-clock time for each run.
* **Honesty.** If something cannot be verified (a dataset is unavailable, a label provenance is unclear,
  the network is blocked), do not guess. Skip it, record why in `V2_REPORT.md`, and continue.

---------------------------------------------------------------------------------------------------

## 1. What the project has established (so you know what to protect and what is open)

Setting: NCDawareRank with `η = (1−r)α`, `μ = rα`. Stage 1 chooses `r` from the mean block escape
probability `P̄`. Stage 2 chooses the total damping `α`. The score is Kendall τ between the ranking of
the perturbed graph and a reference ranking of the complete graph. Perturbations follow [NG13]:
Experiment A keeps a random fraction of arcs; Experiment B removes each in-arc (probability 0.9) of a
random fraction of pages. Truths: NCD (reference NCDawareRank at `(r, α) = (0.118, 0.85)`), NCD95
(reference at `(0.105, 0.95)`), TOT, POP, Q, SELF. Panels: `calib` (64 graphs, fitting), `test` (32),
`scale` (12), `topo` (24), `real` (5: CiteSeer, cnr-2000, Cora, Cora-ML, WikiCS).

Established results you must keep consistent:

1. Neither `P̄`, `h`, `h_v` nor any tested combination predicts the oracle parameters out of sample
   without a baseline; the τ-optimised rules collapse to constants (`r ≈ 0.18–0.24`, `α = 0.810`).
2. The two-stage setting beats the static `(0.118, 0.85)` under NCD: test +0.0110 τ (31/32 graphs,
   p = 9.3e-10), scale +0.0102, topo +0.0036, real +0.0017 (2/5 graphs, p = 0.81). It beats the
   original `(0.105, 0.95)` of [NG13] by +0.074 τ on test (32/32), +0.052 on topo, +0.085 on real (5/5;
   p = 0.0625 is the smallest value an exact two-sided Wilcoxon test can reach with 5 graphs). It loses
   under NCD95 and Q when the constant is mismatched with the reference.
3. **Anchoring:** the oracle damping on the unperturbed graph is 0.852 (NCD), 0.950 (NCD95), 0.547 (TOT),
   0.767 (POP), 0.971 (Q). The best constant is the reference's damping minus a small discount (0.04 for
   a 0.85 reference, 0.02 for a 0.95 reference). Under NCD95, the matched constant 0.93 beats 0.95 by only
   +0.0011 τ on test (19/32, p = 0.062).
4. `h` falls under A and rises under B while `α*` falls under both (inverted V around the graph's own
   unperturbed `h₀`). Both A and B are actually **mixtures** of two kinds of damage: A at 20% of arcs
   raises the dangling share δ from 0.173 to 0.483 and the share ζ of pages without in-links from 0.156
   to 0.506; B at 60% raises δ to 0.298 and ζ to 0.461. They differ in proportion, which is why `h`
   moves in opposite directions.
5. With the graph's own unperturbed value as baseline (non-deployable), a tent in `Δh = h − h₀` gains
   +0.0018 τ in CV and +0.0006 held-out (p = 0.018); a step rule in `Δh_raw` gains +0.0010 held-out
   (p = 6e-5). The plain `h_raw` is nearly blind to A (change −0.010, within-graph correlation with
   `α*` −0.10) and sees B (+0.148, −0.78); `h_raw^rel` and `h_raw^H` rise under both. The oracle
   headroom over the constant is only +0.0052 τ (calibration), so any `α`-selection gain is small.
6. Open questions this extension must settle: (i) generalisation to more real networks; (ii) an
   operational rule for any reference damping; (iii) the mixed-damage case; (iv) realistic baselines
   (damaged or evolved); (v) a usable tool.

---------------------------------------------------------------------------------------------------

## 2. Work packages, in priority order

### WP0 — Text corrections (certain, cheap; do first, no new experiments)
In both papers and both documentation files, using generated macros/tables where numbers appear:
1. Describe Experiments A and B as **mixtures** with different proportions. Add a small generated table
   of `δ`, `ζ`, `h`, `h_raw` (unperturbed → most severe level) for A and B (data: `nonmono_summary.json`,
   `mechanism/trajectories`). Remove or rephrase any wording that calls A "pure dangling" or B "pure
   in-link loss".
2. Add the comparison with `(0.105, 0.95)` to the documentation headline table (§1.2) next to the one
   with `(0.118, 0.85)`, with the anchor-conditional caveat in the same cell or the next line.
3. Add one explicit sentence and a small table (from `pipelines.csv`/`summary.csv`) for the
   **anchor-matched** claim: with the reference at 0.85 use 0.81; with 0.95 use 0.93; mismatch costs
   −0.016 to −0.048 τ on test. State that the choice of reference is a methodological decision.
4. State what [NG13] actually did: Table 2 compares `(η, μ) = (0.75, 0.10)` with PageRank at 0.85 and
   `(0.85, 0.10)` with PageRank at 0.95; the ranking experiments run NCDawareRank at total damping 0.95
   against PageRank at 0.85; no experiment compares 0.85 with 0.95 for ranking quality; the SELF
   protocol makes the reference move with the candidate.
5. State the p-value floor of an exact two-sided Wilcoxon test: 0.0625 for n = 5, 0.0078 for n = 8.
6. Clarify that `h_raw` alone is nearly blind to Experiment A.

### WP1 — Real-network panel expansion (`real2`) — HIGHEST PRIORITY
Goal: turn the weakest claim ("works on real networks", currently 5 graphs, p = 0.81) into a testable one.

* Add a **new** panel `real2` (seeds 17100 + j; do not touch `real`). Target at least 8 new directed
  graphs, ideally 10–12, each with a block partition. Use the existing preparation code pattern
  (`prepare_real_graphs.py`); put raw downloads in `<repo>/raw_data/` (git-ignored; do not commit files
  larger than 50 MB) and commit only the prepared `data/real2/*.npz` plus a `provenance.json` with, for
  every graph: source URL, access date, licence, SHA-256 of the raw file, node and arc counts, how
  blocks were obtained, and whether the partition is a **label partition** (topic, category,
  department) or a **proxy** (detected communities).
* Candidates, in this order. **Verify each one (directedness, label provenance, size, licence) before
  using it; drop what fails.**
  1. CitationFull DBLP and PubMed (same distribution family as the existing g2g files).
  2. ogbn-arxiv (directed citations, subject-area labels).
  3. SNAP email-Eu-core (directed, departments as labels; small).
  4. SNAP web graphs with no host labels: web-NotreDame, web-Stanford; web-BerkStan only if time
     allows. For these, use the same contiguity-constrained greedy directed-modularity partition that
     was used for cnr-2000 and label them as proxy partitions.
  5. Any other directed graph with a trustworthy partition that you can justify in one sentence.
* Size cap: up to about 400,000 nodes and 5 million arcs for the main run; `reps = 1` above 100,000
  nodes, otherwise `reps = 2`. cnr-2000 alone dominated the original 101-minute Stage 2 run, so
  budget accordingly.
* Run the **locked** Stage 1 and Stage 2 pipeline on every graph (same levels, same truths available
  for real graphs: NCD, NCD95, TOT). No refitting of anything.
* **Pre-registered tests** (write them to `PREREG_v2.md` and commit it before computing any `real2`
  outcome): per-graph mean τ differences for
  (a) two-stage `(rule r, α=0.81)` vs `(0.118, 0.85)` under NCD;
  (b) two-stage vs `(0.105, 0.95)` under NCD;
  (c) matched NCD95 constant `(0.105, 0.93)` vs `(0.105, 0.95)` under NCD95;
  each on (i) the new graphs only and (ii) all real graphs (old 5 + new). Exact two-sided Wilcoxon,
  bootstrap 95% CI, wins/n, Holm over the three families. Report label-partition graphs and
  proxy-partition graphs separately as a pre-registered subgroup split.
* Exploratory (label as such): which graph properties (n, mean out-degree, unperturbed δ and ζ, `P̄`,
  intra-block arc share) predict the sign of the gain; the existing real panel loses on CiteSeer,
  Cora-ML and WikiCS.
* Claim "generalises to real networks" only if (a) is significant on the new graphs at 0.05 after Holm
  and there is no Holm-significant loss on any subgroup. Otherwise report exactly what happened.

### WP2 — Operational anchor rule: `α_cal = α_ref − d(α_ref)`
Goal: let a practitioner state the damping they would use on a complete graph and get the calibrated one.

* New truth family `NCD(a)`: reference = NCDawareRank on the complete graph at total damping `a`, with
  `μ = 0.10`, i.e. `r_ref = 0.10/a`, for `a ∈ {0.50, 0.60, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 0.97}`.
  (`a = 0.85` and `a = 0.95` must reproduce the existing NCD and NCD95 truths; this is a built-in
  check.)
* Candidate setting: `r` from the locked Stage 1 rule (primary); `r = 0.10/a` as a sensitivity run.
* This needs new τ(α) surfaces (the stored npz files hold τ only, not rankings), so write a new driver
  mode `stage2x` that recomputes the α-sweeps for panels `calib`, `test`, `topo`, `real` (and `scale` if
  time allows) and evaluates all `NCD(a)` truths in the same pass. Output to `results/stage2x/`.
* Fit on `calib` only: the oracle constant `α_c(a)` and the discount `d(a) = a − α_c(a)`. Pre-register
  the forms `d(a) = d₀` and `d(a) = d₀ + d₁(a − 0.85)`; choose by grouped CV on `calib`.
* Held-out evaluation (`test`, `topo`, `real`, `real2` if available): the fitted rule `a − d̂(a)`
  versus (i) no discount `α = a`, (ii) the best constant per panel (oracle for the constant). Report
  τ differences per `a`, with the same statistics as before, and a figure of `d(a)` with CI.
* Output a table "recommended α for reference damping a" and state the range of `a` where it is
  supported. Do not extrapolate outside it.

### WP3 — Mixed damage (Experiment C) and rules that use a baseline
* **Experiment C:** apply B then A to the same base graph (fixed order; a sensitivity run with the
  order reversed on 12 `calib` graphs). Grid: `keep ∈ {1.0, 0.8, 0.6, 0.4, 0.2}` ×
  `new_pages ∈ {0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6}`, excluding `(1.0, 0)`. Cells with `new_pages = 0` are
  pure A, cells with `keep = 1.0` are pure B, the rest are mixed. Implement
  `mixed_damage(g, keep_frac, page_frac, rng, order)` in `ncd_calibration_lib.py` by composing the
  existing `sparsify_arcs` and `newly_added_pages` (do not alter them). Panels: `calib` (reps 1), `test`,
  `topo`, `real` (for cnr-2000 use only `keep ∈ {0.8, 0.4}` × `new ∈ {0.1, 0.4}`); new seeds
  (21000+). Record for every instance `h`, `h_raw`, `h_raw^rel`, `h_raw^H`, `δ`, `ζ`, `P̄`, r and the
  oracle α* under every truth.
* **Rules** (all with `h₀`, `h_raw,0` etc. from the unperturbed graph in the baseline-ideal setting;
  other baselines are WP4):
  - C0: constant 0.810.
  - C1: tent in `Δh` (re-fit); C2: 3-step in `Δh_raw` (re-fit).
  - C3: tent and 3-step in `Δh_raw^rel`.
  - C4: additive hinge `α = clip(α₀ − a·max(0, Δh_raw − t₁) − b·max(0, Δδ − t₂), 0.40, 0.99)`.
  - C5: the same hinge on `(Δh_raw^rel, Δδ)`.
  - C6: `min(C1, C2)`.
  - C7: 3×3 cell table on `(Δh, Δh_raw)`.
  - C8: gradient boosting on all Δ features (non-deployable ceiling).
  - Single-snapshot counterparts for comparison: `h_raw^rel` alone, `(h_raw^rel, δ)`, `(h_raw^rel, ζ)`.
* **Two fits per rule:** (1) on the original pure-A/B `calib` data, evaluated on Experiment C held-out
  (does a rule fitted on pure damage survive mixtures?); (2) on Experiment C `calib`, evaluated on
  Experiment C held-out.
* **Pre-registered mixture-specific contrast:** in cells where `|Δh| < 0.02` but the instance is
  damaged (oracle `α*` below 0.80), C4/C5/C6/C7 must beat C1 (the tent in `Δh`) in mean τ gain over the
  constant. This is the direct test of the hypothesis that `h` cancels under mixtures.
* Report gain per group (pure A, pure B, mixed), not only pooled.

### WP4 — Realistic baselines (the "is this a sub-sub-case?" question)
The previous experiments take the baseline from the **complete** graph. In practice the previous
snapshot is itself incomplete and the network has grown or changed. Quantify how far the baseline rules
survive.
* **B1 damaged baseline:** baseline statistics from an independent perturbation of the same base graph
  with `keep ∈ {0.9, 0.8, 0.6}` (and a variant with `new_pages ∈ {0.1, 0.3}`).
* **B2 evolved baseline:** baseline from an earlier, smaller graph, namely the subgraph induced on the
  first `(1 − γ)` fraction of nodes, `γ ∈ {0.05, 0.10, 0.20, 0.40}` (growth order for `topo`; a fixed
  random node order for the other panels; state which), additionally with its own crawl loss
  (`keep ∈ {1.0, 0.9, 0.7}`). The current graph is the full graph with the damage applied.
* **B3 baseline policy** on simulated sequences of 8 snapshots with mostly increasing damage and
  occasional repair: compare *reset to last snapshot*, *best so far* (smallest `h_raw`),
  *median of the last k*, and *exponential average*. Metric: mean τ gain over the constant.
* Use the rules and parameters fixed in WP3 (no refitting). Output: a map of the regime where a
  baseline rule still beats the constant (largest tolerable `γ`, largest tolerable baseline damage)
  and the recommended baseline policy. If no regime exists, say so plainly.

### WP5 — Practical tool (`ncd_autotune.py`)
A small, tested module and CLI; this is what turns the study into something usable.
* `tune(src, dst, block, alpha_ref=0.85, baseline=None)` returns a dict: `r`, `alpha`, and diagnostics
  (`P̄`, `h`, `h_raw`, `δ`, `ζ`, the displacement bound `α/(1−α)·h_v` at the chosen α), computed in
  `O(|E|+n)` with the library functions (no reimplementation of the chain).
* `alpha` follows WP2 when validated; otherwise it supports only `alpha_ref ∈ {0.85, 0.95}`
  (0.81, 0.93) and raises a clear error for others. A baseline rule is exposed only if it passes the
  pre-registered rule in WP3 and WP4, and is then flagged as optional; otherwise `baseline` is ignored
  with a warning that no validated rule exists.
* Tests: equality with the locked formulas on a toy graph; consistency with `ncd_calibration_lib`;
  a timing smoke test on a 200k-arc graph. Add a short "Practical recipe" box to both papers (≤ 15
  lines) and a usage section in the documentation and README (add only a new row for the tool; do not
  change existing rows).

### WP6 — Optional, only if time remains: dangling-page treatment sensitivity
The papers state as a limitation that dangling pages use uniform rows rather than the block-confined
treatment of [Nik20]. Read [Nik20]; if its definition is ambiguous, implement the simplest
block-confined variant (a dangling page jumps uniformly inside its own block) and label it that way.
Re-run the locked pipeline on `test` and `topo` with the chosen reference rebuilt under the same
treatment, and report whether `α* ≈ α_ref − d` and the sign of the gain survive. Report only.

---------------------------------------------------------------------------------------------------

## 3. Statistics and the decision rule (fixed; reuse existing code)

* Metrics: Kendall τ-b (primary), Spearman ρ (secondary).
* Per-graph means as the unit; paired differences; exact two-sided Wilcoxon signed-rank; bootstrap
  95% CI over graphs; wins/n; Holm correction over each declared family (declare the family size in
  `PREREG_v2.md`).
* **Decision rule for promoting any new rule** over the locked one: mean τ gain ≥ 0.0005 with p < 0.05
  in grouped 8-fold CV **and** on the pooled held-out panels, and no Holm-significant loss on any
  held-out panel. Rules that need a baseline are labelled non-deployable-in-single-snapshot; the
  rule still applies for the "baseline available" setting.
* Write `PREREG_v2.md` **before** running any held-out outcome: hypotheses, every rule with its
  functional form and number of parameters, the grids, seeds, family sizes, thresholds. Commit it first.
  Count and report every rule you tried, including the failures.
* Held-out panels are never used for fitting, model selection, or choosing thresholds.
* Report effect sizes first, p-values second. State the oracle headroom next to every τ gain so the
  reader sees the scale.

---------------------------------------------------------------------------------------------------

## 4. Integration into the papers and the documentation (EN and GR)

For every WP that produced a result (including negative ones):
1. Extend `make_paper_assets.py` to emit the macros and tables (EN and GR) from the result files. New
   macros must have unique names; table numbering must stay consistent (renumber references
   programmatically, not by hand).
2. **Papers** (`paper/paper_en.tex`, `paper/paper_gr.tex`): add concise subsections (Results §8 for
   WP1–WP4, a short methods paragraph for new designs, the "Practical recipe" box for WP5, Limitations and
   Conclusion updates). Keep the net growth small (about four pages in total); push detail into an
   appendix. Update the abstract, introduction and conclusion **only** with numbers backed by macros and
   only where findings changed. Do not overclaim: mirror the wording of the decision rule.
3. **Documentation** (`DOCUMENTATION_EN.md`, `DOCUMENTATION_GR.md`): new module entries (in the same
   style as §5.x), new tests, new files and tables lists, reproduction steps, the headline table rows,
   layout section, runtime table, and the panel/seed table (`real2`, mixed seeds). Rebuild the PDFs.
4. **Figures**: generated by `generate_figures.py`; Greek versions by `fig_lang.py` (add Greek strings).
5. **Greek style:** use the established glossary exactly as it appears in the existing Greek paper:
   «αποκομμένος γραμμικός (clipped-linear)», «μαντείο (oracle)», «κρεμάμενες (αδιέξοδες) σελίδες
   (dangling nodes)», «βέλτιστες παράμετροι αναφοράς». Reuse the existing Greek wording for recurring
   terms rather than inventing new translations. Math inside section titles needs `\texorpdfstring`.
   English and Greek must carry identical numbers (same macros).
6. **Method disclosure:** update `METHOD_DISCLOSURE_EN_GR.md/.txt` with one paragraph describing the AI
   agent's role in this extension. Do not put AI-disclosure text back into the manuscripts.
7. **Builds:** `bash build_papers.sh` and `bash build_docs.sh` must finish with **0 LaTeX errors and 0
   overfull boxes** for all four PDFs (fix Greek table widths with wrapped first columns/smaller
   `\tabcolsep`; documentation uses `\sloppy` and `emergencystretch`).
8. **Tests and verification:** all existing tests must still pass; add tests for every new module; write
   `verify_v2.py` that recomputes every new headline number from the raw `npz`/CSV files **without
   importing the analysis code** (as `verify_stage2.py` does), and make it pass.
9. Update `README.md` quick start with the new test and verification commands (add rows; do not edit
   existing wording).

---------------------------------------------------------------------------------------------------

## 5. Deliverables and final report

* Branch `v2-extensions` with one commit per WP, then a pull request.
* `PREREG_v2.md` (committed before outcomes), `RESUME.md`, `V2_REPORT.md`.
* `V2_REPORT.md` contains, per WP: status (done / reduced / skipped and why), pre-registered criteria and
  whether each passed, the 3–5 headline numbers with CIs, runtime, files changed, deviations from this
  prompt, and the list of things you could not verify. End with a section "What the evidence now
  supports and what it still does not".
* Updated: code, results (new directories only), figures, `paper/*.tex|pdf`, `DOCUMENTATION_*.md|pdf`,
  `README.md`, `METHOD_DISCLOSURE_EN_GR.*`, plus the synchronised top-level copies.
* Do not create a release zip unless asked. Do not commit raw downloads above 50 MB.

## 6. Acceptance checklist (tick every line in `V2_REPORT.md`)

- [ ] No existing result file or locked rule changed (`git diff --stat` on `results/` shows additions only).
- [ ] `PREREG_v2.md` committed before any held-out outcome was computed.
- [ ] All tests (old and new) pass; `verify_v2.py` passes.
- [ ] No number in any `.tex`/`.md` was typed by hand.
- [ ] Papers and docs, EN and GR: 0 errors, 0 overfull boxes, identical numbers.
- [ ] Negative and null results are reported with the same prominence as positive ones.
- [ ] WP0 corrections applied everywhere (A/B described as mixtures; comparison with (0.105, 0.95);
      anchor-matched constants; what [NG13] did and did not test; Wilcoxon p-floor).
- [ ] Claims about real networks, mixtures and baselines use exactly the decision-rule wording.
- [ ] Top-level copies are byte-identical to the subfolder files.

Begin with WP0 and the reading list, write `PREREG_v2.md`, then WP1, WP2, WP3, WP4, WP5, and WP6 if time
remains. Start by replying with a one-paragraph plan and your estimate of the compute each WP needs;
then execute.
