# V2 extension report

## Status and evidence

| WP | Status | Outcome |
|---|---|---|
| WP0 | done | Corrections and generated A/B mixture and anchor-matching outputs added to both manuscripts and both documentation sources. |
| WP1 | reduced | Eight sources were prepared, but only six new graphs completed the locked Stage 2 outcome before the large web graphs exceeded the measured session budget. The four Geom-GCN graphs, Email-Eu-core and ogbn-arxiv were analysed; Stanford was stopped after 32.9 minutes without an output and NotreDame was not started. The six-graph subgroup is inferentially untestable under Amendment 1. |
| WP2 | reduced | `stage2x` completed the full 64-graph calibration, 32-graph test, 24-graph topology, and five-graph real panels. The fitted anchor discount fails the registered promotion rule: test -0.00076 (14/32), topo -0.00042 (6/24, exact p=0.0340), real +0.00024 (2/5), and pooled held-out -0.00054 (22/61). Anchor reproduction errors are exactly 0.0 at 0.85 and 0.95; the supported range remains 0.50--0.97. |
| WP3 | done | Additive `mixed_experiment.py` completed Experiment C for 4,080 instances over 120 synthetic base graphs and the full five-graph real panel (140 instances, 257.8 s). The graph-unit contrast, pure-A/B/mixed splits, and reverse-order sensitivity are reported; the pre-registered contrast does not promote C4-C7 over C1. |
| WP4 | reduced | A B1 damaged-baseline smoke panel (three calibration graphs, three independent baseline keep levels, nine rows) ran using the fixed WP3 C1 fit. B2 evolved baselines and B3 policies were not run, so no baseline rule is promoted. |
| WP5 | done | `ncd_autotune.py`, strengthened independent tests, build checks, README recipe and independent v2 verification added. |
| WP6 | skipped | No dangling-treatment re-run was performed. |

## WP0 headline evidence

The generated `V2_WP0_mixture_table.md` and `paper/generated/{en,gr}/tab_mixture.tex` show that
both perturbations change more than one structural mechanism. In the existing mechanism replay,
`h_raw` changes by less than two hundredths from A1.0 to A0.2 while it rises materially under B.
The anchor-matched statement is therefore conditional on the reference damping; no universal
reference-free discount is claimed.

## WP5 validation

The locked Stage 1 formula, supported anchor values, diagnostics, baseline warning, mixed-damage
primitive and all existing tests were executed successfully. The strengthened autotune test includes
an independent dense NumPy implementation, duplicate/self-loop and degenerate cases, CLI coverage,
and a 200,000-arc timing smoke test. `verify_v2.py` reads each stage2 NPZ directly, rebuilds
per-graph means, recomputes exact Wilcoxon comparisons, and checks the generated mismatch/mixture
tables, macros and documentation headline rows without importing the analysis modules.

## WP3 continuation evidence

The one-cell `calib_000` probe took 0.74 s. The complete synthetic command
`mixed_experiment.py --panel calib,test,topo --limit 0` took 623.5 s and
produced 4,080 Experiment C instances over 120 base graphs. Outputs are under
`results/mixed/`; `contrast_results.csv` applies the preregistered filter.
The reverse-order sensitivity on 12 calibration graphs took 60.0 s and is
preserved under `results/mixed/reverse/`. The earlier `results/mixed/real/`
output remains a reduced one-instance `cnr2000` result; the complete additive
panel is under `results/mixed/real_full/` with 140 instances over five graphs
and runtime 257.8 s. The four small real graphs and all four required
`cnr2000` cells are included there.

The instance-level filtered contrast remains the preregistered file:
test mixed (52 instances) has C1 0.00515, C4 0.00579, C6 0.00580,
C7 0.00543, C5 0.00285; topo mixed (30 instances) has C1 0.00372,
C4 0.00341, C6 0.00376, C7 0.00076. The new graph-unit analysis is in
`results/mixed/contrast_per_graph_summary.json`; no C4-C7 rule meets the
registered mean/p-value requirements in both calibration and pooled held-out
data, so the verdict is not promoted.

## WP1 continuation evidence

The prepared `real2` panel contains four Geom-GCN label partitions, Email-Eu-core
department labels, OGBN-Arxiv subject labels, and Stanford/NotreDame SNAP web
graphs with proxy partitions. BerkStan was excluded by the 400,000-node /
5-million-arc cap. Six new graphs completed the locked Stage 2 outcome. H1 has
mean gain `-0.00537` (2/6 wins, exact `p = 0.5625`, percentile range
`[-0.03173, 0.01136]`); H2 is `+0.06868` (6/6, `p = 0.03125`); H3b is
`-0.00475` (1/6, `p = 0.15625`). The six-graph subgroup is underpowered by
Amendment 1. The Stanford run produced no output after 32.9 minutes and
NotreDame was not started.

## WP2 continuation evidence

The calibration surface run took 1,489 s, test 336 s, topology 324 s, and the
five-graph real panel 2,233 s; cnr2000 alone took 1,854 s. Anchor reproduction
errors were exactly 0.0 at both 0.85 and 0.95. The fitted constant discount is
0.05592 with linear slope -0.05790, supported only on the registered 0.50--0.97
range. The formal table is in `results/stage2x/verdict.json`: calibration CV
mean +0.00308 (45/64, p<0.00001), while held-out test is -0.00076 (14/32),
topo is -0.00042 (6/24, p=0.0340), real is +0.00024 (2/5), and pooled
held-out is -0.00054 (22/61). The discount is not promoted because it fails
the required held-out gain/p-value rule and has a significant topo loss.

## WP4 continuation evidence

The reduced B1 panel produced nine rows in 3.0 s and mean gain +0.00506 over the
constant. B2 evolved baselines and B3 policies were not run; baseline use remains
disabled.

## Runtime and deviations

The preregistration and WP0 asset generation completed in seconds once the local Python environment
was restored. The first large real2 Stage 1 attempt produced no graph output after 59 minutes and was
stopped; the Stanford real2 Stage 2 attempt likewise produced no output after 32.9 minutes. XeLaTeX,
Pandoc and FreeSerif remain available. The new WP1/WP2/WP3/WP4 numbers have not been integrated into
the papers, so this remains a reduced implementation for the unrun large-graph and baseline work.

## WP1 candidate verification

The reachable graph2gauss raw files were downloaded only to ignored `raw_data/`:

| Candidate | SHA-256 | n | arcs | reverse-arc fraction | labels | outcome |
|---|---|---:|---:|---:|---:|---|
| CitationFull DBLP | `46da600a15c0680d967b3c9ac83979e3c44dfe5a0ef86e1c42a94d8dee1c1da4` | 17,716 | 105,734 | 1.000 | 4 | dropped: essentially symmetric |
| CitationFull PubMed | `7e671409fd689d45ef0715b75be43cacbf416e14d4d84beadb5965ef2e5d028f` | 19,717 | 88,648 | 1.000 | 3 | dropped: essentially symmetric |

Source URLs were `https://raw.githubusercontent.com/abojchevski/graph2gauss/master/data/dblp.npz`
and `https://raw.githubusercontent.com/abojchevski/graph2gauss/master/data/pubmed.npz`.
Both files contain topic labels with no empty ids; the graph2gauss repository is MIT-licensed.
The source URLs, access date (2026-10-27), hashes and directedness check are recorded here rather
than treating these symmetric matrices as directed citation networks. SNAP, LAW and OGB probes
failed at DNS resolution, so they were not substituted or symmetrised.

## Acceptance checklist

- [x] No existing result file or locked rule changed; all new result-related outputs are additive.
- [x] `PREREG_v2.md` was committed before new held-out outcomes.
- [x] Existing tests and new tests pass; `verify_v2.py` passes.
- [ ] No number in any `.tex`/`.md` was typed by hand. Existing legacy text predates this extension;
      this remains to be checked after integration.
- [x] Papers and docs were rebuilt with 0 errors and 0 overfull boxes (done in the review, see below; the original session had no `xelatex`/`pandoc`).
- [x] Negative and null results are reported with the same prominence as positive ones.
- [x] WP0 corrections were applied to the source documentation and manuscripts.
- [x] Claims about unrun real2, mixture and baseline experiments are explicitly limited.
- [x] Top-level copies are byte-identical to the subfolder files.

## What the evidence now supports and what it still does not

The evidence supports the existing locked defaults, the clarification that A and B are mixtures,
the completed synthetic mixed-damage surfaces, and a reproducible NCD(a) surface driver whose
anchor checks pass. It does not support generalisation to more real networks, a promoted
operational discount, a promoted mixture-aware or realistic-baseline rule, or a dangling-page
sensitivity conclusion. The new results are not yet wired into either manuscript.

## Review corrections (applied after the first agent run)

- `wp0_tables()` used the unperturbed A1.0 start values for the B row; the B row of
  `tab_mixture.tex` and `V2_WP0_mixture_table.md` now starts from B0.1 (`delta` 0.185, `zeta` 0.207,
  `h` 0.644, `h_raw` 0.707).
- The placeholder headline row against $(0.105,0.95)$ in both documentation files now carries the
  generated numbers and the anchor caveat; the unused macro `WpZeroNcdVsNg` is used in the text.
- New generated table `tab:mismatch` (`paper/generated/{en,gr}/tab_mismatch.tex`,
  `V2_WP0_mismatch_table.md`) shows the matched and mismatched anchor comparisons on all held-out panels;
  the sentence that pointed to `tab:pipe` for mismatch losses now points to it.
- The tool's pairing at the 0.95 anchor (Stage 1 rule $r$ with $\alpha=0.93$) is now supported by the generated data: the
  calibration-optimal constant for that pairing under NCD95 is 0.920 (0.0002 $\tau$ above 0.93), and the pairing beats the
  [NG13] $(0.105,0.95)$ on `test`, `scale` and `topo` (`V2_WP0_mismatch_table.md`).
- `METHOD_DISCLOSURE_EN_GR.{md,txt}` now states only what exists in this version.
- PDFs rebuilt (papers 24 and 27 pages, documentation 15 and 18 pages) and root-level copies synchronised.
