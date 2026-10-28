# Resume instructions

The canonical directory is `ncdcal_final_-_documentation/`.

```bash
cd ncdcal_final_-_documentation
python3 -m venv .venv
.venv/bin/pip install numpy scipy pandas scikit-learn
.venv/bin/python tests/test_lib.py
.venv/bin/python tests/test_stage2.py
.venv/bin/python tests/test_nonmono.py
.venv/bin/python tests/test_autotune.py
.venv/bin/python verify_v2.py
.venv/bin/python check_build.py
```

Before any new held-out outcome, keep the committed `PREREG_v2.md`. If verified raw datasets are
obtained, add only prepared `data/real2/*.npz` and provenance, then implement the `real2` driver
with seeds `17100 + j` and run the preregistered tests. For WP2, add the `stage2x` sweep under
`results/stage2x/`; for WP3/WP4, keep results in new directories only.

The paper build currently requires `xelatex`, and the documentation build requires `pandoc`.
Use `PATH="$PWD/.venv/bin:$PATH" bash build_papers.sh` after installing the TeX toolchain, then
`bash build_docs.sh`. Finally copy changed canonical files to the repository root and verify each
with `diff -u`.

Continuation pass status:
- WP5 strengthening is complete in the working tree: independent dense-statistics tests, anchor
  validation, cleaning/degenerate cases, CLI coverage, and the 200,000-arc timing smoke test.
- `verify_v2.py` now recomputes per-base-graph comparisons from stage2 NPZ files and checks the
  generated mismatch/mixture tables, macros, and documentation headline rows without importing
  analysis modules.
- Build checks fail on LaTeX errors, overfull boxes, duplicate labels, multiply-defined labels,
  undefined references/citations, and paper-asset writer collisions.
- Toolchain/network probe: XeLaTeX, Pandoc, and FreeSerif are available; graph2gauss PubMed/DBLP
  raw URLs are reachable, while SNAP, LAW, and OGB hosts were unreachable by DNS.
- WP1 candidate check is reduced: graph2gauss DBLP/PubMed were downloaded to ignored `raw_data/`,
  both had reverse-arc fraction 1.000 and were dropped; no `real2` graph or held-out outcome was
  computed. Continue with WP2 only if time remains.
- WP3 primitive checks are complete: endpoint equivalence, deterministic replay, and small
  expectation checks pass. Experiment C itself was not run; WP2 and WP4 remain skipped.

WP3 continuation (2026-10-27):

```bash
.venv/bin/python mixed_experiment.py --panel calib,test,topo --limit 1
.venv/bin/python mixed_experiment.py --panel real --limit 1 --cnr-cells 1
```

`mixed_experiment.py` is the additive Experiment C driver.  It writes
`results/mixed/{instances.csv,surfaces.npz,rule_results.csv,group_results.csv,contrast_results.csv,summary.json}`
and preserves panel-specific reduced outputs in `results/mixed/synthetic/` and
`results/mixed/real/`. The measured `calib_000` one-cell probe was 0.74 s; the
complete synthetic run took 623.5 s and produced 4,080 instances over 120 base
graphs. The `real` output contains exactly one reduced `cnr2000` instance and
took 348.5 s; the four small real graphs were not run. Reverse-order sensitivity
took 60.0 s on 12 calibration graphs and is preserved under
`results/mixed/reverse/`.

WP1/WP2/WP4 continuation (2026-10-27):

```bash
~/venv/bin/python prepare_real2.py
~/venv/bin/python real2_analysis.py
~/venv/bin/python stage2x_analysis.py
~/venv/bin/python baseline_experiment.py
```

`data/real2/` contains eight prepared sources. The completed real2 Stage 2
outcomes cover all eight new graphs. Stanford took 4,006.5 seconds and
NotreDame 3,526.0 seconds; 58 per-instance checkpoints were written under
`results/stage2/real2_instances/`. The first
large real2 Stage 1 attempt was stopped after 59 minutes without an output.

`stage2x.py` completed `calib`, `test`, `topo`, and `real` in
`results/stage2x/`; `stage2x_analysis.py` records exact zero anchor errors and
the supported range `[0.50, 0.97]`. The full synthetic Experiment C run
completed in 623.5 seconds and wrote the complete mixed outputs. The full WP4
B1--B3 panels wrote 720, 1,440 and 480 rows respectively; no baseline rule is
promoted.

Pass 4 completed items (2026-10-28):

```bash
~/venv/bin/python real2_pooled.py
~/venv/bin/python real2_analysis.py
```

The pooled real2 analysis reproduces the reviewer values from the raw NPZ files:
H1 -0.00214 (4/11, exact p=0.4648), H2 +0.07601 (11/11, exact
p=0.0010; anchor-mismatched descriptive), and H3b -0.00372 (2/11,
exact p=0.1016). It writes `results/stage2/real2/pooled.json`,
`pooled_per_graph.csv`, and `pooled_leave_one_out.csv`. The six-new-graph
Holm family remains untestable under Amendment 1. `real2_analysis.py` now
labels the six-graph percentile range separately and adds a 10,000-resample
bootstrap CI of the mean with fixed seed 17100.

WP3/WP2 pass 4 completion (2026-10-28):

```bash
~/venv/bin/python mixed_contrast_analysis.py
~/venv/bin/python mixed_experiment.py --panel real --limit 0 --cnr-cells 4 \
  --output-dir results/mixed/real_full
~/venv/bin/python stage2x_verdict.py
```

The graph-unit mixed contrast writes `results/mixed/contrast_per_graph.csv`
and `contrast_per_graph_summary.json`, with pure-A/pure-B/mixed per-graph
rows and exact Wilcoxon tests for C4/C5/C6/C7 versus C1. The full real panel
took 257.8 s and produced 140 instances over `cnr2000`, `cora_ml`, `citeseer`,
`cora_full`, and `wikics` in the additive `real_full` directory. The fitted
anchor discount verdict is not promoted; its held-out means are test -0.00076
(14/32), topo -0.00042 (6/24), and real +0.00024 (2/5).

WP4 full baseline pass (2026-10-28):

```bash
~/venv/bin/python baseline_experiment_v2.py --panels calib,test --output-dir results/baselines_v2
~/venv/bin/python baseline_experiment_v2.py --panels topo --output-dir results/baselines_v2
~/venv/bin/python baseline_evolved.py
~/venv/bin/python baseline_policy.py
```

B1 covered all synthetic calibration, test, and topology graphs in 9.8 s
(720 rows); B2 covered 1,440 evolved-baseline regimes in 20.2 s; B3 covered
480 policy rows. Mean gains were +0.00012, -0.00121, and +0.00002 respectively,
so no baseline rule is promoted.
