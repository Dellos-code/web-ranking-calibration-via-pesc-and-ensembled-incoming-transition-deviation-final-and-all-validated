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
outcomes cover Cornell, Texas, Wisconsin, Film, Email-Eu-core, and OGBN-Arxiv;
the six-graph subgroup is underpowered by Amendment 1. Stanford was stopped
after 32.9 minutes without an output and NotreDame was not run. The first
large real2 Stage 1 attempt was stopped after 59 minutes without an output.

`stage2x.py` completed `calib`, `test`, `topo`, and `real` in
`results/stage2x/`; `stage2x_analysis.py` records exact zero anchor errors and
the supported range `[0.50, 0.97]`. The full synthetic Experiment C run
completed in 623.5 seconds and wrote the complete mixed outputs. The reduced
WP4 B1 run wrote nine rows in 3.0 seconds; B2 and B3 remain outstanding.
