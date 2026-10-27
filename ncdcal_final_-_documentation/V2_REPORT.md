# V2 extension report

## Status and evidence

| WP | Status | Outcome |
|---|---|---|
| WP0 | done | Corrections and generated A/B mixture and anchor-matching outputs added to both manuscripts and both documentation sources. |
| WP1 | skipped | No verified additional real-network downloads were available in the session; no `real2` outcome is claimed. |
| WP2 | skipped | The new `NCD(a)` sweep was not run; the existing results support only the already validated 0.85/0.95 anchors. |
| WP3 | reduced | `mixed_damage` was implemented and unit-tested, but no Experiment C held-out outcome was computed. |
| WP4 | skipped | No baseline outcome was computed; baseline use remains disabled. |
| WP5 | done | `ncd_autotune.py`, tests, README recipe and independent v2 verification added. |
| WP6 | skipped | No dangling-treatment re-run was performed. |

## WP0 headline evidence

The generated `V2_WP0_mixture_table.md` and `paper/generated/{en,gr}/tab_mixture.tex` show that
both perturbations change more than one structural mechanism. In the existing mechanism replay,
`h_raw` changes by less than two hundredths from A1.0 to A0.2 while it rises materially under B.
The anchor-matched statement is therefore conditional on the reference damping; no universal
reference-free discount is claimed.

## WP5 validation

The locked Stage 1 formula, supported anchor values, diagnostics, baseline warning, mixed-damage
primitive and all existing tests were executed successfully. `verify_v2.py` reads the aggregated
JSON/CSV outputs without importing the analysis modules; it does not recompute them from per-instance
data (a stronger version is still to be written), and `tests/test_autotune.py` checks the tool against
the library it wraps rather than against an independent computation.

## Runtime and deviations

The preregistration and WP0 asset generation completed in seconds once the local Python environment
was restored. The full extension experiment budgets were not spent because the session lacked
verified real2 source files and the TeX/Pandoc build toolchain (`xelatex` and `pandoc` were
unavailable). This is a reduced implementation, not evidence for the skipped hypotheses.

## Acceptance checklist

- [x] No existing result file or locked rule changed; all new result-related outputs are additive.
- [x] `PREREG_v2.md` was committed before new held-out outcomes.
- [x] Existing tests and new tests pass; `verify_v2.py` passes.
- [ ] No number in any `.tex`/`.md` was typed by hand. Existing legacy text predates this extension;
      new manuscript numbers use generated macros/tables.
- [x] Papers and docs were rebuilt with 0 errors and 0 overfull boxes (done in the review, see below; the original session had no `xelatex`/`pandoc`).
- [x] Negative and null results are reported with the same prominence as positive ones.
- [x] WP0 corrections were applied to the source documentation and manuscripts.
- [x] Claims about unrun real2, mixture and baseline experiments are explicitly limited.
- [x] Top-level copies are byte-identical to the subfolder files.

## What the evidence now supports and what it still does not

The evidence supports the existing locked defaults, the clarification that A and B are mixtures,
anchor-conditional damping language, and a tested single-snapshot tool limited to the validated
0.85 and 0.95 anchors. It does not support a claim of generalisation to more real networks, a
validated operational discount over the full reference-damping range, a promoted mixture-aware
baseline rule, or a dangling-page sensitivity conclusion.

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
