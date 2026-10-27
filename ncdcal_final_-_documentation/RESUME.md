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
```

Before any new held-out outcome, keep the committed `PREREG_v2.md`. If verified raw datasets are
obtained, add only prepared `data/real2/*.npz` and provenance, then implement the `real2` driver
with seeds `17100 + j` and run the preregistered tests. For WP2, add the `stage2x` sweep under
`results/stage2x/`; for WP3/WP4, keep results in new directories only.

The paper build currently requires `xelatex`, and the documentation build requires `pandoc`.
Use `PATH="$PWD/.venv/bin:$PATH" bash build_papers.sh` after installing the TeX toolchain, then
`bash build_docs.sh`. Finally copy changed canonical files to the repository root and verify each
with `diff -u`.
