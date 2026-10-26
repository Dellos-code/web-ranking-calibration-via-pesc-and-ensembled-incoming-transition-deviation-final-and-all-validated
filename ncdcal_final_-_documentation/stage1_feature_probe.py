"""
stage1_feature_probe.py
=======================
Diagnostic (not part of the proposed method): does an observable measure of
BLOCK INFORMATIVENESS predict r* where Pbar_esc^(M) does not?

Motivation (variance decomposition of r*): across graphs, r* is driven by the
dispersion of latent page quality and by the share of it that is block-level,
i.e. by how much a page's block says about the page.  That is the quantity an
empirical-Bayes shrinkage weight depends on, and M is a shrinkage-toward-block
operator.  Pre-declared candidate statistics, all O(|E| + n) and built from the
Stage 2 ingredients Sigma_in^(H), Sigma_in^(M):

  corr_HM_log   Pearson corr of log Sigma_in^(H) and log Sigma_in^(M) over pages
                (how well block-level inflow predicts page-level inflow)
  icc_indeg     one-way ANOVA intraclass correlation of log(1 + in-degree)
                by block (method-of-moments, unequal block sizes)
  disp_H        mean |1 - Sigma_in^(H)| (the H half of the Stage 2 statistic h)

The perturbations are replayed with the same RNG stream as run_experiments.py,
and the replay is checked against the stored rows (keep_frac, pesc_M).
"""
import json
import os
import sys
from multiprocessing import Pool

import numpy as np
import pandas as pd

import ncd_calibration_lib as L
import run_experiments as R

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "results", "stage1", "probe_features.csv")


def icc_oneway(y, groups):
    """ANOVA estimator of the intraclass correlation (unequal group sizes)."""
    _, g = np.unique(groups, return_inverse=True)
    k = g.max() + 1
    n = len(y)
    nb = np.bincount(g, minlength=k).astype(float)
    if k < 2 or n - k < 1:
        return np.nan
    mb = np.bincount(g, weights=y, minlength=k) / nb
    m = y.mean()
    ssb = np.sum(nb * (mb - m) ** 2)
    ssw = np.sum((y - mb[g]) ** 2)
    msb, msw = ssb / (k - 1), ssw / (n - k)
    n0 = (n - np.sum(nb ** 2) / n) / (k - 1)
    den = msb + (n0 - 1) * msw
    return float((msb - msw) / den) if den > 0 else np.nan


def probe(ops):
    sh, sm = L.incoming_sums(ops)
    indeg = np.asarray(ops.HT.getnnz(axis=1))
    eps = 1.0 / ops.n
    lh, lm = np.log(sh + eps), np.log(sm + eps)
    c = np.corrcoef(lh, lm)[0, 1] if lh.std() > 0 and lm.std() > 0 else np.nan
    return dict(corr_HM_log=float(c),
                icc_indeg=icc_oneway(np.log1p(indeg), ops.block),
                disp_H=float(np.mean(np.abs(1.0 - sh))))


def replay(spec):
    g, pop, params, rng = R.make_base(spec)
    z = np.load(os.path.join(R.RES, spec["panel"], f"{spec['base']}.npz"))
    rows = json.loads(str(z["meta"]))["rows"]
    out, i = [], 0
    for exp, lvl in R.LEVELS:
        reps = 1 if (exp == "A" and lvl == 1.0) else spec["reps"]
        for rep in range(reps):
            go = (g if lvl == 1.0 else L.sparsify_arcs(g, lvl, rng)) if exp == "A" else L.newly_added_pages(g, lvl, rng)
            oo = L.build_operators(go)
            row = rows[i]
            pm = L.block_escape(oo)[1]
            assert abs(row["keep_frac"] - go.m / g.m) < 1e-12 and abs(row["pesc_M"] - pm) < 1e-12, "replay mismatch"
            out.append(dict(base=spec["base"], exp=exp, level=lvl, rep=rep, **probe(oo)))
            i += 1
    return out


def main():
    panels = sys.argv[1:] or ["calib", "test"]
    specs = [s for p in panels for s in R.panel_specs(p)
             if os.path.exists(os.path.join(R.RES, p, f"{s['base']}.npz"))]
    rows = []
    with Pool(2) as pool:
        for r in pool.imap_unordered(replay, specs):
            rows += r
    pd.DataFrame(rows).to_csv(OUT, index=False)
    print("wrote", OUT, len(rows))


if __name__ == "__main__":
    main()
