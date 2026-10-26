"""
bound_check.py
==============
Numerical check of the displacement bound (Theorem 1 of the paper)

    ||pi(alpha) - v||_1  <=  alpha / (1 - alpha) * h_v(r),     h_v(r) = ||v^T Z_r - v^T||_1,

on (i) the random test graphs of tests/test_stage2.py and (ii) unperturbed calibration
graphs at the Stage 1 ratio.  For every (graph, r, alpha) it records the ratio
lhs / rhs, which must be <= 1 and tends to 1 as alpha -> 0 (first-order tightness).
Writes results/stage2/bound_check.csv and prints the summary quoted in the paper.

    python3 bound_check.py [n_calib_graphs]          (default 4)
"""
import os
import sys

import numpy as np
import pandas as pd

import ncd_calibration_lib as L
import run_experiments as R

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "tests"))
from test_lib import random_graph  # noqa: E402  (same random graphs as the unit tests)

HERE = os.path.dirname(os.path.abspath(__file__))


def ratios(ops, r, source, name):
    hv = L.h_statistics(ops, r)["h_v"]
    S = L.alpha_sweep(ops, [r])
    out = []
    for j, a in enumerate(L.ALPHA_GRID):
        lhs = float(np.abs(S[0, j] - ops.v).sum())
        rhs = a / (1.0 - a) * hv
        out.append(dict(source=source, graph=name, r=r, alpha=a, h_v=hv, lhs=lhs, rhs=rhs,
                        ratio=lhs / rhs if rhs > 0 else np.nan))
    return out


def main(n_calib=4):
    rows = []
    rng = np.random.default_rng(9)
    for i in range(10):
        ops = L.build_operators(random_graph(rng, n=60, m=250, N=6, dangling=5))
        for r in (0.0, 0.1, 0.5, 0.9):
            rows += ratios(ops, r, "random", f"rand_{i}")
    for spec in R.panel_specs("calib")[:n_calib]:
        g = R.make_base(spec)[0]
        ops = L.build_operators(g)
        pbar = L.block_escape(ops)[1]
        r = float(np.clip(R.RULE_NCD[0] + R.RULE_NCD[1] * pbar, 0.01, 0.99))
        rows += ratios(ops, r, "calib", spec["base"])
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(HERE, "results", "stage2", "bound_check.csv"), index=False)
    for src, d in df.groupby("source"):
        lo = d[d.alpha <= 0.051]
        print(f"{src}: {d.graph.nunique()} graphs, {len(d)} (graph, r, alpha) cases; "
              f"max ratio {d.ratio.max():.4f}; ratio at alpha=0.05 in [{lo.ratio.min():.3f}, {lo.ratio.max():.3f}]; "
              f"at alpha=0.85: median {d[np.isclose(d.alpha, 0.85)].ratio.median():.3f}")
    assert (df.ratio <= 1 + 1e-12).all()


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 4)
