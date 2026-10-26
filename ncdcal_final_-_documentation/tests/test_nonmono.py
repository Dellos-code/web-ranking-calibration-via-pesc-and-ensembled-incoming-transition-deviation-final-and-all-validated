"""
Unit tests for stage2_nonmonotone_search.py.

Run:  python3 tests/test_nonmono.py

1. tau_mean_batch (vectorised over parameter vectors) equals the mean of
   stage2_analysis.interp row by row.
2. The exact step-function fits are exact: for random curves, the TAU fit of a
   two- and three-region step attains the brute-force optimum over every pair of
   knot positions on the same quantile grid and every grid value per region, and
   the LS fit attains the brute-force least-squares optimum.
3. Every non-monotone family reduces to the constant when its slopes vanish, and
   the asymmetric tent is a non-increasing function of |x - x0| on each side.
4. The dangling-deconfounded column sums: on random graphs,
   Sigma_in(v; r) - (1 - r) n_dang/n equals the column sums of
   (1 - r) H_raw + r M, where H_raw is H before the dangling rows are patched,
   and its mean equals 1 - (1 - r) n_dang/n.
5. The replay of one calibration graph reproduces the stored h, r and dangling
   fraction of all its instances (skipped when the Stage 2 surfaces are absent).
"""
import itertools
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import ncd_calibration_lib as L  # noqa: E402
import stage2_analysis as S2  # noqa: E402
import stage2_nonmonotone_search as N  # noqa: E402
from test_lib import random_graph  # noqa: E402


def random_curves(rng, n):
    """Concave-ish tau(alpha) curves with random peaks, on the Stage 2 grid."""
    peak = rng.uniform(0.6, 0.95, n)
    width = rng.uniform(0.1, 0.4, n)
    return 0.9 - ((S2.GRID[None, :] - peak[:, None]) / width[:, None]) ** 2 * 0.05 + rng.normal(0, 1e-3, (n, len(S2.GRID)))


def test_tau_mean_batch():
    rng = np.random.default_rng(0)
    C = random_curves(rng, 50)
    A = rng.uniform(0.3, 1.0, (7, 50))
    ref = np.array([S2.interp(C, a).mean() for a in A])
    assert np.allclose(N.tau_mean_batch(C, A), ref, atol=1e-14)


def test_step_exact():
    rng = np.random.default_rng(1)
    n = 60
    x = rng.uniform(0, 1, n)
    C = random_curves(rng, n)
    y = S2.oracle(C)[0]
    qs = np.arange(0.1, 0.91, 0.1)
    o = np.argsort(x, kind="mergesort")
    idx = np.unique(np.clip(np.round(qs * n).astype(int), 1, n - 1))
    Cs, ys = C[o][:, S2.OMASK], y[o]
    tau_seg = lambda a, b: Cs[a:b].sum(axis=0).max()
    ls_seg = lambda a, b: ((ys[a:b] - ys[a:b].mean()) ** 2).sum()
    for k in (2, 3):
        cuts = [(i,) for i in idx] if k == 2 else [c for c in itertools.combinations(idx, 2)]
        best_tau = max(sum(tau_seg(a, b) for a, b in zip((0,) + c, c + (n,))) for c in cuts) / n
        best_ls = min(sum(ls_seg(a, b) for a, b in zip((0,) + c, c + (n,))) for c in cuts)
        for fit in ("TAU", "LS"):
            kn, vv = N.fit_step(x, y, C, k, fit, qs=qs)
            a_hat = vv[np.searchsorted(kn, x, side="right")]
            if fit == "TAU":
                assert abs(S2.interp(C, a_hat).mean() - best_tau) < 1e-12
            else:
                assert abs(((a_hat - y) ** 2).sum() - best_ls) < 1e-10


def test_families():
    z = np.linspace(-0.5, 1.5, 201)
    zb = (0.2, 0.7)
    for fam, p in (("tent", [0.82, 0.4, 0.0, 0.0]), ("symtent", [0.82, 0.4, 0.0]), ("quad", [0.82, 0.4, 0.0]),
                   ("gauss", [0.82, 0.4, 0.1, 0.0]), ("cliplin", [0.82, 0.0]),
                   ("spline2", [0.82, 0.82, 0.82, 0.82, 0.3, 0.6])):
        assert np.allclose(N.FAM[fam](np.array(p, float), z, zb), 0.82), fam
    a = N.FAM["tent"](np.array([0.85, 0.5, 0.2, 0.6]), z, zb)
    j = int(np.searchsorted(z, 0.5))
    assert np.all(np.diff(a[:j]) >= -1e-15) and np.all(np.diff(a[j:]) <= 1e-15)
    assert abs(a.max() - 0.85) < 1e-12
    # the centre is clipped to the admissible range
    b = N.FAM["tent"](np.array([0.85, 5.0, 0.2, 0.2]), z, zb)
    assert abs(z[int(np.argmax(b))] - zb[1]) < 0.011


def test_raw_column_sums():
    rng = np.random.default_rng(5)
    for _ in range(4):
        ops = L.build_operators(random_graph(rng, n=60, m=220, N=5, dangling=6))
        D = L.dense_P(ops, 0.5, 0.3)          # any split: only H and M are used
        H, M = D["H"].copy(), D["M"]
        nd = len(ops.dangling)
        Hraw = H.copy()
        Hraw[ops.dangling, :] = 0.0
        sig_h, sig_m = L.incoming_sums(ops)
        for r in (0.0, 0.118, 0.5):
            s_raw = (r * sig_m + (1 - r) * sig_h) - (1 - r) * nd / ops.n
            assert np.allclose(s_raw, ((1 - r) * Hraw + r * M).sum(axis=0), atol=1e-12)
            assert abs(s_raw.mean() - (1 - (1 - r) * nd / ops.n)) < 1e-12


def test_replay_one_graph():
    path = os.path.join(S2.RES, "calib", "calib_000.npz")
    if not os.path.exists(path):
        print("  (skipped: no Stage 2 surfaces)")
        return
    import json
    import run_experiments as RE
    z = np.load(path)
    rows = pd.DataFrame(json.loads(str(z["meta"]))["rows"])
    rp = pd.DataFrame(N.replay_base(RE.panel_specs("calib")[0]))
    assert len(rp) == len(rows)
    c = RE.R_NAMES.index("rule")
    assert np.max(np.abs(rp["h_rule_rp"].to_numpy() - z["h"][:, c])) < 1e-12
    assert np.max(np.abs(rp["r_rule_rp"].to_numpy() - z["r"][:, c])) < 1e-12
    assert np.max(np.abs(rp["delta"].to_numpy() - rows["dangling_frac"].to_numpy())) < 1e-12


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"ok  {name}")
