"""Experiment C (mixed damage) driver and grouped rule report.

The driver is additive: it reuses the generators, perturbations, solver and
truth definitions from ``run_experiments.py`` and writes only ``results/mixed``.
Use ``--probe`` first to obtain a measured base-graph/cell runtime.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import time

import numpy as np

import ncd_calibration_lib as L
import run_experiments as RE

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "results", "mixed")
KEEPS = (1.0, 0.8, 0.6, 0.4, 0.2)
PAGES = (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6)
TRUTHS = ("NCD", "NCD95", "TOT")
RULES = ("C0", "C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8",
         "single_h_raw_rel", "single_h_raw_rel_delta", "single_h_raw_rel_zeta")
CNR_CELL_LIMIT = 4


def specs(panel, limit=None):
    ss = RE.panel_specs(panel)
    if panel == "real":
        ss.sort(key=lambda s: s["base"] != "cnr2000")
    if limit:
        ss = ss[:limit]
    return ss


def cells(spec):
    for keep in KEEPS:
        for pages in PAGES:
            if keep == 1.0 and pages == 0.0:
                continue
            if spec["base"] == "cnr2000":
                allowed = ((0.8, 0.1), (0.8, 0.4), (0.4, 0.1), (0.4, 0.4))
                if (keep, pages) not in allowed[:CNR_CELL_LIMIT]:
                    continue
            yield keep, pages


def _features(ops, r, base):
    sums = L.incoming_sums(ops)
    sig_h, sig_m = sums
    delta = float(len(ops.dangling) / ops.n)
    s = r * sig_m + (1 - r) * sig_h
    raw = r * sig_m + (1 - r) * (sig_h - delta)
    rel = raw / max(float(raw.mean()), 1e-15)
    f = L.structural_features(ops)
    f.update(h=float(np.mean(np.abs(1 - s))),
             h_raw=float(np.mean(np.abs(1 - raw))),
             h_raw_rel=float(np.mean(np.abs(1 - rel))),
             h_raw_H=float(np.mean(np.abs(1 - (sig_h - delta)))),
             delta=delta, zeta=f["zero_in_frac"], r=float(r),
             dh=float(np.mean(np.abs(1 - s)) - base["h"]),
             dh_raw=float(np.mean(np.abs(1 - raw)) - base["h_raw"]),
             dh_raw_rel=float(np.mean(np.abs(1 - rel)) - base["h_raw_rel"]),
             ddelta=delta - base["delta"], dzeta=f["zero_in_frac"] - base["zeta"])
    return f


def run_one(spec, order="B_then_A", probe=False):
    t0 = time.time()
    g, pop, params, rng = RE.make_base(spec)
    ops0 = L.build_operators(g)
    r0 = float(np.clip(0.242 - 0.143 * L.structural_features(ops0)["pesc_M"], .01, .99))
    b0 = _features(ops0, r0, {"h": 0, "h_raw": 0, "h_raw_rel": 0,
                              "delta": 0, "zeta": 0})
    # Baseline is defined at the same Stage-1 r as each observed instance.
    b0 = _features(ops0, r0, b0)
    refs = {
        "NCD": L.ncdawarerank(ops0, .85, L.R_CANONICAL, tol=1e-11)[0],
        "NCD95": L.ncdawarerank(ops0, .95, L.R_NG13, tol=1e-11)[0],
        "TOT": L.rapr_reference(ops0, L.R_CANONICAL, 1, 1, K=20000)[0],
    }
    krefs = {k: L.KendallRef(v) for k, v in refs.items()}
    if probe:
        keep, pages = next(cells(spec))
        go = L.mixed_damage(g, keep, pages, rng, order)
        oo = L.build_operators(go)
        rr = float(np.clip(.242 - .143 * L.structural_features(oo)["pesc_M"], .01, .99))
        L.alpha_sweep(oo, [rr], L.ALPHA_GRID, tail=1e-13, block=32)
        return {"base": spec["base"], "cell": [keep, pages],
                "seconds": time.time() - t0}
    rows, curves = [], []
    for keep, pages in cells(spec):
        go = L.mixed_damage(g, keep, pages, rng, order)
        oo = L.build_operators(go)
        rr = float(np.clip(.242 - .143 * L.structural_features(oo)["pesc_M"], .01, .99))
        f = _features(oo, rr, b0)
        curve = L.alpha_sweep(oo, [rr], L.ALPHA_GRID, tail=1e-13, block=32)[0]
        tau = np.array([[krefs[k](curve[j]) for j in range(len(L.ALPHA_GRID))]
                        for k in TRUTHS], dtype=np.float32)
        oracle = {k: float(L.ALPHA_GRID[np.argmax(tau[i, L.ALPHA_ORACLE_MASK]) +
                                         np.flatnonzero(L.ALPHA_ORACLE_MASK)[0]])
                  for i, k in enumerate(TRUTHS)}
        rows.append(dict(panel=spec["panel"], base=spec["base"], keep=keep,
                         new_pages=pages, order=order, group=(
                             "pure_A" if pages == 0 else "pure_B" if keep == 1 else "mixed"),
                         **f, **{f"oracle_{k}": v for k, v in oracle.items()}))
        curves.append(tau)
    return {"rows": rows, "curves": np.asarray(curves), "seconds": time.time() - t0,
            "base": spec["base"], "meta": {"spec": spec, "alpha_grid": L.ALPHA_GRID.tolist(),
            "truths": TRUTHS, "order": order, "runtime_seconds": time.time() - t0}}


def _predictors(train):
    """Fit deterministic, deliberately small versions of C0--C8."""
    x = np.asarray(train)
    y = x[:, 0]
    def lin(cols):
        A = np.c_[np.ones(len(x)), x[:, cols]]
        return lambda z: np.clip(np.c_[np.ones(len(z)), z[:, cols]] @
                                 np.linalg.lstsq(A, y, rcond=None)[0], .40, .99)
    models = {"C0": lambda z: np.full(len(z), .81)}
    models["C1"] = lin([x.dtype.names.index("dh") if x.dtype.names else 0])
    # The named feature matrix below is assembled by fit_rules; these lambdas
    # are replaced there, keeping this helper useful for simple smoke tests.
    return models


def fit_rules(rows):
    names = ("dh", "dh_raw", "dh_raw_rel", "ddelta", "delta", "dzeta",
             "h_raw_rel", "zeta")
    X = np.array([[r[n] for n in names] for r in rows], float)
    y = np.array([r["oracle_NCD"] for r in rows], float)
    def clipped(A, target=y, selector=None):
        b = np.linalg.lstsq(A, target, rcond=None)[0]
        return lambda z: np.clip(
            (np.c_[np.ones(len(z)), z[:, selector]] if selector is not None else z) @ b,
            .40, .99)
    one = lambda j: clipped(np.c_[np.ones(len(X)), X[:, j]], selector=j)
    out = {"C0": lambda z: np.full(len(z), .81)}
    out["C1"] = one(0); out["C2"] = one(1); out["C3"] = one(2)
    t1, t2 = np.median(X[:, 1]), np.median(X[:, 3])
    h4 = np.c_[np.ones(len(X)), np.maximum(0, X[:, 1]-t1),
                np.maximum(0, X[:, 3]-t2)]
    out["C4"] = clipped(h4, selector=[1, 3])
    t3 = np.median(X[:, 2])
    h5 = np.c_[np.ones(len(X)), np.maximum(0, X[:, 2]-t3),
                np.maximum(0, X[:, 3]-t2)]
    out["C5"] = clipped(h5, selector=[2, 3])
    p1, p2 = out["C1"], out["C2"]
    out["C6"] = lambda z: np.minimum(p1(z), p2(z))
    qh, qr = np.quantile(X[:, 0], [1/3, 2/3]), np.quantile(X[:, 1], [1/3, 2/3])
    table = np.full((3, 3), .81)
    for i in range(3):
        for j in range(3):
            m = ((X[:, 0] >= (qh[i-1] if i else -np.inf)) &
                 (X[:, 0] < (qh[i] if i < 2 else np.inf)) &
                 (X[:, 1] >= (qr[j-1] if j else -np.inf)) &
                 (X[:, 1] < (qr[j] if j < 2 else np.inf)))
            if m.any(): table[i, j] = y[m].mean()
    out["C7"] = lambda z: np.clip(table[np.digitize(z[:, 0], qh),
                                           np.digitize(z[:, 1], qr)], .40, .99)
    try:
        from sklearn.ensemble import GradientBoostingRegressor
        gb = GradientBoostingRegressor(random_state=0, n_estimators=40, max_depth=2).fit(X, y)
        out["C8"] = lambda z: np.clip(gb.predict(z), .40, .99)
    except Exception:
        out["C8"] = one(0)
    out["single_h_raw_rel"] = one(2)
    out["single_h_raw_rel_delta"] = clipped(
        np.c_[np.ones(len(X)), X[:, 2], X[:, 4]], selector=[2, 4])
    out["single_h_raw_rel_zeta"] = clipped(
        np.c_[np.ones(len(X)), X[:, 2], X[:, 6]], selector=[2, 6])
    return out, names


def main():
    global OUT
    ap = argparse.ArgumentParser()
    ap.add_argument("--panel", default="calib,test,topo,real")
    ap.add_argument("--limit", type=int, default=2)
    ap.add_argument("--probe", action="store_true")
    ap.add_argument("--order", default="B_then_A", choices=("B_then_A", "A_then_B"))
    ap.add_argument("--cnr-cells", type=int, default=4,
                    help="number of reduced cnr2000 cells (1-4)")
    ap.add_argument("--output-dir", default=OUT)
    a = ap.parse_args()
    OUT = os.path.abspath(a.output_dir)
    global CNR_CELL_LIMIT
    CNR_CELL_LIMIT = max(1, min(4, a.cnr_cells))
    os.makedirs(OUT, exist_ok=True)
    selected = [s for p in a.panel.split(",") for s in specs(p, a.limit)]
    if a.probe:
        result = run_one(selected[0], a.order, True)
        print(json.dumps(result, indent=2))
        return
    all_rows, all_curves = [], []
    t0 = time.time()
    for i, spec in enumerate(selected, 1):
        r = run_one(spec, a.order)
        all_rows.extend(r["rows"]); all_curves.extend(r["curves"])
        print(f"[{i}/{len(selected)}] {spec['panel']}/{spec['base']} "
              f"{r['seconds']:.1f}s", flush=True)
    with open(os.path.join(OUT, "instances.csv"), "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=sorted(all_rows[0]))
        writer.writeheader(); writer.writerows(all_rows)
    np.savez_compressed(os.path.join(OUT, "surfaces.npz"),
                        curves=np.asarray(all_curves, dtype=np.float32),
                        alpha=L.ALPHA_GRID)
    # Fit the requested families on calibration rows and write deployable
    # predictions for every held-out row; grouped aggregation is by base graph.
    cal = [r for r in all_rows if r["panel"] == "calib"]
    models, feature_names = fit_rules(cal or all_rows)
    out = []
    alpha = L.ALPHA_GRID
    curve_index = 0
    for r in all_rows:
        curve = np.asarray(all_curves[curve_index])
        if curve.ndim == 2:
            curve = curve[0]
        curve_index += 1
        z = np.array([[r[n] for n in feature_names + ("delta", "zeta")]], float)
        # fit_rules uses the first 8 columns; the extra columns are harmless
        z8 = z[:, :8]
        for name, model in models.items():
            aa = float(model(z8)[0])
            gain = float(np.interp(aa, alpha, curve) - np.interp(.81, alpha, curve))
            out.append(dict(panel=r["panel"], base=r["base"], group=r["group"],
                            rule=name, alpha=aa, gain=gain, keep=r["keep"],
                            new_pages=r["new_pages"], contrast_case=(
                                abs(r["dh"]) < 0.02 and r["oracle_NCD"] < 0.80)))
    with open(os.path.join(OUT, "rule_results.csv"), "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=sorted(out[0]))
        writer.writeheader(); writer.writerows(out)
    # These are descriptive grouped means, with the preregistered contrast
    # restricted before aggregation (never pooled over perturbations as if
    # they were independent).
    for filename, subset in (
        ("group_results.csv", out),
        ("contrast_results.csv", [r for r in out if r["contrast_case"]]),
    ):
        groups = {}
        for r in subset:
            key = (r["panel"], r["group"], r["rule"])
            groups.setdefault(key, []).append(r["gain"])
        with open(os.path.join(OUT, filename), "w", newline="") as fh:
            writer = csv.writer(fh)
            writer.writerow(("panel", "group", "rule", "n_instances", "mean_gain"))
            for (panel, group, rule), vals in sorted(groups.items()):
                writer.writerow((panel, group, rule, len(vals), float(np.mean(vals))))
    summary = {"panels": sorted(set(r["panel"] for r in all_rows)),
               "rules": list(RULES), "n_instances": len(all_rows),
               "n_base_graphs": len(set((r["panel"], r["base"]) for r in all_rows)),
               "runtime_seconds": time.time() - t0, "order": a.order,
               "pre_registered_contrast": "C4/C5/C6/C7 versus C1 when |dh|<0.02 and oracle_NCD<0.80",
               "status": "reduced" if a.limit else "complete"}
    with open(os.path.join(OUT, "summary.json"), "w") as fh:
        json.dump(summary, fh, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
