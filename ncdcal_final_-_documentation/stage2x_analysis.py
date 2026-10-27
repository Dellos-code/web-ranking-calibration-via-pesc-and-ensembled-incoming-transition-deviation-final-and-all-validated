"""Fit the preregistered anchor-discount forms from stage2x surfaces."""
from __future__ import annotations

import csv
import json
import os

import numpy as np
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "results", "stage2x")
A = np.array([.50, .60, .70, .75, .80, .85, .90, .95, .97])
GRID = np.round(np.concatenate([np.arange(.05, .40, .05), np.arange(.40, 1.0, .01)]), 2)
MASK = np.flatnonzero(GRID >= .40)


def load_panel(panel):
    rows, values = [], []
    for fn in sorted(os.listdir(os.path.join(ROOT, panel))):
        if not fn.endswith(".npz"):
            continue
        z = np.load(os.path.join(ROOT, panel, fn))
        meta = json.loads(str(z["meta"]))
        rows.append(meta["rows"])
        values.append(z["tau"])
    return [r for rr in rows for r in rr], np.concatenate(values)


def oracle(tau):
    q = tau[:, :, MASK]
    return GRID[MASK[np.argmax(q, axis=2)]], np.max(q, axis=2)


def fit_discount(rows, tau):
    # Use one mean oracle alpha per base and anchor for the primary r rule.
    al, best = oracle(tau[:, 0])
    by = {}
    for i, r in enumerate(rows):
        by.setdefault(r["base"], []).append(i)
    X, Y = [], []
    for base, ii in by.items():
        # Per-base means are the replication units.
        d = np.mean(A[None, :] - al[ii], axis=0)
        X.append(d)
        Y.append(base)
    D = np.asarray(X)
    d0 = D.mean(axis=0)
    slopes = np.array([np.polyfit(A - .85, row, 1)[0] for row in D])
    slope = float(np.mean(slopes))
    intercept = float(np.mean(D - slope * (A[None, :] - .85)))
    return {"bases": Y, "d_by_anchor": d0.tolist(),
            "constant": float(np.mean(d0)), "linear_slope": slope,
            "linear_intercept": intercept}


def evaluate(panel, fit):
    rows, tau = load_panel(panel)
    al, _ = oracle(tau[:, 0])
    out = []
    by = {}
    for i, r in enumerate(rows):
        by.setdefault(r["base"], []).append(i)
    dconst = fit["constant"]
    for base, ii in by.items():
        gains = []
        for j, a in enumerate(A):
            target = np.clip(a - dconst, .40, .99)
            k = int(np.argmin(np.abs(GRID - target)))
            no = int(np.argmin(np.abs(GRID - a)))
            gains.append(float(np.mean(tau[ii, 0, j, k] - tau[ii, 0, j, no])))
        out.append({"panel": panel, "base": base, "mean_gain_discount_vs_no_discount": float(np.mean(gains))})
    return out


def main():
    rows, tau = load_panel("calib")
    checks = [json.loads(str(np.load(os.path.join(ROOT, "calib", f))["meta"]))["anchor_check"]
              for f in os.listdir(os.path.join(ROOT, "calib")) if f.endswith(".npz")]
    fit = fit_discount(rows, tau)
    summary = {"a_supported": [float(A.min()), float(A.max())],
               "anchor_checks": {"max_085": max(x["anchor_max_abs_085"] for x in checks),
                                 "max_095": max(x["anchor_max_abs_095"] for x in checks)},
               "fit": fit, "held_out": {}}
    for panel in ("test", "topo", "real"):
        summary["held_out"][panel] = evaluate(panel, fit)
    with open(os.path.join(ROOT, "summary.json"), "w") as fh:
        json.dump(summary, fh, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
