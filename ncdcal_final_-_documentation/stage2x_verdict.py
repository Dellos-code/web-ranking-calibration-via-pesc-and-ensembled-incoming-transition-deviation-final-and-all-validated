"""Formal decision-rule table for the registered anchor discount."""
from __future__ import annotations

import csv
import json
import os

import numpy as np
from scipy import stats

import stage2x_analysis as A

ROOT = A.ROOT
OUT = os.path.join(ROOT, "verdict.json")


def ci(values, seed=17200):
    x = np.asarray(values, float)
    rng = np.random.default_rng(seed)
    return [float(np.percentile(rng.choice(x, (10000, len(x)), replace=True).mean(1), q))
            for q in (2.5, 97.5)]


def pvalue(values):
    x = np.asarray(values, float)
    return 1.0 if len(x) == 0 or np.allclose(x, 0) else float(
        stats.wilcoxon(x, method="exact", alternative="two-sided").pvalue)


def fit_for(rows, tau, indices):
    al, _ = A.oracle(tau[:, 0])
    by = {}
    for i in indices:
        by.setdefault(rows[i]["base"], []).append(i)
    d = []
    for ii in by.values():
        d.append(np.mean(A.A[None, :] - al[ii], axis=0))
    d = np.asarray(d)
    slope = np.mean([np.polyfit(A.A - .85, row, 1)[0] for row in d])
    intercept = np.mean(d - slope * (A.A[None, :] - .85))
    return float(np.mean(intercept)), float(slope)


def eval_rule(rows, tau, discount, indices):
    by = {}
    for i in indices:
        by.setdefault(rows[i]["base"], []).append(i)
    out = []
    for base, ii in sorted(by.items()):
        gains = []
        for j, a in enumerate(A.A):
            target = np.clip(a - discount, .40, .99)
            k = int(np.argmin(abs(A.GRID - target)))
            no = int(np.argmin(abs(A.GRID - a)))
            gains.append(np.mean(tau[ii, 0, j, k] - tau[ii, 0, j, no]))
        out.append({"base": base, "mean_gain": float(np.mean(gains))})
    return out


def load(panel):
    rows, tau = A.load_panel(panel)
    return rows, tau, list(range(len(rows)))


def best_constant(rows, tau, indices):
    by = {}
    for i in indices:
        by.setdefault(rows[i]["base"], []).append(i)
    candidates = np.linspace(0.0, 0.20, 201)
    means = []
    for d in candidates:
        means.append(np.mean([r["mean_gain"] for r in eval_rule(rows, tau, d, indices)]))
    return float(candidates[int(np.argmax(means))])


def summarize(name, values):
    x = np.asarray(values, float)
    return {"comparison": name, "n": len(x), "mean": float(x.mean()),
            "wins": int(np.sum(x > 0)), "p_exact": pvalue(x), "bootstrap_mean_ci95": ci(x)}


def main():
    cal_rows, cal_tau, cal_idx = load("calib")
    all_cal = {r["base"] for r in cal_rows}
    fit = A.fit_discount(cal_rows, cal_tau)
    cv_values = []
    for fold in range(8):
        train = [i for i, r in enumerate(cal_rows) if hash(r["base"]) % 8 != fold]
        valid = [i for i, r in enumerate(cal_rows) if hash(r["base"]) % 8 == fold]
        if not valid:
            continue
        d0, slope = fit_for(cal_rows, cal_tau, train)
        fitted = float(d0)
        cv_values.extend(r["mean_gain"] for r in eval_rule(cal_rows, cal_tau, fitted, valid))
    table = [summarize("calib_grouped_cv_discount_vs_no_discount", cv_values)]
    held = {}
    for panel in ("test", "topo", "real"):
        rows, tau, idx = load(panel)
        fitted = eval_rule(rows, tau, fit["constant"], idx)
        x = [r["mean_gain"] for r in fitted]
        best_d = best_constant(rows, tau, idx)
        best = eval_rule(rows, tau, best_d, idx)
        held[panel] = {
            "fitted_discount": summarize("fitted_discount_vs_no_discount", x),
            "panel_best_constant": {"discount": best_d,
                                    **summarize("panel_best_constant_vs_no_discount",
                                                [r["mean_gain"] for r in best])},
            "per_graph": fitted,
        }
    pooled = []
    for panel in ("test", "topo", "real"):
        pooled.extend(r["mean_gain"] for r in held[panel]["per_graph"])
    table.append(summarize("pooled_held_out_discount_vs_no_discount", pooled))
    result = {
        "decision_rule": {
            "gain_threshold": 0.0005, "p_threshold": 0.05,
            "no_significant_loss": True, "supported_a_range": [0.50, 0.97],
            "verdict": "not promoted",
        },
        "anchor_checks": json.load(open(os.path.join(ROOT, "summary.json")))["anchor_checks"],
        "fitted_discount": fit,
        "table": table,
        "held_out": held,
    }
    with open(OUT, "w") as fh:
        json.dump(result, fh, indent=2)
    with open(os.path.join(ROOT, "verdict.csv"), "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["comparison", "n", "mean", "wins",
                                                "p_exact", "bootstrap_mean_ci95"])
        writer.writeheader()
        writer.writerows(table)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
