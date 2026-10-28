"""Summarize the locked real2 comparisons from stage2 NPZ files."""
from __future__ import annotations

import json
import os

import numpy as np
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data", "real2", "provenance.json")
RES = os.path.join(HERE, "results", "stage2", "real2")
GRID = np.round(np.concatenate([np.arange(.05, .40, .05), np.arange(.40, 1., .01)]), 2)


def bootstrap_mean_ci(values, seed=17100, n_resamples=10_000):
    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    samples = rng.choice(values, size=(n_resamples, len(values)), replace=True)
    means = samples.mean(axis=1)
    return [float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))]


def main():
    provenance = json.load(open(DATA))
    records = []
    for fn in sorted(os.listdir(RES)):
        if not fn.endswith(".npz"):
            continue
        z = np.load(os.path.join(RES, fn))
        meta = json.loads(str(z["meta"]))
        rows = meta["rows"]
        tau = z["tau"]
        ia = {float(a): int(np.argmin(abs(GRID - a))) for a in (.81, .85, .93, .95)}
        by = {}
        for i, row in enumerate(rows):
            if row["exp"] == "A" and float(row["level"]) == 1.0:
                continue
            by.setdefault(row["base"], []).append(i)
        for base, ii in by.items():
            ncd_h1 = float(np.mean(tau[ii, 3, 0, ia[.81]] - tau[ii, 0, 0, ia[.85]]))
            ncd_h2 = float(np.mean(tau[ii, 3, 0, ia[.81]] - tau[ii, 1, 0, ia[.95]]))
            ncd95_h3b = float(np.mean(tau[ii, 3, 1, ia[.93]] - tau[ii, 1, 1, ia[.95]]))
            records.append({"base": base, "partition_type": provenance[base]["partition_type"],
                            "h1": ncd_h1, "h2": ncd_h2, "h3b": ncd95_h3b})
    out = {}
    for key in ("h1", "h2", "h3b"):
        x = np.asarray([r[key] for r in records])
        try:
            p = float(stats.wilcoxon(x).pvalue)
        except ValueError:
            p = 1.0
        out[key] = {"n": int(len(x)), "mean": float(x.mean()), "median": float(np.median(x)),
                    "wins": int(np.sum(x > 0)), "p_exact": p,
                    "percentile_range95": [float(np.percentile(x, 2.5)),
                                           float(np.percentile(x, 97.5))],
                    "bootstrap_mean_ci95": bootstrap_mean_ci(x)}
    out["graphs"] = records
    with open(os.path.join(RES, "summary_real2.json"), "w") as fh:
        json.dump(out, fh, indent=2)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
