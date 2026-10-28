"""Recompute the preregistered pooled real2 analysis from raw Stage 2 NPZ files."""
from __future__ import annotations

import csv
import json
import os

import numpy as np
from scipy import stats

from real2_analysis import GRID

HERE = os.path.dirname(os.path.abspath(__file__))
NEW_RES = os.path.join(HERE, "results", "stage2", "real2")
OLD_RES = os.path.join(HERE, "results", "stage2", "real")
PROVENANCE = json.load(open(os.path.join(HERE, "data", "real2", "provenance.json")))
OLD_PROVENANCE = json.load(open(os.path.join(HERE, "data", "real", "summary.json")))
OUT = os.path.join(HERE, "results", "stage2", "real2")


def bootstrap_mean_ci(values, seed=17100, n_resamples=10_000):
    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    means = rng.choice(values, size=(n_resamples, len(values)), replace=True).mean(axis=1)
    return [float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))]


def exact_p(values):
    values = np.asarray(values, dtype=float)
    if not len(values) or np.allclose(values, 0):
        return 1.0
    return float(stats.wilcoxon(values, alternative="two-sided", method="exact").pvalue)


def read_graph(path, provenance):
    z = np.load(path, allow_pickle=True)
    meta = json.loads(str(z["meta"]))
    rows = meta["rows"]
    ia = {float(a): int(np.argmin(abs(GRID - a))) for a in (.81, .85, .93, .95)}
    by = {}
    for i, row in enumerate(rows):
        if row["exp"] == "A" and float(row["level"]) == 1.0:
            continue
        by.setdefault(row["base"], []).append(i)
    records = []
    for base, ii in sorted(by.items()):
        tau = z["tau"]
        records.append({
            "base": base,
            "partition_type": provenance.get(base, {}).get("partition_type", "label"),
            "h1": float(np.mean(tau[ii, 3, 0, ia[.81]] - tau[ii, 0, 0, ia[.85]])),
            "h2": float(np.mean(tau[ii, 3, 0, ia[.81]] - tau[ii, 1, 0, ia[.95]])),
            "h3b": float(np.mean(tau[ii, 3, 1, ia[.93]] - tau[ii, 1, 1, ia[.95]])),
        })
    return records


def all_records():
    records = []
    for fn in sorted(os.listdir(OLD_RES)):
        if fn.endswith(".npz"):
            records.extend(read_graph(os.path.join(OLD_RES, fn),
                                      {os.path.splitext(fn)[0]: {
                                          "partition_type": "proxy" if os.path.splitext(fn)[0] == "cnr2000"
                                          else "label"}}))
    for fn in sorted(os.listdir(NEW_RES)):
        if fn.endswith(".npz"):
            records.extend(read_graph(os.path.join(NEW_RES, fn), PROVENANCE))
    for r in records:
        r["cohort"] = "new" if r["base"] in PROVENANCE else "old"
    return records


def summary(records, name):
    out = {"group": name, "n": len(records)}
    for key in ("h1", "h2", "h3b"):
        x = np.asarray([r[key] for r in records], dtype=float)
        out[key] = {
            "mean": float(x.mean()) if len(x) else None,
            "wins": int(np.sum(x > 0)),
            "p_exact": exact_p(x),
            "bootstrap_mean_ci95": bootstrap_mean_ci(x) if len(x) else None,
        }
    return out


def main():
    records = all_records()
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "pooled_per_graph.csv"), "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["base", "cohort", "partition_type", "h1", "h2", "h3b"])
        writer.writeheader()
        writer.writerows(records)

    groups = {
        "pooled": records,
        "old": [r for r in records if r["cohort"] == "old"],
        "new": [r for r in records if r["cohort"] == "new"],
        "label": [r for r in records if r["partition_type"] == "label"],
        "proxy": [r for r in records if r["partition_type"] == "proxy"],
    }
    pooled = {name: summary(rows, name) for name, rows in groups.items()}
    loo = []
    for i, record in enumerate(records):
        row = {"excluded": record["base"]}
        for key in ("h1", "h2", "h3b"):
            values = [r[key] for j, r in enumerate(records) if i != j]
            row[f"{key}_mean"] = float(np.mean(values))
            row[f"{key}_p_exact"] = exact_p(values)
        loo.append(row)
    with open(os.path.join(OUT, "pooled_leave_one_out.csv"), "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=loo[0].keys())
        writer.writeheader()
        writer.writerows(loo)
    pooled["leave_one_out"] = loo
    pooled["amendment_1"] = {
        "new_graphs": len(groups["new"]),
        "holm_family_size": 3,
        "holm_testable": len(groups["new"]) >= 7,
        "note": ("The three-family Holm claim is testable with at least seven new graphs."
                 if len(groups["new"]) >= 7 else
                 "The three-family Holm claim is untestable with fewer than seven new graphs."),
    }
    with open(os.path.join(OUT, "pooled.json"), "w") as fh:
        json.dump(pooled, fh, indent=2)
    print(json.dumps(pooled, indent=2))


if __name__ == "__main__":
    main()
