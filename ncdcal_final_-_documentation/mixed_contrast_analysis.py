"""Graph-unit analysis of the preregistered mixed-damage contrast."""
from __future__ import annotations

import csv
import json
import os
import argparse

import numpy as np
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
RULES = ("C4", "C5", "C6", "C7")


def exact_p(values):
    values = np.asarray(values, dtype=float)
    if not len(values) or np.allclose(values, 0):
        return 1.0
    return float(stats.wilcoxon(values, method="exact", alternative="two-sided").pvalue)


def load_rows(path):
    with open(path, newline="") as fh:
        return list(csv.DictReader(fh))


def main():
    global RESULTS
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", default=os.path.join(HERE, "results", "mixed"))
    results = os.path.abspath(parser.parse_args().results_dir)
    RESULTS = results
    rows = [r for r in load_rows(os.path.join(RESULTS, "rule_results.csv"))
            if r["contrast_case"].lower() == "true"]
    grouped = {}
    for r in rows:
        key = (r["panel"], r["base"], r["group"], r["rule"])
        grouped.setdefault(key, []).append(float(r["gain"]))
    graph_rows = []
    for (panel, base, group, rule), values in sorted(grouped.items()):
        graph_rows.append({"panel": panel, "base": base, "group": group,
                           "rule": rule, "n_instances": len(values),
                           "mean_gain": float(np.mean(values))})

    summary = {}
    for panel in ("calib", "test", "topo"):
        for group in ("pure_A", "pure_B", "mixed", "all"):
            if group == "all":
                raw = [r for r in rows if r["panel"] == panel]
                all_grouped = {}
                for r in raw:
                    all_grouped.setdefault((r["base"], r["rule"]), []).append(float(r["gain"]))
                subset = [{"base": base, "rule": rule, "mean_gain": float(np.mean(values))}
                          for (base, rule), values in all_grouped.items()]
            else:
                subset = [r for r in graph_rows if r["panel"] == panel and r["group"] == group]
            by_graph = {}
            for r in subset:
                by_graph.setdefault(r["base"], {})[r["rule"]] = r["mean_gain"]
            contrasts = {}
            for rule in RULES:
                diffs = [v[rule] - v["C1"] for v in by_graph.values()
                         if rule in v and "C1" in v]
                contrasts[rule] = {
                    "n_graphs": len(diffs), "mean_gain_vs_C1": float(np.mean(diffs))
                    if diffs else None, "wins": int(np.sum(np.asarray(diffs) > 0)),
                    "p_exact": exact_p(diffs),
                }
            summary[f"{panel}_{group}"] = contrasts

    heldout = {}
    for rule in RULES:
        by_graph = {}
        for r in graph_rows:
            if r["panel"] in ("test", "topo"):
                by_graph.setdefault((r["panel"], r["base"]), {})[r["rule"]] = r["mean_gain"]
        diffs = [v[rule] - v["C1"] for v in by_graph.values()
                 if rule in v and "C1" in v]
        heldout[rule] = {
            "n_graphs": len(diffs), "mean_gain_vs_C1": float(np.mean(diffs)),
            "wins": int(np.sum(np.asarray(diffs) > 0)), "p_exact": exact_p(diffs),
        }

    with open(os.path.join(RESULTS, "contrast_per_graph.csv"), "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["panel", "base", "group",
                                                "rule", "n_instances", "mean_gain"])
        writer.writeheader()
        writer.writerows(graph_rows)
    with open(os.path.join(RESULTS, "contrast_per_graph_summary.json"), "w") as fh:
        json.dump({"contrast_filter": "|dh| < 0.02 and oracle_NCD < 0.80",
                   "rules_vs_C1": summary, "pooled_held_out": heldout,
                   "decision_rule": {
                       "mean_gain_at_least": 0.0005,
                       "p_cv_below": 0.05,
                       "p_pooled_held_out_below": 0.05,
                       "holm_significant_loss_forbidden": True,
                       "verdict": "not promoted"}},
                  fh, indent=2)
    print(json.dumps({"rules_vs_C1": summary, "pooled_held_out": heldout}, indent=2))


if __name__ == "__main__":
    main()
