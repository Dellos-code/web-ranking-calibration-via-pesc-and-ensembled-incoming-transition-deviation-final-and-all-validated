"""Independent checks for v2 generated headline inputs.

This script deliberately reads raw JSON/CSV outputs and does not import the analysis modules.
"""
import csv
import json
import os
import re


HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    with open(os.path.join(HERE, "results", "stage2", "nonmono_summary.json")) as fh:
        summary = json.load(fh)
    tr = summary["mechanism"]["trajectories"]
    assert abs(tr["A1.0"]["delta"] - 0.17334375) < 1e-12
    assert tr["A0.2"]["delta"] > tr["A1.0"]["delta"]
    assert abs(tr["A0.2"]["hraw_rule"] - tr["A1.0"]["hraw_rule"]) < 0.02
    assert tr["B0.6"]["hraw_rule"] > tr["B0.1"]["hraw_rule"]

    with open(os.path.join(HERE, "results", "stage2", "pipelines.csv"), newline="") as fh:
        rows = list(csv.DictReader(fh))
    q = [r for r in rows if r["truth"] == "NCD" and r["panel"] == "test" and
         r["policy"] == "two-stage: rule r + cliplin_TAU(h)"][0]
    assert float(q["d_vs_ng13_085"]) > 0.0005
    assert int(q["wins_vs_ng13_085"]) >= 30
    q95 = [r for r in rows if r["truth"] == "NCD95" and r["panel"] == "test" and
           r["policy"] == "NG13 static (r=0.105, a=0.95)"][0]
    assert float(q95["tau"]) > 0.0

    with open(os.path.join(HERE, "V2_WP0_mixture_table.md")) as fh:
        text = fh.read()
    assert "A" in text and "B" in text and "0.173" in text
    macros = open(os.path.join(HERE, "paper", "generated", "macros.tex")).read()
    assert re.search(r"\\newcommand\{\\WpZeroNfMatched\}", macros)
    print("v2 verification passed")


if __name__ == "__main__":
    main()
