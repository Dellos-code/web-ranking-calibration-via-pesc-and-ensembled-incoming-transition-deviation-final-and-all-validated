"""Independent verification of the v2 generated results.

Only result files and generated text are read here.  The analysis/library modules
are intentionally not imported, so this check can catch shared implementation
errors.
"""
import csv
import glob
import json
import os
import re

import numpy as np
from scipy import stats


HERE = os.path.dirname(os.path.abspath(__file__))
STAGE2 = os.path.join(HERE, "results", "stage2")
TOL = 5e-5


def _alpha_index(meta, alpha):
    grid = np.asarray(meta["alpha_grid"], dtype=float)
    return int(np.argmin(np.abs(grid - alpha)))


def _per_graph(panel, r_name, truth, alpha):
    values = []
    for path in sorted(glob.glob(os.path.join(STAGE2, panel, "*.npz"))):
        with np.load(path, allow_pickle=False) as z:
            meta = json.loads(str(z["meta"]))
            truth_i = meta["truths"].index(truth)
            r_i = meta["r_names"].index(r_name)
            alpha_i = _alpha_index(meta, alpha)
            rows = meta["rows"]
            keep = np.ones(len(rows), dtype=bool)
            if truth in {"NCD", "NCD95", "TOT"}:
                keep = np.array([not (r["exp"] == "A" and r["level"] == 1.0)
                                for r in rows])
            values.append(float(np.mean(z["tau"][keep, r_i, truth_i, alpha_i])))
    if not values:
        raise AssertionError(f"no stage2 files for panel {panel}")
    return np.asarray(values)


def _comparison(panel, truth, left, left_alpha, right, right_alpha):
    diff = (_per_graph(panel, left, truth, left_alpha)
            - _per_graph(panel, right, truth, right_alpha))
    result = stats.wilcoxon(diff, method="exact")
    return float(np.mean(diff)), int(np.sum(diff > 0)), len(diff), float(result.pvalue)


def _assert_close(actual, expected, label):
    if not np.isclose(actual, expected, atol=TOL, rtol=0):
        raise AssertionError(f"{label}: {actual} != {expected}")


def _parse_cell(cell):
    match = re.search(
        r"([+-]?\d+\.\d+)\s*\((\d+)/(\d+);\s*p=([0-9.eE+-]+)", cell)
    if not match:
        raise AssertionError(f"cannot parse mismatch cell: {cell}")
    p_text = match.group(4)
    return (float(match.group(1)), int(match.group(2)), int(match.group(3)),
            float(p_text), p_text)


def check_mismatch_table():
    path = os.path.join(HERE, "V2_WP0_mismatch_table.md")
    rows = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            if line.startswith("| NCD") or line.startswith("| NCD95"):
                rows.append([x.strip() for x in line.strip().strip("|").split("|")])
    if len(rows) != 12:
        raise AssertionError(f"expected 12 mismatch rows, got {len(rows)}")
    candidates = {
        r"rule \$r\$, \$\\alpha=0\.81\$": ("rule", 0.81),
        r"rule \$r\$, \$\\alpha=0\.93\$": ("rule", 0.93),
        r"\$r=0\.105\$, \$\\alpha=0\.93\$": ("ng13", 0.93),
    }
    for fields in rows:
        truth, candidate, comparator = fields[:3]
        left, left_alpha = next((v for pattern, v in candidates.items()
                                 if re.fullmatch(pattern, candidate)), (None, None))
        if left is None:
            raise AssertionError(f"unknown mismatch candidate: {candidate}")
        right, right_alpha = ("can", 0.85) if comparator.startswith("$(0.118") else ("ng13", 0.95)
        for panel, cell in zip(("test", "scale", "topo", "real"), fields[3:]):
            actual = _comparison(panel, truth, left, left_alpha, right, right_alpha)
            expected = _parse_cell(cell)
            _assert_close(actual[0], expected[0], f"{truth} {candidate} {panel} mean")
            if actual[1:3] != expected[1:3]:
                raise AssertionError(f"{truth} {candidate} {panel} wins/n mismatch")
            p_digits = len(expected[4].split(".")[1]) if "." in expected[4] else 0
            if (round(actual[3], p_digits) != round(expected[3], p_digits)
                    and abs(actual[3] - expected[3]) > TOL):
                raise AssertionError(f"{truth} {candidate} {panel} p mismatch")


def _macro_values():
    text = open(os.path.join(HERE, "paper", "generated", "macros.tex"),
                encoding="utf-8").read()
    out = {}
    for line in text.splitlines():
        match = re.match(r"\\newcommand\{\\(\w+)\}\{(.*)\}$", line)
        if match:
            out[match.group(1)] = match.group(2).replace(r"\ensuremath{", "").replace("}", "")
    return out


def _macro_number(value):
    value = value.replace("+", "").replace("{", "")
    sci = re.fullmatch(r"([0-9.]+)\\times10\^(-?\d+)", value)
    if sci:
        return float(sci.group(1)) * 10.0 ** int(sci.group(2))
    return float(value)


def check_test_macros():
    pairs = {
        "MisNcdRuleLoCan": ("NCD", "rule", 0.81, "can", 0.85),
        "MisNcdRuleLoNg": ("NCD", "rule", 0.81, "ng13", 0.95),
        "MisNcdRuleHiCan": ("NCD", "rule", 0.93, "can", 0.85),
        "MisNcdRuleHiNg": ("NCD", "rule", 0.93, "ng13", 0.95),
        "MisNcdNgHiCan": ("NCD", "ng13", 0.93, "can", 0.85),
        "MisNcdNgHiNg": ("NCD", "ng13", 0.93, "ng13", 0.95),
        "MisNcdnfRuleLoCan": ("NCD95", "rule", 0.81, "can", 0.85),
        "MisNcdnfRuleLoNg": ("NCD95", "rule", 0.81, "ng13", 0.95),
        "MisNcdnfRuleHiCan": ("NCD95", "rule", 0.93, "can", 0.85),
        "MisNcdnfRuleHiNg": ("NCD95", "rule", 0.93, "ng13", 0.95),
        "MisNcdnfNgHiCan": ("NCD95", "ng13", 0.93, "can", 0.85),
        "MisNcdnfNgHiNg": ("NCD95", "ng13", 0.93, "ng13", 0.95),
    }
    macros = _macro_values()
    for name, spec in pairs.items():
        mean, wins, n, p = _comparison("test", *spec)
        _assert_close(_macro_number(macros[name]), mean, name)
        if macros[name + "Wins"] != f"{wins}/{n}":
            raise AssertionError(f"{name} wins macro mismatch")
        p_macro = _macro_number(macros[name + "P"])
        if abs(p_macro - p) > TOL and round(p_macro, 3) != round(p, 3):
            raise AssertionError(f"{name} p macro mismatch")


def check_mixture_table():
    summary = json.load(open(os.path.join(STAGE2, "nonmono_summary.json"), encoding="utf-8"))
    trajectories = summary["mechanism"]["trajectories"]
    expected = {
        "A": (trajectories["A1.0"], trajectories["A0.2"]),
        "B": (trajectories["B0.1"], trajectories["B0.6"]),
    }
    with open(os.path.join(HERE, "V2_WP0_mixture_table.md"), encoding="utf-8") as fh:
        lines = [line for line in fh if line.startswith("| A |") or line.startswith("| B |")]
    for line in lines:
        name = line.split("|")[1].strip()
        nums = [float(x) for x in re.findall(r"\d+\.\d+", line)]
        start, end = expected[name]
        values = [start["delta"], end["delta"], start["zero_in_frac"], end["zero_in_frac"],
                  start["h_rule"], end["h_rule"], start["hraw_rule"], end["hraw_rule"]]
        if any(abs(a - b) > 5e-4 for a, b in zip(nums, values)):
            raise AssertionError(f"mixture table mismatch for {name}")
    bline = next(line for line in lines if line.startswith("| B |"))
    if "0.185" not in bline or "0.207" not in bline or "0.644" not in bline:
        raise AssertionError("B row does not use B0.1 start values")


def check_documentation_headlines():
    expected = {
        "+0.0110", "+0.0743", "+0.0011", "+0.0089", "+0.0062", "+0.0034", "-0.0025"
    }
    for lang in ("EN", "GR"):
        text = open(os.path.join(HERE, f"DOCUMENTATION_{lang}.md"), encoding="utf-8").read()
        lines = [line for line in text.splitlines() if line.startswith("| ") and
                 ("Two-stage" in line or "Δύο στάδια" in line or "Anchor-matched" in line
                  or "Σταθερές με αντιστοίχιση" in line)]
        joined = "\n".join(lines)
        missing = expected - set(re.findall(r"[+-]\d+\.\d{4}", joined))
        if missing:
            raise AssertionError(f"{lang} headline values missing: {sorted(missing)}")


def main():
    check_mismatch_table()
    check_test_macros()
    check_mixture_table()
    check_documentation_headlines()
    print("v2 independent verification passed")


if __name__ == "__main__":
    main()
