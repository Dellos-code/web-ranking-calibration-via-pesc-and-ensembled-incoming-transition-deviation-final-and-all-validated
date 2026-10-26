"""
verify_stage2.py
================
Independent recomputation of the Checkpoint-2 headline numbers directly from the
raw Stage 2 result files (no code from stage2_analysis.py is used).

Checks, per panel, under the primary truth NCD (A1.0 excluded):
  mean tau of  (r = 0.10/0.85, alpha = 0.85)   static [NG13] Table 2 split
               (r = 0.10/0.95, alpha = 0.95)   static [NG13] Section 4 setting
               (r = rule,      alpha = 0.85)   Stage 1 only
               (r = rule,      alpha = 0.81)   two-stage with the constant alpha = 0.81
               (r = rule,      alpha = alpha*) oracle
and the Wilcoxon signed-rank p over per-graph means for the two-stage policy
against the static (0.118, 0.85) baseline.  0.81 is an exact grid point, so no
interpolation is involved.
"""
import glob
import json
import os

import numpy as np
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
GRID = np.round(np.concatenate([np.arange(0.05, 0.40, 0.05), np.arange(0.40, 1.0, 0.01)]), 2)


def main():
    for panel in ["calib", "test", "scale", "topo", "real"]:
        files = sorted(glob.glob(os.path.join(HERE, "results", "stage2", panel, "*.npz")))
        if not files:
            continue
        acc = {k: [] for k in ["s085", "s095", "st1", "two", "orc"]}
        per_base = []
        for f in files:
            z = np.load(f)
            meta = json.loads(str(z["meta"]))
            rn, tn = meta["r_names"], meta["truths"]
            c_can, c_ng, c_rule, t_ncd = rn.index("can"), rn.index("ng13"), rn.index("rule"), tn.index("NCD")
            j85, j95, j81 = (int(np.argmin(np.abs(GRID - a))) for a in (0.85, 0.95, 0.81))
            mask = GRID >= 0.40 - 1e-12
            keep = [i for i, row in enumerate(meta["rows"]) if not (row["exp"] == "A" and row["level"] == 1.0)]
            tau = z["tau"][keep].astype(float)
            s085 = tau[:, c_can, t_ncd, j85]
            s095 = tau[:, c_ng, t_ncd, j95]
            st1 = tau[:, c_rule, t_ncd, j85]
            two = tau[:, c_rule, t_ncd, j81]
            orc = np.max(np.where(mask[None, :], tau[:, c_rule, t_ncd, :], -np.inf), axis=1)
            for k, v in zip(acc, (s085, s095, st1, two, orc)):
                acc[k].append(v)
            per_base.append(float(np.mean(two - s085)))
        m = {k: float(np.mean(np.concatenate(v))) for k, v in acc.items()}
        p = stats.wilcoxon(per_base).pvalue if len(per_base) >= 5 else float("nan")
        print(f"{panel:6s} n_graphs={len(files):3d}  NG13(0.118,0.85)={m['s085']:.4f}  NG13(0.105,0.95)={m['s095']:.4f}  "
              f"Stage1(rule,0.85)={m['st1']:.4f}  two-stage(rule,0.81)={m['two']:.4f}  oracle={m['orc']:.4f}  "
              f"two-stage minus NG13(0.85)={m['two'] - m['s085']:+.4f} (Wilcoxon p={p:.2g}, "
              f"{sum(d > 0 for d in per_base)}/{len(per_base)} graphs better)")


if __name__ == "__main__":
    main()
