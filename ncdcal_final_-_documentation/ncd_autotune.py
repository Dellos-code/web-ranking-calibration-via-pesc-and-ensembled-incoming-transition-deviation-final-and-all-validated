"""Practical single-snapshot calibration tool for NCDawareRank."""
from __future__ import annotations

import argparse
import json
import warnings

import numpy as np

import ncd_calibration_lib as L


SUPPORTED = {0.85: 0.81, 0.95: 0.93}


def tune(src, dst, block, alpha_ref=0.85, baseline=None):
    """Return the locked single-snapshot calibration and diagnostics.

    The arrays describe directed arcs ``src -> dst`` and a block id per page.
    ``baseline`` is accepted for API compatibility but is intentionally ignored until
    a baseline rule passes the preregistered WP3/WP4 tests.
    """
    alpha_ref = float(alpha_ref)
    matches = [a for a in SUPPORTED if np.isclose(alpha_ref, a)]
    if not matches:
        raise ValueError("alpha_ref must be one of the validated values 0.85 or 0.95")
    if baseline is not None:
        warnings.warn("No validated baseline rule exists; baseline is ignored.",
                      RuntimeWarning, stacklevel=2)
    block = np.asarray(block, dtype=np.int64)
    src, dst = L.clean_arcs(len(block), src, dst)
    graph = L.Graph(len(block), src, dst, block, name="autotune")
    ops = L.build_operators(graph)
    pbar = L.block_escape(ops)[1]
    r = float(np.clip(0.242 - 0.143 * pbar, 0.01, 0.99))
    sums = L.incoming_sums(ops)
    hs = L.h_statistics(ops, r, sums)
    alpha = SUPPORTED[matches[0]]
    features = L.structural_features(ops)
    return {
        "r": r,
        "alpha": alpha,
        "diagnostics": {
            "Pbar": pbar,
            "h": hs["h"],
            "h_raw": float(np.mean(np.abs(1.0 - (r * sums[1] + (1.0 - r) * sums[0])
                                      + (1.0 - r) * features["dangling_frac"]))),
            "delta": features["dangling_frac"],
            "zeta": features["zero_in_frac"],
            "h_v": hs["h_v"],
            "displacement_bound": float(alpha / (1.0 - alpha) * hs["h_v"]),
        },
    }


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("graph", help="NPZ containing src, dst and block arrays")
    ap.add_argument("--alpha-ref", type=float, default=0.85)
    args = ap.parse_args(argv)
    z = np.load(args.graph)
    print(json.dumps(tune(z["src"], z["dst"], z["block"], args.alpha_ref), indent=2))


if __name__ == "__main__":
    main()
