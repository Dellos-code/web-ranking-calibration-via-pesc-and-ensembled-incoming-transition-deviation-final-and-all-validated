import os
import sys
import warnings

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import ncd_autotune as A  # noqa: E402
import ncd_calibration_lib as L  # noqa: E402


def toy():
    return np.array([0, 1, 2, 2]), np.array([1, 2, 0, 3]), np.array([0, 0, 1, 1])


def test_locked_formulas_and_consistency():
    src, dst, block = toy()
    out = A.tune(src, dst, block)
    graph = L.Graph(4, src, dst, block)
    ops = L.build_operators(graph)
    pbar = L.block_escape(ops)[1]
    assert np.isclose(out["r"], np.clip(0.242 - 0.143 * pbar, 0.01, 0.99))
    assert out["alpha"] == 0.81
    assert np.isclose(out["diagnostics"]["h"], L.h_statistics(ops, out["r"])["h"])


def test_alpha_validation_and_baseline_warning():
    src, dst, block = toy()
    try:
        A.tune(src, dst, block, alpha_ref=0.8)
    except ValueError as exc:
        assert "alpha_ref" in str(exc)
    else:
        raise AssertionError("unsupported alpha_ref accepted")
    with warnings.catch_warnings(record=True) as seen:
        warnings.simplefilter("always")
        A.tune(src, dst, block, baseline={"h": 0.1})
    assert any("baseline is ignored" in str(w.message) for w in seen)


if __name__ == "__main__":
    test_locked_formulas_and_consistency()
    test_alpha_validation_and_baseline_warning()
    print("all autotune tests passed")
