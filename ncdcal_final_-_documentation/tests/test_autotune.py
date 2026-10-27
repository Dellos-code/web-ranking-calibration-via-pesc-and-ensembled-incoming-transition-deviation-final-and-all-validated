import os
import sys
import time
import warnings

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import ncd_autotune as A  # noqa: E402
import ncd_calibration_lib as L  # noqa: E402


def toy():
    return np.array([0, 1, 2, 2]), np.array([1, 2, 0, 3]), np.array([0, 0, 1, 1])


def dense_expected(src, dst, block, r):
    """Independent dense implementation of the definitions in the documentation."""
    n = len(block)
    block = np.asarray(block, dtype=int)
    blocks = sorted(set(block.tolist()))
    block_index = {b: i for i, b in enumerate(blocks)}
    b = np.array([block_index[x] for x in block])
    sizes = np.bincount(b, minlength=len(blocks))
    H = np.zeros((n, n))
    M = np.zeros((n, n))
    outgoing = [[] for _ in range(n)]
    for u, v in zip(src, dst):
        if u != v and v not in outgoing[u]:
            outgoing[u].append(v)
    for u in range(n):
        if outgoing[u]:
            for v in outgoing[u]:
                H[u, v] = 1.0 / len(outgoing[u])
        else:
            H[u, :] = 1.0 / n
        proximal = {b[u]} | {b[v] for v in outgoing[u]}
        for v in range(n):
            if b[v] in proximal:
                M[u, v] = 1.0 / (len(proximal) * sizes[b[v]])
    Z = (1.0 - r) * H + r * M
    sigma = Z.sum(axis=0)
    v = np.array([1.0 / (len(blocks) * sizes[x]) for x in b])
    hv = np.abs(v @ Z - v).sum()
    delta = np.mean([not outgoing[u] for u in range(n)])
    hraw = np.mean(np.abs(1.0 - sigma + (1.0 - r) * delta))
    return {
        "h": np.mean(np.abs(1.0 - sigma)),
        "h_raw": hraw,
        "h_v": hv,
        "delta": delta,
        "zeta": np.mean([not any(v == u for v in src) for u in range(n)]),
    }


def test_locked_formulas_and_consistency():
    src, dst, block = toy()
    out = A.tune(src, dst, block)
    graph = L.Graph(4, src, dst, block)
    ops = L.build_operators(graph)
    pbar = L.block_escape(ops)[1]
    assert np.isclose(out["r"], np.clip(0.242 - 0.143 * pbar, 0.01, 0.99))
    assert out["alpha"] == 0.81
    assert np.isclose(out["diagnostics"]["h"], L.h_statistics(ops, out["r"])["h"])


def test_independent_dense_statistics():
    src = np.array([0, 0, 1, 2, 4, 5])
    dst = np.array([1, 2, 2, 3, 5, 4])
    block = np.array([0, 0, 1, 1, 2, 2])
    out = A.tune(src, dst, block)
    expected = dense_expected(src, dst, block, out["r"])
    for key in ("h", "h_raw", "h_v"):
        assert abs(out["diagnostics"][key] - expected[key]) < 1e-12, key
    assert abs(out["diagnostics"]["delta"] - expected["delta"]) < 1e-12
    assert abs(out["diagnostics"]["zeta"] - expected["zeta"]) < 1e-12


def test_alpha_validation_and_baseline_warning():
    src, dst, block = toy()
    assert A.tune(src, dst, block, alpha_ref=0.95)["alpha"] == 0.93
    assert A.tune(src, dst, block, alpha_ref=0.85)["alpha"] == 0.81
    assert A.tune(src, dst, block, alpha_ref=0.85 + 1e-9)["alpha"] == 0.81
    for alpha_ref in (0.8, 0.9):
        try:
            A.tune(src, dst, block, alpha_ref=alpha_ref)
        except ValueError as exc:
            assert "alpha_ref" in str(exc)
        else:
            raise AssertionError("unsupported alpha_ref accepted")
    with warnings.catch_warnings(record=True) as seen:
        warnings.simplefilter("always")
        plain = A.tune(src, dst, block)
        warned = A.tune(src, dst, block, baseline={"h": 0.1})
    assert any("baseline is ignored" in str(w.message) for w in seen)
    assert plain == warned


def test_cleaning_and_degenerate_graphs():
    src = np.array([0, 0, 0, 1, 2, 2, 3])
    dst = np.array([1, 1, 0, 2, 3, 3, 3])
    block = np.array([0, 0, 1, 1])
    clean = A.tune(np.array([0, 1, 2]), np.array([1, 2, 3]), block)
    assert A.tune(src, dst, block) == clean
    for src2, dst2, block2 in (
        (np.array([0, 1]), np.array([1, 0]), np.array([0, 0])),
        (np.array([], dtype=int), np.array([], dtype=int), np.array([0, 1, 1])),
    ):
        out = A.tune(src2, dst2, block2)
        assert np.isfinite(out["r"])
        assert np.isfinite(out["diagnostics"]["h_v"])


def test_cli_end_to_end(tmp_path):
    path = tmp_path / "graph.npz"
    np.savez(path, src=np.array([0, 1]), dst=np.array([1, 2]),
             block=np.array([0, 0, 1]))
    from io import StringIO
    import contextlib
    with contextlib.redirect_stdout(StringIO()) as buf:
        A.main([str(path), "--alpha-ref", "0.95"])
    parsed = __import__("json").loads(buf.getvalue())
    assert parsed["alpha"] == 0.93
    assert "diagnostics" in parsed


def test_timing_smoke():
    rng = np.random.default_rng(20261027)
    n, m = 50_000, 200_000
    src = rng.integers(0, n, size=m, dtype=np.int64)
    dst = rng.integers(0, n, size=m, dtype=np.int64)
    block = np.arange(n, dtype=np.int64) % 31
    started = time.perf_counter()
    out = A.tune(src, dst, block)
    elapsed = time.perf_counter() - started
    assert elapsed < 10.0, elapsed
    assert np.isfinite(out["r"])


if __name__ == "__main__":
    test_locked_formulas_and_consistency()
    test_independent_dense_statistics()
    test_alpha_validation_and_baseline_warning()
    test_cleaning_and_degenerate_graphs()
    import tempfile
    from pathlib import Path
    with tempfile.TemporaryDirectory() as d:
        test_cli_end_to_end(Path(d))
    test_timing_smoke()
    print("all autotune tests passed")
