"""
decode_cnr2000.py
=================
Decodes the LAW cnr-2000 crawl from its compressed BVGraph files into a plain arc list.

Input  (raw data directory, default ../data/cnr2000/):
    cnr-2000.graph, cnr-2000.offsets, cnr-2000.properties  (and optionally cnr-2000.ef)
    -- the BVGraph files distributed by the Laboratory for Web Algorithmics
       (https://law.di.unimi.it/webdata/cnr-2000/), also mirrored in the webgraph-rs
       repository.
Output:
    cnr2000_edges.npz with arrays  src, dst  (int64, one entry per arc, in BVGraph order)
    and the scalar n (number of pages).  prepare_real_graphs.py reads this file.

Decoding uses the Python bindings of the Rust WebGraph framework (pip package
``webgraph``, class ``BvGraph``).  The result is checked against the arc count stored in
the .properties file.

    python3 decode_cnr2000.py [raw_dir]
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))


def decode(raw_dir):
    import webgraph                                   # pip install webgraph
    g = webgraph.BvGraph(os.path.join(raw_dir, "cnr-2000"))
    n, m = g.num_nodes(), g.num_arcs()
    deg = g.outdegrees().astype(np.int64)
    src = np.repeat(np.arange(n, dtype=np.int64), deg)
    dst = np.empty(m, dtype=np.int64)
    pos = 0
    for u in range(n):
        d = int(deg[u])
        if d:
            dst[pos:pos + d] = np.fromiter(g.successors(u), dtype=np.int64, count=d)
            pos += d
    assert pos == m, (pos, m)
    return n, src, dst


def main(raw_dir):
    n, src, dst = decode(raw_dir)
    props = dict(line.strip().split("=", 1) for line in open(os.path.join(raw_dir, "cnr-2000.properties"))
                 if "=" in line and not line.startswith("#"))
    if "arcs" in props:
        assert int(props["arcs"]) == len(src), (props["arcs"], len(src))
    out = os.path.join(raw_dir, "cnr2000_edges.npz")
    np.savez_compressed(out, src=src, dst=dst, n=n)
    print(f"cnr-2000: n = {n}, arcs = {len(src)} -> {out}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(HERE), "data", "cnr2000"))
