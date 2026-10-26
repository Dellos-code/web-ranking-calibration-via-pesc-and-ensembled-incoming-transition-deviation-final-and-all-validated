"""
prepare_real_graphs.py
======================
Builds the real-network panel used as a transfer test.  Every graph is
written to ``data/real/<name>.npz`` with arrays ``src``, ``dst`` (directed
arcs, self-loops and duplicates removed), ``block`` (NCD block id per page)
and ``n``.

Graphs
------
cnr2000     LAW cnr-2000 web crawl (the dataset of Nikolakopoulos &
            Garofalakis, WSDM 2013, Table 4), decoded from the BVGraph files
            mirrored in the webgraph-rs repository.  The URL list is NOT
            available, so the true host partition is unknown.  LAW numbers
            pages in lexicographic URL order, hence every host is a contiguous
            ID interval; we recover a *host proxy* by contiguity-constrained
            greedy maximisation of directed modularity (adjacent segments are
            merged while the modularity gain is positive).  The resulting
            blocks are an approximation of hosts and are labelled as such.
wikics      WikiCS (Mernyei & Cangea 2020): directed Wikipedia hyperlinks
            between computer-science articles; blocks = 10 topic labels.
cora_full   CitationFull Cora (Bojchevski & Guennemann 2018): directed
            citations, blocks = 70 fine-grained topics.
cora_ml     CitationFull Cora-ML: directed citations, 7 topics.
citeseer    CitationFull CiteSeer: directed citations, 6 topics.

Topic-class blocks are the "content category" decomposition that [4]
(Section 5) names as an admissible NCD criterion.
"""
import heapq
import json
import os
import sys

import numpy as np
import scipy.sparse as sp

HERE = os.path.dirname(os.path.abspath(__file__))
RAW = os.path.join(os.path.dirname(HERE), "data")
OUT = os.path.join(HERE, "data", "real")


def _clean(n, src, dst):
    """Remove self-loops and duplicate arcs."""
    src = np.asarray(src, dtype=np.int64)
    dst = np.asarray(dst, dtype=np.int64)
    keep = src != dst
    key = np.unique(src[keep] * n + dst[keep])
    return key // n, key % n


def contiguous_modularity_segmentation(n, src, dst):
    """Greedy contiguity-constrained directed-modularity segmentation.

    Pages are assumed to be ordered so that every true block is an interval.
    Start from singletons; repeatedly merge the adjacent pair of segments with
    the largest positive modularity gain

        dQ(a,b) = (e_ab + e_ba)/m - (out_a*in_b + out_b*in_a)/m^2 ,

    where e_ab + e_ba is the number of arcs with one end in a and the other in
    b.  Because a and b are adjacent intervals [i,k) and [k,j), that count is
    the number of arcs with min endpoint in [i,k) and max endpoint in [k,j);
    it is answered from arcs sorted by their min endpoint.  Pages left with no
    arcs at all are attached to the segment on their left.
    """
    m = len(src)
    lo = np.minimum(src, dst)
    hi = np.maximum(src, dst)
    order = np.argsort(lo, kind="stable")
    lo_s = lo[order]
    hi_s = hi[order]
    lo_ptr = np.searchsorted(lo_s, np.arange(n + 1))  # arcs with lo == i are lo_ptr[i]:lo_ptr[i+1]
    dout = np.bincount(src, minlength=n).astype(float)
    din = np.bincount(dst, minlength=n).astype(float)
    cdout = np.concatenate([[0.0], np.cumsum(dout)])
    cdin = np.concatenate([[0.0], np.cumsum(din)])

    def cross(i, k, j):
        a, b = lo_ptr[i], lo_ptr[k]
        if a == b:
            return 0
        h = hi_s[a:b]
        return int(np.count_nonzero((h >= k) & (h < j)))

    def gain(i, k, j):
        oa, ia = cdout[k] - cdout[i], cdin[k] - cdin[i]
        ob, ib = cdout[j] - cdout[k], cdin[j] - cdin[k]
        return cross(i, k, j) / m - (oa * ib + ob * ia) / (m * m)

    start = np.arange(n)          # segment id == its start index
    end = np.arange(1, n + 1)     # exclusive end, valid for alive starts
    prev = np.arange(-1, n - 1)
    nxt = np.arange(1, n + 1)
    alive = np.ones(n, dtype=bool)
    ver = np.zeros(n, dtype=np.int64)
    heap = []
    # initial gains between consecutive singletons, vectorised
    adj = (hi - lo) == 1
    cnt = np.bincount(lo[adj], minlength=n)[: n - 1]
    i_idx = np.arange(n - 1)
    g0 = cnt / m - (dout[:-1] * din[1:] + dout[1:] * din[:-1]) / (m * m)
    for i in np.nonzero(g0 > 0)[0]:
        heap.append((-g0[i], int(i), int(i + 1), 0, 0))
    heapq.heapify(heap)
    merges = 0
    while heap:
        ng, a, b, va, vb = heapq.heappop(heap)
        if not (alive[a] and alive[b]) or ver[a] != va or ver[b] != vb or nxt[a] != b:
            continue
        # merge b into a
        end[a] = end[b]
        alive[b] = False
        nb = nxt[b]
        nxt[a] = nb
        if nb < n:
            prev[nb] = a
        ver[a] += 1
        merges += 1
        pa = prev[a]
        if pa >= 0:
            g = gain(pa, a, end[a])
            if g > 0:
                heapq.heappush(heap, (-g, int(pa), int(a), int(ver[pa]), int(ver[a])))
        if nb < n:
            g = gain(a, nb, end[nb])
            if g > 0:
                heapq.heappush(heap, (-g, int(a), int(nb), int(ver[a]), int(ver[nb])))
    starts = np.nonzero(alive)[0]
    block = np.empty(n, dtype=np.int64)
    for bid, s in enumerate(starts):
        block[s:end[s]] = bid
    # attach linkless singletons to the left neighbour segment
    deg = dout + din
    sizes = np.bincount(block)
    lonely = (sizes[block] == 1) & (deg == 0)
    for u in np.nonzero(lonely)[0]:
        if u > 0:
            block[u] = block[u - 1]
        elif n > 1:
            block[u] = block[u + 1]
    _, block = np.unique(block, return_inverse=True)
    return block, merges


def load_cnr2000():
    z = np.load(os.path.join(RAW, "cnr2000", "cnr2000_edges.npz"))
    n = int(z["n"])
    src, dst = _clean(n, z["src"], z["dst"])
    block, merges = contiguous_modularity_segmentation(n, src, dst)
    return n, src, dst, block, {"merges": merges,
                                "partition": "contiguous modularity segmentation (host proxy)"}


def load_wikics():
    with open(os.path.join(RAW, "other", "wikics_data.json")) as fh:
        W = json.load(fh)
    links = W["links"]
    n = len(links)
    src = np.repeat(np.arange(n), [len(l) for l in links])
    dst = np.concatenate([np.asarray(l, dtype=np.int64) for l in links if len(l)])
    src, dst = _clean(n, src, dst)
    return n, src, dst, np.asarray(W["labels"], dtype=np.int64), {"partition": "10 topic labels"}


def load_citationfull(name):
    z = np.load(os.path.join(RAW, "other", f"g2g_{name}.npz"), allow_pickle=False)
    A = sp.csr_matrix((z["adj_data"], z["adj_indices"], z["adj_indptr"]),
                      shape=tuple(z["adj_shape"])).tocoo()
    n = A.shape[0]
    src, dst = _clean(n, A.row, A.col)   # A_ij = 1  <=>  i cites j (in-degree is the heavy tail)
    lab = np.asarray(z["labels"], dtype=np.int64)
    return n, src, dst, lab, {"partition": f"{len(np.unique(lab))} topic labels"}


def main():
    os.makedirs(OUT, exist_ok=True)
    jobs = {
        "cora_ml": lambda: load_citationfull("cora_ml"),
        "citeseer": lambda: load_citationfull("citeseer"),
        "cora_full": lambda: load_citationfull("cora"),
        "wikics": load_wikics,
        "cnr2000": load_cnr2000,
    }
    only = sys.argv[1:] or list(jobs)
    summary = {}
    for name in only:
        n, src, dst, block, meta = jobs[name]()
        _, block = np.unique(block, return_inverse=True)
        np.savez_compressed(os.path.join(OUT, f"{name}.npz"), n=n, src=src, dst=dst, block=block)
        sizes = np.bincount(block)
        intra = float(np.mean(block[src] == block[dst]))
        dout = np.bincount(src, minlength=n)
        meta.update({"n": int(n), "arcs": int(len(src)), "blocks": int(len(sizes)),
                     "block_size_median": float(np.median(sizes)), "block_size_max": int(sizes.max()),
                     "singleton_blocks": int((sizes == 1).sum()), "intra_block_arc_fraction": intra,
                     "dangling_fraction": float(np.mean(dout == 0)), "mean_outdegree": float(dout.mean())})
        summary[name] = meta
        print(name, json.dumps(meta), flush=True)
    path = os.path.join(OUT, "summary.json")
    old = json.load(open(path)) if os.path.exists(path) else {}
    old.update(summary)
    json.dump(old, open(path, "w"), indent=2)


if __name__ == "__main__":
    main()
