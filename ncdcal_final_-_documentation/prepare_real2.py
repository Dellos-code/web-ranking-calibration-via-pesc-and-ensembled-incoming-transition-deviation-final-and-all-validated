"""Prepare the reachable real2 sources into additive compressed graph files."""
from __future__ import annotations

import gzip
import hashlib
import json
import os
from datetime import date

import numpy as np

from prepare_real_graphs import _clean, contiguous_modularity_segmentation

HERE = os.path.dirname(os.path.abspath(__file__))
RAW = os.path.join(HERE, "data", "real2_raw")
GEOM = os.path.join(HERE, "raw_data", "geom_gcn")
OUT = os.path.join(HERE, "data", "real2")
TODAY = "2026-10-27"


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def save(name, n, src, dst, block, meta, raw_paths):
    src, dst = _clean(n, src, dst)
    block = np.asarray(block, dtype=np.int64)
    if len(block) != n or np.any(src >= n) or np.any(dst >= n):
        raise ValueError(f"{name}: invalid endpoint or label range")
    np.savez_compressed(os.path.join(OUT, f"{name}.npz"),
                        n=n, src=src, dst=dst, block=block)
    dout = np.bincount(src, minlength=n)
    meta.update({
        "n": int(n), "arcs": int(len(src)),
        "blocks": int(len(np.unique(block))),
        "reverse_arc_fraction": float(len(np.intersect1d(
            src.astype(np.int64) * n + dst,
            dst.astype(np.int64) * n + src)) / max(len(src), 1)),
        "mean_outdegree": float(len(src) / n),
        "dangling_fraction": float(np.mean(dout == 0)),
        "intra_block_arc_fraction": float(np.mean(block[src] == block[dst])),
        "raw_sha256": {os.path.basename(p): sha256(p) for p in raw_paths},
    })
    return meta


def geom(name):
    rows = []
    with open(os.path.join(GEOM, f"{name}_labels.txt")) as fh:
        next(fh)
        for line in fh:
            node, _feature, label = line.rstrip("\n").split("\t")
            rows.append((int(node), int(label)))
    labels = np.asarray(rows, dtype=np.int64)
    node = labels[:, 0]
    n = int(node.max()) + 1
    if len(np.unique(node)) != n or not np.array_equal(np.sort(node), np.arange(n)):
        raise ValueError(f"{name}: labels do not cover a contiguous node range")
    src, dst = np.loadtxt(os.path.join(GEOM, f"{name}_edges.txt"),
                          dtype=np.int64, unpack=True, skiprows=1)
    return save(
        name, n, src, dst, np.asarray(labels[np.argsort(node), 1], dtype=np.int64),
        {"source_url": f"https://raw.githubusercontent.com/graphdml-uiuc-jlu/geom-gcn/master/new_data/{name}/",
         "access_date": TODAY, "licence": "WebKB data; Geom-GCN repository/paper provenance",
         "partition": "label partition by page category (weaker proxy than a website)",
         "partition_type": "label"},
        [os.path.join(GEOM, f"{name}_edges.txt"),
         os.path.join(GEOM, f"{name}_labels.txt")])


def snap_web(name):
    path = os.path.join(RAW, f"web-{name}.txt.gz")
    edges = np.loadtxt(gzip.open(path, "rt"), comments="#", dtype=np.int64,
                       usecols=(0, 1))
    src, dst = edges[:, 0], edges[:, 1]
    n = int(max(src.max(), dst.max()) + 1)
    src, dst = _clean(n, src, dst)
    block, merges = contiguous_modularity_segmentation(n, src, dst)
    return save(
        f"web-{name}", n, src, dst, block,
        {"source_url": f"https://snap.stanford.edu/data/web-{name}.txt.gz",
         "access_date": TODAY, "licence": "SNAP dataset, Stanford Network Analysis Project",
         "partition": f"contiguity-constrained greedy directed-modularity segmentation ({merges} merges), host proxy",
         "partition_type": "proxy"},
        [path])


def email_eu():
    ep = os.path.join(RAW, "email-Eu-core.txt.gz")
    lp = os.path.join(RAW, "email-Eu-core-department-labels.txt.gz")
    edges = np.loadtxt(gzip.open(ep, "rt"), dtype=np.int64)
    labs = np.loadtxt(gzip.open(lp, "rt"), dtype=np.int64)
    n = int(max(edges.max(), labs[:, 0].max()) + 1)
    block = np.full(n, -1, dtype=np.int64)
    block[labs[:, 0].astype(int)] = labs[:, 1]
    if np.any(block < 0):
        raise ValueError("email-Eu-core: missing node labels")
    return save(
        "email-Eu-core", n, edges[:, 0], edges[:, 1], block,
        {"source_url": "https://snap.stanford.edu/data/email-Eu-core.html",
         "access_date": TODAY, "licence": "SNAP dataset, Stanford Network Analysis Project",
         "partition": "42 department labels", "partition_type": "label"},
        [ep, lp])


def ogbn_arxiv():
    base = os.path.join(RAW, "ogbn-arxiv")
    ep = os.path.join(base, "edge.csv.gz")
    lp = os.path.join(base, "node-label.csv.gz")
    npth = os.path.join(base, "num-node-list.csv.gz")
    edges = np.loadtxt(gzip.open(ep, "rt"), delimiter=",", dtype=np.int64)
    labels = np.loadtxt(gzip.open(lp, "rt"), dtype=np.int64)
    n = int(np.loadtxt(gzip.open(npth, "rt"), dtype=np.int64))
    if len(labels) != n:
        raise ValueError("ogbn-arxiv: label count mismatch")
    return save(
        "ogbn-arxiv", n, edges[:, 0], edges[:, 1], labels,
        {"source_url": "https://ogb.stanford.edu/docs/nodeprop/#ogbn-arxiv",
         "access_date": TODAY, "licence": "Open Graph Benchmark data and paper",
         "partition": "subject-area node labels", "partition_type": "label"},
        [ep, lp, npth])


def main():
    os.makedirs(OUT, exist_ok=True)
    records = {}
    for name in ("cornell", "texas", "wisconsin", "film"):
        records[name] = geom(name)
    for name in ("NotreDame", "Stanford"):
        records[f"web-{name}"] = snap_web(name)
    records["email-Eu-core"] = email_eu()
    records["ogbn-arxiv"] = ogbn_arxiv()
    with open(os.path.join(OUT, "provenance.json"), "w") as fh:
        json.dump(records, fh, indent=2, sort_keys=True)
    print(json.dumps(records, indent=2))


if __name__ == "__main__":
    main()
