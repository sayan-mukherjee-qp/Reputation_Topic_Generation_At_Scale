#!/usr/bin/env python3
"""Joint UMAP x HDBSCAN grid search, scored on event fragmentation.

The reducer comparison in REDUCER_BENCHMARK.md held HDBSCAN fixed at values
tuned for PCA space. That is not a fair test of UMAP: UMAP's whole job is to
pull neighbourhoods tight, so a density clusterer tuned elsewhere will read far
more dense clumps than really exist. This searches both together, as the
stability guide's Step 8 requires.

Scoring deliberately leads with the metric that settled the original benchmark
-- how many topics you must read to see 80% of a real event -- because
silhouette, topic count and noise ratio all reward fragmentation.

Usage:  umap_grid.py [--quick]
"""
from __future__ import annotations

import argparse
import importlib.util
import itertools
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

# Located relative to this file, so a fresh clone works anywhere.
HERE = Path(__file__).resolve().parent
BASE = HERE.parent                  # repo root, where data/ lives
CSV = BASE / "data/twcs_subset_200k.csv"
CACHE = HERE / "embed_cache"        # shared embedding cache (gitignored)
MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

# One pool per difficulty class, each carrying a verified event:
#   VirginTrains  the strike the global manifold lost  (hard, small signal)
#   AppleSupport  iOS 11, worst fragmentation in the baseline (12 topics for 80%)
#   McDonalds     Szechuan sauce, the easy case
POOLS = {
    "VirginTrains": [["strike", "rmt", "industrial action"]],
    "AppleSupport": [["ios 11", "ios11", "ios 11.0", "ios 11.1"]],
    "McDonalds": [["szechuan", "sichuan"]],
}


def load_module():
    spec = importlib.util.spec_from_file_location("rtd", HERE / "reputation_topic_detection.py")
    m = importlib.util.module_from_spec(spec)
    sys.modules["rtd"] = m
    spec.loader.exec_module(m)
    return m


def prepare(m):
    """Reproduce the pipeline's segmentation and discovery split exactly."""
    sys.argv = ["x", str(CSV), "--out", "/tmp/_grid", "--model", MODEL,
                "--min-similarity", "0.50", "--min-cluster-size", "10", "--min-samples", "5",
                "--embed-cache", str(CACHE), "--reducer", "umap"]
    args = m.parse_args()
    df = m.load_data(args.csv, args.text_col, args.time_col, args.brand_col, args.id_col)
    df = df.rename(columns={args.time_col: "event_time", args.brand_col: "brand",
                            args.text_col: "text", args.id_col: "tweet_id"})
    segments = m.segment_records(df, id_col="tweet_id", text_col="text",
                                 max_chars=args.max_segment_chars, overlap_chars=60,
                                 max_segments_per_record=args.max_segments_per_record)
    stop = m.build_content_stopset()
    if args.min_content_words > 0:
        cw = segments["clean_text"].map(lambda t: m.content_word_count(t, stop))
        segments = segments[cw >= args.min_content_words].reset_index(drop=True)

    key = m.embedding_cache_key(args.model, segments["clean_text"].tolist(), "fp32")
    emb = m.load_cached_embeddings(str(CACHE), key)
    if emb is None or len(emb) != len(segments):
        raise SystemExit(f"No cached embeddings for key {key}; run the pipeline once first.")
    print(f"segments={len(segments):,}  embeddings={emb.shape}  (cache hit {key})")

    # Same 80/20 time split the pipeline uses for discovery vs holdout.
    order = np.argsort(pd.to_datetime(segments["event_time"], errors="coerce").to_numpy(), kind="stable")
    cut = int(len(order) * (1 - args.test_fraction))
    train_idx = np.sort(order[:cut])
    return args, segments.iloc[train_idx].reset_index(drop=True), emb[train_idx]


def event_scores(seg_pool: pd.DataFrame, labels: np.ndarray, groups) -> dict:
    """Coverage and topics-to-80% for this pool's event, from raw cluster labels."""
    text = seg_pool["clean_text"].astype(str)
    hit = text.map(lambda t: all(any(k in t.lower() for k in g) for g in groups)).to_numpy()
    rec = seg_pool["record_id"].to_numpy()
    ev_rec = set(rec[hit])
    if not ev_rec:
        return {"ev_records": 0, "coverage": np.nan, "topics_for_80": np.nan, "spread": 0}
    per = {}
    for c in set(labels[hit]):
        if c < 0:
            continue
        per[c] = len(set(rec[hit & (labels == c)]))
    if not per:
        return {"ev_records": len(ev_rec), "coverage": 0.0, "topics_for_80": np.nan, "spread": 0}
    counts = np.array(sorted(per.values(), reverse=True), dtype=float)
    total = len(ev_rec)
    cum = np.cumsum(counts) / total
    return {"ev_records": total,
            "coverage": float(counts[0] / total),
            "topics_for_80": int((cum < 0.8).sum() + 1),
            "spread": int(len(per))}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="smaller grid for a smoke run")
    ap.add_argument("--out", default="grid_results.csv")
    ap.add_argument("--reference-size", type=int, default=50000)
    a = ap.parse_args()

    m = load_module()
    args, seg, emb = prepare(m)

    if a.quick:
        NC, NN, MCS, EPS = [15], [30], [30], [0.0, 0.4]
    else:
        NC, NN, MCS, EPS = [5, 15, 50], [15, 30, 50], [10, 30, 80], [0.0, 0.2, 0.4, 0.8]

    ref_idx = m.umap_reference_sample(seg, emb, a.reference_size, seed=42)
    ref_emb = emb[ref_idx]
    print(f"reference sample: {len(ref_idx):,} segments")

    pool_idx = {b: np.where(seg["brand"].astype(str).to_numpy() == b)[0] for b in POOLS}
    for b, i in pool_idx.items():
        print(f"  pool {b}: {len(i):,} segments")

    rows = []
    for nc, nn in itertools.product(NC, NN):
        t0 = time.perf_counter()
        model = m.fit_umap_model(ref_emb, nc, nn, 0.0, 42, False)
        fit_s = time.perf_counter() - t0
        print(f"\n[UMAP] n_components={nc} n_neighbors={nn} fitted in {fit_s:.0f}s", flush=True)

        proj = {}
        for b, idx in pool_idx.items():
            t1 = time.perf_counter()
            proj[b] = np.asarray(model.transform(emb[idx]), dtype=np.float32)
            print(f"    transform {b}: {time.perf_counter()-t1:.0f}s", flush=True)

        for mcs, eps in itertools.product(MCS, EPS):
            agg = []
            for b, idx in pool_idx.items():
                X = proj[b]
                lab = m.cluster_embeddings(X, mcs, max(5, mcs // 3), "eom", eps)
                q = m.cluster_quality(X, lab)
                sc = event_scores(seg.iloc[idx].reset_index(drop=True), lab, POOLS[b])
                agg.append({"pool": b, "n_clusters": int(q["n_clusters"]),
                            "noise": float(q["noise_ratio"]),
                            "silhouette": q["silhouette"],
                            "max_share": float(m.max_cluster_share(lab)), **sc})
            d = pd.DataFrame(agg)
            rows.append({
                "n_components": nc, "n_neighbors": nn, "min_cluster_size": mcs, "epsilon": eps,
                "umap_fit_s": round(fit_s, 1),
                "clusters": int(d["n_clusters"].sum()),
                "noise": round(float(d["noise"].mean()), 3),
                "silhouette": round(float(pd.to_numeric(d["silhouette"], errors="coerce").mean()), 3),
                "max_share": round(float(d["max_share"].max()), 3),
                "coverage": round(float(d["coverage"].mean()), 3),
                "topics_for_80": round(float(d["topics_for_80"].mean()), 2),
                "spread": round(float(d["spread"].mean()), 1),
                **{f"cov_{r['pool']}": round(r["coverage"], 3) for _, r in d.iterrows()},
                **{f"t80_{r['pool']}": r["topics_for_80"] for _, r in d.iterrows()},
            })
            print(f"    mcs={mcs:<3} eps={eps:<4} clusters={rows[-1]['clusters']:<5} "
                  f"noise={rows[-1]['noise']:.0%} cov={rows[-1]['coverage']:.3f} "
                  f"t80={rows[-1]['topics_for_80']}", flush=True)
        pd.DataFrame(rows).to_csv(a.out, index=False)

    res = pd.DataFrame(rows).sort_values(["topics_for_80", "coverage"], ascending=[True, False])
    res.to_csv(a.out, index=False)
    print(f"\n=== best 12 by topics-to-80% ===\n{res.head(12).to_string(index=False)}")
    print(f"\nWrote {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
