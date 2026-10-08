#!/usr/bin/env python3
"""Build the streaming batch: the next 200k records, in arrival order.

The first 200k (`twcs_subset_200k.csv`) is a time-stratified sample of Oct 1 -
Dec 1 2017 across 12 brands. This takes 200k *different* records for the same
brands from the same corpus and orders them by event time, then cuts them into
equal chunks so they can be fed to the pipeline as successive arrivals.

A note on what this can and cannot test. The corpus ends 2017-12-03, so there is
no genuinely later period to stream; these are disjoint records from the same
window, delivered in time order. That still exercises everything that matters
operationally -- incremental assignment, cross-run topic identity, the rolling
unassigned buffer, manifold drift, emerging-topic detection on unseen text --
but it is a replay at higher volume, not a forward-in-time extrapolation.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

# Located relative to this file, so a fresh clone works anywhere.
BASE = Path(__file__).resolve().parent.parent   # repo root, where data/ lives


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--first", default=str(BASE / "data/twcs_subset_200k.csv"))
    ap.add_argument("--source", default=str(BASE / "data/twcs_prepared.csv"))
    ap.add_argument("--out-dir", default=str(BASE / "data/stream"))
    ap.add_argument("--total", type=int, default=200_000)
    ap.add_argument("--chunks", type=int, default=4)
    a = ap.parse_args()

    first = pd.read_csv(a.first)
    brands = sorted(first.brand.unique())
    used = set(first.tweet_id)

    src = pd.read_csv(a.source)
    src["t"] = pd.to_datetime(src.event_time, errors="coerce")
    pool = src[src.brand.isin(brands) & ~src.tweet_id.isin(used)].dropna(subset=["t"])
    pool = pool[(pool.t >= "2017-10-01") & (pool.t < "2017-12-04")].sort_values("t")
    print(f"candidate pool: {len(pool):,} records not in the first batch")

    # Time-stratified down-sample to `total`, preserving each brand's daily
    # shape the same way the first batch was built -- otherwise the stream
    # would be a dense slice of one period rather than a feed.
    if len(pool) > a.total:
        step = len(pool) / a.total
        idx = (pd.Series(range(a.total)) * step).astype(int).values
        pool = pool.iloc[idx]
    pool = pool.sort_values("t")

    out_dir = Path(a.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cols = [c for c in first.columns if c in pool.columns]

    full = out_dir / "stream_200k.csv"
    pool[cols].to_csv(full, index=False)
    print(f"wrote {len(pool):,} -> {full}")
    print(f"  window {pool.t.min()} -> {pool.t.max()}")

    size = len(pool) // a.chunks
    for i in range(a.chunks):
        lo = i * size
        hi = len(pool) if i == a.chunks - 1 else (i + 1) * size
        c = pool.iloc[lo:hi]
        f = out_dir / f"chunk_{i+1}.csv"
        c[cols].to_csv(f, index=False)
        print(f"  chunk {i+1}: {len(c):>7,}  {c.t.min()} -> {c.t.max()}  -> {f.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
