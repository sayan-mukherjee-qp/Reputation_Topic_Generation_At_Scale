#!/usr/bin/env python3
"""Cut a quick-try slice of the base set and the stream, same shape, 1/10 size.

    python make_slice.py                    # 20k base + 30k stream (fraction 0.1)
    python make_slice.py --fraction 0.05    # 10k base + 15k stream

Each output keeps the full 61-day window and every brand: records are taken
every k-th in time order within each brand, as make_subset.py does, so daily
volume shapes (and the history/holdout time split) survive the downsampling.
Taking the first N rows instead would cover only the first few days.

Writes ../data/twcs_subset_<N>k.csv and ../data/stream<M>/chunk_1..6.csv, the
inputs the dashboard's "Slice" dataset (and BASE_CSV / STREAM_DIR) point at.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

DATA = Path(__file__).resolve().parent.parent / "data"


def stratified(df: pd.DataFrame, fraction: float) -> pd.DataFrame:
    """Every k-th record per brand in time order, then back in arrival order."""
    df = df.assign(_t=pd.to_datetime(df["event_time"], errors="coerce")).sort_values("_t")
    parts = []
    for _, g in df.groupby("brand", sort=False):
        n = max(1, round(len(g) * fraction))
        idx = (pd.Series(range(n)) * (len(g) / n)).astype(int).values
        parts.append(g.iloc[idx])
    return pd.concat(parts).sort_values("_t").drop(columns="_t")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fraction", type=float, default=0.1)
    ap.add_argument("--base", default=str(DATA / "twcs_subset_200k.csv"))
    ap.add_argument("--stream-dir", default=str(DATA / "stream300"))
    a = ap.parse_args()

    base = pd.read_csv(a.base)
    out = stratified(base, a.fraction)
    base_dst = DATA / f"twcs_subset_{round(200 * a.fraction)}k.csv"
    out.to_csv(base_dst, index=False)
    print(f"base   {len(base):>7,} -> {len(out):>6,}  {base_dst.name}  "
          f"({out.event_time.min()} -> {out.event_time.max()}, {out.brand.nunique()} brands)")

    chunks = sorted(Path(a.stream_dir).glob("chunk_*.csv"),
                    key=lambda p: int(p.stem.split("_")[1]))
    stream_dst = DATA / f"stream{round(300 * a.fraction)}"
    stream_dst.mkdir(exist_ok=True)
    total = 0
    for c in chunks:
        df = pd.read_csv(c)
        s = stratified(df, a.fraction)
        s.to_csv(stream_dst / c.name, index=False)
        total += len(s)
        print(f"  {c.name}  {len(df):>6,} -> {len(s):>5,}")
    print(f"stream {total:,} records in {len(chunks)} chunks -> {stream_dst.name}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
