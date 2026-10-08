#!/usr/bin/env python3
"""Build a CPU-tractable topic-detection subset from twcs_prepared.csv.

Keeps a contiguous 61-day window so the daily time series and the 28-day
baseline stay meaningful, then caps each brand with a time-stratified
systematic sample (every k-th record in time order) so the shape of each
brand's daily volume survives the downsampling.
"""
import pandas as pd

SRC = "../data/twcs_prepared.csv"
DST = "../data/twcs_subset_150k.csv"
START, END = "2017-10-01", "2017-12-01"
CAP = 12_850
# Top volume brands, plus the two that hit 100% noise on the small slice.
EXTRA = ["McDonalds", "MicrosoftHelps"]

d = pd.read_csv(SRC)
t = pd.to_datetime(d.event_time)
d = d[(t >= START) & (t < END)]
print(f"window {START}..{END}: {len(d):,} records")

top = d.brand.value_counts().head(10).index.tolist()
brands = top + [b for b in EXTRA if b not in top]
d = d[d.brand.isin(brands)].sort_values("event_time")

parts = []
for b, g in d.groupby("brand", sort=False):
    if len(g) > CAP:
        step = len(g) / CAP
        idx = (pd.Series(range(CAP)) * step).astype(int).values
        g = g.iloc[idx]
    parts.append(g)
    print(f"  {b:<16} {len(g):>7,}")

out = pd.concat(parts).sort_values("event_time")
out.to_csv(DST, index=False)
print(f"\nwrote {len(out):,} rows -> {DST}")
print(f"date range: {out.event_time.min()} -> {out.event_time.max()}")
