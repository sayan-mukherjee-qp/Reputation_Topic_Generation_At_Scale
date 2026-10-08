#!/usr/bin/env python3
"""Rewrite an output's topics.json so every topic carries ALL its tweets.

The reducer-comparison runs were made with --tweets-per-topic 12 to keep their
files small. This reproduces exactly what the pipeline writes with
--tweets-per-topic -1, from `topic_assignments.csv`, rather than spending
another 30 minutes re-clustering to obtain the same bytes.

The ordering below mirrors the pipeline: rank by similarity descending (rows
from candidate/micro discovery carry no similarity and are treated as 1.0,
since they are the only evidence such a topic has), then keep one row per
(topic, record).

Usage:  expand_topic_tweets.py SOURCE_CSV OUT_DIR [OUT_DIR ...]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd


def expand(out_dir: Path, raw_lookup: dict) -> None:
    tj = out_dir / "topics.json"
    topics = json.loads(tj.read_text())

    a = pd.read_csv(out_dir / "topic_assignments.csv",
                    usecols=["topic_id", "record_id", "similarity"],
                    dtype={"record_id": str})
    a = a[a["topic_id"].notna()].copy()
    a["_sim"] = pd.to_numeric(a["similarity"], errors="coerce").fillna(1.0)
    a = a.sort_values("_sim", ascending=False).drop_duplicates(["topic_id", "record_id"])
    by_topic = {tid: g["record_id"].tolist() for tid, g in a.groupby("topic_id", sort=False)}

    total = 0
    for t in topics:
        ids = by_topic.get(t["topic_id"], [])
        t["tweets"] = [raw_lookup.get(str(r), "") for r in ids]
        total += len(t["tweets"])

    tj.write_text(json.dumps(topics, indent=2, ensure_ascii=False), encoding="utf-8")

    _s = pd.read_csv(out_dir / "topic_hot_summary.csv", usecols=["topic_id", "size"])
    sizes = dict(zip(_s["topic_id"], _s["size"]))
    bad = [(t["topic_id"], len(t["tweets"]), sizes.get(t["topic_id"]))
           for t in topics
           if t["topic_id"] in sizes and len(t["tweets"]) != sizes[t["topic_id"]]]
    mb = tj.stat().st_size / 1e6
    print(f"{out_dir.name:<14} topics={len(topics):>5}  tweets={total:>8,}  "
          f"{mb:>6.1f} MB  mismatches={len(bad)}")
    if bad:
        print(f"   first mismatches (topic, stored, size): {bad[:3]}")


def main() -> None:
    if len(sys.argv) < 3:
        raise SystemExit(__doc__)
    src = pd.read_csv(sys.argv[1], usecols=["tweet_id", "text"], dtype={"tweet_id": str})
    raw_lookup = dict(zip(src["tweet_id"].astype(str), src["text"].astype(str)))
    print(f"source records: {len(raw_lookup):,}")
    for d in sys.argv[2:]:
        expand(Path(d), raw_lookup)


if __name__ == "__main__":
    main()
