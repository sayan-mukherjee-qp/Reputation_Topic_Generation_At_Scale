#!/usr/bin/env python3
"""Convert Kaggle twcs.csv into the schema reputation_topic_detection.py expects.

Output columns: tweet_id, event_time, author_id, text, brand

Only inbound (customer-authored) tweets are kept: those are the reputation
signal. Brand is resolved by, in order of preference:
  1. a literal non-numeric @mention that matches a known support account
  2. the author of the tweet this one replies to (in_response_to_tweet_id)
  3. the author of the tweet that replies to this one (response_tweet_id)
Customer handles in twcs are anonymized to digit strings, so "non-numeric
author_id" is a reliable test for a company account.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import pandas as pd

MENTION_RE = re.compile(r"@(\w+)")

SRC = Path(sys.argv[1] if len(sys.argv) > 1 else "../data/twcs.csv")
DST = Path(sys.argv[2] if len(sys.argv) > 2 else "../data/twcs_prepared.csv")


def main() -> None:
    df = pd.read_csv(SRC, dtype=str, keep_default_na=False)
    print(f"loaded {len(df):,} rows")

    df["inbound"] = df["inbound"].str.strip().str.lower() == "true"
    company = set(df.loc[~df["author_id"].str.fullmatch(r"\d+"), "author_id"].unique())
    print(f"support accounts: {len(company)}")

    # tweet_id -> author_id, for resolving both directions of the reply chain.
    author_of = dict(zip(df["tweet_id"], df["author_id"]))

    inb = df[df["inbound"]].copy()
    print(f"inbound (customer) tweets: {len(inb):,}")

    # 1. brand from a literal mention of a known support account.
    def from_mention(text: str) -> str:
        for handle in MENTION_RE.findall(text):
            if handle in company:
                return handle
        return ""

    brand = inb["text"].map(from_mention)

    # 2. fall back to the parent tweet's author.
    def company_author(tid: str) -> str:
        a = author_of.get(tid, "")
        return a if a in company else ""

    need = brand == ""
    brand.loc[need] = inb.loc[need, "in_response_to_tweet_id"].map(company_author)

    # 3. fall back to the author of the first reply.
    need = brand == ""
    first_reply = inb.loc[need, "response_tweet_id"].str.split(",").str[0]
    brand.loc[need] = first_reply.map(company_author)

    inb["brand"] = brand
    resolved = inb[inb["brand"] != ""].copy()
    print(f"brand resolved: {len(resolved):,} ({len(resolved)/len(inb):.1%} of inbound)")

    # twcs stores "Tue Oct 31 22:10:47 +0000 2017".
    ts = pd.to_datetime(resolved["created_at"], format="%a %b %d %H:%M:%S %z %Y", utc=True)
    resolved["event_time"] = ts.dt.tz_convert("UTC").dt.strftime("%Y-%m-%d %H:%M:%S")

    out = resolved[["tweet_id", "event_time", "author_id", "text", "brand"]]
    out = out.sort_values("event_time")
    out.to_csv(DST, index=False)

    print(f"\nwrote {len(out):,} rows -> {DST}")
    print(f"date range: {out['event_time'].min()} -> {out['event_time'].max()}")
    print(f"brands: {out['brand'].nunique()}")
    print("\ntop 20 brands by volume:")
    print(out["brand"].value_counts().head(20).to_string())


if __name__ == "__main__":
    main()
