#!/usr/bin/env python3
"""Event-recall benchmark: did the pipeline surface things that really happened?

Coherence and assignment coverage are both internal metrics -- they describe the
shape of the output, not whether it is useful. This asks the business question
instead: of a list of events known to have occurred in the corpus window, how
many did the system surface, how big was the topic, and how many days after the
event began did the topic's own record stream start?

Matching is deliberately keyword-based rather than embedding-based. Using the
same encoder to both build and grade the topics would make the benchmark
partially self-confirming; independent keyword evidence is weaker per match but
does not share the failure modes of the thing under test.

Usage:  event_recall.py OUTPUT_DIR [OUTPUT_DIR ...]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

# Events verified to have occurred within Oct 1 - Nov 30 2017, each with the
# vocabulary a customer would plausibly use. `all_of` groups are ANDed, so a
# match needs evidence from every group; `any_of` inside a group is ORed.
EVENTS = [
    {
        "name": "McDonald's Szechuan sauce shortage",
        "brand": "McDonalds",
        "onset": "2017-10-07",
        "all_of": [["szechuan", "sichuan"]],
    },
    {
        "name": "iOS 11 complaint wave",
        "brand": "AppleSupport",
        "onset": "2017-10-01",
        "all_of": [["ios 11", "ios11", "ios 11.0", "ios 11.1"]],
    },
    {
        "name": "Windows 10 1709 driver regression",
        "brand": "MicrosoftHelps",
        "onset": "2017-10-17",
        "all_of": [["1709", "fall creators", "creators update", "driver", "radeon"]],
    },
    {
        "name": "Amazon Prime membership / billing",
        "brand": "AmazonHelp",
        "onset": "2017-10-01",
        "all_of": [["prime"], ["charge", "charged", "billing", "member", "membership", "pay"]],
    },
    {
        "name": "Virgin Trains December strike",
        "brand": "VirginTrains",
        "onset": "2017-11-20",
        "all_of": [["strike", "rmt", "industrial action"]],
    },
    {
        "name": "Black Friday / Cyber Monday",
        "brand": None,
        "onset": "2017-11-20",
        "all_of": [["black friday", "cyber monday", "blackfriday"]],
    },
    {
        "name": "Uber driver cancellation / fare disputes",
        "brand": "Uber_Support",
        "onset": "2017-10-01",
        "all_of": [["driver", "drivers"], ["cancel", "cancelled", "charged", "fare", "fee"]],
    },
    {
        "name": "Comcast/Xfinity outage",
        "brand": "comcastcares",
        "onset": "2017-10-01",
        "all_of": [["outage", "down", "no internet", "no service"]],
    },
    {
        "name": "Spotify iOS/playback failures",
        "brand": "SpotifyCares",
        "onset": "2017-10-01",
        "all_of": [["playlist", "playback", "songs", "shuffle", "offline"]],
    },
    {
        "name": "Southwest / American flight delays",
        "brand": None,
        "onset": "2017-10-01",
        "all_of": [["delay", "delayed", "cancelled flight", "cancellation"]],
    },
]


def hits(text: str, groups) -> bool:
    t = str(text).lower()
    return all(any(k in t for k in group) for group in groups)


def evaluate(out_dir: Path) -> pd.DataFrame:
    summary = pd.read_csv(out_dir / "topic_hot_summary.csv")
    assign = pd.read_csv(out_dir / "topic_assignments.csv",
                         usecols=["topic_id", "record_id", "event_time", "clean_text", "brand"])
    assign["dt"] = pd.to_datetime(assign["event_time"], errors="coerce", utc=True, format="mixed")

    rows = []
    for ev in EVENTS:
        m = assign["clean_text"].map(lambda t: hits(t, ev["all_of"]))
        if ev["brand"]:
            m &= assign["brand"].astype(str).eq(ev["brand"])
        ev_rows = assign[m]
        if ev_rows.empty:
            rows.append({"event": ev["name"], "found": False, "matched_records": 0,
                         "topic_id": None, "topic_label": None, "topic_size": 0,
                         "precision_in_topic": None, "onset": ev["onset"],
                         "first_seen": None, "lag_days": None, "is_junk": None,
                         "hot_status": None})
            continue

        # The topic that best represents this event: most matching records.
        best = ev_rows.groupby("topic_id")["record_id"].nunique().idxmax()
        n_in_topic = ev_rows[ev_rows.topic_id == best]["record_id"].nunique()
        srow = summary[summary.topic_id == best]
        size = int(srow["size"].iloc[0]) if len(srow) else 0
        first = ev_rows[ev_rows.topic_id == best]["dt"].min()
        onset = pd.Timestamp(ev["onset"], tz="UTC")
        rows.append({
            "event": ev["name"],
            "found": True,
            "matched_records": int(ev_rows["record_id"].nunique()),
            "topic_id": best,
            "topic_label": srow["label"].iloc[0] if len(srow) else None,
            "topic_size": size,
            # Share of the topic that is actually about the event: low means the
            # event is buried inside a broad topic rather than isolated by it.
            "precision_in_topic": round(n_in_topic / size, 3) if size else None,
            "onset": ev["onset"],
            "first_seen": None if pd.isna(first) else first.date().isoformat(),
            "lag_days": None if pd.isna(first) else (first.normalize() - onset.normalize()).days,
            "is_junk": bool(srow["is_junk"].iloc[0]) if len(srow) and "is_junk" in srow else None,
            "hot_status": srow["status"].iloc[0] if len(srow) else None,
        })
    return pd.DataFrame(rows)


def main() -> None:
    dirs = [Path(d) for d in (sys.argv[1:] or ["out2x2_A"])]
    for d in dirs:
        df = evaluate(d)
        df.to_csv(d / "event_recall.csv", index=False)
        found = int(df.found.sum())
        surfaced = int(((df.found) & (~df.is_junk.fillna(False))).sum())
        print(f"\n=== {d} ===")
        print(f"  events detected      {found}/{len(df)}  ({found/len(df):.0%})")
        print(f"  detected & ranked    {surfaced}/{len(df)}  (not demoted as junk)")
        prec = df.precision_in_topic.dropna()
        if len(prec):
            print(f"  median precision-in-topic {prec.median():.3f}   "
                  f"(share of the matched topic actually about the event)")
        lag = df.lag_days.dropna()
        if len(lag):
            print(f"  median detection lag {lag.median():.0f} days")
        cols = ["event", "found", "matched_records", "topic_id", "topic_size",
                "precision_in_topic", "lag_days", "hot_status", "is_junk"]
        print(df[cols].to_string(index=False))


if __name__ == "__main__":
    main()
