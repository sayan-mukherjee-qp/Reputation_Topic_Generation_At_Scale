#!/usr/bin/env python3
"""Side-by-side benchmark of the three arms written by run_experiments.sh.

  control   reputation_topic_gpu            384-dim MiniLM-L12, frozen UMAP + HDBSCAN
  emb768    reputation_topic_gpu_768        768-dim mpnet-base, frozen UMAP + HDBSCAN
  bertopic  reputation_topic_gpu_bertopic   384-dim MiniLM-L12, BERTopic per pool

Independent coherence is scored with stream_audit's implementation (c-TF-IDF
top-10 terms, epsilon-smoothed C_npmi and C_v) against ONE fixed reference
corpus per phase -- the raw 200k CSV for the base runs, the six raw stream300
chunks for the stream -- so no arm's own segmentation can move the yardstick.

Usage:  compare_arms.py base | stream | all
Writes experiments/results/*.csv and prints the tables.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_mutual_info_score, adjusted_rand_score

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "reputation_topic_gpu"))
from stream_audit import Reference, audit_batch  # noqa: E402

ARMS = {
    "control": ROOT / "reputation_topic_gpu/out/exp_control",
    "emb768": ROOT / "reputation_topic_gpu_768/out/exp_emb768",
    "bertopic": ROOT / "reputation_topic_gpu_bertopic/out/exp_bertopic",
}
LOGS = ROOT / "experiments/logs"
RES = ROOT / "experiments/results"
DATA = ROOT / "data"


def gnu_time(step: str) -> dict:
    """Wall clock, CPU seconds and peak RSS from `/usr/bin/time -v`."""
    f = LOGS / f"{step}.time"
    if not f.exists():
        return {}
    t = f.read_text()

    def grab(pat):
        m = re.search(pat, t)
        return m.group(1) if m else None
    wall = grab(r"Elapsed \(wall clock\) time \(h:mm:ss or m:ss\): (\S+)")
    secs = None
    if wall:
        parts = [float(x) for x in wall.split(":")]
        secs = sum(p * 60 ** i for i, p in enumerate(reversed(parts)))
    user, sys_ = float(grab(r"User time \(seconds\): (\S+)") or 0), float(grab(r"System time \(seconds\): (\S+)") or 0)
    rss = grab(r"Maximum resident set size \(kbytes\): (\d+)")
    return {"wall_s": round(secs, 1) if secs else None,
            "cpu_s": round(user + sys_, 1),
            "cpu_util_x": round((user + sys_) / secs, 2) if secs else None,
            "peak_rss_gib": round(int(rss) / 1024 ** 2, 2) if rss else None}


def primary_topics(run: Path) -> pd.Series:
    """segment_id -> the topic it was assigned to first (its best match)."""
    a = pd.read_csv(run / "topic_assignments.csv", usecols=["segment_id", "topic_id"],
                    dtype=str, low_memory=False)
    return a.dropna().drop_duplicates("segment_id").set_index("segment_id")["topic_id"]


def base() -> pd.DataFrame:
    ref = Reference([DATA / "twcs_subset_200k.csv"], name="fixed 200k")
    rows, prim = [], {}
    for arm, d in ARMS.items():
        run = d / "base_200k"
        if not (run / "topics.json").exists():
            print(f"  {arm}: no base run yet")
            continue
        r, tdf = audit_batch(run, ref)
        tdf.to_csv(run / "audit_topics.csv", index=False)
        meta = json.loads((run / "run_metadata.json").read_text())
        st = json.loads((run / "stage_timing.json").read_text())
        llm = (meta.get("llm") or {}).get("stats") or {}
        r = {"arm": arm, **r,
             "embedding_dim": meta.get("embedding_dimension") or meta.get("embedding_dim"),
             "llm_calls": (meta.get("llm") or {}).get("requested"),
             "llm_failed": (meta.get("llm") or {}).get("failed"),
             "llm_tokens": (llm.get("prompt_tokens", 0) + llm.get("completion_tokens", 0)) or None,
             "capacity_merges": (meta.get("consolidation") or {}).get("capacity_merges"),
             **{f"t_{k}": v for k, v in st.items()},
             **gnu_time(f"base_{arm}")}
        rows.append(r)
        prim[arm] = primary_topics(run)
    df = pd.DataFrame(rows)

    # Agreement with the control: do the arms group the same segments together?
    if "control" in prim:
        for arm, p in prim.items():
            common = prim["control"].index.intersection(p.index)
            a, b = prim["control"].loc[common], p.loc[common]
            df.loc[df.arm == arm, "segments_both_assigned"] = len(common)
            df.loc[df.arm == arm, "AMI_vs_control"] = round(adjusted_mutual_info_score(a, b), 4)
            df.loc[df.arm == arm, "ARI_vs_control"] = round(adjusted_rand_score(a, b), 4)
    df.to_csv(RES / "base_comparison.csv", index=False)
    with pd.option_context("display.max_rows", None, "display.width", 250):
        print(df.set_index("arm").T.to_string())
    return df


def stream() -> pd.DataFrame:
    ref = Reference(sorted(DATA.glob("stream300/chunk_*.csv")), name="fixed stream300")
    rows, summaries, timings = [], [], []
    for arm, d in ARMS.items():
        labels = {}
        for i in range(1, 7):
            run = d / f"stream_{i}"
            if not (run / "topics.json").exists():
                continue
            r, tdf = audit_batch(run, ref)
            tdf.to_csv(run / "audit_topics.csv", index=False)
            rows.append({"arm": arm, **r})
            labels[i] = dict(zip(tdf["topic_id"], tdf["label"]))
            print(f"  {arm} batch {i} audited", flush=True)
        if len(labels) > 1:
            common = set.intersection(*(set(v) for v in labels.values()))
            changed = sum(1 for t in common if len({labels[i][t] for i in labels}) > 1)
            rows.append({"arm": arm, "batch": "CHURN", "entries": len(common),
                         "unique_ids": changed})
        if (d / "stream_summary.csv").exists():
            summaries.append(pd.read_csv(d / "stream_summary.csv").assign(arm=arm))
        if (d / "stage_timing.csv").exists():
            timings.append(pd.read_csv(d / "stage_timing.csv").assign(arm=arm))
        rows.append({"arm": arm, "batch": "PROCESS", **gnu_time(f"stream_{arm}")})
    audit = pd.DataFrame(rows)
    audit.to_csv(RES / "stream_audit.csv", index=False)
    if summaries:
        pd.concat(summaries).to_csv(RES / "stream_summary_all.csv", index=False)
    if timings:
        pd.concat(timings).to_csv(RES / "stream_stage_timing_all.csv", index=False)
    with pd.option_context("display.max_rows", None, "display.max_columns", None, "display.width", 250):
        print(audit.set_index(["arm", "batch"]).T.to_string())
        if summaries:
            print(pd.concat(summaries).to_string(index=False))
        if timings:
            print(pd.concat(timings).set_index(["arm", "run"]).round(1).T.to_string())
    return audit


if __name__ == "__main__":
    RES.mkdir(parents=True, exist_ok=True)
    what = sys.argv[1] if len(sys.argv) > 1 else "all"
    if what in ("base", "all"):
        base()
    if what in ("stream", "all"):
        stream()
