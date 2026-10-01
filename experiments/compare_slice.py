#!/usr/bin/env python3
"""Compare the slice-test arms written by slice_test.sh.

Same audit as compare_arms.py (stream_audit's scorer), against fixed reference
corpora: the raw 20k slice for base runs, the six raw stream30 chunks for the
stream. Adds a brand-switch count per stream batch -- topic IDs present in both
this batch and the previous one whose brand changed -- which is what the
same-brand ID inheritance fix exists to drive to zero.

Usage:  compare_slice.py          (also counts brand switches in the
                                   pre-fix 300k streams, for reference)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "reputation_topic_gpu"))
from stream_audit import Reference, audit_batch  # noqa: E402

DATA = ROOT / "data"
LOGS = ROOT / "experiments/logs_slice"
RES = ROOT / "experiments/results_slice"
ARMS = {
    "control": ROOT / "reputation_topic_gpu/out/slice_control",
    "e768_old": ROOT / "reputation_topic_gpu_768/out/slice_e768_old",
    "e768_cal": ROOT / "reputation_topic_gpu_768/out/slice_e768_cal",
    "e768_margin": ROOT / "reputation_topic_gpu_768/out/slice_e768_margin",
    "e768_floors": ROOT / "reputation_topic_gpu_768/out/slice_e768_floors",
}
PREFIX_300K = {  # yesterday's streams, run before the fix
    "control_300k_prefix": ROOT / "reputation_topic_gpu/out/exp_control",
    "e768_300k_prefix": ROOT / "reputation_topic_gpu_768/out/exp_emb768",
}


def slice_time(step: str) -> dict:
    import re
    f = LOGS / f"{step}.time"
    if not f.exists():
        return {}
    t = f.read_text()
    m = re.search(r"Elapsed \(wall clock\) time \(h:mm:ss or m:ss\): (\S+)", t)
    secs = sum(float(p) * 60 ** i for i, p in enumerate(reversed(m.group(1).split(":")))) if m else None
    rss = re.search(r"Maximum resident set size \(kbytes\): (\d+)", t)
    return {"wall_s": round(secs, 1) if secs else None,
            "peak_rss_gib": round(int(rss.group(1)) / 1024 ** 2, 2) if rss else None}


def brand_switches(prev: Path, cur: Path) -> int:
    def brands(d):
        return {t["topic_id"]: str(t["brand"]) for t in json.loads((d / "topics.json").read_text())}
    p, c = brands(prev), brands(cur)
    return sum(1 for k in p.keys() & c.keys() if p[k] != c[k])


def run_stats(run: Path) -> dict:
    meta = json.loads((run / "run_metadata.json").read_text())
    st = json.loads((run / "stage_timing.json").read_text())
    llm = meta.get("llm") or {}
    s = llm.get("stats") or {}
    return {"thresholds": meta.get("thresholds"),
            "llm_calls": llm.get("requested"), "llm_failed": llm.get("failed"),
            "llm_error": s.get("last_error"),
            "llm_tokens": (s.get("prompt_tokens", 0) + s.get("completion_tokens", 0)) or None,
            "label_merges": llm.get("label_merges"),
            "capacity_merges": (meta.get("consolidation") or {}).get("capacity_merges"),
            "drift": meta.get("umap_drift"),
            "t_embedding": st.get("embedding", st.get("embedding (cached)")),
            "t_total": st.get("TOTAL"),
            "t_excl_embedding": round(st.get("TOTAL", 0) - st.get("embedding", st.get("embedding (cached)", 0)), 1)}


def main() -> int:
    RES.mkdir(parents=True, exist_ok=True)
    rows = []
    ref = Reference([DATA / "twcs_subset_20k.csv"], name="fixed 20k slice")
    for arm, d in ARMS.items():
        run = d / "base_20k"
        if (run / "topics.json").exists():
            r, tdf = audit_batch(run, ref)
            tdf.to_csv(run / "audit_topics.csv", index=False)
            rows.append({"arm": arm, **r, **run_stats(run), **slice_time(f"base_{arm}")})
    sref = Reference(sorted(DATA.glob("stream30/chunk_*.csv")), name="fixed stream30")
    for arm, d in ARMS.items():
        prev = d / "base_20k"
        for i in range(1, 7):
            run = d / f"stream_{i}"
            if not (run / "topics.json").exists():
                continue
            r, tdf = audit_batch(run, sref)
            tdf.to_csv(run / "audit_topics.csv", index=False)
            summ = pd.read_csv(d / "stream_summary.csv")
            srow = summ[summ["run"] == run.name]
            extra = srow.iloc[0][["id_retention", "matched_previous", "new_ids"]].to_dict() if len(srow) else {}
            rows.append({"arm": arm, **r, **run_stats(run), **extra,
                         "brand_switches": brand_switches(prev, run)})
            prev = run
        rows.append({"arm": arm, "batch": "STREAM_PROCESS", **slice_time(f"stream_{arm}")})
    df = pd.DataFrame(rows)
    df.to_csv(RES / "slice_audit.csv", index=False)

    keep = ["arm", "batch", "live_topics", "median_size", "unassigned_window_pct",
            "unassigned_holdout_pct", "inflation", "C_v", "C_npmi", "kept_C_npmi", "kept_C_v",
            "kept_vol_pct", "alerts", "alerts_neg_npmi", "events", "recall", "topics_for_80",
            "unclear_topics", "duplicate_ids", "own_brand_pct", "mis_tagged_topics",
            "id_retention", "new_ids", "brand_switches", "label_merges", "capacity_merges",
            "llm_calls", "llm_failed", "llm_tokens", "drift", "t_embedding", "t_excl_embedding",
            "t_total", "wall_s", "peak_rss_gib"]
    with pd.option_context("display.max_rows", None, "display.max_columns", None, "display.width", 250):
        print(df[[c for c in keep if c in df.columns]].set_index(["arm", "batch"]).T.to_string())

    print("\nBrand switches in the pre-fix 300k streams (same count, for reference):")
    for name, d in PREFIX_300K.items():
        prev, out = d / "base_200k", []
        for i in range(1, 7):
            cur = d / f"stream_{i}"
            if cur.exists():
                out.append(brand_switches(prev, cur))
                prev = cur
        print(f"  {name:22s} {out}")
    print(f"\nwrote {RES / 'slice_audit.csv'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
