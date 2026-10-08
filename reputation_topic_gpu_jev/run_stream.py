#!/usr/bin/env python3
"""Replay the stream chunk by chunk and record how the model holds up.

Each chunk is a separate pipeline run that inherits the previous run's topic
registry (`--previous-topics`) and the frozen per-brand manifolds, and shares
one rolling unassigned buffer. That is the production loop: nothing is
re-discovered from scratch, topic IDs are expected to survive, and drift is
expected to grow until it trips the refit threshold.

With --window N each batch clusters over the last N chunks (the newest is
the incremental holdout, the older ones discovery history), so a ~540-topic
inventory is fed ~3x the evidence per batch without re-embedding anything.

Usage:  run_stream.py --base-run <dir> --chunks data/stream/chunk_*.csv [--window 3]
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from collections import deque
from pathlib import Path

import pandas as pd

# Everything is located relative to this file, so a fresh clone works anywhere.
HERE = Path(__file__).resolve().parent
BASE = HERE.parent                  # repo root, where data/ lives
RUNS = HERE / "out"                 # run outputs, buffers and logs (gitignored)
CACHE = HERE / "embed_cache"        # shared embedding cache (gitignored)
SCRIPT = HERE / "reputation_topic_detection.py"

# Prefer the project venv from `uv sync`; otherwise reuse whatever interpreter
# is running this script.
_VENV_PYTHON = HERE / ".venv/bin/python"
VENV = _VENV_PYTHON if _VENV_PYTHON.exists() else Path(sys.executable)

COMMON = [
    "--model", "sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
    "--min-similarity", "0.50", "--min-cluster-size", "30", "--min-samples", "10",
    "--brand-scoped-assignment",
    "--merge-duplicate-topics", "--duplicate-similarity", "0.92",
    "--flag-junk-topics", "--junk-coherence-veto", "0.10", "--junk-coherence-min-size", "30",
    "--micro-clusters", "--tweets-per-topic", "-1",
    "--reducer", "umap", "--umap-components", "5", "--umap-neighbors", "30",
    "--umap-min-dist", "0.0", "--umap-per-brand",
    "--cluster-selection-epsilon", "0.2", "--candidate-margin", "0.05",
    "--split-max-size", "2500", "--split-max-child-similarity", "0.92",
    "--alert-min-coherence", "0.05", "--alert-min-size", "250",
    "--residual-ack-ratio", "0.40", "--event-recall",
    "--recover-unassigned", "--umap-drift-threshold", "0.45",
    "--umap-reference-size", "50000",
    "--carry-forward-topics", "--topic-max-age-days", "45",
    # out300w review, P1/P5: registry capacity and consolidation.
    "--recover-match-existing", "--max-live-topics", "600",
    "--consolidate-min-similarity", "0.75", "--retire-idle",
    # --secondary-margin 0.05 cut inflation 1.82x -> 1.35x but cost event
    # recall 0.508 -> 0.431 in a same-input batch-6 ablation; left off.
    "--secondary-margin", "0",
    # EXPERIMENT (Laya copy): batches carry the registry and match incoming
    # records to it; a brand's bucket is clustered once it holds 100 segments
    # or its oldest has waited 3 days. Decisions come from TypeSafe's Jev
    # (--decision-backend jev, default; TYPESAFE_API_KEY in .env), GPT via AI
    # Router v1 (--decision-backend gpt) or Laya (--decision-backend laya).
    "--bucket-trigger-size", "100", "--bucket-max-wait-days", "3",
]
DEFAULT_MATCHER = "laya"


def unassigned_rates(run_dir, assign, hold_csv=None):
    """Segment-level unassigned share, over the window and over the holdout.

    topic_assignments.csv holds UNASSIGNED rows only for history; a holdout
    segment that matched nothing lives in unassigned_recent_records.csv. So
    the share is taken against segments.csv: segments with no topic anywhere.
    """
    seg = pd.read_csv(run_dir / "segments.csv", usecols=["segment_id", "record_id"],
                      dtype=str, low_memory=False)
    got = set(assign.loc[assign["topic_id"].notna(), "segment_id"].astype(str))
    miss = ~seg["segment_id"].isin(got)
    win = round(100 * float(miss.mean()), 2)
    hold = None
    if hold_csv and Path(hold_csv).exists():
        ids = set(pd.read_csv(hold_csv, usecols=["tweet_id"])["tweet_id"].astype(str))
        h = seg["record_id"].isin(ids)
        hold = round(100 * float(miss[h].mean()), 2) if h.any() else None
    return win, hold


def summarise(run_dir: Path, prev_dir: Path | None) -> dict:
    topics = json.loads((run_dir / "topics.json").read_text())
    meta = json.loads((run_dir / "run_metadata.json").read_text())
    summary = pd.read_csv(run_dir / "topic_hot_summary.csv")
    assign = pd.read_csv(run_dir / "topic_assignments.csv", low_memory=False)

    ids = [t["topic_id"] for t in topics]
    live = [t for t in topics if t.get("size")]
    ts_max = pd.to_datetime(assign["event_time"], errors="coerce", utc=True, format="mixed").max()
    ls = pd.to_datetime(pd.Series([t.get("last_seen_at") for t in topics]),
                        errors="coerce", utc=True, format="mixed")
    llm = meta.get("llm") or {}
    row = {
        "run": run_dir.name,
        "records": meta.get("records"),
        "holdout_records": (meta.get("window") or {}).get("holdout_records"),
        "topics": len(topics),
        "unique_ids": len(set(ids)),
        "duplicate_ids": len(ids) - len(set(ids)),
        "median_live_size": (int(pd.Series([t["size"] for t in live]).median()) if live else 0),
        "last_seen_beyond_window": int((ls > ts_max).sum()),
        "emerging": sum(1 for t in topics if str(t.get("status", "")).endswith("EMERGING")),
        "recovered": sum(1 for t in topics if str(t["topic_id"]).startswith("T_REC_")),
        "residual": sum(1 for t in topics if t.get("residual_cluster")),
        "junk": sum(1 for t in topics if t.get("is_junk")),
        "dormant": sum(1 for t in topics if t.get("status") == "DORMANT"),
        "zero_size": sum(1 for t in topics if not t.get("size")),
        "alerts": int(summary["status"].isin(["HOT", "TRENDING"]).sum()),
        "alerts_suppressed": int(summary.get("alert_suppressed", pd.Series(dtype=bool)).sum()),
        "drift": meta.get("umap_drift"),
        "refit": bool((meta.get("umap") or {}).get("refit_trigger")),
        # Two definitions, named apart (review P3): over every row the window
        # produced, and over the holdout chunk only -- the new data.
        "unassigned_window_pct": unassigned_rates(
            run_dir, assign, (meta.get("window") or {}).get("holdout_csv"))[0],
        "unassigned_holdout_pct": unassigned_rates(
            run_dir, assign, (meta.get("window") or {}).get("holdout_csv"))[1],
        "mean_coherence": round(float(pd.Series(
            [t["coherence"] for t in topics if t.get("coherence") is not None]).mean()), 4),
        "live_topics": len(live),
        "records_per_live_topic": round((meta.get("records") or 0) / max(1, len(live)), 1),
        # Review step 3: the published set is what consumers see.
        "kept_topics": sum(1 for t in live if not t.get("suppressed")),
        "kept_vol_pct": round(100 * sum(t["size"] for t in live if not t.get("suppressed"))
                              / max(1, sum(t["size"] for t in live)), 1),
        "kept_mean_coherence": round(float(pd.Series(
            [t["coherence"] for t in live
             if not t.get("suppressed") and t.get("coherence") is not None]).mean()), 4),
        "suppressed": sum(1 for t in topics if t.get("suppressed")),
        "campaign": sum(1 for t in topics if t.get("campaign_cluster")),
        "llm_not_substantive": sum(1 for t in topics if t.get("llm_substantive") is False),
        "unclear": sum(1 for t in topics if t.get("label") == "Unclear topic"),
        "llm_requested": llm.get("requested"),
        "llm_failures": llm.get("failed"),
        "llm_reused": llm.get("reused"),
        "llm_label_merges": llm.get("label_merges"),
        "capacity_merges": (meta.get("consolidation") or {}).get("capacity_merges"),
        "retired_idle": (meta.get("consolidation") or {}).get("retired_idle"),
        "llm_tokens": ((llm.get("stats") or {}).get("prompt_tokens", 0)
                       + (llm.get("stats") or {}).get("completion_tokens", 0)) or None,
    }
    # EXPERIMENT (Laya copies): how the matcher filed the batch's new records.
    m = meta.get("matcher") or {}
    row["matcher"] = m.get("matcher")
    for k in ("asked_laya", "laya_topic", "laya_second_topic", "laya_other", "laya_no_issue",
              "laya_low_confidence", "laya_failed_fallback", "cosine_clear", "cosine_miss",
              "centroids_updated", "relabel_requested"):
        if k in m:
            row[k] = m[k]
    if m.get("laya"):
        row["laya_calls"] = m["laya"].get("calls")
        row["laya_seconds"] = m["laya"].get("seconds")
    if m.get("bucket"):
        row["deferred_segments"] = m["bucket"].get("deferred_segments")

    if prev_dir is not None:
        prev_ids = {t["topic_id"] for t in json.loads((prev_dir / "topics.json").read_text())}
        cur_ids = {t["topic_id"] for t in topics}
        dec = run_dir / "topic_stability_decisions.csv"
        if dec.exists():
            d = pd.read_csv(dec)
            row["matched_previous"] = int((d.decision == "MATCH_PREVIOUS").sum())
            row["new_ids"] = int((d.decision == "NEW_STABLE_ID").sum())
        row["id_retention"] = round(len(prev_ids & cur_ids) / max(1, len(prev_ids)), 4)

    er = run_dir / "event_recall.csv"
    if er.exists():
        e = pd.read_csv(er)
        f = e[e["found"]]
        row["events_found"] = f"{len(f)}/{len(e)}"
        row["recall_best_topic"] = round(float(f["recall_in_best_topic"].mean()), 4)
        row["topics_for_80"] = round(float(f["topics_for_80pct"].mean()), 2)
    return row


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-run", required=True)
    ap.add_argument("--model", required=True, help="frozen UMAP artifact to carry forward")
    ap.add_argument("--chunks", nargs="+", required=True)
    ap.add_argument("--out-prefix", default=str(RUNS / "out_stream"))
    ap.add_argument("--buffer", default=str(RUNS / "buf_stream"))
    ap.add_argument("--window", type=int, default=1,
                    help="Chunks per batch: the newest plus up to N-1 earlier ones as history")
    ap.add_argument("--label-method", default="llm", choices=["ctfidf", "llm"])
    ap.add_argument("--label-llm-samples", default="10")
    ap.add_argument("--label-llm-workers", default="12")
    ap.add_argument("--matcher", default=DEFAULT_MATCHER, choices=["centroid", "laya", "hybrid"],
                    help="How each batch files incoming records (see reputation_topic_detection.py)")
    ap.add_argument("--laya-workers", default="4")
    ap.add_argument("--decision-backend", default="jev", choices=["jev", "gpt", "laya"],
                    help="Who answers the matcher's per-record questions (default jev)")
    ap.add_argument("--gpt-workers", default="12")
    ap.add_argument("--jev-workers", default="16")
    ap.add_argument("--jev-max-rpm", default="1000")
    # GPU / embedding, passed straight to every batch.
    ap.add_argument("--device", default=None, help="cuda, cuda:1, cpu (default: auto)")
    ap.add_argument("--fp16", action="store_true")
    ap.add_argument("--max-vram-gb", default=None)
    ap.add_argument("--batch-size", default=None)
    ap.add_argument("--embed-cache", default=str(CACHE),
                    help="Per-chunk embedding cache. Keep it on with --window > 1: it is what "
                         "stops each chunk being re-embedded once per window it appears in. "
                         "'' disables it")
    ap.add_argument("--echo", action="store_true",
                    help="Also echo each batch's log to stdout as it runs, prefixed '  | '")
    a = ap.parse_args()

    Path(a.out_prefix).parent.mkdir(parents=True, exist_ok=True)

    prev = Path(a.base_run)
    rows = [summarise(prev, None) | {"run": "batch0 (" + prev.name + ")"}]
    print(f"base registry: {rows[0]['topics']} topics\n", flush=True)

    timings = []
    t_stream = time.perf_counter()
    for i, chunk in enumerate(a.chunks, start=1):
        out = Path(f"{a.out_prefix}_{i}")
        history = a.chunks[max(0, i - a.window):i - 1]
        cmd = [str(VENV), str(SCRIPT), chunk, "--out", str(out),
               "--previous-topics", str(prev / "topics.json"),
               "--umap-model", a.model, "--buffer", a.buffer] + COMMON + [
               "--label-method", a.label_method,
               "--label-llm-samples", a.label_llm_samples,
               "--label-llm-workers", a.label_llm_workers,
               "--matcher", a.matcher, "--laya-workers", a.laya_workers,
               "--decision-backend", a.decision_backend, "--gpt-workers", a.gpt_workers,
               "--jev-workers", a.jev_workers, "--jev-max-rpm", a.jev_max_rpm,
               # One coherence yardstick for every batch: the base run's corpus.
               "--coherence-reference", str(Path(a.base_run) / "segments.csv")]
        if history:
            cmd += ["--history-csv", *history]
        if a.embed_cache:
            cmd += ["--embed-cache", a.embed_cache]
        if a.device:
            cmd += ["--device", a.device]
        if a.fp16:
            cmd += ["--fp16"]
        if a.max_vram_gb:
            cmd += ["--max-vram-gb", str(a.max_vram_gb)]
        if a.batch_size:
            cmd += ["--batch-size", str(a.batch_size)]
        # The batch log is written as the run goes, not after it, so it can be
        # followed live (tail -f, or the dashboard). Opened before the banner
        # below, so a reader that keys off the banner never sees a stale log.
        log_path = Path(a.out_prefix).parent / f"stream_chunk{i}.log"
        tail: deque[str] = deque(maxlen=40)
        with log_path.open("w", encoding="utf-8") as log:
            print(f"=== chunk {i}: {Path(chunk).name} -> {out.name} ===", flush=True)
            t0 = time.perf_counter()
            proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                    text=True, errors="replace",
                                    env={**os.environ, "PYTHONUNBUFFERED": "1"})
            for line in proc.stdout:
                log.write(line)
                log.flush()
                tail.append(line)
                if a.echo:
                    print(f"  | {line}", end="", flush=True)
            rc = proc.wait()
        dt = time.perf_counter() - t0
        if rc != 0:
            print(f"  FAILED rc={rc}\n{''.join(tail)[-2000:]}", flush=True)
            return 1
        row = summarise(out, prev)
        row["runtime_s"] = round(dt, 1)
        rows.append(row)
        st = out / "stage_timing.json"
        if st.exists():
            timings.append({"run": out.name, **json.loads(st.read_text()),
                            "wall_clock_s": round(dt, 1)})
        print(f"  topics={row['topics']} retention={row.get('id_retention')} "
              f"drift={row['drift']} refit={row['refit']} alerts={row['alerts']} "
              f"events={row.get('events_found')} ({dt/60:.1f} min)", flush=True)
        prev = out

    df = pd.DataFrame(rows)
    out_csv = Path(a.out_prefix).parent / "stream_summary.csv"
    df.to_csv(out_csv, index=False)
    print("\n=== stream summary ===")
    print(df.to_string(index=False))
    print(f"\nWrote {out_csv}")
    if timings:
        tdf = pd.DataFrame(timings).set_index("run")
        tdf.loc["ALL"] = tdf.sum(numeric_only=True)
        t_csv = Path(a.out_prefix).parent / "stage_timing.csv"
        tdf.to_csv(t_csv)
        print("\n=== stage timing (s) ===")
        print(tdf.round(1).T.to_string())
        print(f"\nWrote {t_csv}; stream wall clock {time.perf_counter() - t_stream:.0f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
