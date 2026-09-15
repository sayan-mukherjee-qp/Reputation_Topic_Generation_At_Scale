#!/usr/bin/env python3
"""Replay the stream chunk by chunk and record how the model holds up.

Each chunk is a separate pipeline run that inherits the previous run's topic
registry (`--previous-topics`) and the frozen per-brand manifolds, and shares
one rolling unassigned buffer. That is the production loop: nothing is
re-discovered from scratch, topic IDs are expected to survive, and drift is
expected to grow until it trips the refit threshold.

Usage:  run_stream.py --base-run <dir> --chunks data/stream/chunk_*.csv
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
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
    "--model", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    "--min-similarity", "0.50", "--min-cluster-size", "30", "--min-samples", "10",
    "--brand-scoped-assignment",
    "--merge-duplicate-topics", "--duplicate-similarity", "0.92",
    "--flag-junk-topics", "--junk-coherence-veto", "0.10", "--junk-coherence-min-size", "30",
    "--micro-clusters", "--tweets-per-topic", "-1", "--label-method", "ctfidf",
    "--reducer", "umap", "--umap-components", "5", "--umap-neighbors", "30",
    "--umap-min-dist", "0.0", "--umap-per-brand",
    "--cluster-selection-epsilon", "0.2", "--candidate-margin", "0.05",
    "--split-max-size", "2500", "--split-max-child-similarity", "0.92",
    "--alert-min-coherence", "0.05", "--alert-min-size", "250",
    "--residual-ack-ratio", "0.40", "--event-recall",
    "--recover-unassigned", "--umap-drift-threshold", "0.45",
    "--umap-reference-size", "50000",
    "--carry-forward-topics", "--topic-max-age-days", "45",
    "--embed-cache", str(CACHE),
]


def summarise(run_dir: Path, prev_dir: Path | None) -> dict:
    topics = json.loads((run_dir / "topics.json").read_text())
    meta = json.loads((run_dir / "run_metadata.json").read_text())
    summary = pd.read_csv(run_dir / "topic_hot_summary.csv")
    assign = pd.read_csv(run_dir / "topic_assignments.csv", low_memory=False)

    row = {
        "run": run_dir.name,
        "records": meta.get("records"),
        "topics": len(topics),
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
        "unassigned_share": round(
            float((assign["assignment_type"] == "UNASSIGNED").mean()), 4),
        "mean_coherence": round(float(pd.Series(
            [t["coherence"] for t in topics if t.get("coherence") is not None]).mean()), 4),
    }

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
    a = ap.parse_args()

    Path(a.out_prefix).parent.mkdir(parents=True, exist_ok=True)

    prev = Path(a.base_run)
    rows = [summarise(prev, None) | {"run": "batch0 (" + prev.name + ")"}]
    print(f"base registry: {rows[0]['topics']} topics\n", flush=True)

    for i, chunk in enumerate(a.chunks, start=1):
        out = Path(f"{a.out_prefix}_{i}")
        cmd = [str(VENV), str(SCRIPT), chunk, "--out", str(out),
               "--previous-topics", str(prev / "topics.json"),
               "--umap-model", a.model, "--buffer", a.buffer] + COMMON
        print(f"=== chunk {i}: {Path(chunk).name} -> {out.name} ===", flush=True)
        t0 = time.perf_counter()
        r = subprocess.run(cmd, capture_output=True, text=True)
        dt = time.perf_counter() - t0
        (Path(a.out_prefix).parent / f"stream_chunk{i}.log").write_text(r.stdout + r.stderr)
        if r.returncode != 0:
            print(f"  FAILED rc={r.returncode}\n{r.stderr[-2000:]}", flush=True)
            return 1
        row = summarise(out, prev)
        row["runtime_s"] = round(dt, 1)
        rows.append(row)
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
    return 0


if __name__ == "__main__":
    sys.exit(main())
