#!/usr/bin/env python3
"""Compare topic-discovery runs on quality, fragmentation and stability.

The original PCA-vs-UMAP benchmark showed that every internal metric
(silhouette, topic count, noise ratio, precision-in-topic) rewards cutting
topics smaller, so a fragmenting reducer sweeps all of them and is still the
wrong choice. This therefore reports the internal metrics *and* the event-level
coverage metrics that exposed that, side by side.

Usage:  compare_runs.py NAME=DIR [NAME=DIR ...]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from event_recall import EVENTS, hits  # noqa: E402


def load_run(d: Path) -> dict:
    topics = json.loads((d / "topics.json").read_text())
    assign = pd.read_csv(d / "topic_assignments.csv", low_memory=False)
    summary = pd.read_csv(d / "topic_hot_summary.csv")
    meta_p = d / "run_metadata.json"
    meta = json.loads(meta_p.read_text()) if meta_p.exists() else {}
    return {"dir": d, "topics": topics, "assign": assign, "summary": summary, "meta": meta}


def quality_row(run: dict) -> dict:
    topics, assign = run["topics"], run["assign"]
    coh = np.array([t["coherence"] for t in topics if t.get("coherence") is not None], dtype=float)
    sizes = np.array([t["size"] for t in topics], dtype=float)
    junk = sum(1 for t in topics if t.get("is_junk"))
    # Share of segments no topic claimed. High is healthy here -- most customer
    # tweets are genuinely one-off -- so a big drop signals over-grouping.
    at = assign["assignment_type"].value_counts()
    unassigned = int(at.get("UNASSIGNED", 0))
    return {
        "Topics": len(topics),
        "Flagged junk": junk,
        "Coherent (>=0.1)": int((coh >= 0.1).sum()) if coh.size else 0,
        "Mean NPMI": round(float(coh.mean()), 4) if coh.size else None,
        "Median NPMI": round(float(np.median(coh)), 4) if coh.size else None,
        "Median topic size": int(np.median(sizes)) if sizes.size else 0,
        "Largest topic": int(sizes.max()) if sizes.size else 0,
        "Segments unassigned": unassigned,
        "Unassigned share": round(unassigned / max(1, len(assign)), 3),
    }


def event_rows(run: dict) -> pd.DataFrame:
    """Per event: precision, coverage, and how many topics you must read."""
    assign = run["assign"]
    topics_by_id = {t["topic_id"]: t for t in run["topics"]}
    out = []
    for ev in EVENTS:
        m = assign["clean_text"].astype(str).map(lambda t: hits(t, ev["all_of"]))
        if ev.get("brand"):
            m &= assign["brand"].astype(str).eq(ev["brand"])
        rows = assign[m]
        if rows.empty:
            out.append({"event": ev["name"], "found": False, "matched": 0,
                        "coverage": np.nan, "precision": np.nan,
                        "topics_for_80": np.nan, "spread": 0})
            continue
        per_topic = rows.groupby("topic_id")["record_id"].nunique().sort_values(ascending=False)
        total = rows["record_id"].nunique()
        best = per_topic.index[0]
        best_n = int(per_topic.iloc[0])
        size = topics_by_id.get(best, {}).get("size", 0) or 0
        # Topics needed before 80% of the event's records are visible.
        cum = per_topic.cumsum() / total
        n80 = int((cum < 0.8).sum() + 1)
        out.append({
            "event": ev["name"],
            "found": True,
            "matched": total,
            "coverage": best_n / total,
            "precision": (best_n / size) if size else np.nan,
            "topics_for_80": n80,
            "spread": int(len(per_topic)),
        })
    return pd.DataFrame(out)


def frag_row(ev_df: pd.DataFrame) -> dict:
    f = ev_df[ev_df["found"]]
    return {
        "Events found": f"{int(f.shape[0])}/{len(ev_df)}",
        "Precision-in-topic": round(float(f["precision"].mean()), 3),
        "Coverage of best topic": round(float(f["coverage"].mean()), 3),
        "Topics to see 80%": round(float(f["topics_for_80"].mean()), 1),
        "Topics event spread over": int(f["spread"].mean()),
    }


def main() -> int:
    if len(sys.argv) < 2:
        print(__doc__)
        return 2
    runs = {}
    for arg in sys.argv[1:]:
        name, _, path = arg.partition("=")
        runs[name] = load_run(Path(path))

    qual = pd.DataFrame({n: quality_row(r) for n, r in runs.items()})
    evs = {n: event_rows(r) for n, r in runs.items()}
    frag = pd.DataFrame({n: frag_row(e) for n, e in evs.items()})

    print("\n=== Topic quality ===")
    print(qual.to_string())
    print("\n=== Event-level usefulness (the metrics that settle fragmentation) ===")
    print(frag.to_string())

    print("\n=== Per-event coverage ===")
    cov = pd.DataFrame({n: e.set_index("event")["coverage"] for n, e in evs.items()})
    print(cov.round(3).to_string())
    print("\n=== Per-event topics needed to see 80% ===")
    t80 = pd.DataFrame({n: e.set_index("event")["topics_for_80"] for n, e in evs.items()})
    print(t80.to_string())

    out = Path("comparison_tables.json")
    out.write_text(json.dumps({
        "quality": qual.to_dict(),
        "fragmentation": frag.to_dict(),
        "coverage": cov.to_dict(),
        "topics_for_80": t80.to_dict(),
    }, indent=2, default=str))
    print(f"\nWrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
