"""Read a run's output files into the shapes the dashboard charts.

Everything is derived from files the pipeline already writes; nothing here
recomputes a model result. Parsed files are cached on (path, mtime), so a
run finishing mid-session is picked up on the next request.
"""
from __future__ import annotations

import json
import re
import threading
from pathlib import Path
from typing import Any, Callable

import pandas as pd

_cache: dict[tuple[str, str], tuple[float, Any]] = {}
_cache_lock = threading.Lock()


def _cached(path: Path, kind: str, load: Callable[[Path], Any]) -> Any:
    mtime = path.stat().st_mtime
    key = (str(path), kind)
    with _cache_lock:
        hit = _cache.get(key)
        if hit and hit[0] == mtime:
            return hit[1]
    value = load(path)
    with _cache_lock:
        _cache[key] = (mtime, value)
    return value


def _records(df: pd.DataFrame) -> list[dict]:
    """JSON-safe rows: NaN -> null, numpy scalars -> Python."""
    return json.loads(df.to_json(orient="records", date_format="iso"))


# Stage groups for the timing charts. Twelve stages would need twelve hues;
# five groups tell the same story (embedding dominates) within the palette.
TIMING_GROUPS = {
    "Embedding": ["model load", "embedding", "embedding (cached)"],
    "UMAP": ["umap fit", "umap load"],
    "Clustering & assignment": ["layer 1 filter", "discovery clustering",
                                "history re-assignment", "incremental assignment",
                                "candidate discovery", "consolidation"],
    "Labelling & scoring": ["coherence scoring", "llm labelling"],
    "I/O & metrics": ["load + segment", "metrics + outputs"],
}

TOPIC_FIELDS = [
    "topic_id", "brand", "label", "description", "keywords", "size", "status_lifecycle",
    "status", "hot_score", "recent_volume", "baseline_volume", "growth_rate",
    "velocity_ratio", "anomaly_ratio", "persistence", "coherence", "created_at",
    "last_seen_at", "suppressed", "suppress_reason", "is_junk", "alert_suppressed",
    "alert_suppressed_reason", "label_source", "llm_confidence",
]


# ------------------------------------------------------------------ runs ---

def _run_order(name: str) -> tuple:
    if m := re.fullmatch(r"(.+)_(\d+)", name):
        if not name.startswith("base"):
            return (1, m[1], int(m[2]))
    return (0 if name.startswith("base") else 2, name, 0)


def run_dirs(out_dir: Path) -> list[Path]:
    if not out_dir.is_dir():
        return []
    dirs = [d for d in out_dir.iterdir()
            if d.is_dir() and (d / "run_metadata.json").exists()
            and (d / "topic_hot_summary.csv").exists()]
    return sorted(dirs, key=lambda d: _run_order(d.name))


def previous_run(out_dir: Path, name: str) -> Path | None:
    """The registry a run was matched against: stream_N <- stream_N-1 <- base."""
    names = [d.name for d in run_dirs(out_dir)]
    if name not in names or name.startswith("base"):
        return None
    m = re.fullmatch(r"(.+)_(\d+)", name)
    if m and int(m[2]) > 1 and f"{m[1]}_{int(m[2]) - 1}" in names:
        return out_dir / f"{m[1]}_{int(m[2]) - 1}"
    base = [n for n in names if n.startswith("base")]
    return out_dir / base[0] if base else None


def _summary(path: Path) -> pd.DataFrame:
    return _cached(path, "summary", lambda p: pd.read_csv(p, low_memory=False))


def _json(path: Path) -> dict:
    return _cached(path, "json", lambda p: json.loads(p.read_text()))


def _timing(run: Path) -> dict[str, float]:
    p = run / "stage_timing.json"
    return _json(p) if p.exists() else {}


def grouped_timing(timing: dict[str, float]) -> dict[str, float]:
    out = {g: round(sum(float(timing.get(s, 0.0)) for s in stages), 1)
           for g, stages in TIMING_GROUPS.items()}
    total = float(timing.get("TOTAL", 0.0))
    other = total - sum(out.values())
    if other > 0.5:                     # time between marks, e.g. interpreter start
        out["I/O & metrics"] = round(out["I/O & metrics"] + other, 1)
    return out


def _is_emerging(s: pd.Series) -> pd.Series:
    return s.astype(str).str.endswith("EMERGING")


def run_card(run: Path) -> dict:
    meta = _json(run / "run_metadata.json")
    df = _summary(run / "topic_hot_summary.csv")
    timing = _timing(run)
    prev = previous_run(run.parent, run.name)
    emerging = _is_emerging(df["status_lifecycle"])
    kept = ~df["suppressed"].fillna(False).astype(bool)
    new_emerging = int(emerging.sum())
    if prev is not None:
        prev_ids = set(_summary(prev / "topic_hot_summary.csv")["topic_id"])
        new_emerging = int((emerging & ~df["topic_id"].isin(prev_ids)).sum())
    window = meta.get("window") or {}
    return {
        "name": run.name,
        "kind": "base" if run.name.startswith("base") else "stream",
        "previous": prev.name if prev else None,
        "completed_at": meta.get("completed_at"),
        "records": meta.get("records"),
        "segments": meta.get("segments"),
        "holdout_records": window.get("holdout_records"),
        "window_start": window.get("start"),
        "window_end": window.get("end"),
        "input": Path(str(window.get("holdout_csv") or meta.get("input_csv") or "")).name,
        "device": meta.get("embedding_device"),
        "topics": int(len(df)),
        "live_topics": int((df["size"].fillna(0) > 0).sum()),
        "kept_topics": int((kept & (df["size"].fillna(0) > 0)).sum()),
        "emerging": int(emerging.sum()),
        "emerging_kept": int((emerging & kept).sum()),
        "new_emerging": new_emerging,
        "alerts": int(df["status"].isin(["HOT", "TRENDING"]).sum()),
        "drift": meta.get("umap_drift"),
        "mean_coherence": (round(float(df["coherence"].mean()), 4)
                           if df["coherence"].notna().any() else None),
        "total_seconds": timing.get("TOTAL"),
        "timing": {k: v for k, v in timing.items() if k != "TOTAL"},
        "timing_groups": grouped_timing(timing) if timing else None,
    }


def list_runs(out_dir: Path) -> list[dict]:
    cards = []
    for d in run_dirs(out_dir):
        try:
            cards.append(run_card(d))
        except (OSError, ValueError, KeyError) as exc:   # a run mid-write
            cards.append({"name": d.name, "error": str(exc)})
    return cards


# ---------------------------------------------------------- run overview ---

def _timeseries(run: Path) -> dict:
    p = run / "topic_timeseries.csv"
    if not p.exists():
        return {"dates": [], "counts": {}}

    def load(path: Path) -> dict:
        ts = pd.read_csv(path)
        wide = ts.pivot_table(index="date", columns="topic_id", values="record_count",
                              aggfunc="sum", fill_value=0).sort_index()
        return {"dates": [str(d) for d in wide.index],
                "counts": {c: wide[c].astype(int).tolist() for c in wide.columns}}
    return _cached(p, "timeseries", load)


def _csv_rows(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return _cached(path, "rows", lambda p: _records(pd.read_csv(p)))


def run_overview(run: Path) -> dict:
    card = run_card(run)
    df = _summary(run / "topic_hot_summary.csv")
    cols = [c for c in TOPIC_FIELDS if c in df.columns]
    topics = df[cols].copy()
    prev = previous_run(run.parent, run.name)
    if prev is not None:
        prev_ids = set(_summary(prev / "topic_hot_summary.csv")["topic_id"])
        topics["is_new"] = ~topics["topic_id"].isin(prev_ids)
    else:
        topics["is_new"] = _is_emerging(topics["status_lifecycle"])
    meta = _json(run / "run_metadata.json")
    llm = meta.get("llm") or {}
    return {
        "run": card,
        "meta": {
            "embedding_model": meta.get("embedding_model"),
            "embedding_device": meta.get("embedding_device"),
            "embedding_batch_size": meta.get("embedding_batch_size"),
            "peak_process_vram_mib": meta.get("peak_process_vram_mib"),
            "max_vram_gb": meta.get("max_vram_gb"),
            "reducer": meta.get("reducer"),
            "hdbscan": meta.get("hdbscan"),
            "thresholds": meta.get("thresholds"),
            "llm_model": llm.get("model") if llm.get("enabled") else None,
            "llm_calls": (llm.get("stats") or {}).get("calls"),
        },
        "topics": _records(topics),
        "series": _timeseries(run),
        "events": _csv_rows(run / "event_recall.csv"),
        "candidates": _csv_rows(run / "candidate_topic_decisions.csv"),
    }


def topic_samples(run: Path, topic_id: str, limit: int = 12) -> list[str]:
    """Example tweets for one topic, from topics.json (cached per run, trimmed:
    the file also holds every centroid and can run to tens of MB)."""
    p = run / "topics.json"
    if not p.exists():
        return []

    def load(path: Path) -> dict[str, list[str]]:
        return {str(t["topic_id"]): [str(x) for x in (t.get("tweets") or [])[:40]]
                for t in json.loads(path.read_text())}
    return _cached(p, "samples", load).get(topic_id, [])[:limit]


def stream_table(out_dir: Path) -> list[dict]:
    """stream_summary.csv from run_stream.py (written when the stream ends)."""
    return _csv_rows(out_dir / "stream_summary.csv")
