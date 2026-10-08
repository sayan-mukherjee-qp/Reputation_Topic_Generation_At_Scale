#!/usr/bin/env python3
"""How the dedupe steps behave across a stream, before vs after brand-only ID inheritance.

Per batch, for each run:
  stabilisation   MATCH_PREVIOUS / NEW_STABLE_ID / CARRIED_FORWARD / RETIRED_STALE, and
                  how many new IDs had a same-brand previous topic above the threshold
                  that another cluster took first (one-to-one losers)
  dedupe merges   duplicate_topic_merges.csv by reason, and how many of them merged an
                  ID that already existed in the previous batch into one that did not
                  (an established ID replaced by a just-minted one)
  survivors       same-brand pairs left after every dedupe step: live/live by centroid
                  band, identical labels, and dormant topics shadowing a live one
  brand           ID brand switches vs the previous batch; topics whose members are
                  mostly another brand

Usage: dedupe_audit.py NAME=RUN_DIR [NAME=RUN_DIR ...]
       (RUN_DIR holds base_200k/ or base_20k/ and stream_1..6/)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

UNCLEAR = "Unclear topic"
THRESHOLD = 0.55          # --candidate-similarity, the stabilisation threshold


def base_dir(run: Path) -> Path:
    for b in ("base_200k", "base_20k"):
        if (run / b).exists():
            return run / b
    raise SystemExit(f"no base run under {run}")


def merges(d: Path) -> pd.DataFrame:
    f = d / "duplicate_topic_merges.csv"
    if not f.exists():
        return pd.DataFrame(columns=["merged_topic_id", "into_topic_id", "reason"])
    m = pd.read_csv(f)
    if "reason" not in m.columns:
        m["reason"] = "centroid_similarity"
    m["reason"] = m["reason"].fillna("centroid_similarity")
    return m


def survivors(topics) -> dict:
    by = {}
    for t in topics:
        by.setdefault(t["brand"], []).append(t)
    o = {"live_pairs_ge_092": 0, "live_pairs_085_092": 0, "live_pairs_080_085": 0,
         "identical_label_pairs": 0, "dormant_shadowing_live": 0}
    for grp in by.values():
        if len(grp) < 2:
            continue
        C = np.asarray([t["centroid"] for t in grp], dtype=np.float32)
        C /= np.linalg.norm(C, axis=1, keepdims=True)
        S = C @ C.T
        for i in range(len(grp)):
            for j in range(i + 1, len(grp)):
                a, b, s = grp[i], grp[j], float(S[i, j])
                la, lb = bool(a.get("size")), bool(b.get("size"))
                if la and lb:
                    o["live_pairs_ge_092"] += s >= 0.92
                    o["live_pairs_085_092"] += 0.85 <= s < 0.92
                    o["live_pairs_080_085"] += 0.80 <= s < 0.85
                    o["identical_label_pairs"] += (a["label"] == b["label"] and a["label"] != UNCLEAR)
                elif la != lb:
                    o["dormant_shadowing_live"] += s >= 0.85
    return o


def mis_tagged(d: Path, topics) -> int:
    brand_of = {t["topic_id"]: t["brand"] for t in topics}
    a = pd.read_csv(d / "topic_assignments.csv", usecols=["record_id", "brand", "topic_id"],
                    low_memory=False).dropna(subset=["topic_id"]).drop_duplicates(["topic_id", "record_id"])
    dom = a.groupby("topic_id")["brand"].agg(lambda s: s.astype(str).value_counts().index[0])
    pooled = {"__SMALL_BRANDS__", "GLOBAL"}
    return int(sum(1 for tid, b in dom.items()
                   if tid in brand_of and brand_of[tid] not in pooled and brand_of[tid] != b))


def audit(name: str, run: Path) -> pd.DataFrame:
    rows = []
    prev_dir = base_dir(run)
    prev = json.loads((prev_dir / "topics.json").read_text())
    for i in range(1, 7):
        d = run / f"stream_{i}"
        if not (d / "topics.json").exists():
            continue
        cur = json.loads((d / "topics.json").read_text())
        meta = json.loads((d / "run_metadata.json").read_text())
        dec = pd.read_csv(d / "topic_stability_decisions.csv")
        vc = dec["decision"].value_counts()
        new = dec[dec["decision"] == "NEW_STABLE_ID"]
        prev_ids = {t["topic_id"] for t in prev}
        cur_ids = {t["topic_id"] for t in cur}
        m = merges(d)
        old_into_new = m[m["merged_topic_id"].isin(prev_ids) & ~m["into_topic_id"].isin(prev_ids)]
        pb = {t["topic_id"]: t["brand"] for t in prev}
        cb = {t["topic_id"]: t["brand"] for t in cur}
        row = {"run": name, "batch": i,
               "topics": len(cur), "live": sum(1 for t in cur if t.get("size")),
               "dormant": sum(1 for t in cur if t.get("status") == "DORMANT"),
               "match_previous": int(vc.get("MATCH_PREVIOUS", 0)),
               "new_stable_id": int(vc.get("NEW_STABLE_ID", 0)),
               "new_id_lost_one_to_one": int((new["best_available_similarity"] >= THRESHOLD).sum()),
               "carried_forward": int(vc.get("CARRIED_FORWARD", 0)),
               "retired_stale": int(vc.get("RETIRED_STALE", 0)),
               "retired_idle": (meta.get("consolidation") or {}).get("retired_idle"),
               "dedupe_centroid": int((m["reason"] == "centroid_similarity").sum()),
               "dedupe_label": int((m["reason"] == "identical_llm_label").sum()),
               "dedupe_capacity": int((m["reason"] == "capacity").sum()),
               "old_id_merged_into_new": len(old_into_new),
               "ids_lost_vs_prev": len(prev_ids - cur_ids),
               "brand_switches": sum(1 for k in pb.keys() & cb.keys() if pb[k] != cb[k]),
               "mis_tagged_topics": mis_tagged(d, cur),
               **survivors(cur)}
        rows.append(row)
        prev = cur
        print(f"  {name} batch {i} done", flush=True)
    return pd.DataFrame(rows)


def main() -> int:
    specs = [a.split("=", 1) for a in sys.argv[1:]]
    if not specs:
        print(__doc__)
        return 2
    df = pd.concat([audit(n, Path(p)) for n, p in specs], ignore_index=True)
    out = Path(__file__).resolve().parent / "results_dedupe"
    out.mkdir(exist_ok=True)
    df.to_csv(out / "dedupe_audit.csv", index=False)
    with pd.option_context("display.max_columns", None, "display.width", 250):
        print(df.set_index(["run", "batch"]).T.to_string())
        print("\nTotals over 6 batches:")
        print(df.groupby("run").sum(numeric_only=True).drop(columns="batch").T.to_string())
    return 0


if __name__ == "__main__":
    sys.exit(main())
