#!/usr/bin/env python3
"""Evaluate the P8 near-duplicate arms written by p8_test.sh.

Per arm, base run and six stream batches:
  * duplicates left: same-brand live pairs (excluding 'Unclear topic') with an
    identical label, the same label words, label embeddings >= 0.85, or >= 50%
    shared records with centroid >= 0.70
  * quality: stream_audit's scorer against fixed corpora (raw 20k slice /
    raw stream30 chunks)
  * stream stability: ID retention, plain and alias-aware; brand switches
  * merge correctness (base runs): every merge an arm made, judged by the LLM
    on the two topics AS THEY WERE BEFORE the merge (taken from the baseline
    arm, whose discovery and topic IDs are identical). The review arm is
    excluded from this check -- the same LLM would be grading itself.

Usage: compare_p8.py [--no-judge]
"""
from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
APP = ROOT / "reputation_topic_gpu"
sys.path.insert(0, str(APP))
from stream_audit import Reference, audit_batch  # noqa: E402
from reputation_topic_detection import label_wordset, label_sample  # noqa: E402

OUT = APP / "out/p8"
RES = ROOT / "experiments/results_p8"
DATA = ROOT / "data"
ARMS = ["baseline", "wordset", "semantic", "overlap", "review", "disamb", "groups", "aliases",
        "combo", "combo_cheap"]
MERGING = {"wordset", "semantic", "overlap", "aliases", "combo_cheap"}     # judged (review excluded: circular)
UNCLEAR = "Unclear topic"

_enc = None


def encoder():
    global _enc
    if _enc is None:
        from sentence_transformers import SentenceTransformer
        _enc = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
                                   device="cpu")
    return _enc


def load(run: Path):
    topics = json.loads((run / "topics.json").read_text())
    asg = pd.read_csv(run / "topic_assignments.csv", low_memory=False,
                      usecols=["record_id", "topic_id", "clean_text"])
    return topics, asg


def duplicates_left(topics, asg) -> dict:
    live = [t for t in topics if t.get("size") and t.get("label") != UNCLEAR]
    mem = {k: set(g["record_id"].astype(str)) for k, g in asg.dropna(subset=["topic_id"]).groupby("topic_id")}
    vec = encoder().encode([t["label"] for t in live], normalize_embeddings=True, show_progress_bar=False)
    out = {"dup_identical": 0, "dup_wordset": 0, "dup_label_085": 0, "dup_overlap": 0}
    by = {}
    for i, t in enumerate(live):
        by.setdefault(t["brand"], []).append(i)
    for idx in by.values():
        C = np.asarray([live[i]["centroid"] for i in idx], dtype=np.float32)
        C /= np.linalg.norm(C, axis=1, keepdims=True)
        for x in range(len(idx)):
            for y in range(x + 1, len(idx)):
                a, b = live[idx[x]], live[idx[y]]
                out["dup_identical"] += a["label"].strip().lower() == b["label"].strip().lower()
                out["dup_wordset"] += bool(label_wordset(a["label"])) and \
                    label_wordset(a["label"]) == label_wordset(b["label"])
                out["dup_label_085"] += float(vec[idx[x]] @ vec[idx[y]]) >= 0.85
                ma, mb = mem.get(a["topic_id"], set()), mem.get(b["topic_id"], set())
                c = len(ma & mb) / max(1, min(len(ma), len(mb)))
                out["dup_overlap"] += c >= 0.5 and float(C[x] @ C[y]) >= 0.70
    return out


def merges(run: Path) -> pd.DataFrame:
    f = run / "duplicate_topic_merges.csv"
    if not f.exists():
        return pd.DataFrame()
    m = pd.read_csv(f)
    return m[~m["reason"].isin(["centroid_similarity", "capacity"])]


def brand_switches(prev, cur) -> int:
    p = {t["topic_id"]: t["brand"] for t in prev}
    c = {t["topic_id"]: t["brand"] for t in cur}
    return sum(1 for k in p.keys() & c.keys() if p[k] != c[k])


def alias_retention(prev, cur) -> float:
    """Share of the previous batch's IDs that still resolve: as an ID or an alias."""
    resolvable = {t["topic_id"] for t in cur} | {a for t in cur for a in (t.get("aliases") or [])}
    ids = [t["topic_id"] for t in prev]
    return round(sum(i in resolvable for i in ids) / max(1, len(ids)), 4)


def judge_merges(arm_run: Path, base_topics, base_asg) -> pd.DataFrame:
    """LLM verdict on each merge, using the pre-merge topics from the baseline arm."""
    from airouter_v1_client import AiRouterV1Client, MERGE_REVIEW_PROMPT
    m = merges(arm_run)
    if m.empty:
        return m
    by_id = {t["topic_id"]: t for t in base_topics}
    texts = {k: g["clean_text"].dropna().astype(str).tolist()
             for k, g in base_asg.dropna(subset=["topic_id"]).groupby("topic_id")}
    mem = {k: set(g["record_id"].astype(str)) for k, g in base_asg.dropna(subset=["topic_id"]).groupby("topic_id")}

    def card(tid):
        t = by_id[tid]
        return {"label": t.get("label"), "description": t.get("description"),
                "terms": (t.get("keywords") or [])[:10], "member_count": t.get("size"),
                "samples": [s[:280] for s in label_sample(texts.get(tid, []), 10, tid)]}
    rows, payloads = [], []
    for _, r in m.iterrows():
        a, b = r["merged_topic_id"], r["into_topic_id"]
        if a not in by_id or b not in by_id:
            rows.append({**r.to_dict(), "judge": "missing"})
            payloads.append(None)
            continue
        ma, mb = mem.get(a, set()), mem.get(b, set())
        payloads.append({"brand": by_id[a]["brand"], "topic_a": card(a), "topic_b": card(b),
                         "shared_member_share": round(len(ma & mb) / max(1, min(len(ma), len(mb))), 3),
                         "centroid_similarity": None})
        rows.append(r.to_dict())
    client = AiRouterV1Client()
    todo = [p for p in payloads if p is not None]
    ans = iter(client.ask_json_many(MERGE_REVIEW_PROMPT, todo, workers=8))
    for row, p in zip(rows, payloads):
        if p is None:
            continue
        a = next(ans) or {}
        row["judge"] = str(a.get("decision", "error")).lower()
        row["judge_confidence"] = a.get("confidence")
        row["judge_reason"] = a.get("reason")
    return pd.DataFrame(rows)


def main() -> int:
    judge = "--no-judge" not in sys.argv
    RES.mkdir(parents=True, exist_ok=True)
    ref = Reference([DATA / "twcs_subset_20k.csv"], name="fixed 20k")
    sref = Reference(sorted(DATA.glob("stream30/chunk_*.csv")), name="fixed stream30")
    base_topics, base_asg = load(OUT / "baseline/base_20k")
    rows, judged = [], []
    for arm in ARMS:
        d = OUT / arm
        if not (d / "base_20k/topics.json").exists():
            print(f"  {arm}: not run")
            continue
        prev = None
        for name, run, rf in [("base", d / "base_20k", ref)] + \
                [(f"s{i}", d / f"stream_{i}", sref) for i in range(1, 7)]:
            if not (run / "topics.json").exists():
                continue
            r, tdf = audit_batch(run, rf)
            topics, asg = load(run)
            meta = json.loads((run / "run_metadata.json").read_text())
            llm = meta.get("llm") or {}
            st, p8 = llm.get("stats") or {}, llm.get("p8_stats") or {}
            mg = merges(run)
            star = int(mg.groupby("into_topic_id").size().max()) if len(mg) else 0
            row = {"arm": arm, **r, "batch": name, **duplicates_left(topics, asg),
                   "p8_merges": len(mg), "max_absorbed_by_one": star,
                   "llm_tokens": (st.get("prompt_tokens", 0) + st.get("completion_tokens", 0)
                                  + p8.get("prompt_tokens", 0) + p8.get("completion_tokens", 0)),
                   "llm_calls": (llm.get("requested") or 0) + (p8.get("calls") or 0),
                   "t_total": json.loads((run / "stage_timing.json").read_text()).get("TOTAL")}
            if (run / "topic_groups.csv").exists():
                g = pd.read_csv(run / "topic_groups.csv")
                row["groups"] = len(g)
                row["groups_with_unclear"] = int(g["labels"].str.contains(UNCLEAR).sum())
                row["groups_no_shared_word"] = int(sum(
                    1 for labs in g["labels"]
                    if not set.intersection(*[set(label_wordset(x)) for x in labs.split(" | ")])))
            if prev is not None:
                row["id_retention"] = round(len({t["topic_id"] for t in prev} & {t["topic_id"] for t in topics})
                                            / max(1, len(prev)), 4)
                row["id_retention_alias"] = alias_retention(prev, topics)
                row["brand_switches"] = brand_switches(prev, topics)
            if name == "base":
                ids = {t["topic_id"]: t for t in topics}
                row["T8_T11"] = ("merged" if ("T8" in ids) != ("T11" in ids) or
                                 ("T11" in (ids.get("T8", {}).get("aliases") or []))
                                 else f"separate: {ids.get('T8', {}).get('label')} / {ids.get('T11', {}).get('label')}")
                if judge and arm in MERGING:
                    j = judge_merges(run, base_topics, base_asg)
                    if len(j):
                        j.insert(0, "arm", arm)
                        judged.append(j)
                        row["judged_merges"] = int((j["judge"].isin(["same", "different"])).sum())
                        row["judged_same"] = int((j["judge"] == "same").sum())
            rows.append(row)
            prev = topics
            print(f"  {arm} {name} done", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(RES / "p8_audit.csv", index=False)
    if judged:
        pd.concat(judged).to_csv(RES / "p8_merge_judgements.csv", index=False)
    keep = ["live_topics", "median_size", "dup_identical", "dup_wordset", "dup_label_085", "dup_overlap",
            "p8_merges", "max_absorbed_by_one", "judged_merges", "judged_same",
            "kept_C_npmi", "kept_C_v", "C_npmi", "inflation", "unassigned_holdout_pct",
            "events", "recall", "topics_for_80", "unclear_topics", "alerts",
            "id_retention", "id_retention_alias", "brand_switches", "groups", "groups_with_unclear",
            "groups_no_shared_word", "llm_calls", "llm_tokens", "t_total", "T8_T11"]
    with pd.option_context("display.max_rows", None, "display.max_columns", None, "display.width", 250):
        b = df[df.batch == "base"].set_index("arm")
        print("\n=== BASE 20k ===")
        print(b[[c for c in keep if c in b.columns]].T.to_string())
        s = df[df.batch != "base"]
        agg = s.groupby("arm").agg(**{c: (c, "mean") for c in keep
                                      if c in s.columns and s[c].dtype.kind in "fi"})
        print("\n=== STREAM, mean of 6 batches ===")
        print(agg.reindex([a for a in ARMS if a in agg.index]).T.round(4).to_string())
    print(f"\nwrote {RES / 'p8_audit.csv'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
