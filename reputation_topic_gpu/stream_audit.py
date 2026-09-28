#!/usr/bin/env python3
"""Audit a streaming replay the way the out300 reviews did, so runs compare.

Per batch: registry integrity (duplicate IDs, phantom slots, own-brand rate,
future last_seen_at), evidence per topic (median size, volume by size band),
independent coherence (C_npmi and C_v over c-TF-IDF top-10 terms, scored
against ONE fixed reference corpus so a bigger window cannot move the
yardstick), alert quality, LLM signals, and cross-batch label churn.

Usage:  stream_audit.py --ref data/stream300/chunk_*.csv NAME=DIR_PREFIX ...
        (DIR_PREFIX is e.g. out300/out_s300 -> out300/out_s300_1 .. _6)
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, ENGLISH_STOP_WORDS

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from reputation_topic_detection import ACK_WORDS, normalize_text  # noqa: E402

TOKEN_RE = r"(?u)\b[a-z][a-z']{2,}\b"
STOP = set(ENGLISH_STOP_WORDS) | ACK_WORDS | {"just", "amp", "don't", "i'm", "it's", "can't"}
PLACEHOLDER = re.compile(r"__\w+__|mobile_?care\w*", re.I)
BANDS = [(0, 100, "<=100"), (101, 300, "101-300"), (301, 1000, "301-1000"), (1001, 10**9, "1000+")]
POOLED = {"__SMALL_BRANDS__", "GLOBAL"}
TOP_N = 10
analyzer = CountVectorizer(token_pattern=TOKEN_RE, lowercase=True).build_analyzer()


def clean(t: str) -> str:
    return PLACEHOLDER.sub(" ", normalize_text(str(t)).lower())


class Reference:
    """Binary document-term matrix over a fixed corpus, built lazily per vocabulary."""

    def __init__(self, csvs=(), texts=None, name="reference"):
        if texts is None:
            texts = []
            for c in csvs:
                texts += pd.read_csv(c, usecols=["text"])["text"].fillna("").map(clean).tolist()
        else:
            texts = [clean(t) for t in texts]
        self.texts = texts
        self.cache = {}
        print(f"{name} corpus: {len(texts):,} documents", flush=True)

    def matrix(self, vocab):
        key = tuple(vocab)
        if key not in self.cache:
            cv = CountVectorizer(vocabulary=vocab, token_pattern=TOKEN_RE, binary=True)
            X = cv.fit_transform(self.texts).astype(np.float32).tocsc()
            co = (X.T @ X).toarray()          # vocab is small (a few thousand words)
            self.cache = {key: (co, X.shape[0])}
        return self.cache[key]


def ctfidf_top(member_texts: dict) -> dict:
    tids = list(member_texts)
    docs = [" ".join(clean(x) for x in member_texts[t]) for t in tids]
    cv = CountVectorizer(token_pattern=TOKEN_RE, stop_words=sorted(STOP), min_df=1)
    X = cv.fit_transform(docs).astype(np.float32)
    words = np.asarray(cv.get_feature_names_out())
    tf = X.multiply(1.0 / np.maximum(np.asarray(X.sum(axis=1)), 1)).tocsr()
    f = np.asarray(X.sum(axis=0)).ravel()
    avg = X.sum() / max(1, X.shape[0])
    idf = np.log(1 + avg / np.maximum(f, 1))
    out = {}
    for i, tid in enumerate(tids):
        row = tf.getrow(i)
        if row.nnz == 0:
            continue
        sc = row.toarray().ravel() * idf
        order = np.argsort(-sc)[:TOP_N]
        out[tid] = [words[j] for j in order if sc[j] > 0]
    return out


def coherence(top: dict, ref: Reference) -> pd.DataFrame:
    vocab = sorted({w for ws in top.values() for w in ws})
    co, n = ref.matrix(vocab)
    idx = {w: i for i, w in enumerate(vocab)}
    df_ = np.diag(co)
    eps = 1e-12
    rows = []
    for tid, ws in top.items():
        ids = [idx[w] for w in ws if df_[idx[w]] > 0]
        if len(ids) < 2:
            continue
        p = df_[ids] / n
        pab = co[np.ix_(ids, ids)] / n
        with np.errstate(divide="ignore", invalid="ignore"):
            # C_v uses epsilon-smoothed NPMI context vectors.
            sm = np.log((pab + eps) / np.outer(p, p)) / -np.log(pab + eps)
        # gensim-style epsilon smoothing, so a pair that never co-occurs scores
        # log(eps/(pa*pb))/-log(eps) instead of a flat -1.
        off = ~np.eye(len(ids), dtype=bool)
        c_npmi = float(sm[off].mean())
        np.fill_diagonal(sm, 1.0)
        tot = sm.sum(axis=0)
        cos = [float(sm[i] @ tot / (np.linalg.norm(sm[i]) * np.linalg.norm(tot) + eps))
               for i in range(len(ids))]
        rows.append({"topic_id": tid, "c_npmi": c_npmi, "c_v": float(np.mean(cos)),
                     "terms": " ".join(ws)})
    return pd.DataFrame(rows)


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


def audit_batch(d: Path, ref: Reference, wref: "Reference | None" = None) -> tuple[dict, pd.DataFrame]:
    topics = json.loads((d / "topics.json").read_text())
    meta = json.loads((d / "run_metadata.json").read_text())
    asg = pd.read_csv(d / "topic_assignments.csv", low_memory=False,
                      usecols=["record_id", "segment_id", "event_time", "brand", "topic_id",
                               "clean_text", "assignment_type"])
    ids = [t["topic_id"] for t in topics]
    first = {}
    for t in topics:
        first.setdefault(t["topic_id"], t)
    raw_slots = sum(t.get("size") or 0 for t in topics)
    true_slots = sum(t.get("size") or 0 for t in first.values())
    a = asg[asg["topic_id"].notna()]
    tb = a["topic_id"].map({k: v["brand"] for k, v in first.items()})
    scoped = ~tb.isin(POOLED)
    own = float((a.loc[scoped, "brand"].astype(str) == tb[scoped].astype(str)).mean())
    end = pd.to_datetime(asg["event_time"], errors="coerce", utc=True, format="mixed").max()
    ls = pd.to_datetime(pd.Series([t.get("last_seen_at") for t in first.values()]),
                        errors="coerce", utc=True, format="mixed")

    mem = a.dropna(subset=["clean_text"]).drop_duplicates(["topic_id", "record_id"])
    mt = {tid: g["clean_text"].astype(str).tolist() for tid, g in mem.groupby("topic_id")}
    top = ctfidf_top(mt)
    coh = coherence(top, ref)
    if wref is not None:
        coh = coh.merge(coherence(top, wref)[["topic_id", "c_npmi", "c_v"]]
                        .rename(columns={"c_npmi": "c_npmi_win", "c_v": "c_v_win"}),
                        on="topic_id", how="left")
    # Exact brand integrity: topics whose members are mostly another brand.
    dom = mem.groupby("topic_id")["brand"].agg(lambda x: x.astype(str).value_counts().index[0])
    mis_tagged = int(sum(1 for tid, b in dom.items()
                         if tid in first and str(first[tid]["brand"]) != b))
    hold = (meta.get("window") or {}).get("holdout_csv")
    _un = unassigned_rates(d, asg, hold)      # holdout None for single-chunk runs
    tdf = pd.DataFrame([{**{k: t.get(k) for k in (
        "topic_id", "brand", "label", "size", "status", "hot_status", "is_junk",
        "residual_cluster", "llm_substantive", "suppressed", "campaign_cluster",
        "label_source")}} for t in first.values()])
    tdf = tdf.merge(coh, on="topic_id", how="left")
    live = tdf[tdf["size"].fillna(0) > 0]
    vol = live["size"].sum()

    row = {
        "batch": d.name,
        "window_records": meta.get("records"),
        "entries": len(topics), "unique_ids": len(first),
        "duplicate_ids": len(ids) - len(first),
        "phantom_slots_pct": round(100 * (raw_slots - true_slots) / max(1, raw_slots), 1),
        "own_brand_pct": round(100 * own, 2),
        "mis_tagged_topics": mis_tagged,
        "last_seen_future": int((ls > end).sum()),
        "unassigned_window_pct": _un[0],
        "unassigned_holdout_pct": _un[1],
        "inflation": round(len(a) / max(1, a["segment_id"].nunique()), 2),
        "live_topics": len(live),
        "records_per_live_topic": round((meta.get("records") or 0) / max(1, len(live)), 1),
        "median_size": int(live["size"].median()) if len(live) else 0,
        "reference_corpus_docs": len(ref.texts),
        "window_corpus_docs": None if wref is None else len(wref.texts),
    }
    for lo, hi, name in BANDS:
        m = live["size"].between(lo, hi)
        row[f"vol_{name}_pct"] = round(100 * live.loc[m, "size"].sum() / max(1, vol), 1)
        row[f"npmi_{name}"] = round(float(live.loc[m, "c_npmi"].mean()), 3) if m.any() else None
    row["C_v"] = round(float(tdf["c_v"].mean()), 4)
    row["C_npmi"] = round(float(tdf["c_npmi"].mean()), 4)
    row["C_npmi_size_weighted"] = round(float(np.average(
        live["c_npmi"].fillna(0), weights=live["size"])) if len(live) else 0, 4)
    if wref is not None:
        row["C_v_window"] = round(float(tdf["c_v_win"].mean()), 4)
        row["C_npmi_window"] = round(float(tdf["c_npmi_win"].mean()), 4)

    alerts = tdf[tdf["hot_status"].isin(["HOT", "TRENDING"])]
    row["alerts"] = len(alerts)
    row["alerts_neg_npmi"] = int((alerts["c_npmi"] < 0).sum())
    row["alerts_not_substantive"] = int((alerts["llm_substantive"] == False).sum())  # noqa: E712

    def gap(mask):
        m = mask.map(lambda v: v is True or v is np.True_)
        if not m.any() or m.all():
            return None
        return round(float(tdf.loc[~m, "c_npmi"].mean() - tdf.loc[m, "c_npmi"].mean()), 3)
    row["gap_is_junk"] = gap(tdf["is_junk"])
    row["gap_not_substantive"] = gap(tdf["llm_substantive"].map(lambda v: v is False))
    row["gap_suppressed"] = gap(tdf["suppressed"]) if tdf["suppressed"].notna().any() else None
    if tdf["suppressed"].notna().any():
        kept = live[~live["suppressed"].map(lambda v: v is True)]
        row["kept_vol_pct"] = round(100 * kept["size"].sum() / max(1, vol), 1)
        row["kept_C_npmi"] = round(float(kept["c_npmi"].mean()), 4)
        row["kept_C_v"] = round(float(kept["c_v"].mean()), 4)
        if wref is not None:
            row["kept_C_npmi_window"] = round(float(kept["c_npmi_win"].mean()), 4)
    row["unclear_topics"] = int((tdf["label"] == "Unclear topic").sum())
    row["campaign_topics"] = int(tdf["campaign_cluster"].map(lambda v: v is True).sum())
    er = d / "event_recall.csv"
    if er.exists():
        e = pd.read_csv(er)
        f = e[e["found"]]
        row["events"] = f"{len(f)}/{len(e)}"
        row["recall"] = round(float(f["recall_in_best_topic"].mean()), 3)
        row["topics_for_80"] = round(float(f["topics_for_80pct"].mean()), 2)
    return row, tdf


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", nargs="+", required=True)
    ap.add_argument("--batches", type=int, default=6)
    ap.add_argument("--out", default=str(HERE / "out300w" / "stream_audit.csv"))
    ap.add_argument("--window-ref", default=None,
                    help="Run prefix whose batch-i segments.csv is ALSO used as a common corpus "
                         "for every run's batch i (the review's like-for-like control)")
    ap.add_argument("runs", nargs="+")
    a = ap.parse_args()
    ref = Reference(a.ref)
    specs = [s.split("=", 1) for s in a.runs]
    rows = []
    labels = {name: {} for name, _ in specs}
    for i in range(1, a.batches + 1):
        wref = None
        if a.window_ref:
            seg = Path(f"{a.window_ref}_{i}") / "segments.csv"
            if seg.exists():
                wref = Reference(texts=pd.read_csv(seg, usecols=["clean_text"])["clean_text"]
                                 .fillna("").astype(str).tolist(), name=f"window {i}")
        for name, prefix in specs:
            d = Path(f"{prefix}_{i}")
            if not d.exists():
                continue
            r, tdf = audit_batch(d, ref, wref)
            rows.append({"run": name, **r})
            tdf.to_csv(d / "audit_topics.csv", index=False)
            labels[name][i] = dict(zip(tdf["topic_id"], tdf["label"]))
            print(f"  {name} batch {i} done", flush=True)
    for name, _ in specs:
        labels_ = labels[name]
        # Label churn over topics present (and live-labelled) in every batch.
        if len(labels_) > 1:
            common = set.intersection(*(set(v) for v in labels_.values()))
            distinct = [len({labels_[i][t] for i in labels_}) for t in common]
            rows.append({"run": name, "batch": "CHURN", "entries": len(common),
                         "unique_ids": sum(1 for x in distinct if x > 1),
                         "duplicate_ids": sum(1 for x in distinct if x == len(labels_))})
    rows.sort(key=lambda r: [n for n, _ in specs].index(r["run"]))
    out = pd.DataFrame(rows)
    out.to_csv(a.out, index=False)
    with pd.option_context("display.max_columns", None, "display.width", 250):
        print(out.set_index(["run", "batch"]).T.to_string())
    print(f"\nwrote {a.out}  (CHURN row: entries=topics in all batches, "
          f"unique_ids=changed label at least once, duplicate_ids=distinct label every batch)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
