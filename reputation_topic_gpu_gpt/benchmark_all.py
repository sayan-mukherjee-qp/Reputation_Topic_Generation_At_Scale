#!/usr/bin/env python3
"""One consistent benchmark across every architecture we have run.

Every number here is recomputed from each run's own outputs with the same code,
rather than quoted from the document that introduced it. That matters: earlier
reports measured coherence in at least two different ways, so their figures are
not comparable to each other. These are.

Coherence is computed from the text actually assigned to a topic, not from the
shipped label, so a run cannot score well by labelling well.

  C_NPMI  mean pairwise NPMI over a topic's top-N words. [-1,1]; >0.1 reasonable.
  C_v     the usual sliding-window measure, with the window set to the whole
          document. That is not a shortcut here -- these are tweets, almost all
          far shorter than the standard 110-token window, so the two coincide.
  diversity  share of top-N word slots filled by distinct words across topics;
          low means every topic is described by the same handful of words.

Usage:  benchmark_all.py NAME=DIR [NAME=DIR ...] [--top-n 10] [--out FILE]
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, ENGLISH_STOP_WORDS

TOKEN_RE = r"(?u)\b[a-z][a-z']{2,}\b"

sys.path.insert(0, str(Path(__file__).resolve().parent))
from reputation_topic_detection import VERIFIED_EVENTS, event_keyword_hit  # noqa: E402


def topic_top_words(assign: pd.DataFrame, top_n: int) -> dict:
    analyzer = CountVectorizer(token_pattern=TOKEN_RE, lowercase=True).build_analyzer()
    out = {}
    for tid, g in assign.groupby("topic_id"):
        cnt = Counter()
        for t in g["clean_text"].astype(str):
            cnt.update(w for w in set(analyzer(t)) if w not in ENGLISH_STOP_WORDS)
        words = [w for w, _ in cnt.most_common(top_n)]
        if len(words) >= 2:
            out[tid] = words
    return out


def coherence_scores(topic_words: dict, corpus: pd.Series, top_n: int):
    """Return (per-topic C_NPMI, per-topic C_v, diversity)."""
    vocab = sorted({w for ws in topic_words.values() for w in ws})
    if not vocab:
        return {}, {}, float("nan")
    cv = CountVectorizer(vocabulary=vocab, token_pattern=TOKEN_RE, binary=True, lowercase=True)
    X = cv.fit_transform(corpus.astype(str)).astype(np.float32)
    n_docs = X.shape[0]
    idx = {w: i for i, w in enumerate(vocab)}

    df = np.asarray(X.sum(axis=0)).ravel()          # docs containing each word
    p1 = np.maximum(df, 1e-12) / n_docs
    co = (X.T @ X).toarray()                        # pairwise co-occurrence counts
    eps = 1e-12
    p2 = np.maximum(co, 0) / n_docs

    with np.errstate(divide="ignore", invalid="ignore"):
        pmi = np.log((p2 + eps) / (np.outer(p1, p1) + eps))
        npmi = pmi / (-np.log(p2 + eps))
    npmi = np.nan_to_num(npmi, nan=0.0, posinf=0.0, neginf=-1.0)

    npmi_out, cv_out = {}, {}
    for tid, words in topic_words.items():
        ids = [idx[w] for w in words if w in idx]
        if len(ids) < 2:
            continue
        sub = npmi[np.ix_(ids, ids)]
        iu = np.triu_indices(len(ids), k=1)
        npmi_out[tid] = float(sub[iu].mean())
        # C_v: cosine between each word's NPMI context vector and the topic sum.
        total = sub.sum(axis=0)
        sims = []
        for k in range(len(ids)):
            a, b = sub[k], total
            d = np.linalg.norm(a) * np.linalg.norm(b)
            sims.append(float(a @ b / d) if d > 0 else 0.0)
        cv_out[tid] = float(np.mean(sims))

    slots = sum(len(w) for w in topic_words.values())
    diversity = len({w for ws in topic_words.values() for w in ws}) / max(1, slots)
    return npmi_out, cv_out, diversity


def near_duplicates(topics: list, threshold: float = 0.90) -> int:
    by = defaultdict(list)
    for t in topics:
        by[t.get("brand")].append(t)
    pairs = 0
    for _, ts in by.items():
        if len(ts) < 2:
            continue
        C = np.asarray([t["centroid"] for t in ts], dtype=np.float32)
        S = C @ C.T
        pairs += int((np.triu(S, 1) > threshold).sum())
    return pairs


def evaluate(name: str, d: Path, top_n: int) -> dict:
    topics = json.loads((d / "topics.json").read_text())
    assign = pd.read_csv(d / "topic_assignments.csv", low_memory=True,
                         usecols=lambda c: c in {"topic_id", "clean_text", "record_id",
                                                 "brand", "assignment_type", "segment_id"})
    corpus = pd.read_csv(d / "segments.csv", usecols=["clean_text"])["clean_text"].fillna("")
    a = assign[assign["topic_id"].notna()]

    tw = topic_top_words(a, top_n)
    npmi, cvs, diversity = coherence_scores(tw, corpus, top_n)
    nv = np.array(list(npmi.values())) if npmi else np.array([np.nan])
    cvv = np.array(list(cvs.values())) if cvs else np.array([np.nan])
    sizes = np.array([t["size"] for t in topics], dtype=float)

    row = {
        "architecture": name,
        "corpus_docs": len(corpus),
        "topics": len(topics),
        "median_size": int(np.median(sizes)) if sizes.size else 0,
        "largest": int(sizes.max()) if sizes.size else 0,
        "C_NPMI_mean": round(float(np.nanmean(nv)), 4),
        "C_NPMI_median": round(float(np.nanmedian(nv)), 4),
        "pct_coherent_0.1": round(float(np.nanmean(nv >= 0.1)), 4),
        "pct_negative": round(float(np.nanmean(nv < 0)), 4),
        "C_v_mean": round(float(np.nanmean(cvv)), 4),
        "diversity": round(float(diversity), 4),
        "near_dup_pairs": near_duplicates(topics),
        "junk_flagged": sum(1 for t in topics if t.get("is_junk")),
        "unassigned_share": round(float((assign["assignment_type"] == "UNASSIGNED").mean()), 4),
        "size_inflation": round(float(a.groupby("topic_id")["record_id"].nunique().sum()
                                      / max(1, a["record_id"].nunique())), 3),
    }

    # Event metrics are recomputed here rather than read from each run's
    # event_recall.csv, because that file's columns changed over time and the
    # older runs predate it entirely. Same code for every architecture.
    sizes_by_id = {t["topic_id"]: t["size"] for t in topics}
    found = rec = prec = t80 = 0
    n_found = 0
    for ev in VERIFIED_EVENTS:
        m = a["clean_text"].astype(str).map(lambda x: event_keyword_hit(x, ev["all_of"]))
        if ev.get("brand"):
            m &= a["brand"].astype(str).eq(ev["brand"])
        hit = a[m]
        if hit.empty:
            continue
        per = hit.groupby("topic_id")["record_id"].nunique().sort_values(ascending=False)
        total = int(hit["record_id"].nunique())
        best, best_n = per.index[0], int(per.iloc[0])
        cum = per.cumsum() / total
        n_found += 1
        rec += best_n / total
        prec += best_n / max(1, sizes_by_id.get(best, 0))
        t80 += int((cum < 0.8).sum() + 1)
    if n_found:
        row["events_found"] = f"{n_found}/{len(VERIFIED_EVENTS)}"
        row["event_recall"] = round(rec / n_found, 4)
        row["event_precision"] = round(prec / n_found, 4)
        row["topics_for_80"] = round(t80 / n_found, 2)
    hs = d / "topic_hot_summary.csv"
    if hs.exists():
        s = pd.read_csv(hs)
        row["alerts"] = int(s["status"].isin(["HOT", "TRENDING"]).sum())
        al = s[s["status"].isin(["HOT", "TRENDING"])]
        if len(al) and "coherence" in al:
            row["worst_alert_coherence"] = round(float(al["coherence"].min()), 4)
    return row


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--top-n", type=int, default=10)
    ap.add_argument("--out", default="benchmark_all.csv")
    a = ap.parse_args()

    rows = []
    for spec in a.runs:
        name, _, path = spec.partition("=")
        d = Path(path)
        if not (d / "topics.json").exists():
            print(f"  skip {name}: no topics.json", file=sys.stderr)
            continue
        print(f"  scoring {name} ...", flush=True)
        rows.append(evaluate(name, d, a.top_n))

    df = pd.DataFrame(rows)
    df.to_csv(a.out, index=False)
    pd.set_option("display.width", 250)
    print("\n" + df.to_string(index=False))
    print(f"\nWrote {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
