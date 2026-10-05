#!/usr/bin/env python3
"""C_NPMI topic coherence for an existing run.

Coherence asks whether a topic's top words actually co-occur in real documents.
It correlates with human judgement far better than silhouette, which only
measures geometric separation and happily rewards a tight cluster of "thanks".

NPMI(a,b) = log(P(a,b) / (P(a)P(b))) / -log(P(a,b)), averaged over all pairs of
a topic's top-N words. Range is [-1, 1]: >0.1 is reasonable, >0.2 is good,
<=0 means the words do not belong together.
"""
from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, ENGLISH_STOP_WORDS

OUT = Path(sys.argv[1] if len(sys.argv) > 1 else "output_150k")
TOP_N = 10

assign = pd.read_csv(OUT / "topic_assignments.csv", usecols=["topic_id", "clean_text"]).dropna()
summary = pd.read_csv(OUT / "topic_hot_summary.csv")
corpus = pd.read_csv(OUT / "segments.csv", usecols=["clean_text"]).clean_text.fillna("").astype(str)
print(f"reference corpus: {len(corpus):,} documents")
print(f"topics: {summary.topic_id.nunique()}")

# Top unigrams per topic, from the text actually assigned to it.
token_re = r"(?u)\b[a-z][a-z']{2,}\b"
cv_tok = CountVectorizer(token_pattern=token_re, lowercase=True).build_analyzer()
topic_words: dict[str, list[str]] = {}
for tid, g in assign.groupby("topic_id"):
    cnt = Counter()
    for t in g.clean_text.astype(str):
        cnt.update(w for w in set(cv_tok(t)) if w not in ENGLISH_STOP_WORDS)
    topic_words[tid] = [w for w, _ in cnt.most_common(TOP_N)]

vocab = sorted({w for ws in topic_words.values() for w in ws})
print(f"vocabulary under test: {len(vocab):,} words")

# Binary doc-term matrix over just that vocabulary.
cv = CountVectorizer(vocabulary=vocab, token_pattern=token_re, binary=True, lowercase=True)
X = cv.fit_transform(corpus).astype(bool)
N = X.shape[0]
df_cnt = np.asarray(X.sum(axis=0)).ravel()
idx = {w: i for i, w in enumerate(vocab)}
Xc = X.tocsc()

def npmi(a: str, b: str) -> float:
    ia, ib = idx[a], idx[b]
    na, nb = df_cnt[ia], df_cnt[ib]
    if na == 0 or nb == 0:
        return 0.0
    nab = int(Xc[:, ia].multiply(Xc[:, ib]).sum())
    if nab == 0:
        return -1.0
    pab, pa, pb = nab / N, na / N, nb / N
    return float(np.log(pab / (pa * pb)) / -np.log(pab))

rows = []
for tid, ws in topic_words.items():
    ws = [w for w in ws if w in idx]
    if len(ws) < 2:
        continue
    scores = [npmi(ws[i], ws[j]) for i in range(len(ws)) for j in range(i + 1, len(ws))]
    rows.append({"topic_id": tid, "coherence": float(np.mean(scores)), "n_words": len(ws),
                 "top_words": " ".join(ws)})

coh = pd.DataFrame(rows).merge(
    summary[["topic_id", "brand", "label", "size", "status_lifecycle"]], on="topic_id", how="left")
coh = coh.sort_values("coherence", ascending=False)
coh.to_csv(OUT / "topic_coherence.csv", index=False)

print(f"\n=== C_NPMI coherence over {len(coh)} topics ===")
print(f"  mean   {coh.coherence.mean():.4f}")
print(f"  median {coh.coherence.median():.4f}")
print(f"  std    {coh.coherence.std():.4f}")
print(f"  min    {coh.coherence.min():.4f}   max {coh.coherence.max():.4f}")
for lo, hi, name in [(-1, 0, "incoherent (<=0)"), (0, 0.1, "weak (0-0.1)"),
                     (0.1, 0.2, "reasonable (0.1-0.2)"), (0.2, 0.3, "good (0.2-0.3)"),
                     (0.3, 2, "strong (>0.3)")]:
    n = ((coh.coherence >= lo) & (coh.coherence < hi)).sum()
    print(f"  {name:<24} {n:>4}  {n/len(coh):6.1%}")

print("\n--- 10 most coherent ---")
print(coh.head(10)[["topic_id", "brand", "coherence", "label"]].to_string(index=False))
print("\n--- 10 least coherent ---")
print(coh.tail(10)[["topic_id", "brand", "coherence", "label"]].to_string(index=False))

PLEAS = {"thank", "thanks", "thankyou", "yes", "yep", "nope", "ok", "okay", "lol", "hi",
         "hello", "please", "sorry", "welcome", "yeah", "great", "good", "love", "dm", "dms"}
coh["pleasantry"] = coh.top_words.map(lambda s: sum(w in PLEAS for w in s.split()) >= 3)
print(f"\npleasantry-ish topics: {coh.pleasantry.sum()} of {len(coh)}")
print(f"  mean coherence pleasantry : {coh.loc[coh.pleasantry,'coherence'].mean():.4f}")
print(f"  mean coherence substantive: {coh.loc[~coh.pleasantry,'coherence'].mean():.4f}")
print(f"\nEMERGING topics mean coherence: {coh.loc[coh.status_lifecycle=='EMERGING','coherence'].mean():.4f}")
print(f"ACTIVE   topics mean coherence: {coh.loc[coh.status_lifecycle=='ACTIVE','coherence'].mean():.4f}")
print(f"\nwrote {OUT/'topic_coherence.csv'}")
