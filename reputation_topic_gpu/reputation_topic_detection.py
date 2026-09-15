#!/usr/bin/env python3
"""
Reputation Topic Intelligence prototype
--------------------------------------

Tests a practical architecture on review/social data:

1. Normalize raw text.
2. Segment very long records (sentence-aware with a max-character fallback).
3. Embed semantic units with Sentence Transformers.
4. Discover topics with HDBSCAN (via scikit-learn).
5. Build stable topic centroids/labels.
6. Simulate incremental assignment on a time-based holdout.
7. Cluster unassigned/recent candidate segments to discover new topics.
8. Preserve record-level counting (one record counts at most once per topic).
9. Build daily time series and compute growth/velocity/anomaly/hot score.
10. Save CSV/JSON outputs and plots.

This is an experiment, not a production implementation.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import hashlib
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering, HDBSCAN
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import silhouette_score
from sentence_transformers import SentenceTransformer

try:
    import matplotlib.pyplot as plt
except Exception:
    plt = None


URL_RE = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
MENTION_RE = re.compile(r"(?<!\w)@\w+")
WHITESPACE_RE = re.compile(r"\s+")
MULTI_PUNCT_RE = re.compile(r"([!?.,])\1{2,}")

DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Acknowledgement anchors for Layer 1. Only English is listed on purpose: the
# multilingual encoder places "thanks", "gracias" and "ありがとう" in the same
# neighbourhood, so English anchors transfer to every language in the corpus
# without enumerating any of their vocabulary.
ACK_ANCHORS = [
    "thanks so much", "thank you very much", "ok got it", "yes please", "no thanks",
    "sure thing", "hello there", "sorry about that", "you are welcome",
    "i sent you a dm", "great thanks", "ok thanks", "yeah okay", "thank you",
]

# Layer 2/3 vocabulary: tokens that carry no reputation signal on their own.
ACK_WORDS = {
    "thanks", "thank", "thankyou", "ty", "thx", "yes", "yeah", "yep", "yup", "ok", "okay",
    "sure", "no", "nope", "nah", "hi", "hello", "hey", "please", "pls", "sorry", "welcome",
    "lol", "haha", "hahah", "great", "good", "nice", "cool", "love", "done", "sent", "did",
    "got", "dm", "dms", "dmed", "message", "gracias", "merci", "danke", "obrigado",
    "obrigada", "grazie", "si", "oui", "ja", "nein", "non", "nao", "n\u00e3o", "bien",
    "vielen", "mto", "cheers", "ta", "np",
}

# P8. Tokens that survive normalisation but carry no topic meaning. They are
# artefacts of anonymisation and of brand handles, and without this list they
# head real topics: "mobile_carexi / internet / modem" was five comcastcares
# topics, "__email__" another five.
LABEL_NOISE_WORDS = {
    "__email__", "__url__", "__phone__", "__user__", "__number__", "__date__",
    "mobile_care", "mobile_carexi", "mobilecare", "amp", "http", "https",
    "co", "rt", "via", "pic", "twitter", "com",
}


# The anonymiser emits a stable prefix with an arbitrary suffix --
# mobile_carexi, mobile_carexv, ... -- so an exact word list can never catch
# them all. CountVectorizer's stop_words only matches whole tokens, so these
# are filtered at term-selection instead.
LABEL_NOISE_RE = re.compile(r"^(?:mobile_?care\w*|__\w+__|\d+)$", re.IGNORECASE)


def is_noise_label_term(term: str) -> bool:
    """True when every word of a candidate label phrase is anonymiser noise."""
    words = str(term).split()
    return bool(words) and all(
        LABEL_NOISE_RE.match(w) or w in LABEL_NOISE_WORDS for w in words
    )


def label_stopwords() -> List[str]:
    """Everything suppressed when scoring candidate label terms."""
    try:
        from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
        base = set(ENGLISH_STOP_WORDS)
    except Exception:
        base = set()
    return sorted(base | ACK_WORDS | LABEL_NOISE_WORDS)


WORD_RE = re.compile(r"[^\W\d_]+", re.UNICODE)


@dataclass
class Topic:
    topic_id: str
    brand: str
    label: str
    size: int
    centroid: List[float]
    created_at: Optional[str] = None
    last_seen_at: Optional[str] = None
    status: str = "ACTIVE"
    hot_score: Optional[float] = None
    hot_status: Optional[str] = None
    pca_components: Optional[int] = None
    coherence: Optional[float] = None
    is_junk: Optional[bool] = None
    junk_reason: Optional[str] = None
    residual_cluster: Optional[bool] = None
    residual_reason: Optional[str] = None
    ack_member_ratio: Optional[float] = None
    tweets: Optional[List[str]] = None


@dataclass
class Config:
    model_name: str = DEFAULT_MODEL
    min_cluster_size: int = 20
    min_samples: int = 8
    cluster_selection_method: str = "eom"
    min_topic_similarity: float = 0.68
    new_topic_similarity: float = 0.55
    max_segment_chars: int = 450
    segment_overlap_chars: int = 60
    max_segments_per_record: int = 6
    pca_components: int = 50
    random_state: int = 42
    daily_recent_days: int = 7
    daily_baseline_days: int = 28


class StageTimer:
    """Records wall-clock per pipeline stage so a run can be sized from a smaller one."""

    def __init__(self) -> None:
        self.t0 = time.perf_counter()
        self.marks: List[Tuple[str, float]] = []
        self._last = self.t0

    def mark(self, name: str, detail: str = "") -> None:
        now = time.perf_counter()
        dt = now - self._last
        self._last = now
        self.marks.append((name, dt))
        print(f"[timing] {name:<26} {dt:8.1f}s   {detail}")

    def report(self) -> None:
        total = time.perf_counter() - self.t0
        print("\n=== Stage timing ===")
        for name, dt in self.marks:
            print(f"  {name:<26} {dt:8.1f}s  {dt/total:6.1%}")
        print(f"  {'TOTAL':<26} {total:8.1f}s  ({total/60:.1f} min)")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Test topic discovery + hot topic architecture on a CSV.")
    p.add_argument("csv", help="Path to CSV file")
    p.add_argument("--out", default="topic_test_output", help="Output directory")
    p.add_argument("--text-col", default="text")
    p.add_argument("--time-col", default="event_time")
    p.add_argument("--brand-col", default="brand")
    p.add_argument("--id-col", default="tweet_id")
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--min-cluster-size", type=int, default=20)
    p.add_argument("--min-samples", type=int, default=8)
    p.add_argument("--min-similarity", type=float, default=0.68)
    p.add_argument("--candidate-similarity", type=float, default=0.55)
    p.add_argument("--max-segment-chars", type=int, default=450)
    p.add_argument("--max-segments-per-record", type=int, default=6)
    p.add_argument("--test-fraction", type=float, default=0.20,
                    help="Most-recent fraction used as incremental holdout (0 disables holdout simulation)")
    p.add_argument("--brand-min-records", type=int, default=80,
                    help="Cluster per brand only when enough records exist; otherwise use one global pool")
    p.add_argument("--tune-pca", action="store_true",
                    help="Choose the PCA width per pool instead of using --pca-components, and write the "
                         "chosen widths to pca_config.json in the output directory")
    p.add_argument("--pca-candidates", default="50,100,128",
                    help="Comma-separated PCA widths considered by --tune-pca")
    p.add_argument("--pca-config", default=None,
                    help="JSON of {pool: width} from a previous --tune-pca run. Frozen widths keep results "
                         "reproducible between runs; re-tune on a schedule, not every run.")
    p.add_argument("--max-cluster-share", type=float, default=0.30,
                    help="Reject a clustering where one cluster holds more than this share of the pool; "
                         "that is a catch-all blob, not a topic")
    p.add_argument("--tweets-per-topic", type=int, default=10,
                    help="Texts stored on each topic in topics.json (0 disables, -1 stores every member)")
    p.add_argument("--embed-cache", default=None,
                    help="Directory for cached embeddings, keyed by model + exact text. Development "
                         "convenience only: it changes no output, it just skips re-encoding "
                         "identical input")
    p.add_argument("--brand-scoped-assignment", action="store_true",
                    help="Assign a record only to topics belonging to its own brand (plus the pooled "
                         "buckets). Without this, 54.7%% of assignments land on another brand's topic")
    p.add_argument("--label-method", default="tfidf", choices=["tfidf", "ctfidf", "llm"],
                    help="Topic labelling. 'tfidf' scores terms within a topic only, so common words "
                         "win. 'ctfidf' scores across topics, surfacing distinctive terms. 'llm' "
                         "sends c-TF-IDF terms plus sample tweets to AiRouterV2 for a written label")
    p.add_argument("--label-llm-usecase", default="reputation-topic-label",
                    help="AiRouterV2 use-case name. Its prompt, model and output schema live on the "
                         "use-case in the AiRouterV2 console, not in this client")
    p.add_argument("--label-llm-samples", type=int, default=8,
                    help="Sample tweets sent per topic alongside its terms")
    p.add_argument("--micro-clusters", action="store_true",
                    help="Improvement 5: after normal candidate discovery, sweep the leftover pool for "
                         "very small but very tight same-brand groups. Surfaces issues sitting below "
                         "min_cluster_size, e.g. a 7-record driver regression")
    p.add_argument("--micro-distance", type=float, default=0.6,
                    help="Ward distance ceiling for micro groups. Measured: beyond ~0.8 groups become "
                         "cross-brand semantic mush")
    p.add_argument("--micro-min-size", type=int, default=3)
    p.add_argument("--micro-min-authors", type=int, default=3,
                    help="Distinct authors required. Without this, one person tweeting repeatedly and "
                         "duplicate retweets both fabricate 'topics'")
    p.add_argument("--split-max-size", type=int, default=0,
                    help="Improvement 4: re-cluster any discovery cluster holding more than this many "
                         "segments into sub-topics. 0 disables. Broad topics score low coherence "
                         "because their top words are generic")
    p.add_argument("--split-min-children", type=int, default=2,
                    help="Only accept a split that yields at least this many sub-clusters")
    p.add_argument("--merge-duplicate-topics", action="store_true",
                    help="Improvement 2: merge same-brand topics whose centroids are near-identical")
    p.add_argument("--duplicate-similarity", type=float, default=0.97,
                    help="Centroid cosine similarity above which two same-brand topics are the same topic")
    p.add_argument("--junk-coherence-veto", type=float, default=None,
                    help="Improvement 3: never flag a topic as junk on label/member evidence alone when "
                         "its coherence reaches this. Stops a coherent topic being demoted for merely "
                         "containing polite words")
    p.add_argument("--min-content-words", type=int, default=0,
                    help="Layer 2: drop records with fewer content words than this before embedding. "
                         "1 is conservative (~1.2%% of substantive records lost)")
    p.add_argument("--ack-similarity", type=float, default=0.0,
                    help="Layer 1: exclude records whose max cosine similarity to an acknowledgement "
                         "anchor exceeds this, from clustering and assignment. 0 disables; 0.60 drops "
                         "~23%% of pleasantries at ~0.9%% collateral loss")
    p.add_argument("--flag-junk-topics", action="store_true",
                    help="Layer 3: flag topics with no reputation signal and demote them from the "
                         "HotScore ranking. Nothing is deleted; flagged topics stay in every output")
    p.add_argument("--junk-term-ratio", type=float, default=0.6)
    p.add_argument("--junk-empty-ratio", type=float, default=0.5)
    p.add_argument("--junk-coherence", type=float, default=-0.05,
                    help="Layer 3: flag a topic whose C_NPMI coherence falls below this")
    p.add_argument("--junk-coherence-min-size", type=int, default=30,
                    help="Layer 3: never flag on coherence alone below this many records. NPMI is "
                         "unreliable on small samples, and demoting small topics suppresses exactly "
                         "the emerging signal the buffer exists to surface")
    p.add_argument("--no-coherence", action="store_true",
                    help="Skip inline coherence scoring (saves 1-2 minutes)")
    p.add_argument("--pca-components", type=int, default=50,
                    help="PCA dimensions used for clustering; higher retains more variance at more cost")
    p.add_argument("--buffer", default=None,
                    help="Directory holding a rolling buffer of still-unassigned segments. Enables cross-run "
                         "accumulation so a slow-building topic can reach cluster threshold over several runs.")
    p.add_argument("--buffer-max-age-days", type=int, default=14,
                    help="Discard buffered segments older than this, relative to the newest record in the run")
    p.add_argument("--reducer", default="pca", choices=["pca", "none", "umap"],
                    help="Dimensionality reduction before HDBSCAN. 'none' clusters the raw 384-dim "
                         "embeddings; 'umap' is non-linear and neighbourhood-preserving")
    p.add_argument("--umap-components", type=int, default=50)
    p.add_argument("--umap-neighbors", type=int, default=15,
                    help="UMAP neighbourhood size. Low values favour local structure (more, smaller "
                         "topics); high values favour global structure")
    p.add_argument("--umap-min-dist", type=float, default=0.0,
                    help="0.0 packs points tightly, which is what density clustering wants")
    p.add_argument("--no-pca", action="store_true", help="Disable PCA before HDBSCAN")
    p.add_argument("--plots", action="store_true", help="Write time-series plots if matplotlib is available")
    p.add_argument("--grid-search", action="store_true", help="Run a small HDBSCAN parameter search before final clustering")
    p.add_argument("--previous-topics", default=None, help="Optional topics.json from a previous run; discovered clusters are matched to preserve stable topic IDs")
    p.add_argument("--umap-model", default=None,
                   help="Path to a versioned UMAP artifact (.pkl + .json sidecar). Loaded and "
                        "reused if present, fitted and saved if not. Reusing one artifact is "
                        "what makes topic IDs comparable across runs.")
    p.add_argument("--umap-refit", action="store_true",
                   help="Re-fit and overwrite the --umap-model artifact instead of reusing it.")
    p.add_argument("--umap-reference-size", type=int, default=50000,
                   help="Segments in the UMAP reference sample, drawn evenly across brands and "
                        "time so no single brand defines the manifold. 0 fits on everything.")
    p.add_argument("--carry-forward-topics", action="store_true",
                   help="Keep previous topics that this window did not re-discover, as DORMANT, "
                        "so the registry is cumulative and a quiet topic revives its own ID "
                        "instead of being minted again as new.")
    p.add_argument("--topic-max-age-days", type=int, default=0,
                   help="Retire a carried-forward topic once it has not been seen for this many "
                        "days (0 keeps everything).")
    p.add_argument("--alert-min-coherence", type=float, default=None,
                   help="P1. HOT/TRENDING require at least this coherence. Volume and steadiness "
                        "alone promote conversational grab-bags; an alert earns one extra bar. "
                        "Demoted topics keep their row, score and reason.")
    p.add_argument("--alert-min-size", type=int, default=0,
                   help="P1. HOT/TRENDING also require at least this many records (0 disables).")
    p.add_argument("--residual-ack-ratio", type=float, default=0.40,
                   help="P2. Flag a topic as a residual 'chatter' cluster when at least this share "
                        "of its members are short pure acknowledgements. Measured on members, not "
                        "on the label, which hides it completely.")
    p.add_argument("--split-max-child-similarity", type=float, default=0.92,
                   help="P4. Refuse a split whose children are this similar in the original "
                        "embedding space, so the split pass cannot undo the duplicate merge.")
    p.add_argument("--umap-drift-threshold", type=float, default=0.0,
                   help="P6. Re-fit the frozen manifold when mean distance to its reference set "
                        "exceeds this (0 disables). Past this point every projection is an "
                        "extrapolation, which surfaces as records matching no topic.")
    p.add_argument("--recover-unassigned", action="store_true",
                   help="P7. Give segments that candidate discovery left as noise one more pass, "
                        "per brand, at a smaller cluster size, instead of discarding them.")
    p.add_argument("--recover-min-cluster-size", type=int, default=8,
                   help="P7. Cluster size for the recovery pass.")
    p.add_argument("--recover-min-content-words", type=int, default=4,
                   help="P7. Content words a leftover segment needs before recovery considers it.")
    p.add_argument("--event-recall", action="store_true",
                   help="P3. Write event_recall.csv from the built-in verified-event list, so every "
                        "run carries its own independent check.")
    p.add_argument("--umap-per-brand", action="store_true",
                   help="Fit one frozen manifold per brand instead of one global manifold. A global "
                        "fit averages every brand into one geometry and smooths away small "
                        "brand-local densities; per-brand keeps them, and stays just as reproducible.")
    p.add_argument("--densmap", action="store_true",
                   help="Use densMAP. Plain UMAP normalises away local density variation, which is "
                        "the one thing HDBSCAN actually reads; densMAP preserves it.")
    p.add_argument("--cluster-selection-epsilon", type=float, default=0.0,
                   help="Merge discovered clusters whose centroids sit closer than this fraction of "
                        "the pool's median inter-centroid distance. The anti-fragmentation lever. "
                        "Applied after HDBSCAN because scikit-learn 1.9's own "
                        "cluster_selection_epsilon crashes at any value that has an effect.")
    p.add_argument("--max-cluster-size", type=int, default=0,
                   help="HDBSCAN: reject clusters larger than this many segments (0 disables).")
    p.add_argument("--candidate-margin", type=float, default=0.0,
                   help="A candidate cluster only joins an existing topic if its best match beats "
                        "its second-best by this margin. Guards against a dense topic space where "
                        "every candidate finds something above the threshold and nothing is new.")
    p.add_argument("--umap-refit-per-pool", action="store_true",
                   help="Old unstable behaviour: fit a fresh UMAP per brand pool per run. "
                        "Only for reproducing the original benchmark arm.")
    p.add_argument("--device", default=None,
                   help="Torch device for the embedding model: cuda, cuda:1, cpu, mps. "
                        "Default: cuda when a GPU is visible, else cpu.")
    p.add_argument("--batch-size", type=int, default=None,
                   help="Embedding batch size. Default: 512 on CUDA, 64 on CPU. "
                        "Raise it until the GPU is saturated or it runs out of memory.")
    p.add_argument("--fp16", action="store_true",
                   help="Run the embedding model in half precision on CUDA. Roughly 2x faster "
                        "with a small numerical difference in the embeddings.")
    return p.parse_args()


def content_word_count(text: str, stop: frozenset) -> int:
    """Tokens left once stopwords and acknowledgement words are removed.

    Separates "Ok, thanks." (0) from "Ok. Thanks. I have logged a complaint." (2),
    which a length threshold cannot do -- the second must be kept.
    """
    return sum(1 for w in WORD_RE.findall(str(text).lower())
               if w not in stop and len(w) > 2)


def build_content_stopset() -> frozenset:
    try:
        from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
        return frozenset(set(ENGLISH_STOP_WORDS) | ACK_WORDS)
    except Exception:
        return frozenset(ACK_WORDS)


def acknowledgement_similarity(model: SentenceTransformer, embeddings: np.ndarray) -> np.ndarray:
    """Max cosine similarity of each row to any acknowledgement anchor."""
    anchors = model.encode(ACK_ANCHORS, normalize_embeddings=True,
                           convert_to_numpy=True).astype(np.float32)
    return (embeddings @ anchors.T).max(axis=1)


def npmi_coherence(
    topic_terms: Dict[str, List[str]],
    corpus: Sequence[str],
) -> Dict[str, float]:
    """C_NPMI per topic: do a topic's top words actually co-occur in real text?

    Correlates with human judgement far better than silhouette, which only
    measures geometry and happily rewards a tight cluster of "thanks".
    Range [-1, 1]; >0.1 reasonable, <=0 means the words do not belong together.
    """
    vocab = sorted({w for ws in topic_terms.values() for w in ws})
    if not vocab:
        return {}
    cv = CountVectorizer(vocabulary=vocab, token_pattern=r"(?u)\b[a-z][a-z']{2,}\b",
                         binary=True, lowercase=True)
    X = cv.fit_transform([str(c) for c in corpus]).astype(bool).tocsc()
    n_docs = X.shape[0]
    df_cnt = np.asarray(X.sum(axis=0)).ravel()
    idx = {w: i for i, w in enumerate(vocab)}

    def pair(a: str, b: str) -> float:
        ia, ib = idx[a], idx[b]
        na, nb = df_cnt[ia], df_cnt[ib]
        if na == 0 or nb == 0:
            return 0.0
        nab = int(X[:, ia].multiply(X[:, ib]).sum())
        if nab == 0:
            return -1.0
        pab, pa, pb = nab / n_docs, na / n_docs, nb / n_docs
        return float(np.log(pab / (pa * pb)) / -np.log(pab))

    out = {}
    for tid, ws in topic_terms.items():
        ws = [w for w in ws if w in idx]
        if len(ws) < 2:
            out[tid] = float("nan")
            continue
        scores = [pair(ws[i], ws[j]) for i in range(len(ws)) for j in range(i + 1, len(ws))]
        out[tid] = float(np.mean(scores)) if scores else float("nan")
    return out


def junk_signals(label: str, member_texts: Sequence[str], stop: frozenset) -> Tuple[float, float]:
    """Two independent reads on whether a topic carries reputation signal.

    They fail differently, so either firing is enough: the label test catches
    "thanks / thankyou" but misses a topic of real-but-empty words in another
    language; the member test catches that regardless of the label.
    """
    terms = [t.strip() for t in str(label).split("/") if t.strip()]
    junk_terms = 0
    for t in terms:
        toks = WORD_RE.findall(t.lower())
        if toks and all(w in ACK_WORDS for w in toks):
            junk_terms += 1
    junk_term_ratio = junk_terms / max(1, len(terms))
    if len(member_texts):
        empty = sum(1 for t in member_texts if content_word_count(t, stop) == 0)
        empty_member_ratio = empty / len(member_texts)
    else:
        empty_member_ratio = 0.0
    return junk_term_ratio, empty_member_ratio


def normalize_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = text.replace("\u200b", " ").replace("\r", " ").replace("\n", " ")
    text = URL_RE.sub(" ", text)
    text = MENTION_RE.sub(" ", text)
    # Keep hashtag words but remove the # marker.
    text = text.replace("#", " ")
    text = MULTI_PUNCT_RE.sub(r"\1", text)
    text = WHITESPACE_RE.sub(" ", text).strip()
    return text


def sentence_split(text: str) -> List[str]:
    parts = re.split(r"(?<=[.!?])\s+", text)
    return [p.strip() for p in parts if p.strip()]


def chunk_long_text(text: str, max_chars: int, overlap_chars: int) -> List[str]:
    """Sentence-aware chunking with hard char fallback.

    The key design choice: chunk for semantic representation, but later collapse
    assignments back to DISTINCT record_id for time-series counts.
    """
    if len(text) <= max_chars:
        return [text]

    sentences = sentence_split(text)
    if not sentences:
        return hard_chunk(text, max_chars, overlap_chars)

    chunks: List[str] = []
    current = ""
    for sent in sentences:
        if len(sent) > max_chars:
            if current:
                chunks.append(current.strip())
                current = ""
            chunks.extend(hard_chunk(sent, max_chars, overlap_chars))
            continue

        proposed = f"{current} {sent}".strip() if current else sent
        if len(proposed) <= max_chars:
            current = proposed
        else:
            if current:
                chunks.append(current.strip())
            current = sent

    if current:
        chunks.append(current.strip())

    return chunks or [text]


def hard_chunk(text: str, max_chars: int, overlap_chars: int) -> List[str]:
    chunks = []
    step = max(1, max_chars - overlap_chars)
    for start in range(0, len(text), step):
        part = text[start : start + max_chars].strip()
        if part:
            chunks.append(part)
    return chunks


def segment_records(
    df: pd.DataFrame,
    id_col: str,
    text_col: str,
    max_chars: int,
    overlap_chars: int,
    max_segments_per_record: int,
) -> pd.DataFrame:
    rows = []
    for _, row in df.iterrows():
        raw = str(row.get(text_col, "") or "")
        text = normalize_text(raw)
        if not text:
            continue

        chunks = chunk_long_text(text, max_chars, overlap_chars)
        # Prevent a pathological record from flooding HDBSCAN.
        if len(chunks) > max_segments_per_record:
            idx = np.linspace(0, len(chunks) - 1, max_segments_per_record, dtype=int)
            chunks = [chunks[i] for i in idx]

        for i, chunk in enumerate(chunks):
            rec = row.to_dict()
            rec["record_id"] = str(row[id_col])
            rec["segment_id"] = f"{row[id_col]}::{i}"
            rec["segment_index"] = i
            rec["raw_text"] = raw
            rec["clean_text"] = chunk
            rec["segment_count"] = len(chunks)
            rec["record_weight"] = 1.0 / max(1, len(chunks))
            rows.append(rec)

    out = pd.DataFrame(rows)
    if out.empty:
        raise ValueError("No usable text records after normalization.")
    return out


def l2_normalize(x: np.ndarray) -> np.ndarray:
    denom = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.clip(denom, 1e-12, None)


def cosine_sim(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = a / max(np.linalg.norm(a), 1e-12)
    b = b / np.clip(np.linalg.norm(b, axis=1, keepdims=True), 1e-12, None)
    return b @ a


def embedding_cache_key(model_name: str, texts: Sequence[str], precision: str = "fp32") -> str:
    """Identity of an embedding set: the model, its precision, and the exact text it saw.

    Any change to the text or the model yields a different key, so a stale
    cache cannot be read back by accident. Precision is part of the identity
    because an fp16 run produces numerically different vectors from an fp32
    run over the same text -- without it the two would silently share a cache.
    """
    h = hashlib.sha256()
    h.update(model_name.encode("utf-8"))
    h.update(b"\x00")
    # fp32 deliberately contributes nothing, so it reproduces the key this
    # pipeline used before precision entered the identity. That keeps caches
    # written by the CPU prototype readable here, while fp16 still gets a
    # distinct key instead of silently sharing fp32's vectors.
    if precision != "fp32":
        h.update(precision.encode("utf-8"))
        h.update(b"\x00")
    for t in texts:
        h.update(str(t).encode("utf-8", "replace"))
        h.update(b"\x00")
    return h.hexdigest()[:32]


def load_cached_embeddings(cache_dir: Optional[str], key: str) -> Optional[np.ndarray]:
    if not cache_dir:
        return None
    f = Path(cache_dir) / f"{key}.npy"
    if not f.exists():
        return None
    try:
        return np.load(f).astype(np.float32)
    except Exception:
        return None


def save_cached_embeddings(cache_dir: Optional[str], key: str, emb: np.ndarray) -> None:
    if not cache_dir:
        return
    d = Path(cache_dir)
    d.mkdir(parents=True, exist_ok=True)
    np.save(d / f"{key}.npy", emb.astype(np.float32))


def resolve_device(requested: Optional[str] = None) -> str:
    """Pick the device the embedding model runs on.

    Explicit --device always wins so a run can be pinned to one GPU in a
    multi-GPU box (cuda:1) or forced onto CPU for a comparison run.
    """
    if requested:
        return requested
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return "mps"
    except Exception:
        pass
    return "cpu"


def default_batch_size(device: str) -> int:
    """A GPU is starved by the CPU-era batch of 64; a CPU is not helped by more."""
    return 512 if str(device).startswith("cuda") else 64


def describe_device(device: str) -> str:
    if not str(device).startswith("cuda"):
        return device
    try:
        import torch
        idx = int(device.split(":")[1]) if ":" in device else torch.cuda.current_device()
        name = torch.cuda.get_device_name(idx)
        total = torch.cuda.get_device_properties(idx).total_memory / (1024 ** 3)
        return f"{device} ({name}, {total:.1f} GiB)"
    except Exception:
        return device


def load_embedding_model(model_name: str, device: str, fp16: bool = False) -> SentenceTransformer:
    model = SentenceTransformer(model_name, device=device)
    if fp16 and str(device).startswith("cuda"):
        model = model.half()
    return model


def build_embeddings(model: SentenceTransformer, texts: Sequence[str], batch_size: int = 64) -> np.ndarray:
    emb = model.encode(
        list(texts),
        batch_size=batch_size,
        show_progress_bar=True,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )
    return emb.astype(np.float32)


def reduce_dimensions(emb: np.ndarray, n_components: int, random_state: int) -> Tuple[np.ndarray, Optional[PCA]]:
    if n_components <= 0 or emb.shape[1] <= n_components or emb.shape[0] <= n_components + 2:
        return emb, None
    pca = PCA(n_components=n_components, random_state=random_state)
    reduced = pca.fit_transform(emb)
    return reduced.astype(np.float32), pca


UMAP_ARTIFACT_VERSION = "v1"


def umap_reduce(
    emb: np.ndarray,
    n_components: int,
    n_neighbors: int,
    min_dist: float,
    random_state: int,
    densmap: bool = False,
) -> np.ndarray:
    """Fit-and-transform UMAP in one shot. Unstable; kept only for --umap-refit-per-pool.

    PCA is linear: it keeps the directions of greatest variance and cannot
    unfold structure that is curved in the embedding space. UMAP is non-linear
    and optimises for preserving local neighbourhoods, which is what a density
    clusterer actually consumes -- so in principle it suits HDBSCAN better.

    The cost is stability. Fitting per pool per run means every brand lands in
    its own unrelated coordinate system and nothing is comparable between runs,
    which is exactly what fit_umap_model/UmapModel below exist to fix. Prefer
    those; this remains only to reproduce the old benchmark arm.
    """
    import umap                                   # imported lazily: heavy, optional

    if n_components <= 0 or emb.shape[0] <= n_components + 2:
        return emb
    reducer = umap.UMAP(
        n_components=min(n_components, max(2, emb.shape[0] - 2)),
        n_neighbors=min(n_neighbors, max(2, emb.shape[0] - 1)),
        min_dist=min_dist,
        metric="cosine",
        random_state=random_state,
        densmap=bool(densmap),
        verbose=False,
    )
    return np.asarray(reducer.fit_transform(emb), dtype=np.float32)


def umap_reference_sample(
    segments: pd.DataFrame,
    embeddings: np.ndarray,
    target_size: int,
    seed: int = 42,
) -> np.ndarray:
    """Row indices for a reference set that represents the whole semantic space.

    A plain random sample is dominated by whichever brand tweets most, so the
    manifold gets fitted mostly around that brand. Instead take an equal quota
    per brand, and inside each brand spread the draw evenly over time so a
    single busy week cannot stand in for the whole period. Brands with fewer
    rows than their quota contribute everything they have and their unused
    quota is redistributed over the remaining brands.
    """
    n = len(segments)
    if target_size <= 0 or target_size >= n:
        return np.arange(n)

    rng = np.random.default_rng(seed)
    brands = segments["brand"].astype(str).to_numpy()
    order = pd.to_datetime(segments["event_time"], errors="coerce").to_numpy()

    by_brand: Dict[str, np.ndarray] = {}
    for b in pd.unique(brands):
        by_brand[str(b)] = np.where(brands == b)[0]

    # Smallest pools first, so quota freed by an under-full brand is handed to
    # the brands that can still absorb it.
    remaining = target_size
    names = sorted(by_brand, key=lambda b: len(by_brand[b]))
    picked: List[np.ndarray] = []
    for k, b in enumerate(names):
        quota = remaining // (len(names) - k)
        idx = by_brand[b]
        if len(idx) <= quota:
            take = idx
        else:
            # Even spread over time: bucket by timestamp order, draw per bucket.
            idx = idx[np.argsort(order[idx], kind="stable")]
            buckets = np.array_split(idx, min(quota, len(idx)))
            take = np.array([rng.choice(bk) for bk in buckets if len(bk)], dtype=int)
        picked.append(take)
        remaining -= len(take)
    return np.sort(np.concatenate(picked)) if picked else np.arange(n)


def fit_umap_model(
    emb: np.ndarray,
    n_components: int,
    n_neighbors: int,
    min_dist: float,
    random_state: int,
    densmap: bool = False,
):
    """Fit the one UMAP that every pool and every later run will transform through.

    densMAP is worth the extra fit cost here in principle: plain UMAP explicitly
    normalises away variation in local density, and local density is precisely
    what HDBSCAN consumes. densMAP adds a term that preserves it, so a sparse
    real topic should stay sparse instead of being inflated into something the
    clusterer reads as dense.
    """
    import umap                                   # imported lazily: heavy, optional

    reducer = umap.UMAP(
        n_components=min(n_components, max(2, emb.shape[0] - 2)),
        n_neighbors=min(n_neighbors, max(2, emb.shape[0] - 1)),
        min_dist=min_dist,
        metric="cosine",
        random_state=random_state,
        densmap=bool(densmap),
        verbose=False,
    )
    reducer.fit(emb)
    return reducer


def umap_model_paths(path: str) -> Tuple[Path, Path]:
    """A UMAP artifact is the pickle plus a readable sidecar describing it."""
    p = Path(path)
    if p.suffix != ".pkl":
        p = p.with_suffix(".pkl")
    return p, p.with_name(p.stem + ".json")


def save_umap_model(models, meta: Dict, path: str,
                    refs: Optional[Dict[str, np.ndarray]] = None) -> Tuple[Path, Path]:
    """Pickle the manifolds plus their reference vectors; JSON sidecar describes them.

    The reference vectors ride along because drift cannot be measured without
    them: a reloaded manifold with no memory of what it was fitted on can only
    be trusted, never checked.
    """
    import pickle

    model_path, meta_path = umap_model_paths(path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    if not isinstance(models, dict):
        models = {"__global__": models}
    with open(model_path, "wb") as f:
        pickle.dump({"models": models, "refs": refs or {}}, f, protocol=pickle.HIGHEST_PROTOCOL)
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return model_path, meta_path


def load_umap_model(path: str):
    """Return (models_dict, meta) for a saved artifact, or (None, None) if absent."""
    import pickle

    model_path, meta_path = umap_model_paths(path)
    if not model_path.exists():
        return None, None
    with open(model_path, "rb") as f:
        payload = pickle.load(f)
    meta = {}
    if meta_path.exists():
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except Exception:
            meta = {}
    if isinstance(payload, dict) and "models" in payload:
        meta["_refs"] = payload.get("refs", {})
        return payload["models"], meta
    # An artifact written before manifolds became plural.
    return {"__global__": payload}, meta


def umap_model_is_compatible(meta: Dict, args, input_dim: int) -> Tuple[bool, str]:
    """Refuse a saved model whose space is not the space we are now embedding in.

    A UMAP fitted on different vectors -- another embedding model, another
    width, another parameter set -- would silently project today's data through
    yesterday's unrelated manifold. Better to say so than to produce topics
    nobody can trace.
    """
    if not meta:
        return True, "no metadata recorded; reusing on trust"
    checks = [
        ("embedding_model", meta.get("embedding_model"), args.model),
        ("input_dimension", meta.get("input_dimension"), input_dim),
        ("n_components", meta.get("n_components"), args.umap_components),
        ("n_neighbors", meta.get("n_neighbors"), args.umap_neighbors),
        ("min_dist", meta.get("min_dist"), args.umap_min_dist),
        ("densmap", meta.get("densmap"), bool(args.densmap)),
        ("scope", meta.get("scope"), "per-brand" if args.umap_per_brand else "global"),
    ]
    for name, saved, current in checks:
        if saved is not None and saved != current:
            return False, f"{name}: model has {saved!r}, run asks for {current!r}"
    return True, "compatible"


class UmapProjector:
    """Fitted UMAP manifolds plus the metadata that makes a run reproducible.

    Holds either one global manifold or one per brand. Pools call transform();
    nothing refits. That is what makes two runs over overlapping data land in
    the same coordinate system, which is the precondition for topic IDs that
    survive across runs.

    Every transform is also measured against the data the manifold was fitted
    on. A frozen manifold is only trustworthy while incoming data still looks
    like its reference set; without that measurement "do not refit often"
    quietly becomes "project today's data through a stale manifold".
    """

    def __init__(self, models: Dict[str, object], meta: Dict, refs: Optional[Dict[str, np.ndarray]] = None):
        self.models = models                      # key -> fitted UMAP ("__global__" or brand)
        self.meta = meta
        self._refs = refs or {}                   # key -> reference embeddings, for drift
        self.drift_rows: List[Dict] = []

    @property
    def model(self):
        return self.models.get("__global__") or next(iter(self.models.values()))

    def _key(self, brand: Optional[str]) -> str:
        if brand is not None and brand in self.models:
            return brand
        return "__global__" if "__global__" in self.models else next(iter(self.models))

    def _drift(self, key: str, emb: np.ndarray) -> Optional[float]:
        """Mean cosine distance from each incoming point to its nearest reference point.

        Cheap and interpretable: it rises when incoming data occupies parts of
        the embedding space the manifold never saw, which is the condition that
        should trigger a re-fit.
        """
        ref = self._refs.get(key)
        if ref is None or len(ref) == 0 or emb.shape[0] == 0:
            return None
        rng = np.random.default_rng(0)
        probe = emb if emb.shape[0] <= 2000 else emb[rng.choice(emb.shape[0], 2000, replace=False)]
        anchor = ref if len(ref) <= 5000 else ref[rng.choice(len(ref), 5000, replace=False)]
        # Both sides are L2-normalised, so the dot product is the cosine.
        best = (probe @ anchor.T).max(axis=1)
        return float(np.mean(1.0 - best))

    def transform(self, emb: np.ndarray, brand: Optional[str] = None) -> np.ndarray:
        if emb.shape[0] == 0:
            return emb
        key = self._key(brand)
        d = self._drift(key, emb)
        if d is not None:
            self.drift_rows.append({"pool": brand or key, "manifold": key,
                                    "n": int(emb.shape[0]), "mean_dist_to_reference": round(d, 5)})
        return np.asarray(self.models[key].transform(emb), dtype=np.float32)

    def drift_frame(self) -> pd.DataFrame:
        return pd.DataFrame(self.drift_rows)


def get_or_fit_umap(
    segments: pd.DataFrame,
    embeddings: np.ndarray,
    args,
    timer=None,
) -> UmapProjector:
    """Load the versioned UMAP(s) if present and compatible, otherwise fit and save.

    Fitting is the expensive, stochastic step, so it happens once and is then
    frozen on disk. Everything after this point only transforms.

    With --umap-per-brand the artifact holds one manifold per brand instead of
    one global manifold. A global fit averages 12 brands into a single geometry,
    which smooths away exactly the small brand-local densities a reputation
    system is looking for; a per-brand fit keeps them, and stays just as frozen
    and just as reproducible.
    """
    input_dim = int(embeddings.shape[1])
    path = args.umap_model

    if args.densmap:
        # umap-learn refuses transform() on a densMAP model, so a densMAP
        # manifold can never be frozen and reused -- it is fit_transform only.
        # That is a design incompatibility, not a tuning choice: densMAP and
        # cross-run stable topic IDs cannot both be had.
        raise SystemExit(
            "--densmap cannot be combined with a frozen UMAP manifold: umap-learn does not "
            "support transform() on a densMAP model.\nRun it with --umap-refit-per-pool to "
            "evaluate densMAP quality, accepting that cross-run topic IDs are no longer stable."
        )

    if path and not args.umap_refit:
        payload, meta = load_umap_model(path)
        if payload is not None:
            ok, why = umap_model_is_compatible(meta, args, input_dim)
            if ok:
                models = payload if isinstance(payload, dict) else {"__global__": payload}
                refs = _decode_refs(meta)
                print(f"UMAP model: reusing {umap_model_paths(path)[0]} "
                      f"({meta.get('umap_version', '?')}, scope={meta.get('scope', 'global')}, "
                      f"{len(models)} manifold(s), fitted on "
                      f"{meta.get('reference_size', '?')} segments)")
                if timer is not None:
                    timer.mark("umap load", f"reused {meta.get('umap_version', '?')}")
                return UmapProjector(models, meta, refs)
            raise SystemExit(
                f"Saved UMAP model at {umap_model_paths(path)[0]} does not match this run "
                f"-- {why}.\nRe-fit it with --umap-refit, or point --umap-model at a "
                f"different artifact."
            )

    models: Dict[str, object] = {}
    refs: Dict[str, np.ndarray] = {}
    per_scope: Dict[str, Dict] = {}
    t0 = time.perf_counter()

    if args.umap_per_brand:
        brands = segments["brand"].astype(str)
        # Brands too small to support their own manifold fall back to the shared
        # one, the same way discovery pools them into __SMALL_BRANDS__.
        counts = brands.value_counts()
        big = [b for b, n in counts.items() if n >= max(args.umap_neighbors * 20, 500)]
        print(f"UMAP model: fitting {len(big)} per-brand manifolds + 1 shared fallback"
              f"{' (densMAP)' if args.densmap else ''}")
        for b in big:
            idx = np.where(brands.to_numpy() == b)[0]
            take = umap_reference_sample(segments.iloc[idx].reset_index(drop=True),
                                         embeddings[idx], args.umap_reference_size // max(1, len(big)),
                                         seed=42)
            sub = idx[take]
            models[b] = fit_umap_model(embeddings[sub], args.umap_components, args.umap_neighbors,
                                       args.umap_min_dist, 42, args.densmap)
            refs[b] = embeddings[sub]
            per_scope[b] = {"reference_size": int(len(sub))}
        ref_idx = umap_reference_sample(segments, embeddings, args.umap_reference_size, seed=42)
        models["__global__"] = fit_umap_model(embeddings[ref_idx], args.umap_components,
                                              args.umap_neighbors, args.umap_min_dist, 42, args.densmap)
        refs["__global__"] = embeddings[ref_idx]
    else:
        ref_idx = umap_reference_sample(segments, embeddings, args.umap_reference_size, seed=42)
        ref_brands = segments.iloc[ref_idx]["brand"].astype(str).value_counts().to_dict()
        print(f"UMAP model: fitting on a {len(ref_idx):,}-segment reference sample "
              f"across {len(ref_brands)} brands "
              f"(largest share {max(ref_brands.values()) / max(1, len(ref_idx)):.1%})"
              f"{' (densMAP)' if args.densmap else ''}")
        models["__global__"] = fit_umap_model(embeddings[ref_idx], args.umap_components,
                                              args.umap_neighbors, args.umap_min_dist, 42, args.densmap)
        refs["__global__"] = embeddings[ref_idx]

    fit_s = time.perf_counter() - t0
    ref_times = pd.to_datetime(segments.iloc[ref_idx]["event_time"], errors="coerce")
    meta = {
        "umap_version": UMAP_ARTIFACT_VERSION,
        "scope": "per-brand" if args.umap_per_brand else "global",
        "manifolds": sorted(models.keys()),
        "embedding_model": args.model,
        "input_dimension": input_dim,
        "n_components": int(args.umap_components),
        "n_neighbors": int(args.umap_neighbors),
        "min_dist": float(args.umap_min_dist),
        "densmap": bool(args.densmap),
        "metric": "cosine",
        "random_state": 42,
        "reference_size": int(len(ref_idx)),
        "reference_brands": segments.iloc[ref_idx]["brand"].astype(str).value_counts().to_dict(),
        "per_manifold_reference": per_scope,
        "reference_window": [
            None if pd.isna(ref_times.min()) else ref_times.min().isoformat(),
            None if pd.isna(ref_times.max()) else ref_times.max().isoformat(),
        ],
        "umap_learn_version": _umap_learn_version(),
        "fit_seconds": round(fit_s, 2),
        "fitted_at": pd.Timestamp.utcnow().isoformat(),
    }
    if timer is not None:
        timer.mark("umap fit", f"{len(models)} manifold(s), {len(ref_idx):,} reference segments")

    if path:
        model_path, meta_path = save_umap_model(models, meta, path, refs)
        print(f"UMAP model: saved {model_path} and {meta_path}")
    return UmapProjector(models, meta, refs)


def _decode_refs(meta: Dict) -> Dict[str, np.ndarray]:
    """Reference vectors travel inside the pickle; the sidecar only describes them."""
    return meta.pop("_refs", {}) if isinstance(meta, dict) else {}


def _umap_learn_version() -> Optional[str]:
    try:
        import umap
        return getattr(umap, "__version__", None)
    except Exception:
        return None


def apply_reducer(emb: np.ndarray, args, projector: Optional[UmapProjector] = None,
                  brand: Optional[str] = None) -> np.ndarray:
    """Dispatch to whichever dimensionality reduction the run selected."""
    if args.reducer == "none" or args.no_pca:
        return emb
    if args.reducer == "umap":
        if projector is not None:
            return projector.transform(emb, brand)
        return umap_reduce(emb, args.umap_components, args.umap_neighbors,
                           args.umap_min_dist, 42, args.densmap)
    return reduce_dimensions(emb, min(args.pca_components, emb.shape[1]), 42)[0]


def cluster_embeddings(
    emb: np.ndarray,
    min_cluster_size: int,
    min_samples: int,
    method: str,
    cluster_selection_epsilon: float = 0.0,
    max_cluster_size: int = 0,
) -> np.ndarray:
    """HDBSCAN over whatever space the reducer produced.

    `cluster_selection_epsilon` is HDBSCAN's own anti-fragmentation lever: any
    two clusters whose merge height sits below it are kept merged rather than
    split. It is the direct remedy for a reducer that packs neighbourhoods so
    tightly that one real topic reads as several dense clumps, which is exactly
    what UMAP does. It is scale-dependent, so a value tuned in one embedding
    space means nothing in another.
    """
    clusterer = HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        metric="euclidean",
        cluster_selection_method=method,
        max_cluster_size=(int(max_cluster_size) if max_cluster_size and max_cluster_size > 0 else None),
        allow_single_cluster=False,
    )
    labels = clusterer.fit_predict(emb)
    # Applied here rather than passed to HDBSCAN -- see merge_clusters_by_epsilon.
    return merge_clusters_by_epsilon(emb, labels, cluster_selection_epsilon)


def merge_clusters_by_epsilon(
    X_cluster: np.ndarray,
    labels: np.ndarray,
    epsilon: float,
) -> np.ndarray:
    """Merge clusters whose centroids sit closer than `epsilon` in the reduced space.

    This is our stand-in for HDBSCAN's `cluster_selection_epsilon`, which is the
    textbook anti-fragmentation lever but is unusable in scikit-learn 1.9:
    measured on a real pool, any value below ~0.15 changed nothing and every
    value from 0.15 up raised `TypeError: only 0-dimensional arrays can be
    converted to Python scalars` inside `_tree.pyx: traverse_upwards`, for both
    selection methods and both `allow_single_cluster` settings. In other words
    it crashes at precisely the point it would start doing something.

    `epsilon` here is expressed as a fraction of the pool's median inter-centroid
    distance rather than in raw units. UMAP output has no fixed scale, so a raw
    threshold tuned at one `n_components` is meaningless at another; a relative
    one survives the grid search.
    """
    if epsilon <= 0:
        return labels
    ids = sorted({int(c) for c in labels if c >= 0})
    if len(ids) < 2:
        return labels

    cents = np.vstack([X_cluster[labels == c].mean(axis=0) for c in ids])
    d = np.linalg.norm(cents[:, None, :] - cents[None, :, :], axis=-1)
    iu = np.triu_indices(len(ids), k=1)
    scale = float(np.median(d[iu]))
    if not np.isfinite(scale) or scale <= 0:
        return labels
    cutoff = epsilon * scale

    # Union-find so a chain of close clusters collapses into one group.
    parent = list(range(len(ids)))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for a, b in zip(*iu):
        if d[a, b] < cutoff:
            ra, rb = find(int(a)), find(int(b))
            if ra != rb:
                parent[max(ra, rb)] = min(ra, rb)

    remap = {}
    out = labels.copy()
    for k, c in enumerate(ids):
        root = find(k)
        remap[c] = ids[root]
    for c, target in remap.items():
        if c != target:
            out[labels == c] = target
    return out


def cluster_quality(emb: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
    mask = labels >= 0
    n_noise = int((~mask).sum())
    n_clusters = int(len(set(labels[mask]))) if mask.any() else 0
    result = {
        "noise_ratio": float(n_noise / len(labels)) if len(labels) else 1.0,
        "n_clusters": float(n_clusters),
        "silhouette": float("nan"),
    }
    if n_clusters >= 2 and mask.sum() >= 3:
        try:
            result["silhouette"] = float(silhouette_score(emb[mask], labels[mask], metric="euclidean"))
        except Exception:
            pass
    return result


def silhouette_subsampled(emb: np.ndarray, labels: np.ndarray, cap: int = 5000,
                          rng: Optional[np.random.Generator] = None) -> float:
    """Silhouette on at most `cap` points.

    The full pairwise matrix is O(n^2): at 12k points that is ~1.15 GB per call,
    which is what made --grid-search impractical at this scale.
    """
    rng = rng or np.random.default_rng(42)
    mask = labels >= 0
    if mask.sum() < 3 or len(set(labels[mask])) < 2:
        return float("nan")
    idx = np.where(mask)[0]
    if len(idx) > cap:
        idx = rng.choice(idx, cap, replace=False)
    if len(set(labels[idx])) < 2:
        return float("nan")
    try:
        return float(silhouette_score(emb[idx], labels[idx], metric="euclidean"))
    except Exception:
        return float("nan")


def max_cluster_share(labels: np.ndarray) -> float:
    """Share of the pool held by the single largest cluster."""
    mask = labels >= 0
    if not mask.any():
        return 0.0
    return float(np.bincount(labels[mask]).max() / len(labels))


def choose_pca_width(
    emb: np.ndarray,
    widths: Sequence[int],
    min_cluster_size: int,
    min_samples: int,
    max_share: float,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[int, pd.DataFrame]:
    """Pick the PCA width for one pool.

    Two things make this non-obvious:

    1. Scoring happens in the FULL embedding space, never the reduced one.
       Silhouette shrinks as dimensionality rises, so scoring a 50-d fit in 50-d
       and a 100-d fit in 100-d is biased toward the smaller width before any
       real quality difference is considered.
    2. A clustering where one cluster swallows the pool is rejected outright.
       It scores well on noise ratio while being useless -- one brand collapsed
       to a single blob holding 63% of its records this way.
    """
    rng = rng or np.random.default_rng(42)
    rows = []
    best, best_score = None, -1e9
    for k in widths:
        if k <= 0 or k >= emb.shape[1]:
            X = emb
            k_eff = emb.shape[1]
        else:
            X, _ = reduce_dimensions(emb, k, 42)
            k_eff = k
        labels = cluster_embeddings(X, min_cluster_size, min_samples, "eom")
        mask = labels >= 0
        n_clusters = int(len(set(labels[mask]))) if mask.any() else 0
        noise = float((~mask).mean())
        share = max_cluster_share(labels)
        # Score in the original space so widths are comparable.
        sil = silhouette_subsampled(emb, labels, rng=rng)
        rejected = n_clusters < 2 or share > max_share
        # Noise penalty is deliberately light: 80-95% noise is the healthy
        # regime here, and the share guard now covers the degenerate case.
        score = (-1e9 if rejected
                 else (sil if np.isfinite(sil) else -0.5) + 0.10 * math.log1p(n_clusters) - 0.10 * noise)
        rows.append({"pca_components": k_eff, "n_clusters": n_clusters, "noise_ratio": noise,
                     "max_cluster_share": share, "silhouette_full_space": sil,
                     "rejected": rejected, "objective": None if rejected else score})
        if score > best_score:
            best_score, best = score, k_eff
    return (best if best is not None else min(widths)), pd.DataFrame(rows)


def choose_hdbscan_params(
    emb: np.ndarray,
    candidates: Iterable[Tuple[int, int]],
) -> Tuple[int, int, pd.DataFrame]:
    rows = []
    best = None
    best_score = -1e9
    for min_cluster_size, min_samples in candidates:
        labels = cluster_embeddings(emb, min_cluster_size, min_samples, "eom")
        q = cluster_quality(emb, labels)
        sil = q["silhouette"] if np.isfinite(q["silhouette"]) else -0.5
        noise_penalty = q["noise_ratio"]
        cluster_bonus = math.log1p(q["n_clusters"])
        # Heuristic objective: favor coherent clusters, avoid near-total noise,
        # and avoid the degenerate solution of one giant cluster.
        score = sil + 0.10 * cluster_bonus - 0.35 * noise_penalty
        rows.append({
            "min_cluster_size": min_cluster_size,
            "min_samples": min_samples,
            **q,
            "objective": score,
        })
        if score > best_score:
            best_score = score
            best = (min_cluster_size, min_samples)

    assert best is not None
    return best[0], best[1], pd.DataFrame(rows).sort_values("objective", ascending=False)


def top_terms(texts: Sequence[str], n: int = 6) -> List[str]:
    if not texts:
        return []
    try:
        vec = TfidfVectorizer(
            stop_words="english",
            ngram_range=(1, 2),
            min_df=1,
            max_features=1000,
        )
        X = vec.fit_transform(texts)
        scores = np.asarray(X.mean(axis=0)).ravel()
        terms = np.asarray(vec.get_feature_names_out())
        order = np.argsort(scores)[::-1]
        return terms[order[:n]].tolist()
    except Exception:
        return []


def make_topic_label(texts: Sequence[str]) -> str:
    terms = top_terms(texts, n=5)
    return " / ".join(terms) if terms else "Untitled topic"


def ctfidf_labels(
    member_texts: Dict[str, Sequence[str]],
    top_n: int = 5,
    ngram_range: Tuple[int, int] = (1, 2),
) -> Dict[str, List[str]]:
    """Class-based TF-IDF labels.

    The per-topic TF-IDF this replaces scores a topic's terms only against
    itself, so whatever is frequent wins and every Amazon topic ends up called
    "amazon / order / delivery". c-TF-IDF concatenates each topic into one
    pseudo-document and scores terms ACROSS topics, so a term earns its place by
    being characteristic of this topic rather than merely common in it.

        tf   = count(term, topic) / total terms in topic
        idf  = log(1 + N_topics / documents containing term)

    Terms are then de-duplicated so a label cannot read
    "delivery / late delivery / delivery late".
    """
    tids = [t for t, txts in member_texts.items() if txts]
    if not tids:
        return {}
    docs = [" ".join(str(x) for x in member_texts[t]) for t in tids]
    try:
        # Acknowledgement words are suppressed by name. Scoring within a brand
        # makes "thanks" look distinctive to whichever topic collects the
        # thank-yous, so the cross-brand effect that used to remove it is gone.
        cv = CountVectorizer(stop_words=label_stopwords(),
                             ngram_range=ngram_range,
                             min_df=1, max_features=60000, lowercase=True)
        X = cv.fit_transform(docs).astype(np.float32)
    except ValueError:
        return {}

    counts = np.asarray(X.sum(axis=1)).ravel()
    counts[counts == 0] = 1.0
    tf = X.multiply(1.0 / counts[:, None]).tocsr()

    n_topics = X.shape[0]
    df = np.asarray((X > 0).sum(axis=0)).ravel().astype(np.float32)
    idf = np.log(1.0 + (n_topics / np.maximum(df, 1.0)))

    terms = np.asarray(cv.get_feature_names_out())
    out: Dict[str, List[str]] = {}
    for i, tid in enumerate(tids):
        row = tf.getrow(i)
        if row.nnz == 0:
            out[tid] = []
            continue
        scores = row.toarray().ravel() * idf
        order = np.argsort(scores)[::-1]
        picked: List[str] = []
        for j in order[: top_n * 12]:
            if scores[j] <= 0:
                break
            cand = terms[j]
            if is_noise_label_term(cand):
                continue
            cand_words = set(cand.split())
            # Skip a phrase that merely restates one already chosen.
            if any(cand_words <= set(p.split()) or set(p.split()) <= cand_words
                   for p in picked):
                continue
            picked.append(cand)
            if len(picked) >= top_n:
                break
        out[tid] = picked
    return out


def topic_centroids(labels: np.ndarray, embeddings: np.ndarray) -> Dict[int, np.ndarray]:
    out = {}
    for label in sorted(set(labels)):
        if label < 0:
            continue
        mask = labels == label
        c = embeddings[mask].mean(axis=0)
        c = c / max(np.linalg.norm(c), 1e-12)
        out[int(label)] = c.astype(np.float32)
    return out


def split_oversized_clusters(
    X_cluster: np.ndarray,
    labels: np.ndarray,
    max_size: int,
    min_children: int,
    min_samples: int,
    embeddings: Optional[np.ndarray] = None,
    max_child_similarity: float = 0.0,
) -> np.ndarray:
    """Re-cluster any cluster bigger than `max_size` into sub-clusters.

    Done here, on the raw HDBSCAN labels, so everything downstream -- centroids,
    labels, assignment, counting -- simply sees more and smaller clusters. A
    split is kept only if it actually yields several children; otherwise the
    original cluster stands, because one child is not a split.
    """
    if max_size <= 0:
        return labels
    labels = labels.copy()
    next_id = int(labels.max()) + 1 if (labels >= 0).any() else 0
    for cid in sorted({int(c) for c in labels if c >= 0}):
        idx = np.where(labels == cid)[0]
        if len(idx) <= max_size:
            continue
        # Ward with a fixed child count, not HDBSCAN. HDBSCAN grouped these
        # points because they form ONE density blob, so asking it for density
        # substructure inside them finds nothing. The goal here is not to
        # discover sub-density but to force tighter centroids, which are then
        # more selective about what assignment attracts to them.
        n_children = max(min_children, int(round(len(idx) / max_size)))
        n_children = min(n_children, len(idx) // 10)      # keep children >= ~10
        if n_children < min_children:
            continue
        child = AgglomerativeClustering(n_clusters=n_children, linkage="ward").fit_predict(X_cluster[idx])

        # P4. Refuse a split whose children are near-duplicates of each other in
        # the original embedding space. Without this the split pass and the
        # duplicate-merge pass fight: one cuts a topic in two, the other is
        # asked to glue it back, and the run's granularity depends on which
        # happened to run last.
        if max_child_similarity > 0 and embeddings is not None:
            cents = []
            for c in sorted({int(x) for x in child}):
                sel = idx[child == c]
                v = embeddings[sel].mean(axis=0)
                cents.append(v / max(np.linalg.norm(v), 1e-12))
            if len(cents) >= 2:
                C = np.vstack(cents).astype(np.float32)
                S = C @ C.T
                np.fill_diagonal(S, -1.0)
                if float(S.max()) >= max_child_similarity:
                    continue                      # children too alike; leave it whole

        remap = {c: next_id + i for i, c in enumerate(sorted({int(x) for x in child}))}
        next_id += len(remap)
        for local_i, c in enumerate(child):
            labels[idx[local_i]] = remap[int(c)]
    return labels


def merge_duplicate_topics(
    topics: List[Topic],
    threshold: float,
) -> Tuple[List[Topic], Dict[str, str]]:
    """Collapse same-brand topics whose centroids are near-identical.

    The same topic can be discovered twice inside one pool; 46 topics shared
    only 16 distinct labels in the baseline run, which also double-counts
    records in the daily series.
    """
    if not topics:
        return topics, {}
    order = sorted(range(len(topics)), key=lambda i: -topics[i].size)
    survivors: List[int] = []
    remap: Dict[str, str] = {}
    for i in order:
        t = topics[i]
        ci = np.asarray(t.centroid, dtype=np.float32)
        merged_into = None
        for j in survivors:
            o = topics[j]
            if o.brand != t.brand:
                continue
            cj = np.asarray(o.centroid, dtype=np.float32)
            if float(cosine_sim(ci, cj[None, :])[0]) >= threshold:
                merged_into = o
                break
        if merged_into is None:
            survivors.append(i)
        else:
            remap[t.topic_id] = merged_into.topic_id
            merged_into.size += t.size
    kept = [topics[i] for i in sorted(survivors)]
    return kept, remap


def make_topics(
    segments: pd.DataFrame,
    embeddings: np.ndarray,
    labels: np.ndarray,
    brand: str,
    prefix: str,
) -> Tuple[List[Topic], Dict[int, np.ndarray]]:
    topics: List[Topic] = []
    cents = topic_centroids(labels, embeddings)
    for cluster_id, centroid in cents.items():
        idx = np.where(labels == cluster_id)[0]
        recs = segments.iloc[idx]
        label = make_topic_label(recs["clean_text"].tolist())
        first_ts = pd.to_datetime(recs["event_time"], errors="coerce").min()
        last_ts = pd.to_datetime(recs["event_time"], errors="coerce").max()
        unique_records = int(recs["record_id"].nunique())
        topics.append(
            Topic(
                topic_id=f"{prefix}_{cluster_id}",
                brand=brand,
                label=label,
                size=unique_records,
                centroid=centroid.tolist(),
                created_at=None if pd.isna(first_ts) else first_ts.isoformat(),
                last_seen_at=None if pd.isna(last_ts) else last_ts.isoformat(),
            )
        )
    return topics, cents


def assign_to_topics(
    segments: pd.DataFrame,
    embeddings: np.ndarray,
    topics: List[Topic],
    similarity_threshold: float,
    multi_topic: bool = True,
    max_topics_per_segment: int = 2,
    brand_scoped: bool = False,
) -> pd.DataFrame:
    if not topics:
        result = segments[["record_id", "segment_id", "segment_index", "event_time", "brand", "clean_text"]].copy()
        result["topic_id"] = None
        result["similarity"] = np.nan
        result["assignment_type"] = "UNASSIGNED"
        return result

    topic_matrix = np.asarray([t.centroid for t in topics], dtype=np.float32)
    rows = []
    topic_ids = [t.topic_id for t in topics]

    # Brand scoping. Airlines all discuss delayed flights, so an unscoped
    # nearest-centroid search files a Delta record under an AmericanAir topic:
    # 54.7% of assignments leaked this way. The pooled buckets stay visible
    # because small brands' topics live there.
    allowed_by_brand: Dict[str, np.ndarray] = {}
    if brand_scoped:
        topic_brands = np.asarray([t.brand for t in topics])
        pooled = np.isin(topic_brands, ["__SMALL_BRANDS__", "GLOBAL"])
        for b in segments["brand"].astype(str).unique():
            mask = (topic_brands == b) | pooled
            if not mask.any():          # brand has no topics of its own yet
                mask = np.ones(len(topics), dtype=bool)
            allowed_by_brand[b] = np.where(mask)[0]

    for i in range(len(segments)):
        sims = cosine_sim(embeddings[i], topic_matrix)
        if brand_scoped:
            allowed = allowed_by_brand.get(str(segments.iloc[i]["brand"]))
            if allowed is not None and len(allowed) < len(topics):
                blocked = np.ones(len(topics), dtype=bool)
                blocked[allowed] = False
                sims = sims.copy()
                sims[blocked] = -1.0
        order = np.argsort(sims)[::-1]
        chosen = []
        for j in order:
            score = float(sims[j])
            if score < similarity_threshold:
                break
            chosen.append((topic_ids[j], score))
            if not multi_topic or len(chosen) >= max_topics_per_segment:
                break

        if not chosen:
            rows.append({
                "record_id": segments.iloc[i]["record_id"],
                "segment_id": segments.iloc[i]["segment_id"],
                "segment_index": segments.iloc[i]["segment_index"],
                "event_time": segments.iloc[i]["event_time"],
                "brand": segments.iloc[i]["brand"],
                "clean_text": segments.iloc[i]["clean_text"],
                "topic_id": None,
                "similarity": float(sims[order[0]]) if len(order) else np.nan,
                "assignment_type": "UNASSIGNED",
            })
        else:
            for topic_id, score in chosen:
                rows.append({
                    "record_id": segments.iloc[i]["record_id"],
                    "segment_id": segments.iloc[i]["segment_id"],
                    "segment_index": segments.iloc[i]["segment_index"],
                    "event_time": segments.iloc[i]["event_time"],
                    "brand": segments.iloc[i]["brand"],
                    "clean_text": segments.iloc[i]["clean_text"],
                    "topic_id": topic_id,
                    "similarity": score,
                    "assignment_type": "EXISTING",
                })

    return pd.DataFrame(rows)


def map_clusters_to_existing_topics(
    candidate_segments: pd.DataFrame,
    candidate_embeddings: np.ndarray,
    labels: np.ndarray,
    candidate_centroids: Dict[int, np.ndarray],
    existing_topics: List[Topic],
    new_topic_similarity: float,
    brand: str,
    next_topic_number_start: int,
    margin: float = 0.0,
) -> Tuple[List[Topic], Dict[int, str], pd.DataFrame]:
    existing_matrix = np.asarray([t.centroid for t in existing_topics], dtype=np.float32) if existing_topics else np.empty((0, candidate_embeddings.shape[1]))
    existing_ids = [t.topic_id for t in existing_topics]
    mapping: Dict[int, str] = {}
    new_topics: List[Topic] = []
    rows = []
    next_num = next_topic_number_start

    cids = list(candidate_centroids.keys())

    # Resolve matches one-to-one, strongest first. Without this two candidate
    # clusters can both claim the same existing topic id.
    best_for: Dict[int, Tuple[int, float]] = {}
    if existing_topics:
        sim_matrix = np.vstack([cosine_sim(candidate_centroids[cid], existing_matrix) for cid in cids])
        taken_existing: set = set()
        proposals = sorted(
            ((float(sim_matrix[r, c]), r, c) for r in range(sim_matrix.shape[0]) for c in range(sim_matrix.shape[1])),
            reverse=True,
        )
        claimed_rows: set = set()
        # A cluster sitting almost equally close to two existing topics is not
        # evidence of belonging to either; with a dense topic registry every
        # candidate clears a bare threshold against something, which is how
        # emerging-topic detection silently stops producing anything.
        runner_up = np.full(sim_matrix.shape[0], -1.0, dtype=np.float32)
        if margin > 0 and sim_matrix.shape[1] >= 2:
            part = np.sort(sim_matrix, axis=1)
            runner_up = part[:, -2]

        for score, r, c in proposals:
            if score < new_topic_similarity:
                break
            if r in claimed_rows or c in taken_existing:
                continue
            if margin > 0 and (score - runner_up[r]) < margin:
                continue
            best_for[cids[r]] = (c, score)
            claimed_rows.add(r)
            taken_existing.add(c)
        for r, cid in enumerate(cids):
            if cid not in best_for:
                c = int(np.argmax(sim_matrix[r]))
                best_for[cid] = (-1, float(sim_matrix[r, c]))
    else:
        for cid in cids:
            best_for[cid] = (-1, -1.0)

    for cid in cids:
        centroid = candidate_centroids[cid]
        best_idx, best_score = best_for[cid]

        cluster_rows = candidate_segments.iloc[np.where(labels == cid)[0]]
        label = make_topic_label(cluster_rows["clean_text"].tolist())
        unique_records = int(cluster_rows["record_id"].nunique())

        if best_idx >= 0:
            topic_id = existing_ids[best_idx]
            mapping[cid] = topic_id
            rows.append({
                "candidate_cluster": cid,
                "decision": "MATCH_EXISTING",
                "topic_id": topic_id,
                "similarity_to_existing": best_score,
                "candidate_label": label,
                "unique_records": unique_records,
            })
        else:
            topic_id = f"T_NEW_{next_num}"
            next_num += 1
            mapping[cid] = topic_id
            new_topics.append(
                Topic(
                    topic_id=topic_id,
                    brand=brand,
                    label=label,
                    size=unique_records,
                    centroid=centroid.tolist(),
                    created_at=str(pd.to_datetime(cluster_rows["event_time"], errors="coerce").min()),
                    last_seen_at=str(pd.to_datetime(cluster_rows["event_time"], errors="coerce").max()),
                    status="EMERGING",
                )
            )
            rows.append({
                "candidate_cluster": cid,
                "decision": "NEW_TOPIC",
                "topic_id": topic_id,
                "similarity_to_existing": best_score,
                "candidate_label": label,
                "unique_records": unique_records,
            })

    return new_topics, mapping, pd.DataFrame(rows)


BUFFER_COLS = ["record_id", "segment_id", "segment_index", "event_time", "brand", "clean_text",
               "author_id"]


def _empty_buffer(dim: int) -> Tuple[pd.DataFrame, np.ndarray]:
    return pd.DataFrame(columns=BUFFER_COLS), np.empty((0, dim), dtype=np.float32)


def load_unassigned_buffer(path: Optional[str], expected_dim: int) -> Tuple[pd.DataFrame, np.ndarray]:
    """Load segments that no topic claimed on an earlier run.

    Embeddings from a different model live in a different space, so a width
    mismatch discards the buffer instead of silently mixing the two.
    """
    if not path:
        return _empty_buffer(expected_dim)
    d = Path(path)
    meta_p, emb_p = d / "meta.csv", d / "embeddings.npy"
    if not (meta_p.exists() and emb_p.exists()):
        return _empty_buffer(expected_dim)
    meta = pd.read_csv(meta_p, dtype={"record_id": str, "segment_id": str})
    emb = np.load(emb_p).astype(np.float32)
    if emb.ndim != 2 or len(meta) != len(emb) or emb.shape[1] != expected_dim:
        print(f"Buffer at {d} is incompatible (rows={len(meta)}/{len(emb)}, "
              f"dim={emb.shape[1] if emb.ndim == 2 else 'n/a'} vs {expected_dim}); discarding it.")
        return _empty_buffer(expected_dim)
    for c in BUFFER_COLS:
        if c not in meta.columns:
            print(f"Buffer at {d} is missing column {c!r}; discarding it.")
            return _empty_buffer(expected_dim)
    return meta[BUFFER_COLS], emb


def save_unassigned_buffer(path: Optional[str], meta: pd.DataFrame, emb: np.ndarray) -> None:
    if not path:
        return
    d = Path(path)
    d.mkdir(parents=True, exist_ok=True)
    meta[BUFFER_COLS].to_csv(d / "meta.csv", index=False)
    np.save(d / "embeddings.npy", emb.astype(np.float32))


def prune_buffer(meta: pd.DataFrame, emb: np.ndarray, latest: pd.Timestamp, max_age_days: int):
    """Drop buffered segments past the age limit, so a months-old complaint
    cannot keep propping up a supposedly new topic."""
    if meta.empty or pd.isna(latest):
        return meta, emb
    ts = pd.to_datetime(meta["event_time"], errors="coerce", utc=True, format="mixed")
    keep = (ts >= latest - pd.Timedelta(max_age_days, "D")).to_numpy()
    return meta.loc[keep].reset_index(drop=True), emb[keep]


def topics_in_scope(topics: List[Topic], brand: str) -> List[Topic]:
    """Candidate clusters should match their own brand's topics first; the
    pooled buckets stay visible because small brands live there."""
    scoped = [t for t in topics if t.brand in (brand, "__SMALL_BRANDS__", "GLOBAL")]
    return scoped or topics


def find_micro_clusters(
    meta: pd.DataFrame,
    emb: np.ndarray,
    distance: float,
    min_size: int,
    min_authors: int,
) -> List[np.ndarray]:
    """Very small, very tight, single-brand groups from the leftover pool.

    Diagnostics on this corpus showed 97.3% of leftover segments are singletons
    or pairs at a tight distance, so this is deliberately narrow: it is looking
    for the ~1% that are real, not trying to raise coverage. Three guards keep
    the mush out -- a tight distance ceiling (groups become cross-brand noise
    beyond ~0.8), single brand (72% of tight groups spanned brands), and
    distinct authors (one ranting user and one duplicated tweet each produced a
    fake group).
    """
    out: List[np.ndarray] = []
    if len(meta) < min_size:
        return out
    for brand, g in meta.groupby("brand", sort=False):
        idx = g.index.to_numpy()
        if len(idx) < min_size:
            continue
        X = emb[idx]
        if len(idx) > 2:
            X = reduce_dimensions(X, min(50, X.shape[1]), 42)[0]
        lab = AgglomerativeClustering(n_clusters=None, distance_threshold=distance,
                                      linkage="average", metric="euclidean").fit_predict(X)
        for c in set(lab):
            sel = idx[lab == c]
            if len(sel) < min_size:
                continue
            authors = meta.loc[sel, "author_id"].astype(str).nunique() if "author_id" in meta else len(sel)
            if authors < min_authors:
                continue
            out.append(sel)
    return out


def unique_record_topic_counts(assignments: pd.DataFrame) -> pd.DataFrame:
    x = assignments[assignments["topic_id"].notna()].copy()
    x["date"] = pd.to_datetime(x["event_time"], errors="coerce").dt.date
    # Crucial: one record contributes at most once to a topic/day.
    x = x.drop_duplicates(subset=["record_id", "topic_id", "date"])
    grouped = (
        x.groupby(["date", "topic_id"], as_index=False)
        .agg(record_count=("record_id", "nunique"))
    )
    return grouped


def build_complete_timeseries(counts: pd.DataFrame, topics: List[Topic]) -> pd.DataFrame:
    if counts.empty:
        return counts
    all_dates = pd.date_range(counts["date"].min(), counts["date"].max(), freq="D").date
    topic_ids = [t.topic_id for t in topics]
    grid = pd.MultiIndex.from_product([all_dates, topic_ids], names=["date", "topic_id"]).to_frame(index=False)
    out = grid.merge(counts, on=["date", "topic_id"], how="left")
    out["record_count"] = out["record_count"].fillna(0).astype(int)
    return out


def robust_minmax(s: pd.Series) -> pd.Series:
    if s.empty:
        return s
    lo = s.quantile(0.05)
    hi = s.quantile(0.95)
    if hi <= lo:
        return pd.Series(np.zeros(len(s)), index=s.index)
    return ((s.clip(lo, hi) - lo) / (hi - lo)).fillna(0.0)


def add_hot_scores(ts: pd.DataFrame, recent_days: int = 7, baseline_days: int = 28) -> pd.DataFrame:
    if ts.empty:
        return ts
    ts = ts.copy()
    ts["date"] = pd.to_datetime(ts["date"])
    latest = ts["date"].max()
    recent_start = latest - pd.Timedelta(recent_days - 1, "D")
    baseline_start = latest - pd.Timedelta(baseline_days - 1, "D")
    recent = ts[ts["date"] >= recent_start]
    baseline = ts[(ts["date"] < recent_start) & (ts["date"] >= baseline_start)]

    recent_agg = recent.groupby("topic_id")["record_count"].sum().rename("recent_volume")
    baseline_days_effective = max(1, baseline["date"].nunique())
    baseline_agg = (baseline.groupby("topic_id")["record_count"].sum() / baseline_days_effective * recent_days).rename("baseline_volume")

    # Velocity: compare newest 2 days to older recent days.
    last2 = ts[ts["date"] > latest - pd.Timedelta(2, "D")].groupby("topic_id")["record_count"].sum()
    prev5 = ts[(ts["date"] <= latest - pd.Timedelta(2, "D")) & (ts["date"] >= recent_start)].groupby("topic_id")["record_count"].sum()
    velocity = ((last2 + 1) / (prev5 + 1)).rename("velocity_ratio")

    metrics = pd.concat([recent_agg, baseline_agg, velocity], axis=1).fillna(0.0)
    metrics["growth_rate"] = (metrics["recent_volume"] + 1) / (metrics["baseline_volume"] + 1) - 1
    metrics["anomaly_ratio"] = metrics["recent_volume"] / (metrics["baseline_volume"] + 1)

    # Persistence: fraction of recent days with non-zero activity.
    presence = recent.assign(active=(recent["record_count"] > 0)).groupby("topic_id")["active"].mean().rename("persistence")
    metrics = metrics.join(presence, how="left").fillna(0.0)

    metrics["volume_score"] = robust_minmax(metrics["recent_volume"])
    metrics["growth_score"] = robust_minmax(metrics["growth_rate"].clip(lower=0))
    metrics["velocity_score"] = robust_minmax(np.log1p(metrics["velocity_ratio"].clip(lower=0)))
    metrics["anomaly_score"] = robust_minmax(np.log1p(metrics["anomaly_ratio"].clip(lower=0)))
    metrics["recency_score"] = robust_minmax(metrics["recent_volume"])

    metrics["hot_score"] = (
        0.25 * metrics["volume_score"]
        + 0.25 * metrics["growth_score"]
        + 0.20 * metrics["velocity_score"]
        + 0.15 * metrics["anomaly_score"]
        + 0.10 * metrics["recency_score"]
        + 0.05 * metrics["persistence"]
    )

    # Classification is intentionally simple and explainable for the prototype.
    def status(row: pd.Series) -> str:
        if row["recent_volume"] <= 2:
            return "LOW_EVIDENCE"
        if row["hot_score"] >= 0.75 and row["growth_rate"] > 0.50:
            return "HOT"
        if row["hot_score"] >= 0.55 and row["growth_rate"] > 0.20:
            return "TRENDING"
        if row["growth_rate"] > 0.05:
            return "GROWING"
        if row["growth_rate"] < -0.20:
            return "DECLINING"
        return "STABLE"

    metrics["status"] = metrics.apply(status, axis=1)
    return metrics.reset_index()


def acknowledgement_member_ratio(texts: Sequence[str]) -> float:
    """Share of members that are short pure acknowledgements.

    Measured on members rather than labels. A c-TF-IDF label hides this
    completely: T371 was labelled "comcastoutage / fuck / fucking" while its
    members were "Exactly", "Sent", "Okay thanks".
    """
    if not len(texts):
        return 0.0
    stop = build_content_stopset()
    n = 0
    for t in texts:
        # "Nothing left once stopwords and acknowledgement words are removed" is
        # the right test. Requiring every token to be an ACK word is too strict:
        # "Thank you! Done." fails on "you", yet it is exactly the member these
        # buckets are made of. Emoji- and punctuation-only members score 0 too,
        # which is correct -- they carry no topic either.
        if content_word_count(t, stop) == 0:
            n += 1
    return n / len(texts)


def flag_residual_clusters(
    topics: List[Topic],
    member_texts: Dict[str, List[str]],
    ack_ratio_threshold: float = 0.40,
) -> int:
    """Mark conversational grab-bags for what they are.

    These are the by-product of merging small clusters upward: the short
    replies, thanks and emoji that used to form their own tiny clusters get
    absorbed into large, temporally steady buckets -- and steady volume is
    exactly what HotScore rewards, which is how all six HOT topics ended up
    being chatter. Flagged, not deleted: a "chatter" bucket is legitimate to
    show, never to escalate.
    """
    n = 0
    for t in topics:
        ratio = acknowledgement_member_ratio(member_texts.get(t.topic_id, []))
        t.ack_member_ratio = round(float(ratio), 4)
        reasons = []
        if ratio >= ack_ratio_threshold:
            reasons.append(f"ack_members={ratio:.0%}")
        if t.coherence is not None and t.coherence < 0:
            reasons.append(f"coherence={t.coherence:.3f}")
        t.residual_cluster = bool(reasons)
        t.residual_reason = "; ".join(reasons) if reasons else None
        n += int(bool(reasons))
    return n


def gate_alert_surface(
    summary: pd.DataFrame,
    min_coherence: Optional[float],
    min_size: int,
) -> pd.DataFrame:
    """Demote HOT/TRENDING topics that fail a quality bar.

    HotScore rewards volume, steadiness and growth -- all of which a
    conversational grab-bag has in abundance. An alert is a claim on someone's
    attention, so it earns one extra condition the other statuses do not: the
    topic has to be coherent and substantial. Nothing is hidden; the row, the
    score and the reason all survive, the topic simply stops being an
    escalation.
    """
    if summary.empty:
        return summary
    summary = summary.copy()
    fails = pd.Series(False, index=summary.index)
    reason = pd.Series("", index=summary.index)

    if min_coherence is not None and "coherence" in summary.columns:
        bad = summary["coherence"].fillna(-1.0) < min_coherence
        fails |= bad
        reason = reason.mask(bad & reason.eq(""), "low_coherence")
    if min_size > 0 and "size" in summary.columns:
        small = summary["size"].fillna(0) < min_size
        fails |= small
        reason = reason.mask(small & reason.eq(""), "below_min_size")
    if "residual_cluster" in summary.columns:
        resid = summary["residual_cluster"].fillna(False).astype(bool)
        fails |= resid
        reason = reason.mask(resid & reason.eq(""), "residual_cluster")

    demote = summary["status"].isin(["HOT", "TRENDING"]) & fails
    summary["alert_suppressed"] = demote
    summary["alert_suppressed_reason"] = reason.where(demote, None)
    summary.loc[demote, "status"] = "GROWING"
    return summary


# P3. The independent check travels with the pipeline instead of living in a
# separate script that can be forgotten. These events are verified to have
# occurred inside the Oct-Nov 2017 window; matching is keyword-based on purpose,
# because grading topics with the same encoder that built them would make the
# benchmark partly self-confirming.
VERIFIED_EVENTS = [
    {"name": "McDonald's Szechuan sauce shortage", "brand": "McDonalds",
     "onset": "2017-10-07", "all_of": [["szechuan", "sichuan"]]},
    {"name": "iOS 11 complaint wave", "brand": "AppleSupport",
     "onset": "2017-10-01", "all_of": [["ios 11", "ios11", "ios 11.0", "ios 11.1"]]},
    {"name": "Windows 10 1709 driver regression", "brand": "MicrosoftHelps",
     "onset": "2017-10-17", "all_of": [["1709", "fall creators", "creators update", "driver", "radeon"]]},
    {"name": "Amazon Prime membership / billing", "brand": "AmazonHelp",
     "onset": "2017-10-01", "all_of": [["prime"], ["charge", "charged", "billing", "member", "membership", "pay"]]},
    {"name": "Virgin Trains December strike", "brand": "VirginTrains",
     "onset": "2017-11-20", "all_of": [["strike", "rmt", "industrial action"]]},
    {"name": "Black Friday / Cyber Monday", "brand": None,
     "onset": "2017-11-24", "all_of": [["black friday", "cyber monday", "blackfriday", "cybermonday"]]},
    {"name": "Uber driver cancellation / fare disputes", "brand": "Uber_Support",
     "onset": "2017-10-01", "all_of": [["driver"], ["cancel", "cancelled", "cancelling", "fare", "charge", "charged"]]},
    {"name": "Comcast/Xfinity outage", "brand": "comcastcares",
     "onset": "2017-10-01", "all_of": [["outage", "no internet", "internet down", "service down"]]},
    {"name": "Spotify iOS/playback failures", "brand": "SpotifyCares",
     "onset": "2017-10-01", "all_of": [["playback", "won't play", "wont play", "not playing", "crashing", "crash"]]},
    {"name": "Southwest / American flight delays", "brand": None,
     "onset": "2017-10-01", "all_of": [["delayed", "delay"], ["flight", "flights"]]},
]


def event_keyword_hit(text: str, groups) -> bool:
    t = str(text).lower()
    return all(any(k in t for k in group) for group in groups)


def compute_event_recall(assigned: pd.DataFrame, topics: List[Topic]) -> pd.DataFrame:
    """Per verified event: did we surface it, in how few topics, and how fast?

    Reports recall_in_best_topic and topics_spanned alongside precision, because
    precision alone rises automatically as topics shrink -- which is exactly how
    a fragmenting configuration can look like an improvement.
    """
    if assigned.empty:
        return pd.DataFrame()
    a = assigned[assigned["topic_id"].notna()].copy()
    if "clean_text" not in a.columns:
        return pd.DataFrame()
    a["_t"] = pd.to_datetime(a["event_time"], errors="coerce", utc=True)
    sizes = {t.topic_id: t.size for t in topics}
    labels = {t.topic_id: t.label for t in topics}
    junk = {t.topic_id: bool(t.is_junk) for t in topics}
    resid = {t.topic_id: bool(t.residual_cluster) for t in topics}

    rows = []
    for ev in VERIFIED_EVENTS:
        m = a["clean_text"].astype(str).map(lambda x: event_keyword_hit(x, ev["all_of"]))
        if ev.get("brand"):
            m &= a["brand"].astype(str).eq(ev["brand"])
        hit = a[m]
        if hit.empty:
            rows.append({"event": ev["name"], "brand": ev.get("brand"), "found": False,
                         "matched_records": 0, "topic_id": None, "topic_label": None,
                         "topic_size": 0, "recall_in_best_topic": np.nan,
                         "precision_in_topic": np.nan, "topics_spanned": 0,
                         "topics_for_80pct": np.nan, "onset": ev["onset"],
                         "first_seen": None, "lag_days": np.nan,
                         "is_junk": None, "residual_cluster": None})
            continue
        per = hit.groupby("topic_id")["record_id"].nunique().sort_values(ascending=False)
        total = int(hit["record_id"].nunique())
        best = per.index[0]
        best_n = int(per.iloc[0])
        cum = per.cumsum() / total
        first_seen = hit.loc[hit["topic_id"] == best, "_t"].min()
        onset = pd.Timestamp(ev["onset"], tz="UTC")
        rows.append({
            "event": ev["name"], "brand": ev.get("brand"), "found": True,
            "matched_records": total,
            "topic_id": best, "topic_label": labels.get(best),
            "topic_size": int(sizes.get(best, 0)),
            "recall_in_best_topic": round(best_n / total, 4),
            "precision_in_topic": round(best_n / max(1, sizes.get(best, 0)), 4),
            "topics_spanned": int(len(per)),
            "topics_for_80pct": int((cum < 0.8).sum() + 1),
            "onset": ev["onset"],
            "first_seen": None if pd.isna(first_seen) else first_seen.isoformat(),
            "lag_days": (np.nan if pd.isna(first_seen)
                         else round((first_seen - onset).total_seconds() / 86400, 2)),
            "is_junk": junk.get(best), "residual_cluster": resid.get(best),
        })
    return pd.DataFrame(rows)


def save_json(path: Path, obj) -> None:
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def plot_hot_topics(ts: pd.DataFrame, topics: List[Topic], out_dir: Path, top_n: int = 8) -> None:
    if plt is None or ts.empty:
        return
    summary = add_hot_scores(ts)
    top_ids = summary.sort_values("hot_score", ascending=False)["topic_id"].head(top_n).tolist()
    topic_map = {t.topic_id: t.label for t in topics}

    for tid in top_ids:
        sub = ts[ts["topic_id"] == tid].sort_values("date")
        if sub.empty:
            continue
        plt.figure(figsize=(10, 4))
        plt.plot(sub["date"], sub["record_count"], marker="o")
        plt.title(f"{topic_map.get(tid, tid)} ({tid})")
        plt.xlabel("Date")
        plt.ylabel("Distinct records")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(out_dir / f"{tid}_timeseries.png", dpi=140)
        plt.close()


def load_data(path: str, text_col: str, time_col: str, brand_col: str, id_col: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing = [c for c in [text_col, time_col, brand_col, id_col] if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}. Found: {df.columns.tolist()}")
    df = df.copy()
    df[time_col] = pd.to_datetime(df[time_col], errors="coerce", utc=True)
    df = df[df[time_col].notna()].copy()
    df[brand_col] = df[brand_col].fillna("unknown").astype(str)
    df[text_col] = df[text_col].fillna("").astype(str)
    df[id_col] = df[id_col].astype(str)
    return df.sort_values(time_col).reset_index(drop=True)



def load_previous_topics(path: Optional[str]) -> List[Topic]:
    if not path:
        return []
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    topics = []
    for item in payload:
        topics.append(Topic(**item))
    return topics


def stabilize_topics(
    new_topics: List[Topic],
    previous_topics: List[Topic],
    threshold: float,
    carry_forward: bool = False,
    max_age_days: int = 0,
    window_end: Optional[pd.Timestamp] = None,
) -> Tuple[List[Topic], pd.DataFrame]:
    """Match newly discovered clusters to previous topics by original-space centroid similarity.

    Globally optimal one-to-one matching, not greedy. The distinction matters:
    greedy proposed only each cluster's single best previous topic, so when two
    clusters wanted the same old ID the loser was declared brand new even if it
    had a perfectly good second choice still free. That invented EMERGING topics
    out of nothing more than tie-break order. Solving the whole assignment at
    once (Hungarian) lets the loser take its next-best free topic instead, and
    maximises total similarity over the matching rather than over one row.

    Matches below `threshold` are rejected afterwards, so the optimiser can
    never buy a good pairing by accepting a meaningless one elsewhere.
    """
    if not new_topics:
        return [], pd.DataFrame()
    if not previous_topics:
        return new_topics, pd.DataFrame()

    prev_matrix = np.asarray([t.centroid for t in previous_topics], dtype=np.float32)
    decisions = []
    max_num = 0
    for t in previous_topics:
        m = re.search(r"(\d+)$", t.topic_id)
        if m:
            max_num = max(max_num, int(m.group(1)))
    next_num = max_num + 1

    # Full new x previous similarity matrix in the original embedding space.
    sim_matrix = np.vstack([
        cosine_sim(np.asarray(t.centroid, dtype=np.float32), prev_matrix)
        for t in new_topics
    ])

    matched: Dict[int, int] = {}
    try:
        from scipy.optimize import linear_sum_assignment

        # Only rows/cols that could clear the threshold are worth matching;
        # restricting first keeps the rectangular solve small and stops a
        # sub-threshold pair from occupying a previous topic another cluster needs.
        rows = np.where(sim_matrix.max(axis=1) >= threshold)[0]
        cols = np.where(sim_matrix.max(axis=0) >= threshold)[0]
        if len(rows) and len(cols):
            sub = sim_matrix[np.ix_(rows, cols)]
            r_idx, c_idx = linear_sum_assignment(sub, maximize=True)
            for r, c in zip(r_idx, c_idx):
                if sub[r, c] >= threshold:
                    matched[int(rows[r])] = int(cols[c])
    except ImportError:
        # Same greedy fallback as before, but scanning every free column rather
        # than only the argmax, so a taken best choice does not force a new ID.
        used = set()
        order = np.argsort(-sim_matrix.max(axis=1))
        for i in order:
            row = sim_matrix[i]
            for j in np.argsort(-row):
                if row[j] < threshold:
                    break
                if int(j) not in used:
                    matched[int(i)] = int(j)
                    used.add(int(j))
                    break

    for i, t in enumerate(new_topics):
        j = matched.get(i)
        if j is not None:
            old = previous_topics[j]
            t.topic_id = old.topic_id
            t.created_at = old.created_at or t.created_at
            t.status = "ACTIVE"
            decisions.append({
                "new_label": t.label,
                "previous_topic_id": old.topic_id,
                "similarity": float(sim_matrix[i, j]),
                "best_available_similarity": float(sim_matrix[i].max()),
                "decision": "MATCH_PREVIOUS",
            })
        else:
            t.topic_id = f"T{next_num}"
            next_num += 1
            t.status = "EMERGING"
            decisions.append({
                "new_label": t.label,
                "previous_topic_id": None,
                "similarity": float(sim_matrix[i].max()),
                "best_available_similarity": float(sim_matrix[i].max()),
                "decision": "NEW_STABLE_ID",
            })

    # Carry the registry forward.
    #
    # Without this the topic list is rebuilt from scratch every window and
    # anything not re-discovered is simply gone -- across a four-window replay
    # only 17% of the original topics survived, not because they ended but
    # because no single window re-derived them. A topic with no traffic this
    # window has not ceased to exist; it is dormant. Keeping it (with its
    # centroid) means later traffic revives the original ID instead of minting
    # a duplicate, which is the whole point of a stable registry.
    if carry_forward:
        matched_prev = set(matched.values())
        cutoff = None
        if max_age_days and max_age_days > 0 and window_end is not None:
            cutoff = window_end - pd.Timedelta(days=max_age_days)
        carried = 0
        for j, old_topic in enumerate(previous_topics):
            if j in matched_prev:
                continue
            last = pd.to_datetime(old_topic.last_seen_at, errors="coerce", utc=True)
            if cutoff is not None and not pd.isna(last) and last < cutoff:
                decisions.append({
                    "new_label": old_topic.label, "previous_topic_id": old_topic.topic_id,
                    "similarity": np.nan, "best_available_similarity": np.nan,
                    "decision": "RETIRED_STALE",
                })
                continue
            revived = Topic(**{**asdict(old_topic)})
            revived.status = "DORMANT"
            revived.size = 0                      # recomputed from this window's assignments
            new_topics.append(revived)
            carried += 1
            decisions.append({
                "new_label": old_topic.label, "previous_topic_id": old_topic.topic_id,
                "similarity": np.nan, "best_available_similarity": np.nan,
                "decision": "CARRIED_FORWARD",
            })
        if carried:
            print(f"Registry carry-forward: {carried} previous topics kept as DORMANT")

    return new_topics, pd.DataFrame(decisions)


def run(args: argparse.Namespace) -> int:
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    timer = StageTimer()
    previous_topics = load_previous_topics(args.previous_topics)
    df = load_data(args.csv, args.text_col, args.time_col, args.brand_col, args.id_col)
    # Standard internal names so the rest of the pipeline stays simple.
    df = df.rename(columns={
        args.time_col: "event_time",
        args.brand_col: "brand",
        args.text_col: "text",
        args.id_col: "tweet_id",
    })

    print(f"Loaded {len(df):,} records from {args.csv}")
    print(f"Brands: {df['brand'].nunique():,}")
    print(f"Date range: {df['event_time'].min()} -> {df['event_time'].max()}")

    segments = segment_records(
        df,
        id_col="tweet_id",
        text_col="text",
        max_chars=args.max_segment_chars,
        overlap_chars=60,
        max_segments_per_record=args.max_segments_per_record,
    )
    print(f"Created {len(segments):,} semantic units from {segments['record_id'].nunique():,} records.")
    print(f"Segment expansion factor: {len(segments)/max(1, len(df)):.2f}x")

    content_stop = build_content_stopset()
    if args.min_content_words > 0:
        cw = segments["clean_text"].map(lambda t: content_word_count(t, content_stop))
        keep = cw >= args.min_content_words
        dropped = int((~keep).sum())
        segments = segments[keep].reset_index(drop=True)
        print(f"Layer 2 (min-content-words={args.min_content_words}): dropped {dropped:,} segments "
              f"({dropped/max(1, dropped+len(segments)):.2%}); {len(segments):,} remain")
        if segments.empty:
            raise ValueError("Layer 2 removed every segment; lower --min-content-words.")
    timer.mark("load + segment", f"{len(segments):,} segments")

    _device = resolve_device(getattr(args, "device", None))
    _batch_size = getattr(args, "batch_size", None) or default_batch_size(_device)
    _fp16 = bool(getattr(args, "fp16", False))
    print(f"Embedding device: {describe_device(_device)} "
          f"(batch-size={_batch_size}, fp16={_fp16})")

    _texts = segments["clean_text"].tolist()
    _key = embedding_cache_key(args.model, _texts, "fp16" if _fp16 else "fp32")
    embeddings = load_cached_embeddings(args.embed_cache, _key)
    if embeddings is not None and len(embeddings) == len(_texts):
        model = load_embedding_model(args.model, _device, _fp16)
        timer.mark("model load", f"{args.model.split('/')[-1]} on {_device}")
        timer.mark("embedding (cached)", f"reused {_key}, dim={embeddings.shape[1]}")
    else:
        model = load_embedding_model(args.model, _device, _fp16)
        timer.mark("model load", f"{args.model.split('/')[-1]} on {_device}")
        _emb_t0 = time.perf_counter()
        embeddings = build_embeddings(model, _texts, batch_size=_batch_size)
        _emb_dt = max(1e-9, time.perf_counter() - _emb_t0)
        save_cached_embeddings(args.embed_cache, _key, embeddings)
        timer.mark("embedding", f"{len(segments)/_emb_dt:.1f} seg/s, dim={embeddings.shape[1]}")

    if args.ack_similarity > 0:
        ack = acknowledgement_similarity(model, embeddings)
        keep = ack <= args.ack_similarity
        dropped = int((~keep).sum())
        pd.DataFrame({
            "segment_id": segments.loc[~keep, "segment_id"].values,
            "clean_text": segments.loc[~keep, "clean_text"].values,
            "ack_similarity": ack[~keep],
        }).to_csv(out_dir / "layer1_dropped.csv", index=False)
        segments = segments[keep].reset_index(drop=True)
        embeddings = embeddings[keep]
        print(f"Layer 1 (ack-similarity>{args.ack_similarity}): excluded {dropped:,} segments "
              f"({dropped/max(1, dropped+len(segments)):.2%}); {len(segments):,} remain")
        if segments.empty:
            raise ValueError("Layer 1 removed every segment; raise --ack-similarity.")
        timer.mark("layer 1 filter", f"{dropped:,} excluded")

    # Split by time to emulate real production behavior:
    # older history creates topics; latest period arrives incrementally.
    if args.test_fraction > 0 and 0 < args.test_fraction < 0.5:
        cutoff_idx = int(math.floor(len(df) * (1.0 - args.test_fraction)))
        cutoff_time = df.iloc[cutoff_idx]["event_time"]
        train_mask = segments["event_time"] < cutoff_time
        test_mask = ~train_mask
    else:
        cutoff_time = None
        train_mask = np.ones(len(segments), dtype=bool)
        test_mask = np.zeros(len(segments), dtype=bool)

    train_segments = segments[train_mask].reset_index(drop=True)
    train_embeddings = embeddings[train_mask]
    recent_segments = segments[test_mask].reset_index(drop=True)
    recent_embeddings = embeddings[test_mask]

    print(f"Discovery window: {len(train_segments):,} segments / {train_segments['record_id'].nunique():,} records")
    if cutoff_time is not None:
        print(f"Incremental holdout starts at: {cutoff_time}")

    # One fitted UMAP for the whole run, built from the discovery window only so
    # the incremental holdout never leaks into the manifold. Every pool below
    # transforms through this; nothing refits per pool.
    umap_projector: Optional[UmapProjector] = None
    if args.reducer == "umap" and not args.umap_refit_per_pool:
        umap_projector = get_or_fit_umap(train_segments, train_embeddings, args, timer)

        # P6. Act on the drift signal instead of only recording it. A frozen
        # manifold is trustworthy exactly as long as incoming data still looks
        # like its reference set; past that point every projection is an
        # extrapolation, which shows up downstream as records that match no
        # topic. Measured here, before anything is clustered, so the refit can
        # still happen in the same run.
        if args.umap_drift_threshold > 0 and umap_projector.models:
            d = umap_projector._drift(umap_projector._key(None), train_embeddings)
            if d is not None:
                print(f"UMAP drift check: {d:.4f} against threshold {args.umap_drift_threshold}")
                if d > args.umap_drift_threshold:
                    print(f"  drift exceeds threshold -- refitting the manifold on current data")
                    _forced = argparse.Namespace(**vars(args))
                    _forced.umap_refit = True
                    umap_projector = get_or_fit_umap(train_segments, train_embeddings, _forced, timer)
                    umap_projector.meta["refit_trigger"] = {
                        "reason": "drift_above_threshold",
                        "measured": round(float(d), 5),
                        "threshold": float(args.umap_drift_threshold),
                    }
                    umap_projector.drift_rows.clear()

    # Prefer per-brand discovery when a brand has enough history. Small brands are grouped globally.
    topics: List[Topic] = []
    all_discovery_assignment_frames = []
    next_global_topic_num = 1
    grid_results = []

    pca_widths = [int(x) for x in str(args.pca_candidates).split(",") if x.strip()]
    frozen_pca: Dict[str, int] = {}
    if args.pca_config:
        cfg_p = Path(args.pca_config)
        if cfg_p.exists():
            frozen_pca = {str(k): int(v) for k, v in json.loads(cfg_p.read_text()).items()}
            print(f"PCA widths loaded from {cfg_p} ({len(frozen_pca)} entries)")
        else:
            print(f"PCA config {cfg_p} not found; falling back to --pca-components {args.pca_components}")
    chosen_pca: Dict[str, int] = {}
    pca_search_frames = []

    enough_brand = train_segments.groupby("brand")["record_id"].nunique().to_dict()
    brand_groups = []
    small_idx = []
    for brand, n in enough_brand.items():
        idx = train_segments.index[train_segments["brand"] == brand].to_numpy()
        if n >= args.brand_min_records:
            brand_groups.append((brand, idx))
        else:
            small_idx.extend(idx.tolist())
    if small_idx:
        brand_groups.append(("__SMALL_BRANDS__", np.asarray(small_idx, dtype=int)))

    for brand_key, idx in brand_groups:
        if len(idx) < max(args.min_cluster_size * 2, 50):
            continue
        X = train_embeddings[idx]
        if args.reducer != "pca":
            X_cluster = apply_reducer(X, args, umap_projector, brand_key)
            pool_pca = args.umap_components if args.reducer == "umap" else 0
        elif args.no_pca:
            X_cluster = X
            pool_pca = 0
        else:
            # Precedence: frozen config, then tuning, then the flat default.
            if brand_key in frozen_pca:
                pool_pca = frozen_pca[brand_key]
            elif args.tune_pca:
                pool_pca, search = choose_pca_width(
                    X, pca_widths, args.min_cluster_size, args.min_samples, args.max_cluster_share)
                search["pool"] = brand_key
                pca_search_frames.append(search)
                kept = search[~search["rejected"]]
                print(f"  [tune-pca] {brand_key:20s} chose {pool_pca:>4} "
                      f"(considered {list(search['pca_components'])}, "
                      f"rejected {list(search.loc[search['rejected'], 'pca_components'])})")
            else:
                pool_pca = args.pca_components
            X_cluster, _ = reduce_dimensions(X, min(pool_pca, X.shape[1]), 42)
        chosen_pca[brand_key] = int(pool_pca)
        if args.grid_search:
            candidates = [(10, 5), (15, 5), (20, 8), (25, 8), (30, 10), (40, 10)]
            best_mcs, best_ms, grid = choose_hdbscan_params(X_cluster, candidates)
            grid["brand"] = brand_key
            grid_results.append(grid)
            mcs, ms = best_mcs, best_ms
        else:
            mcs, ms = args.min_cluster_size, args.min_samples

        labels = cluster_embeddings(X_cluster, mcs, ms, "eom",
                                    args.cluster_selection_epsilon, args.max_cluster_size)
        if args.split_max_size > 0:
            before = len({int(c) for c in labels if c >= 0})
            labels = split_oversized_clusters(X_cluster, labels, args.split_max_size,
                                              args.split_min_children, ms,
                                              embeddings=X,
                                              max_child_similarity=args.split_max_child_similarity)
            after = len({int(c) for c in labels if c >= 0})
            if after != before:
                print(f"  [split] {brand_key}: {before} -> {after} clusters "
                      f"(max-size {args.split_max_size})")
        q = cluster_quality(X_cluster, labels)
        share = max_cluster_share(labels)
        print(f"Discovery pool={brand_key:20s} n={len(idx):5d} pca={int(pool_pca):4d} "
              f"clusters={int(q['n_clusters']):3d} noise={q['noise_ratio']:.1%} "
              f"max_share={share:.1%} silhouette={q['silhouette']}")
        if share > args.max_cluster_share:
            print(f"  WARNING: {brand_key} largest cluster holds {share:.1%} of the pool -- "
                  f"catch-all blob, topics for this brand are unreliable.")

        topics_here, _ = make_topics(train_segments.iloc[idx].reset_index(drop=True), X, labels, brand_key, f"T{next_global_topic_num}")
        # Replace local ids with globally unique IDs for this clustering pass.
        for topic in topics_here:
            topic.topic_id = f"T{next_global_topic_num}"
            topic.pca_components = int(pool_pca)
            next_global_topic_num += 1
        topics.extend(topics_here)

        # Assign only discovery segments that are in non-noise clusters, for validation.
        local_seg = train_segments.iloc[idx].reset_index(drop=True)
        for local_i, cluster_id in enumerate(labels):
            if cluster_id < 0:
                continue
            topic = topics_here[cluster_id] if cluster_id < len(topics_here) else None
            if topic:
                all_discovery_assignment_frames.append({
                    "record_id": local_seg.iloc[local_i]["record_id"],
                    "segment_id": local_seg.iloc[local_i]["segment_id"],
                    "event_time": local_seg.iloc[local_i]["event_time"],
                    "brand": local_seg.iloc[local_i]["brand"],
                    "topic_id": topic.topic_id,
                    "similarity": np.nan,
                    "assignment_type": "DISCOVERY",
                })

    timer.mark("discovery clustering", f"{len(topics):,} topics from {len(brand_groups)} pools")

    # Optional cross-run stabilization: match newly discovered centroids to a prior topics.json.
    if previous_topics:
        _win_end = pd.to_datetime(segments["event_time"], errors="coerce", utc=True).max()
        topics, stability = stabilize_topics(
            topics, previous_topics, threshold=args.candidate_similarity,
            carry_forward=args.carry_forward_topics,
            max_age_days=args.topic_max_age_days,
            window_end=_win_end)
        if not stability.empty:
            stability.to_csv(out_dir / "topic_stability_decisions.csv", index=False)
            print("\nCross-run topic stability decisions:")
            print(stability["decision"].value_counts().to_string())

    # If clustering produced nothing, fail loudly with enough diagnostic context.
    if not topics:
        raise RuntimeError(
            "No topics discovered. Try --min-cluster-size 10 --min-samples 5 or inspect text quality."
        )

    if args.merge_duplicate_topics:
        n_before = len(topics)
        topics, dup_remap = merge_duplicate_topics(topics, args.duplicate_similarity)
        if dup_remap:
            pd.DataFrame([{"merged_topic_id": k, "into_topic_id": v}
                          for k, v in dup_remap.items()]).to_csv(
                out_dir / "duplicate_topic_merges.csv", index=False)
        print(f"Duplicate merge: {n_before} -> {len(topics)} topics "
              f"({len(dup_remap)} merged at similarity >= {args.duplicate_similarity})")

    # Re-assign the discovery history to the now-stable topic centroids.
    # This produces the historical time series used as the baseline for hot-topic detection.
    discovery_assignments = assign_to_topics(
        train_segments,
        train_embeddings,
        topics,
        similarity_threshold=max(0.50, args.min_similarity - 0.10),
        multi_topic=True,
        max_topics_per_segment=2,
        brand_scoped=args.brand_scoped_assignment,
    )

    timer.mark("history re-assignment", f"{len(discovery_assignments):,} rows")

    # Incremental assignment against stable topic centroids.
    recent_assignments = assign_to_topics(
        recent_segments,
        recent_embeddings,
        topics,
        similarity_threshold=args.min_similarity,
        multi_topic=True,
        max_topics_per_segment=2,
        brand_scoped=args.brand_scoped_assignment,
    )

    assigned = recent_assignments[recent_assignments["topic_id"].notna()].copy()
    unassigned = recent_assignments[recent_assignments["topic_id"].isna()].copy()
    print(f"Incremental assignment: {len(assigned):,} topic-segment assignments; {unassigned['record_id'].nunique() if not unassigned.empty else 0:,} records had no confident topic assignment.")

    # Candidate discovery. The pool is this window's unassigned segments PLUS
    # anything still carried in the rolling buffer, because a genuinely new issue
    # starts with a handful of mentions -- too few to clear min_cluster_size in
    # any single window. Buffering lets five complaints today and five tomorrow
    # add up instead of being discarded twice.
    timer.mark("incremental assignment", f"{len(recent_assignments):,} rows")

    emb_dim = embeddings.shape[1]
    buffer_meta, buffer_emb = load_unassigned_buffer(args.buffer, emb_dim)
    if len(buffer_meta):
        print(f"Rolling buffer: carried {len(buffer_meta):,} segments in from previous runs.")

    if not unassigned.empty:
        uids = unassigned["segment_id"].tolist()
        seg_index = recent_segments.set_index("segment_id").loc[uids].reset_index()
        emb_lookup = {sid: i for i, sid in enumerate(recent_segments["segment_id"])}
        fresh_emb = np.asarray([recent_embeddings[emb_lookup[sid]] for sid in seg_index["segment_id"]], dtype=np.float32)
        fresh_meta = seg_index[BUFFER_COLS].copy()
    else:
        fresh_meta, fresh_emb = _empty_buffer(emb_dim)

    pool_meta = pd.concat([buffer_meta, fresh_meta], ignore_index=True)
    pool_emb = np.vstack([buffer_emb, fresh_emb]).astype(np.float32) if len(pool_meta) else np.empty((0, emb_dim), dtype=np.float32)

    latest_seen = pd.to_datetime(segments["event_time"], errors="coerce", utc=True, format="mixed").max()
    pool_meta, pool_emb = prune_buffer(pool_meta, pool_emb, latest_seen, args.buffer_max_age_days)

    # Re-running over overlapping data can present the same segment twice.
    if len(pool_meta):
        keep = (~pool_meta["segment_id"].duplicated()).to_numpy()
        pool_meta, pool_emb = pool_meta.loc[keep].reset_index(drop=True), pool_emb[keep]

    print(f"Candidate pool: {len(pool_meta):,} segments "
          f"({len(buffer_meta):,} buffered + {len(fresh_meta):,} new, after age/dedupe pruning).")

    # Cluster the pool per brand so a candidate cannot inherit another brand's
    # topic id. Brands too thin to cluster alone share the pooled bucket.
    new_topics: List[Topic] = []
    candidate_frames = []
    absorbed = np.zeros(len(pool_meta), dtype=bool)

    if len(pool_meta):
        pool_counts = pool_meta.groupby("brand")["record_id"].nunique().to_dict()
        cand_groups = []
        cand_small = []
        for brand, n in pool_counts.items():
            idx = pool_meta.index[pool_meta["brand"] == brand].to_numpy()
            if n >= args.brand_min_records:
                cand_groups.append((brand, idx))
            else:
                cand_small.extend(idx.tolist())
        if cand_small:
            cand_groups.append(("__SMALL_BRANDS__", np.asarray(cand_small, dtype=int)))

        for brand_key, gidx in cand_groups:
            if len(gidx) < max(15, args.min_cluster_size):
                continue
            g_meta = pool_meta.loc[gidx].reset_index(drop=True)
            g_emb = pool_emb[gidx]
            cand_pca = frozen_pca.get(brand_key, chosen_pca.get(brand_key, args.pca_components))
            if args.reducer != "pca":
                Xc = apply_reducer(g_emb, args, umap_projector, brand_key)
            else:
                Xc = g_emb if args.no_pca else reduce_dimensions(g_emb, min(cand_pca, g_emb.shape[1]), 42)[0]
            cand_labels = cluster_embeddings(
                Xc,
                max(10, min(args.min_cluster_size, len(g_emb) // 3)),
                max(5, min(args.min_samples, 10)),
                "eom",
                args.cluster_selection_epsilon,
                args.max_cluster_size,
            )
            cand_centroids = topic_centroids(cand_labels, g_emb)
            if not cand_centroids:
                continue

            brand_new, mapping, cm = map_clusters_to_existing_topics(
                g_meta,
                g_emb,
                cand_labels,
                cand_centroids,
                topics_in_scope(topics, brand_key),
                new_topic_similarity=args.candidate_similarity,
                brand=brand_key,
                next_topic_number_start=next_global_topic_num,
                margin=args.candidate_margin,
            )
            for t in brand_new:
                t.pca_components = int(cand_pca)
            next_global_topic_num += len(brand_new)
            topics.extend(brand_new)
            new_topics.extend(brand_new)
            if not cm.empty:
                cm = cm.copy()
                cm["brand"] = brand_key
                candidate_frames.append(cm)

            # Anything that landed in a cluster is now explained; only the
            # leftover noise stays behind for the next run.
            claimed = []
            for cid, topic_id in mapping.items():
                mask = cand_labels == cid
                absorbed[gidx[mask]] = True
                local = g_meta.loc[mask].copy()
                local["topic_id"] = topic_id
                local["similarity"] = np.nan
                local["assignment_type"] = "NEW_TOPIC_DISCOVERY"
                claimed.append(local)
            if claimed:
                assigned = pd.concat([assigned] + claimed, ignore_index=True)

    # P7. Recovery pass over what candidate discovery left as noise.
    #
    # The leftovers are not junk: measured on the previous run, 78% carried 8+
    # content words and their best existing-topic similarity sat just under the
    # 0.50 floor (median 0.431). They fail the floor and they fail
    # min_cluster_size at the same time, so they are dropped twice over. Rather
    # than lower the floor -- which would let weak matches into good topics --
    # give them one more chance to form topics of their own, per brand, with a
    # smaller cluster size.
    recovered_topics: List[Topic] = []
    if args.recover_unassigned and len(pool_meta):
        rec_idx = np.where(~absorbed)[0]
        if len(rec_idx):
            rec_meta = pool_meta.loc[rec_idx].reset_index(drop=True)
            rec_emb = pool_emb[rec_idx]
            content_ok = rec_meta["clean_text"].map(
                lambda t: content_word_count(t, content_stop) >= args.recover_min_content_words
            ).to_numpy()
            print(f"\nRecovery pass: {len(rec_idx):,} unexplained segments, "
                  f"{int(content_ok.sum()):,} contentful (>= {args.recover_min_content_words} content words)")
            rec_claimed = []
            for brand_key in sorted(rec_meta.loc[content_ok, "brand"].astype(str).unique()):
                sel = np.where(content_ok & (rec_meta["brand"].astype(str).to_numpy() == brand_key))[0]
                if len(sel) < max(2 * args.recover_min_cluster_size, 20):
                    continue
                b_emb = rec_emb[sel]
                Xr = (apply_reducer(b_emb, args, umap_projector, brand_key)
                      if args.reducer != "pca"
                      else (b_emb if args.no_pca
                            else reduce_dimensions(b_emb, min(args.pca_components, b_emb.shape[1]), 42)[0]))
                r_labels = cluster_embeddings(Xr, args.recover_min_cluster_size,
                                              max(3, args.recover_min_cluster_size // 3), "eom",
                                              args.cluster_selection_epsilon, args.max_cluster_size)
                r_cents = topic_centroids(r_labels, b_emb)
                if not r_cents:
                    continue
                b_meta = rec_meta.loc[sel].reset_index(drop=True)
                for cid, centroid in r_cents.items():
                    mask = r_labels == cid
                    rows = b_meta.loc[mask]
                    tid = f"T_REC_{next_global_topic_num}"
                    next_global_topic_num += 1
                    ts_ = pd.to_datetime(rows["event_time"], errors="coerce")
                    recovered_topics.append(Topic(
                        topic_id=tid,
                        brand=brand_key,
                        label=make_topic_label(rows["clean_text"].tolist()),
                        size=int(rows["record_id"].nunique()),
                        centroid=centroid.astype(np.float32).tolist(),
                        created_at=None if ts_.isna().all() else ts_.min().isoformat(),
                        last_seen_at=None if ts_.isna().all() else ts_.max().isoformat(),
                        status="EMERGING",
                    ))
                    local = rows.copy()
                    local["topic_id"] = tid
                    local["similarity"] = np.nan
                    local["assignment_type"] = "RECOVERED_DISCOVERY"
                    rec_claimed.append(local)
                    absorbed[rec_idx[sel[mask]]] = True
            if rec_claimed:
                assigned = pd.concat([assigned] + rec_claimed, ignore_index=True)
                topics.extend(recovered_topics)
                n_rec = sum(len(c) for c in rec_claimed)
                # A recovered segment now has a topic, so it must leave the
                # unassigned artifact -- otherwise that file double-reports it
                # and the "records we dropped" number stays wrong.
                _rec_ids = set(pd.concat(rec_claimed)["segment_id"])
                if not unassigned.empty:
                    unassigned = unassigned[~unassigned["segment_id"].isin(_rec_ids)]
                print(f"  recovered {n_rec:,} segments into {len(recovered_topics)} new topics "
                      f"({len(_rec_ids):,} removed from the unassigned list)")
            else:
                print("  nothing clustered above the recovery threshold")

    # Micro pass over whatever the normal candidate clustering left behind.
    micro_topics: List[Topic] = []
    if args.micro_clusters and len(pool_meta):
        left = pool_meta.loc[~absorbed]
        left_emb = pool_emb[~absorbed]
        left = left.reset_index(drop=True)
        groups = find_micro_clusters(left, left_emb, args.micro_distance,
                                     args.micro_min_size, args.micro_min_authors)
        claimed = []
        for sel in groups:
            rows = left.loc[sel]
            centroid = left_emb[sel].mean(axis=0)
            centroid = centroid / max(np.linalg.norm(centroid), 1e-12)
            tid = f"T_MICRO_{next_global_topic_num}"
            next_global_topic_num += 1
            ts = pd.to_datetime(rows["event_time"], errors="coerce", utc=True, format="mixed")
            t = Topic(
                topic_id=tid,
                brand=str(rows["brand"].iloc[0]),
                label=make_topic_label(rows["clean_text"].astype(str).tolist()),
                size=int(rows["record_id"].nunique()),
                centroid=centroid.astype(np.float32).tolist(),
                created_at=None if ts.isna().all() else ts.min().isoformat(),
                last_seen_at=None if ts.isna().all() else ts.max().isoformat(),
                status="MICRO_EMERGING",
            )
            micro_topics.append(t)
            local = rows.copy()
            local["topic_id"] = tid
            local["similarity"] = np.nan
            local["assignment_type"] = "MICRO_DISCOVERY"
            claimed.append(local)
            # Mark these as explained so they leave the buffer.
            orig = pool_meta.index[pool_meta["segment_id"].isin(rows["segment_id"])]
            absorbed[pool_meta.index.get_indexer(orig)] = True
        if claimed:
            assigned = pd.concat([assigned] + claimed, ignore_index=True)
        topics.extend(micro_topics)
        print(f"Micro pass: {len(micro_topics)} MICRO_EMERGING topics "
              f"(distance<={args.micro_distance}, >={args.micro_min_size} records, "
              f">={args.micro_min_authors} authors) from {len(left):,} leftover segments.")

    candidate_mapping = pd.concat(candidate_frames, ignore_index=True) if candidate_frames else pd.DataFrame()

    print(f"New topic decisions: {len(new_topics):,} topics created as EMERGING.")
    if not candidate_mapping.empty:
        print(candidate_mapping["decision"].value_counts().to_string())

    # Persist the still-unexplained remainder for the next run.
    if args.buffer:
        keep_meta = pool_meta.loc[~absorbed].reset_index(drop=True)
        keep_emb = pool_emb[~absorbed]
        save_unassigned_buffer(args.buffer, keep_meta, keep_emb)
        print(f"Rolling buffer: {len(keep_meta):,} segments held for the next run "
              f"({int(absorbed.sum()):,} absorbed into topics this run).")

    timer.mark("candidate discovery", f"{len(new_topics)} emerging")

    # Combine history + incoming data. We use distinct records for downstream counting.
    assigned = pd.concat([discovery_assignments, assigned], ignore_index=True)

    # Topic size should be distinct records, not chunks.
    topic_sizes = assigned.drop_duplicates(["record_id", "topic_id"]).groupby("topic_id")["record_id"].nunique()
    for topic in topics:
        topic.size = int(topic_sizes.get(topic.topic_id, topic.size))

    # Time-series counts: one record can contribute to multiple topics, but only once per topic/day.
    counts = unique_record_topic_counts(assigned)
    ts = build_complete_timeseries(counts, topics)
    hot = add_hot_scores(ts)

    # Member texts per topic, used by both coherence and the junk signals.
    members = (assigned[assigned["topic_id"].notna()]
               .dropna(subset=["clean_text"])
               .drop_duplicates(["topic_id", "record_id"]))
    member_texts = {tid: g["clean_text"].astype(str).tolist()
                    for tid, g in members.groupby("topic_id", sort=False)}

    if args.label_method in ("ctfidf", "llm"):
        # Score topics against others of the SAME brand. Scoring across all
        # brands makes the brand's own name look distinctive, so Delta topics
        # came back labelled "flight / delta / flights". Within the brand its
        # name appears everywhere and correctly drops out.
        brand_of = {t.topic_id: t.brand for t in topics}
        new_labels: Dict[str, List[str]] = {}
        by_brand: Dict[str, Dict[str, Sequence[str]]] = {}
        for tid, txts in member_texts.items():
            by_brand.setdefault(brand_of.get(tid, "GLOBAL"), {})[tid] = txts
        for _brand, group in by_brand.items():
            # A brand with one topic has nothing to contrast against; fall back
            # to the global pool so it still gets a label.
            new_labels.update(ctfidf_labels(group, top_n=5) if len(group) > 1
                              else ctfidf_labels(member_texts, top_n=5))
        relabelled = 0
        for t in topics:
            terms = new_labels.get(t.topic_id)
            if terms:
                t.label = " / ".join(terms)
                relabelled += 1
        print(f"Labelling: c-TF-IDF relabelled {relabelled:,} of {len(topics):,} topics.")

    if args.label_method == "llm":
        try:
            from air2_client import Air2Client, Air2ConfigError
            client = Air2Client()
            n_ok = 0
            for t in topics:
                texts = member_texts.get(t.topic_id, [])[: args.label_llm_samples]
                label = client.label_topic(
                    use_case=args.label_llm_usecase,
                    brand=t.brand,
                    terms=t.label.split(" / "),
                    samples=texts,
                )
                if label:
                    t.label = label
                    n_ok += 1
            print(f"Labelling: LLM named {n_ok:,} of {len(topics):,} topics.")
        except Exception as exc:
            print(f"Labelling: LLM unavailable ({exc}); keeping c-TF-IDF labels.", file=sys.stderr)

    if not args.no_coherence:
        analyzer = CountVectorizer(token_pattern=r"(?u)\b[a-z][a-z']{2,}\b",
                                   lowercase=True).build_analyzer()
        topic_terms = {}
        for t in topics:
            cnt: Dict[str, int] = {}
            for txt in member_texts.get(t.topic_id, []):
                for w in set(analyzer(txt)):
                    if w not in content_stop:
                        cnt[w] = cnt.get(w, 0) + 1
            topic_terms[t.topic_id] = [w for w, _ in
                                       sorted(cnt.items(), key=lambda kv: -kv[1])[:10]]
        coh_map = npmi_coherence(topic_terms, segments["clean_text"].astype(str).tolist())
        for t in topics:
            c = coh_map.get(t.topic_id, float("nan"))
            t.coherence = None if not np.isfinite(c) else round(float(c), 6)
        timer.mark("coherence scoring", f"{len(coh_map):,} topics")

    # Junk flags. Any one signal firing is enough; they fail differently.
    for t in topics:
        term_ratio, empty_ratio = junk_signals(t.label, member_texts.get(t.topic_id, []), content_stop)
        reasons = []
        # A demonstrably coherent topic should not be demoted merely for
        # containing polite words: "flight / airline / thanks / great / thank"
        # is a real 3,660-record topic that the label test alone would bin.
        vetoed = (args.junk_coherence_veto is not None
                  and t.coherence is not None
                  and t.coherence >= args.junk_coherence_veto)
        if not vetoed:
            if term_ratio >= args.junk_term_ratio:
                reasons.append(f"label_ack={term_ratio:.2f}")
            if empty_ratio >= args.junk_empty_ratio:
                reasons.append(f"empty_members={empty_ratio:.2f}")
        # Coherence needs enough documents to mean anything. A 14-record
        # Black Friday topic at 0.929 event-precision scored -0.058 purely
        # because ten unigrams cannot co-occur across fourteen tweets.
        coherence_usable = (t.size or 0) >= args.junk_coherence_min_size
        if (t.coherence is not None and coherence_usable
                and t.coherence < args.junk_coherence):
            reasons.append(f"coherence={t.coherence:.3f}")
        t.is_junk = bool(reasons)
        t.junk_reason = ";".join(reasons) if reasons else None

    n_junk = sum(1 for t in topics if t.is_junk)
    print(f"\nLayer 3: {n_junk:,} of {len(topics):,} topics carry no reputation signal "
          f"({'demoted from ranking' if args.flag_junk_topics else 'flagged only, ranking unchanged'}).")

    # P5. last_seen_at from every assignment, not just the discovery pass.
    # Discovery only sees the history window, so a topic that kept receiving
    # traffic through the holdout still reported the cutoff date: 253 of 409
    # topics were pinned to the same day, making every lifecycle signal derived
    # from this field meaningless.
    _assigned_any = assigned[assigned["topic_id"].notna()]
    if not _assigned_any.empty:
        _last = (pd.to_datetime(_assigned_any["event_time"], errors="coerce", utc=True)
                 .groupby(_assigned_any["topic_id"]).max())
        _refreshed = 0
        for t in topics:
            v = _last.get(t.topic_id)
            if v is not None and not pd.isna(v):
                prev = pd.to_datetime(t.last_seen_at, errors="coerce", utc=True)
                if pd.isna(prev) or v > prev:
                    t.last_seen_at = v.isoformat()
                    _refreshed += 1
        print(f"\nlast_seen_at refreshed from assignments for {_refreshed} of {len(topics)} topics")

        # Size has to come from this window's assignments, not from whatever the
        # discovery pass happened to seed. A carried-forward topic that received
        # nothing is genuinely 0 this window; one that revived should report what
        # it actually got.
        _counts = _assigned_any.groupby("topic_id")["record_id"].nunique()
        _revived = 0
        for t in topics:
            t.size = int(_counts.get(t.topic_id, 0))
            # A carried-forward topic is marked DORMANT before assignment runs,
            # because at that point nothing has been assigned to anything. Once
            # records land on it, it is active again -- and saying otherwise
            # would put a live 1,900-record topic on a dashboard as dormant.
            if t.status == "DORMANT" and t.size > 0:
                t.status = "ACTIVE"
                _revived += 1
        _dormant = sum(1 for t in topics if t.size == 0)
        if _revived:
            print(f"Revived from DORMANT by this window's traffic: {_revived} topics")
        if _dormant:
            print(f"Idle this window (no records assigned): {_dormant} of {len(topics)} topics")

    # P2. Identify conversational grab-bags before anything ranks them.
    _members = {}
    for tid, grp in _assigned_any.groupby("topic_id"):
        _members[tid] = grp["clean_text"].astype(str).tolist()
    _n_resid = flag_residual_clusters(topics, _members, args.residual_ack_ratio)
    print(f"Residual (chatter) clusters flagged: {_n_resid} of {len(topics)}")

    topic_meta = pd.DataFrame([
        {
            "topic_id": t.topic_id,
            "brand": t.brand,
            "label": t.label,
            "size": t.size,
            "status": t.status,
            "created_at": t.created_at,
            "last_seen_at": t.last_seen_at,
            "pca_components": t.pca_components,
            "coherence": t.coherence,
            "is_junk": bool(t.is_junk),
            "junk_reason": t.junk_reason,
            "residual_cluster": bool(t.residual_cluster),
            "residual_reason": t.residual_reason,
            "ack_member_ratio": t.ack_member_ratio,
        }
        for t in topics
    ])

    summary = topic_meta.merge(hot, on="topic_id", how="left", suffixes=("_lifecycle", ""))
    summary = gate_alert_surface(summary, args.alert_min_coherence, args.alert_min_size)
    _sup = int(summary["alert_suppressed"].sum())
    _alerts = int(summary["status"].isin(["HOT", "TRENDING"]).sum())
    print(f"Alert gate: {_alerts} alerts after suppressing {_sup} "
          f"(min_coherence={args.alert_min_coherence}, min_size={args.alert_min_size})")
    # Demotion sorts junk to the bottom rather than dropping it, so the rows
    # stay auditable and a bad threshold costs nothing to undo.
    if args.flag_junk_topics:
        summary = summary.sort_values(["is_junk", "hot_score", "recent_volume"],
                                      ascending=[True, False, False])
    else:
        summary = summary.sort_values(["hot_score", "recent_volume"], ascending=False)

    # Report quality both ways from the same run: Layer 3 changes no upstream
    # state, so the with/without comparison needs no second run.
    if "coherence" in summary.columns and summary["coherence"].notna().any():
        allc = summary["coherence"].dropna()
        keptc = summary.loc[~summary["is_junk"], "coherence"].dropna()
        print("\n=== Coherence (C_NPMI) ===")
        print(f"  all topics       n={len(allc):>4}  mean={allc.mean():+.4f}  "
              f"median={allc.median():+.4f}  coherent(>=0.1)={int((allc >= 0.1).sum()):>4}")
        print(f"  junk demoted     n={len(keptc):>4}  mean={keptc.mean():+.4f}  "
              f"median={keptc.median():+.4f}  coherent(>=0.1)={int((keptc >= 0.1).sum()):>4}")
        jc = summary.loc[summary["is_junk"], "coherence"].dropna()
        if len(jc):
            print(f"  flagged as junk  n={len(jc):>4}  mean={jc.mean():+.4f}")

    # Save outputs.
    df.to_csv(out_dir / "normalized_records.csv", index=False)
    segments.to_csv(out_dir / "segments.csv", index=False)
    assigned.to_csv(out_dir / "topic_assignments.csv", index=False)
    unassigned.to_csv(out_dir / "unassigned_recent_records.csv", index=False)
    counts.to_csv(out_dir / "topic_daily_counts.csv", index=False)
    ts.to_csv(out_dir / "topic_timeseries.csv", index=False)
    summary.to_csv(out_dir / "topic_hot_summary.csv", index=False)
    if not candidate_mapping.empty:
        candidate_mapping.to_csv(out_dir / "candidate_topic_decisions.csv", index=False)
    if grid_results:
        pd.concat(grid_results, ignore_index=True).to_csv(out_dir / "hdbscan_grid_search.csv", index=False)

    # Attach example tweets so a human can eyeball whether a topic is real.
    # These are the members closest to the centroid, i.e. what most defines the
    # topic. A uniform random sample would be a stricter correctness test.
    if args.tweets_per_topic != 0:
        raw_lookup = dict(zip(df["tweet_id"].astype(str), df["text"].astype(str)))
        ex = assigned[assigned["topic_id"].notna()].copy()
        # Candidate-discovery rows carry no similarity; they are the only
        # evidence an emerging topic has, so rank them first.
        ex["_sim"] = pd.to_numeric(ex["similarity"], errors="coerce").fillna(1.0)
        ex = (ex.sort_values("_sim", ascending=False)
                .drop_duplicates(["topic_id", "record_id"]))
        by_topic = {tid: g for tid, g in ex.groupby("topic_id", sort=False)}
        for t in topics:
            g = by_topic.get(t.topic_id)
            if g is None:
                t.tweets = []
                continue
            ids = (g["record_id"] if args.tweets_per_topic < 0
                   else g["record_id"].head(args.tweets_per_topic))
            t.tweets = [raw_lookup.get(str(r), "") for r in ids.tolist()]

    # Carry momentum onto the topic objects so topics.json records not just where
    # a topic sits but how it was behaving -- --previous-topics reads this back.
    if not hot.empty:
        hot_lookup = hot.set_index("topic_id")[["hot_score", "status"]].to_dict("index")
        for t in topics:
            h = hot_lookup.get(t.topic_id)
            if h is not None:
                t.hot_score = None if pd.isna(h["hot_score"]) else round(float(h["hot_score"]), 6)
                t.hot_status = None if pd.isna(h["status"]) else str(h["status"])

    if args.event_recall:
        _er = compute_event_recall(assigned, topics)
        if not _er.empty:
            _er.to_csv(out_dir / "event_recall.csv", index=False)
            _f = _er[_er["found"]]
            print(f"\nEvent recall: {len(_f)}/{len(_er)} events found | "
                  f"mean recall-in-best-topic {_f['recall_in_best_topic'].mean():.3f} | "
                  f"mean topics-for-80% {_f['topics_for_80pct'].mean():.1f}")

    save_json(out_dir / "topics.json", [asdict(t) for t in topics])

    # Drift: how far each transformed pool sat from the data the frozen manifold
    # was fitted on. Rising values are the signal to re-fit.
    if umap_projector is not None:
        _drift = umap_projector.drift_frame()
        if not _drift.empty:
            _drift.to_csv(out_dir / "umap_drift.csv", index=False)
            _worst = _drift.nlargest(3, "mean_dist_to_reference")
            print("\nUMAP drift (mean cosine distance to reference set):")
            print(f"  overall mean {_drift['mean_dist_to_reference'].mean():.4f}; "
                  f"worst pools: " +
                  ", ".join(f"{r['pool']} {r['mean_dist_to_reference']:.4f}"
                            for _, r in _worst.iterrows()))

    # Everything needed to reproduce this run, kept beside the topics rather
    # than inside them so topics.json keeps exactly the schema it always had.
    save_json(out_dir / "run_metadata.json", {
        "input_csv": args.csv,
        "embedding_model": args.model,
        "embedding_dim": int(embeddings.shape[1]),
        "embedding_precision": "fp16" if _fp16 else "fp32",
        "embedding_device": _device,
        "reducer": args.reducer,
        "umap": (None if args.reducer != "umap" else {
            "mode": "refit-per-pool" if args.umap_refit_per_pool else "versioned-shared-model",
            "artifact": args.umap_model,
            **(umap_projector.meta if umap_projector is not None else {
                "n_components": int(args.umap_components),
                "n_neighbors": int(args.umap_neighbors),
                "min_dist": float(args.umap_min_dist),
            }),
        }),
        "hdbscan": {
            "min_cluster_size": int(args.min_cluster_size),
            "min_samples": int(args.min_samples),
            "cluster_selection_method": "eom",
            "cluster_selection_epsilon": float(args.cluster_selection_epsilon),
            "max_cluster_size": int(args.max_cluster_size),
            "split_max_size": int(args.split_max_size),
        },
        "thresholds": {
            "min_similarity": float(args.min_similarity),
            "candidate_similarity": float(args.candidate_similarity),
            "candidate_margin": float(args.candidate_margin),
            "duplicate_similarity": float(args.duplicate_similarity),
        },
        "umap_drift": (None if umap_projector is None or umap_projector.drift_frame().empty
                       else round(float(umap_projector.drift_frame()["mean_dist_to_reference"].mean()), 5)),
        "previous_topics": args.previous_topics,
        "segments": int(len(segments)),
        "records": int(segments["record_id"].nunique()),
        "topics": len(topics),
        "completed_at": pd.Timestamp.utcnow().isoformat(),
    })

    if args.tune_pca:
        save_json(out_dir / "pca_config.json", chosen_pca)
        if pca_search_frames:
            pd.concat(pca_search_frames, ignore_index=True).to_csv(out_dir / "pca_search.csv", index=False)
        print("\nPCA widths chosen per pool:")
        for k, v in sorted(chosen_pca.items()):
            print(f"  {k:<20} {v}")

    if args.plots:
        plot_hot_topics(ts, topics, out_dir)

    timer.mark("metrics + outputs", f"{len(topics):,} topics scored")

    print("\nTop detected topics:")
    cols = ["topic_id", "label", "size", "recent_volume", "growth_rate", "velocity_ratio", "hot_score", "status", "status_lifecycle"]
    print(summary[cols].head(15).to_string(index=False))
    print(f"\nOutputs written to: {out_dir.resolve()}")
    timer.report()
    return 0


def main() -> None:
    args = parse_args()
    try:
        raise SystemExit(run(args))
    except KeyboardInterrupt:
        print("Interrupted.", file=sys.stderr)
        raise SystemExit(130)
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
