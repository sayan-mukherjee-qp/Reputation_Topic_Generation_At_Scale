#!/usr/bin/env python3
"""Translate similarity thresholds from one embedding space to another.

Every similarity threshold in the pipeline (--min-similarity, --candidate-
similarity, --duplicate-similarity, ...) was tuned in one embedding space. A
different encoder scores the same text on a different scale, so the same
number admits a different share of matches. This finds, for each threshold,
the value in the new space that admits the same share as the tuned value does
in the reference space, using two base runs over the SAME input:

  first/second topic floors   each segment's best and second-best own-brand
                              centroid similarity, in each run's own topics
  centroid-pair thresholds    same-brand topic-centroid pair similarities
  random pairs                a topic-free check: cosine of random segment pairs

Usage:
  calibrate_thresholds.py REF_RUN REF_MODEL NEW_RUN NEW_MODEL
  e.g. calibrate_thresholds.py \\
         reputation_topic_gpu/out/exp_control sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2 \\
         reputation_topic_gpu_768/out/exp_emb768 sentence-transformers/paraphrase-multilingual-mpnet-base-v2

Each RUN is a directory holding base_200k/ (segments.csv, topics.json) and
cache/embed/ (the embedding cache that run wrote).
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "reputation_topic_gpu"))
from reputation_topic_detection import embedding_cache_key  # noqa: E402

# (label, distribution, value tuned in the reference space)
THRESHOLDS = [
    ("--min-similarity / --history-min-similarity", "best", 0.50),
    ("second-topic floor (inflation)", "second", 0.50),
    ("--candidate-similarity", "pairs", 0.55),
    ("--label-merge-similarity", "pairs", 0.70),
    ("--consolidate-min-similarity", "pairs", 0.75),
    ("--duplicate-similarity / --label-reuse-similarity", "pairs", 0.92),
]


def distributions(run: Path, model: str, texts, brands):
    E = np.load(run / "cache/embed" / f"{embedding_cache_key(model, texts)}.npy").astype(np.float32)
    E /= np.linalg.norm(E, axis=1, keepdims=True)
    T = json.loads((run / "base_200k/topics.json").read_text())
    C = np.asarray([t["centroid"] for t in T], np.float32)
    C /= np.linalg.norm(C, axis=1, keepdims=True)
    tb = np.asarray([t["brand"] for t in T])
    best = np.full(len(E), -1.0, np.float32)
    second = best.copy()
    for b in np.unique(brands):
        m, own = brands == b, tb == b
        if not own.any():
            continue
        S = E[m] @ C[own].T
        if S.shape[1] >= 2:
            p = -np.partition(-S, 1, axis=1)[:, :2]
            best[m], second[m] = p.max(1), p.min(1)
        else:
            best[m] = S[:, 0]
    pairs = []
    for b in np.unique(tb):
        Cb = C[tb == b]
        if len(Cb) > 1:
            pairs.append((Cb @ Cb.T)[np.triu_indices(len(Cb), 1)])
    i, j = np.random.default_rng(0).integers(0, len(E), (2, 200_000))
    rand = np.einsum("ij,ij->i", E[i], E[j])
    return {"best": best, "second": second, "pairs": np.concatenate(pairs), "rand": rand}


def equivalent(ref, new, thr):
    """The new-space value admitting the same share the reference admits at thr."""
    return float(np.quantile(new, 1 - np.mean(ref >= thr)))


def main() -> int:
    if len(sys.argv) != 5:
        print(__doc__)
        return 2
    ref_run, ref_model, new_run, new_model = Path(sys.argv[1]), sys.argv[2], Path(sys.argv[3]), sys.argv[4]
    seg = pd.read_csv(ref_run / "base_200k/segments.csv", keep_default_na=False, low_memory=False)
    texts = seg["clean_text"].astype(str).tolist()
    brands = seg["brand"].astype(str).to_numpy()
    ref = distributions(ref_run, ref_model, texts, brands)
    new = distributions(new_run, new_model, texts, brands)
    print(f"{len(texts):,} segments; reference {ref_model.split('/')[-1]}, new {new_model.split('/')[-1]}\n")
    print(f"  {'threshold':50s} {'tuned':>6s} {'equivalent':>10s} {'random-pair check':>18s}")
    for name, key, thr in THRESHOLDS:
        print(f"  {name:50s} {thr:6.2f} {equivalent(ref[key], new[key], thr):10.3f} "
              f"{equivalent(ref['rand'], new['rand'], thr):18.3f}")
    for name, d in (("reference", ref), ("new", new)):
        print(f"\n  {name}: first topic >= 0.50 for {np.mean(d['best'] >= .5):.1%} of segments, "
              f"second >= 0.50 for {np.mean(d['second'] >= .5):.1%}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
