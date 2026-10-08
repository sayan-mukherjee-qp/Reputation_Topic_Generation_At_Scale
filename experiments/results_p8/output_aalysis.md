# Near-Duplicate Topics — Six Fixes Tested

**Date:** 6 Oct 2026. **Code:** `reputation_topic_gpu` (new flags, all off by default). **Data:** 20k base slice + 30k stream slice (6 batches, window 3), MiniLM-384, one frozen UMAP artifact and cached embeddings shared by every version.

**Published report:** https://claude.ai/artifact/D69fR4aYC1LX8kmPP7NTWd (Report 6). **Reproduce:** `experiments/p8_test.sh`, then `experiments/compare_p8.py` → `p8_audit.csv`, `p8_merge_judgements.csv` (this folder).

---

## 1. The case

AmericanAir T8 "Flight cancellations and delays" (590) and T11 "Flight delays and cancellations" (447) in the 20k base run survived every merge:

| Merge step | Rule | Why it missed |
|---|---|---|
| Duplicate merge | centroid ≥ 0.92 | 0.80 |
| Label merge | identical label text, centroid ≥ 0.70 | labels differ by word order |
| Capacity merge | only above 600 live topics | 92 topics |

61% of T11's records were also on T8. T8's tweets are broader (seats, fees, gates); T11's are delays and missed connections.

---

## 2. Ideas and flags

| Idea | Flag | Behaviour |
|---|---|---|
| 1a Label words | `--label-merge-mode wordset` | label merge on the same content words, any order |
| 1b Label meaning | `--label-merge-mode semantic` | label merge on label embeddings ≥ 0.90 |
| 2 Shared records | `--overlap-merge-ratio 0.5` | merge when ≥ 50% of the smaller topic's records are on the other (centroid ≥ 0.70) |
| 3 LLM review | `--llm-merge-review` | LLM judges close pairs, merges "same" at confidence ≥ 0.7 |
| 4 Rename apart | `--disambiguate-labels` | LLM renames colliding same-brand pairs; no merging |
| 5 Parent groups | `--topic-groups` | complete-linkage groups (every pair ≥ 0.75); no merging |
| 6 ID aliases | `--keep-aliases` | absorbed IDs stay resolvable, carried across batches |
| Guards | automatic; `--merge-max-aliases N` | no cross-step chaining within a run; lifetime cap on absorbed IDs |

All merges are same-brand, smaller into larger, and never chained within a step.

---

## 3. Results — stream, mean of 6 batches

| | Baseline | 1a | 1b | 2 | 3 | 4 | 5 | 2+6 | **Best combo** |
|---|---|---|---|---|---|---|---|---|---|
| Identical / same-word labels | 3.7 | 2.5 | 1.7 | 1.0 | 1.0 | **0** | 2.2 | 0.2 | **0** |
| Near-identical labels (≥ 0.85) | 14.8 | 12.8 | 7.8 | 5.7 | 7.2 | **0.2** | 15.0 | 5.3 | **0** |
| ≥ 50% shared-record pairs | 19.8 | 19.7 | 18.3 | **0.7** | 5.8 | 20.3 | 20.2 | **0.8** | 6.8 |
| Published C_npmi | 0.083 | 0.077 | 0.087 | 0.080 | 0.082 | 0.080 | 0.082 | 0.082 | 0.089 |
| Published C_v | 0.529 | 0.527 | 0.533 | 0.532 | 0.536 | 0.525 | 0.528 | 0.533 | **0.545** |
| Event recall | 0.729 | 0.729 | 0.730 | **0.786** | 0.780 | 0.728 | 0.728 | **0.786** | 0.779 |
| Topics for 80% | 1.83 | 1.85 | 1.79 | 1.72 | 1.69 | 1.85 | 1.85 | 1.72 | 1.70 |
| Inflation | 1.50 | 1.50 | 1.49 | **1.28** | 1.32 | 1.50 | 1.50 | **1.28** | 1.39 |
| Live topics | 119 | 121 | 114 | 80 | 89 | 123 | 121 | 80 | 92 |
| ID retention, plain | 0.993 | 0.993 | 0.987 | 0.901 | 0.914 | 0.998 | 0.994 | 0.901 | 0.932 |
| ID retention, with aliases | 0.993 | 0.993 | 0.987 | 0.901 | 0.914 | 0.998 | 0.994 | **0.996** | **0.997** |
| LLM tokens / batch | 38.7k | 42.6k | 38.9k | 32.4k | 63.4k | 43.5k | 40.5k | 32.9k | 37.9k |

Best combo = `--overlap-merge-ratio 0.5 --label-merge-mode semantic --disambiguate-labels --keep-aliases --merge-max-aliases 3` (`combo_cheap`). Brand switches: 0 in every batch of every version. Published C_npmi moves within ±0.006 for every idea, which is within the slice's noise.

### Merge correctness (LLM judge, base run, pre-merge topics)

| Version | Merges | Judged correct | Wrong |
|---|---|---|---|
| 1a | 1 | 1 | — |
| 1b | 2 | 2 | — |
| 2 | 7 | 5 | Spotify "iOS playback" → "Login and access" (59% shared); AmericanAir T12 → T8 (cascade: T8 broadened by its first merge, relabelled to match T12, then label-merged) |
| Best combo | 7 | 6 | the Spotify pair; the cascade is gone |
| 3 | 7 | not judged (circular) | it rejected both of 2's wrong merges itself |

---

## 4. Verdicts

| Idea | Verdict | Evidence |
|---|---|---|
| 1a | Works, tiny reach | 1/1 correct; only reordered labels; subsumed by 1b |
| 1b | **Works** | 2/2 correct; near-identical labels halved; free (reuses the run's encoder) |
| 2 | **Works with guards** | largest effect: inflation −0.22, recall +0.057, −16% tokens; alone: 2/7 wrong and retention 0.90, fixed by the guards and aliases |
| 3 | Works, expensive | most precise; +64% tokens; review + rename + aliases was no better than the best combo at 1.8× its cost |
| 4 | **Works** | collisions → 0 with no merges (retention 0.998); T8/T11 → "Flight cancellations with boarding and seating issues" / "Flight delays causing missed connections and rebooking problems"; +12% tokens |
| 5 | **Does not work as built** | centroid-only groups mix issues: "Late delivery" + "Unclear topic", "Login" + "iOS playback"; on average 9 of 17 stream groups share no label word, 5 contain an Unclear topic |
| 6 | **Works** | alias-aware retention 0.996–0.997 when merges run; free |

---

## 5. Open risks

- **Topic growth.** Largest topic at batch 6: 2,129 (best combo) vs 1,494 (baseline). The alias cap limits absorbed IDs (holds at 3), not size. Check on the 300k stream.
- **One remaining false positive** (Spotify, 59% shared). An overlap threshold of 0.6, or LLM review for the 0.5–0.6 band only, would catch it.
- **Cap gap in the review combo:** the original exact-label merge path does not check `--merge-max-aliases` (one topic reached 4). The best combo uses the semantic path, which does.
- **Default behaviour unchanged:** with no flags, the new code reproduces the 1 Oct slice run exactly (same IDs, brands, sizes, centroids, assignments). The two remaining differences are pre-existing run-to-run variation: 15 LLM labels, and the coherence of 6 small topics, whose top-10 terms break count ties in hash-dependent set order.
- 10% slice, one run per version.

## 6. Next steps

1. Run the best combo on the 200k base + 300k stream; watch largest-topic growth.
2. If it holds, enable it by default and port to `reputation_topic_gpu_768`.
3. Optional: make the coherence top-term tie-break deterministic (sort ties alphabetically).
