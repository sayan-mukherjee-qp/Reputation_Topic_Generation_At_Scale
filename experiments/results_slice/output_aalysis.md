# Slice Test — Tuned 768-dim Copy

**Date:** 1 Oct 2026. **Machine:** Intel Core 7 150U, CPU only (same as every earlier run).

**Data:** `make_slice.py`'s 10% cut of the same corpora, every brand and the full 61-day window kept: `twcs_subset_20k.csv` (20,007 records) and `stream30/chunk_1..6` (29,627 records), stream window 3.

**Versions:** five, each a 20k base run followed by a six-batch stream, all carrying the same-brand ID fix:

| Version | Code | Thresholds |
|---|---|---|
| control | `reputation_topic_gpu` (current) | 384-dim MiniLM, original |
| 768 · original | `reputation_topic_gpu_768` | min 0.50 · candidate 0.55 · duplicate 0.92 · label-merge 0.70 · label-reuse 0.92 · consolidate 0.75 |
| 768 · floors | same | min + history floor 0.53; rest original |
| 768 · calibrated | same | min 0.53 · history 0.53 · candidate 0.60 · duplicate 0.93 · label-merge 0.75 · label-reuse 0.93 · consolidate 0.80 |
| 768 · +margin | same | calibrated + `--secondary-margin 0.03` |

**Reproduce:** `experiments/slice_test.sh`, then `experiments/compare_slice.py` → `slice_audit.csv` (this folder). Published report: https://claude.ai/artifact/D69fR4aYC1LX8kmPP7NTWd (Report 5).

---

## 1. Executive summary

**The same-brand ID fix works in the 768 copy: zero brand switches in all 24 of its stream batches, against 13–34 per batch before the fix. The threshold calibration does not help: every tighter set lowered published-set coherence in the stream. The 768 copy keeps its original thresholds as the default for the 200k GPU run.**

---

## 2. What changed in `reputation_topic_gpu_768`

| Change | Source | Status |
|---|---|---|
| Topic can only inherit an ID from the same brand's previous topics | ported from the original | **kept** — verified §3 |
| LLM failures report last error + base URL | ported | kept |
| Live stream batch logs, `--echo` | ported | kept |
| `make_slice.py`; `BASE_CSV` / `STREAM_DIR` / `BASE_RUN` in compose | ported | kept (dashboard not ported) |
| `--history-min-similarity` replaces a floor hard-coded at 0.50 | new | kept, default 0.50 (no behaviour change) |
| `run_stream.py --thresholds {original, floors, calibrated}` | new | kept, default `original` |
| Calibrated thresholds as default | new | **rejected** — §4 |
| `--secondary-margin 0.03` | tested | **rejected** — §4 |

---

## 3. Brand fix: verified

| Stream | Brand switches per batch 1–6 | ID retention |
|---|---|---|
| Control, 300k, before the fix | 22 · 14 · 34 · 24 · 17 · 12 | 0.945–0.996 |
| 768, 300k, before the fix | 24 · 15 · 34 · 31 · 16 · 13 | 0.974–0.991 |
| Control, 30k slice, with fix | **0 · 0 · 0 · 0 · 0 · 0** | 0.986–1.000 |
| 768 (all four versions), 30k slice, with fix | **0 in all 24 batches** | 0.971–1.000 |

A brand switch is a topic ID present in two consecutive batches under different brands. Every slice batch also had 0 duplicate IDs, 100% own-brand assignment and 0 mis-tagged topics.

---

## 4. Thresholds

### Stream (fixed stream30 corpus, mean of 6 batches)

| | Control | **768 · original** | 768 · floors | 768 · calibrated | 768 · +margin |
|---|---|---|---|---|---|
| **Published C_npmi** | 0.083 | **0.097** | 0.078 | 0.068 | 0.053 |
| Published C_v | 0.529 | **0.556** | 0.529 | 0.521 | 0.510 |
| Event recall · topics-for-80% | 0.729 · 1.83 | **0.748 · 1.72** | 0.741 · 1.84 | 0.724 · 1.86 | 0.629 · 2.45 |
| Inflation (batch 6) | 1.50 (1.58) | 1.65 (1.70) | 1.58 (1.64) | 1.60 (1.68) | **1.17 (1.21)** |
| Unassigned, holdout | 13.6% | 9.9% | 12.0% | 12.5% | 12.4% |
| Live topics b6 · median | 159 · 47 | 138 · 86 | 159 · 61 | 174 · 59 | 175 · 41 |
| Unclear labels (mean) | 18.3 | 17.2 | 21.3 | 21.5 | 25.2 |
| Alerts · negative | 10 · 0 | 7 · 0 | 9 · 1 | 9 · 0 | 6 · 0 |
| LLM calls · tokens | 195 · 245k | **185 · 232k** | 217 · 274k | 249 · 313k | 268 · 337k |

| Published C_npmi by batch | 1 | 2 | 3 | 4 | 5 | 6 |
|---|---|---|---|---|---|---|
| Control | 0.086 | 0.119 | 0.088 | 0.063 | 0.077 | 0.064 |
| 768 · original | 0.125 | 0.107 | 0.099 | 0.101 | 0.081 | 0.066 |
| 768 · floors | 0.072 | 0.110 | 0.108 | 0.078 | 0.063 | 0.035 |
| 768 · calibrated | 0.073 | 0.092 | 0.092 | 0.066 | 0.049 | 0.036 |
| 768 · +margin | 0.050 | 0.060 | 0.073 | 0.062 | 0.039 | 0.035 |

### Base (fixed 20k corpus)

| | Control | 768 · original | 768 · floors | 768 · calibrated | 768 · +margin |
|---|---|---|---|---|---|
| Topics · median | 92 · 163 | 76 · 236 | 83 · 170 | 84 · 168 | 84 · 131 |
| Unassigned · inflation | 34.8% · 1.40 | 26.3% · 1.50 | 31.8% · 1.43 | 31.8% · 1.43 | 31.8% · 1.09 |
| Published C_npmi · C_v | 0.085 · 0.516 | **0.105 · 0.560** | 0.101 · 0.554 | 0.099 · 0.547 | 0.074 · 0.520 |
| Events · recall · for-80% | 10/10 · 0.721 · 2.2 | 10/10 · 0.716 · 2.0 | 10/10 · 0.696 · 2.0 | 10/10 · 0.696 · 2.0 | 10/10 · 0.622 · 2.4 |
| Embedding · rest | 145 s · 95 s | 572 s · 101 s | cached · 101 s | cached · 103 s | cached · 103 s |
| Peak RAM | 1.99 GiB | 3.01 GiB | 2.20 GiB | 2.20 GiB | 2.20 GiB |

### Why calibration hurt

The calibration matched how often each threshold admits a match in each space (`experiments/calibrate_thresholds.py`). That did make 768 behave more like the control: unassigned rose, inflation fell, topic counts rose. But 768's advantage comes from admitting more; its larger topics are the ones that score well. Tighter floors leave more segments unassigned, candidate discovery turns them into small new topics (159–175 by batch 6 against 138), and small topics score at or below zero in every run so far. The calibrated candidate and merge thresholds (0.60 / 0.75 / 0.80) push the same way.

The secondary margin cuts inflation (1.60 → 1.17) but costs event recall (0.724 → 0.629) and coherence, matching the earlier 384 ablation.

---

## 5. How far to trust it

A 10% slice: 76–175 topics rather than 400–600, one run per version. Differences of 0.01–0.02 C_npmi at this scale are within plausible noise. The ordering, though, holds in both the base run and the stream: original first among the 768 versions, calibrated and margin last. The decision rests on that consistency, not on any single gap.

---

## 6. For the 200k GPU run

- `reputation_topic_gpu_768` is ready: original thresholds, same-brand fix, no VRAM cap, live stream logs. The run commands shared earlier apply unchanged.
- Double-counting remains 768's one open trade-off (1.65 vs 1.50 here). If it matters more than coherence, `run_stream.py --thresholds floors` is the mildest option (1.58), at a coherence cost.
- On this CPU the 768 encoder ran at ~35 segments/s, 3.7–4.0× slower than MiniLM. The GPU run will show whether that ratio holds.
