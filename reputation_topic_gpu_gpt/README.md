# Reputation Topic Discovery — EXPERIMENT: GPT decisions + 768-dim embeddings

> **Experimental copy of `../reputation_topic_gpu`** (formerly the Laya copy).
> Two changes from the Laya copy:
>
> 1. **Stream decisions come from GPT** (`--decision-backend gpt`, default).
>    Every incoming record is a question -- "which of its brand's nearest
>    topics does this post belong to, or `other` / `no_issue`?" -- answered by
>    GPT through AI Router v1 (`gpt_decision_client.py`; credentials are the
>    `api_key` / `base_url` / `use_case` already used for LLM labels,
>    `AIROUTER_MODEL` picks the model, default gpt-4.1-mini). The client keeps
>    Laya's contract -- (choice, confidence, probabilities) -- so the gates
>    (`--laya-min-confidence` 0.50, `--laya-secondary-prob` 0.30), fallbacks
>    and outputs are unchanged. About 20 posts of one brand share a call, each
>    with its own shortlist (`--gpt-posts-per-call`, `--gpt-max-options`,
>    `--gpt-workers`). GPT's confidence is its own estimate, not a calibrated
>    probability: re-check the 0.50 gate on real data.
>    `--decision-backend laya` restores Laya (`LAYA_BASE_URL` in `.env`).
>
> 2. **768-dim embeddings**: `sentence-transformers/paraphrase-multilingual-
>    mpnet-base-v2` through sentence-transformers (default everywhere: CLI,
>    `run_stream.py`, Dockerfile, compose). Chosen on the 20k slice (8 Oct
>    2026) over multilingual-e5-base, whose compressed cosine scale (random
>    pairs 0.79 vs 0.17) collapses the tuned thresholds, and over
>    gte-multilingual-base / nomic-embed-text-v2, whose modelling code fails
>    under transformers 5. `--embed-prompt` and `--trust-remote-code` support
>    other encoders (see `ENCODER_SETTINGS`). The UMAP artifact is
>    `umap_v6_768_perbrand.pkl`; compose defaults `MAX_VRAM_GB=4` (the
>    model's fp32 weights are 1.1 GB).
>
> 3. **Default run size: 20k base, then 6 stream batches of ~5k**
>    (`twcs_subset_20k.csv`, `stream30/chunk_1..6`, built by `make_slice.py`;
>    compose `BASE_CSV` / `STREAM_DIR` / `BASE_RUN`). For the full sets use
>    `BASE_CSV=twcs_subset_200k.csv STREAM_DIR=stream300 BASE_RUN=base_200k`.
>
> 4. **Stream coherence uses a fixed reference** (`--coherence-reference`,
>    passed by `run_stream.py`: the base run's `segments.csv`, up to 50k
>    sampled). Scored against a ~5k-record window alone, NPMI drove topics
>    that read +0.10 on the base run to -0.10, the residual flag suppressed 44
>    of 76 topics in one batch, and four brands were left with nothing for the
>    matcher to offer.
>
> Cost guide for GPT decisions: about 2-3k tokens per call of 20 posts, so a
> 49k-record batch is ~2.5k calls. `--matcher hybrid` lets cosine settle the
> clear cases and asks GPT only about the borderline band.

## Original README

# Reputation Topic Discovery + Hot Topic Test (GPU container)

> GPU-enabled copy of the prototype, packaged with `uv` and Docker.
> **See [DOCKER.md](DOCKER.md) for building and running on a cloud GPU box.**
> The notes below describe the pipeline itself and apply to both copies.

This prototype tests:

- long-record segmentation before embedding
- semantic embeddings with Sentence Transformers
- HDBSCAN topic discovery
- incremental assignment to stable topic centroids
- candidate clustering for new-topic discovery
- distinct-record counting so one giant post cannot inflate volume
- daily topic time series
- growth / velocity / anomaly / persistence metrics
- explainable HotScore

## Input

CSV columns used by default:

- `tweet_id`
- `event_time`
- `text`
- `brand`

Example: 

```csv
,tweet_id,event_time,author_id,text,brand
0,708057,2017-10-01 00:02:27,289259,"@118730 you have to stop charging me fees...",Uber_Support
```

## Setup

Containerized (recommended, and the only supported path for GPU runs) —
see [DOCKER.md](DOCKER.md):

```bash
docker build -t reputation-topic-gpu .
docker run --rm --gpus all reputation-topic-gpu /app/gpu_check.py
```

Local, without Docker:

```bash
uv sync --extra cu130      # or --extra cu126 for a driver older than 580, or --extra cpu
source .venv/bin/activate
```

The pipeline's default embedding model is `sentence-transformers/all-MiniLM-L6-v2`; a local run downloads it on first use. The Docker image instead bakes in `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` (the corpus is multilingual) and runs offline, so pass `--model` with that name inside the container, as `docker-compose.yml` does.

## Run

A web dashboard can start the 200k base run and the 300k stream, show each stage live, and chart
the emerging topics and analytics. See [dashboard/README.md](dashboard/README.md).

```bash
python reputation_topic_detection.py /path/to/twitter.csv \
  --out output \
  --grid-search \
  --plots
```

For a first fast experiment:

```bash
python reputation_topic_detection.py /path/to/twitter.csv --out output
```

Useful tuning knobs:

```bash
--min-cluster-size 20
--min-samples 8
--min-similarity 0.68
--candidate-similarity 0.55
--max-segment-chars 450
--max-segments-per-record 6
--test-fraction 0.20
--brand-min-records 80
```

GPU knobs:

```bash
--device cuda:1            # pin to one GPU; default is cuda when one is visible
--max-vram-gb 2            # hard cap on GPU memory (default 2 GiB)
--batch-size 1024          # override the batch; still clamped to the VRAM cap
--fp16                     # half precision on CUDA, ~2x faster, half the memory
```

Embedding is the only stage that uses the GPU, so `--max-vram-gb` caps the whole
run. The batch size is measured against that budget on a probe of the longest
texts in the corpus rather than guessed, the CUDA allocator is capped so the
process cannot quietly grow past it on a shared card, and an unlucky block that
still overflows halves its batch and retries instead of losing the run.

## What the script does

1. Sorts data by event time.
2. Normalizes text by removing URLs, mentions and obvious noise.
3. Splits long records into semantic-ish segments; limits the number of segments per record.
4. Embeds every segment.
5. Uses the oldest ~80% as the topic-discovery history.
6. Simulates incoming data using the most recent ~20%.
7. Assigns recent segments to existing topics by cosine similarity to topic centroids.
8. Clusters low-confidence/unassigned recent segments to discover new candidate topics.
9. Matches candidate clusters back to existing topics by centroid similarity; otherwise creates `EMERGING` topics.
10. Creates daily topic counts using `DISTINCT record_id`, not segment count.
11. Computes a simple HotScore from volume, growth, velocity, anomaly and persistence.

## Outputs

- `topic_hot_summary.csv` — topic-level hot/trending summary
- `topics.json` — topic metadata + centroids
- `topic_assignments.csv` — record/segment → topic assignments
- `topic_daily_counts.csv` — daily distinct-record counts
- `topic_timeseries.csv` — complete daily series
- `candidate_topic_decisions.csv` — existing-vs-new topic decisions
- `hdbscan_grid_search.csv` — optional parameter search
- `*_timeseries.png` — optional charts

## Important interpretation

This is a feasibility experiment. The HDBSCAN objective and HotScore weights are deliberately explainable heuristics. They should be tuned against manually reviewed topic quality and false-alert rate before being used in production.

## Stable topic IDs across runs

After a first run, reuse its `topics.json` when testing a refreshed clustering pass:

```bash
python reputation_topic_detection.py /path/to/twitter.csv \
  --out output_v2 \
  --previous-topics output/topics.json
```

Newly discovered cluster centroids are greedily matched to prior topic centroids. A match above `--candidate-similarity` keeps the previous topic ID; otherwise a new `T<n>` ID is created. In production, match within brand/scope first and resolve one-to-one so two new clusters cannot inherit the same topic ID.
