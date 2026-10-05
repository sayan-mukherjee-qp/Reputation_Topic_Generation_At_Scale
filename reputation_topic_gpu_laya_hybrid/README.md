# Reputation Topic Discovery — EXPERIMENT: cosine + Laya hybrid matching for stream records

> **Experimental copy of `../reputation_topic_gpu`.** The cold-start base run
> (no `--previous-topics`) is the normal pipeline, unchanged. Every stream
> batch changes (`--matcher hybrid`, the default in this copy):
>
> - **No re-discovery.** The previous batch's registry is carried as it
>   stands: every topic starts DORMANT and revives when records land on it,
>   and is retired after `--topic-max-age-days`. The window's history chunks
>   keep the decisions earlier batches made for them, so HotScore's baseline
>   is filed the same way as the new data.
> - **Matching.** Cosine similarity files the clear cases: best >=
>   `--hybrid-high` (0.60) and ahead of the runner-up by `--hybrid-margin`
>   (0.05) is filed exactly as the original pipeline would; best <
>   `--hybrid-low` (0.45) goes straight to the bucket. Laya decides the
>   borderline band between (about 28% of segments on the 300k stream).
>   Laya sees the record's `--laya-shortlist` (12) nearest topics of its own
>   brand plus two exits: `other` sends the record to the bucket, `no_issue`
>   leaves it unassigned and out of the bucket. A topic is accepted at
>   `answer_confidence >= --laya-min-confidence` (0.50), a second topic at
>   probability >= `--laya-secondary-prob` (0.30). Suppressed and "Unclear
>   topic" topics are not offered. Laya's options share a ~256-token budget,
>   so it cannot be shown a brand's whole registry (40-90 topics at 200k).
> - **Centroids follow the matches** (`--centroid-update running`). A topic
>   that drifts below `--label-reuse-similarity` (0.92) from where it was
>   named is sent back to the LLM for a new label.
> - **Scheduled clustering.** The bucket (this batch's misses plus the rolling
>   buffer) is clustered per brand once it holds `--bucket-trigger-size`
>   segments or its oldest has waited `--bucket-max-wait-days` (100 and 3 in
>   `run_stream.py`). Candidate, recovery and micro passes are unchanged.
>
> **Laya** is the open-weight decision model served by `laya-serve`
> (https://github.com/NandhaKishorM/laya). Put `LAYA_BASE_URL` (and
> `LAYA_API_KEY` if the server has one) in `.env`; see `.env.example`. From
> inside the container a server on this machine is
> `http://host.docker.internal:8000`. A batch checks `/health` before it
> embeds anything and aborts if more than `--laya-max-failure-rate` (5%) of
> its questions go unanswered; below that, unanswered records are filed by
> cosine and marked `EXISTING_FALLBACK`. `laya_client.py` run on its own is a
> smoke test of the endpoint.
>
> `--matcher centroid` restores the original stream behaviour exactly.

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
