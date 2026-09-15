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

The default embedding model is `sentence-transformers/all-MiniLM-L6-v2`. The Docker image bakes it in; a local run downloads it on first use.

## Run

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
--batch-size 1024          # default 512 on CUDA, 64 on CPU
--fp16                     # half precision on CUDA, ~2x faster embedding
```

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
