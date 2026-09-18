# Containerized GPU run

A self-contained copy of the prototype, packaged with `uv` and built to run on a
cloud GPU box. The pipeline itself is unchanged apart from four flags (`--device`,
`--max-vram-gb`, `--batch-size`, `--fp16`) that let the embedding stage use the
GPU instead of a CPU-sized batch of 64, within a memory budget you set.

## What runs on the GPU

| Stage | Device | Note |
| --- | --- | --- |
| Sentence-Transformer embedding | **GPU** | The dominant cost on a 200k-record run. Auto-detects CUDA; the batch is sized to fit `--max-vram-gb`. |
| Acknowledgement-anchor similarity | **GPU** | Same model, a handful of vectors. |
| PCA / UMAP reduction | CPU | `umap-learn` and scikit-learn are CPU-only. See *Optional: cuML* below. |
| HDBSCAN, c-TF-IDF, scoring | CPU | scikit-learn / pandas. |

So the GPU buys you the embedding pass. On a 200k-segment corpus that is the
difference between tens of minutes and roughly a minute; the clustering stages
stay CPU-bound, so pick a cloud instance with a decent core count too, not just
a big GPU.

## Choosing the CUDA build

The torch wheels carry their own CUDA runtime, so the image needs no
`nvidia/cuda` base — only the host driver via the NVIDIA Container Toolkit.
Match the build to the driver on the box (`nvidia-smi` shows it):

| Build arg | torch | Requires host driver |
| --- | --- | --- |
| `CUDA_EXTRA=cu130` *(default)* | 2.14.0+cu130 | >= 580 |
| `CUDA_EXTRA=cu126` | 2.14.0+cu126 | >= 560 |
| `CUDA_EXTRA=cpu` | 2.14.0+cpu | — (local sanity run) |

## Build

```bash
docker build -t reputation-topic-gpu .                      # cu130
docker build --build-arg CUDA_EXTRA=cu126 -t reputation-topic-gpu:cu126 .
```

The CUDA build pulls roughly 5 GB of wheels, so the first one is slow. uv's
default 30s HTTP timeout is not enough for the 530 MB torch wheel, so the
builder stage raises it to 900s (`UV_HTTP_TIMEOUT`). If a download still times
out on a slow link, just re-run the build — the BuildKit uv cache mount keeps
what already arrived, so a retry resumes instead of starting over.

The default embedding model is baked into the image, so a run needs no network
access. To bake a different one:

```bash
docker build --build-arg EMBEDDING_MODEL=BAAI/bge-base-en-v1.5 -t reputation-topic-gpu .
```

## Verify the GPU is visible

```bash
docker run --rm --gpus all reputation-topic-gpu /app/gpu_check.py
```

It prints the torch/CUDA versions and every visible device, then runs a matmul
and an embedding call. A non-zero exit means the run would silently fall back to
CPU — usually a missing `--gpus all` or a host driver older than the build.

## Run

```bash
docker run --rm --gpus all \
  -v /path/to/data:/data:ro \
  -v "$PWD/out":/out \
  -v topic-cache:/cache \
  --shm-size=2g \
  reputation-topic-gpu \
  /app/reputation_topic_detection.py /data/twcs_subset_200k.csv \
    --out /out/run1 \
    --embed-cache /cache/embed \
    --max-vram-gb 2 \
    --reducer umap \
    --plots
```

Mount points:

- `/data` — input CSVs, read-only
- `/out` — everything the run writes
- `/cache` — the embedding cache (`/cache/embed`). Keep it on a named volume so
  a re-run over the same model and text skips the embedding pass entirely.

The Hugging Face cache lives at `/opt/hf-cache` inside the image, *not* under
`/cache`, so mounting a volume at `/cache` cannot hide the baked-in model. To
use a different model without rebuilding, mount your own cache over it
(`-v hf-cache:/opt/hf-cache`); the container then needs network access on the
first run.

Any of the other scripts run the same way, since the entrypoint is `python`:

```bash
docker run --rm -v "$PWD/out":/out reputation-topic-gpu /app/event_recall.py ...
docker run --rm -it --entrypoint bash reputation-topic-gpu       # poke around
```

### docker compose

```bash
DATA_DIR=/path/to/data docker compose up --build
CUDA_EXTRA=cu126 DATA_DIR=/path/to/data docker compose up --build   # older driver
```

The service reserves an NVIDIA device, so compose will refuse to start on a
machine without one — for a CPU-only sanity run, use plain `docker run` with a
`CUDA_EXTRA=cpu` image instead of compose.

Override the command for a different run:

```bash
docker compose run --rm topics /app/reputation_topic_detection.py \
  /data/twcs_subset_150k.csv --out /out/run2 --embed-cache /cache/embed
```

## Tuning the GPU pass

- `--max-vram-gb` (default 2) is a hard ceiling, not a hint. The CUDA context
  is measured and subtracted from it, and the caching allocator is capped at
  what is left, so the process cannot exceed the number you give even if the
  batch estimate is wrong. `run_metadata.json` records the peak actually used.
  On a card you have to yourself, raise it — the budget is what sets the batch
  size, so a bigger budget is a bigger batch is a faster pass.
- `--batch-size` is no longer a guess you have to tune. Leave it off and the
  batch is measured: a probe on the longest texts in the corpus gives the cost
  per sample, and the batch is whatever fits the budget. Pass it only to force
  a smaller value; a larger one is clamped to what fits.
- `--fp16` halves the embedding time on any recent card, and halves both the
  weight and activation memory — which on a 2 GiB budget roughly doubles the
  batch that fits. It changes the embeddings slightly; precision is part of the
  embedding-cache key, so an fp16 and an fp32 run can safely share one
  `--embed-cache` directory without reading back each other's vectors.
- A budget too small for the model fails immediately with what it would need,
  rather than dying partway through the embedding pass.
- `--device cuda:1` pins the run to one GPU on a multi-GPU box. The pipeline is
  single-process, so to use several GPUs run several jobs, one per device.
- `OMP_NUM_THREADS` is set to 8 in the image. On a box with many cores, raise
  it (`-e OMP_NUM_THREADS=32`) — HDBSCAN and UMAP are the CPU-bound stages.

## LLM topic labelling

`--label-method llm` calls the QuestionPro AI Router and needs credentials.
Pass them as environment variables, never baked into the image:

```bash
docker run --rm --gpus all --env-file air2.env ... 
```

with `air2.env` holding `AIR2_QP_OAUTH_HOST`, `AIR2_S2S_CLIENT_ID`,
`AIR2_S2S_CLIENT_SECRET`, `AIR2_CONSUMER_ID`, `AIR2_USER_ID`. Compose already
forwards these from the host environment.

## Optional: cuML for GPU clustering

UMAP and HDBSCAN stay on the CPU here. RAPIDS cuML can accelerate both with no
code change, but it is a large dependency with its own numpy/scipy pins, so it
is deliberately not in the image. To try it, add `cuml-cu13` to the
dependencies, rebuild, and run the script under the accelerator:

```bash
python -m cuml.accel /app/reputation_topic_detection.py ...
```

Treat it as an experiment and compare topic output against a CPU run before
trusting it — the GPU implementations are not bit-identical to scikit-learn's.

## Dependency management

`pyproject.toml` + `uv.lock` are the source of truth; `requirements-original.txt`
is kept only as a record of the CPU prototype's loose pins. To change a
dependency:

```bash
uv lock --upgrade-package sentence-transformers
docker build -t reputation-topic-gpu .
```

`uv sync --locked` in the build fails loudly if the lockfile is stale, so the
image can never drift from the lock.
