> **768-dim experimental copy:** the VRAM cap is off by default here
> (`MAX_VRAM_GB=0`). Where this guide says "default 2", read "default 0 (no
> cap)"; set `MAX_VRAM_GB=4` (or pass `--max-vram-gb 4`) to restore one.

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

The image is built on `nvidia/cuda:13.4.1-devel-ubuntu24.04` (full CUDA
toolkit, system Python 3.12); the host driver comes in through the NVIDIA
Container Toolkit and must support the base image's CUDA version. Ubuntu 22.04
tags will not work — they ship Python 3.10. Match the build to the driver on
the box (`nvidia-smi` shows it) and keep the two args paired:

| `CUDA_EXTRA` | `CUDA_BASE` | torch | Requires host driver |
| --- | --- | --- | --- |
| `cu130` *(default)* | `nvidia/cuda:13.4.1-devel-ubuntu24.04` | 2.14.0+cu130 | supports CUDA 13.4 |
| `cu126` | `nvidia/cuda:12.6.3-devel-ubuntu24.04` | 2.14.0+cu126 | >= 560 |
| `cpu` | `ubuntu:24.04` | 2.14.0+cpu | — (local sanity run) |

## Build

```bash
docker build -t reputation-topic-gpu .                      # cu130
docker build --build-arg CUDA_EXTRA=cu126 \
  --build-arg CUDA_BASE=nvidia/cuda:12.6.3-devel-ubuntu24.04 \
  -t reputation-topic-gpu:cu126 .
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

### docker compose: 200k base, then the 300k stream

`docker-compose.yml` defines the two-step run as services:

| Service | Runs | Writes |
|---|---|---|
| `base` | `reputation_topic_detection.py` on `twcs_subset_200k.csv`, LLM labels, fresh per-brand UMAP fit | `$OUT_DIR/base_200k/`, `$OUT_DIR/models/umap_v5_perbrand.*` |
| `stream` | `run_stream.py` over `stream300/chunk_1..6.csv`, 3-chunk sliding window, matched against `base_200k` | `$OUT_DIR/stream_1..6/`, `stream_summary.csv`, `stage_timing.csv` |
| `topics` | ad-hoc (`--help` by default) | — |

```bash
# once, on the GPU box
cp .env.example .env              # fill in api_key= (AI Router v1)
mkdir -p out_docker               # must exist and be writable by uid 1000
export DATA_DIR=/path/to/data     # holds twcs_subset_200k.csv and stream300/chunk_{1..6}.csv
export CUDA_EXTRA=cu130           # cu126 for a driver older than 580
docker compose build
docker compose run --rm topics /app/gpu_check.py

# the run
docker compose run --rm base      # step 1: 200k base registry
docker compose run --rm stream    # step 2: 300k stream against it
```

Knobs, exported in the shell or set in `.env`: `MAX_VRAM_GB` (default `2`),
`GPU_ID` (default `0`), `DATA_DIR` (default `../data`), `OUT_DIR` (default
`./out_docker`), `CUDA_EXTRA`.

Things that matter:

- **VRAM is capped in code, not by Docker.** Docker hands a container whole
  GPUs and has no memory limit for them; `--max-vram-gb` (below) is what holds
  the process at 2 GiB. Check the `VRAM cap recalibrated after warm-up ... (NVML)`
  line in the log and `peak_process_vram_mib` in `run_metadata.json`.
- **Run `base` before `stream`.** `stream` reads `base_200k/topics.json` and the
  UMAP model `base` fitted. There is deliberately no `depends_on`: `compose run
  stream` would otherwise re-run the base every time.
- **The two buffers stay separate.** The base's leftover records run to 30 Nov;
  sharing its buffer would feed them into stream batch 1 (1-14 Oct).
- **The embedding cache volume starts empty and must stay on.** With
  `--window 3` every chunk sits in three windows; the per-chunk cache is what
  makes each one embedded once. `docker volume rm reputation_topic_gpu_topic-cache`
  forces a clean re-embed.
- **Offline model.** The multilingual model is baked into the image and
  `HF_HUB_OFFLINE=1` is set, so no run needs Hugging Face access. The AI
  Router does need network access for `--label-method llm`.
- Compose reserves an NVIDIA device, so it refuses to start on a machine
  without one. For a CPU-only sanity run use plain `docker run` with a
  `CUDA_EXTRA=cpu` image and `--device cpu`.

## Tuning the GPU pass

- `--max-vram-gb` (default 2) caps the **whole process** -- the number
  `nvidia-smi` shows, not just torch's allocator. After the model loads, a
  warm-up encode pulls in every kernel and cuBLAS/cuDNN workspace; that
  overhead is then measured (per process, through NVML) and the caching
  allocator is capped at cap - overhead - 96 MiB. While embedding, the process
  footprint is re-checked every 5k texts and the batch halves if it is ever
  over. `run_metadata.json` records `peak_process_vram_mib` (what nvidia-smi
  saw), `vram_overhead_mib` and the allocator peak. Without `nvidia-ml-py`
  the overhead falls back to a device-wide estimate, which errs smaller.
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

`--label-method llm` calls the QuestionPro AI Router and needs credentials,
passed at run time and never baked into the image. The default backend is AI
Router v1 (`airouter_v1_client.py`, prompt sent inline), which reads `api_key`,
`base_url` and `use_case` -- the keys in `.env.example`. Compose passes `.env`
to every service; with plain docker use `--env-file .env`.

The AiRouterV2 backend (`--label-llm-backend air2`) reads `AIR2_QP_OAUTH_HOST`,
`AIR2_S2S_CLIENT_ID`, `AIR2_S2S_CLIENT_SECRET`, `AIR2_CONSUMER_ID` and
`AIR2_USER_ID` from the same file.

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
