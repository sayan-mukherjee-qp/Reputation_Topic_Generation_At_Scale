# syntax=docker/dockerfile:1.7
#
# GPU image for the reputation topic pipeline.
#
# Built on the nvidia/cuda devel image (full CUDA toolkit, nvcc included) on
# Ubuntu 24.04, whose system Python is 3.12. Ubuntu 22.04 tags will not work:
# they ship Python 3.10 and the project needs 3.12. The host driver is exposed
# by the NVIDIA Container Toolkit (`--gpus all`) and must support the base
# image's CUDA version -- the container refuses to start otherwise.
#
# Pick the CUDA build to match the host driver, and keep the two args paired:
#   CUDA_EXTRA=cu130 CUDA_BASE=nvidia/cuda:13.4.1-devel-ubuntu24.04  (default)
#   CUDA_EXTRA=cu126 CUDA_BASE=nvidia/cuda:12.6.3-devel-ubuntu24.04  driver >= 560
#   CUDA_EXTRA=cpu   CUDA_BASE=ubuntu:24.04                           no GPU, sanity run

ARG CUDA_EXTRA=cu130
ARG CUDA_BASE=nvidia/cuda:13.4.1-devel-ubuntu24.04

FROM ghcr.io/astral-sh/uv:0.8 AS uv

# ---------------------------------------------------------------- builder ---
# Same base as the runtime, so the venv's python symlink resolves in both.
FROM ${CUDA_BASE} AS builder
ARG CUDA_EXTRA

RUN apt-get update \
 && apt-get install -y --no-install-recommends python3.12 ca-certificates \
 && rm -rf /var/lib/apt/lists/*
COPY --from=uv /uv /usr/local/bin/uv

# The CUDA torch wheel is ~530 MB and its NVIDIA deps another few GB; uv's
# default 30s HTTP timeout gives up on them over an ordinary connection.
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never \
    UV_PYTHON=/usr/bin/python3.12 \
    UV_HTTP_TIMEOUT=900 \
    UV_CONCURRENT_DOWNLOADS=4

WORKDIR /app

# Dependency layer on its own: it only rebuilds when the lockfile moves.
COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-dev --extra "${CUDA_EXTRA}"

# ---------------------------------------------------------------- runtime ---
FROM ${CUDA_BASE} AS runtime

# libgomp is the OpenMP runtime scikit-learn, numba and torch all link against.
RUN apt-get update \
 && apt-get install -y --no-install-recommends python3.12 libgomp1 ca-certificates \
 && rm -rf /var/lib/apt/lists/*

# HF_HOME deliberately sits outside /cache: /cache is a mount point, and a
# volume mounted there would otherwise hide the model baked in below.
ENV PATH=/app/.venv/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    HF_HOME=/opt/hf-cache \
    MPLCONFIGDIR=/tmp/matplotlib \
    OMP_NUM_THREADS=8 \
    TOKENIZERS_PARALLELISM=false

# Ubuntu 24.04 images already ship an `ubuntu` user holding uid 1000.
RUN userdel --remove ubuntu 2>/dev/null || true \
 && useradd --create-home --uid 1000 app \
 && mkdir -p /app /data /out /cache/embed /opt/hf-cache \
 && chown -R app:app /out /cache /opt/hf-cache

WORKDIR /app
COPY --from=builder /app/.venv /app/.venv
# --chmod: the container runs as uid 1000 while COPY keeps the host's mode
# bits under root ownership, so a checkout with 600/640 files ("Permission
# denied" on /app/gpu_check.py) would otherwise be unreadable to it.
COPY --chmod=0644 *.py README.md DOCKER.md ./

USER app

# Bake the embedding model in so a run needs no network access. This is the
# multilingual model every run in this repo uses (docker-compose.yml passes it
# too); the corpus mixes English, Spanish, French, German and Japanese.
ARG EMBEDDING_MODEL=sentence-transformers/paraphrase-multilingual-mpnet-base-v2
# Kept as ENV so gpu_check.py loads the model that is actually baked in.
ENV EMBEDDING_MODEL=${EMBEDDING_MODEL}
RUN python -c "\
from sentence_transformers import SentenceTransformer; \
SentenceTransformer('${EMBEDDING_MODEL}', device='cpu')"

# `docker run ... IMAGE <script.py> [args]` -- the default shows the CLI.
ENTRYPOINT ["python"]
CMD ["/app/reputation_topic_detection.py", "--help"]
