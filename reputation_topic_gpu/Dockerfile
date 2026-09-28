# syntax=docker/dockerfile:1.7
#
# GPU image for the reputation topic pipeline.
#
# The torch wheels ship their own CUDA runtime, so this does not need an
# nvidia/cuda base image -- only the host driver, exposed by the NVIDIA
# Container Toolkit (`--gpus all`). That keeps the image a few GB smaller.
#
# Pick the CUDA build to match the host driver:
#   --build-arg CUDA_EXTRA=cu130   (default) driver >= 580
#   --build-arg CUDA_EXTRA=cu126             driver >= 560
#   --build-arg CUDA_EXTRA=cpu               no GPU, for a local sanity run

ARG CUDA_EXTRA=cu130

# ---------------------------------------------------------------- builder ---
FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim AS builder
ARG CUDA_EXTRA

# The CUDA torch wheel is ~530 MB and its NVIDIA deps another few GB; uv's
# default 30s HTTP timeout gives up on them over an ordinary connection.
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never \
    UV_HTTP_TIMEOUT=900 \
    UV_CONCURRENT_DOWNLOADS=4

WORKDIR /app

# Dependency layer on its own: it only rebuilds when the lockfile moves.
COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-dev --extra "${CUDA_EXTRA}"

# ---------------------------------------------------------------- runtime ---
FROM python:3.12-slim-bookworm AS runtime

# libgomp is the OpenMP runtime scikit-learn, numba and torch all link against.
RUN apt-get update \
 && apt-get install -y --no-install-recommends libgomp1 ca-certificates \
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

RUN useradd --create-home --uid 1000 app \
 && mkdir -p /app /data /out /cache/embed /opt/hf-cache \
 && chown -R app:app /out /cache /opt/hf-cache

WORKDIR /app
COPY --from=builder /app/.venv /app/.venv
COPY *.py README.md DOCKER.md ./

USER app

# Bake the embedding model in so a run needs no network access. This is the
# multilingual model every run in this repo uses (docker-compose.yml passes it
# too); the corpus mixes English, Spanish, French, German and Japanese.
ARG EMBEDDING_MODEL=sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
RUN python -c "\
from sentence_transformers import SentenceTransformer; \
SentenceTransformer('${EMBEDDING_MODEL}', device='cpu')"

# `docker run ... IMAGE <script.py> [args]` -- the default shows the CLI.
ENTRYPOINT ["python"]
CMD ["/app/reputation_topic_detection.py", "--help"]
