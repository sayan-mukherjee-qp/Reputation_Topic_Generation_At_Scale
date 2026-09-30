# Dashboard

A web UI for the two-step run in `docker-compose.yml`: a **Start** button runs
the 200k `base` registry and then the 300k `stream`, shows every pipeline stage
live with its timing, and, as each run lands, shows its emerging topics and
analytics.

- **Backend:** FastAPI (`dashboard/server.py`). It reads the `base` and `stream`
  commands from `docker-compose.yml`, so it always runs exactly the documented
  configuration, and follows progress from the pipeline's own log
  (`[timing] <stage>` lines, embedding batches, brand pools).
- **Frontend:** React + TypeScript + Recharts (`dashboard/web`), built by Vite
  into `dashboard/web/dist`, which the backend serves.

## Run it in Docker (GPU box)

```bash
docker compose build
docker compose up dashboard          # http://127.0.0.1:8765 on the GPU box
```

The port is published on `127.0.0.1` only, because the dashboard can start GPU
runs. From your laptop:

```bash
ssh -L 8765:127.0.0.1:8765 you@gpu-box   # then open http://127.0.0.1:8765
```

Inside the container the dashboard spawns the pipeline in the same image, with
the same `/data`, `/out`, `/cache` mounts and `.env`, so nothing else is needed.
`DASHBOARD_PORT` changes the host port.

## Run it without Docker

```bash
uv sync --extra cu130 --extra dashboard      # or --extra cpu
(cd dashboard/web && npm ci && npm run build)
.venv/bin/python -m dashboard                # http://127.0.0.1:8765
```

Paths default to the ones compose uses: `DATA_DIR` (`../data`), `OUT_DIR`
(`./out_docker`), `CACHE_DIR` (`./embed_cache` when it isn't a path), read from
the environment, then `.env`. Override them with flags:

```bash
.venv/bin/python -m dashboard --out-dir out/exp_control   # browse an existing run
.venv/bin/python -m dashboard --device cpu                 # no GPU: override --device=cuda
```

Preflight blocks **Start** if the compose file asks for `--device=cuda` and no
GPU is visible; `--device cpu` is the way through on a laptop.

## Frontend development

```bash
.venv/bin/python -m dashboard                 # API on :8765
cd dashboard/web && npm run dev               # UI on :5173, /api proxied to :8765
```

`DASHBOARD_API=http://host:port npm run dev` points the proxy elsewhere.

## What the options do

- **Continue into the 300k stream:** after the base run succeeds, runs the
  `stream` service (`run_stream.py --echo`, so batch logs stream through).
- **Fresh start:** before each phase, deletes that phase's rolling `--buffer`
  and, for the base run, the `--umap-model` artifact, so leftovers from a
  previous run can't leak into this one. The embedding cache is kept: it only
  skips work, never changes a result.

A run is one process group. **Stop** sends it SIGTERM (SIGKILL after 15 s), and
shutting the dashboard down stops a run in progress. The full output of each
phase is written to `$OUT_DIR/dashboard_logs/`.

## API

| Method | Path | |
|---|---|---|
| GET | `/api/preflight` | inputs, interpreter, GPU, LLM key |
| GET | `/api/job` · `/api/job/events` | job snapshot, and the same as Server-Sent Events |
| POST | `/api/job/start` | `{"include_stream": true, "fresh": true}` |
| POST | `/api/job/stop` | |
| GET | `/api/runs` | one card per finished run in `OUT_DIR` |
| GET | `/api/runs/{name}` | topics, daily series, event recall for one run |
| GET | `/api/runs/{name}/topics/{id}` | example records for one topic |
| GET | `/api/stream` | `stream_summary.csv` |

Interactive docs: `/api/docs`.
