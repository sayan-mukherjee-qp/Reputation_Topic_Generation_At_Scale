"""FastAPI backend: start/stop the pipeline, stream its progress, serve analytics.

    GET  /api/datasets               the input sets a run can use (slice, full)
    GET  /api/preflight              inputs, interpreter, GPU, LLM key
    GET  /api/job                    current job snapshot
    GET  /api/job/events             the same snapshot as Server-Sent Events
    POST /api/job/start              {"dataset": "slice", "include_stream": true, "fresh": true}
    POST /api/job/stop
    GET  /api/runs                   one card per finished run in OUT_DIR
    GET  /api/runs/{name}            topics, daily series, events for one run
    GET  /api/runs/{name}/topics/{id}  example tweets for one topic
    GET  /api/runs/{name}/records    paginated records with their topics as tags
    GET  /api/stream                 stream_summary.csv

The preflight and runs endpoints take ?dataset=slice|full (default full); each
dataset has its own inputs and output directory. Everything else serves the
built React app from dashboard/web/dist.
"""
from __future__ import annotations

import asyncio
import json
import os
import shutil
import subprocess
import threading
from pathlib import Path
from urllib.parse import urlparse

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from . import analytics
from .pipeline import (DATASETS, JobManager, Settings, arg_list, arg_value, read_dotenv,
                       service_command)

WEB_DIST = Path(__file__).resolve().parent / "web" / "dist"


class StartRequest(BaseModel):
    dataset: str = "full"
    include_stream: bool = True
    fresh: bool = True


_rows_cache: dict[str, tuple[float, int]] = {}
_rows_lock = threading.Lock()


def count_rows(path: Path) -> int | None:
    """Records in a CSV (texts span lines, so parse it); cached on mtime."""
    try:
        mtime = path.stat().st_mtime
    except OSError:
        return None
    with _rows_lock:
        hit = _rows_cache.get(str(path))
    if hit and hit[0] == mtime:
        return hit[1]
    import pandas as pd
    n = len(pd.read_csv(path, usecols=[0]))
    with _rows_lock:
        _rows_cache[str(path)] = (mtime, n)
    return n


def _gpu() -> dict:
    if not shutil.which("nvidia-smi"):
        if Path("/dev/nvidiactl").exists():      # a container without the utility tools
            return {"ok": True, "detail": "NVIDIA device present (nvidia-smi not installed)"}
        return {"ok": False, "detail": "nvidia-smi not found: embedding will run on CPU"}
    try:
        r = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total",
                            "--format=csv,noheader"], capture_output=True, text=True, timeout=10)
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"ok": False, "detail": f"nvidia-smi failed: {exc}"}
    gpus = [g.strip() for g in r.stdout.splitlines() if g.strip()]
    if r.returncode != 0 or not gpus:
        return {"ok": False, "detail": (r.stderr or r.stdout).strip()[:300] or "no GPU visible"}
    return {"ok": True, "detail": "; ".join(gpus)}


def create_app(settings: Settings) -> FastAPI:
    app = FastAPI(title="Reputation topic dashboard", docs_url="/api/docs",
                  openapi_url="/api/openapi.json")
    jobs = JobManager(settings)
    app.state.jobs = jobs

    def scoped(dataset: str) -> Settings:
        if dataset not in DATASETS:
            raise HTTPException(404, f"unknown dataset {dataset!r}; one of {', '.join(DATASETS)}")
        return settings.for_dataset(DATASETS[dataset])

    def run_path(name: str, dataset: str) -> Path:
        # Only names the listing produced, so a request can never walk the disk.
        out = scoped(dataset).out_dir
        for d in analytics.run_dirs(out):
            if d.name == name:
                return d
        raise HTTPException(404, f"no finished run named {name!r} in {out}")

    def inputs(s: Settings) -> dict:
        base = service_command(s, "base")
        stream = service_command(s, "stream")
        base_csv = Path(next(a for a in base[2:] if not a.startswith("--")))
        chunks = [Path(c) for c in arg_list(stream, "--chunks")]
        return {"base_csv": base_csv, "chunks": chunks,
                "base_rows": count_rows(base_csv) if base_csv.exists() else None,
                "stream_rows": (sum(count_rows(c) or 0 for c in chunks)
                                if chunks and all(c.exists() for c in chunks) else None)}

    @app.get("/api/datasets")
    def datasets() -> dict:
        out = []
        for ds in DATASETS.values():
            s = settings.for_dataset(ds)
            common = {"key": ds.key, "label": ds.label, "runnable": ds.runnable,
                      "group": ds.group, "description": ds.description}
            if not ds.runnable:
                out.append({**common, **analytics.saved_set_info(s.out_dir)})
                continue
            try:
                i = inputs(s)
            except Exception as exc:              # noqa: BLE001 -- report, don't crash
                out.append({**common, "available": False,
                            "detail": f"{type(exc).__name__}: {exc}"})
                continue
            out.append({
                **common, "available": i["base_csv"].exists(),
                "base_csv": i["base_csv"].name, "base_rows": i["base_rows"],
                "stream_chunks": len(i["chunks"]), "stream_rows": i["stream_rows"],
                "stream_dir": i["chunks"][0].parent.name if i["chunks"] else None,
                "out_dir": str(s.out_dir),
                "detail": None if i["base_csv"].exists() else
                f"{i['base_csv']} is missing" + (": run make_slice.py" if ds.key == "slice" else ""),
            })
        return {"datasets": out}

    @app.get("/api/preflight")
    def preflight(dataset: str = Query("full")) -> dict:
        s = scoped(dataset)
        checks = []

        def add(key, label, ok, detail, level="error"):
            checks.append({"key": key, "label": label, "ok": bool(ok), "detail": detail,
                           "level": "ok" if ok else level})

        add("compose", "Compose file", settings.compose_file.exists(), str(settings.compose_file))
        add("python", "Pipeline interpreter", settings.python.exists(), str(settings.python))
        try:
            base = service_command(s, "base")
            stream = service_command(s, "stream")
        except Exception as exc:                  # noqa: BLE001 -- report, don't crash
            add("commands", "Pipeline commands", False, f"{type(exc).__name__}: {exc}")
            return {"checks": checks, "settings": _settings_view(s)}
        base_csv = Path(next(a for a in base[2:] if not a.startswith("--")))
        rows = count_rows(base_csv) if base_csv.exists() else None
        add("base_input", "Base input" + (f" ({rows:,} records)" if rows else ""),
            base_csv.exists(), str(base_csv) if base_csv.exists() else
            f"{base_csv} is missing" + (" -- run make_slice.py" if dataset == "slice" else ""))
        chunks = [Path(c) for c in arg_list(stream, "--chunks")]
        missing = [c.name for c in chunks if not c.exists()]
        add("stream_input", f"Stream chunks ({len(chunks)})", not missing,
            f"missing: {', '.join(missing)}" if missing
            else (str(chunks[0].parent) if chunks else "no --chunks in the stream service"),
            level="warning")
        try:
            s.out_dir.mkdir(parents=True, exist_ok=True)
            writable = os.access(s.out_dir, os.W_OK)
        except OSError:
            writable = False
        add("out_dir", "Output directory writable", writable, str(s.out_dir))
        gpu = _gpu()
        device = arg_value(base, "--device") or "auto"
        if device.startswith("cuda") and not gpu["ok"]:
            add("gpu", "GPU", False, f"the run asks for --device={device}, but {gpu['detail']}. "
                "Start the dashboard with --device cpu to run on CPU.")
        else:
            add("gpu", f"GPU (--device={device})", gpu["ok"], gpu["detail"], level="warning")
        env = {**read_dotenv(settings.app_dir / ".env"), **os.environ}
        key = env.get("AIROUTER_API_KEY") or env.get("api_key")
        add("llm", "LLM labelling key", key,
            "AI Router key found" if key else
            "no api_key in .env: topics fall back to c-TF-IDF keyword labels", level="warning")
        if arg_value(base, "--label-method") == "llm":
            url = env.get("AIROUTER_BASE_URL") or env.get("base_url") or "https://airouter-api.questionpro.com"
            u = urlparse(url)
            bad = (u.scheme not in ("http", "https") or not u.netloc or "#" in url
                   or any(ch.isspace() for ch in url))
            # A malformed URL fails every labelling call and the run quietly
            # falls back to keyword labels, so it blocks the start.
            add("llm_url", "LLM endpoint", not bad,
                f"base_url={url!r} is malformed (a comment or text merged into the line?); "
                "every labelling call would fail. Fix it in .env." if bad else url)
        return {"checks": checks, "settings": _settings_view(s)}

    @app.get("/api/job")
    def job() -> dict:
        return jobs.snapshot()

    @app.get("/api/job/events")
    async def job_events(request: Request) -> StreamingResponse:
        async def gen():
            seen, idle = -1, 0.0
            while not await request.is_disconnected():
                if jobs.version != seen:
                    seen = jobs.version
                    idle = 0.0
                    yield f"event: state\ndata: {json.dumps(jobs.snapshot())}\n\n"
                elif idle >= 15:
                    idle = 0.0
                    yield ": keep-alive\n\n"
                await asyncio.sleep(0.5)
                idle += 0.5
        return StreamingResponse(gen(), media_type="text/event-stream",
                                 headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

    @app.post("/api/job/start")
    def start(req: StartRequest) -> dict:
        try:
            scoped(req.dataset)
            jobs.start(req.include_stream, req.fresh, req.dataset)
        except RuntimeError as exc:
            raise HTTPException(409, str(exc))
        return jobs.snapshot()

    @app.post("/api/job/stop")
    def stop() -> dict:
        if not jobs.stop():
            raise HTTPException(409, "no run in progress")
        return jobs.snapshot()

    @app.get("/api/runs")
    def runs(dataset: str = Query("full")) -> dict:
        out = scoped(dataset).out_dir
        return {"out_dir": str(out), "runs": analytics.list_runs(out)}

    @app.get("/api/runs/{name}")
    def run(name: str, dataset: str = Query("full")) -> dict:
        path = run_path(name, dataset)
        # Warm the records index (seconds on a full run) before anyone opens Records.
        threading.Thread(target=analytics.warm_records, args=(path,), daemon=True).start()
        return analytics.run_overview(path)

    @app.get("/api/runs/{name}/topics/{topic_id}")
    def topic(name: str, topic_id: str, dataset: str = Query("full")) -> dict:
        return {"topic_id": topic_id,
                "samples": analytics.topic_samples(run_path(name, dataset), topic_id)}

    @app.get("/api/runs/{name}/records")
    def records(name: str, dataset: str = Query("full"), page: int = Query(1, ge=1),
                page_size: int = Query(25, ge=1, le=200), brand: str | None = None,
                topic: str | None = None, q: str | None = None,
                status: str = Query("all", pattern="^(all|assigned|unassigned|emerging)$"),
                order: str = Query("newest", pattern="^(newest|oldest)$")) -> dict:
        return analytics.records_page(run_path(name, dataset), page, page_size, brand or None,
                                      topic or None, (q or "").strip() or None, status,
                                      oldest_first=order == "oldest")

    @app.get("/api/stream")
    def stream(dataset: str = Query("full")) -> dict:
        return {"rows": analytics.stream_table(scoped(dataset).out_dir)}

    @app.on_event("shutdown")
    def _stop_job() -> None:
        # Don't leave a GPU run orphaned when the dashboard goes away.
        jobs.stop()

    if WEB_DIST.is_dir():
        app.mount("/assets", StaticFiles(directory=WEB_DIST / "assets"), name="assets")

        @app.get("/{path:path}", include_in_schema=False)
        def spa(path: str):
            # The app shell is for page routes only. An unknown API route must
            # fail as JSON, or the UI would try to parse this HTML as data.
            if path == "api" or path.startswith("api/"):
                raise HTTPException(404, f"no API route /{path} (is the server older than the UI? "
                                         "restart the dashboard)")
            f = (WEB_DIST / path).resolve()
            if path and f.is_file() and WEB_DIST in f.parents:
                return FileResponse(f)
            return FileResponse(WEB_DIST / "index.html")
    else:
        @app.get("/", include_in_schema=False)
        def no_build():
            return JSONResponse({"detail": "Frontend not built: cd dashboard/web && npm ci && "
                                           "npm run build (or use the Vite dev server)."}, 503)
    return app


def _settings_view(s: Settings) -> dict:
    return {"app_dir": str(s.app_dir), "data_dir": str(s.data_dir), "out_dir": str(s.out_dir),
            "cache_dir": str(s.cache_dir), "python": str(s.python), "device": s.device or ""}
