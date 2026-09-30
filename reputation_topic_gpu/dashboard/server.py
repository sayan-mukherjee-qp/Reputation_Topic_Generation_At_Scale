"""FastAPI backend: start/stop the pipeline, stream its progress, serve analytics.

    GET  /api/preflight              inputs, interpreter, GPU, LLM key
    GET  /api/job                    current job snapshot
    GET  /api/job/events             the same snapshot as Server-Sent Events
    POST /api/job/start              {"include_stream": true, "fresh": true}
    POST /api/job/stop
    GET  /api/runs                   one card per finished run in OUT_DIR
    GET  /api/runs/{name}            topics, daily series, events for one run
    GET  /api/runs/{name}/topics/{id}  example tweets for one topic
    GET  /api/stream                 stream_summary.csv

Everything else serves the built React app from dashboard/web/dist.
"""
from __future__ import annotations

import asyncio
import json
import os
import shutil
import subprocess
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from . import analytics
from .pipeline import JobManager, Settings, arg_list, arg_value, read_dotenv, service_command

WEB_DIST = Path(__file__).resolve().parent / "web" / "dist"


class StartRequest(BaseModel):
    include_stream: bool = True
    fresh: bool = True


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

    def run_path(name: str) -> Path:
        # Only names the listing produced, so a request can never walk the disk.
        for d in analytics.run_dirs(settings.out_dir):
            if d.name == name:
                return d
        raise HTTPException(404, f"no finished run named {name!r} in {settings.out_dir}")

    @app.get("/api/preflight")
    def preflight() -> dict:
        checks = []

        def add(key, label, ok, detail, level="error"):
            checks.append({"key": key, "label": label, "ok": bool(ok), "detail": detail,
                           "level": "ok" if ok else level})

        add("compose", "Compose file", settings.compose_file.exists(), str(settings.compose_file))
        add("python", "Pipeline interpreter", settings.python.exists(), str(settings.python))
        try:
            base = service_command(settings, "base")
            stream = service_command(settings, "stream")
        except Exception as exc:                  # noqa: BLE001 -- report, don't crash
            add("commands", "Pipeline commands", False, f"{type(exc).__name__}: {exc}")
            return {"checks": checks, "settings": _settings_view(settings)}
        base_csv = Path(next(a for a in base[2:] if not a.startswith("--")))
        add("base_input", "Base input (200k)", base_csv.exists(), str(base_csv))
        chunks = [Path(c) for c in arg_list(stream, "--chunks")]
        missing = [c.name for c in chunks if not c.exists()]
        add("stream_input", f"Stream chunks ({len(chunks)})", not missing,
            f"missing: {', '.join(missing)}" if missing
            else (str(chunks[0].parent) if chunks else "no --chunks in the stream service"),
            level="warning")
        try:
            settings.out_dir.mkdir(parents=True, exist_ok=True)
            writable = os.access(settings.out_dir, os.W_OK)
        except OSError:
            writable = False
        add("out_dir", "Output directory writable", writable, str(settings.out_dir))
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
        return {"checks": checks, "settings": _settings_view(settings)}

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
            jobs.start(req.include_stream, req.fresh)
        except RuntimeError as exc:
            raise HTTPException(409, str(exc))
        return jobs.snapshot()

    @app.post("/api/job/stop")
    def stop() -> dict:
        if not jobs.stop():
            raise HTTPException(409, "no run in progress")
        return jobs.snapshot()

    @app.get("/api/runs")
    def runs() -> dict:
        return {"out_dir": str(settings.out_dir), "runs": analytics.list_runs(settings.out_dir)}

    @app.get("/api/runs/{name}")
    def run(name: str) -> dict:
        return analytics.run_overview(run_path(name))

    @app.get("/api/runs/{name}/topics/{topic_id}")
    def topic(name: str, topic_id: str) -> dict:
        return {"topic_id": topic_id,
                "samples": analytics.topic_samples(run_path(name), topic_id)}

    @app.get("/api/stream")
    def stream() -> dict:
        return {"rows": analytics.stream_table(settings.out_dir)}

    @app.on_event("shutdown")
    def _stop_job() -> None:
        # Don't leave a GPU run orphaned when the dashboard goes away.
        jobs.stop()

    if WEB_DIST.is_dir():
        app.mount("/assets", StaticFiles(directory=WEB_DIST / "assets"), name="assets")

        @app.get("/{path:path}", include_in_schema=False)
        def spa(path: str):
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
