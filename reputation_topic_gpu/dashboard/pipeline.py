"""Run the docker-compose pipeline (base, then stream) and track it stage by stage.

The commands are read from docker-compose.yml rather than restated here, so the
dashboard always runs exactly the documented `base` and `stream` services. Inside
the image the container paths (/app, /data, /out, /cache) are used as they are;
on a plain checkout they are mapped onto the host directories in `Settings`.

Progress comes from the pipeline's own log: every stage ends with a
`[timing] <stage> <seconds>s <detail>` line, so the running stage is the next
one not yet marked. A few other lines give progress inside the long stages.
"""
from __future__ import annotations

import json
import os
import sys
import re
import shutil
import signal
import subprocess
import threading
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import yaml

HERE = Path(__file__).resolve().parent.parent      # reputation_topic_gpu/


# --------------------------------------------------------------- settings ---

def read_dotenv(path: Path) -> dict[str, str]:
    """KEY=value pairs, the way docker compose reads the .env beside the file."""
    out: dict[str, str] = {}
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                out[k.strip()] = v.strip().strip("'\"")
    return out


@dataclass
class Settings:
    app_dir: Path        # holds the pipeline scripts (/app in the image)
    data_dir: Path       # mounted at /data in the image
    out_dir: Path        # mounted at /out
    cache_dir: Path      # mounted at /cache
    python: Path         # interpreter the pipeline runs under
    compose_file: Path
    device: str | None = None   # overrides the services' --device (e.g. cpu)

    @classmethod
    def resolve(cls, app_dir=None, data_dir=None, out_dir=None, cache_dir=None,
                python=None, compose_file=None, device=None) -> "Settings":
        """CLI values first, then the same env vars / .env docker compose uses."""
        app = Path(app_dir or HERE).resolve()
        compose = Path(compose_file or app / "docker-compose.yml").resolve()
        env = {**read_dotenv(compose.parent / ".env"), **os.environ}

        def host(value, default):
            # Relative paths are relative to the compose file, as compose does.
            p = Path(value or env.get(default[0]) or default[1])
            return (p if p.is_absolute() else compose.parent / p).resolve()

        # CACHE_DIR may name a docker volume ("topic-cache"), not a path.
        cache = cache_dir or env.get("CACHE_DIR")
        if not cache or "/" not in cache:
            cache = str(app / "embed_cache")
        venv = app / ".venv/bin/python"
        return cls(
            app_dir=app,
            data_dir=host(data_dir, ("DATA_DIR", "../data")),
            out_dir=host(out_dir, ("OUT_DIR", "./out_docker")),
            cache_dir=host(cache, ("", "")),
            python=Path(python) if python else (venv if venv.exists() else Path(sys.executable)),
            compose_file=compose,
            device=device,
        )

    @property
    def mounts(self) -> dict[str, Path]:
        return {"/app": self.app_dir, "/data": self.data_dir,
                "/out": self.out_dir, "/cache": self.cache_dir}


# ------------------------------------------------------- compose commands ---

_VAR = re.compile(r"\$\{(\w+)(?::?-([^}]*))?\}")


def _map_arg(arg: str, mounts: dict[str, Path]) -> str:
    key, eq, val = arg.partition("=") if arg.startswith("--") and "=" in arg else ("", "", arg)
    for prefix, host in mounts.items():
        if val == prefix or val.startswith(prefix + "/"):
            val = str(host) + val[len(prefix):]
            break
    return f"{key}={val}" if eq else val


def service_command(settings: Settings, service: str) -> list[str]:
    """The compose service's command, interpolated and mapped to this host."""
    spec = yaml.safe_load(settings.compose_file.read_text(encoding="utf-8"))
    raw = spec["services"][service]["command"]
    env = {**read_dotenv(settings.compose_file.parent / ".env"), **os.environ}
    args = [_VAR.sub(lambda m: env.get(m[1]) or (m[2] or ""), str(a)) for a in raw]
    if settings.device:
        args = [f"--device={settings.device}" if a.startswith("--device=") else a for a in args]
        if "--device" in args:
            args[args.index("--device") + 1] = settings.device
    return [str(settings.python)] + [_map_arg(a, settings.mounts) for a in args]


def arg_value(cmd: list[str], flag: str) -> str | None:
    for i, a in enumerate(cmd):
        if a == flag and i + 1 < len(cmd):
            return cmd[i + 1]
        if a.startswith(flag + "="):
            return a.split("=", 1)[1]
    return None


def arg_list(cmd: list[str], flag: str) -> list[str]:
    """Values of a nargs='+' flag: everything up to the next --option."""
    if flag not in cmd:
        return []
    out = []
    for a in cmd[cmd.index(flag) + 1:]:
        if a.startswith("--"):
            break
        out.append(a)
    return out


# ------------------------------------------------------------ stage model ---

# (key, label, expected). Optional stages are "expected" only when this
# configuration runs them; the rest are still recognised if they turn up.
STAGES = [
    ("load + segment", "Load & segment records", True),
    ("model load", "Load embedding model", True),
    ("embedding", "Embed segments", True),
    ("layer 1 filter", "Acknowledgement filter", False),
    ("umap", "UMAP manifold (fit / load)", True),
    ("discovery clustering", "HDBSCAN topic discovery", True),
    ("history re-assignment", "History re-assignment", True),
    ("incremental assignment", "Incremental assignment", True),
    ("candidate discovery", "Emerging-topic discovery", True),
    ("consolidation", "Registry consolidation", True),
    ("coherence scoring", "Coherence scoring", True),
    ("llm labelling", "LLM labelling", True),
    ("metrics + outputs", "Hot-score metrics & outputs", True),
]
_STAGE_INDEX = {k: i for i, (k, _, _) in enumerate(STAGES)}


def canonical_stage(name: str) -> str:
    """Timer names -> tracker keys: 'embedding (cached)', 'umap fit/load'."""
    name = name.strip()
    if name.startswith("embedding"):
        return "embedding"
    if name.startswith("umap"):
        return "umap"
    return name


def expected_timings(path: Path) -> dict[str, float]:
    """Stage seconds from a previous run's stage_timing.json, for an ETA."""
    try:
        raw = json.loads(path.read_text())
    except (OSError, ValueError):
        return {}
    out: dict[str, float] = {}
    for k, v in raw.items():
        if k != "TOTAL" and isinstance(v, (int, float)):
            key = canonical_stage(k)
            out[key] = out.get(key, 0.0) + float(v)
    return out


class StageTracker:
    """One pipeline run (reputation_topic_detection.py) as a list of stages."""

    _TIMING = re.compile(r"^\[timing\]\s+(.+?)\s+([\d.]+)s\s*(.*)$")
    _EMBEDDED = re.compile(r"^\s*embedded ([\d,]+)/([\d,]+)")
    _TQDM = re.compile(r"Batches:\s+(\d+)%\|[^|]*\|\s*(\d+)/(\d+)")
    _POOL = re.compile(r"^Discovery pool=(\S+)")
    _BRANDS = re.compile(r"^Brands: (\d+)")
    _LOADED = re.compile(r"^Loaded ([\d,]+) records")
    _CREATED = re.compile(r"^Created ([\d,]+) semantic units from ([\d,]+) records")
    _DEVICE = re.compile(r"^Embedding device: (.+?) \(fp16")
    _EMERGING = re.compile(r"^New topic decisions: (\d+) topics created as EMERGING")
    _MICRO = re.compile(r"^Micro pass: (\d+) MICRO_EMERGING")
    _ERROR = re.compile(r"^ERROR: (.*)")

    def __init__(self, expected: dict[str, float] | None = None,
                 enabled: set[str] | None = None) -> None:
        expected = expected or {}
        self.stages = [{
            "key": k, "label": label, "status": "pending", "seconds": None,
            "detail": "", "started_at": None, "expected_seconds": expected.get(k),
            "expected": exp or (k in (enabled or set())), "progress": None,
        } for k, label, exp in STAGES]
        self.info: dict[str, object] = {}
        self.error: str | None = None
        self._pools = 0

    def start(self) -> None:
        self._advance(-1)

    def _advance(self, after: int) -> None:
        for s in self.stages[after + 1:]:
            if s["status"] == "pending" and s["expected"]:
                s["status"], s["started_at"] = "running", time.time()
                return

    def _running(self) -> dict | None:
        return next((s for s in self.stages if s["status"] == "running"), None)

    def feed(self, line: str) -> None:
        if m := self._TIMING.match(line):
            key = canonical_stage(m[1])
            idx = _STAGE_INDEX.get(key)
            if idx is None:
                return
            for s in self.stages[:idx]:
                if s["status"] in ("pending", "running"):
                    s["status"], s["progress"] = "skipped", None
            s = self.stages[idx]
            s["status"], s["progress"] = "done", None
            s["seconds"] = round((s["seconds"] or 0.0) + float(m[2]), 2)
            s["detail"] = m[3].strip()
            if s["started_at"] is None:
                s["started_at"] = time.time() - float(m[2])
            self._advance(idx)
            return
        cur = self._running()
        if cur is not None and cur["key"] == "embedding":
            if m := self._EMBEDDED.match(line):
                done, total = int(m[1].replace(",", "")), int(m[2].replace(",", ""))
                cur["progress"] = {"done": done, "total": total, "unit": "segments"}
            elif m := self._TQDM.search(line):
                cur["progress"] = {"done": int(m[2]), "total": int(m[3]), "unit": "batches"}
        if m := self._POOL.match(line):
            self._pools += 1
            s = self.stages[_STAGE_INDEX["discovery clustering"]]
            if s["status"] == "running":
                s["progress"] = {"done": self._pools, "total": self.info.get("brands"),
                                 "unit": "brand pools", "last": m[1]}
        elif m := self._BRANDS.match(line):
            self.info["brands"] = int(m[1])
        elif m := self._LOADED.match(line):
            self.info["records"] = int(m[1].replace(",", ""))
        elif m := self._CREATED.match(line):
            self.info["segments"] = int(m[1].replace(",", ""))
        elif m := self._DEVICE.match(line):
            self.info["device"] = m[1]
        elif m := self._EMERGING.match(line):
            self.info["emerging_created"] = int(m[1])
        elif m := self._MICRO.match(line):
            self.info["micro_emerging_created"] = int(m[1])
        elif m := self._ERROR.match(line):
            self.error = m[1]

    def finish(self, ok: bool, stopped: bool = False) -> None:
        for s in self.stages:
            if s["status"] == "running":
                s["status"] = "skipped" if ok else ("stopped" if stopped else "failed")
                if not ok and s["started_at"]:
                    s["seconds"] = round(time.time() - s["started_at"], 2)
            elif s["status"] == "pending":
                s["status"] = "skipped" if ok else "pending"
            s["progress"] = None

    def snapshot(self) -> dict:
        return {"stages": [dict(s) for s in self.stages], "info": dict(self.info),
                "error": self.error}


# -------------------------------------------------------------------- job ---

class PipelineJob:
    """Base run, then (optionally) the stream, each a subprocess we read line by line."""

    _CHUNK = re.compile(r"^=== chunk (\d+): (\S+) -> (\S+) ===")
    _CHUNK_DONE = re.compile(r"^\s+topics=(\d+) retention=(\S+) drift=(\S+) refit=(\S+) "
                             r"alerts=(\d+) events=(\S+) \(([\d.]+) min\)")
    _CHUNK_FAIL = re.compile(r"^\s+FAILED rc=(-?\d+)")

    def __init__(self, settings: Settings, include_stream: bool, fresh: bool,
                 on_change: Callable[[], None]) -> None:
        self.settings = settings
        self.include_stream = include_stream
        self.fresh = fresh
        self._on_change = on_change
        self.lock = threading.RLock()
        self.status = "running"
        self.started_at = time.time()
        self.finished_at: float | None = None
        self.error: str | None = None
        self.completed_runs: list[str] = []
        self.log: deque[str] = deque(maxlen=400)
        self._proc: subprocess.Popen | None = None
        self._stopping = False
        self.log_dir = settings.out_dir / "dashboard_logs"
        stamp = time.strftime("%Y%m%d-%H%M%S", time.localtime(self.started_at))

        base_cmd = service_command(settings, "base")
        base_out = Path(arg_value(base_cmd, "--out") or settings.out_dir / "base_200k")
        self.phases: list[dict] = [{
            "key": "base", "title": "Base registry",
            "subtitle": Path(next(a for a in base_cmd[2:] if not a.startswith("--"))).name,
            "status": "pending", "started_at": None, "finished_at": None,
            "returncode": None, "cmd": base_cmd, "run": base_out.name,
            "log_file": str(self.log_dir / f"{stamp}_base.log"),
            "tracker": StageTracker(expected_timings(base_out / "stage_timing.json")),
        }]
        if include_stream:
            stream_cmd = service_command(settings, "stream") + ["--echo"]
            prefix = Path(arg_value(stream_cmd, "--out-prefix") or settings.out_dir / "stream")
            chunks = arg_list(stream_cmd, "--chunks")
            self.phases.append({
                "key": "stream", "title": "Stream replay",
                "subtitle": f"{len(chunks)} batches, window {arg_value(stream_cmd, '--window') or 1}",
                "status": "pending", "started_at": None, "finished_at": None,
                "returncode": None, "cmd": stream_cmd,
                "log_file": str(self.log_dir / f"{stamp}_stream.log"),
                "batches": [{
                    "index": i, "chunk": Path(c).name, "run": f"{prefix.name}_{i}",
                    "status": "pending", "started_at": None, "finished_at": None,
                    "summary": None,
                    "tracker": StageTracker(expected_timings(
                        prefix.parent / f"{prefix.name}_{i}" / "stage_timing.json")),
                } for i, c in enumerate(chunks, start=1)],
            })
        self._thread = threading.Thread(target=self._run, name="pipeline-job", daemon=True)

    # ---- lifecycle
    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        with self.lock:
            self._stopping = True
            proc = self._proc
        if proc and proc.poll() is None:
            try:
                os.killpg(proc.pid, signal.SIGTERM)
            except ProcessLookupError:
                return
            threading.Thread(target=self._kill_later, args=(proc,), daemon=True).start()

    @staticmethod
    def _kill_later(proc: subprocess.Popen) -> None:
        try:
            proc.wait(timeout=15)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass

    def _changed(self) -> None:
        self._on_change()

    def _log(self, line: str) -> None:
        # tqdm redraws its bar with \r; keep only the latest frame in the tail.
        if line.startswith("Batches:") and self.log and self.log[-1].startswith("Batches:"):
            self.log[-1] = line
        else:
            self.log.append(line)

    # ---- the work
    def _clean(self, phase: dict) -> None:
        """A fresh run starts without the previous run's buffer and UMAP model,
        as .work/final_stream.sh did. The embedding cache is kept: it only
        skips work, never changes a result."""
        cmd = phase["cmd"]
        targets = []
        if buf := arg_value(cmd, "--buffer"):
            targets.append(Path(buf))
        if phase["key"] == "base" and (umap := arg_value(cmd, "--umap-model")):
            p = Path(umap).with_suffix(".pkl")
            targets += [p, p.with_suffix(".json")]
        for t in targets:
            if t.is_dir():
                shutil.rmtree(t)
            elif t.exists():
                t.unlink()
            self._log(f"[dashboard] removed {t}")

    def _run(self) -> None:
        try:
            self.log_dir.mkdir(parents=True, exist_ok=True)
            for phase in self.phases:
                if self._stopping:
                    break
                ok = self._run_phase(phase)
                if not ok:
                    break
            with self.lock:
                if self._stopping:
                    self.status = "stopped"
                elif any(p["status"] == "failed" for p in self.phases):
                    self.status = "failed"
                else:
                    self.status = "succeeded"
        except Exception as exc:                      # never leave the job "running"
            with self.lock:
                self.status, self.error = "failed", f"{type(exc).__name__}: {exc}"
                for p in self.phases:
                    if p["status"] == "running":
                        p["status"], p["finished_at"] = "failed", time.time()
                self._log(f"[dashboard] {self.error}")
        finally:
            with self.lock:
                self.finished_at = time.time()
            self._changed()

    def _run_phase(self, phase: dict) -> bool:
        with self.lock:
            phase["status"], phase["started_at"] = "running", time.time()
            if self.fresh:
                self._clean(phase)
            if phase["key"] == "base":
                phase["tracker"].start()
            self._log(f"[dashboard] $ {' '.join(phase['cmd'])}")
        self._changed()

        proc = subprocess.Popen(
            phase["cmd"], cwd=self.settings.app_dir, stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, text=True, errors="replace", bufsize=1,
            env={**os.environ, "PYTHONUNBUFFERED": "1"}, start_new_session=True)
        with self.lock:
            self._proc = proc
        last_push = 0.0
        with open(phase["log_file"], "w", encoding="utf-8") as logf:
            for raw in proc.stdout:                   # universal newlines: \r splits too
                logf.write(raw)
                line = raw.rstrip("\n")
                with self.lock:
                    self._log(line)
                    self._parse(phase, line)
                now = time.monotonic()
                if now - last_push > 0.25:
                    last_push = now
                    logf.flush()                      # keep the file tail -f-able
                    self._changed()
        rc = proc.wait()
        with self.lock:
            self._proc = None
            phase["returncode"], phase["finished_at"] = rc, time.time()
            ok = rc == 0 and not self._stopping
            phase["status"] = "succeeded" if ok else ("stopped" if self._stopping else "failed")
            if phase["key"] == "base":
                phase["tracker"].finish(ok, stopped=self._stopping)
                if ok:
                    self.completed_runs.append(phase["run"])
                elif not self._stopping:
                    self.error = phase["tracker"].error or f"base run exited with code {rc}"
            else:
                for b in phase["batches"]:
                    if b["status"] == "running":
                        b["status"] = "stopped" if self._stopping else "failed"
                        b["finished_at"] = time.time()
                        b["tracker"].finish(False, stopped=self._stopping)
                        if not self._stopping:
                            self.error = (b["tracker"].error
                                          or f"stream batch {b['index']} failed (exit {rc})")
                if not ok and not self._stopping and not self.error:
                    self.error = f"stream exited with code {rc}"
        self._changed()
        return ok

    def _parse(self, phase: dict, line: str) -> None:
        if phase["key"] == "base":
            phase["tracker"].feed(line)
            return
        batches = phase["batches"]
        cur = next((b for b in batches if b["status"] == "running"), None)
        if line.startswith("  | "):
            if cur:
                cur["tracker"].feed(line[4:])
        elif m := self._CHUNK.match(line):
            i = int(m[1])
            b = batches[i - 1]
            b["status"], b["started_at"] = "running", time.time()
            b["tracker"].start()
        elif (m := self._CHUNK_DONE.match(line)) and cur:
            cur["status"], cur["finished_at"] = "succeeded", time.time()
            cur["tracker"].finish(True)
            cur["summary"] = {
                "topics": int(m[1]),
                "id_retention": None if m[2] == "None" else float(m[2]),
                "drift": None if m[3] == "None" else float(m[3]),
                "refit": m[4] == "True", "alerts": int(m[5]),
                "events": None if m[6] == "None" else m[6],
                "minutes": float(m[7]),
            }
            self.completed_runs.append(cur["run"])
        elif (m := self._CHUNK_FAIL.match(line)) and cur:
            cur["status"], cur["finished_at"] = "failed", time.time()
            cur["tracker"].finish(False)

    # ---- state for the UI
    def snapshot(self) -> dict:
        with self.lock:
            phases = []
            for p in self.phases:
                d = {k: v for k, v in p.items() if k not in ("tracker", "batches", "cmd")}
                d["command"] = " ".join(p["cmd"])
                if "tracker" in p:
                    d.update(p["tracker"].snapshot())
                if "batches" in p:
                    d["batches"] = [{**{k: v for k, v in b.items() if k != "tracker"},
                                     **b["tracker"].snapshot()} for b in p["batches"]]
                phases.append(d)
            return {
                "status": self.status, "started_at": self.started_at,
                "finished_at": self.finished_at, "error": self.error,
                "include_stream": self.include_stream, "fresh": self.fresh,
                "completed_runs": list(self.completed_runs),
                "phases": phases, "log": list(self.log),
            }


class JobManager:
    """At most one job at a time; the last one is kept for the UI."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.job: PipelineJob | None = None
        self.version = 0
        self._lock = threading.Lock()

    def bump(self) -> None:
        with self._lock:
            self.version += 1

    def start(self, include_stream: bool, fresh: bool) -> PipelineJob:
        with self._lock:
            if self.job and self.job.status == "running":
                raise RuntimeError("a run is already in progress")
            self.job = PipelineJob(self.settings, include_stream, fresh, self.bump)
        self.job.start()
        self.bump()
        return self.job

    def stop(self) -> bool:
        job = self.job
        if not job or job.status != "running":
            return False
        job.stop()
        self.bump()
        return True

    def snapshot(self) -> dict:
        return {"version": self.version, "server_time": time.time(),
                "job": self.job.snapshot() if self.job else None}
