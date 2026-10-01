"""python -m dashboard [--port 8765] [--out-dir DIR] ...

Paths default to the same DATA_DIR / OUT_DIR / CACHE_DIR docker compose uses
(environment, then .env beside docker-compose.yml, then compose's defaults).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import uvicorn

from .pipeline import DATASETS_FILE, Settings, load_saved_datasets
from .server import create_app


def main() -> None:
    ap = argparse.ArgumentParser(prog="python -m dashboard", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--host", default="127.0.0.1",
                    help="Bind address. The dashboard can start GPU runs; keep it on "
                         "localhost and reach it over an SSH tunnel (default 127.0.0.1)")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--app-dir", help="Pipeline scripts (default: this checkout)")
    ap.add_argument("--data-dir", help="Host dir mounted at /data (DATA_DIR)")
    ap.add_argument("--out-dir", help="Host dir mounted at /out (OUT_DIR)")
    ap.add_argument("--cache-dir", help="Host dir mounted at /cache (CACHE_DIR)")
    ap.add_argument("--python", help="Interpreter for the pipeline (default: .venv/bin/python)")
    ap.add_argument("--compose-file", help="Default: docker-compose.yml in --app-dir")
    ap.add_argument("--device", help="Override the services' --device=cuda, e.g. 'cpu' on a "
                                     "machine without a GPU, or 'cuda:1'")
    ap.add_argument("--datasets-file", default=str(DATASETS_FILE),
                    help="JSON list of saved result sets to show as extra tabs "
                         "(default: dashboard/datasets.local.json, if present)")
    a = ap.parse_args()
    for problem in load_saved_datasets(Path(a.datasets_file)):
        print(f"warning: {problem}", flush=True)
    settings = Settings.resolve(a.app_dir, a.data_dir, a.out_dir, a.cache_dir,
                                a.python, a.compose_file, a.device)
    print(f"Dashboard: http://{a.host}:{a.port}   (OUT_DIR={settings.out_dir})", flush=True)
    app = create_app(settings)
    try:
        # An open browser tab holds a live-updates (SSE) connection that never
        # closes by itself; without a timeout, Ctrl+C would wait on it forever
        # and the run would keep going with no server to stop it.
        uvicorn.run(app, host=a.host, port=a.port, log_level="warning",
                    timeout_graceful_shutdown=3)
    finally:
        app.state.jobs.stop()          # never leave a pipeline run orphaned


if __name__ == "__main__":
    main()
