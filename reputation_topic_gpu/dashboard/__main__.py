"""python -m dashboard [--port 8765] [--out-dir DIR] ...

Paths default to the same DATA_DIR / OUT_DIR / CACHE_DIR docker compose uses
(environment, then .env beside docker-compose.yml, then compose's defaults).
"""
from __future__ import annotations

import argparse

import uvicorn

from .pipeline import Settings
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
    a = ap.parse_args()
    settings = Settings.resolve(a.app_dir, a.data_dir, a.out_dir, a.cache_dir,
                                a.python, a.compose_file, a.device)
    print(f"Dashboard: http://{a.host}:{a.port}   (OUT_DIR={settings.out_dir})", flush=True)
    uvicorn.run(create_app(settings), host=a.host, port=a.port, log_level="warning")


if __name__ == "__main__":
    main()
