"""Detached GPU worker: python -m app.worker

Owns the models and the CUDA context. Runs independently of any harness so a
Claude Code / OpenCode restart costs nothing.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

from app.config import HOST, LOG_DIR, OUTPUT_ROOT, PORT

PIDFILE = OUTPUT_ROOT / "worker.lock"


def _write_pidfile() -> None:
    try:
        PIDFILE.parent.mkdir(parents=True, exist_ok=True)
        PIDFILE.write_text(str(os.getpid()), encoding="utf-8")
    except OSError:
        pass


def _clear_pidfile() -> None:
    try:
        PIDFILE.unlink(missing_ok=True)
    except OSError:
        pass


def main() -> None:
    import atexit

    import uvicorn

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    _write_pidfile()
    atexit.register(_clear_pidfile)

    print(f"[localgen] worker pid={os.getpid()} on http://{HOST}:{PORT}", flush=True)
    print(f"[localgen] logs: {LOG_DIR}", flush=True)

    uvicorn.run("app.main:app", host=HOST, port=PORT, log_level="info", access_log=False)


if __name__ == "__main__":
    # Allow `python app/worker.py` as well as `python -m app.worker`.
    if __package__ in (None, ""):
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    main()
