"""Worker discovery and auto-spawn. MUST stay torch-free.

This is imported by the MCP adapter, which harnesses spawn on every session.
Importing torch here would put a 5-15 s stall (and a CUDA context) into every
harness startup, which is exactly what this architecture exists to avoid.
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

import httpx

from app.config import AUTOSPAWN, BASE_URL, LOG_DIR, OUTPUT_ROOT

REPO_ROOT = Path(__file__).resolve().parent.parent
LOCKFILE = OUTPUT_ROOT / "spawn.lock"
SPAWN_TIMEOUT_S = 90.0
POLL_INTERVAL_S = 0.25


class WorkerUnavailable(RuntimeError):
    pass


def _manual_hint() -> str:
    return (
        f"Start it manually:\n"
        f'    cd "{REPO_ROOT}"\n'
        f"    .venv\\Scripts\\python.exe -m app.worker\n"
        f"Log: {LOG_DIR / 'worker.log'}"
    )


def is_alive(base_url: str = BASE_URL, timeout: float = 1.0) -> bool:
    try:
        r = httpx.get(f"{base_url}/health", timeout=timeout)
        return r.status_code == 200
    except Exception:  # noqa: BLE001
        return False


def _take_spawn_lock() -> int | None:
    """Exclusive create; None means another process is already spawning."""
    try:
        LOCKFILE.parent.mkdir(parents=True, exist_ok=True)
        fd = os.open(str(LOCKFILE), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        os.write(fd, str(os.getpid()).encode())
        return fd
    except FileExistsError:
        # Stale lock from a crashed spawn: clear it if it is old.
        try:
            if time.time() - LOCKFILE.stat().st_mtime > SPAWN_TIMEOUT_S:
                LOCKFILE.unlink(missing_ok=True)
        except OSError:
            pass
        return None
    except OSError:
        return None


def _release_spawn_lock(fd: int | None) -> None:
    if fd is None:
        return
    try:
        os.close(fd)
    except OSError:
        pass
    try:
        LOCKFILE.unlink(missing_ok=True)
    except OSError:
        pass


def _spawn() -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    logfile = open(LOG_DIR / "worker.log", "ab", buffering=0)  # noqa: SIM115

    creationflags = 0
    if os.name == "nt":
        # Detached so the worker outlives the harness that spawned it.
        creationflags = getattr(subprocess, "DETACHED_PROCESS", 0x00000008) | getattr(
            subprocess, "CREATE_NEW_PROCESS_GROUP", 0x00000200
        )

    subprocess.Popen(
        [sys.executable, "-m", "app.worker"],
        cwd=str(REPO_ROOT),
        stdout=logfile,
        stderr=subprocess.STDOUT,
        stdin=subprocess.DEVNULL,
        creationflags=creationflags,
        close_fds=True,
    )


def ensure_worker(base_url: str = BASE_URL, autospawn: bool | None = None) -> str:
    """Return a base URL known to be serving, spawning the worker if needed."""
    if is_alive(base_url):
        return base_url

    if autospawn is None:
        autospawn = AUTOSPAWN
    if not autospawn:
        raise WorkerUnavailable(
            f"LocalGen worker is not running at {base_url} and autospawn is disabled.\n"
            + _manual_hint()
        )

    fd = _take_spawn_lock()
    if fd is not None:
        try:
            _spawn()
        except Exception as exc:  # noqa: BLE001
            _release_spawn_lock(fd)
            raise WorkerUnavailable(
                f"Could not start the LocalGen worker: {exc}\n" + _manual_hint()
            ) from exc

    # Poll regardless of who won the lock -- /health never touches the GPU, so
    # it answers as soon as uvicorn binds.
    deadline = time.time() + SPAWN_TIMEOUT_S
    try:
        while time.time() < deadline:
            if is_alive(base_url):
                return base_url
            time.sleep(POLL_INTERVAL_S)
    finally:
        _release_spawn_lock(fd)

    raise WorkerUnavailable(
        f"LocalGen worker did not come up within {SPAWN_TIMEOUT_S:.0f}s at {base_url}.\n"
        + _manual_hint()
    )
