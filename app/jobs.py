"""SQLite-backed job queue for work too slow to run inside a request.

Video takes minutes, well past any MCP tool timeout, so it always goes through
here: submit -> job_id -> poll. Persistence means a long job survives a worker
restart being observed, and the caller can come back to it later.

A SINGLE runner thread drains the queue and calls `ModelManager.run()`, which
itself funnels onto the one GPU thread. There is exactly one place GPU work
happens, so nothing can race for VRAM.
"""

from __future__ import annotations

import json
import sqlite3
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Callable

from app.config import JOBS_DB

QUEUED, RUNNING, DONE, ERROR, CANCELLED = "queued", "running", "done", "error", "cancelled"
TERMINAL = {DONE, ERROR, CANCELLED}
RETENTION_DAYS = 7

_SCHEMA = """
CREATE TABLE IF NOT EXISTS jobs (
    id                TEXT PRIMARY KEY,
    kind              TEXT NOT NULL,
    status            TEXT NOT NULL,
    params_json       TEXT NOT NULL,
    result_json       TEXT,
    error             TEXT,
    progress          REAL DEFAULT 0.0,
    progress_msg      TEXT,
    cancel_requested  INTEGER DEFAULT 0,
    created_at        REAL NOT NULL,
    started_at        REAL,
    finished_at       REAL
);
CREATE INDEX IF NOT EXISTS idx_jobs_status ON jobs(status);
CREATE INDEX IF NOT EXISTS idx_jobs_created ON jobs(created_at DESC);
"""


class JobStore:
    def __init__(self, db_path: Path = JOBS_DB) -> None:
        db_path = Path(db_path)
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(str(db_path), check_same_thread=False)
        self._db.row_factory = sqlite3.Row
        self._lock = threading.RLock()
        with self._lock:
            self._db.execute("PRAGMA journal_mode=WAL")
            self._db.executescript(_SCHEMA)
            self._db.commit()
        self._recover()
        self._prune()

    def _recover(self) -> None:
        """A job marked running cannot survive the process that ran it."""
        with self._lock:
            self._db.execute(
                "UPDATE jobs SET status=?, error=?, finished_at=? WHERE status IN (?, ?)",
                (ERROR, "worker restarted before this job finished", time.time(), RUNNING, QUEUED),
            )
            self._db.commit()

    def _prune(self) -> None:
        cutoff = time.time() - RETENTION_DAYS * 86400
        with self._lock:
            self._db.execute("DELETE FROM jobs WHERE created_at < ?", (cutoff,))
            self._db.commit()

    def submit(self, kind: str, params: dict[str, Any]) -> str:
        job_id = f"{kind[:3]}-{uuid.uuid4().hex[:8]}"
        with self._lock:
            self._db.execute(
                "INSERT INTO jobs (id, kind, status, params_json, created_at) VALUES (?,?,?,?,?)",
                (job_id, kind, QUEUED, json.dumps(params, default=str), time.time()),
            )
            self._db.commit()
        return job_id

    def get(self, job_id: str) -> dict[str, Any] | None:
        with self._lock:
            row = self._db.execute("SELECT * FROM jobs WHERE id=?", (job_id,)).fetchone()
        return self._row_to_dict(row) if row else None

    def list(self, status: str | None = None, kind: str | None = None, limit: int = 20) -> list[dict[str, Any]]:
        sql = "SELECT * FROM jobs"
        clauses, args = [], []
        if status:
            clauses.append("status=?")
            args.append(status)
        if kind:
            clauses.append("kind=?")
            args.append(kind)
        if clauses:
            sql += " WHERE " + " AND ".join(clauses)
        sql += " ORDER BY created_at DESC LIMIT ?"
        args.append(limit)
        with self._lock:
            rows = self._db.execute(sql, args).fetchall()
        return [self._row_to_dict(r) for r in rows]

    def claim_next(self) -> dict[str, Any] | None:
        with self._lock:
            row = self._db.execute(
                "SELECT * FROM jobs WHERE status=? ORDER BY created_at ASC LIMIT 1", (QUEUED,)
            ).fetchone()
            if row is None:
                return None
            self._db.execute(
                "UPDATE jobs SET status=?, started_at=? WHERE id=?",
                (RUNNING, time.time(), row["id"]),
            )
            self._db.commit()
        return self.get(row["id"])

    def set_progress(self, job_id: str, fraction: float, message: str) -> None:
        with self._lock:
            self._db.execute(
                "UPDATE jobs SET progress=?, progress_msg=? WHERE id=?",
                (round(float(fraction), 4), message, job_id),
            )
            self._db.commit()

    def finish(self, job_id: str, result: dict[str, Any]) -> None:
        with self._lock:
            self._db.execute(
                "UPDATE jobs SET status=?, result_json=?, progress=1.0, finished_at=? WHERE id=?",
                (DONE, json.dumps(result, default=str), time.time(), job_id),
            )
            self._db.commit()

    def fail(self, job_id: str, error: str, status: str = ERROR) -> None:
        with self._lock:
            self._db.execute(
                "UPDATE jobs SET status=?, error=?, finished_at=? WHERE id=?",
                (status, error[:2000], time.time(), job_id),
            )
            self._db.commit()

    def request_cancel(self, job_id: str) -> dict[str, Any] | None:
        with self._lock:
            row = self._db.execute("SELECT status FROM jobs WHERE id=?", (job_id,)).fetchone()
            if row is None:
                return None
            if row["status"] == QUEUED:
                # Never started, so cancel it outright.
                self._db.execute(
                    "UPDATE jobs SET status=?, cancel_requested=1, finished_at=? WHERE id=?",
                    (CANCELLED, time.time(), job_id),
                )
            else:
                self._db.execute("UPDATE jobs SET cancel_requested=1 WHERE id=?", (job_id,))
            self._db.commit()
        return self.get(job_id)

    def is_cancelled(self, job_id: str) -> bool:
        with self._lock:
            row = self._db.execute(
                "SELECT cancel_requested FROM jobs WHERE id=?", (job_id,)
            ).fetchone()
        return bool(row and row["cancel_requested"])

    @staticmethod
    def _row_to_dict(row: sqlite3.Row) -> dict[str, Any]:
        d = dict(row)
        d["params"] = json.loads(d.pop("params_json") or "{}")
        result = d.pop("result_json", None)
        d["result"] = json.loads(result) if result else None
        d["cancel_requested"] = bool(d.get("cancel_requested"))
        for key in ("created_at", "started_at", "finished_at"):
            if d.get(key):
                d[key] = round(d[key], 3)
        started, finished = d.get("started_at"), d.get("finished_at")
        if started:
            d["elapsed_s"] = round((finished or time.time()) - started, 1)
        return d

    def close(self) -> None:
        with self._lock:
            self._db.close()


class JobRunner:
    """Single background thread draining the queue."""

    def __init__(self, store: JobStore, handlers: dict[str, Callable[..., Any]]) -> None:
        self._store = store
        self._handlers = handlers
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, daemon=True, name="job-runner")

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()

    def _loop(self) -> None:
        while not self._stop.is_set():
            job = None
            try:
                job = self._store.claim_next()
            except Exception:  # noqa: BLE001 - the runner must never die
                pass
            if job is None:
                self._stop.wait(1.0)
                continue
            self._run(job)

    def _run(self, job: dict[str, Any]) -> None:
        from app.backends.base import JobCancelled

        job_id, kind = job["id"], job["kind"]
        handler = self._handlers.get(kind)
        if handler is None:
            self._store.fail(job_id, f"no handler registered for kind {kind!r}")
            return

        def progress(fraction: float, message: str) -> None:
            self._store.set_progress(job_id, fraction, message)

        try:
            artifact = handler(
                params=job["params"],
                progress=progress,
                is_cancelled=lambda: self._store.is_cancelled(job_id),
            )
            self._store.finish(job_id, artifact.to_dict())
        except JobCancelled as exc:
            self._store.fail(job_id, str(exc), status=CANCELLED)
        except Exception as exc:  # noqa: BLE001
            self._store.fail(job_id, f"{type(exc).__name__}: {exc}")


_store: JobStore | None = None
_runner: JobRunner | None = None
_init_lock = threading.Lock()


def get_store() -> JobStore:
    global _store, _runner
    with _init_lock:
        if _store is not None:
            return _store
        from app import service

        _store = JobStore()
        _runner = JobRunner(_store, service.job_handlers())
        _runner.start()
        return _store


def shutdown() -> None:
    if _runner is not None:
        _runner.stop()
    if _store is not None:
        _store.close()
