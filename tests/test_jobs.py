"""Job store semantics. No GPU, no models."""

from __future__ import annotations

import time

import pytest

from app.backends.base import Artifact, JobCancelled
from app.jobs import CANCELLED, DONE, ERROR, QUEUED, RUNNING, JobRunner, JobStore


@pytest.fixture
def store(tmp_path):
    s = JobStore(tmp_path / "jobs.db")
    yield s
    s.close()


def test_submit_returns_queued_job(store):
    jid = store.submit("video", {"prompt": "x"})
    job = store.get(jid)
    assert job["status"] == QUEUED
    assert job["kind"] == "video"
    assert job["params"] == {"prompt": "x"}


def test_job_ids_are_unique(store):
    ids = {store.submit("video", {}) for _ in range(200)}
    assert len(ids) == 200


def test_claim_next_is_fifo_and_marks_running(store):
    first = store.submit("video", {"n": 1})
    second = store.submit("video", {"n": 2})
    claimed = store.claim_next()
    assert claimed["id"] == first
    assert claimed["status"] == RUNNING
    assert store.claim_next()["id"] == second
    assert store.claim_next() is None


def test_progress_updates(store):
    jid = store.submit("video", {})
    store.claim_next()
    store.set_progress(jid, 0.42, "denoise 8/20")
    job = store.get(jid)
    assert job["progress"] == pytest.approx(0.42)
    assert job["progress_msg"] == "denoise 8/20"


def test_finish_records_result(store):
    jid = store.submit("video", {})
    store.claim_next()
    store.finish(jid, {"path": "out.mp4", "kind": "video"})
    job = store.get(jid)
    assert job["status"] == DONE
    assert job["result"]["path"] == "out.mp4"
    assert job["progress"] == 1.0


def test_fail_records_error(store):
    jid = store.submit("video", {})
    store.claim_next()
    store.fail(jid, "boom")
    assert store.get(jid)["status"] == ERROR
    assert "boom" in store.get(jid)["error"]


def test_cancel_queued_job_terminates_immediately(store):
    jid = store.submit("video", {})
    assert store.request_cancel(jid)["status"] == CANCELLED
    assert store.claim_next() is None, "a cancelled job must not be claimable"


def test_cancel_running_job_sets_flag(store):
    jid = store.submit("video", {})
    store.claim_next()
    job = store.request_cancel(jid)
    assert job["status"] == RUNNING, "a running job stays running until it notices"
    assert store.is_cancelled(jid) is True


def test_cancel_unknown_job_returns_none(store):
    assert store.request_cancel("nope-123") is None


def test_recovery_marks_orphaned_jobs_errored(tmp_path):
    """A job cannot survive the process that was running it."""
    db = tmp_path / "jobs.db"
    s1 = JobStore(db)
    jid = s1.submit("video", {})
    s1.claim_next()
    assert s1.get(jid)["status"] == RUNNING
    s1.close()

    s2 = JobStore(db)  # simulates a worker restart
    job = s2.get(jid)
    assert job["status"] == ERROR
    assert "restarted" in job["error"]
    s2.close()


def test_list_filters(store):
    a = store.submit("video", {})
    store.submit("image", {})
    store.claim_next()
    store.finish(a, {"ok": True})
    assert len(store.list(kind="video")) == 1
    assert len(store.list(status=DONE)) == 1
    assert len(store.list()) == 2


# --- runner ------------------------------------------------------------------


def _wait(store, jid, timeout=5.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        job = store.get(jid)
        if job["status"] in (DONE, ERROR, CANCELLED):
            return job
        time.sleep(0.02)
    raise AssertionError(f"job {jid} did not finish: {store.get(jid)}")


def test_runner_executes_and_records(store):
    def handler(params, progress, is_cancelled):
        progress(0.5, "halfway")
        return Artifact(path=params["out"], kind="video", mime="video/mp4")

    runner = JobRunner(store, {"video": handler})
    runner.start()
    try:
        jid = store.submit("video", {"out": "clip.mp4"})
        job = _wait(store, jid)
        assert job["status"] == DONE
        assert job["result"]["path"] == "clip.mp4"
    finally:
        runner.stop()


def test_runner_reports_handler_failure(store):
    def handler(params, progress, is_cancelled):
        raise RuntimeError("gpu exploded")

    runner = JobRunner(store, {"video": handler})
    runner.start()
    try:
        job = _wait(store, store.submit("video", {}))
        assert job["status"] == ERROR
        assert "gpu exploded" in job["error"]
    finally:
        runner.stop()


def test_runner_marks_cancellation_distinctly(store):
    def handler(params, progress, is_cancelled):
        raise JobCancelled("cancelled during denoise")

    runner = JobRunner(store, {"video": handler})
    runner.start()
    try:
        job = _wait(store, store.submit("video", {}))
        assert job["status"] == CANCELLED, "cancellation must not look like a failure"
    finally:
        runner.stop()


def test_runner_rejects_unknown_kind(store):
    runner = JobRunner(store, {})
    runner.start()
    try:
        job = _wait(store, store.submit("video", {}))
        assert job["status"] == ERROR
        assert "no handler" in job["error"]
    finally:
        runner.stop()
