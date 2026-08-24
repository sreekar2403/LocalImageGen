"""ModelManager semantics, exercised with fake backends (no GPU, no models)."""

from __future__ import annotations

import threading

import pytest

from app.backends.base import Artifact
from app.manager import ModelManager


class FakeBackend:
    def __init__(self, name: str, needs_gpu: bool = True) -> None:
        self.name = name
        self.kinds = (name,)
        self.needs_gpu = needs_gpu
        self.vram_estimate_mb = 100
        self._loaded = False
        self.load_calls = 0
        self.unload_calls = 0
        self.threads: list[str] = []

    @property
    def loaded(self) -> bool:
        return self._loaded

    def load(self) -> None:
        self._loaded = True
        self.load_calls += 1

    def unload(self) -> None:
        self._loaded = False
        self.unload_calls += 1

    def generate(self, params, progress=None) -> Artifact:
        self.threads.append(threading.current_thread().name)
        return Artifact(path=params.get("out_path", ""), kind=self.name, mime="text/plain")


@pytest.fixture
def mgr():
    m = ModelManager(idle_evict_s=10_000, min_residency_s=0)
    yield m
    m.shutdown()


def test_first_run_loads_backend(mgr):
    a = FakeBackend("a")
    mgr.register(a)
    mgr.run("a", {})
    assert a.load_calls == 1 and a.loaded


def test_repeat_run_reuses_resident_backend(mgr):
    a = FakeBackend("a")
    mgr.register(a)
    for _ in range(3):
        mgr.run("a", {})
    assert a.load_calls == 1, "backend reloaded despite already being resident"


def test_switching_backend_evicts_the_incumbent(mgr):
    a, b = FakeBackend("a"), FakeBackend("b")
    mgr.register(a)
    mgr.register(b)
    mgr.run("a", {})
    mgr.run("b", {})
    assert a.unload_calls == 1, "previous backend was not evicted -- both would hold VRAM"
    assert not a.loaded and b.loaded
    assert mgr.status()["resident_backend"] == "b"
    assert mgr.status()["swaps"] == 1


def test_explicit_evict_frees(mgr):
    a = FakeBackend("a")
    mgr.register(a)
    mgr.run("a", {})
    mgr.evict()
    assert a.unload_calls == 1 and not a.loaded
    assert mgr.status()["resident_backend"] is None


def test_all_gpu_work_runs_on_one_thread(mgr):
    """Serialization must be structural, not advisory."""
    a = FakeBackend("a")
    mgr.register(a)
    threads = []
    workers = [threading.Thread(target=lambda: mgr.run("a", {})) for _ in range(8)]
    for t in workers:
        t.start()
    for t in workers:
        t.join()
    threads = set(a.threads)
    assert len(threads) == 1, f"GPU work ran on {len(threads)} threads: {threads}"
    assert threads.pop().startswith("gpu")


def test_cpu_backend_bypasses_the_gpu_thread(mgr):
    cpu = FakeBackend("cpu", needs_gpu=False)
    mgr.register(cpu)
    mgr.run("cpu", {})
    assert cpu.threads and not cpu.threads[0].startswith("gpu")
    assert cpu.load_calls == 0, "CPU backend should never be leased or loaded"
    assert mgr.status()["resident_backend"] is None


def test_cpu_backend_does_not_evict_gpu_backend(mgr):
    a, cpu = FakeBackend("a"), FakeBackend("cpu", needs_gpu=False)
    mgr.register(a)
    mgr.register(cpu)
    mgr.run("a", {})
    mgr.run("cpu", {})
    assert a.loaded and a.unload_calls == 0
    assert mgr.status()["resident_backend"] == "a"


def test_unknown_backend_raises(mgr):
    with pytest.raises(KeyError):
        mgr.run("nope", {})


def test_for_kind_lookup(mgr):
    a = FakeBackend("a")
    mgr.register(a)
    assert mgr.for_kind("a") is a
    with pytest.raises(KeyError):
        mgr.for_kind("missing")
