"""VRAM arbiter.

With ~7 GB free on an 8 GB card, exactly ONE gpu backend can be resident. The
mechanism is a dedicated single-worker thread pool rather than a lock: every
piece of GPU work -- sync HTTP requests and queued jobs alike -- is submitted to
the same thread. Serialization becomes structural, eviction is trivially safe,
and no CUDA context is ever touched from two threads.

Backends with `needs_gpu = False` (LLM-driven SVG authoring, prompt enhancement)
bypass the GPU thread entirely and run concurrently.
"""

from __future__ import annotations

import gc
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from app.backends.base import Artifact, Backend, Progress
from app.config import IDLE_EVICT_S, MIN_RESIDENCY_S

MB = 1024 * 1024


def _torch():
    import torch

    return torch


class ModelManager:
    def __init__(self, idle_evict_s: float = IDLE_EVICT_S, min_residency_s: float = MIN_RESIDENCY_S) -> None:
        self._exec = ThreadPoolExecutor(max_workers=1, thread_name_prefix="gpu")
        self._lock = threading.RLock()  # guards bookkeeping only, never CUDA
        self._backends: dict[str, Backend] = {}
        self._resident: str | None = None
        self._resident_since = 0.0
        self._last_used = time.monotonic()
        self._swaps = 0
        self._idle_evict_s = idle_evict_s
        self._min_residency_s = min_residency_s
        self._stop = threading.Event()
        self._reaper = threading.Thread(target=self._reap_loop, daemon=True, name="idle-reaper")
        self._reaper.start()

    # --- registry ------------------------------------------------------------

    def register(self, backend: Backend) -> None:
        with self._lock:
            self._backends[backend.name] = backend

    def get(self, name: str) -> Backend:
        with self._lock:
            if name not in self._backends:
                raise KeyError(f"unknown backend {name!r}; have {sorted(self._backends)}")
            return self._backends[name]

    def for_kind(self, kind: str) -> Backend:
        with self._lock:
            for b in self._backends.values():
                if kind in b.kinds:
                    return b
        raise KeyError(f"no backend handles kind {kind!r}")

    # --- execution -----------------------------------------------------------

    def run(self, name: str, params: dict[str, Any], progress: Progress | None = None) -> Artifact:
        """The only entry point for generation work.

        CPU-only backends run inline (concurrently); GPU backends are funnelled
        onto the single GPU thread.
        """
        backend = self.get(name)
        if not backend.needs_gpu:
            return backend.generate(params, progress)
        return self._exec.submit(self._run_on_gpu_thread, name, params, progress).result()

    def _run_on_gpu_thread(self, name: str, params: dict[str, Any], progress: Progress | None) -> Artifact:
        backend = self.get(name)
        self._ensure_resident(name)
        try:
            return backend.generate(params, progress)
        finally:
            self._last_used = time.monotonic()
            try:
                _torch().cuda.empty_cache()
            except Exception:  # noqa: BLE001
                pass

    def _ensure_resident(self, name: str) -> None:
        with self._lock:
            if self._resident == name and self._backends[name].loaded:
                return
            if self._resident is not None and self._resident != name:
                self._evict_locked()
                self._swaps += 1
            # LM Studio holds ~5.7 GB when a model is resident, which leaves
            # nowhere near enough for FLUX. Make it give the GPU back first.
            if self._backends[name].needs_gpu:
                try:
                    from app.backends.llm import release_gpu

                    release_gpu()
                except Exception:  # noqa: BLE001 - never block on arbitration
                    pass
            self._backends[name].load()
            self._resident = name
            self._resident_since = time.monotonic()

    # --- eviction ------------------------------------------------------------

    def _evict_locked(self) -> None:
        if self._resident is None:
            return
        backend = self._backends.get(self._resident)
        self._resident = None
        if backend is not None:
            backend.unload()
        gc.collect()
        try:
            torch = _torch()
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
        except Exception:  # noqa: BLE001
            pass
        gc.collect()

    def evict(self, name: str | None = None) -> dict[str, Any]:
        """Public eviction. Routed through the GPU thread so it can never race
        a generation in progress."""

        def _do() -> dict[str, Any]:
            with self._lock:
                if name is not None and self._resident != name:
                    return self.status()
                self._evict_locked()
            return self.status()

        return self._exec.submit(_do).result()

    def _reap_loop(self) -> None:
        while not self._stop.wait(30.0):
            try:
                with self._lock:
                    if self._resident is None:
                        continue
                    now = time.monotonic()
                    idle = now - self._last_used
                    resident_for = now - self._resident_since
                    if idle < self._idle_evict_s or resident_for < self._min_residency_s:
                        continue
                self.evict()
            except Exception:  # noqa: BLE001 - a reaper must never die
                continue

    def shutdown(self) -> None:
        self._stop.set()
        try:
            self.evict()
        except Exception:  # noqa: BLE001
            pass
        self._exec.shutdown(wait=False)

    # --- introspection -------------------------------------------------------

    def vram(self) -> tuple[int, int]:
        """(free_mb, total_mb) as the driver sees it, including other processes."""
        try:
            free, total = _torch().cuda.mem_get_info()
            return round(free / MB), round(total / MB)
        except Exception:  # noqa: BLE001 - no CUDA
            return (0, 0)

    def status(self) -> dict[str, Any]:
        free, total = self.vram()
        with self._lock:
            resident = self._resident
            backends = {
                b.name: {
                    "kinds": list(b.kinds),
                    "needs_gpu": b.needs_gpu,
                    "loaded": b.loaded,
                    "vram_estimate_mb": b.vram_estimate_mb,
                }
                for b in self._backends.values()
            }
            swaps = self._swaps
            last_used = self._last_used
        return {
            "resident_backend": resident,
            "vram_free_mb": free,
            "vram_total_mb": total,
            "swaps": swaps,
            "idle_s": round(time.monotonic() - last_used, 1),
            "idle_evict_s": self._idle_evict_s,
            "backends": backends,
        }


_manager: ModelManager | None = None
_manager_lock = threading.Lock()


def get_manager() -> ModelManager:
    """Process-wide manager with every backend registered."""
    global _manager
    with _manager_lock:
        if _manager is not None:
            return _manager

        mgr = ModelManager()

        from app.backends.image_flux import ImageFluxBackend

        mgr.register(ImageFluxBackend())

        # CPU-only backends: never leased, safe to run alongside a GPU job.
        try:
            from app.backends.enhance import EnhanceBackend
            from app.backends.svg import SvgAuthorBackend, SvgTraceBackend

            mgr.register(EnhanceBackend())
            mgr.register(SvgAuthorBackend())
            mgr.register(SvgTraceBackend(mgr))
        except Exception:  # noqa: BLE001 - SVG deps optional at import time
            pass

        _manager = mgr
        return _manager
