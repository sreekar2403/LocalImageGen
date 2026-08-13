"""Unified GPU/CPU memory manager.

Implements :class:`localimagegen.core.ports.MemoryManagerPort`. Tracks the
models loaded by the various adapters together with their device locations,
exposes VRAM usage statistics, and provides helpers to offload models back to
CPU and release cached CUDA memory.
"""

from __future__ import annotations

import gc
from typing import Any, Callable

import torch

from localimagegen.services.logging import get_logger


class UnifiedMemoryManager:
    """Coordinate GPU/CPU memory across model adapters.

    Adapters register the models they load via :meth:`register`; the manager
    keeps track of each model's device and an optional callback that performs
    the actual offload (e.g. moving a diffusers pipeline back to CPU).
    """

    def __init__(self) -> None:
        self._models: dict[str, str] = {}
        self._offload_callbacks: dict[str, Callable[[], None]] = {}
        self._logger = get_logger(__name__)

    def register(
        self,
        name: str,
        device: str,
        offload: Callable[[], None] | None = None,
    ) -> None:
        """Track a loaded model and how to offload it."""
        self._models[name] = device
        if offload is not None:
            self._offload_callbacks[name] = offload
        self._logger.debug("model_registered", model=name, device=device)

    def offload(self, name: str) -> None:
        """Move a tracked model back to CPU and free its GPU memory."""
        if name not in self._models:
            return
        callback = self._offload_callbacks.get(name)
        if callback is not None:
            callback()
        self._models[name] = "cpu"
        self._logger.info("model_offloaded", model=name)

    def offload_all(self) -> None:
        """Offload every tracked model."""
        for name in list(self._models):
            self.offload(name)

    def stats(self) -> dict[str, Any]:
        """Return current memory usage statistics."""
        stats: dict[str, Any] = {"models": dict(self._models)}
        if torch.cuda.is_available():
            stats["cuda"] = {
                "allocated_mb": round(torch.cuda.memory_allocated() / 1024**2, 2),
                "reserved_mb": round(torch.cuda.memory_reserved() / 1024**2, 2),
                "max_allocated_mb": round(torch.cuda.max_memory_allocated() / 1024**2, 2),
            }
        return stats

    def cleanup(self) -> None:
        """Run garbage collection and release cached CUDA memory."""
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        self._logger.debug("memory_cleanup_complete")