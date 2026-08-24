"""Backend protocol and the single artifact type every modality returns."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Protocol, runtime_checkable

# fraction 0..1, human-readable message
Progress = Callable[[float, str], None]


@dataclass(slots=True)
class Artifact:
    """One generated thing. Uniform across image / svg / video.

    `preview_path` is always something a harness can render as an image: for
    images it equals `path`; for SVG it is the rasterised PNG; for video it
    would be a contact sheet. `text` carries inline source (SVG only) -- it is
    cheap in tokens and lets the caller edit the result directly.
    """

    path: Path
    kind: str
    mime: str
    width: int | None = None
    height: int | None = None
    preview_path: Path | None = None
    text: str | None = None
    prompt_used: str | None = None
    model: str | None = None
    backend: str | None = None
    quantization: str | None = None
    seed: int | None = None
    steps: int | None = None
    elapsed_s: float = 0.0
    warnings: list[str] = field(default_factory=list)
    meta: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "path": str(self.path),
            "preview_path": str(self.preview_path) if self.preview_path else None,
            "mime": self.mime,
            "width": self.width,
            "height": self.height,
            "text": self.text,
            "prompt_used": self.prompt_used,
            "model": self.model,
            "backend": self.backend,
            "quantization": self.quantization,
            "seed": self.seed,
            "steps": self.steps,
            "elapsed_s": round(self.elapsed_s, 2),
            "warnings": list(self.warnings),
            "meta": dict(self.meta),
        }


@runtime_checkable
class Backend(Protocol):
    """A generation backend.

    Only ONE gpu backend may be resident at a time on an 8GB card, which the
    ModelManager enforces. Backends with `needs_gpu = False` (Ollama-driven
    ones) are never leased and may run concurrently with a GPU job.
    """

    name: str
    kinds: tuple[str, ...]
    needs_gpu: bool
    vram_estimate_mb: int

    @property
    def loaded(self) -> bool: ...

    def load(self) -> None: ...

    def unload(self) -> None: ...

    def generate(self, params: dict[str, Any], progress: Progress | None = None) -> Artifact: ...


class JobCancelled(RuntimeError):
    """Raised from a diffusion step callback to abort a running job."""
