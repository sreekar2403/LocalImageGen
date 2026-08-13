"""Port (interface) definitions for LocalImageGen.

Ports are the contracts that infrastructure adapters must satisfy. They are
defined in the domain layer and depend only on the domain models, keeping the
core free of any infrastructure knowledge. Concrete implementations (e.g. an
Ollama-backed LLM adapter or a diffusers-backed diffusion adapter) are provided
in the ``adapters`` package.
"""

from __future__ import annotations

from typing import Any, Callable, Protocol, runtime_checkable

from localimagegen.core.models import (
    GeneratedImage,
    GenerationMetadata,
    GenerationParams,
    Prompt,
)


@runtime_checkable
class LLMPort(Protocol):
    """Contract for a local language model used to enhance prompts."""

    def enhance(self, prompt: Prompt, style: str) -> Prompt:
        """Return a new :class:`Prompt` with an enhanced version of the text.

        The returned prompt should carry the optimized text in ``enhanced``.
        """
        ...


@runtime_checkable
class DiffusionPort(Protocol):
    """Contract for a local diffusion model that renders images."""

    def generate(self, prompt: Prompt, params: GenerationParams) -> GeneratedImage:
        """Generate an image from a prompt and return it with metadata."""
        ...

    def metadata(self) -> GenerationMetadata:
        """Return metadata describing the underlying diffusion model."""
        ...


@runtime_checkable
class StoragePort(Protocol):
    """Contract for persisting generated images."""

    def save(self, image: GeneratedImage, name: str) -> str:
        """Persist an image and return the location (path/URI) it was saved to."""
        ...


@runtime_checkable
class MemoryManagerPort(Protocol):
    """Contract for coordinating GPU/CPU memory across model adapters.

    Adapters register the models they load together with an optional callback
    that performs the actual offload (e.g. moving a diffusers pipeline back to
    CPU). The manager tracks device locations, exposes usage statistics, and
    can trigger garbage collection / CUDA cache release.
    """

    def register(
        self,
        name: str,
        device: str,
        offload: Callable[[], None] | None = None,
    ) -> None:
        """Track a loaded model and how to offload it."""
        ...

    def offload(self, name: str) -> None:
        """Move a tracked model back to CPU and free its GPU memory."""
        ...

    def offload_all(self) -> None:
        """Offload every tracked model."""
        ...

    def stats(self) -> dict[str, Any]:
        """Return current memory usage statistics."""
        ...

    def cleanup(self) -> None:
        """Run garbage collection and release cached GPU memory."""
        ...
