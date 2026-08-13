"""Port (interface) definitions for LocalImageGen.

Ports are the contracts that infrastructure adapters must satisfy. They are
defined in the domain layer and depend only on the domain models, keeping the
core free of any infrastructure knowledge. Concrete implementations (e.g. an
Ollama-backed LLM adapter or a diffusers-backed diffusion adapter) will be
provided in later phases.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

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
