"""Domain models for LocalImageGen.

These models describe the core business concepts of the application. They live
in the innermost Clean Architecture layer and therefore depend only on the
Python standard library and ``pydantic`` for validation - never on
infrastructure concerns such as HTTP clients, model backends, or the
filesystem.
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class Prompt(BaseModel):
    """A text prompt used to drive image generation.

    ``text`` is the raw user idea; ``enhanced`` is the optional, LLM-optimized
    version produced by the prompt-enhancement step.
    """

    text: str = Field(..., min_length=1, description="The raw user prompt.")
    enhanced: Optional[str] = Field(
        default=None,
        description="LLM-optimized prompt, if enhancement has been applied.",
    )

    @property
    def effective(self) -> str:
        """Return the prompt to feed the diffusion model.

        Prefers the enhanced version when available, otherwise falls back to the
        raw text.
        """
        return self.enhanced if self.enhanced else self.text


class GenerationParams(BaseModel):
    """Parameters controlling a single diffusion generation run."""

    width: int = Field(default=1024, ge=64, le=4096)
    height: int = Field(default=768, ge=64, le=4096)
    num_inference_steps: int = Field(default=12, ge=1, le=1000)
    seed: int = Field(default=42)
    device: str = Field(default="cuda")
    guidance_scale: Optional[float] = Field(
        default=None, ge=0.0, description="Classifier-free guidance scale, if supported."
    )


class GenerationMetadata(BaseModel):
    """Metadata describing how and when an image was generated."""

    model: str = Field(..., description="Diffusion model identifier used.")
    prompt: Optional[str] = Field(
        default=None,
        description="The effective prompt used for generation. None for model-level metadata.",
    )
    params: GenerationParams = Field(..., description="Parameters used for generation.")
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="UTC timestamp of generation.",
    )
    duration_seconds: Optional[float] = Field(
        default=None, ge=0.0, description="Wall-clock time of the generation run."
    )


class GeneratedImage(BaseModel):
    """A generated image together with its metadata.

    The image payload is stored as raw bytes so that the domain model remains
    independent of any particular image library or storage backend.
    """

    data: bytes = Field(..., description="Raw encoded image bytes (e.g. PNG).")
    format: str = Field(default="png", description="Image encoding format.")
    metadata: GenerationMetadata = Field(..., description="Generation metadata.")


class PromptStyle(str, Enum):
    """Supported prompt-enhancement styles."""

    FLUX = "flux"
    SDXL = "sdxl"
    MIDJOURNEY = "midjourney"
