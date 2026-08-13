"""Base pipeline abstract class for image generation workflows."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

from localimagegen.core.models import (
    GeneratedImage,
    GenerationParams,
    Prompt,
    PromptStyle,
)


class BasePipeline(ABC):
    """Abstract base for generation pipelines.

    Subclasses wire together the adapters (LLM, diffusion, storage, memory)
    to implement a complete generation workflow.
    """

    @abstractmethod
    def run(
        self,
        prompt: Prompt,
        style: PromptStyle = PromptStyle.FLUX,
        params: GenerationParams | None = None,
        output_name: str | None = None,
        enhance: bool = True,
    ) -> Path:
        """Execute the full generation pipeline.

        Args:
            prompt: The raw (or partially enhanced) prompt.
            style: Which prompt-enhancement style to use.
            params: Generation parameters (width, height, steps, seed, etc.).
                If ``None``, sensible defaults are used.
            output_name: Optional base filename (without extension) for the
                saved image. Auto-generated if omitted.
            enhance: Whether to run prompt enhancement via the LLM.

        Returns:
            Path to the saved image file.
        """
        ...
