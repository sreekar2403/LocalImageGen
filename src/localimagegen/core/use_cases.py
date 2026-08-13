"""Application use cases orchestrating the generation pipeline."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

from localimagegen.config.settings import Settings
from localimagegen.core.models import (
    GeneratedImage,
    GenerationParams,
    Prompt,
    PromptStyle,
)
from localimagegen.pipelines.flux import FluxPipeline

log = logging.getLogger(__name__)


@dataclass
class GenerateImageRequest:
    """Input for a single image generation request."""

    prompt: str
    style: PromptStyle = PromptStyle.FLUX
    width: int | None = None
    height: int | None = None
    steps: int | None = None
    seed: int | None = None
    output_name: str | None = None
    enhance: bool = True


@dataclass
class GenerateImageResponse:
    """Output from a single image generation request."""

    image_path: Path
    prompt_used: str
    enhanced_prompt: str | None = None


class GenerateImageUseCase:
    """Orchestrates a single image generation request through the pipeline."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.pipeline = FluxPipeline(settings)

    def execute(self, request: GenerateImageRequest) -> GenerateImageResponse:
        """Run the full generation pipeline and return the result."""
        prompt = Prompt(text=request.prompt)

        params = GenerationParams(
            width=request.width or self.settings.width,
            height=request.height or self.settings.height,
            num_inference_steps=request.steps or self.settings.num_inference_steps,
            seed=request.seed or self.settings.seed,
            device=self.settings.device,
        )

        style = request.style

        image_path = self.pipeline.run(
            prompt=prompt,
            style=style,
            params=params,
            output_name=request.output_name,
        )

        return GenerateImageResponse(
            image_path=image_path,
            prompt_used=prompt.text,
            enhanced_prompt=prompt.enhanced,
        )
