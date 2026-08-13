"""FLUX image generation pipeline."""

from __future__ import annotations

import logging
from pathlib import Path

from localimagegen.adapters.diffusion.flux import FluxAdapter
from localimagegen.adapters.llm.ollama import OllamaAdapter
from localimagegen.adapters.memory.unified import UnifiedMemoryManager
from localimagegen.adapters.storage.local import LocalFileStorage
from localimagegen.config.settings import Settings
from localimagegen.core.models import GenerationParams, Prompt, PromptStyle
from localimagegen.pipelines.base import BasePipeline

log = logging.getLogger(__name__)


class FluxPipeline(BasePipeline):
    """Full generation pipeline: Ollama prompt enhancement → FLUX diffusion → save."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.memory = UnifiedMemoryManager()
        self.llm = OllamaAdapter(settings, self.memory)
        self.diffusion = FluxAdapter(settings, self.memory)
        self.storage = LocalFileStorage(settings)

    def run(
        self,
        prompt: Prompt,
        style: PromptStyle = PromptStyle.FLUX,
        params: GenerationParams | None = None,
        output_name: str | None = None,
    ) -> Path:
        """Run the full pipeline: enhance → generate → save."""
        # Step 1: Enhance prompt via Ollama
        log.info("Enhancing prompt with Ollama (model=%s)", self.settings.ollama_model)
        enhanced = self.llm.enhance(prompt, style)
        log.info("Enhanced prompt: %s", enhanced.effective[:120])

        # Step 2: Build default params if not provided
        if params is None:
            params = GenerationParams(
                width=self.settings.width,
                height=self.settings.height,
                num_inference_steps=self.settings.num_inference_steps,
                seed=self.settings.seed,
                device=self.settings.device,
            )

        # Step 3: Generate image via FLUX
        log.info(
            "Generating image (model=%s, %dx%d, steps=%d, seed=%d)",
            self.settings.vision_model,
            params.width,
            params.height,
            params.num_inference_steps,
            params.seed,
        )
        image = self.diffusion.generate(enhanced, params)

        # Step 4: Save to disk
        path = self.storage.save(image, output_name)
        log.info("Saved image to %s", path)

        return path
