"""Diffusers-backed FLUX diffusion adapter.

Implements :class:`localimagegen.core.ports.DiffusionPort` using the
``diffusers`` ``Flux2KleinPipeline``. The adapter loads the pipeline from the
configured Hugging Face model id, runs a single generation, encodes the result
as PNG bytes, and releases GPU memory afterwards.
"""

from __future__ import annotations

import gc
import io
import time

import torch
from diffusers import Flux2KleinPipeline

from localimagegen.config.settings import Settings
from localimagegen.core.models import (
    GeneratedImage,
    GenerationMetadata,
    GenerationParams,
    Prompt,
)
from localimagegen.core.ports import MemoryManagerPort
from localimagegen.services.logging import get_logger


class FluxAdapter:
    """Generate images with a local FLUX.2 Klein diffusion model.

    Args:
        settings: Application settings (``vision_model`` selects the model).
        memory_manager: Optional memory manager used to track and offload the
            loaded pipeline.
    """

    def __init__(
        self,
        settings: Settings,
        memory_manager: MemoryManagerPort | None = None,
    ) -> None:
        self._settings = settings
        self._memory_manager = memory_manager
        self._logger = get_logger(__name__)
        self._pipe: Flux2KleinPipeline | None = None
        self._device = self._resolve_device(settings.device)

        # Match the legacy scripts' TF32 optimizations for Ampere+ GPUs.
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

        if self._memory_manager is not None:
            self._memory_manager.register("flux", self._device, offload=self._offload_pipe)

    @staticmethod
    def _resolve_device(requested: str) -> str:
        """Return ``cuda`` when requested and available, otherwise ``cpu``."""
        if requested == "cuda" and torch.cuda.is_available():
            return "cuda"
        return "cpu"

    def _dtype(self) -> torch.dtype:
        """Pick bfloat16 on CUDA and float32 on CPU."""
        return torch.bfloat16 if self._device == "cuda" else torch.float32

    def _load_pipeline(self) -> Flux2KleinPipeline:
        """Load (or reuse) the FLUX pipeline from the configured model id."""
        if self._pipe is None:
            self._logger.info(
                "loading_flux_pipeline",
                model=self._settings.vision_model,
                device=self._device,
            )
            self._pipe = Flux2KleinPipeline.from_pretrained(
                self._settings.vision_model,
                torch_dtype=self._dtype(),
            )
            self._pipe.enable_model_cpu_offload()
            self._pipe.enable_attention_slicing()
        return self._pipe

    def _offload_pipe(self) -> None:
        """Move the pipeline back to CPU and drop the reference."""
        if self._pipe is not None:
            try:
                self._pipe.to("cpu")
            except Exception:  # pragma: no cover - defensive cleanup
                pass
            self._pipe = None

    def generate(self, prompt: Prompt, params: GenerationParams) -> GeneratedImage:
        """Generate an image from ``prompt`` and return it with metadata.

        GPU memory is released in a ``finally`` block so that even a failed
        generation does not leave the model resident on the device.
        """
        start = time.perf_counter()
        try:
            pipe = self._load_pipeline()
            generator = torch.Generator(device=self._device).manual_seed(params.seed)
            image = pipe(
                prompt=prompt.effective,
                height=params.height,
                width=params.width,
                num_inference_steps=params.num_inference_steps,
                generator=generator,
            ).images[0]

            buffer = io.BytesIO()
            image.save(buffer, format="PNG")
            data = buffer.getvalue()
        finally:
            self.cleanup()

        duration = time.perf_counter() - start
        metadata = GenerationMetadata(
            model=self._settings.vision_model,
            prompt=prompt.effective,
            params=params,
            duration_seconds=duration,
        )
        return GeneratedImage(data=data, format="png", metadata=metadata)

    def metadata(self) -> GenerationMetadata:
        """Return metadata describing the underlying diffusion model."""
        return GenerationMetadata(
            model=self._settings.vision_model,
            prompt="",
            params=GenerationParams(device=self._device),
        )

    def cleanup(self) -> None:
        """Offload the pipeline and release GPU memory."""
        if self._memory_manager is not None:
            self._memory_manager.offload("flux")
        else:
            self._offload_pipe()
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()