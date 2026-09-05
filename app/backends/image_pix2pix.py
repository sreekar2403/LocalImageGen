"""InstructPix2Pix: instruction-based image editing.

Replaces the FLUX.1-Kontext-dev GGUF attempt -- same enable_sequential_cpu_
offload()-vs-GGUF incompatibility that broke FLUX.1-dev applies equally here,
since Kontext-dev shares the same loading path. See app/backends/image_sd3.py
and bench.md "Phase 1 (abandoned)" for the root cause.

InstructPix2Pix (timbrooks/instruct-pix2pix, SD1.5-based, ~4GB) is one of the
most mature instruction-editing models in the diffusers ecosystem: plain
fp16 from_pretrained() + enable_model_cpu_offload(), no quantization of any
kind needed, and genuinely supports DUAL classifier-free guidance --
`guidance_scale` (how closely to follow the text instruction) and
`image_guidance_scale` (how closely to preserve the original image), both
real knobs on this pipeline.
"""

from __future__ import annotations

import gc
import time
from pathlib import Path
from typing import Any

import torch

from app.backends._imaging import apply_text_overlay
from app.backends.base import Artifact, JobCancelled, Progress
from app.config import (
    DEFAULT_GUIDANCE_SCALE_PIX2PIX,
    DEFAULT_IMAGE_GUIDANCE_SCALE_PIX2PIX,
    DEFAULT_STEPS_PIX2PIX,
    PIX2PIX_MODEL,
)

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

DEVICE = "cuda"


class Pix2PixBackend:
    name = "image.instruct-pix2pix"
    kinds = ("edit",)
    needs_gpu = True
    vram_estimate_mb = 4000  # unverified -- see bench.md "Phase 2"

    def __init__(self, model_name: str = PIX2PIX_MODEL) -> None:
        self.model_name = model_name
        self._pipe = None
        self._quantization = "none"
        self._load_error: str | None = None

    # --- lifecycle -----------------------------------------------------------

    @property
    def loaded(self) -> bool:
        return self._pipe is not None

    @property
    def quantization(self) -> str:
        return self._quantization

    @property
    def load_error(self) -> str | None:
        return self._load_error

    def load(self) -> None:
        if self._pipe is not None:
            return
        from diffusers import StableDiffusionInstructPix2PixPipeline

        try:
            pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16,
                safety_checker=None,
            )
            self._pipe = pipe
            self._quantization = "fp16"
            self._load_error = None
        except Exception as exc:  # noqa: BLE001
            self._load_error = f"{type(exc).__name__}: {exc}"
            raise RuntimeError(f"InstructPix2Pix load failed -- {self._load_error}") from exc

        self._pipe.enable_model_cpu_offload()
        # NOT enable_tiling(): unnecessary at this model's normal resolutions
        # (SD1.5-based, typically <=768x768, decodes in one pass comfortably).
        # NOTE: a hard vertical "split" artifact was seen in testing and
        # initially suspected to be a tiling seam -- it was NOT: it reproduces
        # identically with tiling disabled. Root cause (confirmed via direct
        # A/B testing): it's a genuine InstructPix2Pix limitation for DRASTIC
        # full-scene transformations (e.g. "turn an indoor desk scene into an
        # ocean sunset") that conflict with image_guidance_scale's pull toward
        # preserving the original composition -- the model produces a
        # half-blended compositing artifact instead of a clean result.
        # Localized edits (lighting, color grading, day/night, style filters,
        # add/remove an object) tested clean with no artifact at all. This is
        # a model capability limit, not a bug in this loading code -- left
        # enable_slicing() in place since it's harmless (a no-op at batch
        # size 1) and costs nothing.
        self._pipe.vae.enable_slicing()

    def unload(self) -> None:
        pipe, self._pipe = self._pipe, None
        if pipe is None:
            return
        try:
            pipe.remove_all_hooks()
        except Exception:  # noqa: BLE001
            pass
        for component in ("unet", "text_encoder", "vae"):
            try:
                setattr(pipe, component, None)
            except Exception:  # noqa: BLE001
                pass
        del pipe
        self._quantization = "none"
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

    # --- generation ------------------------------------------------------------

    def generate(self, params: dict[str, Any], progress: Progress | None = None) -> Artifact:
        from PIL import Image

        if not self.loaded:
            self.load()
        pipe = self._pipe

        prompt: str = params["prompt"]
        width: int = params["width"]
        height: int = params["height"]
        steps: int = params.get("steps") or DEFAULT_STEPS_PIX2PIX
        guidance_scale = params.get("guidance_scale")
        if guidance_scale is None:
            guidance_scale = DEFAULT_GUIDANCE_SCALE_PIX2PIX
        image_guidance_scale = params.get("image_guidance_scale")
        if image_guidance_scale is None:
            image_guidance_scale = DEFAULT_IMAGE_GUIDANCE_SCALE_PIX2PIX
        seed = params.get("seed")
        out_path = Path(params["out_path"])
        warnings: list[str] = list(params.get("warnings", []))

        text_overlay = params.get("text_overlay")
        text_position = params.get("text_position", "bottom")
        text_color = params.get("text_color", "#FFFFFF")
        text_bg_color = params.get("text_bg_color", "#00000080")
        text_font_size = params.get("text_font_size", 48)
        text_padding = params.get("text_padding", 20)

        refs = params.get("reference_images") or []
        if not refs:
            raise ValueError(f"{self.name} requires a reference image")
        if len(refs) > 1:
            warnings.append(
                f"{self.name}: multiple reference images supplied but "
                "InstructPix2Pix only edits one image at a time -- only the "
                "first is used"
            )
        source = Image.open(refs[0]).convert("RGB").resize((width, height))

        generator = None
        if seed is not None:
            generator = torch.Generator(device=DEVICE).manual_seed(int(seed))

        call: dict[str, Any] = {
            "prompt": prompt,
            "image": source,
            "num_inference_steps": steps,
            "guidance_scale": guidance_scale,
            "image_guidance_scale": image_guidance_scale,
            "generator": generator,
        }

        cancelled = params.get("is_cancelled")
        if progress or cancelled:
            def _cb(pipeline, i, t, cbk):
                if cancelled and cancelled():
                    raise JobCancelled("cancelled by request")
                if progress:
                    progress((i + 1) / max(steps, 1), f"denoise {i + 1}/{steps}")
                return cbk

            call["callback_on_step_end"] = _cb

        torch.cuda.reset_peak_memory_stats()
        start = time.perf_counter()
        image = pipe(**call).images[0]
        elapsed = time.perf_counter() - start

        if text_overlay:
            image = apply_text_overlay(
                image, text_overlay, text_position, text_color,
                text_bg_color, text_font_size, text_padding
            )

        out_path.parent.mkdir(parents=True, exist_ok=True)
        image.save(out_path)

        peak_mb = round(torch.cuda.max_memory_allocated() / (1024 * 1024))

        return Artifact(
            path=out_path,
            kind="edit",
            mime="image/png",
            width=width,
            height=height,
            preview_path=out_path,
            prompt_used=prompt,
            model=self.model_name,
            backend=self.name,
            quantization=self._quantization,
            seed=seed,
            steps=steps,
            elapsed_s=elapsed,
            warnings=warnings,
            meta={
                "peak_vram_mb": peak_mb,
                "reference_images": [str(r) for r in refs],
                "guidance_scale": guidance_scale,
                "image_guidance_scale": image_guidance_scale,
            },
        )
