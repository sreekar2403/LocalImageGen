"""Stable Diffusion 3.5 Medium: text-to-image generation.

Replaces the FLUX.1-dev GGUF attempt: enable_sequential_cpu_offload() is
fundamentally incompatible with GGUF-quantized weights in diffusers 0.39.0 --
accelerate's device-hook machinery loses a GGUFParameter's `quant_type`
metadata when moving it to the `meta` device, and diffusers' own GGUF code
then crashes with a bare `KeyError(None)` inside GGML_QUANT_SIZES[quant_type].
That is what "generation failed: None" was -- see bench.md "Phase 1 (abandoned)".

SD3.5 Medium (2.5B, MMDiT) needs no GGUF at all: plain bf16 from_pretrained()
+ enable_model_cpu_offload() -- the same simple pattern klein already uses
successfully. Its optional third text encoder (T5-XXL, ~9GB standalone) is
dropped entirely (`text_encoder_3=None`), which is a documented low-VRAM SD3.5
pattern, not a workaround invented here -- CLIP-L + OpenCLIP-G alone still
give real prompt conditioning and real classifier-free guidance, just with
somewhat less fine-grained prompt adherence than the full triple-encoder
setup would.
"""

from __future__ import annotations

import gc
import time
from pathlib import Path
from typing import Any

import torch

from app.backends._imaging import apply_text_overlay
from app.backends.base import Artifact, JobCancelled, Progress
from app.config import DEFAULT_GUIDANCE_SCALE_SD3, DEFAULT_STEPS_SD3, SD3_MODEL

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

DEVICE = "cuda"


class SD3Backend:
    name = "image.sd3.5-medium"
    kinds = ("image",)
    needs_gpu = True
    vram_estimate_mb = 6000  # unverified -- see bench.md "Phase 2"

    def __init__(self, model_name: str = SD3_MODEL) -> None:
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
        from diffusers import StableDiffusion3Pipeline

        try:
            # Drop the optional T5-XXL third encoder entirely -- it alone is
            # ~9GB in fp16/bf16, the exact problem that broke the FLUX.1-dev
            # attempt. CLIP-L + OpenCLIP-G still give real conditioning + CFG.
            pipe = StableDiffusion3Pipeline.from_pretrained(
                self.model_name,
                torch_dtype=torch.bfloat16,
                text_encoder_3=None,
                tokenizer_3=None,
            )
            self._pipe = pipe
            self._quantization = "bf16-no-t5"
            self._load_error = None
        except Exception as exc:  # noqa: BLE001
            self._load_error = f"{type(exc).__name__}: {exc}"
            raise RuntimeError(f"SD3.5 load failed -- {self._load_error}") from exc

        self._pipe.enable_model_cpu_offload()
        self._pipe.vae.enable_slicing()
        self._pipe.vae.enable_tiling()

    def unload(self) -> None:
        """Same remove_all_hooks()-then-drop pattern as ImageFluxBackend."""
        pipe, self._pipe = self._pipe, None
        if pipe is None:
            return
        try:
            pipe.remove_all_hooks()
        except Exception:  # noqa: BLE001
            pass
        for component in ("transformer", "text_encoder", "text_encoder_2", "vae"):
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
        if not self.loaded:
            self.load()
        pipe = self._pipe

        prompt: str = params["prompt"]
        width: int = params["width"]
        height: int = params["height"]
        steps: int = params.get("steps") or DEFAULT_STEPS_SD3
        guidance_scale = params.get("guidance_scale")
        if guidance_scale is None:
            guidance_scale = DEFAULT_GUIDANCE_SCALE_SD3
        negative_prompt = params.get("negative_prompt")
        seed = params.get("seed")
        out_path = Path(params["out_path"])
        warnings: list[str] = list(params.get("warnings", []))

        text_overlay = params.get("text_overlay")
        text_position = params.get("text_position", "bottom")
        text_color = params.get("text_color", "#FFFFFF")
        text_bg_color = params.get("text_bg_color", "#00000080")
        text_font_size = params.get("text_font_size", 48)
        text_padding = params.get("text_padding", 20)

        generator = None
        if seed is not None:
            generator = torch.Generator(device=DEVICE).manual_seed(int(seed))

        call: dict[str, Any] = {
            "prompt": prompt,
            "height": height,
            "width": width,
            "num_inference_steps": steps,
            "guidance_scale": guidance_scale,
            "generator": generator,
        }
        if negative_prompt:
            call["negative_prompt"] = negative_prompt

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
        if peak_mb > 6500:
            warnings.append(
                f"peak VRAM {peak_mb} MiB is close to the limit -- the driver may be "
                "spilling to system RAM, which is very slow"
            )

        return Artifact(
            path=out_path,
            kind="image",
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
            meta={"peak_vram_mb": peak_mb, "guidance_scale": guidance_scale},
        )
