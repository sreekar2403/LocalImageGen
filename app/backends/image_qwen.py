"""Qwen-Image family: text-to-image generation and instruction editing.

Replaces the SD3.5-Medium / InstructPix2Pix split with the Qwen family:

  * Qwen/Qwen-Image (or Qwen/Qwen-Image-2512) — MMDiT 20B + Qwen2.5-VL 7B
    for pure text-to-image generation. Documented as strong at text
    rendering, bilingual, and general image quality.

  * Qwen/Qwen-Image-Edit-2511 — same MMDiT transformer but trained for
    instruction-guided editing (single and multi-image). The
    QwenImageEditPlusPipeline supports up to 3 reference images and
    preserves the same text encoder / VAE as the generation model,
    so the pair is at least a unified architecture family unlike
    SD3 (MMDiT) vs pix2pix (SD1.5 UNet).

Qwen-Image-2.0 claims single-checkpoint unification of T2I+TI2I, but
Edit-2511 is the latest proven edit checkpoint with diffusers support
in 0.39.0 and Apache-2.0 license. Generation and edit therefore share
the same manager GPU slot but are two loads. A true single-checkpoint
path (load Edit 2511 for both and pass image=None) is not supported
by the EditPlus pipeline's preprocessing (it expects image), so the
split stays — just within one vendor family.

VRAM on RTX 4060 Laptop 8GB (8188 MiB, ~7GB free):
  Qwen-Image is 20B DiT + 7B VL + VAE. BF16 requires >20GB, FP8 ~20GB,
  Q4_K_M ~13GB, Q3_K_M ~9.8GB, Q2_K 7.5GB. Even with
  enable_model_cpu_offload() a whole-module 7B encoder (14GB BF16)
  exceeds free VRAM, so OOM is expected on 8GB without quantization.
  We try bf16 + enable_model_cpu_offload() first, then fall back to
  enable_sequential_cpu_offload() (<3GB but slower). For 8GB production
  the recommended path is GGUF Q3/Q2 + offloaded VL to CPU RAM (see
  Qwen-Image Comfy docs) or Qwen-Image-Lightning 4-step LoRA. This
  backend keeps the simple offload path and surfaces OOM clearly so
  the user can opt into GGUF if needed.

Flux.2-klein stays registered as image.flux2-klein fallback and
svg.trace continues to use it until its raster quality is proven
replaceable by Qwen.

See bench.md Phase 1/2 for the flux/pix2pix history.
"""

from __future__ import annotations

import gc
import time
from pathlib import Path
from typing import Any

import torch

from app.backends._imaging import apply_text_overlay
from app.backends.base import Artifact, JobCancelled, Progress

# Default models — prefer 2511/2512 if available, fall back to base names.
# Env overrides allow local mirrors or GGUF variants.
from app.config import (
    QWEN_IMAGE_MODEL,
    QWEN_EDIT_MODEL,
    DEFAULT_STEPS_QWEN,
    DEFAULT_TRUE_CFG_QWEN,
)

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

DEVICE = "cuda"


def _load_with_offload(pipe):
    """Pick the best offload strategy for the available VRAM.

    model_cpu_offload loads entire modules (~14GB for the 7B VL encoder)
    to GPU at once — OOMs on 8GB. Sequential offload moves one layer at
    a time (<3GB peak VRAM) but is slower. We detect VRAM and skip
    directly to sequential for cards with <12GB free.
    """
    try:
        free, _ = torch.cuda.mem_get_info(0)
        free_gb = free / (1024 ** 3)
    except Exception:
        free_gb = 0.0

    if free_gb >= 12.0:
        try:
            pipe.enable_model_cpu_offload()
            return "cpu_offload"
        except Exception:
            pass

    # <12GB or model_cpu_offload failed — use sequential (safe for 8GB)
    try:
        pipe.enable_sequential_cpu_offload()
        return "sequential"
    except Exception:
        return "none"


class QwenImageBackend:
    """Pure text-to-image via Qwen/Qwen-Image (or 2512)."""

    name = "image.qwen-image"
    kinds = ("image",)
    needs_gpu = True
    vram_estimate_mb = 8000  # BF16 estimate — actual with offload ~7-9GB peak

    def __init__(self, model_name: str = QWEN_IMAGE_MODEL) -> None:
        self.model_name = model_name
        self._pipe = None
        self._offload = "none"
        self._quantization = "bf16"
        self._load_error: str | None = None

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
        try:
            from diffusers import QwenImagePipeline
        except Exception as exc:
            raise RuntimeError(f"QwenImagePipeline not available in diffusers 0.39: {exc}") from exc

        try:
            pipe = QwenImagePipeline.from_pretrained(
                self.model_name,
                torch_dtype=torch.bfloat16,
            )
            self._offload = _load_with_offload(pipe)
            self._pipe = pipe
            self._quantization = f"bf16-{self._offload}"
            self._load_error = None
        except Exception as exc:  # noqa: BLE001
            self._load_error = f"{type(exc).__name__}: {exc}"
            raise RuntimeError(f"Qwen-Image load failed -- {self._load_error}") from exc

    def unload(self) -> None:
        pipe, self._pipe = self._pipe, None
        if pipe is None:
            return
        try:
            pipe.remove_all_hooks()
        except Exception:  # noqa: BLE001
            pass
        for component in ("transformer", "text_encoder", "vae", "processor"):
            try:
                setattr(pipe, component, None)
            except Exception:  # noqa: BLE001
                pass
        del pipe
        self._quantization = "none"
        self._offload = "none"
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

    def generate(self, params: dict[str, Any], progress: Progress | None = None) -> Artifact:
        if not self.loaded:
            self.load()
        pipe = self._pipe

        prompt: str = params["prompt"]
        width: int = params["width"]
        height: int = params["height"]
        steps: int = params.get("steps") or DEFAULT_STEPS_QWEN
        guidance = params.get("guidance_scale")
        # Qwen uses true_cfg_scale for CFG (4.0 default), not guidance_scale
        true_cfg = float(guidance) if guidance is not None else DEFAULT_TRUE_CFG_QWEN
        negative_prompt = params.get("negative_prompt")
        # Qwen requires negative_prompt for CFG when true_cfg>1
        if true_cfg > 1.0 and not negative_prompt:
            negative_prompt = " "  # empty string enables CFG
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
            "true_cfg_scale": true_cfg,
            "generator": generator,
        }
        if negative_prompt is not None:
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
        if peak_mb > 7500:
            warnings.append(
                f"peak VRAM {peak_mb} MiB — driver may spill to system RAM; "
                "consider GGUF Q3_K_M/Q2_K or sequential offload"
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
            meta={"peak_vram_mb": peak_mb, "true_cfg_scale": true_cfg, "offload": self._offload},
        )


class QwenEditBackend:
    """Instruction edit via Qwen/Qwen-Image-Edit-2511 (Plus pipeline, up to 3 images)."""

    name = "image.qwen-edit-2511"
    kinds = ("edit",)
    needs_gpu = True
    vram_estimate_mb = 8000

    def __init__(self, model_name: str = QWEN_EDIT_MODEL) -> None:
        self.model_name = model_name
        self._pipe = None
        self._offload = "none"
        self._quantization = "bf16"
        self._load_error: str | None = None

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
        # Prefer Plus pipeline (2509/2511 multi-image); fall back to base Edit if unavailable
        try:
            from diffusers import QwenImageEditPlusPipeline
            PipeClass = QwenImageEditPlusPipeline
        except Exception:
            from diffusers import QwenImageEditPipeline
            PipeClass = QwenImageEditPipeline  # type: ignore

        try:
            pipe = PipeClass.from_pretrained(
                self.model_name,
                torch_dtype=torch.bfloat16,
            )
            self._offload = _load_with_offload(pipe)
            self._pipe = pipe
            self._quantization = f"bf16-{self._offload}"
            self._load_error = None
        except Exception as exc:  # noqa: BLE001
            self._load_error = f"{type(exc).__name__}: {exc}"
            raise RuntimeError(f"Qwen-Edit load failed -- {self._load_error}") from exc

    def unload(self) -> None:
        pipe, self._pipe = self._pipe, None
        if pipe is None:
            return
        try:
            pipe.remove_all_hooks()
        except Exception:  # noqa: BLE001
            pass
        for component in ("transformer", "text_encoder", "vae", "processor"):
            try:
                setattr(pipe, component, None)
            except Exception:  # noqa: BLE001
                pass
        del pipe
        self._quantization = "none"
        self._offload = "none"
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

    def generate(self, params: dict[str, Any], progress: Progress | None = None) -> Artifact:
        from PIL import Image

        if not self.loaded:
            self.load()
        pipe = self._pipe

        prompt: str = params["prompt"]
        width: int = params.get("width")  # optional — Plus pipeline sizes from image if None
        height: int = params.get("height")
        steps: int = params.get("steps") or DEFAULT_STEPS_QWEN
        guidance = params.get("guidance_scale")
        true_cfg = float(guidance) if guidance is not None else DEFAULT_TRUE_CFG_QWEN
        # image_guidance_scale is not a native Qwen knob; keep for API compat but warn
        image_guidance = params.get("image_guidance_scale")
        seed = params.get("seed")
        out_path = Path(params["out_path"])
        warnings: list[str] = list(params.get("warnings", []))

        if image_guidance is not None:
            warnings.append(
                "image_guidance_scale is not a native Qwen-Image-Edit knob and is ignored "
                "(Qwen uses VAE+VL dual encoding instead)"
            )

        text_overlay = params.get("text_overlay")
        text_position = params.get("text_position", "bottom")
        text_color = params.get("text_color", "#FFFFFF")
        text_bg_color = params.get("text_bg_color", "#00000080")
        text_font_size = params.get("text_font_size", 48)
        text_padding = params.get("text_padding", 20)

        refs = params.get("reference_images") or []
        if not refs:
            raise ValueError(f"{self.name} requires a reference image")
        # Plus pipeline supports up to 3 images; preserve all, warn if >3
        if len(refs) > 3:
            warnings.append(
                f"{self.name}: {len(refs)} reference images supplied, Qwen-Edit-Plus supports up to 3 — only first 3 used"
            )
            refs = refs[:3]

        # Load images as PIL; do not resize to width/height — Plus sizes from first image
        # and uses calculate_dimensions internally. Respect explicit width/height if given.
        pil_images = [Image.open(p).convert("RGB") for p in refs]
        # If service requested explicit dimensions, pass them; else let pipeline infer from image
        # (pipeline_qwenimage_edit_plus: image.size -> 1024*1024 area)
        call: dict[str, Any] = {
            "prompt": prompt,
            "image": pil_images if len(pil_images) > 1 else pil_images[0],
            "num_inference_steps": steps,
            "true_cfg_scale": true_cfg,
            "negative_prompt": " ",
        }
        if width and height:
            call["width"] = width
            call["height"] = height

        generator = None
        if seed is not None:
            generator = torch.Generator(device=DEVICE).manual_seed(int(seed))
            call["generator"] = generator

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

        # Apply text overlay if requested — image is already at pipeline output size,
        # but we report service's requested width/height or actual image size
        if text_overlay:
            image = apply_text_overlay(
                image, text_overlay, text_position, text_color,
                text_bg_color, text_font_size, text_padding
            )

        out_path.parent.mkdir(parents=True, exist_ok=True)
        image.save(out_path)

        peak_mb = round(torch.cuda.max_memory_allocated() / (1024 * 1024))
        # Report actual output size (Qwen may override)
        out_w, out_h = image.size

        return Artifact(
            path=out_path,
            kind="edit",
            mime="image/png",
            width=out_w,
            height=out_h,
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
                "true_cfg_scale": true_cfg,
                "offload": self._offload,
            },
        )
