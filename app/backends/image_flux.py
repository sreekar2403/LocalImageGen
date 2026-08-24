"""FLUX.2-klein-4B image backend: text-to-image and multi-reference editing.

Measured on an RTX 4060 Laptop (8GB) -- see bench.md:

    bf16 + cpu offload   115.7 s warm, peak 7996 MiB, reserved 8654 MiB
    FP8 (both modules)    10.4 s warm, peak 5001 MiB, reserved 5808 MiB

The bf16 path reserved MORE than the 8188 MiB the card physically has, i.e. the
driver was spilling to system RAM over PCIe on every denoise step. Cause:
`model_cpu_offload_seq = "text_encoder->transformer->vae"` moves one whole
module at a time, and in bf16 each of the two big ones exceeds the ~7 GB free:

    text_encoder (Qwen3ForCausalLM)      7673 MB   <-- the BIGGER one
    transformer  (Flux2Transformer2DModel) 7393 MB
    vae                                    161 MB

So both must be quantized. Quantizing only the transformer does not fix it.
"""

from __future__ import annotations

import gc
import time
from pathlib import Path
from typing import Any

import torch

from app.backends.base import Artifact, JobCancelled, Progress
from app.config import (
    MODEL_NAME,
    WARN_GUIDANCE,
    WARN_NEGATIVE,
)

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

DEVICE = "cuda"


def _torchao_config():
    """torchao FP8 config objects, one per owning library.

    diffusers 0.39 / transformers 5.15 REJECT string quant types ("float8dq",
    "int8wo", ...). An AOBaseConfig object is required, and the transformer
    needs diffusers' wrapper while the text_encoder needs transformers'.
    """
    from diffusers import PipelineQuantizationConfig
    from diffusers import TorchAoConfig as DiffusersTorchAo
    from torchao.quantization import Float8DynamicActivationFloat8WeightConfig
    from transformers import TorchAoConfig as TransformersTorchAo

    return PipelineQuantizationConfig(
        quant_mapping={
            "transformer": DiffusersTorchAo(Float8DynamicActivationFloat8WeightConfig()),
            "text_encoder": TransformersTorchAo(Float8DynamicActivationFloat8WeightConfig()),
        }
    )


def _int8_config():
    from diffusers import PipelineQuantizationConfig
    from diffusers import TorchAoConfig as DiffusersTorchAo
    from torchao.quantization import Int8WeightOnlyConfig
    from transformers import TorchAoConfig as TransformersTorchAo

    return PipelineQuantizationConfig(
        quant_mapping={
            "transformer": DiffusersTorchAo(Int8WeightOnlyConfig()),
            "text_encoder": TransformersTorchAo(Int8WeightOnlyConfig()),
        }
    )


class ImageFluxBackend:
    name = "image.flux2-klein"
    kinds = ("image", "edit")
    needs_gpu = True
    vram_estimate_mb = 5100  # measured peak under FP8

    def __init__(self, model_name: str = MODEL_NAME) -> None:
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
        from diffusers import Flux2KleinPipeline

        # Ladder: fp8 -> int8 -> bf16. fp8 is the only mode that actually fits
        # this card; the others exist for different hardware.
        errors: list[str] = []
        for label, factory in (("fp8", _torchao_config), ("int8", _int8_config), ("bf16", None)):
            try:
                kwargs: dict[str, Any] = {"torch_dtype": torch.bfloat16}
                if factory is not None:
                    kwargs["quantization_config"] = factory()
                # Never pass a pipeline-level device_map here:
                # enable_model_cpu_offload() raises ValueError when
                # _is_pipeline_device_mapped() is true (pipeline_utils.py:1207).
                pipe = Flux2KleinPipeline.from_pretrained(self.model_name, **kwargs)
                self._pipe = pipe
                self._quantization = label
                break
            except Exception as exc:  # noqa: BLE001 - try the next rung
                errors.append(f"{label}: {type(exc).__name__}: {exc}")
                continue

        if self._pipe is None:
            self._load_error = " | ".join(errors)
            raise RuntimeError(f"all quantization modes failed -- {self._load_error}")

        self._load_error = " | ".join(errors) or None
        self._pipe.enable_model_cpu_offload()
        self._pipe.enable_attention_slicing()

    def unload(self) -> None:
        """Actually free the VRAM.

        `enable_model_cpu_offload()` installs accelerate hook chains that hold
        module references, so dropping `_pipe` alone leaves the weights alive.
        remove_all_hooks() (pipeline_utils.py:1181) must run first. The old
        cleanup() did `pipe.to("cpu")` and KEPT `_pipe`, which freed nothing.
        """
        pipe, self._pipe = self._pipe, None
        if pipe is None:
            return
        try:
            pipe.remove_all_hooks()
        except Exception:  # noqa: BLE001 - not all pipelines expose it
            pass
        for component in ("transformer", "text_encoder", "vae"):
            try:
                setattr(pipe, component, None)
            except Exception:  # noqa: BLE001
                pass
        del pipe
        self._quantization = "none"
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

    # --- generation ----------------------------------------------------------

    def generate(self, params: dict[str, Any], progress: Progress | None = None) -> Artifact:
        from PIL import Image

        if not self.loaded:
            self.load()
        pipe = self._pipe

        prompt: str = params["prompt"]
        width: int = params["width"]
        height: int = params["height"]
        steps: int = params.get("steps") or 4
        seed = params.get("seed")
        out_path = Path(params["out_path"])
        warnings: list[str] = list(params.get("warnings", []))

        # Both of these are inert on a distilled model. Accept them for API
        # compatibility, ignore them, and say so rather than lying.
        if params.get("guidance_scale") is not None and not params.get("_suppress_guidance_warning"):
            warnings.append(WARN_GUIDANCE)
        if params.get("negative_prompt"):
            warnings.append(WARN_NEGATIVE)

        generator = None
        if seed is not None:
            generator = torch.Generator(device=DEVICE).manual_seed(int(seed))

        call: dict[str, Any] = {
            "prompt": prompt,
            "height": height,
            "width": width,
            "num_inference_steps": steps,
            "generator": generator,
        }

        # Multi-reference editing is native: `image` is __call__'s first arg.
        refs = params.get("reference_images") or []
        if refs:
            images = [Image.open(p).convert("RGB") for p in refs]
            call["image"] = images if len(images) > 1 else images[0]

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
            kind="edit" if refs else "image",
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
            meta={"peak_vram_mb": peak_mb, "reference_images": [str(r) for r in refs]},
        )
