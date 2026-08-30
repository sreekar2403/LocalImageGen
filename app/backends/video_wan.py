"""Wan2.1-T2V-1.3B video backend.

THE CONSTRAINT. Component sizes on disk (both stored fp32):

    text_encoder  UMT5EncoderModel, d_model 4096, 24 layers   21.7 GB  -> ~10.8 GB bf16
    transformer   30 layers, dim 1536                          5.4 GB  -> ~2.7 GB bf16
    vae           AutoencoderKLWan                             0.5 GB  -> ~0.25 GB

The transformer and VAE fit on an 8 GB card comfortably. The UMT5-XXL text
encoder never will -- not at 10.8 GB against ~7 GB usable. A naive
`WanPipeline.from_pretrained(...).enable_model_cpu_offload()` with a `prompt=`
would thrash or die.

So this runs in two stages:

    A. CPU: load the text encoder, compute prompt embeddings, free it. Costs
       ~11 GB of the 32 GB system RAM, briefly.
    B. GPU: load the pipeline with `text_encoder=None` and pass the precomputed
       embeddings. `encode_prompt()` short-circuits when embeds are supplied
       (`pipeline_wan.py:246`), and `check_inputs` requires exactly one of
       prompt/prompt_embeds -- so `prompt` must be None.

Unlike FLUX.2-klein, Wan is NOT distilled: `do_classifier_free_guidance` is
`guidance_scale > 1.0`, so guidance and negative prompts genuinely work, and
each step runs two forward passes.
"""

from __future__ import annotations

import gc
import time
from pathlib import Path
from typing import Any

import torch

from app.backends.base import Artifact, JobCancelled, Progress
from app.config import VIDEO_MODEL

# The reference implementation's negative prompt, in English.
DEFAULT_NEGATIVE = (
    "bright tones, overexposed, static, blurred details, subtitles, style, artwork, "
    "painting, picture, still, overall gray, worst quality, low quality, JPEG artifacts, "
    "ugly, deformed, extra fingers, poorly drawn hands, poorly drawn face, malformed limbs, "
    "fused fingers, motionless frame, cluttered background, three legs, crowded background, "
    "walking backwards"
)

MAX_SEQUENCE_LENGTH = 226


def snap_frames(n: int) -> int:
    """Wan's temporal VAE requires (num_frames - 1) % 4 == 0."""
    n = max(5, int(n))
    return n - ((n - 1) % 4)


def _prompt_clean(text: str) -> str:
    from diffusers.pipelines.wan.pipeline_wan import prompt_clean

    return prompt_clean(text)


def encode_prompts_on_cpu(
    prompt: str,
    negative_prompt: str,
    model_name: str = VIDEO_MODEL,
    progress: Progress | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Stage A. Replicates `_get_t5_prompt_embeds` (pipeline_wan.py:158) on CPU.

    The text encoder is loaded, used, and dropped inside this function so it is
    never resident while the GPU stages run.
    """
    from transformers import AutoTokenizer, UMT5EncoderModel

    if progress:
        progress(0.02, "loading text encoder (CPU)")

    tokenizer = AutoTokenizer.from_pretrained(model_name, subfolder="tokenizer")
    text_encoder = UMT5EncoderModel.from_pretrained(
        model_name, subfolder="text_encoder", torch_dtype=torch.bfloat16
    )
    text_encoder.eval()

    def embed(text: str) -> torch.Tensor:
        cleaned = [_prompt_clean(text)]
        inputs = tokenizer(
            cleaned,
            padding="max_length",
            max_length=MAX_SEQUENCE_LENGTH,
            truncation=True,
            add_special_tokens=True,
            return_attention_mask=True,
            return_tensors="pt",
        )
        ids, mask = inputs.input_ids, inputs.attention_mask
        seq_lens = mask.gt(0).sum(dim=1).long()
        with torch.no_grad():
            out = text_encoder(ids, mask).last_hidden_state
        out = out.to(dtype=torch.bfloat16)
        trimmed = [u[:v] for u, v in zip(out, seq_lens)]
        return torch.stack(
            [
                torch.cat([u, u.new_zeros(MAX_SEQUENCE_LENGTH - u.size(0), u.size(1))])
                for u in trimmed
            ],
            dim=0,
        )

    try:
        if progress:
            progress(0.06, "encoding prompt (CPU)")
        positive = embed(prompt)
        negative = embed(negative_prompt)
    finally:
        del text_encoder, tokenizer
        gc.collect()

    return positive, negative


class VideoWanBackend:
    name = "video.wan21-t2v-1.3b"
    kinds = ("video",)
    needs_gpu = True
    vram_estimate_mb = 4200

    def __init__(self, model_name: str = VIDEO_MODEL) -> None:
        self.model_name = model_name
        self._pipe = None
        self._load_error: str | None = None

    @property
    def loaded(self) -> bool:
        return self._pipe is not None

    @property
    def quantization(self) -> str:
        return "bf16"

    @property
    def load_error(self) -> str | None:
        return self._load_error

    def load(self) -> None:
        """Stage B setup. Loads WITHOUT the text encoder."""
        if self._pipe is not None:
            return
        from diffusers import AutoencoderKLWan, UniPCMultistepScheduler, WanPipeline

        # The VAE is published in fp32; keeping it there avoids decode artifacts
        # and costs only ~0.5 GB.
        vae = AutoencoderKLWan.from_pretrained(
            self.model_name, subfolder="vae", torch_dtype=torch.float32
        )
        pipe = WanPipeline.from_pretrained(
            self.model_name,
            vae=vae,
            text_encoder=None,  # deliberate: 21.7 GB, handled on CPU in stage A
            torch_dtype=torch.bfloat16,
        )
        assert pipe.text_encoder is None, "text encoder must not be resident on the GPU"

        # flow_shift 3.0 is the reference setting for 480p (5.0 for 720p).
        pipe.scheduler = UniPCMultistepScheduler.from_config(
            pipe.scheduler.config, flow_shift=3.0
        )
        pipe.vae.enable_tiling()  # AutoencoderKLWan.enable_tiling, line 1093
        pipe.enable_model_cpu_offload()

        self._pipe = pipe

    def unload(self) -> None:
        pipe, self._pipe = self._pipe, None
        if pipe is None:
            return
        try:
            pipe.remove_all_hooks()
        except Exception:  # noqa: BLE001
            pass
        for component in ("transformer", "transformer_2", "vae"):
            try:
                setattr(pipe, component, None)
            except Exception:  # noqa: BLE001
                pass
        del pipe
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

    def generate(self, params: dict[str, Any], progress: Progress | None = None) -> Artifact:
        from app.ffmpeg import contact_sheet, encode_video

        prompt: str = params["prompt"]
        negative: str = params.get("negative_prompt") or DEFAULT_NEGATIVE
        width: int = int(params.get("width") or 832)
        height: int = int(params.get("height") or 480)
        steps: int = int(params.get("steps") or 20)
        guidance: float = float(params.get("guidance_scale") or 5.0)
        fps: int = int(params.get("fps") or 16)
        seed = params.get("seed")
        out_path = Path(params["out_path"])
        warnings: list[str] = list(params.get("warnings", []))

        num_frames = snap_frames(params.get("num_frames") or 33)
        if num_frames != (params.get("num_frames") or 33):
            warnings.append(
                f"num_frames adjusted to {num_frames} ((n-1) must be divisible by 4)"
            )
        if width % 16 or height % 16:
            raise ValueError(f"width and height must be divisible by 16, got {width}x{height}")

        cancelled = params.get("is_cancelled")
        start = time.perf_counter()

        # --- Stage A: CPU text encoding -------------------------------------
        positive_embeds, negative_embeds = encode_prompts_on_cpu(
            prompt, negative, self.model_name, progress
        )
        encode_s = time.perf_counter() - start

        if cancelled and cancelled():
            raise JobCancelled("cancelled before denoise")

        # --- Stage B: GPU denoise -------------------------------------------
        if progress:
            progress(0.12, "loading transformer + VAE (GPU)")
        if not self.loaded:
            self.load()
        pipe = self._pipe

        device = pipe._execution_device
        positive_embeds = positive_embeds.to(device)
        negative_embeds = negative_embeds.to(device)

        generator = None
        if seed is not None:
            generator = torch.Generator(device="cuda").manual_seed(int(seed))

        def _cb(pipeline, i, t, cbk):
            if cancelled and cancelled():
                raise JobCancelled("cancelled during denoise")
            if progress:
                frac = 0.15 + 0.7 * ((i + 1) / max(steps, 1))
                progress(frac, f"denoise {i + 1}/{steps}")
            return cbk

        torch.cuda.reset_peak_memory_stats()
        denoise_start = time.perf_counter()
        result = pipe(
            prompt=None,  # check_inputs rejects prompt AND prompt_embeds together
            negative_prompt=None,
            prompt_embeds=positive_embeds,
            negative_prompt_embeds=negative_embeds,
            height=height,
            width=width,
            num_frames=num_frames,
            num_inference_steps=steps,
            guidance_scale=guidance,
            generator=generator,
            output_type="np",
            callback_on_step_end=_cb,
        )
        frames = result.frames[0]
        denoise_s = time.perf_counter() - denoise_start
        peak_mb = round(torch.cuda.max_memory_allocated() / (1024 * 1024))

        # --- Stage C: encode -------------------------------------------------
        if progress:
            progress(0.9, "encoding mp4")
        encode_video(frames, out_path, fps=fps, codec=params.get("codec") or "libx264")

        sheet = out_path.with_suffix(".contact.png")
        contact_sheet(frames, sheet)

        if progress:
            progress(1.0, "done")

        duration = num_frames / fps
        return Artifact(
            path=out_path,
            kind="video",
            mime="video/mp4",
            width=width,
            height=height,
            preview_path=sheet,
            prompt_used=prompt,
            model=self.model_name,
            backend=self.name,
            quantization="bf16",
            seed=seed,
            steps=steps,
            elapsed_s=time.perf_counter() - start,
            warnings=warnings,
            meta={
                "num_frames": num_frames,
                "fps": fps,
                "duration_s": round(duration, 2),
                "guidance_scale": guidance,
                "peak_vram_mb": peak_mb,
                "text_encode_s": round(encode_s, 1),
                "denoise_s": round(denoise_s, 1),
                "contact_sheet": str(sheet),
            },
        )
