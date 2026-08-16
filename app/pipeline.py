import gc
import importlib.util
import threading

import torch

from app.config import MODEL_NAME

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

DEVICE = "cuda"

_HAVE_TORCHAO = importlib.util.find_spec("torchao") is not None
_HAVE_BITSANDBYTES = importlib.util.find_spec("bitsandbytes") is not None


class ImagePipeline:
    """Lazily loads a FLUX.2-klein-4B pipeline quantized to fit 8GB VRAM.

    Quantization strategy (best-effort, graceful fallback):
      1. torchao FP8 for the transformer if torchao is installed
      2. Plain bfloat16 + full CPU offload otherwise (proven on this machine)
    """

    def __init__(self):
        self._pipe = None
        self._lock = threading.Lock()
        self._quantization = "none"
        self._load_error = None

    @property
    def quantization(self):
        return self._quantization

    @property
    def load_error(self):
        return self._load_error

    @property
    def loaded(self):
        return self._pipe is not None

    def load(self):
        with self._lock:
            if self._pipe is not None:
                return self._pipe
            self._pipe, self._quantization = self._build_pipe()
            return self._pipe

    def _build_pipe(self):
        from diffusers import Flux2KleinPipeline

        # 1. Try torchao FP8 transformer if available
        transformer = None
        quantization = "none"
        if _HAVE_TORCHAO:
            try:
                from diffusers import Flux2Transformer2DModel, TorchAoConfig
                from torchao.quantization import (
                    Float8DynamicActivationFloat8WeightConfig,
                )

                transformer = Flux2Transformer2DModel.from_pretrained(
                    MODEL_NAME,
                    subfolder="transformer",
                    torch_dtype=torch.bfloat16,
                    quantization_config=TorchAoConfig(
                        Float8DynamicActivationFloat8WeightConfig()
                    ),
                    device_map="cuda",
                )
                quantization = "fp8"
            except Exception as exc:  # noqa: BLE001 - fall back gracefully
                transformer = None
                quantization = "none"
                self._load_error = f"torchao FP8 failed, falling back to bf16: {exc}"

        # 2. Load pipeline (with FP8 transformer if we got one)
        from_pretrained_kwargs = {"torch_dtype": torch.bfloat16}
        if transformer is not None:
            from_pretrained_kwargs["transformer"] = transformer

        pipe = Flux2KleinPipeline.from_pretrained(
            MODEL_NAME,
            **from_pretrained_kwargs,
        )

        # 3. Memory optimizations — essential on 8GB VRAM.
        # Proven config: model CPU offload + attention slicing keeps peak
        # VRAM under 8GB for FLUX.2-klein-4B (matches vision_model.py).
        pipe.enable_model_cpu_offload()
        pipe.enable_attention_slicing()

        return pipe, quantization

    def generate(self, prompt, width, height, steps, guidance_scale, seed, negative_prompt=None):
        if not self.loaded:
            self.load()
        pipe = self._pipe

        generator = None
        if seed is not None:
            generator = torch.Generator(device=DEVICE).manual_seed(seed)

        kwargs = {
            "prompt": prompt,
            "height": height,
            "width": width,
            "num_inference_steps": steps,
            "guidance_scale": guidance_scale,
            "generator": generator,
        }
        if negative_prompt:
            kwargs["negative_prompt"] = negative_prompt

        try:
            image = pipe(**kwargs).images[0]
        finally:
            self.cleanup()

        return image

    def cleanup(self):
        """Free GPU memory after each generation."""
        pipe = self._pipe
        if pipe is not None:
            try:
                pipe.to("cpu")
            except Exception:  # noqa: BLE001 - offload already happened
                pass
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


_pipeline = ImagePipeline()


def get_pipeline():
    return _pipeline