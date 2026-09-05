"""Central configuration. Self-contained on purpose.

This module must import cleanly from ANY working directory: MCP harnesses
(Claude Desktop, pi, deepseek) spawn the server without a useful cwd. The root
`config.py` is consulted only as an optional user override, never required.
"""

from __future__ import annotations

import os
from pathlib import Path

# --- models ------------------------------------------------------------------

_DEFAULT_IMAGE_MODEL = "black-forest-labs/FLUX.2-klein-4B"
_DEFAULT_OLLAMA_MODEL = "qwen2.5-coder:7b"  # SVG authoring is a code task
_DEFAULT_VIDEO_MODEL = "Wan-AI/Wan2.1-T2V-1.3B-Diffusers"

try:  # optional user override; absence is normal and must never break import
    from config import vision_model as _override_image
except Exception:  # noqa: BLE001
    _override_image = None
try:
    from config import ollama_model as _override_ollama
except Exception:  # noqa: BLE001
    _override_ollama = None

MODEL_NAME = os.environ.get("LOCALIMAGEGEN_IMAGE_MODEL") or _override_image or _DEFAULT_IMAGE_MODEL
OLLAMA_MODEL = os.environ.get("LOCALIMAGEGEN_OLLAMA_MODEL") or _override_ollama or _DEFAULT_OLLAMA_MODEL
VIDEO_MODEL = os.environ.get("LOCALIMAGEGEN_VIDEO_MODEL") or _DEFAULT_VIDEO_MODEL

# Image generation history (see bench.md): FLUX.1-dev/Kontext-dev via GGUF was
# tried and abandoned (sequential offload incompatible with GGUF weights);
# SD3.5 Medium + InstructPix2Pix were measured working; Qwen-Image family was
# prototyped. All removed: FLUX.2-klein-4B is now the single image backend.

# --- generation defaults -----------------------------------------------------

DEFAULT_STEPS = 4
DEFAULT_SEED = None

# FLUX.2-klein is a DISTILLED model: classifier-free guidance is disabled, so
# guidance_scale and negative_prompt have NO effect (see bench.md). The v1 API
# and CLI no longer accept them at all. DEFAULT_GUIDANCE_SCALE survives only
# because the legacy /generate contract (app/schemas.py: GenerateRequest) is
# frozen byte-for-byte and still echoes a guidance_scale value in its response.
DEFAULT_GUIDANCE_SCALE = 3.5

# Latent geometry: FLUX2 requires dimensions that are multiples of 16.
DIMENSION_MULTIPLE = 16
MIN_DIMENSION = 256
MAX_DIMENSION = int(os.environ.get("LOCALIMAGEGEN_MAX_DIMENSION", 1024))

# --- runtime -----------------------------------------------------------------

HOST = os.environ.get("LOCALIMAGEGEN_HOST", "127.0.0.1")
PORT = int(os.environ.get("LOCALIMAGEGEN_PORT", 8765))
BASE_URL = os.environ.get("LOCALIMAGEGEN_URL", f"http://{HOST}:{PORT}")
AUTOSPAWN = os.environ.get("LOCALIMAGEGEN_AUTOSPAWN", "1") not in ("0", "false", "False")

# Reload after FP8 costs ~24s (bench.md), so idle eviction can be fairly eager.
IDLE_EVICT_S = float(os.environ.get("LOCALIMAGEGEN_IDLE_EVICT_S", 600))
MIN_RESIDENCY_S = float(os.environ.get("LOCALIMAGEGEN_MIN_RESIDENCY_S", 60))

# Ollama runs CPU-only by default so it never contends for the 8GB card.
# NOTE: the old `OLLAMA_GPU=0` env trick was a no-op -- Ollama is a separate
# server process and that is not a real Ollama variable. The working mechanism
# is options={"num_gpu": 0} on the chat call.
# LLM provider for prompt enhancement and SVG authoring.
#   lmstudio -> OpenAI-compatible server, GPU-accelerated, needs VRAM arbitration
#   ollama   -> pinned to CPU (num_gpu=0), never contends for VRAM
LLM_PROVIDER = os.environ.get("LOCALIMAGEGEN_LLM_PROVIDER", "lmstudio").lower()
LMSTUDIO_URL = os.environ.get("LOCALIMAGEGEN_LMSTUDIO_URL", "http://127.0.0.1:1234")
# qwen2.5-3b-instruct: best all-round sub-5B model for prompt rewriting across
# image/SVG/video tasks -- strong instruction-following, good at structured
# descriptive rewrites, small enough to leave VRAM for the image/video models.
_DEFAULT_LLM_MODEL = {"lmstudio": "google/gemma-4-e4b", "ollama": "qwen2.5:3b-instruct"}
LLM_MODEL = (
    os.environ.get("LOCALIMAGEGEN_LLM_MODEL")
    or _DEFAULT_LLM_MODEL.get(LLM_PROVIDER, _DEFAULT_OLLAMA_MODEL)
)

OLLAMA_GPU = os.environ.get("LOCALIMAGEGEN_OLLAMA_GPU", "0") not in ("0", "false", "False")
OLLAMA_NUM_GPU = -1 if OLLAMA_GPU else 0
OLLAMA_KEEP_ALIVE = os.environ.get("LOCALIMAGEGEN_OLLAMA_KEEP_ALIVE", "0")

# --- presets -----------------------------------------------------------------

PLATFORMS = {
    "default": {"width": 1024, "height": 1024, "aspect": "1:1"},
    "youtube": {"width": 1024, "height": 576, "aspect": "16:9"},
    "youtube-shorts": {"width": 576, "height": 1024, "aspect": "9:16"},
    "reels": {"width": 576, "height": 1024, "aspect": "9:16"},
    "instagram": {"width": 1024, "height": 1024, "aspect": "1:1"},
    "instagram-story": {"width": 576, "height": 1024, "aspect": "9:16"},
    "instagram-portrait": {"width": 896, "height": 1120, "aspect": "4:5"},
    "twitter": {"width": 1024, "height": 576, "aspect": "16:9"},
    "facebook": {"width": 1024, "height": 1024, "aspect": "1:1"},
    "linkedin": {"width": 1024, "height": 544, "aspect": "1.88:1"},
    "whatsapp": {"width": 576, "height": 1024, "aspect": "9:16"},
}

VIDEO_PRESETS = {
    # Wan requires dimensions divisible by 16 and (num_frames - 1) % 4 == 0.
    # 480p is the resolution the 1.3B model was trained for.
    "short-480p":  {"width": 832, "height": 480, "num_frames": 33, "steps": 20, "fps": 16},
    "tiny-480p":   {"width": 832, "height": 480, "num_frames": 17, "steps": 15, "fps": 16},
    "long-480p":   {"width": 832, "height": 480, "num_frames": 49, "steps": 25, "fps": 16},
    "square-480p": {"width": 480, "height": 480, "num_frames": 33, "steps": 20, "fps": 16},
    "portrait":    {"width": 480, "height": 832, "num_frames": 33, "steps": 20, "fps": 16},
}

SVG_PRESETS = {
    "icon": {"size": 512, "hint": "single centred glyph, 2-4 colours, bold simple geometry"},
    "logo": {"size": 512, "hint": "wordmark or monogram, flat, high contrast, memorable silhouette"},
    "diagram": {"size": 1024, "hint": "boxes, arrows and labels, generous spacing, readable at a glance"},
    "chart": {"size": 1024, "hint": "axes, gridlines, plotted series, legible tick labels"},
    "illustration": {"size": 1024, "hint": "layered flat-vector scene, cohesive limited palette"},
}

# --- output ------------------------------------------------------------------

DEFAULT_OUTPUT_DIR = Path.home() / "LocalImageGen"
OUTPUT_ROOT = Path(os.environ.get("LOCALIMAGEGEN_ROOT", DEFAULT_OUTPUT_DIR))
# Back-compat: LOCALIMAGEGEN_DIR historically pointed at the images dir itself.
OUTPUT_DIR = Path(os.environ.get("LOCALIMAGEGEN_DIR", OUTPUT_ROOT / "images"))
LOG_DIR = OUTPUT_ROOT / "logs"
JOBS_DB = OUTPUT_ROOT / "jobs.db"

KIND_DIRS = {
    "image": OUTPUT_DIR,
    "edit": OUTPUT_DIR,
    "svg": OUTPUT_ROOT / "svg",
    "video": OUTPUT_ROOT / "video",
}

# --- helpers -----------------------------------------------------------------


def resolve_platform(name):
    return PLATFORMS.get(name, PLATFORMS["default"])


def snap_dimension(value: int) -> int:
    """Clamp into range and snap to the latent grid FLUX2 requires."""
    value = max(MIN_DIMENSION, min(int(value), MAX_DIMENSION))
    return max(DIMENSION_MULTIPLE, round(value / DIMENSION_MULTIPLE) * DIMENSION_MULTIPLE)


def resolve_dimensions(platform, width, height, warnings: list[str] | None = None):
    """Resolve final (width, height), snapped to a multiple of 16.

    Explicit width+height override the platform preset, matching prior behaviour.
    """
    if width and height:
        requested = (int(width), int(height))
    else:
        preset = resolve_platform(platform)
        requested = (preset["width"], preset["height"])

    snapped = (snap_dimension(requested[0]), snap_dimension(requested[1]))
    if warnings is not None and snapped != requested:
        warnings.append(
            f"dimensions adjusted {requested[0]}x{requested[1]} -> {snapped[0]}x{snapped[1]} "
            f"(must be a multiple of {DIMENSION_MULTIPLE}, max {MAX_DIMENSION})"
        )
    return snapped
