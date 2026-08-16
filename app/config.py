import os
from pathlib import Path

from config import vision_model

MODEL_NAME = vision_model
DEFAULT_STEPS = 4
DEFAULT_GUIDANCE_SCALE = 3.5
DEFAULT_SEED = None

MAX_DIMENSION = 1024

# Aspect-ratio aware platform presets. Dimensions are multiples of 16.
# Resolution is capped at 1024 per side to fit 8GB VRAM while preserving
# the platform's native aspect ratio.
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

# Root output directory. Overridable via LOCALIMAGEGEN_DIR env var.
# Defaults to a user-specific location outside the repo so generated
# images survive re-clones and never touch git.
DEFAULT_OUTPUT_DIR = Path.home() / "LocalImageGen" / "images"
OUTPUT_DIR = Path(os.environ.get("LOCALIMAGEGEN_DIR", DEFAULT_OUTPUT_DIR))


def resolve_platform(name):
    return PLATFORMS.get(name, PLATFORMS["default"])


def resolve_dimensions(platform, width, height):
    if width and height:
        return int(width), int(height)
    preset = resolve_platform(platform)
    return preset["width"], preset["height"]
