import re
import time
from pathlib import Path

from app.config import OUTPUT_DIR


def sanitize_user_id(user_id):
    if not user_id:
        return "default"
    cleaned = re.sub(r"[^A-Za-z0-9._-]", "-", user_id).strip(".-")
    return cleaned or "default"


def resolve_user_dir(user_id=None):
    """User-specific output directory: OUTPUT_DIR/<user_id>/."""
    user_dir = OUTPUT_DIR / sanitize_user_id(user_id)
    user_dir.mkdir(parents=True, exist_ok=True)
    return user_dir


def unique_image_path(user_dir, extension=".png"):
    """Timestamped filename to avoid collisions between concurrent requests."""
    return user_dir / f"{int(time.time() * 1000)}{extension}"