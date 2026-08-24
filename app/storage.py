"""Output path resolution, shared by every backend."""

from __future__ import annotations

import os
import time
import uuid
from pathlib import Path

from app.config import KIND_DIRS, OUTPUT_DIR

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
SVG_EXTENSIONS = {".svg"}
VIDEO_EXTENSIONS = {".mp4", ".webm", ".gif"}
ALL_EXTENSIONS = IMAGE_EXTENSIONS | SVG_EXTENSIONS | VIDEO_EXTENSIONS

DEFAULT_EXTENSION = {"image": ".png", "edit": ".png", "svg": ".svg", "video": ".mp4"}


def unique_name(kind: str = "image", extension: str | None = None) -> str:
    """Collision-proof filename.

    The previous scheme was `int(time.time() * 1000)`, which DOES collide for
    concurrent requests despite its docstring claiming otherwise. A uuid4 suffix
    removes the race while keeping the name sortable by time.
    """
    ext = extension or DEFAULT_EXTENSION.get(kind, ".png")
    stamp = time.strftime("%Y%m%d-%H%M%S")
    return f"{kind}-{stamp}-{uuid.uuid4().hex[:6]}{ext}"


def unique_image_path(target_dir, extension=".png"):
    """Back-compat shim for the pre-refactor helper."""
    return Path(target_dir) / unique_name("image", extension)


def kind_dir(kind: str = "image") -> Path:
    d = KIND_DIRS.get(kind, OUTPUT_DIR)
    d.mkdir(parents=True, exist_ok=True)
    return d


def resolve_output_path(path=None, kind: str = "image", extension: str | None = None) -> Path:
    """Resolve where a generated artifact should be saved.

    - `path` is None                     -> <kind dir>/<generated name>
    - `path` ends with a known extension -> used as-is (parent dirs created)
    - anything else                      -> treated as a directory
    """
    ext = extension or DEFAULT_EXTENSION.get(kind, ".png")

    if path is None:
        return kind_dir(kind) / unique_name(kind, ext)

    target = Path(path)
    if target.suffix.lower() in ALL_EXTENSIONS:
        target.parent.mkdir(parents=True, exist_ok=True)
        return target

    target.mkdir(parents=True, exist_ok=True)
    return target / unique_name(kind, ext)


def sibling_path(path: Path, extension: str) -> Path:
    """Companion file next to an artifact (e.g. an SVG's PNG preview)."""
    return Path(path).with_suffix(extension)


# Any path separator or parent-ref makes a filename unsafe. Built from os.sep
# rather than a regex literal so no backslash escaping is involved.
_SEPARATORS = tuple(c for c in ("/", os.sep, os.altsep) if c)


def _is_unsafe(filename: str) -> bool:
    return ".." in filename or any(sep in filename for sep in _SEPARATORS)


def safe_join(root: Path, filename: str) -> Path:
    """Resolve `filename` under `root`, refusing traversal.

    Guards `GET /images/{filename}`, which previously had no check at all and
    would happily serve ../../../Windows/win.ini.
    """
    if _is_unsafe(filename):
        raise ValueError("invalid filename")
    root = Path(root).resolve()
    candidate = (root / filename).resolve()
    if candidate != root and root not in candidate.parents:
        raise ValueError("path escapes root")
    return candidate
