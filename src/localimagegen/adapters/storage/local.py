"""Local filesystem storage adapter.

Implements :class:`localimagegen.core.ports.StoragePort`. Saves generated
images (and their metadata) to a local directory, optionally organized into
``YYYY/MM/DD`` subdirectories based on the generation timestamp.
"""

from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path

from localimagegen.core.models import GeneratedImage
from localimagegen.services.logging import get_logger


class LocalFileStorage:
    """Persist generated images to the local filesystem.

    Args:
        base_dir: Root directory for saved images.
        organize_by_date: When ``True`` (default) images are stored under
            ``base_dir/YYYY/MM/DD`` using the generation timestamp.
    """

    def __init__(
        self,
        base_dir: Path | str = Path("images"),
        *,
        organize_by_date: bool = True,
    ) -> None:
        self._base_dir = Path(base_dir)
        self._organize_by_date = organize_by_date
        self._logger = get_logger(__name__)

    def save(self, image: GeneratedImage, name: str) -> str:
        """Persist ``image`` and return the path it was saved to.

        A JSON sidecar file containing the generation metadata is written next
        to the image.
        """
        safe_name = self._sanitize_name(name, image.format)
        target_dir = self._target_dir(image.metadata.created_at)
        target_dir.mkdir(parents=True, exist_ok=True)

        image_path = target_dir / safe_name
        image_path.write_bytes(image.data)
        self._write_metadata(image, image_path)

        self._logger.info("image_saved", path=str(image_path))
        return str(image_path)

    def _sanitize_name(self, name: str, fmt: str) -> str:
        """Return a filesystem-safe file name with the correct extension."""
        cleaned = re.sub(r"[^\w\-.]", "-", name.strip()).strip("-")
        cleaned = cleaned or "image"
        extension = f".{fmt.lower()}" if fmt else ".png"
        if not cleaned.lower().endswith(extension):
            cleaned += extension
        return cleaned

    def _target_dir(self, created_at: datetime) -> Path:
        """Return the directory for an image based on its creation time."""
        if not self._organize_by_date:
            return self._base_dir
        return (
            self._base_dir
            / f"{created_at.year:04d}"
            / f"{created_at.month:02d}"
            / f"{created_at.day:02d}"
        )

    def _write_metadata(self, image: GeneratedImage, image_path: Path) -> None:
        """Write a JSON sidecar file with the generation metadata."""
        metadata_path = image_path.with_suffix(".json")
        payload = image.metadata.model_dump(mode="json")
        metadata_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")