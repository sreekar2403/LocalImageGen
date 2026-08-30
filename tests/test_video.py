"""Video frame math and encoding. CPU only -- no Wan model is loaded."""

from __future__ import annotations

import numpy as np
import pytest

from app.backends.video_wan import snap_frames
from app.config import VIDEO_PRESETS
from app.ffmpeg import contact_sheet, encode_video, probe


@pytest.mark.parametrize("n", [5, 9, 17, 33, 49, 81, 161])
def test_valid_frame_counts_are_unchanged(n):
    assert snap_frames(n) == n


@pytest.mark.parametrize("raw", range(5, 60))
def test_snap_always_satisfies_wan_constraint(raw):
    """Wan's temporal VAE requires (num_frames - 1) % 4 == 0."""
    snapped = snap_frames(raw)
    assert (snapped - 1) % 4 == 0
    assert snapped <= raw
    assert snapped >= 5


def test_snap_clamps_tiny_values():
    assert snap_frames(1) == 5
    assert snap_frames(0) == 5


def test_every_video_preset_is_valid():
    for name, cfg in VIDEO_PRESETS.items():
        assert cfg["width"] % 16 == 0, f"{name}: width must be divisible by 16"
        assert cfg["height"] % 16 == 0, f"{name}: height must be divisible by 16"
        assert (cfg["num_frames"] - 1) % 4 == 0, f"{name}: bad frame count"
        assert cfg["steps"] > 0 and cfg["fps"] > 0


def _frames(count=9, h=64, w=96):
    rng = np.random.default_rng(0)
    return rng.random((count, h, w, 3), dtype=np.float32)


def test_encode_video_produces_playable_mp4(tmp_path):
    out = encode_video(_frames(), tmp_path / "clip.mp4", fps=16)
    assert out.is_file() and out.stat().st_size > 0
    info = probe(out)
    if info:  # ffprobe present
        assert info.get("width") == "96"
        assert info.get("height") == "64"


def test_encode_accepts_uint8_frames(tmp_path):
    frames = (_frames() * 255).astype(np.uint8)
    assert encode_video(frames, tmp_path / "u8.mp4").is_file()


def test_encode_rejects_wrong_shape(tmp_path):
    from app.ffmpeg import FfmpegError

    with pytest.raises(FfmpegError):
        encode_video(np.zeros((4, 8, 8)), tmp_path / "bad.mp4")


def test_contact_sheet_tiles_first_middle_last(tmp_path):
    from PIL import Image

    sheet = contact_sheet(_frames(count=9, h=64, w=96), tmp_path / "sheet.png")
    assert sheet.is_file()
    # 3 picks (first, middle, last) laid out in one row of 96px tiles
    assert Image.open(sheet).size == (96 * 3, 64)


def test_contact_sheet_handles_short_clips(tmp_path):
    assert contact_sheet(_frames(count=2, h=32, w=32), tmp_path / "s.png").is_file()
