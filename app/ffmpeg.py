"""Frame sequence -> mp4, plus a contact sheet.

Encoding uses the ffmpeg CLI already on PATH (9.0 full build) over stdin, so
there are no intermediate PNGs and no extra Python dependency -- `imageio-ffmpeg`
would ship a second ffmpeg binary for no benefit.

libx264, not NVENC, deliberately: we have just freed the GPU after a long
diffusion run, a 2-second 480p clip encodes in well under a second on 20 CPU
threads, and x264 gives better quality per byte. NVENC stays available behind a
flag for anyone encoding something long.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import numpy as np


class FfmpegError(RuntimeError):
    pass


def ffmpeg_binary() -> str:
    found = shutil.which("ffmpeg")
    if not found:
        raise FfmpegError(
            "ffmpeg not found on PATH. Install it (winget install Gyan.FFmpeg) "
            "or set it on PATH; it is required to encode video."
        )
    return found


def _to_uint8(frames: np.ndarray) -> np.ndarray:
    """diffusers returns float32 in [0, 1] when output_type='np'."""
    if frames.dtype == np.uint8:
        return frames
    return (np.clip(frames, 0.0, 1.0) * 255.0).round().astype(np.uint8)


def encode_video(
    frames: np.ndarray,
    out_path: Path,
    fps: int = 16,
    codec: str = "libx264",
    crf: int = 18,
) -> Path:
    """Write frames (F, H, W, 3) to an mp4."""
    frames = _to_uint8(np.asarray(frames))
    if frames.ndim != 4 or frames.shape[-1] != 3:
        raise FfmpegError(f"expected frames shaped (F, H, W, 3), got {frames.shape}")

    count, height, width, _ = frames.shape
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if codec == "h264_nvenc":
        codec_args = ["-c:v", "h264_nvenc", "-preset", "p5", "-cq", str(crf)]
    else:
        codec_args = ["-c:v", "libx264", "-preset", "slow", "-crf", str(crf)]

    cmd = [
        ffmpeg_binary(), "-y",
        "-f", "rawvideo", "-pix_fmt", "rgb24",
        "-s", f"{width}x{height}", "-r", str(fps),
        "-i", "-",
        *codec_args,
        "-pix_fmt", "yuv420p",
        "-movflags", "+faststart",
        str(out_path),
    ]

    proc = subprocess.run(cmd, input=frames.tobytes(), capture_output=True)
    if proc.returncode != 0 or not out_path.is_file():
        tail = proc.stderr.decode("utf-8", "replace")[-800:]
        raise FfmpegError(f"ffmpeg failed (exit {proc.returncode}):\n{tail}")

    return out_path


def contact_sheet(frames: np.ndarray, out_path: Path, columns: int = 3) -> Path:
    """Tile a few frames into one PNG.

    This is what makes video usable from an agent harness: no harness can watch
    an mp4, but any of them can read a PNG and judge the result.
    """
    from PIL import Image

    frames = _to_uint8(np.asarray(frames))
    count = frames.shape[0]
    picks = sorted({0, count // 2, count - 1}) if count >= 3 else list(range(count))
    columns = min(columns, len(picks))

    tiles = [Image.fromarray(frames[i]) for i in picks]
    w, h = tiles[0].size
    rows = (len(tiles) + columns - 1) // columns

    sheet = Image.new("RGB", (w * columns, h * rows), (16, 16, 16))
    for idx, tile in enumerate(tiles):
        sheet.paste(tile, ((idx % columns) * w, (idx // columns) * h))

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out_path)
    return out_path


def probe(path: Path) -> dict[str, str]:
    """ffprobe summary; empty dict if ffprobe is unavailable."""
    binary = shutil.which("ffprobe")
    if not binary:
        return {}
    proc = subprocess.run(
        [binary, "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=width,height,nb_frames,r_frame_rate,duration",
         "-of", "default=noprint_wrappers=1", str(path)],
        capture_output=True, text=True,
    )
    out: dict[str, str] = {}
    for line in proc.stdout.splitlines():
        if "=" in line:
            k, v = line.split("=", 1)
            out[k.strip()] = v.strip()
    return out
