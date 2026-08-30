"""Build a YouTube Short from multiple looping scenes.

Each scene prompt is generated once (Wan2.1-T2V, portrait 480x832), optionally
turned into a seamless boomerang (forward + reversed), then the scene clips are
repeated and concatenated until the Short reaches --min-seconds.

Scenes are written as a beat sheet -- hook, problem, mechanism, payoff, CTA --
rather than six interchangeable abstract loops, so the Short reads as an
explanation instead of a screensaver.

Usage:
    python scripts/make_short.py                       # uses SCENES below
    python scripts/make_short.py --topic "loop engineering" --min-seconds 70
    python scripts/make_short.py --scenes a.mp4 b.mp4 --out final.mp4

Requires ffmpeg on PATH (the same one app/ffmpeg.py uses).
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path

# Running as `python scripts/make_short.py` puts scripts/ on sys.path, not the
# repo root, so `app` would not be importable without this.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import service  # noqa: E402
from app.config import VIDEO_PRESETS  # noqa: E402


# --- shared look --------------------------------------------------------------
# Every scene inherits this so cuts feel like one film, not six stock clips.
STYLE = (
    "dark navy blueprint grid background, neon cyan and amber accents, "
    "flat vector motion-graphics, crisp edges, centred composition, "
    "vertical portrait framing, subject fills the middle third"
)
NEGATIVE = (
    "text, letters, watermark, logo, human faces, hands, photorealistic, "
    "camera shake, motion blur, jitter, warping, extra objects, cluttered"
)

# --- the beat sheet: each entry is one scene of the Short ----------------------
# (label, motion prompt, seconds of screen time it deserves)
# Motion is described as ONE continuous cyclical action so repeats blend.
SCENES: list[tuple[str, str, float]] = [
    # HOOK -- big, fast, one idea. Must land in the first 2 seconds.
    ("hook",
     "a single glowing arrow launches forward, curves hard, and slams back into "
     "its own tail forming a bright ring that pulses once, fast confident motion",
     6.0),

    # PROBLEM -- the open-ended line that never comes back.
    ("problem",
     "a straight line of parts marches off the top of frame and never returns, "
     "the line thins and fades to grey, slow relentless drift, cold empty space "
     "left behind",
     8.0),

    # MECHANISM 1 -- measure.
    ("measure",
     "a sensor node on a pipeline lights up amber as each part passes, emitting a "
     "small ripple of measurement rings, steady rhythmic pulse, one pulse per part",
     9.0),

    # MECHANISM 2 -- feed back.
    ("feedback",
     "the measurement ripple travels backwards along a curved return path and nudges "
     "a control dial at the start of the line, dial ticks, cycle repeats endlessly",
     10.0),

    # MECHANISM 3 -- the loop tightens (shows the concept doing work).
    ("tighten",
     "interlocking gears drive a flywheel that drives the gears again, the whole "
     "assembly speeds up smoothly and settles into a stable steady rotation, "
     "isometric schematic",
     10.0),

    # PAYOFF -- pull back, the loop was the whole system all along.
    ("payoff",
     "camera pulls back to reveal the whole feedback circuit glowing as one closed "
     "ring, nested smaller loops orbiting inside it, calm continuous rotation",
     9.0),

    # CTA -- quiet, low motion, room for an end card / caption.
    ("cta",
     "a minimal loop symbol folding into itself at the centre of empty dark space, "
     "very slow calm breathing motion, generous negative space above and below",
     8.0),
]


def _progress(frac: float, msg: str) -> None:
    print(f"  [{frac * 100:5.1f}%] {msg:<40s}", end="\r", flush=True)


def build_prompt(motion: str, topic: str) -> str:
    head = f"{topic}: " if topic else ""
    return f"{head}seamless looping animation, {motion}, {STYLE}"


def generate_scene(motion: str, idx: int, label: str, preset: str,
                   topic: str, seed: int | None) -> Path:
    """Generate one scene clip. Generated once -- repetition happens at concat."""
    base = Path.home() / "LocalImageGen" / "video" / "scenes"
    base.mkdir(parents=True, exist_ok=True)
    out = base / f"scene_{idx:02d}_{label}.mp4"
    prompt = build_prompt(motion, topic)
    print(f"\n[scene {idx + 1}: {label}] {motion[:60]}...")
    art = service.generate_video(
        prompt=prompt,
        preset=preset,
        path=str(out),
        negative_prompt=NEGATIVE,
        seed=seed,
        progress=_progress,
    )
    print(f"  -> {art.path}  ({art.meta['duration_s']}s)")
    return Path(art.path)


def boomerang(clip: Path) -> Path:
    """Forward + reversed copy: turns any clip into a genuinely seamless loop."""
    out = clip.with_name(clip.stem + "_boom.mp4")
    cmd = [
        "ffmpeg", "-y", "-i", str(clip),
        "-filter_complex",
        "[0:v]split[a][b];[b]reverse,trim=start_frame=1[r];[a][r]concat=n=2:v=1[v]",
        "-map", "[v]", "-an", "-pix_fmt", "yuv420p", str(out),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0 or not out.is_file():
        print(f"  ! boomerang failed for {clip.name}, using clip as-is")
        return clip
    return out


def plan_playlist(clips: list[Path], weights: list[float],
                  min_seconds: float) -> list[Path]:
    """Repeat each scene clip until the Short is at least min_seconds long.

    Screen time is distributed by the beat-sheet weights, so the hook stays short
    and the mechanism beats get room to breathe.
    """
    durations = [__probe_duration(c) or 2.0 for c in clips]
    total_weight = sum(weights) or float(len(clips))
    reps = []
    for dur, want in zip(durations, weights):
        share = max(min_seconds * (want / total_weight), dur)
        reps.append(max(1, round(share / dur)))
    # Rounding can land under target -- top up the longest beats until we clear it.
    order = sorted(range(len(clips)), key=lambda i: -weights[i])
    i = 0
    while sum(r * d for r, d in zip(reps, durations)) < min_seconds:
        reps[order[i % len(order)]] += 1
        i += 1

    playlist: list[Path] = []
    for clip, n in zip(clips, reps):
        playlist.extend([clip] * n)
    return playlist


def concat(clips: list[Path], out_path: Path) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as f:
        list_path = Path(f.name)
        for c in clips:
            # concat demuxer needs forward-slash paths, escaped
            f.write("file '" + c.as_posix() + "'\n")
    base = ["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", str(list_path)]
    # Stream copy is fastest, but only valid when every clip shares a codec/size.
    proc = subprocess.run(base + ["-c", "copy", str(out_path)],
                          capture_output=True, text=True)
    if proc.returncode != 0 or not out_path.is_file():
        print("  ! stream copy failed, re-encoding")
        proc = subprocess.run(
            base + ["-c:v", "libx264", "-preset", "medium", "-crf", "20",
                    "-pix_fmt", "yuv420p", str(out_path)],
            capture_output=True, text=True)
    list_path.unlink(missing_ok=True)
    if proc.returncode != 0 or not out_path.is_file():
        raise RuntimeError(f"ffmpeg concat failed:\n{proc.stderr[-800:]}")
    return out_path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenes", nargs="*", help="existing mp4 clips to concat")
    ap.add_argument("--topic", default="", help="prefix injected into each SCENE prompt")
    ap.add_argument("--preset", default="portrait", choices=list(VIDEO_PRESETS))
    ap.add_argument("--seed", type=int, default=None,
                    help="fixed seed keeps the look consistent across scenes")
    ap.add_argument("--no-boomerang", action="store_true",
                    help="skip the forward+reverse pass that makes clips loop")
    ap.add_argument("--min-seconds", type=float, default=60.0)
    ap.add_argument("--out", default=str(Path.home() / "LocalImageGen" / "video" / "short_final.mp4"))
    args = ap.parse_args()

    if args.scenes:
        clips = [Path(s) for s in args.scenes]
        weights = [1.0] * len(clips)
    else:
        clips = []
        weights = []
        for i, (label, motion, want) in enumerate(SCENES):
            clips.append(generate_scene(motion, i, label, args.preset,
                                        args.topic, args.seed))
            weights.append(want)

    if not args.no_boomerang:
        print("\nBuilding seamless loops...")
        clips = [boomerang(c) for c in clips]

    playlist = plan_playlist(clips, weights, args.min_seconds)
    out = concat(playlist, Path(args.out))
    est = sum(__probe_duration(c) for c in playlist)
    print(f"\nFinal Short: {out}  (~{est:.1f}s across {len(playlist)} clips)")


def __probe_duration(path: Path) -> float:
    from app.ffmpeg import probe

    info = probe(path)
    if info.get("duration"):
        try:
            return float(info["duration"])
        except ValueError:
            pass
    if info.get("nb_frames") and info.get("r_frame_rate"):
        num, den = info["r_frame_rate"].split("/")
        fps = float(num) / float(den or 1)
        return float(info["nb_frames"]) / fps
    return 0.0


if __name__ == "__main__":
    main()
