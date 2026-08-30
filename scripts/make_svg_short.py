"""Build a YouTube Short by rendering hand-authored SVG frames -- no diffusion.

Why this exists: a diffusion video model has a weak prior for abstract
motion-graphics (blueprint grids, arrows, dials), so it improvises and the
result reads as neon soup. A feedback loop is a *diagram*, and diagrams are
exactly what vector graphics are for. Every frame here is deterministic,
1080x1920, dead sharp, and -- crucially -- carries readable text, which a
diffusion model cannot.

Pipeline:  python builds one SVG string per frame
        -> resvg rasterises it to PNG (the renderer app/svgtool.py validates with)
        -> ffmpeg muxes the PNG stream into H.264

Usage:
    python scripts/make_svg_short.py                    # full Short
    python scripts/make_svg_short.py --stills           # 1 PNG per beat, no video
    python scripts/make_svg_short.py --fps 60 --out x.mp4

Needs only resvg-py + ffmpeg, both already in requirements.txt. No GPU.
"""

from __future__ import annotations

import argparse
import math
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# --- canvas ------------------------------------------------------------------
W, H = 1080, 1920
# YouTube overlays its own chrome top and bottom; keep anything that must be
# read inside this band.
SAFE_TOP, SAFE_BOTTOM = 300, 1560

# --- palette -----------------------------------------------------------------
BG = "#070B18"
GRID = "#111E36"
CYAN = "#22D3EE"
AMBER = "#F59E0B"
ROSE = "#F43F5E"
INK = "#E8EFF9"
MUTED = "#7A8BA6"
DIM = "#2A3852"

FONT = "Segoe UI, Inter, DejaVu Sans, sans-serif"


# --- easing ------------------------------------------------------------------
def clamp01(x: float) -> float:
    return 0.0 if x < 0 else 1.0 if x > 1 else x


def seg(t: float, a: float, b: float) -> float:
    """Progress of `t` through the window [a, b], clamped to 0..1."""
    return clamp01((t - a) / (b - a)) if b > a else 1.0


def ease_out(t: float) -> float:
    return 1 - (1 - clamp01(t)) ** 3


def ease_in_out(t: float) -> float:
    t = clamp01(t)
    return 4 * t**3 if t < 0.5 else 1 - ((-2 * t + 2) ** 3) / 2


# --- svg primitives ----------------------------------------------------------
def esc(s: str) -> str:
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def text(x, y, s, size=52, fill=INK, weight=600, anchor="middle",
         opacity=1.0, spacing=0.0) -> str:
    if opacity <= 0.001:
        return ""
    return (
        f'<text x="{x:.1f}" y="{y:.1f}" font-family="{FONT}" font-size="{size}" '
        f'font-weight="{weight}" fill="{fill}" text-anchor="{anchor}" '
        f'letter-spacing="{spacing}" opacity="{opacity:.3f}">{esc(s)}</text>'
    )


def dash(length: float, frac: float) -> str:
    """Draw-on attributes for a stroke of known `length`."""
    return (
        f'stroke-dasharray="{length:.1f}" '
        f'stroke-dashoffset="{length * (1 - clamp01(frac)):.1f}"'
    )


def node(cx, cy, size=104, fill=DIM, stroke=CYAN, sw=3, opacity=1.0, r=22) -> str:
    if opacity <= 0.001:
        return ""
    return (
        f'<rect x="{cx - size/2:.1f}" y="{cy - size/2:.1f}" width="{size}" height="{size}" '
        f'rx="{r}" fill="{fill}" stroke="{stroke}" stroke-width="{sw}" opacity="{opacity:.3f}"/>'
    )


def background() -> str:
    """Blueprint grid + vignette. Static, so it never competes with the motion."""
    lines = []
    for x in range(0, W + 1, 90):
        lines.append(
            f'<line x1="{x}" y1="0" x2="{x}" y2="{H}" stroke="{GRID}" stroke-width="1.5"/>'
        )
    for y in range(0, H + 1, 90):
        lines.append(
            f'<line x1="0" y1="{y}" x2="{W}" y2="{y}" stroke="{GRID}" stroke-width="1.5"/>'
        )
    return (
        f'<rect width="{W}" height="{H}" fill="{BG}"/>'
        f'<g opacity="0.55">{"".join(lines)}</g>'
        f'<rect width="{W}" height="{H}" fill="url(#vig)"/>'
    )


def chip(label: str, opacity: float) -> str:
    """The small step badge above the stage, e.g. '01 / MEASURE'."""
    if opacity <= 0.001:
        return ""
    w = 40 + len(label) * 19
    x = (W - w) / 2
    return (
        f'<g opacity="{opacity:.3f}">'
        f'<rect x="{x:.1f}" y="{SAFE_TOP - 12}" width="{w}" height="58" rx="29" '
        f'fill="none" stroke="{AMBER}" stroke-width="2.5" opacity="0.75"/>'
        + text(W / 2, SAFE_TOP + 30, label, size=27, fill=AMBER, weight=700, spacing=3.4)
        + "</g>"
    )


def caption(lines, t: float, start=0.25, stagger=0.16) -> str:
    """Bottom text block. Each line fades and rises independently."""
    out = []
    y = 1320
    for i, (s, size, fill) in enumerate(lines):
        p = ease_out(seg(t, start + i * stagger, start + i * stagger + 0.55))
        out.append(
            text(W / 2, y + (1 - p) * 34, s, size=size, fill=fill,
                 weight=700 if size >= 62 else 500, opacity=p)
        )
        y += size + 34
    return "".join(out)


# --- the stage: a shared pipeline the beats mutate ---------------------------
LINE_Y = 830
LINE_X0, LINE_X1 = 130, 950
NODE_XS = [190, 380, 570, 760, 940]


def pipeline(progress: float, grey_from: float = 1.0, opacity=1.0) -> str:
    """The open production line: five nodes on a rail, drawn on by `progress`."""
    length = LINE_X1 - LINE_X0
    parts = [
        f'<line x1="{LINE_X0}" y1="{LINE_Y}" x2="{LINE_X1}" y2="{LINE_Y}" '
        f'stroke="{DIM}" stroke-width="6" stroke-linecap="round" '
        f'{dash(length, progress)} opacity="{opacity:.3f}"/>'
    ]
    for i, x in enumerate(NODE_XS):
        appear = clamp01((progress * len(NODE_XS)) - i)
        greyed = x / W > grey_from
        parts.append(
            node(x, LINE_Y, size=96,
                 stroke=MUTED if greyed else CYAN,
                 opacity=ease_out(appear) * opacity)
        )
    return "".join(parts)


# --- beats -------------------------------------------------------------------
def beat_hook(t: float, d: float) -> str:
    p = ease_out(seg(t, 0.15, 1.1))
    # An arrow launches, then leaves frame entirely -- the visual thesis.
    x = -160 + ease_in_out(seg(t, 0.9, 2.6)) * (W + 400)
    trail = "".join(
        f'<line x1="{x - 120 - k*90:.0f}" y1="{LINE_Y}" x2="{x - 40 - k*90:.0f}" y2="{LINE_Y}" '
        f'stroke="{CYAN}" stroke-width="{10 - k*2}" stroke-linecap="round" '
        f'opacity="{0.5 - k*0.14:.2f}"/>'
        for k in range(3)
    )
    return (
        text(W / 2, 560, "YOUR SYSTEM", size=64, fill=MUTED, weight=700,
             opacity=p, spacing=8)
        + text(W / 2, 668, "RUNS BLIND.", size=104, fill=INK, weight=800,
               opacity=ease_out(seg(t, 0.45, 1.4)))
        + f'<g opacity="{ease_out(seg(t, 0.8, 1.3)):.3f}">{trail}'
        + f'<circle cx="{x:.0f}" cy="{LINE_Y}" r="20" fill="{CYAN}"/></g>'
        + caption([("It ships output.", 54, MUTED),
                   ("It never learns from it.", 54, INK)], t, start=2.0)
    )


def beat_problem(t: float, d: float) -> str:
    draw = ease_out(seg(t, 0.1, 1.3))
    # The part travels the line and exits; everything it passes goes grey.
    travel = ease_in_out(seg(t, 1.2, 3.6))
    x = LINE_X0 + travel * (W + 200 - LINE_X0)
    fade = seg(t, 3.4, 4.4)
    return (
        chip("OPEN LOOP", ease_out(seg(t, 0.0, 0.6)))
        + pipeline(draw, grey_from=clamp01(travel * 1.15), opacity=1 - fade * 0.45)
        + f'<circle cx="{x:.0f}" cy="{LINE_Y}" r="18" fill="{AMBER}" '
          f'opacity="{ease_out(seg(t, 1.1, 1.4)) * (1 - seg(t, 3.2, 3.9)):.3f}"/>'
        + text(W / 2, 1080, "no return path", size=40, fill=ROSE, weight=600,
               opacity=ease_out(seg(t, 3.5, 4.2)) * 0.9, spacing=4)
        + caption([("Output leaves.", 58, INK),
                   ("Nothing comes back.", 58, ROSE)], t, start=3.6)
    )


def beat_measure(t: float, d: float) -> str:
    sx, sy = 760, 600
    # One ripple per pass -- the rhythm sells "instrumented", not decorative.
    rings = []
    for k in range(3):
        rp = ((t - 1.0 - k * 0.55) % 1.9) / 1.9
        if t > 1.0 + k * 0.55 and 0 < rp < 1:
            rings.append(
                f'<circle cx="{sx}" cy="{sy}" r="{28 + rp*150:.0f}" fill="none" '
                f'stroke="{AMBER}" stroke-width="3" opacity="{(1-rp)*0.65:.3f}"/>'
            )
    pulse = 1 + 0.14 * math.sin(t * 5.2)
    return (
        chip("01 / MEASURE", ease_out(seg(t, 0.0, 0.5)))
        + pipeline(1.0)
        + f'<g opacity="{ease_out(seg(t, 0.5, 1.1)):.3f}">'
        + f'<line x1="{sx}" y1="{sy+30}" x2="{sx}" y2="{LINE_Y-50}" '
          f'stroke="{AMBER}" stroke-width="4" stroke-dasharray="10 10"/>'
        + "".join(rings)
        + f'<circle cx="{sx}" cy="{sy}" r="{26*pulse:.1f}" fill="{AMBER}"/>'
        + text(sx, sy - 68, "SENSOR", size=26, fill=AMBER, weight=700, spacing=4)
        + "</g>"
        + caption([("Instrument the output.", 58, INK),
                   ("What you can't see,", 46, MUTED),
                   ("you can't correct.", 46, MUTED)], t, start=1.4)
    )


def beat_feedback(t: float, d: float) -> str:
    sx, sy = 760, 600
    dx, dy = 190, 600
    # The return arc is the whole point of the beat, so it draws on slowly.
    arc_len = 1250.0
    draw = ease_in_out(seg(t, 0.7, 2.9))
    arc = f"M {sx} {sy} C {sx} 330, {dx} 330, {dx} {dy}"
    # The needle only starts ticking once the signal has actually arrived.
    tick = math.sin(max(0.0, t - 2.9) * 4.0) * 34 if t > 2.9 else 0.0
    ang = math.radians(-90 + tick)
    return (
        chip("02 / FEED BACK", ease_out(seg(t, 0.0, 0.5)))
        + pipeline(1.0)
        + f'<circle cx="{sx}" cy="{sy}" r="26" fill="{AMBER}"/>'
        + f'<path d="{arc}" fill="none" stroke="{CYAN}" stroke-width="6" '
          f'stroke-linecap="round" {dash(arc_len, draw)}/>'
        + f'<g opacity="{ease_out(seg(t, 2.7, 3.2)):.3f}">'
        + f'<circle cx="{dx}" cy="{dy}" r="52" fill="{BG}" stroke="{CYAN}" stroke-width="5"/>'
        + f'<line x1="{dx}" y1="{dy}" x2="{dx+math.cos(ang)*38:.1f}" '
          f'y2="{dy+math.sin(ang)*38:.1f}" stroke="{ROSE}" stroke-width="6" '
          f'stroke-linecap="round"/>'
        + text(dx, dy + 100, "INPUT", size=26, fill=CYAN, weight=700, spacing=4)
        + "</g>"
        + caption([("Route the signal", 58, INK),
                   ("back to the input.", 58, CYAN)], t, start=3.1)
    )


def beat_tighten(t: float, d: float) -> str:
    cx, cy = W / 2, 800
    # Radius and error collapse together: the loop tightening IS the point.
    k = ease_in_out(seg(t, 0.6, 3.8))
    r = 300 - k * 118
    circ = 2 * math.pi * r
    spin = t * 68
    err = 96 - k * 88
    cycle = 42 - k * 39
    bars = "".join(
        f'<rect x="{cx - 150 + i*76:.0f}" y="{1090 - (err * (1 - i*0.12)):.0f}" width="34" '
        f'rx="8" height="{max(6, err * (1 - i*0.12)):.0f}" fill="{ROSE}" opacity="0.8"/>'
        for i in range(5)
    )
    return (
        chip("03 / TIGHTEN", ease_out(seg(t, 0.0, 0.5)))
        + f'<g opacity="{ease_out(seg(t, 0.2, 0.9)):.3f}">'
        + f'<circle cx="{cx}" cy="{cy}" r="{r:.1f}" fill="none" stroke="{DIM}" stroke-width="8"/>'
        + f'<circle cx="{cx}" cy="{cy}" r="{r:.1f}" fill="none" stroke="{CYAN}" stroke-width="8" '
          f'stroke-linecap="round" stroke-dasharray="{circ*0.3:.0f} {circ:.0f}" '
          f'transform="rotate({spin:.1f} {cx} {cy})"/>'
        + text(cx, cy + 16, f"{cycle:.0f}", size=110, fill=INK, weight=800)
        + text(cx, cy + 74, "MIN CYCLE", size=26, fill=MUTED, weight=700, spacing=4)
        + "</g>"
        + f'<g opacity="{ease_out(seg(t, 1.0, 1.6)):.3f}">{bars}</g>'
        + caption([("Shorten the delay.", 58, INK),
                   ("A slow loop is an open one.", 46, MUTED)], t, start=3.4)
    )


def beat_payoff(t: float, d: float) -> str:
    cx, cy = W / 2, 790
    R = 270
    a = ease_out(seg(t, 0.1, 1.0))
    glow = 0.55 + 0.45 * math.sin(t * 2.1)
    inner = "".join(
        f'<circle cx="{cx + math.cos(math.radians(t*46 + i*120))*118:.1f}" '
        f'cy="{cy + math.sin(math.radians(t*46 + i*120))*118:.1f}" r="46" fill="none" '
        f'stroke="{AMBER}" stroke-width="4" opacity="{0.75*a:.3f}"/>'
        for i in range(3)
    )
    return (
        f'<circle cx="{cx}" cy="{cy}" r="{R}" fill="none" stroke="{CYAN}" stroke-width="12" '
        f'stroke-linecap="round" opacity="{a:.3f}"/>'
        + f'<circle cx="{cx}" cy="{cy}" r="{R}" fill="none" stroke="{CYAN}" stroke-width="26" '
          f'opacity="{0.16*glow*a:.3f}"/>'
        + inner
        + caption([("The loop isn't a feature.", 52, MUTED),
                   ("It's the system.", 76, CYAN)], t, start=1.6)
    )


def beat_cta(t: float, d: float) -> str:
    cx, cy = W / 2, 800
    breathe = 1 + 0.05 * math.sin(t * 1.9)
    a = ease_out(seg(t, 0.1, 0.8))
    return (
        f'<circle cx="{cx}" cy="{cy}" r="{120*breathe:.1f}" fill="none" stroke="{CYAN}" '
        f'stroke-width="9" opacity="{a:.3f}"/>'
        + f'<circle cx="{cx}" cy="{cy}" r="{54*breathe:.1f}" fill="none" stroke="{AMBER}" '
          f'stroke-width="6" opacity="{a*0.85:.3f}"/>'
        + caption([("Close one loop", 66, INK),
                   ("this week.", 66, CYAN)], t, start=0.6)
    )


# (label, render fn, seconds)
BEATS = [
    ("hook",     beat_hook,     3.8),
    ("problem",  beat_problem,  5.4),
    ("measure",  beat_measure,  5.6),
    ("feedback", beat_feedback, 6.0),
    ("tighten",  beat_tighten,  5.8),
    ("payoff",   beat_payoff,   5.2),
    ("cta",      beat_cta,      3.6),
]

DEFS = (
    '<defs><radialGradient id="vig" cx="50%" cy="42%" r="78%">'
    f'<stop offset="55%" stop-color="{BG}" stop-opacity="0"/>'
    '<stop offset="100%" stop-color="#000000" stop-opacity="0.72"/>'
    "</radialGradient></defs>"
)


def frame_svg(beat_idx: int, t: float) -> str:
    """One complete frame. Beats fade at their own edges -- no crossfade
    bookkeeping, and it reads as a deliberate cut rather than a dissolve."""
    label, fn, dur = BEATS[beat_idx]
    alpha = min(seg(t, 0.0, 0.28), 1 - seg(t, dur - 0.3, dur))
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" '
        f'viewBox="0 0 {W} {H}">{DEFS}{background()}'
        f'<g opacity="{clamp01(alpha):.3f}">{fn(t, dur)}</g></svg>'
    )


def render_png(svg: str) -> bytes:
    import resvg_py

    out = resvg_py.svg_to_bytes(svg_string=svg, width=W, height=H)
    return bytes(out) if isinstance(out, list) else out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--stills", action="store_true",
                    help="write one representative PNG per beat and exit")
    ap.add_argument("--out", default=str(
        Path.home() / "LocalImageGen" / "video" / "loop_svg_short.mp4"))
    args = ap.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if args.stills:
        for i, (label, _, dur) in enumerate(BEATS):
            p = out_path.parent / f"svg_beat_{i:02d}_{label}.png"
            p.write_bytes(render_png(frame_svg(i, dur * 0.72)))
            print(f"  {p}")
        return

    total = sum(d for _, _, d in BEATS)
    n_total = int(total * args.fps)
    print(f"Rendering {total:.1f}s @ {args.fps}fps -> {n_total} frames at {W}x{H}")

    proc = subprocess.Popen(
        ["ffmpeg", "-y", "-f", "image2pipe", "-framerate", str(args.fps), "-i", "-",
         "-c:v", "libx264", "-preset", "slow", "-crf", "17",
         "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(out_path)],
        stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
    )
    n = 0
    try:
        for i, (label, _, dur) in enumerate(BEATS):
            for f in range(int(dur * args.fps)):
                proc.stdin.write(render_png(frame_svg(i, f / args.fps)))
                n += 1
                if n % 30 == 0:
                    print(f"  [{n/n_total*100:5.1f}%] {label:<9s}", end="\r", flush=True)
        proc.stdin.close()
    except BrokenPipeError:
        pass
    if proc.wait() != 0:
        raise RuntimeError(f"ffmpeg failed:\n{proc.stderr.read().decode()[-800:]}")
    print(f"\nWrote {out_path}  ({n} frames, {n/args.fps:.1f}s)")


if __name__ == "__main__":
    main()
