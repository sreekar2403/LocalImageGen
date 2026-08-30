"""Answer one question: can FLUX.2-klein produce a still worth animating?

Every frame judged so far came from Wan2.1-T2V-1.3B, never from the image
model -- `jobs.db` has no image jobs at all. So before spending any more GPU
time on video, establish the image ceiling directly.

Two variables, held against the same fixed seed so the comparison is honest:

  * steps -- config.DEFAULT_STEPS is 4, the speed floor for a distilled model,
    not its quality setting. 10 is the control.
  * subject -- an abstract "neon blueprint feedback loop" (a weak prior these
    models improvise around) against a concrete photographic subject (a strong
    prior). If the concrete one is sharp and the abstract one is mush, the
    problem was never the model.

Usage:
    python scripts/flux_still_test.py
    python scripts/flux_still_test.py --steps 4 10 20 --seed 7
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import service  # noqa: E402

OUT_DIR = Path.home() / "LocalImageGen" / "images" / "still_test"

# (slug, prompt) -- one weak prior, one strong, one in between.
SUBJECTS: list[tuple[str, str]] = [
    ("abstract",
     "dark navy blueprint grid background, a single glowing cyan feedback loop "
     "arrow curving back into itself, amber accent nodes, flat vector "
     "motion-graphics, crisp edges, centred composition"),
    ("concrete",
     "a brass and steel industrial governor mechanism on a dark workbench, "
     "spinning flyweights, shallow depth of field, dramatic side lighting, "
     "macro product photography, sharp focus"),
    ("diagrammatic",
     "isometric technical illustration of a closed-loop control system, sensor, "
     "controller and actuator connected in a ring, muted blueprint palette, "
     "clean line art, white background"),
]


def _progress(frac: float, msg: str) -> None:
    print(f"    [{frac*100:5.1f}%] {msg:<28s}", end="\r", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, nargs="+", default=[4, 10])
    ap.add_argument("--seed", type=int, default=11,
                    help="fixed so step count is the only variable")
    ap.add_argument("--size", type=int, default=1024)
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = []

    for slug, prompt in SUBJECTS:
        for steps in args.steps:
            out = OUT_DIR / f"{slug}_s{steps:02d}.png"
            print(f"\n[{slug} @ {steps} steps]")
            t0 = time.time()
            art = service.generate_image(
                prompt=prompt,
                width=args.size,
                height=args.size,
                steps=steps,
                seed=args.seed,
                path=str(out),
                progress=_progress,
            )
            dt = time.time() - t0
            rows.append((slug, steps, dt, art.meta.get("quantization", "?"), art.path))
            print(f"  -> {art.path}  ({dt:.1f}s)")

    print("\n--- summary -------------------------------------------------")
    print(f"{'subject':<14}{'steps':>6}{'sec':>8}  quant")
    for slug, steps, dt, quant, _ in rows:
        print(f"{slug:<14}{steps:>6}{dt:>8.1f}  {quant}")
    print(f"\nAll images in {OUT_DIR}")
    print("Judge: is ANY of these a frame you'd post as-is? If no, drop the")
    print("diffusion path and build the Short with scripts/make_svg_short.py.")


if __name__ == "__main__":
    main()
