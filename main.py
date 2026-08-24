"""LocalGen CLI.

    python main.py serve                       start the GPU worker
    python main.py image "a lighthouse"        generate an image
    python main.py edit out.png "make it night"
    python main.py svg "rocket icon"           LLM-authored SVG
    python main.py svg "fox sticker" --trace   FLUX + vectorize
    python main.py enhance "a cat"
    python main.py status | evict | bench

This file used to be empty ("do not use" per AGENTS.md). It is now the single
entrypoint; the old batch_generate.py / generate_single.py duplicated the
pipeline three ways and are gone.
"""

from __future__ import annotations

import argparse
import json
import sys


def _print_artifact(art) -> None:
    d = art.to_dict() if hasattr(art, "to_dict") else art
    print(f"\n  {d['kind']}: {d['path']}")
    if d.get("preview_path") and d["preview_path"] != d["path"]:
        print(f"  preview: {d['preview_path']}")
    meta = d.get("meta") or {}
    bits = []
    if d.get("width"):
        bits.append(f"{d['width']}x{d['height']}")
    if d.get("quantization"):
        bits.append(d["quantization"])
    if meta.get("peak_vram_mb"):
        bits.append(f"{meta['peak_vram_mb']} MiB peak")
    if meta.get("path_count") is not None:
        bits.append(f"{meta['path_count']} paths")
    bits.append(f"{d.get('elapsed_s', 0)}s")
    print("  " + " · ".join(bits))
    for w in d.get("warnings") or []:
        print(f"  note: {w}")


def cmd_serve(args) -> None:
    from app.worker import main as worker_main

    worker_main()


def cmd_image(args) -> None:
    from app import service

    art = service.generate_image(
        prompt=args.prompt, platform=args.platform, width=args.width,
        height=args.height, steps=args.steps, seed=args.seed, path=args.out,
        enhance=args.enhance, style=args.style,
    )
    _print_artifact(art)


def cmd_edit(args) -> None:
    from app import service

    art = service.generate_image(
        prompt=args.prompt, reference_images=args.images, steps=args.steps,
        seed=args.seed, path=args.out,
    )
    _print_artifact(art)


def cmd_svg(args) -> None:
    from app import service

    art = service.generate_svg(
        prompt=args.prompt, mode="trace" if args.trace else "author",
        svg_kind=args.kind, size=args.size, path=args.out, colors=args.colors,
        seed=args.seed, model=args.model, enhance=args.enhance,
    )
    _print_artifact(art)
    if art.text and len(art.text) < 4000:
        print("\n" + art.text)


def cmd_enhance(args) -> None:
    from app.backends.enhance import enhance_prompt

    print(enhance_prompt(args.prompt, args.style)["prompt"])


def cmd_status(args) -> None:
    from app import service

    print(json.dumps(service.health(), indent=2, default=str))


def cmd_evict(args) -> None:
    from app.manager import get_manager

    print(json.dumps(get_manager().evict(), indent=2, default=str))


def cmd_bench(args) -> None:
    sys.argv = ["bench.py", "--mode", args.mode, "--runs", str(args.runs)]
    import runpy

    runpy.run_path("scripts/bench.py", run_name="__main__")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="localgen", description="Local image + SVG generation")
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("serve", help="start the GPU worker").set_defaults(func=cmd_serve)

    i = sub.add_parser("image", help="generate an image")
    i.add_argument("prompt")
    i.add_argument("--platform", default="default")
    i.add_argument("--width", type=int)
    i.add_argument("--height", type=int)
    i.add_argument("--steps", type=int, default=4)
    i.add_argument("--seed", type=int)
    i.add_argument("--out")
    i.add_argument("--enhance", action="store_true")
    i.add_argument("--style")
    i.set_defaults(func=cmd_image)

    e = sub.add_parser("edit", help="edit images with a text instruction")
    e.add_argument("images", nargs="+")
    e.add_argument("--prompt", required=True)
    e.add_argument("--steps", type=int, default=4)
    e.add_argument("--seed", type=int)
    e.add_argument("--out")
    e.set_defaults(func=cmd_edit)

    s = sub.add_parser("svg", help="generate an SVG")
    s.add_argument("prompt")
    s.add_argument("--trace", action="store_true", help="FLUX raster + vectorize instead of LLM authoring")
    s.add_argument("--kind", default="icon", choices=["icon", "logo", "diagram", "chart", "illustration"])
    s.add_argument("--size", type=int)
    s.add_argument("--colors", type=int, default=12)
    s.add_argument("--seed", type=int)
    s.add_argument("--model")
    s.add_argument("--out")
    s.add_argument("--enhance", action="store_true")
    s.set_defaults(func=cmd_svg)

    n = sub.add_parser("enhance", help="expand a prompt with a local LLM")
    n.add_argument("prompt")
    n.add_argument("--style")
    n.set_defaults(func=cmd_enhance)

    sub.add_parser("status", help="worker + VRAM status").set_defaults(func=cmd_status)
    sub.add_parser("evict", help="unload the resident model").set_defaults(func=cmd_evict)

    b = sub.add_parser("bench", help="measure generation performance")
    b.add_argument("--mode", default="fp8", choices=["fp8", "bf16"])
    b.add_argument("--runs", type=int, default=3)
    b.set_defaults(func=cmd_bench)

    return p


def main() -> None:
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
