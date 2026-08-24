"""Modality orchestration shared by the HTTP routes and the CLI.

Both front-ends previously duplicated the resolve -> load -> generate -> save ->
format sequence. It lives here once now.
"""

from __future__ import annotations

from typing import Any

from app.backends.base import Artifact
from app.config import (
    DEFAULT_STEPS,
    MODEL_NAME,
    SVG_PRESETS,
    resolve_dimensions,
)
from app.manager import get_manager
from app.storage import resolve_output_path


def generate_image(
    prompt: str,
    platform: str = "default",
    width: int | None = None,
    height: int | None = None,
    steps: int | None = None,
    seed: int | None = None,
    path: str | None = None,
    negative_prompt: str | None = None,
    guidance_scale: float | None = None,
    reference_images: list[str] | None = None,
    enhance: bool = False,
    style: str | None = None,
    progress=None,
) -> Artifact:
    warnings: list[str] = []
    w, h = resolve_dimensions(platform, width, height, warnings)
    mgr = get_manager()

    prompt_used = prompt
    if enhance:
        from app.backends.enhance import enhance_prompt

        try:
            result = enhance_prompt(prompt, style)
            prompt_used = result["prompt"]
            warnings.append(f"prompt enhanced with {result['model']} ({result['style']})")
        except Exception as exc:  # noqa: BLE001 - enhancement is best-effort
            warnings.append(f"prompt enhancement failed, using original: {exc}")

    kind = "edit" if reference_images else "image"
    out_path = resolve_output_path(path, kind)

    return mgr.run(
        "image.flux2-klein",
        {
            "prompt": prompt_used,
            "width": w,
            "height": h,
            "steps": steps or DEFAULT_STEPS,
            "seed": seed,
            "out_path": out_path,
            "negative_prompt": negative_prompt,
            "guidance_scale": guidance_scale,
            "reference_images": reference_images or [],
            "warnings": warnings,
        },
        progress,
    )


def generate_svg(
    prompt: str,
    mode: str = "author",
    svg_kind: str = "icon",
    size: int | None = None,
    path: str | None = None,
    colors: int = 12,
    seed: int | None = None,
    model: str | None = None,
    max_repairs: int = 2,
    enhance: bool = False,
    style: str | None = None,
    progress=None,
) -> Artifact:
    warnings: list[str] = []
    mgr = get_manager()
    out_path = resolve_output_path(path, "svg")
    resolved_size = int(size or SVG_PRESETS.get(svg_kind, {}).get("size", 512))

    prompt_used = prompt
    if enhance:
        from app.backends.enhance import enhance_prompt

        try:
            result = enhance_prompt(prompt, style)
            prompt_used = result["prompt"]
            warnings.append(f"prompt enhanced with {result['model']}")
        except Exception as exc:  # noqa: BLE001
            warnings.append(f"prompt enhancement failed, using original: {exc}")

    backend = "svg.trace" if mode == "trace" else "svg.author"
    return mgr.run(
        backend,
        {
            "prompt": prompt_used,
            "svg_kind": svg_kind,
            "size": resolved_size,
            "colors": colors,
            "seed": seed,
            "model": model,
            "max_repairs": max_repairs,
            "out_path": out_path,
            "warnings": warnings,
        },
        progress,
    )


def edit_svg(svg: str, instruction: str, path: str | None = None, model: str | None = None) -> Artifact:
    """Edit an existing SVG given a path or literal source."""
    import time
    from pathlib import Path

    from app.backends.llm import chat
    from app.prompts import SVG_PROMPTS
    from app.storage import sibling_path
    from app.svgtool import SvgError, extract, path_count, rasterize, sanitize

    source = svg
    candidate = Path(svg)
    try:
        if candidate.suffix.lower() == ".svg" and candidate.is_file():
            source = candidate.read_text(encoding="utf-8")
    except OSError:
        pass

    start = time.perf_counter()
    warnings: list[str] = []
    raw, _ = chat(
        system=SVG_PROMPTS["icon"],
        user=(
            f"Apply this change: {instruction}\n\n"
            f"Return the COMPLETE modified SVG, preserving everything not "
            f"mentioned.\n\nCurrent SVG:\n{source}"
        ),
        model=model,
        temperature=0.3,
    )

    out_path = resolve_output_path(path, "svg")
    try:
        clean, notes = sanitize(extract(raw))
        png = rasterize(clean)
        valid = True
    except SvgError as exc:
        warnings.append(f"edited SVG failed validation: {exc}")
        clean, notes, png, valid = raw, [], None, False

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(clean, encoding="utf-8")
    preview = None
    if png:
        preview = sibling_path(out_path, ".png")
        preview.write_bytes(png)

    return Artifact(
        path=out_path,
        kind="svg",
        mime="image/svg+xml",
        preview_path=preview,
        text=clean if valid else raw,
        prompt_used=instruction,
        backend="svg.edit",
        elapsed_s=time.perf_counter() - start,
        warnings=warnings,
        meta={"valid": valid, "mode": "edit", "path_count": path_count(clean), "notes": notes},
    )


def health() -> dict[str, Any]:
    mgr = get_manager()
    status = mgr.status()
    image = mgr.get("image.flux2-klein")
    return {
        "status": "ok",
        "model": MODEL_NAME,
        "loaded": image.loaded,
        "quantization": getattr(image, "quantization", "none"),
        "load_error": getattr(image, "load_error", None),
        **status,
    }
