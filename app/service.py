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
    text_overlay: str | None = None,
    text_position: str = "bottom",
    text_color: str = "#FFFFFF",
    text_bg_color: str = "#00000080",
    text_font_size: int = 48,
    text_padding: int = 20,
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
            "text_overlay": text_overlay,
            "text_position": text_position,
            "text_color": text_color,
            "text_bg_color": text_bg_color,
            "text_font_size": text_font_size,
            "text_padding": text_padding,
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


def generate_video(
    prompt: str,
    preset: str = "short-480p",
    width: int | None = None,
    height: int | None = None,
    num_frames: int | None = None,
    steps: int | None = None,
    guidance_scale: float | None = None,
    fps: int | None = None,
    seed: int | None = None,
    negative_prompt: str | None = None,
    path: str | None = None,
    codec: str | None = None,
    enhance: bool = False,
    style: str | None = None,
    progress=None,
    is_cancelled=None,
) -> Artifact:
    from app.config import VIDEO_PRESETS

    warnings: list[str] = []
    cfg = VIDEO_PRESETS.get(preset) or VIDEO_PRESETS["short-480p"]
    if preset not in VIDEO_PRESETS:
        warnings.append(f"unknown preset {preset!r}, using short-480p")

    prompt_used = prompt
    if enhance:
        from app.backends.enhance import enhance_prompt

        try:
            result = enhance_prompt(prompt, style)
            prompt_used = result["prompt"]
            warnings.append(f"prompt enhanced with {result['model']}")
        except Exception as exc:  # noqa: BLE001
            warnings.append(f"prompt enhancement failed, using original: {exc}")

    out_path = resolve_output_path(path, "video")
    return get_manager().run(
        "video.wan21-t2v-1.3b",
        {
            "prompt": prompt_used,
            "negative_prompt": negative_prompt,
            "width": width or cfg["width"],
            "height": height or cfg["height"],
            "num_frames": num_frames or cfg["num_frames"],
            "steps": steps or cfg["steps"],
            "fps": fps or cfg["fps"],
            "guidance_scale": guidance_scale if guidance_scale is not None else 5.0,
            "seed": seed,
            "codec": codec,
            "out_path": out_path,
            "warnings": warnings,
            "is_cancelled": is_cancelled,
        },
        progress,
    )


def job_handlers() -> dict[str, Any]:
    """Kind -> callable used by the job runner.

    Each takes (params, progress, is_cancelled) and returns an Artifact.
    """

    def _video(params, progress, is_cancelled):
        return generate_video(progress=progress, is_cancelled=is_cancelled, **params)

    def _image(params, progress, is_cancelled):
        return generate_image(progress=progress, **params)

    def _svg(params, progress, is_cancelled):
        return generate_svg(progress=progress, **params)

    return {"video": _video, "image": _image, "svg": _svg}
