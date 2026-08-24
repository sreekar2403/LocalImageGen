"""SVG generation, two modes.

  author  A local LLM (LM Studio or Ollama) writes SVG source directly, then it
          is sanitized and RENDERED to prove it is valid. Best for icons, logos,
          diagrams and charts.
  trace   FLUX renders a flat-colour raster, Pillow posterizes it, vtracer
          converts it to paths. Best for illustrations and stickers.

Deliberately NOT in the repair loop: feeding the rendered PNG back to the local
model for visual critique. The small local models are not multimodal, and none
is worth the VRAM for it. The tool returns a PNG preview so the *calling* agent
can look at the result and ask for changes -- that is the better critic.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from app.backends.base import Artifact, Progress
from app.backends.llm import chat
from app.config import LLM_MODEL as OLLAMA_MODEL, SVG_PRESETS
from app.prompts import SVG_PROMPTS, SVG_REPAIR_PROMPT, TRACE_STYLE_SUFFIX
from app.storage import sibling_path
from app.svgtool import SvgError, extract, path_count, rasterize, sanitize

MAX_REPAIRS = 2


def _write(out_path: Path, svg: str, png: bytes | None) -> Path | None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(svg, encoding="utf-8")
    if png is None:
        return None
    preview = sibling_path(out_path, ".png")
    preview.write_bytes(png)
    return preview


class SvgAuthorBackend:
    """LLM-authored SVG with a bounded sanitize -> render -> repair loop."""

    name = "svg.author"
    kinds = ("svg",)
    needs_gpu = False
    vram_estimate_mb = 0

    @property
    def loaded(self) -> bool:
        return True

    def load(self) -> None:
        return

    def unload(self) -> None:
        return

    def generate(self, params: dict[str, Any], progress: Progress | None = None) -> Artifact:
        prompt: str = params["prompt"]
        kind: str = params.get("svg_kind") or "icon"
        size: int = int(params.get("size") or SVG_PRESETS.get(kind, {}).get("size", 512))
        model = params.get("model")
        max_repairs = int(params.get("max_repairs", MAX_REPAIRS))
        out_path = Path(params["out_path"])
        warnings: list[str] = list(params.get("warnings", []))

        system = SVG_PROMPTS.get(kind, SVG_PROMPTS["icon"])
        hint = SVG_PRESETS.get(kind, {}).get("hint", "")
        user = f"{prompt}\n\nTarget canvas: {size}x{size}. Style: {hint}"

        start = time.perf_counter()
        notes: list[str] = []
        attempts = 0
        last_error: str | None = None
        raw = ""

        if progress:
            progress(0.1, f"authoring {kind} with {model or OLLAMA_MODEL}")

        raw, _ = chat(system=system, user=user, model=model, temperature=0.4)

        while True:
            try:
                svg_src = extract(raw)
                clean, sanitize_notes = sanitize(svg_src, size, size)
                png = rasterize(clean)
                notes.extend(sanitize_notes)
                break
            except SvgError as exc:
                last_error = str(exc)
                if attempts >= max_repairs:
                    # Return the best effort rather than failing outright, so the
                    # caller can see and fix the source themselves.
                    warnings.append(
                        f"SVG failed validation after {attempts} repair attempt(s): {last_error}"
                    )
                    preview = _write(out_path, raw, None)
                    return Artifact(
                        path=out_path,
                        kind="svg",
                        mime="image/svg+xml",
                        width=size,
                        height=size,
                        preview_path=preview,
                        text=raw,
                        prompt_used=prompt,
                        model=model or OLLAMA_MODEL,
                        backend=self.name,
                        elapsed_s=time.perf_counter() - start,
                        warnings=warnings,
                        meta={"valid": False, "repairs": attempts, "error": last_error, "mode": "author"},
                    )
                attempts += 1
                if progress:
                    progress(0.4, f"repair {attempts}/{max_repairs}: {last_error[:60]}")
                notes.append(f"repair {attempts}: {last_error}")
                raw, _ = chat(
                    system=SVG_REPAIR_PROMPT,
                    user=f"Error:\n{last_error}\n\nSVG source:\n{raw}",
                    model=model,
                    temperature=0.2,
                )

        preview = _write(out_path, clean, png)
        if progress:
            progress(1.0, "done")

        return Artifact(
            path=out_path,
            kind="svg",
            mime="image/svg+xml",
            width=size,
            height=size,
            preview_path=preview,
            text=clean,
            prompt_used=prompt,
            model=model or OLLAMA_MODEL,
            backend=self.name,
            elapsed_s=time.perf_counter() - start,
            warnings=warnings,
            meta={
                "valid": True,
                "repairs": attempts,
                "mode": "author",
                "svg_kind": kind,
                "path_count": path_count(clean),
                "bytes": len(clean),
                "notes": notes,
            },
        )


class SvgTraceBackend:
    """FLUX raster -> posterize -> vtracer paths.

    Holds a manager reference so the raster step goes through the normal GPU
    lease rather than loading a second copy of FLUX.
    """

    name = "svg.trace"
    kinds = ("svg_trace",)
    needs_gpu = False  # the nested image call takes the lease itself
    vram_estimate_mb = 0

    def __init__(self, manager) -> None:
        self._manager = manager

    @property
    def loaded(self) -> bool:
        return True

    def load(self) -> None:
        return

    def unload(self) -> None:
        return

    def generate(self, params: dict[str, Any], progress: Progress | None = None) -> Artifact:
        import vtracer
        from PIL import Image

        prompt: str = params["prompt"]
        size: int = int(params.get("size") or 1024)
        colors: int = int(params.get("colors") or 12)
        out_path = Path(params["out_path"])
        warnings: list[str] = list(params.get("warnings", []))

        start = time.perf_counter()
        if progress:
            progress(0.1, "rendering raster with FLUX")

        raster_path = sibling_path(out_path, ".raster.png")
        image_artifact = self._manager.run(
            "image.flux2-klein",
            {
                "prompt": prompt + TRACE_STYLE_SUFFIX,
                "width": size,
                "height": size,
                "steps": params.get("steps") or 4,
                "seed": params.get("seed"),
                "out_path": raster_path,
                "_suppress_guidance_warning": True,
            },
        )
        warnings.extend(image_artifact.warnings)

        if progress:
            progress(0.6, f"posterizing to {colors} colours")

        # Posterizing first is what keeps vtracer from emitting a 2000-path mess.
        img = Image.open(image_artifact.path).convert("RGB")
        img.quantize(colors=colors, method=Image.Quantize.MEDIANCUT).convert("RGB").save(raster_path)

        if progress:
            progress(0.75, "tracing to vector paths")

        out_path.parent.mkdir(parents=True, exist_ok=True)
        vtracer.convert_image_to_svg_py(
            str(raster_path),
            str(out_path),
            colormode="color",
            mode="spline",
            filter_speckle=8,
            color_precision=6,
            path_precision=3,
        )

        raw = out_path.read_text(encoding="utf-8")
        try:
            clean, notes = sanitize(raw, size, size)
            png = rasterize(clean)
            valid = True
        except SvgError as exc:  # vtracer output should always be clean; be honest if not
            warnings.append(f"traced SVG failed validation: {exc}")
            clean, notes, png, valid = raw, [], None, False

        preview = _write(out_path, clean, png)
        paths = path_count(clean)
        size_bytes = len(clean)

        # Trace output is far too large to ever inline into a tool result.
        if size_bytes > 12000:
            warnings.append(
                f"traced SVG is {size_bytes // 1024} KB across {paths} paths -- "
                "returned as a file path, not inline"
            )

        return Artifact(
            path=out_path,
            kind="svg",
            mime="image/svg+xml",
            width=size,
            height=size,
            preview_path=preview,
            text=None,  # never inline traced output
            prompt_used=prompt,
            model=image_artifact.model,
            backend=self.name,
            quantization=image_artifact.quantization,
            seed=params.get("seed"),
            elapsed_s=time.perf_counter() - start,
            warnings=warnings,
            meta={
                "valid": valid,
                "mode": "trace",
                "path_count": paths,
                "bytes": size_bytes,
                "colors": colors,
                "raster_path": str(image_artifact.path),
                "notes": notes,
            },
        )
