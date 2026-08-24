"""HTTP worker. This is the ONLY process that touches the GPU.

Run it with:  python -m app.worker      (or: uvicorn app.main:app)
"""

from __future__ import annotations

import os

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse

from app import service
from app.config import (
    DEFAULT_GUIDANCE_SCALE,
    MODEL_NAME,
    OUTPUT_DIR,
    PLATFORMS,
    SVG_PRESETS,
)
from app.manager import get_manager
from app.schemas import (
    ArtifactResponse,
    EditRequest,
    EnhanceRequest,
    GenerateRequest,
    GenerateResponse,
    HealthResponse,
    ImageRequest,
    SvgEditRequest,
    SvgRequest,
)
from app.storage import safe_join

app = FastAPI(
    title="LocalGen",
    description=(
        "Local image and SVG generation (FLUX.2-klein-4B + Ollama). "
        "One GPU-owning worker, fronted by MCP adapters for any harness."
    ),
    version="0.2.0",
)


# --- status ------------------------------------------------------------------


@app.get("/health", response_model=HealthResponse)
def health():
    """Never touches the GPU, so auto-spawn can poll it during startup."""
    data = service.health()
    return HealthResponse(worker_pid=os.getpid(), **data)


@app.get("/platforms")
def platforms():
    return PLATFORMS


@app.get("/capabilities")
def capabilities():
    return {
        "kinds": ["image", "edit", "svg", "enhance"],
        "image_model": MODEL_NAME,
        "platforms": PLATFORMS,
        "svg_presets": SVG_PRESETS,
        "svg_modes": ["author", "trace"],
        "notes": {
            "guidance_scale": "ignored (distilled model)",
            "negative_prompt": "ignored (distilled model)",
        },
    }


@app.get("/models")
def models():
    return get_manager().status()


# --- generation --------------------------------------------------------------


@app.post("/generate", response_model=GenerateResponse)
def generate(req: GenerateRequest):
    """Legacy contract, preserved byte-for-byte apart from the added `warnings`."""
    try:
        art = service.generate_image(
            prompt=req.prompt,
            platform=req.platform,
            width=req.width,
            height=req.height,
            steps=req.num_inference_steps,
            seed=req.seed,
            path=req.path,
            negative_prompt=req.negative_prompt,
            guidance_scale=req.guidance_scale,
            enhance=req.enhance,
            style=req.style,
        )
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"generation failed: {exc}") from exc

    return GenerateResponse(
        image_path=str(art.path),
        model=art.model or MODEL_NAME,
        width=art.width or 0,
        height=art.height or 0,
        steps=art.steps or 0,
        guidance_scale=req.guidance_scale or DEFAULT_GUIDANCE_SCALE,
        seed=art.seed,
        quantization=art.quantization or "none",
        generation_time_s=round(art.elapsed_s, 2),
        warnings=art.warnings,
    )


def _artifact_response(art) -> ArtifactResponse:
    return ArtifactResponse(**art.to_dict())


@app.post("/v1/image", response_model=ArtifactResponse)
def v1_image(req: ImageRequest):
    try:
        return _artifact_response(
            service.generate_image(
                prompt=req.prompt, platform=req.platform, width=req.width,
                height=req.height, steps=req.steps, seed=req.seed, path=req.path,
                negative_prompt=req.negative_prompt, guidance_scale=req.guidance_scale,
                enhance=req.enhance, style=req.style,
            )
        )
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"generation failed: {exc}") from exc


@app.post("/v1/image/edit", response_model=ArtifactResponse)
def v1_edit(req: EditRequest):
    for p in req.image_paths:
        if not os.path.isfile(p):
            raise HTTPException(status_code=400, detail=f"reference image not found: {p}")
    try:
        return _artifact_response(
            service.generate_image(
                prompt=req.prompt, width=req.width, height=req.height,
                steps=req.steps, seed=req.seed, path=req.path,
                reference_images=req.image_paths,
            )
        )
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"edit failed: {exc}") from exc


@app.post("/v1/svg", response_model=ArtifactResponse)
def v1_svg(req: SvgRequest):
    if req.mode not in ("author", "trace"):
        raise HTTPException(status_code=422, detail="mode must be 'author' or 'trace'")
    try:
        return _artifact_response(
            service.generate_svg(
                prompt=req.prompt, mode=req.mode, svg_kind=req.svg_kind,
                size=req.size, path=req.path, colors=req.colors, seed=req.seed,
                model=req.model, max_repairs=req.max_repairs,
                enhance=req.enhance, style=req.style,
            )
        )
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"svg generation failed: {exc}") from exc


@app.post("/v1/svg/edit", response_model=ArtifactResponse)
def v1_svg_edit(req: SvgEditRequest):
    try:
        return _artifact_response(
            service.edit_svg(req.svg, req.instruction, req.path, req.model)
        )
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"svg edit failed: {exc}") from exc


@app.post("/v1/enhance")
def v1_enhance(req: EnhanceRequest):
    from app.backends.enhance import enhance_prompt

    try:
        return enhance_prompt(req.prompt, req.style, req.model)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=503, detail=f"enhancement failed: {exc}") from exc


# --- files & admin -----------------------------------------------------------


@app.get("/images/{filename}")
def serve_image(filename: str):
    """Traversal-guarded. Previously this had no check at all."""
    try:
        path = safe_join(OUTPUT_DIR, filename)
    except ValueError:
        raise HTTPException(status_code=400, detail="invalid filename") from None
    if not path.is_file():
        raise HTTPException(status_code=404, detail="image not found")
    return FileResponse(path)


@app.post("/admin/evict")
def admin_evict(backend: str | None = None):
    return get_manager().evict(backend)


@app.post("/admin/shutdown", status_code=202)
def admin_shutdown():
    """Clean stop for an auto-spawned worker."""
    import threading

    def _stop():
        import time

        time.sleep(0.2)
        get_manager().shutdown()
        os._exit(0)

    threading.Thread(target=_stop, daemon=True).start()
    return {"stopping": True}


if __name__ == "__main__":
    import uvicorn

    from app.config import HOST, PORT

    uvicorn.run(app, host=HOST, port=PORT)
