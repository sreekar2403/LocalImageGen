"""HTTP worker. This is the ONLY process that touches the GPU.

Run it with:  python -m app.worker      (or: uvicorn app.main:app)
"""

from __future__ import annotations

import os
import time
import uuid
from pathlib import Path

from fastapi import FastAPI, HTTPException, UploadFile, File as FastAPIFile
from fastapi.responses import FileResponse, HTMLResponse

from app import service
from app.config import (
    DEFAULT_GUIDANCE_SCALE,
    MODEL_NAME,
    OUTPUT_DIR,
    OUTPUT_ROOT,
    PLATFORMS,
    SVG_PRESETS,
    VIDEO_PRESETS,
)
from app.manager import get_manager
from app.schemas import (
    ArtifactResponse,
    JobRef,
    JobStatus,
    JobSubmit,
    VideoRequest,
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
        "kinds": ["image", "edit", "svg", "enhance", "video"],
        "image_model": MODEL_NAME,
        "platforms": PLATFORMS,
        "svg_presets": SVG_PRESETS,
        "svg_modes": ["author", "trace"],
        "video_presets": VIDEO_PRESETS,
        "video_model": __import__("app.config", fromlist=["VIDEO_MODEL"]).VIDEO_MODEL,
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
                text_overlay=req.text_overlay, text_position=req.text_position,
                text_color=req.text_color, text_bg_color=req.text_bg_color,
                text_font_size=req.text_font_size, text_padding=req.text_padding,
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
        return enhance_prompt(req.prompt, req.style, req.model, req.suggest_overlays)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=503, detail=f"enhancement failed: {exc}") from exc


# --- video (always async) ----------------------------------------------------


@app.post("/v1/video", response_model=JobRef, status_code=202)
def v1_video(req: VideoRequest):
    """Video always goes through the job queue -- it takes minutes, not seconds."""
    from app.jobs import get_store

    params = req.model_dump(exclude_none=True)
    job_id = get_store().submit("video", params)
    job = get_store().get(job_id)
    return JobRef(job_id=job_id, kind="video", status=job["status"], created_at=job["created_at"])


# --- jobs --------------------------------------------------------------------


@app.post("/jobs", response_model=JobRef, status_code=202)
def submit_job(req: JobSubmit):
    from app.jobs import get_store

    if req.kind not in ("video", "image", "svg"):
        raise HTTPException(status_code=422, detail=f"unknown job kind {req.kind!r}")
    job_id = get_store().submit(req.kind, req.params)
    job = get_store().get(job_id)
    return JobRef(job_id=job_id, kind=req.kind, status=job["status"], created_at=job["created_at"])


@app.get("/jobs", response_model=list[JobStatus])
def list_jobs(status: str | None = None, kind: str | None = None, limit: int = 20):
    from app.jobs import get_store

    return [JobStatus(**j) for j in get_store().list(status, kind, limit)]


@app.get("/jobs/{job_id}", response_model=JobStatus)
def get_job(job_id: str):
    from app.jobs import get_store

    job = get_store().get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"no such job: {job_id}")
    return JobStatus(**job)


@app.post("/jobs/{job_id}/cancel", response_model=JobStatus)
def cancel_job(job_id: str):
    from app.jobs import get_store

    job = get_store().request_cancel(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"no such job: {job_id}")
    return JobStatus(**job)


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


# --- web UI ------------------------------------------------------------------

_WEB_DIR = Path(__file__).resolve().parent / "web"


@app.get("/", response_class=HTMLResponse)
def index():
    """Serve the single-page chat UI. Same process, so server + UI share one command."""
    html = (_WEB_DIR / "index.html").read_text(encoding="utf-8")
    return HTMLResponse(html)


@app.post("/v1/upload")
async def upload(file: UploadFile = FastAPIFile(...)):
    """Persist an attachment on the worker's disk; the same machine runs the UI,
    so the returned absolute path can be handed straight to /v1/image/edit.
    """
    root = OUTPUT_ROOT / "uploads"
    root.mkdir(parents=True, exist_ok=True)
    ext = Path(file.filename or "bin").suffix
    name = f"up-{time.strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:6]}{ext}"
    dest = root / name
    dest.write_bytes(await file.read())
    return {"path": str(dest), "filename": name}


@app.get("/file")
def serve_file(p: str):
    """Traversal-guarded file server for any generated or uploaded artifact."""
    allowed = [OUTPUT_ROOT.resolve(), OUTPUT_DIR.resolve()]
    candidate = Path(p).resolve()
    if not any(candidate == a or a in candidate.parents for a in allowed):
        raise HTTPException(status_code=400, detail="file not accessible")
    if not candidate.is_file():
        raise HTTPException(status_code=404, detail="file not found")
    return FileResponse(candidate)


if __name__ == "__main__":
    import uvicorn

    from app.config import HOST, PORT

    uvicorn.run(app, host=HOST, port=PORT)
