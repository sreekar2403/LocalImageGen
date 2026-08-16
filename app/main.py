import time

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse

from app.config import (
    DEFAULT_GUIDANCE_SCALE,
    DEFAULT_SEED,
    DEFAULT_STEPS,
    MODEL_NAME,
    PLATFORMS,
    resolve_dimensions,
)
from app.pipeline import get_pipeline
from app.schemas import GenerateRequest, GenerateResponse, HealthResponse
from app.storage import resolve_user_dir, unique_image_path

app = FastAPI(
    title="Local Image Gen",
    description="FLUX.2-klein-4B image generation API for OpenCode/Hermes harnesses",
    version="0.1.0",
)


@app.get("/health", response_model=HealthResponse)
def health():
    pipe = get_pipeline()
    return HealthResponse(
        status="ok" if pipe.loaded else "loading",
        model=MODEL_NAME,
        loaded=pipe.loaded,
        quantization=pipe.quantization,
        load_error=pipe.load_error,
    )


@app.get("/platforms")
def platforms():
    return PLATFORMS


@app.post("/generate", response_model=GenerateResponse)
def generate(req: GenerateRequest):
    width, height = resolve_dimensions(req.platform, req.width, req.height)
    steps = req.num_inference_steps or DEFAULT_STEPS
    guidance = req.guidance_scale or DEFAULT_GUIDANCE_SCALE
    seed = req.seed if req.seed is not None else DEFAULT_SEED

    pipe = get_pipeline()
    if not pipe.loaded:
        try:
            pipe.load()
        except Exception as exc:  # noqa: BLE001 - surface load failure to caller
            raise HTTPException(status_code=503, detail=f"model load failed: {exc}")

    user_dir = resolve_user_dir(req.user_id)
    out_path = unique_image_path(user_dir)

    start = time.time()
    try:
        image = pipe.generate(
            prompt=req.prompt,
            width=width,
            height=height,
            steps=steps,
            guidance_scale=guidance,
            seed=seed,
            negative_prompt=req.negative_prompt,
        )
        image.save(out_path)
    except Exception as exc:  # noqa: BLE001 - surface generation failure
        raise HTTPException(status_code=500, detail=f"generation failed: {exc}")

    elapsed = time.time() - start

    return GenerateResponse(
        image_path=str(out_path),
        model=MODEL_NAME,
        width=width,
        height=height,
        steps=steps,
        guidance_scale=guidance,
        seed=seed,
        quantization=pipe.quantization,
        generation_time_s=round(elapsed, 2),
    )


@app.get("/images/{user_id}/{filename}")
def serve_image(user_id: str, filename: str):
    user_dir = resolve_user_dir(user_id)
    path = user_dir / filename
    if not path.is_file():
        raise HTTPException(status_code=404, detail="image not found")
    return FileResponse(path, media_type="image/png")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)