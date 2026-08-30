from typing import Any, List, Optional

from pydantic import BaseModel, Field

from app.config import DEFAULT_GUIDANCE_SCALE, DEFAULT_STEPS


class GenerateRequest(BaseModel):
    """Frozen legacy contract for POST /generate. Field names unchanged."""

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "prompt": "a cinematic YouTube video poster with a dramatic hero shot, bold title text, and a vibrant gradient overlay",
                    "platform": "youtube",
                    "num_inference_steps": 4,
                    "seed": 42,
                    "path": "C:/Users/me/Pictures/youtube-poster.png",
                }
            ]
        }
    }

    prompt: str = Field(..., min_length=1, description="Text prompt for image generation")
    platform: str = Field("default", description="Platform preset that determines default dimensions")
    width: Optional[int] = Field(None, ge=256, le=1024, description="Width (overrides platform preset); snapped to a multiple of 16")
    height: Optional[int] = Field(None, ge=256, le=1024, description="Height (overrides platform preset); snapped to a multiple of 16")
    num_inference_steps: int = Field(DEFAULT_STEPS, ge=1, le=50)
    guidance_scale: float = Field(
        DEFAULT_GUIDANCE_SCALE,
        ge=0.1,
        le=15.0,
        description="DEPRECATED / IGNORED. FLUX.2-klein is distilled, so classifier-free guidance is disabled. Accepted for compatibility; returns a warning.",
    )
    seed: Optional[int] = Field(None, description="Seed for reproducibility (None = random)")
    negative_prompt: Optional[str] = Field(
        None,
        description="DEPRECATED / IGNORED. Negative prompt embeddings are never applied on a distilled model. Accepted for compatibility; returns a warning.",
    )
    path: Optional[str] = Field(
        None,
        description="Explicit output path. A directory saves a generated filename inside; a filename ending in an image extension is used as-is. Defaults to ~/LocalImageGen/images/",
    )
    enhance: bool = Field(False, description="Expand the prompt with a local Ollama model first")
    style: Optional[str] = Field(None, description="Enhancement style: flux2 (default), flux, sdxl, midjourney")


class GenerateResponse(BaseModel):
    """Frozen legacy shape. `warnings` is the only additive field."""

    image_path: str
    model: str
    width: int
    height: int
    steps: int
    guidance_scale: float
    seed: Optional[int]
    quantization: str
    generation_time_s: float
    warnings: List[str] = []


class ArtifactResponse(BaseModel):
    """Uniform shape for every modality on the /v1 routes."""

    kind: str
    path: str
    preview_path: Optional[str] = None
    mime: str
    width: Optional[int] = None
    height: Optional[int] = None
    text: Optional[str] = None
    prompt_used: Optional[str] = None
    model: Optional[str] = None
    backend: Optional[str] = None
    quantization: Optional[str] = None
    seed: Optional[int] = None
    steps: Optional[int] = None
    elapsed_s: float = 0.0
    warnings: List[str] = []
    meta: dict = {}


class ImageRequest(BaseModel):
    prompt: str = Field(..., min_length=1)
    platform: str = "default"
    width: Optional[int] = Field(None, ge=256, le=1024)
    height: Optional[int] = Field(None, ge=256, le=1024)
    steps: int = Field(DEFAULT_STEPS, ge=1, le=50)
    seed: Optional[int] = None
    path: Optional[str] = None
    enhance: bool = False
    style: Optional[str] = None
    negative_prompt: Optional[str] = Field(None, description="IGNORED on this distilled model")
    guidance_scale: Optional[float] = Field(None, description="IGNORED on this distilled model")
    text_overlay: Optional[str] = Field(None, description="Text to overlay on the generated image")
    text_position: str = Field("bottom", description="Position of text overlay: top, bottom, center, top-left, top-right, bottom-left, bottom-right")
    text_color: str = Field("#FFFFFF", description="Text color in hex format")
    text_bg_color: str = Field("#00000080", description="Text background color in hex format with alpha")
    text_font_size: int = Field(48, ge=8, le=200, description="Text font size in pixels")
    text_padding: int = Field(20, ge=0, le=100, description="Padding around text in pixels")


class EditRequest(BaseModel):
    prompt: str = Field(..., min_length=1)
    image_paths: List[str] = Field(..., min_length=1, description="One or more reference images (FLUX.2 supports multi-reference editing)")
    width: Optional[int] = Field(None, ge=256, le=1024)
    height: Optional[int] = Field(None, ge=256, le=1024)
    steps: int = Field(DEFAULT_STEPS, ge=1, le=50)
    seed: Optional[int] = None
    path: Optional[str] = None


class SvgRequest(BaseModel):
    model_config = {
        "json_schema_extra": {
            "examples": [
                {"prompt": "a rocket taking off", "mode": "author", "svg_kind": "icon", "size": 512}
            ]
        }
    }

    prompt: str = Field(..., min_length=1)
    mode: str = Field("author", description="author = LLM writes SVG source; trace = FLUX raster then vectorize")
    svg_kind: str = Field("icon", description="icon | logo | diagram | chart | illustration")
    size: Optional[int] = Field(None, ge=64, le=4096)
    colors: int = Field(12, ge=2, le=64, description="Posterization colours for trace mode")
    seed: Optional[int] = None
    model: Optional[str] = Field(None, description="Ollama model override")
    max_repairs: int = Field(2, ge=0, le=5)
    path: Optional[str] = None
    enhance: bool = False
    style: Optional[str] = None


class SvgEditRequest(BaseModel):
    svg: str = Field(..., description="Path to an .svg file, or literal SVG source")
    instruction: str = Field(..., min_length=1)
    path: Optional[str] = None
    model: Optional[str] = None


class EnhanceRequest(BaseModel):
    prompt: str = Field(..., min_length=1)
    style: Optional[str] = Field(None, description="flux2 (default), flux, sdxl, midjourney")
    model: Optional[str] = None
    suggest_overlays: bool = Field(False, description="Also suggest text overlay settings (text, position, color, font_size)")


class HealthResponse(BaseModel):
    """Keeps all five original fields; everything else is additive."""

    status: str
    model: str
    loaded: bool
    quantization: str
    load_error: Optional[str] = None
    resident_backend: Optional[str] = None
    vram_free_mb: int = 0
    vram_total_mb: int = 0
    swaps: int = 0
    idle_s: float = 0.0
    backends: dict[str, Any] = {}
    worker_pid: Optional[int] = None
    version: str = "0.2.0"


class VideoRequest(BaseModel):
    model_config = {
        "json_schema_extra": {
            "examples": [{"prompt": "a paper boat drifting down a rain gutter", "preset": "short-480p"}]
        }
    }

    prompt: str = Field(..., min_length=1)
    preset: str = Field("short-480p", description="short-480p | tiny-480p | long-480p | square-480p | portrait")
    width: Optional[int] = Field(None, ge=256, le=1280, description="Must be divisible by 16")
    height: Optional[int] = Field(None, ge=256, le=1280, description="Must be divisible by 16")
    num_frames: Optional[int] = Field(None, ge=5, le=161, description="(n-1) must be divisible by 4; snapped if not")
    steps: Optional[int] = Field(None, ge=1, le=50)
    guidance_scale: Optional[float] = Field(None, ge=1.0, le=15.0, description="Wan is NOT distilled, so this genuinely applies")
    fps: Optional[int] = Field(None, ge=1, le=60)
    seed: Optional[int] = None
    negative_prompt: Optional[str] = None
    path: Optional[str] = None
    codec: Optional[str] = Field(None, description="libx264 (default) or h264_nvenc")
    enhance: bool = False
    style: Optional[str] = None


class JobRef(BaseModel):
    job_id: str
    kind: str
    status: str
    created_at: float


class JobStatus(BaseModel):
    id: str
    kind: str
    status: str
    progress: float = 0.0
    progress_msg: Optional[str] = None
    error: Optional[str] = None
    result: Optional[dict] = None
    created_at: Optional[float] = None
    started_at: Optional[float] = None
    finished_at: Optional[float] = None
    elapsed_s: Optional[float] = None
    cancel_requested: bool = False


class JobSubmit(BaseModel):
    kind: str = Field(..., description="video | image | svg")
    params: dict = Field(default_factory=dict)
