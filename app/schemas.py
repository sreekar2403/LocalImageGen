from typing import Optional

from pydantic import BaseModel, Field


class GenerateRequest(BaseModel):
    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "prompt": "a cinematic YouTube video poster with a dramatic hero shot, bold title text, and a vibrant gradient overlay",
                    "platform": "youtube",
                    "num_inference_steps": 4,
                    "guidance_scale": 3.5,
                    "seed": 42,
                    "user_id": "alice",
                }
            ]
        }
    }

    prompt: str = Field(
        ..., min_length=1, description="Text prompt for image generation", example="a cinematic video poster with a dramatic hero shot and gradient overlay"
    )
    platform: str = Field("default", description="Platform preset that determines default dimensions", example="youtube")
    width: Optional[int] = Field(None, ge=256, le=1024, description="Width (overrides platform preset)", example=None)
    height: Optional[int] = Field(None, ge=256, le=1024, description="Height (overrides platform preset)", example=None)
    num_inference_steps: int = Field(4, ge=1, le=50, example=4)
    guidance_scale: float = Field(3.5, ge=0.1, le=15.0, example=3.5)
    seed: Optional[int] = Field(None, description="Seed for reproducibility (None = random)", example=42)
    negative_prompt: Optional[str] = None
    user_id: Optional[str] = Field(None, description="Per-user output subdirectory", example="alice")


class GenerateResponse(BaseModel):
    image_path: str
    model: str
    width: int
    height: int
    steps: int
    guidance_scale: float
    seed: Optional[int]
    quantization: str
    generation_time_s: float


class HealthResponse(BaseModel):
    status: str
    model: str
    loaded: bool
    quantization: str
    load_error: Optional[str]