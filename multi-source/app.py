"""
FastAPI Microservice - Text to Image/Video Generation

Endpoints:
  POST /generate - Generate image/video from text prompt
  GET  /health    - Health check endpoint
"""

from fastapi import FastAPI, Request, HTTPException, File, UploadFile, Query
from pydantic import BaseModel
import io
import numpy as np
from pathlib import Path
from typing import Optional
import base64

# Try to import local generation modules (gracefully fallback for demo)
try:
    from .generator import ImageGenerator, VideoGenerator
except ImportError as e:
    print(f
try:
    from .generator import ImageGenerator, VideoGenerator
except ImportError:
    # Demo generators that create test patterns
    class ImageGenerator:
        def generate(self, prompt: str, size: int = 512, fmt: str = "png"):
            return self._demo_generate(prompt, size, fmt)

        def _demo_generate(self, prompt: str, size: int, fmt: str):
            from PIL import Image, ImageDraw
            img = Image.new("RGB", (size, size), color="black")
            draw = ImageDraw.Draw(img)
            draw.text((10, 10), "Image: " + prompt[:30], fill=(255, 255, 255))
            buffer = io.BytesIO()
            img.save(buffer, format=fmt.upper())
            return buffer.getvalue(), fmt.upper().lower()

    class VideoGenerator:
        def generate(self, prompt: str, duration: int = 30, fps: int = 30):
            # Demo video generator using ffmpeg or simple frames
            import subprocess
            output_path = Path("/tmp/generated_video.mp4")
            
            # Create a placeholder video frame
            cmd = [
                "ffmpeg", "-y",
                "-framerate", str(fps),
                "-f", "lavfi",
                "-i", "color=c=blue:s=512x512:d=" + str(1/fps),
                "-c:v", "libx264",
                "-c:a", "aac",
                output_path
            ]
            
            try:
                subprocess.run(cmd, capture_output=True)
                with open(output_path, "rb") as f:
                    return f.read(), "mp4"
            except Exception as e:
                raise HTTPException(status_code=503, detail=f"Video generation failed: {str(e)}")

app = FastAPI(
    title="Local Image Generation Microservice",
    description="Text-to-Image/Video API powered by local AI models",
    version="0.1.0"
)


class GenerateRequest(BaseModel):
    """Request body for image/video generation"""
    prompt: str = Query(..., min_length=1, max_length=5000, description="Text prompt for generation")
    fmt: Optional[str] = Query(
        "png", 
        regex=r"^(png|jpeg|jpg)$",
        description="Output image format (png or jpeg)"
    )
    size: int = Query(512, ge=64, le=2048, description="Image dimensions (square)")
    steps: Optional[int] = Query(30, ge=1, le=100, description="Generation steps for images")
    guidance_scale: Optional[float] = Query(7.5, gt=0, description="Classifier-free guidance scale")


class GenerateVideoRequest(BaseModel):
    """Request body for video generation"""
    prompt: str = Query(..., min_length=1, max_length=5000)
    duration: int = Query(30, ge=1, le=600, description="Video duration in seconds")
    fps: int = Query(24, ge=8, le=60, description="Frames per second")
    resolution: str = Query("512x512", regex=r"^\d+x\d+$", description="Resolution (widthxheight)")


# Initialize generators
try:
    from .generator import ImageGenerator, VideoGenerator
    image_gen = ImageGenerator()
    video_gen = VideoGenerator()
    print('Successfully imported generators from generator.py')
except Exception as e:
    print(f'Import error: {e}')
    # Demo generators that create test patterns
    class ImageGenerator:
        def generate(self, prompt: str, size: int = 512, fmt: str = 
video_gen = VideoGenerator()


@app.get("/health", tags=["Health"])
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "local-image-gen-api",
        "version": "0.1.0"
    }


@app.post(
    "/generate/image",
    response_model=dict,
    tags=["Image Generation"],
    summary="Generate image from text prompt",
    description="Generate a PNG/JPEG image based on the text prompt"
)
async def generate_image(request: GenerateRequest):
    """
    Generate an image from a text prompt.
    
    **Parameters:**
    - `prompt`: Text description of the desired image
    - `fmt`: Output format - png or jpeg (default: png)
    - `size`: Image dimension in pixels (64-2048, default: 512)
    - `steps`: Number of diffusion steps (default: 30)
    - `guidance_scale`: Guidance scale for sampling (default: 7.5)
    
    **Returns:**
    - Binary file content in the specified format
    """
    try:
        image_bytes, fmt = image_gen.generate(
            prompt=request.prompt,
            size=request.size,
            fmt=request.fmt.lower()
        )
        
        return {
            "success": True,
            "format": fmt,
            "size_bytes": len(image_bytes),
            "data": base64.b64encode(image_bytes).decode("utf-8")
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Image generation failed: {str(e)}")


@app.post(
    "/generate/video",
    response_model=dict,
    tags=["Video Generation"],
    summary="Generate video from text prompt",
    description="Generate an MP4 video based on the text prompt"
)
async def generate_video(request: GenerateVideoRequest):
    """
    Generate a video from a text prompt.
    
    **Parameters:**
    - `prompt`: Text description of the desired video scene
    - `duration`: Video duration in seconds (default: 30)
    - `fps`: Frames per second (default: 24)
    - `resolution`: Resolution as widthxheight (e.g., "512x512")
    
    **Returns:**
    - Binary file content as MP4, or base64 encoded data
    """
    try:
        # Parse resolution
        parts = request.resolution.split("x")
        if len(parts) != 2:
            raise HTTPException(status_code=400, detail="Invalid resolution format (use WxH)")
        
        video_bytes, fmt = video_gen.generate(
            prompt=request.prompt,
            duration=request.duration,
            fps=request.fps
        )
        
        return {
            "success": True,
            "format": fmt,
            "size_bytes": len(video_bytes),
            "duration_seconds": request.duration,
            "fps": request.fps,
            "data": base64.b64encode(video_bytes).decode("utf-8")
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Video generation failed: {str(e)}")


@app.get("/generate/image/batch", tags=["Image Generation"])
async def generate_image_batch(
    prompts: list[str] = Query(..., min_items=1, max_items=10),
    fmt: str = Query("png", regex=r"^(png|jpeg)$"),
    size: int = Query(512)
):
    """Generate multiple images from text prompts in a single request."""
    results = []
    for i, prompt in enumerate(prompts):
        try:
            image_bytes, img_fmt = image_gen.generate(
                prompt=prompt,
                size=size,
                fmt=fmt.lower()
            )
            results.append({
                "index": i,
                "prompt": prompt[:50],  # Truncate for display
                "format": img_fmt,
                "size_bytes": len(image_bytes),
                "data": base64.b64encode(image_bytes).decode("utf-8")
            })
        except Exception as e:
            results.append({
                "index": i,
                "prompt": prompt[:50],
                "error": str(e)
            })
    
    return {"generated_count": len(results), "results": results}


# Run with uvicorn
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
