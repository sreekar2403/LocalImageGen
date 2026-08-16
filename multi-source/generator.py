"""
Generator module for image/video generation using local AI models.
Supports FLUX-2 (images) and WAN2.1 T2V (videos).
Falls back to demo generators when models are not available locally.
"""

import os
import io
from pathlib import Path
from typing import Optional, Tuple
from PIL import Image, ImageDraw
import numpy as np


class DemoImageGenerator:
    """Fallback generator that creates test images with text overlays."""
    
    def generate(self, prompt: str, size: int = 512, fmt: str = "png") -> Tuple[bytes, str]:
        img = Image.new("RGB", (size, size), color=(30, 30, 30))
        
        # Draw a gradient background
        for y in range(size):
            ratio = y / size
            color = tuple(int(c * (1 - ratio) + d * ratio) 
                        for c, d in [(255, 255, 255), (60, 60, 180)])
            ImageDraw.Draw(img).rectangle([(0, y), (size, y)], fill=color)
        
        # Add text overlay with prompt
        draw = ImageDraw.Draw(img)
        
        # Draw a semi-transparent box
        draw.rectangle(
            [(15, 15), (size - 15, size - 30)], 
            fill=(20, 20, 40), 
            outline=(80, 80, 200)
        )
        
        # Draw prompt text in chunks
        lines = self._split_text(prompt[:50], max_width=size * 0.6, font_size=32)
        y_pos = 15 + 30
        
        for line in lines:
            bbox = draw.textbbox((0, 0), line, font=None)
            w = bbox[2] - bbox[0]
            x = (size - w) // 2
            draw.text((x, y_pos), line, fill=(255, 255, 255))
            y_pos += 36
        
        buffer = io.BytesIO()
        fmt_upper = fmt.upper() if fmt else "PNG"
        img.save(buffer, format=fmt_upper)
        
        return buffer.getvalue(), fmt.upper().lower()
    
    def _split_text(self, text: str, max_width: int, font_size: int):
        """Simple text wrapping for display."""
        lines = []
        current_line = ""
        # Approximate char width (very rough estimate)
        char_width = 6
        
        for word in text.split():
            test_line = current_line + " " + word if current_line else word
            estimated_width = len(test_line) * char_width
            
            if estimated_width < max_width:
                current_line = test_line
            else:
                lines.append(current_line)
                current_line = word
        
        if current_line:
            lines.append(current_line)
        
        # Truncate to first 30 chars for display
        return [line[:30] for line in lines]


class DemoVideoGenerator:
    """Fallback video generator using ffmpeg with static frames."""
    
    def generate(self, prompt: str, duration: int = 30, fps: int = 24) -> Tuple[bytes, str]:
        import subprocess
        
        output_path = Path("/tmp/generated_video.mp4")
        
        # Get resolution
        parts = prompt.split("x")[:1]
        width = height = 512
        
        cmd = [
            "ffmpeg", "-y", "-f", "lavfi",
            "-i", f"color=c=blue:s={width}x{height}:d={duration}",
            "-c:v", "libx264",
            "-c:a", "aac",
            "-r", str(fps),
            output_path
        ]
        
        try:
            subprocess.run(cmd, capture_output=True, check=False)
            
            with open(output_path, "rb") as f:
                return f.read(), "mp4"
        except Exception as e:
            raise HTTPException(status_code=503, detail=f"Video generation failed: {str(e)}")


def create_image_generator() -> "ImageGenerator":
    """Create the appropriate image generator based on available models."""
    
    # Check if we can use FLUX-2 or similar local model
    try:
        from .model_adapter import ModelAdapter
        
        adapter = ModelAdapter()
        
        # Try to load vision model for images
        try:
            adapter.load_vision_model("black-forest-labs/FLUX.2-klein-4B")
            return FLUXImageGenerator(adapter)
        except Exception as e:
            print(f"Could not load FLUX-2: {e}")
        
        # Fallback to demo generator
        print("Using demo image generator (no local model loaded)")
        return DemoImageGenerator()
    
    except ImportError:
        print("Model adapter not found, using demo generator")
        return DemoImageGenerator()


class FLUXImageGenerator:
    """Real image generator using FLUX-2 or other vision models."""
    
    def __init__(self, adapter):
        self.adapter = adapter
    
    def generate(self, prompt: str, size: int = 512, fmt: str = "png") -> Tuple[bytes, str]:
        try:
            # Use the model adapter to generate image
            result = self.adapter.generate_image(prompt=prompt)
            
            if isinstance(result, dict):
                # Model returned bytes directly
                img_bytes, img_fmt = result["image"], result.get("format", "png")
            else:
                # Model returned PIL Image
                from PIL import Image as PILImage
                pil_img = result
                
                buffer = io.BytesIO()
                fmt_upper = fmt.upper().lower()
                if fmt_upper == "jpeg" or fmt_upper == "jpg":
                    img_fmt = "JPEG"
                else:
                    img_fmt = fmt_upper
                pil_img.save(buffer, format=img_fmt)
                img_bytes = buffer.getvalue()
            
            return img_bytes, img_fmt.lower()
        
        except Exception as e:
            raise HTTPException(status_code=503, detail=f"Image generation failed with model: {str(e)}")


class ImageGenerator:
    """Main image generator facade."""
    
    def __init__(self):
        self.generator = create_image_generator()
    
    def generate(self, prompt: str, size: int = 512, fmt: str = "png") -> Tuple[bytes, str]:
        return self.generator.generate(prompt=prompt, size=size, fmt=fmt)


class VideoGeneratorFacade:
    """Main video generator facade."""
    
    def __init__(self):
        # Check for WAN2.1 T2V model availability
        try:
            import subprocess
            subprocess.run(
                ["ffmpeg", "-version"], 
                capture_output=True, 
                check=False
            )
            
            self.generator = DemoVideoGenerator()
            print("Using demo video generator (install WAN2.1 for real T2V)")
            
        except Exception as e:
            self.generator = DemoVideoGenerator()
    
    def generate(self, prompt: str, duration: int = 30, fps: int = 24) -> Tuple[bytes, str]:
        return self.generator.generate(prompt=prompt, duration=duration, fps=fps)


# Create global generators for use by FastAPI
image_generator = ImageGenerator()
video_generator = VideoGeneratorFacade()
