"""MCP server exposing image generation as tools for OpenCode/Hermes.

Run standalone (stdio transport):
    python -m app.mcp_server

Register in opencode.json as a local MCP server, then tools such as
`generate_image` become available to the harness.
"""

import time
from typing import Optional

from app.config import (
    DEFAULT_GUIDANCE_SCALE,
    DEFAULT_STEPS,
    MODEL_NAME,
    PLATFORMS,
    resolve_dimensions,
)
from app.pipeline import get_pipeline
from app.storage import resolve_user_dir, unique_image_path


def build_server():
    from mcp.server import MCPServer

    mcp = MCPServer("LocalImageGen")

    @mcp.tool()
    def generate_image(
        prompt: str,
        platform: str = "default",
        width: Optional[int] = None,
        height: Optional[int] = None,
        num_inference_steps: int = DEFAULT_STEPS,
        guidance_scale: float = DEFAULT_GUIDANCE_SCALE,
        seed: Optional[int] = None,
        negative_prompt: Optional[str] = None,
        user_id: Optional[str] = None,
    ) -> str:
        """Generate an image with FLUX.2-klein-4B and return its saved path."""
        dim_w, dim_h = resolve_dimensions(platform, width, height)

        pipe = get_pipeline()
        if not pipe.loaded:
            pipe.load()

        user_dir = resolve_user_dir(user_id)
        out_path = unique_image_path(user_dir)

        start = time.time()
        image = pipe.generate(
            prompt=prompt,
            width=dim_w,
            height=dim_h,
            steps=num_inference_steps,
            guidance_scale=guidance_scale,
            seed=seed,
            negative_prompt=negative_prompt,
        )
        image.save(out_path)
        elapsed = round(time.time() - start, 2)

        return (
            f"Image saved to {out_path} "
            f"({dim_w}x{dim_h}, {num_inference_steps} steps, "
            f"guidance {guidance_scale}, seed {seed}, {elapsed}s, "
            f"quantization {pipe.quantization})"
        )

    @mcp.tool()
    def list_platforms() -> str:
        """Return supported platform dimension presets (aspect-ratio aware)."""
        return "\n".join(
            f"{name}: {cfg['width']}x{cfg['height']} ({cfg['aspect']})"
            for name, cfg in PLATFORMS.items()
        )

    @mcp.tool()
    def health() -> str:
        """Return model load status and active quantization mode."""
        pipe = get_pipeline()
        return (
            f"model={MODEL_NAME} loaded={pipe.loaded} "
            f"quantization={pipe.quantization} error={pipe.load_error}"
        )

    return mcp


def main():
    build_server().run()


if __name__ == "__main__":
    main()