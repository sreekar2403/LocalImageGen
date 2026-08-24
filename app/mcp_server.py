"""MCP adapter for any harness: Claude Code, Claude Desktop, OpenCode, pi, deepseek.

    python -m app.mcp_server                 # stdio (default)
    python -m app.mcp_server --transport http

THIN BY DESIGN. This module must never import torch or diffusers. It is a pure
httpx client over the GPU worker, so it starts in milliseconds, any number of
harnesses can run it at once, and a harness restart costs nothing instead of
reloading 15 GB of weights.

Tools return FILE PATHS, never base64. Claude Code does not convert MCP
ImageContent into a native image block -- base64 lands as text at roughly
15-25k tokens per image and trips "result exceeds maximum allowed tokens" (the
API caps base64 at 5 MB, Claude Desktop caps tool results near 1 MB). Handing
back a path lets the harness read the file itself, which costs ~600 tokens.
SVG is the exception: it is text, so the source is returned inline under a
budget, letting the agent edit it directly.
"""

from __future__ import annotations

import argparse
from typing import Any, Optional

import httpx

from app.config import BASE_URL
from app.launcher import WorkerUnavailable, ensure_worker

# Generation is slow relative to normal HTTP; give it plenty of room.
REQUEST_TIMEOUT = httpx.Timeout(connect=10.0, read=900.0, write=30.0, pool=10.0)

# Inline SVG source up to this many characters; beyond it, return the path only.
INLINE_SVG_BUDGET = 12_000
# Raises the harness's persist-to-disk threshold for this tool's results.
SVG_META = {"anthropic/maxResultSizeChars": 120_000}


def _call(method: str, path: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
    base = ensure_worker(BASE_URL)
    url = f"{base}{path}"
    with httpx.Client(timeout=REQUEST_TIMEOUT) as client:
        resp = client.request(method, url, json=payload)
    if resp.status_code >= 400:
        try:
            detail = resp.json().get("detail", resp.text)
        except Exception:  # noqa: BLE001
            detail = resp.text
        raise RuntimeError(f"worker returned {resp.status_code}: {detail}")
    return resp.json()


def _warnings_block(art: dict[str, Any]) -> str:
    warns = art.get("warnings") or []
    return "".join(f"\nnote: {w}" for w in warns)


def _image_summary(art: dict[str, Any]) -> str:
    meta = art.get("meta") or {}
    bits = [f"{art.get('width')}x{art.get('height')}"]
    if art.get("steps"):
        bits.append(f"{art['steps']} steps")
    if art.get("seed") is not None:
        bits.append(f"seed {art['seed']}")
    if art.get("quantization"):
        bits.append(art["quantization"])
    if meta.get("peak_vram_mb"):
        bits.append(f"{meta['peak_vram_mb']} MiB peak")
    bits.append(f"{art.get('elapsed_s', 0)}s")
    return (
        f"Saved: {art['path']}\n"
        + " · ".join(bits)
        + "\nRead the file to view it."
        + _warnings_block(art)
    )


def build_server():
    from mcp.server import MCPServer

    mcp = MCPServer(
        "LocalGen",
        instructions=(
            "Local image and SVG generation on this machine (FLUX.2-klein-4B + Ollama). "
            "Tools return file paths; use your file reader to view an image. "
            "guidance_scale and negative_prompt are accepted but IGNORED because the "
            "image model is distilled."
        ),
    )

    @mcp.tool()
    def generate_image(
        prompt: str,
        platform: str = "default",
        width: Optional[int] = None,
        height: Optional[int] = None,
        steps: int = 4,
        seed: Optional[int] = None,
        path: Optional[str] = None,
        enhance: bool = False,
        style: Optional[str] = None,
    ) -> str:
        """Generate an image locally with FLUX.2-klein-4B and return its file path.

        Takes roughly 10-15s. `platform` picks preset dimensions (youtube, reels,
        instagram, ...). Set `enhance` to expand the prompt with a local LLM first.
        """
        art = _call(
            "POST",
            "/v1/image",
            {
                "prompt": prompt, "platform": platform, "width": width,
                "height": height, "steps": steps, "seed": seed, "path": path,
                "enhance": enhance, "style": style,
            },
        )
        return _image_summary(art)

    @mcp.tool()
    def edit_image(
        prompt: str,
        image_paths: list[str],
        width: Optional[int] = None,
        height: Optional[int] = None,
        steps: int = 4,
        seed: Optional[int] = None,
        path: Optional[str] = None,
    ) -> str:
        """Edit or recombine existing images with a text instruction.

        FLUX.2 supports multi-reference editing: pass several paths to blend or
        transfer style between them.
        """
        art = _call(
            "POST",
            "/v1/image/edit",
            {
                "prompt": prompt, "image_paths": image_paths, "width": width,
                "height": height, "steps": steps, "seed": seed, "path": path,
            },
        )
        return _image_summary(art)

    @mcp.tool(meta=SVG_META)
    def generate_svg(
        prompt: str,
        svg_kind: str = "icon",
        mode: str = "author",
        size: Optional[int] = None,
        colors: int = 12,
        path: Optional[str] = None,
        max_repairs: int = 2,
        model: Optional[str] = None,
    ) -> str:
        """Generate an SVG locally and return its source plus a PNG preview path.

        mode="author" (default) has a local LLM write real vector paths -- best for
        icons, logos, diagrams and charts. mode="trace" renders with FLUX then
        vectorizes it -- best for illustrations and stickers. svg_kind is one of
        icon, logo, diagram, chart, illustration.
        """
        art = _call(
            "POST",
            "/v1/svg",
            {
                "prompt": prompt, "svg_kind": svg_kind, "mode": mode, "size": size,
                "colors": colors, "path": path, "max_repairs": max_repairs,
                "model": model,
            },
        )
        meta = art.get("meta") or {}
        head = (
            f"SVG:     {art['path']}\n"
            f"Preview: {art.get('preview_path')}\n"
            f"{meta.get('mode')} · {meta.get('path_count', '?')} paths · "
            f"{meta.get('bytes', 0)} bytes · valid={meta.get('valid')} · "
            f"repairs={meta.get('repairs', 0)} · {art.get('elapsed_s', 0)}s"
            + _warnings_block(art)
        )
        source = art.get("text")
        if source and len(source) <= INLINE_SVG_BUDGET:
            return f"{head}\n\n{source}"
        if source:
            return f"{head}\n\n(source is {len(source)} chars -- read the file to see it)"
        return f"{head}\n\nRead the preview PNG to view it."

    @mcp.tool(meta=SVG_META)
    def edit_svg(svg: str, instruction: str, path: Optional[str] = None) -> str:
        """Modify an existing SVG. `svg` is a path to an .svg file or literal source."""
        art = _call("POST", "/v1/svg/edit", {"svg": svg, "instruction": instruction, "path": path})
        meta = art.get("meta") or {}
        head = (
            f"SVG:     {art['path']}\n"
            f"Preview: {art.get('preview_path')}\n"
            f"valid={meta.get('valid')} · {art.get('elapsed_s', 0)}s"
            + _warnings_block(art)
        )
        source = art.get("text")
        if source and len(source) <= INLINE_SVG_BUDGET:
            return f"{head}\n\n{source}"
        return head

    @mcp.tool()
    def enhance_prompt(prompt: str, style: Optional[str] = None) -> str:
        """Expand a short idea into a detailed image prompt using a local LLM.

        style: flux2 (default, prose -- correct for FLUX.2), flux, sdxl, midjourney.
        """
        data = _call("POST", "/v1/enhance", {"prompt": prompt, "style": style})
        return data["prompt"]

    @mcp.tool()
    def list_presets() -> str:
        """List platform dimension presets and SVG kinds."""
        caps = _call("GET", "/capabilities")
        lines = ["Platforms:"]
        lines += [
            f"  {n}: {c['width']}x{c['height']} ({c['aspect']})"
            for n, c in caps["platforms"].items()
        ]
        lines.append("SVG kinds:")
        lines += [f"  {n}: {c['size']}px -- {c['hint']}" for n, c in caps["svg_presets"].items()]
        lines.append("SVG modes: " + ", ".join(caps["svg_modes"]))
        return "\n".join(lines)

    # Kept so existing OpenCode/Hermes registrations keep working unchanged.
    @mcp.tool()
    def list_platforms() -> str:
        """Deprecated alias for list_presets."""
        return list_presets()

    @mcp.tool()
    def service_status() -> str:
        """Worker status: VRAM, which model is resident, and load state.

        Also the cheapest way to warm the worker up before generating.
        """
        h = _call("GET", "/health")
        return (
            f"worker pid={h.get('worker_pid')} · {h.get('status')}\n"
            f"model={h.get('model')} loaded={h.get('loaded')} "
            f"quantization={h.get('quantization')}\n"
            f"resident={h.get('resident_backend')} · "
            f"VRAM {h.get('vram_free_mb')}/{h.get('vram_total_mb')} MiB free · "
            f"swaps={h.get('swaps')} · idle={h.get('idle_s')}s\n"
            f"backends: {', '.join(h.get('backends', {}))}"
        )

    @mcp.tool()
    def evict_models() -> str:
        """Unload the resident model to free VRAM for another application."""
        s = _call("POST", "/admin/evict")
        return f"evicted. resident={s.get('resident_backend')} VRAM free {s.get('vram_free_mb')} MiB"

    # `health` was the original tool name; keep it working.
    @mcp.tool()
    def health() -> str:
        """Deprecated alias for service_status."""
        return service_status()

    return mcp


def main() -> None:
    parser = argparse.ArgumentParser(description="LocalGen MCP adapter")
    parser.add_argument("--transport", default="stdio", choices=["stdio", "sse", "streamable-http", "http"])
    args = parser.parse_args()
    transport = "streamable-http" if args.transport == "http" else args.transport

    try:
        build_server().run(transport=transport)
    except WorkerUnavailable as exc:
        raise SystemExit(str(exc)) from exc


if __name__ == "__main__":
    main()
