"""Prompt enhancement via Ollama. CPU-only, so it never takes a GPU lease.

This closes the gap the README always described but the service never had: the
`app/` stack went straight from raw prompt to diffusion, while only the legacy
CLI scripts enhanced anything.
"""

from __future__ import annotations

import time
from typing import Any

from app.backends.base import Artifact, Progress
from app.backends.llm import chat
from app.config import LLM_MODEL as OLLAMA_MODEL
from app.prompts import SYSTEM_PROMPTS

DEFAULT_STYLE = "flux2"


def resolve_style(style: str | None) -> str:
    """Accept a style name, or a 1-based index into the prompt list.

    The old `nlp_model.py` menu printed numbered options but then looked the
    *number* up as a dict key, so it silently always fell through to "flux".
    Accepting both forms fixes that.
    """
    if not style:
        return DEFAULT_STYLE
    key = str(style).strip()
    if key in SYSTEM_PROMPTS:
        return key
    if key.isdigit():
        idx = int(key) - 1
        names = list(SYSTEM_PROMPTS)
        if 0 <= idx < len(names):
            return names[idx]
    return DEFAULT_STYLE


def enhance_prompt(prompt: str, style: str | None = None, model: str | None = None, suggest_overlays: bool = False) -> dict[str, Any]:
    if suggest_overlays:
        resolved = "flux2_overlay"
    else:
        resolved = resolve_style(style)
    content, elapsed = chat(
        system=SYSTEM_PROMPTS[resolved],
        user=prompt,
        model=model,
        temperature=0.8,
    )
    result: dict[str, Any] = {
        "prompt": content.strip(),
        "original": prompt,
        "style": resolved,
        "model": model or OLLAMA_MODEL,
        "elapsed_s": round(elapsed, 2),
    }
    if suggest_overlays:
        import json as _json
        text = content.strip()
        try:
            # Try to parse as JSON
            parsed = _json.loads(text)
            result["prompt"] = parsed.get("prompt", text)
            result["overlay"] = parsed.get("overlay")
        except _json.JSONDecodeError:
            # If JSON parsing fails, try to extract JSON from the response
            import re
            json_match = re.search(r'\{[\s\S]*"prompt"[\s\S]*\}', text)
            if json_match:
                try:
                    parsed = _json.loads(json_match.group())
                    result["prompt"] = parsed.get("prompt", text)
                    result["overlay"] = parsed.get("overlay")
                except _json.JSONDecodeError:
                    result["overlay"] = None
            else:
                result["overlay"] = None
    return result


class EnhanceBackend:
    name = "enhance.ollama"
    kinds = ("enhance",)
    needs_gpu = False
    vram_estimate_mb = 0

    @property
    def loaded(self) -> bool:
        return True

    def load(self) -> None:  # nothing to hold resident
        return

    def unload(self) -> None:
        return

    def generate(self, params: dict[str, Any], progress: Progress | None = None) -> Artifact:
        start = time.perf_counter()
        result = enhance_prompt(
            params["prompt"], params.get("style"), params.get("model")
        )
        return Artifact(
            path=params.get("out_path") or "",
            kind="enhance",
            mime="text/plain",
            text=result["prompt"],
            prompt_used=result["prompt"],
            model=result["model"],
            backend=self.name,
            elapsed_s=time.perf_counter() - start,
            meta=result,
        )
