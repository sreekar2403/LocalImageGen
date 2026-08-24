"""Back-compat shim over `app/backends/enhance.py`.

Two bugs from the original are fixed by delegating:

1. The style menu printed numbered options but looked the *number* up as a dict
   key, so it always silently fell through to "flux". `resolve_style()` now
   accepts either the number or the name.
2. `os.environ["OLLAMA_GPU"] = "0"` plus a `subprocess` call to `ollama stop`
   did nothing useful -- Ollama is a separate server, and that is not a real
   Ollama variable. VRAM is now controlled with `options={"num_gpu": 0}` and
   `keep_alive=0` on the chat call itself.
"""

from __future__ import annotations

from app.backends.enhance import enhance_prompt, resolve_style
from app.prompts import SYSTEM_PROMPTS

__all__ = ["generate_prompt", "enhance_prompt", "resolve_style", "SYSTEM_PROMPTS"]


def generate_prompt(user_prompt: str, style: str | None = None) -> str:
    """Enhance a prompt. Prompts interactively for a style only if not given."""
    if style is None:
        names = list(SYSTEM_PROMPTS)
        print("Prompt styles:")
        for i, name in enumerate(names, 1):
            print(f"  [{i}]: {name}")
        style = input("Enter option (number or name): ").strip()

    return enhance_prompt(user_prompt, style)["prompt"]
