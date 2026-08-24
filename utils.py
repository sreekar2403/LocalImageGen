"""Back-compat shim. The prompt library now lives in `app/prompts.py`."""

from app.prompts import SVG_PROMPTS, SVG_REPAIR_PROMPT, SYSTEM_PROMPTS

__all__ = ["SYSTEM_PROMPTS", "SVG_PROMPTS", "SVG_REPAIR_PROMPT"]
