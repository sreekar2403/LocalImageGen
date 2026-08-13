"""Command-line interface for LocalImageGen.

The CLI is built with ``typer`` and renders output with ``rich``. It exposes the
following commands:

- ``generate``: generate a single image from a prompt.
- ``batch``: generate many images from a YAML file of prompts.
- ``enhance``: enhance a prompt with the local LLM (no image generation).
- ``models list``: list models available in the local Ollama server.
- ``config show``: display the resolved application settings.
"""

from .app import app

__all__ = ["app"]
