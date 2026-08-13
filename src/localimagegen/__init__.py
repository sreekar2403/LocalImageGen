"""LocalImageGen - Local AI image generation pipeline.

This package implements the Clean Architecture foundation for the LocalImageGen
project. It is organized into three layers:

- ``config``: Multi-source configuration loading (CLI args > env vars > YAML > defaults).
- ``core``: Domain models and port (interface) definitions with zero infrastructure
  dependencies.
- ``services``: Cross-cutting infrastructure services such as structured logging.
"""

from localimagegen.config.settings import Settings

__all__ = ["Settings"]
