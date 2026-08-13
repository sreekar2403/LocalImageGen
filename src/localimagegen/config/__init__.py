"""Configuration loading for LocalImageGen.

Provides a single :class:`Settings` object backed by ``pydantic-settings`` that
merges configuration from multiple sources with the following precedence
(highest wins):

1. CLI arguments
2. Environment variables (including ``.env`` files)
3. YAML configuration file
4. Built-in defaults
"""

from localimagegen.config.settings import Settings

__all__ = ["Settings"]
