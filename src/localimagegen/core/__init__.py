"""Core domain layer for LocalImageGen.

This package contains the domain models and port (interface) definitions. It is
the innermost layer of the Clean Architecture and must have **zero**
infrastructure dependencies - it imports only from the Python standard library
and ``pydantic`` for data validation.
"""

from localimagegen.core.models import (
    GeneratedImage,
    GenerationMetadata,
    GenerationParams,
    Prompt,
)
from localimagegen.core.ports import DiffusionPort, LLMPort, StoragePort

__all__ = [
    "DiffusionPort",
    "GeneratedImage",
    "GenerationMetadata",
    "GenerationParams",
    "LLMPort",
    "Prompt",
    "StoragePort",
]
