"""Cross-cutting services for LocalImageGen.

This package holds infrastructure-adjacent services (such as structured
logging) that are shared across the application but are not part of the domain
layer.
"""

from localimagegen.services.logging import configure_logging, get_logger

__all__ = ["configure_logging", "get_logger"]
