"""Main Typer application for LocalImageGen.

Registers the top-level commands (``generate``, ``batch``, ``enhance``) and the
nested sub-apps (``models``, ``config``) on a single :class:`typer.Typer`
instance. Structured logging is configured once, before any command runs, from
the resolved application settings.
"""

from __future__ import annotations

import typer

from .commands.batch import batch
from .commands.config import config_app
from .commands.enhance import enhance
from .commands.generate import generate
from .commands.models import models_app
from ..config.settings import Settings
from ..services.logging import configure_logging

app = typer.Typer(
    name="localimagegen",
    help="Local AI image generation pipeline.",
    no_args_is_help=True,
    add_completion=False,
    pretty_exceptions_show_locals=False,
)


@app.callback()
def _configure_runtime() -> None:
    """Configure structured logging from the resolved settings."""
    settings = Settings.from_cli_args()
    configure_logging(settings.log_level)


app.command(name="generate")(generate)
app.command(name="batch")(batch)
app.command(name="enhance")(enhance)
app.add_typer(models_app)
app.add_typer(config_app)


if __name__ == "__main__":
    app()