"""``config`` subcommand: inspect the resolved application settings."""

from __future__ import annotations

import typer
from rich.console import Console
from rich.table import Table

from ..config.settings import Settings

console = Console()

config_app = typer.Typer(
    name="config",
    help="Inspect the resolved application configuration.",
    no_args_is_help=True,
)


@config_app.command("show")
def config_show() -> None:
    """Display the current resolved application settings."""
    try:
        settings = Settings.from_cli_args()
    except Exception as exc:
        console.print(f"[bold red]Failed to load settings:[/bold red] {exc}")
        raise typer.Exit(code=1) from exc

    table = Table(title="LocalImageGen configuration")
    table.add_column("Setting", style="cyan")
    table.add_column("Value", style="white")

    for field, value in settings.model_dump(mode="json").items():
        table.add_row(field, str(value))

    console.print(table)
