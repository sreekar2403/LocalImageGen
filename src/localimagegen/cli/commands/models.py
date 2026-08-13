"""``models`` subcommand: inspect the local Ollama server."""

from __future__ import annotations

import typer
from rich.console import Console
from rich.table import Table

from ..adapters.llm.ollama import OllamaAdapter
from ..config.settings import Settings

console = Console()

models_app = typer.Typer(
    name="models",
    help="Inspect models available in the local Ollama server.",
    no_args_is_help=True,
)


def _format_size(size: int) -> str:
    """Format a byte count into a human-readable string."""
    value = float(size)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if value < 1024 or unit == "TB":
            return f"{int(value)} {unit}" if unit == "B" else f"{value:.1f} {unit}"
        value /= 1024
    return f"{value:.1f} TB"


@models_app.command("list")
def models_list() -> None:
    """List models available in the local Ollama server."""
    try:
        settings = Settings.from_cli_args()
        adapter = OllamaAdapter(settings)
        models = adapter.list_models()
    except Exception as exc:
        console.print(f"[bold red]Failed to list Ollama models:[/bold red] {exc}")
        raise typer.Exit(code=1) from exc

    if not models:
        console.print("[yellow]No models found in the local Ollama server.[/yellow]")
        return

    table = Table(title="Available Ollama models")
    table.add_column("Name", style="cyan")
    table.add_column("Size", justify="right")
    table.add_column("Modified", style="magenta")

    for model in models:
        table.add_row(
            str(model.get("name", "")),
            _format_size(int(model.get("size", 0) or 0)),
            str(model.get("modified_at", "")),
        )

    console.print(table)
