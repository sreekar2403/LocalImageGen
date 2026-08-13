"""``enhance`` command: enhance a prompt with the local LLM (no image generation)."""

from __future__ import annotations

import typer
from rich.console import Console
from rich.panel import Panel

from ..adapters.llm.ollama import OllamaAdapter
from ..config.settings import Settings
from ..core.models import Prompt, PromptStyle

console = Console()


def enhance(
    prompt: str = typer.Argument(
        ...,
        help="Text prompt to enhance.",
    ),
    style: PromptStyle = typer.Option(
        PromptStyle.FLUX,
        "--style",
        help="Prompt-enhancement style used by the local LLM (flux, sdxl, midjourney).",
    ),
) -> None:
    """Enhance a prompt with the local Ollama model and print the result."""
    try:
        settings = Settings.from_cli_args()
        adapter = OllamaAdapter(settings)
        result = adapter.enhance(Prompt(text=prompt), style)
    except Exception as exc:
        console.print(f"[bold red]Enhancement failed:[/bold red] {exc}")
        raise typer.Exit(code=1) from exc

    console.print(
        Panel(
            result.effective,
            title=f"Enhanced prompt (style: {style.value})",
            border_style="cyan",
        )
    )
