"""``batch`` command: generate images from a YAML file of prompts."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import typer
import yaml
from rich.console import Console
from rich.table import Table

from ..config.settings import Settings
from ..core.models import GenerationParams, Prompt, PromptStyle
from ..pipelines.flux import FluxPipeline

console = Console()


def _load_entries(path: Path) -> list[dict[str, Any]]:
    """Load and validate the batch YAML file into a list of prompt entries.

    Supports either a bare list of entries or a mapping with a ``prompts`` key.
    Each entry must be a mapping containing at least a non-empty ``prompt`` key.
    """
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or []
    if isinstance(data, dict):
        data = data.get("prompts", [])
    if not isinstance(data, list):
        raise ValueError(
            f"Batch file '{path}' must contain a list of prompts "
            "or a mapping with a 'prompts' list."
        )
    entries: list[dict[str, Any]] = []
    for index, item in enumerate(data, start=1):
        if not isinstance(item, dict) or not str(item.get("prompt", "")).strip():
            raise ValueError(
                f"Batch entry #{index} must be a mapping with a non-empty 'prompt'."
            )
        entries.append(item)
    return entries


def _entry_style(entry: dict[str, Any]) -> PromptStyle:
    """Resolve the entry's ``style`` field to a :class:`PromptStyle`."""
    raw = entry.get("style")
    if raw is None:
        return PromptStyle.FLUX
    try:
        return PromptStyle(str(raw))
    except ValueError:
        choices = ", ".join(style.value for style in PromptStyle)
        raise ValueError(f"Unsupported style '{raw}'; expected one of: {choices}.")


def _entry_params(entry: dict[str, Any], settings: Settings) -> GenerationParams:
    """Build generation parameters from an entry, falling back to settings."""
    return GenerationParams(
        width=int(entry.get("width", settings.width)),
        height=int(entry.get("height", settings.height)),
        num_inference_steps=int(entry.get("steps", settings.num_inference_steps)),
        seed=int(entry.get("seed", settings.seed)),
        device=settings.device,
    )


def batch(
    yaml_file: Path = typer.Argument(
        ...,
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True,
        help="Path to a YAML file containing the prompts to generate.",
    ),
    no_enhance: bool = typer.Option(
        False,
        "--no-enhance",
        help="Skip LLM prompt enhancement for every entry.",
    ),
) -> None:
    """Generate images for every prompt in a YAML batch file."""
    try:
        entries = _load_entries(yaml_file)
    except Exception as exc:
        console.print(f"[bold red]Failed to read batch file:[/bold red] {exc}")
        raise typer.Exit(code=1) from exc

    try:
        settings = Settings.from_cli_args()
        pipeline = FluxPipeline(settings)
    except Exception as exc:
        console.print(f"[bold red]Failed to initialize pipeline:[/bold red] {exc}")
        raise typer.Exit(code=1) from exc

    results = Table(title=f"Batch generation results ({yaml_file})")
    results.add_column("Name", style="cyan")
    results.add_column("Status", justify="center")
    results.add_column("Output", style="magenta")

    for index, entry in enumerate(entries, start=1):
        name = str(entry.get("name", f"entry-{index}"))
        console.print(f"[bold]Generating {index}/{len(entries)}:[/bold] {name}")
        try:
            style = _entry_style(entry)
            params = _entry_params(entry, settings)
            path = pipeline.run(
                Prompt(text=str(entry["prompt"])),
                style=style,
                params=params,
                output_name=name,
                enhance=not no_enhance,
            )
            results.add_row(name, "[green]ok[/green]", str(path))
            console.print(f"  [green]Saved:[/green] {path}")
        except Exception as exc:
            results.add_row(name, "[red]failed[/red]", str(exc))
            console.print(f"  [red]Failed:[/red] {exc}")

    console.print()
    console.print(results)
