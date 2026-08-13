"""``generate`` command: render a single image from a prompt."""

from __future__ import annotations

from typing import Optional

import typer
from rich.console import Console
from rich.panel import Panel

from ..config.settings import Settings
from ..core.models import GenerationParams, Prompt, PromptStyle
from ..pipelines.flux import FluxPipeline

console = Console()


def generate(
    prompt: str = typer.Argument(
        ...,
        help="Text prompt describing the image to generate.",
    ),
    style: PromptStyle = typer.Option(
        PromptStyle.FLUX,
        "--style",
        help="Prompt-enhancement style used by the local LLM (flux, sdxl, midjourney).",
    ),
    width: Optional[int] = typer.Option(
        None,
        "--width",
        min=64,
        max=4096,
        help="Output image width in pixels (defaults to settings).",
    ),
    height: Optional[int] = typer.Option(
        None,
        "--height",
        min=64,
        max=4096,
        help="Output image height in pixels (defaults to settings).",
    ),
    steps: Optional[int] = typer.Option(
        None,
        "--steps",
        min=1,
        max=1000,
        help="Number of diffusion inference steps (defaults to settings).",
    ),
    seed: Optional[int] = typer.Option(
        None,
        "--seed",
        help="Random seed for reproducibility (defaults to settings).",
    ),
    output: Optional[str] = typer.Option(
        None,
        "--output",
        help="Base file name (without extension) for the saved image.",
    ),
    no_enhance: bool = typer.Option(
        False,
        "--no-enhance",
        help="Skip LLM prompt enhancement and use the raw prompt as-is.",
    ),
) -> None:
    """Generate a single image from a text prompt."""
    try:
        settings = Settings.from_cli_args()
        params = GenerationParams(
            width=width if width is not None else settings.width,
            height=height if height is not None else settings.height,
            num_inference_steps=steps if steps is not None else settings.num_inference_steps,
            seed=seed if seed is not None else settings.seed,
            device=settings.device,
        )

        pipeline = FluxPipeline(settings)
        path = pipeline.run(
            Prompt(text=prompt),
            style=style,
            params=params,
            output_name=output,
            enhance=not no_enhance,
        )
    except Exception as exc:
        console.print(f"[bold red]Generation failed:[/bold red] {exc}")
        raise typer.Exit(code=1) from exc

    console.print(
        Panel(
            f"[bold green]Image generated successfully[/bold green]\n\n"
            f"Prompt: [white]{prompt}[/white]\n"
            f"Style: [cyan]{style.value}[/cyan]\n"
            f"Size: [cyan]{params.width}x{params.height}[/cyan]\n"
            f"Steps: [cyan]{params.num_inference_steps}[/cyan]\n"
            f"Seed: [cyan]{params.seed}[/cyan]\n"
            f"Saved to: [magenta]{path}[/magenta]",
            title="Generation complete",
            border_style="green",
        )
    )
