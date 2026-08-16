# AGENTS.md

## Project Overview

Local AI image/video generation pipeline. Two components:
- **`legacy/`** — Standalone scripts (FLUX.2 for images, WAN 2.1 for video, Ollama for prompt enhancement)
- **`multi-source/`** — FastAPI microservice wrapping the same models

## Setup & Run

```bash
# Install dependencies (uses uv with PyTorch CUDA 12.6 index)
uv pip install -e .

# Run the FastAPI service
cd multi-source
uvicorn app:app --reload --host 0.0.0.0 --port 8000

# API docs at http://localhost:8000/docs
```

Legacy scripts run directly: `python legacy/vision_model.py` or `python legacy/batch_generate.py`

## Key Models

| Task | Model | Source |
|------|-------|--------|
| Image generation | `black-forest-labs/FLUX.2-klein-4B` | HuggingFace |
| Video generation | `Wan-AI/Wan2.1-T2V-1.3B-Diffusers` | HuggingFace |
| Prompt enhancement | `gemma4:e4b` | Ollama (local) |

Models are downloaded on first run. Requires CUDA GPU with sufficient VRAM.

## Known Issues

- `multi-source/app.py` has syntax errors (duplicate import blocks, truncated class). Will fail on import.
- `pyproject.toml` defines CLI entry `localimagegen = "localimagegen.cli:app"` but no `localimagegen/` package exists.
- No tests, no linter config, no CI pipeline.
- Video generation scripts write to `/tmp/generated_video.mp4` — not cross-platform on Windows.

## Conventions

- Python 3.10+
- `uv` is the package manager (not pip/poetry)
- PyTorch installed from `pytorch-cu126` index (CUDA 12.6)
- No linting/formatting tools configured
