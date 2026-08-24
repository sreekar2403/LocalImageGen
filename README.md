# LocalGen

Local **image** and **SVG** generation on your own GPU, reachable from any MCP
harness — Claude Code, Claude Desktop, OpenCode, pi, deepseek. Fully offline
after the model downloads.

Built and measured on an RTX 4060 Laptop (8 GB VRAM).

| | |
|---|---|
| Image | **10.4 s** at 1024x1024, FLUX.2-klein-4B, FP8, 5001 MiB peak |
| SVG (author) | 25–90 s, LLM-written paths, sanitized and render-validated |
| SVG (trace) | ~8 s, FLUX raster then vectorized — best for artwork |
| Video | not built (see AGENTS.md) |

## How it works

One background worker owns the GPU. Harnesses talk to it through a thin MCP
adapter that imports no torch and starts in under a second, so several harnesses
can be connected at once and restarting one costs nothing.

```
harness --stdio--> app/mcp_server.py --HTTP--> app/main.py --> GPU
                   (thin, ~680ms)              (owns models)
```

Only one model fits in 8 GB, so a `ModelManager` runs every GPU task on a single
thread and evicts whatever is resident before loading something else. It also
reclaims VRAM from LM Studio, which otherwise holds ~5.7 GB.

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Then either LM Studio (default — load `qwen3.5-4b`, start the local server) or
Ollama (`LOCALIMAGEGEN_LLM_PROVIDER=ollama`, which runs it CPU-only).

## Use

```powershell
python main.py serve                      # start the worker
python main.py image "a lighthouse at dusk" --platform youtube
python main.py svg "rocket icon"
python main.py svg "fox mascot sticker" --trace
python main.py status
```

HTTP: `POST /generate`, `/v1/image`, `/v1/image/edit`, `/v1/svg`, `/v1/svg/edit`,
`/v1/enhance`. Swagger at <http://127.0.0.1:8765/docs>.

From a harness: `generate_image`, `edit_image`, `generate_svg`, `edit_svg`,
`enhance_prompt`, `list_presets`, `service_status`, `evict_models`.
See [docs/mcp-registration.md](docs/mcp-registration.md).

## Notes worth knowing

- **`guidance_scale` and `negative_prompt` are ignored.** FLUX.2-klein is
  distilled, so classifier-free guidance is disabled. Both are accepted for
  compatibility and return a warning rather than pretending to work.
- **FP8 quantization is what makes this usable** — 11x faster than bf16 on this
  card, because in bf16 the model does not fit and the driver spills to system
  RAM on every step. See [bench.md](bench.md).
- **Image editing is free** — FLUX.2 takes reference images natively, so
  `edit_image` needs no extra model.
- Generated files land in `~/LocalImageGen/` by default.

## Layout

```
app/mcp_server.py   MCP adapter (no torch)
app/launcher.py     worker discovery + auto-spawn
app/worker.py       GPU-owning process
app/main.py         FastAPI routes
app/manager.py      VRAM arbiter
app/service.py      modality orchestration
app/backends/       image_flux, svg, enhance, llm
app/svgtool.py      SVG sanitize + render-validate
main.py             CLI
scripts/bench.py    performance measurement
tests/              51 tests, no GPU required
```
