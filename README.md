# LocalGen

Local **image** and **SVG** generation on your own GPU, reachable from any MCP
harness — Claude Code, Claude Desktop, OpenCode, pi, deepseek. Fully offline
after the model downloads.

Built and measured on an RTX 4060 Laptop (8 GB VRAM).

| | |
|---|---|
| Image (generate) | FLUX.2-klein-4B, FP8, **10.4 s** at 1024x1024/4 steps, 5001 MiB peak — the single backend |
| Image (edit) | FLUX.2-klein-4B native reference conditioning, same 4-step path |
| Prompts | auto-normalized to klein (60–140w prose, positive-only); `enhance` opts into LLM `flux2` rewrite |
| SVG (author) | 25–90 s, LLM-written paths, few-shot + budget-constrained, sanitized and render-validated |
| SVG (trace) | ~8 s, FLUX raster then vectorized, adaptive palette + density fallback — best for artwork |
| Video | Wan2.1-T2V-1.3B, two-stage CPU-encode/GPU-denoise, 480p presets incl. medium-480p (~4 s) |

History: FLUX.1-dev/Kontext-dev via GGUF was tried and abandoned
(`enable_sequential_cpu_offload()` incompatible with GGUF weights, see
bench.md "Phase 1"). SD3.5 Medium + InstructPix2Pix were measured working
(bench.md "Phase 2") then removed with the Qwen-Image prototype when the
project standardized on klein-only for simplicity: one backend, one 4-step
path, deterministic prompt normalization instead of chasing CFG dials.

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
reclaims VRAM from LM Studio, which otherwise holds ~5.7 GB. Image generation
and editing share the one slot via `image.flux2-klein`; video (`video.wan21-t2v-1.3b`)
takes the same slot when a video job runs.

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

- **Prompts are auto-normalized for klein; `guidance_scale`/`negative_prompt`/`image_guidance_scale` are ignored with a warning.** FLUX.2-klein-4B is distilled (`is_distilled: true`, no CFG), so the only dial is the words themselves: 60–140w natural prose, explicit spatial relations, quoted text, single focal idea. The normalizer strips quality-tag salad, folds negations to positives, and truncates overlong prompts. `enhance=true` opts into an LLM `flux2` rewrite on top.
- **FLUX.1-dev/Kontext-dev via GGUF was tried and abandoned** —
  `enable_sequential_cpu_offload()` is fundamentally incompatible with
  GGUF-quantized weights in diffusers 0.39.0 (confirmed by direct
  reproduction: a `KeyError(None)` deep in diffusers' own GGUF code). See
  bench.md "Phase 1" for the full diagnosis. SD3.5 Medium/InstructPix2Pix
  replaced them and are measured working on this hardware (bench.md "Phase 2").
- **FP8 quantization is what makes klein usable** — 11x faster than bf16 on
  this card, because in bf16 the model does not fit and the driver spills to
  system RAM on every step. See [bench.md](bench.md).
- **Image editing is native reference conditioning** — no second model, no extra download; pass reference images to the same klein backend.
- **Video prompts are normalized too** — Wan2.1 wants subject + motion verb + camera move + lighting in 40–120 words; the normalizer strips tag salad, warns on missing motion and on dialogue/text-in-frame, and the default negative prompt applies unless overridden. `medium-480p` (~4 s) added; contact sheet is 5 tiles.
- Generated files land in `~/LocalImageGen/` by default.

## Layout

```
app/mcp_server.py   MCP adapter (no torch)
app/launcher.py     worker discovery + auto-spawn
app/worker.py       GPU-owning process
app/main.py         FastAPI routes
app/manager.py      VRAM arbiter
app/service.py      modality orchestration
app/backends/       image_flux (klein), svg (author+trace), enhance, llm, video_wan
app/svgtool.py      SVG sanitize + render-validate
main.py             CLI
scripts/bench.py    performance measurement
tests/              51 tests, no GPU required
```
