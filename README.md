# LocalGen

Local **image** and **SVG** generation on your own GPU, reachable from any MCP
harness — Claude Code, Claude Desktop, OpenCode, pi, deepseek. Fully offline
after the model downloads.

Built and measured on an RTX 4060 Laptop (8 GB VRAM).

| | |
|---|---|
| Image (generate) | Stable Diffusion 3.5 Medium, T5 encoder dropped, bf16. **Measured:** 24.8 s at 768x768/20 steps, 4960 MiB peak |
| Image (edit) | InstructPix2Pix, fp16. Real end-to-end, no quantization needed |
| Image (fallback) | **10.4 s** at 1024x1024, FLUX.2-klein-4B, FP8, 5001 MiB peak — still registered as `image.flux2-klein`, just no longer the default |
| SVG (author) | 25–90 s, LLM-written paths, sanitized and render-validated |
| SVG (trace) | ~8 s, FLUX raster then vectorized — best for artwork |
| Video | not built (see AGENTS.md) |

FLUX.2-klein-4B was the default until it became clear its permanently-disabled
classifier-free guidance gave no real dial for prompt adherence (see "Notes
worth knowing"). FLUX.1-dev/Kontext-dev via GGUF quantization was tried next
and abandoned — `enable_sequential_cpu_offload()` turned out to be
fundamentally incompatible with GGUF-quantized weights in diffusers 0.39.0
(see bench.md "Phase 1"). Stable Diffusion 3.5 Medium (generation) and
InstructPix2Pix (editing) replaced them: both are plain `from_pretrained()` +
`enable_model_cpu_offload()`, no GGUF, no quantization ladder, and both are
now measured working on this exact hardware (bench.md "Phase 2") — the
simpler, more reliable path. Trade-off: somewhat less fine-grained prompt
adherence than FLUX would have given, and InstructPix2Pix's edit quality is
noticeably rougher than a modern instruction-editor (it's a 2023,
SD1.5-based model). Licensing is also simpler than FLUX.1's outright
non-commercial ban: SD3.5 Medium is under the Stability Community License
(free including commercial use under $1M annual revenue), and
InstructPix2Pix is MIT.

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
reclaims VRAM from LM Studio, which otherwise holds ~5.7 GB. There are now
three image backends sharing that one slot — `image.sd3.5-medium`
(generation), `image.instruct-pix2pix` (editing), and `image.flux2-klein`
(fast fallback) — routed to automatically based on whether reference images
are supplied.

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

- **`guidance_scale`/`negative_prompt` are real on `/v1/image`; `guidance_scale`
  and `image_guidance_scale` are real on `/v1/image/edit`.** SD3.5 Medium and
  InstructPix2Pix are not distilled, so classifier-free guidance genuinely
  applies -- unlike the old default, FLUX.2-klein-4B, which had it
  permanently disabled (`is_distilled: true`). InstructPix2Pix has dual
  guidance by design: `image_guidance_scale` controls how closely the edit
  preserves the original image, on top of the usual text-instruction
  guidance. The legacy `/generate` endpoint's frozen field set is unchanged,
  but now actually applies the values it was previously accepting-and-ignoring.
- **FLUX.1-dev/Kontext-dev via GGUF was tried and abandoned** —
  `enable_sequential_cpu_offload()` is fundamentally incompatible with
  GGUF-quantized weights in diffusers 0.39.0 (confirmed by direct
  reproduction: a `KeyError(None)` deep in diffusers' own GGUF code). See
  bench.md "Phase 1" for the full diagnosis. SD3.5 Medium/InstructPix2Pix
  replaced them and are measured working on this hardware (bench.md "Phase 2").
- **FP8 quantization is what makes klein usable** — 11x faster than bf16 on
  this card, because in bf16 the model does not fit and the driver spills to
  system RAM on every step. See [bench.md](bench.md). SD3.5 Medium/
  InstructPix2Pix need no quantization at all: SD3.5 drops its optional
  T5-XXL text encoder entirely (a documented low-VRAM pattern), and
  InstructPix2Pix is small enough (SD1.5-based) to just fit in fp16.
- **Image editing needs a second model now** — FLUX.2-klein took reference
  images natively at no extra cost; InstructPix2Pix is a separate ~4GB
  download, purpose-built for instruction-based editing.
- **InstructPix2Pix is for in-place edits, not scene replacement.** Confirmed
  via controlled testing: "make it nighttime" / "turn it into black and
  white" on the same image both came out clean, but "turn this indoor desk
  scene into a sunset over the ocean" reproducibly produces a hard visible
  seam (the model can't reconcile preserving the original composition with a
  completely different requested scene). Ask for lighting/color/style/small
  add-remove edits, not full scene swaps. See bench.md "Phase 2" for the full
  A/B test.
- **Licensing is simpler than the FLUX.1 path would have been** — SD3.5
  Medium is under the Stability Community License (free including commercial
  use under $1M annual revenue) and InstructPix2Pix is MIT, versus FLUX.1-dev/
  Kontext-dev's outright non-commercial ban.
- Generated files land in `~/LocalImageGen/` by default.

## Layout

```
app/mcp_server.py   MCP adapter (no torch)
app/launcher.py     worker discovery + auto-spawn
app/worker.py       GPU-owning process
app/main.py         FastAPI routes
app/manager.py      VRAM arbiter
app/service.py      modality orchestration
app/backends/       image_flux (klein), image_sd3, image_pix2pix, svg, enhance, llm
app/svgtool.py      SVG sanitize + render-validate
main.py             CLI
scripts/bench.py    performance measurement
tests/              51 tests, no GPU required
```
