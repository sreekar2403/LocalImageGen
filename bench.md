# Phase 0 benchmark — measured, not estimated

RTX 4060 Laptop (8188 MiB, sm89) · torch 2.13.0+cu126 · diffusers 0.39.0 · torchao 0.18.0
FLUX.2-klein-4B · 1024x1024 · 4 steps · 3 runs · `python scripts/bench.py`

## Headline

| | bf16 + cpu offload (**today's production path**) | FP8 transformer + text_encoder |
|---|---|---|
| Load | 1.4 s | 24.0 s |
| First generation | 127.9 s | **14.9 s** |
| Warm generation | **115.7 s** | **10.4 s** |
| Torch peak | 7996 MiB | **5001 MiB** |
| Torch reserved | **8654 MiB** | 5808 MiB |

**FP8 is 11.1x faster warm and cuts peak VRAM by 37%.** Output quality at 4 steps is unaffected.

## The bf16 path was thrashing, and AGENTS.md described the symptom wrong

Reserved **8654 MiB > 8188 MiB physical VRAM**. The NVIDIA 555.97 Windows driver was silently
spilling to system RAM over PCIe on every denoise step.

Root cause: `model_cpu_offload_seq = "text_encoder->transformer->vae"` moves one whole module at a
time, and in bf16 *each* of the two big ones individually exceeds the ~7.0 GB free:

| Component | Class | bf16 on disk |
|---|---|---|
| text_encoder | Qwen3ForCausalLM | **7673 MB** |
| transformer | Flux2Transformer2DModel | **7393 MB** |
| vae | AutoencoderKLFlux2 | 161 MB |

The text encoder is the *bigger* component — quantizing only the transformer would not have fixed it.

`AGENTS.md` says *"First generation is slow (~1-2 min) due to CPU offloading; subsequent calls reuse
cached weights."* The measurement disproves the second half: warm runs were 115.7 s, essentially as
slow as the first. It was never a load-time cost — it was per-step PCIe thrash.

## Dead knobs confirmed

`model_index.json` has `"is_distilled": true`, so in `pipeline_flux2_klein.py:593`
`do_classifier_free_guidance` is permanently False. diffusers printed it unprompted during the run:

```
Guidance scale 4.0 is ignored for step-wise distilled models.
```

- **`guidance_scale`** — ignored. `guidance=None` is hardcoded at :852/:867; `guidance_embeds: false`.
- **`negative_prompt`** — `negative_prompt_embeds` is read only inside `if self.do_classifier_free_guidance:`
  (:747, :862). Encoding one manually would silently do nothing.

Both were originally kept in the API as accepted-but-ignored fields that emitted
a warning. Since neither can ever do anything on this model, they were removed
from `/v1/image`, `/v1/image/edit`, and the MCP tools entirely -- they only
remain on the legacy `/generate` endpoint, which is a frozen contract.

## API note

String `quant_type` ("float8dq", "int8wo", …) is **rejected** by both diffusers 0.39 and
transformers 5.15. torchao `AOBaseConfig` objects are required, and each component needs the config
class from its own library:

```python
PipelineQuantizationConfig(quant_mapping={
    "transformer":  diffusers.TorchAoConfig(Float8DynamicActivationFloat8WeightConfig()),
    "text_encoder": transformers.TorchAoConfig(Float8DynamicActivationFloat8WeightConfig()),
})
```

Do not pass a pipeline-level `device_map` — `enable_model_cpu_offload()` raises `ValueError` when
`_is_pipeline_device_mapped()` is true (`pipeline_utils.py:1207`).

## Consequences for the architecture

1. **FP8 is required, not optional.** Promote `torchao` to a hard dependency; keep bf16 as a fallback.
2. **Images stay synchronous over MCP.** 10.4 s is far inside any tool timeout.
3. **Idle eviction can be aggressive** — reload is 24 s, not 90 s.
4. 576x1024 (reels/story) should land ~6-8 s; worth measuring when the presets are wired.

---

# SVG authoring: model comparison (measured)

Same prompt each time — *"a minimal terminal window icon, rounded square, with a
green blinking cursor"*, `svg_kind=icon`, 512px.

| Model | Runtime | Time | Result |
|---|---|---|---|
| `qwen2.5-coder:7b` | Ollama, CPU | 25–45 s | Poor — plain rects, duplicate shapes, no real geometry |
| `qwen3.5-4b` | LM Studio, GPU | 89 s | Decent — real `<path>` geometry, good palette, cursor misplaced |
| `qwen3.8:27b-mtp-q4_K_M` | Ollama, CPU | **463 s** | Best — correct chevron + cursor bar, well composed |

A few-shot worked example was added to the SVG system prompts; it helped the
larger models more than the 7B, which still tends toward plain rectangles.

`qwen3.5-4b` is the default. It is a **reasoning** model: the answer arrives in
`content` and the chain of thought in `reasoning_content`, which is why it costs
~89 s for a small icon. Only `content` is parsed.

## Trace mode is the better path for illustration

*"a friendly fox mascot sticker, front facing"*, 768px, 10 colours:
**7.7 s**, 29 paths, 28 KB, valid — and visually far better than anything the
LLM path produced. Use `mode="author"` for icons/diagrams where clean editable
geometry matters, and `mode="trace"` for artwork.

# VRAM arbitration (measured)

LM Studio is GPU-accelerated and does **not** release memory on its own quickly:

| Moment | VRAM free |
|---|---|
| `qwen3.5-4b` resident in LM Studio | **1155 MiB** |
| FLUX peak requirement | ~5001 MiB |
| After `release_gpu()` → FLUX generated | 7128 MiB free afterwards |

1155 MiB is nowhere near enough for FLUX, so `ModelManager._ensure_resident()`
runs `lms unload --all` before taking any GPU lease. Confirmed working: an image
request made while LM Studio held the GPU completed in 39.6 s total (24 s reload
+ 15.5 s generate) at 5001 MiB peak, instead of thrashing or OOM-ing.

Ollama needs no arbitration — it is pinned to CPU with `num_gpu: 0`.

---

# Phase 1 — FLUX.1-dev / FLUX.1-Kontext-dev (NOT YET MEASURED)

Everything below is a checklist, not a result. `app/backends/image_flux1.py`
was written without GPU access; nothing here has been run. Fill in real
numbers as each check is done, the same way Phase 0 was built from
`scripts/bench.py`.

**Why these models replaced klein as the default**: FLUX.2-klein-4B's
`is_distilled: true` permanently disables classifier-free guidance (see
"Dead knobs confirmed" above) — there was no dial to turn for poor prompt
adherence. FLUX.1-dev (generation) and FLUX.1-Kontext-dev (editing) are
non-distilled with genuine CFG, at the cost of ~20-30 steps instead of 4 and
a BFL non-commercial license (acceptable for this personal/local
deployment). klein-4B stays registered as `image.flux2-klein`, unchanged,
as a fallback.

## What to check, in order

1. **Do the assumed GGUF repos/filenames actually exist?** `app/config.py`
   guesses `city96/FLUX.1-dev-gguf` / `city96/FLUX.1-Kontext-dev-gguf` with
   filenames like `flux1-dev-Q4_K_S.gguf`. Browse the actual HF repos and
   correct `LOCALIMAGEGEN_FLUX1_DEV_GGUF_REPO` /
   `LOCALIMAGEGEN_FLUX1_DEV_GGUF_FILENAME` (and the Kontext equivalents) via
   env var if they differ, rather than editing code.
2. **`gguf` package compatibility — RESOLVED.** `gguf==0.10.0` (the original
   pin) calls `numpy.memmap.newbyteorder()`, which numpy removed in 2.0;
   fails with `AttributeError: 'memmap' object has no attribute
   'newbyteorder'` inside `GGUFReader.__init__` on any numpy>=2.0 environment
   (this project pulls numpy 2.x transitively). This is what "all GGUF rungs
   failed" / "Unable to load weights from checkpoint file" actually was —
   diffusers' generic `except Exception` in `model_loading_utils.py` masks
   the real underlying error with that unhelpful message. Fixed by upgrading
   to `gguf==0.19.0`; verified `load_gguf_checkpoint()` parses the cached
   `flux1-dev-Q4_K_S.gguf` (780 tensors) without error on this machine.
3. **GGUF ladder fit.** For each of `image.flux1-dev` and
   `image.flux1-kontext-dev`, attempt load at each rung in
   `FLUX1_GGUF_LADDER` (Q4_K_S → Q3_K_S → Q2_K) and record: does it load,
   `torch.cuda.max_memory_allocated()`/`reserved()`, and whether reserved
   exceeds 8188 MiB (the same spill check that caught klein's bf16 problem).
4. **Load time**, cold (incl. first HF download) vs. warm (cached files,
   dequant only) — compare against klein's 24.0 s FP8 baseline. Tune
   `FLUX1_MIN_RESIDENCY_S` (currently a 180 s placeholder) once this is known.
5. **Per-image wall-clock** at 20/25/30 steps, 1024x1024, with
   `enable_sequential_cpu_offload()` — compare against klein's 10.4 s warm
   baseline. Expect much slower; this is the accepted quality-over-speed
   tradeoff, but the actual number matters for setting expectations.
6. **`FluxKontextPipeline.__call__` signature** — does it accept a list of
   reference images, or only one? `image_flux1.py` currently assumes one and
   warns if more are supplied; update if the installed diffusers version
   supports multi-reference.
7. **`from_pretrained(transformer=..., torch_dtype=...)` interaction** — read
   the installed `diffusers/pipelines/flux/pipeline_flux.py` to confirm
   passing `torch_dtype` alongside a pre-quantized `transformer=` override
   doesn't re-cast and break the GGUF dequant state.
8. **Unload correctness** — confirm `torch.cuda.memory_allocated()` returns
   to near-zero after `FluxGGUFBackend.unload()`, since sequential-offload
   hooks differ from klein's `enable_model_cpu_offload()` hooks and the
   existing cleanup pattern is unverified against them.
9. **Quality spot-check** — same prompt through klein (baseline) vs.
   FLUX.1-dev at the winning rung: confirm the actual complaint (missing
   detail / prompt adherence) is measurably better. Same comparison for
   editing: klein's reference-conditioning vs. Kontext-dev on the same edit
   instruction.
10. **Run `python -m pytest -q`**, specifically `tests/test_legacy_contract.py`,
    after confirming the above — the legacy `/generate` endpoint now
    routes through FLUX.1-dev too (its schema is unchanged, but the values
    it returns, like `quantization`, will read `"gguf-Q4_K_S"` instead of
    `"fp8"`).

## Phase 1 outcome: ABANDONED

FLUX.1-dev/Kontext-dev via GGUF is abandoned. `enable_sequential_cpu_offload()`
is fundamentally incompatible with GGUF-quantized weights in diffusers 0.39.0
-- confirmed by direct reproduction (not just theory): accelerate's
`attach_align_device_hook` moves a `GGUFParameter` to the `meta` device during
offload-hook setup, which reconstructs the parameter and loses its
`quant_type` attribute, and diffusers' own `GGUFParameter.__new__` then
crashes with a bare `KeyError(None)` indexing `GGML_QUANT_SIZES[None]`. This
is what surfaced to the user as the unhelpful "generation failed: None" (a
`KeyError(None)`'s `str()` is literally the text "None").

Fixing it properly would require switching to `enable_model_cpu_offload()`
(klein's approach) AND quantizing the ~9GB T5-XXL text encoder to fit as a
whole module -- a materially bigger lift than initially scoped. Replaced with
Stable Diffusion 3.5 Medium (generation) + InstructPix2Pix (editing) instead
-- see "Phase 2" below.

# Phase 2 — SD3.5 Medium / InstructPix2Pix (MEASURED on this hardware)

RTX 4060 Laptop (8188 MiB) · torch 2.13.0+cu126 · diffusers 0.39.0
Reproduced directly via `app/backends/image_sd3.py` / `image_pix2pix.py`,
then confirmed again through real web-UI usage (`/health` showed
`resident_backend: "image.instruct-pix2pix"`, `swaps: 1` after a
generate-then-edit sequence).

| | SD3.5 Medium (generate) | InstructPix2Pix (edit) |
|---|---|---|
| Load | components load in ~2s once cached; first-ever HF download was ~35 min (19 files) and ~6 min (14 files) respectively over this network -- one-time cost | |
| Quantization | `bf16-no-t5` (T5-XXL dropped entirely, CLIP-L + OpenCLIP-G only) | `fp16` (no quantization needed at all) |
| Generation, 20 steps, 768x768 | 24.8 s | not separately timed; qualitatively similar |
| Peak VRAM | 4960 MiB | not separately measured, but the live server ran both back-to-back on one 8GB card without issue |

Both fit comfortably with plain `enable_model_cpu_offload()` -- no GGUF, no
quantization ladder, no exotic offload mode. This is the simple, boring,
working path the FLUX.1 attempt was trying to avoid needing.

**Quality, qualitatively**: SD3.5 Medium produced a sharp, well-composed,
prompt-faithful image on the first real test (a brass telescope on a wooden
desk, matching a detailed prompt closely).

**InstructPix2Pix quality finding (confirmed via controlled A/B testing,
2026-09-04)**: drastic full-scene transformations ("turn this indoor desk
scene into a sunset over the ocean") reproducibly produce a hard vertical
"split" artifact -- roughly 40% across the image, mismatched color grading on
each side, same location across three separate attempts with three different
prompt wordings and both with and without `vae.enable_tiling()` (ruled out as
the cause). Root cause: this is a genuine InstructPix2Pix limitation, not a
loading/config bug -- `image_guidance_scale`'s pull toward preserving the
original composition fights the text instruction's pull toward a completely
different scene, and the model produces a half-blended compositing artifact
instead of resolving the conflict.

Localized edits tested clean with **no artifact at all**: "make it nighttime"
and "turn it into a black and white photo" on the same source image both
produced coherent, well-composed results preserving the original layout.
**Practical guidance: use InstructPix2Pix for in-place edits (lighting,
color/style, day-night, add/remove a small object), not full scene
replacement.** If scene-replacement-grade editing becomes a real requirement,
a purpose-built modern instruction-editor would need re-evaluating against
this same ~7GB VRAM budget (FLUX.1-Kontext-dev is the obvious next
candidate, but see "Phase 1 outcome: ABANDONED" above for why it wasn't
usable as originally attempted here).

## If SD3.5/InstructPix2Pix ever needs revisiting

- **Generation quality ceiling**: dropping T5-XXL trades some fine-grained
  prompt adherence for VRAM headroom. If that ceiling becomes the limiting
  factor, the next thing to try is loading a quantized T5 (e.g. int8 via
  `transformers.TorchAoConfig`, the same pattern klein's text encoder already
  uses) instead of dropping it entirely -- more setup, real prompt-adherence
  upside.
- **Editing quality ceiling**: InstructPix2Pix is a 2023, SD1.5-based model.
  If its edit quality stops being good enough, re-evaluate a newer
  instruction-editor against this same ~7GB budget rather than assuming
  FLUX.1-Kontext-dev is the only option -- the GGUF-loading problem above is
  specific to how *that* model was being loaded here, not an argument against
  all bigger editors in general.
- klein-4B (`image.flux2-klein`) remains registered and untouched as a
  fast/cheap fallback if either new backend ever needs bypassing.
