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

Both stay in the API as accepted-but-ignored fields that emit a warning. Failing loudly beats lying.

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
