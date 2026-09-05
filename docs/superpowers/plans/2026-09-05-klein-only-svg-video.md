# Klein-only + prompt auto-match + SVG/Wan upgrades Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make FLUX.2-klein-4B the single image backend with automatic prompt normalization, and substantially upgrade SVG authoring/tracing and Wan video quality/adherence/length.

**Architecture:** Delete SD3/pix2pix/Qwen backends and their config surface; add a pure deterministic `normalize_klein_prompt()` called unconditionally in `service.generate_image`; flip LLM enhance default to `flux2`; fix schemas/manager/capabilities to klein reality; upgrade SVG few-shot + vtracer auto-tune + validation; add video prompt rules + medium preset + 5-tile sheet.

**Tech Stack:** Python 3, diffusers 0.39.0 Flux2KleinPipeline FP8 via torchao, FastAPI, Pydantic v2, vtracer, resvg-py, Pillow, Wan2.1-T2V-1.3B, pytest CPU-only.

## Global Constraints

- FLUX.2-klein-4B is the ONLY image backend (`image.flux2-klein`, 4 steps, FP8, no CFG, `guidance_scale`/`negative_prompt` accepted-but-ignored with warning).
- `app/mcp_server.py` must never import torch (thin httpx client).
- All GPU work funnels through `ModelManager` single GPU thread; one resident backend.
- Tests are CPU-only: `python -m pytest tests -q` must pass without GPU.
- MCP tools return file paths, never base64 (SVG source inline only under 12k budget).
- YAGNI: no new backends, no GGUF, no compat flags for deleted models.

---

### Task 1: Klein-only deletion

**Files:**
- Delete: `app/backends/image_sd3.py`, `app/backends/image_pix2pix.py`, `app/backends/image_qwen.py`
- Modify: `app/manager.py:219-229`, `app/config.py:32-72`, `app/schemas.py:85-147`, `app/service.py:59-64`, `app/main.py:70-81`
- Test: `tests/test_legacy_contract.py`, `tests/test_manager.py`

**Interfaces:**
- Consumes: nothing new.
- Produces: `get_manager()` registers only `ImageFluxBackend + VideoWanBackend + Enhance/SvgAuthor/SvgTrace`; `service.generate_image` always uses `"image.flux2-klein"`.

- [ ] **Step 1: Delete dead backend files**

```bash
Remove-Item -LiteralPath "app/backends/image_sd3.py", "app/backends/image_pix2pix.py", "app/backends/image_qwen.py"
```

- [ ] **Step 2: manager.py registers klein only**

```python
from app.backends.image_flux import ImageFluxBackend
mgr.register(ImageFluxBackend())
# SD3/pix2pix/Qwen registrations removed
```

- [ ] **Step 3: config.py remove SD3/pix2pix/Qwen/FLUX1 surface**

Delete `_DEFAULT_SD3_MODEL`, `_DEFAULT_PIX2PIX_MODEL`, `SD3_MODEL`, `PIX2PIX_MODEL`, `DEFAULT_STEPS_SD3`, `DEFAULT_GUIDANCE_SCALE_SD3`, `DEFAULT_STEPS_PIX2PIX`, `DEFAULT_GUIDANCE_SCALE_PIX2PIX`, `DEFAULT_IMAGE_GUIDANCE_SCALE_PIX2PIX`, `DEFAULT_STEPS_QWEN`, `DEFAULT_TRUE_CFG_QWEN`, `FLUX1_MIN_RESIDENCY_S`, `FLUX1_IDLE_EVICT_S` and the Qwen-replaces-SD3 comment block. Keep `MODEL_NAME`, `DEFAULT_STEPS=4`, `DEFAULT_GUIDANCE_SCALE=3.5` (legacy echo only).

- [ ] **Step 4: schemas.py klein reality**

```python
class ImageRequest(BaseModel):
    steps: int = Field(4, ge=1, le=12, description="FLUX.2-klein distilled; 4 steps is the tuned default.")
    guidance_scale: Optional[float] = Field(None, description="Ignored on klein (no CFG); accepted for compat, warns.")
    negative_prompt: Optional[str] = Field(None, description="Ignored on klein (no CFG); accepted for compat, warns.")
```

Same for `EditRequest` (steps 4, `image_guidance_scale` ignored-with-warning). Remove `DEFAULT_STEPS_QWEN` import.

- [ ] **Step 5: service.py collapse routing**

```python
kind = "edit" if reference_images else "image"
out_path = resolve_output_path(path, kind)
backend_name = "image.flux2-klein"  # single backend
```

- [ ] **Step 6: Run tests**

Run: `.\.venv\Scripts\python.exe -m pytest tests/test_manager.py tests/test_legacy_contract.py -q`
Expected: PASS (update fixtures if they reference sd3/qwen names).

- [ ] **Step 7: Commit**

```bash
git add -A
git commit -m "feat: klein-only image backends, drop SD3/pix2pix/Qwen"
```

### Task 2: Klein prompt auto-match

**Files:**
- Modify: `app/prompts.py`, `app/backends/enhance.py:18-22`, `app/service.py:44-57`, `app/mcp_server.py:78-90`
- Test: `tests/test_klein_prompt.py` (new)

**Interfaces:**
- Consumes: raw user `prompt: str`.
- Produces: `normalize_klein_prompt(prompt: str) -> tuple[str, list[str]]` returning `(clean, warnings)`; `service.generate_image` calls it unconditionally before enhance/lease.

- [ ] **Step 1: Write failing test**

```python
def test_strips_quality_tags():
    clean, warns = normalize_klein_prompt("a lighthouse, masterpiece, 8k, ultra detailed")
    assert "masterpiece" not in clean and "8k" not in clean
    assert any("tag" in w.lower() for w in warns)

def test_folds_negative_to_positive():
    clean, _ = normalize_klein_prompt("a portrait, no blur, without extra fingers")
    assert "not blurry" not in clean
    assert "sharp" in clean.lower() or "five fingers" in clean.lower()

def test_truncates_long_prompt():
    clean, warns = normalize_klein_prompt("word " * 300)
    assert len(clean.split()) <= 150
    assert warns
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.\.venv\Scripts\python.exe -m pytest tests/test_klein_prompt.py -q`
Expected: FAIL with "normalize_klein_prompt not defined"

- [ ] **Step 3: Implement normalizer in app/prompts.py**

```python
_QUALITY_TAGS = {"masterpiece","best quality","ultra detailed","8k","4k","trending on artstation","octane render","unreal engine"}
_NEGATIVE_FOLDS = [("no blur", "sharp, crisp focus"), ("not blurry", "sharp, crisp focus"), ("without extra fingers", "a single steady hand with five fingers"), ("no extra fingers", "a single steady hand with five fingers")]

def normalize_klein_prompt(prompt: str) -> tuple[str, list[str]]:
    warnings: list[str] = []
    clean = " ".join(prompt.strip().split())
    lowered = clean.lower()
    for tag in _QUALITY_TAGS:
        if tag in lowered:
            warnings.append(f"removed quality tag {tag!r} (klein resolves 4 steps, tags add noise)")
    import re
    for tag in _QUALITY_TAGS:
        clean = re.sub(re.escape(tag), "", clean, flags=re.IGNORECASE)
    for bad, good in _NEGATIVE_FOLDS:
        if bad in clean.lower():
            clean = re.sub(re.escape(bad), good, clean, flags=re.IGNORECASE)
            warnings.append(f"rewrote {bad!r} as positive {good!r} (klein has no negative prompt)")
    clean = re.sub(r"\s*,\s*", ", ", clean)
    clean = re.sub(r"\s{2,}", " ", clean).strip(" ,.")
    words = clean.split()
    if len(words) > 140:
        cut = " ".join(words[:140])
        period = cut.rfind(".")
        clean = cut[:period+1] if period > 80 else cut
        warnings.append(f"truncated {len(words)} -> {len(clean.split())} words (klein 60-140w budget)")
    return clean, warnings
```

- [ ] **Step 4: Wire into service + flip enhance default**

```python
# app/backends/enhance.py
DEFAULT_STYLE = "flux2"
_OVERLAY_VARIANTS = {"flux": "flux_overlay", "flux2": "flux2_overlay", "flux1": "flux1_overlay", "sdxl": "sdxl_overlay", "qwen": "qwen_overlay", "midjourney": "midjourney"}
```

```python
# app/service.py top of generate_image
from app.prompts import normalize_klein_prompt
clean, norm_warns = normalize_klein_prompt(prompt)
warnings.extend(norm_warns)
prompt_used = clean
```

- [ ] **Step 5: Run tests**

Run: `.\.venv\Scripts\python.exe -m pytest tests/test_klein_prompt.py tests/test_legacy_contract.py -q`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add app/prompts.py app/backends/enhance.py app/service.py tests/test_klein_prompt.py
git commit -m "feat: klein prompt auto-normalize, flux2 enhance default"
```

### Task 3: SVG upgrades (author + trace + validation)

**Files:**
- Modify: `app/prompts.py:323-398`, `app/backends/svg.py`, `app/svgtool.py:156-158`
- Test: `tests/test_svgtool.py`, `tests/test_svg_klein.py` (new, CPU-only with fake chat/vtracer)

**Interfaces:**
- Consumes: `SVG_PROMPTS[kind]`, `chat()`, `vtracer.convert_image_to_svg_py`, `sanitize/rasterize`.
- Produces: same `Artifact` shape; `meta` gains `path_count`, `bytes`, `colors`, `simplified` notes.

- [ ] **Step 1: Author prompts — 3 few-shots + budgets**

Append two more worked examples (logo wordmark, diagram boxes+arrows) after `_SVG_EXAMPLE`; change author `temperature 0.4 -> 0.3`, `max_repairs 2 -> 3`; repair prompt gains: "Keep under 80 paths and 15KB. Prefer real <path> geometry over rects."

- [ ] **Step 2: Trace auto-tune + simplify**

```python
kind_hint = params.get("svg_kind") or "icon"
default_colors = {"icon": 8, "logo": 8, "diagram": 10, "chart": 10, "illustration": 14}.get(kind_hint, 12)
colors = int(params.get("colors") or default_colors)
# after vtracer: drop tiny paths, round precision
import re
clean = re.sub(r"M-?\d+\.\d{3,}", lambda m: m.group(0)[:m.group(0).find(".")+3], clean)
```

Try `mode="spline"` then fallback `mode="polygon"` if `path_count > 400` or bytes > 100_000; record `warnings` + `meta["simplified"]`.

- [ ] **Step 3: Fix kinds routing**

```python
class SvgTraceBackend:
    kinds = ("svg",)  # was ("svg_trace",) which for_kind("svg") never matched
```

`SvgAuthorBackend` keeps `("svg",)`; service routes by explicit `mode` so both coexist — document that `for_kind` returns author first.

- [ ] **Step 4: Tests**

Add CPU-only tests: sanitize keeps viewBox synthesis, `path_count` counts, trace simplify regex reduces precision, `kinds` fix. Run `pytest tests/test_svgtool.py -q` PASS.

- [ ] **Step 5: Commit**

```bash
git add app/prompts.py app/backends/svg.py app/svgtool.py tests/
git commit -m "feat: svg author few-shot, trace auto-tune, kinds fix"
```

### Task 4: Wan video upgrades

**Files:**
- Modify: `app/prompts.py` (add `VIDEO_PROMPT_RULES` + `normalize_video_prompt`), `app/backends/video_wan.py`, `app/config.py:VIDEO_PRESETS`, `app/ffmpeg.py:contact_sheet`
- Test: extend `tests/test_video.py`

**Interfaces:**
- Consumes: user `prompt`, `preset`, `VIDEO_PRESETS`.
- Produces: same `Artifact`; `meta` gains `prompt_warnings`, `probe` info; new preset `medium-480p`.

- [ ] **Step 1: Video prompt cleaner**

```python
def normalize_video_prompt(prompt: str) -> tuple[str, list[str]]:
    # strip quality tags, require motion verb + camera move, cap ~120 words, warn on dialogue/text-in-frame
```

Wire unconditionally in `VideoWanBackend.generate` before `encode_prompts_on_cpu`; always send `DEFAULT_NEGATIVE` unless user overrides; warn when tokenizer truncates past 226 tokens.

- [ ] **Step 2: Quality defaults**

`short-480p steps 20 -> 25`; add `medium-480p: {832x480, 65 frames, 25 steps, 16fps}`; `contact_sheet` picks 5 frames (0/25/50/75/100%); add `probe()` output to `meta`; warn if output <10KB/frame; check `is_cancelled` before Stage C encode.

- [ ] **Step 3: Tests**

```python
def test_video_prompt_strips_tags():
    clean, warns = normalize_video_prompt("a cat, masterpiece 8k, panning shot")
    assert "masterpiece" not in clean and warns

def test_medium_preset_valid():
    cfg = VIDEO_PRESETS["medium-480p"]
    assert (cfg["num_frames"]-1) % 4 == 0 and cfg["width"] % 16 == 0
```

Run `pytest tests/test_video.py -q` PASS.

- [ ] **Step 4: Commit**

```bash
git add app/prompts.py app/backends/video_wan.py app/config.py app/ffmpeg.py tests/test_video.py
git commit -m "feat: wan prompt adherence, medium preset, 5-tile sheet"
```

### Task 5: Docs + verification

**Files:**
- Modify: `README.md`, `AGENTS.md`, `bench.md`, `app/main.py:description/capabilities`, `app/mcp_server.py:instructions`
- Test: full `pytest tests -q`

**Interfaces:**
- Consumes: all prior tasks.
- Produces: docs match klein-only reality; `/capabilities` and MCP instructions say 4 steps, auto-normalized prompts, guidance ignored.

- [ ] **Step 1: Rewrite docs tables to klein-only; move SD3/pix2pix/Qwen numbers to History section.**

- [ ] **Step 2: Full test run**

Run: `.\.venv\Scripts\python.exe -m pytest tests -q`
Expected: PASS, 0 GPU required.

- [ ] **Step 3: Smoke (manual, GPU box)**

`python main.py status`, one `image`, one `edit`, one `svg --kind icon`, one `tiny-480p` video job. Record timings in bench.md.

- [ ] **Step 4: Commit**

```bash
git add -A
git commit -m "docs: klein-only reality, prompt contract, svg/video notes"
```
