"""Phase 0 benchmark: measure what actually happens on this box before refactoring.

Answers three questions the architecture depends on:
  1. Does torchao FP8 load on sm_89 / torch 2.13 / Windows?
  2. What is peak VRAM with the text_encoder quantized too (not just the transformer)?
  3. Is a sync MCP image call viable (<= ~90 s), or must images go through the job queue?

Usage:
    python scripts/bench.py --mode fp8   --runs 2
    python scripts/bench.py --mode bf16  --runs 2
"""

from __future__ import annotations

import argparse
import gc
import json
import time
from pathlib import Path

import torch

MODEL = "black-forest-labs/FLUX.2-klein-4B"
MB = 1024 * 1024


def vram() -> tuple[float, float]:
    """(free_mb, total_mb) as the driver sees it -- includes non-torch allocations."""
    free, total = torch.cuda.mem_get_info()
    return free / MB, total / MB


def peak_mb() -> float:
    return torch.cuda.max_memory_allocated() / MB


def reserved_mb() -> float:
    return torch.cuda.max_memory_reserved() / MB


def build(mode: str):
    """Returns (pipe, quantization_label, notes)."""
    from diffusers import Flux2KleinPipeline

    notes: list[str] = []
    kwargs = {"torch_dtype": torch.bfloat16}
    label = "bf16"

    if mode == "fp8":
        # String quant_type is gone in diffusers 0.39 / transformers 5.15 -- torchao
        # AOBaseConfig objects are required, and each component needs the config class
        # from ITS OWN library (transformer=diffusers, text_encoder=transformers).
        from diffusers import PipelineQuantizationConfig
        from diffusers import TorchAoConfig as DiffusersTorchAo
        from transformers import TorchAoConfig as TransformersTorchAo
        from torchao.quantization import Float8DynamicActivationFloat8WeightConfig

        kwargs["quantization_config"] = PipelineQuantizationConfig(
            quant_mapping={
                "transformer": DiffusersTorchAo(Float8DynamicActivationFloat8WeightConfig()),
                "text_encoder": TransformersTorchAo(Float8DynamicActivationFloat8WeightConfig()),
            }
        )
        label = "fp8"
        notes.append("quantized transformer AND text_encoder")

    t0 = time.perf_counter()
    pipe = Flux2KleinPipeline.from_pretrained(MODEL, **kwargs)
    load_s = time.perf_counter() - t0

    # NOTE: never pass a pipeline-level device_map here -- enable_model_cpu_offload()
    # raises ValueError when _is_pipeline_device_mapped() is true.
    pipe.enable_model_cpu_offload()
    pipe.enable_attention_slicing()

    return pipe, label, load_s, notes


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["fp8", "bf16"], default="fp8")
    ap.add_argument("--runs", type=int, default=2)
    ap.add_argument("--steps", type=int, default=4)
    ap.add_argument("--size", default="1024x1024")
    ap.add_argument("--out", default="bench_results")
    args = ap.parse_args()

    w, h = (int(x) for x in args.size.lower().split("x"))
    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)

    torch.cuda.reset_peak_memory_stats()
    free0, total = vram()
    print(f"[gpu] {torch.cuda.get_device_name(0)}  sm{''.join(map(str, torch.cuda.get_device_capability()))}")
    print(f"[gpu] free {free0:.0f} / {total:.0f} MiB before load")
    print(f"[run] mode={args.mode} size={w}x{h} steps={args.steps} runs={args.runs}\n")

    result: dict = {
        "mode": args.mode, "size": f"{w}x{h}", "steps": args.steps,
        "gpu": torch.cuda.get_device_name(0),
        "vram_total_mb": round(total), "vram_free_before_mb": round(free0),
    }

    try:
        pipe, label, load_s, notes = build(args.mode)
    except Exception as e:
        print(f"[FAIL] {args.mode} load: {type(e).__name__}: {e}")
        result["load_error"] = f"{type(e).__name__}: {e}"
        (outdir / f"{args.mode}.json").write_text(json.dumps(result, indent=2))
        return

    result.update(quantization=label, load_s=round(load_s, 1), notes=notes,
                  peak_after_load_mb=round(peak_mb()))
    print(f"[load] {load_s:.1f}s  quantization={label}  {'; '.join(notes)}")
    print(f"[load] torch peak {peak_mb():.0f} MiB, free now {vram()[0]:.0f} MiB\n")

    times = []
    for i in range(args.runs):
        torch.cuda.reset_peak_memory_stats()
        t0 = time.perf_counter()
        img = pipe(
            prompt="a lighthouse on a cliff at dusk, long exposure, moody sea",
            height=h, width=w,
            num_inference_steps=args.steps,
            generator=torch.Generator(device="cuda").manual_seed(42 + i),
        ).images[0]
        dt = time.perf_counter() - t0
        times.append(dt)
        p, r, f = peak_mb(), reserved_mb(), vram()[0]
        img.save(outdir / f"{args.mode}_{i}.png")
        tag = "first/cold" if i == 0 else "warm"
        print(f"[gen {i}] {dt:6.1f}s  peak {p:5.0f} MiB  reserved {r:5.0f} MiB  free {f:5.0f} MiB  ({tag})")
        result.setdefault("runs", []).append(
            {"i": i, "seconds": round(dt, 1), "peak_mb": round(p),
             "reserved_mb": round(r), "free_after_mb": round(f)}
        )

    warm = times[1:] or times
    result["warm_avg_s"] = round(sum(warm) / len(warm), 1)
    result["first_s"] = round(times[0], 1)
    print(f"\n[result] first={times[0]:.1f}s  warm avg={result['warm_avg_s']:.1f}s  quant={label}")

    (outdir / f"{args.mode}.json").write_text(json.dumps(result, indent=2))
    print(f"[saved] {outdir / (args.mode + '.json')}")

    del pipe
    gc.collect(); torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
