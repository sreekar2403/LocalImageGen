"""
Batch image generation script for portfolio project screenshots.
Uses the same FLUX pipeline as vision_model.py but runs non-interactively.

Usage: python batch_generate.py
"""

import os
import gc
import torch
from diffusers import Flux2KleinPipeline
from config import vision_model
from nlp_model import generate_prompt

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

device = "cuda"

# Project descriptions for image generation
PROJECTS = [
    {
        "name": "friday-multi-agentic",
        "prompt": "A futuristic AI assistant interface on a dark monitor screen, showing multiple AI agents working together, holographic workflow diagrams, neural network visualizations, Python code snippets floating in the air, cyberpunk aesthetic with blue and purple neon glow, cinematic lighting, ultra detailed, 4K quality",
    },
    {
        "name": "local-image-gen",
        "prompt": "A creative AI workstation showing local image generation pipeline, split screen with prompt text on left transforming into a beautiful generated image on right, FLUX model architecture visualization, GPU memory usage graphs, dark developer environment with green terminal accents, cinematic product render, octane render style",
    },
]


def generate_batch():
    print("Loading FLUX model...")
    pipe = Flux2KleinPipeline.from_pretrained(
        vision_model,
        torch_dtype=torch.bfloat16,
    )
    pipe.enable_model_cpu_offload()
    pipe.enable_attention_slicing()

    output_dir = os.path.join(os.path.dirname(__file__), "images")
    os.makedirs(output_dir, exist_ok=True)

    for project in PROJECTS:
        print(f"\n{'='*50}")
        print(f"Generating: {project['name']}")
        print(f"{'='*50}")

        # Use the pre-written prompt directly (skip Ollama for speed)
        prompt = project["prompt"]
        print(f"Prompt: {prompt[:100]}...")

        generator = torch.Generator(device=device).manual_seed(42)
        try:
            image = pipe(
                prompt=prompt,
                height=768,
                width=1024,
                num_inference_steps=12,
                generator=generator,
            ).images[0]

            output_path = os.path.join(output_dir, f"{project['name']}.png")
            image.save(output_path)
            print(f"Saved: {output_path}")

        except Exception as e:
            print(f"Error generating {project['name']}: {e}")
            continue

    # Cleanup
    pipe.to("cpu")
    del pipe
    gc.collect()
    torch.cuda.empty_cache()

    print(f"\n{'='*50}")
    print("All images generated!")
    print(f"Output directory: {output_dir}")
    print(f"{'='*50}")


if __name__ == "__main__":
    generate_batch()
