"""
Single image generation for LocalImageGen project screenshot.
"""

import os
import gc
import torch
from diffusers import Flux2KleinPipeline
from config import vision_model

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

device = "cuda"

prompt = "A mesmerizing 3D collectible figurine of an AI engineer as a cosmic weaver, floating cross-legged in a nebula void, wearing a flowing robe made of woven light and stardust. Eight translucent arms extend in a mandala pattern, each hand manipulating threads of quantum code that stitch together floating dimensional portals — each portal shows a different project universe (web app, ML model, cloud infra, mobile app). The threads glow in shifting aurora colors. The figure's calm face has subtle circuit patterns under the skin like constellations. The base is a crystalline dodecahedron with a miniature galaxy inside.. Deep cosmic purple, nebula pink, aurora teal, starlight white. Ethereal volumetric lighting, octane render, 8K, mystical sci-fi collectible aesthetic."

def generate():
    print("Loading FLUX model...")
    pipe = Flux2KleinPipeline.from_pretrained(
        vision_model,
        torch_dtype=torch.bfloat16,
    )
    pipe.enable_model_cpu_offload()
    pipe.enable_attention_slicing()

    output_dir = os.path.join(os.path.dirname(__file__), "images")
    os.makedirs(output_dir, exist_ok=True)

    print(f"Prompt: {prompt[:100]}...")

    generator = torch.Generator(device=device).manual_seed(42)
    try:
        image = pipe(
            prompt=prompt,
            height=768,
            width=768,
            num_inference_steps=12,
            generator=generator,
        ).images[0]

        output_path = os.path.join(output_dir, "image2.png")
        image.save(output_path)
        print(f"Saved: {output_path}")

    except Exception as e:
        print(f"Error: {e}")
    finally:
        pipe.to("cpu")
        del pipe
        gc.collect()
        torch.cuda.empty_cache()


if __name__ == "__main__":
    generate()
