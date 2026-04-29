import os
import gc
import torch
from diffusers import Flux2KleinPipeline
from config import vision_model
from nlp_model import generate_prompt

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

device = "cuda"

def generate_image():
    prompt = generate_prompt(input("Enter prompt: "))
    print("\nPrompt Generated...")
    print(f"\n\nPrompt: {prompt}\n\n")
    image_file_name = input("Enter image file name: ").replace(" ", "-")
    image_file_name = image_file_name + (".png" if not image_file_name.endswith((".png", ".jpg", ".jpeg")) else "")
    print("Generating Image...")

    pipe = Flux2KleinPipeline.from_pretrained(
        vision_model,
        torch_dtype=torch.bfloat16,
    )
    pipe.enable_model_cpu_offload()

    # supported optimization
    pipe.enable_attention_slicing()

    generator = torch.Generator(device=device).manual_seed(0)
    try:
        image = pipe(
            prompt=prompt,
            height=768,
            width=1024,
            num_inference_steps=12,
            generator=generator
        ).images[0]
        print("Image Generated...")
        image.save(f"./images/{image_file_name}")
        image.show()
        print(f"Image saved as /images/{image_file_name}")
    finally:
        # Cleanup: free GPU memory
        # Move model back to CPU before deleting to ensure all tensors are released
        pipe.to("cpu")
        del pipe
        
        gc.collect()
        torch.cuda.empty_cache()



# Run the image generation only when this script is executed directly
if __name__ == "__main__":
    generate_image()