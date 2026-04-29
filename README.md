# LocalImageGen

Local AI image generation pipeline that uses Ollama to convert simple ideas into optimized prompts for local diffusion models like FLUX and Stable Diffusion. Runs fully offline, privacy-focused, customizable, and designed for fast high-quality image generation without cloud APIs.

## Features

- **Prompt Enhancement**: Leverages local LLMs via Ollama to expand simple ideas into highly detailed, cinematic prompts optimized for diffusion models.
- **Local Image Generation**: Uses `diffusers` to run powerful models like FLUX locally.
- **Memory Optimized**: Dynamically offloads the LLM and the Diffusion model between CPU and GPU memory to prevent Out of Memory (OOM) errors, allowing it to run on consumer hardware (e.g., 8GB VRAM).
- **Fully Offline**: No internet connection required after initial model downloads.

## Prerequisites

1. **Python 3.10+**
2. **Ollama**: You must have [Ollama](https://ollama.com/) installed and running on your system.
3. Download the necessary Ollama model used in your `config.py` (e.g., `gemma4:e4b`).
   ```bash
   ollama pull gemma4:e4b
   ```

## Installation

1. Clone or download this repository.
2. Install the required Python dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Configuration

Edit `config.py` to change the models used for generation:

```python
vision_model = "black-forest-labs/FLUX.2-klein-4B"
ollama_model = "gemma4:e4b" # Make sure this model is downloaded in Ollama
```

## Usage

Run the main script to generate an image:

```bash
python vision_model.py
```

1. Enter your simple prompt when prompted.
2. Enter the desired output image file name.
3. Choose the style of prompt enhancement (e.g., FLUX, SDXL, Midjourney).
4. The script will use Ollama to generate a detailed prompt, offload Ollama to free up VRAM, and then generate the image using the specified vision model.
5. The generated image will be saved in the `./images/` directory and opened automatically.
