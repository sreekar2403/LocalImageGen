"""Optional user overrides.

`app/config.py` reads these defensively -- deleting this file is safe. Environment
variables (LOCALIMAGEGEN_IMAGE_MODEL / LOCALIMAGEGEN_OLLAMA_MODEL) take precedence.
"""

vision_model = "black-forest-labs/FLUX.2-klein-4B"

# Was "gemma4:e4b", which is not pulled on this machine. qwen2.5-coder:7b is
# present and is the better choice for SVG authoring (a code task).
ollama_model = "qwen2.5-coder:7b"
