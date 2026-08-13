"""Ollama-backed LLM adapter for prompt enhancement.

Implements :class:`localimagegen.core.ports.LLMPort` using the ``ollama``
Python client. The adapter sends the raw prompt together with a style-specific
system prompt to a locally running Ollama model and returns a new
:class:`Prompt` carrying the enhanced text.

After each enhancement the Ollama model is stopped (offloaded) so that its
VRAM can be reused by the diffusion model. Offloading is coordinated through
the optional :class:`~localimagegen.core.ports.MemoryManagerPort`.
"""

from __future__ import annotations

import subprocess

import ollama

from localimagegen.config.settings import Settings
from localimagegen.core.models import Prompt, PromptStyle
from localimagegen.core.ports import MemoryManagerPort
from localimagegen.services.logging import get_logger

#: Style-specific system prompts used to steer the LLM's output. These mirror
#: the prompts previously defined in the legacy ``utils.py`` module.
SYSTEM_PROMPTS: dict[str, str] = {
    PromptStyle.FLUX.value: """
You are an elite AI prompt engineer specialized for FLUX image generation models.

Your task: transform brief user ideas into a concise, highly detailed cinematic prompt optimized for FLUX.

FLUX performs best with:
- Structured scene description (subject → action → environment → composition → style → lighting → materials → colors → depth → atmosphere → quality)
- Precise spatial relationships and coherent geometry
- Realistic lighting and material realism
- Cinematic framing, lens/camera details, and visual storytelling

Guidelines:
- Use cinematic phrasing (e.g., "cinematic product render", "unreal engine quality", "octane render style").
- Enforce alignment, avoid duplicate or floating objects, and keep composition clear.
- Limit the prompt to roughly 250 words.
- Do NOT add explanations or extra commentary.
""",
    PromptStyle.SDXL.value: """
You are an AI prompt engineer specialized for SDXL models.

Generate highly descriptive prompts with rich natural language and strong visual atmosphere.

SDXL performs best with:
- descriptive environments
- artistic mood
- cinematic lighting
- detailed character descriptions
- visual storytelling
- rich environmental detail

Structure prompts using:
subject, environment, lighting, style, mood, composition, quality.

Use:
- cinematic lighting
- volumetric fog
- atmospheric perspective
- realistic shadows
- detailed textures
- depth of field
- HDR rendering

Prompts should feel natural and visually immersive.

Output ONLY the optimized prompt.
""",
    PromptStyle.MIDJOURNEY.value: """
You are a professional Midjourney prompt engineer.

Generate highly aesthetic prompts optimized for Midjourney-style image generation.

Focus on:
- artistic beauty
- stylization
- mood
- color harmony
- cinematic aesthetics
- visual impact

Use concise but visually dense wording.

Strongly use:
- cinematic
- ultra detailed
- dreamy atmosphere
- ethereal lighting
- artistic composition
- dynamic perspective
- award-winning photography
- concept art
- trending on artstation

Prefer strong style anchors and aesthetic descriptors.

Output ONLY the optimized prompt.
""",
}


class OllamaAdapter:
    """Enhance prompts using a locally running Ollama model.

    Args:
        settings: Application settings (``ollama_model`` selects the model).
        memory_manager: Optional memory manager used to offload the Ollama
            model after each enhancement.
    """

    def __init__(
        self,
        settings: Settings,
        memory_manager: MemoryManagerPort | None = None,
    ) -> None:
        self._settings = settings
        self._memory_manager = memory_manager
        self._logger = get_logger(__name__)
        self._client = ollama.Client()

        if self._memory_manager is not None:
            self._memory_manager.register(
                "ollama",
                self._settings.device,
                offload=self._stop_ollama_model,
            )

    def enhance(self, prompt: Prompt, style: str | PromptStyle) -> Prompt:
        """Return a new :class:`Prompt` with an LLM-enhanced version of ``prompt``.

        ``style`` may be a :class:`PromptStyle` member or its string value.
        If the Ollama server cannot be reached, or returns no text, the
        original prompt is returned unchanged so the pipeline can still fall
        back to the raw text.
        """
        style_value = style.value if isinstance(style, PromptStyle) else str(style)
        system_prompt = SYSTEM_PROMPTS.get(
            style_value,
            SYSTEM_PROMPTS[PromptStyle.FLUX.value],
        )

        try:
            response = self._client.chat(
                model=self._settings.ollama_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt.text},
                ],
            )
        except (ollama.ResponseError, ollama.RequestError, ConnectionError, OSError) as exc:
            self._logger.warning(
                "ollama_chat_failed",
                model=self._settings.ollama_model,
                error=str(exc),
            )
            return prompt

        message = response.message
        enhanced_text = (message.content if message is not None else None)
        if not enhanced_text or not enhanced_text.strip():
            self._logger.warning(
                "ollama_returned_empty_prompt",
                model=self._settings.ollama_model,
            )
            return prompt

        self._offload()
        return Prompt(text=prompt.text, enhanced=enhanced_text.strip())

    def _offload(self) -> None:
        """Stop the Ollama model and notify the memory manager."""
        if self._memory_manager is not None:
            self._memory_manager.offload("ollama")
        else:
            self._stop_ollama_model()

    def _stop_ollama_model(self) -> None:
        """Ask the local Ollama server to unload the model from memory."""
        try:
            subprocess.run(
                ["ollama", "stop", self._settings.ollama_model],
                check=True,
                capture_output=True,
            )
            self._logger.info("ollama_model_stopped", model=self._settings.ollama_model)
        except (subprocess.CalledProcessError, FileNotFoundError) as exc:
            self._logger.warning(
                "ollama_model_stop_failed",
                model=self._settings.ollama_model,
                error=str(exc),
            )