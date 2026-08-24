"""Unified local-LLM client. Providers: LM Studio (default) and Ollama.

VRAM arbitration is the whole problem here. Measured on this box:

    qwen3.5-4b loaded in LM Studio  ->  1142 MiB free  (LM Studio holds ~5.7 GB)
    FLUX.2-klein peak               ->  ~5000 MiB needed

They cannot co-reside. LM Studio applies its own idle TTL (3 min by default),
but waiting that out before every image would be absurd, so `release_gpu()`
unloads it explicitly and the ModelManager calls that before any GPU lease.

Ollama is handled differently: it is told `num_gpu=0` per request, so it never
takes VRAM in the first place and needs no arbitration.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import time

import httpx

from app.config import (
    LLM_MODEL,
    LLM_PROVIDER,
    LMSTUDIO_URL,
    OLLAMA_KEEP_ALIVE,
    OLLAMA_NUM_GPU,
)


class LLMError(RuntimeError):
    pass


# --- LM Studio ----------------------------------------------------------------


def _lms_binary() -> str | None:
    found = shutil.which("lms")
    if found:
        return found
    candidate = os.path.expanduser("~/.lmstudio/bin/lms.exe")
    if os.path.isfile(candidate):
        return candidate
    candidate = os.path.expanduser("~/.lmstudio/bin/lms")
    return candidate if os.path.isfile(candidate) else None


def lmstudio_loaded_mb() -> float:
    """Rough VRAM held by LM Studio, via `lms ps`. 0.0 when nothing is loaded."""
    binary = _lms_binary()
    if not binary:
        return 0.0
    try:
        out = subprocess.run(
            [binary, "ps"], capture_output=True, text=True, timeout=15
        ).stdout
    except Exception:  # noqa: BLE001
        return 0.0
    total = 0.0
    for line in out.splitlines():
        if " GB" in line:
            for token in line.split():
                try:
                    idx = line.split().index(token)
                    if line.split()[idx + 1].startswith("GB"):
                        total += float(token) * 1024
                        break
                except (ValueError, IndexError):
                    continue
    return total


def release_gpu(timeout: float = 30.0) -> bool:
    """Free VRAM held by the local LLM runtime. Returns True if it acted."""
    if LLM_PROVIDER != "lmstudio":
        return False  # Ollama runs CPU-only; nothing to release
    binary = _lms_binary()
    if not binary:
        return False
    try:
        subprocess.run(
            [binary, "unload", "--all"], capture_output=True, text=True, timeout=timeout
        )
        return True
    except Exception:  # noqa: BLE001 - best effort, never block generation
        return False


def _lmstudio_chat(system: str, user: str, model: str, temperature: float, num_predict: int | None):
    payload: dict[str, object] = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "temperature": temperature,
        # Auto-unload after idle so the GPU comes back without manual action.
        "ttl": 300,
    }
    if num_predict:
        payload["max_tokens"] = num_predict

    try:
        resp = httpx.post(
            f"{LMSTUDIO_URL}/v1/chat/completions", json=payload, timeout=900.0
        )
    except Exception as exc:  # noqa: BLE001
        raise LLMError(
            f"LM Studio unreachable at {LMSTUDIO_URL}: {type(exc).__name__}: {exc}. "
            "Is the LM Studio local server running?"
        ) from exc

    if resp.status_code >= 400:
        raise LLMError(f"LM Studio returned {resp.status_code}: {resp.text[:300]}")

    data = resp.json()
    message = data["choices"][0]["message"]
    # qwen3.5-4b is a reasoning model: the answer is in `content`, the chain of
    # thought in `reasoning_content`. Only `content` is wanted.
    content = message.get("content") or ""
    if not content.strip():
        raise LLMError(
            "LM Studio returned empty content "
            f"(finish_reason={data['choices'][0].get('finish_reason')}). "
            "For a reasoning model, raise max_tokens so it can finish thinking."
        )
    return content


def _ollama_chat(system: str, user: str, model: str, temperature: float, num_predict: int | None):
    import ollama

    options: dict[str, object] = {"num_gpu": OLLAMA_NUM_GPU, "temperature": temperature}
    if num_predict:
        options["num_predict"] = num_predict
    try:
        resp = ollama.chat(
            model=model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            options=options,
            keep_alive=OLLAMA_KEEP_ALIVE,
        )
    except Exception as exc:  # noqa: BLE001
        raise LLMError(f"Ollama call failed for {model!r}: {type(exc).__name__}: {exc}") from exc

    message = getattr(resp, "message", None)
    content = (
        getattr(message, "content", "")
        if message is not None
        else (resp.get("message") or {}).get("content", "")
    )
    if not content.strip():
        raise LLMError(f"Ollama returned an empty response for {model!r}")
    return content


def chat(
    system: str,
    user: str,
    model: str | None = None,
    temperature: float = 0.7,
    num_predict: int | None = None,
    provider: str | None = None,
) -> tuple[str, float]:
    """Return (content, elapsed_seconds)."""
    provider = (provider or LLM_PROVIDER).lower()
    model = model or LLM_MODEL

    start = time.perf_counter()
    if provider == "lmstudio":
        content = _lmstudio_chat(system, user, model, temperature, num_predict)
    elif provider == "ollama":
        content = _ollama_chat(system, user, model, temperature, num_predict)
    else:
        raise LLMError(f"unknown LLM provider {provider!r}; use 'lmstudio' or 'ollama'")
    return content, time.perf_counter() - start
