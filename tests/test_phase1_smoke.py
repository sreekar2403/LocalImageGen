"""Smoke tests for Phase 1 foundation — config, models, ports, logging."""

from pathlib import Path

import pytest


def test_settings_defaults():
    """Settings load with sensible defaults."""
    from localimagegen.config.settings import Settings

    s = Settings()
    assert s.vision_model == "black-forest-labs/FLUX.2-klein-4B"
    assert s.ollama_model == "gemma4:e4b"
    assert s.width == 1024
    assert s.height == 768
    assert s.num_inference_steps == 12
    assert s.seed == 42
    assert s.device == "cuda"


def test_settings_validation():
    """Settings reject invalid values."""
    from pydantic import ValidationError

    from localimagegen.config.settings import Settings

    with pytest.raises(ValidationError):
        Settings(device="tpu")

    with pytest.raises(ValidationError):
        Settings(log_level="VERBOSE")


def test_settings_yaml_loading(tmp_path: Path):
    """Settings load from YAML file."""
    from localimagegen.config.settings import Settings

    config = tmp_path / "test.yaml"
    config.write_text("width: 512\nheight: 512\nseed: 99\n")

    s = Settings.from_yaml(config)
    assert s.width == 512
    assert s.height == 512
    assert s.seed == 99
    assert s.vision_model == "black-forest-labs/FLUX.2-klein-4B"  # default preserved


def test_settings_cli_overrides(tmp_path: Path):
    """CLI overrides take highest precedence."""
    from localimagegen.config.settings import Settings

    config = tmp_path / "test.yaml"
    config.write_text("width: 512\n")

    s = Settings.from_cli_args({"width": 2048}, config_file=config)
    assert s.width == 2048  # CLI override wins


def test_prompt_effective():
    """Prompt.effective returns enhanced text when available."""
    from localimagegen.core.models import Prompt

    p = Prompt(text="a cat")
    assert p.effective == "a cat"

    p2 = Prompt(text="a cat", enhanced="a majestic tabby cat in a garden")
    assert p2.effective == "a majestic tabby cat in a garden"


def test_generation_params_validation():
    """GenerationParams rejects out-of-range values."""
    from pydantic import ValidationError

    from localimagegen.core.models import GenerationParams

    with pytest.raises(ValidationError):
        GenerationParams(width=10)  # below min 64

    with pytest.raises(ValidationError):
        GenerationParams(height=5000)  # above max 4096


def test_prompt_style_enum():
    """PromptStyle enum has expected values."""
    from localimagegen.core.models import PromptStyle

    assert PromptStyle.FLUX.value == "flux"
    assert PromptStyle.SDXL.value == "sdxl"
    assert PromptStyle.MIDJOURNEY.value == "midjourney"


def test_logging_setup():
    """Logging can be configured without errors."""
    from localimagegen.services.logging import configure_logging, get_logger

    configure_logging(level="DEBUG", json=False)
    logger = get_logger("test")
    assert logger is not None
