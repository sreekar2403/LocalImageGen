"""Application settings with multi-source configuration loading.

Configuration is resolved with the following precedence (highest wins):

1. **CLI arguments** - explicit overrides passed to :meth:`Settings.from_cli_args`.
2. **Environment variables** - including values loaded from a ``.env`` file.
3. **YAML configuration file** - a ``config.yaml`` (or path given via
   ``LOCALIMAGEGEN_CONFIG`` / ``--config``).
4. **Built-in defaults** - defined on the model fields.

This is implemented on top of ``pydantic-settings`` so that every field is
validated and typed, and environment variables map automatically to fields via
the ``env_prefix``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

#: Default YAML configuration file name looked up relative to the project root.
DEFAULT_CONFIG_FILE = "config.yaml"

#: Environment variable prefix used for all settings (e.g. ``LOCALIMAGEGEN_*``).
ENV_PREFIX = "LOCALIMAGEGEN_"


class Settings(BaseSettings):
    """Typed application configuration.

    All fields have sensible defaults so the application can run with zero
    configuration. Values may be overridden through environment variables,
    a YAML file, or CLI arguments (in increasing order of precedence).
    """

    model_config = SettingsConfigDict(
        env_prefix=ENV_PREFIX,
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # --- Model selection -------------------------------------------------
    vision_model: str = Field(
        default="black-forest-labs/FLUX.2-klein-4B",
        description="Hugging Face model id used for image diffusion.",
    )
    ollama_model: str = Field(
        default="gemma4:e4b",
        description="Ollama model used for prompt enhancement.",
    )

    # --- Generation parameters ------------------------------------------
    width: int = Field(default=1024, ge=64, le=4096, description="Output image width.")
    height: int = Field(default=768, ge=64, le=4096, description="Output image height.")
    num_inference_steps: int = Field(
        default=12, ge=1, le=1000, description="Number of diffusion inference steps."
    )
    seed: int = Field(default=42, description="Random seed for reproducibility.")
    device: str = Field(
        default="cuda",
        description="Device to run inference on ('cuda' or 'cpu').",
    )

    # --- Paths -----------------------------------------------------------
    output_dir: Path = Field(
        default=Path("images"),
        description="Directory where generated images are saved.",
    )
    config_file: Path = Field(
        default=Path(DEFAULT_CONFIG_FILE),
        description="Path to the YAML configuration file.",
    )

    # --- Logging ---------------------------------------------------------
    log_level: str = Field(
        default="INFO",
        description="Logging level for the structured logger.",
    )

    @field_validator("device")
    @classmethod
    def _validate_device(cls, value: str) -> str:
        normalized = value.strip().lower()
        if normalized not in {"cuda", "cpu"}:
            raise ValueError(f"Unsupported device '{value}'; expected 'cuda' or 'cpu'.")
        return normalized

    @field_validator("log_level")
    @classmethod
    def _validate_log_level(cls, value: str) -> str:
        normalized = value.strip().upper()
        if normalized not in {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}:
            raise ValueError(f"Unsupported log level '{value}'.")
        return normalized

    # --- Loading helpers -------------------------------------------------

    @classmethod
    def from_yaml(cls, path: Path | str) -> "Settings":
        """Build settings from a YAML file, falling back to defaults for gaps.

        The YAML file is optional; if it does not exist an empty mapping is used
        so that defaults and other sources still apply.
        """
        yaml_path = Path(path)
        data: Mapping[str, Any] = {}
        if yaml_path.exists():
            import yaml

            with yaml_path.open("r", encoding="utf-8") as handle:
                loaded = yaml.safe_load(handle) or {}
                if not isinstance(loaded, dict):
                    raise ValueError(
                        f"Config file '{yaml_path}' must contain a YAML mapping."
                    )
                data = loaded
        return cls(**data)

    @classmethod
    def from_cli_args(
        cls,
        overrides: Mapping[str, Any] | None = None,
        *,
        config_file: Path | str | None = None,
    ) -> "Settings":
        """Build settings applying CLI argument overrides on top of all sources.

        ``overrides`` should map setting field names to their CLI-provided
        values. These take the highest precedence. If ``config_file`` is given
        it is used as the YAML source; otherwise the ``config_file`` field
        (from env/defaults) is honored.
        """
        overrides = dict(overrides or {})

        # Determine the YAML source: explicit CLI path wins, otherwise fall back
        # to whatever the environment/defaults resolved for config_file.
        yaml_source: Path | None = None
        if config_file is not None:
            yaml_source = Path(config_file)
        else:
            # Resolve the default config path from env/defaults first.
            probe = cls()
            yaml_source = probe.config_file

        settings = cls.from_yaml(yaml_source)
        if overrides:
            settings = settings.model_copy(update=overrides)
        return settings

    def to_yaml(self, path: Path | str) -> None:
        """Serialize the current settings to a YAML file (for scaffolding)."""
        import yaml

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            yaml.safe_dump(self.model_dump(mode="json"), handle, sort_keys=False)
