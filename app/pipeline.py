"""DEPRECATED back-compat shim.

The FLUX logic moved to `app/backends/image_flux.py` and GPU arbitration to
`app/manager.py`. This adapter keeps `get_pipeline()` working for anything that
still imports it. Slated for removal after one release.
"""

from __future__ import annotations

import warnings

from app.manager import get_manager

BACKEND = "image.flux2-klein"


class _PipelineAdapter:
    @property
    def _backend(self):
        return get_manager().get(BACKEND)

    @property
    def loaded(self) -> bool:
        return self._backend.loaded

    @property
    def quantization(self) -> str:
        return getattr(self._backend, "quantization", "none")

    @property
    def load_error(self):
        return getattr(self._backend, "load_error", None)

    def load(self):
        get_manager().run  # noqa: B018 - ensure manager is constructed
        self._backend.load()
        return self._backend

    def generate(self, prompt, width, height, steps, guidance_scale=None, seed=None,
                 negative_prompt=None, out_path=None):
        from app.storage import resolve_output_path

        art = get_manager().run(
            BACKEND,
            {
                "prompt": prompt, "width": width, "height": height, "steps": steps,
                "seed": seed, "negative_prompt": negative_prompt,
                "guidance_scale": guidance_scale,
                "out_path": out_path or resolve_output_path(None, "image"),
            },
        )
        from PIL import Image

        return Image.open(art.path)

    def cleanup(self):
        get_manager().evict(BACKEND)


_adapter = _PipelineAdapter()


def get_pipeline():
    warnings.warn(
        "app.pipeline.get_pipeline() is deprecated; use app.service or app.manager",
        DeprecationWarning,
        stacklevel=2,
    )
    return _adapter
