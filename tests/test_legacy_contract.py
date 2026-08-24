"""The POST /generate contract is frozen. These tests fail if it drifts.

Reconstructed from the original repo-root `test.txt`, which was a real captured
request body.
"""

import json
from pathlib import Path

from app.schemas import GenerateRequest, GenerateResponse

FIXTURE = Path(__file__).parent / "fixtures" / "legacy_generate.json"

# Exactly the fields the pre-refactor GenerateResponse returned.
FROZEN_RESPONSE_FIELDS = {
    "image_path", "model", "width", "height", "steps",
    "guidance_scale", "seed", "quantization", "generation_time_s",
}

FROZEN_REQUEST_FIELDS = {
    "prompt", "platform", "width", "height", "num_inference_steps",
    "guidance_scale", "seed", "negative_prompt", "path",
}


def test_captured_body_still_validates():
    req = GenerateRequest(**json.loads(FIXTURE.read_text()))
    assert req.platform == "default" and req.seed == 42
    assert req.num_inference_steps == 4


def test_request_keeps_every_legacy_field():
    missing = FROZEN_REQUEST_FIELDS - set(GenerateRequest.model_fields)
    assert not missing, f"legacy request fields dropped: {missing}"


def test_response_keeps_every_legacy_field():
    missing = FROZEN_RESPONSE_FIELDS - set(GenerateResponse.model_fields)
    assert not missing, f"legacy response fields dropped: {missing}"


def test_response_additions_are_only_warnings():
    added = set(GenerateResponse.model_fields) - FROZEN_RESPONSE_FIELDS
    assert added == {"warnings"}, f"unexpected additive fields: {added}"


def test_deprecated_fields_are_still_accepted():
    """They are ignored on a distilled model, but must not become errors."""
    req = GenerateRequest(prompt="x", negative_prompt="blurry", guidance_scale=7.5)
    assert req.negative_prompt == "blurry" and req.guidance_scale == 7.5
