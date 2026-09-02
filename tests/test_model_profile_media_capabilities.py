import pytest

from model_profiles import ModelProfile, ProfileContractError


def payload(input_modalities=None):
    capabilities = {
        "streaming": True,
        "structured_output": False,
        "tools": False,
        "reasoning_controls": False,
        "cache_strategies": ["no_prompt_cache_v1"],
        "cache_ttls": [],
        "usage_fields": [],
    }
    if input_modalities is not None:
        capabilities["input_modalities"] = input_modalities
    return {
        "profile_id": "media-profile",
        "display_name": "Media Profile",
        "enabled": True,
        "test_status": "passed",
        "provider": "test",
        "protocol": "openai_chat_completions",
        "base_url": "https://example.invalid/v1",
        "route_id": "media-route",
        "model": "model",
        "adapter_version": "adapter.v1",
        "credential_ref": "env:MEDIA_KEY",
        "headers": {"Authorization": "Bearer ${credential}"},
        "capabilities": capabilities,
        "cache_strategy": "no_prompt_cache_v1",
        "requested_cache_ttl": None,
        "revision": 1,
    }


def test_profile_input_modalities_are_explicit_and_do_not_infer_from_protocol():
    legacy = ModelProfile.from_dict(payload())
    media = ModelProfile.from_dict(payload(["text", "image", "document"]))
    assert legacy.capabilities.input_modalities == ("text",)
    assert media.capabilities.input_modalities == ("text", "image", "document")


def test_profile_rejects_unknown_or_missing_text_modality():
    with pytest.raises(ProfileContractError):
        ModelProfile.from_dict(payload(["image"]))
    with pytest.raises(ProfileContractError):
        ModelProfile.from_dict(payload(["text", "vision-by-model-name"]))
