import pytest

from model_profiles import (
    ModelBinding,
    ModelProfile,
    ProfileCapabilities,
    ProfileContractError,
    resolve_feature_flags,
)


def _profile_payload(**overrides):
    payload = {
        "profile_id": "ofox-claude",
        "display_name": "OFOX Claude",
        "enabled": True,
        "test_status": "passed",
        "provider": "ofox",
        "protocol": "anthropic_messages",
        "base_url": "https://api.example.invalid/anthropic",
        "route_id": "ofox-anthropic",
        "model": "anthropic/claude-opus-4.6",
        "adapter_version": "anthropic-messages.v1",
        "credential_ref": "env:OFOX_ANTHROPIC_KEY",
        "capabilities": {
            "streaming": True,
            "structured_output": False,
            "tools": True,
            "reasoning_controls": False,
            "cache_strategies": ["anthropic_prefix_anchored_v1"],
            "cache_ttls": ["5m"],
            "usage_fields": ["input_tokens", "output_tokens"],
        },
        "cache_strategy": "anthropic_prefix_anchored_v1",
        "requested_cache_ttl": "5m",
        "revision": 1,
    }
    payload.update(overrides)
    return payload


def test_profile_has_no_actor_identity_field():
    with pytest.raises(ProfileContractError, match="actor_id"):
        ModelProfile.from_dict(_profile_payload(actor_id="jiao"))


def test_any_tested_profile_can_bind_to_either_actor():
    profile = ModelProfile.from_dict(_profile_payload())

    jiao = ModelBinding.create(actor_id="jiao", default_profile=profile)
    laoke = ModelBinding.create(actor_id="laoke", default_profile=profile)

    assert jiao.default_profile_id == profile.profile_id
    assert laoke.default_profile_id == profile.profile_id


def test_unprobed_profile_cannot_be_selected():
    profile = ModelProfile.from_dict(_profile_payload(test_status="unverified"))

    with pytest.raises(ProfileContractError, match="tested"):
        ModelBinding.create(actor_id="jiao", default_profile=profile)


def test_fallback_requires_explicit_ordered_profile_ids():
    profile = ModelProfile.from_dict(_profile_payload())
    binding = ModelBinding.create(
        actor_id="jiao",
        default_profile=profile,
        approved_fallback_profile_ids=("gemini-tested", "deepseek-tested"),
    )

    assert binding.approved_fallback_profile_ids == (
        "gemini-tested",
        "deepseek-tested",
    )
    with pytest.raises(ProfileContractError, match="explicit"):
        ModelBinding.create(
            actor_id="jiao",
            default_profile=profile,
            approved_fallback_profile_ids=("*",),
        )


def test_cache_ttl_must_be_observed_for_the_route():
    with pytest.raises(ProfileContractError, match="TTL"):
        ModelProfile.from_dict(_profile_payload(requested_cache_ttl="1h"))


def test_new_feature_flags_default_false(monkeypatch):
    for name in (
        "MODEL_EXECUTION_ENABLED",
        "MODEL_PROFILE_MANAGEMENT_ENABLED",
        "MODEL_PROFILE_PWA_ENABLED",
    ):
        monkeypatch.delenv(name, raising=False)

    assert resolve_feature_flags() == {
        "model_execution": False,
        "model_profile_management": False,
        "model_profile_pwa": False,
    }


def test_capabilities_do_not_infer_cache_from_protocol_name():
    capabilities = ProfileCapabilities.from_dict(
        {
            "streaming": True,
            "structured_output": False,
            "tools": False,
            "reasoning_controls": False,
            "cache_strategies": [],
            "cache_ttls": [],
            "usage_fields": [],
        }
    )
    assert capabilities.cache_strategies == ()
    assert capabilities.cache_ttls == ()
