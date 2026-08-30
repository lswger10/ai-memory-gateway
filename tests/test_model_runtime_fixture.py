import json

import pytest

from model_profile_store import InMemoryModelProfileStore
from model_runtime_fixture import bootstrap_ephemeral_model_profiles


def _profile(profile_id):
    return {
        "profile_id": profile_id,
        "display_name": profile_id,
        "enabled": True,
        "test_status": "passed",
        "provider": "local-fake",
        "protocol": "openai_chat_completions",
        "base_url": "http://127.0.0.1:3099/v1",
        "route_id": profile_id,
        "model": profile_id,
        "adapter_version": "fake.v1",
        "credential_ref": "env:FAKE_KEY",
        "headers": {"Authorization": "Bearer ${credential}"},
        "capabilities": {
            "streaming": True,
            "structured_output": False,
            "tools": False,
            "reasoning_controls": False,
            "cache_strategies": ["no_prompt_cache_v1"],
            "cache_ttls": [],
            "usage_fields": [],
        },
        "cache_strategy": "no_prompt_cache_v1",
        "requested_cache_ttl": None,
        "revision": 1,
    }


@pytest.mark.anyio
async def test_ephemeral_fixture_bootstraps_explicit_profiles_bindings_and_fallbacks(tmp_path):
    path = tmp_path / "profiles.json"
    path.write_text(
        json.dumps(
            {
                "profiles": [_profile("primary"), _profile("fallback")],
                "actor_defaults": {"jiao": "primary", "laoke": "fallback"},
                "approved_fallbacks": {"jiao": ["fallback"], "laoke": []},
                "room_overrides": [
                    {
                        "room_id": "room_group_home",
                        "actor_id": "jiao",
                        "profile_id": "fallback",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    store = InMemoryModelProfileStore()

    await bootstrap_ephemeral_model_profiles(store, path)

    private = await store.resolve("jiao", "room_weiwei_jiao")
    group = await store.resolve("jiao", "room_group_home")
    assert private.primary.profile_id == "primary"
    assert [item.profile_id for item in private.fallbacks] == ["fallback"]
    assert group.primary.profile_id == "fallback"
    assert group.actor_id == "jiao"

