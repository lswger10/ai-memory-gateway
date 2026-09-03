import pytest
from dataclasses import replace

from cache_probe import GatewayCacheProbeService
from model_execution import ProviderChunk
from model_execution_contracts import ProviderUsage
from model_profile_store import InMemoryModelProfileStore
from model_profiles import ModelProfile


def _profile(*, strategy="anthropic_prefix_anchored_v1", ttl="1h"):
    ttls = [ttl] if ttl else []
    return ModelProfile.from_dict(
        {
            "profile_id": "profile-1",
            "display_name": "Profile 1",
            "enabled": True,
            "test_status": "unverified",
            "provider": "test-provider",
            "protocol": "anthropic_messages_compatible",
            "base_url": "https://provider.invalid",
            "route_id": "route-1",
            "model": "model-1",
            "adapter_version": "adapter-v1",
            "credential_ref": "env:TEST_PROVIDER_KEY",
            "headers": {},
            "capabilities": {
                "streaming": True,
                "structured_output": False,
                "tools": False,
                "reasoning_controls": False,
                "cache_strategies": [strategy],
                "cache_ttls": ttls,
                "usage_fields": [
                    "cache_creation_input_tokens",
                    "cache_read_input_tokens",
                ],
            },
            "cache_strategy": strategy,
            "requested_cache_ttl": ttl,
            "revision": 1,
        }
    )


class _Runner:
    def __init__(self, usages):
        self.usages = list(usages)
        self.calls = []

    async def run(
        self,
        *,
        profile,
        request,
        context,
        cache_namespace,
        max_output_tokens=None,
    ):
        self.calls.append(
            (profile, request, context, cache_namespace, max_output_tokens)
        )
        yield ProviderChunk("final", {"text": "CACHE_PROBE_OK"})
        usage = self.usages.pop(0)
        yield ProviderChunk(
            "usage",
            {"usage": usage, "observed_cache_support": "unverified"},
        )


@pytest.mark.anyio
async def test_double_send_probe_freezes_every_input_and_promotes_verified_cache():
    store = InMemoryModelProfileStore()
    await store.put_profile(_profile())
    runner = _Runner(
        [
            ProviderUsage.from_provider_values(cache_creation_input_tokens=400),
            ProviderUsage.from_provider_values(cache_read_input_tokens=380),
        ]
    )
    service = GatewayCacheProbeService(profiles=store, provider_runner=runner)

    result = await service.run(
        profile_id="profile-1",
        actor_id="jiao",
        room_id="room_weiwei_jiao",
        conversation_id="canonical-conversation-1",
    )

    assert result.status == "verified"
    assert len(runner.calls) == 2
    first, second = runner.calls
    assert first[1] == second[1]
    assert first[2] == second[2]
    assert first[3] == second[3]
    assert first[2].dynamic_tail == ("cache-probe-dynamic-tail-v1",)
    assert (await store.get_profile("profile-1")).test_status == "passed"


@pytest.mark.anyio
async def test_cache_write_only_probe_verifies_route_without_claiming_cache_hit():
    store = InMemoryModelProfileStore()
    await store.put_profile(_profile())
    runner = _Runner(
        [
            ProviderUsage.from_provider_values(cache_creation_input_tokens=400),
            ProviderUsage.from_provider_values(cache_creation_input_tokens=20),
        ]
    )
    service = GatewayCacheProbeService(profiles=store, provider_runner=runner)

    result = await service.run(
        profile_id="profile-1",
        actor_id="jiao",
        room_id="room_weiwei_jiao",
        conversation_id="canonical-conversation-1",
    )

    assert result.status == "unverified"
    profile = await store.get_profile("profile-1")
    assert profile.test_status == "passed"
    assert profile.selectable is True


@pytest.mark.anyio
async def test_cache_miss_does_not_unverify_an_already_verified_route():
    store = InMemoryModelProfileStore()
    await store.put_profile(replace(_profile(), test_status="passed"))
    runner = _Runner(
        [
            ProviderUsage.from_provider_values(input_tokens=100, cached_tokens=0),
            ProviderUsage.from_provider_values(input_tokens=100, cached_tokens=0),
        ]
    )
    service = GatewayCacheProbeService(profiles=store, provider_runner=runner)

    result = await service.run(
        profile_id="profile-1",
        actor_id="jiao",
        room_id="room_weiwei_jiao",
        conversation_id="canonical-conversation-1",
    )

    assert result.status == "unverified"
    assert (await store.get_profile("profile-1")).test_status == "passed"


@pytest.mark.anyio
async def test_no_cache_profile_can_pass_route_probe_without_fabricated_cache_support():
    store = InMemoryModelProfileStore()
    await store.put_profile(_profile(strategy="no_prompt_cache_v1", ttl=None))
    runner = _Runner(
        [
            ProviderUsage.from_provider_values(input_tokens=30, output_tokens=3),
            ProviderUsage.from_provider_values(input_tokens=30, output_tokens=3),
        ]
    )
    service = GatewayCacheProbeService(profiles=store, provider_runner=runner)

    result = await service.run(
        profile_id="profile-1",
        actor_id="laoke",
        room_id="room_weiwei_laoke",
        conversation_id="canonical-conversation-2",
    )

    assert result.status == "not_applicable"
    assert (await store.get_profile("profile-1")).test_status == "passed"
    assert result.second.cache_read_input_tokens is None


@pytest.mark.anyio
async def test_paid_cache_probe_caps_provider_output_to_minimal_tokens():
    store = InMemoryModelProfileStore()
    await store.put_profile(_profile())
    runner = _Runner(
        [
            ProviderUsage.from_provider_values(cache_creation_input_tokens=400),
            ProviderUsage.from_provider_values(cache_read_input_tokens=380),
        ]
    )
    service = GatewayCacheProbeService(profiles=store, provider_runner=runner)

    await service.run(
        profile_id="profile-1",
        actor_id="jiao",
        room_id="room_weiwei_jiao",
        conversation_id="canonical-conversation-1",
    )

    assert [call[4] for call in runner.calls] == [32, 32]


@pytest.mark.anyio
async def test_paid_cache_probe_uses_a_distinct_stable_prefix_per_profile():
    store = InMemoryModelProfileStore()
    first_profile = _profile(strategy="no_prompt_cache_v1", ttl=None)
    second_profile = replace(
        first_profile,
        profile_id="profile-2",
        route_id="route-2",
    )
    await store.put_profile(first_profile)
    await store.put_profile(second_profile)
    runner = _Runner(
        [
            ProviderUsage.from_provider_values(input_tokens=30, output_tokens=3),
            ProviderUsage.from_provider_values(input_tokens=30, output_tokens=3),
            ProviderUsage.from_provider_values(input_tokens=30, output_tokens=3),
            ProviderUsage.from_provider_values(input_tokens=30, output_tokens=3),
        ]
    )
    service = GatewayCacheProbeService(profiles=store, provider_runner=runner)

    await service.run(
        profile_id="profile-1",
        actor_id="laoke",
        room_id="room_weiwei_laoke",
        conversation_id="canonical-conversation-1",
    )
    await service.run(
        profile_id="profile-2",
        actor_id="laoke",
        room_id="room_weiwei_laoke",
        conversation_id="canonical-conversation-1",
    )

    first_context = runner.calls[0][2]
    second_context = runner.calls[2][2]
    assert first_context.static_system != second_context.static_system
