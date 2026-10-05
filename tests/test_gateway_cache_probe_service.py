from cache_strategies import PromptSegment
from model_usage_store import InMemoryModelUsageStore
import pytest
from dataclasses import replace

from cache_probe import GatewayCacheProbeService
from model_execution import ProviderChunk
from model_execution_contracts import ProviderUsage
from model_profile_store import InMemoryModelProfileStore
from model_profile_store import ProfileStoreError
from model_profiles import ModelProfile


@pytest.mark.anyio
@pytest.mark.parametrize("protocol,mode", [
    ("anthropic_messages_compatible", "anthropic_web_search_20250305"),
    ("openai_responses", "openai_web_search"),
])
async def test_search_probe_reaches_transport_through_real_request_renderer(protocol, mode):
    import json
    import httpx
    from gateway_provider_runner import GatewayProviderRunner
    from provider_transport import PooledHttpTransport
    from test_gateway_provider_runner import Resolver

    payload = _profile().to_dict()
    payload["protocol"] = protocol
    payload["headers"] = {"Authorization": "Bearer ${credential}"}
    payload["capabilities"]["web_search"] = mode
    if protocol == "openai_responses":
        payload.update(cache_strategy="openai_stable_prefix_v1", requested_cache_ttl=None)
        payload["capabilities"].update(cache_strategies=["openai_stable_prefix_v1"], cache_ttls=[])
    profiles = InMemoryModelProfileStore()
    await profiles.put_profile(ModelProfile.from_dict(payload))
    usage = InMemoryModelUsageStore()
    sent = []

    def respond(request):
        sent.append(json.loads(request.content))
        # A normal answer without server-search evidence must NOT enable search.
        if protocol == "openai_responses":
            body = 'event: response.output_text.delta\ndata: {"delta":"No search evidence."}\n\nevent: response.completed\ndata: {"response":{"output":[],"usage":{"input_tokens":31,"output_tokens":2}}}\n\n'
        else:
            body = 'event: message_start\ndata: {"message":{"usage":{"input_tokens":31,"output_tokens":2}}}\n\nevent: content_block_delta\ndata: {"delta":{"type":"text_delta","text":"No search evidence."}}\n\nevent: message_stop\ndata: {}\n\n'
        return httpx.Response(200, text=body)

    transport = PooledHttpTransport(client_factory=lambda **kwargs:
        httpx.AsyncClient(transport=httpx.MockTransport(respond), **kwargs))
    service = GatewayCacheProbeService(profiles=profiles, usage_store=usage,
        provider_runner=GatewayProviderRunner(transport=transport, credential_resolver=Resolver()))
    try:
        result = await service.run_search(profile_id="profile-1", profile_revision=1,
            actor_id="jiao", room_id="room_weiwei_jiao", conversation_id="synthetic")
        assert len(sent) == 1
        assert sent[0]["tools"][0]["type"] == ("web_search" if protocol == "openai_responses" else "web_search_20250305")
        assert sent[0]["max_output_tokens" if protocol == "openai_responses" else "max_tokens"] == 512
        assert result["status"] == "unverified"
        rows = await usage.list_receipts()
        assert len(rows) == 1
        assert rows[0].execution_purpose == "web_search_probe"
        assert rows[0].usage.input_tokens == 31
        assert not await profiles.has_verified_probe("profile-1", 1, "native_web_search")
    finally:
        await transport.close()


@pytest.mark.anyio
@pytest.mark.parametrize("evidence", [True, False])
async def test_search_probe_is_one_small_call_and_requires_provider_search_evidence(evidence):
    store = InMemoryModelProfileStore()
    original = _profile()
    profile = replace(original, capabilities=replace(original.capabilities, web_search="anthropic_web_search_20250305"))
    await store.put_profile(profile)
    class SearchRunner(_Runner):
        async def run(self, **kwargs):
            yield ProviderChunk("web_search", {"verified": evidence})
            async for item in super().run(**kwargs):
                yield item
    runner = SearchRunner([ProviderUsage.from_provider_values(input_tokens=40, output_tokens=15)])
    service = GatewayCacheProbeService(profiles=store, provider_runner=runner, usage_store=InMemoryModelUsageStore())
    result = await service.run_search(profile_id=profile.profile_id, profile_revision=1,
        actor_id="jiao", room_id="room_weiwei_jiao", conversation_id="synthetic")
    assert len(runner.calls) == 1
    assert runner.calls[0][2].web_search_enabled
    assert len(str(runner.calls[0][2].static_system)) < 1000
    assert result["status"] == ("verified" if evidence else "unverified")
    assert await store.has_verified_probe(profile.profile_id, 1, "native_web_search") == evidence
    assert (await store.get_profile(profile.profile_id)).test_status == ("passed" if evidence else profile.test_status)
    assert not await store.has_verified_probe(profile.profile_id, 1, "frozen_double_send_cache")
    await store.put_profile(replace(profile, revision=2))
    assert not await store.has_verified_probe(profile.profile_id, 2, "native_web_search")


@pytest.mark.anyio
@pytest.mark.parametrize("second_fails", [False, True])
async def test_real_postgres_explicit_probe_records_both_http_attempts(isolated_postgres, monkeypatch, second_fails):
    import database
    import httpx
    from gateway_provider_runner import GatewayProviderRunner
    from postgres_model_stores import PostgresModelProfileStore, PostgresModelUsageStore
    from provider_transport import PooledHttpTransport
    from test_gateway_provider_runner import Resolver
    async def factory():
        return isolated_postgres
    monkeypatch.setattr(database, "get_pool", factory)
    monkeypatch.setattr(database, "MEMORY_VECTOR_ENABLED", False)
    await database.init_tables()
    profiles = PostgresModelProfileStore(factory)
    payload = _profile().to_dict()
    payload["headers"] = {"x-api-key": "${credential}"}
    await profiles.put_profile(ModelProfile.from_dict(payload))
    usage = PostgresModelUsageStore(factory)
    calls = []
    def respond(request):
        calls.append(request)
        if second_fails and len(calls) == 2:
            return httpx.Response(503)
        return httpx.Response(200, text='event: message_start\ndata: {"message":{"usage":{"input_tokens":31}}}\n\nevent: message_stop\ndata: {}\n\n')
    transport = PooledHttpTransport(client_factory=lambda **kwargs:
        httpx.AsyncClient(transport=httpx.MockTransport(respond), **kwargs))
    service = GatewayCacheProbeService(profiles=profiles, usage_store=usage,
        provider_runner=GatewayProviderRunner(transport=transport, credential_resolver=Resolver()))
    try:
        await service.run(profile_id="profile-1", actor_id="jiao", room_id="room_weiwei_jiao", conversation_id="probe-test")
        rows = await usage.list_receipts()
        assert len(calls) == len(rows) == 2
        assert {row.execution_purpose for row in rows} == {"cache_probe"}
        assert [row.status for row in rows] == ["failed" if second_fails else "succeeded", "succeeded"]
        assert [row.usage.input_tokens for row in rows] == [None if second_fails else 31, 31]
        assert all(row.usage.output_tokens is None for row in rows)
    finally:
        await transport.close()


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
        on_attempt=None,
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
        if on_attempt:
            import uuid
            await on_attempt(str(uuid.uuid4()), usage, "succeeded", True, "unverified")


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
    service = GatewayCacheProbeService(profiles=store, provider_runner=runner, usage_store=InMemoryModelUsageStore())

    result = await service.run(
        profile_id="profile-1",
        actor_id="jiao",
        room_id="room_weiwei_jiao",
        conversation_id="canonical-conversation-1",
    )

    assert result.status == "verified"
    assert result.profile_id == "profile-1"
    assert result.profile_revision == 1
    assert len(runner.calls) == 2
    first, second = runner.calls
    assert first[1] == second[1]
    assert first[2] == second[2]
    assert first[3] == second[3]
    assert first[2].dynamic_tail == (PromptSegment("request_metadata", "cache-probe-dynamic-tail-v1"),)
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
    service = GatewayCacheProbeService(profiles=store, provider_runner=runner, usage_store=InMemoryModelUsageStore())

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
    service = GatewayCacheProbeService(profiles=store, provider_runner=runner, usage_store=InMemoryModelUsageStore())

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
    service = GatewayCacheProbeService(profiles=store, provider_runner=runner, usage_store=InMemoryModelUsageStore())

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
    service = GatewayCacheProbeService(profiles=store, provider_runner=runner, usage_store=InMemoryModelUsageStore())

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
    service = GatewayCacheProbeService(profiles=store, provider_runner=runner, usage_store=InMemoryModelUsageStore())

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


@pytest.mark.anyio
async def test_inflight_old_probe_does_not_certify_edited_profile():
    store = InMemoryModelProfileStore()
    original = _profile()
    await store.put_profile(original)

    class EditingRunner(_Runner):
        async def run(self, **kwargs):
            if not self.calls:
                await store.put_profile(replace(original, model="new-model", revision=2))
            async for chunk in super().run(**kwargs):
                yield chunk

    runner = EditingRunner([ProviderUsage.from_provider_values(cache_read_input_tokens=80)] * 2)
    service = GatewayCacheProbeService(profiles=store, provider_runner=runner, usage_store=InMemoryModelUsageStore())
    with pytest.raises(ProfileStoreError, match="revision"):
        await service.run(profile_id="profile-1", actor_id="jiao",
            room_id="room_weiwei_jiao", conversation_id="synthetic-conversation")
    assert (await store.get_profile("profile-1")).test_status == "unverified"
    assert not await store.has_verified_probe("profile-1", 2, "frozen_double_send_cache")


@pytest.mark.anyio
async def test_stale_dashboard_probe_revision_rejected_before_provider():
    store = InMemoryModelProfileStore()
    await store.put_profile(replace(_profile(), revision=2))
    runner = _Runner([])
    service = GatewayCacheProbeService(profiles=store, provider_runner=runner, usage_store=InMemoryModelUsageStore())
    with pytest.raises(ProfileStoreError, match="revision"):
        await service.run(profile_id="profile-1", profile_revision=1, actor_id="jiao",
            room_id="room_weiwei_jiao", conversation_id="synthetic")
    assert runner.calls == []
