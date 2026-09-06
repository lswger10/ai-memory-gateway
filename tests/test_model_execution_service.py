import asyncio

import pytest

from model_execution import (
    ContextBundle,
    GatewayModelExecutionService,
    ProviderChunk,
    ProviderRunUnavailable,
)
from model_execution_contracts import GatewayExecutionRequest, ProviderUsage
from model_profile_store import InMemoryModelProfileStore
from model_profiles import ModelProfile
from model_usage_store import InMemoryModelUsageStore


def _profile(profile_id: str) -> ModelProfile:
    return ModelProfile.from_dict(
        {
            "profile_id": profile_id,
            "display_name": profile_id,
            "enabled": True,
            "test_status": "passed",
            "provider": "fake",
            "protocol": "anthropic_messages",
            "base_url": "https://example.invalid",
            "route_id": f"route-{profile_id}",
            "model": f"model-{profile_id}",
            "adapter_version": "fake.v1",
            "credential_ref": f"env:{profile_id.upper()}_KEY",
            "headers": {"x-api-key": "${credential}"},
            "capabilities": {
                "streaming": True,
                "structured_output": False,
                "tools": False,
                "reasoning_controls": False,
                "cache_strategies": ["anthropic_prefix_anchored_v1"],
                "cache_ttls": ["5m"],
                "usage_fields": [
                    "input_tokens",
                    "output_tokens",
                    "cache_creation_input_tokens",
                    "cache_read_input_tokens",
                ],
            },
            "cache_strategy": "anthropic_prefix_anchored_v1",
            "requested_cache_ttl": "5m",
            "revision": 1,
        }
    )


def _request(*, binding_revision=1):
    return GatewayExecutionRequest.from_dict(
        {
            "contract_version": "gateway-model-execution.v1.0",
            "execution_kind": "full",
            "actor_id": "jiao",
            "room_id": "room_group_home",
            "conversation_id": "conversation-1",
            "current_event_id": 101,
            "generation_request_id": "generation-1",
            "execution_mode": "group",
            "fence": {
                "room_id": "room_group_home",
                "conversation_id": "conversation-1",
                "burst_id": "burst-1",
                "trigger_event_id": 101,
                "fence_epoch": 1,
                "lease_epoch": 1,
                "orchestrator_instance": "orch-1",
            },
            "bedroom_session_id": None,
            "binding_revision": binding_revision,
        }
    )


class _ContextBuilder:
    def __init__(self, cache_conversation_id=None):
        self.requests = []
        self.cache_conversation_id = cache_conversation_id

    async def build(self, request):
        self.requests.append(request)
        return ContextBundle(
            static_system=("kernel", "actor", "room"),
            stable_summary="summary",
            stable_history=("history",),
            dynamic_tail=("memory", "current"),
            actor_prompt_version="jiao.v1",
            runtime_kernel_version="kernel.v1",
            room_policy_version="group.v1",
            tool_schema_hash="tools.none",
            cache_conversation_id=self.cache_conversation_id,
        )


class _Runner:
    def __init__(self, *, fail_profiles=()):
        self.fail_profiles = set(fail_profiles)
        self.calls = []
        self.cancelled = False

    async def run(self, *, profile, request, context, cache_namespace, on_attempt):
        import uuid
        self.calls.append((profile.profile_id, request, context, cache_namespace))
        failed = profile.profile_id in self.fail_profiles
        usage = (ProviderUsage.from_provider_values() if failed else ProviderUsage.from_provider_values(
            input_tokens=100, output_tokens=10, cache_creation_input_tokens=80, cache_read_input_tokens=20))
        try:
            if failed:
                raise ProviderRunUnavailable("sanitized failure")
            yield ProviderChunk("delta", {"text": "hello"})
            yield ProviderChunk("final", {"text": "hello"})
            yield ProviderChunk("usage", {"usage": usage, "observed_cache_support": "verified"})
        finally:
            self.cancelled = True
            await on_attempt(str(uuid.uuid4()), usage, "failed" if failed else "succeeded",
                not failed, "unverified" if failed else "verified")


async def _service(*, fail_profiles=(), fallbacks=()):
    profiles = InMemoryModelProfileStore()
    for profile_id in ("primary", *fallbacks):
        await profiles.put_profile(_profile(profile_id))
    await profiles.set_actor_default("jiao", "primary")
    if fallbacks:
        await profiles.set_approved_fallbacks("jiao", tuple(fallbacks))
    context = _ContextBuilder()
    runner = _Runner(fail_profiles=fail_profiles)
    usage = InMemoryModelUsageStore()
    return (
        GatewayModelExecutionService(
            profiles=profiles,
            context_builder=context,
            provider_runner=runner,
            usage_store=usage,
        ),
        context,
        runner,
        usage,
    )


@pytest.mark.anyio
async def test_gateway_resolves_profile_without_orchestrator_model_input():
    service, _, runner, _ = await _service()
    events = [event async for event in service.stream(_request())]

    assert events[0].event == "profile"
    assert events[0].data["profile_id"] == "primary"
    assert runner.calls[0][0] == "primary"
    assert all("api_key" not in str(event.data).lower() for event in events)


@pytest.mark.anyio
async def test_gateway_fetches_relay_facts_and_builds_actor_pack_once():
    service, context, _, _ = await _service()
    _ = [event async for event in service.stream(_request())]
    assert context.requests == [_request()]


@pytest.mark.anyio
async def test_profile_fallback_uses_only_explicit_allowlist():
    service, _, runner, _ = await _service(
        fail_profiles=("primary",), fallbacks=("approved",)
    )
    events = [event async for event in service.stream(_request(binding_revision=2))]

    assert [call[0] for call in runner.calls] == ["primary", "approved"]
    profiles = [event.data for event in events if event.event == "profile"]
    assert profiles[-1]["profile_id"] == "approved"
    assert profiles[-1]["fallback_used"] is True
    assert profiles[-1]["fallback_from_profile_id"] == "primary"


class _BlockingRunner:
    def __init__(self):
        self.cancelled = asyncio.Event()

    async def run(self, **kwargs):
        try:
            yield ProviderChunk("delta", {"text": "partial"})
            await asyncio.Event().wait()
        finally:
            self.cancelled.set()


@pytest.mark.anyio
async def test_cancelled_execution_never_yields_accepted_final():
    profiles = InMemoryModelProfileStore()
    await profiles.put_profile(_profile("primary"))
    await profiles.set_actor_default("jiao", "primary")
    runner = _BlockingRunner()
    usage = InMemoryModelUsageStore()
    service = GatewayModelExecutionService(
        profiles=profiles,
        context_builder=_ContextBuilder(),
        provider_runner=runner,
        usage_store=usage,
    )
    stream = service.stream(_request())
    assert (await anext(stream)).event == "profile"
    assert (await anext(stream)).event == "delta"
    await stream.aclose()

    await asyncio.wait_for(runner.cancelled.wait(), timeout=1)
    assert await usage.list_receipts() == ()


@pytest.mark.anyio
async def test_execution_receipt_records_actual_fallback_and_usage():
    service, _, _, usage_store = await _service(
        fail_profiles=("primary",), fallbacks=("approved",)
    )
    _ = [event async for event in service.stream(_request(binding_revision=2))]
    receipts = await usage_store.list_receipts()

    assert len(receipts) == 2
    assert receipts[1].status == "failed"
    assert receipts[1].usage.input_tokens is None
    receipt = receipts[0]
    assert receipt.profile_id == "approved"
    assert receipt.fallback_used is True
    assert receipt.fallback_from_profile_id == "primary"
    assert receipt.usage.cache_read_input_tokens == 20
    assert receipt.stable_prefix_hash
    assert receipt.prompt_cache_key is None
    assert receipt.runtime_kernel_version == "kernel.v1"
    assert receipt.persona_version == "jiao.v1"
    assert receipt.room_policy_version == "group.v1"
    assert receipt.tool_schema_hash == "tools.none"
    assert receipt.summary_version == 1
    assert receipt.compressed_up_to_event_id == 0
    assert receipt.provider_usage_received is True


@pytest.mark.anyio
async def test_binding_revision_mismatch_rejects_before_provider_call():
    service, _, runner, _ = await _service()
    request = GatewayExecutionRequest.from_dict(
        {**_request_to_dict(_request()), "binding_revision": 99}
    )
    with pytest.raises(ValueError, match="binding revision"):
        _ = [event async for event in service.stream(request)]
    assert runner.calls == []


@pytest.mark.anyio
async def test_bedroom_provider_cache_uses_session_partition_not_private_room():
    from model_usage_store import build_cache_namespace

    profiles = InMemoryModelProfileStore()
    await profiles.put_profile(_profile("primary"))
    await profiles.set_actor_default("jiao", "primary")
    context = _ContextBuilder("bedroom:bedroom-1")
    runner = _Runner()
    service = GatewayModelExecutionService(
        profiles=profiles, context_builder=context, provider_runner=runner,
        usage_store=InMemoryModelUsageStore(),
    )
    _ = [event async for event in service.stream(_request())]
    expected = build_cache_namespace(
        actor_id="jiao", conversation_id="bedroom:bedroom-1",
        profile_id="primary", profile_revision=1, execution_mode="group",
        actor_prompt_version="jiao.v1", runtime_kernel_version="kernel.v1",
        room_policy_version="group.v1", tool_schema_hash="tools.none",
        cache_strategy_version="anthropic_prefix_anchored_v1",
    )
    assert runner.calls[0][3] == expected


def _request_to_dict(request):
    return {
        "contract_version": request.contract_version,
        "execution_kind": request.execution_kind,
        "actor_id": request.actor_id,
        "room_id": request.room_id,
        "conversation_id": request.conversation_id,
        "current_event_id": request.current_event_id,
        "generation_request_id": request.generation_request_id,
        "execution_mode": request.execution_mode,
        "fence": {
            "room_id": request.fence.room_id,
            "conversation_id": request.fence.conversation_id,
            "burst_id": request.fence.burst_id,
            "trigger_event_id": request.fence.trigger_event_id,
            "fence_epoch": request.fence.fence_epoch,
            "lease_epoch": request.fence.lease_epoch,
            "orchestrator_instance": request.fence.orchestrator_instance,
        },
        "bedroom_session_id": request.bedroom_session_id,
        "binding_revision": request.binding_revision,
    }


@pytest.mark.anyio
@pytest.mark.parametrize("outcome", ["failed", "cancelled", "protocol_error", "truncated"])
@pytest.mark.parametrize("protocol", ["anthropic_messages", "openai_chat_completions", "openai_responses"])
async def test_real_http_attempt_preserves_usage_received_before_interruption(protocol, outcome):
    import httpx
    from gateway_provider_runner import GatewayProviderRunner
    from provider_transport import PooledHttpTransport
    from test_gateway_provider_runner import Resolver
    service, _, _, usage_store = await _service()
    payload = _profile("primary").to_dict()
    payload.update(protocol=protocol, cache_strategy="no_prompt_cache_v1", requested_cache_ttl=None)
    payload["capabilities"].update(cache_strategies=["no_prompt_cache_v1"], cache_ttls=[])
    # A fresh store keeps the exact known binding revision while changing protocol.
    profiles = InMemoryModelProfileStore()
    await profiles.put_profile(ModelProfile.from_dict(payload))
    await profiles.set_actor_default("jiao", "primary")
    service._profiles = profiles
    packets = {
        "anthropic_messages": b'event: message_start\ndata: {"message":{"usage":{"input_tokens":19}}}\n\n',
        "openai_chat_completions": b'data: {"choices":[],"usage":{"prompt_tokens":19}}\n\n',
        "openai_responses": b'event: response.incomplete\ndata: {"response":{"usage":{"input_tokens":19}}}\n\n',
    }
    if outcome in {"failed", "cancelled"}:
        packets["openai_responses"] = b'event: response.completed\ndata: {"response":{"usage":{"input_tokens":19}}}\n\n'
    elif outcome == "truncated":
        packets["openai_responses"] = b'event: response.output_text.delta\ndata: {"delta":"partial"}\n\n'
    calls = []
    class InterruptedBody(httpx.AsyncByteStream):
        async def __aiter__(self):
            yield packets[protocol]
            if outcome == "cancelled":
                raise asyncio.CancelledError()
            if outcome == "failed":
                raise httpx.ReadError("synthetic interrupted stream")
            if outcome == "protocol_error":
                yield b'event: error\ndata: {"error":{"type":"synthetic_provider_failure"}}\n\n'
    def respond(request):
        calls.append(request)
        return httpx.Response(200, stream=InterruptedBody())
    transport = PooledHttpTransport(client_factory=lambda **kwargs:
        httpx.AsyncClient(transport=httpx.MockTransport(respond), **kwargs))
    service._provider_runner = GatewayProviderRunner(transport=transport, credential_resolver=Resolver())
    try:
        if outcome == "cancelled":
            with pytest.raises(asyncio.CancelledError):
                _ = [event async for event in service.stream(_request())]
        else:
            events = [event async for event in service.stream(_request())]
            assert events[-1].event == "unavailable"
        assert len(calls) == 1
        receipts = await usage_store.list_receipts()
        assert len(receipts) == 1
        assert receipts[0].status == ("cancelled" if outcome == "cancelled" else "failed")
        assert receipts[0].usage.input_tokens == (None if protocol == "openai_responses" and outcome == "truncated" else 19)
        assert receipts[0].usage.output_tokens is None
        assert receipts[0].usage.cache_read_input_tokens is None
        assert receipts[0].generation_request_id == "generation-1"
    finally:
        await transport.close()


@pytest.mark.anyio
async def test_real_http_fallback_has_separate_failed_and_successful_receipts():
    import httpx
    from gateway_provider_runner import GatewayProviderRunner
    from provider_transport import PooledHttpTransport
    from test_gateway_provider_runner import Resolver
    service, _, _, usage_store = await _service(fallbacks=("approved",))
    calls = []
    def respond(request):
        calls.append(request)
        if len(calls) == 1:
            return httpx.Response(503)
        return httpx.Response(200, text='event: message_start\ndata: {"message":{"usage":{"input_tokens":5}}}\n\nevent: content_block_delta\ndata: {"delta":{"text":"hello"}}\n\nevent: message_delta\ndata: {"usage":{"output_tokens":2}}\n\nevent: message_stop\ndata: {}\n\n')
    transport = PooledHttpTransport(client_factory=lambda **kwargs:
        httpx.AsyncClient(transport=httpx.MockTransport(respond), **kwargs))
    service._provider_runner = GatewayProviderRunner(transport=transport, credential_resolver=Resolver())
    try:
        events = [event async for event in service.stream(_request(binding_revision=2))]
        assert sum(event.event == "final" for event in events) == 1
        assert len(calls) == 2
        receipts = await usage_store.list_receipts()
        assert len(receipts) == 2
        by_profile = {receipt.profile_id: receipt for receipt in receipts}
        assert by_profile["primary"].status == "failed"
        assert by_profile["primary"].usage.input_tokens is None
        assert by_profile["approved"].status == "succeeded"
        assert by_profile["approved"].usage.input_tokens == 5
        assert by_profile["approved"].fallback_used
        assert len({receipt.receipt_id for receipt in receipts}) == 2
        assert len({receipt.generation_request_id for receipt in receipts}) == 1
    finally:
        await transport.close()


@pytest.mark.anyio
@pytest.mark.parametrize("blocked", [False, True])
async def test_usage_store_failure_does_not_dispatch_paid_fallback(blocked):
    import httpx
    from gateway_provider_runner import GatewayProviderRunner
    from model_usage_store import UsageRecordingError
    from provider_transport import PooledHttpTransport
    from test_gateway_provider_runner import Resolver
    service, _, _, _ = await _service(fallbacks=("approved",))
    class BrokenStore:
        async def record(self, draft):
            if blocked:
                import anyio
                await anyio.sleep_forever()
            raise RuntimeError("synthetic database unavailable")
    service._usage_store = BrokenStore()
    calls = []
    def respond(request):
        calls.append(request)
        return httpx.Response(200, text='event: message_stop\ndata: {}\n\n')
    transport = PooledHttpTransport(client_factory=lambda **kwargs:
        httpx.AsyncClient(transport=httpx.MockTransport(respond), **kwargs))
    service._provider_runner = GatewayProviderRunner(transport=transport, credential_resolver=Resolver())
    try:
        with pytest.raises(UsageRecordingError):
            _ = [event async for event in service.stream(_request(binding_revision=2))]
        assert len(calls) == 1
    finally:
        await transport.close()


@pytest.mark.anyio
@pytest.mark.parametrize("with_staged_tool", [False, True])
async def test_asgi_level_cancellation_persists_received_usage_in_postgres(isolated_postgres, monkeypatch, with_staged_tool):
    import anyio
    import database
    import httpx
    from gateway_provider_runner import GatewayProviderRunner
    from postgres_model_stores import PostgresModelProfileStore, PostgresModelUsageStore
    from provider_transport import PooledHttpTransport
    from test_gateway_provider_runner import Resolver
    from dataclasses import replace
    import json
    from actor_memory_tools import ActorMemoryExecutionContext, ActorMemoryToolLibrary, PostgresActorMemoryToolStore
    async def factory():
        return isolated_postgres
    monkeypatch.setattr(database, "get_pool", factory)
    monkeypatch.setattr(database, "MEMORY_VECTOR_ENABLED", False)
    await database.init_tables()
    profile = _profile("primary")
    if with_staged_tool:
        profile = replace(profile, capabilities=replace(profile.capabilities, tools=True))
    await PostgresModelProfileStore(factory).put_profile(profile)
    service, builder, _, _ = await _service()
    profiles = InMemoryModelProfileStore()
    await profiles.put_profile(profile)
    await profiles.set_actor_default("jiao", "primary")
    service._profiles = profiles
    tools = None
    if with_staged_tool:
        tools = ActorMemoryToolLibrary(PostgresActorMemoryToolStore(factory))
        original_build = builder.build
        async def build(request, *args, **kwargs):
            return replace(await original_build(request), tool_schema_hash="actor-memory-tools.v1",
                actor_memory_context=ActorMemoryExecutionContext(actor_id="jiao", room_id="room_group_home",
                    conversation_id="conversation-1", generation_request_id="generation-1", source_event_id=101,
                    execution_mode="group", profile_id="primary"))
        builder.build = build
    service._usage_store = PostgresModelUsageStore(factory)
    class CancelledBody(httpx.AsyncByteStream):
        async def __aiter__(self):
            yield b'event: message_start\ndata: {"message":{"usage":{"input_tokens":19}}}\n\n'
            scope.cancel()
            await anyio.sleep(0)
    calls = []
    def respond(request):
        calls.append(request)
        if with_staged_tool and len(calls) == 1:
            tool = {"index": 0, "content_block": {"type": "tool_use", "id": "staged-write", "name": "write_memory",
                "input": {"content": "remember rain", "scope": "group", "memory_type": "fact", "perspective": "jiao",
                    "confidential": False, "importance": 5, "evidence_event_ids": [101]}}}
            return httpx.Response(200, text="event: content_block_start\ndata: " + json.dumps(tool) + "\n\nevent: message_stop\ndata: {}\n\n")
        return httpx.Response(200, stream=CancelledBody())
    transport = PooledHttpTransport(client_factory=lambda **kwargs:
        httpx.AsyncClient(transport=httpx.MockTransport(respond), **kwargs))
    service._provider_runner = GatewayProviderRunner(transport=transport, credential_resolver=Resolver(), memory_tools=tools)
    try:
        with anyio.CancelScope() as scope:
            _ = [event async for event in service.stream(_request())]
        receipts = await service._usage_store.list_receipts()
        assert len(receipts) == (2 if with_staged_tool else 1)
        assert receipts[0].usage.input_tokens == 19
        assert receipts[0].status == "cancelled"
        if with_staged_tool:
            async with isolated_postgres.acquire() as conn:
                stages = await conn.fetch("SELECT status FROM actor_memory_tool_stages")
                assert [row["status"] for row in stages] == ["discarded"]
    finally:
        await transport.close()


@pytest.mark.anyio
@pytest.mark.parametrize("second_fails", [False, True])
async def test_real_http_tool_rounds_have_individual_receipts_and_unknown_totals(second_fails, isolated_postgres, monkeypatch):
    from dataclasses import replace
    import httpx
    from actor_memory_tools import ActorMemoryExecutionContext, ActorMemoryToolLibrary, InMemoryActorMemoryToolStore
    from gateway_provider_runner import GatewayProviderRunner
    from provider_transport import PooledHttpTransport
    from test_gateway_provider_runner import Resolver
    import database
    from postgres_model_stores import PostgresModelProfileStore, PostgresModelUsageStore
    service, builder, _, usage_store = await _service()
    profiles = InMemoryModelProfileStore()
    payload = _profile("primary").to_dict()
    payload["capabilities"]["tools"] = True
    await profiles.put_profile(ModelProfile.from_dict(payload))
    await profiles.set_actor_default("jiao", "primary")
    service._profiles = profiles
    async def factory():
        return isolated_postgres
    monkeypatch.setattr(database, "get_pool", factory)
    monkeypatch.setattr(database, "MEMORY_VECTOR_ENABLED", False)
    await database.init_tables()
    await PostgresModelProfileStore(factory).put_profile(ModelProfile.from_dict(payload))
    usage_store = PostgresModelUsageStore(factory)
    service._usage_store = usage_store
    original = builder.build
    async def build(request, *args, **kwargs):
        return replace(await original(request), tool_schema_hash="actor-memory-tools.v1",
            actor_memory_context=ActorMemoryExecutionContext(actor_id="jiao", room_id="room_group_home",
                conversation_id="conversation-1", generation_request_id="generation-1", source_event_id=101,
                execution_mode="group", profile_id="primary"))
    builder.build = build
    calls = []
    def respond(request):
        calls.append(request)
        if len(calls) == 1:
            return httpx.Response(200, text='event: message_start\ndata: {"message":{"usage":{"input_tokens":7}}}\n\nevent: content_block_start\ndata: {"index":0,"content_block":{"type":"tool_use","id":"read-1","name":"search_memory","input":{"query":"rain"}}}\n\nevent: message_stop\ndata: {}\n\n')
        if second_fails:
            return httpx.Response(503)
        return httpx.Response(200, text='event: content_block_delta\ndata: {"delta":{"text":"hello"}}\n\nevent: message_stop\ndata: {}\n\n')
    transport = PooledHttpTransport(client_factory=lambda **kwargs:
        httpx.AsyncClient(transport=httpx.MockTransport(respond), **kwargs))
    service._provider_runner = GatewayProviderRunner(transport=transport, credential_resolver=Resolver(),
        memory_tools=ActorMemoryToolLibrary(InMemoryActorMemoryToolStore()))
    try:
        events = [event async for event in service.stream(_request())]
        assert len(calls) == 2
        receipts = await usage_store.list_receipts()
        assert len(receipts) == 2
        assert [receipt.status for receipt in receipts] == ["failed" if second_fails else "succeeded", "succeeded"]
        assert [receipt.usage.input_tokens for receipt in receipts] == [None, 7]
        if not second_fails:
            assert next(event for event in events if event.event == "usage").data["input_tokens"] is None
            assert events[-1].data["execution_receipt_id"] == receipts[0].receipt_id
    finally:
        await transport.close()
