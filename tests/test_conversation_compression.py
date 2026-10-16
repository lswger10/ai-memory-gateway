from cache_strategies import PromptSegment
from types import SimpleNamespace

import pytest

from anchored_history import InMemoryAnchoredHistoryStore, SummaryCompressionError
from conversation_partitions import ConversationFact, InMemoryConversationPartitionStore
from model_execution import ProviderChunk, ProviderRunUnavailable
from model_execution_contracts import ProviderUsage
from model_usage_store import InMemoryModelUsageStore
from tests.test_execution_context_builder import _GroupContext, _event, _profile


@pytest.mark.anyio
async def test_responses_summary_uses_bounded_isolated_cache_key_and_preserves_facts():
    import json
    from contextlib import asynccontextmanager
    from conversation_compression import ConversationCompressionService
    from gateway_provider_runner import GatewayProviderRunner
    from model_profiles import ModelProfile
    from tests.test_gateway_provider_runner import Resolver, profile

    payload = profile().to_dict()
    payload.update(protocol="openai_responses", cache_strategy="openai_stable_prefix_v1",
                   requested_cache_ttl=None)
    payload["capabilities"].update(cache_strategies=["openai_stable_prefix_v1"], cache_ttls=[])
    selected = ModelProfile.from_dict(payload)
    calls = []

    class Response:
        status_code = 200

        async def aiter_lines(self):
            yield "event: response.output_text.delta"
            yield "data: " + json.dumps({"type": "response.output_text.delta", "delta": "A bounded summary."})
            yield ""
            yield "event: response.completed"
            yield "data: " + json.dumps({"type": "response.completed", "response": {
                "status": "completed", "output": [], "usage": {"input_tokens": 100, "output_tokens": 8}}})
            yield ""

    class Transport:
        async def open_stream(self, **kwargs):
            body = kwargs["json_body"]
            # Provider schema boundary: the previous namespace + suffix was 74 chars.
            assert len(body["prompt_cache_key"]) <= 64
            assert "tools" not in body
            calls.append(body)
            @asynccontextmanager
            async def opened(): yield Response()
            return opened()

    class Sync:
        async def ensure_relay_synced(self, **kwargs): pass

    class Profiles:
        async def resolve(self, *args): return SimpleNamespace(primary=selected)

    facts, history, usage = InMemoryConversationPartitionStore(), InMemoryAnchoredHistoryStore(), InMemoryModelUsageStore()
    await facts.append_accepted_facts(tuple(ConversationFact.from_relay_event(_event(i)) for i in range(1, 65)))
    builder = SimpleNamespace(group_context=_GroupContext(), conversation_sync=Sync(),
        conversation_store=facts, history_store=history)
    service = ConversationCompressionService(builder=builder, profiles=Profiles(),
        runner=GatewayProviderRunner(transport=Transport(), credential_resolver=Resolver()), usage_store=usage)
    target = dict(actor_id="jiao", room_id="room_weiwei_jiao", conversation_id="conversation-1", current_event_id=64)
    result = await service.compress(**target)
    assert result["compressed_up_to_event_id"] == 16
    assert await facts.count_facts("conversation-1") == 64
    assert len(calls) == 1
    assert calls[0]["prompt_cache_key"] not in history._states
    await facts.append_accepted_facts(tuple(ConversationFact.from_relay_event(_event(i)) for i in range(65, 73)))
    await service.compress(**{**target, "current_event_id": 72})
    assert calls[0]["prompt_cache_key"] != calls[1]["prompt_cache_key"]
    receipts = await usage.list_receipts()
    assert len(receipts) == len(calls) == 2
    assert {r.prompt_cache_key for r in receipts} == {c["prompt_cache_key"] for c in calls}
    assert all(r.status == "succeeded" for r in receipts)


@pytest.mark.anyio
@pytest.mark.parametrize("automatic", [False, True])
@pytest.mark.parametrize("protocol", ["anthropic_messages", "openai_chat_completions"])
async def test_chinese_summary_has_output_headroom_and_completes_without_retry(automatic, protocol):
    from contextlib import asynccontextmanager
    from dataclasses import replace
    from anchored_history import AnchoredHistoryCompactor
    from conversation_compression import ConversationCompressionService
    from execution_context_builder import GatewayExecutionContextBuilder
    from gateway_provider_runner import GatewayProviderRunner
    from model_profiles import ModelProfile
    from tests.test_execution_context_builder import _Relay, _request
    from tests.test_gateway_provider_runner import Resolver, profile

    # Provider boundary fixture: a valid bounded Chinese summary needs more
    # than the old output ceiling including reasoning. Exercise the real SSE parser.
    summary = "双方确认保留完整聊天事实，压缩只更新摘要和游标。" * 60
    calls = []
    selected = profile()
    if protocol == "openai_chat_completions":
        payload = selected.to_dict()
        payload.update(protocol=protocol, model="google/gemini-3.1-pro-preview",
                       cache_strategy="no_prompt_cache_v1", requested_cache_ttl=None)
        payload["capabilities"].update(cache_strategies=["no_prompt_cache_v1"], cache_ttls=[])
        selected = ModelProfile.from_dict(payload)

    class Response:
        status_code = 200

        def __init__(self, maximum): self.maximum = maximum

        async def aiter_lines(self):
            import json
            # Reasoning consumes the same output allowance as the visible summary.
            truncated = self.maximum < 5000
            if protocol == "openai_chat_completions":
                yield "data: " + json.dumps({"choices": [{"delta": {"content": summary},
                    "finish_reason": "length" if truncated else "stop"}],
                    "usage": {"prompt_tokens": 2000, "completion_tokens": min(self.maximum, 5000)}})
                yield ""
                yield "data: [DONE]"
                yield ""
                return
            frames = [
                ("message_start", {"message": {"usage": {"input_tokens": 2000}}}),
                ("content_block_delta", {"delta": {"type": "text_delta", "text": summary}}),
                ("message_delta", {"delta": {"stop_reason": "max_tokens" if truncated else "end_turn"},
                                   "usage": {"output_tokens": min(self.maximum, 5000)}}),
                ("message_stop", {}),
            ]
            for name, data in frames:
                yield f"event: {name}"
                yield "data: " + json.dumps(data)
                yield ""

    class Transport:
        async def open_stream(self, **kwargs):
            calls.append(kwargs["json_body"])
            @asynccontextmanager
            async def opened(): yield Response(kwargs["json_body"]["max_tokens"])
            return opened()

    class Sync:
        async def ensure_relay_synced(self, **kwargs): pass

    class Profiles:
        async def resolve(self, *args): return SimpleNamespace(primary=selected)

    facts, history, usage = InMemoryConversationPartitionStore(), InMemoryAnchoredHistoryStore(), InMemoryModelUsageStore()
    await facts.append_accepted_facts(tuple(ConversationFact.from_relay_event(_event(i)) for i in range(1, 65)))
    builder = SimpleNamespace(group_context=_GroupContext(), conversation_sync=Sync(),
        conversation_store=facts, history_store=history)
    service = ConversationCompressionService(builder=builder, profiles=Profiles(),
        runner=GatewayProviderRunner(transport=Transport(), credential_resolver=Resolver()), usage_store=usage)
    if automatic:
        relay, group = _Relay(), _GroupContext()
        relay.events = [_event(i) for i in range(1, 66)]
        group.relay_client = relay
        execution = GatewayExecutionContextBuilder(group_context=group, bedroom_context=object(),
            conversation_store=facts, history_store=history,
            history_compactor=AnchoredHistoryCompactor(compact_after_events=63))
        execution.summary_service = service
        prior = _request()
        request = replace(prior, current_event_id=65, fence=replace(prior.fence, trigger_event_id=65))
        result = await execution.build(request, selected, resolved_room_id="room_weiwei_jiao",
            resolved_conversation_id="conversation-1")
        assert result.stable_summary == summary
        assert result.compressed_up_to_event_id == 16
        assert len(result.stable_history) == 48
    else:
        result = await service.compress(actor_id="jiao", room_id="room_weiwei_jiao",
            conversation_id="conversation-1", current_event_id=64)
        assert result["summary"] == summary
        assert result["compressed_up_to_event_id"] == 16
    assert await facts.count_facts("conversation-1") == (65 if automatic else 64)
    assert len(calls) == 1
    assert calls[0]["max_tokens"] <= 8192
    assert "tools" not in calls[0]
    receipts = await usage.list_receipts()
    assert len(receipts) == 1
    assert receipts[0].usage.output_tokens == 5000
    assert receipts[0].execution_purpose == "conversation_compression"


@pytest.mark.anyio
@pytest.mark.parametrize("failure_mode", ["exception", "truncated"])
async def test_manual_summary_is_atomic_scoped_and_does_not_delete_facts(failure_mode):
    from conversation_compression import ConversationCompressionService

    class Sync:
        async def ensure_relay_synced(self, **kwargs): pass

    class Profiles:
        async def resolve(self, actor_id, room_id): return SimpleNamespace(primary=_profile())

    class Runner:
        calls = 0
        fail = True
        inputs = []

        async def run(self, **kwargs):
            self.calls += 1
            self.inputs.append(kwargs["context"].dynamic_tail)
            assert kwargs["context"].actor_memory_context is None
            assert kwargs["context"].current_media_references == ()
            if self.fail and failure_mode == "exception":
                raise ProviderRunUnavailable("synthetic failure")
            await kwargs["on_attempt"](f"summary-attempt-{self.calls}", ProviderUsage.from_provider_values(input_tokens=100, output_tokens=10), "succeeded", True, "unverified")
            yield ProviderChunk("final", {"text": "The user and actor discussed synthetic events 1 through 16.",
                "truncated": self.fail and failure_mode == "truncated"})

    facts = InMemoryConversationPartitionStore()
    await facts.append_accepted_facts(tuple(ConversationFact.from_relay_event(_event(i)) for i in range(1, 65)))
    history = InMemoryAnchoredHistoryStore()
    builder = SimpleNamespace(group_context=_GroupContext(), conversation_sync=Sync(),
        conversation_store=facts, history_store=history)
    runner, usage = Runner(), InMemoryModelUsageStore()
    service = ConversationCompressionService(builder=builder, profiles=Profiles(), runner=runner, usage_store=usage)
    target = dict(actor_id="jiao", room_id="room_weiwei_jiao", conversation_id="conversation-1", current_event_id=64)
    with pytest.raises(SummaryCompressionError):
        await service.compress(**target)
    assert all(state.compressed_up_to_event_id == 0 for state in history._states.values())
    runner.fail = False
    result = await service.compress(**target)
    assert result["status"] == "compressed"
    assert result["compressed_up_to_event_id"] == 16
    assert await facts.count_facts("conversation-1") == 64
    assert (await service.compress(**target))["status"] == "unchanged"
    assert runner.calls == 2
    receipts = await usage.list_receipts()
    assert receipts[0].execution_purpose == "conversation_compression"
    assert all(identity["actor_id"] == "jiao" for identity in history._identities.values())
    await facts.append_accepted_facts(tuple(ConversationFact.from_relay_event(_event(i)) for i in range(65, 73)))
    result = await service.compress(**{**target, "current_event_id": 72})
    assert result["compressed_up_to_event_id"] == 24
    import json
    incremental = json.loads(runner.inputs[-1][0].content)
    assert incremental["prior_summary"] == "The user and actor discussed synthetic events 1 through 16."
    assert [event["event_id"] for event in incremental["new_events"]] == list(range(17, 25))
    assert await facts.count_facts("conversation-1") == 72


def test_compression_endpoint_requires_admin_and_rejects_extra_payload(monkeypatch):
    from fastapi.testclient import TestClient
    import main

    monkeypatch.setattr(main, "GATEWAY_SECRET", "summary-admin")
    monkeypatch.setenv("MODEL_PROFILE_MANAGEMENT_ENABLED", "true")
    client = TestClient(main.app)
    assert client.post("/api/conversation-compression", json={}).status_code == 401
    response = client.post("/api/conversation-compression", json={"unexpected": True},
        headers={"X-Gateway-Key": "summary-admin"})
    assert response.status_code == 422


def test_resume_endpoint_is_admin_only_and_cannot_create_summary_state(monkeypatch):
    from unittest.mock import AsyncMock
    from fastapi.testclient import TestClient
    from anchored_history import AnchoredHistoryError
    import main
    class Compression:
        async def resume(self, **kwargs):
            assert kwargs["execution_mode"] == "group"
            raise AnchoredHistoryError("cache state revision conflict")
    monkeypatch.setattr(main, "GATEWAY_SECRET", "summary-admin")
    monkeypatch.setenv("ACTOR_PERSONA_PROXY_SECRET", "persona-only")
    monkeypatch.setenv("MODEL_PROFILE_MANAGEMENT_ENABLED", "true")
    monkeypatch.setattr(main, "_refresh_actor_prompt_store", AsyncMock())
    monkeypatch.setattr(main, "_get_model_execution_service", AsyncMock())
    monkeypatch.setattr(main, "_conversation_compression_service", Compression())
    payload = dict(action="resume", actor_id="jiao", room_id="room_group_home",
                   conversation_id="missing", execution_mode="group", expected_revision=1)
    client = TestClient(main.app)
    assert client.post("/api/conversation-compression", json=payload).status_code == 401
    assert client.post("/api/conversation-compression", json=payload,
        headers={"X-Gateway-Persona-Key": "persona-only"}).status_code == 401
    response = client.post("/api/conversation-compression", json=payload,
        headers={"X-Gateway-Key": "summary-admin"})
    assert response.status_code == 409
    assert response.json() == {"detail": "compression_state_conflict"}


@pytest.mark.anyio
@pytest.mark.parametrize("mode,conversation", [("group", "living-room"), ("bedroom", "bedroom:session-1")])
async def test_explicit_resume_only_reopens_selected_summary_without_provider(mode, conversation):
    import asyncio
    from anchored_history import AnchoredHistoryCompactor, build_summary_namespace
    from conversation_compression import ConversationCompressionService
    store, compactor = InMemoryAnchoredHistoryStore(), AnchoredHistoryCompactor()
    service = ConversationCompressionService(builder=SimpleNamespace(history_store=store, history_compactor=compactor),
        profiles=None, runner=None, usage_store=None)
    identity = dict(actor_id="laoke", room_id="room_group_home" if mode == "group" else "room_bedroom",
                    conversation_id=conversation, execution_mode=mode)
    key = build_summary_namespace(**identity)
    state = await store.get_or_create(key, identity=identity)
    await store.set_compression_error(key, expected_revision=1, reason="summary_empty")
    other = dict(identity, conversation_id="bedroom:session-2" if mode == "bedroom" else "other-group")
    otherkey = build_summary_namespace(**other)
    await store.get_or_create(otherkey, identity=other)
    await store.set_compression_error(otherkey, expected_revision=1, reason="summary_empty")
    events = tuple(dict(event_id=i, actor_id="weiwei", content="synthetic") for i in range(1, 131))
    calls = []
    async def fail(*args):
        calls.append(1)
        raise SummaryCompressionError("summary_empty")
    with pytest.raises(SummaryCompressionError):
        await compactor.maybe_compact(store=store, cache_namespace=key, state=state, events=events, summarize=fail)
    assert calls == []
    await compactor._lock.acquire()
    pending = asyncio.create_task(service.resume(**identity, expected_revision=1))
    try:
        await asyncio.sleep(0)
        assert not pending.done()
        assert (await store.get_or_create(key)).compression_error == "summary_empty"
    finally:
        compactor._lock.release()
    assert await pending == {"status": "resumed"}
    assert calls == [] and (await store.get_or_create(key)).compression_error is None
    assert (await store.get_or_create(otherkey)).compression_error == "summary_empty"
    for _ in range(2):
        with pytest.raises(SummaryCompressionError):
            await compactor.maybe_compact(store=store, cache_namespace=key, state=state, events=events, summarize=fail)
    assert calls == [1]


def test_compression_endpoint_reports_input_budget_not_state_conflict(monkeypatch):
    from unittest.mock import AsyncMock
    from fastapi.testclient import TestClient
    from anchored_history import AnchoredHistoryCompactor
    import main

    class Compression:
        async def compress(self, **kwargs):
            store = InMemoryAnchoredHistoryStore()
            state = await store.get_or_create("summary")
            async def forbidden_summary(*args):
                pytest.fail("oversized input must not reach a provider")
            await AnchoredHistoryCompactor().maybe_compact(store=store, cache_namespace="summary",
                state=state, events=tuple(dict(event_id=i, actor_id="weiwei", content="x" * 3000)
                                         for i in range(1, 131)), summarize=forbidden_summary)
    monkeypatch.setattr(main, "GATEWAY_SECRET", "summary-admin")
    monkeypatch.setenv("MODEL_PROFILE_MANAGEMENT_ENABLED", "true")
    monkeypatch.setattr(main, "_refresh_actor_prompt_store", AsyncMock())
    monkeypatch.setattr(main, "_get_model_execution_service", AsyncMock())
    monkeypatch.setattr(main, "_conversation_compression_service", Compression())
    response = TestClient(main.app).post("/api/conversation-compression",
        json=dict(actor_id="jiao", room_id="room_weiwei_jiao", conversation_id="test", current_event_id=130),
        headers={"X-Gateway-Key": "summary-admin"})
    assert response.status_code == 413
    assert response.json() == {"detail": "summary_input_budget_exceeded"}


@pytest.mark.anyio
@pytest.mark.parametrize("failure", ["empty", "truncated", "oversized", "transport"])
async def test_failed_automatic_summary_pauses_across_turns_and_profile_changes_until_manual_retry(failure):
    from dataclasses import replace
    from conversation_compression import ConversationCompressionService
    from execution_context_builder import GatewayExecutionContextBuilder
    from model_execution import GatewayModelExecutionService
    from tests.test_execution_context_builder import _Relay, _request

    group = _GroupContext()
    group.relay_client = _Relay()
    group.relay_client.events = [_event(i) for i in range(1, 132)]
    builder = GatewayExecutionContextBuilder(group_context=group, bedroom_context=object())
    class Profiles:
        selected = _profile()
        async def resolve(self, *args):
            return SimpleNamespace(primary=self.selected, fallbacks=(), binding_revision=1)
    class Runner:
        calls = 0
        fail = True
        async def run(self, **kwargs):
            self.calls += 1
            assert kwargs["context"].tool_schema_hash == "conversation-summary.v1"
            if self.fail and failure == "transport":
                raise ProviderRunUnavailable("synthetic upstream error")
            text = "" if failure == "empty" else "x" * 4097 if failure == "oversized" else "Complete summary"
            yield ProviderChunk("final", {"text": text if self.fail else "Complete summary",
                "truncated": self.fail and failure == "truncated"})
    profiles, runner, usage = Profiles(), Runner(), InMemoryModelUsageStore()
    summary = ConversationCompressionService(builder=builder, profiles=profiles, runner=runner, usage_store=usage)
    builder.summary_service = summary
    service = GatewayModelExecutionService(profiles=profiles, context_builder=builder,
        provider_runner=runner, usage_store=usage)
    prior = _request()
    for event_id, profile_id in [(131, "gemini"), (132, "claude"), (133, "gpt")]:
        if event_id > 131:
            group.relay_client.events.append(_event(event_id))
        profiles.selected = replace(profiles.selected, profile_id=profile_id)
        req = replace(prior, current_event_id=event_id, fence=replace(prior.fence, trigger_event_id=event_id))
        output = [item async for item in service.stream(req)]
        assert [item.event for item in output] == ["unavailable"]
    assert runner.calls == 1  # No chat, fallback or repeated compression dispatch.
    before = tuple(await builder.conversation_store.list_facts("conversation-1"))
    state = next(iter(builder.history_store._states.values()))
    assert state.compression_error and state.compressed_up_to_event_id == 0
    runner.fail = False
    result = await summary.compress(actor_id="jiao", room_id="room_weiwei_jiao",
        conversation_id="conversation-1", current_event_id=133)
    assert result["status"] == "compressed" and runner.calls == 2
    assert tuple(await builder.conversation_store.list_facts("conversation-1")) == before
    state = next(iter(builder.history_store._states.values()))
    assert state.compression_error is None and state.compressed_up_to_event_id == 85
