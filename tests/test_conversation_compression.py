from cache_strategies import PromptSegment
from types import SimpleNamespace

import pytest

from anchored_history import InMemoryAnchoredHistoryStore
from conversation_partitions import ConversationFact, InMemoryConversationPartitionStore
from model_execution import ProviderChunk, ProviderRunUnavailable
from model_execution_contracts import ProviderUsage
from model_usage_store import InMemoryModelUsageStore
from tests.test_execution_context_builder import _GroupContext, _event, _profile


@pytest.mark.anyio
@pytest.mark.parametrize("automatic", [False, True])
async def test_chinese_summary_has_output_headroom_and_completes_without_retry(automatic):
    from contextlib import asynccontextmanager
    from dataclasses import replace
    from anchored_history import AnchoredHistoryCompactor
    from conversation_compression import ConversationCompressionService
    from execution_context_builder import GatewayExecutionContextBuilder
    from gateway_provider_runner import GatewayProviderRunner
    from tests.test_execution_context_builder import _Relay, _request
    from tests.test_gateway_provider_runner import Resolver, profile

    # Provider boundary fixture: a valid bounded Chinese summary needs more
    # than the old 1,024-token ceiling. Exercise the actual renderer/SSE parser.
    summary = "双方确认保留完整聊天事实，压缩只更新摘要和游标。" * 60
    calls = []

    class Response:
        status_code = 200

        def __init__(self, maximum): self.maximum = maximum

        async def aiter_lines(self):
            import json
            truncated = self.maximum < 1800
            frames = [
                ("message_start", {"message": {"usage": {"input_tokens": 2000}}}),
                ("content_block_delta", {"delta": {"type": "text_delta", "text": summary}}),
                ("message_delta", {"delta": {"stop_reason": "max_tokens" if truncated else "end_turn"},
                                   "usage": {"output_tokens": min(self.maximum, 1800)}}),
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
        async def resolve(self, *args): return SimpleNamespace(primary=profile())

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
        result = await execution.build(request, profile(), resolved_room_id="room_weiwei_jiao",
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
    assert calls[0]["max_tokens"] == 4096
    assert "tools" not in calls[0]
    receipts = await usage.list_receipts()
    assert len(receipts) == 1
    assert receipts[0].usage.output_tokens == 1800
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
    with pytest.raises(ProviderRunUnavailable):
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
