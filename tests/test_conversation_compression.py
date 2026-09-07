from types import SimpleNamespace

import pytest

from anchored_history import InMemoryAnchoredHistoryStore
from conversation_partitions import ConversationFact, InMemoryConversationPartitionStore
from model_execution import ProviderChunk, ProviderRunUnavailable
from model_execution_contracts import ProviderUsage
from model_usage_store import InMemoryModelUsageStore
from tests.test_execution_context_builder import _GroupContext, _event, _profile


@pytest.mark.anyio
async def test_manual_summary_is_atomic_scoped_and_does_not_delete_facts():
    from conversation_compression import ConversationCompressionService

    class Sync:
        async def ensure_relay_synced(self, **kwargs): pass

    class Profiles:
        async def resolve(self, actor_id, room_id): return SimpleNamespace(primary=_profile())

    class Runner:
        calls = 0
        fail = True

        async def run(self, **kwargs):
            self.calls += 1
            assert kwargs["context"].actor_memory_context is None
            assert kwargs["context"].current_media_references == ()
            if self.fail:
                raise ProviderRunUnavailable("synthetic failure")
            await kwargs["on_attempt"]("summary-attempt", ProviderUsage.from_provider_values(input_tokens=100, output_tokens=10), "succeeded", True, "unverified")
            yield ProviderChunk("final", {"text": "The user and actor discussed synthetic events 1 through 16."})

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
