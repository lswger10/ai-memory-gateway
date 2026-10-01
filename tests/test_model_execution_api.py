from fastapi.testclient import TestClient
import asyncio
from types import SimpleNamespace

import pytest

import main
from model_execution import ExecutionStreamEvent
from actor_memory_tools import ActorMemoryToolLibrary, InMemoryActorMemoryToolStore
from conversation_partitions import InMemoryConversationPartitionStore
from execution_context_builder import GatewayExecutionContextBuilder
from relay_group_client import RelayGroupClient
from test_conversation_sync import FakeRelay, relay_event


def _accepted_runtime(monkeypatch, events=(), bedroom=None):
    facts = FakeRelay(events, bedroom)
    relay = RelayGroupClient("http://relay.invalid", "test-key")
    monkeypatch.setattr(relay, "fetch_model_history_facts", facts.fetch_model_history_facts)
    monkeypatch.setattr(relay, "fetch_bedroom_facts", facts.fetch_bedroom_facts)
    store = InMemoryConversationPartitionStore()
    builder = GatewayExecutionContextBuilder(
        group_context=SimpleNamespace(relay_client=relay),
        bedroom_context=SimpleNamespace(relay_client=relay),
        conversation_store=store,
    )
    monkeypatch.setenv("MODEL_EXECUTION_ENABLED", "true")
    monkeypatch.setenv("GROUP_ORCHESTRATOR_SERVICE_KEY", "service-key")
    monkeypatch.setattr(main, "_actor_memory_relay", relay)
    monkeypatch.setattr(main, "_model_context_builder", builder)
    monkeypatch.setattr(main, "_model_execution_service", object())
    monkeypatch.setattr(main, "_actor_memory_tools", ActorMemoryToolLibrary(InMemoryActorMemoryToolStore()))
    return store


def _payload():
    return {
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
        "binding_revision": 1,
    }


def _headers():
    return {
        "Authorization": "Bearer service-key",
        "X-Gateway-Execution-Version": "gateway-model-execution.v1.0",
    }


def test_model_execution_endpoint_is_disabled_by_default(monkeypatch):
    monkeypatch.delenv("MODEL_EXECUTION_ENABLED", raising=False)
    response = TestClient(main.app).post(
        "/internal/model-execution/stream", json=_payload(), headers=_headers()
    )
    assert response.status_code == 404
    assert response.json()["error"]["code"] == "model_execution_disabled"


def test_model_execution_endpoint_requires_orchestrator_principal(monkeypatch):
    monkeypatch.setenv("MODEL_EXECUTION_ENABLED", "true")
    monkeypatch.setenv("GROUP_ORCHESTRATOR_SERVICE_KEY", "service-key")
    response = TestClient(main.app).post(
        "/internal/model-execution/stream",
        json=_payload(),
        headers={"X-Gateway-Execution-Version": "gateway-model-execution.v1.0"},
    )
    assert response.status_code == 403
    assert response.json()["error"]["code"] == "principal_not_allowed"


def test_model_execution_endpoint_streams_normalized_events(monkeypatch):
    class Service:
        async def stream(self, request):
            yield ExecutionStreamEvent("delta", {"text": "hello"})
            yield ExecutionStreamEvent(
                "done",
                {
                    "generation_request_id": request.generation_request_id,
                    "execution_receipt_id": "receipt-1",
                },
            )

    monkeypatch.setenv("MODEL_EXECUTION_ENABLED", "true")
    monkeypatch.setenv("GROUP_ORCHESTRATOR_SERVICE_KEY", "service-key")
    monkeypatch.setattr(main, "_model_execution_service", Service())
    response = TestClient(main.app).post(
        "/internal/model-execution/stream", json=_payload(), headers=_headers()
    )
    assert response.status_code == 200
    assert "event: delta" in response.text
    assert '"execution_receipt_id": "receipt-1"' in response.text


def test_memory_mutations_commit_only_after_relay_accepted_final(monkeypatch):
    event = relay_event(202, actor_id="jiao", conversation_id="conversation-1")
    event["provenance"] = {"generation_request_id": "generation-1"}
    _accepted_runtime(monkeypatch, (event,))

    store = InMemoryActorMemoryToolStore()
    tools = ActorMemoryToolLibrary(store)
    context = main.ActorMemoryExecutionContext(
        actor_id="jiao", room_id="room_group_home", conversation_id="conversation-1",
        generation_request_id="generation-1", source_event_id=101,
        execution_mode="group", profile_id="profile-1",
    )
    asyncio.run(tools.call(context, "tool-1", "write_memory", {
        "content": "accepted only", "scope": "group", "memory_type": "fact",
        "perspective": "jiao", "confidential": False, "importance": 7,
        "evidence_event_ids": [101],
    }))
    monkeypatch.setenv("MODEL_EXECUTION_ENABLED", "true")
    monkeypatch.setenv("GROUP_ORCHESTRATOR_SERVICE_KEY", "service-key")
    monkeypatch.setattr(main, "_actor_memory_tools", tools)

    response = TestClient(main.app).post(
        "/internal/model-execution/memory/accepted",
        json={"execution": _payload(), "accepted_event_id": 202}, headers=_headers(),
    )

    assert response.status_code == 200
    assert response.json()["status"] == "committed"
    assert len(asyncio.run(store.list_active())) == 1


@pytest.mark.parametrize("actor,room,mode", [
    ("jiao", "room_weiwei_jiao", "private"),
    ("laoke", "room_weiwei_laoke", "private"),
    ("jiao", "room_group_home", "group"),
    ("laoke", "room_group_home", "group"),
])
def test_accepted_final_persists_complete_history_without_another_generation(monkeypatch, actor, room, mode):
    events = [relay_event(i, actor_id="weiwei" if i == 1 else actor,
                          room_id=room, conversation_id="conversation-1") for i in (1, 2)]
    events[1]["provenance"] = {"generation_request_id": "generation-1"}
    store = _accepted_runtime(monkeypatch, events)
    execution = _payload()
    execution.update(actor_id=actor, room_id=room, execution_mode=mode, current_event_id=1)
    execution["fence"].update(room_id=room, trigger_event_id=1)
    for _ in range(2):  # Replayed acceptance must not duplicate either fact.
        response = TestClient(main.app).post(
            "/internal/model-execution/memory/accepted",
            json={"execution": execution, "accepted_event_id": 2}, headers=_headers(),
        )
        assert response.status_code == 200
        facts = asyncio.run(store.list_facts("conversation-1"))
        assert [(f.source_event_id, f.actor_id, f.content) for f in facts] == [
            (1, "weiwei", "message-1"), (2, actor, "message-2"),
        ]
        assert asyncio.run(store.synced_through_event_id("conversation-1")) == 2


@pytest.mark.parametrize("policy", ["no-retention", "summary-only", "full-bedroom-archive"])
def test_bedroom_accepted_final_stays_in_its_session_partition(monkeypatch, policy):
    bedroom = {
        "session": {"bedroom_session_id": "bedroom-1", "actor_id": "laoke",
                    "room_id": "room_weiwei_laoke", "conversation_id": "private-laoke",
                    "status": "active", "retention_policy": policy},
        "turns": [
            {"turn_id": 1, "actor_id": "weiwei", "role": "human", "text": "scene question",
             "request_id": "b1", "created_at": "2026-09-30T00:00:01Z", "provenance_json": None},
            {"turn_id": 2, "actor_id": "laoke", "role": "agent", "text": "scene final",
             "request_id": "b2", "created_at": "2026-09-30T00:00:02Z",
             "provenance_json": {"generation_request_id": "generation-1"}},
        ],
    }
    store = _accepted_runtime(monkeypatch, bedroom=bedroom)
    execution = _payload()
    for key in ("room_id", "conversation_id", "fence"):
        execution.pop(key)
    execution.update(actor_id="laoke", execution_mode="bedroom", current_event_id=1,
                     bedroom_session_id="bedroom-1", bedroom_turn_epoch=1)
    response = TestClient(main.app).post(
        "/internal/model-execution/memory/accepted",
        json={"execution": execution, "accepted_event_id": 2}, headers=_headers(),
    )
    assert response.status_code == 200
    facts = asyncio.run(store.list_facts("bedroom:bedroom-1"))
    assert [f.content for f in facts] == ["scene question", "scene final"]
    assert all(f.retention_policy == policy for f in facts)
    assert asyncio.run(store.list_facts("private-laoke")) == ()


@pytest.mark.parametrize("invalid", ["missing", "wrong_actor", "unaccepted_generation"])
def test_unverified_final_never_enters_cognitive_history(monkeypatch, invalid):
    event = relay_event(2, actor_id="laoke" if invalid == "wrong_actor" else "jiao",
                        conversation_id="conversation-1")
    event["provenance"] = {"generation_request_id": "other" if invalid == "unaccepted_generation" else "generation-1"}
    store = _accepted_runtime(monkeypatch, () if invalid == "missing" else (event,))
    response = TestClient(main.app).post(
        "/internal/model-execution/memory/accepted",
        json={"execution": _payload(), "accepted_event_id": 2}, headers=_headers(),
    )
    assert response.status_code == 409
    assert asyncio.run(store.list_facts("conversation-1")) == ()


def test_acceptance_reports_incomplete_history_instead_of_success(monkeypatch):
    event = relay_event(2, actor_id="jiao", conversation_id="conversation-1")
    event["provenance"] = {"generation_request_id": "generation-1"}
    store = _accepted_runtime(monkeypatch)

    async def incomplete_history(**coordinates):
        return (event,) if coordinates["after_event_id"] == 1 else ()

    monkeypatch.setattr(main._actor_memory_relay, "fetch_model_history_facts", incomplete_history)
    response = TestClient(main.app).post(
        "/internal/model-execution/memory/accepted",
        json={"execution": _payload(), "accepted_event_id": 2}, headers=_headers(),
    )
    assert response.status_code == 503
    assert response.json()["error"]["code"] == "conversation_sync_incomplete"
    assert asyncio.run(store.list_facts("conversation-1")) == ()


def test_memory_mutations_can_be_discarded_without_relay_lookup(monkeypatch):
    class Tools:
        async def discard(self, context):
            return {"status": "discarded", "generation_request_id": context.generation_request_id}

    monkeypatch.setenv("MODEL_EXECUTION_ENABLED", "true")
    monkeypatch.setenv("GROUP_ORCHESTRATOR_SERVICE_KEY", "service-key")
    monkeypatch.setattr(main, "_actor_memory_tools", Tools())
    response = TestClient(main.app).post(
        "/internal/model-execution/memory/discarded",
        json={"execution": _payload()}, headers=_headers(),
    )
    assert response.status_code == 200
    assert response.json()["status"] == "discarded"
