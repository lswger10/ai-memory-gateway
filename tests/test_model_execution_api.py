from fastapi.testclient import TestClient

import main
from model_execution import ExecutionStreamEvent
from actor_memory_tools import ActorMemoryToolLibrary, InMemoryActorMemoryToolStore


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
    class Relay:
        async def verify_accepted_execution_final(self, **coordinates):
            assert coordinates["accepted_event_id"] == 202
            assert coordinates["generation_request_id"] == "generation-1"
            return {"event_id": 202}

    store = InMemoryActorMemoryToolStore()
    tools = ActorMemoryToolLibrary(store)
    context = main.ActorMemoryExecutionContext(
        actor_id="jiao", room_id="room_group_home", conversation_id="conversation-1",
        generation_request_id="generation-1", source_event_id=101,
        execution_mode="group", profile_id="profile-1",
    )
    import asyncio
    asyncio.run(tools.call(context, "tool-1", "write_memory", {
        "content": "accepted only", "scope": "group", "memory_type": "fact",
        "perspective": "jiao", "confidential": False, "importance": 7,
        "evidence_event_ids": [101],
    }))
    monkeypatch.setenv("MODEL_EXECUTION_ENABLED", "true")
    monkeypatch.setenv("GROUP_ORCHESTRATOR_SERVICE_KEY", "service-key")
    monkeypatch.setattr(main, "_actor_memory_tools", tools)
    monkeypatch.setattr(main, "_actor_memory_relay", Relay())

    response = TestClient(main.app).post(
        "/internal/model-execution/memory/accepted",
        json={"execution": _payload(), "accepted_event_id": 202}, headers=_headers(),
    )

    assert response.status_code == 200
    assert response.json()["status"] == "committed"
    assert len(asyncio.run(store.list_active())) == 1


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
