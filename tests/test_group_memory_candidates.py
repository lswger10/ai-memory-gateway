import json
import asyncio
import inspect
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest
from fastapi.testclient import TestClient


FIXTURE_ROOT = (
    Path(__file__).resolve().parents[1] / "contracts" / "group-room" / "v1" / "fixtures"
)
CANDIDATE = json.loads(
    (FIXTURE_ROOT / "memory-candidate-request.json").read_text(encoding="utf-8")
)
AGENT_FINAL = json.loads(
    (FIXTURE_ROOT / "group-event-agent-final.json").read_text(encoding="utf-8")
)
FACTS = json.loads(
    (FIXTURE_ROOT / "context-facts-response-active.json").read_text(encoding="utf-8")
)


def accepted_facts():
    facts = json.loads(json.dumps(FACTS))
    facts["accepted_burst_public_events"] = [json.loads(json.dumps(AGENT_FINAL))]
    facts["recent_public_events"] = [json.loads(json.dumps(AGENT_FINAL))]
    facts["accepted_event_range"] = {"first_event_id": 101, "last_event_id": 201}
    return facts


class FakeRelay:
    def __init__(self, facts=None):
        self.facts = accepted_facts() if facts is None else facts
        self.calls = []

    async def verify_candidate_source(self, request, actor_id):
        self.calls.append((request.to_dict(), actor_id))
        return self.facts


class FakeRepository:
    def __init__(self):
        self.rows = {}
        self.writes = []

    async def persist(self, *, identity, payload_hash, write, auth_context):
        existing = self.rows.get(identity)
        if existing:
            if existing[0] != payload_hash:
                raise ValueError("candidate identity conflict")
            return existing[1]
        memory_id = f"memory-test-{len(self.rows) + 1}"
        self.rows[identity] = (payload_hash, memory_id)
        self.writes.append((write, auth_context))
        return memory_id


def test_duplicate_candidate_returns_same_memory_id():
    from group_contracts import MemoryCandidateRequest
    from group_memory import CandidateIngressService

    repository = FakeRepository()
    service = CandidateIngressService(FakeRelay(), repository.persist)
    request = MemoryCandidateRequest.from_dict(CANDIDATE)

    first = asyncio.run(service.accept("jiao", request))
    second = asyncio.run(service.accept("jiao", request))

    assert first.to_dict() == second.to_dict()
    assert len(repository.writes) == 1


@pytest.mark.parametrize("mutation", ["room_id", "conversation_id", "trigger_event_id"])
def test_candidate_rejects_mismatched_factual_coordinates(mutation):
    from group_contracts import MemoryCandidateRequest
    from group_memory import CandidateIngressService, StaleCandidateError

    body = json.loads(json.dumps(CANDIDATE))
    body["fence"][mutation] = (
        "room_weiwei_jiao"
        if mutation == "room_id"
        else ("wrong" if mutation == "conversation_id" else 999)
    )
    service = CandidateIngressService(FakeRelay(), FakeRepository().persist)

    with pytest.raises(StaleCandidateError):
        asyncio.run(service.accept("jiao", MemoryCandidateRequest.from_dict(body)))


def test_candidate_requires_exact_relay_accepted_final_and_generation():
    from group_contracts import MemoryCandidateRequest
    from group_memory import CandidateIngressService, StaleCandidateError

    facts = accepted_facts()
    facts["accepted_burst_public_events"][0]["provenance"][
        "generation_request_id"
    ] = "different-generation"
    service = CandidateIngressService(FakeRelay(facts), FakeRepository().persist)

    with pytest.raises(StaleCandidateError):
        asyncio.run(service.accept("jiao", MemoryCandidateRequest.from_dict(CANDIDATE)))


def test_candidate_scope_comes_from_visible_evidence_not_agent_payload():
    from group_contracts import MemoryCandidateRequest
    from group_memory import CandidateIngressService

    repository = FakeRepository()
    service = CandidateIngressService(FakeRelay(), repository.persist)
    asyncio.run(service.accept("jiao", MemoryCandidateRequest.from_dict(CANDIDATE)))

    write, auth = repository.writes[0]
    assert write.scope.value == "weiwei-jiao"
    assert write.confidential is False
    assert write.perspective.value == "jiao"
    assert auth.actor_id == "jiao"


def test_candidate_with_only_its_source_final_cannot_widen_to_group_scope():
    """Untrusted candidate text must never turn missing evidence into Group scope."""
    from group_contracts import MemoryCandidateRequest
    from group_memory import CandidateIngressService

    body = json.loads(json.dumps(CANDIDATE))
    body["candidate"]["evidence_event_ids"] = [body["source_event_id"]]
    repository = FakeRepository()
    service = CandidateIngressService(FakeRelay(), repository.persist)

    asyncio.run(service.accept("jiao", MemoryCandidateRequest.from_dict(body)))

    write, _auth = repository.writes[0]
    assert write.scope.value == "weiwei-jiao"


def test_candidate_cannot_launder_private_context_through_three_actor_evidence():
    """Stage-B candidates stay in the submitter's pairwise namespace.

    Public evidence can support a proposal, but model-authored content is not a
    trustworthy classifier for widening it to a namespace visible to another
    actor. Trusted closed-burst extraction remains responsible for shared scope.
    """
    from group_contracts import MemoryCandidateRequest
    from group_memory import CandidateIngressService

    body = json.loads(json.dumps(CANDIDATE))
    facts = accepted_facts()
    extra = json.loads(json.dumps(AGENT_FINAL))
    extra.update({"event_id": 202, "actor_id": "laoke"})
    extra["provenance"]["generation_request_id"] = "laoke-generation"
    facts["recent_public_events"].append(extra)
    body["candidate"]["evidence_event_ids"] = [101, 201, 202]
    repository = FakeRepository()
    service = CandidateIngressService(FakeRelay(facts), repository.persist)

    asyncio.run(service.accept("jiao", MemoryCandidateRequest.from_dict(body)))

    write, _auth = repository.writes[0]
    assert write.scope.value == "weiwei-jiao"


def test_candidate_cannot_cite_an_event_outside_relay_visible_facts():
    from group_contracts import MemoryCandidateRequest
    from group_memory import CandidateIngressService, StaleCandidateError

    body = json.loads(json.dumps(CANDIDATE))
    body["candidate"]["evidence_event_ids"] = [101, 201, 999]
    service = CandidateIngressService(FakeRelay(), FakeRepository().persist)

    with pytest.raises(StaleCandidateError):
        asyncio.run(service.accept("jiao", MemoryCandidateRequest.from_dict(body)))


def test_database_boundary_reasserts_private_candidate_class_and_cross_source_dedupe():
    import database

    source = inspect.getsource(database.persist_group_memory_candidate)
    assert '"jiao": "weiwei-jiao"' in source
    assert '"laoke": "weiwei-laoke"' in source
    assert 'write.source_kind.value != "agent_candidate"' in source
    assert "_persist_or_merge_group_memory" in source


def test_candidate_and_batch_share_one_content_lock_and_dedupe_helper():
    import database

    candidate_source = inspect.getsource(database.persist_group_memory_candidate)
    batch_source = inspect.getsource(database.persist_group_extracted_memory)
    helper_source = inspect.getsource(database._persist_or_merge_group_memory)

    assert "_persist_or_merge_group_memory" in candidate_source
    assert "_persist_or_merge_group_memory" in batch_source
    assert "pg_advisory_xact_lock" in helper_source
    assert helper_source.index("pg_advisory_xact_lock") < helper_source.index(
        "SELECT id, evidence, provenance"
    )


def test_shared_dedupe_helper_merges_evidence_and_source_links():
    import database
    from memory_policy import (
        MemoryScope,
        MemoryType,
        MemoryWrite,
        Perspective,
        SourceKind,
    )

    class FakeConnection:
        def __init__(self):
            self.calls = []

        async def execute(self, sql, *args):
            self.calls.append(("execute", sql, args))

        async def fetchrow(self, sql, *args):
            self.calls.append(("fetchrow", sql, args))
            return {
                "id": 77,
                "evidence": [101],
                "provenance": {
                    "source_links": [
                        {"source_kind": "explicit_user_memory", "source_event_id": 101}
                    ]
                },
            }

    write = MemoryWrite(
        content="same normalized memory",
        scope=MemoryScope.WEIWEI_JIAO,
        memory_type=MemoryType.FACT,
        perspective=Perspective.JIAO,
        confidential=False,
        source_kind=SourceKind.AGENT_CANDIDATE,
        evidence_count=2,
        provenance={
            "source_event_id": 201,
            "generation_request_id": "gen-merge",
            "candidate_index": 0,
            "evidence_event_ids": [101, 201],
        },
    )
    conn = FakeConnection()

    memory_id = asyncio.run(database._persist_or_merge_group_memory(conn, write))

    assert memory_id == 77
    update = next(call for call in conn.calls if "UPDATE memories" in call[1])
    merged_evidence = json.loads(update[2][0])
    merged_provenance = json.loads(update[2][2])
    assert merged_evidence == [101, 201]
    assert {link["source_kind"] for link in merged_provenance["source_links"]} == {
        "explicit_user_memory",
        "agent_candidate",
    }


def candidate_headers(key):
    return {
        "Authorization": f"Bearer {key}",
        "X-Group-Contract-Version": "group-room.v1.0",
    }


def test_candidate_endpoint_is_independently_feature_gated():
    import main

    client = TestClient(main.app)
    with patch.dict("os.environ", {}, clear=True):
        response = client.post(
            "/internal/group/memory-candidates",
            headers=candidate_headers("jiao-key"),
            json=CANDIDATE,
        )
    assert response.status_code == 404
    assert response.json()["error"]["code"] == "group_feature_disabled"


def test_candidate_actor_is_derived_from_key_and_orchestrator_key_is_rejected():
    import main
    from group_contracts import MemoryCandidateReceipt

    class FakeService:
        def __init__(self):
            self.actors = []

        async def accept(self, actor_id, request):
            self.actors.append(actor_id)
            return MemoryCandidateReceipt.from_dict(
                {
                    "contract_version": "group-room.v1.0",
                    "accepted": True,
                    "memory_id": "memory-1",
                    "source_event_id": request.to_dict()["source_event_id"],
                }
            )

    service = FakeService()
    env = {
        "GATEWAY_GROUP_MEMORY_ENABLED": "true",
        "GROUP_AGENT_CANDIDATES_ENABLED": "true",
        "GROUP_JIAO_MEMORY_CANDIDATE_KEY": "jiao-key",
        "GROUP_LAOKE_MEMORY_CANDIDATE_KEY": "laoke-key",
        "GROUP_ORCHESTRATOR_SERVICE_KEY": "orchestrator-key",
    }
    client = TestClient(main.app)
    with patch.dict("os.environ", env, clear=True), patch.object(
        main, "_group_candidate_service", service
    ):
        accepted = client.post(
            "/internal/group/memory-candidates",
            headers=candidate_headers("jiao-key"),
            json=CANDIDATE,
        )
        rejected = client.post(
            "/internal/group/memory-candidates",
            headers=candidate_headers("orchestrator-key"),
            json=CANDIDATE,
        )

    assert accepted.status_code == 200
    assert service.actors == ["jiao"]
    assert rejected.status_code == 403
    assert rejected.json()["error"]["code"] == "principal_not_allowed"


def test_candidate_body_cannot_spoof_actor_or_choose_scope():
    import main

    env = {
        "GATEWAY_GROUP_MEMORY_ENABLED": "true",
        "GROUP_AGENT_CANDIDATES_ENABLED": "true",
        "GROUP_JIAO_MEMORY_CANDIDATE_KEY": "jiao-key",
    }
    client = TestClient(main.app)
    for field, value in (("actor_id", "laoke"), ("scope", "weiwei-laoke")):
        body = {**CANDIDATE, field: value}
        with patch.dict("os.environ", env, clear=True):
            response = client.post(
                "/internal/group/memory-candidates",
                headers=candidate_headers("jiao-key"),
                json=body,
            )
        assert response.status_code == 422
        assert response.json()["error"]["code"] == "invalid_group_payload"
