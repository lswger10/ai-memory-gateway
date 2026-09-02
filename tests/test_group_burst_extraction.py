import asyncio
import json
from pathlib import Path
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient


FIXTURE_ROOT = (
    Path(__file__).resolve().parents[1] / "contracts" / "group-room" / "v1" / "fixtures"
)
REQUEST = json.loads(
    (FIXTURE_ROOT / "closed-burst-extraction-request.json").read_text(encoding="utf-8")
)
CLOSED_FACTS = json.loads(
    (FIXTURE_ROOT / "context-facts-response-closed.json").read_text(encoding="utf-8")
)


class FakeRelay:
    def __init__(self, facts=None):
        self.facts = json.loads(json.dumps(facts or CLOSED_FACTS))
        self.requests = []

    async def fetch_closed_burst_facts(self, request):
        self.requests.append(request.to_dict())
        return json.loads(json.dumps(self.facts))


class MemoryQueue:
    def __init__(self):
        self.rows = {}

    async def enqueue(self, closed_ref, *, estimated_tokens=0):
        key = (closed_ref["burst_id"], closed_ref["fence_epoch"])
        self.rows.setdefault(
            key,
            {
                "closed_fence": dict(closed_ref),
                "estimated_tokens": estimated_tokens,
                "status": "queued",
            },
        )
        return dict(self.rows[key])

    async def pending(self):
        return [dict(row) for row in self.rows.values() if row["status"] == "queued"]

    async def complete(self, closed_ref):
        key = (closed_ref["burst_id"], closed_ref["fence_epoch"])
        self.rows[key]["status"] = "processed"


def test_active_or_mismatched_fence_is_never_queued():
    from group_contracts import ClosedBurstExtractionRequest
    from group_memory import ClosedBurstExtractionService, UnstableBurstError

    active = json.loads(json.dumps(CLOSED_FACTS))
    active["fence_status"] = "active"
    active["close_reason"] = None
    queue = MemoryQueue()
    service = ClosedBurstExtractionService(FakeRelay(active), queue)

    with pytest.raises(UnstableBurstError):
        asyncio.run(service.enqueue(ClosedBurstExtractionRequest.from_dict(REQUEST)))
    assert queue.rows == {}


@pytest.mark.parametrize("mutation", ["room_id", "conversation_id", "trigger_event_id"])
def test_closed_burst_ref_rejects_mismatched_factual_coordinates(mutation):
    from group_contracts import ClosedBurstExtractionRequest
    from group_memory import ClosedBurstExtractionService, UnstableBurstError

    body = json.loads(json.dumps(REQUEST))
    body["closed_fence"][mutation] = (
        "room_weiwei_jiao"
        if mutation == "room_id"
        else ("wrong-conversation" if mutation == "conversation_id" else 999)
    )
    service = ClosedBurstExtractionService(FakeRelay(), MemoryQueue())
    with pytest.raises(UnstableBurstError):
        asyncio.run(service.enqueue(ClosedBurstExtractionRequest.from_dict(body)))


def test_extraction_uses_relay_closed_facts_not_caller_messages():
    from group_contracts import ClosedBurstExtractionRequest, ContractError
    from group_memory import ClosedBurstExtractionService

    forged = {**REQUEST, "messages": [{"content": "forged"}]}
    with pytest.raises(ContractError):
        ClosedBurstExtractionRequest.from_dict(forged)

    seen = []

    async def extractor(facts):
        seen.append(facts)

    relay = FakeRelay()
    queue = MemoryQueue()
    service = ClosedBurstExtractionService(relay, queue, extractor=extractor)
    request = ClosedBurstExtractionRequest.from_dict(REQUEST)
    asyncio.run(service.enqueue(request))
    asyncio.run(service.process_once())

    assert relay.requests == [REQUEST, REQUEST]
    assert seen[0]["accepted_burst_public_events"][0]["content"] == "我先接住这个话题。"
    assert "forged" not in json.dumps(seen, ensure_ascii=False)


def test_reaction_is_auxiliary_evidence_not_an_extraction_unit():
    from group_contracts import ClosedBurstExtractionRequest
    from group_memory import ClosedBurstExtractionService

    seen = []

    async def extractor(facts):
        seen.append(facts)

    service = ClosedBurstExtractionService(FakeRelay(), MemoryQueue(), extractor=extractor)
    request = ClosedBurstExtractionRequest.from_dict(REQUEST)
    asyncio.run(service.enqueue(request))
    asyncio.run(service.process_once())

    assert service.threshold_burst_count == 1
    assert service.extraction_unit_count == 1
    assert seen[0]["reactions_by_event"] == {"101": {"jiao": "❤️"}}


def test_duplicate_enqueue_keeps_one_cursor_without_copied_text():
    from group_contracts import ClosedBurstExtractionRequest
    from group_memory import ClosedBurstExtractionService

    queue = MemoryQueue()
    service = ClosedBurstExtractionService(FakeRelay(), queue)
    request = ClosedBurstExtractionRequest.from_dict(REQUEST)
    first = asyncio.run(service.enqueue(request))
    second = asyncio.run(service.enqueue(request))

    assert first == second
    assert len(queue.rows) == 1
    serialized = json.dumps(list(queue.rows.values()))
    assert "椒椒" not in serialized
    assert "content" not in serialized


def test_unconfigured_extractor_never_marks_durable_cursor_processed():
    from group_contracts import ClosedBurstExtractionRequest
    from group_memory import ClosedBurstExtractionService, GroupExtractorUnavailable

    queue = MemoryQueue()
    service = ClosedBurstExtractionService(FakeRelay(), queue)
    request = ClosedBurstExtractionRequest.from_dict(REQUEST)
    asyncio.run(service.enqueue(request))

    with pytest.raises(GroupExtractorUnavailable):
        asyncio.run(service.process_once())

    assert next(iter(queue.rows.values()))["status"] == "queued"
    assert service.extraction_unit_count == 0


def test_group_batch_pipeline_validates_scope_and_perspective_before_persisting():
    from group_memory import GroupBatchExtractionPipeline

    writes = []

    async def model_extract(_facts):
        return [
            {
                "content": "椒椒对薇薇承诺会记住兰花。",
                "scope": "weiwei-jiao",
                "memory_type": "fact",
                "perspective": "jiao",
                "confidence": 0.9,
                "evidence_event_ids": [101, 201],
            }
        ]

    async def persist(write, auth_context):
        writes.append((write, auth_context))
        return 1

    pipeline = GroupBatchExtractionPipeline(model_extract, persist=persist)
    asyncio.run(pipeline(CLOSED_FACTS))

    write, auth = writes[0]
    assert write.scope.value == "weiwei-jiao"
    assert write.perspective.value == "jiao"
    assert write.source_kind.value == "chat_extraction"
    assert auth.actor_id == "weiwei"
    assert auth.room_id == "room_group_home"


def test_group_batch_pipeline_rejects_pairwise_scope_with_third_party_evidence():
    from group_memory import GroupBatchExtractionPipeline, ForbiddenMemoryWrite

    facts = json.loads(json.dumps(CLOSED_FACTS))
    other = json.loads(json.dumps(facts["accepted_burst_public_events"][0]))
    other.update({"event_id": 202, "actor_id": "laoke"})
    facts["accepted_burst_public_events"].append(other)

    async def model_extract(_facts):
        return [
            {
                "content": "不应进入椒椒私域的三方内容",
                "scope": "weiwei-jiao",
                "memory_type": "inference",
                "perspective": "jiao",
                "confidence": 0.7,
                "evidence_event_ids": [101, 201, 202],
            }
        ]

    pipeline = GroupBatchExtractionPipeline(model_extract, persist=lambda *_: None)
    with pytest.raises(ForbiddenMemoryWrite):
        asyncio.run(pipeline(facts))


def test_group_extractor_without_key_keeps_queue_retryable():
    import memory_extractor

    with patch.object(memory_extractor, "MEMORY_API_KEY", ""), patch.object(
        memory_extractor, "MEMORY_API_BASE_URL", ""
    ), patch.object(memory_extractor, "MEMORY_MODEL", ""):
        with pytest.raises(memory_extractor.GroupExtractionUnavailable):
            asyncio.run(memory_extractor.extract_group_memories(CLOSED_FACTS))


def test_gateway_lifespan_owns_group_extraction_worker():
    from pathlib import Path

    source = Path("main.py").read_text(encoding="utf-8")
    assert "_group_extraction_worker" in source
    assert "asyncio.create_task(_group_extraction_worker" in source


def test_persisted_threshold_defers_incomplete_batch_without_losing_cursor():
    from group_contracts import ClosedBurstExtractionRequest
    from group_memory import ClosedBurstExtractionService

    seen = []

    async def extractor(facts):
        seen.append(facts)

    queue = MemoryQueue()
    service = ClosedBurstExtractionService(
        FakeRelay(),
        queue,
        extractor=extractor,
        burst_threshold=2,
        token_threshold=0,
        max_wait_seconds=0,
    )
    asyncio.run(service.enqueue(ClosedBurstExtractionRequest.from_dict(REQUEST)))

    assert asyncio.run(service.process_once()) == 0
    assert next(iter(queue.rows.values()))["status"] == "queued"
    assert next(iter(queue.rows.values()))["estimated_tokens"] > 0
    assert seen == []


def test_extraction_progress_schema_persists_counter_without_message_text():
    import database

    sql = database.SCOPED_MEMORY_MIGRATION_SQL
    progress = sql[sql.index("CREATE TABLE IF NOT EXISTS group_burst_extraction_progress") :]
    progress = progress[: progress.index("CREATE TABLE IF NOT EXISTS cold_archive_raw")]
    assert "estimated_tokens" in progress
    assert "content" not in progress
    assert "message" not in progress


def extraction_headers(key="relay-key"):
    return {
        "Authorization": f"Bearer {key}",
        "X-Group-Contract-Version": "group-room.v1.0",
    }


def test_closed_burst_endpoint_is_gated_and_relay_authenticated():
    import main

    client = TestClient(main.app)
    enabled = {
        "GATEWAY_GROUP_MEMORY_ENABLED": "true",
        "GROUP_BURST_EXTRACTION_ENABLED": "true",
        "GROUP_RELAY_SERVICE_KEY": "relay-key",
    }
    with patch.dict("os.environ", {}, clear=True):
        disabled = client.post(
            "/internal/group/extraction/closed-bursts",
            headers=extraction_headers(),
            json=REQUEST,
        )
    with patch.dict("os.environ", enabled, clear=True):
        forbidden = client.post(
            "/internal/group/extraction/closed-bursts",
            headers=extraction_headers("actor-key"),
            json=REQUEST,
        )

    assert disabled.status_code == 404
    assert disabled.json()["error"]["code"] == "group_feature_disabled"
    assert forbidden.status_code == 403
    assert forbidden.json()["error"]["code"] == "principal_not_allowed"
