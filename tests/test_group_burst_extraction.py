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

    async def enqueue(self, closed_ref):
        key = (closed_ref["burst_id"], closed_ref["fence_epoch"])
        self.rows.setdefault(key, {"closed_fence": dict(closed_ref), "status": "queued"})
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
