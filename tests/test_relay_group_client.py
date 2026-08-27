import json
from pathlib import Path
import unittest


FIXTURE_ROOT = (
    Path(__file__).resolve().parents[1] / "contracts" / "group-room" / "v1" / "fixtures"
)


class FakeResponse:
    def __init__(self, status_code, payload):
        self.status_code = status_code
        self._payload = payload

    def json(self):
        return self._payload


class RecordingHttpClient:
    def __init__(self, response):
        self.response = response
        self.calls = []

    async def post(self, url, *, headers, json):
        self.calls.append({"url": url, "headers": headers, "json": json})
        return self.response


class RelayGroupClientTests(unittest.IsolatedAsyncioTestCase):
    async def test_fetch_context_facts_uses_versioned_gateway_principal_contract(self):
        from group_contracts import ContextPackRequest
        from relay_group_client import RelayGroupClient

        request_payload = json.loads(
            (FIXTURE_ROOT / "context-pack-request.json").read_text(encoding="utf-8")
        )
        expected_facts_request = json.loads(
            (FIXTURE_ROOT / "context-facts-request.json").read_text(encoding="utf-8")
        )
        response_payload = json.loads(
            (FIXTURE_ROOT / "context-facts-response-active.json").read_text(encoding="utf-8")
        )
        transport = RecordingHttpClient(FakeResponse(200, response_payload))
        client = RelayGroupClient(
            "http://relay.internal:3011", "gateway-group-test-key", http_client=transport
        )

        facts = await client.fetch_context_facts(ContextPackRequest.from_dict(request_payload))

        self.assertEqual(response_payload, facts.to_dict())
        self.assertEqual(1, len(transport.calls))
        call = transport.calls[0]
        self.assertEqual(
            "http://relay.internal:3011/internal/group/context-facts", call["url"]
        )
        self.assertEqual(expected_facts_request, call["json"])
        self.assertEqual(
            "Bearer gateway-group-test-key", call["headers"]["Authorization"]
        )
        self.assertEqual(
            "group-room.v1.0", call["headers"]["X-Group-Contract-Version"]
        )

    async def test_mismatched_relay_facts_are_rejected_not_guessed(self):
        from group_contracts import ContextPackRequest
        from relay_group_client import RelayFactsMismatch, RelayGroupClient

        request_payload = json.loads(
            (FIXTURE_ROOT / "context-pack-request.json").read_text(encoding="utf-8")
        )
        response_payload = json.loads(
            (FIXTURE_ROOT / "context-facts-response-active.json").read_text(encoding="utf-8")
        )
        response_payload["conversation_id"] = "wrong-conversation"
        transport = RecordingHttpClient(FakeResponse(200, response_payload))
        client = RelayGroupClient("http://relay.internal:3011", "key", http_client=transport)

        with self.assertRaises(RelayFactsMismatch):
            await client.fetch_context_facts(ContextPackRequest.from_dict(request_payload))

    async def test_recent_public_facts_may_come_from_an_older_burst(self):
        from group_contracts import ContextPackRequest
        from relay_group_client import RelayGroupClient

        request_payload = json.loads(
            (FIXTURE_ROOT / "context-pack-request.json").read_text(encoding="utf-8")
        )
        response_payload = json.loads(
            (FIXTURE_ROOT / "context-facts-response-active.json").read_text(encoding="utf-8")
        )
        older = dict(response_payload["trigger_event"])
        older.update(event_id=88, burst_id="older-burst", request_id="older-request")
        response_payload["recent_public_events"] = [older]
        transport = RecordingHttpClient(FakeResponse(200, response_payload))
        client = RelayGroupClient("http://relay.internal:3011", "key", http_client=transport)

        facts = await client.fetch_context_facts(
            ContextPackRequest.from_dict(request_payload)
        )
        self.assertEqual("older-burst", facts.to_dict()["recent_public_events"][0]["burst_id"])

    async def test_context_pack_facts_must_still_have_an_active_fence(self):
        from group_contracts import ContextPackRequest
        from relay_group_client import RelayFactsMismatch, RelayGroupClient

        request_payload = json.loads(
            (FIXTURE_ROOT / "context-pack-request.json").read_text(encoding="utf-8")
        )
        response_payload = json.loads(
            (FIXTURE_ROOT / "context-facts-response-closed.json").read_text(encoding="utf-8")
        )
        transport = RecordingHttpClient(FakeResponse(200, response_payload))
        client = RelayGroupClient("http://relay.internal:3011", "key", http_client=transport)

        with self.assertRaises(RelayFactsMismatch):
            await client.fetch_context_facts(ContextPackRequest.from_dict(request_payload))


if __name__ == "__main__":
    unittest.main()
