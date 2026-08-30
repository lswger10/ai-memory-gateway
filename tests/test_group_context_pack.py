import ast
import json
from pathlib import Path
import unittest
from unittest.mock import AsyncMock, patch

from fastapi.testclient import TestClient

from memory_policy import AuthorizedMemorySearchResult, CandidateAudit


FIXTURE_ROOT = (
    Path(__file__).resolve().parents[1] / "contracts" / "group-room" / "v1" / "fixtures"
)
PACK_REQUEST = json.loads(
    (FIXTURE_ROOT / "context-pack-request.json").read_text(encoding="utf-8")
)
FACTS = json.loads(
    (FIXTURE_ROOT / "context-facts-response-active.json").read_text(encoding="utf-8")
)


class FakeRelayClient:
    def __init__(self, facts):
        from group_contracts import PublicContextFacts

        self.facts = PublicContextFacts.from_dict(facts)
        self.requests = []

    async def fetch_context_facts(self, request):
        self.requests.append(request.to_dict())
        return self.facts


class FakeContextService:
    async def build(self, request, pack_kind):
        from group_contracts import OpaqueContextPack

        return OpaqueContextPack.from_dict(
            {
                "contract_version": "group-room.v1.0",
                "pack_id": "test-pack",
                "pack_kind": pack_kind,
                "actor_id": request.to_dict()["actor_id"],
                "room_id": request.to_dict()["room_id"],
                "conversation_id": request.to_dict()["conversation_id"],
                "current_event_id": request.to_dict()["current_event_id"],
                "burst_id": request.to_dict()["burst_id"],
                "fence_epoch": request.to_dict()["fence_epoch"],
                "provider_neutral_messages": [{"role": "user", "content": "facts"}],
                "token_budget": 512 if pack_kind == "probe" else 12000,
            }
        )


class GroupContextPackServiceTests(unittest.IsolatedAsyncioTestCase):
    async def test_synthetic_fixture_search_filters_before_candidate_creation(self):
        from group_memory import build_synthetic_scoped_search
        from memory_policy import build_retrieval_policy, room_members

        rows = [
            {"id": 901, "content": "椒椒可见", "scope": "weiwei-jiao", "confidential": False, "source_kind": "synthetic_test"},
            {"id": 902, "content": "老克私密", "scope": "weiwei-laoke", "confidential": True, "source_kind": "synthetic_test"},
            {"id": 903, "content": "群共享", "scope": "group", "confidential": False, "source_kind": "synthetic_test"},
        ]
        search = build_synthetic_scoped_search(rows)
        policy = build_retrieval_policy("jiao", "room_group_home", room_members("room_group_home"))

        result = await search("椒椒", policy, 10)

        self.assertEqual(result.candidate_ids, (901, 903))
        self.assertNotIn(902, result.candidate_ids)
        self.assertEqual([row["id"] for row in result.memories], [901, 903])

    async def test_gateway_fetches_relay_facts_and_builds_actor_pack(self):
        from group_contracts import ContextPackRequest
        from group_memory import GroupContextPackService

        relay = FakeRelayClient(FACTS)
        allowed = {
            "id": 1,
            "content": "椒椒和薇薇的已授权 synthetic memory",
            "scope": "weiwei-laoke",
            "source_kind": "synthetic_test",
        }
        search_result = AuthorizedMemorySearchResult(
            memories=(allowed,), candidate_ids=(1,), audit=CandidateAudit(sql_candidate_ids=(1,))
        )
        search = AsyncMock(return_value=search_result)
        service = GroupContextPackService(relay, search=search)

        pack = await service.build(
            ContextPackRequest.from_dict(PACK_REQUEST), pack_kind="full"
        )

        self.assertEqual([PACK_REQUEST], relay.requests)
        self.assertEqual("laoke", pack.to_dict()["actor_id"])
        serialized = json.dumps(pack.to_dict(), ensure_ascii=False)
        self.assertIn("已授权 synthetic memory", serialized)
        self.assertNotIn("candidate_ids", serialized)
        self.assertNotIn("retrieval", serialized)
        policy = search.await_args.args[1]
        self.assertEqual("laoke", policy.actor_id)
        self.assertEqual("room_group_home", policy.room_id)

    async def test_probe_is_small_and_uses_relay_mentions_without_reparsing(self):
        from group_contracts import ContextPackRequest
        from group_memory import GroupContextPackService

        relay = FakeRelayClient(FACTS)
        service = GroupContextPackService(
            relay,
            search=AsyncMock(
                return_value=AuthorizedMemorySearchResult((), (), CandidateAudit())
            ),
        )
        pack = await service.build(
            ContextPackRequest.from_dict(PACK_REQUEST), pack_kind="probe"
        )
        payload = pack.to_dict()
        self.assertLessEqual(payload["token_budget"], 512)
        self.assertIn("jiao", json.dumps(payload["provider_neutral_messages"]))

    async def test_execution_probe_contract_is_dynamic_tail_not_static_prefix(self):
        from group_contracts import ContextPackRequest
        from group_memory import GroupContextPackService

        relay = FakeRelayClient(FACTS)
        service = GroupContextPackService(
            relay,
            search=AsyncMock(
                return_value=AuthorizedMemorySearchResult((), (), CandidateAudit())
            ),
        )

        probe = await service.build_execution_components(
            ContextPackRequest.from_dict(PACK_REQUEST), pack_kind="probe"
        )
        full = await service.build_execution_components(
            ContextPackRequest.from_dict(PACK_REQUEST), pack_kind="full"
        )

        probe_contract = "Probe response contract"
        self.assertNotIn(probe_contract, "\n".join(probe["static_system"]))
        self.assertIn(probe_contract, "\n".join(probe["dynamic_tail"]))
        self.assertNotIn(probe_contract, "\n".join(full["dynamic_tail"]))

    async def test_gateway_does_not_reparse_names_as_strong_mentions(self):
        from group_contracts import ContextPackRequest
        from group_memory import GroupContextPackService

        facts = json.loads(json.dumps(FACTS))
        facts["trigger_event"]["mentions"] = []
        facts["trigger_event"]["content"] = "我只是自然地提到 jiao 这个名字。"
        relay = FakeRelayClient(facts)
        service = GroupContextPackService(
            relay,
            search=AsyncMock(
                return_value=AuthorizedMemorySearchResult((), (), CandidateAudit())
            ),
        )
        pack = await service.build(
            ContextPackRequest.from_dict(PACK_REQUEST), pack_kind="probe"
        )
        rendered = json.dumps(pack.to_dict(), ensure_ascii=False)
        self.assertNotIn("Relay-normalized strong mentions:", rendered)

    async def test_private_room_uses_the_same_policy_builder(self):
        from group_contracts import ContextPackRequest
        from group_memory import GroupContextPackService

        request = dict(PACK_REQUEST)
        request.update(
            actor_id="jiao",
            room_id="room_weiwei_jiao",
            conversation_id="private-conversation",
        )
        facts = dict(FACTS)
        facts.update(
            room_id="room_weiwei_jiao", conversation_id="private-conversation"
        )
        facts["trigger_event"] = dict(facts["trigger_event"])
        facts["trigger_event"].update(
            room_id="room_weiwei_jiao", conversation_id="private-conversation"
        )
        relay = FakeRelayClient(facts)
        search = AsyncMock(
            return_value=AuthorizedMemorySearchResult((), (), CandidateAudit())
        )
        service = GroupContextPackService(relay, search=search)
        await service.build(ContextPackRequest.from_dict(request), pack_kind="full")
        policy = search.await_args.args[1]
        self.assertIn("weiwei-jiao", policy.confidential_scopes)
        self.assertNotIn("weiwei-laoke", policy.allowed_scopes)

    def test_group_memory_cannot_use_legacy_or_no_policy_search(self):
        source = (Path(__file__).resolve().parents[1] / "group_memory.py").read_text(
            encoding="utf-8"
        )
        tree = ast.parse(source)
        names = {
            alias.name
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom) and node.module == "database"
            for alias in node.names
        }
        self.assertIn("search_authorized_memories", names)
        self.assertNotIn("search_legacy_memories", names)
        self.assertNotIn("search_memories", names)


class GroupContextPackEndpointTests(unittest.TestCase):
    def test_group_context_endpoint_requires_its_service_specific_bearer(self):
        import main

        client = TestClient(main.app)
        headers = {
            "Authorization": "Bearer wrong-key",
            "X-Group-Contract-Version": "group-room.v1.0",
        }
        with patch.dict(
            "os.environ",
            {
                "GATEWAY_GROUP_MEMORY_ENABLED": "true",
                "GROUP_ORCHESTRATOR_SERVICE_KEY": "orchestrator-test-key",
            },
            clear=True,
        ):
            response = client.post(
                "/internal/group/context-packs/full", headers=headers, json=PACK_REQUEST
            )
        self.assertEqual(401, response.status_code)
        self.assertEqual("invalid_service_key", response.json()["error"]["code"])

    def test_group_context_endpoint_is_feature_gated_and_rejects_caller_facts(self):
        import main

        client = TestClient(main.app)
        headers = {
            "Authorization": "Bearer orchestrator-test-key",
            "X-Group-Contract-Version": "group-room.v1.0",
        }
        with patch.dict("os.environ", {}, clear=True):
            disabled = client.post(
                "/internal/group/context-packs/full", headers=headers, json=PACK_REQUEST
            )
        self.assertEqual(404, disabled.status_code)
        self.assertEqual("group_feature_disabled", disabled.json()["error"]["code"])

        polluted = {**PACK_REQUEST, "timeline": [], "scopes": ["group"]}
        with (
            patch.dict(
                "os.environ",
                {
                    "GATEWAY_GROUP_MEMORY_ENABLED": "true",
                    "GROUP_ORCHESTRATOR_SERVICE_KEY": "orchestrator-test-key",
                },
                clear=True,
            ),
            patch.object(main, "_group_context_service", FakeContextService()),
        ):
            rejected = client.post(
                "/internal/group/context-packs/full", headers=headers, json=polluted
            )
        self.assertEqual(422, rejected.status_code)
        self.assertEqual("invalid_group_payload", rejected.json()["error"]["code"])

    def test_authenticated_endpoint_returns_opaque_pack(self):
        import main

        client = TestClient(main.app)
        headers = {
            "Authorization": "Bearer orchestrator-test-key",
            "X-Group-Contract-Version": "group-room.v1.0",
        }
        with (
            patch.dict(
                "os.environ",
                {
                    "GATEWAY_GROUP_MEMORY_ENABLED": "true",
                    "GROUP_ORCHESTRATOR_SERVICE_KEY": "orchestrator-test-key",
                },
                clear=True,
            ),
            patch.object(main, "_group_context_service", FakeContextService()),
        ):
            response = client.post(
                "/internal/group/context-packs/probe", headers=headers, json=PACK_REQUEST
            )
        self.assertEqual(200, response.status_code, response.text)
        self.assertEqual("probe", response.json()["pack_kind"])
        self.assertNotIn("candidate_ids", response.json())


if __name__ == "__main__":
    unittest.main()
