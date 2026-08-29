import unittest
from unittest.mock import AsyncMock

from memory_policy import AuthorizedMemorySearchResult, CandidateAudit


def facts(room="room_weiwei_jiao", actor="jiao", policy="summary-only"):
    return {
        "contract_version": "bedroom-room.v1.0",
        "session": {
            "contract_version": "bedroom-room.v1.0",
            "bedroom_session_id": "bedroom-1",
            "room_id": room,
            "conversation_id": "private-conversation",
            "actor_id": actor,
            "retention_policy": policy,
            "scene_context": "雨夜暖灯",
            "status": "active",
            "turn_epoch": 1,
            "created_at": "2026-08-29T12:00:00Z",
            "ended_at": None,
            "retention_receipt": None,
        },
        "turns": [
            {"turn_id": 1, "turn_epoch": 1, "actor_id": "weiwei", "role": "human", "text": "椒椒，我在", "request_id": "h1", "created_at": "now", "provenance_json": None},
        ],
    }


class FakeRelay:
    def __init__(self, value):
        self.value = value
    async def fetch_bedroom_facts(self, session_id):
        return self.value


class FakeRepo:
    def __init__(self):
        self.summaries = []
        self.archives = []
    async def persist_summary(self, **kwargs):
        self.summaries.append(kwargs)
        return "bedroom-summary:1"
    async def persist_archive(self, **kwargs):
        self.archives.append(kwargs)
        return "bedroom-archive:1"


class BedroomMemoryTests(unittest.IsolatedAsyncioTestCase):
    async def test_private_pack_uses_actor_prompt_scene_and_private_acl(self):
        from bedroom_memory import BedroomContextPackService, BedroomPackRequest

        search = AsyncMock(
            return_value=AuthorizedMemorySearchResult(
                memories=({"id": 1, "content": "椒椒私密", "scope": "weiwei-jiao"},),
                candidate_ids=(1,),
                audit=CandidateAudit(sql_candidate_ids=(1,)),
            )
        )
        service = BedroomContextPackService(
            FakeRelay(facts()), search=search, summary_search=AsyncMock(return_value=())
        )
        pack = await service.build(
            BedroomPackRequest.from_dict(
                {"contract_version":"bedroom-room.v1.0","bedroom_session_id":"bedroom-1","turn_id":1,"turn_epoch":1,"actor_id":"jiao"}
            )
        )
        system = pack.to_dict()["provider_neutral_messages"][0]["content"]
        self.assertIn("雨夜暖灯", system)
        self.assertIn("椒椒私密", system)
        self.assertNotIn("LAOKE", system)
        policy = search.await_args.args[1]
        self.assertEqual(policy.allowed_scopes, ("weiwei-jiao", "group"))
        self.assertEqual(policy.confidential_scopes, ("weiwei-jiao",))

    async def test_laoke_pack_cannot_reuse_jiao_actor_coordinate(self):
        from bedroom_memory import BedroomContextPackService, BedroomContractError, BedroomPackRequest

        service = BedroomContextPackService(
            FakeRelay(facts("room_weiwei_laoke", "laoke")),
            search=AsyncMock(return_value=AuthorizedMemorySearchResult((), (), CandidateAudit())),
            summary_search=AsyncMock(return_value=()),
        )
        with self.assertRaises(BedroomContractError):
            await service.build(BedroomPackRequest("bedroom-1", 1, 1, "jiao"))

    async def test_summary_and_archive_receipts_are_scope_bound_and_idempotency_ready(self):
        from bedroom_memory import BedroomRetentionService

        repo = FakeRepo()
        service = BedroomRetentionService(repo)
        summary = await service.persist(facts(policy="summary-only"))
        archive = await service.persist(facts(policy="full-bedroom-archive"))
        self.assertEqual(summary["receipt_id"], "bedroom-summary:1")
        self.assertEqual(repo.summaries[0]["scope"], "weiwei-jiao")
        self.assertEqual(archive["receipt_id"], "bedroom-archive:1")
        self.assertEqual(repo.archives[0]["scope"], "weiwei-jiao")


if __name__ == "__main__":
    unittest.main()
