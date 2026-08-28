import json
import unittest
from pathlib import Path
from unittest.mock import AsyncMock

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


class Relay:
    async def fetch_context_facts(self, _request):
        from group_contracts import PublicContextFacts

        return PublicContextFacts.from_dict(FACTS)


class RelationshipSummaryTests(unittest.IsolatedAsyncioTestCase):
    async def test_summary_selection_never_loads_forbidden_pairwise_rows(self):
        from group_contracts import ContextPackRequest
        from group_memory import GroupContextPackService, build_scoped_summary_search

        rows = (
            {"id": 1001, "scope": "weiwei-laoke", "content": "老克与薇薇允许的关系摘要", "confidential": False},
            {"id": 1002, "scope": "weiwei-jiao", "content": "椒椒与薇薇不应给老克看", "confidential": False},
            {"id": 1003, "scope": "jiao-laoke", "content": "两位 agent 的共同关系摘要", "confidential": False},
            {"id": 1004, "scope": "group", "content": "客厅摘要", "confidential": False},
            {"id": 1005, "scope": "weiwei-laoke", "content": "私聊 confidential 摘要", "confidential": True},
        )
        summary_search = build_scoped_summary_search(rows)
        service = GroupContextPackService(
            Relay(),
            search=AsyncMock(
                return_value=AuthorizedMemorySearchResult((), (), CandidateAudit())
            ),
            summary_search=summary_search,
        )
        pack = await service.build(
            ContextPackRequest.from_dict(PACK_REQUEST), pack_kind="full"
        )
        rendered = json.dumps(pack.to_dict(), ensure_ascii=False)
        self.assertIn("老克与薇薇允许的关系摘要", rendered)
        self.assertIn("两位 agent 的共同关系摘要", rendered)
        self.assertIn("客厅摘要", rendered)
        self.assertNotIn("椒椒与薇薇不应给老克看", rendered)
        self.assertNotIn("私聊 confidential 摘要", rendered)
        self.assertEqual((1001, 1003, 1004), service.last_summary_candidate_ids)

    async def test_probe_pack_does_not_load_persisted_relationship_summaries(self):
        from group_contracts import ContextPackRequest
        from group_memory import GroupContextPackService

        summaries = AsyncMock(return_value=())
        service = GroupContextPackService(
            Relay(),
            search=AsyncMock(
                return_value=AuthorizedMemorySearchResult((), (), CandidateAudit())
            ),
            summary_search=summaries,
        )
        await service.build(
            ContextPackRequest.from_dict(PACK_REQUEST), pack_kind="probe"
        )
        summaries.assert_not_awaited()

    async def test_relationship_summary_does_not_mutate_actor_identity(self):
        from group_memory import ScopeAwareMemoryService

        service = ScopeAwareMemoryService(
            identity_profiles={"jiao": {"prompt_identity": "JIAO-v1"}}
        )
        before = await service.identity_profile("jiao")
        await service.refresh_relationship_summary(
            "weiwei-jiao",
            "椒椒和薇薇的关系摘要",
            evidence_event_ids=(101, 102),
            confidential=False,
        )
        self.assertEqual(before, await service.identity_profile("jiao"))

    def test_summary_schema_is_separate_from_actor_identity_profiles(self):
        import database

        sql = database.SCOPED_MEMORY_MIGRATION_SQL
        self.assertIn("CREATE TABLE IF NOT EXISTS relationship_summaries", sql)
        self.assertIn("evidence_event_ids", sql)
        self.assertNotIn("relationship_summary", sql.split("actor_identity_profiles", 1)[1].split(");", 1)[0])


if __name__ == "__main__":
    unittest.main()
