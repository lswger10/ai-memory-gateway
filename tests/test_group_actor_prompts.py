import json
from pathlib import Path
import unittest
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


class FakeRelayClient:
    def __init__(self, facts):
        from group_contracts import PublicContextFacts

        self.facts = PublicContextFacts.from_dict(facts)

    async def fetch_context_facts(self, _request):
        return self.facts


def request_and_facts(actor_id):
    request = dict(PACK_REQUEST)
    request["actor_id"] = actor_id
    facts = json.loads(json.dumps(FACTS))
    return request, facts


class GroupActorPromptTests(unittest.IsolatedAsyncioTestCase):
    def test_prompt_profiles_are_actor_versioned_and_provider_independent(self):
        from actor_prompt_profiles import load_actor_prompt_profiles

        profiles = load_actor_prompt_profiles()
        self.assertEqual({"jiao", "laoke"}, set(profiles))
        self.assertNotEqual(profiles["jiao"].prompt_text, profiles["laoke"].prompt_text)
        self.assertTrue(profiles["jiao"].prompt_version.startswith("jiao."))
        self.assertTrue(profiles["laoke"].prompt_version.startswith("laoke."))
        serialized = json.dumps(
            {actor: profile.to_dict() for actor, profile in profiles.items()},
            ensure_ascii=False,
        )
        self.assertNotIn("provider", serialized.lower())
        self.assertNotIn("model", serialized.lower())

    async def test_jiao_and_laoke_packs_contain_only_their_actor_prompt(self):
        from actor_prompt_profiles import load_actor_prompt_profiles
        from group_contracts import ContextPackRequest
        from group_memory import GroupContextPackService

        profiles = load_actor_prompt_profiles()
        memory = {
            "id": 1,
            "content": "authorized relationship fact",
            "scope": "group",
            "source_kind": "synthetic_test",
        }
        search = AsyncMock(
            return_value=AuthorizedMemorySearchResult(
                memories=(memory,), candidate_ids=(1,), audit=CandidateAudit()
            )
        )
        system_by_actor = {}
        for actor_id in ("jiao", "laoke"):
            request, facts = request_and_facts(actor_id)
            service = GroupContextPackService(
                FakeRelayClient(facts), search=search, prompt_profiles=profiles
            )
            pack = await service.build(
                ContextPackRequest.from_dict(request), pack_kind="full"
            )
            system = pack.to_dict()["provider_neutral_messages"][0]
            self.assertEqual("system", system["role"])
            system_by_actor[actor_id] = system["content"]

        self.assertIn(profiles["jiao"].prompt_text, system_by_actor["jiao"])
        self.assertNotIn(profiles["laoke"].prompt_text, system_by_actor["jiao"])
        self.assertIn(profiles["laoke"].prompt_text, system_by_actor["laoke"])
        self.assertNotIn(profiles["jiao"].prompt_text, system_by_actor["laoke"])
        for content in system_by_actor.values():
            self.assertIn("Group runtime kernel", content)
            self.assertIn("Room policy", content)
            self.assertIn("authorized relationship fact", content)

    async def test_probe_pack_also_uses_actor_prompt_without_expanded_history(self):
        from actor_prompt_profiles import load_actor_prompt_profiles
        from group_contracts import ContextPackRequest
        from group_memory import GroupContextPackService

        profiles = load_actor_prompt_profiles()
        request, facts = request_and_facts("jiao")
        service = GroupContextPackService(
            FakeRelayClient(facts),
            search=AsyncMock(
                return_value=AuthorizedMemorySearchResult((), (), CandidateAudit())
            ),
            prompt_profiles=profiles,
        )
        pack = await service.build(
            ContextPackRequest.from_dict(request), pack_kind="probe"
        )
        messages = pack.to_dict()["provider_neutral_messages"]
        self.assertEqual("system", messages[0]["role"])
        self.assertIn(profiles["jiao"].prompt_text, messages[0]["content"])
        self.assertNotIn(profiles["laoke"].prompt_text, messages[0]["content"])


if __name__ == "__main__":
    unittest.main()
