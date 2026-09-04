import asyncio

import pytest

from actor_memory_tools import (
    ACTOR_MEMORY_TOOL_NAMES,
    ActorMemoryExecutionContext,
    ActorMemoryToolLibrary,
    InMemoryActorMemoryToolStore,
    actor_memory_tool_definitions,
)


def context(actor="jiao", room="room_weiwei_jiao", generation="gen-1", profile="profile-a"):
    return ActorMemoryExecutionContext(
        actor_id=actor,
        room_id=room,
        conversation_id=f"conversation-{room}",
        generation_request_id=generation,
        source_event_id=101,
        execution_mode="group" if room == "room_group_home" else "private",
        profile_id=profile,
    )


def test_all_actor_memory_tools_are_provider_neutral_and_never_accept_actor_identity():
    definitions = actor_memory_tool_definitions()
    assert {item["name"] for item in definitions} == ACTOR_MEMORY_TOOL_NAMES
    for item in definitions:
        assert "actor_id" not in item["input_schema"].get("properties", {})

    by_name = {item["name"]: item for item in definitions}
    expected = {"weiwei", "jiao", "laoke", "shared"}
    assert set(by_name["write_memory"]["input_schema"]["properties"]["perspective"]["enum"]) == expected
    assert set(by_name["set_perspective"]["input_schema"]["properties"]["perspective"]["enum"]) == expected


@pytest.mark.parametrize("perspective", ["weiwei", "laoke", "shared"])
def test_actor_mutation_cannot_claim_unconfirmed_perspective(perspective):
    async def run():
        tools = ActorMemoryToolLibrary(InMemoryActorMemoryToolStore())
        with pytest.raises(PermissionError):
            await tools.call(
                context(),
                f"perspective-{perspective}",
                "write_memory",
                {
                    "content": "不能由椒椒单方面确认",
                    "scope": "weiwei-jiao",
                    "memory_type": "fact",
                    "perspective": perspective,
                    "confidential": False,
                    "importance": 5,
                    "evidence_event_ids": [101],
                },
            )

    asyncio.run(run())


def test_write_is_staged_until_accepted_final_and_discard_never_persists():
    async def run():
        store = InMemoryActorMemoryToolStore()
        tools = ActorMemoryToolLibrary(store)
        staged = await tools.call(
        context(),
        "call-write",
        "write_memory",
        {
            "content": "薇薇喜欢在雨天听音乐",
            "scope": "weiwei-jiao",
            "memory_type": "fact",
            "perspective": "jiao",
            "confidential": True,
            "importance": 8,
            "evidence_event_ids": [101],
        },
    )
        assert staged["status"] == "staged"
        assert await store.list_active() == []
        committed = await tools.commit_accepted(context(), accepted_event_id=102)
        assert committed["status"] == "committed"
        assert (await store.list_active())[0]["content"] == "薇薇喜欢在雨天听音乐"

        discarded_context = context(generation="gen-discard")
        await tools.call(
        discarded_context,
        "call-discard",
        "write_memory",
        {
            "content": "不应保存",
            "scope": "weiwei-jiao",
            "memory_type": "fact",
            "perspective": "jiao",
            "confidential": False,
            "importance": 5,
            "evidence_event_ids": [101],
        },
    )
        await tools.discard(discarded_context)
        assert [row["content"] for row in await store.list_active()] == ["薇薇喜欢在雨天听音乐"]
    asyncio.run(run())


def test_reads_are_immediate_policy_scoped_and_profile_independent():
    async def run():
        store = InMemoryActorMemoryToolStore()
        store.seed(content="椒椒私密", scope="weiwei-jiao", perspective="jiao", confidential=True)
        store.seed(content="老克私密", scope="weiwei-laoke", perspective="laoke", confidential=True)
        store.seed(content="小家公开", scope="group", perspective="shared", confidential=False)
        tools = ActorMemoryToolLibrary(store)

        first = await tools.call(context(profile="gpt-a"), "read-1", "search_memory", {"query": "私密", "limit": 20})
        second = await tools.call(context(profile="claude-b"), "read-2", "list_memories", {"status": "active", "limit": 20})
        assert [row["content"] for row in first["memories"]] == ["椒椒私密"]
        assert {row["content"] for row in second["memories"]} == {"椒椒私密", "小家公开"}
        with pytest.raises(PermissionError):
            await tools.call(context(), "read-3", "get_memory", {"memory_id": 2})
    asyncio.run(run())


def test_mutation_library_updates_tombstones_restores_evidence_merges_and_supersedes():
    async def run():
        store = InMemoryActorMemoryToolStore()
        one = store.seed(content="旧偏好", scope="weiwei-jiao", perspective="jiao")
        two = store.seed(content="重复偏好", scope="weiwei-jiao", perspective="jiao")
        movable = store.seed(content="可公开事实", scope="weiwei-jiao", perspective="jiao")
        replaceable = store.seed(content="旧理解", scope="weiwei-jiao", perspective="jiao")
        tools = ActorMemoryToolLibrary(store)
        ctx = context(room="room_group_home", generation="gen-mutations")
        calls = [
        ("update_memory", {"memory_id": one, "content": "新偏好"}),
        ("set_importance", {"memory_id": one, "importance": 9}),
        ("set_memory_type", {"memory_id": one, "memory_type": "inference"}),
        ("set_confidential", {"memory_id": one, "confidential": True}),
        ("set_perspective", {"memory_id": one, "perspective": "jiao"}),
        ("add_evidence", {"memory_id": one, "event_id": 101}),
        ("remove_evidence", {"memory_id": one, "event_id": 101}),
        ("set_memory_status", {"memory_id": one, "status": "stale"}),
        ("restore_memory", {"memory_id": one}),
        ("change_scope", {"memory_id": movable, "scope": "group"}),
        ("merge_memories", {"memory_ids": [one, two], "content": "合并偏好", "importance": 8}),
        ("supersede_memory", {"memory_id": replaceable, "content": "替代理解", "memory_type": "fact", "importance": 8}),
        ("delete_memory", {"memory_id": movable}),
        ]
        for index, (name, arguments) in enumerate(calls):
            result = await tools.call(ctx, f"mutation-{index}", name, arguments)
            assert result["status"] == "staged"
        receipt = await tools.commit_accepted(ctx, accepted_event_id=103)
        assert receipt["status"] == "committed"
        rows = await store.all_records()
        assert any(row["content"] == "合并偏好" for row in rows)
        assert any(row["content"] == "替代理解" for row in rows)
        assert next(row for row in rows if row["id"] == two)["status"] == "superseded"
    asyncio.run(run())


@pytest.mark.parametrize(
    ("actor", "room", "scope"),
    [
        ("jiao", "room_weiwei_jiao", "weiwei-jiao"),
        ("laoke", "room_weiwei_laoke", "weiwei-laoke"),
    ],
)
def test_both_actors_can_commit_direct_memory_and_candidate(actor, room, scope):
    async def run():
        store = InMemoryActorMemoryToolStore()
        tools = ActorMemoryToolLibrary(store)
        ctx = context(actor=actor, room=room, generation=f"gen-{actor}")
        for index, name in enumerate(("write_memory", "propose_memory_candidate")):
            await tools.call(
            ctx,
            f"{actor}-{index}",
            name,
            {
                "content": f"{actor}-{name}",
                "scope": scope,
                "memory_type": "fact",
                "perspective": actor,
                "confidential": False,
                "importance": 6,
                "evidence_event_ids": [101],
            },
            )
        await tools.commit_accepted(ctx, accepted_event_id=104)
        rows = await store.all_records()
        assert [row["source_kind"] for row in rows] == ["actor_tool", "agent_candidate"]
    asyncio.run(run())


@pytest.mark.parametrize(("actor", "scope"), [("jiao", "weiwei-jiao"), ("laoke", "weiwei-laoke")])
def test_complete_memory_tool_library_is_callable_for_each_bound_actor(actor, scope):
    async def run():
        store = InMemoryActorMemoryToolStore()
        first = store.seed(content=f"{actor} first", scope=scope, perspective=actor)
        second = store.seed(content=f"{actor} second", scope=scope, perspective=actor)
        third = store.seed(content=f"{actor} third", scope=scope, perspective=actor)
        tools = ActorMemoryToolLibrary(store)
        ctx = context(actor=actor, room="room_group_home", generation=f"all-tools-{actor}")
        calls = {
            "search_memory": {"query": actor, "limit": 20},
            "get_memory": {"memory_id": first},
            "list_memories": {"status": "active", "limit": 20},
            "write_memory": {"content": f"{actor} write", "scope": scope, "memory_type": "fact", "perspective": actor, "confidential": False, "importance": 5, "evidence_event_ids": [101]},
            "propose_memory_candidate": {"content": f"{actor} candidate", "scope": scope, "memory_type": "inference", "perspective": actor, "confidential": False, "importance": 6, "evidence_event_ids": [101]},
            "update_memory": {"memory_id": first, "content": "updated"},
            "delete_memory": {"memory_id": first},
            "restore_memory": {"memory_id": first},
            "change_scope": {"memory_id": third, "scope": "group"},
            "set_confidential": {"memory_id": first, "confidential": True},
            "set_perspective": {"memory_id": first, "perspective": actor},
            "set_memory_type": {"memory_id": first, "memory_type": "inference"},
            "set_memory_status": {"memory_id": first, "status": "stale"},
            "set_importance": {"memory_id": first, "importance": 8},
            "add_evidence": {"memory_id": first, "event_id": 101},
            "remove_evidence": {"memory_id": first, "event_id": 101},
            "merge_memories": {"memory_ids": [first, second], "content": "merged", "importance": 7},
            "supersede_memory": {"memory_id": third, "content": "superseded", "memory_type": "fact", "importance": 7},
        }

        observed = set()
        for index, (name, arguments) in enumerate(calls.items()):
            result = await tools.call(ctx, f"{actor}-{index}", name, arguments)
            observed.add(name)
            assert "memories" in result or "memory" in result or result["status"] == "staged"

        assert observed == ACTOR_MEMORY_TOOL_NAMES

    asyncio.run(run())


def test_confidential_and_scope_mutations_cannot_escape_actor_acl():
    async def run():
        store = InMemoryActorMemoryToolStore()
        other = store.seed(content="老克私密", scope="weiwei-laoke", perspective="laoke", confidential=True)
        group = store.seed(content="群体事实", scope="group", perspective="shared")
        tools = ActorMemoryToolLibrary(store)
        with pytest.raises(PermissionError):
            await tools.call(context(), "bad-update", "update_memory", {"memory_id": other, "content": "越权"})
        with pytest.raises(PermissionError):
            await tools.call(context(), "bad-secret", "set_confidential", {"memory_id": group, "confidential": True})
        with pytest.raises(PermissionError):
            await tools.call(context(), "bad-scope", "change_scope", {"memory_id": group, "scope": "weiwei-laoke"})
    asyncio.run(run())


def test_duplicate_tool_call_and_acceptance_are_idempotent():
    async def run():
        store = InMemoryActorMemoryToolStore()
        tools = ActorMemoryToolLibrary(store)
        ctx = context()
        args = {
        "content": "只保存一次", "scope": "weiwei-jiao", "memory_type": "fact",
        "perspective": "jiao", "confidential": False, "importance": 5,
        "evidence_event_ids": [101],
        }
        assert await tools.call(ctx, "same-call", "write_memory", args) == await tools.call(ctx, "same-call", "write_memory", args)
        first = await tools.commit_accepted(ctx, accepted_event_id=105)
        second = await tools.commit_accepted(ctx, accepted_event_id=105)
        assert first == second
        assert len(await store.all_records()) == 1
    asyncio.run(run())


def test_dashboard_audit_exposes_coordinates_but_not_private_arguments():
    async def run():
        store = InMemoryActorMemoryToolStore()
        tools = ActorMemoryToolLibrary(store)
        await tools.call(context(), "tool-audit", "write_memory", {
            "content": "private tool content", "scope": "weiwei-jiao",
            "memory_type": "fact", "perspective": "jiao",
            "confidential": False, "importance": 5,
            "evidence_event_ids": [101],
        })
        rows = await store.audit()
        assert rows[0]["actor_id"] == "jiao"
        assert rows[0]["action"] == "write_memory"
        assert "arguments" not in rows[0]
        assert "content" not in rows[0]
    asyncio.run(run())
