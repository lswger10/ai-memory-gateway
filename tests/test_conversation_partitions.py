import pytest
from dataclasses import replace


def relay_event(event_id, *, room_id="room_weiwei_jiao", actor_id="weiwei", content=None):
    return {
        "event_id": event_id,
        "room_id": room_id,
        "conversation_id": "conversation-1" if room_id != "room_group_home" else "group-1",
        "burst_id": f"burst-{event_id}",
        "actor_id": actor_id,
        "role": "human" if actor_id == "weiwei" else "agent",
        "event_type": "human_message" if actor_id == "weiwei" else "agent_final",
        "content": content or f"message-{event_id}",
        "mentions": [],
        "reply_to_event_id": None,
        "created_at": f"2026-08-30T00:00:{event_id:02d}Z",
        "request_id": f"request-{event_id}",
        "visibility": "public",
        "provenance": None if actor_id == "weiwei" else {
            "provider": "ofox",
            "provider_family": "anthropic",
            "model": "claude",
            "model_snapshot": None,
            "adapter_version": "v1",
            "generation_request_id": f"gen-{event_id}",
            "fallback_used": False,
        },
    }


@pytest.mark.anyio
async def test_private_partition_persists_complete_accepted_history():
    from conversation_partitions import ConversationFact, InMemoryConversationPartitionStore

    store = InMemoryConversationPartitionStore()
    await store.append_accepted_facts(
        tuple(ConversationFact.from_relay_event(relay_event(i)) for i in (1, 2, 3))
    )

    facts = await store.list_facts("conversation-1")
    assert [fact.source_event_id for fact in facts] == [1, 2, 3]
    assert [fact.content for fact in facts] == ["message-1", "message-2", "message-3"]


@pytest.mark.anyio
async def test_group_transcript_is_stored_once_for_both_actor_contexts():
    from conversation_partitions import ConversationFact, InMemoryConversationPartitionStore

    store = InMemoryConversationPartitionStore()
    fact = ConversationFact.from_relay_event(
        relay_event(10, room_id="room_group_home", actor_id="jiao")
    )
    await store.append_accepted_facts((fact,))
    await store.append_accepted_facts((fact,))

    assert await store.count_facts("group-1") == 1
    assert (await store.list_facts("group-1"))[0].actor_id == "jiao"


@pytest.mark.anyio
async def test_duplicate_fact_identity_rejects_changed_content():
    from conversation_partitions import (
        ConversationFact,
        ConversationPartitionConflict,
        InMemoryConversationPartitionStore,
    )

    store = InMemoryConversationPartitionStore()
    await store.append_accepted_facts((ConversationFact.from_relay_event(relay_event(2)),))

    with pytest.raises(ConversationPartitionConflict):
        await store.append_accepted_facts(
            (ConversationFact.from_relay_event(relay_event(2, content="changed")),)
        )


def test_attachment_history_keeps_references_and_rejects_embedded_bytes():
    from conversation_partitions import ConversationFact, ConversationPartitionError

    event = relay_event(4)
    event["attachments"] = [
        {"attachment_id": "upload-1", "filename": "photo.jpg", "mime": "image/jpeg", "size": 42}
    ]
    fact = ConversationFact.from_relay_event(event)
    assert fact.attachments[0]["attachment_id"] == "upload-1"

    event["attachments"][0]["base64"] = "AAAA"
    with pytest.raises(ConversationPartitionError):
        ConversationFact.from_relay_event(event)


@pytest.mark.anyio
async def test_bedroom_partition_is_session_scoped_and_deletable():
    from conversation_partitions import ConversationFact, InMemoryConversationPartitionStore

    store = InMemoryConversationPartitionStore()
    session = {
        "bedroom_session_id": "bedroom-1",
        "room_id": "room_weiwei_laoke",
        "conversation_id": "private-laoke",
        "retention_policy": "no-retention",
    }
    turn = {
        "turn_id": 7,
        "turn_epoch": 1,
        "actor_id": "weiwei",
        "role": "human",
        "text": "private scene",
        "request_id": "bedroom-request",
        "created_at": "2026-08-30T00:00:07Z",
        "provenance": None,
    }
    await store.append_accepted_facts((ConversationFact.from_bedroom_turn(session, turn),))

    assert [fact.content for fact in await store.list_facts("bedroom:bedroom-1")] == ["private scene"]
    await store.delete_bedroom_partition("bedroom-1")
    assert await store.list_facts("bedroom:bedroom-1") == ()


@pytest.mark.anyio
async def test_public_group_recall_excludes_private_bedroom_legacy_future_and_drafts():
    from conversation_partitions import ConversationFact, InMemoryConversationPartitionStore

    store = InMemoryConversationPartitionStore()
    public = ConversationFact.from_relay_event(relay_event(10,
        room_id="room_group_home", actor_id="laoke", content="缓存保活每50分钟续期"))
    await store.append_accepted_facts((public,
        ConversationFact.from_relay_event(relay_event(11, content="缓存私聊秘密")),
        ConversationFact.from_relay_event(relay_event(12, room_id="room_weiwei_laoke", content="缓存另一私聊")),
        replace(public, fact_identity="bedroom:x:13", partition_id="bedroom:x",
            source_event_id=13, source_kind="bedroom_turn", bedroom_session_id="x", content="缓存卧室秘密"),
        replace(public, fact_identity="legacy:14", source_event_id=14, source_kind="legacy_unscoped"),
        replace(public, fact_identity="draft:15", source_event_id=15, event_type="agent_draft"),
        replace(public, fact_identity="future:30", source_event_id=30),
    ))
    results = await store.search_public_group_facts(keywords=("缓存",), before_event_id=20)
    assert results == ({"room_id": "room_group_home", "conversation_id": "group-1",
        "event_id": 10, "actor_id": "laoke", "created_at": "2026-08-30T00:00:10Z",
        "content": "缓存保活每50分钟续期", "truncated": False},)
    assert await store.search_public_group_facts(keywords=("火锅",), before_event_id=20) == ()
    assert await store.search_public_group_facts(keywords=(), before_event_id=20) == ()


@pytest.mark.anyio
async def test_public_group_recall_is_ranked_bounded_and_never_rewrites_facts():
    from conversation_partitions import ConversationFact, InMemoryConversationPartitionStore

    store = InMemoryConversationPartitionStore()
    facts = tuple(ConversationFact.from_relay_event(relay_event(i,
        room_id="room_group_home", content="缓存" + "长" * 900)) for i in range(1, 7))
    strongest = ConversationFact.from_relay_event(relay_event(7,
        room_id="room_group_home", content="缓存保活方案"))
    await store.append_accepted_facts((*facts, strongest))
    results = await store.search_public_group_facts(keywords=("缓存", "保活"), before_event_id=20)
    assert [row["event_id"] for row in results] == [7, 6, 5, 4]
    assert results[1]["truncated"] is True
    assert len(results[1]["content"]) == 600
    assert await store.list_facts("group-1") == (*facts, strongest)
    recent = await store.search_public_group_facts(keywords=(), before_event_id=6, include_recent=True)
    assert [row["event_id"] for row in recent] == [5, 4, 3, 2]


@pytest.mark.anyio
async def test_public_group_recall_excerpt_contains_late_keyword_not_just_message_start():
    from conversation_partitions import ConversationFact, InMemoryConversationPartitionStore
    store = InMemoryConversationPartitionStore()
    await store.append_accepted_facts((ConversationFact.from_relay_event(relay_event(1,
        room_id="room_group_home", content="开场" * 500 + "缓存保活每50分钟一次。")),))
    results = await store.search_public_group_facts(keywords=("缓存",), before_event_id=2)
    assert "缓存保活每50分钟一次。" in results[0]["content"]
    assert results[0]["truncated"] is True
    assert len(results[0]["content"]) <= 600

