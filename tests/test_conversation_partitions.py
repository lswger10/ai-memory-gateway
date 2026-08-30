import pytest


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

