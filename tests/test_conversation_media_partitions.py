import json

import pytest


def media_event(event_id=1, *, room_id="room_group_home", conversation_id="group-media"):
    return {
        "contract_version": "group-room.v1.1",
        "event_id": event_id,
        "room_id": room_id,
        "conversation_id": conversation_id,
        "burst_id": f"burst-{event_id}",
        "actor_id": "weiwei",
        "role": "human",
        "event_type": "human_message",
        "message_kind": "image",
        "content": "看图",
        "attachments": [
            {
                "attachment_id": "photo-1.png",
                "name": "photo.png",
                "media_type": "image/png",
                "size": 42,
                "category": "image",
                "purpose": "attachment",
                "source": {"type": "relay", "path": "/uploads/photo-1.png"},
                "derived_text": None,
                "semantic_label": None,
            }
        ],
        "reply_to_event_id": None,
        "mentions": [],
        "created_at": "2026-09-01T00:00:00Z",
        "request_id": f"media-{event_id}",
        "visibility": "room",
        "provenance": None,
    }


def test_v11_fact_persists_exact_reference_and_message_kind_without_bytes():
    from conversation_partitions import ConversationFact

    fact = ConversationFact.from_relay_event(media_event())
    assert fact.message_kind == "image"
    assert fact.attachments[0]["source"] == {
        "type": "relay", "path": "/uploads/photo-1.png"
    }
    assert fact.to_history_event()["message_kind"] == "image"
    assert "bytes" not in json.dumps(fact.to_history_event())


def test_v11_fact_rejects_unknown_or_embedded_media_fields():
    from conversation_partitions import ConversationFact, ConversationPartitionError

    for field in ("base64", "bytes", "data_url", "unknown"):
        event = media_event()
        event["attachments"][0][field] = "AAAA"
        with pytest.raises(ConversationPartitionError):
            ConversationFact.from_relay_event(event)


@pytest.mark.anyio
async def test_group_media_transcript_is_stored_once_for_both_actor_syncs():
    from conversation_partitions import ConversationFact, InMemoryConversationPartitionStore

    fact = ConversationFact.from_relay_event(media_event())
    store = InMemoryConversationPartitionStore()
    await store.append_accepted_facts((fact,))
    await store.append_accepted_facts((fact,))
    assert await store.count_facts("group-media") == 1


@pytest.mark.anyio
async def test_delayed_v11_read_repair_reaches_current_media_event():
    from conversation_partitions import InMemoryConversationPartitionStore
    from conversation_sync import ConversationSyncService

    class Relay:
        async def fetch_model_history_facts(self, **kwargs):
            assert kwargs["current_event_id"] == 2
            return (media_event(1), media_event(2))

    store = InMemoryConversationPartitionStore()
    await ConversationSyncService(Relay(), store).ensure_relay_synced(
        actor_id="jiao",
        room_id="room_group_home",
        conversation_id="group-media",
        current_event_id=2,
    )
    facts = await store.list_facts("group-media")
    assert [fact.source_event_id for fact in facts] == [1, 2]
    assert all(fact.message_kind == "image" for fact in facts)
