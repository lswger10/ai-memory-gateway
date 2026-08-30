import pytest


def relay_event(event_id, *, actor_id="weiwei", room_id="room_group_home", conversation_id="group-1"):
    return {
        "contract_version": "group-room.v1.0",
        "event_id": event_id,
        "room_id": room_id,
        "conversation_id": conversation_id,
        "burst_id": f"burst-{event_id}",
        "actor_id": actor_id,
        "role": "human" if actor_id == "weiwei" else "agent",
        "event_type": "human_message" if actor_id == "weiwei" else "agent_final",
        "content": f"message-{event_id}",
        "reply_to_event_id": None,
        "mentions": [],
        "created_at": f"2026-08-30T00:00:{event_id:02d}Z",
        "request_id": f"request-{event_id}",
        "visibility": "room",
        "provenance": None,
    }


class FakeRelay:
    def __init__(self, events=(), bedroom=None):
        self.events = tuple(events)
        self.bedroom = bedroom
        self.history_calls = []

    async def fetch_model_history_facts(self, **kwargs):
        self.history_calls.append(kwargs)
        return tuple(
            event for event in self.events
            if event["event_id"] > kwargs["after_event_id"]
            and event["event_id"] <= kwargs["through_event_id"]
        )

    async def fetch_bedroom_facts(self, session_id):
        assert self.bedroom["session"]["bedroom_session_id"] == session_id
        return self.bedroom


@pytest.mark.anyio
async def test_delayed_partition_is_read_repaired_through_current_event():
    from conversation_partitions import InMemoryConversationPartitionStore
    from conversation_sync import ConversationSyncService

    relay = FakeRelay(relay_event(i) for i in (1, 2, 3))
    store = InMemoryConversationPartitionStore()
    receipt = await ConversationSyncService(relay, store).ensure_relay_synced(
        actor_id="jiao", room_id="room_group_home", conversation_id="group-1",
        current_event_id=3,
    )

    assert receipt.synced_through_event_id == 3
    assert [fact.source_event_id for fact in await store.list_facts("group-1")] == [1, 2, 3]
    assert relay.history_calls[0]["include_current_event"] is True


@pytest.mark.anyio
async def test_restart_store_deduplicates_replayed_relay_facts():
    from conversation_partitions import ConversationFact, InMemoryConversationPartitionStore
    from conversation_sync import ConversationSyncService

    store = InMemoryConversationPartitionStore()
    await store.append_accepted_facts((ConversationFact.from_relay_event(relay_event(1)),))
    relay = FakeRelay(relay_event(i) for i in (1, 2))
    service = ConversationSyncService(relay, store)
    await service.ensure_relay_synced(
        actor_id="jiao", room_id="room_group_home", conversation_id="group-1",
        current_event_id=2,
    )
    await service.ensure_relay_synced(
        actor_id="laoke", room_id="room_group_home", conversation_id="group-1",
        current_event_id=2,
    )
    assert await store.count_facts("group-1") == 2


@pytest.mark.anyio
async def test_sync_watermark_repairs_older_fact_even_when_current_row_already_exists():
    from conversation_partitions import ConversationFact, InMemoryConversationPartitionStore
    from conversation_sync import ConversationSyncService

    store = InMemoryConversationPartitionStore()
    await store.append_accepted_facts((ConversationFact.from_relay_event(relay_event(2)),))
    relay = FakeRelay((relay_event(1), relay_event(2)))
    await ConversationSyncService(relay, store).ensure_relay_synced(
        actor_id="jiao", room_id="room_group_home", conversation_id="group-1",
        current_event_id=2,
    )
    assert relay.history_calls[0]["after_event_id"] == 0
    assert [fact.source_event_id for fact in await store.list_facts("group-1")] == [1, 2]
    assert await store.synced_through_event_id("group-1") == 2


@pytest.mark.anyio
async def test_missing_current_fact_blocks_generation_instead_of_using_partial_history():
    from conversation_partitions import InMemoryConversationPartitionStore
    from conversation_sync import ConversationSyncIncomplete, ConversationSyncService

    service = ConversationSyncService(
        FakeRelay((relay_event(1), relay_event(2))),
        InMemoryConversationPartitionStore(),
    )
    with pytest.raises(ConversationSyncIncomplete):
        await service.ensure_relay_synced(
            actor_id="jiao", room_id="room_group_home", conversation_id="group-1",
            current_event_id=3,
        )


@pytest.mark.anyio
async def test_wrong_room_fact_is_rejected_during_read_repair():
    from conversation_partitions import InMemoryConversationPartitionStore
    from conversation_sync import ConversationSyncIncomplete, ConversationSyncService

    wrong = relay_event(1, room_id="room_weiwei_jiao", conversation_id="group-1")
    with pytest.raises(ConversationSyncIncomplete):
        await ConversationSyncService(FakeRelay((wrong,)), InMemoryConversationPartitionStore()).ensure_relay_synced(
            actor_id="jiao", room_id="room_group_home", conversation_id="group-1",
            current_event_id=1,
        )


@pytest.mark.anyio
async def test_bedroom_read_repair_is_session_partitioned_and_requires_current_turn():
    from conversation_partitions import InMemoryConversationPartitionStore
    from conversation_sync import ConversationSyncService

    bedroom = {
        "session": {
            "bedroom_session_id": "bedroom-1", "room_id": "room_weiwei_jiao",
            "conversation_id": "private-jiao", "actor_id": "jiao",
            "retention_policy": "summary-only",
        },
        "turns": [
            {"turn_id": 1, "actor_id": "weiwei", "role": "human", "text": "hi", "request_id": "b1", "created_at": "now", "provenance_json": None},
            {"turn_id": 2, "actor_id": "jiao", "role": "agent", "text": "hello", "request_id": "b2", "created_at": "later", "provenance_json": None},
        ],
    }
    store = InMemoryConversationPartitionStore()
    receipt = await ConversationSyncService(FakeRelay(bedroom=bedroom), store).ensure_bedroom_synced(
        bedroom_session_id="bedroom-1", current_turn_id=2, actor_id="jiao"
    )
    assert receipt.partition_id == "bedroom:bedroom-1"
    assert [fact.content for fact in await store.list_facts(receipt.partition_id)] == ["hi", "hello"]
