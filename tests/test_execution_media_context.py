import pytest

from tests.test_conversation_media_partitions import media_event
from tests.test_execution_context_builder import (
    _BedroomContext,
    _GroupContext,
    _event,
    _profile,
    _request,
)
from dataclasses import replace


@pytest.mark.anyio
async def test_followup_retains_unanswered_image_until_this_actor_replies():
    from conversation_partitions import ConversationFact, InMemoryConversationPartitionStore
    from execution_context_builder import GatewayExecutionContextBuilder

    class Sync:
        async def ensure_relay_synced(self, **kwargs):
            return None

    store = InMemoryConversationPartitionStore()
    events = [media_event(1), _event(2, room_id="room_group_home", conversation_id="group-media")]
    await store.append_accepted_facts(tuple(ConversationFact.from_relay_event(e) for e in events))
    builder = GatewayExecutionContextBuilder(group_context=_GroupContext(),
        bedroom_context=_BedroomContext(), conversation_store=store, conversation_sync=Sync())
    req = replace(_request(), room_id="room_group_home", conversation_id="group-media", current_event_id=2)
    bundle = await builder.build(req, _profile(), resolved_room_id=req.room_id, resolved_conversation_id=req.conversation_id)
    assert [x["attachment_id"] for x in bundle.current_media_references] == ["photo-1.png"]

    await store.append_accepted_facts(tuple(ConversationFact.from_relay_event(e) for e in (
        _event(3, actor_id="jiao", room_id=req.room_id, conversation_id=req.conversation_id),
        _event(4, room_id=req.room_id, conversation_id=req.conversation_id))))
    bundle = await builder.build(replace(req, current_event_id=4), _profile(),
        resolved_room_id=req.room_id, resolved_conversation_id=req.conversation_id)
    assert bundle.current_media_references == ()


@pytest.mark.anyio
async def test_current_media_reference_is_dynamic_and_never_enters_stable_history():
    from conversation_partitions import ConversationFact, InMemoryConversationPartitionStore
    from execution_context_builder import GatewayExecutionContextBuilder

    class Sync:
        async def ensure_relay_synced(self, **_kwargs):
            return None

    store = InMemoryConversationPartitionStore()
    await store.append_accepted_facts(
        (
            ConversationFact.from_relay_event(media_event(1)),
            ConversationFact.from_relay_event(media_event(2)),
        )
    )
    builder = GatewayExecutionContextBuilder(
        group_context=_GroupContext(),
        bedroom_context=_BedroomContext(),
        conversation_store=store,
        conversation_sync=Sync(),
    )
    bundle = await builder.build(
        replace(
            _request(),
            room_id="room_group_home",
            conversation_id="group-media",
            current_event_id=2,
        ),
        _profile(),
        resolved_room_id="room_group_home",
        resolved_conversation_id="group-media",
    )
    assert "photo-1.png" in bundle.stable_history[0]
    assert "photo-1.png" not in "".join(bundle.dynamic_tail)
    assert bundle.current_media_references[0]["attachment_id"] == "photo-1.png"
    assert all("data_url" not in item for item in bundle.current_media_references)


@pytest.mark.anyio
async def test_group_reprobe_keeps_trigger_media_available_to_second_actor():
    from conversation_partitions import ConversationFact, InMemoryConversationPartitionStore
    from execution_context_builder import GatewayExecutionContextBuilder

    class Sync:
        async def ensure_relay_synced(self, **_kwargs):
            return None

    trigger = media_event(1)
    trigger["burst_id"] = "burst-shared"
    accepted_final = _event(
        2,
        actor_id="jiao",
        room_id="room_group_home",
        conversation_id="group-media",
    )
    accepted_final["burst_id"] = "burst-shared"
    store = InMemoryConversationPartitionStore()
    await store.append_accepted_facts(
        (
            ConversationFact.from_relay_event(trigger),
            ConversationFact.from_relay_event(accepted_final),
        )
    )
    builder = GatewayExecutionContextBuilder(
        group_context=_GroupContext(),
        bedroom_context=_BedroomContext(),
        conversation_store=store,
        conversation_sync=Sync(),
    )

    bundle = await builder.build(
        replace(
            _request(),
            actor_id="laoke",
            room_id="room_group_home",
            conversation_id="group-media",
            current_event_id=2,
        ),
        _profile(),
        resolved_room_id="room_group_home",
        resolved_conversation_id="group-media",
    )

    assert [item["attachment_id"] for item in bundle.current_media_references] == [
        "photo-1.png"
    ]
