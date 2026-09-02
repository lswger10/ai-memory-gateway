import pytest

from tests.test_conversation_media_partitions import media_event
from tests.test_execution_context_builder import (
    _BedroomContext,
    _GroupContext,
    _profile,
    _request,
)
from dataclasses import replace


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
