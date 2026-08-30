from __future__ import annotations

from dataclasses import dataclass
import json
from typing import Any

from conversation_partitions import ConversationFact


class ConversationSyncIncomplete(RuntimeError):
    """Relay could not prove the complete accepted partition through current fact."""


@dataclass(frozen=True, slots=True)
class ConversationSyncReceipt:
    partition_id: str
    synced_through_event_id: int
    inserted_count: int


class ConversationSyncService:
    def __init__(self, relay_client: Any, store: Any) -> None:
        self.relay_client = relay_client
        self.store = store

    async def ensure_relay_synced(
        self,
        *,
        actor_id: str,
        room_id: str,
        conversation_id: str,
        current_event_id: int,
    ) -> ConversationSyncReceipt:
        latest = await self.store.latest_event_id(conversation_id)
        events = await self.relay_client.fetch_model_history_facts(
            actor_id=actor_id,
            room_id=room_id,
            conversation_id=conversation_id,
            current_event_id=current_event_id,
            after_event_id=min(latest, current_event_id),
            through_event_id=current_event_id,
            include_current_event=True,
        )
        facts = []
        for event in events:
            if (
                event.get("room_id") != room_id
                or event.get("conversation_id") != conversation_id
                or event.get("visibility") not in {"room", "public"}
            ):
                raise ConversationSyncIncomplete("Relay returned mismatched conversation fact")
            facts.append(ConversationFact.from_relay_event(event))
        inserted = await self.store.append_accepted_facts(tuple(facts))
        persisted = await self.store.list_facts(
            conversation_id,
            through_event_id=current_event_id,
        )
        current = next(
            (fact for fact in persisted if fact.source_event_id == current_event_id),
            None,
        )
        if current is None or current.room_id != room_id or current.conversation_id != conversation_id:
            raise ConversationSyncIncomplete("cognitive history is not synced through current event")
        return ConversationSyncReceipt(conversation_id, current_event_id, inserted)

    async def ensure_bedroom_synced(
        self,
        *,
        bedroom_session_id: str,
        current_turn_id: int,
        actor_id: str,
    ) -> ConversationSyncReceipt:
        payload = await self.relay_client.fetch_bedroom_facts(bedroom_session_id)
        session = payload.get("session") if isinstance(payload, dict) else None
        turns = payload.get("turns") if isinstance(payload, dict) else None
        if (
            not isinstance(session, dict)
            or session.get("bedroom_session_id") != bedroom_session_id
            or session.get("actor_id") != actor_id
            or not isinstance(turns, list)
        ):
            raise ConversationSyncIncomplete("Bedroom facts do not match execution identity")
        facts = []
        for raw in turns:
            if not isinstance(raw, dict):
                raise ConversationSyncIncomplete("Bedroom fact is malformed")
            turn_id = raw.get("turn_id")
            if isinstance(turn_id, bool) or not isinstance(turn_id, int):
                raise ConversationSyncIncomplete("Bedroom turn identity is malformed")
            if turn_id > current_turn_id:
                continue
            turn = dict(raw)
            provenance = turn.pop("provenance_json", turn.get("provenance"))
            if isinstance(provenance, str):
                provenance = json.loads(provenance)
            turn["provenance"] = provenance
            facts.append(ConversationFact.from_bedroom_turn(session, turn))
        inserted = await self.store.append_accepted_facts(tuple(facts))
        partition_id = f"bedroom:{bedroom_session_id}"
        persisted = await self.store.list_facts(partition_id, through_event_id=current_turn_id)
        if not any(fact.source_event_id == current_turn_id for fact in persisted):
            raise ConversationSyncIncomplete("Bedroom history is not synced through current turn")
        return ConversationSyncReceipt(partition_id, current_turn_id, inserted)
