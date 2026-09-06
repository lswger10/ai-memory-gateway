from __future__ import annotations

from dataclasses import dataclass, replace
import json
from typing import Any

from conversation_partitions import ConversationFact, _timestamp_value


class ConversationSyncIncomplete(RuntimeError):
    """Relay could not prove the complete accepted partition through current fact."""


@dataclass(frozen=True, slots=True)
class ConversationSyncReceipt:
    partition_id: str
    synced_through_event_id: int
    inserted_count: int


class ConversationSyncService:
    def __init__(self, relay_client: Any, store: Any, history_store: Any = None) -> None:
        self.relay_client = relay_client
        self.store = store
        self.history_store = history_store

    async def ensure_relay_synced(
        self,
        *,
        actor_id: str,
        room_id: str,
        conversation_id: str,
        current_event_id: int,
    ) -> ConversationSyncReceipt:
        # ponytail: full O(n) replay until Relay supplies an authoritative integrity proof.
        events = await self.relay_client.fetch_model_history_facts(
            actor_id=actor_id,
            room_id=room_id,
            conversation_id=conversation_id,
            current_event_id=current_event_id,
            after_event_id=0,
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
        return await self._reconcile(conversation_id, current_event_id, tuple(facts))

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
        return await self._reconcile(f"bedroom:{bedroom_session_id}", current_turn_id, tuple(facts))

    async def _reconcile(self, partition_id, current_event_id, facts):
        expected = {fact.fact_identity: fact for fact in facts}
        if (
            len(expected) != len(facts)
            or len({fact.source_event_id for fact in facts}) != len(facts)
            or any(fact.partition_id != partition_id or fact.source_event_id > current_event_id for fact in facts)
            or not any(fact.source_event_id == current_event_id for fact in facts)
        ):
            raise ConversationSyncIncomplete("Relay did not supply a complete, unique current history")
        cursor = await self.store.synced_through_event_id(partition_id)
        existing = {fact.fact_identity: fact for fact in await self.store.list_facts(
            partition_id, through_event_id=current_event_id)}
        if existing.keys() - expected.keys():
            raise ConversationSyncIncomplete("Relay history omits previously accepted facts")
        changed = tuple(fact for identity, fact in expected.items()
                        if identity not in existing or not _same_fact(existing[identity], fact))
        repairs_history = any(fact.fact_identity in existing or fact.source_event_id <= cursor
                              for fact in changed)
        inserted = await self.store.append_accepted_facts(
            changed, repair=True, history_store=self.history_store if repairs_history else None,
        ) if changed else 0
        persisted = {fact.fact_identity: fact for fact in await self.store.list_facts(
            partition_id, through_event_id=current_event_id)}
        if persisted.keys() != expected.keys() or any(
            not _same_fact(persisted[identity], fact) for identity, fact in expected.items()
        ):
            raise ConversationSyncIncomplete("cognitive history does not match Relay")
        await self.store.mark_synced_through(partition_id, current_event_id)
        return ConversationSyncReceipt(partition_id, current_event_id, inserted)


def _same_fact(left: ConversationFact, right: ConversationFact) -> bool:
    # PostgreSQL normalizes timestamp spelling; this is not a change in factual content.
    if replace(left, created_at=right.created_at) != right:
        return False
    if left.created_at == right.created_at:
        return True
    return _timestamp_value(left.created_at) == _timestamp_value(right.created_at)
