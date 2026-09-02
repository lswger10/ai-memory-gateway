from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import inspect
import json
from typing import Any, Iterable

from group_contracts_v11 import (
    CONTRACT_VERSION as MEDIA_CONTRACT_VERSION,
    ContractError as MediaContractError,
    MediaReference,
    validate_room_event,
)


class ConversationPartitionError(ValueError):
    pass


class ConversationPartitionConflict(ConversationPartitionError):
    pass


_FORBIDDEN_ATTACHMENT_FIELDS = {
    "base64", "bytes", "content", "data", "data_url", "raw", "payload"
}


def _bounded_mapping(value: Any, field: str) -> dict[str, Any] | None:
    if value is None:
        return None
    if not isinstance(value, dict):
        raise ConversationPartitionError(f"{field} must be an object or null")
    return json.loads(json.dumps(value, ensure_ascii=False, sort_keys=True))


def _attachment_references(
    value: Any,
    *,
    contract_version: str | None = None,
) -> tuple[dict[str, Any], ...]:
    if value is None:
        return ()
    if not isinstance(value, list):
        raise ConversationPartitionError("attachments must be an array")
    result: list[dict[str, Any]] = []
    for item in value:
        if not isinstance(item, dict) or _FORBIDDEN_ATTACHMENT_FIELDS.intersection(item):
            raise ConversationPartitionError("conversation history accepts attachment references only")
        if contract_version == MEDIA_CONTRACT_VERSION:
            try:
                item = MediaReference.from_dict(item).to_dict()
            except MediaContractError as exc:
                raise ConversationPartitionError(str(exc)) from exc
        result.append(json.loads(json.dumps(item, ensure_ascii=False, sort_keys=True)))
    return tuple(result)


@dataclass(frozen=True, slots=True)
class ConversationFact:
    fact_identity: str
    partition_id: str
    room_id: str
    conversation_id: str
    source_event_id: int
    actor_id: str
    role: str
    content: str
    request_id: str
    created_at: str
    source_kind: str
    provenance: dict[str, Any] | None = None
    attachments: tuple[dict[str, Any], ...] = ()
    bedroom_session_id: str | None = None
    retention_policy: str | None = None
    burst_id: str | None = None
    event_type: str | None = None
    reply_to_event_id: int | None = None
    mentions: tuple[str, ...] = ()
    message_kind: str = "text"

    def __post_init__(self) -> None:
        strings = (
            self.fact_identity, self.partition_id, self.room_id,
            self.conversation_id, self.actor_id, self.role,
            self.request_id, self.created_at, self.source_kind,
        )
        if any(not isinstance(value, str) or not value for value in strings):
            raise ConversationPartitionError("conversation fact identity fields are required")
        if isinstance(self.source_event_id, bool) or self.source_event_id < 1:
            raise ConversationPartitionError("source_event_id must be positive")
        if self.role not in {"human", "agent"}:
            raise ConversationPartitionError("only accepted human/agent facts are cognitive history")
        if not isinstance(self.content, str):
            raise ConversationPartitionError("content must be text")

    @classmethod
    def from_relay_event(cls, event: dict[str, Any]) -> "ConversationFact":
        if not isinstance(event, dict) or event.get("visibility") not in {"room", "public"}:
            raise ConversationPartitionError("Relay event is not an accepted public fact")
        contract_version = event.get("contract_version")
        if contract_version == MEDIA_CONTRACT_VERSION:
            try:
                validate_room_event(event)
            except MediaContractError as exc:
                raise ConversationPartitionError(str(exc)) from exc
        room_id = str(event.get("room_id") or "")
        conversation_id = str(event.get("conversation_id") or "")
        event_id = event.get("event_id")
        if isinstance(event_id, bool) or not isinstance(event_id, int):
            raise ConversationPartitionError("event_id must be an integer")
        return cls(
            fact_identity=f"relay:{room_id}:{conversation_id}:{event_id}",
            partition_id=conversation_id,
            room_id=room_id,
            conversation_id=conversation_id,
            source_event_id=event_id,
            actor_id=str(event.get("actor_id") or ""),
            role=str(event.get("role") or ""),
            content=str(event.get("content") or ""),
            request_id=str(event.get("request_id") or ""),
            created_at=str(event.get("created_at") or ""),
            source_kind="relay_event",
            provenance=_bounded_mapping(event.get("provenance"), "provenance"),
            attachments=_attachment_references(
                event.get("attachments"), contract_version=contract_version
            ),
            burst_id=event.get("burst_id"),
            event_type=event.get("event_type"),
            reply_to_event_id=event.get("reply_to_event_id"),
            mentions=tuple(event.get("mentions") or ()),
            message_kind=str(event.get("message_kind") or "text"),
        )

    @classmethod
    def from_bedroom_turn(
        cls, session: dict[str, Any], turn: dict[str, Any]
    ) -> "ConversationFact":
        session_id = str(session.get("bedroom_session_id") or "")
        turn_id = turn.get("turn_id")
        if isinstance(turn_id, bool) or not isinstance(turn_id, int):
            raise ConversationPartitionError("Bedroom turn_id must be an integer")
        return cls(
            fact_identity=f"bedroom:{session_id}:{turn_id}",
            partition_id=f"bedroom:{session_id}",
            room_id=str(session.get("room_id") or ""),
            conversation_id=str(session.get("conversation_id") or ""),
            source_event_id=turn_id,
            actor_id=str(turn.get("actor_id") or ""),
            role=str(turn.get("role") or ""),
            content=str(turn.get("text") or ""),
            request_id=str(turn.get("request_id") or ""),
            created_at=str(turn.get("created_at") or ""),
            source_kind="bedroom_turn",
            provenance=_bounded_mapping(turn.get("provenance"), "provenance"),
            bedroom_session_id=session_id,
            retention_policy=str(session.get("retention_policy") or ""),
        )

    @property
    def content_hash(self) -> str:
        payload = {
            "fact_identity": self.fact_identity,
            "partition_id": self.partition_id,
            "room_id": self.room_id,
            "conversation_id": self.conversation_id,
            "source_event_id": self.source_event_id,
            "actor_id": self.actor_id,
            "role": self.role,
            "content": self.content,
            "request_id": self.request_id,
            "created_at": self.created_at,
            "source_kind": self.source_kind,
            "provenance": self.provenance,
            "attachments": self.attachments,
            "bedroom_session_id": self.bedroom_session_id,
            "retention_policy": self.retention_policy,
            "burst_id": self.burst_id,
            "event_type": self.event_type,
            "reply_to_event_id": self.reply_to_event_id,
            "mentions": self.mentions,
            "message_kind": self.message_kind,
        }
        canonical = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    def to_history_event(self) -> dict[str, Any]:
        return {
            "event_id": self.source_event_id,
            "room_id": self.room_id,
            "conversation_id": self.conversation_id,
            "burst_id": self.burst_id,
            "actor_id": self.actor_id,
            "role": self.role,
            "event_type": self.event_type,
            "content": self.content,
            "reply_to_event_id": self.reply_to_event_id,
            "mentions": list(self.mentions),
            "created_at": self.created_at,
            "request_id": self.request_id,
            "visibility": "room",
            "provenance": self.provenance,
            "attachments": list(self.attachments),
            "message_kind": self.message_kind,
        }


class InMemoryConversationPartitionStore:
    def __init__(self) -> None:
        self._facts: dict[str, ConversationFact] = {}
        self._sync_watermarks: dict[str, int] = {}
        self._lock = asyncio.Lock()

    async def append_accepted_facts(self, facts: Iterable[ConversationFact]) -> int:
        inserted = 0
        async with self._lock:
            for fact in facts:
                existing = self._facts.get(fact.fact_identity)
                if existing is not None:
                    if existing.content_hash != fact.content_hash:
                        raise ConversationPartitionConflict("accepted fact identity changed")
                    continue
                self._facts[fact.fact_identity] = fact
                inserted += 1
        return inserted

    async def list_facts(
        self,
        partition_id: str,
        *,
        after_event_id: int = 0,
        through_event_id: int | None = None,
    ) -> tuple[ConversationFact, ...]:
        async with self._lock:
            result = [
                fact for fact in self._facts.values()
                if fact.partition_id == partition_id
                and fact.source_event_id > after_event_id
                and (through_event_id is None or fact.source_event_id <= through_event_id)
            ]
        return tuple(sorted(result, key=lambda fact: fact.source_event_id))

    async def latest_event_id(self, partition_id: str) -> int:
        facts = await self.list_facts(partition_id)
        return facts[-1].source_event_id if facts else 0

    async def count_facts(self, partition_id: str) -> int:
        return len(await self.list_facts(partition_id))

    async def synced_through_event_id(self, partition_id: str) -> int:
        async with self._lock:
            return self._sync_watermarks.get(partition_id, 0)

    async def mark_synced_through(self, partition_id: str, event_id: int) -> None:
        async with self._lock:
            self._sync_watermarks[partition_id] = max(
                event_id, self._sync_watermarks.get(partition_id, 0)
            )

    async def delete_bedroom_partition(self, bedroom_session_id: str) -> None:
        partition_id = f"bedroom:{bedroom_session_id}"
        async with self._lock:
            self._facts = {
                identity: fact for identity, fact in self._facts.items()
                if fact.partition_id != partition_id
            }
            self._sync_watermarks.pop(partition_id, None)


def _json_value(value: Any, fallback: Any) -> Any:
    if value is None:
        return fallback
    if isinstance(value, str):
        return json.loads(value)
    return value


def _timestamp_text(value: Any) -> str:
    if isinstance(value, str):
        return value
    if hasattr(value, "isoformat"):
        return value.isoformat().replace("+00:00", "Z")
    return str(value)


def _timestamp_value(value: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
    except (AttributeError, TypeError, ValueError) as exc:
        raise ConversationPartitionError("created_at must be an ISO timestamp") from exc
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed


class PostgresConversationPartitionStore:
    """Persistent view of Relay-accepted facts in the existing conversations table."""

    def __init__(self, pool_factory) -> None:
        self._pool_factory = pool_factory

    async def _pool(self):
        value = self._pool_factory()
        return await value if inspect.isawaitable(value) else value

    async def append_accepted_facts(self, facts: Iterable[ConversationFact]) -> int:
        pool = await self._pool()
        inserted = 0
        async with pool.acquire() as conn:
            async with conn.transaction():
                for fact in facts:
                    row = await conn.fetchrow(
                        """
                        INSERT INTO conversations (
                            session_id, room_id, canonical_conversation_id,
                            source_event_id, actor_id, event_role,
                            fact_identity, fact_hash, content, request_id,
                            created_at, source_kind, provenance_json,
                            attachments_json, message_kind, bedroom_session_id,
                            retention_policy, role, accepted_at, burst_id,
                            event_type, reply_to_event_id, mentions_json
                        ) VALUES (
                            $1,$2,$3,$4,$5,$6,$7,$8,$9,$10,
                            $11::timestamptz,$12,$13::jsonb,$14::jsonb,$15,$16,$17,
                            CASE WHEN $6 = 'human' THEN 'user' ELSE 'assistant' END,
                            NOW(),$18,$19,$20,$21::jsonb
                        )
                        ON CONFLICT (fact_identity) WHERE fact_identity IS NOT NULL
                        DO UPDATE SET fact_identity = EXCLUDED.fact_identity
                        RETURNING fact_hash
                        """,
                        fact.partition_id,
                        fact.room_id,
                        fact.conversation_id,
                        fact.source_event_id,
                        fact.actor_id,
                        fact.role,
                        fact.fact_identity,
                        fact.content_hash,
                        fact.content,
                        fact.request_id,
                        _timestamp_value(fact.created_at),
                        fact.source_kind,
                        json.dumps(fact.provenance, ensure_ascii=False),
                        json.dumps(fact.attachments, ensure_ascii=False),
                        fact.message_kind,
                        fact.bedroom_session_id,
                        fact.retention_policy,
                        fact.burst_id,
                        fact.event_type,
                        fact.reply_to_event_id,
                        json.dumps(fact.mentions, ensure_ascii=False),
                    )
                    if row is None or row["fact_hash"] != fact.content_hash:
                        raise ConversationPartitionConflict("accepted fact identity changed")
                    # PostgreSQL cannot distinguish insert from the idempotent no-op in
                    # this compact statement. Returning the number accepted is sufficient
                    # for callers; factual identity is still stored exactly once.
                    inserted += 1
        return inserted

    async def list_facts(
        self,
        partition_id: str,
        *,
        after_event_id: int = 0,
        through_event_id: int | None = None,
    ) -> tuple[ConversationFact, ...]:
        pool = await self._pool()
        async with pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT session_id, room_id, canonical_conversation_id,
                       source_event_id, actor_id, event_role, fact_identity,
                       fact_hash, content, request_id, created_at, source_kind,
                       provenance_json, attachments_json, message_kind, bedroom_session_id,
                       retention_policy, burst_id, event_type,
                       reply_to_event_id, mentions_json
                FROM conversations
                WHERE session_id = $1
                  AND fact_identity IS NOT NULL
                  AND source_event_id > $2
                  AND ($3::bigint IS NULL OR source_event_id <= $3)
                ORDER BY source_event_id ASC
                """,
                partition_id,
                after_event_id,
                through_event_id,
            )
        return tuple(self._from_row(row) for row in rows)

    async def latest_event_id(self, partition_id: str) -> int:
        pool = await self._pool()
        async with pool.acquire() as conn:
            value = await conn.fetchval(
                """SELECT COALESCE(MAX(source_event_id), 0)
                   FROM conversations
                   WHERE session_id = $1 AND fact_identity IS NOT NULL""",
                partition_id,
            )
        return int(value or 0)

    async def count_facts(self, partition_id: str) -> int:
        return len(await self.list_facts(partition_id))

    async def synced_through_event_id(self, partition_id: str) -> int:
        pool = await self._pool()
        async with pool.acquire() as conn:
            value = await conn.fetchval(
                "SELECT synced_through_event_id FROM conversation_partition_sync_state WHERE partition_id=$1",
                partition_id,
            )
        return int(value or 0)

    async def mark_synced_through(self, partition_id: str, event_id: int) -> None:
        pool = await self._pool()
        async with pool.acquire() as conn:
            await conn.execute(
                """INSERT INTO conversation_partition_sync_state(
                       partition_id,synced_through_event_id,updated_at
                     ) VALUES($1,$2,NOW())
                     ON CONFLICT(partition_id) DO UPDATE SET
                       synced_through_event_id=GREATEST(
                         conversation_partition_sync_state.synced_through_event_id,
                         EXCLUDED.synced_through_event_id
                       ), updated_at=NOW()""",
                partition_id,
                event_id,
            )

    async def delete_bedroom_partition(self, bedroom_session_id: str) -> None:
        pool = await self._pool()
        async with pool.acquire() as conn:
            async with conn.transaction():
                await conn.execute(
                    "DELETE FROM conversations WHERE session_id = $1",
                    f"bedroom:{bedroom_session_id}",
                )
                await conn.execute(
                    "DELETE FROM conversation_partition_sync_state WHERE partition_id = $1",
                    f"bedroom:{bedroom_session_id}",
                )

    @staticmethod
    def _from_row(row: Any) -> ConversationFact:
        return ConversationFact(
            fact_identity=row["fact_identity"],
            partition_id=row["session_id"],
            room_id=row["room_id"],
            conversation_id=row["canonical_conversation_id"],
            source_event_id=int(row["source_event_id"]),
            actor_id=row["actor_id"],
            role=row["event_role"],
            content=row["content"] or "",
            request_id=row["request_id"],
            created_at=_timestamp_text(row["created_at"]),
            source_kind=row["source_kind"],
            provenance=_json_value(row["provenance_json"], None),
            attachments=tuple(_json_value(row["attachments_json"], [])),
            message_kind=row.get("message_kind") or "text",
            bedroom_session_id=row["bedroom_session_id"],
            retention_policy=row["retention_policy"],
            burst_id=row.get("burst_id"),
            event_type=row.get("event_type"),
            reply_to_event_id=row.get("reply_to_event_id"),
            mentions=tuple(_json_value(row.get("mentions_json"), [])),
        )
