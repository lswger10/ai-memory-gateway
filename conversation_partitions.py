from __future__ import annotations

import asyncio
from dataclasses import dataclass
import hashlib
import inspect
import json
from typing import Any, Iterable


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


def _attachment_references(value: Any) -> tuple[dict[str, Any], ...]:
    if value is None:
        return ()
    if not isinstance(value, list):
        raise ConversationPartitionError("attachments must be an array")
    result: list[dict[str, Any]] = []
    for item in value:
        if not isinstance(item, dict) or _FORBIDDEN_ATTACHMENT_FIELDS.intersection(item):
            raise ConversationPartitionError("conversation history accepts attachment references only")
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
        if not isinstance(event, dict) or event.get("visibility") != "public":
            raise ConversationPartitionError("Relay event is not an accepted public fact")
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
            attachments=_attachment_references(event.get("attachments")),
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
        }
        canonical = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


class InMemoryConversationPartitionStore:
    def __init__(self) -> None:
        self._facts: dict[str, ConversationFact] = {}
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

    async def delete_bedroom_partition(self, bedroom_session_id: str) -> None:
        partition_id = f"bedroom:{bedroom_session_id}"
        async with self._lock:
            self._facts = {
                identity: fact for identity, fact in self._facts.items()
                if fact.partition_id != partition_id
            }


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
                            attachments_json, bedroom_session_id,
                            retention_policy, role, accepted_at
                        ) VALUES (
                            $1,$2,$3,$4,$5,$6,$7,$8,$9,$10,
                            $11::timestamptz,$12,$13::jsonb,$14::jsonb,$15,$16,
                            CASE WHEN $6 = 'human' THEN 'user' ELSE 'assistant' END,
                            NOW()
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
                        fact.created_at,
                        fact.source_kind,
                        json.dumps(fact.provenance, ensure_ascii=False),
                        json.dumps(fact.attachments, ensure_ascii=False),
                        fact.bedroom_session_id,
                        fact.retention_policy,
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
                       provenance_json, attachments_json, bedroom_session_id,
                       retention_policy
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

    async def delete_bedroom_partition(self, bedroom_session_id: str) -> None:
        pool = await self._pool()
        async with pool.acquire() as conn:
            async with conn.transaction():
                await conn.execute(
                    "DELETE FROM conversations WHERE session_id = $1",
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
            bedroom_session_id=row["bedroom_session_id"],
            retention_policy=row["retention_policy"],
        )
