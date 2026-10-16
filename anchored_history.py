from __future__ import annotations

import asyncio
import hashlib
import json
from dataclasses import dataclass, replace


class AnchoredHistoryError(ValueError):
    pass


class SummaryInputBudgetExceeded(AnchoredHistoryError):
    """The whole target batch cannot fit the bounded summary request."""


class SummaryCompressionError(AnchoredHistoryError):
    def __init__(self, reason: str):
        self.reason = reason
        super().__init__(f"compression paused: {reason}; history unchanged")


def build_summary_namespace(*, actor_id: str, room_id: str, conversation_id: str,
                            execution_mode: str) -> str:
    """Conversation cognition survives Profile/Persona/tool changes; scenes do not mix."""
    if actor_id not in {"jiao", "laoke"} or execution_mode not in {"private", "group", "bedroom"}:
        raise AnchoredHistoryError("invalid summary identity")
    if not isinstance(conversation_id, str) or not conversation_id.strip():
        raise AnchoredHistoryError("summary conversation is required")
    if (not isinstance(room_id, str) or not room_id.strip()
            or conversation_id.startswith("bedroom:") != (execution_mode == "bedroom")
            or conversation_id == "bedroom:"):
        raise AnchoredHistoryError("summary room/scene mismatch")
    return hashlib.sha256(json.dumps([actor_id, room_id, conversation_id, execution_mode],
        ensure_ascii=False, separators=(",", ":")).encode("utf-8")).hexdigest()


@dataclass(frozen=True, slots=True)
class AnchoredHistoryQuery:
    conversation_id: str
    after_event_id: int
    through_event_id: int
    ordering: str

    def __post_init__(self) -> None:
        if not isinstance(self.conversation_id, str) or not self.conversation_id:
            raise AnchoredHistoryError("conversation_id must be non-empty")
        if self.ordering != "ascending_after_cursor":
            raise AnchoredHistoryError("sliding or descending history is forbidden")
        if (
            isinstance(self.after_event_id, bool)
            or not isinstance(self.after_event_id, int)
            or self.after_event_id < 0
        ):
            raise AnchoredHistoryError("after_event_id must be non-negative")
        if (
            isinstance(self.through_event_id, bool)
            or not isinstance(self.through_event_id, int)
            or self.through_event_id < self.after_event_id
        ):
            raise AnchoredHistoryError("through_event_id must follow the anchor")


@dataclass(frozen=True, slots=True)
class AnchoredHistoryState:
    # This historical field name now holds only the conversation-summary namespace.
    cache_namespace: str
    compressed_up_to_event_id: int
    summary: str
    summary_token_count: int
    state_revision: int
    compression_error: str | None = None


class AnchoredHistoryCompactor:
    """Select complete exchanges; advance only after a bounded semantic summary."""

    def __init__(
        self,
        *,
        compact_after_events: int = 128,
        retain_raw_events: int = 48,
        summary_token_limit: int = 1024,
    ) -> None:
        if (
            isinstance(compact_after_events, bool)
            or isinstance(retain_raw_events, bool)
            or isinstance(summary_token_limit, bool)
            or compact_after_events < 2
            or retain_raw_events < 1
            or retain_raw_events >= compact_after_events
            or summary_token_limit < 1
        ):
            raise AnchoredHistoryError("invalid anchored compaction bounds")
        self.compact_after_events = compact_after_events
        self.retain_raw_events = retain_raw_events
        self.summary_token_limit = summary_token_limit
        self._lock = asyncio.Lock()

    @staticmethod
    def _estimated_tokens(text: str) -> int:
        return max(1, (len(text) + 3) // 4)

    async def resume(self, *, store, namespace, expected_revision):
        async with self._lock:
            await store.set_compression_error(namespace, expected_revision=expected_revision, reason=None)

    async def maybe_compact(
        self,
        *,
        store,
        cache_namespace: str,
        state: AnchoredHistoryState,
        events: tuple[dict, ...],
        summarize=None,
        force: bool = False,
        identity: dict | None = None,
    ) -> tuple[AnchoredHistoryState, tuple[dict, ...]]:
        if summarize is None or (not force and len(events) <= self.compact_after_events):
            return state, events
        cut = len(events) - self.retain_raw_events
        # Retain the whole human-led exchange, including both Group actors.
        while cut > 0 and events[cut]["actor_id"] != "weiwei":
            cut -= 1
        if cut <= 0:
            return state, events
        # Bound one paid summary without cutting a message or silently dropping a unit.
        if len(json.dumps({"prior_summary": state.summary,
                "new_events": events[:cut]}, ensure_ascii=False)) > 192000:
            raise SummaryInputBudgetExceeded(
                "complete batch exceeds summary input budget; bounded catch-up required; history unchanged")
        compressed = events[:cut]
        tail = events[cut:]
        # ponytail: summaries queue within this single-replica Gateway. Multi-replica
        # execution needs a DB claim to prevent paying twice (CAS protects state only).
        async with self._lock:
            latest = await store.get_or_create(cache_namespace, identity=identity)
            if latest.state_revision != state.state_revision:
                raise AnchoredHistoryError("cache state revision conflict before summary")
            if latest.compression_error and not force:
                raise SummaryCompressionError(latest.compression_error)
            # Persist before spending: cancellation/crash must not buy another attempt.
            await store.set_compression_error(cache_namespace, expected_revision=state.state_revision,
                                              reason="summary_attempt_incomplete")
            try:
                replacement = await summarize(state.summary, compressed)
                if not isinstance(replacement, str) or not replacement.strip():
                    raise SummaryCompressionError("summary_empty")
                token_count = self._estimated_tokens(replacement)
                if token_count > self.summary_token_limit:
                    raise SummaryCompressionError("summary_too_long")
            except SummaryCompressionError as exc:
                await store.set_compression_error(cache_namespace, expected_revision=state.state_revision,
                                                  reason=exc.reason)
                raise
            updated = await store.apply_compression(
                cache_namespace,
                expected_revision=state.state_revision,
                replacement_summary=replacement,
                summary_token_count=token_count,
                compressed_up_to_event_id=int(compressed[-1]["event_id"]),
            )
        return updated, tail


class InMemoryAnchoredHistoryStore:
    def __init__(self, *, summary_token_limit: int = 1024) -> None:
        if summary_token_limit < 1:
            raise AnchoredHistoryError("summary token limit must be positive")
        self._summary_token_limit = summary_token_limit
        self._states: dict[str, AnchoredHistoryState] = {}
        self._identities: dict[str, dict] = {}
        self._lock = asyncio.Lock()

    async def get_or_create(
        self, cache_namespace: str, *, identity: dict | None = None
    ) -> AnchoredHistoryState:
        async with self._lock:
            state = self._states.get(cache_namespace)
            if state is None:
                state = AnchoredHistoryState(
                    cache_namespace=cache_namespace,
                    compressed_up_to_event_id=0,
                    summary="",
                    summary_token_count=0,
                    state_revision=1,
                )
                self._states[cache_namespace] = state
            if identity is not None:
                prior = self._identities.get(cache_namespace)
                if prior is not None and prior != identity:
                    raise AnchoredHistoryError("cache namespace identity changed")
                self._identities[cache_namespace] = dict(identity)
            return state

    async def observe_appended_events(
        self, cache_namespace: str, event_ids: tuple[int, ...]
    ) -> None:
        state = await self.get_or_create(cache_namespace)
        if tuple(sorted(event_ids)) != tuple(event_ids):
            raise AnchoredHistoryError("factual events must be ascending")
        if any(event_id <= state.compressed_up_to_event_id for event_id in event_ids):
            raise AnchoredHistoryError("appended event must follow compressed cursor")
        # Observation intentionally does not mutate the compression cursor/state.

    async def set_compression_error(self, cache_namespace, *, expected_revision, reason):
        async with self._lock:
            current = self._states.get(cache_namespace)
            if current is None or current.state_revision != expected_revision:
                raise AnchoredHistoryError("cache state revision conflict")
            self._states[cache_namespace] = replace(current, compression_error=reason)

    async def apply_compression(
        self,
        cache_namespace: str,
        *,
        expected_revision: int,
        replacement_summary: str | None,
        summary_token_count: int,
        compressed_up_to_event_id: int,
    ) -> AnchoredHistoryState:
        async with self._lock:
            current = self._states.get(cache_namespace)
            if current is None:
                current = AnchoredHistoryState(
                    cache_namespace=cache_namespace,
                    compressed_up_to_event_id=0,
                    summary="",
                    summary_token_count=0,
                    state_revision=1,
                )
                self._states[cache_namespace] = current
            if current.state_revision != expected_revision:
                raise AnchoredHistoryError("cache state revision conflict")
            if not isinstance(replacement_summary, str) or not replacement_summary:
                raise AnchoredHistoryError("replacement summary is required")
            if (
                isinstance(summary_token_count, bool)
                or not isinstance(summary_token_count, int)
                or summary_token_count < 0
            ):
                raise AnchoredHistoryError("summary token count is invalid")
            if summary_token_count > self._summary_token_limit:
                raise AnchoredHistoryError("replacement summary exceeds token limit")
            if compressed_up_to_event_id <= current.compressed_up_to_event_id:
                raise AnchoredHistoryError("compressed cursor must move forward")
            replacement = AnchoredHistoryState(
                cache_namespace=cache_namespace,
                compressed_up_to_event_id=compressed_up_to_event_id,
                summary=replacement_summary,
                summary_token_count=summary_token_count,
                state_revision=current.state_revision + 1,
            )
            self._states[cache_namespace] = replacement
            return replacement

    async def invalidate_conversation_state(self, conversation_id: str) -> None:
        async with self._lock:
            for namespace, state in self._states.items():
                if self._identities.get(namespace, {}).get("conversation_id") == conversation_id:
                    self._states[namespace] = replace(
                        state, summary="", summary_token_count=0, compressed_up_to_event_id=0,
                        state_revision=state.state_revision + 1,
                    )

    async def delete_conversation_state(self, conversation_id: str) -> None:
        async with self._lock:
            self._states = {
                namespace: state for namespace, state in self._states.items()
                if self._identities.get(namespace, {}).get("conversation_id")
                != conversation_id
            }
            self._identities = {
                namespace: identity
                for namespace, identity in self._identities.items()
                if identity.get("conversation_id") != conversation_id
            }


def _state_from_row(row) -> AnchoredHistoryState:
    return AnchoredHistoryState(
        cache_namespace=str(row["cache_namespace"]),
        compressed_up_to_event_id=int(row["compressed_up_to_event_id"]),
        summary=str(row["summary"]),
        summary_token_count=int(row["summary_token_count"]),
        state_revision=int(row["state_revision"]),
        compression_error=row.get("compression_error"),
    )


class PostgresAnchoredHistoryStore:
    REQUIRED_IDENTITY = {
        "actor_id",
        "room_id",
        "conversation_id",
        "execution_mode",
    }

    def __init__(self, pool_factory, *, summary_token_limit: int = 1024) -> None:
        if summary_token_limit < 1:
            raise AnchoredHistoryError("summary token limit must be positive")
        self._pool_factory = pool_factory
        self._summary_token_limit = summary_token_limit

    async def get_or_create(
        self, cache_namespace: str, *, identity: dict | None = None
    ) -> AnchoredHistoryState:
        if not cache_namespace:
            raise AnchoredHistoryError("cache namespace is required")
        if not isinstance(identity, dict) or set(identity) != self.REQUIRED_IDENTITY:
            raise AnchoredHistoryError("complete summary identity is required")
        if cache_namespace != build_summary_namespace(**identity):
            raise AnchoredHistoryError("summary namespace identity mismatch")
        pool = await self._pool_factory()
        async with pool.acquire() as conn:
            async with conn.transaction():
                # Coordinate new namespaces with atomic factual repair; no persisted lock state.
                await conn.execute(
                    "SELECT pg_advisory_xact_lock_shared(hashtextextended('history-repair:' || $1, 0))",
                    identity["conversation_id"],
                )
                # Serialize first adoption; a second Profile must see the same state.
                await conn.execute("SELECT pg_advisory_xact_lock(hashtextextended($1, 0))", cache_namespace)
                row = await conn.fetchrow(
                    "SELECT summary_namespace AS cache_namespace,* FROM conversation_summary_state WHERE summary_namespace=$1",
                    cache_namespace)
                if row is None:
                    prior = await self._legacy_summary(conn, identity)
                    row = await conn.fetchrow(
                        """INSERT INTO conversation_summary_state(
                             summary_namespace,actor_id,room_id,conversation_id,execution_mode,
                             summary,summary_token_count,compressed_up_to_event_id,state_revision)
                           VALUES($1,$2,$3,$4,$5,$6,$7,$8,$9)
                           RETURNING summary_namespace AS cache_namespace,*""",
                        cache_namespace, identity["actor_id"], identity["room_id"],
                        identity["conversation_id"], identity["execution_mode"],
                        prior["summary"] if prior else "", prior["summary_token_count"] if prior else 0,
                        prior["compressed_up_to_event_id"] if prior else 0,
                        prior["state_revision"] if prior else 1)
        if row is None:
            raise AnchoredHistoryError("cache state could not be created")
        return _state_from_row(row)

    async def _legacy_summary(self, conn, identity):
        """One-time adoption, never a model call or a write to the old summary rows."""
        candidates = await conn.fetch(
            """SELECT * FROM model_cache_state WHERE actor_id=$1 AND conversation_id=$2
               AND execution_mode=$3 AND compressed_up_to_event_id>0
               ORDER BY compressed_up_to_event_id DESC,cache_namespace""",
            identity["actor_id"], identity["conversation_id"], identity["execution_mode"])
        if not candidates:
            return None
        facts = await conn.fetch(
            """SELECT source_event_id,room_id,canonical_conversation_id,source_kind,bedroom_session_id
               FROM conversations WHERE session_id=$1 AND fact_identity IS NOT NULL""",
            identity["conversation_id"])
        bedroom = identity["execution_mode"] == "bedroom"
        if not facts or any(f["room_id"] != identity["room_id"]
                or f["source_kind"] != ("bedroom_turn" if bedroom else "relay_event")
                or (f["bedroom_session_id"] != identity["conversation_id"][8:] if bedroom
                    else f["canonical_conversation_id"] != identity["conversation_id"])
                for f in facts):
            raise AnchoredHistoryError("legacy summary provenance mismatch; adoption requires review")
        event_ids = {int(f["source_event_id"]) for f in facts}
        valid = [c for c in candidates if c["compressed_up_to_event_id"] in event_ids
                 and isinstance(c["summary"], str) and c["summary"].strip()
                 and len(c["summary"]) <= self._summary_token_limit * 4
                 and 0 <= c["summary_token_count"] <= self._summary_token_limit
                 and c["state_revision"] >= 1]
        if not valid:
            raise AnchoredHistoryError("legacy summary cursor/bounds invalid; adoption requires review")
        best = valid[0]
        if any(c["summary"] != best["summary"] for c in valid
               if c["compressed_up_to_event_id"] == best["compressed_up_to_event_id"]):
            raise AnchoredHistoryError("conflicting legacy summaries; adoption requires review")
        return best

    async def observe_appended_events(
        self, cache_namespace: str, event_ids: tuple[int, ...]
    ) -> None:
        if tuple(sorted(event_ids)) != tuple(event_ids):
            raise AnchoredHistoryError("factual events must be ascending")
        if any(event_id < 1 for event_id in event_ids):
            raise AnchoredHistoryError("factual event ids must be positive")
        # Append observation deliberately does not move compressed_up_to.

    async def set_compression_error(self, cache_namespace, *, expected_revision, reason):
        pool = await self._pool_factory()
        async with pool.acquire() as conn:
            row = await conn.fetchrow(
                """UPDATE conversation_summary_state SET compression_error=$3,updated_at=NOW()
                   WHERE summary_namespace=$1 AND state_revision=$2 RETURNING summary_namespace""",
                cache_namespace, expected_revision, reason)
            if row is None:
                raise AnchoredHistoryError("cache state revision conflict")

    async def apply_compression(
        self,
        cache_namespace: str,
        *,
        expected_revision: int,
        replacement_summary: str | None,
        summary_token_count: int,
        compressed_up_to_event_id: int,
    ) -> AnchoredHistoryState:
        if not isinstance(replacement_summary, str) or not replacement_summary:
            raise AnchoredHistoryError("replacement summary is required")
        if (
            isinstance(summary_token_count, bool)
            or not isinstance(summary_token_count, int)
            or summary_token_count < 0
            or summary_token_count > self._summary_token_limit
        ):
            raise AnchoredHistoryError("replacement summary exceeds token limit")
        if compressed_up_to_event_id < 1:
            raise AnchoredHistoryError("compressed cursor must move forward")
        pool = await self._pool_factory()
        async with pool.acquire() as conn:
            async with conn.transaction():
                current = await conn.fetchrow(
                    "SELECT * FROM conversation_summary_state WHERE summary_namespace=$1 FOR UPDATE",
                    cache_namespace,
                )
                if current is None:
                    raise AnchoredHistoryError("cache state does not exist")
                if int(current["state_revision"]) != expected_revision:
                    raise AnchoredHistoryError("cache state revision conflict")
                if compressed_up_to_event_id <= int(
                    current["compressed_up_to_event_id"]
                ):
                    raise AnchoredHistoryError("compressed cursor must move forward")
                row = await conn.fetchrow(
                    """UPDATE conversation_summary_state SET
                         summary=$3,summary_token_count=$4,
                         compressed_up_to_event_id=$5,
                         state_revision=state_revision+1,compression_error=NULL,updated_at=NOW()
                       WHERE summary_namespace=$1 AND state_revision=$2
                       RETURNING summary_namespace AS cache_namespace,compressed_up_to_event_id,summary,
                                 summary_token_count,state_revision,compression_error""",
                    cache_namespace,
                    expected_revision,
                    replacement_summary,
                    summary_token_count,
                    compressed_up_to_event_id,
                )
                if row is None:
                    raise AnchoredHistoryError("cache state revision conflict")
        return _state_from_row(row)

    async def invalidate_conversation_state(self, conversation_id: str, *, connection=None) -> None:
        if connection is None:
            pool = await self._pool_factory()
            async with pool.acquire() as conn:
                await self.invalidate_conversation_state(conversation_id, connection=conn)
            return
        await connection.execute(
            """UPDATE conversation_summary_state SET summary='',summary_token_count=0,
                 compressed_up_to_event_id=0,state_revision=state_revision+1,updated_at=NOW()
               WHERE conversation_id=$1""", conversation_id)
        # Repair also invalidates rollback copies so later adoption cannot resurrect stale text.
        await connection.execute(
            """UPDATE model_cache_state SET summary='',summary_token_count=0,
                 compressed_up_to_event_id=0,stable_prefix_hash=NULL,
                 state_revision=state_revision+1,updated_at=NOW()
               WHERE conversation_id=$1""",
            conversation_id,
        )

    async def delete_conversation_state(self, conversation_id: str) -> None:
        pool = await self._pool_factory()
        async with pool.acquire() as conn:
            async with conn.transaction():
                await conn.execute("SELECT pg_advisory_xact_lock(hashtextextended('history-repair:' || $1, 0))", conversation_id)
                await conn.execute("DELETE FROM conversation_summary_state WHERE conversation_id=$1", conversation_id)
                await conn.execute("DELETE FROM model_cache_state WHERE conversation_id=$1", conversation_id)
