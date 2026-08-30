from __future__ import annotations

import asyncio
from dataclasses import dataclass


class AnchoredHistoryError(ValueError):
    pass


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
    cache_namespace: str
    compressed_up_to_event_id: int
    summary: str
    summary_token_count: int
    state_revision: int


class AnchoredHistoryCompactor:
    """Advance an anchored cursor only at a bounded, atomic compression event.

    The replacement is deliberately deterministic and extractive. Long-term
    semantic memory remains owned by the memory pipeline; this summary exists
    only to keep the provider cache prefix bounded without a hidden paid model
    call.
    """

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

    @staticmethod
    def _event_line(event: dict) -> str:
        event_id = event.get("event_id")
        actor_id = event.get("actor_id")
        content = event.get("content")
        if (
            isinstance(event_id, bool)
            or not isinstance(event_id, int)
            or event_id < 1
            or not isinstance(actor_id, str)
            or not actor_id
            or not isinstance(content, str)
        ):
            raise AnchoredHistoryError("invalid factual event for compression")
        normalized = " ".join(content.split())
        return f"#{event_id} {actor_id}: {normalized}"

    @staticmethod
    def _estimated_tokens(text: str) -> int:
        return max(1, (len(text) + 3) // 4)

    def _replacement_summary(
        self, prior_summary: str, compressed_events: tuple[dict, ...]
    ) -> tuple[str, int]:
        lines = [line for line in (prior_summary.strip(),) if line]
        lines.extend(self._event_line(event) for event in compressed_events)
        max_chars = self.summary_token_limit * 4
        summary = "\n".join(lines)
        if len(summary) > max_chars:
            summary = summary[-max_chars:]
            first_newline = summary.find("\n")
            if first_newline >= 0:
                summary = summary[first_newline + 1 :]
            summary = "[older compressed facts omitted]\n" + summary
            summary = summary[-max_chars:]
        if not summary:
            raise AnchoredHistoryError("compression produced an empty summary")
        return summary, self._estimated_tokens(summary)

    async def maybe_compact(
        self,
        *,
        store,
        cache_namespace: str,
        state: AnchoredHistoryState,
        events: tuple[dict, ...],
    ) -> tuple[AnchoredHistoryState, tuple[dict, ...]]:
        if len(events) <= self.compact_after_events:
            return state, events
        cut = len(events) - self.retain_raw_events
        compressed = events[:cut]
        tail = events[cut:]
        replacement, token_count = self._replacement_summary(
            state.summary, compressed
        )
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
    )


class PostgresAnchoredHistoryStore:
    REQUIRED_IDENTITY = {
        "actor_id",
        "conversation_id",
        "profile_id",
        "profile_revision",
        "execution_mode",
        "actor_prompt_version",
        "runtime_kernel_version",
        "room_policy_version",
        "tool_schema_hash",
        "cache_strategy_version",
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
            raise AnchoredHistoryError("complete cache identity is required")
        pool = await self._pool_factory()
        async with pool.acquire() as conn:
            row = await conn.fetchrow(
                """WITH inserted AS (
                     INSERT INTO model_cache_state(
                       cache_namespace,actor_id,conversation_id,profile_id,
                       profile_revision,execution_mode,actor_prompt_version,
                       runtime_kernel_version,room_policy_version,tool_schema_hash,
                       cache_strategy_version
                     ) VALUES($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11)
                     ON CONFLICT(cache_namespace) DO NOTHING
                   )
                   SELECT cache_namespace,compressed_up_to_event_id,summary,
                          summary_token_count,state_revision
                   FROM model_cache_state WHERE cache_namespace=$1""",
                cache_namespace,
                identity["actor_id"],
                identity["conversation_id"],
                identity["profile_id"],
                identity["profile_revision"],
                identity["execution_mode"],
                identity["actor_prompt_version"],
                identity["runtime_kernel_version"],
                identity["room_policy_version"],
                identity["tool_schema_hash"],
                identity["cache_strategy_version"],
            )
        if row is None:
            raise AnchoredHistoryError("cache state could not be created")
        return _state_from_row(row)

    async def observe_appended_events(
        self, cache_namespace: str, event_ids: tuple[int, ...]
    ) -> None:
        if tuple(sorted(event_ids)) != tuple(event_ids):
            raise AnchoredHistoryError("factual events must be ascending")
        if any(event_id < 1 for event_id in event_ids):
            raise AnchoredHistoryError("factual event ids must be positive")
        # Append observation deliberately does not move compressed_up_to.

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
                    "SELECT * FROM model_cache_state WHERE cache_namespace=$1 FOR UPDATE",
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
                    """UPDATE model_cache_state SET
                         summary=$3,summary_token_count=$4,
                         compressed_up_to_event_id=$5,
                         state_revision=state_revision+1,updated_at=NOW()
                       WHERE cache_namespace=$1 AND state_revision=$2
                       RETURNING cache_namespace,compressed_up_to_event_id,summary,
                                 summary_token_count,state_revision""",
                    cache_namespace,
                    expected_revision,
                    replacement_summary,
                    summary_token_count,
                    compressed_up_to_event_id,
                )
                if row is None:
                    raise AnchoredHistoryError("cache state revision conflict")
        return _state_from_row(row)

    async def delete_conversation_state(self, conversation_id: str) -> None:
        pool = await self._pool_factory()
        async with pool.acquire() as conn:
            await conn.execute(
                "DELETE FROM model_cache_state WHERE conversation_id=$1",
                conversation_id,
            )
