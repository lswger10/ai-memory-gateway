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


class InMemoryAnchoredHistoryStore:
    def __init__(self, *, summary_token_limit: int = 1024) -> None:
        if summary_token_limit < 1:
            raise AnchoredHistoryError("summary token limit must be positive")
        self._summary_token_limit = summary_token_limit
        self._states: dict[str, AnchoredHistoryState] = {}
        self._lock = asyncio.Lock()

    async def get_or_create(self, cache_namespace: str) -> AnchoredHistoryState:
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
