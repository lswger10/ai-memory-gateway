import asyncio

import pytest

from anchored_history import (
    AnchoredHistoryCompactor,
    AnchoredHistoryError,
    AnchoredHistoryQuery,
    InMemoryAnchoredHistoryStore,
)


def test_sliding_limit_history_is_rejected():
    with pytest.raises(AnchoredHistoryError, match="sliding"):
        AnchoredHistoryQuery(
            conversation_id="conversation-1",
            after_event_id=0,
            through_event_id=100,
            ordering="descending_limit",
        )


def test_anchored_query_is_ascending_after_cursor():
    query = AnchoredHistoryQuery(
        conversation_id="conversation-1",
        after_event_id=40,
        through_event_id=100,
        ordering="ascending_after_cursor",
    )
    assert query.after_event_id == 40
    assert query.ordering == "ascending_after_cursor"


@pytest.mark.anyio
async def test_normal_append_does_not_advance_compressed_up_to():
    store = InMemoryAnchoredHistoryStore()
    before = await store.get_or_create("namespace-1")
    await store.observe_appended_events("namespace-1", (41, 42, 43))
    after = await store.get_or_create("namespace-1")

    assert after.compressed_up_to_event_id == before.compressed_up_to_event_id == 0
    assert after.summary == ""
    assert after.state_revision == before.state_revision


@pytest.mark.anyio
async def test_compression_atomically_replaces_capped_summary_and_cursor():
    store = InMemoryAnchoredHistoryStore(summary_token_limit=20)
    first = await store.get_or_create("namespace-1")
    updated = await store.apply_compression(
        "namespace-1",
        expected_revision=first.state_revision,
        replacement_summary="complete bounded summary",
        summary_token_count=12,
        compressed_up_to_event_id=40,
    )

    assert updated.summary == "complete bounded summary"
    assert updated.summary_token_count == 12
    assert updated.compressed_up_to_event_id == 40
    assert updated.state_revision == first.state_revision + 1


@pytest.mark.anyio
async def test_failed_summary_does_not_advance_cursor():
    store = InMemoryAnchoredHistoryStore(summary_token_limit=20)
    before = await store.get_or_create("namespace-1")
    with pytest.raises(AnchoredHistoryError):
        await store.apply_compression(
            "namespace-1",
            expected_revision=before.state_revision,
            replacement_summary=None,
            summary_token_count=0,
            compressed_up_to_event_id=40,
        )
    assert await store.get_or_create("namespace-1") == before


@pytest.mark.anyio
async def test_summary_over_limit_does_not_partially_advance_cursor():
    store = InMemoryAnchoredHistoryStore(summary_token_limit=10)
    before = await store.get_or_create("namespace-1")
    with pytest.raises(AnchoredHistoryError, match="limit"):
        await store.apply_compression(
            "namespace-1",
            expected_revision=before.state_revision,
            replacement_summary="too large",
            summary_token_count=11,
            compressed_up_to_event_id=40,
        )
    assert await store.get_or_create("namespace-1") == before


@pytest.mark.anyio
async def test_compression_cannot_move_cursor_backwards():
    store = InMemoryAnchoredHistoryStore()
    state = await store.get_or_create("namespace-1")
    state = await store.apply_compression(
        "namespace-1",
        expected_revision=state.state_revision,
        replacement_summary="first",
        summary_token_count=1,
        compressed_up_to_event_id=40,
    )
    with pytest.raises(AnchoredHistoryError, match="forward"):
        await store.apply_compression(
            "namespace-1",
            expected_revision=state.state_revision,
            replacement_summary="older",
            summary_token_count=1,
            compressed_up_to_event_id=39,
        )


@pytest.mark.anyio
async def test_compactor_overwrites_bounded_summary_advances_cursor_and_keeps_tail():
    store = InMemoryAnchoredHistoryStore(summary_token_limit=16)
    state = await store.get_or_create("namespace-1")
    events = tuple(
        {"event_id": event_id, "actor_id": "weiwei", "content": f"fact-{event_id}"}
        for event_id in range(1, 7)
    )
    compactor = AnchoredHistoryCompactor(
        compact_after_events=4,
        retain_raw_events=2,
        summary_token_limit=16,
    )

    async def summarize(prior_summary, compressed_events):
        assert prior_summary == ""
        assert [event["event_id"] for event in compressed_events] == [1, 2, 3, 4]
        return "fact-4 and the earlier agreement"

    updated, stable_tail = await compactor.maybe_compact(
        store=store,
        cache_namespace="namespace-1",
        state=state,
        events=events,
        summarize=summarize,
    )

    assert updated.compressed_up_to_event_id == 4
    assert updated.state_revision == state.state_revision + 1
    assert updated.summary_token_count <= 16
    assert "fact-4" in updated.summary
    assert [event["event_id"] for event in stable_tail] == [5, 6]


@pytest.mark.anyio
async def test_compactor_does_not_move_cursor_during_normal_append():
    store = InMemoryAnchoredHistoryStore(summary_token_limit=32)
    state = await store.get_or_create("namespace-1")
    compactor = AnchoredHistoryCompactor(
        compact_after_events=4,
        retain_raw_events=2,
        summary_token_limit=32,
    )

    unchanged, tail = await compactor.maybe_compact(
        store=store,
        cache_namespace="namespace-1",
        state=state,
        events=({"event_id": 1, "actor_id": "weiwei", "content": "hello"},),
    )

    assert unchanged == state
    assert tail[0]["event_id"] == 1


@pytest.mark.anyio
async def test_no_summarizer_never_substitutes_lossy_tail_for_history():
    store = InMemoryAnchoredHistoryStore()
    state = await store.get_or_create("namespace-1")
    events = tuple({"event_id": i, "actor_id": "weiwei", "content": str(i)} for i in range(1, 9))
    compactor = AnchoredHistoryCompactor(compact_after_events=4, retain_raw_events=2)
    after, tail = await compactor.maybe_compact(store=store, cache_namespace="namespace-1", state=state, events=events)
    assert after == state
    assert tail == events


@pytest.mark.anyio
async def test_summary_cut_preserves_complete_human_and_multi_actor_turn():
    store = InMemoryAnchoredHistoryStore()
    state = await store.get_or_create("namespace-1")
    events = tuple({"event_id": i, "actor_id": actor, "content": str(i)} for i, actor in enumerate(
        ["weiwei", "jiao", "laoke", "weiwei", "jiao", "laoke", "weiwei"], 1))
    seen = []

    async def summarize(prior_summary, compressed_events):
        seen.extend(compressed_events)
        return "A full earlier exchange, including both actors."

    compactor = AnchoredHistoryCompactor(compact_after_events=5, retain_raw_events=2)
    updated, tail = await compactor.maybe_compact(store=store, cache_namespace="namespace-1", state=state,
        events=events, summarize=summarize)
    assert [event["event_id"] for event in seen] == [1, 2, 3]
    assert [event["event_id"] for event in tail] == [4, 5, 6, 7]
    assert updated.compressed_up_to_event_id == 3


@pytest.mark.anyio
async def test_summary_failure_or_oversize_keeps_previous_summary_and_cursor():
    store = InMemoryAnchoredHistoryStore()
    state = await store.get_or_create("namespace-1")
    state = await store.apply_compression("namespace-1", expected_revision=1,
        replacement_summary="An important older correction", summary_token_count=10, compressed_up_to_event_id=1)
    events = tuple({"event_id": i, "actor_id": "weiwei", "content": str(i)} for i in range(2, 9))
    compactor = AnchoredHistoryCompactor(compact_after_events=4, retain_raw_events=2)

    async def oversized(prior_summary, compressed_events):
        assert prior_summary == "An important older correction"
        return "x" * 5000

    with pytest.raises(AnchoredHistoryError):
        await compactor.maybe_compact(store=store, cache_namespace="namespace-1", state=state,
            events=events, summarize=oversized)
    assert await store.get_or_create("namespace-1") == state


@pytest.mark.anyio
async def test_concurrent_rooms_wait_and_stale_state_never_pays_twice():
    store = InMemoryAnchoredHistoryStore()
    first = await store.get_or_create("first")
    second = await store.get_or_create("second")
    events = tuple({"event_id": i, "actor_id": "weiwei", "content": str(i)} for i in range(1, 9))
    compactor = AnchoredHistoryCompactor(compact_after_events=4, retain_raw_events=2)
    started, release = asyncio.Event(), asyncio.Event()
    calls = []

    async def summarize(prior_summary, compressed_events):
        calls.append(compressed_events)
        started.set()
        await release.wait()
        return "An accepted complete exchange."

    async def compact(namespace, state):
        return await compactor.maybe_compact(store=store, cache_namespace=namespace,
            state=state, events=events, summarize=summarize)

    first_task = asyncio.create_task(compact("first", first))
    await started.wait()
    second_task = asyncio.create_task(compact("second", second))
    stale_task = asyncio.create_task(compact("first", first))
    await asyncio.sleep(0)
    release.set()
    results = await asyncio.gather(first_task, second_task, stale_task, return_exceptions=True)
    assert not isinstance(results[0], BaseException)
    assert not isinstance(results[1], BaseException)
    assert isinstance(results[2], AnchoredHistoryError)
    assert len(calls) == 2
