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
    from dataclasses import replace
    assert await store.get_or_create("namespace-1") == replace(state, compression_error="summary_too_long")


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


@pytest.mark.anyio
async def test_oversized_full_batch_never_buys_small_fragments_on_subsequent_turns():
    store = InMemoryAnchoredHistoryStore()
    state = await store.get_or_create("conversation-summary")
    compactor = AnchoredHistoryCompactor()
    calls = []

    async def summarize(prior, events):
        calls.append(events)
        return "A summary."

    for count in (130, 131):
        events = tuple({"event_id": i, "actor_id": "weiwei", "content": "x" * 3000}
                       for i in range(1, count + 1))
        with pytest.raises(AnchoredHistoryError, match="summary input budget"):
            await compactor.maybe_compact(store=store, cache_namespace="conversation-summary",
                state=state, events=events, summarize=summarize)
        assert await store.get_or_create("conversation-summary") == state
    assert calls == []


@pytest.mark.anyio
@pytest.mark.parametrize("input_chars, permitted", [(191999, True), (192000, True), (192001, False)])
async def test_full_batch_input_bound_includes_prior_summary_and_event_metadata(input_chars, permitted):
    import json
    store = InMemoryAnchoredHistoryStore()
    await store.get_or_create("summary")
    state = await store.apply_compression("summary", expected_revision=1,
        replacement_summary="先前摘要" * 1000, summary_token_count=1000, compressed_up_to_event_id=1)
    events = tuple(dict(event_id=i, actor_id="weiwei", content="") for i in range(2, 132))
    overhead = len(json.dumps({"prior_summary": state.summary, "new_events": events[:82]}, ensure_ascii=False))
    events[0]["content"] = "字" * (input_chars - overhead)
    calls = []
    async def summarize(prior, selected):
        calls.append(selected)
        return "New complete summary"
    kwargs = dict(store=store, cache_namespace="summary", state=state, events=events, summarize=summarize)
    if permitted:
        updated, tail = await AnchoredHistoryCompactor().maybe_compact(**kwargs)
        assert updated.compressed_up_to_event_id == 83 and len(tail) == 48
        assert calls == [events[:82]]
    else:
        with pytest.raises(AnchoredHistoryError, match="summary input budget"):
            await AnchoredHistoryCompactor().maybe_compact(**kwargs)
        assert calls == [] and await store.get_or_create("summary") == state


@pytest.mark.anyio
async def test_summary_state_isolates_actor_room_conversation_and_bedroom_session():
    from anchored_history import build_summary_namespace
    identities = [
        dict(actor_id="jiao", room_id="room_weiwei_jiao", conversation_id="private-jiao", execution_mode="private"),
        dict(actor_id="laoke", room_id="room_weiwei_laoke", conversation_id="private-laoke", execution_mode="private"),
        dict(actor_id="jiao", room_id="room_group_home", conversation_id="group", execution_mode="group"),
        dict(actor_id="laoke", room_id="room_group_home", conversation_id="group", execution_mode="group"),
        dict(actor_id="jiao", room_id="room_weiwei_jiao", conversation_id="bedroom:a", execution_mode="bedroom"),
        dict(actor_id="jiao", room_id="room_weiwei_jiao", conversation_id="bedroom:b", execution_mode="bedroom"),
    ]
    store = InMemoryAnchoredHistoryStore()
    keys = [build_summary_namespace(**identity) for identity in identities]
    assert len(set(keys)) == 6
    for i, (key, identity) in enumerate(zip(keys, identities)):
        await store.get_or_create(key, identity=identity)
        await store.apply_compression(key, expected_revision=1, replacement_summary=f"scope {i}",
                                      summary_token_count=2, compressed_up_to_event_id=10)
    await store.delete_conversation_state("bedroom:a")
    for i, (key, identity) in enumerate(zip(keys, identities)):
        state = await store.get_or_create(key, identity=identity)
        assert state.summary == ("" if i == 4 else f"scope {i}")
    await store.invalidate_conversation_state("group")
    assert (await store.get_or_create(keys[0])).summary == "scope 0"
    assert (await store.get_or_create(keys[2])).summary == ""
    assert (await store.get_or_create(keys[3])).summary == ""


@pytest.mark.anyio
async def test_full_batch_leaves_tail_and_does_not_summarize_next_turn():
    store = InMemoryAnchoredHistoryStore()
    state = await store.get_or_create("summary")
    compactor = AnchoredHistoryCompactor()
    calls = []
    events = tuple(dict(event_id=i, actor_id="weiwei", content=str(i)) for i in range(1, 131))
    async def summarize(prior, selected):
        calls.append([event["event_id"] for event in selected])
        return "Complete earlier history"
    updated, tail = await compactor.maybe_compact(store=store, cache_namespace="summary",
        state=state, events=events, summarize=summarize)
    assert updated.compressed_up_to_event_id == 82
    assert calls == [list(range(1, 83))]
    assert len(tail) == 48
    next_tail = tail + (dict(event_id=131, actor_id="weiwei", content="next"),)
    again, remaining = await compactor.maybe_compact(store=store, cache_namespace="summary",
        state=updated, events=next_tail, summarize=summarize)
    assert again == updated and remaining == next_tail and len(calls) == 1


@pytest.mark.anyio
async def test_interrupted_summary_stays_paused_without_changing_prefix_or_other_scopes():
    import asyncio
    store = InMemoryAnchoredHistoryStore()
    state = await store.get_or_create("summary")
    events = tuple(dict(event_id=i, actor_id="weiwei", content="synthetic") for i in range(1, 131))
    calls = []
    async def interrupted(*args):
        calls.append(1)
        raise asyncio.CancelledError()
    with pytest.raises(asyncio.CancelledError):
        await AnchoredHistoryCompactor().maybe_compact(store=store, cache_namespace="summary",
            state=state, events=events, summarize=interrupted)
    restored = await store.get_or_create("summary")
    assert restored.compression_error == "summary_attempt_incomplete"
    assert (restored.summary, restored.state_revision, restored.compressed_up_to_event_id) == ("", 1, 0)
    with pytest.raises(AnchoredHistoryError, match="paused"):
        await AnchoredHistoryCompactor().maybe_compact(store=store, cache_namespace="summary",
            state=state, events=events, summarize=interrupted)
    assert calls == [1]
    other = await store.get_or_create("other-actor-room")
    async def success(*args): return "Other scope summary"
    result, tail = await AnchoredHistoryCompactor().maybe_compact(store=store, cache_namespace="other-actor-room",
        state=other, events=events, summarize=success)
    assert result.compressed_up_to_event_id == 82 and len(tail) == 48
