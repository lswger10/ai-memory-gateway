import pytest

from anchored_history import (
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
