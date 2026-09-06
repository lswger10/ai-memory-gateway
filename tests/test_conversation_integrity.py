"""F02: real database mutation boundaries and canonical read repair."""
import asyncio
from unittest.mock import AsyncMock

import httpx
import pytest

import database
import main
from anchored_history import AnchoredHistoryError, PostgresAnchoredHistoryStore
from conversation_partitions import ConversationFact, PostgresConversationPartitionStore
from conversation_sync import ConversationSyncService, ConversationSyncIncomplete
from test_conversation_sync import FakeRelay, relay_event


@pytest.fixture
async def history_db(monkeypatch, isolated_postgres):
    async def pool_factory():
        return isolated_postgres

    monkeypatch.setattr(database, "get_pool", pool_factory)
    monkeypatch.setattr(main, "MEMORY_ENABLED", True)
    monkeypatch.setattr(main, "GATEWAY_SECRET", "test-admin")
    await database.init_tables()
    await database.ensure_token_usage_table()
    store = PostgresConversationPartitionStore(pool_factory)
    await store.append_accepted_facts(tuple(ConversationFact.from_relay_event(relay_event(i)) for i in (1, 2)))
    await store.mark_synced_through("group-1", 2)
    async with isolated_postgres.acquire() as conn:
        derived = await conn.fetchval("SELECT id FROM conversations WHERE source_event_id=1")
        legacy = await conn.fetchval("INSERT INTO conversations(session_id,role,content) VALUES('legacy','user','original') RETURNING id")
    return isolated_postgres, store, derived, legacy


@pytest.mark.anyio
@pytest.mark.parametrize("mutation", ["edit", "single_delete", "delete", "batch_delete", "merge_source", "merge_target"])
async def test_legacy_http_mutations_reject_derived_rows_without_partial_writes(history_db, mutation):
    pool, store, derived, legacy = history_db
    actions = {
        "edit": ("PATCH", f"/api/chat/messages/{derived}", {"content": "changed"}),
        "single_delete": ("DELETE", f"/api/chat/messages/{derived}", None),
        "delete": ("DELETE", "/api/conversations/group-1", None),
        "batch_delete": ("POST", "/api/conversations/batch-delete", {"session_ids": ["legacy", "group-1"]}),
        "merge_source": ("POST", "/api/admin/merge-sessions", {"source_ids": ["group-1"], "target_id": "legacy"}),
        "merge_target": ("POST", "/api/admin/merge-sessions", {"source_ids": ["legacy"], "target_id": "group-1"}),
    }
    method, path, body = actions[mutation]
    before = await store.list_facts("group-1")
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=main.app), base_url="http://test", headers={"X-Gateway-Key": "test-admin"}) as client:
        response = await client.request(method, path, json=body)
    assert response.status_code == 409
    assert response.json()["error"] == "relay_derived_conversation_read_only"
    assert await store.list_facts("group-1") == before
    async with pool.acquire() as conn:
        assert await conn.fetchval("SELECT content FROM conversations WHERE id=$1", legacy) == "original"
        assert await conn.fetchval("SELECT session_id FROM conversations WHERE id=$1", legacy) == "legacy"


@pytest.mark.anyio
async def test_legacy_management_and_derived_reads_remain_available(history_db):
    pool, store, derived, legacy = history_db
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=main.app), base_url="http://test", headers={"X-Gateway-Key": "test-admin"}) as client:
        assert (await client.patch(f"/api/chat/messages/{legacy}", json={"content": "edited"})).json()["status"] == "ok"
        exported = await client.get("/api/conversations/export")
        assert exported.status_code == 200
        assert "message-1" in exported.text
        merged = await client.post("/api/admin/merge-sessions", json={"source_ids": ["legacy"], "target_id": "legacy-target"})
        assert merged.json()["merged_messages"] == 1
        assert (await client.delete("/api/conversations/legacy-target")).json()["status"] == "ok"
    assert len(await store.list_facts("group-1")) == 2
    async with pool.acquire() as conn:
        assert await conn.fetchval("SELECT count(*) FROM conversations WHERE fact_identity IS NULL") == 0


@pytest.mark.anyio
@pytest.mark.parametrize("damage", ["missing", "content", "metadata"])
async def test_history_before_watermark_is_repaired_from_relay(history_db, damage):
    pool, store, derived, _ = history_db
    async with pool.acquire() as conn:
        if damage == "missing":
            await conn.execute("DELETE FROM conversations WHERE id=$1", derived)
        elif damage == "content":
            await conn.execute("UPDATE conversations SET content='corrupted' WHERE id=$1", derived)
        else:
            await conn.execute("UPDATE conversations SET actor_id='laoke' WHERE id=$1", derived)
    relay = FakeRelay(relay_event(i) for i in (1, 2, 3))
    receipt = await ConversationSyncService(relay, store).ensure_relay_synced(
        actor_id="jiao", room_id="room_group_home", conversation_id="group-1", current_event_id=3)
    facts = await store.list_facts("group-1")
    assert [f.source_event_id for f in facts] == [1, 2, 3]
    assert facts[0].content == "message-1"
    assert facts[0].actor_id == "weiwei"
    assert receipt.synced_through_event_id == 3


@pytest.mark.anyio
async def test_missing_current_fact_cannot_advance_the_watermark(history_db):
    _, store, _, _ = history_db
    with pytest.raises(ConversationSyncIncomplete):
        await ConversationSyncService(FakeRelay((relay_event(1), relay_event(2))), store).ensure_relay_synced(
            actor_id="jiao", room_id="room_group_home", conversation_id="group-1", current_event_id=3)
    assert await store.synced_through_event_id("group-1") == 2


@pytest.mark.anyio
@pytest.mark.parametrize("damage", [False, True])
async def test_repair_invalidates_summary_and_fences_old_compression(history_db, damage):
    pool, store, derived, _ = history_db
    history = PostgresAnchoredHistoryStore(database.get_pool)
    identity = {"actor_id": "jiao", "conversation_id": "group-1", "profile_id": "test",
                "profile_revision": 1, "execution_mode": "group", "actor_prompt_version": "v1",
                "runtime_kernel_version": "v1", "room_policy_version": "v1",
                "tool_schema_hash": "v1", "cache_strategy_version": "v1"}
    async with pool.acquire() as conn:
        await conn.execute("INSERT INTO model_profiles(profile_id,profile_json) VALUES('test','{}')")
    initial = await history.get_or_create("test-namespace", identity=identity)
    compressed = await history.apply_compression("test-namespace", expected_revision=initial.state_revision,
        replacement_summary="old summary", summary_token_count=2, compressed_up_to_event_id=2)
    if damage:
        async with pool.acquire() as conn:
            await conn.execute("DELETE FROM conversations WHERE id=$1", derived)
    service = ConversationSyncService(FakeRelay(relay_event(i) for i in (1, 2, 3)), store, history)
    await service.ensure_relay_synced(actor_id="jiao", room_id="room_group_home", conversation_id="group-1", current_event_id=3)
    state = await history.get_or_create("test-namespace", identity=identity)
    if damage:
        assert state.summary == ""
        assert state.compressed_up_to_event_id == 0
        assert state.state_revision > compressed.state_revision
        with pytest.raises(AnchoredHistoryError, match="revision"):
            await history.apply_compression("test-namespace", expected_revision=compressed.state_revision,
                replacement_summary="late stale summary", summary_token_count=3, compressed_up_to_event_id=3)
    else:
        assert state == compressed
    await service.ensure_relay_synced(actor_id="laoke", room_id="room_group_home", conversation_id="group-1", current_event_id=3)
    assert await history.get_or_create("test-namespace", identity=identity) == state


@pytest.mark.anyio
async def test_failed_summary_invalidation_prevents_repair_and_watermark_advance(history_db):
    pool, store, derived, _ = history_db
    async with pool.acquire() as conn:
        await conn.execute("DELETE FROM conversations WHERE id=$1", derived)
    history = AsyncMock()
    history.invalidate_conversation_state.side_effect = RuntimeError("invalidation unavailable")
    with pytest.raises(RuntimeError, match="invalidation unavailable"):
        await ConversationSyncService(FakeRelay(relay_event(i) for i in (1, 2, 3)), store, history).ensure_relay_synced(
            actor_id="jiao", room_id="room_group_home", conversation_id="group-1", current_event_id=3)
    assert [f.source_event_id for f in await store.list_facts("group-1")] == [2]
    assert await store.synced_through_event_id("group-1") == 2


@pytest.mark.anyio
@pytest.mark.parametrize("events", [(1, 1, 2, 3), (2, 3)])
async def test_duplicate_or_truncated_authority_history_fails_without_mutation(history_db, events):
    _, store, _, _ = history_db
    before = await store.list_facts("group-1")
    with pytest.raises(ConversationSyncIncomplete):
        await ConversationSyncService(FakeRelay(relay_event(i) for i in events), store).ensure_relay_synced(
            actor_id="jiao", room_id="room_group_home", conversation_id="group-1", current_event_id=3)
    assert await store.list_facts("group-1") == before
    assert await store.synced_through_event_id("group-1") == 2


@pytest.mark.anyio
@pytest.mark.parametrize("existing_namespace", [False, True])
async def test_cache_reader_waits_for_atomic_repair_even_for_new_namespace(history_db, monkeypatch, existing_namespace):
    pool, store, derived, _ = history_db
    history = PostgresAnchoredHistoryStore(database.get_pool)
    identity = {"actor_id": "jiao", "conversation_id": "group-1", "profile_id": "test",
                "profile_revision": 1, "execution_mode": "group", "actor_prompt_version": "v1",
                "runtime_kernel_version": "v1", "room_policy_version": "v1",
                "tool_schema_hash": "v1", "cache_strategy_version": "v1"}
    async with pool.acquire() as conn:
        await conn.execute("INSERT INTO model_profiles(profile_id,profile_json) VALUES('test','{}')")
        await conn.execute("UPDATE conversations SET content='CORRUPTED' WHERE id=$1", derived)
    if existing_namespace:
        await history.get_or_create("concurrent", identity=identity)
    reached, release = asyncio.Event(), asyncio.Event()
    invalidate = history.invalidate_conversation_state

    async def paused_invalidation(*args, **kwargs):
        await invalidate(*args, **kwargs)
        reached.set()
        await release.wait()

    monkeypatch.setattr(history, "invalidate_conversation_state", paused_invalidation)
    service = ConversationSyncService(FakeRelay(relay_event(i) for i in (1, 2, 3)), store, history)
    repair = asyncio.create_task(service.ensure_relay_synced(
        actor_id="jiao", room_id="room_group_home", conversation_id="group-1", current_event_id=3))
    reader = None
    try:
        await asyncio.wait_for(reached.wait(), 5)
        reader = asyncio.create_task(history.get_or_create("concurrent", identity=identity))
        async with asyncio.timeout(5):
            async with pool.acquire() as conn:
                while not await conn.fetchval("SELECT count(*) FROM pg_locks WHERE locktype='advisory' AND NOT granted"):
                    await asyncio.sleep(0.01)
        assert not reader.done()
    finally:
        release.set()
        await repair
        if reader is not None:
            state = await reader
    facts = await store.list_facts("group-1", after_event_id=state.compressed_up_to_event_id)
    assert facts[0].content == "message-1"
    assert state.summary == ""
    assert await store.synced_through_event_id("group-1") == 3
