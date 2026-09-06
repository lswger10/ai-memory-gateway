"""F03/F04: replacement must preserve classification and its original sources."""
from datetime import date
import asyncio
import json
import re

import httpx
import pytest

import database
import main
from memory_policy import MemoryScope, MemoryType, MemoryWrite, Perspective, SourceKind


@pytest.fixture
async def memory_db(monkeypatch, isolated_postgres):
    async def factory():
        return isolated_postgres

    monkeypatch.setattr(database, "get_pool", factory)
    monkeypatch.setattr(database, "MEMORY_VECTOR_ENABLED", False)
    await database.init_tables()

    async def seed(content, *, scope="weiwei-jiao", confidential=True, perspective="jiao", evidence=(101,), source_kind="chat_extraction"):
        write = MemoryWrite(content=content, scope=MemoryScope(scope), memory_type=MemoryType.FACT,
            perspective=Perspective(perspective), confidential=confidential, source_kind=SourceKind(source_kind),
            provenance={"conversation_id": "original-session", "evidence_event_ids": list(evidence), "source_event_id": evidence[0]})
        memory_id = await database.create_typed_memory(write)
        async with isolated_postgres.acquire() as conn:
            await conn.execute("UPDATE memories SET created_at='2026-09-06T00:00:00Z' WHERE id=$1", memory_id)
        return memory_id

    return isolated_postgres, seed


def provider(monkeypatch, output):
    requests = []
    original_client = httpx.AsyncClient

    async def respond(request):
        prompt = json.loads(request.content)["messages"][0]["content"]
        ids = [int(value) for value in re.findall(r"\[ID=(\d+)\]", prompt)]
        requests.append(ids)
        events = await output(ids) if callable(output) else output
        return httpx.Response(200, json={"choices": [{"message": {"content": json.dumps(events)}}]})

    monkeypatch.setattr(main, "MEMORY_API_BASE_URL", "http://synthetic/memory")
    monkeypatch.setattr(main, "MEMORY_MODEL", "synthetic")
    monkeypatch.setattr(main, "get_memory_api_key", lambda: "synthetic")
    monkeypatch.setattr(main.httpx, "AsyncClient", lambda **kwargs: original_client(transport=httpx.MockTransport(respond), **kwargs))
    return requests


async def rows(pool):
    async with pool.acquire() as conn:
        return [dict(row) for row in await conn.fetch("SELECT * FROM memories ORDER BY id")]


@pytest.mark.anyio
async def test_manual_merge_preserves_scope_confidentiality_perspective_and_evidence(memory_db):
    pool, seed = memory_db
    ids = [await seed("first", evidence=(101,)), await seed("second", evidence=(102,))]
    new_id = await database.merge_memories(ids, "combined", "combined facts", 8)
    data = await rows(pool)
    merged = next(row for row in data if row["id"] == new_id)
    assert (merged["scope"], merged["confidential"], merged["perspective"], merged["memory_type"], merged["source_kind"]) == (
        "weiwei-jiao", True, "jiao", "fact", "chat_extraction")
    assert json.loads(merged["evidence"]) == [101, 102]
    assert merged["evidence_count"] == 2
    assert merged["merged_from"] == ids
    provenance = json.loads(merged["provenance"])
    assert {source["id"] for source in provenance["merged_sources"]} == set(ids)
    assert all(source["source_session"] == "original-session" for source in provenance["merged_sources"])
    assert all(not row["is_active"] and row["superseded_by"] == new_id for row in data if row["id"] in ids)


@pytest.mark.anyio
@pytest.mark.parametrize("difference", [{"scope": "weiwei-laoke"}, {"confidential": False}, {"perspective": "weiwei"}])
async def test_manual_merge_cannot_cross_classification_boundary(memory_db, difference):
    pool, seed = memory_db
    ids = [await seed("first"), await seed("second", **difference)]
    before = await rows(pool)
    with pytest.raises(ValueError):
        await database.merge_memories(ids, "mixed", "mixed result", 5)
    assert await rows(pool) == before


@pytest.mark.anyio
async def test_empty_consolidation_is_explicit_no_change(memory_db, monkeypatch):
    pool, seed = memory_db
    await seed("keep this")
    before = await rows(pool)
    provider(monkeypatch, [])
    result = await main.consolidate_memories_for_date(date(2026, 9, 6))
    assert result["status"] == "no_changes"
    assert result["events_created"] == 0
    assert await rows(pool) == before


@pytest.mark.anyio
async def test_partial_consolidation_only_retires_covered_sources(memory_db, monkeypatch):
    pool, seed = memory_db
    covered, retained = await seed("covered"), await seed("retained")
    provider(monkeypatch, [{"title": "one", "content": "replacement", "importance": 5, "merged_ids": [covered]}])
    result = await main.consolidate_memories_for_date(date(2026, 9, 6))
    assert result["status"] == "ok"
    data = {row["id"]: row for row in await rows(pool)}
    assert data[retained]["is_active"]
    assert not data[covered]["is_active"]
    assert result["fragments_processed"] == 1
    replacement = data[data[covered]["superseded_by"]]
    assert replacement["scope"] == "weiwei-jiao" and replacement["confidential"]


@pytest.mark.anyio
async def test_second_replacement_database_failure_rolls_back_whole_batch(memory_db, monkeypatch):
    pool, seed = memory_db
    ids = [await seed("first"), await seed("second")]
    async with pool.acquire() as conn:
        await conn.execute("ALTER TABLE memories ADD CONSTRAINT synthetic_failure CHECK (content <> 'second-result')")
    before = await rows(pool)
    provider(monkeypatch, [{"title": "result", "content": content, "importance": 5, "merged_ids": [memory_id]}
        for memory_id, content in zip(ids, ("first-result", "second-result"))])
    result = await main.consolidate_memories_for_date(date(2026, 9, 6))
    assert result["status"] == "error"
    assert await rows(pool) == before


@pytest.mark.anyio
@pytest.mark.parametrize("invalid", ["external", "duplicate"])
async def test_untrusted_output_cannot_reuse_or_invent_sources(memory_db, monkeypatch, invalid):
    pool, seed = memory_db
    memory_id = await seed("original")
    before = await rows(pool)
    ids = [memory_id, 999999] if invalid == "external" else [memory_id, memory_id]
    provider(monkeypatch, [{"title": "bad", "content": "bad result", "importance": 5, "merged_ids": ids}])
    result = await main.consolidate_memories_for_date(date(2026, 9, 6))
    assert result["status"] == "error"
    assert await rows(pool) == before


@pytest.mark.anyio
async def test_consolidation_provider_inputs_are_partitioned_before_call(memory_db, monkeypatch):
    pool, seed = memory_db
    ids = [await seed("private-secret"), await seed("private-open", confidential=False),
           await seed("other-private", scope="weiwei-laoke", perspective="laoke"),
           await seed("group", scope="group", confidential=False)]
    before = await rows(pool)
    requests = provider(monkeypatch, [])
    result = await main.consolidate_memories_for_date(date(2026, 9, 6))
    assert sorted(requests) == [[memory_id] for memory_id in ids]
    assert result["status"] == "no_changes"
    assert await rows(pool) == before


@pytest.mark.anyio
async def test_source_edit_during_provider_call_is_not_superseded(memory_db, monkeypatch):
    pool, seed = memory_db
    memory_id = await seed("before edit")

    async def output(ids):
        async with pool.acquire() as conn:
            await conn.execute("UPDATE memories SET content='concurrent edit' WHERE id=$1", memory_id)
        return [{"title": "stale", "content": "stale result", "merged_ids": ids}]

    provider(monkeypatch, output)
    result = await main.consolidate_memories_for_date(date(2026, 9, 6))
    assert result["status"] == "error"
    data = await rows(pool)
    assert len(data) == 1 and data[0]["content"] == "concurrent edit" and data[0]["is_active"]


@pytest.mark.anyio
async def test_duplicate_replacement_does_not_supersede_itself(memory_db):
    pool, seed = memory_db
    memory_id = await seed("same content")
    before = await rows(pool)
    with pytest.raises(ValueError, match="duplicate"):
        await database.merge_memories([memory_id], "same", "same content", 5)
    assert await rows(pool) == before


@pytest.mark.anyio
async def test_concurrent_merge_claims_sources_only_once(memory_db):
    pool, seed = memory_db
    ids = [await seed("first"), await seed("second")]
    results = await asyncio.gather(
        database.merge_memories(ids, "one", "result one", 5),
        database.merge_memories(ids, "two", "result two", 5), return_exceptions=True)
    assert sum(type(result) is int for result in results) == 1
    assert sum(isinstance(result, ValueError) for result in results) == 1
    data = await rows(pool)
    assert len(data) == 3 and sum(row["is_active"] for row in data) == 1


@pytest.mark.anyio
async def test_legacy_merge_remains_quarantined_and_revert_restores_sources(memory_db):
    pool, _ = memory_db
    async with pool.acquire() as conn:
        first = await conn.fetchval("INSERT INTO memories(content) VALUES('legacy first') RETURNING id")
        second = await conn.fetchval("INSERT INTO memories(content) VALUES('legacy second') RETURNING id")
    new_id = await database.merge_memories([first, second], "legacy", "legacy merged", 5)
    data = await rows(pool)
    assert all(row["scope"] == "legacy_unscoped" and row["perspective"] is None for row in data)
    reverted = await database.revert_merge(new_id)
    assert reverted["status"] == "ok"
    data = await rows(pool)
    assert [row["id"] for row in data] == [first, second]
    assert all(row["is_active"] and row["status"] == "active" and row["superseded_by"] is None for row in data)


@pytest.mark.anyio
@pytest.mark.parametrize("invalid", ["mixed_confidential", "duplicate_content"])
async def test_actor_merge_uses_same_replacement_boundary(memory_db, invalid):
    from actor_memory_tools import PostgresActorMemoryToolStore
    from test_actor_memory_tools import context
    pool, seed = memory_db
    ids = [await seed("first", confidential=False, source_kind="actor_tool"),
           await seed("second", confidential=invalid == "mixed_confidential", source_kind="actor_tool")]
    before = await rows(pool)
    store = PostgresActorMemoryToolStore(database.get_pool)
    async with pool.acquire() as conn:
        with pytest.raises(ValueError):
            async with conn.transaction():
                await store._apply(conn, context(), "merge_memories",
                    {"memory_ids": ids, "content": "first" if invalid == "duplicate_content" else "combined", "importance": 5}, 999)
    assert await rows(pool) == before


@pytest.mark.anyio
async def test_missing_permanently_deleted_source_blocks_revert(memory_db):
    pool, seed = memory_db
    first, second = await seed("first"), await seed("second")
    replacement = await database.merge_memories([first, second], "both", "both facts", 5)
    await database.delete_memory(first)
    before = await rows(pool)
    result = await database.revert_merge(replacement)
    assert "error" in result
    assert await rows(pool) == before


@pytest.mark.anyio
async def test_actor_merge_cannot_override_source_classification_or_evidence(memory_db):
    from actor_memory_tools import PostgresActorMemoryToolStore, ActorMemoryToolLibrary
    from test_actor_memory_tools import context
    pool, seed = memory_db
    ids = [await seed("first"), await seed("second")]
    args = {"memory_ids": ids, "content": "public leak", "importance": 5, "scope": "group",
            "confidential": False, "perspective": "shared", "evidence_event_ids": []}
    before = await rows(pool)
    store = PostgresActorMemoryToolStore(database.get_pool)
    with pytest.raises(ValueError):
        await ActorMemoryToolLibrary(store).call(context(), "injected", "merge_memories", args)
    # Old persisted stages must also be rejected at the actual mutation boundary.
    async with pool.acquire() as conn:
        with pytest.raises(ValueError):
            async with conn.transaction():
                await store._apply(conn, context(), "merge_memories", args, 999)
    assert await rows(pool) == before


@pytest.mark.anyio
async def test_actor_generations_cannot_replace_the_same_sources_twice(memory_db):
    from actor_memory_tools import PostgresActorMemoryToolStore, ActorMemoryToolLibrary
    from test_actor_memory_tools import context
    pool, seed = memory_db
    ids = [await seed("first"), await seed("second")]
    library = ActorMemoryToolLibrary(PostgresActorMemoryToolStore(database.get_pool))
    contexts = [context(generation="merge-one"), context(generation="merge-two")]
    for i, ctx in enumerate(contexts):
        await library.call(ctx, "merge", "merge_memories", {"memory_ids": ids, "content": f"result {i}", "importance": 5})
    await library.commit_accepted(contexts[0], accepted_event_id=999)
    before = await rows(pool)
    with pytest.raises(ValueError):
        await library.commit_accepted(contexts[1], accepted_event_id=1000)
    assert await rows(pool) == before
