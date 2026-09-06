import pytest

from model_execution_contracts import ProviderUsage
from model_usage_store import (
    ExecutionReceiptDraft,
    InMemoryModelUsageStore,
    UsageStoreConflict,
    build_cache_namespace,
)


def _draft(**overrides):
    values = {
        "generation_request_id": "gen-1",
        "actor_id": "jiao",
        "room_id": "room_group_home",
        "conversation_id": "conversation-1",
        "profile_id": "profile-a",
        "profile_revision": 1,
        "provider": "provider-a",
        "protocol": "anthropic_messages",
        "route_id": "route-a",
        "model": "model-a",
        "adapter_version": "adapter.v1",
        "cache_strategy": "anthropic_prefix_anchored_v1",
        "requested_cache_ttl": "5m",
        "observed_cache_support": "verified",
        "fallback_used": False,
        "fallback_from_profile_id": None,
        "usage": ProviderUsage.from_provider_values(
            input_tokens=500,
            output_tokens=40,
            cache_creation_input_tokens=300,
            cache_read_input_tokens=200,
            cached_tokens=None,
        ),
        "status": "succeeded",
        "stable_prefix_hash": "stable-prefix-hash",
        "prompt_cache_key": None,
        "runtime_kernel_version": "kernel.v2",
        "persona_version": "jiao.v3",
        "room_policy_version": "private.v1",
        "tool_schema_hash": "tools-none",
        "summary_version": 4,
        "compressed_up_to_event_id": 88,
        "provider_usage_received": True,
        "execution_purpose": "generation",
    }
    values.update(overrides)
    return ExecutionReceiptDraft(**values)


@pytest.mark.anyio
async def test_execution_receipt_idempotent_by_generation_request_id():
    store = InMemoryModelUsageStore()
    first = await store.record(_draft())
    retry = await store.record(_draft())

    assert first is retry
    assert first.receipt_id == retry.receipt_id
    assert len(await store.list_receipts()) == 1


@pytest.mark.anyio
async def test_dashboard_receipts_are_newest_first_with_observation_time():
    store = InMemoryModelUsageStore()
    await store.record(_draft(generation_request_id="old"))
    await store.record(_draft(generation_request_id="new"))
    rows = await store.list_receipts()
    assert [r.generation_request_id for r in rows] == ["new", "old"]
    assert rows[0].created_at is not None


@pytest.mark.anyio
async def test_generation_identity_conflict_is_rejected():
    store = InMemoryModelUsageStore()
    await store.record(_draft())
    with pytest.raises(UsageStoreConflict):
        await store.record(_draft(model="different-model"))


@pytest.mark.anyio
@pytest.mark.parametrize("persistent", [False, True])
async def test_attempt_receipts_allow_one_generation_and_deduplicate_exact_payload(persistent, isolated_postgres, monkeypatch):
    import asyncio
    import database
    from postgres_model_stores import PostgresModelProfileStore, PostgresModelUsageStore
    from test_model_execution_service import _profile
    async def factory():
        return isolated_postgres
    if persistent:
        monkeypatch.setattr(database, "get_pool", factory)
        monkeypatch.setattr(database, "MEMORY_VECTOR_ENABLED", False)
        await database.init_tables()
        await PostgresModelProfileStore(factory).put_profile(_profile("profile-a"))
        store = PostgresModelUsageStore(factory)
    else:
        store = InMemoryModelUsageStore()
    first = _draft(receipt_id="attempt-one", status="failed")
    final = _draft(receipt_id="attempt-two")
    repeated = await asyncio.gather(*(store.record(first) for _ in range(3)))
    assert {item.receipt_id for item in repeated} == {"attempt-one"}
    await store.record(final)
    assert len(await store.list_receipts()) == 2
    with pytest.raises(UsageStoreConflict):
        await store.record(_draft(receipt_id="attempt-one", status="failed", model="different-model"))
    assert {item.model for item in await store.list_receipts()} == {"model-a"}


@pytest.mark.anyio
async def test_receipt_schema_upgrade_preserves_existing_history(isolated_postgres, monkeypatch):
    import database
    from postgres_model_stores import PostgresModelProfileStore, PostgresModelUsageStore
    from test_model_execution_service import _profile
    async def factory():
        return isolated_postgres
    monkeypatch.setattr(database, "get_pool", factory)
    monkeypatch.setattr(database, "MEMORY_VECTOR_ENABLED", False)
    await database.init_tables()
    await PostgresModelProfileStore(factory).put_profile(_profile("profile-a"))
    store = PostgresModelUsageStore(factory)
    original = await store.record(_draft())
    async with isolated_postgres.acquire() as conn:
        await conn.execute("ALTER TABLE model_execution_receipts ADD CONSTRAINT model_execution_receipts_generation_request_id_key UNIQUE(generation_request_id)")
        before = dict(await conn.fetchrow("SELECT * FROM model_execution_receipts"))
    await database.init_tables()
    async with isolated_postgres.acquire() as conn:
        assert dict(await conn.fetchrow("SELECT * FROM model_execution_receipts")) == before
    await store.record(_draft(receipt_id="new-attempt"))
    rows = await store.list_receipts()
    assert {row.receipt_id for row in rows} == {original.receipt_id, "new-attempt"}
    assert len({row.generation_request_id for row in rows}) == 1


@pytest.mark.anyio
async def test_provider_usage_round_trips_exact_values_and_nulls():
    store = InMemoryModelUsageStore()
    usage = ProviderUsage.from_provider_values(
        input_tokens=11,
        output_tokens=3,
        cache_creation_input_tokens=None,
        cache_read_input_tokens=None,
        cached_tokens=7,
    )
    receipt = await store.record(_draft(usage=usage))

    assert receipt.usage == usage
    assert receipt.usage.cache_creation_input_tokens is None
    assert receipt.usage.cached_tokens == 7
    assert receipt.stable_prefix_hash == "stable-prefix-hash"
    assert receipt.summary_version == 4
    assert receipt.compressed_up_to_event_id == 88
    assert receipt.provider_usage_received is True
    assert receipt.execution_purpose == "generation"


@pytest.mark.anyio
async def test_execution_purpose_distinguishes_cache_keepalive_from_result_status():
    store = InMemoryModelUsageStore()

    receipt = await store.record(
        _draft(status="succeeded", execution_purpose="cache_keepalive")
    )

    assert receipt.status == "succeeded"
    assert receipt.execution_purpose == "cache_keepalive"


def test_cache_namespace_includes_actor_conversation_profile_and_versions():
    baseline = build_cache_namespace(
        actor_id="jiao",
        conversation_id="conversation-1",
        profile_id="profile-a",
        profile_revision=1,
        execution_mode="private",
        actor_prompt_version="jiao.v3",
        runtime_kernel_version="kernel.v2",
        room_policy_version="private.v1",
        tool_schema_hash="tools-none",
        cache_strategy_version="anthropic_prefix_anchored_v1",
    )
    dimensions = {
        "actor_id": "laoke",
        "conversation_id": "conversation-2",
        "profile_id": "profile-b",
        "profile_revision": 2,
        "execution_mode": "bedroom",
        "actor_prompt_version": "jiao.v4",
        "runtime_kernel_version": "kernel.v3",
        "room_policy_version": "bedroom.v1",
        "tool_schema_hash": "tools-v2",
        "cache_strategy_version": "no_prompt_cache_v1",
    }
    defaults = {
        "actor_id": "jiao",
        "conversation_id": "conversation-1",
        "profile_id": "profile-a",
        "profile_revision": 1,
        "execution_mode": "private",
        "actor_prompt_version": "jiao.v3",
        "runtime_kernel_version": "kernel.v2",
        "room_policy_version": "private.v1",
        "tool_schema_hash": "tools-none",
        "cache_strategy_version": "anthropic_prefix_anchored_v1",
    }
    for field, replacement in dimensions.items():
        changed = {**defaults, field: replacement}
        assert build_cache_namespace(**changed) != baseline
