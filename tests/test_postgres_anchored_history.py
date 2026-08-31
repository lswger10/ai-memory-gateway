import json

import pytest

from anchored_history import AnchoredHistoryError, PostgresAnchoredHistoryStore


class _Transaction:
    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        return False


class _Connection:
    def __init__(self, rows):
        self.rows = rows

    def transaction(self):
        return _Transaction()

    async def fetchrow(self, sql, *args):
        if "INSERT INTO model_cache_state" in sql:
            namespace = args[0]
            self.rows.setdefault(
                namespace,
                {
                    "cache_namespace": namespace,
                    "compressed_up_to_event_id": 0,
                    "summary": "",
                    "summary_token_count": 0,
                    "state_revision": 1,
                },
            )
            return dict(self.rows[namespace])
        if "FOR UPDATE" in sql:
            return dict(self.rows[args[0]])
        if "UPDATE model_cache_state" in sql:
            namespace, expected, summary, count, cursor = args
            row = self.rows[namespace]
            if row["state_revision"] != expected:
                return None
            row.update(
                summary=summary,
                summary_token_count=count,
                compressed_up_to_event_id=cursor,
                state_revision=expected + 1,
            )
            return dict(row)
        raise AssertionError(sql)


class _Acquire:
    def __init__(self, conn):
        self.conn = conn

    async def __aenter__(self):
        return self.conn

    async def __aexit__(self, *args):
        return False


class _Pool:
    def __init__(self):
        self.rows = {}
        self.conn = _Connection(self.rows)

    def acquire(self):
        return _Acquire(self.conn)


@pytest.mark.anyio
async def test_postgres_cache_state_survives_gateway_store_recreation():
    pool = _Pool()
    first = PostgresAnchoredHistoryStore(lambda: _pool(pool))
    state = await first.get_or_create(
        "namespace-1",
        identity={
            "actor_id": "jiao",
            "conversation_id": "conversation-1",
            "profile_id": "profile-1",
            "profile_revision": 1,
            "execution_mode": "private",
            "actor_prompt_version": "actor.v1",
            "runtime_kernel_version": "runtime.v1",
            "room_policy_version": "room.v1",
            "tool_schema_hash": "tools.v1",
            "cache_strategy_version": "anthropic_prefix_anchored_v1",
        },
    )
    await first.apply_compression(
        "namespace-1",
        expected_revision=state.state_revision,
        replacement_summary="bounded summary",
        summary_token_count=2,
        compressed_up_to_event_id=42,
    )

    recreated = PostgresAnchoredHistoryStore(lambda: _pool(pool))
    restored = await recreated.get_or_create(
        "namespace-1",
        identity={
            "actor_id": "jiao",
            "conversation_id": "conversation-1",
            "profile_id": "profile-1",
            "profile_revision": 1,
            "execution_mode": "private",
            "actor_prompt_version": "actor.v1",
            "runtime_kernel_version": "runtime.v1",
            "room_policy_version": "room.v1",
            "tool_schema_hash": "tools.v1",
            "cache_strategy_version": "anthropic_prefix_anchored_v1",
        },
    )

    assert restored.compressed_up_to_event_id == 42
    assert restored.summary == "bounded summary"
    assert restored.state_revision == 2


@pytest.mark.anyio
async def test_postgres_compression_rejects_stale_revision_without_partial_cursor_move():
    pool = _Pool()
    store = PostgresAnchoredHistoryStore(lambda: _pool(pool))
    identity = {
        "actor_id": "laoke",
        "conversation_id": "conversation-2",
        "profile_id": "profile-2",
        "profile_revision": 1,
        "execution_mode": "private",
        "actor_prompt_version": "actor.v1",
        "runtime_kernel_version": "runtime.v1",
        "room_policy_version": "room.v1",
        "tool_schema_hash": "tools.v1",
        "cache_strategy_version": "anthropic_prefix_anchored_v1",
    }
    await store.get_or_create("namespace-2", identity=identity)

    with pytest.raises(AnchoredHistoryError, match="revision conflict"):
        await store.apply_compression(
            "namespace-2",
            expected_revision=99,
            replacement_summary="must not persist",
            summary_token_count=3,
            compressed_up_to_event_id=100,
        )

    row = pool.rows["namespace-2"]
    assert row["summary"] == ""
    assert row["compressed_up_to_event_id"] == 0
    assert row["state_revision"] == 1


async def _pool(pool):
    return pool


class _PostgresCteVisibilityConnection:
    """Model PostgreSQL's same-statement visibility for data-modifying CTEs."""

    async def fetchrow(self, sql, *args):
        if "INSERT INTO model_cache_state" not in sql:
            raise AssertionError(sql)
        if "FROM inserted" not in sql:
            return None
        return {
            "cache_namespace": args[0],
            "compressed_up_to_event_id": 0,
            "summary": "",
            "summary_token_count": 0,
            "state_revision": 1,
        }


class _PostgresCteVisibilityPool:
    def acquire(self):
        return _Acquire(_PostgresCteVisibilityConnection())


@pytest.mark.anyio
async def test_postgres_cache_state_creation_reads_insert_returning_row():
    pool = _PostgresCteVisibilityPool()
    store = PostgresAnchoredHistoryStore(lambda: _pool(pool))

    state = await store.get_or_create(
        "namespace-new",
        identity={
            "actor_id": "laoke",
            "conversation_id": "conversation-new",
            "profile_id": "profile-new",
            "profile_revision": 1,
            "execution_mode": "private",
            "actor_prompt_version": "actor.v2",
            "runtime_kernel_version": "runtime.v1",
            "room_policy_version": "room.v1",
            "tool_schema_hash": "tools.v1",
            "cache_strategy_version": "anthropic_prefix_anchored_v1",
        },
    )

    assert state.cache_namespace == "namespace-new"
    assert state.state_revision == 1
