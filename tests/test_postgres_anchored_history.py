import json

import pytest

from anchored_history import AnchoredHistoryError, PostgresAnchoredHistoryStore, build_summary_namespace


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

    async def execute(self, sql, *args):
        assert "pg_advisory_xact_lock" in sql
        assert len(args) == 1

    async def fetch(self, sql, *args):
        assert "FROM model_cache_state" in sql
        return []

    async def fetchrow(self, sql, *args):
        if sql.startswith("SELECT summary_namespace"):
            return self.rows.get(args[0])
        if "INSERT INTO conversation_summary_state" in sql:
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
        if "SET compression_error=$3" in sql:
            namespace, expected, reason = args
            row = self.rows[namespace]
            if row["state_revision"] != expected:
                return None
            row["compression_error"] = reason
            return dict(row)
        if "UPDATE conversation_summary_state" in sql:
            namespace, expected, summary, count, cursor = args
            row = self.rows[namespace]
            if row["state_revision"] != expected:
                return None
            row.update(
                summary=summary,
                summary_token_count=count,
                compressed_up_to_event_id=cursor,
                state_revision=expected + 1,
                compression_error=None,
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
        build_summary_namespace(actor_id="jiao", room_id="room_weiwei_jiao", conversation_id="conversation-1", execution_mode="private"),
        identity={
            "actor_id": "jiao",
            "room_id": "room_weiwei_jiao",
            "conversation_id": "conversation-1",
            "execution_mode": "private",
        },
    )
    await first.apply_compression(
        build_summary_namespace(actor_id="jiao", room_id="room_weiwei_jiao", conversation_id="conversation-1", execution_mode="private"),
        expected_revision=state.state_revision,
        replacement_summary="bounded summary",
        summary_token_count=2,
        compressed_up_to_event_id=42,
    )

    recreated = PostgresAnchoredHistoryStore(lambda: _pool(pool))
    restored = await recreated.get_or_create(
        build_summary_namespace(actor_id="jiao", room_id="room_weiwei_jiao", conversation_id="conversation-1", execution_mode="private"),
        identity={
            "actor_id": "jiao",
            "room_id": "room_weiwei_jiao",
            "conversation_id": "conversation-1",
            "execution_mode": "private",
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
            "room_id": "room_weiwei_laoke",
        "conversation_id": "conversation-2",
        "execution_mode": "private",
    }
    await store.get_or_create(build_summary_namespace(actor_id="laoke", room_id="room_weiwei_laoke", conversation_id="conversation-2", execution_mode="private"), identity=identity)

    with pytest.raises(AnchoredHistoryError, match="revision conflict"):
        await store.apply_compression(
            build_summary_namespace(actor_id="laoke", room_id="room_weiwei_laoke", conversation_id="conversation-2", execution_mode="private"),
            expected_revision=99,
            replacement_summary="must not persist",
            summary_token_count=3,
            compressed_up_to_event_id=100,
        )

    row = pool.rows[build_summary_namespace(actor_id="laoke", room_id="room_weiwei_laoke", conversation_id="conversation-2", execution_mode="private")]
    assert row["summary"] == ""
    assert row["compressed_up_to_event_id"] == 0
    assert row["state_revision"] == 1


async def _pool(pool):
    return pool


async def _assert_persistent_pause(factory, restart):
    from anchored_history import AnchoredHistoryCompactor, SummaryCompressionError
    identity = dict(actor_id="laoke", room_id="room_weiwei_laoke",
                    conversation_id="pause-test", execution_mode="private")
    key = build_summary_namespace(**identity)
    store = PostgresAnchoredHistoryStore(factory)
    state = await store.get_or_create(key, identity=identity)
    events = tuple(dict(event_id=i, actor_id="weiwei", content=f"synthetic-{i}") for i in range(1, 131))
    calls = []

    async def fail(*args):
        calls.append(1)
        raise SummaryCompressionError("summary_empty")

    with pytest.raises(SummaryCompressionError):
        await AnchoredHistoryCompactor().maybe_compact(store=store, cache_namespace=key,
            state=state, events=events, summarize=fail, identity=identity)
    await restart()
    restarted = PostgresAnchoredHistoryStore(factory)
    restored = await restarted.get_or_create(key, identity=identity)
    assert restored.compression_error == "summary_empty"
    assert (restored.summary, restored.state_revision, restored.compressed_up_to_event_id) == ("", 1, 0)
    for _ in range(2):
        with pytest.raises(SummaryCompressionError, match="summary_empty"):
            await AnchoredHistoryCompactor().maybe_compact(store=restarted, cache_namespace=key,
                state=restored, events=events, summarize=fail, identity=identity)
    assert len(calls) == 1
    for changes in (dict(actor_id="jiao", room_id="room_weiwei_jiao"),
                    dict(room_id="room_group_home", execution_mode="group"),
                    dict(conversation_id="bedroom:pause-test", execution_mode="bedroom")):
        other = dict(identity, **changes)
        assert (await restarted.get_or_create(build_summary_namespace(**other), identity=other)).compression_error is None

    async def succeed(*args):
        calls.append(1)
        return "Complete synthetic summary"

    recovered, tail = await AnchoredHistoryCompactor().maybe_compact(store=restarted,
        cache_namespace=key, state=restored, events=events, summarize=succeed, force=True, identity=identity)
    assert len(calls) == 2 and len(tail) == 48
    assert recovered.compression_error is None and recovered.compressed_up_to_event_id == 82
    with pytest.raises(AnchoredHistoryError, match="revision conflict"):
        await store.set_compression_error(key, expected_revision=1, reason="summary_empty")
    assert await restarted.get_or_create(key, identity=identity) == recovered


@pytest.mark.anyio
async def test_postgres_pause_store_recreation_and_explicit_recovery():
    pool = _Pool()
    async def restart(): pass
    await _assert_persistent_pause(lambda: _pool(pool), restart)


@pytest.mark.anyio
async def test_real_postgres_pause_survives_new_connections_and_schema_reinitialization(monkeypatch, isolated_postgres):
    import database
    async def factory(): return isolated_postgres
    monkeypatch.setattr(database, "get_pool", factory)
    await database.init_tables()

    async def restart():
        await isolated_postgres.expire_connections()
        await database.init_tables()

    await _assert_persistent_pause(factory, restart)


class _PostgresCteVisibilityConnection:
    """Model PostgreSQL's same-statement visibility for data-modifying CTEs."""

    transaction = _Connection.transaction
    execute = _Connection.execute
    fetch = _Connection.fetch

    async def fetchrow(self, sql, *args):
        if sql.startswith("SELECT summary_namespace"):
            return None
        if "INSERT INTO conversation_summary_state" not in sql:
            raise AssertionError(sql)
        if "RETURNING summary_namespace" not in sql:
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
        build_summary_namespace(actor_id="laoke", room_id="room_weiwei_laoke", conversation_id="conversation-new", execution_mode="private"),
        identity={
            "actor_id": "laoke",
            "room_id": "room_weiwei_laoke",
            "conversation_id": "conversation-new",
            "execution_mode": "private",
        },
    )

    assert state.cache_namespace == build_summary_namespace(actor_id="laoke", room_id="room_weiwei_laoke", conversation_id="conversation-new", execution_mode="private")
    assert state.state_revision == 1


def _summary_identity(**changes):
    return dict(actor_id="jiao", room_id="room_weiwei_jiao", conversation_id="conversation-1",
                execution_mode="private", **changes)


class _MigrationConnection(_Connection):
    def __init__(self, rows, candidates, facts):
        super().__init__(rows)
        self.candidates, self.facts = candidates, facts
        self.adoptions = 0

    async def fetch(self, sql, *args):
        if "FROM model_cache_state" in sql:
            assert args == ("jiao", "conversation-1", "private")
            return sorted(self.candidates, key=lambda c: -c["compressed_up_to_event_id"])
        assert "FROM conversations" in sql
        assert args == ("conversation-1",)
        return self.facts

    async def fetchrow(self, sql, *args):
        if "INSERT INTO conversation_summary_state" in sql:
            self.adoptions += 1
            key, actor, room, conversation, mode, summary, count, cursor, revision = args
            self.rows[key] = dict(cache_namespace=key, summary=summary, summary_token_count=count,
                                  compressed_up_to_event_id=cursor, state_revision=revision)
        return await super().fetchrow(sql, *args)


def _candidate(cursor, summary, revision):
    return dict(compressed_up_to_event_id=cursor, summary=summary, summary_token_count=10,
                state_revision=revision)


@pytest.mark.anyio
async def test_adoption_reuses_most_advanced_valid_summary_once_not_latest_profile():
    pool = _Pool()
    candidates = [_candidate(80, "original paid summary", 21), _candidate(24, "new profile partial", 2)]
    facts = [dict(source_event_id=i, room_id="room_weiwei_jiao", canonical_conversation_id="conversation-1",
                  source_kind="relay_event", bedroom_session_id=None) for i in (24, 80)]
    pool.conn = _MigrationConnection(pool.rows, candidates, facts)
    async def factory(): return pool
    identity = _summary_identity()
    key = build_summary_namespace(**identity)
    state = await PostgresAnchoredHistoryStore(factory).get_or_create(key, identity=identity)
    assert (state.summary, state.state_revision, state.compressed_up_to_event_id) == ("original paid summary", 21, 80)
    candidates[0]["summary"] = "old rollback row must no longer be live"
    restored = await PostgresAnchoredHistoryStore(factory).get_or_create(key, identity=identity)
    assert restored == state
    assert pool.conn.adoptions == 1


@pytest.mark.anyio
@pytest.mark.parametrize("damage", ["room", "source", "cursor", "conflict", "oversize", "missing_facts"])
async def test_ambiguous_or_invalid_adoption_never_creates_empty_restart_state(damage):
    pool = _Pool()
    candidates = [_candidate(80, "paid original", 21)]
    facts = [dict(source_event_id=80, room_id="room_weiwei_jiao", canonical_conversation_id="conversation-1",
                  source_kind="relay_event", bedroom_session_id=None)]
    if damage == "room": facts[0]["room_id"] = "room_weiwei_laoke"
    if damage == "source": facts[0]["source_kind"] = "bedroom_turn"
    if damage == "cursor": candidates[0]["compressed_up_to_event_id"] = 81
    if damage == "conflict": candidates.append(_candidate(80, "different summary at same cursor", 22))
    if damage == "oversize": candidates[0]["summary"] = "x" * 5000
    if damage == "missing_facts": facts = []
    pool.conn = _MigrationConnection(pool.rows, candidates, facts)
    async def factory(): return pool
    identity = _summary_identity()
    with pytest.raises(AnchoredHistoryError, match="adoption requires review"):
        await PostgresAnchoredHistoryStore(factory).get_or_create(build_summary_namespace(**identity), identity=identity)
    assert pool.rows == {}


@pytest.mark.anyio
async def test_real_postgres_summary_adoption_restart_repair_and_bedroom_cleanup(monkeypatch, isolated_postgres):
    import database
    from dataclasses import replace
    from conversation_partitions import ConversationFact, PostgresConversationPartitionStore
    from tests.test_execution_context_builder import _event

    async def factory(): return isolated_postgres
    monkeypatch.setattr(database, "get_pool", factory)
    await database.init_tables()
    facts = PostgresConversationPartitionStore(factory)
    original = tuple(ConversationFact.from_relay_event(_event(i)) for i in range(1, 81))
    await facts.append_accepted_facts(original)
    identity = _summary_identity()
    key = build_summary_namespace(**identity)
    async with isolated_postgres.acquire() as conn:
        await conn.execute("INSERT INTO model_profiles(profile_id,profile_json) VALUES('original','{}'),('new','{}')")
        for profile_id, cursor, summary, revision in [("original",80,"Original paid summary",21),("new",24,"Partial",2)]:
            await conn.execute("""INSERT INTO model_cache_state(cache_namespace,actor_id,conversation_id,profile_id,
                profile_revision,execution_mode,actor_prompt_version,runtime_kernel_version,room_policy_version,
                tool_schema_hash,cache_strategy_version,summary,summary_token_count,compressed_up_to_event_id,state_revision)
                VALUES($1,'jiao','conversation-1',$1,1,'private','v1','v1','v1','v1','v1',$2,10,$3,$4)""",
                profile_id, summary, cursor, revision)
    history = PostgresAnchoredHistoryStore(factory)
    state = await history.get_or_create(key, identity=identity)
    assert (state.summary, state.compressed_up_to_event_id, state.state_revision) == ("Original paid summary",80,21)
    # New store instance is a process restart, and schema initialization is repeatable.
    await database.init_tables()
    restarted = PostgresAnchoredHistoryStore(factory)
    assert await restarted.get_or_create(key, identity=identity) == state
    assert await facts.list_facts("conversation-1") == original
    async with isolated_postgres.acquire() as conn:
        async with conn.transaction():
            await history.invalidate_conversation_state("conversation-1", connection=conn)
    assert (await restarted.get_or_create(key, identity=identity)).summary == ""
    with pytest.raises(AnchoredHistoryError, match="revision conflict"):
        await restarted.apply_compression(key, expected_revision=21, replacement_summary="stale", summary_token_count=2,
                                         compressed_up_to_event_id=81)
    bedroom = dict(identity, conversation_id="bedroom:isolated", execution_mode="bedroom")
    bedkey = build_summary_namespace(**bedroom)
    await history.get_or_create(bedkey, identity=bedroom)
    await history.apply_compression(bedkey, expected_revision=1, replacement_summary="bedroom-only", summary_token_count=5,
                                    compressed_up_to_event_id=1)
    await history.delete_conversation_state("bedroom:isolated")
    async with isolated_postgres.acquire() as conn:
        assert await conn.fetchval("SELECT count(*) FROM conversation_summary_state WHERE conversation_id='bedroom:isolated'") == 0
        assert await conn.fetchval("SELECT count(*) FROM conversation_summary_state WHERE conversation_id='conversation-1'") == 1
        assert await conn.fetchval("SELECT count(*) FROM model_cache_state WHERE summary<>''") == 0
    assert await facts.list_facts("conversation-1") == original
