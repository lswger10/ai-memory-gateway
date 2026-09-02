from datetime import datetime

import pytest


class RecordingConnection:
    def __init__(self):
        self.statements = []

    async def execute(self, sql, *args):
        self.statements.append((sql, args))


@pytest.mark.anyio
async def test_conversation_partition_schema_is_additive_and_idempotent():
    from database import apply_conversation_partition_schema

    conn = RecordingConnection()
    await apply_conversation_partition_schema(conn)
    await apply_conversation_partition_schema(conn)

    sql = "\n".join(statement for statement, _ in conn.statements)
    assert "ALTER TABLE conversations" in sql
    assert "ADD COLUMN IF NOT EXISTS fact_identity" in sql
    assert "ADD COLUMN IF NOT EXISTS request_id" in sql
    assert "ADD COLUMN IF NOT EXISTS message_kind" in sql
    assert "CREATE UNIQUE INDEX IF NOT EXISTS idx_conversations_fact_identity" in sql
    assert "DROP TABLE" not in sql
    assert "DELETE FROM conversations" not in sql


class Transaction:
    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        return False


class PartitionConnection:
    def __init__(self):
        self.rows = {}

    def transaction(self):
        return Transaction()

    async def fetchrow(self, sql, *args):
        if "INSERT INTO conversations" in sql:
            identity = args[6]
            self.rows.setdefault(identity, args)
            return {"fact_hash": self.rows[identity][7]}
        raise AssertionError(sql)

    async def fetch(self, sql, *args):
        if "FROM conversations" not in sql:
            raise AssertionError(sql)
        partition_id, after_event_id, through_event_id = args
        values = [row for row in self.rows.values() if row[0] == partition_id]
        values = [
            row for row in values
            if row[3] > after_event_id and (through_event_id is None or row[3] <= through_event_id)
        ]
        return [
            {
                "session_id": row[0], "room_id": row[1],
                "canonical_conversation_id": row[2], "source_event_id": row[3],
                "actor_id": row[4], "event_role": row[5],
                "fact_identity": row[6], "fact_hash": row[7],
                "content": row[8], "request_id": row[9], "created_at": row[10],
                "source_kind": row[11], "provenance_json": row[12],
                "attachments_json": row[13], "message_kind": row[14],
                "bedroom_session_id": row[15],
                "retention_policy": row[16],
            }
            for row in sorted(values, key=lambda item: item[3])
        ]

    async def fetchval(self, sql, *args):
        if "MAX(source_event_id)" in sql:
            values = [row[3] for row in self.rows.values() if row[0] == args[0]]
            return max(values, default=0)
        raise AssertionError(sql)

    async def execute(self, sql, *args):
        if "DELETE FROM conversations" in sql:
            target = f"bedroom:{args[0]}"
            self.rows = {key: row for key, row in self.rows.items() if row[0] != target}
            return "DELETE"
        raise AssertionError(sql)


class Acquire:
    def __init__(self, connection):
        self.connection = connection

    async def __aenter__(self):
        return self.connection

    async def __aexit__(self, *args):
        return False


class Pool:
    def __init__(self):
        self.connection = PartitionConnection()

    def acquire(self):
        return Acquire(self.connection)


@pytest.mark.anyio
async def test_postgres_partition_survives_store_recreation_and_deduplicates():
    from conversation_partitions import ConversationFact, PostgresConversationPartitionStore

    pool = Pool()
    event = {
        "event_id": 11, "room_id": "room_group_home", "conversation_id": "group-1",
        "actor_id": "weiwei", "role": "human", "content": "hello",
        "request_id": "request-11", "created_at": "2026-08-30T00:00:11Z",
        "visibility": "public", "provenance": None,
    }
    fact = ConversationFact.from_relay_event(event)
    first = PostgresConversationPartitionStore(lambda: _pool(pool))
    await first.append_accepted_facts((fact, fact))
    stored_args = pool.connection.rows[fact.fact_identity]
    assert isinstance(stored_args[10], datetime)

    recreated = PostgresConversationPartitionStore(lambda: _pool(pool))
    restored = await recreated.list_facts("group-1")
    assert [item.source_event_id for item in restored] == [11]
    assert await recreated.latest_event_id("group-1") == 11


async def _pool(pool):
    return pool
