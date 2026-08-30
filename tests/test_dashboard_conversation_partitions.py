import pytest


class _Acquire:
    def __init__(self, connection):
        self.connection = connection
    async def __aenter__(self):
        return self.connection
    async def __aexit__(self, *args):
        return False


class _Pool:
    def __init__(self, rows):
        self.connection = _Connection(rows)
    def acquire(self):
        return _Acquire(self.connection)


class _Connection:
    def __init__(self, rows):
        self.rows = rows
    async def fetchval(self, sql, *args):
        return len(self.rows)
    async def fetch(self, sql, *args):
        assert "source_event_id" in sql
        return self.rows


@pytest.mark.anyio
async def test_dashboard_conversations_projects_complete_cognitive_fact_coordinates(monkeypatch):
    import main

    rows = [
        {
            "id": 2, "role": "assistant", "content": "second",
            "created_at": None, "room_id": "room_group_home",
            "canonical_conversation_id": "group-1", "source_event_id": 2,
            "actor_id": "jiao", "event_role": "agent",
            "source_kind": "relay_event", "bedroom_session_id": None,
            "retention_policy": None, "attachments_json": [],
            "provenance_json": {"generation_request_id": "gen-2"},
        },
        {
            "id": 1, "role": "user", "content": "first",
            "created_at": None, "room_id": "room_group_home",
            "canonical_conversation_id": "group-1", "source_event_id": 1,
            "actor_id": "weiwei", "event_role": "human",
            "source_kind": "relay_event", "bedroom_session_id": None,
            "retention_policy": None, "attachments_json": [],
            "provenance_json": None,
        },
    ]

    async def get_pool():
        return _Pool(rows)

    monkeypatch.setattr(main, "MEMORY_ENABLED", True)
    monkeypatch.setattr(main, "get_pool", get_pool)
    payload = await main.api_conversation_messages("group-1", limit=50, offset=0)
    assert payload["total"] == 2
    assert [row["source_event_id"] for row in payload["messages"]] == [2, 1]
    assert payload["messages"][0]["actor_id"] == "jiao"
    assert payload["messages"][0]["conversation_id"] == "group-1"
    assert payload["messages"][0]["provenance"]["generation_request_id"] == "gen-2"
