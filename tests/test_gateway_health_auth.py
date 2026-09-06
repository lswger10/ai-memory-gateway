"""HTTP regressions for fail-closed management and truthful readiness (F01/F10)."""

from unittest.mock import AsyncMock

import pytest
from fastapi.testclient import TestClient

import main


@pytest.mark.parametrize("method,path", [
    ("GET", "/api/memory-settings"),
    ("DELETE", "/api/memories/7"),
    ("GET", "/dashboard"),
])
def test_missing_management_credentials_never_reach_data_handlers(monkeypatch, method, path):
    monkeypatch.setattr(main, "GATEWAY_SECRET", "")
    monkeypatch.setattr(main, "ACTOR_PERSONA_PROXY_SECRET", "")
    monkeypatch.setattr(main, "MEMORY_ENABLED", True)
    writes = []

    async def delete(memory_id):
        writes.append(memory_id)

    monkeypatch.setattr(main, "delete_memory", delete)
    monkeypatch.setattr(main, "get_all_gateway_config", AsyncMock(return_value={}))
    response = TestClient(main.app).request(method, path)
    assert response.status_code == 503
    assert writes == []


def test_admin_key_still_grants_settings_but_wrong_key_does_not(monkeypatch):
    monkeypatch.setattr(main, "GATEWAY_SECRET", "test-admin")
    monkeypatch.setattr(main, "get_all_gateway_config", AsyncMock(return_value={}))
    client = TestClient(main.app)
    assert client.get("/api/memory-settings").status_code == 401
    assert client.get("/api/memory-settings", headers={"X-Gateway-Key": "wrong"}).status_code == 401
    assert client.get("/api/memory-settings", headers={"X-Gateway-Key": "test-admin"}).status_code == 200


def test_database_query_failure_is_not_an_empty_database_and_recovers(monkeypatch):
    monkeypatch.setattr(main, "MEMORY_ENABLED", True)
    monkeypatch.setattr(main.app.state, "database_initialized", True, raising=False)
    count = AsyncMock(side_effect=[ConnectionError("private connection detail"), 0, 3])
    monkeypatch.setattr(main, "get_all_memories_count", count)
    client = TestClient(main.app)
    failed = client.get("/")
    assert failed.status_code == 503
    assert failed.json()["ready"] is False
    assert failed.json()["memory_count"] is None
    assert "private connection detail" not in failed.text
    assert client.get("/health").status_code == 200
    empty = client.get("/")
    assert empty.status_code == 200
    assert empty.json()["ready"] is True
    assert empty.json()["memory_count"] == 0
    assert client.get("/").json()["memory_count"] == 3


@pytest.mark.parametrize("failure", ["init_tables", "_ensure_actor_prompt_store", "ensure_token_usage_table"])
def test_failed_initialization_is_live_but_not_ready(monkeypatch, failure):
    monkeypatch.setattr(main, "MEMORY_ENABLED", True)
    for step in ("init_tables", "_ensure_actor_prompt_store", "ensure_token_usage_table"):
        monkeypatch.setattr(main, step, AsyncMock(
            side_effect=ConnectionError("test initialization failure") if step == failure else None))
    monkeypatch.setattr(main, "get_all_memories_count", AsyncMock(return_value=0))
    monkeypatch.setattr(main, "close_pool", AsyncMock())
    monkeypatch.setattr(main, "_model_provider_runner", None)
    monkeypatch.setattr(main, "resolve_feature_flags", lambda: {"model_execution": True})
    monkeypatch.setattr(main, "group_memory_features_from_env", lambda: {"group_memory": True, "burst_extraction": True})
    started = []

    async def worker():
        started.append(True)

    monkeypatch.setattr(main, "_group_extraction_worker", worker)
    monkeypatch.setattr(main, "_conversation_cache_pin_worker", worker)
    with TestClient(main.app) as client:
        assert client.get("/health").status_code == 200
        response = client.get("/")
        assert response.status_code == 503
        assert response.json()["ready"] is False
        assert response.json()["memory_count"] is None
    assert started == []


def test_successful_initialization_marks_an_empty_database_ready(monkeypatch):
    monkeypatch.setattr(main, "MEMORY_ENABLED", True)
    for step in ("init_tables", "_ensure_actor_prompt_store", "ensure_token_usage_table", "close_pool"):
        monkeypatch.setattr(main, step, AsyncMock())
    monkeypatch.setattr(main, "get_all_gateway_config", AsyncMock(return_value={}))
    monkeypatch.setattr(main, "get_all_memories_count", AsyncMock(return_value=0))
    monkeypatch.setattr(main, "_model_provider_runner", None)
    monkeypatch.setattr(main, "resolve_feature_flags", lambda: {"model_execution": False})
    monkeypatch.setattr(main, "group_memory_features_from_env", lambda: {"group_memory": False})
    with TestClient(main.app) as client:
        response = client.get("/")
        assert response.status_code == 200
        assert response.json()["ready"] is True
        assert response.json()["memory_count"] == 0


def test_disabled_persistence_is_not_database_readiness(monkeypatch):
    monkeypatch.setattr(main, "MEMORY_ENABLED", False)
    response = TestClient(main.app).get("/")
    assert response.status_code == 503
    assert response.json()["ready"] is False
    assert response.json()["memory_count"] is None


@pytest.mark.anyio
async def test_real_postgres_lifespan_and_readiness_recovery(monkeypatch, isolated_postgres):
    import httpx

    async def pool_factory():
        return isolated_postgres

    monkeypatch.setattr(main, "MEMORY_ENABLED", True)
    monkeypatch.setattr(main._db_module, "DATABASE_URL", "postgresql://isolated-test")
    monkeypatch.setattr(main._db_module, "get_pool", pool_factory)
    monkeypatch.setattr(main, "get_pool", pool_factory)
    monkeypatch.setattr(main, "close_pool", AsyncMock())  # Fixture owns the pool.
    monkeypatch.setattr(main, "_actor_prompt_store", None)
    monkeypatch.setattr(main, "_actor_prompt_mapping", None)
    monkeypatch.setattr(main, "_model_provider_runner", None)
    monkeypatch.setattr(main, "resolve_feature_flags", lambda: {"model_execution": False})
    monkeypatch.setattr(main, "group_memory_features_from_env", lambda: {"group_memory": False})
    async with main.lifespan(main.app):
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=main.app), base_url="http://test") as client:
            empty = await client.get("/")
            assert empty.status_code == 200
            assert empty.json()["memory_count"] == 0
            async with isolated_postgres.acquire() as conn:
                await conn.execute("ALTER TABLE memories RENAME TO temporarily_unavailable_memories")
            failed = await client.get("/")
            assert failed.status_code == 503
            assert failed.json()["memory_count"] is None
            assert (await client.get("/health")).status_code == 200
            async with isolated_postgres.acquire() as conn:
                await conn.execute("ALTER TABLE temporarily_unavailable_memories RENAME TO memories")
            recovered = await client.get("/")
            assert recovered.status_code == 200
            assert recovered.json()["ready"] is True
            assert recovered.json()["memory_count"] == 0
