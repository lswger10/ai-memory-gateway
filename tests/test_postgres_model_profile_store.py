import json
import asyncio
import os
import uuid
from dataclasses import replace

import pytest

from model_profiles import ModelProfile
from postgres_model_stores import PostgresModelProfileStore
from model_profile_store import ProfileStoreError


def _profile_payload() -> dict:
    return {
        "profile_id": "ofox-claude-cache-5m",
        "display_name": "OFOX Claude cache",
        "enabled": True,
        "test_status": "unverified",
        "provider": "ofox",
        "protocol": "anthropic_messages_compatible",
        "base_url": "https://api.ofox.ai/anthropic",
        "route_id": "ofox-claude-anthropic-cache-5m",
        "model": "anthropic/claude-opus-4.6",
        "adapter_version": "gateway-anthropic-v1",
        "credential_ref": "env:LAOKE_OFOX_API_KEY",
        "headers": {
            "anthropic-version": "2023-06-01",
            "x-api-key": "${credential}",
        },
        "capabilities": {
            "streaming": True,
            "structured_output": False,
            "tools": False,
            "reasoning_controls": False,
            "cache_strategies": ["anthropic_prefix_anchored_v1"],
            "cache_ttls": ["5m"],
            "usage_fields": [
                "input_tokens",
                "output_tokens",
                "cache_creation_input_tokens",
                "cache_read_input_tokens",
            ],
        },
        "cache_strategy": "anthropic_prefix_anchored_v1",
        "requested_cache_ttl": "5m",
        "revision": 1,
    }


class _Connection:
    def __init__(self, profile_json):
        self.profile_json = profile_json

    async def fetchrow(self, sql, *args):
        return {"profile_json": self.profile_json}

    async def fetch(self, sql, *args):
        return ({"profile_json": self.profile_json},)


class _Acquire:
    def __init__(self, connection):
        self.connection = connection

    async def __aenter__(self):
        return self.connection

    async def __aexit__(self, *args):
        return False


class _Pool:
    def __init__(self, profile_json):
        self.connection = _Connection(profile_json)

    def acquire(self):
        return _Acquire(self.connection)


async def _pool(value):
    return value


@pytest.mark.anyio
@pytest.mark.parametrize("database_value", [json.dumps(_profile_payload()), _profile_payload()])
async def test_postgres_profile_store_decodes_asyncpg_jsonb_values(database_value):
    pool = _Pool(database_value)
    store = PostgresModelProfileStore(lambda: _pool(pool))

    profile = await store.get_profile("ofox-claude-cache-5m")
    profiles = await store.list_profiles()

    assert isinstance(profile, ModelProfile)
    assert profile.profile_id == "ofox-claude-cache-5m"
    assert profiles == (profile,)


class _ResolveConnection:
    def __init__(self):
        primary = _profile_payload()
        primary["test_status"] = "passed"
        fallback = dict(primary)
        fallback.update(
            profile_id="ofox-claude-no-cache",
            display_name="OFOX Claude no cache",
            route_id="ofox-claude-no-cache",
            cache_strategy="no_prompt_cache_v1",
            requested_cache_ttl=None,
            capabilities={**primary["capabilities"], "cache_strategies": ["no_prompt_cache_v1"], "cache_ttls": []},
        )
        self.profiles = {
            primary["profile_id"]: json.dumps(primary),
            fallback["profile_id"]: json.dumps(fallback),
        }

    async def fetchrow(self, sql, *args):
        if "FROM model_actor_bindings" in sql:
            return {
                "actor_id": "laoke",
                "default_profile_id": "ofox-claude-cache-5m",
                "approved_fallback_profile_ids": json.dumps(["ofox-claude-no-cache"]),
                "revision": 2,
            }
        if "FROM model_room_overrides" in sql:
            return None
        if "SELECT profile_json FROM model_profiles" in sql:
            return {"profile_json": self.profiles[args[0]]}
        raise AssertionError(sql)


class _ResolvePool:
    def __init__(self):
        self.connection = _ResolveConnection()

    def acquire(self):
        return _Acquire(self.connection)


@pytest.mark.anyio
async def test_postgres_profile_resolution_decodes_asyncpg_jsonb_fallback_ids():
    pool = _ResolvePool()
    store = PostgresModelProfileStore(lambda: _pool(pool))

    resolved = await store.resolve("laoke", "room_weiwei_laoke")

    assert resolved.primary.profile_id == "ofox-claude-cache-5m"
    assert [profile.profile_id for profile in resolved.fallbacks] == [
        "ofox-claude-no-cache"
    ]


class _ProbeConnection:
    async def fetchrow(self, sql, *args):
        assert "status='verified'" in sql
        assert args == ("ofox-claude-cache-5m", 1, "frozen_double_send_cache")
        return {"?column?": 1}


@pytest.mark.anyio
async def test_postgres_profile_store_requires_verified_probe_for_cache_capability():
    pool = _Pool(None)
    pool.connection = _ProbeConnection()
    store = PostgresModelProfileStore(lambda: _pool(pool))

    assert await store.has_verified_probe(
        "ofox-claude-cache-5m", 1, "frozen_double_send_cache"
    )


@pytest.mark.anyio
async def test_real_postgres_model_settings_atomicity_and_restart():
    """Opt-in test DSN only; all writes are confined to a disposable schema."""
    dsn = os.environ.get("GATEWAY_MODEL_SETTINGS_TEST_DSN")
    if not dsn or os.environ.get("GATEWAY_TEST_POSTGRES_APPROVED") != "true":
        pytest.skip("BLOCKED: explicit isolated test PostgreSQL authorization/DSN required")
    import asyncpg
    from database import apply_model_execution_schema
    schema = "group_e2e_" + uuid.uuid4().hex
    admin = await asyncpg.connect(dsn)
    pool = None
    try:
        await admin.execute(f'CREATE SCHEMA "{schema}"')
        pool = await asyncpg.create_pool(dsn, min_size=1, max_size=3, server_settings={"search_path": schema})
        async with pool.acquire() as conn:
            assert await conn.fetchval("SELECT current_schema()") == schema
            await apply_model_execution_schema(conn)
        store = PostgresModelProfileStore(lambda: _pool(pool))
        profiles = [ModelProfile.from_dict({**_profile_payload(), "profile_id": name, "test_status": "passed"}) for name in ("first", "next", "backup")]
        for p in profiles:
            await store.put_profile(p)
        original = await store.save_actor_binding("jiao", "first", ("backup",), expected_revision=0)
        with pytest.raises(ProfileStoreError):
            await store.save_actor_binding("jiao", "next", ("missing",), expected_revision=original.revision)
        assert await store.get_actor_binding("jiao") == original
        saves = await asyncio.gather(
            store.save_actor_binding("jiao", "next", ("backup",), expected_revision=1),
            store.save_actor_binding("jiao", "backup", ("next",), expected_revision=1), return_exceptions=True)
        assert sum(isinstance(s, ProfileStoreError) for s in saves) == 1
        edits = await asyncio.gather(
            store.put_profile(replace(profiles[0], model="edited-a", revision=2)),
            store.put_profile(replace(profiles[0], model="edited-b", revision=2)), return_exceptions=True)
        assert sum(isinstance(s, ProfileStoreError) for s in edits) == 1
        with pytest.raises(ProfileStoreError, match="revision"):
            await store.set_test_status("first", "passed", expected_revision=1)
        assert (await store.get_profile("first")).test_status == "unverified"
        persisted = await store.get_actor_binding("jiao")
        await pool.close()
        pool = await asyncpg.create_pool(dsn, min_size=1, max_size=2, server_settings={"search_path": schema})
        restarted = PostgresModelProfileStore(lambda: _pool(pool))
        assert await restarted.get_actor_binding("jiao") == persisted
        assert (await restarted.get_profile("first")).revision == 2
    finally:
        if pool is not None:
            await pool.close()
        await admin.execute(f'DROP SCHEMA "{schema}" CASCADE')
        assert not await admin.fetchval("SELECT EXISTS(SELECT 1 FROM pg_namespace WHERE nspname=$1)", schema)
        await admin.close()
