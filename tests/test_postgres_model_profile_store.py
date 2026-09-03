import json

import pytest

from model_profiles import ModelProfile
from postgres_model_stores import PostgresModelProfileStore


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
