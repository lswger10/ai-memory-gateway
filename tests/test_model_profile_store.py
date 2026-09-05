import pytest
from dataclasses import replace

from database import MODEL_EXECUTION_MIGRATION_SQL, apply_model_execution_schema
from model_profile_store import InMemoryModelProfileStore, ProfileStoreError
from model_profiles import ModelProfile


def _profile(profile_id: str, *, status: str = "passed") -> ModelProfile:
    return ModelProfile.from_dict(
        {
            "profile_id": profile_id,
            "display_name": profile_id,
            "enabled": True,
            "test_status": status,
            "provider": "test-provider",
            "protocol": "openai_chat_completions",
            "base_url": "https://example.invalid/v1",
            "route_id": f"route-{profile_id}",
            "model": f"model-{profile_id}",
            "adapter_version": "test.v1",
            "credential_ref": f"env:{profile_id.upper()}_KEY",
            "capabilities": {
                "streaming": True,
                "structured_output": False,
                "tools": False,
                "reasoning_controls": False,
                "cache_strategies": ["no_prompt_cache_v1"],
                "cache_ttls": [],
                "usage_fields": [],
            },
            "cache_strategy": "no_prompt_cache_v1",
            "requested_cache_ttl": None,
            "revision": 1,
        }
    )


class _ConnectionRecorder:
    def __init__(self):
        self.calls = []

    async def execute(self, sql):
        self.calls.append(sql)


@pytest.mark.anyio
async def test_profile_schema_is_additive_and_idempotent():
    conn = _ConnectionRecorder()
    await apply_model_execution_schema(conn)
    await apply_model_execution_schema(conn)

    assert conn.calls == [MODEL_EXECUTION_MIGRATION_SQL, MODEL_EXECUTION_MIGRATION_SQL]
    assert "CREATE TABLE IF NOT EXISTS model_profiles" in MODEL_EXECUTION_MIGRATION_SQL
    assert "CREATE TABLE IF NOT EXISTS model_execution_receipts" in MODEL_EXECUTION_MIGRATION_SQL
    assert "ADD COLUMN IF NOT EXISTS stable_prefix_hash" in MODEL_EXECUTION_MIGRATION_SQL
    assert "ADD COLUMN IF NOT EXISTS provider_usage_received" in MODEL_EXECUTION_MIGRATION_SQL
    assert "DROP TABLE" not in MODEL_EXECUTION_MIGRATION_SQL.upper()
    assert "UPDATE memories" not in MODEL_EXECUTION_MIGRATION_SQL


@pytest.mark.anyio
async def test_actor_default_and_room_override_are_revisioned():
    store = InMemoryModelProfileStore()
    await store.put_profile(_profile("profile-a"))
    await store.put_profile(_profile("profile-b"))

    default = await store.set_actor_default("jiao", "profile-a")
    override = await store.set_room_override(
        "room_group_home", "jiao", "profile-b", expected_revision=None
    )

    assert default.revision == 1
    assert override.revision == 1
    replaced = await store.set_room_override(
        "room_group_home", "jiao", "profile-a", expected_revision=1
    )
    assert replaced.revision == 2
    with pytest.raises(ProfileStoreError, match="revision"):
        await store.set_room_override(
            "room_group_home", "jiao", "profile-b", expected_revision=1
        )


@pytest.mark.anyio
async def test_resolver_prefers_room_override_then_actor_default():
    store = InMemoryModelProfileStore()
    await store.put_profile(_profile("default"))
    await store.put_profile(_profile("room-choice"))
    await store.set_actor_default("jiao", "default")
    await store.set_room_override("room_group_home", "jiao", "room-choice")

    group = await store.resolve("jiao", "room_group_home")
    private = await store.resolve("jiao", "room_weiwei_jiao")

    assert group.primary.profile_id == "room-choice"
    assert group.source == "room_override"
    assert private.primary.profile_id == "default"
    assert private.source == "actor_default"


@pytest.mark.anyio
async def test_resolver_returns_only_approved_fallback_order():
    store = InMemoryModelProfileStore()
    for profile_id in ("default", "approved-2", "approved-1", "not-approved"):
        await store.put_profile(_profile(profile_id))
    await store.set_actor_default("jiao", "default")
    await store.set_approved_fallbacks(
        "jiao", ("approved-2", "approved-1")
    )

    resolved = await store.resolve("jiao", "room_group_home")

    assert [item.profile_id for item in resolved.fallbacks] == [
        "approved-2",
        "approved-1",
    ]
    assert "not-approved" not in [item.profile_id for item in resolved.fallbacks]


@pytest.mark.anyio
async def test_cross_family_profile_binding_preserves_actor_id():
    store = InMemoryModelProfileStore()
    await store.put_profile(_profile("gemini-route"))
    await store.set_actor_default("jiao", "gemini-route")
    await store.set_actor_default("laoke", "gemini-route")

    assert (await store.resolve("jiao", "room_group_home")).actor_id == "jiao"
    assert (await store.resolve("laoke", "room_group_home")).actor_id == "laoke"


@pytest.mark.anyio
async def test_unverified_fallback_is_rejected_before_resolution():
    store = InMemoryModelProfileStore()
    await store.put_profile(_profile("default"))
    await store.put_profile(_profile("unverified", status="unverified"))
    await store.set_actor_default("jiao", "default")
    with pytest.raises(ProfileStoreError, match="selectable"):
        await store.set_approved_fallbacks("jiao", ("unverified",))


@pytest.mark.anyio
async def test_changed_profile_requires_next_revision_and_invalidates_certification():
    store = InMemoryModelProfileStore()
    original = _profile("default")
    await store.put_profile(original)
    await store.record_probe_result(profile_id="default", profile_revision=1,
        probe_kind="frozen_double_send_cache", status="verified", observed_capabilities={})
    with pytest.raises(ProfileStoreError, match="revision"):
        await store.put_profile(replace(original, model="changed"))
    assert await store.get_profile("default") == original
    updated = await store.put_profile(replace(original, model="changed", revision=2))
    assert updated.test_status == "unverified"
    assert not await store.has_verified_probe("default", 2, "frozen_double_send_cache")


@pytest.mark.anyio
async def test_probe_status_cannot_certify_a_different_revision():
    store = InMemoryModelProfileStore()
    await store.put_profile(_profile("default", status="unverified"))
    await store.put_profile(replace(_profile("default", status="unverified"), revision=2))
    with pytest.raises(ProfileStoreError, match="revision"):
        await store.set_test_status("default", "passed", expected_revision=1)
    assert (await store.get_profile("default")).test_status == "unverified"


@pytest.mark.anyio
async def test_save_binding_is_atomic_and_rejects_stale_revision():
    store = InMemoryModelProfileStore()
    for name in ("first", "next", "backup"):
        await store.put_profile(_profile(name))
    original = await store.set_actor_default("jiao", "first")
    with pytest.raises(ProfileStoreError):
        await store.save_actor_binding("jiao", "next", ("missing",), expected_revision=original.revision)
    assert (await store.resolve("jiao", "room_weiwei_jiao")).primary.profile_id == "first"
    saved = await store.save_actor_binding("jiao", "next", ("backup",), expected_revision=original.revision)
    assert saved.revision == original.revision + 1
    with pytest.raises(ProfileStoreError, match="revision"):
        await store.save_actor_binding("jiao", "first", (), expected_revision=original.revision)
    resolved = await store.resolve("jiao", "room_weiwei_jiao")
    assert resolved.primary.profile_id == "next"
    assert [p.profile_id for p in resolved.fallbacks] == ["backup"]
