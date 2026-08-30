import hashlib
import asyncio
from datetime import datetime, timezone
import inspect

import pytest


def test_private_markdown_is_preserved_versioned_and_reversible():
    from actor_prompt_profiles import load_actor_prompt_profiles
    from actor_prompt_store import InMemoryActorPromptVersionStore

    store = InMemoryActorPromptVersionStore(load_actor_prompt_profiles())
    original = "# synthetic actor prompt\r\n\r\nexact text.  \r\n"
    first = asyncio.run(store.create_version("laoke", "laoke-v2.md", original))

    assert first.content_sha256 == hashlib.sha256(original.encode("utf-8")).hexdigest()
    assert asyncio.run(store.export_text("laoke", first.version_id)) == original
    assert asyncio.run(store.get_active("laoke")).prompt_version == "laoke.v1"

    activated = asyncio.run(store.activate("laoke", first.version_id, expected_revision=0))
    assert activated.revision == 1
    assert asyncio.run(store.get_active("laoke")).prompt_text == original

    second = asyncio.run(store.create_version("laoke", "laoke-v3.md", "# synthetic v3\n"))
    asyncio.run(store.activate("laoke", second.version_id, expected_revision=1))
    rolled_back = asyncio.run(store.activate("laoke", first.version_id, expected_revision=2))

    assert rolled_back.revision == 3
    assert asyncio.run(store.get_active("laoke")).prompt_text == original
    assert {item.version_id for item in asyncio.run(store.list_versions("laoke"))} >= {
        "builtin:laoke.v1",
        first.version_id,
        second.version_id,
    }


def test_duplicate_upload_is_idempotent_and_actor_versions_are_isolated():
    from actor_prompt_profiles import load_actor_prompt_profiles
    from actor_prompt_store import InMemoryActorPromptVersionStore

    store = InMemoryActorPromptVersionStore(load_actor_prompt_profiles())
    jiao = asyncio.run(store.create_version("jiao", "jiao.md", "# JIAO v2\n"))
    duplicate = asyncio.run(store.create_version("jiao", "renamed.md", "# JIAO v2\n"))

    assert duplicate.version_id == jiao.version_id
    assert len(asyncio.run(store.list_versions("jiao"))) == 2
    assert all(item.actor_id == "jiao" for item in asyncio.run(store.list_versions("jiao")))
    with pytest.raises(KeyError):
        asyncio.run(store.activate("laoke", jiao.version_id, expected_revision=0))


def test_active_mapping_hot_switches_without_provider_or_identity_fields():
    from actor_prompt_profiles import load_actor_prompt_profiles
    from actor_prompt_store import ActiveActorPromptMapping, InMemoryActorPromptVersionStore

    store = InMemoryActorPromptVersionStore(load_actor_prompt_profiles())
    mapping = ActiveActorPromptMapping(store)
    assert mapping["jiao"].prompt_version == "jiao.v1"

    version = asyncio.run(store.create_version("jiao", "jiao-v2.md", "# JIAO v2\n完整正文"))
    asyncio.run(store.activate("jiao", version.version_id, expected_revision=0))

    active = mapping["jiao"]
    assert active.prompt_text == "# JIAO v2\n完整正文"
    assert active.prompt_version.startswith("jiao.private.")
    assert "provider" not in active.to_dict()
    assert "model" not in active.to_dict()


def test_schema_is_additive_and_keeps_prompt_body_out_of_git_fallback():
    from actor_prompt_store import ACTOR_PROMPT_MIGRATION_SQL, PostgresActorPromptVersionStore

    assert "CREATE TABLE IF NOT EXISTS actor_prompt_versions" in ACTOR_PROMPT_MIGRATION_SQL
    assert "CREATE TABLE IF NOT EXISTS actor_prompt_active" in ACTOR_PROMPT_MIGRATION_SQL
    assert "UNIQUE (actor_id, content_sha256)" in ACTOR_PROMPT_MIGRATION_SQL
    assert "prompt_text TEXT NOT NULL" in ACTOR_PROMPT_MIGRATION_SQL
    assert "pg_advisory_xact_lock" in inspect.getsource(
        PostgresActorPromptVersionStore.activate
    )


def test_postgres_active_revision_refreshes_across_gateway_workers():
    from actor_prompt_profiles import load_actor_prompt_profiles
    from actor_prompt_store import PostgresActorPromptVersionStore

    now = datetime.now(timezone.utc)
    state = {"versions": [], "active": []}

    class Connection:
        async def fetch(self, query, *args):
            if "LEFT JOIN actor_prompt_versions" in query:
                joined = []
                for active in state["active"]:
                    version = next(
                        (row for row in state["versions"] if row["version_id"] == active["version_id"]),
                        {},
                    )
                    joined.append({**version, **active})
                return joined
            if "FROM actor_prompt_versions WHERE actor_id" in query:
                return [row for row in state["versions"] if row["actor_id"] == args[0]]
            if "FROM actor_prompt_versions ORDER BY" in query:
                return list(state["versions"])
            if "FROM actor_prompt_active" in query:
                return list(state["active"])
            raise AssertionError(query)

    class Acquire:
        async def __aenter__(self):
            return Connection()

        async def __aexit__(self, *_args):
            return False

    class Pool:
        def acquire(self):
            return Acquire()

    async def pool_factory():
        return Pool()

    store = PostgresActorPromptVersionStore(pool_factory, load_actor_prompt_profiles())
    asyncio.run(store.initialize())
    assert store.get_active_cached("laoke").prompt_version == "laoke.v1"

    body = "# activated by another worker\n"
    digest = hashlib.sha256(body.encode("utf-8")).hexdigest()
    version_id = f"private:laoke:{digest}"
    state["versions"].append(
        {
            "version_id": version_id,
            "actor_id": "laoke",
            "prompt_version": f"laoke.private.{digest[:16]}",
            "prompt_text": body,
            "content_sha256": digest,
            "source_filename": "laoke.md",
            "created_at": now,
        }
    )
    state["active"].append(
        {
            "actor_id": "laoke",
            "version_id": version_id,
            "revision": 1,
            "activated_at": now,
        }
    )

    asyncio.run(store.refresh_active())
    assert store.get_active_cached("laoke").prompt_text == body


def test_every_context_execution_path_refreshes_active_persona_revision():
    import main

    assert "await _refresh_actor_prompt_store()" in inspect.getsource(
        main._stream_gateway_execution
    )
    assert "await _refresh_actor_prompt_store()" in inspect.getsource(
        main._build_group_context_pack
    )
    assert "await _refresh_actor_prompt_store()" in inspect.getsource(
        main.bedroom_context_pack_full
    )
