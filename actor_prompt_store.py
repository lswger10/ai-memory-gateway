"""Private, versioned actor Persona storage with immutable Markdown bodies."""

from __future__ import annotations

import asyncio
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
from typing import Callable

from actor_prompt_profiles import ActorPromptProfile


ACTORS = ("jiao", "laoke")

ACTOR_PROMPT_MIGRATION_SQL = """
CREATE TABLE IF NOT EXISTS actor_prompt_versions (
    version_id TEXT PRIMARY KEY,
    actor_id TEXT NOT NULL CHECK (actor_id IN ('jiao','laoke')),
    prompt_version TEXT NOT NULL,
    prompt_text TEXT NOT NULL,
    content_sha256 TEXT NOT NULL,
    source_filename TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE (actor_id, content_sha256)
);

CREATE INDEX IF NOT EXISTS idx_actor_prompt_versions_actor_created
    ON actor_prompt_versions(actor_id, created_at DESC);

CREATE TABLE IF NOT EXISTS actor_prompt_active (
    actor_id TEXT PRIMARY KEY CHECK (actor_id IN ('jiao','laoke')),
    version_id TEXT NOT NULL,
    revision INTEGER NOT NULL DEFAULT 1 CHECK (revision > 0),
    activated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
"""


class ActorPromptStoreError(ValueError):
    pass


class ActorPromptRevisionConflict(ActorPromptStoreError):
    pass


@dataclass(frozen=True)
class ActorPromptVersion:
    actor_id: str
    version_id: str
    prompt_version: str
    prompt_text: str
    content_sha256: str
    source_filename: str
    created_at: datetime | None
    source: str

    def to_profile(self) -> ActorPromptProfile:
        return ActorPromptProfile(
            actor_id=self.actor_id,
            prompt_version=self.prompt_version,
            prompt_text=self.prompt_text,
        )

    def to_public_dict(self, *, active: bool = False) -> dict:
        return {
            "actor_id": self.actor_id,
            "version_id": self.version_id,
            "prompt_version": self.prompt_version,
            "content_sha256": self.content_sha256,
            "content_bytes": len(self.prompt_text.encode("utf-8")),
            "source_filename": self.source_filename,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "source": self.source,
            "active": active,
        }


@dataclass(frozen=True)
class ActorPromptActiveState:
    actor_id: str
    version_id: str
    revision: int
    activated_at: datetime | None

    def to_public_dict(self) -> dict:
        return {
            "actor_id": self.actor_id,
            "version_id": self.version_id,
            "revision": self.revision,
            "activated_at": self.activated_at.isoformat() if self.activated_at else None,
        }


def _validate_actor(actor_id: str) -> None:
    if actor_id not in ACTORS:
        raise KeyError(actor_id)


def _builtin_version(profile: ActorPromptProfile) -> ActorPromptVersion:
    digest = hashlib.sha256(profile.prompt_text.encode("utf-8")).hexdigest()
    return ActorPromptVersion(
        actor_id=profile.actor_id,
        version_id=f"builtin:{profile.prompt_version}",
        prompt_version=profile.prompt_version,
        prompt_text=profile.prompt_text,
        content_sha256=digest,
        source_filename=f"{profile.prompt_version}.md",
        created_at=None,
        source="builtin_fallback",
    )


def _private_version(actor_id: str, source_filename: str, prompt_text: str) -> ActorPromptVersion:
    _validate_actor(actor_id)
    digest = hashlib.sha256(prompt_text.encode("utf-8")).hexdigest()
    return ActorPromptVersion(
        actor_id=actor_id,
        version_id=f"private:{actor_id}:{digest}",
        prompt_version=f"{actor_id}.private.{digest[:16]}",
        prompt_text=prompt_text,
        content_sha256=digest,
        source_filename=source_filename,
        created_at=datetime.now(timezone.utc),
        source="private_upload",
    )


class InMemoryActorPromptVersionStore:
    def __init__(self, builtins: Mapping[str, ActorPromptProfile]) -> None:
        if set(builtins) != set(ACTORS):
            raise ActorPromptStoreError("actor prompt builtins are incomplete")
        self._versions = {
            actor_id: {_builtin_version(profile).version_id: _builtin_version(profile)}
            for actor_id, profile in builtins.items()
        }
        self._active = {
            actor_id: ActorPromptActiveState(
                actor_id, f"builtin:{profile.prompt_version}", 0, None
            )
            for actor_id, profile in builtins.items()
        }

    async def initialize(self) -> None:
        return None

    async def refresh_active(self) -> None:
        return None

    def get_active_cached(self, actor_id: str) -> ActorPromptProfile:
        _validate_actor(actor_id)
        state = self._active[actor_id]
        return self._versions[actor_id][state.version_id].to_profile()

    async def get_active(self, actor_id: str) -> ActorPromptProfile:
        return self.get_active_cached(actor_id)

    async def get_active_state(self, actor_id: str) -> ActorPromptActiveState:
        _validate_actor(actor_id)
        return self._active[actor_id]

    async def list_versions(self, actor_id: str) -> tuple[ActorPromptVersion, ...]:
        _validate_actor(actor_id)
        rows = self._versions[actor_id].values()
        return tuple(sorted(rows, key=lambda row: (row.created_at is not None, row.created_at or datetime.min.replace(tzinfo=timezone.utc)), reverse=True))

    async def create_version(
        self, actor_id: str, source_filename: str, prompt_text: str
    ) -> ActorPromptVersion:
        version = _private_version(actor_id, source_filename, prompt_text)
        existing = self._versions[actor_id].get(version.version_id)
        if existing is not None:
            return existing
        self._versions[actor_id][version.version_id] = version
        return version

    async def activate(
        self, actor_id: str, version_id: str, *, expected_revision: int
    ) -> ActorPromptActiveState:
        _validate_actor(actor_id)
        if version_id not in self._versions[actor_id]:
            raise KeyError(version_id)
        current = self._active[actor_id]
        if current.version_id == version_id:
            return current
        if expected_revision != current.revision:
            raise ActorPromptRevisionConflict("actor prompt revision changed")
        state = ActorPromptActiveState(
            actor_id, version_id, current.revision + 1, datetime.now(timezone.utc)
        )
        self._active[actor_id] = state
        return state

    async def export_text(self, actor_id: str, version_id: str) -> str:
        _validate_actor(actor_id)
        try:
            return self._versions[actor_id][version_id].prompt_text
        except KeyError as exc:
            raise KeyError(version_id) from exc


class PostgresActorPromptVersionStore(InMemoryActorPromptVersionStore):
    """PostgreSQL persistence plus a hot in-process active-profile cache."""

    def __init__(self, pool_factory: Callable, builtins: Mapping[str, ActorPromptProfile]) -> None:
        super().__init__(builtins)
        self._pool_factory = pool_factory
        self._initialized = False
        self._lock = asyncio.Lock()

    async def initialize(self) -> None:
        if self._initialized:
            return
        async with self._lock:
            if self._initialized:
                return
            pool = await self._pool_factory()
            async with pool.acquire() as conn:
                versions = await conn.fetch(
                    "SELECT version_id, actor_id, prompt_version, prompt_text, "
                    "content_sha256, source_filename, created_at "
                    "FROM actor_prompt_versions ORDER BY created_at"
                )
                active = await conn.fetch(
                    "SELECT actor_id, version_id, revision, activated_at "
                    "FROM actor_prompt_active"
                )
            for row in versions:
                version = ActorPromptVersion(
                    actor_id=str(row["actor_id"]),
                    version_id=str(row["version_id"]),
                    prompt_version=str(row["prompt_version"]),
                    prompt_text=str(row["prompt_text"]),
                    content_sha256=str(row["content_sha256"]),
                    source_filename=str(row["source_filename"]),
                    created_at=row["created_at"],
                    source="private_upload",
                )
                self._versions[version.actor_id][version.version_id] = version
            for row in active:
                actor_id = str(row["actor_id"])
                version_id = str(row["version_id"])
                if version_id in self._versions.get(actor_id, {}):
                    self._active[actor_id] = ActorPromptActiveState(
                        actor_id, version_id, int(row["revision"]), row["activated_at"]
                    )
            self._initialized = True

    @staticmethod
    def _version_from_row(row) -> ActorPromptVersion:
        return ActorPromptVersion(
            actor_id=str(row["actor_id"]),
            version_id=str(row["version_id"]),
            prompt_version=str(row["prompt_version"]),
            prompt_text=str(row["prompt_text"]),
            content_sha256=str(row["content_sha256"]),
            source_filename=str(row["source_filename"]),
            created_at=row["created_at"],
            source="private_upload",
        )

    async def _refresh_actor_versions(self, actor_id: str) -> None:
        _validate_actor(actor_id)
        pool = await self._pool_factory()
        async with pool.acquire() as conn:
            rows = await conn.fetch(
                "SELECT version_id, actor_id, prompt_version, prompt_text, "
                "content_sha256, source_filename, created_at "
                "FROM actor_prompt_versions WHERE actor_id=$1 ORDER BY created_at",
                actor_id,
            )
        for row in rows:
            version = self._version_from_row(row)
            self._versions[actor_id][version.version_id] = version

    async def refresh_active(self) -> None:
        await self.initialize()
        pool = await self._pool_factory()
        async with pool.acquire() as conn:
            rows = await conn.fetch(
                "SELECT a.actor_id, a.version_id, a.revision, a.activated_at, "
                "v.prompt_version, v.prompt_text, v.content_sha256, "
                "v.source_filename, v.created_at "
                "FROM actor_prompt_active a LEFT JOIN actor_prompt_versions v "
                "ON v.version_id=a.version_id"
            )
        for row in rows:
            actor_id = str(row["actor_id"])
            version_id = str(row["version_id"])
            if version_id not in self._versions[actor_id] and row["prompt_text"] is not None:
                version = self._version_from_row(row)
                self._versions[actor_id][version.version_id] = version
            if version_id in self._versions[actor_id]:
                self._active[actor_id] = ActorPromptActiveState(
                    actor_id, version_id, int(row["revision"]), row["activated_at"]
                )

    async def list_versions(self, actor_id: str) -> tuple[ActorPromptVersion, ...]:
        await self.initialize()
        await self._refresh_actor_versions(actor_id)
        return await super().list_versions(actor_id)

    async def create_version(
        self, actor_id: str, source_filename: str, prompt_text: str
    ) -> ActorPromptVersion:
        await self.initialize()
        proposed = _private_version(actor_id, source_filename, prompt_text)
        pool = await self._pool_factory()
        async with pool.acquire() as conn:
            row = await conn.fetchrow(
                "INSERT INTO actor_prompt_versions "
                "(version_id, actor_id, prompt_version, prompt_text, content_sha256, source_filename) "
                "VALUES ($1,$2,$3,$4,$5,$6) "
                "ON CONFLICT (actor_id, content_sha256) DO UPDATE "
                "SET actor_id = EXCLUDED.actor_id "
                "RETURNING version_id, actor_id, prompt_version, prompt_text, "
                "content_sha256, source_filename, created_at",
                proposed.version_id,
                proposed.actor_id,
                proposed.prompt_version,
                proposed.prompt_text,
                proposed.content_sha256,
                proposed.source_filename,
            )
        version = self._version_from_row(row)
        self._versions[actor_id][version.version_id] = version
        return version

    async def activate(
        self, actor_id: str, version_id: str, *, expected_revision: int
    ) -> ActorPromptActiveState:
        await self.initialize()
        _validate_actor(actor_id)
        await self._refresh_actor_versions(actor_id)
        if version_id not in self._versions[actor_id]:
            raise KeyError(version_id)
        pool = await self._pool_factory()
        async with pool.acquire() as conn:
            async with conn.transaction():
                await conn.execute(
                    "SELECT pg_advisory_xact_lock(hashtext($1))",
                    f"actor-prompt:{actor_id}",
                )
                row = await conn.fetchrow(
                    "SELECT version_id, revision, activated_at FROM actor_prompt_active "
                    "WHERE actor_id=$1 FOR UPDATE",
                    actor_id,
                )
                current = self._active[actor_id]
                if row is not None:
                    current = ActorPromptActiveState(
                        actor_id,
                        str(row["version_id"]),
                        int(row["revision"]),
                        row["activated_at"],
                    )
                if current.version_id == version_id:
                    self._active[actor_id] = current
                    return current
                if expected_revision != current.revision:
                    raise ActorPromptRevisionConflict("actor prompt revision changed")
                row = await conn.fetchrow(
                    "INSERT INTO actor_prompt_active(actor_id, version_id, revision, activated_at) "
                    "VALUES ($1,$2,$3,NOW()) "
                    "ON CONFLICT(actor_id) DO UPDATE SET "
                    "version_id=EXCLUDED.version_id, revision=EXCLUDED.revision, "
                    "activated_at=EXCLUDED.activated_at "
                    "RETURNING version_id, revision, activated_at",
                    actor_id,
                    version_id,
                    current.revision + 1,
                )
        state = ActorPromptActiveState(
            actor_id, str(row["version_id"]), int(row["revision"]), row["activated_at"]
        )
        self._active[actor_id] = state
        return state

    async def export_text(self, actor_id: str, version_id: str) -> str:
        await self.initialize()
        if version_id not in self._versions.get(actor_id, {}):
            await self._refresh_actor_versions(actor_id)
        return await super().export_text(actor_id, version_id)


class ActiveActorPromptMapping(Mapping[str, ActorPromptProfile]):
    """A live Mapping used by Group/Bedroom services without owning Persona state."""

    def __init__(self, store: InMemoryActorPromptVersionStore) -> None:
        self.store = store

    def __getitem__(self, actor_id: str) -> ActorPromptProfile:
        return self.store.get_active_cached(actor_id)

    def __iter__(self) -> Iterator[str]:
        return iter(ACTORS)

    def __len__(self) -> int:
        return len(ACTORS)
