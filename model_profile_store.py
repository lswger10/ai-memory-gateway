from __future__ import annotations

import asyncio
from dataclasses import dataclass, replace
from datetime import datetime, timedelta, timezone

from model_profiles import ModelProfile


class ProfileStoreError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class StoredBinding:
    actor_id: str
    default_profile_id: str
    approved_fallback_profile_ids: tuple[str, ...]
    revision: int


@dataclass(frozen=True, slots=True)
class RoomOverride:
    room_id: str
    actor_id: str
    profile_id: str
    revision: int


@dataclass(frozen=True, slots=True)
class ResolvedProfiles:
    actor_id: str
    room_id: str
    primary: ModelProfile
    fallbacks: tuple[ModelProfile, ...]
    source: str
    binding_revision: int


class InMemoryModelProfileStore:
    """Deterministic repository used by the resolver and contract tests.

    The production PostgreSQL repository uses the same operations. Keeping the
    resolution policy here prevents a caller from supplying arbitrary scopes or
    Profiles to model execution.
    """

    def __init__(self) -> None:
        self._profiles: dict[str, ModelProfile] = {}
        self._bindings: dict[str, StoredBinding] = {}
        self._overrides: dict[tuple[str, str], RoomOverride] = {}
        self._probe_results: dict[tuple[str, int, str], dict] = {}
        self._lock = asyncio.Lock()

    async def put_profile(self, profile: ModelProfile) -> ModelProfile:
        async with self._lock:
            existing = self._profiles.get(profile.profile_id)
            if existing is not None:
                if replace(profile, test_status=existing.test_status) == existing:
                    return existing
                if profile.revision != existing.revision + 1:
                    raise ProfileStoreError("Profile revision conflict")
                profile = replace(profile, test_status="unverified")
            self._profiles[profile.profile_id] = profile
            return profile

    async def get_profile(self, profile_id: str) -> ModelProfile:
        async with self._lock:
            profile = self._profiles.get(profile_id)
            if profile is None:
                raise ProfileStoreError(f"unknown Profile: {profile_id}")
            return profile

    async def list_profiles(self) -> tuple[ModelProfile, ...]:
        async with self._lock:
            return tuple(self._profiles.values())

    async def set_test_status(self, profile_id: str, status: str, *, expected_revision: int) -> ModelProfile:
        async with self._lock:
            profile = self._profiles.get(profile_id)
            if profile is None:
                raise ProfileStoreError(f"unknown Profile: {profile_id}")
            if profile.revision != expected_revision:
                raise ProfileStoreError("Profile revision changed during probe")
            updated = replace(profile, test_status=status)
            self._profiles[profile_id] = updated
            return updated

    async def record_probe_result(
        self,
        *,
        profile_id: str,
        profile_revision: int,
        probe_kind: str,
        status: str,
        observed_capabilities: dict,
        sanitized_detail: str | None = None,
    ) -> None:
        async with self._lock:
            self._probe_results[(profile_id, profile_revision, probe_kind)] = {
                "status": status, "observed_capabilities": observed_capabilities,
                "sanitized_detail": sanitized_detail, "tested_at": datetime.now(timezone.utc),
            }

    async def has_verified_probe(
        self, profile_id: str, profile_revision: int, probe_kind: str
    ) -> bool:
        async with self._lock:
            return (
                self._probe_results.get((profile_id, profile_revision, probe_kind), {}).get("status")
                == "verified"
            )

    async def claim_health_check(self, profile_id: str, revision: int) -> bool:
        async with self._lock:
            key = (profile_id, revision, "model_health")
            now = datetime.now(timezone.utc)
            profile = self._profiles.get(profile_id)
            if profile is None or profile.revision != revision or not profile.enabled:
                return False
            if any(row["tested_at"] > now - timedelta(days=1)
                   for (pid, _, kind), row in self._probe_results.items()
                   if pid == profile_id and kind == "model_health"):
                return False
            self._probe_results.setdefault(key, {"status": "unverified", "observed_capabilities": {}})["tested_at"] = now
            return True

    async def list_health_results(self) -> tuple[dict, ...]:
        async with self._lock:
            return tuple({"profile_id": p.profile_id, "profile_revision": p.revision,
                "display_name": p.display_name, "status": row["status"],
                "tested_at": row["tested_at"].isoformat(), **row["observed_capabilities"]}
                for p in self._profiles.values() if p.enabled
                if (row := self._probe_results.get((p.profile_id, p.revision, "model_health"))))

    def _selectable(self, profile_id: str) -> ModelProfile:
        profile = self._profiles.get(profile_id)
        if profile is None:
            raise ProfileStoreError(f"unknown Profile: {profile_id}")
        if not profile.selectable:
            raise ProfileStoreError(f"Profile is not selectable: {profile_id}")
        return profile

    async def get_actor_binding(self, actor_id: str) -> StoredBinding | None:
        async with self._lock:
            return self._bindings.get(actor_id)

    async def save_actor_binding(
        self, actor_id: str, profile_id: str, profile_ids: tuple[str, ...], *, expected_revision: int,
    ) -> StoredBinding:
        async with self._lock:
            current = self._bindings.get(actor_id)
            if (current.revision if current else 0) != expected_revision:
                raise ProfileStoreError("binding revision conflict")
            if len(set(profile_ids)) != len(profile_ids) or "*" in profile_ids or profile_id in profile_ids:
                raise ProfileStoreError("fallbacks must be unique and exclude default")
            for selected in (profile_id, *profile_ids):
                self._selectable(selected)
            saved = StoredBinding(actor_id, profile_id, tuple(profile_ids), expected_revision + 1)
            self._bindings[actor_id] = saved
            return saved

    async def set_actor_default(
        self,
        actor_id: str,
        profile_id: str,
        *,
        expected_revision: int | None = None,
    ) -> StoredBinding:
        async with self._lock:
            self._selectable(profile_id)
            previous = self._bindings.get(actor_id)
            if expected_revision is not None:
                actual = previous.revision if previous else None
                if actual != expected_revision:
                    raise ProfileStoreError("binding revision conflict")
            next_revision = 1 if previous is None else previous.revision + 1
            binding = StoredBinding(
                actor_id=actor_id,
                default_profile_id=profile_id,
                approved_fallback_profile_ids=(
                    previous.approved_fallback_profile_ids if previous else ()
                ),
                revision=next_revision,
            )
            self._bindings[actor_id] = binding
            return binding

    async def set_approved_fallbacks(
        self,
        actor_id: str,
        profile_ids: tuple[str, ...],
        *,
        expected_revision: int | None = None,
    ) -> StoredBinding:
        async with self._lock:
            binding = self._bindings.get(actor_id)
            if binding is None:
                raise ProfileStoreError("actor default must be configured first")
            if expected_revision is not None and binding.revision != expected_revision:
                raise ProfileStoreError("binding revision conflict")
            ordered = tuple(profile_ids)
            if len(set(ordered)) != len(ordered) or "*" in ordered:
                raise ProfileStoreError("fallbacks must be an explicit unique order")
            if binding.default_profile_id in ordered:
                raise ProfileStoreError("default Profile cannot repeat as fallback")
            for profile_id in ordered:
                self._selectable(profile_id)
            updated = StoredBinding(
                actor_id=actor_id,
                default_profile_id=binding.default_profile_id,
                approved_fallback_profile_ids=ordered,
                revision=binding.revision + 1,
            )
            self._bindings[actor_id] = updated
            return updated

    async def set_room_override(
        self,
        room_id: str,
        actor_id: str,
        profile_id: str,
        *,
        expected_revision: int | None = None,
    ) -> RoomOverride:
        async with self._lock:
            self._selectable(profile_id)
            key = (room_id, actor_id)
            previous = self._overrides.get(key)
            actual = previous.revision if previous else None
            if expected_revision is not None and actual != expected_revision:
                raise ProfileStoreError("room override revision conflict")
            override = RoomOverride(
                room_id=room_id,
                actor_id=actor_id,
                profile_id=profile_id,
                revision=1 if previous is None else previous.revision + 1,
            )
            self._overrides[key] = override
            return override

    async def resolve(self, actor_id: str, room_id: str) -> ResolvedProfiles:
        async with self._lock:
            binding = self._bindings.get(actor_id)
            if binding is None:
                raise ProfileStoreError(f"actor has no Model Profile binding: {actor_id}")
            override = self._overrides.get((room_id, actor_id))
            profile_id = override.profile_id if override else binding.default_profile_id
            primary = self._profiles.get(profile_id)
            if primary is None:
                raise ProfileStoreError(f"unknown Profile: {profile_id}")
            fallback_ids = tuple(
                item
                for item in binding.approved_fallback_profile_ids
                if item != primary.profile_id
            )
            fallbacks = tuple(p for item in fallback_ids
                              if (p := self._profiles.get(item)) is not None and p.selectable)
            revision = override.revision if override else binding.revision
            return ResolvedProfiles(
                actor_id=actor_id,
                room_id=room_id,
                primary=primary,
                fallbacks=fallbacks,
                source="room_override" if override else "actor_default",
                binding_revision=revision,
            )
