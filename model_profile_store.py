from __future__ import annotations

import asyncio
from dataclasses import dataclass, replace

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
        self._probe_results: dict[tuple[str, int, str], str] = {}
        self._lock = asyncio.Lock()

    async def put_profile(self, profile: ModelProfile) -> ModelProfile:
        async with self._lock:
            existing = self._profiles.get(profile.profile_id)
            if existing is not None and profile.revision < existing.revision:
                raise ProfileStoreError("Profile revision cannot move backwards")
            self._profiles[profile.profile_id] = profile
            return profile

    async def get_profile(self, profile_id: str) -> ModelProfile:
        async with self._lock:
            profile = self._profiles.get(profile_id)
            if profile is None:
                raise ProfileStoreError(f"unknown Profile: {profile_id}")
            return profile

    async def set_test_status(self, profile_id: str, status: str) -> ModelProfile:
        async with self._lock:
            profile = self._profiles.get(profile_id)
            if profile is None:
                raise ProfileStoreError(f"unknown Profile: {profile_id}")
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
        del observed_capabilities, sanitized_detail
        async with self._lock:
            self._probe_results[(profile_id, profile_revision, probe_kind)] = status

    async def has_verified_probe(
        self, profile_id: str, profile_revision: int, probe_kind: str
    ) -> bool:
        async with self._lock:
            return (
                self._probe_results.get((profile_id, profile_revision, probe_kind))
                == "verified"
            )

    def _selectable(self, profile_id: str) -> ModelProfile:
        profile = self._profiles.get(profile_id)
        if profile is None:
            raise ProfileStoreError(f"unknown Profile: {profile_id}")
        if not profile.selectable:
            raise ProfileStoreError(f"Profile is not selectable: {profile_id}")
        return profile

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
            primary = self._selectable(profile_id)
            fallback_ids = tuple(
                item
                for item in binding.approved_fallback_profile_ids
                if item != primary.profile_id
            )
            fallbacks = tuple(self._selectable(item) for item in fallback_ids)
            revision = override.revision if override else binding.revision
            return ResolvedProfiles(
                actor_id=actor_id,
                room_id=room_id,
                primary=primary,
                fallbacks=fallbacks,
                source="room_override" if override else "actor_default",
                binding_revision=revision,
            )
