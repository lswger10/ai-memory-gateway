from __future__ import annotations

import json
from pathlib import Path

from model_profiles import ModelProfile


async def bootstrap_ephemeral_model_profiles(store, path: Path) -> None:
    """Load explicit local-acceptance Profiles into an in-memory repository.

    This is activated only by the test-only environment path in ``main``. It
    never infers actor identity, model family, fallback order, or provider
    capabilities.
    """
    payload = json.loads(path.read_text(encoding="utf-8"))
    expected = {
        "profiles",
        "actor_defaults",
        "approved_fallbacks",
        "room_overrides",
    }
    if not isinstance(payload, dict) or set(payload) != expected:
        raise ValueError("invalid ephemeral Model Profile fixture")
    profiles = payload["profiles"]
    if not isinstance(profiles, list):
        raise ValueError("fixture profiles must be a list")
    for item in profiles:
        await store.put_profile(ModelProfile.from_dict(item))
    defaults = payload["actor_defaults"]
    fallbacks = payload["approved_fallbacks"]
    if not isinstance(defaults, dict) or not isinstance(fallbacks, dict):
        raise ValueError("fixture bindings must be objects")
    for actor_id in ("jiao", "laoke"):
        await store.set_actor_default(actor_id, str(defaults[actor_id]))
        ordered = fallbacks.get(actor_id, [])
        if not isinstance(ordered, list):
            raise ValueError("fixture fallbacks must be explicit lists")
        await store.set_approved_fallbacks(
            actor_id, tuple(str(item) for item in ordered)
        )
    overrides = payload["room_overrides"]
    if not isinstance(overrides, list):
        raise ValueError("fixture room_overrides must be a list")
    for item in overrides:
        if not isinstance(item, dict) or set(item) != {
            "room_id",
            "actor_id",
            "profile_id",
        }:
            raise ValueError("invalid room override fixture")
        await store.set_room_override(
            str(item["room_id"]),
            str(item["actor_id"]),
            str(item["profile_id"]),
        )
