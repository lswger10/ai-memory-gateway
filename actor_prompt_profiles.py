"""Versioned Group actor prompts, separate from identity and memory storage."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Mapping


DEFAULT_PROFILE_PATH = Path(__file__).with_name("actor_prompt_profiles.json")


@dataclass(frozen=True)
class ActorPromptProfile:
    actor_id: str
    prompt_version: str
    prompt_text: str

    def to_dict(self) -> dict[str, str]:
        return asdict(self)


def load_actor_prompt_profiles(
    path: Path = DEFAULT_PROFILE_PATH,
) -> Mapping[str, ActorPromptProfile]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if set(payload) != {"jiao", "laoke"}:
        raise ValueError("actor prompt profiles must define exactly jiao and laoke")
    profiles = {}
    for actor_id, value in payload.items():
        if set(value) != {"actor_id", "prompt_version", "prompt_text"}:
            raise ValueError("actor prompt profile has non-canonical fields")
        if value["actor_id"] != actor_id:
            raise ValueError("actor prompt profile identity mismatch")
        if not value["prompt_version"].startswith(f"{actor_id}."):
            raise ValueError("actor prompt version must be actor-specific")
        if not value["prompt_text"].strip():
            raise ValueError("actor prompt text must be non-empty")
        profiles[actor_id] = ActorPromptProfile(**value)
    return profiles
