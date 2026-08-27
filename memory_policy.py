"""Gateway-owned Group memory types and policy primitives."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import os
from typing import Any, Mapping


class MemoryScope(str, Enum):
    WEIWEI_JIAO = "weiwei-jiao"
    WEIWEI_LAOKE = "weiwei-laoke"
    JIAO_LAOKE = "jiao-laoke"
    GROUP = "group"
    LEGACY_UNSCOPED = "legacy_unscoped"


class MemoryType(str, Enum):
    FACT = "fact"
    INFERENCE = "inference"


class Perspective(str, Enum):
    SHARED = "shared"
    WEIWEI = "weiwei"
    JIAO = "jiao"
    LAOKE = "laoke"


class SourceKind(str, Enum):
    CHAT_EXTRACTION = "chat_extraction"
    EXPLICIT_USER_MEMORY = "explicit_user_memory"
    AGENT_CANDIDATE = "agent_candidate"
    USER_ATTESTED_MEMORY = "user_attested_memory"
    SYNTHETIC_TEST = "synthetic_test"


class MemoryStatus(str, Enum):
    ACTIVE = "active"
    STALE = "stale"
    SUPERSEDED = "superseded"


def _enabled(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in {"1", "true", "yes", "on"}


def group_memory_features_from_env() -> dict[str, bool]:
    return {
        "group_memory": _enabled("GATEWAY_GROUP_MEMORY_ENABLED"),
        "agent_candidates": _enabled("GROUP_AGENT_CANDIDATES_ENABLED"),
    }


@dataclass(frozen=True)
class MemoryWrite:
    content: str
    scope: MemoryScope
    memory_type: MemoryType
    perspective: Perspective
    confidential: bool
    source_kind: SourceKind
    status: MemoryStatus = MemoryStatus.ACTIVE
    confidence: float | None = None
    evidence_count: int = 0
    provenance: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        if not self.content.strip():
            raise ValueError("memory content must be non-empty")
        if self.scope is MemoryScope.LEGACY_UNSCOPED:
            raise ValueError("typed writes cannot target legacy_unscoped")
        if self.confidence is not None and not 0 <= self.confidence <= 1:
            raise ValueError("confidence must be between 0 and 1")
        if self.evidence_count < 0:
            raise ValueError("evidence_count cannot be negative")


def quarantine_legacy_memory_row(row: Mapping[str, Any]) -> dict[str, Any]:
    """Project an unmapped legacy row without inferring identity or relationship."""
    result = dict(row)
    result["scope"] = MemoryScope.LEGACY_UNSCOPED.value
    result["memory_type"] = None
    result["perspective"] = None
    result["source_kind"] = None
    result["provenance"] = dict(result.get("provenance") or {})
    return result
