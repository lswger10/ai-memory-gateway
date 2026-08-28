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
        "burst_extraction": _enabled("GROUP_BURST_EXTRACTION_ENABLED"),
    }


@dataclass(frozen=True)
class RetrievalPolicy:
    actor_id: str
    room_id: str
    present_actor_ids: frozenset[str]
    allowed_scopes: tuple[str, ...]
    confidential_scopes: tuple[str, ...]


@dataclass(frozen=True)
class CandidateAudit:
    sql_candidate_ids: tuple[int, ...] = ()
    vector_candidate_ids: tuple[int, ...] = ()
    python_vector_loaded_ids: tuple[int, ...] = ()
    archive_candidate_ids: tuple[int, ...] = ()
    summary_candidate_ids: tuple[int, ...] = ()
    rerank_candidate_ids: tuple[int, ...] = ()


@dataclass(frozen=True)
class AuthorizedMemorySearchResult:
    memories: tuple[Mapping[str, Any], ...]
    candidate_ids: tuple[int, ...]
    audit: CandidateAudit


_ROOM_MEMBERS = {
    "room_weiwei_jiao": frozenset({"weiwei", "jiao"}),
    "room_weiwei_laoke": frozenset({"weiwei", "laoke"}),
    "room_group_home": frozenset({"weiwei", "jiao", "laoke"}),
}


def room_members(room_id: str) -> frozenset[str]:
    members = _ROOM_MEMBERS.get(room_id)
    if members is None:
        raise ValueError("unknown room")
    return members


def build_retrieval_policy(
    actor_id: str,
    room_id: str,
    present_actor_ids: frozenset[str],
) -> RetrievalPolicy:
    expected_members = _ROOM_MEMBERS.get(room_id)
    if expected_members is None or present_actor_ids != expected_members:
        raise ValueError("room membership does not match the canonical registry")
    if actor_id not in {"jiao", "laoke"} or actor_id not in present_actor_ids:
        raise ValueError("actor is not an authorized room member")

    if room_id == "room_group_home":
        scopes = (
            (MemoryScope.WEIWEI_JIAO, MemoryScope.JIAO_LAOKE, MemoryScope.GROUP)
            if actor_id == "jiao"
            else (MemoryScope.WEIWEI_LAOKE, MemoryScope.JIAO_LAOKE, MemoryScope.GROUP)
        )
        confidential_scopes: tuple[MemoryScope, ...] = ()
    elif room_id == "room_weiwei_jiao" and actor_id == "jiao":
        scopes = (MemoryScope.WEIWEI_JIAO, MemoryScope.GROUP)
        confidential_scopes = (MemoryScope.WEIWEI_JIAO,)
    elif room_id == "room_weiwei_laoke" and actor_id == "laoke":
        scopes = (MemoryScope.WEIWEI_LAOKE, MemoryScope.GROUP)
        confidential_scopes = (MemoryScope.WEIWEI_LAOKE,)
    else:
        raise ValueError("actor cannot retrieve memory in this room")

    return RetrievalPolicy(
        actor_id=actor_id,
        room_id=room_id,
        present_actor_ids=present_actor_ids,
        allowed_scopes=tuple(scope.value for scope in scopes),
        confidential_scopes=tuple(scope.value for scope in confidential_scopes),
    )


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
