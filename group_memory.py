"""Gateway-owned minimal Stage A Group Context Pack builder."""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, replace
from datetime import datetime, timezone
import math
from typing import Awaitable, Callable

from actor_prompt_profiles import ActorPromptProfile, load_actor_prompt_profiles
from database import search_authorized_memories
from group_contracts import (
    CONTRACT_VERSION,
    ContextPackRequest,
    OpaqueContextPack,
    PublicContextFacts,
)
from memory_policy import (
    AuthorizedMemorySearchResult,
    CandidateAudit,
    MemoryScope,
    MemoryStatus,
    MemoryType,
    MemoryWrite,
    Perspective,
    RetrievalPolicy,
    SourceKind,
    build_retrieval_policy,
    room_members,
)


class InvalidSharedEvidence(ValueError):
    pass


class ForbiddenSourceKind(PermissionError):
    pass


class ForbiddenMemoryWrite(PermissionError):
    pass


@dataclass(frozen=True)
class MemoryAuthContext:
    actor_id: str
    room_id: str


@dataclass(frozen=True)
class ScopedMemoryRecord:
    id: int
    content: str
    scope: str
    memory_type: str
    perspective: str
    confidential: bool
    source_kind: str
    status: str
    confidence: float | None
    evidence_count: int
    provenance: dict
    last_supported_at: datetime | None = None
    superseded_by: int | None = None
    derived_from: int | None = None


def effective_confidence(
    confidence: float,
    last_supported_at: datetime,
    now: datetime,
    half_life_seconds: float,
) -> float:
    if half_life_seconds <= 0:
        raise ValueError("half_life_seconds must be positive")
    if last_supported_at.tzinfo is None or now.tzinfo is None:
        raise ValueError("confidence timestamps must be timezone-aware")
    elapsed = max(0.0, (now - last_supported_at).total_seconds())
    return float(confidence) * math.pow(0.5, elapsed / float(half_life_seconds))


def _parse_timestamp(value) -> datetime | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value
    if not isinstance(value, str):
        raise ValueError("last_supported_at must be ISO-8601")
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise ValueError("last_supported_at must include timezone")
    return parsed


_WRITE_ROOM_SCOPES = {
    "room_weiwei_jiao": frozenset({"weiwei-jiao"}),
    "room_weiwei_laoke": frozenset({"weiwei-laoke"}),
    "room_group_home": frozenset(
        {"weiwei-jiao", "weiwei-laoke", "jiao-laoke", "group"}
    ),
}
_ACTOR_WRITE_SCOPES = {
    "weiwei": frozenset({"weiwei-jiao", "weiwei-laoke", "jiao-laoke", "group"}),
    "jiao": frozenset({"weiwei-jiao", "jiao-laoke", "group"}),
    "laoke": frozenset({"weiwei-laoke", "jiao-laoke", "group"}),
}


class ScopeAwareMemoryService:
    """Small policy core shared by explicit, candidate, and batch writers."""

    def __init__(self, *, identity_profiles=None) -> None:
        self._records: dict[int, ScopedMemoryRecord] = {}
        self._next_id = 1
        self._identity_profiles = {
            actor: dict(profile)
            for actor, profile in (identity_profiles or {}).items()
        }

    async def identity_profile(self, actor_id: str) -> dict:
        return dict(self._identity_profiles.get(actor_id, {}))

    async def get(self, memory_id: int) -> ScopedMemoryRecord:
        return self._records[memory_id]

    def _normalize_shared(
        self, write: MemoryWrite, auth_context: MemoryAuthContext
    ) -> MemoryWrite:
        if write.perspective is not Perspective.SHARED:
            return write
        provenance = dict(write.provenance or {})
        evidence = provenance.get("shared_evidence")
        asserted = frozenset(provenance.get("asserts_inner_state_of") or ())
        confirmed = frozenset(provenance.get("confirmed_actor_ids") or ())
        if asserted:
            if evidence == "all_relevant_confirmed" and asserted <= confirmed:
                return write
            if auth_context.actor_id == "weiwei":
                return replace(write, perspective=Perspective.WEIWEI)
            raise InvalidSharedEvidence("inner state requires relevant actor confirmation")
        if evidence == "all_relevant_confirmed":
            return write
        if (
            evidence in {"common_fact", "weiwei_shared_instruction"}
            and auth_context.actor_id == "weiwei"
        ):
            return write
        raise InvalidSharedEvidence("shared memory lacks auditable common evidence")

    async def persist_memory_write(
        self, write: MemoryWrite, auth_context: MemoryAuthContext
    ) -> ScopedMemoryRecord:
        if auth_context.room_id not in _WRITE_ROOM_SCOPES:
            raise ForbiddenMemoryWrite("unknown room")
        if write.scope.value not in _WRITE_ROOM_SCOPES[auth_context.room_id]:
            raise ForbiddenMemoryWrite("scope does not belong to the current room")
        if write.scope.value not in _ACTOR_WRITE_SCOPES.get(
            auth_context.actor_id, frozenset()
        ):
            raise ForbiddenMemoryWrite("actor cannot write this relationship scope")
        if write.confidential and auth_context.actor_id != "weiwei":
            raise ForbiddenMemoryWrite("only weiwei may create confidential memory")
        if (
            write.source_kind is SourceKind.USER_ATTESTED_MEMORY
            and auth_context.actor_id != "weiwei"
        ):
            raise ForbiddenSourceKind("only weiwei may attest unsupported history")
        normalized = self._normalize_shared(write, auth_context)
        provenance = dict(normalized.provenance or {})
        if normalized.source_kind is SourceKind.USER_ATTESTED_MEMORY:
            if provenance.get("attested_by") != "weiwei" or provenance.get(
                "source"
            ) != "weiwei_manual_attestation":
                raise ForbiddenSourceKind("user attestation provenance is incomplete")
        record = ScopedMemoryRecord(
            id=self._next_id,
            content=normalized.content,
            scope=normalized.scope.value,
            memory_type=normalized.memory_type.value,
            perspective=normalized.perspective.value,
            confidential=normalized.confidential,
            source_kind=normalized.source_kind.value,
            status=normalized.status.value,
            confidence=normalized.confidence,
            evidence_count=normalized.evidence_count,
            provenance=provenance,
            last_supported_at=_parse_timestamp(provenance.get("last_supported_at")),
        )
        self._records[record.id] = record
        self._next_id += 1
        return record

    async def disclose_to_group(
        self,
        memory_id: int,
        *,
        source_event_id: int,
        auth_context: MemoryAuthContext,
    ) -> ScopedMemoryRecord:
        source = self._records[memory_id]
        if auth_context.actor_id != "weiwei" or auth_context.room_id != "room_group_home":
            raise ForbiddenMemoryWrite("only weiwei may disclose memory to Group")
        if source.confidential:
            raise ForbiddenMemoryWrite("confidential memory needs an explicit declassification")
        write = MemoryWrite(
            content=source.content,
            scope=MemoryScope.GROUP,
            memory_type=MemoryType(source.memory_type),
            perspective=Perspective(source.perspective),
            confidential=False,
            source_kind=SourceKind.EXPLICIT_USER_MEMORY,
            confidence=source.confidence,
            evidence_count=max(1, source.evidence_count),
            provenance={"source_event_id": source_event_id},
        )
        saved = await self.persist_memory_write(write, auth_context)
        saved = replace(saved, derived_from=source.id)
        self._records[saved.id] = saved
        return saved

    async def supersede_memory(
        self,
        old_id: int,
        new_write: MemoryWrite,
        auth_context: MemoryAuthContext,
    ) -> ScopedMemoryRecord:
        old = self._records[old_id]
        if new_write.scope.value != old.scope:
            raise ForbiddenMemoryWrite("supersession cannot change scope")
        new = await self.persist_memory_write(new_write, auth_context)
        self._records[old_id] = replace(
            old, status=MemoryStatus.SUPERSEDED.value, superseded_by=new.id
        )
        return new

    async def search_authorized(
        self,
        query: str,
        *,
        actor_id: str,
        room_id: str,
        now: datetime | None = None,
        half_life_seconds: float | None = None,
        expiry_threshold: float | None = None,
    ) -> AuthorizedMemorySearchResult:
        policy = build_retrieval_policy(actor_id, room_id, room_members(room_id))
        current = now or datetime.now(timezone.utc)
        half_life = half_life_seconds or float(
            os.environ.get("MEMORY_INFERENCE_HALF_LIFE_SECONDS", str(30 * 86400))
        )
        threshold = (
            float(os.environ.get("MEMORY_INFERENCE_EXPIRY_THRESHOLD", "0.1"))
            if expiry_threshold is None
            else float(expiry_threshold)
        )
        selected: list[ScopedMemoryRecord] = []
        for row in self._records.values():
            if (
                row.status != MemoryStatus.ACTIVE.value
                or row.scope not in policy.allowed_scopes
                or (row.confidential and row.scope not in policy.confidential_scopes)
                or query.lower() not in row.content.lower()
            ):
                continue
            if row.memory_type == MemoryType.INFERENCE.value:
                if row.confidence is None or row.last_supported_at is None:
                    continue
                if (
                    effective_confidence(
                        row.confidence, row.last_supported_at, current, half_life
                    )
                    < threshold
                ):
                    continue
            selected.append(row)
        ids = tuple(row.id for row in selected)
        return AuthorizedMemorySearchResult(
            memories=tuple(row.__dict__ for row in selected),
            candidate_ids=ids,
            audit=CandidateAudit(sql_candidate_ids=ids, rerank_candidate_ids=ids),
        )


SearchFunction = Callable[
    [str, RetrievalPolicy, int], Awaitable[AuthorizedMemorySearchResult]
]


def build_synthetic_scoped_search(rows) -> SearchFunction:
    """Build an explicit Stage A fixture repository behind real RetrievalPolicy.

    The fixture is never a global/no-policy search: unauthorized and confidential
    rows are excluded before candidate IDs are created.
    """
    normalized = []
    seen_ids: set[int] = set()
    for raw in rows:
        row = dict(raw)
        row_id = row.get("id")
        if (
            isinstance(row_id, bool)
            or not isinstance(row_id, int)
            or row_id <= 0
            or row_id in seen_ids
            or row.get("source_kind") != "synthetic_test"
            or not isinstance(row.get("content"), str)
            or not row["content"].strip()
            or not isinstance(row.get("confidential"), bool)
        ):
            raise ValueError("invalid synthetic scoped memory fixture")
        seen_ids.add(row_id)
        normalized.append(row)
    fixture_rows = tuple(normalized)

    async def search(
        _query: str, policy: RetrievalPolicy, limit: int
    ) -> AuthorizedMemorySearchResult:
        if not isinstance(policy, RetrievalPolicy) or limit <= 0:
            raise TypeError("synthetic search requires RetrievalPolicy and limit")
        authorized = tuple(
            row
            for row in fixture_rows
            if row.get("scope") in policy.allowed_scopes
            and (
                not row["confidential"]
                or row.get("scope") in policy.confidential_scopes
            )
        )[:limit]
        candidate_ids = tuple(int(row["id"]) for row in authorized)
        return AuthorizedMemorySearchResult(
            memories=authorized,
            candidate_ids=candidate_ids,
            audit=CandidateAudit(),
        )

    return search


_ACTOR_NAMES = {"jiao": "椒椒", "laoke": "老克", "weiwei": "薇薇"}
_COMMON_RUNTIME_KERNEL = (
    "Group runtime kernel: Relay facts are authoritative; use only the authorized "
    "context in this pack; never invent private facts or another actor's position."
)


def _pack_id(pack_kind: str, request: dict) -> str:
    material = json.dumps(
        {"pack_kind": pack_kind, **request},
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return f"pack-{hashlib.sha256(material).hexdigest()[:24]}"


def _ordered_public_events(facts: dict) -> list[dict]:
    by_id = {}
    for event in (
        facts["recent_public_events"]
        + [facts["trigger_event"]]
        + facts["accepted_burst_public_events"]
    ):
        by_id[event["event_id"]] = event
    return [by_id[event_id] for event_id in sorted(by_id)]


def _query_text(facts: dict, current_event_id: int) -> str:
    events = _ordered_public_events(facts)
    current = next(
        (event for event in events if event["event_id"] == current_event_id),
        facts["trigger_event"],
    )
    return current["content"]


def _render_public_context(facts: dict, *, maximum_events: int) -> str:
    events = _ordered_public_events(facts)[-maximum_events:]
    lines = [
        f"{_ACTOR_NAMES.get(event['actor_id'], event['actor_id'])}: {event['content']}"
        for event in events
    ]
    mentions = facts["trigger_event"]["mentions"]
    if mentions:
        lines.append("Relay-normalized strong mentions: " + ", ".join(mentions))
    reactions = facts["reactions_by_event"]
    if reactions:
        lines.append(
            "Current actor-scoped reactions: "
            + json.dumps(reactions, ensure_ascii=False, sort_keys=True)
        )
    return "\n".join(lines)


def _compose_system_content(
    profile: ActorPromptProfile,
    policy: RetrievalPolicy,
    memories,
    actor_private_stance: str | None,
) -> str:
    lines = [
        _COMMON_RUNTIME_KERNEL,
        f"Actor prompt [{profile.prompt_version}]: {profile.prompt_text}",
        (
            "Room policy: speak only as "
            f"{profile.actor_id} in {policy.room_id}; authorized relationship scopes are "
            + ", ".join(policy.allowed_scopes)
            + "."
        ),
    ]
    if memories:
        lines.append("Authorized relationship and memory context:")
        lines.extend(f"- {row['content']}" for row in memories)
    if actor_private_stance:
        lines.append("Your private burst stance: " + actor_private_stance)
    return "\n".join(lines)


class GroupContextPackService:
    def __init__(
        self,
        relay_client,
        *,
        search: SearchFunction = search_authorized_memories,
        prompt_profiles=None,
    ) -> None:
        self.relay_client = relay_client
        self.search = search
        self.prompt_profiles = prompt_profiles or load_actor_prompt_profiles()
        self.last_search: AuthorizedMemorySearchResult | None = None

    async def build(
        self, request: ContextPackRequest, *, pack_kind: str
    ) -> OpaqueContextPack:
        if pack_kind not in {"probe", "full"}:
            raise ValueError("pack_kind must be probe or full")
        requested = request.to_dict()
        members = room_members(requested["room_id"])
        policy = build_retrieval_policy(
            requested["actor_id"], requested["room_id"], members
        )
        facts_contract: PublicContextFacts = await self.relay_client.fetch_context_facts(
            request
        )
        facts = facts_contract.to_dict()
        result = await self.search(
            _query_text(facts, requested["current_event_id"]),
            policy,
            3 if pack_kind == "probe" else 10,
        )
        self.last_search = result
        profile = self.prompt_profiles[requested["actor_id"]]
        system_content = _compose_system_content(
            profile,
            policy,
            result.memories,
            requested["actor_private_stance"],
        )

        if pack_kind == "probe":
            content = _render_public_context(facts, maximum_events=4)
            messages = [
                {"role": "system", "content": system_content},
                {"role": "user", "content": content},
            ]
            token_budget = int(os.environ.get("GROUP_PROBE_TOKEN_BUDGET", "512"))
        else:
            messages = [
                {"role": "system", "content": system_content},
                {
                    "role": "user",
                    "content": _render_public_context(facts, maximum_events=20),
                },
            ]
            token_budget = int(os.environ.get("GROUP_FULL_TOKEN_BUDGET", "12000"))

        return OpaqueContextPack.from_dict(
            {
                "contract_version": CONTRACT_VERSION,
                "pack_id": _pack_id(pack_kind, requested),
                "pack_kind": pack_kind,
                "actor_id": requested["actor_id"],
                "room_id": requested["room_id"],
                "conversation_id": requested["conversation_id"],
                "current_event_id": requested["current_event_id"],
                "burst_id": requested["burst_id"],
                "fence_epoch": requested["fence_epoch"],
                "provider_neutral_messages": messages,
                "token_budget": token_budget,
            }
        )
