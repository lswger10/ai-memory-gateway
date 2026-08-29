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
from database import (
    complete_group_closed_burst,
    enqueue_group_closed_burst,
    get_pending_group_closed_bursts,
    persist_group_memory_candidate,
    persist_group_extracted_memory,
    search_authorized_memories,
    search_authorized_summary_candidates,
)
from group_contracts import (
    CONTRACT_VERSION,
    ClosedBurstExtractionRequest,
    ContractError,
    ContextPackRequest,
    MemoryCandidateReceipt,
    MemoryCandidateRequest,
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


class StaleCandidateError(RuntimeError):
    pass


class SensitiveCandidateError(PermissionError):
    pass


class UnstableBurstError(RuntimeError):
    def __init__(self, message: str, code: str = "stale_fence"):
        super().__init__(message)
        self.code = code


class GroupExtractorUnavailable(RuntimeError):
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
        self._relationship_summaries: dict[str, dict] = {}

    async def identity_profile(self, actor_id: str) -> dict:
        return dict(self._identity_profiles.get(actor_id, {}))

    async def refresh_relationship_summary(
        self,
        scope: str,
        content: str,
        *,
        evidence_event_ids: tuple[int, ...],
        confidential: bool,
    ) -> dict:
        if scope not in {"weiwei-jiao", "weiwei-laoke", "jiao-laoke", "group"}:
            raise ValueError("invalid summary scope")
        if not content.strip() or not evidence_event_ids:
            raise ValueError("summary content and evidence are required")
        summary = {
            "scope": scope,
            "content": content,
            "evidence_event_ids": tuple(evidence_event_ids),
            "confidential": bool(confidential),
        }
        self._relationship_summaries[scope] = summary
        return dict(summary)

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


_ACTOR_CANDIDATE_SCOPE = {
    "jiao": MemoryScope.WEIWEI_JIAO,
    "laoke": MemoryScope.WEIWEI_LAOKE,
}


def _candidate_scope(actor_id: str) -> MemoryScope:
    """Keep untrusted agent proposals inside their private relationship scope.

    Evidence links prove that the cited public events exist; they do not prove
    that model-authored candidate text is safe to widen to Group or to the
    other-agent relationship. Trusted closed-burst extraction owns those wider
    classifications.
    """
    return _ACTOR_CANDIDATE_SCOPE[actor_id]


class CandidateIngressService:
    """Actor-bound accepted-final gate for untrusted memory proposals."""

    def __init__(self, relay_client, persist=persist_group_memory_candidate) -> None:
        self.relay_client = relay_client
        self.persist = persist

    async def accept(
        self, actor_id: str, request: MemoryCandidateRequest
    ) -> MemoryCandidateReceipt:
        if actor_id not in {"jiao", "laoke"}:
            raise PermissionError("actor principal is not allowed")
        payload = request.to_dict()
        try:
            facts = await self.relay_client.verify_candidate_source(request, actor_id)
        except Exception as exc:
            from relay_group_client import RelayFactsMismatch

            if isinstance(exc, RelayFactsMismatch):
                raise StaleCandidateError("candidate final is stale") from exc
            raise

        fence = payload["fence"]
        exact = {
            "room_id": fence["room_id"],
            "conversation_id": fence["conversation_id"],
            "current_event_id": payload["source_event_id"],
            "burst_id": fence["burst_id"],
            "fence_epoch": fence["fence_epoch"],
        }
        if any(facts.get(key) != value for key, value in exact.items()):
            raise StaleCandidateError("candidate coordinates do not match Relay facts")
        if facts.get("fence_status") != "active":
            raise StaleCandidateError("candidate fence is not active")
        trigger = facts.get("trigger_event") or {}
        if trigger.get("event_id") != fence["trigger_event_id"]:
            raise StaleCandidateError("candidate trigger does not match Relay facts")

        visible = {
            int(event["event_id"]): event
            for event in [trigger]
            + list(facts.get("accepted_burst_public_events") or [])
            + list(facts.get("recent_public_events") or [])
            if event.get("visibility") == "room"
            and event.get("room_id") == fence["room_id"]
            and event.get("conversation_id") == fence["conversation_id"]
        }
        accepted_finals = {
            int(event["event_id"]): event
            for event in facts.get("accepted_burst_public_events") or []
        }
        source = accepted_finals.get(payload["source_event_id"])
        if (
            not source
            or source.get("burst_id") != fence["burst_id"]
            or source.get("actor_id") != actor_id
            or source.get("role") != "agent"
            or source.get("event_type") != "agent_final"
            or (source.get("provenance") or {}).get("generation_request_id")
            != payload["generation_request_id"]
        ):
            raise StaleCandidateError("candidate source is not the exact accepted final")

        candidate = payload["candidate"]
        evidence_ids = tuple(candidate["evidence_event_ids"])
        if payload["source_event_id"] not in evidence_ids or any(
            event_id not in visible for event_id in evidence_ids
        ):
            raise StaleCandidateError("candidate cites unauthorized evidence")
        if candidate["sensitivity_hint"]:
            raise SensitiveCandidateError(
                "agent candidates cannot create confidential memory in v1"
            )

        evidence = {event_id: visible[event_id] for event_id in evidence_ids}
        perspective = (
            Perspective(actor_id)
            if candidate["perspective"] != actor_id
            else Perspective(candidate["perspective"])
        )
        provenance = {
            "actor_id": actor_id,
            "room_id": fence["room_id"],
            "conversation_id": fence["conversation_id"],
            "burst_id": fence["burst_id"],
            "trigger_event_id": fence["trigger_event_id"],
            "fence_epoch": fence["fence_epoch"],
            "source_event_id": payload["source_event_id"],
            "generation_request_id": payload["generation_request_id"],
            "candidate_index": payload["candidate_index"],
            "evidence_event_ids": list(evidence_ids),
        }
        write = MemoryWrite(
            content=candidate["content"],
            scope=_candidate_scope(actor_id),
            memory_type=MemoryType(candidate["memory_type"]),
            perspective=perspective,
            confidential=False,
            source_kind=SourceKind.AGENT_CANDIDATE,
            confidence=candidate["confidence"],
            evidence_count=len(evidence_ids),
            provenance=provenance,
        )
        auth_context = MemoryAuthContext(actor_id=actor_id, room_id=fence["room_id"])
        identity = (
            actor_id,
            payload["generation_request_id"],
            payload["candidate_index"],
        )
        payload_hash = hashlib.sha256(
            json.dumps(
                payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
            ).encode("utf-8")
        ).hexdigest()
        try:
            memory_id = await self.persist(
                identity=identity,
                payload_hash=payload_hash,
                write=write,
                auth_context=auth_context,
            )
        except ValueError as exc:
            raise ContractError("candidate identity conflict") from exc
        return MemoryCandidateReceipt.from_dict(
            {
                "contract_version": CONTRACT_VERSION,
                "accepted": True,
                "memory_id": memory_id,
                "source_event_id": payload["source_event_id"],
            }
        )


class DatabaseClosedBurstQueue:
    async def enqueue(self, closed_ref: dict, *, estimated_tokens: int = 0) -> dict:
        return await enqueue_group_closed_burst(
            closed_ref, estimated_tokens=estimated_tokens
        )

    async def pending(self) -> list[dict]:
        return await get_pending_group_closed_bursts()

    async def complete(self, closed_ref: dict) -> None:
        await complete_group_closed_burst(closed_ref)


async def _no_group_extractor(_facts: dict) -> None:
    """Never acknowledge durable work when no extractor was configured."""
    raise GroupExtractorUnavailable("Group extractor is not configured")


_BATCH_SCOPE_ACTORS = {
    MemoryScope.WEIWEI_JIAO: frozenset({"weiwei", "jiao"}),
    MemoryScope.WEIWEI_LAOKE: frozenset({"weiwei", "laoke"}),
    MemoryScope.JIAO_LAOKE: frozenset({"jiao", "laoke"}),
}


class GroupBatchExtractionPipeline:
    """Validate model output against Relay public facts before typed persistence."""

    def __init__(self, model_extract, *, persist=persist_group_extracted_memory):
        self.model_extract = model_extract
        self.persist = persist

    async def __call__(self, facts: dict) -> tuple[int, ...]:
        events = [facts["trigger_event"]] + list(
            facts.get("accepted_burst_public_events") or []
        )
        visible = {
            int(event["event_id"]): event
            for event in events
            if event.get("visibility") == "room"
        }
        proposals = await self.model_extract(facts)
        persisted: list[int] = []
        for proposal in proposals:
            evidence_ids = tuple(int(value) for value in proposal["evidence_event_ids"])
            if not evidence_ids or any(value not in visible for value in evidence_ids):
                raise ForbiddenMemoryWrite("batch extraction cites non-public evidence")
            evidence_actors = frozenset(
                visible[value]["actor_id"] for value in evidence_ids
            )
            scope = MemoryScope(proposal["scope"])
            scoped_actors = _BATCH_SCOPE_ACTORS.get(scope)
            if scoped_actors is not None and not evidence_actors <= scoped_actors:
                raise ForbiddenMemoryWrite("pairwise extraction includes a third party")
            perspective = Perspective(proposal["perspective"])
            if perspective is Perspective.SHARED:
                raise InvalidSharedEvidence(
                    "automatic extraction cannot promote an observation to shared"
                )
            if perspective.value not in evidence_actors:
                raise ForbiddenMemoryWrite("perspective actor lacks cited evidence")
            supported = max(
                (
                    str(visible[value].get("created_at") or "")
                    for value in evidence_ids
                ),
                default="",
            ) or None
            provenance = {
                "room_id": facts["room_id"],
                "conversation_id": facts["conversation_id"],
                "burst_id": facts["burst_id"],
                "trigger_event_id": facts["trigger_event"]["event_id"],
                "fence_epoch": facts["fence_epoch"],
                "evidence_event_ids": list(evidence_ids),
                "last_supported_at": supported,
            }
            write = MemoryWrite(
                content=proposal["content"],
                scope=scope,
                memory_type=MemoryType(proposal["memory_type"]),
                perspective=perspective,
                confidential=False,
                source_kind=SourceKind.CHAT_EXTRACTION,
                confidence=proposal.get("confidence"),
                evidence_count=len(evidence_ids),
                provenance=provenance,
            )
            persisted.append(
                await self.persist(
                    write,
                    MemoryAuthContext(actor_id="weiwei", room_id=facts["room_id"]),
                )
            )
        return tuple(persisted)


class ClosedBurstExtractionService:
    """Durable coordinate queue whose source text is always re-fetched from Relay."""

    _CLOSE_REASONS = {
        "budget_exhausted", "all_pass", "reaction_only",
        "no_available_candidate", "user_preempted", "crash_terminated",
    }

    def __init__(
        self,
        relay_client,
        queue=None,
        *,
        extractor=None,
        burst_threshold: int = 1,
        token_threshold: int = 0,
        max_wait_seconds: float = 0,
    ) -> None:
        self.relay_client = relay_client
        self.queue = queue or DatabaseClosedBurstQueue()
        self.extractor = extractor or _no_group_extractor
        if burst_threshold < 1 or token_threshold < 0 or max_wait_seconds < 0:
            raise ValueError("invalid Group extraction thresholds")
        self.burst_threshold = int(burst_threshold)
        self.token_threshold = int(token_threshold)
        self.max_wait_seconds = float(max_wait_seconds)
        self._seen: set[tuple[str, int]] = set()
        self.threshold_burst_count = 0
        self.extraction_unit_count = 0

    @staticmethod
    def _validate_facts(ref: dict, facts: dict) -> None:
        expected = {
            "room_id": ref["room_id"],
            "conversation_id": ref["conversation_id"],
            "burst_id": ref["burst_id"],
            "fence_epoch": ref["fence_epoch"],
        }
        if any(facts.get(key) != value for key, value in expected.items()):
            raise UnstableBurstError("closed burst coordinates do not match Relay")
        if facts.get("fence_status") != "closed":
            raise UnstableBurstError("burst is not stably closed", "burst_not_closed")
        if facts.get("close_reason") not in ClosedBurstExtractionService._CLOSE_REASONS:
            raise UnstableBurstError("closed burst reason is not canonical")
        trigger = facts.get("trigger_event") or {}
        if trigger.get("event_id") != ref["trigger_event_id"]:
            raise UnstableBurstError("closed burst trigger does not match Relay")
        for event in [trigger] + list(facts.get("accepted_burst_public_events") or []):
            if (
                event.get("room_id") != ref["room_id"]
                or event.get("conversation_id") != ref["conversation_id"]
                or event.get("burst_id") != ref["burst_id"]
                or event.get("visibility") != "room"
            ):
                raise UnstableBurstError("closed burst contains mismatched facts")

    async def enqueue(self, request: ClosedBurstExtractionRequest) -> dict:
        ref = request.to_dict()["closed_fence"]
        try:
            facts = await self.relay_client.fetch_closed_burst_facts(request)
        except Exception as exc:
            from relay_group_client import RelayFactsMismatch

            if isinstance(exc, RelayFactsMismatch):
                raise UnstableBurstError("Relay rejected closed burst") from exc
            raise
        self._validate_facts(ref, facts)
        event_text = "".join(
            str(event.get("content") or "")
            for event in [facts["trigger_event"]]
            + list(facts.get("accepted_burst_public_events") or [])
        )
        estimated_tokens = max(1, (len(event_text) + 3) // 4)
        row = await self.queue.enqueue(ref, estimated_tokens=estimated_tokens)
        key = (ref["burst_id"], ref["fence_epoch"])
        if key not in self._seen:
            self._seen.add(key)
            self.threshold_burst_count += 1
        return row

    async def process_once(self) -> int:
        pending = await self.queue.pending()
        if not pending:
            return 0
        token_total = sum(int(row.get("estimated_tokens") or 0) for row in pending)
        oldest = pending[0].get("enqueued_at")
        if isinstance(oldest, str):
            oldest = _parse_timestamp(oldest)
        elapsed_ready = False
        if self.max_wait_seconds and isinstance(oldest, datetime):
            elapsed_ready = (
                datetime.now(timezone.utc) - oldest.astimezone(timezone.utc)
            ).total_seconds() >= self.max_wait_seconds
        if not (
            len(pending) >= self.burst_threshold
            or (self.token_threshold > 0 and token_total >= self.token_threshold)
            or elapsed_ready
        ):
            return 0
        processed = 0
        for row in pending:
            request = ClosedBurstExtractionRequest.from_dict(
                {"contract_version": CONTRACT_VERSION, "closed_fence": row["closed_fence"]}
            )
            facts = await self.relay_client.fetch_closed_burst_facts(request)
            self._validate_facts(row["closed_fence"], facts)
            await self.extractor(facts)
            await self.queue.complete(row["closed_fence"])
            self.extraction_unit_count += 1
            processed += 1
        return processed


SearchFunction = Callable[
    [str, RetrievalPolicy, int], Awaitable[AuthorizedMemorySearchResult]
]
SummarySearchFunction = Callable[
    [str, RetrievalPolicy, int], Awaitable[tuple[dict, ...]]
]


async def _empty_summary_search(
    _query: str, _policy: RetrievalPolicy, _limit: int
) -> tuple[dict, ...]:
    return ()


def build_scoped_summary_search(rows) -> SummarySearchFunction:
    normalized = tuple(dict(row) for row in rows)

    async def search(
        _query: str, policy: RetrievalPolicy, limit: int
    ) -> tuple[dict, ...]:
        if not isinstance(policy, RetrievalPolicy):
            raise TypeError("summary search requires RetrievalPolicy")
        return tuple(
            row
            for row in normalized
            if row.get("scope") in policy.allowed_scopes
            and (
                not row.get("confidential", False)
                or row.get("scope") in policy.confidential_scopes
            )
        )[:limit]

    return search


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
    summaries,
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
    if summaries:
        lines.append("Authorized relationship summaries:")
        lines.extend(f"- [{row['scope']}] {row['content']}" for row in summaries)
    if actor_private_stance:
        lines.append("Your private burst stance: " + actor_private_stance)
    return "\n".join(lines)


class GroupContextPackService:
    def __init__(
        self,
        relay_client,
        *,
        search: SearchFunction = search_authorized_memories,
        summary_search: SummarySearchFunction | None = None,
        prompt_profiles=None,
    ) -> None:
        self.relay_client = relay_client
        self.search = search
        self.summary_search = (
            search_authorized_summary_candidates
            if summary_search is None and search is search_authorized_memories
            else summary_search or _empty_summary_search
        )
        self.prompt_profiles = prompt_profiles or load_actor_prompt_profiles()
        self.last_search: AuthorizedMemorySearchResult | None = None
        self.last_summary_candidate_ids: tuple[int, ...] = ()

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
        summaries: tuple[dict, ...] = ()
        if pack_kind == "full":
            summaries = await self.summary_search(
                _query_text(facts, requested["current_event_id"]), policy, 6
            )
        self.last_summary_candidate_ids = tuple(int(row["id"]) for row in summaries)
        profile = self.prompt_profiles[requested["actor_id"]]
        system_content = _compose_system_content(
            profile,
            policy,
            result.memories,
            summaries,
            requested["actor_private_stance"],
        )

        if pack_kind == "probe":
            system_content += (
                "\nProbe response contract: Return exactly one JSON object with all "
                "of these fields and no markdown: action, urge, reason_code, "
                "reaction, reaction_target_event_id, reply_to_event_id, "
                "burst_stance. action must be pass, react, reply, or react+reply; "
                "urge must be strong, normal, weak, or pass. Use null when no "
                "reaction or target applies, and keep burst_stance to at most "
                "two short lines."
            )
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
