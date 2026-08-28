"""Gateway-local validators for the vendored Group Room JSON contract."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any


CONTRACT_VERSION = "group-room.v1.0"
ROOM_IDS = frozenset({"room_weiwei_jiao", "room_weiwei_laoke", "room_group_home"})
AGENT_ACTOR_IDS = frozenset({"jiao", "laoke"})


class ContractError(ValueError):
    pass


def _exact_object(payload: Any, keys: set[str], label: str) -> dict[str, Any]:
    if not isinstance(payload, dict) or set(payload) != keys:
        raise ContractError(f"{label} has non-canonical fields")
    if payload.get("contract_version") != CONTRACT_VERSION:
        raise ContractError("contract_version mismatch")
    return payload


def _nonempty(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ContractError(f"{label} must be non-empty")
    return value


def _positive(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ContractError(f"{label} must be positive")
    return value


def _fence_token(value: Any) -> dict[str, Any]:
    keys = {
        "room_id", "conversation_id", "burst_id", "trigger_event_id",
        "fence_epoch", "lease_epoch", "orchestrator_instance",
    }
    if not isinstance(value, dict) or set(value) != keys:
        raise ContractError("fence has non-canonical fields")
    if value["room_id"] not in ROOM_IDS:
        raise ContractError("invalid fence room")
    _nonempty(value["conversation_id"], "conversation_id")
    _nonempty(value["burst_id"], "burst_id")
    _positive(value["trigger_event_id"], "trigger_event_id")
    _positive(value["fence_epoch"], "fence_epoch")
    _positive(value["lease_epoch"], "lease_epoch")
    _nonempty(value["orchestrator_instance"], "orchestrator_instance")
    return value


@dataclass(frozen=True)
class FrozenPayload:
    contract_version: str
    _json: str

    @classmethod
    def _freeze(cls, payload: dict[str, Any]):
        return cls(
            CONTRACT_VERSION,
            json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")),
        )

    def to_dict(self) -> dict[str, Any]:
        return json.loads(self._json)


@dataclass(frozen=True)
class ContextPackRequest(FrozenPayload):
    @classmethod
    def from_dict(cls, payload: Any) -> "ContextPackRequest":
        value = _exact_object(
            payload,
            {
                "contract_version", "actor_id", "room_id", "conversation_id",
                "current_event_id", "burst_id", "fence_epoch", "actor_private_stance",
            },
            "context pack request",
        )
        if value["actor_id"] not in AGENT_ACTOR_IDS or value["room_id"] not in ROOM_IDS:
            raise ContractError("invalid actor or room")
        _nonempty(value["conversation_id"], "conversation_id")
        _nonempty(value["burst_id"], "burst_id")
        _positive(value["current_event_id"], "current_event_id")
        _positive(value["fence_epoch"], "fence_epoch")
        stance = value["actor_private_stance"]
        if stance is not None:
            _nonempty(stance, "actor_private_stance")
        return cls._freeze(value)


@dataclass(frozen=True)
class ContextFactsRequest(FrozenPayload):
    @classmethod
    def from_dict(cls, payload: Any) -> "ContextFactsRequest":
        value = _exact_object(
            payload,
            {
                "contract_version", "room_id", "conversation_id", "current_event_id",
                "burst_id", "fence_epoch", "recent_limit", "require_closed",
            },
            "context facts request",
        )
        if value["room_id"] not in ROOM_IDS:
            raise ContractError("invalid facts room")
        _nonempty(value["conversation_id"], "conversation_id")
        _nonempty(value["burst_id"], "burst_id")
        _positive(value["current_event_id"], "current_event_id")
        _positive(value["fence_epoch"], "fence_epoch")
        _positive(value["recent_limit"], "recent_limit")
        if not isinstance(value["require_closed"], bool):
            raise ContractError("require_closed must be boolean")
        return cls._freeze(value)


def _validate_group_event(value: Any) -> None:
    required = {
        "contract_version", "event_id", "room_id", "conversation_id", "burst_id",
        "actor_id", "role", "event_type", "content", "reply_to_event_id", "mentions",
        "created_at", "request_id", "visibility", "provenance",
    }
    event = _exact_object(value, required, "group event")
    if event["room_id"] not in ROOM_IDS or event["visibility"] != "room":
        raise ContractError("invalid event room or visibility")
    _positive(event["event_id"], "event_id")
    if not isinstance(event["mentions"], list):
        raise ContractError("mentions must be an array")


@dataclass(frozen=True)
class PublicContextFacts(FrozenPayload):
    @classmethod
    def from_dict(cls, payload: Any) -> "PublicContextFacts":
        value = _exact_object(
            payload,
            {
                "contract_version", "room_id", "conversation_id", "current_event_id",
                "burst_id", "fence_epoch", "fence_status", "close_reason",
                "trigger_event", "accepted_burst_public_events", "recent_public_events",
                "reactions_by_event", "accepted_event_range",
            },
            "public context facts",
        )
        if value["room_id"] not in ROOM_IDS or value["fence_status"] not in {"active", "closed"}:
            raise ContractError("invalid facts room or fence status")
        _validate_group_event(value["trigger_event"])
        for field in ("accepted_burst_public_events", "recent_public_events"):
            if not isinstance(value[field], list):
                raise ContractError(f"{field} must be an array")
            for event in value[field]:
                _validate_group_event(event)
        if not isinstance(value["reactions_by_event"], dict):
            raise ContractError("reactions_by_event must be an object")
        return cls._freeze(value)


@dataclass(frozen=True)
class OpaqueContextPack(FrozenPayload):
    @classmethod
    def from_dict(cls, payload: Any) -> "OpaqueContextPack":
        value = _exact_object(
            payload,
            {
                "contract_version", "pack_id", "pack_kind", "actor_id", "room_id",
                "conversation_id", "current_event_id", "burst_id", "fence_epoch",
                "provider_neutral_messages", "token_budget",
            },
            "opaque context pack",
        )
        if value["pack_kind"] not in {"probe", "full"}:
            raise ContractError("invalid pack kind")
        if value["actor_id"] not in AGENT_ACTOR_IDS or value["room_id"] not in ROOM_IDS:
            raise ContractError("invalid pack actor or room")
        if not isinstance(value["provider_neutral_messages"], list):
            raise ContractError("pack messages must be an array")
        for message in value["provider_neutral_messages"]:
            if set(message) != {"role", "content"} or message["role"] not in {"system", "user", "assistant"} or not isinstance(message["content"], str):
                raise ContractError("invalid provider-neutral message")
        _positive(value["token_budget"], "token_budget")
        return cls._freeze(value)


@dataclass(frozen=True)
class MemoryCandidateReceipt(FrozenPayload):
    @classmethod
    def from_dict(cls, payload: Any) -> "MemoryCandidateReceipt":
        value = _exact_object(
            payload,
            {"contract_version", "accepted", "memory_id", "source_event_id"},
            "memory candidate receipt",
        )
        if value["accepted"] is not True:
            raise ContractError("candidate receipt must be accepted")
        _nonempty(value["memory_id"], "memory_id")
        _positive(value["source_event_id"], "source_event_id")
        return cls._freeze(value)


@dataclass(frozen=True)
class MemoryCandidateRequest(FrozenPayload):
    @classmethod
    def from_dict(cls, payload: Any) -> "MemoryCandidateRequest":
        value = _exact_object(
            payload,
            {
                "contract_version", "source_event_id", "fence",
                "generation_request_id", "candidate_index", "candidate",
            },
            "memory candidate request",
        )
        _positive(value["source_event_id"], "source_event_id")
        _fence_token(value["fence"])
        _nonempty(value["generation_request_id"], "generation_request_id")
        if value["candidate_index"] != 0:
            raise ContractError("candidate_index must be zero")
        candidate = value["candidate"]
        required = {
            "content", "memory_type", "perspective", "confidence",
            "evidence_event_ids", "sensitivity_hint",
        }
        if not isinstance(candidate, dict) or set(candidate) != required:
            raise ContractError("memory candidate has non-canonical fields")
        _nonempty(candidate["content"], "candidate content")
        if candidate["memory_type"] not in {"fact", "inference"}:
            raise ContractError("invalid candidate memory_type")
        if candidate["perspective"] not in {"shared", "weiwei", "jiao", "laoke"}:
            raise ContractError("invalid candidate perspective")
        confidence = candidate["confidence"]
        if confidence is not None and (
            isinstance(confidence, bool)
            or not isinstance(confidence, (int, float))
            or not 0 <= confidence <= 1
        ):
            raise ContractError("invalid candidate confidence")
        evidence = candidate["evidence_event_ids"]
        if (
            not isinstance(evidence, list)
            or any(isinstance(item, bool) or not isinstance(item, int) or item < 1 for item in evidence)
            or len(evidence) != len(set(evidence))
        ):
            raise ContractError("invalid evidence_event_ids")
        if not isinstance(candidate["sensitivity_hint"], bool):
            raise ContractError("sensitivity_hint must be boolean")
        return cls._freeze(value)


@dataclass(frozen=True)
class ContractBundleReport:
    contract_version: str
    fixture_count: int
    hash_errors: tuple[str, ...]


def verify_sha256sums(manifest_path: Path) -> tuple[str, ...]:
    errors = []
    root = manifest_path.parent
    listed = set()
    for raw_line in manifest_path.read_text(encoding="utf-8").splitlines():
        digest, relative = raw_line.split("  ", 1)
        listed.add(relative)
        target = root / Path(relative)
        if not target.is_file() or hashlib.sha256(target.read_bytes()).hexdigest() != digest:
            errors.append(relative)
    actual = {
        path.relative_to(root).as_posix()
        for path in (root / "fixtures").glob("*.json")
    } | {"group-room.schema.json"}
    errors.extend(sorted(actual - listed))
    return tuple(sorted(set(errors)))


def validate_contract_bundle(root: Path) -> ContractBundleReport:
    schema = json.loads((root / "group-room.schema.json").read_text(encoding="utf-8"))
    if schema.get("properties", {}).get("contract_version", {}).get("const") != CONTRACT_VERSION:
        raise ContractError("schema contract version mismatch")
    fixtures = sorted((root / "fixtures").glob("*.json"))
    for fixture in fixtures:
        payload = json.loads(fixture.read_text(encoding="utf-8"))
        if payload.get("contract_version") != CONTRACT_VERSION:
            raise ContractError(f"fixture version mismatch: {fixture.name}")
    return ContractBundleReport(
        CONTRACT_VERSION,
        len(fixtures),
        verify_sha256sums(root / "SHA256SUMS"),
    )
