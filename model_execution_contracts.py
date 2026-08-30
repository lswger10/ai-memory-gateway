from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping


CONTRACT_VERSION = "gateway-model-execution.v1.0"


class ExecutionContractError(ValueError):
    pass


def _required_string(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ExecutionContractError(f"{field} must be a non-empty string")
    return value.strip()


def _positive_int(value: Any, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ExecutionContractError(f"{field} must be a positive integer")
    return value


@dataclass(frozen=True, slots=True)
class ExecutionFence:
    room_id: str
    conversation_id: str
    burst_id: str
    trigger_event_id: int
    fence_epoch: int
    lease_epoch: int
    orchestrator_instance: str

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ExecutionFence":
        fields = {
            "room_id",
            "conversation_id",
            "burst_id",
            "trigger_event_id",
            "fence_epoch",
            "lease_epoch",
            "orchestrator_instance",
        }
        unknown = set(payload) - fields
        missing = fields - set(payload)
        if unknown or missing:
            raise ExecutionContractError(
                f"invalid fence fields: missing={sorted(missing)} unknown={sorted(unknown)}"
            )
        return cls(
            room_id=_required_string(payload["room_id"], "fence.room_id"),
            conversation_id=_required_string(
                payload["conversation_id"], "fence.conversation_id"
            ),
            burst_id=_required_string(payload["burst_id"], "fence.burst_id"),
            trigger_event_id=_positive_int(
                payload["trigger_event_id"], "fence.trigger_event_id"
            ),
            fence_epoch=_positive_int(payload["fence_epoch"], "fence.fence_epoch"),
            lease_epoch=_positive_int(payload["lease_epoch"], "fence.lease_epoch"),
            orchestrator_instance=_required_string(
                payload["orchestrator_instance"], "fence.orchestrator_instance"
            ),
        )


_REQUEST_FIELDS = {
    "contract_version",
    "execution_kind",
    "actor_id",
    "room_id",
    "conversation_id",
    "current_event_id",
    "generation_request_id",
    "execution_mode",
    "fence",
    "bedroom_session_id",
    "bedroom_turn_epoch",
    "actor_private_stance",
    "binding_revision",
}


@dataclass(frozen=True, slots=True)
class GatewayExecutionRequest:
    contract_version: str
    execution_kind: str
    actor_id: str
    room_id: str | None
    conversation_id: str | None
    current_event_id: int
    generation_request_id: str
    execution_mode: str
    fence: ExecutionFence | None
    bedroom_session_id: str | None
    bedroom_turn_epoch: int | None
    actor_private_stance: str | None
    binding_revision: int | None

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "GatewayExecutionRequest":
        unknown = set(payload) - _REQUEST_FIELDS
        if unknown:
            raise ExecutionContractError(
                f"forbidden execution request field(s): {', '.join(sorted(unknown))}"
            )
        common = {
            "contract_version", "execution_kind", "execution_mode", "actor_id",
            "current_event_id", "generation_request_id", "binding_revision",
        }
        missing = common - set(payload)
        if missing:
            raise ExecutionContractError(f"missing execution request field(s): {', '.join(sorted(missing))}")
        if payload["contract_version"] != CONTRACT_VERSION:
            raise ExecutionContractError("unsupported contract_version")
        kind = payload["execution_kind"]
        if kind not in {"probe", "full"}:
            raise ExecutionContractError("execution_kind must be probe or full")
        mode = payload["execution_mode"]
        if mode not in {"group", "private", "bedroom"}:
            raise ExecutionContractError("invalid execution_mode")
        if mode == "bedroom":
            required = {"bedroom_session_id", "bedroom_turn_epoch"}
            missing = required - set(payload)
            forbidden = {"room_id", "conversation_id", "fence", "actor_private_stance"} & set(payload)
            if missing or forbidden:
                raise ExecutionContractError("invalid bedroom execution coordinates")
            bedroom_session_id = _required_string(payload["bedroom_session_id"], "bedroom_session_id")
            bedroom_turn_epoch = _positive_int(payload["bedroom_turn_epoch"], "bedroom_turn_epoch")
            room_id = None
            conversation_id = None
            fence = None
            actor_private_stance = None
        else:
            required = {"room_id", "conversation_id", "fence", "bedroom_session_id"}
            missing = required - set(payload)
            if missing or "bedroom_turn_epoch" in payload:
                raise ExecutionContractError("invalid group/private execution coordinates")
            fence_value = payload["fence"]
            if not isinstance(fence_value, Mapping):
                raise ExecutionContractError("fence must be an object")
            fence = ExecutionFence.from_dict(fence_value)
            room_id = _required_string(payload["room_id"], "room_id")
            conversation_id = _required_string(payload["conversation_id"], "conversation_id")
            if fence.room_id != room_id or fence.conversation_id != conversation_id:
                raise ExecutionContractError("fence room/conversation mismatch")
            if payload["bedroom_session_id"] is not None:
                raise ExecutionContractError("bedroom_session_id is allowed only for bedroom mode")
            bedroom_session_id = None
            bedroom_turn_epoch = None
            stance = payload.get("actor_private_stance")
            if stance is not None:
                if not isinstance(stance, str) or len(stance) > 2000:
                    raise ExecutionContractError("actor_private_stance must be bounded text or null")
            actor_private_stance = stance
        binding_revision = payload["binding_revision"]
        if binding_revision is not None:
            binding_revision = _positive_int(binding_revision, "binding_revision")
        return cls(
            contract_version=CONTRACT_VERSION,
            execution_kind=kind,
            actor_id=_required_string(payload["actor_id"], "actor_id"),
            room_id=room_id,
            conversation_id=conversation_id,
            current_event_id=_positive_int(payload["current_event_id"], "current_event_id"),
            generation_request_id=_required_string(
                payload["generation_request_id"], "generation_request_id"
            ),
            execution_mode=mode,
            fence=fence,
            bedroom_session_id=bedroom_session_id,
            bedroom_turn_epoch=bedroom_turn_epoch,
            actor_private_stance=actor_private_stance,
            binding_revision=binding_revision,
        )


def _nullable_usage_int(value: Any, field: str) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ExecutionContractError(f"{field} must be a non-negative integer or null")
    return value


@dataclass(frozen=True, slots=True)
class ProviderUsage:
    input_tokens: int | None
    output_tokens: int | None
    cache_creation_input_tokens: int | None
    cache_read_input_tokens: int | None
    cached_tokens: int | None

    @classmethod
    def from_provider_values(
        cls,
        *,
        input_tokens: Any = None,
        output_tokens: Any = None,
        cache_creation_input_tokens: Any = None,
        cache_read_input_tokens: Any = None,
        cached_tokens: Any = None,
    ) -> "ProviderUsage":
        return cls(
            input_tokens=_nullable_usage_int(input_tokens, "input_tokens"),
            output_tokens=_nullable_usage_int(output_tokens, "output_tokens"),
            cache_creation_input_tokens=_nullable_usage_int(
                cache_creation_input_tokens, "cache_creation_input_tokens"
            ),
            cache_read_input_tokens=_nullable_usage_int(
                cache_read_input_tokens, "cache_read_input_tokens"
            ),
            cached_tokens=_nullable_usage_int(cached_tokens, "cached_tokens"),
        )
