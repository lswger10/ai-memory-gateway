from __future__ import annotations

import asyncio
import hashlib
import json
import uuid
import anyio
from dataclasses import dataclass, replace
from datetime import datetime, timezone

from model_execution_contracts import ProviderUsage


class UsageStoreConflict(ValueError):
    pass


class UsageRecordingError(RuntimeError):
    """Accounting failure must never be treated as provider fallback permission."""


@dataclass(frozen=True, slots=True)
class ExecutionReceiptDraft:
    generation_request_id: str
    actor_id: str
    room_id: str
    conversation_id: str
    profile_id: str
    profile_revision: int
    provider: str
    protocol: str
    route_id: str
    model: str
    adapter_version: str
    cache_strategy: str
    requested_cache_ttl: str | None
    observed_cache_support: str
    fallback_used: bool
    fallback_from_profile_id: str | None
    usage: ProviderUsage
    status: str
    stable_prefix_hash: str | None = None
    prompt_cache_key: str | None = None
    runtime_kernel_version: str | None = None
    persona_version: str | None = None
    room_policy_version: str | None = None
    tool_schema_hash: str | None = None
    summary_version: int | None = None
    compressed_up_to_event_id: int | None = None
    provider_usage_received: bool = False
    execution_purpose: str = "generation"
    receipt_id: str | None = None


@dataclass(frozen=True, slots=True)
class ExecutionReceipt:
    receipt_id: str
    generation_request_id: str
    actor_id: str
    room_id: str
    conversation_id: str
    profile_id: str
    profile_revision: int
    provider: str
    protocol: str
    route_id: str
    model: str
    adapter_version: str
    cache_strategy: str
    requested_cache_ttl: str | None
    observed_cache_support: str
    fallback_used: bool
    fallback_from_profile_id: str | None
    usage: ProviderUsage
    status: str
    stable_prefix_hash: str | None = None
    prompt_cache_key: str | None = None
    runtime_kernel_version: str | None = None
    persona_version: str | None = None
    room_policy_version: str | None = None
    tool_schema_hash: str | None = None
    summary_version: int | None = None
    compressed_up_to_event_id: int | None = None
    provider_usage_received: bool = False
    execution_purpose: str = "generation"
    created_at: str | None = None


class InMemoryModelUsageStore:
    def __init__(self) -> None:
        self._receipts: dict[str, tuple[ExecutionReceiptDraft, ExecutionReceipt]] = {}
        self._lock = asyncio.Lock()

    async def record(self, draft: ExecutionReceiptDraft) -> ExecutionReceipt:
        async with self._lock:
            receipt_id = draft.receipt_id or str(uuid.uuid5(uuid.NAMESPACE_URL, draft.generation_request_id))
            existing = self._receipts.get(receipt_id)
            if existing is not None:
                existing_draft, receipt = existing
                if existing_draft != draft:
                    raise UsageStoreConflict(
                        "receipt_id already has different execution provenance"
                    )
                return receipt
            receipt = ExecutionReceipt(
                receipt_id=receipt_id,
                generation_request_id=draft.generation_request_id,
                actor_id=draft.actor_id,
                room_id=draft.room_id,
                conversation_id=draft.conversation_id,
                profile_id=draft.profile_id,
                profile_revision=draft.profile_revision,
                provider=draft.provider,
                protocol=draft.protocol,
                route_id=draft.route_id,
                model=draft.model,
                adapter_version=draft.adapter_version,
                cache_strategy=draft.cache_strategy,
                requested_cache_ttl=draft.requested_cache_ttl,
                observed_cache_support=draft.observed_cache_support,
                fallback_used=draft.fallback_used,
                fallback_from_profile_id=draft.fallback_from_profile_id,
                usage=draft.usage,
                status=draft.status,
                stable_prefix_hash=draft.stable_prefix_hash,
                prompt_cache_key=draft.prompt_cache_key,
                runtime_kernel_version=draft.runtime_kernel_version,
                persona_version=draft.persona_version,
                room_policy_version=draft.room_policy_version,
                tool_schema_hash=draft.tool_schema_hash,
                summary_version=draft.summary_version,
                compressed_up_to_event_id=draft.compressed_up_to_event_id,
                provider_usage_received=draft.provider_usage_received,
                execution_purpose=draft.execution_purpose,
                created_at=datetime.now(timezone.utc).isoformat(),
            )
            self._receipts[receipt_id] = (draft, receipt)
            return receipt

    async def list_receipts(self, *, limit: int = 200) -> tuple[ExecutionReceipt, ...]:
        async with self._lock:
            values = tuple(item[1] for item in self._receipts.values())
            return tuple(reversed(values[-limit:]))


def execution_receipt_draft(*, profile, generation_request_id, actor_id, room_id,
                            conversation_id, context, cache_namespace,
                            fallback_from_profile_id=None, execution_purpose="generation"):
    """Shared provenance for generation, Pin and explicit cache-probe attempts."""
    return ExecutionReceiptDraft(
        generation_request_id=generation_request_id, actor_id=actor_id,
        room_id=room_id, conversation_id=conversation_id,
        profile_id=profile.profile_id, profile_revision=profile.revision,
        provider=profile.provider, protocol=profile.protocol, route_id=profile.route_id,
        model=profile.model, adapter_version=profile.adapter_version,
        cache_strategy=profile.cache_strategy, requested_cache_ttl=profile.requested_cache_ttl,
        observed_cache_support="unverified", fallback_used=fallback_from_profile_id is not None,
        fallback_from_profile_id=fallback_from_profile_id,
        usage=ProviderUsage.from_provider_values(), status="failed",
        stable_prefix_hash=context.stable_prefix_hash or build_stable_prefix_hash(
            static_system=context.static_system, stable_summary=context.stable_summary,
            stable_history=context.stable_history),
        prompt_cache_key=cache_namespace if profile.cache_strategy == "openai_stable_prefix_v1" else None,
        runtime_kernel_version=context.runtime_kernel_version,
        persona_version=context.actor_prompt_version, room_policy_version=context.room_policy_version,
        tool_schema_hash=context.tool_schema_hash, summary_version=context.summary_version or 1,
        compressed_up_to_event_id=context.compressed_up_to_event_id or 0,
        execution_purpose=execution_purpose,
    )


async def record_provider_attempt(store, draft, attempt_id, usage, status,
                                  provider_usage_received, observed_cache_support):
    try:
        # ASGI uses level cancellation: shield only the bounded accounting write.
        with anyio.fail_after(5, shield=True):
            return await store.record(replace(draft, receipt_id=attempt_id, usage=usage,
                status=status, provider_usage_received=provider_usage_received,
                observed_cache_support=observed_cache_support))
    except Exception as exc:
        raise UsageRecordingError("provider attempt receipt could not be persisted") from exc


def build_cache_namespace(
    *,
    actor_id: str,
    conversation_id: str,
    profile_id: str,
    profile_revision: int,
    execution_mode: str,
    actor_prompt_version: str,
    runtime_kernel_version: str,
    room_policy_version: str,
    tool_schema_hash: str,
    cache_strategy_version: str,
) -> str:
    payload = {
        "actor_id": actor_id,
        "conversation_id": conversation_id,
        "profile_id": profile_id,
        "profile_revision": profile_revision,
        "execution_mode": execution_mode,
        "actor_prompt_version": actor_prompt_version,
        "runtime_kernel_version": runtime_kernel_version,
        "room_policy_version": room_policy_version,
        "tool_schema_hash": tool_schema_hash,
        "cache_strategy_version": cache_strategy_version,
    }
    for field, value in payload.items():
        if field == "profile_revision":
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError("profile_revision must be positive")
        elif not isinstance(value, str) or not value:
            raise ValueError(f"{field} must be a non-empty string")
    canonical = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def build_stable_prefix_hash(
    *,
    static_system: tuple[str, ...],
    stable_summary: str,
    stable_history: tuple[str, ...],
) -> str:
    canonical = json.dumps(
        {
            "static_system": static_system,
            "stable_summary": stable_summary,
            "stable_history": stable_history,
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()
