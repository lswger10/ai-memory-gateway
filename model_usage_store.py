from __future__ import annotations

import asyncio
import hashlib
import json
import uuid
from dataclasses import dataclass

from model_execution_contracts import ProviderUsage


class UsageStoreConflict(ValueError):
    pass


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


class InMemoryModelUsageStore:
    def __init__(self) -> None:
        self._receipts: dict[str, tuple[ExecutionReceiptDraft, ExecutionReceipt]] = {}
        self._lock = asyncio.Lock()

    async def record(self, draft: ExecutionReceiptDraft) -> ExecutionReceipt:
        async with self._lock:
            existing = self._receipts.get(draft.generation_request_id)
            if existing is not None:
                existing_draft, receipt = existing
                if existing_draft != draft:
                    raise UsageStoreConflict(
                        "generation_request_id already has different execution provenance"
                    )
                return receipt
            receipt = ExecutionReceipt(
                receipt_id=str(
                    uuid.uuid5(uuid.NAMESPACE_URL, draft.generation_request_id)
                ),
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
            )
            self._receipts[draft.generation_request_id] = (draft, receipt)
            return receipt

    async def list_receipts(self) -> tuple[ExecutionReceipt, ...]:
        async with self._lock:
            return tuple(item[1] for item in self._receipts.values())


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
