from __future__ import annotations

from dataclasses import dataclass
import logging
from typing import Any, AsyncIterator, Protocol

from model_execution_contracts import GatewayExecutionRequest, ProviderUsage
from model_profile_store import InMemoryModelProfileStore
from model_usage_store import (
    execution_receipt_draft,
    record_provider_attempt,
    UsageRecordingError,
    InMemoryModelUsageStore,
    build_cache_namespace,
)
from provider_adapters import build_provider_provenance


class ProviderRunUnavailable(RuntimeError):
    """A sanitized, retryable provider-attempt failure."""


@dataclass(frozen=True, slots=True)
class ContextBundle:
    static_system: tuple[str, ...]
    stable_summary: str
    stable_history: tuple[str, ...]
    dynamic_tail: tuple[str, ...]
    actor_prompt_version: str
    runtime_kernel_version: str
    room_policy_version: str
    tool_schema_hash: str
    cache_conversation_id: str | None = None
    stable_prefix_hash: str | None = None
    summary_version: int | None = None
    compressed_up_to_event_id: int | None = None
    current_media_references: tuple[dict[str, Any], ...] = ()
    actor_memory_context: Any | None = None


@dataclass(frozen=True, slots=True)
class ProviderChunk:
    event: str
    data: dict[str, Any]


@dataclass(frozen=True, slots=True)
class ExecutionStreamEvent:
    event: str
    data: dict[str, Any]


class ContextBuilder(Protocol):
    async def resolve_coordinates(
        self, request: GatewayExecutionRequest
    ) -> tuple[str, str]: ...

    async def build(
        self,
        request: GatewayExecutionRequest,
        profile: Any,
        *,
        resolved_room_id: str,
        resolved_conversation_id: str,
    ) -> ContextBundle: ...


class ProviderRunner(Protocol):
    def run(
        self,
        *,
        profile: Any,
        request: GatewayExecutionRequest,
        context: ContextBundle,
        cache_namespace: str,
        on_attempt,
    ) -> AsyncIterator[ProviderChunk]: ...


class GatewayModelExecutionService:
    def __init__(
        self,
        *,
        profiles: InMemoryModelProfileStore,
        context_builder: ContextBuilder,
        provider_runner: ProviderRunner,
        usage_store: InMemoryModelUsageStore,
    ) -> None:
        self._profiles = profiles
        self._context_builder = context_builder
        self._provider_runner = provider_runner
        self._usage_store = usage_store

    async def stream(
        self, request: GatewayExecutionRequest
    ) -> AsyncIterator[ExecutionStreamEvent]:
        if hasattr(self._context_builder, "resolve_coordinates"):
            room_id, conversation_id = await self._context_builder.resolve_coordinates(request)
        else:
            if request.room_id is None or request.conversation_id is None:
                raise ValueError("execution coordinates are unresolved")
            room_id, conversation_id = request.room_id, request.conversation_id
        resolved = await self._profiles.resolve(request.actor_id, room_id)
        if (
            request.binding_revision is not None
            and request.binding_revision != resolved.binding_revision
        ):
            raise ValueError("binding revision changed before execution")
        attempts = (resolved.primary, *resolved.fallbacks)
        origin_profile_id = resolved.primary.profile_id
        last_unavailable: ProviderRunUnavailable | None = None

        for index, profile in enumerate(attempts):
            try:
                context = await self._context_builder.build(
                    request,
                    profile,
                    resolved_room_id=room_id,
                    resolved_conversation_id=conversation_id,
                )
            except TypeError:
                # Transitional support for injected deterministic test builders.
                context = await self._context_builder.build(request)
            fallback_used = index > 0
            fallback_from = origin_profile_id if fallback_used else None
            provenance = build_provider_provenance(
                profile,
                generation_request_id=request.generation_request_id,
                fallback_used=fallback_used,
                fallback_from_profile_id=fallback_from,
            )
            yield ExecutionStreamEvent("profile", provenance.to_dict())
            namespace = build_cache_namespace(
                actor_id=request.actor_id,
                conversation_id=context.cache_conversation_id or conversation_id,
                profile_id=profile.profile_id,
                profile_revision=profile.revision,
                execution_mode=request.execution_mode,
                actor_prompt_version=context.actor_prompt_version,
                runtime_kernel_version=context.runtime_kernel_version,
                room_policy_version=context.room_policy_version,
                tool_schema_hash=context.tool_schema_hash,
                cache_strategy_version=profile.cache_strategy,
            )
            usage = ProviderUsage.from_provider_values()
            provider_usage_received = False
            observed_cache_support = "unverified"
            final_seen = False
            final_data: dict[str, Any] | None = None
            receipt = None
            draft = execution_receipt_draft(
                profile=profile, generation_request_id=request.generation_request_id,
                actor_id=request.actor_id, room_id=room_id, conversation_id=conversation_id,
                context=context, cache_namespace=namespace, fallback_from_profile_id=fallback_from,
                execution_purpose="generation" if request.execution_kind == "full" else "generation_probe")
            async def on_attempt(attempt_id, usage, status, provider_usage_received, observed_cache_support):
                nonlocal receipt
                receipt = await record_provider_attempt(self._usage_store, draft, attempt_id,
                    usage, status, provider_usage_received, observed_cache_support)
            provider_stream = self._provider_runner.run(
                profile=profile,
                request=request,
                context=context,
                cache_namespace=namespace,
                on_attempt=on_attempt,
            )
            try:
                async for chunk in provider_stream:
                    if chunk.event == "usage":
                        candidate = chunk.data.get("usage")
                        if not isinstance(candidate, ProviderUsage):
                            raise ProviderRunUnavailable("provider usage shape is invalid")
                        usage = candidate
                        provider_usage_received = bool(
                            chunk.data.get("provider_usage_received", False)
                            or any(
                                value is not None
                                for value in (
                                    candidate.input_tokens,
                                    candidate.output_tokens,
                                    candidate.cache_creation_input_tokens,
                                    candidate.cache_read_input_tokens,
                                    candidate.cached_tokens,
                                )
                            )
                        )
                        observed_cache_support = str(
                            chunk.data.get("observed_cache_support", "unverified")
                        )
                        continue
                    if chunk.event == "final":
                        final_seen = True
                        final_data = dict(chunk.data)
                        continue
                    if chunk.event not in {"delta", "probe", "final"}:
                        raise ProviderRunUnavailable("provider emitted an unsupported event")
                    yield ExecutionStreamEvent(chunk.event, dict(chunk.data))
                if not final_seen:
                    raise ProviderRunUnavailable("provider stream ended without final")
            except ProviderRunUnavailable as exc:
                logging.getLogger(__name__).warning(
                    "Model execution unavailable generation=%s profile=%s reason=%s cause=%s",
                    request.generation_request_id, profile.profile_id, str(exc),
                    type(exc.__cause__).__name__ if exc.__cause__ else "none",
                )
                last_unavailable = exc
                continue
            finally:
                close = getattr(provider_stream, "aclose", None)
                if close is not None:
                    await close()

            if receipt is None:
                raise UsageRecordingError("provider completed without an attempt receipt")
            yield ExecutionStreamEvent("final", final_data or {})
            yield ExecutionStreamEvent(
                "usage",
                {
                    "input_tokens": usage.input_tokens,
                    "output_tokens": usage.output_tokens,
                    "cache_creation_input_tokens": usage.cache_creation_input_tokens,
                    "cache_read_input_tokens": usage.cache_read_input_tokens,
                    "cached_tokens": usage.cached_tokens,
                    "observed_cache_support": observed_cache_support,
                },
            )
            yield ExecutionStreamEvent(
                "done",
                {
                    "generation_request_id": request.generation_request_id,
                    "execution_receipt_id": receipt.receipt_id,
                },
            )
            return

        yield ExecutionStreamEvent(
            "unavailable",
            {
                "generation_request_id": request.generation_request_id,
                "reason_code": "generation_unavailable",
            },
        )
        if last_unavailable is None:
            raise ProviderRunUnavailable("no approved Model Profile attempt was available")
