from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from model_execution import ContextBundle
from model_execution_contracts import CONTRACT_VERSION, GatewayExecutionRequest
from model_execution_contracts import ProviderUsage
from model_usage_store import build_cache_namespace


@dataclass(frozen=True, slots=True)
class FrozenProbeInput:
    actor_id: str
    conversation_id: str
    profile_id: str
    static_prefix_hash: str
    history_hash: str
    dynamic_tail_hash: str


@dataclass(frozen=True, slots=True)
class ProbeResult:
    status: str
    first: ProviderUsage
    second: ProviderUsage


class ProbeRunner(Protocol):
    async def run_once(self, frozen_input: FrozenProbeInput) -> ProviderUsage: ...


class CacheAcceptanceProbe:
    async def run(
        self, frozen_input: FrozenProbeInput, runner: ProbeRunner
    ) -> ProbeResult:
        empty = ProviderUsage.from_provider_values()
        try:
            first = await runner.run_once(frozen_input)
            second = await runner.run_once(frozen_input)
        except Exception:
            return ProbeResult(status="failed", first=empty, second=empty)
        read = second.cache_read_input_tokens
        cached = second.cached_tokens
        verified = (read is not None and read > 0) or (
            cached is not None and cached > 0
        )
        return ProbeResult(
            status="verified" if verified else "unverified",
            first=first,
            second=second,
        )


class GatewayCacheProbeService:
    """Runs the only paid cache test: an explicit, frozen two-send probe.

    Creating or editing a Model Profile never calls this service.  Callers must
    expose it behind an authenticated management action so there is no idle
    prompt-cache keepalive or surprise provider spend.
    """

    def __init__(self, *, profiles, provider_runner) -> None:
        self.profiles = profiles
        self.provider_runner = provider_runner

    @staticmethod
    def _request(*, actor_id: str, room_id: str, conversation_id: str, profile_id: str):
        return GatewayExecutionRequest.from_dict(
            {
                "contract_version": CONTRACT_VERSION,
                "execution_kind": "full",
                "actor_id": actor_id,
                "room_id": room_id,
                "conversation_id": conversation_id,
                "current_event_id": 1,
                "generation_request_id": f"cache-probe:{profile_id}:{actor_id}",
                "execution_mode": "private" if room_id != "room_group_home" else "group",
                "fence": {
                    "room_id": room_id,
                    "conversation_id": conversation_id,
                    "burst_id": "cache-probe-burst-v1",
                    "trigger_event_id": 1,
                    "fence_epoch": 1,
                    "lease_epoch": 1,
                    "orchestrator_instance": "gateway-cache-probe-v1",
                },
                "bedroom_session_id": None,
                "binding_revision": None,
            }
        )

    @staticmethod
    def _context() -> ContextBundle:
        # Deliberately large, deterministic stable prefix. Some Anthropic model
        # families only cache once the prefix passes their minimum token size.
        anchor = "gateway-cache-probe-static-anchor-v1 " * 1400
        return ContextBundle(
            static_system=(
                "gateway-cache-probe-runtime-v1\n" + anchor,
                "gateway-cache-probe-actor-v1",
                "gateway-cache-probe-room-policy-v1",
            ),
            stable_summary="gateway-cache-probe-summary-v1",
            stable_history=("gateway-cache-probe-history-v1",),
            dynamic_tail=("cache-probe-dynamic-tail-v1",),
            actor_prompt_version="cache-probe.actor.v1",
            runtime_kernel_version="cache-probe.runtime.v1",
            room_policy_version="cache-probe.room.v1",
            tool_schema_hash="cache-probe.tools.v1",
        )

    async def _once(self, *, profile, request, context, cache_namespace) -> ProviderUsage:
        usage = ProviderUsage.from_provider_values()
        final_seen = False
        stream = self.provider_runner.run(
            profile=profile,
            request=request,
            context=context,
            cache_namespace=cache_namespace,
            max_output_tokens=32,
        )
        try:
            async for chunk in stream:
                if chunk.event == "final":
                    final_seen = True
                elif chunk.event == "usage":
                    candidate = chunk.data.get("usage")
                    if isinstance(candidate, ProviderUsage):
                        usage = candidate
            if not final_seen:
                raise RuntimeError("cache probe provider stream omitted final")
            return usage
        finally:
            close = getattr(stream, "aclose", None)
            if close is not None:
                await close()

    async def run(
        self,
        *,
        profile_id: str,
        actor_id: str,
        room_id: str,
        conversation_id: str,
    ) -> ProbeResult:
        if actor_id not in {"jiao", "laoke"}:
            raise ValueError("cache probe actor is invalid")
        profile = await self.profiles.get_profile(profile_id)
        request = self._request(
            actor_id=actor_id,
            room_id=room_id,
            conversation_id=conversation_id,
            profile_id=profile_id,
        )
        context = self._context()
        namespace = build_cache_namespace(
            actor_id=actor_id,
            conversation_id=conversation_id,
            profile_id=profile.profile_id,
            profile_revision=profile.revision,
            execution_mode=request.execution_mode,
            actor_prompt_version=context.actor_prompt_version,
            runtime_kernel_version=context.runtime_kernel_version,
            room_policy_version=context.room_policy_version,
            tool_schema_hash=context.tool_schema_hash,
            cache_strategy_version=profile.cache_strategy,
        )

        frozen = FrozenProbeInput(
            actor_id=actor_id,
            conversation_id=conversation_id,
            profile_id=profile_id,
            static_prefix_hash=namespace,
            history_hash="cache-probe-history-v1",
            dynamic_tail_hash="cache-probe-dynamic-tail-v1",
        )

        class _FrozenRunner:
            async def run_once(inner_self, ignored):
                if ignored != frozen:
                    raise RuntimeError("cache probe input changed between sends")
                return await self._once(
                    profile=profile,
                    request=request,
                    context=context,
                    cache_namespace=namespace,
                )

        result = await CacheAcceptanceProbe().run(frozen, _FrozenRunner())
        if profile.cache_strategy == "no_prompt_cache_v1" and result.status != "failed":
            result = ProbeResult("not_applicable", result.first, result.second)
            await self.profiles.set_test_status(profile_id, "passed")
        elif result.status == "verified":
            await self.profiles.set_test_status(profile_id, "passed")
        else:
            await self.profiles.set_test_status(profile_id, "unverified")
        recorder = getattr(self.profiles, "record_probe_result", None)
        if recorder is not None:
            stored_status = "verified" if result.status == "not_applicable" else result.status
            await recorder(
                profile_id=profile.profile_id,
                profile_revision=profile.revision,
                probe_kind=(
                    "route_generation"
                    if profile.cache_strategy == "no_prompt_cache_v1"
                    else "frozen_double_send_cache"
                ),
                status=stored_status,
                observed_capabilities={
                    "cache_strategy": profile.cache_strategy,
                    "requested_cache_ttl": profile.requested_cache_ttl,
                    "cache_read_input_tokens": result.second.cache_read_input_tokens,
                    "cached_tokens": result.second.cached_tokens,
                },
            )
        return result
