import asyncio

import pytest

from model_execution import (
    ContextBundle,
    GatewayModelExecutionService,
    ProviderChunk,
    ProviderRunUnavailable,
)
from model_execution_contracts import GatewayExecutionRequest, ProviderUsage
from model_profile_store import InMemoryModelProfileStore
from model_profiles import ModelProfile
from model_usage_store import InMemoryModelUsageStore


def _profile(profile_id: str) -> ModelProfile:
    return ModelProfile.from_dict(
        {
            "profile_id": profile_id,
            "display_name": profile_id,
            "enabled": True,
            "test_status": "passed",
            "provider": "fake",
            "protocol": "anthropic_messages",
            "base_url": "https://example.invalid",
            "route_id": f"route-{profile_id}",
            "model": f"model-{profile_id}",
            "adapter_version": "fake.v1",
            "credential_ref": f"env:{profile_id.upper()}_KEY",
            "headers": {"x-api-key": "${credential}"},
            "capabilities": {
                "streaming": True,
                "structured_output": False,
                "tools": False,
                "reasoning_controls": False,
                "cache_strategies": ["anthropic_prefix_anchored_v1"],
                "cache_ttls": ["5m"],
                "usage_fields": [
                    "input_tokens",
                    "output_tokens",
                    "cache_creation_input_tokens",
                    "cache_read_input_tokens",
                ],
            },
            "cache_strategy": "anthropic_prefix_anchored_v1",
            "requested_cache_ttl": "5m",
            "revision": 1,
        }
    )


def _request(*, binding_revision=1):
    return GatewayExecutionRequest.from_dict(
        {
            "contract_version": "gateway-model-execution.v1.0",
            "execution_kind": "full",
            "actor_id": "jiao",
            "room_id": "room_group_home",
            "conversation_id": "conversation-1",
            "current_event_id": 101,
            "generation_request_id": "generation-1",
            "execution_mode": "group",
            "fence": {
                "room_id": "room_group_home",
                "conversation_id": "conversation-1",
                "burst_id": "burst-1",
                "trigger_event_id": 101,
                "fence_epoch": 1,
                "lease_epoch": 1,
                "orchestrator_instance": "orch-1",
            },
            "bedroom_session_id": None,
            "binding_revision": binding_revision,
        }
    )


class _ContextBuilder:
    def __init__(self):
        self.requests = []

    async def build(self, request):
        self.requests.append(request)
        return ContextBundle(
            static_system=("kernel", "actor", "room"),
            stable_summary="summary",
            stable_history=("history",),
            dynamic_tail=("memory", "current"),
            actor_prompt_version="jiao.v1",
            runtime_kernel_version="kernel.v1",
            room_policy_version="group.v1",
            tool_schema_hash="tools.none",
        )


class _Runner:
    def __init__(self, *, fail_profiles=()):
        self.fail_profiles = set(fail_profiles)
        self.calls = []
        self.cancelled = False

    async def run(self, *, profile, request, context, cache_namespace):
        self.calls.append((profile.profile_id, request, context, cache_namespace))
        try:
            if profile.profile_id in self.fail_profiles:
                raise ProviderRunUnavailable("sanitized failure")
            yield ProviderChunk("delta", {"text": "hello"})
            yield ProviderChunk("final", {"text": "hello"})
            yield ProviderChunk(
                "usage",
                {
                    "usage": ProviderUsage.from_provider_values(
                        input_tokens=100,
                        output_tokens=10,
                        cache_creation_input_tokens=80,
                        cache_read_input_tokens=20,
                    ),
                    "observed_cache_support": "verified",
                },
            )
        finally:
            self.cancelled = True


async def _service(*, fail_profiles=(), fallbacks=()):
    profiles = InMemoryModelProfileStore()
    for profile_id in ("primary", *fallbacks):
        await profiles.put_profile(_profile(profile_id))
    await profiles.set_actor_default("jiao", "primary")
    if fallbacks:
        await profiles.set_approved_fallbacks("jiao", tuple(fallbacks))
    context = _ContextBuilder()
    runner = _Runner(fail_profiles=fail_profiles)
    usage = InMemoryModelUsageStore()
    return (
        GatewayModelExecutionService(
            profiles=profiles,
            context_builder=context,
            provider_runner=runner,
            usage_store=usage,
        ),
        context,
        runner,
        usage,
    )


@pytest.mark.anyio
async def test_gateway_resolves_profile_without_orchestrator_model_input():
    service, _, runner, _ = await _service()
    events = [event async for event in service.stream(_request())]

    assert events[0].event == "profile"
    assert events[0].data["profile_id"] == "primary"
    assert runner.calls[0][0] == "primary"
    assert all("api_key" not in str(event.data).lower() for event in events)


@pytest.mark.anyio
async def test_gateway_fetches_relay_facts_and_builds_actor_pack_once():
    service, context, _, _ = await _service()
    _ = [event async for event in service.stream(_request())]
    assert context.requests == [_request()]


@pytest.mark.anyio
async def test_profile_fallback_uses_only_explicit_allowlist():
    service, _, runner, _ = await _service(
        fail_profiles=("primary",), fallbacks=("approved",)
    )
    events = [event async for event in service.stream(_request(binding_revision=2))]

    assert [call[0] for call in runner.calls] == ["primary", "approved"]
    profiles = [event.data for event in events if event.event == "profile"]
    assert profiles[-1]["profile_id"] == "approved"
    assert profiles[-1]["fallback_used"] is True
    assert profiles[-1]["fallback_from_profile_id"] == "primary"


class _BlockingRunner:
    def __init__(self):
        self.cancelled = asyncio.Event()

    async def run(self, **kwargs):
        try:
            yield ProviderChunk("delta", {"text": "partial"})
            await asyncio.Event().wait()
        finally:
            self.cancelled.set()


@pytest.mark.anyio
async def test_cancelled_execution_never_yields_accepted_final():
    profiles = InMemoryModelProfileStore()
    await profiles.put_profile(_profile("primary"))
    await profiles.set_actor_default("jiao", "primary")
    runner = _BlockingRunner()
    usage = InMemoryModelUsageStore()
    service = GatewayModelExecutionService(
        profiles=profiles,
        context_builder=_ContextBuilder(),
        provider_runner=runner,
        usage_store=usage,
    )
    stream = service.stream(_request())
    assert (await anext(stream)).event == "profile"
    assert (await anext(stream)).event == "delta"
    await stream.aclose()

    await asyncio.wait_for(runner.cancelled.wait(), timeout=1)
    assert await usage.list_receipts() == ()


@pytest.mark.anyio
async def test_execution_receipt_records_actual_fallback_and_usage():
    service, _, _, usage_store = await _service(
        fail_profiles=("primary",), fallbacks=("approved",)
    )
    _ = [event async for event in service.stream(_request(binding_revision=2))]
    receipts = await usage_store.list_receipts()

    assert len(receipts) == 1
    receipt = receipts[0]
    assert receipt.profile_id == "approved"
    assert receipt.fallback_used is True
    assert receipt.fallback_from_profile_id == "primary"
    assert receipt.usage.cache_read_input_tokens == 20


@pytest.mark.anyio
async def test_binding_revision_mismatch_rejects_before_provider_call():
    service, _, runner, _ = await _service()
    request = GatewayExecutionRequest.from_dict(
        {**_request_to_dict(_request()), "binding_revision": 99}
    )
    with pytest.raises(ValueError, match="binding revision"):
        _ = [event async for event in service.stream(request)]
    assert runner.calls == []


def _request_to_dict(request):
    return {
        "contract_version": request.contract_version,
        "execution_kind": request.execution_kind,
        "actor_id": request.actor_id,
        "room_id": request.room_id,
        "conversation_id": request.conversation_id,
        "current_event_id": request.current_event_id,
        "generation_request_id": request.generation_request_id,
        "execution_mode": request.execution_mode,
        "fence": {
            "room_id": request.fence.room_id,
            "conversation_id": request.fence.conversation_id,
            "burst_id": request.fence.burst_id,
            "trigger_event_id": request.fence.trigger_event_id,
            "fence_epoch": request.fence.fence_epoch,
            "lease_epoch": request.fence.lease_epoch,
            "orchestrator_instance": request.fence.orchestrator_instance,
        },
        "bedroom_session_id": request.bedroom_session_id,
        "binding_revision": request.binding_revision,
    }
