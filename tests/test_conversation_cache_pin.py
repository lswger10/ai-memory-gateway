from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest
from fastapi.testclient import TestClient

from conversation_cache_pin import CachePinService, InMemoryConversationCachePinStore
from model_execution import ContextBundle, ProviderChunk
from model_execution_contracts import ProviderUsage
from model_profile_store import InMemoryModelProfileStore
from model_profiles import ModelProfile
from model_usage_store import InMemoryModelUsageStore


NOW = datetime(2026, 9, 2, 12, 0, tzinfo=timezone.utc)


@pytest.mark.anyio
async def test_pin_api_separates_running_state_from_real_receipt_outcome():
    import main
    from dataclasses import replace
    from cache_dashboard import build_cache_usage_view
    service, _, _, _ = await _service()
    pin = await service.set_pin(room_id="room_weiwei_laoke", conversation_id="conversation-laoke",
        execution_mode="private", enabled=True)
    await service.run_due_once()
    pin = await service.get_pin(pin.pin_id)
    receipt = (await service.usage_store.list_receipts())[0]
    for read, creation, expected in ((118, 2, "HIT"), (0, 120, "OBSERVED_MISS"), (None, None, "UNOBSERVABLE")):
        observed = replace(receipt, usage=ProviderUsage.from_provider_values(
            cache_read_input_tokens=read, cache_creation_input_tokens=creation))
        view = main._cache_pin_public(pin, build_cache_usage_view((observed,)))
        assert view["actors"]["laoke"]["status"] == "active"
        assert view["actors"]["laoke"]["cache_outcome"] == expected
    other = replace(receipt, generation_request_id="cache-pin:private:other:laoke:test")
    assert main._cache_pin_public(pin, build_cache_usage_view((other,)))["actors"]["laoke"]["cache_outcome"] == "UNOBSERVABLE"


def _profile(profile_id: str, *, ttl: str | None = "1h") -> ModelProfile:
    strategy = "anthropic_prefix_anchored_v1" if ttl else "no_prompt_cache_v1"
    return ModelProfile.from_dict(
        {
            "profile_id": profile_id,
            "display_name": profile_id,
            "enabled": True,
            "test_status": "passed",
            "provider": "fake",
            "protocol": "anthropic_messages_compatible",
            "base_url": "https://example.invalid/anthropic",
            "route_id": f"route-{profile_id}",
            "model": "anthropic/claude-opus-4.6",
            "adapter_version": "gateway-anthropic-v1",
            "credential_ref": "env:TEST_ONLY_KEY",
            "headers": {"x-api-key": "${credential}"},
            "capabilities": {
                "streaming": True,
                "structured_output": False,
                "tools": False,
                "reasoning_controls": False,
                "cache_strategies": [strategy],
                "cache_ttls": [ttl] if ttl else [],
                "usage_fields": [
                    "input_tokens",
                    "output_tokens",
                    "cache_creation_input_tokens",
                    "cache_read_input_tokens",
                ],
                "input_modalities": ["text"],
            },
            "cache_strategy": strategy,
            "requested_cache_ttl": ttl,
            "revision": 1,
        }
    )


class _ContextBuilder:
    def __init__(self):
        self.calls = []

    async def build_cache_keepalive(self, **kwargs):
        self.calls.append(kwargs)
        return ContextBundle(
            static_system=("kernel", f"actor:{kwargs['actor_id']}", "room"),
            stable_summary="summary",
            stable_history=("accepted fact",),
            dynamic_tail=("cache continuity maintenance",),
            actor_prompt_version=f"{kwargs['actor_id']}.v2",
            runtime_kernel_version="kernel.v1",
            room_policy_version="room.v1",
            tool_schema_hash="tools.none.v1",
            cache_conversation_id=kwargs["cache_conversation_id"],
            stable_prefix_hash="stable-prefix",
            summary_version=2,
            compressed_up_to_event_id=9,
        )


class _Runner:
    def __init__(self):
        self.calls = []

    async def run(self, **kwargs):
        self.calls.append(kwargs)
        yield ProviderChunk("final", {"text": "ignored"})
        yield ProviderChunk(
            "usage",
            {
                "usage": ProviderUsage.from_provider_values(
                    input_tokens=120,
                    output_tokens=1,
                    cache_creation_input_tokens=0,
                    cache_read_input_tokens=118,
                ),
                "observed_cache_support": "verified",
                "provider_usage_received": True,
            },
        )


async def _service():
    profiles = InMemoryModelProfileStore()
    await profiles.put_profile(_profile("jiao-natural", ttl=None))
    await profiles.put_profile(_profile("laoke-1h"))
    await profiles.record_probe_result(
        profile_id="laoke-1h",
        profile_revision=1,
        probe_kind="frozen_double_send_cache",
        status="verified",
        observed_capabilities={"requested_cache_ttl": "1h"},
    )
    await profiles.set_actor_default("jiao", "jiao-natural")
    await profiles.set_actor_default("laoke", "laoke-1h")
    store = InMemoryConversationCachePinStore()
    builder = _ContextBuilder()
    runner = _Runner()
    usage = InMemoryModelUsageStore()
    service = CachePinService(
        store=store,
        profiles=profiles,
        context_builder=builder,
        provider_runner=runner,
        usage_store=usage,
        now=lambda: NOW,
        interval=timedelta(minutes=50),
    )
    return service, store, builder, runner


@pytest.mark.anyio
async def test_living_room_pin_runs_only_verified_1h_actor_namespace():
    service, _, builder, runner = await _service()
    pin = await service.set_pin(
        room_id="room_group_home",
        conversation_id="conversation-group",
        execution_mode="group",
        enabled=True,
    )

    result = await service.run_due_once()

    assert pin.enabled is True
    assert result.calls == 1
    assert [call["actor_id"] for call in builder.calls] == ["laoke"]
    assert runner.calls[0]["max_output_tokens"] == 1
    view = await service.get_pin(pin.pin_id)
    assert view.status == "active"
    assert view.actors["laoke"].cache_read_input_tokens == 118
    assert view.actors["laoke"].call_count == 1
    assert view.actors["jiao"].status == "paused"
    assert view.actors["jiao"].profile_id == "jiao-natural"


@pytest.mark.anyio
async def test_private_pin_survives_service_restart_and_respects_due_time():
    service, store, _, runner = await _service()
    pin = await service.set_pin(
        room_id="room_weiwei_laoke",
        conversation_id="conversation-laoke",
        execution_mode="private",
        enabled=True,
    )
    await service.run_due_once()

    restarted = CachePinService(
        store=store,
        profiles=service.profiles,
        context_builder=_ContextBuilder(),
        provider_runner=_Runner(),
        now=lambda: NOW + timedelta(minutes=10),
        interval=timedelta(minutes=50),
    )
    result = await restarted.run_due_once()

    assert result.calls == 0
    persisted = await restarted.get_pin(pin.pin_id)
    assert persisted.enabled is True
    assert persisted.actors["laoke"].next_keepalive_at == NOW + timedelta(minutes=50)
    assert len(runner.calls) == 1


@pytest.mark.anyio
async def test_edited_unverified_profile_pauses_pin_without_provider_call():
    from dataclasses import replace
    service, _, _, runner = await _service()
    pin = await service.set_pin(room_id="room_weiwei_laoke", conversation_id="conversation-laoke",
        execution_mode="private", enabled=True)
    await service.run_due_once()
    profile = await service.profiles.get_profile("laoke-1h")
    await service.profiles.put_profile(replace(profile, model="changed-model", revision=2))
    result = await service.run_due_once()
    assert result.calls == 0
    assert len(runner.calls) == 1
    assert (await service.get_pin(pin.pin_id)).actors["laoke"].status == "paused"


@pytest.mark.anyio
async def test_profile_without_verified_1h_keeps_pin_enabled_but_paused():
    service, _, builder, runner = await _service()
    pin = await service.set_pin(
        room_id="room_weiwei_jiao",
        conversation_id="conversation-jiao",
        execution_mode="private",
        enabled=True,
    )

    result = await service.run_due_once()
    view = await service.get_pin(pin.pin_id)

    assert result.calls == 0
    assert view.enabled is True
    assert view.status == "paused"
    assert view.actors["jiao"].status == "paused"
    assert builder.calls == []
    assert runner.calls == []


@pytest.mark.anyio
async def test_declared_one_hour_cache_without_verified_probe_stays_paused():
    profiles = InMemoryModelProfileStore()
    await profiles.put_profile(_profile("declared-only-1h"))
    await profiles.set_actor_default("laoke", "declared-only-1h")
    store = InMemoryConversationCachePinStore()
    builder = _ContextBuilder()
    runner = _Runner()
    service = CachePinService(
        store=store,
        profiles=profiles,
        context_builder=builder,
        provider_runner=runner,
        now=lambda: NOW,
    )
    await service.set_pin(
        room_id="room_weiwei_laoke",
        conversation_id="conversation-laoke",
        execution_mode="private",
        enabled=True,
    )

    result = await service.run_due_once()
    view = await service.get_pin("private:conversation-laoke")

    assert result.calls == 0
    assert view.enabled is True
    assert view.actors["laoke"].status == "paused"
    assert view.actors["laoke"].last_error == "profile_has_no_verified_1h_cache"
    assert builder.calls == []
    assert runner.calls == []


@pytest.mark.anyio
async def test_new_group_pin_immediately_exposes_both_actor_states():
    service, _, _, _ = await _service()
    pin = await service.set_pin(
        room_id="room_group_home",
        conversation_id="conversation-group",
        execution_mode="group",
        enabled=True,
    )

    assert set(pin.actors) == {"jiao", "laoke"}
    assert {state.status for state in pin.actors.values()} == {"pending"}


@pytest.mark.anyio
async def test_ending_bedroom_session_disables_its_pin():
    service, _, _, _ = await _service()
    pin = await service.set_pin(
        room_id="room_weiwei_laoke",
        conversation_id="conversation-laoke",
        execution_mode="bedroom",
        bedroom_session_id="bedroom-1",
        actor_id="laoke",
        enabled=True,
    )

    await service.end_bedroom("bedroom-1")

    disabled = await service.get_pin(pin.pin_id)
    assert disabled.enabled is False
    assert disabled.status == "off"


@pytest.mark.anyio
async def test_keepalive_never_publishes_or_mutates_conversation_history():
    service, _, builder, runner = await _service()
    await service.set_pin(
        room_id="room_weiwei_laoke",
        conversation_id="conversation-laoke",
        execution_mode="private",
        enabled=True,
    )

    await service.run_due_once()

    assert builder.calls[0]["cache_conversation_id"] == "conversation-laoke"
    request = runner.calls[0]["request"]
    assert request.execution_kind == "full"
    assert request.generation_request_id.startswith("cache-pin:")
    assert not hasattr(request, "fence")
    receipts = await service.usage_store.list_receipts()
    assert receipts[0].status == "succeeded"
    assert receipts[0].execution_purpose == "cache_keepalive"
    assert receipts[0].usage.cache_read_input_tokens == 118


def test_postgres_schema_persists_pin_and_per_actor_runtime_state():
    from database import MODEL_EXECUTION_MIGRATION_SQL

    assert "CREATE TABLE IF NOT EXISTS conversation_cache_pins" in MODEL_EXECUTION_MIGRATION_SQL
    assert "CREATE TABLE IF NOT EXISTS conversation_cache_pin_actor_state" in MODEL_EXECUTION_MIGRATION_SQL
    assert "PRIMARY KEY (pin_id, actor_id)" in MODEL_EXECUTION_MIGRATION_SQL


def test_management_api_exposes_conversation_pin_without_provider_secrets(monkeypatch):
    import main

    service, _, _, _ = _run(_service())

    async def get_service():
        return service

    monkeypatch.setattr(main, "MEMORY_ENABLED", True)
    monkeypatch.setenv("MODEL_PROFILE_MANAGEMENT_ENABLED", "true")
    monkeypatch.setattr(main, "_get_cache_pin_service", get_service)
    client = TestClient(main.app)
    response = client.put(
        "/api/cache-pins",
        json={
            "room_id": "room_weiwei_laoke",
            "conversation_id": "conversation-laoke",
            "execution_mode": "private",
            "enabled": True,
        },
    )

    assert response.status_code == 200
    assert response.json()["pin_id"] == "private:conversation-laoke"
    listing = client.get("/api/cache-pins")
    assert listing.status_code == 200
    assert listing.json()["pins"][0]["enabled"] is True
    assert "credential" not in listing.text.lower()


def test_bedroom_retention_success_is_not_reversed_by_pin_cleanup_failure(monkeypatch):
    import main

    class Retention:
        async def persist(self, payload):
            return {"accepted": True, "bedroom_session_id": payload["session"]["bedroom_session_id"]}

    class BrokenPins:
        async def end_bedroom(self, bedroom_session_id):
            raise RuntimeError("cache pin store unavailable")

    async def get_pins():
        return BrokenPins()

    monkeypatch.setattr(main, "_bedroom_enabled", lambda: True)
    monkeypatch.setattr(main, "_get_bedroom_retention_service", lambda: Retention())
    monkeypatch.setattr(main, "_get_cache_pin_service", get_pins)
    monkeypatch.setenv("BEDROOM_GATEWAY_SERVICE_KEY", "bedroom-key")
    client = TestClient(main.app)
    response = client.post(
        "/internal/bedroom/retention",
        headers={
            "Authorization": "Bearer bedroom-key",
            "X-Bedroom-Contract-Version": main.BEDROOM_CONTRACT_VERSION,
        },
        json={"session": {"bedroom_session_id": "bedroom-1"}},
    )

    assert response.status_code == 200
    assert response.json()["accepted"] is True


def _run(awaitable):
    import asyncio

    return asyncio.run(awaitable)
