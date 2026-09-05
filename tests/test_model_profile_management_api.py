from fastapi.testclient import TestClient

import main
from model_profiles import ModelProfile
from cache_probe import ProbeResult
from model_execution_contracts import ProviderUsage
import asyncio
from model_profile_store import InMemoryModelProfileStore


def _profile_payload(**overrides):
    payload = {
        "profile_id": "ofox-claude",
        "display_name": "OFOX Claude",
        "enabled": True,
        "test_status": "passed",
        "provider": "ofox",
        "protocol": "anthropic_messages_compatible",
        "base_url": "https://api.ofox.ai/anthropic",
        "route_id": "ofox-anthropic",
        "model": "anthropic/claude-opus-4.6",
        "adapter_version": "anthropic.v1",
        "credential_ref": "env:OFOX_API_KEY",
        "headers": {"x-provider-key": "env:OFOX_EXTRA_HEADER"},
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
    payload.update(overrides)
    return payload


class _ProfileStore:
    def __init__(self):
        self.saved = None

    async def put_profile(self, profile):
        self.saved = profile
        return profile

    async def list_profiles(self):
        return (self.saved,) if self.saved else ()


def test_management_api_redacts_credentials_and_cannot_self_assert_passed(monkeypatch):
    store = _ProfileStore()

    async def runtime():
        main._model_profile_store = store
        return object()

    monkeypatch.setenv("MODEL_PROFILE_MANAGEMENT_ENABLED", "true")
    monkeypatch.setattr(main, "_get_model_execution_service", runtime)
    response = TestClient(main.app).put("/api/model-profiles", json=_profile_payload())

    assert response.status_code == 200
    assert store.saved.test_status == "unverified"
    body = response.json()
    assert body["test_status"] == "unverified"
    assert body["credential_configured"] is False
    assert "credential_ref" not in body
    assert "headers" not in body
    assert body["header_names"] == ["x-provider-key"]
    assert "OFOX_API_KEY" not in response.text
    assert "OFOX_EXTRA_HEADER" not in response.text


def test_safe_profile_never_serializes_secret_values(monkeypatch):
    monkeypatch.setenv("OFOX_API_KEY", "secret-api-value")
    profile = ModelProfile.from_dict(_profile_payload(test_status="unverified"))
    rendered = main._safe_profile(profile)
    text = str(rendered)

    assert rendered["credential_configured"] is True
    assert "secret-api-value" not in text
    assert "OFOX_API_KEY" not in text
    assert "OFOX_EXTRA_HEADER" not in text


def test_cache_probe_requires_explicit_charge_confirmation(monkeypatch):
    class ProbeService:
        calls = []

        async def run(self, **kwargs):
            self.calls.append(kwargs)
            usage = ProviderUsage.from_provider_values(
                input_tokens=100,
                cache_read_input_tokens=80,
            )
            return ProbeResult("verified", usage, usage)

    service = ProbeService()
    monkeypatch.setenv("MODEL_PROFILE_MANAGEMENT_ENABLED", "true")
    monkeypatch.setattr(main, "_cache_probe_service", service)
    client = TestClient(main.app)
    body = {
        "profile_id": "profile-1",
        "actor_id": "jiao",
        "room_id": "room_weiwei_jiao",
        "conversation_id": "canonical-conversation-1",
    }

    rejected = client.post("/api/cache-probes", json=body)
    accepted = client.post(
        "/api/cache-probes",
        json={**body, "confirm_provider_charges": True},
    )

    assert rejected.status_code == 409
    assert service.calls == [body]
    assert accepted.status_code == 200
    assert accepted.json()["status"] == "verified"
    assert accepted.json()["second"]["cache_read_input_tokens"] == 80


def _client_with_store(monkeypatch, store):
    async def runtime():
        return object()
    monkeypatch.setenv("MODEL_PROFILE_MANAGEMENT_ENABLED", "true")
    monkeypatch.setattr(main, "_model_profile_store", store)
    monkeypatch.setattr(main, "_get_model_execution_service", runtime)
    return TestClient(main.app)


def test_profile_edit_preserves_hidden_credentials_and_rejects_stale_save(monkeypatch):
    store = InMemoryModelProfileStore()
    original = ModelProfile.from_dict(_profile_payload())
    asyncio.run(store.put_profile(original))
    client = _client_with_store(monkeypatch, store)
    edit = _profile_payload(display_name="Edited", revision=2)
    del edit["credential_ref"]
    del edit["headers"]
    saved = client.put("/api/model-profiles", json=edit)
    assert saved.status_code == 200
    current = asyncio.run(store.get_profile(original.profile_id))
    assert current.credential_ref == original.credential_ref
    assert current.headers == original.headers
    assert current.test_status == "unverified"
    stale = client.put("/api/model-profiles", json={**edit, "model": "stale-model"})
    assert stale.status_code == 409
    assert asyncio.run(store.get_profile(original.profile_id)) == current


def test_binding_save_is_one_atomic_request_and_reads_actual_actor_default(monkeypatch):
    store = InMemoryModelProfileStore()
    async def seed():
        for name in ("first", "second"):
            await store.put_profile(ModelProfile.from_dict(_profile_payload(profile_id=name)))
        await store.set_actor_default("jiao", "first")
        await store.set_room_override("room_weiwei_jiao", "jiao", "second")
        await store.set_room_override("room_weiwei_jiao", "jiao", "second", expected_revision=1)
    asyncio.run(seed())
    client = _client_with_store(monkeypatch, store)
    snapshot = client.get("/api/model-bindings").json()
    assert snapshot["actor_defaults"]["jiao"] == {
        "profile_id": "first", "approved_fallback_profile_ids": [], "revision": 1}
    failed = client.put("/api/model-bindings", json={
        "action": "save_actor_binding", "actor_id": "jiao", "profile_id": "second",
        "profile_ids": ["missing"], "expected_revision": 1})
    assert failed.status_code == 409
    assert client.get("/api/model-bindings").json()["actor_defaults"] == snapshot["actor_defaults"]
    accepted = client.put("/api/model-bindings", json={
        "action": "save_actor_binding", "actor_id": "jiao", "profile_id": "second",
        "profile_ids": ["first"], "expected_revision": 1})
    assert accepted.status_code == 200
    assert accepted.json()["revision"] == 2


def test_invalid_profile_returns_safe_field_errors_before_any_provider_call(monkeypatch):
    client = _client_with_store(monkeypatch, InMemoryModelProfileStore())
    response = client.put("/api/model-profiles", json=_profile_payload(base_url="", model=""))
    assert response.status_code == 422
    assert response.json()["invalid_fields"] == ["base_url", "model"]
