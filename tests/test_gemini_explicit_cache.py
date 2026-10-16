import asyncio
import json
from dataclasses import replace
from datetime import datetime, timedelta, timezone

import pytest

from gemini_explicit_cache import GeminiExplicitCache, InMemoryGeminiCacheStore
from model_execution import ExplicitCacheUnavailable
from model_profile_store import InMemoryModelProfileStore
from model_usage_store import InMemoryModelUsageStore
from conversation_cache_pin import InMemoryConversationCachePinStore, ConversationCachePin
from gateway_provider_runner import GatewayProviderRunner
from test_gemini_native import native_profile, context
from test_gateway_provider_runner import request


def test_runtime_build_context_can_import_gateway(tmp_path):
    import os
    from pathlib import Path
    import shutil
    import subprocess
    import sys

    root = Path(__file__).resolve().parents[1]
    # The image uses a literal allowlist; test imports without the source checkout.
    for line in (root / ".dockerignore").read_text().splitlines():
        if line.startswith("!"):
            relative = line[1:]
            target = tmp_path / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(root / relative, target)
    result = subprocess.run(
        [sys.executable, "-I", "-B", "-c", "import sys; sys.path.insert(0, '.'); import main"],
        cwd=tmp_path, capture_output=True, text=True, timeout=30,
        env={**os.environ, "PYTHONPATH": ""},
    )
    assert result.returncode == 0, result.stderr


class CreateTransport:
    def __init__(self, now, error=False): self.now, self.error, self.calls = now, error, []
    async def request(self, **kwargs):
        self.calls.append(kwargs)
        await asyncio.sleep(0)
        if self.error: raise TimeoutError("do not expose body/key")
        class Response:
            status_code = 200
            def json(inner):
                return {"name": "cachedContents/synthetic", "model": "google/gemini-test",
                        "expireTime": (self.now + timedelta(hours=1)).isoformat(),
                        "usageMetadata": {"totalTokenCount": 5000}}
        return Response()


async def setup_cache(*, verified=True, error=False):
    now = datetime(2026, 10, 16, tzinfo=timezone.utc)
    profiles, pins, usage = InMemoryModelProfileStore(), InMemoryConversationCachePinStore(), InMemoryModelUsageStore()
    selected = native_profile()
    await profiles.put_profile(selected)
    if verified:
        await profiles.record_probe_result(profile_id=selected.profile_id, profile_revision=1,
            probe_kind="gemini_explicit_cache", status="verified", observed_capabilities={})
    await pins.upsert_pin(ConversationCachePin("group:conversation-1", "room_group_home", "conversation-1", "group", True))
    cache = GeminiExplicitCache(store=InMemoryGeminiCacheStore(), pins=pins, profiles=profiles, usage=usage, now=lambda: now)
    return cache, CreateTransport(now, error), selected


async def prepare(cache, transport, selected, ctx=None, req=None, namespace="namespace"):
    ctx, req = ctx or context(), req or request()
    body = GatewayProviderRunner()._render(selected, req, ctx, namespace).json_body
    return await cache.prepare(profile=selected, request=req, context=ctx, namespace=namespace,
        body=body, headers={"x-goog-api-key": "synthetic-only"}, transport=transport)


@pytest.mark.anyio
async def test_reuses_frozen_prefix_and_sends_new_history_only_in_tail():
    cache, transport, selected = await setup_cache()
    first = await prepare(cache, transport, selected)
    extra = json.dumps({"event_id": 2, "actor_id": "weiwei", "role": "human", "content": "new fact"})
    second = await prepare(cache, transport, selected, replace(context(), stable_history=context().stable_history + (extra,)))
    assert len(transport.calls) == 1
    assert first["cachedContent"] == second["cachedContent"]
    assert "systemInstruction" not in second and "tools" not in second
    assert "new fact" in json.dumps(second["contents"][0])
    assert "earlier" not in json.dumps(second)
    assert "current_event" not in json.dumps(transport.calls[0]["json_body"])
    assert transport.calls[0]["json_body"]["ttl"] == "3600s"
    receipts = await cache.usage.list_receipts()
    assert len(receipts) == 1 and receipts[0].execution_purpose == "cache_create"
    assert receipts[0].usage.cache_creation_input_tokens == 5000


@pytest.mark.anyio
async def test_creation_failure_survives_service_restart_and_does_not_fallback_or_retry():
    cache, transport, selected = await setup_cache(error=True)
    with pytest.raises(ExplicitCacheUnavailable): await prepare(cache, transport, selected)
    restarted = GeminiExplicitCache(store=cache.store, pins=cache.pins, profiles=cache.profiles, usage=cache.usage, now=cache.now)
    for namespace in ("namespace", "other-profile-namespace"):
        with pytest.raises(ExplicitCacheUnavailable): await prepare(restarted, transport, selected, namespace=namespace)
    assert len(transport.calls) == 1
    assert (await cache.store.list_resources())[0].status == "paused"


@pytest.mark.anyio
async def test_unfinished_creation_blocks_after_reconstruction_and_repaired_history_rebuilds():
    cache, transport, selected = await setup_cache()
    await prepare(cache, transport, selected)
    changed = replace(context(), stable_history=(json.dumps({"event_id": 1, "actor_id": "weiwei",
        "role": "human", "content": "repaired original fact"}),))
    await prepare(cache, transport, selected, changed)
    assert len(transport.calls) == 2
    row = (await cache.store.list_resources())[0]
    assert await cache.store.reserve(replace(row, status="creating", attempt_id="interrupted-attempt"), row.attempt_id)
    restarted = GeminiExplicitCache(store=cache.store, pins=cache.pins, profiles=cache.profiles, usage=cache.usage, now=cache.now)
    with pytest.raises(ExplicitCacheUnavailable, match="paused"):
        await prepare(restarted, transport, selected, changed)
    assert len(transport.calls) == 2


@pytest.mark.anyio
async def test_parallel_creators_only_one_post():
    cache, transport, selected = await setup_cache()
    results = await asyncio.gather(*(prepare(cache, transport, selected) for _ in range(3)), return_exceptions=True)
    assert len(transport.calls) == 1
    assert any(isinstance(item, dict) for item in results)


@pytest.mark.anyio
async def test_expiry_summary_or_history_repair_rebuild_on_demand_only():
    cache, transport, selected = await setup_cache()
    await prepare(cache, transport, selected)
    cache.now = lambda: transport.now + timedelta(hours=2)
    assert len(transport.calls) == 1
    transport.now += timedelta(hours=2)
    # Use a fixed clock, not a lambda referring to the mutable transport clock.
    current = transport.now
    cache.now = lambda: current
    await prepare(cache, transport, selected)
    await prepare(cache, transport, selected, replace(context(), stable_summary="revised", summary_version=2))
    assert len(transport.calls) == 3


@pytest.mark.anyio
async def test_unverified_pin_off_bedroom_and_actor_scope():
    cache, transport, selected = await setup_cache(verified=False)
    with pytest.raises(ExplicitCacheUnavailable, match="unverified"):
        await prepare(cache, transport, selected)
    assert not transport.calls
    pin = (await cache.pins.list_pins())[0]
    await cache.pins.upsert_pin(replace(pin, enabled=False))
    assert "cachedContent" not in await prepare(cache, transport, selected)
    assert "cachedContent" not in await prepare(cache, transport, selected, req=replace(request(), execution_mode="bedroom"))
    assert not transport.calls
    cache, transport, selected = await setup_cache()
    await prepare(cache, transport, selected)
    await prepare(cache, transport, selected, req=replace(request(), actor_id="laoke"))
    assert len(transport.calls) == 2


@pytest.mark.anyio
async def test_paused_unknown_attempt_requires_exact_explicit_resume_after_ttl():
    cache, transport, selected = await setup_cache(error=True)
    with pytest.raises(ExplicitCacheUnavailable): await prepare(cache, transport, selected)
    row = (await cache.store.list_resources())[0]
    assert not await cache.store.resume(row.cache_key, row.attempt_id, row.scope, cache.now())
    later = cache.now() + timedelta(hours=2)
    assert not await cache.store.resume(row.cache_key, "stale", row.scope, later)
    assert not await cache.store.resume(row.cache_key, row.attempt_id, ("laoke", *row.scope[1:]), later)
    assert await cache.store.resume(row.cache_key, row.attempt_id, row.scope, later)
    assert len(transport.calls) == 1


@pytest.mark.anyio
async def test_cached_generation_error_stops_model_fallback_and_pauses():
    from model_execution import GatewayModelExecutionService
    from test_model_execution_service import _profile
    from test_gateway_provider_runner import Resolver
    cache, transport, selected = await setup_cache()
    await cache.profiles.set_actor_default("jiao", selected.profile_id)
    await cache.profiles.put_profile(_profile("fallback"))
    await cache.profiles.set_approved_fallbacks("jiao", ("fallback",))
    class Builder:
        async def build(self, *args, **kwargs): return context()
    attempts = []
    async def fail(**kwargs):
        attempts.append(kwargs)
        raise TimeoutError("synthetic")
    transport.open_stream = fail
    runner = GatewayProviderRunner(explicit_cache=cache, transport=transport, credential_resolver=Resolver())
    service = GatewayModelExecutionService(profiles=cache.profiles, context_builder=Builder(), provider_runner=runner, usage_store=cache.usage)
    for _ in range(2):
        chunks = [item async for item in service.stream(request())]
        assert chunks[-1].event == "unavailable"
    assert len(attempts) == 1 and len(transport.calls) == 1
    assert (await cache.store.list_resources())[0].status == "paused"
    assert sorted(row.execution_purpose for row in await cache.usage.list_receipts()) == ["cache_create", "generation"]


@pytest.mark.anyio
@pytest.mark.parametrize("status", [401, 404])
async def test_cached_http_error_retains_status_without_response_body(status):
    import httpx
    from provider_transport import PooledHttpTransport
    from test_gateway_provider_runner import Resolver
    cache, creation, selected = await setup_cache()
    await prepare(cache, creation, selected)
    transport = PooledHttpTransport(client_factory=lambda **kwargs: httpx.AsyncClient(
        transport=httpx.MockTransport(lambda req: httpx.Response(status, text="private-provider-body")), **kwargs))
    try:
        runner = GatewayProviderRunner(explicit_cache=cache, transport=transport, credential_resolver=Resolver())
        with pytest.raises(ExplicitCacheUnavailable, match=f"gemini_cached_http_{status}"):
            async for _ in runner.run(profile=selected, request=request(), context=context(),
                                      cache_namespace="namespace", use_explicit_cache=True):
                pass
        assert (await cache.store.list_resources())[0].last_error == f"gemini_cached_http_{status}"
    finally:
        await transport.close()


@pytest.mark.anyio
async def test_admin_resource_listing_and_exact_resume_do_not_call_provider(monkeypatch):
    import main
    from types import SimpleNamespace
    from fastapi.testclient import TestClient
    cache, transport, selected = await setup_cache(error=True)
    with pytest.raises(ExplicitCacheUnavailable):
        await prepare(cache, transport, selected)
    row = (await cache.store.list_resources())[0]
    async def get_pins(): return SimpleNamespace(list_pins=cache.pins.list_pins)
    monkeypatch.setattr(main, "_get_cache_pin_service", get_pins)
    monkeypatch.setattr(main, "_model_provider_runner", SimpleNamespace(explicit_cache=cache))
    monkeypatch.setattr(main, "_model_execution_service", object())
    monkeypatch.setattr(main, "_model_usage_store", cache.usage)
    monkeypatch.setattr(main, "GATEWAY_SECRET", "synthetic-admin")
    monkeypatch.setattr(main, "MEMORY_ENABLED", True)
    monkeypatch.setenv("MODEL_PROFILE_MANAGEMENT_ENABLED", "true")
    monkeypatch.setenv("ACTOR_PERSONA_PROXY_SECRET", "synthetic-persona")
    client = TestClient(main.app)
    body = {"action": "resume_explicit_cache", "cache_key": row.cache_key, "attempt_id": row.attempt_id,
        "actor_id": row.actor_id, "room_id": row.room_id, "conversation_id": row.conversation_id,
        "execution_mode": row.execution_mode}
    assert client.put("/api/cache-pins", json=body).status_code == 401
    assert client.put("/api/cache-pins", json=body, headers={"X-Gateway-Persona-Key": "synthetic-persona"}).status_code == 401
    client.headers["X-Gateway-Key"] = "synthetic-admin"
    listing = client.get("/api/cache-pins")
    assert listing.status_code == 200
    assert listing.json()["explicit_resources"][0]["status"] == "paused"
    assert "resource_name" not in listing.text and "systemInstruction" not in listing.text
    assert client.put("/api/cache-pins", json=body).status_code == 409
    cache.now = lambda: row.created_at + timedelta(hours=2)
    assert client.put("/api/cache-pins", json={**body, "attempt_id": "stale"}).status_code == 409
    assert client.put("/api/cache-pins", json={**body, "cache_key": 3}).status_code == 422
    assert client.put("/api/cache-pins", json=body).json() == {"status": "resumed", "provider_calls": 0}
    assert len(transport.calls) == 1


@pytest.mark.anyio
@pytest.mark.parametrize("cache_read", [4000, None])
@pytest.mark.parametrize("resource_name", [
    "cachedContents/probe",
    "projects/123456789012/locations/us-central1/cachedContents/synthetic_probe-1",
])
async def test_native_probe_uses_real_runner_with_mock_http_and_does_not_bind(cache_read, resource_name):
    import httpx
    from cache_probe import GatewayCacheProbeService
    from provider_transport import PooledHttpTransport
    from test_gateway_provider_runner import Resolver
    cache, _, selected = await setup_cache(verified=False)
    sent = []
    def respond(req):
        sent.append((req.url.path, json.loads(req.content)))
        if req.url.path.endswith("cachedContents"):
            return httpx.Response(200, json={"name": resource_name, "model": selected.model,
                "expireTime": (cache.now() + timedelta(hours=1)).isoformat(), "usageMetadata": {"totalTokenCount": 4000}})
        assert json.loads(req.content)["cachedContent"] == resource_name
        frame = {"candidates": [{"content": {"parts": [{"text": "synthetic ok"}]}, "finishReason": "STOP"}],
                 "usageMetadata": {"promptTokenCount": 4100, "candidatesTokenCount": 3, "cachedContentTokenCount": cache_read}}
        return httpx.Response(200, text="data: " + json.dumps(frame) + "\n\n")
    transport = PooledHttpTransport(client_factory=lambda **kwargs: httpx.AsyncClient(transport=httpx.MockTransport(respond), **kwargs))
    try:
        runner = GatewayProviderRunner(explicit_cache=cache, transport=transport, credential_resolver=Resolver())
        result = await GatewayCacheProbeService(profiles=cache.profiles, provider_runner=runner, usage_store=cache.usage).run(
            profile_id=selected.profile_id, actor_id="jiao", room_id="room_weiwei_jiao", conversation_id="never-read-real-history")
        assert (result.status == "verified") == (cache_read is not None)
        assert len(sent) == 3
        assert (await cache.store.list_resources())[0].resource_name == resource_name
        assert await cache.profiles.has_verified_probe(selected.profile_id, 1, "gemini_explicit_cache") == (cache_read is not None)
        assert await cache.profiles.get_actor_binding("jiao") is None
        assert sorted(row.execution_purpose for row in await cache.usage.list_receipts()) == ["cache_create", "cache_probe", "cache_probe"]
    finally:
        await transport.close()


@pytest.mark.anyio
async def test_gemini_pin_scheduler_never_calls_provider():
    from conversation_cache_pin import CachePinService
    cache, transport, selected = await setup_cache()
    await cache.profiles.set_actor_default("jiao", selected.profile_id)
    await cache.profiles.set_actor_default("laoke", selected.profile_id)
    service = CachePinService(store=cache.pins, profiles=cache.profiles, provider_runner=object(),
        context_builder=object(), usage_store=cache.usage, now=cache.now)
    assert (await service.run_due_once()).calls == 0
    assert (await service.run_due_once()).calls == 0
    pin = (await cache.pins.list_pins())[0]
    assert all(state.next_keepalive_at is None and state.last_error == "gemini_keepalive_unsupported" for state in pin.actors.values())


@pytest.mark.anyio
async def test_postgres_cache_reservation_reinitialization_and_schema_are_durable(isolated_postgres):
    from database import apply_model_execution_schema
    from gemini_explicit_cache import PostgresGeminiCacheStore
    async def pool_factory(): return isolated_postgres
    async with isolated_postgres.acquire() as conn:
        await apply_model_execution_schema(conn)
        await apply_model_execution_schema(conn)
    cache, transport, selected = await setup_cache(error=True)
    cache.store = PostgresGeminiCacheStore(pool_factory)
    with pytest.raises(ExplicitCacheUnavailable): await prepare(cache, transport, selected)
    cache.store = PostgresGeminiCacheStore(pool_factory)
    with pytest.raises(ExplicitCacheUnavailable): await prepare(cache, transport, selected)
    assert len(transport.calls) == 1
    row = (await cache.store.list_resources())[0]
    assert await cache.store.resume(row.cache_key, row.attempt_id, row.scope, cache.now() + timedelta(hours=2))


@pytest.mark.anyio
async def test_private_rooms_never_share_resources_and_profile_switch_keeps_old_summary():
    cache, transport, selected = await setup_cache()
    req = replace(request(), room_id="room_weiwei_jiao", execution_mode="private")
    await cache.pins.upsert_pin(ConversationCachePin("private:conversation-1", req.room_id, req.conversation_id, "private", True))
    original = context()
    await prepare(cache, transport, selected, original, req)
    await prepare(cache, transport, selected, original, req, namespace="new-profile")
    await prepare(cache, transport, selected, original, req)
    assert len(transport.calls) == 2 and original == context()
    wrong = replace(req, actor_id="laoke", room_id="room_weiwei_laoke")
    assert "cachedContent" not in await prepare(cache, transport, selected, original, wrong)
    await cache.pins.upsert_pin(ConversationCachePin("private:conversation-1", wrong.room_id, wrong.conversation_id, "private", True))
    await prepare(cache, transport, selected, original, wrong)
    assert len(transport.calls) == 3


@pytest.mark.anyio
async def test_profile_edit_drops_verification_before_any_new_request():
    cache, transport, selected = await setup_cache()
    await prepare(cache, transport, selected)
    updated = await cache.profiles.put_profile(replace(selected, revision=2, display_name="new revision"))
    with pytest.raises(ExplicitCacheUnavailable, match="unverified"):
        await prepare(cache, transport, updated, namespace="revision2")
    assert len(transport.calls) == 1


@pytest.mark.anyio
@pytest.mark.parametrize("resource_name", [
    "https://private.invalid/secret",
    "projects/secret/locations/us-central1/cachedContents/../other",
    "projects/secret/locations/us-central1/cachedContents/id?key=secret",
    "projects/secret/locations/us-central1/cachedContents/id/extra",
    "projects//locations/us-central1/cachedContents/secret",
    "projects/secret/locations/us-central1/cachedContents/%2Fother",
])
async def test_cache_invalid_resource_pauses_without_exposing_response(resource_name):
    cache, transport, selected = await setup_cache()
    async def malicious(**kwargs):
        class Response:
            status_code = 200
            def json(self): return {"name": resource_name, "model": selected.model,
                "expireTime": (cache.now() + timedelta(hours=1)).isoformat()}
        return Response()
    transport.request = malicious
    with pytest.raises(ExplicitCacheUnavailable, match="resource_invalid"):
        await prepare(cache, transport, selected)
    row = (await cache.store.list_resources())[0]
    assert row.status == "paused" and row.resource_name is None
    assert "secret" not in str(row)


@pytest.mark.anyio
async def test_summary_and_probe_do_not_implicitly_create_resources():
    from types import SimpleNamespace
    from test_gateway_provider_runner import Resolver
    from test_gemini_native import NativeResponse
    cache, transport, selected = await setup_cache()
    class Stream:
        async def __aenter__(self):
            return NativeResponse([{"candidates": [{"content": {"parts": [{"text": "ok"}]}, "finishReason": "STOP"}]}])
        async def __aexit__(self, *args): return False
    async def open_stream(**kwargs):
        assert "cachedContent" not in kwargs["json_body"]
        return Stream()
    transport.open_stream = open_stream
    runner = GatewayProviderRunner(explicit_cache=cache, transport=transport, credential_resolver=Resolver())
    summary_request = SimpleNamespace(execution_kind="full", generation_request_id="summary")
    _ = [item async for item in runner.run(profile=selected, request=summary_request,
        context=context(), cache_namespace="summary")]
    assert not transport.calls and not await cache.store.list_resources()
