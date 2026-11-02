import asyncio
from dataclasses import replace

import pytest

from model_health import ModelHealthService
from model_execution import ProviderChunk, ProviderRunUnavailable
from model_execution_contracts import ProviderUsage
from model_profile_store import InMemoryModelProfileStore
from model_usage_store import InMemoryModelUsageStore
from test_model_profile_store import _profile


class Runner:
    def __init__(self, failure=None):
        self.calls = []
        self.failure = failure

    async def run(self, **kwargs):
        self.calls.append(kwargs)
        await asyncio.sleep(0)
        if self.failure:
            raise self.failure
        yield ProviderChunk("final", {"text": "OK"})
        yield ProviderChunk("usage", {"usage": ProviderUsage.from_provider_values(output_tokens=1)})


@pytest.mark.anyio
async def test_gemini_health_allows_thinking_before_visible_reply():
    import json
    import httpx
    from gateway_provider_runner import GatewayProviderRunner
    from provider_transport import PooledHttpTransport
    from test_gateway_provider_runner import Resolver
    from test_gemini_native import native_profile

    profiles = InMemoryModelProfileStore()
    profile = replace(native_profile(), test_status="unverified")
    await profiles.put_profile(profile)

    def respond(req):
        budget = json.loads(req.content)["generationConfig"]["maxOutputTokens"]
        # The live provider used 114 thinking tokens before its one-token reply.
        enough = budget >= 115
        frame = {"candidates": [{"content": {"parts": [{"text": "OK" if enough else ""}]},
                                "finishReason": "STOP" if enough else "MAX_TOKENS"}]}
        return httpx.Response(200, text="data: " + json.dumps(frame) + "\n\n")

    transport = PooledHttpTransport(client_factory=lambda **kwargs:
        httpx.AsyncClient(transport=httpx.MockTransport(respond), **kwargs))
    try:
        service = ModelHealthService(profiles=profiles,
            provider_runner=GatewayProviderRunner(transport=transport, credential_resolver=Resolver()),
            usage_store=InMemoryModelUsageStore())
        result = await service.check(profile.profile_id, profile_revision=profile.revision)
        assert result["status"] == "verified"
        assert (await profiles.get_profile(profile.profile_id)).selectable
    finally:
        await transport.close()


@pytest.mark.anyio
async def test_daily_check_is_bounded_synthetic_and_claimed_once_across_workers():
    profiles = InMemoryModelProfileStore()
    await profiles.put_profile(_profile("active"))
    await profiles.put_profile(replace(_profile("disabled"), enabled=False))
    runner = Runner()
    first = ModelHealthService(profiles=profiles, provider_runner=runner, usage_store=InMemoryModelUsageStore())
    second = ModelHealthService(profiles=profiles, provider_runner=runner, usage_store=InMemoryModelUsageStore())
    await asyncio.gather(first.run_due_once(), second.run_due_once())
    await first.run_due_once()
    assert len(runner.calls) == 1
    call = runner.calls[0]
    assert call["profile"].profile_id == "active"
    assert call["max_output_tokens"] == 32
    assert call["context"].stable_history == ()
    assert call["context"].actor_memory_context is None
    assert call["context"].tool_schema_hash == "tools.none.v1"
    assert call["context"].web_search_enabled is False
    assert call["request"].conversation_id.startswith("model-health:")
    results = await profiles.list_health_results()
    assert results[0]["status"] == "verified"
    await profiles.put_profile(replace(_profile("active"), revision=2))
    await first.run_due_once()
    assert len(runner.calls) == 1  # Editing must not multiply scheduled provider charges.


@pytest.mark.anyio
@pytest.mark.parametrize("status,expected_status", [(410, "failed"), (400, "failed"), (429, "passed"), (503, "passed")])
async def test_daily_failure_alert_and_only_permanent_failure_revokes_profile(status, expected_status):
    profiles = InMemoryModelProfileStore()
    await profiles.put_profile(_profile("primary"))
    await profiles.put_profile(_profile("gemini-backup"))
    await profiles.save_actor_binding("laoke", "primary", ("gemini-backup",), expected_revision=0)
    failure = ProviderRunUnavailable(f"provider HTTP {status}")
    failure.provider_status = status
    failure.reason_code = "provider_model_retired" if status == 410 else "provider_unavailable"
    service = ModelHealthService(profiles=profiles, provider_runner=Runner(failure), usage_store=InMemoryModelUsageStore())
    await service.check("primary", profile_revision=1)
    assert (await profiles.get_profile("primary")).test_status == expected_status
    resolved = await profiles.resolve("laoke", "room_weiwei_laoke")
    assert resolved.primary.profile_id == "primary"
    assert [p.profile_id for p in resolved.fallbacks] == ["gemini-backup"]
    result = (await profiles.list_health_results())[0]
    assert result["status"] == "failed"
    assert result["provider_status"] == status
    assert result["reason_code"] == failure.reason_code


@pytest.mark.anyio
async def test_manual_success_verifies_saved_profile_and_failure_does_not_escape_to_next_profile():
    profiles = InMemoryModelProfileStore()
    await profiles.put_profile(_profile("new", status="unverified"))
    await profiles.put_profile(_profile("other"))
    runner = Runner()
    service = ModelHealthService(profiles=profiles, provider_runner=runner, usage_store=InMemoryModelUsageStore())
    result = await service.check("new", profile_revision=1)
    assert result["status"] == "verified"
    assert (await profiles.get_profile("new")).selectable
    assert [call["profile"].profile_id for call in runner.calls] == ["new"]


@pytest.mark.anyio
async def test_changed_revision_during_check_cannot_be_certified_or_shown_as_current():
    profiles = InMemoryModelProfileStore()
    await profiles.put_profile(_profile("changed", status="unverified"))

    class EditingRunner(Runner):
        async def run(self, **kwargs):
            await profiles.put_profile(replace(kwargs["profile"], model="new-model", revision=2))
            async for chunk in super().run(**kwargs):
                yield chunk

    service = ModelHealthService(profiles=profiles, provider_runner=EditingRunner(), usage_store=InMemoryModelUsageStore())
    await service.check("changed", profile_revision=1)
    assert (await profiles.get_profile("changed")).test_status == "unverified"
    assert await profiles.list_health_results() == ()


@pytest.mark.anyio
async def test_unhealthy_backup_is_skipped_without_blocking_other_approved_routes():
    profiles = InMemoryModelProfileStore()
    for name in ("primary", "bad", "good"):
        await profiles.put_profile(_profile(name))
    await profiles.save_actor_binding("jiao", "primary", ("bad", "good"), expected_revision=0)
    await profiles.set_test_status("bad", "failed", expected_revision=1)
    assert [p.profile_id for p in (await profiles.resolve("jiao", "room_weiwei_jiao")).fallbacks] == ["good"]


@pytest.mark.anyio
async def test_failed_primary_executes_only_approved_backup_with_correct_provenance():
    from model_execution import GatewayModelExecutionService
    from test_model_execution_service import _ContextBuilder, _Runner, _request
    profiles = InMemoryModelProfileStore()
    await profiles.put_profile(_profile("primary"))
    await profiles.put_profile(_profile("gemini-backup"))
    await profiles.save_actor_binding("jiao", "primary", ("gemini-backup",), expected_revision=0)
    await profiles.set_test_status("primary", "failed", expected_revision=1)
    runner = _Runner()
    service = GatewayModelExecutionService(profiles=profiles, context_builder=_ContextBuilder(),
        provider_runner=runner, usage_store=InMemoryModelUsageStore())
    events = [event async for event in service.stream(_request())]
    assert [call[0] for call in runner.calls] == ["gemini-backup"]
    provenance = next(event.data for event in events if event.event == "profile")
    assert provenance["fallback_used"] is True
    assert provenance["fallback_from_profile_id"] == "primary"
    assert events[-1].event == "done"


@pytest.mark.anyio
async def test_postgres_daily_claim_is_atomic_and_old_revisions_are_hidden(isolated_postgres):
    from database import apply_model_execution_schema
    from postgres_model_stores import PostgresModelProfileStore
    async with isolated_postgres.acquire() as conn:
        await apply_model_execution_schema(conn)
    async def pool_factory():
        return isolated_postgres
    profiles = PostgresModelProfileStore(pool_factory)
    await profiles.put_profile(_profile("primary"))
    await profiles.put_profile(_profile("backup"))
    await profiles.save_actor_binding("jiao", "primary", ("backup",), expected_revision=0)
    claims = await asyncio.gather(*(profiles.claim_health_check("primary", 1) for _ in range(5)))
    assert claims.count(True) == 1
    await profiles.record_probe_result(profile_id="primary", profile_revision=1,
        probe_kind="model_health", status="failed", observed_capabilities={"reason_code": "provider_model_retired"})
    await profiles.set_test_status("primary", "failed", expected_revision=1)
    assert (await profiles.resolve("jiao", "room_weiwei_jiao")).fallbacks[0].profile_id == "backup"
    assert (await profiles.list_health_results())[0]["reason_code"] == "provider_model_retired"
    await profiles.put_profile(replace(_profile("primary"), revision=2))
    assert await profiles.list_health_results() == ()
    assert not await profiles.claim_health_check("primary", 2)
    async with isolated_postgres.acquire() as conn:
        await conn.execute("UPDATE model_profile_probe_results SET tested_at=NOW()-INTERVAL '2 days'")
    assert await profiles.claim_health_check("primary", 2)
