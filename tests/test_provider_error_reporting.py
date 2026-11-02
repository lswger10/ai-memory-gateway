import json
from dataclasses import replace

import httpx
import pytest

from gateway_provider_runner import GatewayProviderRunner
from provider_transport import PooledHttpTransport
from test_model_execution_service import _request, _service


class Credentials:
    def resolve(self, reference):
        return "credential-secret"


async def execute(handler, *, fallbacks=(), context=None, headers=None):
    service, builder, _, usage = await _service(fallbacks=fallbacks)
    if headers:
        from test_model_execution_service import _profile
        await service._profiles.put_profile(replace(_profile("primary"), headers=tuple(headers.items()), revision=2))
        await service._profiles.set_test_status("primary", "passed", expected_revision=2)
    if context is not None:
        class Builder:
            async def build(self, request, *args, **kwargs):
                return context
        service._context_builder = Builder()
    transport = PooledHttpTransport(client_factory=lambda **kwargs: httpx.AsyncClient(
        transport=httpx.MockTransport(handler), **kwargs))
    service._provider_runner = GatewayProviderRunner(
        transport=transport, credential_resolver=Credentials())
    try:
        return [event async for event in service.stream(_request(binding_revision=None))], usage
    finally:
        await transport.close()


@pytest.mark.anyio
@pytest.mark.parametrize("status,reason", [
    (400, "provider_bad_request"), (401, "provider_auth_failed"),
    (403, "provider_auth_failed"), (402, "provider_payment_required"),
    (410, "provider_model_retired"), (429, "provider_rate_limited"),
    (500, "provider_unavailable"), (503, "provider_unavailable"),
    (404, "generation_unavailable"),
])
async def test_http_failures_retain_status_and_reason_without_provider_text(status, reason, caplog):
    events, usage = await execute(lambda request: httpx.Response(
        status, json={"error": {"message": "model has been retired"}}))
    assert events[-1].data == {"generation_request_id": "generation-1",
                               "reason_code": reason, "provider_status": status}
    assert "model has been retired" in caplog.text
    assert "model has been retired" not in str(events)
    assert len(await usage.list_receipts()) == 1


@pytest.mark.anyio
@pytest.mark.parametrize("error", [httpx.ReadTimeout, httpx.ConnectError])
async def test_transport_failure_has_structured_reason_without_raw_exception(error, caplog):
    def handler(request):
        raise error("sensitive-url?key=credential-secret", request=request)
    events, _ = await execute(handler)
    assert events[-1].data == {"generation_request_id": "generation-1",
                               "reason_code": "provider_transport_failed"}
    assert "credential-secret" not in caplog.text + str(events)


@pytest.mark.anyio
async def test_fallback_reports_last_failure_and_keeps_successful_fallback():
    for final_status in (429, 200):
        responses = iter([httpx.Response(410, json={"error": {"message": "retired"}}),
            httpx.Response(final_status, text=(
                'event: content_block_delta\ndata: {"delta":{"type":"text_delta","text":"hello"}}\n\n'
                'event: message_stop\ndata: {}\n\n'))])
        events, usage = await execute(lambda request: next(responses), fallbacks=("backup",))
        assert len(await usage.list_receipts()) == 2
        if final_status == 429:
            assert events[-1].data["reason_code"] == "provider_rate_limited"
        else:
            assert events[-1].event == "done"
            assert next(event.data["text"] for event in events if event.event == "final") == "hello"


@pytest.mark.anyio
async def test_error_logs_redact_credentials_and_echoed_input_and_limit_summary(caplog):
    from test_model_execution_service import _ContextBuilder
    context = replace(await _ContextBuilder().build(_request()),
                      stable_summary="private conversation content")
    def handler(request):
        return httpx.Response(400, json={"error": {"message":
            "invalid credential-secret private conversation content\n" + "x" * 500},
            "request": json.loads(request.content)})
    events, _ = await execute(handler, context=context)
    assert "credential-secret" not in caplog.text + str(events)
    assert "private conversation content" not in caplog.text + str(events)
    record = next(record for record in caplog.records if "provider_error=" in record.message)
    summary = record.message.split("provider_error=", 1)[1]
    assert "invalid" in summary and "[redacted]" in summary
    assert len(summary) <= 200 and "\n" not in summary


@pytest.mark.anyio
async def test_error_body_read_is_bounded_and_http_reason_survives_broken_body():
    class HugeBody(httpx.AsyncByteStream):
        async def __aiter__(self):
            for _ in range(16):
                yield b"x" * 1024
            pytest.fail("error response was read without a limit")

    class BrokenBody(httpx.AsyncByteStream):
        async def __aiter__(self):
            raise httpx.ReadTimeout("credential-secret")
            yield b""

    for body in (HugeBody(), BrokenBody()):
        events, _ = await execute(lambda request: httpx.Response(410, stream=body))
        assert events[-1].data["reason_code"] == "provider_model_retired"


@pytest.mark.anyio
async def test_plain_text_error_redacts_header_tokens_and_json_history_content(caplog):
    from test_model_execution_service import _ContextBuilder
    context = replace(await _ContextBuilder().build(_request()),
        stable_history=(json.dumps({"actor_id": "weiwei", "content": "private-history-text"}),))
    events, _ = await execute(lambda request: httpx.Response(401, text=
        "authentication failed: header-token api-secret credential-secret private-history-text"),
        context=context, headers={"Authorization": "Bearer header-token", "x-api-key": "api-secret"})
    assert "authentication failed" in caplog.text
    for sensitive in ("header-token", "api-secret", "credential-secret", "private-history-text"):
        assert sensitive not in caplog.text + str(events)


@pytest.mark.anyio
@pytest.mark.parametrize("fallbacks", [(), ("backup",)])
async def test_unhealthy_primary_is_skipped_before_context_and_provider(fallbacks, monkeypatch):
    service, context, runner, usage = await _service(fallbacks=fallbacks)
    resolved = await service._profiles.resolve("jiao", "room_group_home")
    async def resolve(actor_id, room_id):
        return replace(resolved, primary=replace(resolved.primary, test_status="failed"))
    monkeypatch.setattr(service._profiles, "resolve", resolve)
    events = [event async for event in service.stream(_request(binding_revision=None))]
    assert [call[0] for call in runner.calls] == list(fallbacks)
    assert len(context.requests) == len(fallbacks)
    assert len(await usage.list_receipts()) == len(fallbacks)
    if fallbacks:
        assert events[-1].event == "done"
        assert events[0].data["fallback_from_profile_id"] == "primary"
    else:
        assert events[-1].data["reason_code"] == "generation_unavailable"
