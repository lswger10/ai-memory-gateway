import json
from dataclasses import replace

import pytest

from gateway_provider_runner import GatewayProviderRunner, _continue_with_tool_results
from model_execution import ContextBundle, ProviderRunUnavailable
from cache_strategies import PromptSegment
from model_profiles import ModelProfile
from test_gateway_provider_runner import profile, request, Resolver


def native_profile():
    data = profile().to_dict()
    data.update(protocol="gemini_native", model="google/gemini-test", base_url="https://api.example.invalid",
                cache_strategy="gemini_explicit_cache_v1", requested_cache_ttl="1h",
                headers={"x-goog-api-key": "${credential}"})
    data["capabilities"].update(cache_strategies=["gemini_explicit_cache_v1"], cache_ttls=["1h"],
                                tools=True, input_modalities=["text", "image", "document"])
    return ModelProfile.from_dict(data)


def context():
    return ContextBundle(("kernel", "actor", "room"), "summary", (
        json.dumps({"event_id": 1, "actor_id": "jiao", "role": "agent", "event_type": "agent_final", "content": "earlier"}),
    ), (PromptSegment("current_time", "now"), PromptSegment("current_event", "question")),
        "actor.v1", "runtime.v1", "room.v1", "actor-memory-tools.v1")


def test_native_render_preserves_roles_context_labels_media_and_tools():
    rendered = GatewayProviderRunner()._render(native_profile(), request(), context(), "namespace", media_parts=(
        {"kind": "image", "media_type": "image/png", "data": "aW1hZ2U="},
        {"kind": "document", "media_type": "application/pdf", "data": "cGRm"},
    ))
    assert rendered.path == "/gemini/v1beta/models/google/gemini-test:streamGenerateContent?alt=sse"
    body = rendered.json_body
    assert body["systemInstruction"]["parts"][0]["text"] == "kernel\n\nactor\n\nroom"
    assert body["contents"][1] == {"role": "model", "parts": [{"text": "earlier"}]}
    assert body["contents"][2]["parts"][0]["text"] == "[current_time]\nnow"
    assert body["contents"][3]["parts"][0]["text"] == "[current_event]\nquestion"
    assert body["contents"][-1]["parts"][-1]["inlineData"]["mimeType"] == "application/pdf"
    assert body["tools"][0]["functionDeclarations"][0]["parameters"]


class NativeResponse:
    status_code = 200
    def __init__(self, frames): self.frames = frames
    async def aiter_lines(self):
        for frame in self.frames:
            yield "data: " + json.dumps(frame)
            yield ""


@pytest.mark.anyio
async def test_native_stream_hides_thought_and_requires_terminal_preserves_usage_and_signature():
    parts = [{"text": "private reasoning", "thought": True, "thoughtSignature": "sig-1"},
             {"functionCall": {"name": "search_memory", "args": {"query": "rain"}}, "thoughtSignature": "sig-2"}]
    frames = [{"candidates": [{"content": {"role": "model", "parts": parts}, "finishReason": "STOP"}],
               "usageMetadata": {"promptTokenCount": 100, "candidatesTokenCount": 5, "cachedContentTokenCount": 80}}]
    chunks = [item async for item in GatewayProviderRunner()._gemini(NativeResponse(frames), native_profile(), request())]
    assert not any(item.event == "delta" for item in chunks)
    tool = next(item.data for item in chunks if item.event == "tool_calls")
    continued = _continue_with_tool_results("gemini_native", {"contents": []}, tool["calls"], [{"ok": True}], assistant_content=tool["assistant_content"])
    assert continued["contents"][0] == {"role": "model", "parts": parts}
    assert continued["contents"][1]["parts"][0]["functionResponse"]["name"] == "search_memory"
    usage = next(item.data["usage"] for item in chunks if item.event == "usage")
    assert (usage.input_tokens, usage.output_tokens, usage.cached_tokens) == (100, 5, 80)
    with pytest.raises(ProviderRunUnavailable, match="terminal"):
        _ = [item async for item in GatewayProviderRunner()._gemini(NativeResponse([
            {"candidates": [{"content": {"parts": [{"text": "partial"}]}}]}]), native_profile(), request())]


@pytest.mark.anyio
@pytest.mark.parametrize("finish", ["STOP", "MAX_TOKENS", "SAFETY"])
async def test_native_stream_finish_reason(finish):
    response = NativeResponse([{"candidates": [{"content": {"parts": [{"text": "reply"}]}, "finishReason": finish}]}])
    if finish == "SAFETY":
        with pytest.raises(ProviderRunUnavailable):
            _ = [item async for item in GatewayProviderRunner()._gemini(response, native_profile(), request())]
    else:
        chunks = [item async for item in GatewayProviderRunner()._gemini(response, native_profile(), request())]
        final = next(item for item in chunks if item.event == "final")
        assert final.data["text"] == "reply"
        assert bool(final.data.get("truncated")) == (finish == "MAX_TOKENS")
        usage = next(item.data["usage"] for item in chunks if item.event == "usage")
        assert usage.cached_tokens is None


def test_native_usage_counts_billed_thought_tokens_and_calendar_bytes_are_not_json_text():
    from provider_adapters import GeminiNativeAdapter
    usage = GeminiNativeAdapter().parse_usage({"promptTokenCount": 40, "candidatesTokenCount": 5, "thoughtsTokenCount": 100})
    assert usage.output_tokens == 105
    result = {"calendar_content": [{"type": "text", "text": "page"},
        {"type": "image", "mimeType": "image/png", "data": "c3ludGhldGlj"}]}
    body = _continue_with_tool_results("gemini_native", {"contents": []},
        [{"id": "internal", "name": "calendar"}], [result],
        assistant_content={"role": "model", "parts": [{"functionCall": {"name": "calendar", "args": {}}}]})
    assert "c3ludGhldGlj" not in json.dumps(body["contents"][1])
    assert body["contents"][2]["parts"][1]["inlineData"]["data"] == "c3ludGhldGlj"


@pytest.mark.anyio
async def test_native_real_runner_tool_roundtrip_preserves_signature_and_scope():
    import httpx
    from provider_transport import PooledHttpTransport
    sent, invoked = [], []
    signature = {"functionCall": {"name": "search_memory", "args": {"query": "synthetic"}}, "thoughtSignature": "opaque"}
    def respond(req):
        body = json.loads(req.content)
        sent.append(body)
        if len(sent) == 1:
            # OFOX returns STOP with empty parts for parametersJsonSchema.
            declarations = body["tools"][0]["functionDeclarations"]
            assert all("parametersJsonSchema" not in tool for tool in declarations)
            search = next(tool for tool in declarations if tool["name"] == "search_memory")
            assert search["parameters"]["required"] == ["query"]
            assert search["parameters"]["properties"]["query"] == {"type": "string"}
            parts = [signature]
        else:
            assert body["contents"][-2] == {"role": "model", "parts": [signature]}
            assert body["contents"][-1]["parts"][0]["functionResponse"]["response"]["is_error"]
            parts = [{"text": "Scope respected."}]
        return httpx.Response(200, text="data: " + json.dumps({"candidates": [{"content": {"parts": parts}, "finishReason": "STOP"}]}) + "\n\n")
    class Memory:
        async def call(self, ctx, key, name, arguments):
            invoked.append((ctx, name, arguments))
            raise PermissionError("synthetic scope denial")
        async def discard(self, ctx): pass
    transport = PooledHttpTransport(client_factory=lambda **kwargs: httpx.AsyncClient(transport=httpx.MockTransport(respond), **kwargs))
    actor_context = object()
    try:
        runner = GatewayProviderRunner(transport=transport, credential_resolver=Resolver(), memory_tools=Memory())
        items = [item async for item in runner.run(profile=native_profile(), request=request(),
            context=replace(context(), actor_memory_context=actor_context), cache_namespace="namespace")]
        assert invoked == [(actor_context, "search_memory", {"query": "synthetic"})]
        assert len(sent) == 2
        assert next(item.data["text"] for item in items if item.event == "final") == "Scope respected."
    finally:
        await transport.close()
