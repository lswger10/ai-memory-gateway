import json
import httpx
import pytest
from gateway_provider_runner import _continue_with_tool_results
from shared_page_client import SharedPageClient


@pytest.mark.anyio
async def test_calendar_actor_bound_and_write_timeout_not_retried():
    calls = []
    def handler(request):
        calls.append(request)
        assert json.loads(request.content)["actor"] == "laoke"
        assert request.headers["X-Calendar-Token"] == "synthetic"
        raise httpx.ReadTimeout("synthetic")
    client = SharedPageClient("http://calendar", "synthetic", transport=httpx.MockTransport(handler))
    result = await client.call("laoke", {"action": "create", "title": "SYNTHETIC"}, images_enabled=True)
    assert result["is_error"] and len(calls) == 1
    assert "unknown" in result["error"]["code"]
    result = await client.call("laoke", {"action": "create", "actor": "jiao"}, images_enabled=True)
    assert result["is_error"] and len(calls) == 1


@pytest.mark.parametrize("protocol", ["anthropic_messages_compatible", "openai_chat_completions", "openai_responses"])
def test_calendar_page_image_reaches_provider_as_image(protocol):
    calls = [{"id": "synthetic-call", "name": "calendar", "arguments": {"action": "see"}}]
    result = {"calendar_content": [{"type": "text", "text": "synthetic page"},
        {"type": "image", "data": "cG5n", "mimeType": "image/png"}]}
    body = _continue_with_tool_results(protocol, {"messages": [], "input": []}, calls, [result])
    serialized = json.dumps(body)
    assert "synthetic page" in serialized
    assert '"image"' in serialized or '"image_url"' in serialized or '"input_image"' in serialized
    assert "calendar_content" not in serialized
