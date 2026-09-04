import httpx
import pytest

from cache_strategies import AnthropicPrefixAnchoredV1, PromptSegment
from actor_memory_tools import actor_memory_tool_definitions
from model_profiles import ModelProfile
from provider_adapters import (
    AnthropicMessagesAdapter,
    OpenAIChatCompletionsAdapter,
    OpenAIResponsesAdapter,
    build_provider_provenance,
)
from provider_transport import EnvironmentCredentialResolver, PooledHttpTransport


def _profile(
    protocol: str,
    strategy: str,
    *,
    ttl="5m",
    cache_strategies=None,
    cache_ttls=None,
):
    strategies = cache_strategies or [strategy]
    ttls = cache_ttls if cache_ttls is not None else ([ttl] if ttl else [])
    headers = (
        {"x-api-key": "${credential}", "anthropic-version": "2023-06-01"}
        if protocol == "anthropic_messages"
        else {"Authorization": "Bearer ${credential}"}
    )
    return ModelProfile.from_dict(
        {
            "profile_id": f"profile-{protocol}",
            "display_name": protocol,
            "enabled": True,
            "test_status": "passed",
            "provider": "test-provider",
            "protocol": protocol,
            "base_url": "https://example.invalid",
            "route_id": f"route-{protocol}",
            "model": f"model-{protocol}",
            "adapter_version": "adapter.v1",
            "credential_ref": "env:TEST_PROVIDER_KEY",
            "headers": headers,
            "capabilities": {
                "streaming": True,
                "structured_output": False,
                "tools": True,
                "reasoning_controls": False,
                "cache_strategies": strategies,
                "cache_ttls": ttls,
                "usage_fields": [
                    "input_tokens",
                    "output_tokens",
                    "cache_creation_input_tokens",
                    "cache_read_input_tokens",
                    "cached_tokens",
                ],
            },
            "cache_strategy": strategy,
            "requested_cache_ttl": ttl,
            "revision": 1,
        }
    )


def _anthropic_layout(ttl="5m"):
    return AnthropicPrefixAnchoredV1().build_layout(
        tools=({"name": "fixed_tool", "input_schema": {"type": "object"}},),
        stable_segments=(
            PromptSegment("runtime_kernel", "kernel"),
            PromptSegment("actor_prompt", "actor"),
            PromptSegment("room_policy", "room"),
            PromptSegment("factual_history", "history"),
        ),
        dynamic_segments=(
            PromptSegment("retrieved_memory", "memory"),
            PromptSegment("current_event", "current"),
        ),
        capabilities=_profile(
            "anthropic_messages", "anthropic_prefix_anchored_v1", ttl=ttl
        ).capabilities,
        requested_ttl=ttl,
    )


def test_anthropic_adapter_renders_only_verified_cache_control_and_ttl():
    profile = _profile(
        "anthropic_messages", "anthropic_prefix_anchored_v1", ttl="1h"
    )
    request = AnthropicMessagesAdapter().render(
        profile=profile,
        layout=_anthropic_layout("1h"),
        max_output_tokens=300,
    )

    assert request.path == "/v1/messages"
    assert request.json_body["model"] == profile.model
    assert request.json_body["messages"][-1]["content"][0]["text"].endswith(
        "current"
    )
    breakpoints = [
        block["cache_control"]
        for message in request.json_body["messages"]
        for block in message["content"]
        if "cache_control" in block
    ]
    assert breakpoints == [{"type": "ephemeral", "ttl": "1h"}]
    assert "stream_options" not in request.json_body


def test_openai_strategy_never_receives_anthropic_cache_control():
    profile = _profile(
        "openai_chat_completions",
        "openai_stable_prefix_v1",
        ttl=None,
        cache_ttls=[],
    )
    request = OpenAIChatCompletionsAdapter().render(
        profile=profile,
        system_content="static system",
        messages=({"role": "user", "content": "hello"},),
        prompt_cache_key="namespace-hash",
        max_output_tokens=100,
        tools=actor_memory_tool_definitions(),
    )
    assert "cache_control" not in str(request.json_body)
    assert request.json_body["prompt_cache_key"] == "namespace-hash"
    assert request.json_body["stream_options"] == {"include_usage": True}
    assert request.json_body["tools"][0]["type"] == "function"
    assert "actor_id" not in request.json_body["tools"][0]["function"]["parameters"].get("properties", {})


def test_openai_responses_renders_profile_model_and_cache_key():
    profile = _profile(
        "openai_responses", "openai_stable_prefix_v1", ttl=None, cache_ttls=[]
    )
    request = OpenAIResponsesAdapter().render(
        profile=profile,
        instructions="static",
        input_items=({"role": "user", "content": "hello"},),
        prompt_cache_key="namespace-hash",
        max_output_tokens=100,
        tools=actor_memory_tool_definitions(),
    )
    assert request.path == "/v1/responses"
    assert request.json_body["model"] == profile.model
    assert request.json_body["prompt_cache_key"] == "namespace-hash"
    assert request.json_body["tools"][0]["type"] == "function"
    assert "function" not in request.json_body["tools"][0]


def test_secrets_are_absent_from_provenance():
    profile = _profile(
        "anthropic_messages", "anthropic_prefix_anchored_v1", ttl="5m"
    )
    provenance = build_provider_provenance(
        profile,
        generation_request_id="gen-1",
        fallback_used=True,
        fallback_from_profile_id="profile-old",
    )
    rendered = provenance.to_dict()
    assert rendered["profile_id"] == profile.profile_id
    assert "credential_ref" not in rendered
    assert "base_url" not in rendered
    assert "api_key" not in str(rendered).lower()


def test_environment_credential_resolver_requires_env_reference(monkeypatch):
    monkeypatch.setenv("TEST_PROVIDER_KEY", "secret-value")
    resolver = EnvironmentCredentialResolver()
    assert resolver.resolve("env:TEST_PROVIDER_KEY") == "secret-value"
    with pytest.raises(ValueError):
        resolver.resolve("literal-secret")


class _FakeResponse:
    status_code = 200


class _FakeClient:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.calls = []
        self.closed = False

    async def request(self, method, path, **kwargs):
        self.calls.append((method, path, kwargs))
        return _FakeResponse()

    async def aclose(self):
        self.closed = True


@pytest.mark.anyio
async def test_transport_reuses_one_async_client_across_requests():
    created = []

    def factory(**kwargs):
        client = _FakeClient(**kwargs)
        created.append(client)
        return client

    transport = PooledHttpTransport(client_factory=factory)
    await transport.request(
        pool_key="route-a",
        base_url="https://example.invalid",
        headers={"Authorization": "Bearer secret"},
        method="POST",
        path="/v1/responses",
        json_body={"model": "a"},
    )
    await transport.request(
        pool_key="route-a",
        base_url="https://example.invalid",
        headers={"Authorization": "Bearer secret"},
        method="POST",
        path="/v1/responses",
        json_body={"model": "a"},
    )
    assert len(created) == 1
    assert len(created[0].calls) == 2
    await transport.close()
    assert created[0].closed is True


@pytest.mark.anyio
@pytest.mark.parametrize(
    ("base_url", "endpoint_path", "expected_url"),
    (
        (
            "https://api.ofox.ai/v1",
            "/v1/chat/completions",
            "https://api.ofox.ai/v1/chat/completions",
        ),
        (
            "https://api.openai.com",
            "/v1/chat/completions",
            "https://api.openai.com/v1/chat/completions",
        ),
        (
            "https://api.ofox.ai/anthropic",
            "/v1/messages",
            "https://api.ofox.ai/anthropic/v1/messages",
        ),
    ),
)
async def test_transport_joins_profile_base_url_without_duplicate_route_segments(
    base_url, endpoint_path, expected_url
):
    observed = []

    async def handler(request):
        observed.append(str(request.url))
        return httpx.Response(200, json={"ok": True})

    def factory(**kwargs):
        return httpx.AsyncClient(
            **kwargs,
            transport=httpx.MockTransport(handler),
        )

    transport = PooledHttpTransport(client_factory=factory)
    try:
        await transport.request(
            pool_key="route-under-test",
            base_url=base_url,
            headers={"Authorization": "Bearer test-only"},
            method="POST",
            path=endpoint_path,
            json_body={"model": "test-model"},
        )
    finally:
        await transport.close()

    assert observed == [expected_url]
