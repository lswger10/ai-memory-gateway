import pytest

from gateway_provider_runner import GatewayProviderRunner
from model_execution import ContextBundle
from model_execution_contracts import GatewayExecutionRequest
from model_profiles import ModelProfile


def profile():
    return ModelProfile.from_dict(
        {
            "profile_id": "anthropic-ofox",
            "display_name": "OFOX Claude",
            "enabled": True,
            "test_status": "passed",
            "provider": "ofox",
            "protocol": "anthropic_messages_compatible",
            "base_url": "https://api.example.invalid/anthropic",
            "route_id": "ofox-anthropic",
            "model": "anthropic/claude-opus-4.6",
            "adapter_version": "gateway.anthropic.v1",
            "credential_ref": "env:TEST_KEY",
            "headers": {"x-api-key": "${credential}"},
            "capabilities": {
                "streaming": True,
                "structured_output": False,
                "tools": False,
                "reasoning_controls": False,
                "cache_strategies": ["anthropic_prefix_anchored_v1"],
                "cache_ttls": ["5m"],
                "usage_fields": ["cache_creation_input_tokens", "cache_read_input_tokens"],
            },
            "cache_strategy": "anthropic_prefix_anchored_v1",
            "requested_cache_ttl": "5m",
            "revision": 1,
        }
    )


def request():
    return GatewayExecutionRequest.from_dict(
        {
            "contract_version": "gateway-model-execution.v1.0",
            "execution_kind": "full",
            "execution_mode": "group",
            "actor_id": "jiao",
            "room_id": "room_group_home",
            "conversation_id": "conversation-1",
            "current_event_id": 2,
            "generation_request_id": "generation-1",
            "fence": {
                "room_id": "room_group_home",
                "conversation_id": "conversation-1",
                "burst_id": "burst-1",
                "trigger_event_id": 2,
                "fence_epoch": 1,
                "lease_epoch": 1,
                "orchestrator_instance": "orch-1",
            },
            "bedroom_session_id": None,
            "actor_private_stance": "dynamic stance",
            "binding_revision": None,
        }
    )


class Resolver:
    def resolve(self, credential_ref):
        return "secret"


class Response:
    status_code = 200

    async def aiter_lines(self):
        lines = [
            "event: message_start",
            'data: {"message":{"usage":{"input_tokens":100,"cache_creation_input_tokens":80,"cache_read_input_tokens":20}}}',
            "",
            "event: content_block_delta",
            'data: {"delta":{"type":"text_delta","text":"hi"}}',
            "",
            "event: message_delta",
            'data: {"usage":{"output_tokens":2}}',
            "",
        ]
        for line in lines:
            yield line


class StreamContext:
    async def __aenter__(self):
        return Response()

    async def __aexit__(self, *args):
        return False


class Transport:
    def __init__(self):
        self.calls = []

    async def open_stream(self, **kwargs):
        self.calls.append(kwargs)
        return StreamContext()


@pytest.mark.anyio
async def test_anthropic_runner_keeps_dynamic_tail_after_cache_breakpoint(monkeypatch):
    transport = Transport()
    runner = GatewayProviderRunner(transport=transport, credential_resolver=Resolver())
    context = ContextBundle(
        static_system=("kernel", "actor", "room"),
        stable_summary="summary",
        stable_history=("old fact",),
        dynamic_tail=("retrieved memory", "current event"),
        actor_prompt_version="actor.v1",
        runtime_kernel_version="kernel.v1",
        room_policy_version="room.v1",
        tool_schema_hash="tools.none.v1",
    )
    chunks = [
        chunk
        async for chunk in runner.run(
            profile=profile(), request=request(), context=context, cache_namespace="cache-1"
        )
    ]
    body = transport.calls[0]["json_body"]
    assert body["messages"][1]["content"][0]["cache_control"]["ttl"] == "5m"
    assert "cache_control" not in body["messages"][2]["content"][0]
    assert "retrieved memory" in body["messages"][2]["content"][0]["text"]
    usage = next(chunk for chunk in chunks if chunk.event == "usage")
    assert usage.data["usage"].cache_read_input_tokens == 20
    assert usage.data["observed_cache_support"] == "verified"
