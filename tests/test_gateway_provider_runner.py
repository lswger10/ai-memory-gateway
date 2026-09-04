import json

import pytest

from actor_memory_tools import ACTOR_MEMORY_TOOL_NAMES, ActorMemoryExecutionContext, ActorMemoryToolLibrary, InMemoryActorMemoryToolStore
from gateway_provider_runner import GatewayProviderRunner
from model_execution import ContextBundle, ProviderRunUnavailable
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


class ToolTransport(Transport):
    async def open_stream(self, **kwargs):
        self.calls.append(kwargs)
        return ToolStreamContext(len(self.calls))


class ToolResponse(Response):
    def __init__(self, round_number):
        self.round_number = round_number

    async def aiter_lines(self):
        if self.round_number == 1:
            lines = [
                "event: content_block_start",
                'data: {"index":0,"content_block":{"type":"tool_use","id":"tool-1","name":"write_memory","input":{}}}',
                "",
                "event: content_block_delta",
                'data: {"index":0,"delta":{"type":"input_json_delta","partial_json":"{\\"content\\":\\"记住雨天\\",\\"scope\\":\\"weiwei-jiao\\",\\"memory_type\\":\\"fact\\",\\"perspective\\":\\"jiao\\",\\"confidential\\":false,\\"importance\\":7,\\"evidence_event_ids\\":[101]}"}}',
                "",
            ]
        else:
            lines = [
                "event: content_block_delta",
                'data: {"delta":{"type":"text_delta","text":"已经记下。"}}',
                "",
            ]
        for line in lines:
            yield line


class ToolStreamContext(StreamContext):
    def __init__(self, round_number): self.round_number = round_number
    async def __aenter__(self): return ToolResponse(self.round_number)


class OpenAIToolResponse(Response):
    def __init__(self, protocol, round_number):
        self.protocol = protocol
        self.round_number = round_number

    async def aiter_lines(self):
        arguments = '{"content":"remember rain","scope":"weiwei-jiao","memory_type":"fact","perspective":"jiao","confidential":false,"importance":7,"evidence_event_ids":[101]}'
        if self.protocol == "openai_chat_completions":
            payload = (
                {"choices": [{"delta": {"tool_calls": [{"index": 0, "id": "tool-1", "function": {"name": "write_memory", "arguments": arguments}}]}}]}
                if self.round_number == 1
                else {"choices": [{"delta": {"content": "saved"}}]}
            )
            lines = [f"data: {json.dumps(payload)}", ""]
        elif self.round_number == 1:
            lines = [
                "event: response.output_item.added",
                f'data: {{"output_index":0,"item":{{"type":"function_call","call_id":"tool-1","name":"write_memory","arguments":{json.dumps(arguments)}}}}}',
                "",
            ]
        else:
            lines = ["event: response.output_text.delta", 'data: {"delta":"saved"}', ""]
        for line in lines:
            yield line


class OpenAIToolTransport(Transport):
    def __init__(self, protocol):
        super().__init__()
        self.protocol = protocol

    async def open_stream(self, **kwargs):
        self.calls.append(kwargs)
        response = OpenAIToolResponse(self.protocol, len(self.calls))

        class Context:
            async def __aenter__(self): return response
            async def __aexit__(self, *args): return False

        return Context()


class FailingSecondToolTransport(OpenAIToolTransport):
    async def open_stream(self, **kwargs):
        self.calls.append(kwargs)
        if len(self.calls) == 1:
            response = OpenAIToolResponse(self.protocol, 1)
        else:
            response = Response()
            response.status_code = 503

        class Context:
            async def __aenter__(self): return response
            async def __aexit__(self, *args): return False

        return Context()


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


@pytest.mark.anyio
async def test_anthropic_tool_call_is_private_staged_and_followed_by_tool_result():
    payload = profile().to_dict()
    payload["capabilities"]["tools"] = True
    tool_profile = ModelProfile.from_dict(payload)
    store = InMemoryActorMemoryToolStore()
    transport = ToolTransport()
    runner = GatewayProviderRunner(
        transport=transport,
        credential_resolver=Resolver(),
        memory_tools=ActorMemoryToolLibrary(store),
    )
    memory_context = ActorMemoryExecutionContext(
        actor_id="jiao", room_id="room_weiwei_jiao",
        conversation_id="conversation-1", generation_request_id="generation-1",
        source_event_id=101, execution_mode="private", profile_id=tool_profile.profile_id,
    )
    context = ContextBundle(
        static_system=("kernel", "actor", "room"), stable_summary="",
        stable_history=(), dynamic_tail=("current event",),
        actor_prompt_version="actor.v1", runtime_kernel_version="kernel.v1",
        room_policy_version="room.v1", tool_schema_hash="actor-memory-tools.v1",
        actor_memory_context=memory_context,
    )
    chunks = [item async for item in runner.run(profile=tool_profile, request=request(), context=context, cache_namespace="cache-tools")]
    assert len(transport.calls) == 2
    assert transport.calls[0]["json_body"]["tools"]
    assert transport.calls[1]["json_body"]["messages"][-1]["content"][0]["type"] == "tool_result"
    assert [item.data["text"] for item in chunks if item.event == "final"] == ["已经记下。"]
    assert await store.list_active() == []
    assert len([stage for stage in store.stages.values() if stage["status"] == "staged"]) == 1


@pytest.mark.anyio
@pytest.mark.parametrize("protocol", ["openai_chat_completions", "openai_responses"])
async def test_openai_tool_call_is_private_staged_and_followed_by_native_tool_result(protocol):
    payload = profile().to_dict()
    payload.update(protocol=protocol, cache_strategy="openai_stable_prefix_v1", requested_cache_ttl=None)
    payload["capabilities"]["tools"] = True
    payload["capabilities"]["cache_strategies"] = ["openai_stable_prefix_v1"]
    payload["capabilities"]["cache_ttls"] = []
    tool_profile = ModelProfile.from_dict(payload)
    store = InMemoryActorMemoryToolStore()
    transport = OpenAIToolTransport(protocol)
    runner = GatewayProviderRunner(
        transport=transport, credential_resolver=Resolver(),
        memory_tools=ActorMemoryToolLibrary(store),
    )
    memory_context = ActorMemoryExecutionContext(
        actor_id="jiao", room_id="room_weiwei_jiao",
        conversation_id="conversation-1", generation_request_id="generation-1",
        source_event_id=101, execution_mode="private", profile_id=tool_profile.profile_id,
    )
    context = ContextBundle(
        static_system=("kernel", "actor", "room"), stable_summary="",
        stable_history=(), dynamic_tail=("current event",),
        actor_prompt_version="actor.v1", runtime_kernel_version="kernel.v1",
        room_policy_version="room.v1", tool_schema_hash="actor-memory-tools.v1",
        actor_memory_context=memory_context,
    )

    chunks = [item async for item in runner.run(profile=tool_profile, request=request(), context=context, cache_namespace="cache-tools")]

    assert len(transport.calls) == 2
    assert transport.calls[0]["json_body"]["tools"]
    second = transport.calls[1]["json_body"]
    if protocol == "openai_chat_completions":
        assert second["messages"][-1]["role"] == "tool"
    else:
        assert second["input"][-1]["type"] == "function_call_output"
    assert [item.data["text"] for item in chunks if item.event == "final"] == ["saved"]
    assert len([stage for stage in store.stages.values() if stage["status"] == "staged"]) == 1


@pytest.mark.anyio
async def test_provider_failure_after_tool_call_discards_private_staged_mutation():
    payload = profile().to_dict()
    payload.update(protocol="openai_chat_completions", cache_strategy="openai_stable_prefix_v1", requested_cache_ttl=None)
    payload["capabilities"]["tools"] = True
    payload["capabilities"]["cache_strategies"] = ["openai_stable_prefix_v1"]
    payload["capabilities"]["cache_ttls"] = []
    tool_profile = ModelProfile.from_dict(payload)
    store = InMemoryActorMemoryToolStore()
    runner = GatewayProviderRunner(
        transport=FailingSecondToolTransport("openai_chat_completions"),
        credential_resolver=Resolver(), memory_tools=ActorMemoryToolLibrary(store),
    )
    memory_context = ActorMemoryExecutionContext(
        actor_id="jiao", room_id="room_weiwei_jiao",
        conversation_id="conversation-1", generation_request_id="generation-1",
        source_event_id=101, execution_mode="private", profile_id=tool_profile.profile_id,
    )
    context = ContextBundle(
        static_system=("kernel", "actor", "room"), stable_summary="",
        stable_history=(), dynamic_tail=("current event",),
        actor_prompt_version="actor.v1", runtime_kernel_version="kernel.v1",
        room_policy_version="room.v1", tool_schema_hash="actor-memory-tools.v1",
        actor_memory_context=memory_context,
    )

    with pytest.raises(ProviderRunUnavailable):
        await _consume(runner.run(profile=tool_profile, request=request(), context=context, cache_namespace="cache-tools"))

    assert {stage["status"] for stage in store.stages.values()} == {"discarded"}
    assert await store.list_active() == []


@pytest.mark.anyio
async def test_anthropic_no_cache_profile_sends_no_cache_control():
    payload = profile().to_dict()
    payload["test_status"] = "passed"
    payload["capabilities"]["cache_strategies"] = ["no_prompt_cache_v1"]
    payload["capabilities"]["cache_ttls"] = []
    payload["cache_strategy"] = "no_prompt_cache_v1"
    payload["requested_cache_ttl"] = None
    uncached = ModelProfile.from_dict(payload)
    transport = Transport()
    runner = GatewayProviderRunner(transport=transport, credential_resolver=Resolver())
    context = ContextBundle(
        static_system=("kernel", "actor", "room"),
        stable_summary="summary",
        stable_history=("old fact",),
        dynamic_tail=("current event",),
        actor_prompt_version="actor.v1",
        runtime_kernel_version="kernel.v1",
        room_policy_version="room.v1",
        tool_schema_hash="tools.none.v1",
    )

    await _consume(
        runner.run(
            profile=uncached,
            request=request(),
            context=context,
            cache_namespace="cache-uncached",
        )
    )

    assert "cache_control" not in str(transport.calls[0]["json_body"])


def test_openai_render_keeps_compressed_summary_before_anchored_history():
    payload = profile().to_dict()
    payload.update(
        protocol="openai_chat_completions",
        cache_strategy="openai_stable_prefix_v1",
        requested_cache_ttl=None,
    )
    payload["capabilities"]["cache_strategies"] = ["openai_stable_prefix_v1"]
    payload["capabilities"]["cache_ttls"] = []
    openai = ModelProfile.from_dict(payload)
    context = ContextBundle(
        static_system=("kernel", "actor", "room"),
        stable_summary="bounded compressed summary",
        stable_history=("anchored fact",),
        dynamic_tail=("current event",),
        actor_prompt_version="actor.v1",
        runtime_kernel_version="kernel.v1",
        room_policy_version="room.v1",
        tool_schema_hash="tools.none.v1",
    )

    rendered = GatewayProviderRunner()._render(
        openai, request(), context, "cache-openai"
    )

    assert rendered.json_body["messages"][1]["content"] == "bounded compressed summary"
    assert rendered.json_body["messages"][2]["content"] == "anchored fact"
    assert rendered.json_body["messages"][3]["content"] == "current event"


def test_openai_chat_render_includes_actor_memory_tools_when_profile_allows_them():
    payload = profile().to_dict()
    payload.update(
        protocol="openai_chat_completions",
        cache_strategy="openai_stable_prefix_v1",
        requested_cache_ttl=None,
    )
    payload["capabilities"]["tools"] = True
    payload["capabilities"]["cache_strategies"] = ["openai_stable_prefix_v1"]
    payload["capabilities"]["cache_ttls"] = []
    openai = ModelProfile.from_dict(payload)
    context = ContextBundle(
        static_system=("kernel", "actor", "room"), stable_summary="",
        stable_history=(), dynamic_tail=("current event",),
        actor_prompt_version="actor.v1", runtime_kernel_version="kernel.v1",
        room_policy_version="room.v1", tool_schema_hash="actor-memory-tools.v1",
    )

    rendered = GatewayProviderRunner()._render(openai, request(), context, "cache-openai-tools")

    assert {tool["function"]["name"] for tool in rendered.json_body["tools"]} == ACTOR_MEMORY_TOOL_NAMES


async def _consume(stream):
    return [item async for item in stream]
