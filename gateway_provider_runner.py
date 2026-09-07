from __future__ import annotations

import asyncio
import json
import uuid
import anyio
from typing import Any, AsyncIterator, Mapping

from actor_memory_tools import ActorMemoryToolLibrary, actor_memory_tool_definitions
from cache_strategies import (
    AnthropicPrefixAnchoredV1,
    AnthropicPromptLayout,
    CacheBreakpoint,
    PromptSegment,
)
from model_execution import ContextBundle, ProviderChunk, ProviderRunUnavailable
from model_execution_contracts import GatewayExecutionRequest, ProviderUsage
from model_usage_store import UsageRecordingError
from model_profiles import ModelProfile
from media_materialization import (
    RelayMediaReader,
    prepare_media_for_profile,
    render_media_tail,
)
from provider_adapters import (
    AnthropicMessagesAdapter,
    OpenAIChatCompletionsAdapter,
    OpenAIResponsesAdapter,
    resolve_profile_headers,
)
from provider_transport import EnvironmentCredentialResolver, PooledHttpTransport


def _observed_cache_support(usage: ProviderUsage) -> str:
    values = (usage.cache_read_input_tokens, usage.cached_tokens)
    if any(value is not None and value > 0 for value in values):
        return "verified"
    if all(value is None for value in values) and usage.cache_creation_input_tokens is None:
        return "unavailable"
    return "unverified"


class GatewayProviderRunner:
    def __init__(
        self,
        *,
        transport: PooledHttpTransport | None = None,
        credential_resolver: EnvironmentCredentialResolver | None = None,
        media_reader: RelayMediaReader | None = None,
        memory_tools: ActorMemoryToolLibrary | None = None,
    ) -> None:
        self.transport = transport or PooledHttpTransport()
        self.credentials = credential_resolver or EnvironmentCredentialResolver()
        self.media_reader = media_reader
        self.memory_tools = memory_tools

    def _render(
        self,
        profile: ModelProfile,
        request: GatewayExecutionRequest,
        context: ContextBundle,
        cache_namespace: str,
        max_output_tokens: int | None = None,
        media_parts: tuple[dict[str, Any], ...] = (),
    ):
        maximum = (
            max_output_tokens
            if max_output_tokens is not None
            else (512 if request.execution_kind == "probe" else 12000)
        )
        system_kinds = ("runtime_kernel", "actor_prompt", "room_policy")
        stable = tuple(
            PromptSegment(kind, text)
            for kind, text in zip(system_kinds, context.static_system, strict=True)
        )
        if context.stable_summary:
            stable += (PromptSegment("compressed_summary", context.stable_summary),)
        stable += tuple(
            PromptSegment("factual_history", text) for text in context.stable_history
        )
        dynamic = tuple(
            PromptSegment("current_event", text) for text in context.dynamic_tail
        )
        tools = (
            actor_memory_tool_definitions()
            if profile.capabilities.tools
            and context.tool_schema_hash == "actor-memory-tools.v1"
            else ()
        )
        if profile.protocol in {"anthropic_messages", "anthropic_messages_compatible"}:
            cache_enabled = profile.cache_strategy == "anthropic_prefix_anchored_v1"
            if cache_enabled:
                layout = AnthropicPrefixAnchoredV1().build_layout(
                    tools=tools,
                    stable_segments=stable,
                    dynamic_segments=dynamic,
                    capabilities=profile.capabilities,
                    requested_ttl=profile.requested_cache_ttl,
                )
            elif profile.cache_strategy == "no_prompt_cache_v1":
                layout = AnthropicPromptLayout(
                    tools=tools,
                    system=stable[:3],
                    stable_messages=stable[3:],
                    dynamic_messages=dynamic,
                    breakpoint=CacheBreakpoint("none", None),
                )
            else:
                raise ProviderRunUnavailable(
                    "Anthropic route received incompatible cache strategy"
                )
            return AnthropicMessagesAdapter().render(
                profile=profile,
                layout=layout,
                max_output_tokens=maximum,
                apply_cache_control=cache_enabled,
                media_parts=media_parts,
            )

        instructions = "\n\n".join(context.static_system)
        stable_text = (
            *((context.stable_summary,) if context.stable_summary else ()),
            *context.stable_history,
        )
        messages = tuple(
            {"role": "user", "content": text}
            for text in (*stable_text, *context.dynamic_tail)
        )
        cache_key = (
            cache_namespace
            if profile.cache_strategy == "openai_stable_prefix_v1"
            else None
        )
        if profile.protocol == "openai_responses":
            input_items = tuple(
                {"role": item["role"], "content": [{"type": "input_text", "text": item["content"]}]}
                for item in messages
            )
            return OpenAIResponsesAdapter().render(
                profile=profile,
                instructions=instructions,
                input_items=input_items,
                prompt_cache_key=cache_key,
                max_output_tokens=maximum,
                media_parts=media_parts,
                tools=tools,
            )
        if profile.protocol == "openai_chat_completions":
            return OpenAIChatCompletionsAdapter().render(
                profile=profile,
                system_content=instructions,
                messages=messages,
                prompt_cache_key=cache_key,
                max_output_tokens=maximum,
                media_parts=media_parts,
                tools=tools,
            )
        raise ProviderRunUnavailable("unsupported provider protocol")

    async def run(
        self,
        *,
        profile: ModelProfile,
        request: GatewayExecutionRequest,
        context: ContextBundle,
        cache_namespace: str,
        max_output_tokens: int | None = None,
        on_attempt=None,
    ) -> AsyncIterator[ProviderChunk]:
        try:
            credential = self.credentials.resolve(profile.credential_ref)
            headers = resolve_profile_headers(profile, credential)
            media_parts: tuple[dict[str, Any], ...] = ()
            if context.current_media_references:
                if self.media_reader is None:
                    raise ProviderRunUnavailable("Relay media reader is not configured")
                prepared = await prepare_media_for_profile(
                    profile, context.current_media_references, self.media_reader
                )
                media_parts = render_media_tail(profile, prepared)
            rendered = self._render(
                profile,
                request,
                context,
                cache_namespace,
                max_output_tokens=max_output_tokens,
                media_parts=media_parts,
            )
            body = rendered.json_body
            aggregate_usage = None
            provider_usage_received = False
            for _ in range(8):
                attempt_id = str(uuid.uuid4())
                attempt_usage = ProviderUsage.from_provider_values()
                attempt_usage_received = False
                attempt_status = "failed"
                round_items = []
                try:
                    stream_context = await self.transport.open_stream(
                        pool_key=profile.route_id, base_url=profile.base_url, headers=headers,
                        method=rendered.method, path=rendered.path, json_body=body)
                    async with stream_context as response:
                        if response.status_code >= 400:
                            raise ProviderRunUnavailable(f"provider HTTP {response.status_code}")
                        parser = (self._anthropic if profile.protocol in {"anthropic_messages", "anthropic_messages_compatible"}
                                  else self._openai_responses if profile.protocol == "openai_responses" else self._openai_chat)
                        async for item in parser(response, profile, request):
                            if item.event == "usage":
                                attempt_usage = item.data["usage"]
                                attempt_usage_received |= bool(item.data.get("provider_usage_received"))
                            elif item.event == "delta":
                                yield item
                            else:
                                round_items.append(item)
                    if any(item.event == "final" and not str(item.data.get("text", "")).strip()
                           for item in round_items):
                        raise ProviderRunUnavailable("provider completed without reply text")
                    attempt_status = "succeeded"
                except (asyncio.CancelledError, GeneratorExit):
                    attempt_status = "cancelled"
                    raise
                finally:
                    if on_attempt is not None:
                        await on_attempt(attempt_id, attempt_usage, attempt_status,
                            attempt_usage_received, _observed_cache_support(attempt_usage))
                tool_item = next((item for item in round_items if item.event == "tool_calls"), None)
                aggregate_usage = attempt_usage if aggregate_usage is None else _add_usage(aggregate_usage, attempt_usage)
                provider_usage_received |= attempt_usage_received
                if tool_item is None:
                    for item in round_items:
                        if item.event not in {"usage", "tool_calls"}:
                            yield item
                    yield ProviderChunk("usage", {
                        "usage": aggregate_usage,
                        "observed_cache_support": _observed_cache_support(aggregate_usage),
                        "provider_usage_received": provider_usage_received,
                    })
                    return
                if self.memory_tools is None or context.actor_memory_context is None:
                    raise ProviderRunUnavailable("provider requested unavailable memory tools")
                calls = tool_item.data["calls"]
                results = []
                for call in calls:
                    result = await self.memory_tools.call(
                        context.actor_memory_context,
                        f"{profile.profile_id}:{call['id']}",
                        call["name"], call["arguments"],
                    )
                    results.append(result)
                body = _continue_with_tool_results(profile.protocol, body, calls, results)
            raise ProviderRunUnavailable("provider tool loop exceeded limit")
        except (ProviderRunUnavailable, UsageRecordingError, asyncio.CancelledError, GeneratorExit):
            if self.memory_tools is not None and context.actor_memory_context is not None:
                with anyio.fail_after(5, shield=True):
                    await self.memory_tools.discard(context.actor_memory_context)
            raise
        except Exception as exc:
            if self.memory_tools is not None and context.actor_memory_context is not None:
                with anyio.fail_after(5, shield=True):
                    await self.memory_tools.discard(context.actor_memory_context)
            raise ProviderRunUnavailable("provider transport failed") from exc

    async def _anthropic(self, response, profile, request):
        adapter = AnthropicMessagesAdapter()
        text = ""
        usage_values: dict[str, Any] = {}
        calls: dict[int, dict[str, Any]] = {}
        terminal_seen = False
        async for event, data in _sse_json(response):
            if event == "error" or data.get("error"):
                raise ProviderRunUnavailable("provider stream reported an error")
            if event == "message_stop":
                terminal_seen = True
            if event == "content_block_start":
                block = data.get("content_block", {})
                if isinstance(block, Mapping) and block.get("type") == "tool_use":
                    calls[int(data.get("index", 0))] = {
                        "id": str(block.get("id", "")), "name": str(block.get("name", "")),
                        "json": json.dumps(block.get("input", {})) if block.get("input") else "",
                    }
            if event == "content_block_delta":
                delta = data.get("delta", {})
                value = delta.get("text") if isinstance(delta, Mapping) else None
                if isinstance(value, str):
                    text += value
                    yield ProviderChunk("delta", {"text": value})
                if isinstance(delta, Mapping) and delta.get("type") == "input_json_delta":
                    calls[int(data.get("index", 0))]["json"] += str(delta.get("partial_json", ""))
            if event == "message_start":
                message = data.get("message", {})
                if isinstance(message, Mapping) and isinstance(message.get("usage"), Mapping):
                    usage_values.update(message["usage"])
                    yield _usage_chunk(adapter, usage_values)
            if event == "message_delta" and isinstance(data.get("usage"), Mapping):
                usage_values.update(data["usage"])
                yield _usage_chunk(adapter, usage_values)
        if not terminal_seen:
            raise ProviderRunUnavailable("provider stream omitted its terminal event")
        if calls:
            yield ProviderChunk("tool_calls", {"calls": _finish_calls(calls)})
        elif request.execution_kind == "probe":
            yield ProviderChunk("probe", _parse_probe(text))
        if not calls:
            yield ProviderChunk("final", {"text": text})
        usage = adapter.parse_usage(usage_values)
        yield ProviderChunk(
            "usage",
            {
                "usage": usage,
                "observed_cache_support": _observed_cache_support(usage),
                "provider_usage_received": bool(usage_values),
            },
        )

    async def _openai_chat(self, response, profile, request):
        adapter = OpenAIChatCompletionsAdapter()
        text = ""
        usage_values: dict[str, Any] = {}
        calls: dict[int, dict[str, Any]] = {}
        terminal_seen = False
        async for event, data in _sse_json(response):
            if event == "done":
                terminal_seen = True
            if event == "error" or data.get("error"):
                raise ProviderRunUnavailable("provider stream reported an error")
            choices = data.get("choices")
            if isinstance(choices, list) and choices:
                terminal_seen |= choices[0].get("finish_reason") is not None
                delta = choices[0].get("delta", {})
                value = delta.get("content") if isinstance(delta, Mapping) else None
                if isinstance(value, str):
                    text += value
                    yield ProviderChunk("delta", {"text": value})
                for call in delta.get("tool_calls", ()) if isinstance(delta, Mapping) else ():
                    index = int(call.get("index", 0))
                    current = calls.setdefault(index, {"id": "", "name": "", "json": ""})
                    current["id"] += str(call.get("id", ""))
                    function = call.get("function", {})
                    if isinstance(function, Mapping):
                        current["name"] += str(function.get("name", ""))
                        current["json"] += str(function.get("arguments", ""))
            if isinstance(data.get("usage"), Mapping):
                usage_values.update(data["usage"])
                yield _usage_chunk(adapter, usage_values)
        if not terminal_seen:
            raise ProviderRunUnavailable("provider stream omitted its terminal event")
        if calls:
            yield ProviderChunk("tool_calls", {"calls": _finish_calls(calls)})
        elif request.execution_kind == "probe":
            yield ProviderChunk("probe", _parse_probe(text))
        if not calls:
            yield ProviderChunk("final", {"text": text})
        usage = adapter.parse_usage(usage_values)
        yield ProviderChunk(
            "usage",
            {
                "usage": usage,
                "observed_cache_support": _observed_cache_support(usage),
                "provider_usage_received": bool(usage_values),
            },
        )

    async def _openai_responses(self, response, profile, request):
        adapter = OpenAIResponsesAdapter()
        text = ""
        usage_values: dict[str, Any] = {}
        calls: dict[int, dict[str, Any]] = {}
        terminal_seen = False
        async for event, data in _sse_json(response):
            if event == "error" or data.get("error"):
                raise ProviderRunUnavailable("provider stream reported an error")
            if event == "response.output_text.delta":
                value = data.get("delta")
                if isinstance(value, str):
                    text += value
                    yield ProviderChunk("delta", {"text": value})
            if event == "response.output_item.added":
                item = data.get("item", {})
                if isinstance(item, Mapping) and item.get("type") == "function_call":
                    calls[int(data.get("output_index", 0))] = {
                        "id": str(item.get("call_id") or item.get("id") or ""),
                        "name": str(item.get("name", "")), "json": str(item.get("arguments", "")),
                    }
            if event in {"response.function_call_arguments.delta", "response.function_call_arguments.done"}:
                current = calls.setdefault(int(data.get("output_index", 0)), {"id": str(data.get("call_id", "")), "name": str(data.get("name", "")), "json": ""})
                fragment = data.get("delta") if event.endswith(".delta") else data.get("arguments")
                if fragment and (event.endswith(".delta") or not current["json"]):
                    current["json"] += str(fragment)
            response_value = data.get("response")
            if event in {"response.completed", "response.failed", "response.incomplete"} and isinstance(response_value, Mapping):
                if isinstance(response_value.get("usage"), Mapping):
                    usage_values.update(response_value["usage"])
                    yield _usage_chunk(adapter, usage_values)
            if event in {"response.failed", "response.incomplete"}:
                raise ProviderRunUnavailable("provider response did not complete")
            if event == "response.completed":
                terminal_seen = True
        if not terminal_seen:
            raise ProviderRunUnavailable("provider stream omitted its terminal event")
        if calls:
            yield ProviderChunk("tool_calls", {"calls": _finish_calls(calls)})
        elif request.execution_kind == "probe":
            yield ProviderChunk("probe", _parse_probe(text))
        if not calls:
            yield ProviderChunk("final", {"text": text})
        usage = adapter.parse_usage(usage_values)
        yield ProviderChunk(
            "usage",
            {
                "usage": usage,
                "observed_cache_support": _observed_cache_support(usage),
                "provider_usage_received": bool(usage_values),
            },
        )


def _finish_calls(calls: Mapping[int, Mapping[str, str]]) -> list[dict[str, Any]]:
    result = []
    for _, call in sorted(calls.items()):
        try:
            arguments = json.loads(call["json"] or "{}")
        except json.JSONDecodeError as exc:
            raise ProviderRunUnavailable("provider tool arguments were not valid JSON") from exc
        if not call["id"] or not call["name"] or not isinstance(arguments, dict):
            raise ProviderRunUnavailable("provider tool call was incomplete")
        result.append({"id": call["id"], "name": call["name"], "arguments": arguments})
    return result


def _add_usage(left: ProviderUsage, right: ProviderUsage) -> ProviderUsage:
    def add(a, b):
        return None if a is None or b is None else a + b
    return ProviderUsage.from_provider_values(
        input_tokens=add(left.input_tokens, right.input_tokens),
        output_tokens=add(left.output_tokens, right.output_tokens),
        cache_creation_input_tokens=add(left.cache_creation_input_tokens, right.cache_creation_input_tokens),
        cache_read_input_tokens=add(left.cache_read_input_tokens, right.cache_read_input_tokens),
        cached_tokens=add(left.cached_tokens, right.cached_tokens),
    )


def _continue_with_tool_results(protocol: str, body: dict[str, Any], calls, results) -> dict[str, Any]:
    body = json.loads(json.dumps(body))
    if protocol in {"anthropic_messages", "anthropic_messages_compatible"}:
        body["messages"].append({"role": "assistant", "content": [
            {"type": "tool_use", "id": call["id"], "name": call["name"], "input": call["arguments"]}
            for call in calls
        ]})
        body["messages"].append({"role": "user", "content": [
            {"type": "tool_result", "tool_use_id": call["id"], "content": json.dumps(result, ensure_ascii=False)}
            for call, result in zip(calls, results, strict=True)
        ]})
    elif protocol == "openai_chat_completions":
        body["messages"].append({"role": "assistant", "tool_calls": [
            {"id": call["id"], "type": "function", "function": {"name": call["name"], "arguments": json.dumps(call["arguments"], ensure_ascii=False)}}
            for call in calls
        ]})
        body["messages"].extend(
            {"role": "tool", "tool_call_id": call["id"], "content": json.dumps(result, ensure_ascii=False)}
            for call, result in zip(calls, results, strict=True)
        )
    else:
        body["input"].extend(
            item
            for call, result in zip(calls, results, strict=True)
            for item in (
                {"type": "function_call", "call_id": call["id"], "name": call["name"], "arguments": json.dumps(call["arguments"], ensure_ascii=False)},
                {"type": "function_call_output", "call_id": call["id"], "output": json.dumps(result, ensure_ascii=False)},
            )
        )
    return body


async def _sse_json(response) -> AsyncIterator[tuple[str, dict[str, Any]]]:
    event = "message"
    data_lines: list[str] = []
    async for line in response.aiter_lines():
        if not line:
            if data_lines:
                raw = "\n".join(data_lines)
                if raw == "[DONE]":
                    yield "done", {}
                else:
                    value = json.loads(raw)
                    if isinstance(value, dict):
                        yield event, value
            event, data_lines = "message", []
        elif line.startswith("event:"):
            event = line.removeprefix("event:").strip()
        elif line.startswith("data:"):
            data_lines.append(line.removeprefix("data:").lstrip())


def _parse_probe(text: str) -> dict[str, Any]:
    try:
        value = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ProviderRunUnavailable("provider probe was not valid JSON") from exc
    if not isinstance(value, dict):
        raise ProviderRunUnavailable("provider probe was not an object")
    return value


def _usage_chunk(adapter, values):
    usage = adapter.parse_usage(values)
    return ProviderChunk("usage", {"usage": usage,
        "observed_cache_support": _observed_cache_support(usage),
        "provider_usage_received": bool(values)})
