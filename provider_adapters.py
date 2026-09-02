from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping

from cache_strategies import AnthropicPromptLayout, PromptSegment
from model_execution_contracts import ProviderUsage
from model_profiles import ModelProfile


class ProviderAdapterError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class RenderedProviderRequest:
    method: str
    path: str
    json_body: dict[str, Any]


@dataclass(frozen=True, slots=True)
class ProviderProvenance:
    profile_id: str
    profile_revision: int
    provider: str
    protocol: str
    route_id: str
    model: str
    adapter_version: str
    generation_request_id: str
    fallback_used: bool
    fallback_from_profile_id: str | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def build_provider_provenance(
    profile: ModelProfile,
    *,
    generation_request_id: str,
    fallback_used: bool,
    fallback_from_profile_id: str | None,
) -> ProviderProvenance:
    return ProviderProvenance(
        profile_id=profile.profile_id,
        profile_revision=profile.revision,
        provider=profile.provider,
        protocol=profile.protocol,
        route_id=profile.route_id,
        model=profile.model,
        adapter_version=profile.adapter_version,
        generation_request_id=generation_request_id,
        fallback_used=fallback_used,
        fallback_from_profile_id=fallback_from_profile_id,
    )


def resolve_profile_headers(profile: ModelProfile, credential: str) -> dict[str, str]:
    if not profile.headers:
        raise ProviderAdapterError("Model Profile has no explicit header templates")
    return {
        name: value.replace("${credential}", credential)
        for name, value in profile.headers
    }


def _cache_control(ttl: str | None) -> dict[str, str]:
    value = {"type": "ephemeral"}
    if ttl is not None:
        value["ttl"] = ttl
    return value


def _text(segment: PromptSegment) -> str:
    if isinstance(segment.content, str):
        return segment.content
    raise ProviderAdapterError(
        f"segment {segment.source_kind} requires provider-neutral text"
    )


def _anthropic_media(parts: tuple[dict[str, Any], ...]) -> list[dict[str, Any]]:
    rendered = []
    for part in parts:
        if part["kind"] == "text":
            rendered.append({"type": "text", "text": part["text"]})
        elif part["kind"] in {"image", "document"}:
            rendered.append(
                {
                    "type": part["kind"],
                    "source": {
                        "type": "base64",
                        "media_type": part["media_type"],
                        "data": part["data"],
                    },
                }
            )
    return rendered


def _openai_chat_media(parts: tuple[dict[str, Any], ...]) -> list[dict[str, Any]]:
    rendered = []
    for part in parts:
        if part["kind"] == "text":
            rendered.append({"type": "text", "text": part["text"]})
        elif part["kind"] == "image":
            rendered.append(
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:{part['media_type']};base64,{part['data']}"
                    },
                }
            )
        elif part["kind"] == "document":
            rendered.append(
                {
                    "type": "file",
                    "file": {
                        "filename": part["name"],
                        "file_data": f"data:{part['media_type']};base64,{part['data']}",
                    },
                }
            )
    return rendered


def _openai_responses_media(parts: tuple[dict[str, Any], ...]) -> list[dict[str, Any]]:
    rendered = []
    for part in parts:
        if part["kind"] == "text":
            rendered.append({"type": "input_text", "text": part["text"]})
        elif part["kind"] == "image":
            rendered.append(
                {
                    "type": "input_image",
                    "image_url": f"data:{part['media_type']};base64,{part['data']}",
                }
            )
        elif part["kind"] == "document":
            rendered.append(
                {
                    "type": "input_file",
                    "filename": part["name"],
                    "file_data": f"data:{part['media_type']};base64,{part['data']}",
                }
            )
    return rendered


class AnthropicMessagesAdapter:
    protocol = "anthropic_messages"

    def render(
        self,
        *,
        profile: ModelProfile,
        layout: AnthropicPromptLayout,
        max_output_tokens: int,
        apply_cache_control: bool = True,
        media_parts: tuple[dict[str, Any], ...] = (),
    ) -> RenderedProviderRequest:
        if profile.protocol not in {"anthropic_messages", "anthropic_messages_compatible"}:
            raise ProviderAdapterError("Profile protocol is not Anthropic Messages")
        expected_strategy = (
            "anthropic_prefix_anchored_v1"
            if apply_cache_control
            else "no_prompt_cache_v1"
        )
        if profile.cache_strategy != expected_strategy:
            raise ProviderAdapterError("Anthropic layout cache strategy mismatch")
        system = [
            {"type": "text", "text": _text(segment)} for segment in layout.system
        ]
        messages: list[dict[str, Any]] = []
        for segment in layout.stable_messages:
            messages.append(
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": f"[{segment.source_kind}]\n{_text(segment)}",
                        }
                    ],
                }
            )
        if apply_cache_control:
            breakpoint = _cache_control(layout.breakpoint.requested_ttl)
            if messages:
                messages[-1]["content"][-1]["cache_control"] = breakpoint
            else:
                system[-1]["cache_control"] = breakpoint
        for segment in layout.dynamic_messages:
            messages.append(
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": f"[{segment.source_kind}]\n{_text(segment)}",
                        }
                    ],
                }
            )
        if media_parts:
            if not messages or not layout.dynamic_messages:
                messages.append({"role": "user", "content": []})
            messages[-1]["content"].extend(_anthropic_media(media_parts))
        body: dict[str, Any] = {
            "model": profile.model,
            "system": system,
            "messages": messages,
            "max_tokens": max_output_tokens,
            "stream": True,
        }
        if layout.tools:
            if not profile.capabilities.tools:
                raise ProviderAdapterError("Profile does not support tools")
            body["tools"] = list(layout.tools)
        return RenderedProviderRequest(method="POST", path="/v1/messages", json_body=body)

    def parse_usage(self, payload: Mapping[str, Any]) -> ProviderUsage:
        return ProviderUsage.from_provider_values(
            input_tokens=payload.get("input_tokens"),
            output_tokens=payload.get("output_tokens"),
            cache_creation_input_tokens=payload.get("cache_creation_input_tokens"),
            cache_read_input_tokens=payload.get("cache_read_input_tokens"),
            cached_tokens=payload.get("cached_tokens"),
        )


class OpenAIResponsesAdapter:
    protocol = "openai_responses"

    def render(
        self,
        *,
        profile: ModelProfile,
        instructions: str,
        input_items: tuple[dict[str, Any], ...],
        prompt_cache_key: str | None,
        max_output_tokens: int,
        media_parts: tuple[dict[str, Any], ...] = (),
    ) -> RenderedProviderRequest:
        if profile.protocol != self.protocol:
            raise ProviderAdapterError("Profile protocol is not OpenAI Responses")
        body: dict[str, Any] = {
            "model": profile.model,
            "instructions": instructions,
            "input": list(input_items),
            "max_output_tokens": max_output_tokens,
            "stream": True,
            "store": False,
        }
        if media_parts:
            if not body["input"]:
                body["input"].append({"role": "user", "content": []})
            body["input"][-1]["content"].extend(
                _openai_responses_media(media_parts)
            )
        if prompt_cache_key is not None:
            if profile.cache_strategy != "openai_stable_prefix_v1":
                raise ProviderAdapterError("prompt_cache_key requires OpenAI cache strategy")
            body["prompt_cache_key"] = prompt_cache_key
        return RenderedProviderRequest(method="POST", path="/v1/responses", json_body=body)

    def parse_usage(self, payload: Mapping[str, Any]) -> ProviderUsage:
        details = payload.get("input_tokens_details")
        cached = details.get("cached_tokens") if isinstance(details, Mapping) else None
        return ProviderUsage.from_provider_values(
            input_tokens=payload.get("input_tokens"),
            output_tokens=payload.get("output_tokens"),
            cached_tokens=cached,
        )


class OpenAIChatCompletionsAdapter:
    protocol = "openai_chat_completions"

    def render(
        self,
        *,
        profile: ModelProfile,
        system_content: str,
        messages: tuple[dict[str, Any], ...],
        prompt_cache_key: str | None,
        max_output_tokens: int,
        media_parts: tuple[dict[str, Any], ...] = (),
    ) -> RenderedProviderRequest:
        if profile.protocol != self.protocol:
            raise ProviderAdapterError("Profile protocol is not Chat Completions")
        body: dict[str, Any] = {
            "model": profile.model,
            "messages": [
                {"role": "system", "content": system_content},
                *messages,
            ],
            "max_tokens": max_output_tokens,
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        if media_parts:
            if len(body["messages"]) == 1:
                body["messages"].append({"role": "user", "content": []})
            target = body["messages"][-1]
            if isinstance(target["content"], str):
                target["content"] = [
                    {"type": "text", "text": target["content"]}
                ]
            target["content"].extend(_openai_chat_media(media_parts))
        if prompt_cache_key is not None:
            if profile.cache_strategy != "openai_stable_prefix_v1":
                raise ProviderAdapterError("prompt_cache_key requires OpenAI cache strategy")
            body["prompt_cache_key"] = prompt_cache_key
        return RenderedProviderRequest(
            method="POST", path="/v1/chat/completions", json_body=body
        )

    def parse_usage(self, payload: Mapping[str, Any]) -> ProviderUsage:
        details = payload.get("prompt_tokens_details")
        cached = details.get("cached_tokens") if isinstance(details, Mapping) else None
        return ProviderUsage.from_provider_values(
            input_tokens=payload.get("prompt_tokens"),
            output_tokens=payload.get("completion_tokens"),
            cached_tokens=cached,
        )
