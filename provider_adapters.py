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


class AnthropicMessagesAdapter:
    protocol = "anthropic_messages"

    def render(
        self,
        *,
        profile: ModelProfile,
        layout: AnthropicPromptLayout,
        max_output_tokens: int,
        apply_cache_control: bool = True,
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
        }
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
