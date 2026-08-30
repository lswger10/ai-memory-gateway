from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from model_profiles import ProfileCapabilities


class CacheStrategyError(ValueError):
    pass


_SYSTEM_KINDS = {"runtime_kernel", "actor_prompt", "room_policy"}
_STABLE_MESSAGE_KINDS = {
    "compressed_summary",
    "factual_history",
    "older_image_description",
}
_DYNAMIC_KINDS = {
    "current_time",
    "request_metadata",
    "retrieved_memory",
    "relationship_summary",
    "burst_stance",
    "current_event",
    "group_burst",
    "bedroom_context",
    "debug_trace",
    "recent_raw_image",
}


@dataclass(frozen=True, slots=True)
class PromptSegment:
    source_kind: str
    content: Any

    def __post_init__(self) -> None:
        if not isinstance(self.source_kind, str) or not self.source_kind:
            raise CacheStrategyError("source_kind must be a non-empty string")
        if self.content is None:
            raise CacheStrategyError("segment content cannot be null")


@dataclass(frozen=True, slots=True)
class CacheBreakpoint:
    after_source_kind: str
    requested_ttl: str | None


@dataclass(frozen=True, slots=True)
class AnthropicPromptLayout:
    tools: tuple[dict[str, Any], ...]
    system: tuple[PromptSegment, ...]
    stable_messages: tuple[PromptSegment, ...]
    dynamic_messages: tuple[PromptSegment, ...]
    breakpoint: CacheBreakpoint
    provider_order: tuple[str, str, str] = ("tools", "system", "messages")


class AnthropicPrefixAnchoredV1:
    strategy_id = "anthropic_prefix_anchored_v1"

    def build_layout(
        self,
        *,
        tools: tuple[dict[str, Any], ...],
        stable_segments: tuple[PromptSegment, ...],
        dynamic_segments: tuple[PromptSegment, ...],
        capabilities: ProfileCapabilities,
        requested_ttl: str | None,
    ) -> AnthropicPromptLayout:
        if self.strategy_id not in capabilities.cache_strategies:
            raise CacheStrategyError("route did not verify anthropic cache strategy")
        if requested_ttl is not None and requested_ttl not in capabilities.cache_ttls:
            raise CacheStrategyError(
                f"route did not verify requested cache TTL {requested_ttl}"
            )
        system: list[PromptSegment] = []
        messages: list[PromptSegment] = []
        reached_messages = False
        for segment in stable_segments:
            if segment.source_kind in _DYNAMIC_KINDS:
                raise CacheStrategyError(
                    f"{segment.source_kind} is forbidden in stable prefix"
                )
            if segment.source_kind in _SYSTEM_KINDS:
                if reached_messages:
                    raise CacheStrategyError(
                        "system segments cannot appear after stable history"
                    )
                system.append(segment)
            elif segment.source_kind in _STABLE_MESSAGE_KINDS:
                reached_messages = True
                messages.append(segment)
            else:
                raise CacheStrategyError(
                    f"unsupported stable prefix source: {segment.source_kind}"
                )
        if not system:
            raise CacheStrategyError("stable prefix requires static system content")
        for segment in dynamic_segments:
            if segment.source_kind not in _DYNAMIC_KINDS:
                raise CacheStrategyError(
                    f"{segment.source_kind} does not belong in dynamic tail"
                )
        after = messages[-1].source_kind if messages else system[-1].source_kind
        return AnthropicPromptLayout(
            tools=tuple(tools),
            system=tuple(system),
            stable_messages=tuple(messages),
            dynamic_messages=tuple(dynamic_segments),
            breakpoint=CacheBreakpoint(
                after_source_kind=after,
                requested_ttl=requested_ttl,
            ),
        )


class OpenAIStablePrefixV1:
    strategy_id = "openai_stable_prefix_v1"


class NoPromptCacheV1:
    strategy_id = "no_prompt_cache_v1"
