import pytest

from cache_strategies import (
    AnthropicPrefixAnchoredV1,
    CacheStrategyError,
    PromptSegment,
)
from model_profiles import ProfileCapabilities
from model_usage_store import build_cache_namespace


def _capabilities(*, ttls=("5m",)):
    return ProfileCapabilities.from_dict(
        {
            "streaming": True,
            "structured_output": False,
            "tools": True,
            "reasoning_controls": False,
            "cache_strategies": ["anthropic_prefix_anchored_v1"],
            "cache_ttls": list(ttls),
            "usage_fields": [
                "input_tokens",
                "output_tokens",
                "cache_creation_input_tokens",
                "cache_read_input_tokens",
            ],
        }
    )


def _stable_segments():
    return (
        PromptSegment("runtime_kernel", "kernel-v1"),
        PromptSegment("actor_prompt", "JIAO prompt"),
        PromptSegment("room_policy", "private policy"),
        PromptSegment("compressed_summary", "bounded summary"),
        PromptSegment("factual_history", "weiwei: hello"),
    )


def _dynamic_segments():
    return (
        PromptSegment("retrieved_memory", "authorized memory"),
        PromptSegment("current_event", "weiwei: now"),
    )


def test_anthropic_prefix_orders_tools_system_then_messages():
    layout = AnthropicPrefixAnchoredV1().build_layout(
        tools=({"name": "propose_memory_candidate", "input_schema": {}},),
        stable_segments=_stable_segments(),
        dynamic_segments=_dynamic_segments(),
        capabilities=_capabilities(),
        requested_ttl="5m",
    )

    assert layout.provider_order == ("tools", "system", "messages")
    assert [item.source_kind for item in layout.system] == [
        "runtime_kernel",
        "actor_prompt",
        "room_policy",
    ]
    assert [item.source_kind for item in layout.stable_messages] == [
        "compressed_summary",
        "factual_history",
    ]
    assert layout.breakpoint.after_source_kind == "factual_history"
    assert layout.dynamic_messages == _dynamic_segments()


@pytest.mark.parametrize("source_kind", ["current_time", "request_metadata"])
def test_dynamic_time_before_breakpoint_is_rejected(source_kind):
    with pytest.raises(CacheStrategyError, match="stable prefix"):
        AnthropicPrefixAnchoredV1().build_layout(
            tools=(),
            stable_segments=(*_stable_segments(), PromptSegment(source_kind, "changes")),
            dynamic_segments=_dynamic_segments(),
            capabilities=_capabilities(),
            requested_ttl="5m",
        )


def test_retrieved_memory_before_breakpoint_is_rejected():
    with pytest.raises(CacheStrategyError, match="retrieved_memory"):
        AnthropicPrefixAnchoredV1().build_layout(
            tools=(),
            stable_segments=(*_stable_segments(), PromptSegment("retrieved_memory", "x")),
            dynamic_segments=(PromptSegment("current_event", "now"),),
            capabilities=_capabilities(),
            requested_ttl="5m",
        )


def test_bedroom_transcript_stays_after_breakpoint():
    bedroom = PromptSegment("bedroom_context", "temporary transcript")
    layout = AnthropicPrefixAnchoredV1().build_layout(
        tools=(),
        stable_segments=_stable_segments(),
        dynamic_segments=(bedroom, PromptSegment("current_event", "now")),
        capabilities=_capabilities(),
        requested_ttl="5m",
    )
    assert bedroom not in layout.stable_messages
    assert bedroom in layout.dynamic_messages


def test_unsupported_route_cannot_request_one_hour_ttl():
    with pytest.raises(CacheStrategyError, match="1h"):
        AnthropicPrefixAnchoredV1().build_layout(
            tools=(),
            stable_segments=_stable_segments(),
            dynamic_segments=_dynamic_segments(),
            capabilities=_capabilities(ttls=("5m",)),
            requested_ttl="1h",
        )


def test_actor_prompt_and_room_policy_versions_invalidate_namespace():
    common = {
        "actor_id": "jiao",
        "conversation_id": "conversation-1",
        "profile_id": "profile-a",
        "profile_revision": 1,
        "execution_mode": "private",
        "actor_prompt_version": "jiao.v1",
        "runtime_kernel_version": "kernel.v1",
        "room_policy_version": "private.v1",
        "tool_schema_hash": "tools.v1",
        "cache_strategy_version": "anthropic_prefix_anchored_v1",
    }
    baseline = build_cache_namespace(**common)
    assert build_cache_namespace(**{**common, "actor_prompt_version": "jiao.v2"}) != baseline
    assert build_cache_namespace(**{**common, "room_policy_version": "private.v2"}) != baseline
    assert build_cache_namespace(**{**common, "actor_id": "laoke"}) != baseline


def test_dynamic_segment_cannot_be_omitted_from_tail_by_misclassification():
    with pytest.raises(CacheStrategyError, match="dynamic tail"):
        AnthropicPrefixAnchoredV1().build_layout(
            tools=(),
            stable_segments=_stable_segments(),
            dynamic_segments=(PromptSegment("factual_history", "wrong zone"),),
            capabilities=_capabilities(),
            requested_ttl="5m",
        )
