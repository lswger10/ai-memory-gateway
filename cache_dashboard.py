from __future__ import annotations

from model_usage_store import ExecutionReceipt


def _cache_outcome(receipt: ExecutionReceipt) -> str:
    usage = receipt.usage
    if (usage.cache_read_input_tokens or 0) > 0 or (usage.cached_tokens or 0) > 0:
        return "HIT"
    provider_usage_evidence = receipt.provider_usage_received or any(
        value is not None
        for value in (
            usage.input_tokens,
            usage.output_tokens,
            usage.cache_creation_input_tokens,
            usage.cache_read_input_tokens,
            usage.cached_tokens,
        )
    )
    if not provider_usage_evidence:
        return "UNOBSERVABLE"
    if receipt.protocol in {"anthropic_messages", "anthropic_messages_compatible"}:
        observable = (
            usage.cache_creation_input_tokens is not None
            and usage.cache_read_input_tokens is not None
        )
    elif receipt.protocol in {"openai_responses", "openai_chat_completions"}:
        observable = usage.cached_tokens is not None
    else:
        observable = any(
            value is not None
            for value in (
                usage.cache_creation_input_tokens,
                usage.cache_read_input_tokens,
                usage.cached_tokens,
            )
        )
    return "OBSERVED_MISS" if observable else "UNOBSERVABLE"


def build_cache_usage_view(
    receipts: tuple[ExecutionReceipt, ...],
) -> tuple[dict, ...]:
    result = []
    for receipt in receipts:
        outcome = _cache_outcome(receipt)
        result.append(
            {
                "generation_request_id": receipt.generation_request_id,
                "actor_id": receipt.actor_id,
                "room_id": receipt.room_id,
                "profile_id": receipt.profile_id,
                "provider": receipt.provider,
                "protocol": receipt.protocol,
                "route_id": receipt.route_id,
                "model": receipt.model,
                "cache_strategy": receipt.cache_strategy,
                "requested_cache_ttl": receipt.requested_cache_ttl,
                "observed_cache_support": receipt.observed_cache_support,
                "cache_outcome": outcome,
                "cache_verified": outcome == "HIT",
                "input_tokens": receipt.usage.input_tokens,
                "output_tokens": receipt.usage.output_tokens,
                "cache_creation_input_tokens": (
                    receipt.usage.cache_creation_input_tokens
                ),
                "cache_read_input_tokens": receipt.usage.cache_read_input_tokens,
                "cached_tokens": receipt.usage.cached_tokens,
                "fallback_used": receipt.fallback_used,
                "fallback_from_profile_id": receipt.fallback_from_profile_id,
                "stable_prefix_hash": receipt.stable_prefix_hash,
                "prompt_cache_key": receipt.prompt_cache_key,
                "runtime_kernel_version": receipt.runtime_kernel_version,
                "persona_version": receipt.persona_version,
                "room_policy_version": receipt.room_policy_version,
                "tool_schema_hash": receipt.tool_schema_hash,
                "summary_version": receipt.summary_version,
                "compressed_up_to_event_id": receipt.compressed_up_to_event_id,
                "provider_usage_received": receipt.provider_usage_received,
            }
        )
    return tuple(result)


def build_cache_observability_summary(
    receipts: tuple[ExecutionReceipt, ...],
) -> dict:
    outcomes = tuple(_cache_outcome(receipt) for receipt in receipts)
    hits = outcomes.count("HIT")
    misses = outcomes.count("OBSERVED_MISS")
    unobservable = outcomes.count("UNOBSERVABLE")
    observable = hits + misses
    total = len(outcomes)
    return {
        "total_requests": total,
        "observable_requests": observable,
        "hit_requests": hits,
        "observed_miss_requests": misses,
        "unobservable_requests": unobservable,
        "observable_hit_ratio": hits / observable if observable else None,
        "telemetry_coverage_ratio": observable / total if total else None,
    }
