from __future__ import annotations

from model_usage_store import ExecutionReceipt


def _cache_outcome(receipt: ExecutionReceipt) -> str:
    usage = receipt.usage
    if (usage.cache_read_input_tokens or 0) > 0 or (usage.cached_tokens or 0) > 0:
        return "read_hit"
    if (usage.cache_creation_input_tokens or 0) > 0:
        return "write_only_unverified"
    if (
        usage.cache_creation_input_tokens is None
        and usage.cache_read_input_tokens is None
        and usage.cached_tokens is None
    ):
        return "metrics_unavailable"
    return "miss"


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
                "cache_verified": outcome == "read_hit",
                "input_tokens": receipt.usage.input_tokens,
                "output_tokens": receipt.usage.output_tokens,
                "cache_creation_input_tokens": (
                    receipt.usage.cache_creation_input_tokens
                ),
                "cache_read_input_tokens": receipt.usage.cache_read_input_tokens,
                "cached_tokens": receipt.usage.cached_tokens,
                "fallback_used": receipt.fallback_used,
                "fallback_from_profile_id": receipt.fallback_from_profile_id,
            }
        )
    return tuple(result)
