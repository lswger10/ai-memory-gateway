from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from model_execution_contracts import ProviderUsage


@dataclass(frozen=True, slots=True)
class FrozenProbeInput:
    actor_id: str
    conversation_id: str
    profile_id: str
    static_prefix_hash: str
    history_hash: str
    dynamic_tail_hash: str


@dataclass(frozen=True, slots=True)
class ProbeResult:
    status: str
    first: ProviderUsage
    second: ProviderUsage


class ProbeRunner(Protocol):
    async def run_once(self, frozen_input: FrozenProbeInput) -> ProviderUsage: ...


class CacheAcceptanceProbe:
    async def run(
        self, frozen_input: FrozenProbeInput, runner: ProbeRunner
    ) -> ProbeResult:
        empty = ProviderUsage.from_provider_values()
        try:
            first = await runner.run_once(frozen_input)
            second = await runner.run_once(frozen_input)
        except Exception:
            return ProbeResult(status="failed", first=empty, second=empty)
        read = second.cache_read_input_tokens
        cached = second.cached_tokens
        verified = (read is not None and read > 0) or (
            cached is not None and cached > 0
        )
        return ProbeResult(
            status="verified" if verified else "unverified",
            first=first,
            second=second,
        )
