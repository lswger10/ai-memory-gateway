import pytest

from cache_probe import CacheAcceptanceProbe, FrozenProbeInput, ProbeResult
from model_execution_contracts import ProviderUsage


class _ProbeRunner:
    def __init__(self, usages):
        self.usages = list(usages)
        self.inputs = []

    async def run_once(self, frozen_input):
        self.inputs.append(frozen_input)
        return self.usages.pop(0)


def _frozen():
    return FrozenProbeInput(
        actor_id="jiao",
        conversation_id="probe-conversation",
        profile_id="profile-a",
        static_prefix_hash="stable-hash",
        history_hash="history-hash",
        dynamic_tail_hash="frozen-tail-hash",
    )


@pytest.mark.anyio
async def test_double_send_probe_freezes_all_prefix_inputs():
    runner = _ProbeRunner(
        [
            ProviderUsage.from_provider_values(cache_creation_input_tokens=100),
            ProviderUsage.from_provider_values(cache_read_input_tokens=90),
        ]
    )
    result = await CacheAcceptanceProbe().run(_frozen(), runner)
    assert runner.inputs == [_frozen(), _frozen()]
    assert result.status == "verified"


@pytest.mark.anyio
async def test_second_send_with_cached_tokens_marks_verified():
    runner = _ProbeRunner(
        [
            ProviderUsage.from_provider_values(cached_tokens=0),
            ProviderUsage.from_provider_values(cached_tokens=80),
        ]
    )
    assert (await CacheAcceptanceProbe().run(_frozen(), runner)).status == "verified"


@pytest.mark.anyio
async def test_cache_write_without_read_marks_unverified():
    runner = _ProbeRunner(
        [
            ProviderUsage.from_provider_values(cache_creation_input_tokens=100),
            ProviderUsage.from_provider_values(cache_creation_input_tokens=10),
        ]
    )
    result = await CacheAcceptanceProbe().run(_frozen(), runner)
    assert result.status == "unverified"
    assert result.second.cache_read_input_tokens is None


@pytest.mark.anyio
async def test_absent_cache_metrics_marks_unverified_not_supported():
    runner = _ProbeRunner(
        [ProviderUsage.from_provider_values(), ProviderUsage.from_provider_values()]
    )
    result = await CacheAcceptanceProbe().run(_frozen(), runner)
    assert result == ProbeResult(
        status="unverified", first=runner.inputs and ProviderUsage.from_provider_values(), second=ProviderUsage.from_provider_values()
    )


@pytest.mark.anyio
async def test_probe_failure_is_bounded():
    class FailingRunner:
        async def run_once(self, frozen_input):
            raise RuntimeError("provider body with secrets")

    result = await CacheAcceptanceProbe().run(_frozen(), FailingRunner())
    assert result.status == "failed"
    assert not hasattr(result, "provider_error")
