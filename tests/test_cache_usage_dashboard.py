import cache_dashboard
from cache_dashboard import build_cache_usage_view
from model_execution_contracts import ProviderUsage
from model_usage_store import ExecutionReceipt


def _receipt(
    *,
    observed,
    creation=None,
    read=None,
    cached=None,
    usage_received=True,
    protocol="anthropic_messages_compatible",
):
    return ExecutionReceipt(
        "receipt-1",
        "generation-1",
        "jiao",
        "room_group_home",
        "conversation-1",
        "profile-1",
        3,
        "provider",
        protocol,
        "route-1",
        "model-1",
        "adapter-v1",
        "anthropic_prefix_anchored_v1",
        "1h",
        observed,
        False,
        None,
        ProviderUsage.from_provider_values(
            input_tokens=120,
            output_tokens=20,
            cache_creation_input_tokens=creation,
            cache_read_input_tokens=read,
            cached_tokens=cached,
        ),
        "delivered",
        stable_prefix_hash="stable-prefix",
        prompt_cache_key="prompt-cache-key",
        runtime_kernel_version="kernel.v1",
        persona_version="jiao.v2",
        room_policy_version="group.v1",
        tool_schema_hash="tools.none",
        summary_version=3,
        compressed_up_to_event_id=90,
        provider_usage_received=usage_received,
    )


def test_dashboard_distinguishes_hit_observed_miss_and_unobservable():
    values = build_cache_usage_view(
        (
            _receipt(observed="verified", creation=10, read=90),
            _receipt(observed="unverified", creation=100),
            _receipt(observed="unverified", creation=0, read=0),
            _receipt(observed="unavailable", usage_received=False),
        )
    )

    assert [item["cache_outcome"] for item in values] == [
        "HIT",
        "UNOBSERVABLE",
        "OBSERVED_MISS",
        "UNOBSERVABLE",
    ]
    assert values[0]["cache_read_input_tokens"] == 90
    assert values[3]["cache_read_input_tokens"] is None
    assert values[3]["cached_tokens"] is None
    assert values[0]["stable_prefix_hash"] == "stable-prefix"
    assert values[0]["persona_version"] == "jiao.v2"
    assert values[0]["compressed_up_to_event_id"] == 90


def test_cache_write_alone_never_marks_profile_cache_verified():
    value = build_cache_usage_view(
        (_receipt(observed="verified", creation=1000, read=None, cached=None),)
    )[0]

    assert value["cache_outcome"] == "UNOBSERVABLE"
    assert value["cache_verified"] is False


def test_dashboard_separates_observable_hit_ratio_from_telemetry_coverage():
    receipts = (
        _receipt(observed="verified", read=90),
        _receipt(observed="unverified", creation=0, read=0),
        _receipt(observed="unavailable", usage_received=False),
    )

    summary = cache_dashboard.build_cache_observability_summary(receipts)

    assert summary == {
        "total_requests": 3,
        "observable_requests": 2,
        "hit_requests": 1,
        "observed_miss_requests": 1,
        "unobservable_requests": 1,
        "observable_hit_ratio": 0.5,
        "telemetry_coverage_ratio": 2 / 3,
    }


def test_pre_migration_receipts_use_real_cache_fields_as_usage_evidence():
    values = build_cache_usage_view(
        (
            _receipt(
                observed="unverified",
                protocol="openai_chat_completions",
                cached=0,
                usage_received=False,
            ),
            _receipt(
                observed="unverified",
                creation=0,
                read=0,
                usage_received=False,
            ),
        )
    )

    assert [value["cache_outcome"] for value in values] == [
        "OBSERVED_MISS",
        "OBSERVED_MISS",
    ]
