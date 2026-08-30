from cache_dashboard import build_cache_usage_view
from model_execution_contracts import ProviderUsage
from model_usage_store import ExecutionReceipt


def _receipt(*, observed, creation=None, read=None, cached=None):
    return ExecutionReceipt(
        "receipt-1",
        "generation-1",
        "jiao",
        "room_group_home",
        "conversation-1",
        "profile-1",
        3,
        "provider",
        "anthropic_messages_compatible",
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
    )


def test_dashboard_distinguishes_cache_read_write_miss_and_unverified():
    values = build_cache_usage_view(
        (
            _receipt(observed="verified", creation=10, read=90),
            _receipt(observed="unverified", creation=100),
            _receipt(observed="unverified", creation=0, read=0, cached=0),
            _receipt(observed="unavailable"),
        )
    )

    assert [item["cache_outcome"] for item in values] == [
        "read_hit",
        "write_only_unverified",
        "miss",
        "metrics_unavailable",
    ]
    assert values[0]["cache_read_input_tokens"] == 90
    assert values[3]["cache_read_input_tokens"] is None
    assert values[3]["cached_tokens"] is None


def test_cache_write_alone_never_marks_profile_cache_verified():
    value = build_cache_usage_view(
        (_receipt(observed="verified", creation=1000, read=None, cached=None),)
    )[0]

    assert value["cache_outcome"] == "write_only_unverified"
    assert value["cache_verified"] is False

