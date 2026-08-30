import pytest

from model_execution_contracts import (
    ExecutionContractError,
    GatewayExecutionRequest,
    ProviderUsage,
)


def _request_payload(**overrides):
    payload = {
        "contract_version": "gateway-model-execution.v1.0",
        "execution_kind": "full",
        "actor_id": "jiao",
        "room_id": "room_group_home",
        "conversation_id": "conversation-1",
        "current_event_id": 101,
        "generation_request_id": "generation-1",
        "execution_mode": "group",
        "fence": {
            "room_id": "room_group_home",
            "conversation_id": "conversation-1",
            "burst_id": "burst-1",
            "trigger_event_id": 101,
            "fence_epoch": 4,
            "lease_epoch": 2,
            "orchestrator_instance": "orchestrator-1",
        },
        "bedroom_session_id": None,
        "binding_revision": 3,
        "actor_private_stance": None,
    }
    payload.update(overrides)
    return payload


def _bedroom_request_payload(**overrides):
    payload = {
        "contract_version": "gateway-model-execution.v1.0",
        "execution_kind": "full",
        "execution_mode": "bedroom",
        "actor_id": "laoke",
        "bedroom_session_id": "bedroom-1",
        "current_event_id": 9,
        "bedroom_turn_epoch": 3,
        "generation_request_id": "bedroom:1:3:laoke",
        "binding_revision": None,
    }
    payload.update(overrides)
    return payload


@pytest.mark.parametrize(
    "forbidden,value",
    [
        ("provider", "ofox"),
        ("model", "anthropic/claude-opus-4.6"),
        ("api_key", "secret"),
        ("messages", [{"role": "user", "content": "hidden"}]),
        ("prompt", "hidden"),
        ("cache_strategy", "anthropic_prefix_anchored_v1"),
        ("memory_scopes", ["group"]),
    ],
)
def test_execution_request_rejects_provider_model_key_and_messages(forbidden, value):
    with pytest.raises(ExecutionContractError, match=forbidden):
        GatewayExecutionRequest.from_dict(_request_payload(**{forbidden: value}))


def test_execution_request_accepts_only_identifier_coordinates():
    request = GatewayExecutionRequest.from_dict(_request_payload())
    assert request.actor_id == "jiao"
    assert request.fence is not None
    assert request.fence.trigger_event_id == 101


def test_group_execution_accepts_only_optional_actor_private_stance():
    request = GatewayExecutionRequest.from_dict(
        _request_payload(actor_private_stance="opaque actor stance")
    )
    assert request.actor_private_stance == "opaque actor stance"


def test_usage_fields_are_nullable_and_never_estimated():
    usage = ProviderUsage.from_provider_values(
        input_tokens=120,
        output_tokens=31,
        cache_creation_input_tokens=None,
        cache_read_input_tokens=None,
        cached_tokens=None,
    )
    assert usage.input_tokens == 120
    assert usage.output_tokens == 31
    assert usage.cache_creation_input_tokens is None
    assert usage.cache_read_input_tokens is None
    assert usage.cached_tokens is None


def test_usage_rejects_boolean_and_negative_provider_values():
    with pytest.raises(ExecutionContractError):
        ProviderUsage.from_provider_values(input_tokens=True)
    with pytest.raises(ExecutionContractError):
        ProviderUsage.from_provider_values(output_tokens=-1)


def test_bedroom_execution_uses_session_turn_coordinates_not_group_fence():
    request = GatewayExecutionRequest.from_dict(_bedroom_request_payload())
    assert request.execution_mode == "bedroom"
    assert request.bedroom_session_id == "bedroom-1"
    assert request.bedroom_turn_epoch == 3
    assert request.fence is None
    assert request.room_id is None
    assert request.conversation_id is None
