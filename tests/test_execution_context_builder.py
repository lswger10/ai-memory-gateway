import pytest

from anchored_history import AnchoredHistoryState
from execution_context_builder import GatewayExecutionContextBuilder
from model_execution_contracts import GatewayExecutionRequest
from model_profiles import ModelProfile


class _Relay:
    async def fetch_model_history_facts(self, **kwargs):
        return ()


class _GroupContext:
    relay_client = _Relay()

    async def build_execution_components(self, request):
        return {
            "static_system": ("runtime", "actor", "room"),
            "dynamic_tail": ("dynamic",),
            "actor_prompt_version": "actor.v1",
            "runtime_kernel_version": "runtime.v1",
            "room_policy_version": "room.v1",
            "tool_schema_hash": "tools.v1",
        }


class _HistoryStore:
    def __init__(self):
        self.identity = None

    async def get_or_create(self, namespace, *, identity):
        self.identity = identity
        return AnchoredHistoryState(namespace, 0, "", 0, 1)

    async def observe_appended_events(self, namespace, event_ids):
        pass


def _profile():
    return ModelProfile.from_dict(
        {
            "profile_id": "profile-1",
            "display_name": "Profile",
            "enabled": True,
            "test_status": "passed",
            "provider": "provider",
            "protocol": "openai_chat_completions",
            "base_url": "https://provider.invalid/v1",
            "route_id": "route-1",
            "model": "model-1",
            "adapter_version": "adapter-v1",
            "credential_ref": "env:KEY",
            "headers": {},
            "capabilities": {
                "streaming": True,
                "structured_output": False,
                "tools": False,
                "reasoning_controls": False,
                "cache_strategies": ["no_prompt_cache_v1"],
                "cache_ttls": [],
                "usage_fields": [],
            },
            "cache_strategy": "no_prompt_cache_v1",
            "requested_cache_ttl": None,
            "revision": 4,
        }
    )


def _request():
    return GatewayExecutionRequest.from_dict(
        {
            "contract_version": "gateway-model-execution.v1.0",
            "execution_kind": "full",
            "actor_id": "jiao",
            "room_id": "room_weiwei_jiao",
            "conversation_id": "conversation-1",
            "current_event_id": 2,
            "generation_request_id": "generation-1",
            "execution_mode": "private",
            "fence": {
                "room_id": "room_weiwei_jiao",
                "conversation_id": "conversation-1",
                "burst_id": "burst-1",
                "trigger_event_id": 2,
                "fence_epoch": 1,
                "lease_epoch": 1,
                "orchestrator_instance": "orch-1",
            },
            "bedroom_session_id": None,
            "binding_revision": 1,
        }
    )


@pytest.mark.anyio
async def test_context_builder_persists_complete_cache_identity_before_history_read():
    history = _HistoryStore()
    builder = GatewayExecutionContextBuilder(
        group_context=_GroupContext(),
        bedroom_context=object(),
        history_store=history,
    )

    await builder.build(
        _request(),
        _profile(),
        resolved_room_id="room_weiwei_jiao",
        resolved_conversation_id="conversation-1",
    )

    assert history.identity == {
        "actor_id": "jiao",
        "conversation_id": "conversation-1",
        "profile_id": "profile-1",
        "profile_revision": 4,
        "execution_mode": "private",
        "actor_prompt_version": "actor.v1",
        "runtime_kernel_version": "runtime.v1",
        "room_policy_version": "room.v1",
        "tool_schema_hash": "tools.v1",
        "cache_strategy_version": "no_prompt_cache_v1",
    }

