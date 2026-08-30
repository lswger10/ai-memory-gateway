import pytest
from dataclasses import replace

from anchored_history import AnchoredHistoryState
from execution_context_builder import GatewayExecutionContextBuilder
from model_execution_contracts import GatewayExecutionRequest
from model_profiles import ModelProfile


class _Relay:
    def __init__(self):
        self.events = [_event(1), _event(2)]

    async def fetch_model_history_facts(self, **kwargs):
        return tuple(
            event for event in self.events
            if event["event_id"] > kwargs["after_event_id"]
            and event["event_id"] <= kwargs["through_event_id"]
        )


def _event(
    event_id, *, actor_id="weiwei", room_id="room_weiwei_jiao",
    conversation_id="conversation-1"
):
    return {
        "event_id": event_id,
        "room_id": room_id,
        "conversation_id": conversation_id,
        "burst_id": f"burst-{event_id}",
        "actor_id": actor_id,
        "role": "human" if actor_id == "weiwei" else "agent",
        "event_type": "human_message" if actor_id == "weiwei" else "agent_final",
        "content": f"message-{event_id}",
        "reply_to_event_id": None,
        "mentions": [],
        "created_at": f"2026-08-30T00:00:0{event_id}Z",
        "request_id": f"request-{event_id}",
        "visibility": "room",
        "provenance": None,
    }


class _GroupContext:
    relay_client = _Relay()

    async def build_execution_components(self, request, *, pack_kind):
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

    bundle = await builder.build(
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
    assert '"event_id":1' in bundle.stable_history[0]
    assert '"event_id":2' not in "".join(bundle.stable_history)


@pytest.mark.anyio
async def test_group_cognitive_transcript_is_shared_while_actor_cache_identity_is_separate():
    from conversation_partitions import InMemoryConversationPartitionStore

    relay = _Relay()
    relay.events = [
        _event(i, room_id="room_group_home", conversation_id="group-1")
        for i in (1, 2)
    ]
    context = _GroupContext()
    context.relay_client = relay
    store = InMemoryConversationPartitionStore()
    jiao_history = _HistoryStore()
    jiao = GatewayExecutionContextBuilder(
        group_context=context, bedroom_context=object(), history_store=jiao_history,
        conversation_store=store,
    )
    await jiao.build(
        replace(
            _request(), room_id="room_group_home", conversation_id="group-1",
            fence=replace(
                _request().fence, room_id="room_group_home", conversation_id="group-1"
            ),
        ),
        _profile(), resolved_room_id="room_group_home",
        resolved_conversation_id="group-1",
    )

    relay.events.append(
        _event(3, actor_id="jiao", room_id="room_group_home", conversation_id="group-1")
    )
    prior = _request()
    request = replace(
        prior,
        actor_id="laoke",
        current_event_id=3,
        generation_request_id="generation-2",
        room_id="room_group_home",
        conversation_id="group-1",
        fence=replace(
            prior.fence, room_id="room_group_home", conversation_id="group-1",
            trigger_event_id=3,
        ),
    )
    laoke_history = _HistoryStore()
    laoke = GatewayExecutionContextBuilder(
        group_context=context, bedroom_context=object(), history_store=laoke_history,
        conversation_store=store,
    )
    bundle = await laoke.build(
        request, _profile(),
        resolved_room_id="room_group_home", resolved_conversation_id="group-1",
    )

    assert await store.count_facts("group-1") == 3
    assert jiao_history.identity["actor_id"] == "jiao"
    assert laoke_history.identity["actor_id"] == "laoke"
    assert '"event_id":2' in "".join(bundle.stable_history)
