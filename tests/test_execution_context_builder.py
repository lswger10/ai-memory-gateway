from cache_strategies import PromptSegment
import pytest
import json
from dataclasses import replace

from anchored_history import AnchoredHistoryState
from execution_context_builder import GatewayExecutionContextBuilder
from model_execution_contracts import GatewayExecutionRequest
from model_profiles import ModelProfile


class _Relay:
    def __init__(self):
        self.events = [_event(1), _event(2)]



    async def fetch_interaction_context(self, **kwargs):
        return {"context": None, "accepted_at": "2026-10-04T09:01:00Z"}

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


@pytest.mark.anyio
async def test_live_clock_changes_only_dynamic_tail(monkeypatch):
    from datetime import datetime, timezone
    from unittest.mock import Mock
    clock = Mock()
    clock.now.return_value = datetime(2026, 10, 4, 9, 1, 0, tzinfo=timezone.utc)
    monkeypatch.setattr("execution_context_builder.datetime", clock, raising=False)
    builder = GatewayExecutionContextBuilder(group_context=_GroupContext(), bedroom_context=object())
    args = dict(resolved_room_id="room_weiwei_jiao", resolved_conversation_id="conversation-1")
    before = await builder.build(_request(), _profile(), **args)
    clock.now.return_value = datetime(2026, 10, 4, 9, 2, 1, tzinfo=timezone.utc)
    after = await builder.build(_request(), _profile(), **args)
    first = json.loads(next(s.content for s in before.dynamic_tail if s.source_kind == "current_time"))
    second = json.loads(next(s.content for s in after.dynamic_tail if s.source_kind == "current_time"))
    assert first["now_utc"] == "2026-10-04T09:01:00+00:00"
    assert second["now_utc"] == "2026-10-04T09:02:01+00:00"
    assert first["current_event_created_at"] == _event(2)["created_at"]
    assert first["timezone"] == "UTC"
    assert before.static_system == after.static_system
    assert before.stable_history == after.stable_history
    assert before.stable_summary == after.stable_summary
    assert before.stable_prefix_hash == after.stable_prefix_hash
    assert before.summary_version == after.summary_version
    assert await builder.conversation_store.count_facts("conversation-1") == 2


@pytest.mark.anyio
async def test_device_clock_uses_original_trigger_and_only_changes_dynamic_tail(monkeypatch):
    from datetime import datetime, timezone
    from unittest.mock import Mock, AsyncMock
    clock = Mock(wraps=datetime)
    clock.now.return_value = datetime(2026, 10, 4, 9, 2, 0, tzinfo=timezone.utc)
    monkeypatch.setattr("execution_context_builder.datetime", clock)
    group = _GroupContext()
    group.relay_client = _Relay()
    group.relay_client.fetch_interaction_context = AsyncMock(return_value={
        "accepted_at": "2026-10-04T09:01:00Z", "context": {
            "device_time": "2026-10-04T09:00:00Z", "timezone": "Asia/Shanghai",
            "utc_offset_minutes": 480, "web_search_enabled": False}})
    builder = GatewayExecutionContextBuilder(group_context=group, bedroom_context=object())
    args = dict(resolved_room_id="room_weiwei_jiao", resolved_conversation_id="conversation-1")
    before = await builder.build(_request(), _profile(), **args)
    stamp = json.loads(before.dynamic_tail[0].content)
    assert stamp["now_local"] == "2026-10-04T17:01:00+08:00"
    assert stamp["timezone"] == "Asia/Shanghai"
    assert group.relay_client.fetch_interaction_context.call_args.kwargs["trigger_event_id"] == _request().fence.trigger_event_id
    clock.now.return_value = datetime(2026, 10, 4, 9, 3, 0, tzinfo=timezone.utc)
    after = await builder.build(_request(), _profile(), **args)
    assert before.stable_prefix_hash == after.stable_prefix_hash
    assert before.static_system == after.static_system
    assert before.dynamic_tail != after.dynamic_tail


@pytest.mark.anyio
async def test_search_requires_opt_in_and_current_revision_probe_without_forking_summary():
    from unittest.mock import AsyncMock
    from model_profile_store import InMemoryModelProfileStore
    group = _GroupContext()
    group.relay_client = _Relay()
    device = {"device_time": "2026-10-04T09:00:00Z", "timezone": "UTC",
              "utc_offset_minutes": 0, "web_search_enabled": True}
    group.relay_client.fetch_interaction_context = AsyncMock(return_value={
        "context": device, "accepted_at": "2026-10-04T09:00:00Z"})
    profiles = InMemoryModelProfileStore()
    profile = _profile()
    profile = replace(profile, capabilities=replace(profile.capabilities, web_search="anthropic_web_search_20250305"))
    await profiles.put_profile(profile)
    builder = GatewayExecutionContextBuilder(group_context=group, bedroom_context=object(), profiles=profiles)
    args = dict(resolved_room_id="room_weiwei_jiao", resolved_conversation_id="conversation-1")
    with pytest.raises(ValueError, match="web_search_unverified"):
        await builder.build(_request(), profile, **args)
    await profiles.record_probe_result(profile_id=profile.profile_id, profile_revision=profile.revision,
        probe_kind="native_web_search", status="verified", observed_capabilities={})
    enabled = await builder.build(_request(), profile, **args)
    assert enabled.web_search_enabled
    assert "+web-search:" in enabled.tool_schema_hash
    device["web_search_enabled"] = False
    disabled = await builder.build(_request(), profile, **args)
    assert not disabled.web_search_enabled
    assert "+web-search:" not in disabled.tool_schema_hash
    assert enabled.stable_history == disabled.stable_history
    assert enabled.summary_version == disabled.summary_version
    device["web_search_enabled"] = True
    with pytest.raises(ValueError, match="web_search_unverified"):
        await builder.build(_request(), replace(profile, revision=profile.revision+1), **args)


class _GroupContext:
    relay_client = _Relay()

    async def build_execution_components(self, request, *, pack_kind):
        return {
            "static_system": ("runtime", "actor", "room"),
            "dynamic_tail": (PromptSegment("current_event", "dynamic"),),
            "actor_prompt_version": "actor.v1",
            "runtime_kernel_version": "runtime.v1",
            "room_policy_version": "room.v1",
            "tool_schema_hash": "tools.v1",
        }

    def build_stable_execution_components(self, actor_id, room_id):
        return {
            "static_system": ("runtime", f"actor:{actor_id}", f"room:{room_id}"),
            "actor_prompt_version": "actor.v1",
            "runtime_kernel_version": "runtime.v1",
            "room_policy_version": "room.v1",
            "tool_schema_hash": "tools.v1",
        }


class _BedroomRelay:
    async def fetch_interaction_context(self, **kwargs):
        return {"context": None, "accepted_at": "2026-10-04T09:01:00Z"}

    def __init__(self):
        self.payload = {
            "session": {
                "bedroom_session_id": "bedroom-1",
                "room_id": "room_weiwei_jiao",
                "conversation_id": "conversation-1",
                "actor_id": "jiao",
                "retention_policy": "no-retention",
            },
            "turns": [
                {"turn_id": 1, "actor_id": "jiao", "role": "agent", "text": "prior", "request_id": "b1", "created_at": "then", "provenance_json": None},
                {"turn_id": 2, "actor_id": "weiwei", "role": "human", "text": "current", "request_id": "b2", "created_at": "now", "provenance_json": None},
            ],
        }

    async def fetch_bedroom_facts(self, session_id):
        assert session_id == "bedroom-1"
        return self.payload


class _BedroomContext:
    def __init__(self):
        self.relay_client = _BedroomRelay()

    async def build_execution_components(self, request):
        return {
            "static_system": ("runtime", "actor", "room"),
            "dynamic_tail": (PromptSegment("current_event", "current"),),
            "actor_prompt_version": "actor.v1",
            "runtime_kernel_version": "runtime.v1",
            "room_policy_version": "bedroom.v1",
            "tool_schema_hash": "tools.v1",
        }

    def build_stable_execution_components(self, actor_id, room_id):
        return {
            "static_system": ("runtime", f"actor:{actor_id}", f"room:{room_id}"),
            "actor_prompt_version": "actor.v1",
            "runtime_kernel_version": "runtime.v1",
            "room_policy_version": "bedroom.v1",
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
@pytest.mark.parametrize("actor_id", ["jiao", "laoke"])
@pytest.mark.parametrize("query", ["缓存保活还记得吗？", "刚才在客厅聊了什么？",
    "还记得客厅的聊天吗？", "客厅里刚刚说了什么？"])
async def test_private_recall_adds_public_excerpts_only_to_dynamic_tail(actor_id, query):
    from conversation_partitions import ConversationFact, InMemoryConversationPartitionStore
    from gateway_provider_runner import GatewayProviderRunner

    room_id = f"room_weiwei_{actor_id}"
    group = _GroupContext()
    group.relay_client = _Relay()
    group.relay_client.events = [_event(i, room_id=room_id) for i in (1, 9)]
    group.relay_client.events[-1]["content"] = query
    store = InMemoryConversationPartitionStore()
    builder = GatewayExecutionContextBuilder(group_context=group, bedroom_context=object(), conversation_store=store)
    prior = _request()
    request = replace(prior, actor_id=actor_id, room_id=room_id, current_event_id=9,
        fence=replace(prior.fence, room_id=room_id, trigger_event_id=9))
    before = await builder.build(request, _profile(), resolved_room_id=room_id, resolved_conversation_id="conversation-1")
    public = _event(3, actor_id="laoke", room_id="room_group_home", conversation_id="public-group")
    public["content"] = "缓存保活每50分钟一次，不代表永久命中。"
    await store.append_accepted_facts((ConversationFact.from_relay_event(public),))
    after = await builder.build(request, _profile(), resolved_room_id=room_id, resolved_conversation_id="conversation-1")
    recall = next(segment.content for segment in after.dynamic_tail if segment.source_kind == "context_recall")
    payload = json.loads(recall)
    assert payload["public_group_recall"][0]["actor_id"] == "laoke"
    assert payload["public_group_recall"][0]["event_id"] == 3
    assert payload["public_group_recall"][0]["conversation_id"] == "public-group"
    assert after.static_system == before.static_system
    assert after.stable_history == before.stable_history
    assert after.stable_prefix_hash == before.stable_prefix_hash
    assert after.summary_version == before.summary_version
    assert await store.count_facts("conversation-1") == 2
    assert await store.count_facts("public-group") == 1
    rendered = GatewayProviderRunner()._render(_profile(), request, after, "test").json_body
    assert public["content"] in json.dumps(rendered, ensure_ascii=False)
    probe = await builder.build(replace(request, execution_kind="probe"), _profile(),
        resolved_room_id=room_id, resolved_conversation_id="conversation-1")
    assert all(segment.source_kind != "context_recall" for segment in probe.dynamic_tail)


@pytest.mark.anyio
@pytest.mark.parametrize("mode", ["private", "group", "bedroom", "wrong_actor"])
async def test_recall_does_not_inject_unrelated_or_non_private_context(mode):
    from conversation_partitions import ConversationFact, InMemoryConversationPartitionStore

    group = _GroupContext()
    group.relay_client = _Relay()
    request = _request()
    if mode == "group":
        request = replace(request, execution_mode="group", room_id="room_group_home",
            fence=replace(request.fence, room_id="room_group_home"))
    elif mode == "bedroom":
        request = replace(request, execution_mode="bedroom", room_id=None, conversation_id=None,
            fence=None, bedroom_session_id="bedroom-1", bedroom_turn_epoch=2)
    elif mode == "wrong_actor":
        request = replace(request, actor_id="laoke")
    room_id = request.room_id or "room_weiwei_jiao"
    group.relay_client.events = [_event(i, room_id=room_id) for i in (1, 2)]
    group.relay_client.events[-1]["content"] = "晚饭吃火锅" if mode == "private" else "缓存保活"
    store = InMemoryConversationPartitionStore()
    public = _event(1, room_id="room_group_home", conversation_id="public-group")
    public["content"] = "缓存保活"
    await store.append_accepted_facts((ConversationFact.from_relay_event(public),))
    bedroom = _BedroomContext()
    bedroom.relay_client.payload["turns"][-1]["text"] = "缓存保活"
    builder = GatewayExecutionContextBuilder(group_context=group, bedroom_context=bedroom, conversation_store=store)
    bundle = await builder.build(request, _profile(), resolved_room_id=room_id, resolved_conversation_id="conversation-1")
    assert all(segment.source_kind != "context_recall" for segment in bundle.dynamic_tail)


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
async def test_full_tool_capable_execution_gets_actor_bound_memory_context():
    payload = _profile().to_dict()
    payload["capabilities"]["tools"] = True
    profile = ModelProfile.from_dict(payload)
    builder = GatewayExecutionContextBuilder(
        group_context=_GroupContext(), bedroom_context=object(),
    )

    bundle = await builder.build(
        _request(), profile, resolved_room_id="room_weiwei_jiao",
        resolved_conversation_id="conversation-1",
    )

    assert bundle.tool_schema_hash == "actor-memory-tools.v1"
    assert bundle.actor_memory_context.actor_id == "jiao"
    assert bundle.actor_memory_context.room_id == "room_weiwei_jiao"
    assert bundle.actor_memory_context.writable_scopes == {"weiwei-jiao"}


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


@pytest.mark.anyio
async def test_compressed_cursor_never_deletes_complete_conversation_partition():
    from anchored_history import AnchoredHistoryCompactor, InMemoryAnchoredHistoryStore
    from conversation_partitions import InMemoryConversationPartitionStore

    relay = _Relay()
    relay.events = [_event(i) for i in range(1, 6)]
    context = _GroupContext()
    context.relay_client = relay
    conversations = InMemoryConversationPartitionStore()
    cache = InMemoryAnchoredHistoryStore()
    builder = GatewayExecutionContextBuilder(
        group_context=context, bedroom_context=object(),
        conversation_store=conversations, history_store=cache,
        history_compactor=AnchoredHistoryCompactor(
            compact_after_events=3, retain_raw_events=1, summary_token_limit=128
        ),
    )
    class Summary:
        calls = 0
        async def summarize(self, **kwargs):
            self.calls += 1
            assert kwargs["actor_id"] == "jiao"
            assert [event["event_id"] for event in kwargs["events"]] == [1, 2, 3]
            return "Events one through three, without changing the factual record."
    summary = Summary()
    builder.summary_service = summary
    prior = _request()
    request = replace(
        prior, current_event_id=5, fence=replace(prior.fence, trigger_event_id=5)
    )
    selected = replace(_profile(), capabilities=replace(_profile().capabilities, tools=True))
    bundle = await builder.build(
        request, selected, resolved_room_id="room_weiwei_jiao",
        resolved_conversation_id="conversation-1",
    )
    assert bundle.stable_summary
    assert summary.calls == 1
    probe = await builder.build(replace(request, execution_kind="probe"), selected,
        resolved_room_id="room_weiwei_jiao", resolved_conversation_id="conversation-1")
    assert probe.stable_summary == bundle.stable_summary
    assert probe.compressed_up_to_event_id == bundle.compressed_up_to_event_id
    assert probe.stable_history == bundle.stable_history
    assert probe.actor_memory_context is None
    assert probe.tool_schema_hash != bundle.tool_schema_hash
    assert summary.calls == 1
    assert await conversations.count_facts("conversation-1") == 5
    assert [fact.source_event_id for fact in await conversations.list_facts("conversation-1")] == [1, 2, 3, 4, 5]


@pytest.mark.anyio
async def test_probe_and_keepalive_never_trigger_paid_summary_or_advance_cursor():
    from anchored_history import AnchoredHistoryCompactor, InMemoryAnchoredHistoryStore
    from conversation_partitions import ConversationFact, InMemoryConversationPartitionStore
    relay = _Relay()
    relay.events = [_event(i) for i in range(1, 7)]
    group = _GroupContext()
    group.relay_client = relay
    conversations, history = InMemoryConversationPartitionStore(), InMemoryAnchoredHistoryStore()
    await conversations.append_accepted_facts(tuple(ConversationFact.from_relay_event(event) for event in relay.events))
    builder = GatewayExecutionContextBuilder(group_context=group, bedroom_context=object(),
        conversation_store=conversations, history_store=history,
        history_compactor=AnchoredHistoryCompactor(compact_after_events=3, retain_raw_events=1))
    class ForbiddenSummary:
        async def summarize(self, **kwargs):
            raise AssertionError("paid summary outside full chat")
    builder.summary_service = ForbiddenSummary()
    prior = _request()
    probe = replace(prior, execution_kind="probe", current_event_id=6, fence=replace(prior.fence, trigger_event_id=6))
    result = await builder.build(probe, _profile(), resolved_room_id="room_weiwei_jiao", resolved_conversation_id="conversation-1")
    assert result.compressed_up_to_event_id == 0
    assert len(result.stable_history) == 5
    maintenance = await builder.build_cache_keepalive(actor_id="jiao", room_id="room_weiwei_jiao",
        conversation_id="conversation-1", execution_mode="private", bedroom_session_id=None,
        cache_conversation_id="conversation-1", profile=_profile())
    assert maintenance.compressed_up_to_event_id == 0
    assert len(maintenance.stable_history) == 6


@pytest.mark.anyio
async def test_bedroom_build_uses_session_partition_and_prior_accepted_turns():
    from conversation_partitions import InMemoryConversationPartitionStore

    payload = {
        "contract_version": "gateway-model-execution.v1.0",
        "execution_kind": "full",
        "actor_id": "jiao",
        "current_event_id": 2,
        "generation_request_id": "bedroom-generation-1",
        "execution_mode": "bedroom",
        "bedroom_session_id": "bedroom-1",
        "bedroom_turn_epoch": 2,
        "binding_revision": 1,
    }
    request = GatewayExecutionRequest.from_dict(payload)
    store = InMemoryConversationPartitionStore()
    bedroom = _BedroomContext()
    builder = GatewayExecutionContextBuilder(
        group_context=_GroupContext(), bedroom_context=bedroom,
        conversation_store=store,
    )
    bundle = await builder.build(
        request, _profile(), resolved_room_id="room_weiwei_jiao",
        resolved_conversation_id="conversation-1",
    )
    assert bundle.cache_conversation_id == "bedroom:bedroom-1"
    assert '"event_id":1' in "".join(bundle.stable_history)
    assert '"event_id":2' not in "".join(bundle.stable_history)
    assert await store.count_facts("bedroom:bedroom-1") == 2
    from gateway_provider_runner import GatewayProviderRunner
    rendered = GatewayProviderRunner()._render(_profile(), request, bundle, "bedroom-test").json_body
    assert rendered["messages"][1]["role"] == "assistant"
    assert "prior" in rendered["messages"][1]["content"]


@pytest.mark.anyio
async def test_cache_keepalive_uses_all_persisted_facts_and_no_dynamic_context_pack():
    from conversation_partitions import ConversationFact, InMemoryConversationPartitionStore

    store = InMemoryConversationPartitionStore()
    await store.append_accepted_facts(tuple(
        ConversationFact.from_relay_event(_event(i)) for i in (1, 2)
    ))
    context = _GroupContext()
    builder = GatewayExecutionContextBuilder(
        group_context=context,
        bedroom_context=_BedroomContext(),
        conversation_store=store,
    )

    bundle = await builder.build_cache_keepalive(
        actor_id="jiao",
        room_id="room_weiwei_jiao",
        conversation_id="conversation-1",
        execution_mode="private",
        bedroom_session_id=None,
        cache_conversation_id="conversation-1",
        profile=_profile(),
    )

    assert '"event_id":1' in "".join(bundle.stable_history)
    assert '"event_id":2' in "".join(bundle.stable_history)
    assert bundle.dynamic_tail == (PromptSegment("request_metadata", "Cache continuity maintenance request."),)
    assert bundle.static_system == (
        "runtime", "actor:jiao", "room:room_weiwei_jiao"
    )
