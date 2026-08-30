from __future__ import annotations

import json

from anchored_history import AnchoredHistoryCompactor, InMemoryAnchoredHistoryStore
from bedroom_memory import BedroomContextPackService, BedroomPackRequest
from group_contracts import CONTRACT_VERSION as GROUP_CONTRACT_VERSION, ContextPackRequest
from group_memory import GroupContextPackService
from model_execution import ContextBundle
from model_execution_contracts import GatewayExecutionRequest
from model_profiles import ModelProfile
from model_usage_store import build_cache_namespace


class GatewayExecutionContextBuilder:
    """Builds provider-neutral cache-safe segments from Relay facts and Gateway ACL."""

    def __init__(
        self,
        *,
        group_context: GroupContextPackService,
        bedroom_context: BedroomContextPackService,
        history_store: InMemoryAnchoredHistoryStore | None = None,
        history_compactor: AnchoredHistoryCompactor | None = None,
    ) -> None:
        self.group_context = group_context
        self.bedroom_context = bedroom_context
        self.history_store = history_store or InMemoryAnchoredHistoryStore()
        self.history_compactor = history_compactor or AnchoredHistoryCompactor()

    async def resolve_coordinates(
        self, request: GatewayExecutionRequest
    ) -> tuple[str, str]:
        if request.execution_mode != "bedroom":
            assert request.room_id is not None and request.conversation_id is not None
            return request.room_id, request.conversation_id
        facts = await self.bedroom_context.relay_client.fetch_bedroom_facts(
            request.bedroom_session_id
        )
        session = facts.get("session", {})
        room_id = session.get("room_id")
        conversation_id = session.get("conversation_id")
        if not isinstance(room_id, str) or not isinstance(conversation_id, str):
            raise ValueError("Bedroom facts omitted canonical coordinates")
        return room_id, conversation_id

    async def build(
        self,
        request: GatewayExecutionRequest,
        profile: ModelProfile,
        *,
        resolved_room_id: str,
        resolved_conversation_id: str,
    ) -> ContextBundle:
        if request.execution_mode == "bedroom":
            components = await self.bedroom_context.build_execution_components(
                BedroomPackRequest(
                    request.bedroom_session_id,
                    request.current_event_id,
                    request.bedroom_turn_epoch,
                    request.actor_id,
                )
            )
            return ContextBundle(
                static_system=components["static_system"],
                stable_summary="",
                stable_history=(),
                dynamic_tail=components["dynamic_tail"],
                actor_prompt_version=components["actor_prompt_version"],
                runtime_kernel_version=components["runtime_kernel_version"],
                room_policy_version=components["room_policy_version"],
                tool_schema_hash=components["tool_schema_hash"],
            )

        assert request.fence is not None
        pack_request = ContextPackRequest.from_dict(
            {
                "contract_version": GROUP_CONTRACT_VERSION,
                "actor_id": request.actor_id,
                "room_id": resolved_room_id,
                "conversation_id": resolved_conversation_id,
                "current_event_id": request.current_event_id,
                "burst_id": request.fence.burst_id,
                "fence_epoch": request.fence.fence_epoch,
                "actor_private_stance": request.actor_private_stance,
            }
        )
        components = await self.group_context.build_execution_components(
            pack_request, pack_kind=request.execution_kind
        )
        namespace = build_cache_namespace(
            actor_id=request.actor_id,
            conversation_id=resolved_conversation_id,
            profile_id=profile.profile_id,
            profile_revision=profile.revision,
            execution_mode=request.execution_mode,
            actor_prompt_version=components["actor_prompt_version"],
            runtime_kernel_version=components["runtime_kernel_version"],
            room_policy_version=components["room_policy_version"],
            tool_schema_hash=components["tool_schema_hash"],
            cache_strategy_version=profile.cache_strategy,
        )
        state = await self.history_store.get_or_create(
            namespace,
            identity={
                "actor_id": request.actor_id,
                "conversation_id": resolved_conversation_id,
                "profile_id": profile.profile_id,
                "profile_revision": profile.revision,
                "execution_mode": request.execution_mode,
                "actor_prompt_version": components["actor_prompt_version"],
                "runtime_kernel_version": components["runtime_kernel_version"],
                "room_policy_version": components["room_policy_version"],
                "tool_schema_hash": components["tool_schema_hash"],
                "cache_strategy_version": profile.cache_strategy,
            },
        )
        history = await self.group_context.relay_client.fetch_model_history_facts(
            actor_id=request.actor_id,
            room_id=resolved_room_id,
            conversation_id=resolved_conversation_id,
            current_event_id=request.current_event_id,
            after_event_id=state.compressed_up_to_event_id,
            through_event_id=max(
                state.compressed_up_to_event_id, request.current_event_id - 1
            ),
        )
        event_ids = tuple(int(event["event_id"]) for event in history)
        await self.history_store.observe_appended_events(namespace, event_ids)
        state, history = await self.history_compactor.maybe_compact(
            store=self.history_store,
            cache_namespace=namespace,
            state=state,
            events=tuple(history),
        )
        stable_history = tuple(
            json.dumps(event, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
            for event in history
        )
        return ContextBundle(
            static_system=components["static_system"],
            stable_summary=state.summary,
            stable_history=stable_history,
            dynamic_tail=components["dynamic_tail"],
            actor_prompt_version=components["actor_prompt_version"],
            runtime_kernel_version=components["runtime_kernel_version"],
            room_policy_version=components["room_policy_version"],
            tool_schema_hash=components["tool_schema_hash"],
        )
