from __future__ import annotations

import json

from anchored_history import AnchoredHistoryCompactor, InMemoryAnchoredHistoryStore
from bedroom_memory import BedroomContextPackService, BedroomPackRequest
from conversation_partitions import InMemoryConversationPartitionStore
from conversation_sync import ConversationSyncService
from group_contracts import CONTRACT_VERSION as GROUP_CONTRACT_VERSION, ContextPackRequest
from group_memory import GroupContextPackService
from model_execution import ContextBundle
from model_execution_contracts import GatewayExecutionRequest
from model_profiles import ModelProfile
from model_usage_store import build_cache_namespace, build_stable_prefix_hash


class GatewayExecutionContextBuilder:
    """Builds provider-neutral cache-safe segments from Relay facts and Gateway ACL."""

    def __init__(
        self,
        *,
        group_context: GroupContextPackService,
        bedroom_context: BedroomContextPackService,
        history_store: InMemoryAnchoredHistoryStore | None = None,
        history_compactor: AnchoredHistoryCompactor | None = None,
        conversation_store=None,
        conversation_sync: ConversationSyncService | None = None,
        bedroom_conversation_sync: ConversationSyncService | None = None,
    ) -> None:
        self.group_context = group_context
        self.bedroom_context = bedroom_context
        self.history_store = history_store or InMemoryAnchoredHistoryStore()
        self.history_compactor = history_compactor or AnchoredHistoryCompactor()
        self.conversation_store = conversation_store or InMemoryConversationPartitionStore()
        self.conversation_sync = conversation_sync or ConversationSyncService(
            group_context.relay_client, self.conversation_store
        )
        self.bedroom_conversation_sync = (
            bedroom_conversation_sync
            or ConversationSyncService(
                getattr(bedroom_context, "relay_client", group_context.relay_client),
                self.conversation_store,
            )
        )

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
            receipt = await self.bedroom_conversation_sync.ensure_bedroom_synced(
                bedroom_session_id=request.bedroom_session_id,
                current_turn_id=request.current_event_id,
                actor_id=request.actor_id,
            )
            components = await self.bedroom_context.build_execution_components(
                BedroomPackRequest(
                    request.bedroom_session_id,
                    request.current_event_id,
                    request.bedroom_turn_epoch,
                    request.actor_id,
                )
            )
            return await self._assemble(
                request=request,
                profile=profile,
                components=components,
                cache_conversation_id=receipt.partition_id,
                partition_id=receipt.partition_id,
                through_stable_event_id=max(0, request.current_event_id - 1),
            )

        assert request.fence is not None
        await self.conversation_sync.ensure_relay_synced(
            actor_id=request.actor_id,
            room_id=resolved_room_id,
            conversation_id=resolved_conversation_id,
            current_event_id=request.current_event_id,
        )
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
        return await self._assemble(
            request=request,
            profile=profile,
            components=components,
            cache_conversation_id=resolved_conversation_id,
            partition_id=resolved_conversation_id,
            through_stable_event_id=max(0, request.current_event_id - 1),
        )

    async def build_cache_keepalive(
        self,
        *,
        actor_id: str,
        room_id: str,
        conversation_id: str,
        execution_mode: str,
        bedroom_session_id: str | None,
        cache_conversation_id: str,
        profile: ModelProfile,
    ) -> ContextBundle:
        """Rebuild the existing stable prefix without fetching dynamic context.

        A keepalive is Gateway-internal cache maintenance.  It consumes only
        Relay-accepted facts already present in the cognitive partition and
        therefore cannot become a public event or a memory source.
        """
        if execution_mode == "bedroom":
            if not bedroom_session_id:
                raise ValueError("Bedroom cache keepalive requires a session")
            components = self.bedroom_context.build_stable_execution_components(
                actor_id, room_id
            )
            partition_id = f"bedroom:{bedroom_session_id}"
        else:
            components = self.group_context.build_stable_execution_components(
                actor_id, room_id
            )
            partition_id = conversation_id

        namespace = build_cache_namespace(
            actor_id=actor_id,
            conversation_id=cache_conversation_id,
            profile_id=profile.profile_id,
            profile_revision=profile.revision,
            execution_mode=execution_mode,
            actor_prompt_version=components["actor_prompt_version"],
            runtime_kernel_version=components["runtime_kernel_version"],
            room_policy_version=components["room_policy_version"],
            tool_schema_hash=components["tool_schema_hash"],
            cache_strategy_version=profile.cache_strategy,
        )
        state = await self.history_store.get_or_create(
            namespace,
            identity={
                "actor_id": actor_id,
                "conversation_id": cache_conversation_id,
                "profile_id": profile.profile_id,
                "profile_revision": profile.revision,
                "execution_mode": execution_mode,
                "actor_prompt_version": components["actor_prompt_version"],
                "runtime_kernel_version": components["runtime_kernel_version"],
                "room_policy_version": components["room_policy_version"],
                "tool_schema_hash": components["tool_schema_hash"],
                "cache_strategy_version": profile.cache_strategy,
            },
        )
        facts = await self.conversation_store.list_facts(
            partition_id, after_event_id=state.compressed_up_to_event_id
        )
        history = tuple(fact.to_history_event() for fact in facts)
        await self.history_store.observe_appended_events(
            namespace, tuple(int(event["event_id"]) for event in history)
        )
        state, history = await self.history_compactor.maybe_compact(
            store=self.history_store,
            cache_namespace=namespace,
            state=state,
            events=history,
        )
        stable_history = tuple(
            json.dumps(event, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
            for event in history
        )
        return ContextBundle(
            static_system=components["static_system"],
            stable_summary=state.summary,
            stable_history=stable_history,
            dynamic_tail=("Cache continuity maintenance request.",),
            actor_prompt_version=components["actor_prompt_version"],
            runtime_kernel_version=components["runtime_kernel_version"],
            room_policy_version=components["room_policy_version"],
            tool_schema_hash=components["tool_schema_hash"],
            cache_conversation_id=cache_conversation_id,
            stable_prefix_hash=build_stable_prefix_hash(
                static_system=components["static_system"],
                stable_summary=state.summary,
                stable_history=stable_history,
            ),
            summary_version=state.state_revision,
            compressed_up_to_event_id=state.compressed_up_to_event_id,
        )

    async def _assemble(
        self,
        *,
        request: GatewayExecutionRequest,
        profile: ModelProfile,
        components: dict,
        cache_conversation_id: str,
        partition_id: str,
        through_stable_event_id: int,
    ) -> ContextBundle:
        namespace = build_cache_namespace(
            actor_id=request.actor_id,
            conversation_id=cache_conversation_id,
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
                "conversation_id": cache_conversation_id,
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
        facts = await self.conversation_store.list_facts(
            partition_id,
            after_event_id=state.compressed_up_to_event_id,
            through_event_id=max(state.compressed_up_to_event_id, through_stable_event_id),
        )
        history = tuple(fact.to_history_event() for fact in facts)
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
        stable_prefix_hash = build_stable_prefix_hash(
            static_system=components["static_system"],
            stable_summary=state.summary,
            stable_history=stable_history,
        )
        current_facts = await self.conversation_store.list_facts(
            partition_id,
            after_event_id=max(0, request.current_event_id - 1),
            through_event_id=request.current_event_id,
        )
        current_fact = (
            current_facts[-1]
            if current_facts and current_facts[-1].source_event_id == request.current_event_id
            else None
        )
        generation_facts = (current_fact,) if current_fact is not None else ()
        if current_fact is not None and current_fact.burst_id:
            generation_facts = tuple(
                fact
                for fact in await self.conversation_store.list_facts(
                    partition_id, through_event_id=request.current_event_id
                )
                if fact.burst_id == current_fact.burst_id
            )
        current_media_references = []
        seen_attachment_ids: set[str] = set()
        for fact in generation_facts:
            for reference in fact.attachments:
                attachment_id = reference["attachment_id"]
                if attachment_id not in seen_attachment_ids:
                    seen_attachment_ids.add(attachment_id)
                    current_media_references.append(reference)
        return ContextBundle(
            static_system=components["static_system"],
            stable_summary=state.summary,
            stable_history=stable_history,
            dynamic_tail=components["dynamic_tail"],
            actor_prompt_version=components["actor_prompt_version"],
            runtime_kernel_version=components["runtime_kernel_version"],
            room_policy_version=components["room_policy_version"],
            tool_schema_hash=components["tool_schema_hash"],
            cache_conversation_id=cache_conversation_id,
            stable_prefix_hash=stable_prefix_hash,
            summary_version=state.state_revision,
            compressed_up_to_event_id=state.compressed_up_to_event_id,
            current_media_references=tuple(current_media_references),
        )
