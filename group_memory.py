"""Gateway-owned minimal Stage A Group Context Pack builder."""

from __future__ import annotations

import hashlib
import json
import os
from typing import Awaitable, Callable

from database import search_authorized_memories
from group_contracts import (
    CONTRACT_VERSION,
    ContextPackRequest,
    OpaqueContextPack,
    PublicContextFacts,
)
from memory_policy import (
    AuthorizedMemorySearchResult,
    RetrievalPolicy,
    build_retrieval_policy,
    room_members,
)


SearchFunction = Callable[
    [str, RetrievalPolicy, int], Awaitable[AuthorizedMemorySearchResult]
]


_ACTOR_NAMES = {"jiao": "椒椒", "laoke": "老克", "weiwei": "薇薇"}


def _pack_id(pack_kind: str, request: dict) -> str:
    material = json.dumps(
        {"pack_kind": pack_kind, **request},
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return f"pack-{hashlib.sha256(material).hexdigest()[:24]}"


def _ordered_public_events(facts: dict) -> list[dict]:
    by_id = {}
    for event in (
        facts["recent_public_events"]
        + [facts["trigger_event"]]
        + facts["accepted_burst_public_events"]
    ):
        by_id[event["event_id"]] = event
    return [by_id[event_id] for event_id in sorted(by_id)]


def _query_text(facts: dict, current_event_id: int) -> str:
    events = _ordered_public_events(facts)
    current = next(
        (event for event in events if event["event_id"] == current_event_id),
        facts["trigger_event"],
    )
    return current["content"]


def _render_public_context(facts: dict, *, maximum_events: int) -> str:
    events = _ordered_public_events(facts)[-maximum_events:]
    lines = [
        f"{_ACTOR_NAMES.get(event['actor_id'], event['actor_id'])}: {event['content']}"
        for event in events
    ]
    mentions = facts["trigger_event"]["mentions"]
    if mentions:
        lines.append("Relay-normalized strong mentions: " + ", ".join(mentions))
    reactions = facts["reactions_by_event"]
    if reactions:
        lines.append(
            "Current actor-scoped reactions: "
            + json.dumps(reactions, ensure_ascii=False, sort_keys=True)
        )
    return "\n".join(lines)


class GroupContextPackService:
    def __init__(
        self,
        relay_client,
        *,
        search: SearchFunction = search_authorized_memories,
    ) -> None:
        self.relay_client = relay_client
        self.search = search
        self.last_search: AuthorizedMemorySearchResult | None = None

    async def build(
        self, request: ContextPackRequest, *, pack_kind: str
    ) -> OpaqueContextPack:
        if pack_kind not in {"probe", "full"}:
            raise ValueError("pack_kind must be probe or full")
        requested = request.to_dict()
        members = room_members(requested["room_id"])
        policy = build_retrieval_policy(
            requested["actor_id"], requested["room_id"], members
        )
        facts_contract: PublicContextFacts = await self.relay_client.fetch_context_facts(
            request
        )
        facts = facts_contract.to_dict()
        result = await self.search(
            _query_text(facts, requested["current_event_id"]),
            policy,
            3 if pack_kind == "probe" else 10,
        )
        self.last_search = result

        if pack_kind == "probe":
            content = _render_public_context(facts, maximum_events=4)
            if requested["actor_private_stance"]:
                content += "\nYour private burst stance: " + requested["actor_private_stance"]
            messages = [{"role": "user", "content": content}]
            token_budget = int(os.environ.get("GROUP_PROBE_TOKEN_BUDGET", "512"))
        else:
            actor_name = _ACTOR_NAMES[requested["actor_id"]]
            system_lines = [f"你是{actor_name}。只根据以下已授权上下文回应。"]
            if result.memories:
                system_lines.append("Authorized memories:")
                system_lines.extend(f"- {row['content']}" for row in result.memories)
            if requested["actor_private_stance"]:
                system_lines.append(
                    "Your private burst stance: " + requested["actor_private_stance"]
                )
            messages = [
                {"role": "system", "content": "\n".join(system_lines)},
                {
                    "role": "user",
                    "content": _render_public_context(facts, maximum_events=20),
                },
            ]
            token_budget = int(os.environ.get("GROUP_FULL_TOKEN_BUDGET", "12000"))

        return OpaqueContextPack.from_dict(
            {
                "contract_version": CONTRACT_VERSION,
                "pack_id": _pack_id(pack_kind, requested),
                "pack_kind": pack_kind,
                "actor_id": requested["actor_id"],
                "room_id": requested["room_id"],
                "conversation_id": requested["conversation_id"],
                "current_event_id": requested["current_event_id"],
                "burst_id": requested["burst_id"],
                "fence_epoch": requested["fence_epoch"],
                "provider_neutral_messages": messages,
                "token_budget": token_budget,
            }
        )
