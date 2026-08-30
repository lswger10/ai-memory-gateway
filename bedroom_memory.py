"""Gateway-owned Bedroom Context Pack and durable retention helpers."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Any

from actor_prompt_profiles import load_actor_prompt_profiles
from group_contracts import CONTRACT_VERSION, OpaqueContextPack
from group_memory import (
    _COMMON_RUNTIME_KERNEL,
    search_authorized_memories,
    search_authorized_summary_candidates,
)
from memory_policy import build_retrieval_policy, room_members


BEDROOM_CONTRACT_VERSION = "bedroom-room.v1.0"
PRIVATE_ROOM_ACTORS = {
    "room_weiwei_jiao": "jiao",
    "room_weiwei_laoke": "laoke",
}
PRIVATE_ROOM_SCOPES = {
    "room_weiwei_jiao": "weiwei-jiao",
    "room_weiwei_laoke": "weiwei-laoke",
}


class BedroomContractError(ValueError):
    pass


def validate_bedroom_facts(payload: Any) -> dict:
    if not isinstance(payload, dict) or payload.get("contract_version") != BEDROOM_CONTRACT_VERSION:
        raise BedroomContractError("invalid Bedroom facts")
    if set(payload) != {"contract_version", "session", "turns"}:
        raise BedroomContractError("unexpected Bedroom facts fields")
    session = payload.get("session")
    turns = payload.get("turns")
    if not isinstance(session, dict) or not isinstance(turns, list):
        raise BedroomContractError("invalid Bedroom facts shape")
    room_id = session.get("room_id")
    actor_id = session.get("actor_id")
    if PRIVATE_ROOM_ACTORS.get(room_id) != actor_id:
        raise BedroomContractError("Bedroom actor does not match room")
    if session.get("status") not in {"active", "ending"}:
        raise BedroomContractError("Bedroom session is not readable")
    if session.get("retention_policy") not in {
        "no-retention", "summary-only", "full-bedroom-archive"
    }:
        raise BedroomContractError("invalid retention policy")
    for turn in turns:
        if (
            not isinstance(turn, dict)
            or turn.get("actor_id") not in {"weiwei", actor_id}
            or turn.get("role") not in {"human", "agent"}
            or not isinstance(turn.get("text"), str)
            or not isinstance(turn.get("turn_id"), int)
        ):
            raise BedroomContractError("invalid Bedroom turn")
    return payload


@dataclass(frozen=True)
class BedroomPackRequest:
    bedroom_session_id: str
    turn_id: int
    turn_epoch: int
    actor_id: str

    @classmethod
    def from_dict(cls, payload: Any) -> "BedroomPackRequest":
        if not isinstance(payload, dict) or set(payload) != {
            "contract_version", "bedroom_session_id", "turn_id", "turn_epoch", "actor_id"
        }:
            raise BedroomContractError("invalid Bedroom pack request")
        if payload.get("contract_version") != BEDROOM_CONTRACT_VERSION:
            raise BedroomContractError("Bedroom contract mismatch")
        if payload.get("actor_id") not in {"jiao", "laoke"}:
            raise BedroomContractError("invalid Bedroom actor")
        if not isinstance(payload.get("turn_id"), int) or not isinstance(payload.get("turn_epoch"), int):
            raise BedroomContractError("invalid Bedroom turn coordinates")
        return cls(
            str(payload["bedroom_session_id"]), int(payload["turn_id"]),
            int(payload["turn_epoch"]), str(payload["actor_id"]),
        )


class BedroomContextPackService:
    def __init__(self, relay_client, *, search=search_authorized_memories, summary_search=search_authorized_summary_candidates, prompt_profiles=None):
        self.relay_client = relay_client
        self.search = search
        self.summary_search = summary_search
        self.prompt_profiles = prompt_profiles or load_actor_prompt_profiles()
        self.last_candidate_ids: tuple[int, ...] = ()

    async def build(self, request: BedroomPackRequest) -> OpaqueContextPack:
        facts = validate_bedroom_facts(
            await self.relay_client.fetch_bedroom_facts(request.bedroom_session_id)
        )
        session = facts["session"]
        if session["actor_id"] != request.actor_id or session["turn_epoch"] != request.turn_epoch:
            raise BedroomContractError("stale Bedroom generation")
        if not any(turn["turn_id"] == request.turn_id and turn["role"] == "human" for turn in facts["turns"]):
            raise BedroomContractError("Bedroom trigger turn is missing")
        room_id = session["room_id"]
        policy = build_retrieval_policy(request.actor_id, room_id, room_members(room_id))
        query = next(turn["text"] for turn in facts["turns"] if turn["turn_id"] == request.turn_id)
        memories = await self.search(query, policy, 10)
        summaries = await self.summary_search(query, policy, 6)
        self.last_candidate_ids = tuple(memories.candidate_ids)
        profile = self.prompt_profiles[request.actor_id]
        system_lines = [
            _COMMON_RUNTIME_KERNEL,
            f"Actor prompt [{profile.prompt_version}]: {profile.prompt_text}",
            f"Room policy: speak only as {request.actor_id} in {room_id}; authorized relationship scopes are " + ", ".join(policy.allowed_scopes) + ".",
            "Temporary Bedroom scene layer (not identity or permanent memory): " + str(session.get("scene_context") or "private relationship scene"),
        ]
        if memories.memories:
            system_lines.append("Authorized relationship and memory context:")
            system_lines.extend(f"- {row['content']}" for row in memories.memories)
        if summaries:
            system_lines.append("Authorized relationship summaries:")
            system_lines.extend(f"- [{row['scope']}] {row['content']}" for row in summaries)
        transcript = "\n".join(
            f"{'薇薇' if turn['actor_id']=='weiwei' else profile.actor_id}: {turn['text']}"
            for turn in facts["turns"][-30:]
        )
        pack_id = hashlib.sha256(
            f"{request.bedroom_session_id}|{request.turn_id}|{request.turn_epoch}|{request.actor_id}".encode()
        ).hexdigest()
        return OpaqueContextPack.from_dict(
            {
                "contract_version": CONTRACT_VERSION,
                "pack_id": f"bedroom-{pack_id}",
                "pack_kind": "full",
                "actor_id": request.actor_id,
                "room_id": room_id,
                "conversation_id": session["conversation_id"],
                "current_event_id": request.turn_id,
                "burst_id": f"bedroom:{request.bedroom_session_id}:{request.turn_epoch}",
                "fence_epoch": request.turn_epoch,
                "provider_neutral_messages": [
                    {"role": "system", "content": "\n".join(system_lines)},
                    {"role": "user", "content": transcript},
                ],
                "token_budget": 12000,
            }
        )

    async def build_execution_components(self, request: BedroomPackRequest) -> dict:
        """Bedroom transcript is deliberately dynamic and never joins private cache history."""
        facts = validate_bedroom_facts(
            await self.relay_client.fetch_bedroom_facts(request.bedroom_session_id)
        )
        session = facts["session"]
        if session["actor_id"] != request.actor_id or session["turn_epoch"] != request.turn_epoch:
            raise BedroomContractError("stale Bedroom generation")
        if not any(
            turn["turn_id"] == request.turn_id and turn["role"] == "human"
            for turn in facts["turns"]
        ):
            raise BedroomContractError("Bedroom trigger turn is missing")
        room_id = session["room_id"]
        policy = build_retrieval_policy(request.actor_id, room_id, room_members(room_id))
        query = next(
            turn["text"] for turn in facts["turns"] if turn["turn_id"] == request.turn_id
        )
        memories = await self.search(query, policy, 10)
        summaries = await self.summary_search(query, policy, 6)
        profile = self.prompt_profiles[request.actor_id]
        static_system = (
            _COMMON_RUNTIME_KERNEL,
            f"Actor prompt [{profile.prompt_version}]: {profile.prompt_text}",
            (
                f"Room policy: speak only as {request.actor_id} in {room_id}; "
                "authorized relationship scopes are "
                + ", ".join(policy.allowed_scopes)
                + "."
            ),
        )
        dynamic: list[str] = [
            "Temporary Bedroom scene layer (not identity or permanent memory): "
            + str(session.get("scene_context") or "private relationship scene")
        ]
        if memories.memories:
            dynamic.append(
                "Authorized relationship and memory context:\n"
                + "\n".join(f"- {row['content']}" for row in memories.memories)
            )
        if summaries:
            dynamic.append(
                "Authorized relationship summaries:\n"
                + "\n".join(f"- [{row['scope']}] {row['content']}" for row in summaries)
            )
        dynamic.append(
            "\n".join(
                f"{'薇薇' if turn['actor_id']=='weiwei' else profile.actor_id}: {turn['text']}"
                for turn in facts["turns"]
            )
        )
        return {
            "room_id": room_id,
            "conversation_id": session["conversation_id"],
            "static_system": static_system,
            "dynamic_tail": tuple(dynamic),
            "actor_prompt_version": profile.prompt_version,
            "runtime_kernel_version": "group-runtime-kernel.v1",
            "room_policy_version": f"{room_id}.bedroom.v1",
            "tool_schema_hash": "tools.none.v1",
        }


def bounded_relationship_summary(facts: dict, maximum_chars: int = 1600) -> str:
    lines = [
        f"{'薇薇' if turn['actor_id']=='weiwei' else turn['actor_id']}: {turn['text']}"
        for turn in facts["turns"]
    ]
    content = "\n".join(lines)
    return content if len(content) <= maximum_chars else content[-maximum_chars:]


class BedroomRetentionService:
    def __init__(self, repository):
        self.repository = repository

    async def persist(self, payload: Any) -> dict:
        facts = validate_bedroom_facts(payload)
        session = facts["session"]
        policy = session["retention_policy"]
        if policy == "no-retention":
            raise BedroomContractError("no-retention must not enter Gateway")
        scope = PRIVATE_ROOM_SCOPES[session["room_id"]]
        canonical = json.dumps(facts, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        content_hash = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
        if policy == "summary-only":
            receipt_id = await self.repository.persist_summary(
                session_id=session["bedroom_session_id"],
                actor_id=session["actor_id"],
                scope=scope,
                content=bounded_relationship_summary(facts),
                content_hash=content_hash,
            )
        else:
            receipt_id = await self.repository.persist_archive(
                session_id=session["bedroom_session_id"],
                actor_id=session["actor_id"],
                room_id=session["room_id"],
                scope=scope,
                facts=facts,
                content_hash=content_hash,
            )
        return {
            "contract_version": BEDROOM_CONTRACT_VERSION,
            "accepted": True,
            "receipt_id": receipt_id,
        }


class BedroomPostgresRepository:
    """Idempotent durable receipt writer using Gateway's existing PostgreSQL pool."""

    async def _existing(self, conn, session_id: str, content_hash: str):
        row = await conn.fetchrow(
            "SELECT receipt_id,content_hash FROM bedroom_retention_receipts WHERE session_id=$1",
            session_id,
        )
        if row is None:
            return None
        if row["content_hash"] != content_hash:
            raise BedroomContractError("Bedroom retention payload changed")
        return row["receipt_id"]

    async def persist_summary(self, *, session_id, actor_id, scope, content, content_hash):
        from database import get_pool

        pool = await get_pool()
        async with pool.acquire() as conn:
            async with conn.transaction():
                await conn.execute("SELECT pg_advisory_xact_lock(hashtext($1))", "bedroom:" + session_id)
                existing = await self._existing(conn, session_id, content_hash)
                if existing:
                    return existing
                summary_id = await conn.fetchval(
                    "INSERT INTO relationship_summaries "
                    "(scope,content,confidential,evidence_event_ids,evidence_hash) "
                    "VALUES ($1,$2,TRUE,ARRAY[]::BIGINT[],$3) RETURNING id",
                    scope,
                    content,
                    "bedroom:" + content_hash,
                )
                receipt_id = "bedroom-summary:" + hashlib.sha256(
                    f"{session_id}|{content_hash}".encode()
                ).hexdigest()
                await conn.execute(
                    "INSERT INTO bedroom_retention_receipts "
                    "(session_id,retention_policy,actor_id,scope,content_hash,relationship_summary_id,receipt_id) "
                    "VALUES ($1,'summary-only',$2,$3,$4,$5,$6)",
                    session_id, actor_id, scope, content_hash, summary_id, receipt_id,
                )
                return receipt_id

    async def persist_archive(self, *, session_id, actor_id, room_id, scope, facts, content_hash):
        from database import get_pool

        pool = await get_pool()
        async with pool.acquire() as conn:
            async with conn.transaction():
                await conn.execute("SELECT pg_advisory_xact_lock(hashtext($1))", "bedroom:" + session_id)
                existing = await self._existing(conn, session_id, content_hash)
                if existing:
                    return existing
                archive_id = await conn.fetchval(
                    "INSERT INTO bedroom_archive_raw "
                    "(session_id,actor_id,room_id,scope,facts_json,content_hash) "
                    "VALUES ($1,$2,$3,$4,$5::jsonb,$6) RETURNING id",
                    session_id, actor_id, room_id, scope,
                    json.dumps(facts, ensure_ascii=False), content_hash,
                )
                receipt_id = "bedroom-archive:" + hashlib.sha256(
                    f"{session_id}|{content_hash}".encode()
                ).hexdigest()
                await conn.execute(
                    "INSERT INTO bedroom_retention_receipts "
                    "(session_id,retention_policy,actor_id,scope,content_hash,archive_id,receipt_id) "
                    "VALUES ($1,'full-bedroom-archive',$2,$3,$4,$5,$6)",
                    session_id, actor_id, scope, content_hash, archive_id, receipt_id,
                )
                return receipt_id
