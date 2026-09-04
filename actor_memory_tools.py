"""Provider-neutral actor memory tools with accepted-final staging."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Any, Mapping

from database import _persist_or_merge_group_memory, get_pool, search_authorized_memories
from memory_policy import MemoryScope, MemoryStatus, MemoryType, MemoryWrite, Perspective, SourceKind
from memory_policy import build_retrieval_policy, room_members


ACTOR_MEMORY_TOOL_NAMES = frozenset(
    {
        "search_memory", "get_memory", "list_memories", "write_memory",
        "update_memory", "delete_memory", "restore_memory", "change_scope",
        "set_confidential", "set_perspective", "set_memory_type",
        "set_memory_status", "set_importance", "add_evidence",
        "remove_evidence", "merge_memories", "supersede_memory",
        "propose_memory_candidate",
    }
)
ACTOR_MEMORY_TOOL_SCHEMA_HASH = "actor-memory-tools.v1"
_READ_TOOLS = {"search_memory", "get_memory", "list_memories"}
_PAIRWISE_SCOPE = {"jiao": "weiwei-jiao", "laoke": "weiwei-laoke"}
_ROOM_WRITE_SCOPES = {
    "room_weiwei_jiao": {"weiwei-jiao"},
    "room_weiwei_laoke": {"weiwei-laoke"},
    "room_group_home": {"weiwei-jiao", "weiwei-laoke", "jiao-laoke", "group"},
}
_ACTOR_WRITE_SCOPES = {
    "jiao": {"weiwei-jiao", "jiao-laoke", "group"},
    "laoke": {"weiwei-laoke", "jiao-laoke", "group"},
}


def _schema(properties: dict[str, Any], required: tuple[str, ...] = ()) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": properties,
        "required": list(required),
        "additionalProperties": False,
    }


def actor_memory_tool_definitions() -> tuple[dict[str, Any], ...]:
    integer = {"type": "integer", "minimum": 1}
    memory_id = {"memory_id": integer}
    definitions = {
        "search_memory": _schema({"query": {"type": "string"}, "limit": integer}, ("query",)),
        "get_memory": _schema(memory_id, ("memory_id",)),
        "list_memories": _schema({"status": {"enum": ["active", "stale", "superseded"]}, "limit": integer}),
        "write_memory": _schema(
            {
                "content": {"type": "string"}, "scope": {"type": "string"},
                "memory_type": {"enum": ["fact", "inference"]},
                "perspective": {"enum": ["weiwei", "jiao", "laoke", "shared"]},
                "confidential": {"type": "boolean"}, "importance": {"type": "integer", "minimum": 1, "maximum": 10},
                "evidence_event_ids": {"type": "array", "items": integer, "uniqueItems": True},
            },
            ("content", "scope", "memory_type", "perspective", "confidential", "importance", "evidence_event_ids"),
        ),
        "update_memory": _schema({**memory_id, "content": {"type": "string"}, "importance": {"type": "integer", "minimum": 1, "maximum": 10}}, ("memory_id",)),
        "delete_memory": _schema(memory_id, ("memory_id",)),
        "restore_memory": _schema(memory_id, ("memory_id",)),
        "change_scope": _schema({**memory_id, "scope": {"type": "string"}}, ("memory_id", "scope")),
        "set_confidential": _schema({**memory_id, "confidential": {"type": "boolean"}}, ("memory_id", "confidential")),
        "set_perspective": _schema({**memory_id, "perspective": {"enum": ["weiwei", "jiao", "laoke", "shared"]}}, ("memory_id", "perspective")),
        "set_memory_type": _schema({**memory_id, "memory_type": {"enum": ["fact", "inference"]}}, ("memory_id", "memory_type")),
        "set_memory_status": _schema({**memory_id, "status": {"enum": ["active", "stale"]}}, ("memory_id", "status")),
        "set_importance": _schema({**memory_id, "importance": {"type": "integer", "minimum": 1, "maximum": 10}}, ("memory_id", "importance")),
        "add_evidence": _schema({**memory_id, "event_id": integer}, ("memory_id", "event_id")),
        "remove_evidence": _schema({**memory_id, "event_id": integer}, ("memory_id", "event_id")),
        "merge_memories": _schema({"memory_ids": {"type": "array", "items": integer, "minItems": 2, "uniqueItems": True}, "content": {"type": "string"}, "importance": {"type": "integer", "minimum": 1, "maximum": 10}}, ("memory_ids", "content", "importance")),
        "supersede_memory": _schema({**memory_id, "content": {"type": "string"}, "memory_type": {"enum": ["fact", "inference"]}, "importance": {"type": "integer", "minimum": 1, "maximum": 10}}, ("memory_id", "content", "memory_type", "importance")),
    }
    definitions["propose_memory_candidate"] = definitions["write_memory"]
    return tuple(
        {"name": name, "description": f"Gateway actor memory operation: {name}", "input_schema": definitions[name]}
        for name in sorted(definitions)
    )


@dataclass(frozen=True, slots=True)
class ActorMemoryExecutionContext:
    actor_id: str
    room_id: str
    conversation_id: str
    generation_request_id: str
    source_event_id: int
    execution_mode: str
    profile_id: str

    def __post_init__(self) -> None:
        if self.actor_id not in {"jiao", "laoke"}:
            raise ValueError("actor memory tools require a bound actor")
        if self.room_id not in _ROOM_WRITE_SCOPES:
            raise ValueError("unknown room")
        if self.execution_mode not in {"private", "group", "bedroom"}:
            raise ValueError("invalid execution mode")

    @property
    def policy(self):
        return build_retrieval_policy(self.actor_id, self.room_id, room_members(self.room_id))

    @property
    def writable_scopes(self) -> frozenset[str]:
        return frozenset(_ROOM_WRITE_SCOPES[self.room_id] & _ACTOR_WRITE_SCOPES[self.actor_id])


def _positive_int(value: Any, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{field} must be a positive integer")
    return value


def _bounded_text(value: Any, field: str, maximum: int = 20000) -> str:
    if not isinstance(value, str) or not value.strip() or len(value) > maximum:
        raise ValueError(f"{field} must be bounded text")
    return value.strip()


class ActorMemoryToolLibrary:
    def __init__(self, store) -> None:
        self.store = store

    async def call(
        self,
        context: ActorMemoryExecutionContext,
        tool_call_id: str,
        name: str,
        arguments: Mapping[str, Any],
    ) -> dict[str, Any]:
        if name not in ACTOR_MEMORY_TOOL_NAMES or not isinstance(arguments, Mapping):
            raise ValueError("unknown actor memory tool")
        if not isinstance(tool_call_id, str) or not tool_call_id or len(tool_call_id) > 200:
            raise ValueError("tool_call_id is invalid")
        args = dict(arguments)
        if "actor_id" in args:
            raise PermissionError("actor identity cannot come from tool arguments")
        if name == "search_memory":
            return {"memories": await self.store.search(_bounded_text(args.get("query"), "query", 2000), context.policy, int(args.get("limit", 20)))}
        if name == "list_memories":
            status = args.get("status", "active")
            if status not in {"active", "stale", "superseded"}:
                raise ValueError("invalid memory status")
            return {"memories": await self.store.list(context.policy, status, int(args.get("limit", 50)))}
        if name == "get_memory":
            return {"memory": await self.store.get(_positive_int(args.get("memory_id"), "memory_id"), context.policy)}

        await self._validate_mutation(context, name, args)
        return await self.store.stage(context, tool_call_id, name, args)

    async def _validate_mutation(self, context, name: str, args: dict[str, Any]) -> None:
        if name in {"write_memory", "propose_memory_candidate"}:
            _bounded_text(args.get("content"), "content")
            self._validate_scope(context, args.get("scope"))
            self._validate_perspective(context, args.get("perspective"))
            if args.get("memory_type") not in {"fact", "inference"}:
                raise ValueError("invalid memory_type")
            if not isinstance(args.get("confidential"), bool):
                raise ValueError("confidential must be boolean")
            self._validate_confidential(context, args["scope"], args["confidential"])
            self._validate_importance(args.get("importance"))
            self._validate_evidence(args.get("evidence_event_ids"))
            return
        target_ids = args.get("memory_ids") if name == "merge_memories" else [args.get("memory_id")]
        if not isinstance(target_ids, list) or len(target_ids) < (2 if name == "merge_memories" else 1):
            raise ValueError("memory target is invalid")
        records = [await self.store.get(_positive_int(value, "memory_id"), context.policy) for value in target_ids]
        if name == "change_scope":
            self._validate_scope(context, args.get("scope"))
            if records[0].get("confidential"):
                self._validate_confidential(context, args["scope"], True)
        elif name == "set_confidential":
            if not isinstance(args.get("confidential"), bool):
                raise ValueError("confidential must be boolean")
            self._validate_confidential(context, records[0]["scope"], args["confidential"])
        elif name == "set_perspective":
            self._validate_perspective(context, args.get("perspective"))
        elif name == "set_memory_type" and args.get("memory_type") not in {"fact", "inference"}:
            raise ValueError("invalid memory_type")
        elif name == "set_memory_status" and args.get("status") not in {"active", "stale"}:
            raise ValueError("invalid memory status")
        elif name == "set_importance":
            self._validate_importance(args.get("importance"))
        elif name in {"add_evidence", "remove_evidence"}:
            _positive_int(args.get("event_id"), "event_id")
        elif name == "update_memory":
            if "content" not in args and "importance" not in args:
                raise ValueError("update requires content or importance")
            if "content" in args:
                _bounded_text(args["content"], "content")
            if "importance" in args:
                self._validate_importance(args["importance"])
        elif name == "merge_memories":
            if len({row["scope"] for row in records}) != 1:
                raise ValueError("merged memories must share a scope")
            _bounded_text(args.get("content"), "content")
            self._validate_importance(args.get("importance"))
        elif name == "supersede_memory":
            _bounded_text(args.get("content"), "content")
            if args.get("memory_type") not in {"fact", "inference"}:
                raise ValueError("invalid memory_type")
            self._validate_importance(args.get("importance"))

    @staticmethod
    def _validate_importance(value: Any) -> None:
        if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= 10:
            raise ValueError("importance must be between 1 and 10")

    @staticmethod
    def _validate_evidence(value: Any) -> None:
        if not isinstance(value, list) or any(_positive_int(item, "evidence_event_id") < 1 for item in value):
            raise ValueError("evidence_event_ids are invalid")

    @staticmethod
    def _validate_perspective(context, value: Any) -> None:
        if value != context.actor_id:
            raise PermissionError("actor cannot claim an unconfirmed perspective")

    @staticmethod
    def _validate_confidential(context, scope: str, value: bool) -> None:
        if value and scope != _PAIRWISE_SCOPE[context.actor_id]:
            raise PermissionError("actor confidential memory is limited to its own pairwise scope")

    @staticmethod
    def _validate_scope(context, scope: Any) -> None:
        if scope not in context.writable_scopes:
            raise PermissionError("memory scope is not writable in this execution context")

    async def commit_accepted(self, context: ActorMemoryExecutionContext, *, accepted_event_id: int) -> dict[str, Any]:
        return await self.store.commit(context, _positive_int(accepted_event_id, "accepted_event_id"))

    async def discard(self, context: ActorMemoryExecutionContext) -> dict[str, Any]:
        return await self.store.discard(context)


class InMemoryActorMemoryToolStore:
    """Deterministic fixture store; production uses the PostgreSQL store below."""

    def __init__(self) -> None:
        self.records: dict[int, dict[str, Any]] = {}
        self.stages: dict[tuple[str, str, str], dict[str, Any]] = {}
        self.receipts: dict[tuple[str, str], dict[str, Any]] = {}
        self.next_id = 1

    def seed(self, *, content: str, scope: str, perspective: str, confidential: bool = False, source_kind: str = "synthetic_test") -> int:
        memory_id = self.next_id
        self.next_id += 1
        self.records[memory_id] = {
            "id": memory_id, "content": content, "scope": scope,
            "memory_type": "fact", "perspective": perspective,
            "confidential": confidential, "source_kind": source_kind,
            "status": "active", "importance": 5, "evidence": [],
            "superseded_by": None,
        }
        return memory_id

    @staticmethod
    def _visible(row: Mapping[str, Any], policy) -> bool:
        return row["scope"] in policy.allowed_scopes and (
            not row["confidential"] or row["scope"] in policy.confidential_scopes
        )

    async def search(self, query: str, policy, limit: int) -> list[dict[str, Any]]:
        return [dict(row) for row in self.records.values() if row["status"] == "active" and self._visible(row, policy) and query.casefold() in row["content"].casefold()][:max(1, min(limit, 100))]

    async def list(self, policy, status: str, limit: int) -> list[dict[str, Any]]:
        return [dict(row) for row in self.records.values() if row["status"] == status and self._visible(row, policy)][:max(1, min(limit, 100))]

    async def get(self, memory_id: int, policy) -> dict[str, Any]:
        row = self.records.get(memory_id)
        if row is None:
            raise KeyError(memory_id)
        if not self._visible(row, policy):
            raise PermissionError("memory is outside actor policy")
        return dict(row)

    async def stage(self, context, tool_call_id: str, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        key = (context.actor_id, context.generation_request_id, tool_call_id)
        payload_hash = hashlib.sha256(json.dumps({"name": name, "arguments": arguments}, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        existing = self.stages.get(key)
        if existing:
            if existing["payload_hash"] != payload_hash:
                raise ValueError("tool call identity conflict")
            return dict(existing["ack"])
        ack = {"status": "staged", "tool_call_id": tool_call_id, "action": name}
        self.stages[key] = {"context": context, "name": name, "arguments": dict(arguments), "payload_hash": payload_hash, "status": "staged", "ack": ack}
        return dict(ack)

    async def commit(self, context, accepted_event_id: int) -> dict[str, Any]:
        receipt_key = (context.actor_id, context.generation_request_id)
        if receipt_key in self.receipts:
            return dict(self.receipts[receipt_key])
        resulting = []
        for stage in self.stages.values():
            if stage["context"].actor_id != context.actor_id or stage["context"].generation_request_id != context.generation_request_id or stage["status"] != "staged":
                continue
            stage_ids = self._apply(context, stage["name"], stage["arguments"], accepted_event_id)
            resulting.extend(stage_ids)
            stage["status"] = "committed"
            stage["resulting_memory_ids"] = list(stage_ids)
        receipt = {"status": "committed", "actor_id": context.actor_id, "generation_request_id": context.generation_request_id, "accepted_event_id": accepted_event_id, "resulting_memory_ids": sorted(set(resulting))}
        self.receipts[receipt_key] = receipt
        return dict(receipt)

    async def discard(self, context) -> dict[str, Any]:
        for stage in self.stages.values():
            if stage["context"].actor_id == context.actor_id and stage["context"].generation_request_id == context.generation_request_id and stage["status"] == "staged":
                stage["status"] = "discarded"
        return {"status": "discarded", "actor_id": context.actor_id, "generation_request_id": context.generation_request_id}

    def _apply(self, context, name: str, args: dict[str, Any], accepted_event_id: int) -> list[int]:
        if name in {"write_memory", "propose_memory_candidate"}:
            normalized = " ".join(args["content"].split()).casefold()
            for row in self.records.values():
                if row["status"] == "active" and row["scope"] == args["scope"] and row["perspective"] == args["perspective"] and " ".join(row["content"].split()).casefold() == normalized:
                    row["evidence"] = sorted(set(row["evidence"]) | set(args["evidence_event_ids"]))
                    return [row["id"]]
            memory_id = self.seed(content=args["content"], scope=args["scope"], perspective=args["perspective"], confidential=args["confidential"], source_kind="agent_candidate" if name == "propose_memory_candidate" else "actor_tool")
            row = self.records[memory_id]
            row.update(memory_type=args["memory_type"], importance=args["importance"], evidence=sorted(set(args["evidence_event_ids"])), provenance={"actor_id": context.actor_id, "generation_request_id": context.generation_request_id, "source_event_id": accepted_event_id, "tool_action": name})
            return [memory_id]
        ids = args.get("memory_ids") or [args["memory_id"]]
        rows = [self.records[int(memory_id)] for memory_id in ids]
        if name == "update_memory":
            for field in ("content", "importance"):
                if field in args:
                    rows[0][field] = args[field]
        elif name == "delete_memory": rows[0]["status"] = "stale"
        elif name == "restore_memory": rows[0]["status"], rows[0]["superseded_by"] = "active", None
        elif name == "change_scope": rows[0]["scope"] = args["scope"]
        elif name == "set_confidential": rows[0]["confidential"] = args["confidential"]
        elif name == "set_perspective": rows[0]["perspective"] = args["perspective"]
        elif name == "set_memory_type": rows[0]["memory_type"] = args["memory_type"]
        elif name == "set_memory_status": rows[0]["status"] = args["status"]
        elif name == "set_importance": rows[0]["importance"] = args["importance"]
        elif name == "add_evidence": rows[0]["evidence"] = sorted(set(rows[0]["evidence"]) | {args["event_id"]})
        elif name == "remove_evidence": rows[0]["evidence"] = [item for item in rows[0]["evidence"] if item != args["event_id"]]
        elif name == "merge_memories":
            memory_id = self.seed(content=args["content"], scope=rows[0]["scope"], perspective=context.actor_id, confidential=any(row["confidential"] for row in rows), source_kind="actor_tool")
            self.records[memory_id]["importance"] = args["importance"]
            self.records[memory_id]["evidence"] = sorted({item for row in rows for item in row["evidence"]})
            for row in rows:
                row["status"], row["superseded_by"] = "superseded", memory_id
            return [memory_id, *ids]
        elif name == "supersede_memory":
            memory_id = self.seed(content=args["content"], scope=rows[0]["scope"], perspective=rows[0]["perspective"], confidential=rows[0]["confidential"], source_kind="actor_tool")
            self.records[memory_id].update(memory_type=args["memory_type"], importance=args["importance"], evidence=list(rows[0]["evidence"]))
            rows[0]["status"], rows[0]["superseded_by"] = "superseded", memory_id
            return [memory_id, rows[0]["id"]]
        return [int(value) for value in ids]

    async def list_active(self) -> list[dict[str, Any]]:
        return [dict(row) for row in self.records.values() if row["status"] == "active"]

    async def all_records(self) -> list[dict[str, Any]]:
        return [dict(row) for row in self.records.values()]

    async def audit(self, limit: int = 100) -> list[dict[str, Any]]:
        rows = []
        for stage in reversed(tuple(self.stages.values())):
            context = stage["context"]
            rows.append({
                "actor_id": context.actor_id,
                "generation_request_id": context.generation_request_id,
                "room_id": context.room_id,
                "conversation_id": context.conversation_id,
                "source_event_id": context.source_event_id,
                "action": stage["name"], "status": stage["status"],
                "resulting_memory_ids": stage.get("resulting_memory_ids", []),
            })
        return rows[:max(1, min(limit, 500))]


class PostgresActorMemoryToolStore:
    """Durable staged mutations and policy-filtered reads for real executions."""

    def __init__(self, pool_factory=get_pool) -> None:
        self.pool_factory = pool_factory

    @staticmethod
    def _clean_row(row) -> dict[str, Any]:
        value = dict(row)
        for field in ("provenance", "evidence", "resulting_memory_ids"):
            if isinstance(value.get(field), str):
                value[field] = json.loads(value[field])
        return value

    async def search(self, query: str, policy, limit: int) -> list[dict[str, Any]]:
        result = await search_authorized_memories(query, policy, limit=max(1, min(limit, 100)))
        return [dict(row) for row in result.memories]

    async def list(self, policy, status: str, limit: int) -> list[dict[str, Any]]:
        pool = await self.pool_factory()
        async with pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT id,content,importance,scope,memory_type,perspective,
                       confidential,source_kind,status,evidence,superseded_by,provenance
                FROM memories
                WHERE status=$1 AND scope=ANY($2::text[])
                  AND (confidential=FALSE OR scope=ANY($3::text[]))
                ORDER BY id DESC LIMIT $4
                """,
                status, list(policy.allowed_scopes), list(policy.confidential_scopes),
                max(1, min(limit, 100)),
            )
        return [self._clean_row(row) for row in rows]

    async def get(self, memory_id: int, policy) -> dict[str, Any]:
        pool = await self.pool_factory()
        async with pool.acquire() as conn:
            row = await self._fetch_authorized(conn, memory_id, policy)
        if row is None:
            raise PermissionError("memory is outside actor policy")
        return self._clean_row(row)

    async def audit(self, limit: int = 100) -> list[dict[str, Any]]:
        pool = await self.pool_factory()
        async with pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT actor_id,generation_request_id,room_id,conversation_id,
                       source_event_id,action,status,accepted_event_id,
                       resulting_memory_ids,created_at,committed_at,discarded_at
                FROM actor_memory_tool_stages
                ORDER BY created_at DESC LIMIT $1
                """,
                max(1, min(limit, 500)),
            )
        return [self._clean_row(row) for row in rows]

    async def _fetch_authorized(self, conn, memory_id: int, policy, *, lock: bool = False):
        return await conn.fetchrow(
            """
            SELECT id,content,importance,scope,memory_type,perspective,
                   confidential,source_kind,status,evidence,superseded_by,provenance
            FROM memories
            WHERE id=$1 AND scope=ANY($2::text[])
              AND (confidential=FALSE OR scope=ANY($3::text[]))
            """ + (" FOR UPDATE" if lock else ""),
            memory_id, list(policy.allowed_scopes), list(policy.confidential_scopes),
        )

    async def stage(self, context, tool_call_id: str, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        payload = json.dumps(arguments, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        payload_hash = hashlib.sha256(f"{name}:{payload}".encode()).hexdigest()
        pool = await self.pool_factory()
        async with pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                INSERT INTO actor_memory_tool_stages (
                    actor_id,generation_request_id,tool_call_id,room_id,
                    conversation_id,source_event_id,execution_mode,profile_id,
                    action,arguments_json,payload_hash,status
                ) VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10::jsonb,$11,'staged')
                ON CONFLICT (actor_id,generation_request_id,tool_call_id) DO NOTHING
                RETURNING payload_hash,status
                """,
                context.actor_id, context.generation_request_id, tool_call_id,
                context.room_id, context.conversation_id, context.source_event_id,
                context.execution_mode, context.profile_id, name, payload, payload_hash,
            )
            if row is None:
                row = await conn.fetchrow(
                    "SELECT payload_hash,status FROM actor_memory_tool_stages "
                    "WHERE actor_id=$1 AND generation_request_id=$2 AND tool_call_id=$3",
                    context.actor_id, context.generation_request_id, tool_call_id,
                )
            if row is None or row["payload_hash"] != payload_hash:
                raise ValueError("tool call identity conflict")
        return {"status": row["status"], "tool_call_id": tool_call_id, "action": name}

    async def commit(self, context, accepted_event_id: int) -> dict[str, Any]:
        pool = await self.pool_factory()
        async with pool.acquire() as conn:
            async with conn.transaction():
                await conn.execute(
                    "SELECT pg_advisory_xact_lock(hashtext($1))",
                    f"actor-memory:{context.actor_id}:{context.generation_request_id}",
                )
                stages = await conn.fetch(
                    """
                    SELECT tool_call_id,room_id,conversation_id,source_event_id,
                           execution_mode,profile_id,action,arguments_json,status,
                           accepted_event_id,resulting_memory_ids
                    FROM actor_memory_tool_stages
                    WHERE actor_id=$1 AND generation_request_id=$2
                    ORDER BY created_at,tool_call_id FOR UPDATE
                    """,
                    context.actor_id, context.generation_request_id,
                )
                if not stages:
                    return {"status": "committed", "actor_id": context.actor_id, "generation_request_id": context.generation_request_id, "accepted_event_id": accepted_event_id, "resulting_memory_ids": []}
                stages = [row for row in stages if row["status"] != "discarded"]
                if not stages:
                    return {"status": "committed", "actor_id": context.actor_id, "generation_request_id": context.generation_request_id, "accepted_event_id": accepted_event_id, "resulting_memory_ids": []}
                if all(row["status"] == "committed" for row in stages):
                    prior = {row["accepted_event_id"] for row in stages}
                    if prior != {accepted_event_id}:
                        raise ValueError("accepted final identity conflict")
                    ids = sorted({int(item) for row in stages for item in (row["resulting_memory_ids"] or [])})
                    return {"status": "committed", "actor_id": context.actor_id, "generation_request_id": context.generation_request_id, "accepted_event_id": accepted_event_id, "resulting_memory_ids": ids}
                if any(
                    row["room_id"] != context.room_id
                    or row["conversation_id"] != context.conversation_id
                    or row["source_event_id"] != context.source_event_id
                    or row["execution_mode"] != context.execution_mode
                    for row in stages
                ):
                    raise PermissionError("memory stage coordinates changed")
                resulting: list[int] = []
                for stage in stages:
                    args = stage["arguments_json"]
                    if isinstance(args, str):
                        args = json.loads(args)
                    ids = await self._apply(conn, context, stage["action"], dict(args), accepted_event_id)
                    resulting.extend(ids)
                    await conn.execute(
                        """
                        UPDATE actor_memory_tool_stages
                        SET status='committed',accepted_event_id=$1,
                            resulting_memory_ids=$2::jsonb,committed_at=NOW()
                        WHERE actor_id=$3 AND generation_request_id=$4 AND tool_call_id=$5
                        """,
                        accepted_event_id, json.dumps(ids), context.actor_id,
                        context.generation_request_id, stage["tool_call_id"],
                    )
        return {"status": "committed", "actor_id": context.actor_id, "generation_request_id": context.generation_request_id, "accepted_event_id": accepted_event_id, "resulting_memory_ids": sorted(set(resulting))}

    async def discard(self, context) -> dict[str, Any]:
        pool = await self.pool_factory()
        async with pool.acquire() as conn:
            await conn.execute(
                """
                UPDATE actor_memory_tool_stages SET status='discarded',discarded_at=NOW()
                WHERE actor_id=$1 AND generation_request_id=$2 AND status='staged'
                """,
                context.actor_id, context.generation_request_id,
            )
        return {"status": "discarded", "actor_id": context.actor_id, "generation_request_id": context.generation_request_id}

    async def _authorized_rows(self, conn, context, ids: list[int]):
        rows = []
        for memory_id in ids:
            row = await self._fetch_authorized(conn, int(memory_id), context.policy, lock=True)
            if row is None:
                raise PermissionError("memory mutation target is outside actor policy")
            rows.append(self._clean_row(row))
        return rows

    @staticmethod
    def _write(context, args, accepted_event_id, *, source_kind: SourceKind, template=None):
        evidence = sorted({int(item) for item in args.get("evidence_event_ids", template.get("evidence", []) if template else [])})
        return MemoryWrite(
            content=args["content"],
            scope=MemoryScope(args.get("scope", template["scope"] if template else _PAIRWISE_SCOPE[context.actor_id])),
            memory_type=MemoryType(args.get("memory_type", template["memory_type"] if template else "fact")),
            perspective=Perspective(args.get("perspective", template["perspective"] if template else context.actor_id)),
            confidential=bool(args.get("confidential", template["confidential"] if template else False)),
            source_kind=source_kind,
            status=MemoryStatus.ACTIVE,
            evidence_count=len(evidence),
            provenance={
                "actor_id": context.actor_id,
                "room_id": context.room_id,
                "conversation_id": context.conversation_id,
                "generation_request_id": context.generation_request_id,
                "source_event_id": accepted_event_id,
                "evidence_event_ids": evidence,
                "tool_action": "actor_memory_tool",
            },
        )

    async def _apply(self, conn, context, name: str, args: dict[str, Any], accepted_event_id: int) -> list[int]:
        if name in {"write_memory", "propose_memory_candidate"}:
            write = self._write(
                context, args, accepted_event_id,
                source_kind=SourceKind.AGENT_CANDIDATE if name == "propose_memory_candidate" else SourceKind.ACTOR_TOOL,
            )
            memory_id = await _persist_or_merge_group_memory(conn, write)
            await conn.execute(
                "UPDATE memories SET importance=$1,updated_at=NOW() WHERE id=$2",
                args["importance"], memory_id,
            )
            return [memory_id]
        ids = [int(item) for item in (args.get("memory_ids") or [args["memory_id"]])]
        rows = await self._authorized_rows(conn, context, ids)
        if name == "update_memory":
            fields, values = [], []
            for field in ("content", "importance"):
                if field in args:
                    values.append(args[field]); fields.append(f"{field}=${len(values)}")
            values.append(ids[0])
            await conn.execute(f"UPDATE memories SET {','.join(fields)},updated_at=NOW() WHERE id=${len(values)}", *values)
        elif name == "delete_memory":
            await conn.execute("UPDATE memories SET status='stale',is_active=FALSE,updated_at=NOW() WHERE id=$1", ids[0])
        elif name == "restore_memory":
            await conn.execute("UPDATE memories SET status='active',is_active=TRUE,superseded_by=NULL,updated_at=NOW() WHERE id=$1", ids[0])
        elif name == "change_scope":
            await conn.execute("UPDATE memories SET scope=$1,updated_at=NOW() WHERE id=$2", args["scope"], ids[0])
        elif name == "set_confidential":
            await conn.execute("UPDATE memories SET confidential=$1,updated_at=NOW() WHERE id=$2", args["confidential"], ids[0])
        elif name == "set_perspective":
            await conn.execute("UPDATE memories SET perspective=$1,updated_at=NOW() WHERE id=$2", args["perspective"], ids[0])
        elif name == "set_memory_type":
            await conn.execute("UPDATE memories SET memory_type=$1,updated_at=NOW() WHERE id=$2", args["memory_type"], ids[0])
        elif name == "set_memory_status":
            await conn.execute("UPDATE memories SET status=$1,is_active=($1='active'),updated_at=NOW() WHERE id=$2", args["status"], ids[0])
        elif name == "set_importance":
            await conn.execute("UPDATE memories SET importance=$1,updated_at=NOW() WHERE id=$2", args["importance"], ids[0])
        elif name in {"add_evidence", "remove_evidence"}:
            evidence = {int(item) for item in rows[0].get("evidence") or []}
            if name == "add_evidence": evidence.add(int(args["event_id"]))
            else: evidence.discard(int(args["event_id"]))
            await conn.execute("UPDATE memories SET evidence=$1::jsonb,evidence_count=$2,updated_at=NOW() WHERE id=$3", json.dumps(sorted(evidence)), len(evidence), ids[0])
        elif name == "merge_memories":
            if len({row["scope"] for row in rows}) != 1:
                raise ValueError("merged memories must share a scope")
            template = dict(rows[0]); template["evidence"] = sorted({int(item) for row in rows for item in row.get("evidence") or []})
            new_id = await _persist_or_merge_group_memory(conn, self._write(context, args, accepted_event_id, source_kind=SourceKind.ACTOR_TOOL, template=template))
            await conn.execute("UPDATE memories SET importance=$1,updated_at=NOW() WHERE id=$2", args["importance"], new_id)
            await conn.execute("UPDATE memories SET status='superseded',is_active=FALSE,superseded_by=$1,updated_at=NOW() WHERE id=ANY($2::int[])", new_id, ids)
            return [new_id, *ids]
        elif name == "supersede_memory":
            new_id = await _persist_or_merge_group_memory(conn, self._write(context, args, accepted_event_id, source_kind=SourceKind.ACTOR_TOOL, template=rows[0]))
            await conn.execute("UPDATE memories SET importance=$1,updated_at=NOW() WHERE id=$2", args["importance"], new_id)
            await conn.execute("UPDATE memories SET status='superseded',is_active=FALSE,superseded_by=$1,updated_at=NOW() WHERE id=$2", new_id, ids[0])
            return [new_id, ids[0]]
        return ids
