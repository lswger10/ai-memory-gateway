"""Gateway-owned Gemini resource handles; no transcript or summary writes."""
from __future__ import annotations

import asyncio
import hashlib
import json
import re
import uuid
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timedelta, timezone

import anyio

from model_execution import ExplicitCacheUnavailable
from model_execution_contracts import ProviderUsage
from model_usage_store import execution_receipt_draft, record_provider_attempt


@dataclass(frozen=True)
class GeminiCacheResource:
    cache_key: str
    actor_id: str
    room_id: str
    conversation_id: str
    execution_mode: str
    profile_id: str
    profile_revision: int
    namespace: str
    attempt_id: str
    prefix_hash: str
    history_count: int
    cutoff_event_id: int
    summary_version: int
    created_at: datetime
    status: str = "creating"
    resource_name: str | None = None
    expires_at: datetime | None = None
    last_error: str | None = None

    @property
    def scope(self):
        return (self.actor_id, self.room_id, self.conversation_id, self.execution_mode)


class InMemoryGeminiCacheStore:
    def __init__(self):
        self.resources = {}
        self.lock = asyncio.Lock()

    async def list_resources(self, scope=None):
        return tuple(row for row in self.resources.values() if scope is None or row.scope == scope)

    async def resume(self, key, attempt_id, scope, now):
        async with self.lock:
            row = self.resources.get(key)
            if row is None or row.attempt_id != attempt_id or row.scope != scope or row.status == "ready" or now < row.created_at + timedelta(hours=1):
                return False
            del self.resources[key]
            return True

    async def reserve(self, item, expected_attempt):
        async with self.lock:
            old = self.resources.get(item.cache_key)
            if (old.attempt_id if old else None) != expected_attempt or any(
                row.scope == item.scope and row.status != "ready" for row in self.resources.values()
            ):
                return False
            self.resources[item.cache_key] = item
            return True

    async def save(self, item):
        async with self.lock:
            old = self.resources.get(item.cache_key)
            if old is None or old.attempt_id != item.attempt_id:
                raise ExplicitCacheUnavailable("gemini_cache_state_conflict")
            self.resources[item.cache_key] = item


class PostgresGeminiCacheStore:
    def __init__(self, pool_factory): self.pool_factory = pool_factory

    async def list_resources(self, scope=None):
        pool = await self.pool_factory()
        async with pool.acquire() as conn:
            rows = await conn.fetch("SELECT * FROM gemini_explicit_cache_resources" +
                (" WHERE actor_id=$1 AND room_id=$2 AND conversation_id=$3 AND execution_mode=$4" if scope else ""), *(scope or ()))
        return tuple(GeminiCacheResource(**dict(row)) for row in rows)

    async def resume(self, key, attempt_id, scope, now):
        pool = await self.pool_factory()
        async with pool.acquire() as conn:
            result = await conn.execute("""DELETE FROM gemini_explicit_cache_resources
                WHERE cache_key=$1 AND attempt_id=$2 AND actor_id=$3 AND room_id=$4
                AND conversation_id=$5 AND execution_mode=$6 AND status <> 'ready'
                AND created_at <= $7""", key, attempt_id, *scope, now - timedelta(hours=1))
            return result == "DELETE 1"

    async def reserve(self, item, expected_attempt):
        pool = await self.pool_factory()
        async with pool.acquire() as conn, conn.transaction():
            # Serialize missing-row claims too, including alternate Profiles in the same scope.
            await conn.execute("SELECT pg_advisory_xact_lock(hashtextextended($1,0))", json.dumps(item.scope))
            blocked = await conn.fetchval("""SELECT EXISTS(SELECT 1 FROM gemini_explicit_cache_resources
                WHERE actor_id=$1 AND room_id=$2 AND conversation_id=$3 AND execution_mode=$4
                AND status <> 'ready')""", *item.scope)
            old = await conn.fetchval("SELECT attempt_id FROM gemini_explicit_cache_resources WHERE cache_key=$1", item.cache_key)
            if blocked or old != expected_attempt:
                return False
            values = asdict(item)
            columns = tuple(values)  # Dataclass-owned identifiers, never request input.
            await conn.execute("INSERT INTO gemini_explicit_cache_resources(" + ",".join(columns) + ") VALUES(" +
                ",".join(f"${i}" for i in range(1, len(columns) + 1)) + ") ON CONFLICT(cache_key) DO UPDATE SET " +
                ",".join(f"{key}=EXCLUDED.{key}" for key in columns if key != "cache_key"), *values.values())
            return True

    async def save(self, item):
        pool = await self.pool_factory()
        async with pool.acquire() as conn:
            result = await conn.execute("""UPDATE gemini_explicit_cache_resources
                SET status=$3,resource_name=$4,expires_at=$5,last_error=$6
                WHERE cache_key=$1 AND attempt_id=$2""", item.cache_key, item.attempt_id,
                item.status, item.resource_name, item.expires_at, item.last_error)
            if result != "UPDATE 1":
                raise ExplicitCacheUnavailable("gemini_cache_state_conflict")


def _fingerprint(context, body, history_count):
    payload = [context.static_system, context.stable_summary, context.summary_version,
               context.compressed_up_to_event_id, context.stable_history[:history_count], body.get("tools", [])]
    return hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


class GeminiExplicitCache:
    def __init__(self, *, store, pins, profiles, usage, now=None):
        self.store, self.pins, self.profiles, self.usage = store, pins, profiles, usage
        self.now = now or (lambda: datetime.now(timezone.utc))

    async def prepare(self, *, profile, request, context, namespace, body, headers, transport, probe=False):
        if request.execution_mode not in {"private", "group"} or request.execution_kind != "full":
            return body
        scope = (request.actor_id, request.room_id, request.conversation_id, request.execution_mode)
        pin = await self.pins.get_pin(f"{request.execution_mode}:{request.conversation_id}")
        if probe and (not request.generation_request_id.startswith("cache-probe:") or context.tool_schema_hash != "cache-probe.tools.v1"):
            raise ExplicitCacheUnavailable("gemini_cache_probe_scope_invalid")
        if not probe and (pin is None or not pin.enabled or pin.room_id != request.room_id or request.actor_id not in pin.actors):
            return body
        if not probe and (not profile.selectable or not await self.profiles.has_verified_probe(
            profile.profile_id, profile.revision, "gemini_explicit_cache"
        )):
            raise ExplicitCacheUnavailable("gemini_explicit_cache_unverified")
        key = hashlib.sha256(json.dumps([scope, namespace]).encode()).hexdigest()
        rows = await self.store.list_resources(scope)
        if any(row.scope == scope and row.status != "ready" for row in rows):
            raise ExplicitCacheUnavailable("gemini_cache_creation_paused")
        resource = next((row for row in rows if row.cache_key == key), None)
        now = self.now()
        if resource is None or resource.expires_at <= now or resource.history_count > len(context.stable_history) or (
            resource.prefix_hash != _fingerprint(context, body, resource.history_count)
        ):
            expected = resource.attempt_id if resource else None
            cutoff = context.compressed_up_to_event_id or 0
            if context.stable_history:
                try:
                    cutoff = int(json.loads(context.stable_history[-1])["event_id"])
                except (ValueError, KeyError, TypeError) as exc:
                    raise ExplicitCacheUnavailable("gemini_cache_history_identity_invalid") from exc
            resource = GeminiCacheResource(key, *scope, profile.profile_id, profile.revision, namespace,
                str(uuid.uuid4()), _fingerprint(context, body, len(context.stable_history)),
                len(context.stable_history), cutoff, context.summary_version or 0, now)
            if not await self.store.reserve(resource, expected):
                raise ExplicitCacheUnavailable("gemini_cache_creation_in_progress_or_paused")
            resource = await self._create(resource, profile, request, context, body, headers, transport)
        tail = dict(body)
        tail.pop("systemInstruction", None)
        tail.pop("tools", None)
        tail["contents"] = body["contents"][resource.history_count + bool(context.stable_summary):]
        tail["cachedContent"] = resource.resource_name
        return tail

    async def _create(self, resource, profile, request, context, body, headers, transport):
        payload = {"model": profile.model, "ttl": "3600s", "systemInstruction": body["systemInstruction"],
                   "contents": body["contents"][:resource.history_count + bool(context.stable_summary)]}
        if body.get("tools"):
            payload["tools"] = body["tools"]
        draft = execution_receipt_draft(profile=profile, generation_request_id=request.generation_request_id,
            actor_id=request.actor_id, room_id=request.room_id, conversation_id=request.conversation_id,
            context=context, cache_namespace=resource.namespace, execution_purpose="cache_create")
        usage, received, status = ProviderUsage.from_provider_values(), False, "failed"
        try:
            response = await transport.request(pool_key=profile.route_id, base_url=profile.base_url,
                headers=headers, method="POST", path="/gemini/v1beta/cachedContents", json_body=payload)
            if response.status_code >= 400:
                raise ExplicitCacheUnavailable(f"gemini_cache_create_http_{response.status_code}")
            value = response.json()
            metadata = value.get("usageMetadata", {})
            usage = ProviderUsage.from_provider_values(cache_creation_input_tokens=metadata.get("totalTokenCount"))
            received = usage.cache_creation_input_tokens is not None
            name = value.get("name", "")
            expires = datetime.fromisoformat(value["expireTime"].replace("Z", "+00:00"))
            if (not re.fullmatch(
                    r"(?:projects/[A-Za-z0-9_-]+/locations/[A-Za-z0-9_-]+/)?cachedContents/[A-Za-z0-9_-]+", name)
                or expires.tzinfo is None or not self.now() < expires <= self.now() + timedelta(seconds=3605)
                or value.get("model", "").removeprefix("models/") != profile.model.removeprefix("models/")):
                raise ExplicitCacheUnavailable("gemini_cache_resource_invalid")
            resource = replace(resource, status="ready", resource_name=name, expires_at=expires)
            await self.store.save(resource)
            status = "succeeded"
            return resource
        except (Exception, asyncio.CancelledError) as exc:
            reason = str(exc) if isinstance(exc, ExplicitCacheUnavailable) else "gemini_cache_create_outcome_unknown"
            with anyio.fail_after(5, shield=True):
                await self.store.save(replace(resource, status="paused", last_error=reason))
            if isinstance(exc, asyncio.CancelledError):
                raise
            raise ExplicitCacheUnavailable(reason) from exc
        finally:
            await record_provider_attempt(self.usage, draft, resource.attempt_id, usage, status, received, "unverified")

    async def pause_generation(self, *, request, namespace, error):
        http_status = re.fullmatch(r"provider HTTP ([45][0-9]{2})", str(error))
        reason = (f"gemini_cached_http_{http_status[1]}" if http_status else {
            "provider stream omitted its terminal event": "gemini_cached_incomplete_stream",
            "provider Gemini response did not complete": "gemini_cached_incomplete_response",
            "provider Gemini request blocked or failed": "gemini_cached_provider_blocked",
            "provider completed without reply text": "gemini_cached_empty_reply",
            "provider tool loop exceeded limit": "gemini_cached_tool_limit",
        }.get(str(error), "gemini_cached_generation_failed"))
        if isinstance(error, TimeoutError) or type(error).__name__ in {"ReadTimeout", "ConnectTimeout", "WriteTimeout", "PoolTimeout"}:
            reason = "gemini_cached_transport_timeout"
        scope = (request.actor_id, request.room_id, request.conversation_id, request.execution_mode)
        with anyio.fail_after(5, shield=True):
            for row in await self.store.list_resources(scope):
                if row.scope == scope and row.namespace == namespace:
                    await self.store.save(replace(row, status="paused", last_error=reason))
        return reason
