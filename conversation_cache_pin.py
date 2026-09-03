from __future__ import annotations

import asyncio
import uuid
from dataclasses import dataclass, field, replace
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from typing import Any, Mapping, Protocol

from model_execution import ContextBundle, ProviderRunUnavailable
from model_execution_contracts import ProviderUsage
from model_usage_store import (
    ExecutionReceiptDraft,
    build_cache_namespace,
    build_stable_prefix_hash,
)


PRIVATE_ROOM_ACTORS = {
    "room_weiwei_jiao": "jiao",
    "room_weiwei_laoke": "laoke",
}
GROUP_ROOM_ID = "room_group_home"


class CachePinError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class CachePinActorState:
    actor_id: str
    status: str = "pending"
    profile_id: str | None = None
    last_keepalive_at: datetime | None = None
    next_keepalive_at: datetime | None = None
    call_count: int = 0
    cache_read_input_tokens: int | None = None
    last_error: str | None = None


@dataclass(frozen=True, slots=True)
class ConversationCachePin:
    pin_id: str
    room_id: str
    conversation_id: str
    execution_mode: str
    enabled: bool
    bedroom_session_id: str | None = None
    bedroom_actor_id: str | None = None
    actors: Mapping[str, CachePinActorState] = field(default_factory=dict)

    @property
    def status(self) -> str:
        if not self.enabled:
            return "off"
        if any(state.status == "active" for state in self.actors.values()):
            return "active"
        return "paused"


@dataclass(frozen=True, slots=True)
class CachePinRunResult:
    calls: int


class ConversationCachePinStore(Protocol):
    async def upsert_pin(self, pin: ConversationCachePin) -> ConversationCachePin: ...
    async def get_pin(self, pin_id: str) -> ConversationCachePin | None: ...
    async def list_pins(self) -> tuple[ConversationCachePin, ...]: ...
    async def save_actor_state(
        self, pin_id: str, state: CachePinActorState
    ) -> None: ...


class InMemoryConversationCachePinStore:
    def __init__(self) -> None:
        self._pins: dict[str, ConversationCachePin] = {}
        self._lock = asyncio.Lock()

    async def upsert_pin(self, pin: ConversationCachePin) -> ConversationCachePin:
        async with self._lock:
            previous = self._pins.get(pin.pin_id)
            actors = dict(previous.actors) if previous is not None else dict(pin.actors)
            saved = replace(pin, actors=actors)
            self._pins[pin.pin_id] = saved
            return saved

    async def get_pin(self, pin_id: str) -> ConversationCachePin | None:
        async with self._lock:
            return self._pins.get(pin_id)

    async def list_pins(self) -> tuple[ConversationCachePin, ...]:
        async with self._lock:
            return tuple(self._pins[key] for key in sorted(self._pins))

    async def save_actor_state(
        self, pin_id: str, state: CachePinActorState
    ) -> None:
        async with self._lock:
            pin = self._pins[pin_id]
            actors = dict(pin.actors)
            actors[state.actor_id] = state
            self._pins[pin_id] = replace(pin, actors=actors)


class PostgresConversationCachePinStore:
    def __init__(self, pool_factory) -> None:
        self._pool_factory = pool_factory

    @staticmethod
    def _actor_state(row) -> CachePinActorState:
        return CachePinActorState(
            actor_id=row["actor_id"],
            status=row["status"],
            profile_id=row["profile_id"],
            last_keepalive_at=row["last_keepalive_at"],
            next_keepalive_at=row["next_keepalive_at"],
            call_count=int(row["call_count"] or 0),
            cache_read_input_tokens=row["cache_read_input_tokens"],
            last_error=row["last_error"],
        )

    async def _from_row(self, conn, row) -> ConversationCachePin:
        states = await conn.fetch(
            "SELECT * FROM conversation_cache_pin_actor_state WHERE pin_id=$1 ORDER BY actor_id",
            row["pin_id"],
        )
        return ConversationCachePin(
            pin_id=row["pin_id"],
            room_id=row["room_id"],
            conversation_id=row["conversation_id"],
            execution_mode=row["execution_mode"],
            enabled=bool(row["enabled"]),
            bedroom_session_id=row["bedroom_session_id"],
            bedroom_actor_id=row["bedroom_actor_id"],
            actors={state["actor_id"]: self._actor_state(state) for state in states},
        )

    async def upsert_pin(self, pin: ConversationCachePin) -> ConversationCachePin:
        pool = await self._pool_factory()
        async with pool.acquire() as conn:
            await conn.execute(
                """INSERT INTO conversation_cache_pins(
                     pin_id,room_id,conversation_id,execution_mode,
                     bedroom_session_id,bedroom_actor_id,enabled
                   ) VALUES($1,$2,$3,$4,$5,$6,$7)
                   ON CONFLICT(pin_id) DO UPDATE SET
                     room_id=EXCLUDED.room_id,
                     conversation_id=EXCLUDED.conversation_id,
                     execution_mode=EXCLUDED.execution_mode,
                     bedroom_session_id=EXCLUDED.bedroom_session_id,
                     bedroom_actor_id=EXCLUDED.bedroom_actor_id,
                     enabled=EXCLUDED.enabled,
                     updated_at=NOW()""",
                pin.pin_id,
                pin.room_id,
                pin.conversation_id,
                pin.execution_mode,
                pin.bedroom_session_id,
                pin.bedroom_actor_id,
                pin.enabled,
            )
            row = await conn.fetchrow(
                "SELECT * FROM conversation_cache_pins WHERE pin_id=$1", pin.pin_id
            )
            return await self._from_row(conn, row)

    async def get_pin(self, pin_id: str) -> ConversationCachePin | None:
        pool = await self._pool_factory()
        async with pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT * FROM conversation_cache_pins WHERE pin_id=$1", pin_id
            )
            return None if row is None else await self._from_row(conn, row)

    async def list_pins(self) -> tuple[ConversationCachePin, ...]:
        pool = await self._pool_factory()
        async with pool.acquire() as conn:
            rows = await conn.fetch(
                "SELECT * FROM conversation_cache_pins ORDER BY pin_id"
            )
            return tuple([await self._from_row(conn, row) for row in rows])

    async def save_actor_state(
        self, pin_id: str, state: CachePinActorState
    ) -> None:
        pool = await self._pool_factory()
        async with pool.acquire() as conn:
            await conn.execute(
                """INSERT INTO conversation_cache_pin_actor_state(
                     pin_id,actor_id,status,profile_id,last_keepalive_at,
                     next_keepalive_at,call_count,cache_read_input_tokens,last_error
                   ) VALUES($1,$2,$3,$4,$5,$6,$7,$8,$9)
                   ON CONFLICT(pin_id,actor_id) DO UPDATE SET
                     status=EXCLUDED.status,
                     profile_id=EXCLUDED.profile_id,
                     last_keepalive_at=EXCLUDED.last_keepalive_at,
                     next_keepalive_at=EXCLUDED.next_keepalive_at,
                     call_count=EXCLUDED.call_count,
                     cache_read_input_tokens=EXCLUDED.cache_read_input_tokens,
                     last_error=EXCLUDED.last_error,
                     updated_at=NOW()""",
                pin_id,
                state.actor_id,
                state.status,
                state.profile_id,
                state.last_keepalive_at,
                state.next_keepalive_at,
                state.call_count,
                state.cache_read_input_tokens,
                state.last_error,
            )


def _pin_identity(
    *, execution_mode: str, conversation_id: str, bedroom_session_id: str | None
) -> str:
    if execution_mode == "bedroom":
        if not bedroom_session_id:
            raise CachePinError("Bedroom pin requires bedroom_session_id")
        return f"bedroom:{bedroom_session_id}"
    return f"{execution_mode}:{conversation_id}"


def _pin_actors(pin: ConversationCachePin) -> tuple[str, ...]:
    if pin.execution_mode == "group":
        return ("jiao", "laoke")
    if pin.execution_mode == "private":
        actor = PRIVATE_ROOM_ACTORS.get(pin.room_id)
        if actor is None:
            raise CachePinError("private pin requires a canonical typed private room")
        return (actor,)
    if pin.execution_mode == "bedroom" and pin.bedroom_actor_id in {"jiao", "laoke"}:
        return (pin.bedroom_actor_id,)
    raise CachePinError("invalid cache pin execution coordinates")


def _supports_verified_one_hour_cache(profile: Any) -> bool:
    return bool(
        profile.selectable
        and profile.cache_strategy == "anthropic_prefix_anchored_v1"
        and profile.requested_cache_ttl == "1h"
        and "1h" in profile.capabilities.cache_ttls
        and "anthropic_prefix_anchored_v1" in profile.capabilities.cache_strategies
    )


class CachePinService:
    def __init__(
        self,
        *,
        store: ConversationCachePinStore,
        profiles: Any,
        context_builder: Any,
        provider_runner: Any,
        usage_store: Any | None = None,
        now=None,
        interval: timedelta = timedelta(minutes=50),
    ) -> None:
        self.store = store
        self.profiles = profiles
        self.context_builder = context_builder
        self.provider_runner = provider_runner
        self.usage_store = usage_store
        self.now = now or (lambda: datetime.now(timezone.utc))
        self.interval = interval

    async def set_pin(
        self,
        *,
        room_id: str,
        conversation_id: str,
        execution_mode: str,
        enabled: bool,
        bedroom_session_id: str | None = None,
        actor_id: str | None = None,
    ) -> ConversationCachePin:
        if execution_mode not in {"private", "group", "bedroom"}:
            raise CachePinError("invalid cache pin execution_mode")
        if execution_mode == "group" and room_id != GROUP_ROOM_ID:
            raise CachePinError("group pin requires room_group_home")
        pin = ConversationCachePin(
            pin_id=_pin_identity(
                execution_mode=execution_mode,
                conversation_id=conversation_id,
                bedroom_session_id=bedroom_session_id,
            ),
            room_id=room_id,
            conversation_id=conversation_id,
            execution_mode=execution_mode,
            enabled=enabled,
            bedroom_session_id=bedroom_session_id,
            bedroom_actor_id=actor_id,
        )
        actors = _pin_actors(pin)
        saved = await self.store.upsert_pin(pin)
        for actor_id in actors:
            if actor_id not in saved.actors:
                await self.store.save_actor_state(
                    saved.pin_id, CachePinActorState(actor_id=actor_id)
                )
        refreshed = await self.store.get_pin(saved.pin_id)
        if refreshed is None:
            raise CachePinError("cache pin disappeared after save")
        return refreshed

    async def get_pin(self, pin_id: str) -> ConversationCachePin:
        pin = await self.store.get_pin(pin_id)
        if pin is None:
            raise CachePinError("cache pin not found")
        return pin

    async def list_pins(self) -> tuple[ConversationCachePin, ...]:
        return await self.store.list_pins()

    async def end_bedroom(self, bedroom_session_id: str) -> None:
        pin_id = f"bedroom:{bedroom_session_id}"
        pin = await self.store.get_pin(pin_id)
        if pin is not None:
            await self.store.upsert_pin(replace(pin, enabled=False))

    async def run_due_once(self) -> CachePinRunResult:
        calls = 0
        now = self.now()
        for pin in await self.store.list_pins():
            if not pin.enabled:
                continue
            for actor_id in _pin_actors(pin):
                state = pin.actors.get(actor_id, CachePinActorState(actor_id))
                resolved = await self.profiles.resolve(actor_id, pin.room_id)
                profile = resolved.primary
                cache_verified = _supports_verified_one_hour_cache(profile) and (
                    await self.profiles.has_verified_probe(
                        profile.profile_id,
                        profile.revision,
                        "frozen_double_send_cache",
                    )
                )
                if not cache_verified:
                    await self.store.save_actor_state(
                        pin.pin_id,
                        replace(
                            state,
                            status="paused",
                            profile_id=profile.profile_id,
                            next_keepalive_at=now + self.interval,
                            last_error="profile_has_no_verified_1h_cache",
                        ),
                    )
                    continue
                if state.next_keepalive_at is not None and state.next_keepalive_at > now:
                    continue
                try:
                    cache_conversation_id = (
                        f"bedroom:{pin.bedroom_session_id}"
                        if pin.execution_mode == "bedroom"
                        else pin.conversation_id
                    )
                    context: ContextBundle = await self.context_builder.build_cache_keepalive(
                        actor_id=actor_id,
                        room_id=pin.room_id,
                        conversation_id=pin.conversation_id,
                        execution_mode=pin.execution_mode,
                        bedroom_session_id=pin.bedroom_session_id,
                        cache_conversation_id=cache_conversation_id,
                        profile=profile,
                    )
                    namespace = build_cache_namespace(
                        actor_id=actor_id,
                        conversation_id=context.cache_conversation_id or cache_conversation_id,
                        profile_id=profile.profile_id,
                        profile_revision=profile.revision,
                        execution_mode=pin.execution_mode,
                        actor_prompt_version=context.actor_prompt_version,
                        runtime_kernel_version=context.runtime_kernel_version,
                        room_policy_version=context.room_policy_version,
                        tool_schema_hash=context.tool_schema_hash,
                        cache_strategy_version=profile.cache_strategy,
                    )
                    generation_request_id = (
                        f"cache-pin:{pin.pin_id}:{actor_id}:{uuid.uuid4()}"
                    )
                    request = SimpleNamespace(
                        execution_kind="full",
                        generation_request_id=generation_request_id,
                    )
                    usage = ProviderUsage.from_provider_values()
                    observed_cache_support = "unverified"
                    provider_usage_received = False
                    stream = self.provider_runner.run(
                        profile=profile,
                        request=request,
                        context=context,
                        cache_namespace=namespace,
                        max_output_tokens=1,
                    )
                    async for chunk in stream:
                        if chunk.event == "usage" and isinstance(
                            chunk.data.get("usage"), ProviderUsage
                        ):
                            usage = chunk.data["usage"]
                            observed_cache_support = str(
                                chunk.data.get("observed_cache_support", "unverified")
                            )
                            provider_usage_received = bool(
                                chunk.data.get("provider_usage_received", False)
                            )
                    if self.usage_store is not None:
                        await self.usage_store.record(
                            ExecutionReceiptDraft(
                                generation_request_id=generation_request_id,
                                actor_id=actor_id,
                                room_id=pin.room_id,
                                conversation_id=pin.conversation_id,
                                profile_id=profile.profile_id,
                                profile_revision=profile.revision,
                                provider=profile.provider,
                                protocol=profile.protocol,
                                route_id=profile.route_id,
                                model=profile.model,
                                adapter_version=profile.adapter_version,
                                cache_strategy=profile.cache_strategy,
                                requested_cache_ttl=profile.requested_cache_ttl,
                                observed_cache_support=observed_cache_support,
                                fallback_used=False,
                                fallback_from_profile_id=None,
                                usage=usage,
                                status="succeeded",
                                stable_prefix_hash=(
                                    context.stable_prefix_hash
                                    or build_stable_prefix_hash(
                                        static_system=context.static_system,
                                        stable_summary=context.stable_summary,
                                        stable_history=context.stable_history,
                                    )
                                ),
                                prompt_cache_key=None,
                                runtime_kernel_version=context.runtime_kernel_version,
                                persona_version=context.actor_prompt_version,
                                room_policy_version=context.room_policy_version,
                                tool_schema_hash=context.tool_schema_hash,
                                summary_version=context.summary_version or 1,
                                compressed_up_to_event_id=(
                                    context.compressed_up_to_event_id or 0
                                ),
                                provider_usage_received=provider_usage_received,
                                execution_purpose="cache_keepalive",
                            )
                        )
                    calls += 1
                    await self.store.save_actor_state(
                        pin.pin_id,
                        CachePinActorState(
                            actor_id=actor_id,
                            status="active",
                            profile_id=profile.profile_id,
                            last_keepalive_at=now,
                            next_keepalive_at=now + self.interval,
                            call_count=state.call_count + 1,
                            cache_read_input_tokens=usage.cache_read_input_tokens,
                        ),
                    )
                except ProviderRunUnavailable:
                    await self.store.save_actor_state(
                        pin.pin_id,
                        replace(
                            state,
                            status="paused",
                            profile_id=profile.profile_id,
                            next_keepalive_at=now + self.interval,
                            last_error="provider_unavailable",
                        ),
                    )
        return CachePinRunResult(calls=calls)
