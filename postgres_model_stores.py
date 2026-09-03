from __future__ import annotations

import json
import uuid
from dataclasses import replace

from model_execution_contracts import ProviderUsage
from model_profile_store import ProfileStoreError, ResolvedProfiles, RoomOverride, StoredBinding
from model_profiles import ModelProfile
from model_usage_store import ExecutionReceipt, ExecutionReceiptDraft, UsageStoreConflict


def _json_object(value):
    if isinstance(value, str):
        value = json.loads(value)
    if not isinstance(value, dict):
        raise ProfileStoreError("stored Profile JSON must be an object")
    return value


def _json_array(value):
    if isinstance(value, str):
        value = json.loads(value)
    if not isinstance(value, list):
        raise ProfileStoreError("stored Profile list must be an array")
    return value


class PostgresModelProfileStore:
    def __init__(self, pool_factory) -> None:
        self._pool_factory = pool_factory

    async def put_profile(self, profile: ModelProfile) -> ModelProfile:
        pool = await self._pool_factory()
        payload = json.dumps(profile.to_dict(), ensure_ascii=False)
        async with pool.acquire() as conn:
            row = await conn.fetchrow("SELECT revision FROM model_profiles WHERE profile_id=$1", profile.profile_id)
            if row is not None and int(row["revision"]) > profile.revision:
                raise ProfileStoreError("Profile revision cannot move backwards")
            await conn.execute(
                """INSERT INTO model_profiles(profile_id,profile_json,enabled,test_status,revision)
                   VALUES($1,$2::jsonb,$3,$4,$5)
                   ON CONFLICT(profile_id) DO UPDATE SET
                     profile_json=EXCLUDED.profile_json, enabled=EXCLUDED.enabled,
                     test_status=EXCLUDED.test_status, revision=EXCLUDED.revision,
                     updated_at=NOW()""",
                profile.profile_id, payload, profile.enabled, profile.test_status, profile.revision,
            )
        return profile

    async def list_profiles(self) -> tuple[ModelProfile, ...]:
        pool = await self._pool_factory()
        async with pool.acquire() as conn:
            rows = await conn.fetch("SELECT profile_json FROM model_profiles ORDER BY profile_id")
        return tuple(ModelProfile.from_dict(_json_object(row["profile_json"])) for row in rows)

    async def get_profile(self, profile_id: str) -> ModelProfile:
        pool = await self._pool_factory()
        async with pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT profile_json FROM model_profiles WHERE profile_id=$1",
                profile_id,
            )
        if row is None:
            raise ProfileStoreError(f"unknown Profile: {profile_id}")
        return ModelProfile.from_dict(_json_object(row["profile_json"]))

    async def set_test_status(self, profile_id: str, status: str) -> ModelProfile:
        profile = await self.get_profile(profile_id)
        updated = replace(profile, test_status=status)
        pool = await self._pool_factory()
        async with pool.acquire() as conn:
            result = await conn.execute(
                """UPDATE model_profiles
                   SET profile_json=$2::jsonb,test_status=$3,updated_at=NOW()
                   WHERE profile_id=$1 AND revision=$4""",
                profile_id,
                json.dumps(updated.to_dict(), ensure_ascii=False),
                status,
                profile.revision,
            )
        if result.endswith("0"):
            raise ProfileStoreError("profile changed during probe")
        return updated

    async def record_probe_result(
        self,
        *,
        profile_id: str,
        profile_revision: int,
        probe_kind: str,
        status: str,
        observed_capabilities: dict,
        sanitized_detail: str | None = None,
    ) -> None:
        pool = await self._pool_factory()
        async with pool.acquire() as conn:
            await conn.execute(
                """INSERT INTO model_profile_probe_results(
                     profile_id,profile_revision,probe_kind,status,
                     observed_capabilities,sanitized_detail
                   ) VALUES($1,$2,$3,$4,$5::jsonb,$6)
                   ON CONFLICT(profile_id,profile_revision,probe_kind) DO UPDATE SET
                     status=EXCLUDED.status,
                     observed_capabilities=EXCLUDED.observed_capabilities,
                     sanitized_detail=EXCLUDED.sanitized_detail,
                     tested_at=NOW()""",
                profile_id,
                profile_revision,
                probe_kind,
                status,
                json.dumps(observed_capabilities),
                sanitized_detail,
            )

    async def _profile(self, conn, profile_id: str) -> ModelProfile:
        row = await conn.fetchrow("SELECT profile_json FROM model_profiles WHERE profile_id=$1", profile_id)
        if row is None:
            raise ProfileStoreError(f"unknown Profile: {profile_id}")
        profile = ModelProfile.from_dict(_json_object(row["profile_json"]))
        if not profile.selectable:
            raise ProfileStoreError(f"Profile is not selectable: {profile_id}")
        return profile

    async def set_actor_default(self, actor_id: str, profile_id: str, *, expected_revision: int | None = None) -> StoredBinding:
        pool = await self._pool_factory()
        async with pool.acquire() as conn:
            async with conn.transaction():
                await self._profile(conn, profile_id)
                row = await conn.fetchrow("SELECT * FROM model_actor_bindings WHERE actor_id=$1 FOR UPDATE", actor_id)
                actual = int(row["revision"]) if row else None
                if expected_revision is not None and actual != expected_revision:
                    raise ProfileStoreError("binding revision conflict")
                revision = 1 if row is None else actual + 1
                fallbacks = _json_array(row["approved_fallback_profile_ids"]) if row else []
                await conn.execute(
                    """INSERT INTO model_actor_bindings(actor_id,default_profile_id,approved_fallback_profile_ids,revision)
                       VALUES($1,$2,$3::jsonb,$4) ON CONFLICT(actor_id) DO UPDATE SET
                       default_profile_id=EXCLUDED.default_profile_id, revision=EXCLUDED.revision, updated_at=NOW()""",
                    actor_id, profile_id, json.dumps(fallbacks), revision,
                )
        return StoredBinding(actor_id, profile_id, tuple(fallbacks), revision)

    async def set_approved_fallbacks(self, actor_id: str, profile_ids: tuple[str, ...], *, expected_revision: int | None = None) -> StoredBinding:
        if len(set(profile_ids)) != len(profile_ids) or "*" in profile_ids:
            raise ProfileStoreError("fallbacks must be an explicit unique order")
        pool = await self._pool_factory()
        async with pool.acquire() as conn:
            async with conn.transaction():
                row = await conn.fetchrow("SELECT * FROM model_actor_bindings WHERE actor_id=$1 FOR UPDATE", actor_id)
                if row is None:
                    raise ProfileStoreError("actor default must be configured first")
                actual = int(row["revision"])
                if expected_revision is not None and actual != expected_revision:
                    raise ProfileStoreError("binding revision conflict")
                if row["default_profile_id"] in profile_ids:
                    raise ProfileStoreError("default Profile cannot repeat as fallback")
                for profile_id in profile_ids:
                    await self._profile(conn, profile_id)
                revision = actual + 1
                await conn.execute(
                    "UPDATE model_actor_bindings SET approved_fallback_profile_ids=$2::jsonb,revision=$3,updated_at=NOW() WHERE actor_id=$1",
                    actor_id, json.dumps(list(profile_ids)), revision,
                )
        return StoredBinding(actor_id, row["default_profile_id"], tuple(profile_ids), revision)

    async def set_room_override(self, room_id: str, actor_id: str, profile_id: str, *, expected_revision: int | None = None) -> RoomOverride:
        pool = await self._pool_factory()
        async with pool.acquire() as conn:
            async with conn.transaction():
                await self._profile(conn, profile_id)
                row = await conn.fetchrow("SELECT revision FROM model_room_overrides WHERE room_id=$1 AND actor_id=$2 FOR UPDATE", room_id, actor_id)
                actual = int(row["revision"]) if row else None
                if expected_revision is not None and actual != expected_revision:
                    raise ProfileStoreError("room override revision conflict")
                revision = 1 if row is None else actual + 1
                await conn.execute(
                    """INSERT INTO model_room_overrides(room_id,actor_id,profile_id,revision)
                       VALUES($1,$2,$3,$4) ON CONFLICT(room_id,actor_id) DO UPDATE SET
                       profile_id=EXCLUDED.profile_id,revision=EXCLUDED.revision,updated_at=NOW()""",
                    room_id, actor_id, profile_id, revision,
                )
        return RoomOverride(room_id, actor_id, profile_id, revision)

    async def resolve(self, actor_id: str, room_id: str) -> ResolvedProfiles:
        pool = await self._pool_factory()
        async with pool.acquire() as conn:
            binding = await conn.fetchrow("SELECT * FROM model_actor_bindings WHERE actor_id=$1", actor_id)
            if binding is None:
                raise ProfileStoreError(f"actor has no Model Profile binding: {actor_id}")
            override = await conn.fetchrow("SELECT * FROM model_room_overrides WHERE room_id=$1 AND actor_id=$2", room_id, actor_id)
            primary_id = override["profile_id"] if override else binding["default_profile_id"]
            primary = await self._profile(conn, primary_id)
            fallback_ids = tuple(
                item
                for item in _json_array(binding["approved_fallback_profile_ids"])
                if item != primary_id
            )
            fallbacks = tuple([await self._profile(conn, item) for item in fallback_ids])
            revision = int(override["revision"] if override else binding["revision"])
        return ResolvedProfiles(actor_id, room_id, primary, fallbacks, "room_override" if override else "actor_default", revision)


class PostgresModelUsageStore:
    def __init__(self, pool_factory) -> None:
        self._pool_factory = pool_factory

    async def record(self, draft: ExecutionReceiptDraft) -> ExecutionReceipt:
        pool = await self._pool_factory()
        receipt_id = str(uuid.uuid5(uuid.NAMESPACE_URL, draft.generation_request_id))
        values = (
            receipt_id, draft.generation_request_id, draft.actor_id, draft.room_id,
            draft.conversation_id, draft.profile_id, draft.profile_revision, draft.provider,
            draft.protocol, draft.route_id, draft.model, draft.adapter_version,
            draft.cache_strategy, draft.requested_cache_ttl, draft.observed_cache_support,
            draft.fallback_used, draft.fallback_from_profile_id, draft.usage.input_tokens,
            draft.usage.output_tokens, draft.usage.cache_creation_input_tokens,
            draft.usage.cache_read_input_tokens, draft.usage.cached_tokens, draft.status,
            draft.stable_prefix_hash, draft.prompt_cache_key,
            draft.runtime_kernel_version, draft.persona_version,
            draft.room_policy_version, draft.tool_schema_hash,
            draft.summary_version, draft.compressed_up_to_event_id,
            draft.provider_usage_received, draft.execution_purpose,
        )
        async with pool.acquire() as conn:
            existing = await conn.fetchrow("SELECT * FROM model_execution_receipts WHERE generation_request_id=$1", draft.generation_request_id)
            if existing is None:
                await conn.execute(
                    """INSERT INTO model_execution_receipts(
                    receipt_id,generation_request_id,actor_id,room_id,conversation_id,
                    profile_id,profile_revision,provider,protocol,route_id,model,
                    adapter_version,cache_strategy,requested_cache_ttl,
                    observed_cache_support,fallback_used,fallback_from_profile_id,
                    input_tokens,output_tokens,cache_creation_input_tokens,
                    cache_read_input_tokens,cached_tokens,status,stable_prefix_hash,
                    prompt_cache_key,runtime_kernel_version,persona_version,
                    room_policy_version,tool_schema_hash,summary_version,
                    compressed_up_to_event_id,provider_usage_received,
                    execution_purpose,created_at
                    ) VALUES(
                    $1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12,$13,$14,$15,$16,$17,
                    $18,$19,$20,$21,$22,$23,$24,$25,$26,$27,$28,$29,$30,$31,$32,$33,NOW())""",
                    *values,
                )
            else:
                if existing["receipt_id"] != receipt_id:
                    raise UsageStoreConflict("generation request receipt conflict")
        return ExecutionReceipt(
            receipt_id, draft.generation_request_id, draft.actor_id, draft.room_id,
            draft.conversation_id, draft.profile_id, draft.profile_revision, draft.provider,
            draft.protocol, draft.route_id, draft.model, draft.adapter_version,
            draft.cache_strategy, draft.requested_cache_ttl, draft.observed_cache_support,
            draft.fallback_used, draft.fallback_from_profile_id, draft.usage, draft.status,
            draft.stable_prefix_hash, draft.prompt_cache_key,
            draft.runtime_kernel_version, draft.persona_version,
            draft.room_policy_version, draft.tool_schema_hash,
            draft.summary_version, draft.compressed_up_to_event_id,
            draft.provider_usage_received, draft.execution_purpose,
        )

    async def list_receipts(self, *, limit: int = 200) -> tuple[ExecutionReceipt, ...]:
        pool = await self._pool_factory()
        async with pool.acquire() as conn:
            rows = await conn.fetch("SELECT * FROM model_execution_receipts ORDER BY created_at DESC LIMIT $1", limit)
        return tuple(
            ExecutionReceipt(
                row["receipt_id"], row["generation_request_id"], row["actor_id"], row["room_id"],
                row["conversation_id"], row["profile_id"], row["profile_revision"], row["provider"],
                row["protocol"], row["route_id"], row["model"], row["adapter_version"],
                row["cache_strategy"], row["requested_cache_ttl"], row["observed_cache_support"],
                row["fallback_used"], row["fallback_from_profile_id"],
                ProviderUsage.from_provider_values(
                    input_tokens=row["input_tokens"], output_tokens=row["output_tokens"],
                    cache_creation_input_tokens=row["cache_creation_input_tokens"],
                    cache_read_input_tokens=row["cache_read_input_tokens"], cached_tokens=row["cached_tokens"],
                ), row["status"], row["stable_prefix_hash"],
                row["prompt_cache_key"], row["runtime_kernel_version"],
                row["persona_version"], row["room_policy_version"],
                row["tool_schema_hash"], row["summary_version"],
                row["compressed_up_to_event_id"], row["provider_usage_received"],
                row["execution_purpose"],
            ) for row in rows
        )
