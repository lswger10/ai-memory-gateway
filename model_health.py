from __future__ import annotations

import asyncio
import logging
import uuid

from cache_probe import GatewayCacheProbeService
from cache_strategies import PromptSegment
from model_execution import ContextBundle, ProviderRunUnavailable
from model_profile_store import ProfileStoreError
from model_usage_store import UsageRecordingError


class ModelHealthService:
    """One synthetic generation per enabled profile per day; never chat facts."""

    def __init__(self, *, profiles, provider_runner, usage_store):
        self.profiles = profiles
        self.probe = GatewayCacheProbeService(profiles=profiles, provider_runner=provider_runner, usage_store=usage_store)
        self._lock = asyncio.Lock()

    async def check(self, profile_id: str, *, profile_revision: int) -> dict:
        async with self._lock:
            profile = await self.profiles.get_profile(profile_id)
            if profile.revision != profile_revision:
                raise ProfileStoreError("Profile revision changed before health check")
            if not profile.enabled:
                raise ValueError("enable the profile before testing")
            request = self.probe._request(actor_id="jiao", room_id="room_weiwei_jiao",
                conversation_id="model-health:" + str(uuid.uuid4()), profile_id=profile_id)
            context = ContextBundle(static_system=("Reply with OK.", "Connectivity test.", "No user data."),
                stable_summary="", stable_history=(),
                dynamic_tail=(PromptSegment("current_event", "OK?"),),
                actor_prompt_version="model-health.v1", runtime_kernel_version="model-health.v1",
                room_policy_version="model-health.v1", tool_schema_hash="tools.none.v1")
            status, reason, http_status = "verified", None, None
            # Gemini counts thinking against the output limit before producing reply text.
            max_output_tokens = 512 if profile.protocol == "gemini_native" else 32
            try:
                async with asyncio.timeout(150):
                    await self.probe._once(profile=profile, request=request, context=context,
                        cache_namespace=request.conversation_id, execution_purpose="model_health", max_output_tokens=max_output_tokens)
            except UsageRecordingError:
                raise
            except ProviderRunUnavailable as exc:
                status, reason = "failed", exc.reason_code
                http_status = exc.provider_status
            except Exception as exc:
                status, reason = "failed", "provider_transport_failed"
                logging.getLogger(__name__).warning("Model health check failed profile=%s cause=%s",
                    profile_id, type(exc).__name__)
            details = {"reason_code": reason, "provider_status": http_status}
            await self.profiles.record_probe_result(profile_id=profile_id, profile_revision=profile.revision,
                probe_kind="model_health", status=status, observed_capabilities=details)
            try:
                if status == "verified":
                    await self.profiles.set_test_status(profile_id, "passed", expected_revision=profile.revision)
                elif http_status in {400, 401, 402, 403, 404, 410}:
                    await self.profiles.set_test_status(profile_id, "failed", expected_revision=profile.revision)
            except ProfileStoreError:
                # A concurrent edit must remain unverified; old results are revision-scoped.
                pass
            return {"profile_id": profile_id, "profile_revision": profile.revision, "status": status, **details}

    async def run_due_once(self):
        for profile in await self.profiles.list_profiles():
            if not profile.enabled or not await self.profiles.claim_health_check(profile.profile_id, profile.revision):
                continue
            try:
                await self.check(profile.profile_id, profile_revision=profile.revision)
            except (ProfileStoreError, ValueError):
                continue
