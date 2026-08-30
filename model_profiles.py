from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Mapping


class ProfileContractError(ValueError):
    pass


_PROFILE_FIELDS = {
    "profile_id",
    "display_name",
    "enabled",
    "test_status",
    "provider",
    "protocol",
    "base_url",
    "route_id",
    "model",
    "adapter_version",
    "credential_ref",
    "headers",
    "capabilities",
    "cache_strategy",
    "requested_cache_ttl",
    "revision",
}
_CAPABILITY_FIELDS = {
    "streaming",
    "structured_output",
    "tools",
    "reasoning_controls",
    "cache_strategies",
    "cache_ttls",
    "usage_fields",
}
_TEST_STATUSES = {"unverified", "passed", "failed", "unsupported"}
_CACHE_STRATEGIES = {
    "anthropic_prefix_anchored_v1",
    "openai_stable_prefix_v1",
    "no_prompt_cache_v1",
}


def _required_string(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ProfileContractError(f"{field} must be a non-empty string")
    return value.strip()


def _string_tuple(value: Any, field: str) -> tuple[str, ...]:
    if not isinstance(value, (list, tuple)):
        raise ProfileContractError(f"{field} must be a list")
    result = tuple(_required_string(item, field) for item in value)
    if len(set(result)) != len(result):
        raise ProfileContractError(f"{field} must not contain duplicates")
    return result


def _flag(name: str) -> bool:
    return os.getenv(name, "false").strip().lower() in {"1", "true", "yes", "on"}


def resolve_feature_flags() -> dict[str, bool]:
    return {
        "model_execution": _flag("MODEL_EXECUTION_ENABLED"),
        "model_profile_management": _flag("MODEL_PROFILE_MANAGEMENT_ENABLED"),
        "model_profile_pwa": _flag("MODEL_PROFILE_PWA_ENABLED"),
    }


@dataclass(frozen=True, slots=True)
class ProfileCapabilities:
    streaming: bool
    structured_output: bool
    tools: bool
    reasoning_controls: bool
    cache_strategies: tuple[str, ...]
    cache_ttls: tuple[str, ...]
    usage_fields: tuple[str, ...]

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ProfileCapabilities":
        unknown = set(payload) - _CAPABILITY_FIELDS
        if unknown:
            raise ProfileContractError(
                f"unknown capability fields: {', '.join(sorted(unknown))}"
            )
        booleans: dict[str, bool] = {}
        for field in (
            "streaming",
            "structured_output",
            "tools",
            "reasoning_controls",
        ):
            value = payload.get(field)
            if not isinstance(value, bool):
                raise ProfileContractError(f"capabilities.{field} must be boolean")
            booleans[field] = value
        cache_strategies = _string_tuple(
            payload.get("cache_strategies"), "capabilities.cache_strategies"
        )
        if any(value not in _CACHE_STRATEGIES for value in cache_strategies):
            raise ProfileContractError("unsupported cache strategy capability")
        cache_ttls = _string_tuple(
            payload.get("cache_ttls"), "capabilities.cache_ttls"
        )
        if any(value not in {"5m", "1h"} for value in cache_ttls):
            raise ProfileContractError("cache TTL capability must be 5m or 1h")
        return cls(
            **booleans,
            cache_strategies=cache_strategies,
            cache_ttls=cache_ttls,
            usage_fields=_string_tuple(
                payload.get("usage_fields"), "capabilities.usage_fields"
            ),
        )


@dataclass(frozen=True, slots=True)
class ModelProfile:
    profile_id: str
    display_name: str
    enabled: bool
    test_status: str
    provider: str
    protocol: str
    base_url: str
    route_id: str
    model: str
    adapter_version: str
    credential_ref: str
    headers: tuple[tuple[str, str], ...]
    capabilities: ProfileCapabilities
    cache_strategy: str
    requested_cache_ttl: str | None
    revision: int

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ModelProfile":
        unknown = set(payload) - _PROFILE_FIELDS
        if unknown:
            raise ProfileContractError(
                f"unknown Model Profile fields: {', '.join(sorted(unknown))}"
            )
        enabled = payload.get("enabled")
        if not isinstance(enabled, bool):
            raise ProfileContractError("enabled must be boolean")
        test_status = _required_string(payload.get("test_status"), "test_status")
        if test_status not in _TEST_STATUSES:
            raise ProfileContractError("invalid test_status")
        revision = payload.get("revision")
        if isinstance(revision, bool) or not isinstance(revision, int) or revision < 1:
            raise ProfileContractError("revision must be a positive integer")
        capabilities_value = payload.get("capabilities")
        if not isinstance(capabilities_value, Mapping):
            raise ProfileContractError("capabilities must be an object")
        capabilities = ProfileCapabilities.from_dict(capabilities_value)
        strategy = _required_string(payload.get("cache_strategy"), "cache_strategy")
        if strategy not in _CACHE_STRATEGIES:
            raise ProfileContractError("unsupported cache_strategy")
        if strategy not in capabilities.cache_strategies:
            raise ProfileContractError("cache_strategy was not observed for this route")
        ttl = payload.get("requested_cache_ttl")
        if ttl is not None:
            ttl = _required_string(ttl, "requested_cache_ttl")
            if ttl not in capabilities.cache_ttls:
                raise ProfileContractError(
                    "requested cache TTL was not observed for this route"
                )
        headers_value = payload.get("headers", {})
        if not isinstance(headers_value, Mapping):
            raise ProfileContractError("headers must be an object")
        headers: list[tuple[str, str]] = []
        for name, value in headers_value.items():
            headers.append(
                (
                    _required_string(name, "headers name"),
                    _required_string(value, f"headers.{name}"),
                )
            )
        return cls(
            profile_id=_required_string(payload.get("profile_id"), "profile_id"),
            display_name=_required_string(payload.get("display_name"), "display_name"),
            enabled=enabled,
            test_status=test_status,
            provider=_required_string(payload.get("provider"), "provider"),
            protocol=_required_string(payload.get("protocol"), "protocol"),
            base_url=_required_string(payload.get("base_url"), "base_url"),
            route_id=_required_string(payload.get("route_id"), "route_id"),
            model=_required_string(payload.get("model"), "model"),
            adapter_version=_required_string(
                payload.get("adapter_version"), "adapter_version"
            ),
            credential_ref=_required_string(
                payload.get("credential_ref"), "credential_ref"
            ),
            headers=tuple(sorted(headers, key=lambda item: item[0].lower())),
            capabilities=capabilities,
            cache_strategy=strategy,
            requested_cache_ttl=ttl,
            revision=revision,
        )

    @property
    def selectable(self) -> bool:
        return self.enabled and self.test_status == "passed"


@dataclass(frozen=True, slots=True)
class ModelBinding:
    actor_id: str
    default_profile_id: str
    approved_fallback_profile_ids: tuple[str, ...]
    revision: int = 1

    @classmethod
    def create(
        cls,
        *,
        actor_id: str,
        default_profile: ModelProfile,
        approved_fallback_profile_ids: tuple[str, ...] = (),
        revision: int = 1,
    ) -> "ModelBinding":
        actor = _required_string(actor_id, "actor_id")
        if actor == "weiwei":
            raise ProfileContractError("weiwei cannot have a provider binding")
        if not default_profile.selectable:
            raise ProfileContractError("default Profile must be enabled and tested")
        fallbacks = tuple(approved_fallback_profile_ids)
        if any(not isinstance(item, str) or not item.strip() for item in fallbacks):
            raise ProfileContractError("fallback Profile IDs must be explicit strings")
        if "*" in fallbacks:
            raise ProfileContractError("fallbacks must be an explicit ordered list")
        if len(set(fallbacks)) != len(fallbacks):
            raise ProfileContractError("fallback order must not contain duplicates")
        if default_profile.profile_id in fallbacks:
            raise ProfileContractError("default Profile cannot repeat in fallback order")
        if isinstance(revision, bool) or not isinstance(revision, int) or revision < 1:
            raise ProfileContractError("binding revision must be positive")
        return cls(
            actor_id=actor,
            default_profile_id=default_profile.profile_id,
            approved_fallback_profile_ids=fallbacks,
            revision=revision,
        )
