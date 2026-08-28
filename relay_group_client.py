"""Read-only Relay facts client for Gateway-owned Group Context Packs."""

from __future__ import annotations

from typing import Any

import httpx

from group_contracts import (
    CONTRACT_VERSION,
    ContextFactsRequest,
    ContextPackRequest,
    ClosedBurstExtractionRequest,
    MemoryCandidateRequest,
    PublicContextFacts,
)


class RelayGroupError(RuntimeError):
    def __init__(self, status_code: int, code: str):
        super().__init__(code)
        self.status_code = status_code
        self.code = code


class RelayFactsMismatch(RelayGroupError):
    def __init__(self):
        super().__init__(409, "stale_fence")


class RelayGroupClient:
    def __init__(
        self,
        base_url: str,
        service_key: str,
        *,
        http_client: Any | None = None,
        timeout_seconds: float = 10.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.service_key = service_key
        self.http_client = http_client
        self.timeout_seconds = timeout_seconds

    async def fetch_context_facts(
        self, request: ContextPackRequest
    ) -> PublicContextFacts:
        pack = request.to_dict()
        facts_request = ContextFactsRequest.from_dict(
            {
                "contract_version": CONTRACT_VERSION,
                "room_id": pack["room_id"],
                "conversation_id": pack["conversation_id"],
                "current_event_id": pack["current_event_id"],
                "burst_id": pack["burst_id"],
                "fence_epoch": pack["fence_epoch"],
                "recent_limit": 20,
                "require_closed": False,
            }
        )
        facts = await self._post_context_facts(facts_request)
        factual = facts.to_dict()
        if factual["fence_status"] != "active":
            raise RelayFactsMismatch()
        coordinates = (
            "room_id", "conversation_id", "current_event_id", "burst_id", "fence_epoch"
        )
        if any(factual[field] != pack[field] for field in coordinates):
            raise RelayFactsMismatch()
        current_burst_events = [factual["trigger_event"]] + factual[
            "accepted_burst_public_events"
        ]
        for event in current_burst_events:
            if (
                event["room_id"] != pack["room_id"]
                or event["conversation_id"] != pack["conversation_id"]
                or event["burst_id"] != pack["burst_id"]
            ):
                raise RelayFactsMismatch()
        for event in factual["recent_public_events"]:
            if (
                event["room_id"] != pack["room_id"]
                or event["conversation_id"] != pack["conversation_id"]
            ):
                raise RelayFactsMismatch()
        return facts

    async def _post_context_facts(
        self, facts_request: ContextFactsRequest
    ) -> PublicContextFacts:
        headers = {
            "Authorization": f"Bearer {self.service_key}",
            "X-Group-Contract-Version": CONTRACT_VERSION,
        }
        url = f"{self.base_url}/internal/group/context-facts"
        try:
            if self.http_client is not None:
                response = await self.http_client.post(
                    url, headers=headers, json=facts_request.to_dict()
                )
            else:
                async with httpx.AsyncClient(timeout=self.timeout_seconds) as client:
                    response = await client.post(
                        url, headers=headers, json=facts_request.to_dict()
                    )
        except httpx.HTTPError as exc:
            raise RelayGroupError(503, "dependency_unavailable") from exc

        payload = response.json()
        if response.status_code != 200:
            error = payload.get("error", {}) if isinstance(payload, dict) else {}
            raise RelayGroupError(
                response.status_code,
                str(error.get("code") or "dependency_unavailable"),
            )
        return PublicContextFacts.from_dict(payload)

    async def verify_candidate_source(
        self, request: MemoryCandidateRequest, actor_id: str
    ) -> dict[str, Any]:
        candidate = request.to_dict()
        fence = candidate["fence"]
        facts_request = ContextFactsRequest.from_dict(
            {
                "contract_version": CONTRACT_VERSION,
                "room_id": fence["room_id"],
                "conversation_id": fence["conversation_id"],
                "current_event_id": candidate["source_event_id"],
                "burst_id": fence["burst_id"],
                "fence_epoch": fence["fence_epoch"],
                "recent_limit": 20,
                "require_closed": False,
            }
        )
        factual = (await self._post_context_facts(facts_request)).to_dict()
        if factual["fence_status"] != "active":
            raise RelayFactsMismatch()
        exact = {
            "room_id": fence["room_id"],
            "conversation_id": fence["conversation_id"],
            "current_event_id": candidate["source_event_id"],
            "burst_id": fence["burst_id"],
            "fence_epoch": fence["fence_epoch"],
        }
        if any(factual[key] != value for key, value in exact.items()):
            raise RelayFactsMismatch()
        if factual["trigger_event"]["event_id"] != fence["trigger_event_id"]:
            raise RelayFactsMismatch()
        events = factual["accepted_burst_public_events"]
        source = next(
            (event for event in events if event["event_id"] == candidate["source_event_id"]),
            None,
        )
        if (
            source is None
            or source["actor_id"] != actor_id
            or source["role"] != "agent"
            or source["event_type"] != "agent_final"
            or not source.get("provenance")
            or source["provenance"].get("generation_request_id")
            != candidate["generation_request_id"]
        ):
            raise RelayFactsMismatch()
        return factual

    async def fetch_closed_burst_facts(
        self, request: ClosedBurstExtractionRequest
    ) -> dict[str, Any]:
        ref = request.to_dict()["closed_fence"]
        facts_request = ContextFactsRequest.from_dict(
            {
                "contract_version": CONTRACT_VERSION,
                "room_id": ref["room_id"],
                "conversation_id": ref["conversation_id"],
                "current_event_id": ref["trigger_event_id"],
                "burst_id": ref["burst_id"],
                "fence_epoch": ref["fence_epoch"],
                "recent_limit": 20,
                "require_closed": True,
            }
        )
        factual = (await self._post_context_facts(facts_request)).to_dict()
        if factual["fence_status"] != "closed":
            raise RelayFactsMismatch()
        exact = {
            "room_id": ref["room_id"],
            "conversation_id": ref["conversation_id"],
            "burst_id": ref["burst_id"],
            "fence_epoch": ref["fence_epoch"],
        }
        if any(factual[key] != value for key, value in exact.items()):
            raise RelayFactsMismatch()
        if factual["trigger_event"]["event_id"] != ref["trigger_event_id"]:
            raise RelayFactsMismatch()
        return factual
