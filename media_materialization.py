from __future__ import annotations

import base64
from dataclasses import dataclass
from typing import Any
from urllib.parse import quote

import httpx

from group_contracts_v11 import ContractError, MediaReference
from model_profiles import ModelProfile


class MediaMaterializationError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class PreparedMedia:
    reference: dict[str, Any]
    data: bytes | None


class RelayMediaReader:
    def __init__(
        self,
        base_url: str,
        service_key: str,
        *,
        http_client: httpx.AsyncClient | None = None,
        timeout_seconds: float = 20.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.service_key = service_key
        self.http_client = http_client
        self.timeout_seconds = timeout_seconds

    async def fetch(self, reference: dict[str, Any]) -> PreparedMedia:
        try:
            value = MediaReference.from_dict(reference).to_dict()
        except ContractError as exc:
            raise MediaMaterializationError(str(exc)) from exc
        attachment_id = value["attachment_id"]
        url = f"{self.base_url}/internal/group/media/{quote(attachment_id, safe='')}"
        headers = {
            "Authorization": f"Bearer {self.service_key}",
            "X-Group-Contract-Version": "group-room.v1.1",
        }
        try:
            if self.http_client is not None:
                response = await self.http_client.get(url, headers=headers)
            else:
                async with httpx.AsyncClient(timeout=self.timeout_seconds) as client:
                    response = await client.get(url, headers=headers)
        except httpx.HTTPError as exc:
            raise MediaMaterializationError("Relay media read failed") from exc
        if response.status_code != 200:
            raise MediaMaterializationError("Relay media was not authorized or available")
        data = response.content
        content_type = response.headers.get("content-type", "").split(";", 1)[0].lower()
        if len(data) != value["size"] or content_type != value["media_type"]:
            raise MediaMaterializationError("Relay media metadata did not match bytes")
        if content_type == "image/png" and not data.startswith(b"\x89PNG\r\n\x1a\n"):
            raise MediaMaterializationError("Relay PNG signature is invalid")
        if content_type in {"image/jpeg", "image/jpg"} and not data.startswith(b"\xff\xd8\xff"):
            raise MediaMaterializationError("Relay JPEG signature is invalid")
        if content_type == "image/gif" and not data.startswith((b"GIF87a", b"GIF89a")):
            raise MediaMaterializationError("Relay GIF signature is invalid")
        if content_type == "application/pdf" and not data.startswith(b"%PDF-"):
            raise MediaMaterializationError("Relay PDF signature is invalid")
        return PreparedMedia(value, data)


async def prepare_media_for_profile(
    profile: ModelProfile,
    references: tuple[dict[str, Any], ...],
    reader: RelayMediaReader,
) -> tuple[PreparedMedia, ...]:
    result = []
    for reference in references:
        category = reference.get("category")
        modality = "document" if category in {"text", "document", "file"} else category
        raw_supported = modality in profile.capabilities.input_modalities and modality in {
            "image", "document"
        }
        result.append(
            await reader.fetch(reference)
            if raw_supported
            else PreparedMedia(reference, None)
        )
    return tuple(result)


def _fallback_text(reference: dict[str, Any]) -> str:
    category = "sticker" if reference.get("purpose") == "sticker" else reference["category"]
    name = reference["name"]
    detail = reference.get("derived_text") or reference.get("semantic_label")
    suffix = (
        f" {detail}"
        if detail
        else " (raw content was not supplied to this text-only Model Profile)"
    )
    return f"[{category}: {name}]{suffix}"


def render_media_tail(
    profile: ModelProfile,
    prepared: tuple[PreparedMedia, ...],
) -> tuple[dict[str, Any], ...]:
    parts = []
    for item in prepared:
        reference = item.reference
        category = reference["category"]
        modality = "document" if category in {"text", "document", "file"} else category
        semantic_emitted = False
        if reference.get("purpose") == "sticker" and reference.get("semantic_label"):
            parts.append({"kind": "text", "text": _fallback_text(reference)})
            semantic_emitted = True
        if item.data is not None and modality in profile.capabilities.input_modalities:
            if modality == "image":
                parts.append(
                    {
                        "kind": "image",
                        "media_type": reference["media_type"],
                        "data": base64.b64encode(item.data).decode("ascii"),
                    }
                )
                continue
            if modality == "document" and reference["media_type"] == "application/pdf":
                parts.append(
                    {
                        "kind": "document",
                        "name": reference["name"],
                        "media_type": reference["media_type"],
                        "data": base64.b64encode(item.data).decode("ascii"),
                    }
                )
                continue
        if not semantic_emitted:
            parts.append({"kind": "text", "text": _fallback_text(reference)})
    return tuple(parts)
