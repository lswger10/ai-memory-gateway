from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path, PurePosixPath
from typing import Any


CONTRACT_VERSION = "group-room.v1.1"
MESSAGE_KINDS = {"text", "attachment", "image", "sticker", "voice_message"}
_MEDIA_KEYS = {
    "attachment_id", "name", "media_type", "size", "category", "purpose",
    "source", "derived_text", "semantic_label",
}


class ContractError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class MediaReference:
    value: dict[str, Any]

    @classmethod
    def from_dict(cls, payload: Any) -> "MediaReference":
        if not isinstance(payload, dict) or set(payload) != _MEDIA_KEYS:
            raise ContractError("media reference fields are invalid")
        if not isinstance(payload["attachment_id"], str) or not payload["attachment_id"]:
            raise ContractError("attachment_id is required")
        name = payload["name"]
        if not isinstance(name, str) or PurePosixPath(name).name != name or "\\" in name:
            raise ContractError("media name must be a basename")
        if not isinstance(payload["size"], int) or isinstance(payload["size"], bool) or payload["size"] < 1:
            raise ContractError("media size must be positive")
        source = payload["source"]
        if not isinstance(source, dict) or set(source) != {"type", "path"}:
            raise ContractError("media source fields are invalid")
        path = source.get("path")
        if source.get("type") != "relay" or not isinstance(path, str) or not path.startswith("/uploads/"):
            raise ContractError("media source must be Relay")
        if "/" in path[len("/uploads/"):] or "\\" in path:
            raise ContractError("media source path is unsafe")
        if payload["purpose"] not in {"attachment", "sticker", "voice_message"}:
            raise ContractError("media purpose is invalid")
        if payload["category"] not in {"text", "document", "image", "audio", "video", "file"}:
            raise ContractError("media category is invalid")
        if payload["derived_text"] is not None and not isinstance(payload["derived_text"], str):
            raise ContractError("derived_text is invalid")
        label = payload["semantic_label"]
        if label is not None and (not isinstance(label, str) or not label.strip()):
            raise ContractError("semantic_label is invalid")
        return cls(json.loads(json.dumps(payload, ensure_ascii=False, sort_keys=True)))

    def to_dict(self) -> dict[str, Any]:
        return json.loads(json.dumps(self.value, ensure_ascii=False))


def validate_room_event(payload: Any) -> dict[str, Any]:
    if not isinstance(payload, dict) or payload.get("contract_version") != CONTRACT_VERSION:
        raise ContractError("event contract version mismatch")
    if payload.get("message_kind") not in MESSAGE_KINDS or not isinstance(payload.get("attachments"), list):
        raise ContractError("event media fields are invalid")
    attachments = [MediaReference.from_dict(item).to_dict() for item in payload["attachments"]]
    kind = payload["message_kind"]
    content = payload.get("content")
    if not isinstance(content, str) or (not content and not attachments):
        raise ContractError("event requires text or attachment")
    if kind == "text" and attachments:
        raise ContractError("text event cannot contain attachments")
    if kind == "sticker" and (
        len(attachments) != 1
        or attachments[0]["purpose"] != "sticker"
        or attachments[0]["semantic_label"] is None
    ):
        raise ContractError("sticker event is invalid")
    if kind == "voice_message" and any(
        item["purpose"] != "voice_message" or item["category"] != "audio"
        for item in attachments
    ):
        raise ContractError("voice message event is invalid")
    return json.loads(json.dumps(payload, ensure_ascii=False))


def verify_contract_bundle(root: Path) -> dict[str, str]:
    entries = {}
    for line in (root / "SHA256SUMS").read_text(encoding="utf-8").splitlines():
        digest, relative = line.split("  ", 1)
        entries[relative] = digest
    actual = {"group-room.schema.json", *(f"fixtures/{path.name}" for path in (root / "fixtures").glob("*.json"))}
    if set(entries) != actual:
        raise ContractError("contract manifest coverage mismatch")
    for relative, digest in entries.items():
        if hashlib.sha256((root / relative).read_bytes()).hexdigest() != digest:
            raise ContractError(f"contract hash mismatch: {relative}")
    return entries
