"""Immutable Cold Archive primitives with conversational ACL before scanning."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Any, Mapping

from memory_policy import RetrievalPolicy


ANNOTATION_TYPES = frozenset(
    {"correction", "identity_mapping", "timestamp_fix", "redaction", "note"}
)


class ImmutableArchiveError(RuntimeError):
    pass


class InvalidArchiveAnnotation(ValueError):
    pass


class ArchiveIdentityConflict(ValueError):
    pass


@dataclass(frozen=True)
class ArchiveRawInput:
    source_system: str
    source_conversation_id: str
    source_message_id: str
    raw_actor_label: str
    raw_payload: Mapping[str, Any]
    raw_content: str
    raw_timestamp: str | None
    mapped_actor: str | None
    mapped_scope: str | None
    confidential: bool
    manifest_hash: str | None


@dataclass(frozen=True)
class ArchiveRawRecord(ArchiveRawInput):
    id: int
    content_hash: str


@dataclass(frozen=True)
class ArchiveAnnotation:
    id: int
    archive_id: int
    annotation_type: str
    payload: Mapping[str, Any]


@dataclass(frozen=True)
class ArchiveSearchResult:
    records: tuple[ArchiveRawRecord, ...]
    candidate_ids: tuple[int, ...]
    scanned_ids: tuple[int, ...]


def _content_hash(value: ArchiveRawInput) -> str:
    material = json.dumps(
        {
            "raw_payload": value.raw_payload,
            "raw_content": value.raw_content,
            "raw_actor_label": value.raw_actor_label,
            "raw_timestamp": value.raw_timestamp,
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(material).hexdigest()


class ColdArchiveService:
    def __init__(self) -> None:
        self._raw: dict[int, ArchiveRawRecord] = {}
        self._source_index: dict[tuple[str, str, str], int] = {}
        self._annotations: list[ArchiveAnnotation] = []
        self._next_raw_id = 1
        self._next_annotation_id = 1

    @property
    def raw_count(self) -> int:
        return len(self._raw)

    def raw(self, archive_id: int) -> ArchiveRawRecord:
        return self._raw[archive_id]

    def append_raw(self, value: ArchiveRawInput) -> ArchiveRawRecord:
        if not value.source_system or not value.source_conversation_id or not value.source_message_id:
            raise ValueError("archive source identity is required")
        identity = (
            value.source_system,
            value.source_conversation_id,
            value.source_message_id,
        )
        digest = _content_hash(value)
        existing_id = self._source_index.get(identity)
        if existing_id is not None:
            existing = self._raw[existing_id]
            if existing.content_hash != digest:
                raise ArchiveIdentityConflict("source identity content changed")
            return existing
        record = ArchiveRawRecord(
            **value.__dict__, id=self._next_raw_id, content_hash=digest
        )
        self._raw[record.id] = record
        self._source_index[identity] = record.id
        self._next_raw_id += 1
        return record

    def update_raw(self, _archive_id: int, _content: str) -> None:
        raise ImmutableArchiveError("Cold Archive raw rows are immutable")

    def delete_raw(self, _archive_id: int) -> None:
        raise ImmutableArchiveError("Cold Archive raw rows are immutable")

    def append_annotation(
        self, archive_id: int, annotation_type: str, payload: Mapping[str, Any]
    ) -> ArchiveAnnotation:
        if archive_id not in self._raw:
            raise KeyError(archive_id)
        if annotation_type not in ANNOTATION_TYPES:
            raise InvalidArchiveAnnotation("unsupported annotation type")
        annotation = ArchiveAnnotation(
            id=self._next_annotation_id,
            archive_id=archive_id,
            annotation_type=annotation_type,
            payload=dict(payload),
        )
        self._annotations.append(annotation)
        self._next_annotation_id += 1
        return annotation

    def normalized_view(self, archive_id: int) -> dict[str, Any]:
        raw = self._raw[archive_id]
        view = {
            "id": raw.id,
            "content": raw.raw_content,
            "actor_id": raw.mapped_actor,
            "scope": raw.mapped_scope,
            "timestamp": raw.raw_timestamp,
            "redacted": False,
        }
        for annotation in self._annotations:
            if annotation.archive_id != archive_id:
                continue
            if annotation.annotation_type == "correction":
                view["content"] = annotation.payload.get("content", view["content"])
            elif annotation.annotation_type == "identity_mapping":
                view["actor_id"] = annotation.payload.get("actor_id", view["actor_id"])
                view["scope"] = annotation.payload.get("scope", view["scope"])
            elif annotation.annotation_type == "timestamp_fix":
                view["timestamp"] = annotation.payload.get("timestamp", view["timestamp"])
            elif annotation.annotation_type == "redaction":
                view["content"] = "[REDACTED]"
                view["redacted"] = True
        return view

    def search(self, query: str, policy: RetrievalPolicy) -> ArchiveSearchResult:
        if not isinstance(policy, RetrievalPolicy):
            raise TypeError("Cold Archive search requires RetrievalPolicy")
        authorized = tuple(
            row
            for row in self._raw.values()
            if row.mapped_scope in policy.allowed_scopes
            and (
                not row.confidential
                or row.mapped_scope in policy.confidential_scopes
            )
        )
        matches = tuple(
            row for row in authorized if query.lower() in row.raw_content.lower()
        )
        return ArchiveSearchResult(
            records=matches,
            candidate_ids=tuple(row.id for row in matches),
            scanned_ids=tuple(row.id for row in authorized),
        )

