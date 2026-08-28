"""No-guess archive parser protocol and source inventory."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Protocol, Sequence


@dataclass(frozen=True)
class ArchiveRecord:
    source_system: str
    source_conversation_id: str
    source_message_id: str
    raw_actor_label: str
    raw_payload: Mapping[str, Any]
    raw_content: str
    raw_timestamp: str | None = None


class ArchiveSourceParser(Protocol):
    format_id: str

    def detect(self, raw: Any) -> bool: ...

    def iter_records(
        self, raw: Any, approved_mapping: Mapping[str, str]
    ) -> Iterable[ArchiveRecord]: ...


@dataclass(frozen=True)
class SourceInventory:
    source_system: str
    top_level_type: str
    field_paths: tuple[str, ...]
    counts: Mapping[str, int]
    detected_format: str | None
    records: tuple[ArchiveRecord, ...]
    requires_human_field_mapping: bool
    suggested_actor_mapping: Mapping[str, str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_system": self.source_system,
            "top_level_type": self.top_level_type,
            "field_paths": list(self.field_paths),
            "counts": dict(self.counts),
            "detected_format": self.detected_format,
            "records": [],
            "requires_human_field_mapping": self.requires_human_field_mapping,
            "suggested_actor_mapping": {},
        }


def _field_paths(value: Any, prefix: str = "") -> set[str]:
    paths: set[str] = set()
    if isinstance(value, dict):
        for key, child in value.items():
            path = f"{prefix}.{key}" if prefix else str(key)
            paths.add(path)
            paths.update(_field_paths(child, path))
    elif isinstance(value, list):
        list_path = f"{prefix}[]" if prefix else "[]"
        paths.add(list_path)
        for child in value[:20]:
            paths.update(_field_paths(child, list_path))
    return paths


def _counts(value: Any) -> dict[str, int]:
    if isinstance(value, dict):
        return {"top_level_fields": len(value), "top_level_items": 1}
    if isinstance(value, list):
        return {"top_level_fields": 0, "top_level_items": len(value)}
    return {"top_level_fields": 0, "top_level_items": 1}


def inventory_archive(
    raw: Any,
    *,
    source_system: str,
    parsers: Sequence[ArchiveSourceParser] = (),
) -> SourceInventory:
    """Describe structure only; detection never authorizes parsing or identity mapping."""

    detected = next(
        (parser.format_id for parser in parsers if parser.detect(raw)), None
    )
    return SourceInventory(
        source_system=source_system,
        top_level_type=type(raw).__name__,
        field_paths=tuple(sorted(_field_paths(raw))),
        counts=_counts(raw),
        detected_format=detected,
        records=(),
        requires_human_field_mapping=True,
        suggested_actor_mapping={},
    )

