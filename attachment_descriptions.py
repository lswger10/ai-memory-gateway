from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any


class AttachmentDescriptionError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class AttachmentDescription:
    attachment_identity: str
    description_version: str
    description: str


class InMemoryAttachmentDescriptionStore:
    def __init__(self) -> None:
        self._items: dict[tuple[str, str], AttachmentDescription] = {}
        self._lock = asyncio.Lock()

    async def put_once(
        self, attachment_identity: str, description_version: str, description: str
    ) -> AttachmentDescription:
        if not all(
            isinstance(value, str) and value
            for value in (attachment_identity, description_version, description)
        ):
            raise AttachmentDescriptionError("description coordinates must be non-empty")
        async with self._lock:
            key = (attachment_identity, description_version)
            existing = self._items.get(key)
            if existing is not None:
                if existing.description != description:
                    raise AttachmentDescriptionError(
                        "persisted attachment description is immutable"
                    )
                return existing
            value = AttachmentDescription(
                attachment_identity=attachment_identity,
                description_version=description_version,
                description=description,
            )
            self._items[key] = value
            return value

    async def get(
        self, attachment_identity: str, description_version: str
    ) -> AttachmentDescription | None:
        async with self._lock:
            return self._items.get((attachment_identity, description_version))


@dataclass(frozen=True, slots=True)
class ImageHistoryItem:
    attachment_identity: str
    raw_block: dict[str, Any]


@dataclass(frozen=True, slots=True)
class PlannedImageHistoryItem:
    attachment_identity: str
    kind: str
    content: str | dict[str, Any]


async def plan_image_history(
    items: tuple[ImageHistoryItem, ...],
    *,
    recent_raw_limit: int,
    description_version: str,
    store: InMemoryAttachmentDescriptionStore,
) -> tuple[PlannedImageHistoryItem, ...]:
    if isinstance(recent_raw_limit, bool) or recent_raw_limit < 0:
        raise AttachmentDescriptionError("recent_raw_limit must be non-negative")
    raw_start = max(0, len(items) - recent_raw_limit)
    result: list[PlannedImageHistoryItem] = []
    for index, item in enumerate(items):
        if index >= raw_start:
            result.append(
                PlannedImageHistoryItem(
                    attachment_identity=item.attachment_identity,
                    kind="raw",
                    content=item.raw_block,
                )
            )
            continue
        description = await store.get(item.attachment_identity, description_version)
        if description is None:
            raise AttachmentDescriptionError(
                f"stable description is required for old image {item.attachment_identity}"
            )
        result.append(
            PlannedImageHistoryItem(
                attachment_identity=item.attachment_identity,
                kind="description",
                content=description.description,
            )
        )
    return tuple(result)
