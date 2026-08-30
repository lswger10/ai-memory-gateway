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


class PostgresAttachmentDescriptionStore:
    def __init__(self, pool_factory) -> None:
        self._pool_factory = pool_factory

    @staticmethod
    def _validate(
        attachment_identity: str, description_version: str, description: str | None = None
    ) -> None:
        values = (attachment_identity, description_version)
        if description is not None:
            values += (description,)
        if not all(isinstance(value, str) and value for value in values):
            raise AttachmentDescriptionError(
                "description coordinates must be non-empty"
            )

    @staticmethod
    def _from_row(row) -> AttachmentDescription | None:
        if row is None:
            return None
        return AttachmentDescription(
            attachment_identity=str(row["attachment_identity"]),
            description_version=str(row["description_version"]),
            description=str(row["description"]),
        )

    async def put_once(
        self, attachment_identity: str, description_version: str, description: str
    ) -> AttachmentDescription:
        self._validate(attachment_identity, description_version, description)
        pool = await self._pool_factory()
        async with pool.acquire() as conn:
            row = await conn.fetchrow(
                """WITH inserted AS (
                     INSERT INTO model_attachment_descriptions(
                       attachment_identity,description_version,description
                     ) VALUES($1,$2,$3)
                     ON CONFLICT(attachment_identity,description_version) DO NOTHING
                   )
                   SELECT attachment_identity,description_version,description
                   FROM model_attachment_descriptions
                   WHERE attachment_identity=$1 AND description_version=$2""",
                attachment_identity,
                description_version,
                description,
            )
        value = self._from_row(row)
        if value is None:
            raise AttachmentDescriptionError("description could not be persisted")
        if value.description != description:
            raise AttachmentDescriptionError(
                "persisted attachment description is immutable"
            )
        return value

    async def get(
        self, attachment_identity: str, description_version: str
    ) -> AttachmentDescription | None:
        self._validate(attachment_identity, description_version)
        pool = await self._pool_factory()
        async with pool.acquire() as conn:
            row = await conn.fetchrow(
                """SELECT attachment_identity,description_version,description
                   FROM model_attachment_descriptions
                   WHERE attachment_identity=$1 AND description_version=$2""",
                attachment_identity,
                description_version,
            )
        return self._from_row(row)


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
