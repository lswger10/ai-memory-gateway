import pytest

from attachment_descriptions import (
    AttachmentDescriptionError,
    ImageHistoryItem,
    InMemoryAttachmentDescriptionStore,
    PostgresAttachmentDescriptionStore,
    plan_image_history,
)


class _Connection:
    def __init__(self):
        self.rows = {}

    async def fetchrow(self, sql, *args):
        key = (args[0], args[1])
        if "INSERT INTO model_attachment_descriptions" in sql:
            self.rows.setdefault(key, args[2])
        value = self.rows.get(key)
        if value is None:
            return None
        return {
            "attachment_identity": key[0],
            "description_version": key[1],
            "description": value,
        }


class _Acquire:
    def __init__(self, connection):
        self.connection = connection

    async def __aenter__(self):
        return self.connection

    async def __aexit__(self, *args):
        return False


class _Pool:
    def __init__(self):
        self.connection = _Connection()

    def acquire(self):
        return _Acquire(self.connection)


async def _pool(value):
    return value


@pytest.mark.anyio
async def test_old_images_use_persisted_descriptions_and_recent_images_remain_raw():
    store = InMemoryAttachmentDescriptionStore()
    await store.put_once("sha256:old", "vision-description.v1", "old beach photo")
    items = (
        ImageHistoryItem("sha256:old", {"type": "image", "data": "old-bytes"}),
        ImageHistoryItem("sha256:new", {"type": "image", "data": "new-bytes"}),
    )

    planned = await plan_image_history(
        items,
        recent_raw_limit=1,
        description_version="vision-description.v1",
        store=store,
    )

    assert planned[0].kind == "description"
    assert planned[0].content == "old beach photo"
    assert planned[1].kind == "raw"
    assert planned[1].content == {"type": "image", "data": "new-bytes"}


@pytest.mark.anyio
async def test_missing_old_image_description_requires_generation_before_cache_build():
    store = InMemoryAttachmentDescriptionStore()
    with pytest.raises(AttachmentDescriptionError, match="description"):
        await plan_image_history(
            (ImageHistoryItem("sha256:old", {"type": "image", "data": "bytes"}),),
            recent_raw_limit=0,
            description_version="vision-description.v1",
            store=store,
        )


@pytest.mark.anyio
async def test_description_is_idempotent_and_conflicts_are_rejected():
    store = InMemoryAttachmentDescriptionStore()
    first = await store.put_once("sha256:a", "v1", "stable description")
    retry = await store.put_once("sha256:a", "v1", "stable description")
    assert first is retry
    with pytest.raises(AttachmentDescriptionError, match="immutable"):
        await store.put_once("sha256:a", "v1", "different")


@pytest.mark.anyio
async def test_postgres_description_survives_store_recreation_and_rejects_conflict():
    pool = _Pool()
    first = PostgresAttachmentDescriptionStore(lambda: _pool(pool))
    saved = await first.put_once("sha256:a", "vision.v1", "stable description")

    recreated = PostgresAttachmentDescriptionStore(lambda: _pool(pool))
    restored = await recreated.get("sha256:a", "vision.v1")

    assert restored == saved
    with pytest.raises(AttachmentDescriptionError, match="immutable"):
        await recreated.put_once("sha256:a", "vision.v1", "different")
