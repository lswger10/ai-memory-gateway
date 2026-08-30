import pytest

from attachment_descriptions import (
    AttachmentDescriptionError,
    ImageHistoryItem,
    InMemoryAttachmentDescriptionStore,
    plan_image_history,
)


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
