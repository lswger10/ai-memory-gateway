import pytest

from cold_archive import (
    ArchiveRawInput,
    ColdArchiveService,
    ImmutableArchiveError,
    InvalidArchiveAnnotation,
)
from memory_policy import build_retrieval_policy, room_members


def _record(message_id, scope, content, *, confidential=False):
    return ArchiveRawInput(
        source_system="synthetic-export",
        source_conversation_id="conversation-2026",
        source_message_id=message_id,
        raw_actor_label="assistant",
        raw_payload={"id": message_id, "text": content},
        raw_content=content,
        raw_timestamp="2026-08-01T00:00:00Z",
        mapped_actor="jiao" if scope == "weiwei-jiao" else "laoke",
        mapped_scope=scope,
        confidential=confidential,
        manifest_hash="a" * 64,
    )


def test_raw_archive_cannot_be_updated_or_deleted():
    service = ColdArchiveService()
    raw = service.append_raw(_record("m1", "weiwei-jiao", "private archive"))
    with pytest.raises(ImmutableArchiveError):
        service.update_raw(raw.id, "rewritten")
    with pytest.raises(ImmutableArchiveError):
        service.delete_raw(raw.id)
    assert service.raw(raw.id).raw_content == "private archive"


def test_append_only_annotations_use_bounded_types_and_normalized_view():
    service = ColdArchiveService()
    raw = service.append_raw(_record("m2", "weiwei-jiao", "typo"))
    service.append_annotation(raw.id, "correction", {"content": "corrected"})
    with pytest.raises(InvalidArchiveAnnotation):
        service.append_annotation(raw.id, "rewrite", {"content": "forbidden"})
    assert service.raw(raw.id).raw_content == "typo"
    assert service.normalized_view(raw.id)["content"] == "corrected"


def test_archive_search_applies_policy_before_candidate_creation():
    service = ColdArchiveService()
    allowed = service.append_raw(_record("m3", "weiwei-jiao", "private clue"))
    forbidden = service.append_raw(_record("m4", "weiwei-laoke", "private clue"))
    confidential = service.append_raw(
        _record("m5", "weiwei-jiao", "private clue", confidential=True)
    )
    policy = build_retrieval_policy(
        "jiao", "room_group_home", room_members("room_group_home")
    )
    result = service.search("private", policy)
    assert result.candidate_ids == (allowed.id,)
    assert forbidden.id not in result.scanned_ids
    assert confidential.id not in result.scanned_ids


def test_duplicate_source_identity_and_hash_is_idempotent():
    service = ColdArchiveService()
    first = service.append_raw(_record("m6", "weiwei-jiao", "same"))
    second = service.append_raw(_record("m6", "weiwei-jiao", "same"))
    assert second.id == first.id
    assert service.raw_count == 1


def test_postgres_schema_rejects_raw_update_and_delete():
    import database

    sql = database.SCOPED_MEMORY_MIGRATION_SQL
    assert "cold_archive_raw_immutable" in sql
    assert "BEFORE UPDATE OR DELETE ON cold_archive_raw" in sql
    assert "Cold Archive raw rows are immutable" in sql
