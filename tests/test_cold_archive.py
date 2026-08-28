import asyncio
import pytest
from unittest.mock import AsyncMock, patch

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


def test_archive_search_uses_correction_instead_of_immutable_raw_content():
    service = ColdArchiveService()
    raw = service.append_raw(_record("m2-correct", "weiwei-jiao", "wrong flower"))
    service.append_annotation(raw.id, "correction", {"content": "moonlit orchid"})
    policy = build_retrieval_policy(
        "jiao", "room_weiwei_jiao", room_members("room_weiwei_jiao")
    )

    corrected = service.search("orchid", policy)
    stale = service.search("wrong flower", policy)

    assert corrected.candidate_ids == (raw.id,)
    assert corrected.records[0]["content"] == "moonlit orchid"
    assert stale.candidate_ids == ()


def test_archive_redaction_removes_row_before_candidate_creation():
    service = ColdArchiveService()
    raw = service.append_raw(_record("m2-redact", "weiwei-jiao", "private secret"))
    service.append_annotation(raw.id, "redaction", {"reason": "user request"})
    policy = build_retrieval_policy(
        "jiao", "room_weiwei_jiao", room_members("room_weiwei_jiao")
    )

    result = service.search("secret", policy)

    assert raw.id not in result.scanned_ids
    assert raw.id not in result.candidate_ids


def test_database_archive_search_applies_annotations_before_matching():
    import inspect
    import database

    source = inspect.getsource(database.search_authorized_archive_candidates)
    assert "cold_archive_annotations" in source
    assert "redaction" in source
    assert "correction" in source


def test_authorized_archive_candidate_is_returned_to_context_pack_search():
    import database

    class EmptyConnection:
        async def fetch(self, *_args):
            return []

        async def execute(self, *_args):
            return None

    class Acquire:
        async def __aenter__(self):
            return EmptyConnection()

        async def __aexit__(self, *_args):
            return False

    class Pool:
        def acquire(self):
            return Acquire()

    policy = build_retrieval_policy(
        "jiao", "room_weiwei_jiao", room_members("room_weiwei_jiao")
    )
    archive = ({
        "id": 71,
        "content": "moonlit orchid",
        "scope": "weiwei-jiao",
        "confidential": False,
    },)
    with (
        patch.object(database, "get_pool", AsyncMock(return_value=Pool())),
        patch.object(
            database,
            "search_authorized_archive_candidates",
            AsyncMock(return_value=archive),
        ),
        patch.object(
            database,
            "search_authorized_summary_candidates",
            AsyncMock(return_value=()),
        ),
        patch.object(database, "MEMORY_VECTOR_ENABLED", False),
    ):
        result = asyncio.run(
            database.search_authorized_memories("orchid", policy, limit=10)
        )

    assert result.memories == ({**archive[0], "source_kind": "cold_archive"},)
    assert result.audit.archive_candidate_ids == (71,)


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
