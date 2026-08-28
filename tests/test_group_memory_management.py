import json
from pathlib import Path
from unittest.mock import AsyncMock, patch

from fastapi.testclient import TestClient


FIXTURE_ROOT = (
    Path(__file__).resolve().parents[1] / "contracts" / "group-room" / "v1" / "fixtures"
)
PACK_REQUEST = json.loads(
    (FIXTURE_ROOT / "context-pack-request.json").read_text(encoding="utf-8")
)


def test_admin_can_list_all_typed_fields_including_jiao_laoke_confidential():
    import main

    row = {
        "id": 77,
        "content": "椒椒和老克之间的管理员可见记录",
        "importance": 5,
        "source_session": "group-home-conversation",
        "created_at": None,
        "layer": 1,
        "title": None,
        "is_active": True,
        "merged_from": None,
        "event_date": None,
        "scope": "jiao-laoke",
        "memory_type": "inference",
        "perspective": "laoke",
        "confidential": True,
        "status": "active",
        "confidence": 0.8,
        "evidence_count": 2,
        "last_supported_at": None,
        "superseded_by": None,
        "derived_from": None,
        "source_kind": "chat_extraction",
        "provenance": {"source_event_id": 201},
        "evidence": [101, 201],
    }
    loader = AsyncMock(return_value=[row])
    with patch.object(main, "MEMORY_ENABLED", True), patch.object(
        main, "get_all_memories_detail", loader
    ), patch.object(main, "get_layer_statistics", AsyncMock(return_value=None)):
        response = TestClient(main.app).get(
            "/api/memories?scope=jiao-laoke&confidential=true"
        )

    assert response.status_code == 200
    assert response.json()["memories"][0]["confidential"] is True
    assert response.json()["memories"][0]["scope"] == "jiao-laoke"
    loader.assert_awaited_once_with(
        layer=None, active_only=None, scope="jiao-laoke", confidential=True
    )


def test_admin_credential_cannot_be_reused_as_actor_or_orchestrator_credential():
    import main

    env = {
        "GATEWAY_GROUP_MEMORY_ENABLED": "true",
        "GROUP_ORCHESTRATOR_SERVICE_KEY": "orchestrator-key",
    }
    with patch.dict("os.environ", env, clear=True), patch.object(
        main, "GATEWAY_SECRET", "admin-key"
    ):
        response = TestClient(main.app).post(
            "/internal/group/context-packs/full",
            headers={
                "X-Gateway-Key": "admin-key",
                "X-Group-Contract-Version": "group-room.v1.0",
            },
            json=PACK_REQUEST,
        )
    assert response.status_code == 401
    assert response.json()["error"]["code"] == "invalid_service_key"


def test_archive_management_is_read_plus_append_only_annotation():
    import main

    archive_rows = [{"id": 5, "raw_content": "raw", "annotations": []}]
    list_fn = AsyncMock(return_value=archive_rows)
    append_fn = AsyncMock(return_value={"id": 9, "archive_id": 5, "annotation_type": "note", "payload": {"text": "checked"}})
    client = TestClient(main.app)
    with patch.object(main, "MEMORY_ENABLED", True), patch.object(
        main, "list_cold_archive_for_management", list_fn
    ), patch.object(main, "append_cold_archive_annotation", append_fn):
        listed = client.get("/api/archive")
        appended = client.post(
            "/api/archive/5/annotations",
            json={"annotation_type": "note", "payload": {"text": "checked"}},
        )
        raw_update = client.put("/api/archive/5", json={"raw_content": "changed"})

    assert listed.json()["archive"] == archive_rows
    assert appended.status_code == 200
    assert appended.json()["annotation"]["id"] == 9
    append_fn.assert_awaited_once()
    assert raw_update.status_code in {404, 405}


def test_archive_annotation_rejects_unbounded_type():
    import main

    with patch.object(main, "MEMORY_ENABLED", True):
        response = TestClient(main.app).post(
            "/api/archive/5/annotations",
            json={"annotation_type": "rewrite_raw", "payload": {}},
        )
    assert response.status_code == 422
