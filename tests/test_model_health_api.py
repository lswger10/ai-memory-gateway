from unittest.mock import AsyncMock

from fastapi.testclient import TestClient

import main


def test_gateway_query_secret_is_not_authentication(monkeypatch):
    monkeypatch.setattr(main, "GATEWAY_SECRET", "admin-test")
    monkeypatch.setattr(main, "get_all_gateway_config", AsyncMock(return_value={}))
    client = TestClient(main.app)
    assert client.get("/api/memory-settings?gateway_key=admin-test").status_code == 401
    assert client.get("/api/memory-settings", headers={"X-Gateway-Key": "admin-test"}).status_code == 200


def test_health_check_requires_charge_confirmation_and_current_revision(monkeypatch):
    monkeypatch.setattr(main, "GATEWAY_SECRET", "admin-test")
    monkeypatch.setenv("MODEL_PROFILE_MANAGEMENT_ENABLED", "true")
    service = AsyncMock()
    service.check.return_value = {"profile_id": "new", "profile_revision": 2, "status": "verified"}
    monkeypatch.setattr(main, "_get_model_health_service", AsyncMock(return_value=service))
    client = TestClient(main.app, headers={"X-Gateway-Key": "admin-test"})
    assert client.post("/api/model-health", json=[]).status_code == 422
    assert client.post("/api/model-health", json={"profile_id": "new", "profile_revision": 2}).status_code == 409
    service.check.assert_not_called()
    assert client.post("/api/model-health", json={"profile_id": "new", "profile_revision": True,
        "confirm_provider_charges": True}).status_code == 422
    result = client.post("/api/model-health", json={"profile_id": "new", "profile_revision": 2,
        "confirm_provider_charges": True})
    assert result.status_code == 200
    assert result.json()["status"] == "verified"
    service.check.assert_awaited_once_with("new", profile_revision=2)
