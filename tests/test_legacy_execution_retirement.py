from pathlib import Path
from unittest.mock import AsyncMock, patch

from fastapi.testclient import TestClient

import main


ROOT = Path(__file__).resolve().parents[1]


def test_legacy_openai_execution_and_global_ai_settings_are_not_routes():
    paths = {route.path for route in main.app.routes}
    assert "/v1/chat/completions" not in paths
    assert "/v1/models" not in paths
    assert "/api/settings" not in paths
    assert "/api/models" not in paths
    assert not any(path.startswith("/api/partition") for path in paths)
    assert "/api/memory-settings" in paths
    assert "/api/cache-pins" in paths


def test_memory_settings_exclude_retired_global_execution_authority():
    with patch.object(main, "get_all_gateway_config", AsyncMock(return_value={})):
        response = TestClient(main.app).get("/api/memory-settings")
    assert response.status_code == 200
    settings = response.json()["settings"]
    assert not {
        "API_BASE_URL",
        "API_KEY",
        "DEFAULT_MODEL",
        "systemPrompt",
        "FORCE_STREAM",
        "REASONING_EFFORT",
        "CACHE_PARTITION_ENABLED",
        "CACHE_TTL",
    } & settings.keys()


def test_health_names_the_single_current_configuration_authority(monkeypatch):
    monkeypatch.setattr(main, "MEMORY_ENABLED", False)
    response = TestClient(main.app).get("/")
    assert response.status_code == 200
    assert response.json()["configuration_authority"] == "model_profiles_and_actor_personas"
    assert "system_prompt_loaded" not in response.json()


def test_dashboard_uses_profile_persona_and_memory_sources_only():
    html = (ROOT / "templates" / "dashboard.html").read_text(encoding="utf-8")
    javascript = (ROOT / "static" / "js" / "dashboard.js").read_text(
        encoding="utf-8"
    )
    assert "set-API_BASE_URL" not in html
    assert "set-DEFAULT_MODEL" not in html
    assert "set-systemPrompt" not in html
    assert 'data-section="threads"' not in html
    assert 'id="section-threads"' not in html
    assert "/api/settings" not in javascript
    assert "/api/partition" not in javascript
    assert "/api/models" not in javascript
    assert "/api/memory-settings" in javascript
