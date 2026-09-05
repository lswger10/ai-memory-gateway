"""Real DOM checks; every HTTP request is intercepted, never a live provider."""
import json
from pathlib import Path
import sys
from urllib.parse import urlsplit

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def browser():
    playwright = pytest.importorskip("playwright.sync_api")
    with playwright.sync_playwright() as runtime:
        browser = runtime.chromium.launch(channel="msedge" if sys.platform == "win32" else "chromium", headless=True)
        yield browser
        browser.close()


@pytest.fixture
def dashboard(browser):
    from main import _safe_profile
    from test_model_profile_management_api import _profile_payload
    from model_profiles import ModelProfile
    profiles = [_safe_profile(ModelProfile.from_dict(_profile_payload(
        profile_id=name, display_name=name, revision=revision, test_status=status)))
        for name, revision, status in (("claude", 4, "passed"), ("gpt", 3, "passed"), ("untested", 1, "unverified"))]
    responses = {
        "/api/model-profiles": {"profiles": profiles},
        "/api/model-bindings": {"actor_defaults": {
            "jiao": {"profile_id": "gpt", "approved_fallback_profile_ids": ["claude"], "revision": 7},
            "laoke": {"profile_id": "claude", "approved_fallback_profile_ids": [], "revision": 2}},
            "bindings": {"jiao": {"room_group_home": {"profile_id": "claude", "source": "room_override", "binding_revision": 1}}}},
        "/api/model-usage/summary": {"cache_view": [], "cache_observability": {}},
        "/api/cache-pins": {"pins": []},
        "/api/actor-memory-tools/audit": {"items": []},
    }
    writes = []
    page = browser.new_page()

    def route(request):
        path = urlsplit(request.request.url).path
        if path == "/":
            return request.fulfill(body=(ROOT / "templates/dashboard.html").read_text(encoding="utf-8"), content_type="text/html")
        if path.startswith("/static/"):
            return request.fulfill(path=str(ROOT / path.lstrip("/")))
        if request.request.method != "GET":
            writes.append((path, request.request.post_data_json))
            return request.fulfill(json={"revision": 8, "accepted": True})
        value = responses.get(path, {})
        if isinstance(value, tuple):
            return request.fulfill(status=value[0], json=value[1])
        request.fulfill(json=value)

    page.route("**/*", route)
    page.goto("http://dashboard.test/")
    page.locator('[data-section="models"]').click()
    page.wait_for_function("document.querySelector('#probe-profile').options.length === 3")
    yield page, responses, writes
    page.close()


def test_binding_form_displays_actual_default_and_saves_once(dashboard):
    page, _, writes = dashboard
    assert page.locator("#binding-default-profile").input_value() == "gpt"
    assert page.locator("#binding-fallbacks").input_value() == "claude"
    assert page.locator('#binding-default-profile option[value="untested"]').count() == 0
    page.get_by_role("button", name="保存默认与 fallback").click()
    page.wait_for_function("document.querySelector('#models-msg').textContent.includes('已保存')")
    assert writes == [("/api/model-bindings", {"action": "save_actor_binding", "actor_id": "jiao",
        "profile_id": "gpt", "profile_ids": ["claude"], "expected_revision": 7})]
    page.select_option("#binding-actor", "laoke")
    assert page.locator("#binding-default-profile").input_value() == "claude"
    assert page.locator("#binding-fallbacks").input_value() == ""


def test_profile_editor_loads_real_values_and_advances_revision(dashboard):
    page, _, writes = dashboard
    page.locator('#model-profile-list button[data-profile-id="claude"]').click(timeout=1500)
    assert page.locator("#model-profile-base-url").input_value() == "https://api.ofox.ai/anthropic"
    assert page.locator("#model-profile-key-env").input_value() == ""
    page.fill("#model-profile-name", "edited name")
    page.locator('#model-profile-save').click()
    page.wait_for_function("document.querySelector('#models-msg').textContent.includes('已保存')")
    assert len(writes) == 1
    payload = writes[0][1]
    assert payload["revision"] == 5
    assert payload["display_name"] == "edited name"
    assert "credential_ref" not in payload
    assert "headers" not in payload


def test_failed_api_is_visible_not_empty_success(dashboard):
    page, responses, _ = dashboard
    responses["/api/actor-memory-tools/audit"] = (401, {"detail": "unauthorized"})
    page.evaluate("loadModelProfilesAndUsage()")
    assert "401" in page.locator("#actor-memory-audit-body").inner_text()
    assert "暂无" not in page.locator("#actor-memory-audit-body").inner_text()
    assert page.locator("#binding-default-profile").input_value() == "gpt"


def test_active_pin_shows_observed_cache_result_separately(dashboard):
    page, responses, _ = dashboard
    responses["/api/cache-pins"] = {"pins": [{"room_id": "room_weiwei_laoke", "execution_mode": "private",
        "conversation_id": "conversation-1", "enabled": True, "status": "active", "actors": {
            "laoke": {"status": "active", "profile_id": "claude", "call_count": 68,
                "cache_outcome": "OBSERVED_MISS", "last_error": None,
                "cache_read_input_tokens": 0}}}]}
    page.evaluate("loadModelProfilesAndUsage()")
    text = page.locator("#conversation-cache-pin-list").inner_text()
    assert "active" in text
    assert "可观测未命中" in text


def test_double_click_probe_only_sends_once_for_displayed_revision(dashboard):
    page, _, writes = dashboard
    page.fill("#probe-conversation", "synthetic-conversation")
    page.evaluate("Promise.all([runCacheProbe(), runCacheProbe()])")
    probes = [body for path, body in writes if path == "/api/cache-probes"]
    assert len(probes) == 1
    assert probes[0]["profile_revision"] == 4


def test_blank_new_profile_does_not_submit_placeholder_values(dashboard):
    page, _, writes = dashboard
    page.locator("#model-profile-save").click()
    assert not page.locator("#model-profile-form").evaluate("form => form.checkValidity()")
    assert writes == []


def test_failed_binding_reload_cannot_save_stale_defaults(dashboard):
    page, responses, writes = dashboard
    responses["/api/model-bindings"] = (503, {"detail": "unavailable"})
    page.evaluate("saveActorBinding()")
    assert page.locator("#binding-save").is_disabled()
    page.select_option("#binding-actor", "laoke")
    assert page.locator("#binding-save").is_disabled()
    assert len(writes) == 1
