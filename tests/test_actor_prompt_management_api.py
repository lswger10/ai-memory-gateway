import base64
import hashlib

from fastapi.testclient import TestClient

import main


class _PromptStore:
    def __init__(self):
        from actor_prompt_profiles import load_actor_prompt_profiles
        from actor_prompt_store import InMemoryActorPromptVersionStore

        self.inner = InMemoryActorPromptVersionStore(load_actor_prompt_profiles())

    def __getattr__(self, name):
        return getattr(self.inner, name)


def _enable(monkeypatch):
    monkeypatch.setenv("ACTOR_PERSONA_MANAGEMENT_ENABLED", "true")
    monkeypatch.setattr(main, "GATEWAY_SECRET", "persona-test-secret")
    monkeypatch.setattr(main._db_module, "DATABASE_URL", "postgresql://test")
    store = _PromptStore()
    monkeypatch.setattr(main, "_actor_prompt_store", store)
    client = TestClient(main.app, headers={"X-Gateway-Key": "persona-test-secret"})
    return store, client


def _payload(filename: str, text: str) -> dict:
    raw = text.encode("utf-8")
    return {
        "source_filename": filename,
        "prompt_base64": base64.b64encode(raw).decode("ascii"),
        "content_sha256": hashlib.sha256(raw).hexdigest(),
    }


def test_management_upload_list_activate_and_export_keep_body_private(monkeypatch):
    _store, client = _enable(monkeypatch)
    prompt = "\ufeff# synthetic actor prompt\r\n\r\nexact body.  \r\n"

    created = client.post(
        "/api/actor-prompts/jiao/versions",
        json=_payload("jiao-v2.md", prompt),
    )
    assert created.status_code == 200
    metadata = created.json()["version"]
    assert "prompt_text" not in metadata
    assert prompt not in created.text

    listed = client.get("/api/actor-prompts").json()
    assert prompt not in str(listed)
    assert listed["actors"]["jiao"]["active_version_id"] == "builtin:jiao.v1"

    activated = client.post(
        "/api/actor-prompts/jiao/activate",
        json={"version_id": metadata["version_id"], "expected_revision": 0},
    )
    assert activated.status_code == 200
    assert activated.json()["active"]["version_id"] == metadata["version_id"]

    exported = client.get(
        f"/api/actor-prompts/jiao/versions/{metadata['version_id']}/export"
    )
    assert exported.status_code == 200
    assert exported.content.decode("utf-8") == prompt
    assert exported.headers["content-type"].startswith("text/markdown")
    assert "attachment" in exported.headers["content-disposition"]


def test_management_rejects_wrong_actor_oversize_and_non_markdown(monkeypatch):
    _store, client = _enable(monkeypatch)
    monkeypatch.setenv("ACTOR_PROMPT_MAX_BYTES", "16")

    assert client.post(
        "/api/actor-prompts/weiwei/versions",
        json=_payload("x.md", "valid"),
    ).status_code == 404
    assert client.post(
        "/api/actor-prompts/jiao/versions",
        json=_payload("x.txt", "valid"),
    ).status_code == 422
    assert client.post(
        "/api/actor-prompts/jiao/versions",
        json=_payload("folder\\x.md", "valid"),
    ).status_code == 422
    assert client.post(
        "/api/actor-prompts/jiao/versions",
        json=_payload("x.md", "x" * 17),
    ).status_code == 413


def test_persona_management_is_default_off(monkeypatch):
    monkeypatch.delenv("ACTOR_PERSONA_MANAGEMENT_ENABLED", raising=False)
    monkeypatch.setattr(main, "GATEWAY_SECRET", "persona-test-secret")
    response = TestClient(main.app).get(
        "/api/actor-prompts", headers={"X-Gateway-Key": "persona-test-secret"}
    )
    assert response.status_code == 404


def test_persona_management_refuses_to_run_without_gateway_auth(monkeypatch):
    monkeypatch.setenv("ACTOR_PERSONA_MANAGEMENT_ENABLED", "true")
    monkeypatch.setattr(main, "GATEWAY_SECRET", "")
    monkeypatch.setattr(main._db_module, "DATABASE_URL", "postgresql://test")
    response = TestClient(main.app).get("/api/actor-prompts")
    assert response.status_code == 503


def test_persona_management_refuses_nonpersistent_storage(monkeypatch):
    monkeypatch.setenv("ACTOR_PERSONA_MANAGEMENT_ENABLED", "true")
    monkeypatch.setattr(main, "GATEWAY_SECRET", "persona-test-secret")
    monkeypatch.setattr(main._db_module, "DATABASE_URL", "")
    response = TestClient(main.app).get(
        "/api/actor-prompts", headers={"X-Gateway-Key": "persona-test-secret"}
    )
    assert response.status_code == 503
