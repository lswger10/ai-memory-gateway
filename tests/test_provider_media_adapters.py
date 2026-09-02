import base64

import httpx
import pytest

from media_materialization import (
    MediaMaterializationError,
    PreparedMedia,
    RelayMediaReader,
    render_media_tail,
)
from provider_adapters import (
    AnthropicMessagesAdapter,
    OpenAIChatCompletionsAdapter,
    OpenAIResponsesAdapter,
)
from tests.test_gateway_provider_adapters import _anthropic_layout, _profile
from model_profiles import ModelProfile


REFERENCE = {
    "attachment_id": "photo-1.png",
    "name": "photo.png",
    "media_type": "image/png",
    "size": 12,
    "category": "image",
    "purpose": "attachment",
    "source": {"type": "relay", "path": "/uploads/photo-1.png"},
    "derived_text": None,
    "semantic_label": None,
}


def media_enabled(profile):
    payload = profile.to_dict()
    payload["capabilities"]["input_modalities"] = ["text", "image"]
    return ModelProfile.from_dict(payload)


@pytest.mark.anyio
async def test_relay_media_reader_uses_internal_relay_endpoint_and_validates_bytes():
    calls = []

    async def handler(request):
        calls.append((str(request.url), request.headers.get("authorization")))
        return httpx.Response(200, content=b"\x89PNG\r\n\x1a\nbody", headers={"content-type": "image/png"})

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        reader = RelayMediaReader("https://relay.invalid", "gateway-key", http_client=client)
        prepared = await reader.fetch(REFERENCE)
    assert prepared.data.startswith(b"\x89PNG")
    assert calls == [
        ("https://relay.invalid/internal/group/media/photo-1.png", "Bearer gateway-key")
    ]


@pytest.mark.anyio
async def test_relay_media_reader_rejects_size_mime_signature_and_non_relay_source():
    async def wrong(_request):
        return httpx.Response(200, content=b"not-a-png!!", headers={"content-type": "text/plain"})

    async with httpx.AsyncClient(transport=httpx.MockTransport(wrong)) as client:
        reader = RelayMediaReader("https://relay.invalid", "gateway-key", http_client=client)
        with pytest.raises(MediaMaterializationError):
            await reader.fetch(REFERENCE)
        with pytest.raises(MediaMaterializationError):
            await reader.fetch({**REFERENCE, "source": {"type": "external", "path": "/x"}})


def test_provider_specific_image_shapes_use_same_prepared_media_after_breakpoint():
    prepared = PreparedMedia(REFERENCE, b"\x89PNG\r\n\x1a\nbody")
    encoded = base64.b64encode(prepared.data).decode("ascii")

    anthropic_profile = media_enabled(
        _profile("anthropic_messages", "anthropic_prefix_anchored_v1")
    )
    anthropic_parts = render_media_tail(anthropic_profile, (prepared,))
    anthropic = AnthropicMessagesAdapter().render(
        profile=anthropic_profile,
        layout=_anthropic_layout(),
        max_output_tokens=100,
        media_parts=anthropic_parts,
    ).json_body
    assert anthropic["messages"][-1]["content"][-1] == {
        "type": "image",
        "source": {"type": "base64", "media_type": "image/png", "data": encoded},
    }
    assert "cache_control" not in anthropic["messages"][-1]["content"][-1]

    for protocol, adapter in (
        ("openai_chat_completions", OpenAIChatCompletionsAdapter()),
        ("openai_responses", OpenAIResponsesAdapter()),
    ):
        profile = media_enabled(_profile(
            protocol, "openai_stable_prefix_v1", ttl=None, cache_ttls=[]
        )
        )
        parts = render_media_tail(profile, (prepared,))
        if protocol == "openai_chat_completions":
            body = adapter.render(
                profile=profile,
                system_content="system",
                messages=({"role": "user", "content": "current"},),
                prompt_cache_key="key",
                max_output_tokens=100,
                media_parts=parts,
            ).json_body
            assert body["messages"][-1]["content"][-1]["image_url"]["url"].endswith(encoded)
        else:
            body = adapter.render(
                profile=profile,
                instructions="system",
                input_items=({"role": "user", "content": [{"type": "input_text", "text": "current"}]},),
                prompt_cache_key="key",
                max_output_tokens=100,
                media_parts=parts,
            ).json_body
            assert body["input"][-1]["content"][-1]["image_url"].endswith(encoded)


def test_unverified_modality_uses_truthful_text_fallback_without_fetch_guessing():
    profile = _profile(
        "openai_chat_completions", "openai_stable_prefix_v1", ttl=None, cache_ttls=[]
    )
    prepared = PreparedMedia(
        {**REFERENCE, "derived_text": "一张稳定描述的海边照片"}, None
    )
    assert render_media_tail(profile, (prepared,)) == (
        {"kind": "text", "text": "[image: photo.png] 一张稳定描述的海边照片"},
    )
