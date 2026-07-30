import asyncio
import inspect
import unittest
import uuid
from unittest.mock import AsyncMock, patch

from fastapi import HTTPException

import database
import main as gateway


class FakeRequest:
    def __init__(self, body, headers=None):
        self._body = dict(body)
        self.headers = headers or {}

    async def json(self):
        return dict(self._body)


class FakeResponse:
    status_code = 200
    text = ""

    def json(self):
        return {
            "choices": [{"message": {"content": "gateway answer"}}]
        }


class RecordingAsyncClient:
    calls = []

    def __init__(self, *args, **kwargs):
        pass

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    async def post(self, url, headers=None, json=None):
        self.__class__.calls.append(
            {"url": url, "headers": dict(headers or {}), "json": dict(json or {})}
        )
        return FakeResponse()


class GatewayRequestIdentityContractTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        RecordingAsyncClient.calls = []
        self.session_id = str(uuid.uuid4())
        self.request_id = str(uuid.uuid4())
        self.body = {
            "model": "contract-model",
            "stream": False,
            "messages": [{"role": "user", "content": "hello"}],
        }
        self.global_patchers = (
            patch.object(gateway, "API_KEY", "test-api-key"),
            patch.object(
                gateway,
                "API_BASE_URL",
                "https://upstream.test/v1/chat/completions",
            ),
            patch.object(gateway, "DEFAULT_MODEL", "contract-model"),
            patch.object(gateway, "PARTITION_SESSION_ID", "legacy-global"),
            patch.object(gateway, "CACHE_PARTITION_ENABLED", True),
            patch.object(gateway, "MEMORY_ENABLED", True),
            patch.object(gateway, "MEMORY_EXTRACT_ENABLED", False),
            patch.object(gateway, "SYSTEM_PROMPT", ""),
            patch.object(gateway, "REASONING_EFFORT", ""),
            patch.object(gateway, "FORCE_STREAM", False),
        )
        for patcher in self.global_patchers:
            patcher.start()

    async def asyncTearDown(self):
        for patcher in reversed(self.global_patchers):
            patcher.stop()
        await asyncio.sleep(0)

    async def invoke(self, headers=None, body=None):
        conversation_reads = AsyncMock(return_value=[])
        cache_builds = AsyncMock(
            side_effect=lambda _sid, messages, _prompt, _user: messages
        )
        background_sessions = []

        async def record_background(session_id, *_args, **_kwargs):
            background_sessions.append(session_id)

        with (
            patch.object(
                gateway, "get_conversation_messages", conversation_reads
            ),
            patch.object(gateway, "build_partitioned_messages", cache_builds),
            patch.object(
                gateway, "process_memories_background", record_background
            ),
            patch.object(gateway.httpx, "AsyncClient", RecordingAsyncClient),
        ):
            response = await gateway._chat_completions_inner(
                FakeRequest(body or self.body, headers)
            )
            await asyncio.sleep(0)

        return response, conversation_reads, cache_builds, background_sessions

    def tidal_headers(self, session_id=None, request_id=None):
        return {
            "X-Gateway-Session-ID": session_id or self.session_id,
            "X-Gateway-Request-ID": request_id or self.request_id,
        }

    async def test_tidal_headers_override_global_session_for_all_window_state(self):
        _, conversation_reads, cache_builds, background_sessions = await self.invoke(
            self.tidal_headers()
        )

        actual_sessions = (
            conversation_reads.await_args.args[0],
            cache_builds.await_args.args[0],
            background_sessions[0],
        )
        self.assertEqual(
            (self.session_id, self.session_id, self.session_id),
            actual_sessions,
        )
        self.assertNotIn(self.request_id, actual_sessions)

    async def test_missing_both_identity_headers_keeps_legacy_compatibility(self):
        _, conversation_reads, cache_builds, background_sessions = await self.invoke()
        self.assertEqual("legacy-global", conversation_reads.await_args.args[0])
        self.assertEqual("legacy-global", cache_builds.await_args.args[0])
        self.assertEqual(["legacy-global"], background_sessions)

    async def test_request_id_changes_do_not_change_the_session_partition(self):
        first_request_id = str(uuid.uuid4())
        second_request_id = str(uuid.uuid4())
        _, first_reads, _, _ = await self.invoke(
            self.tidal_headers(request_id=first_request_id)
        )
        _, second_reads, _, _ = await self.invoke(
            self.tidal_headers(request_id=second_request_id)
        )

        self.assertEqual(
            (self.session_id, self.session_id),
            (
                first_reads.await_args.args[0],
                second_reads.await_args.args[0],
            ),
        )

    async def test_partial_or_invalid_tidal_identity_is_rejected(self):
        invalid_headers = (
            {"X-Gateway-Session-ID": self.session_id},
            {"X-Gateway-Request-ID": self.request_id},
            self.tidal_headers(session_id="not-a-window-id"),
            self.tidal_headers(request_id="not-a-request-id"),
        )
        for headers in invalid_headers:
            with self.subTest(headers=headers):
                with self.assertRaises(HTTPException) as raised:
                    await self.invoke(headers)
                self.assertEqual(400, raised.exception.status_code)

    async def test_legacy_main_is_a_valid_stable_tidal_session(self):
        _, conversation_reads, _, _ = await self.invoke(
            self.tidal_headers(session_id="legacy-main")
        )
        self.assertEqual("legacy-main", conversation_reads.await_args.args[0])

    async def test_internal_identity_fields_are_not_forwarded_to_upstream(self):
        body = dict(self.body)
        body.update(
            {
                "window_id": self.session_id,
                "request_id": self.request_id,
                "session_id": "internal-session",
                "gateway_session_id": self.session_id,
                "gateway_request_id": self.request_id,
                "source_session": self.session_id,
                "provenance": {"source_type": "chat_extraction"},
            }
        )

        await self.invoke(self.tidal_headers(), body)
        upstream_call = RecordingAsyncClient.calls[-1]
        internal_body_fields = {
            "window_id",
            "request_id",
            "session_id",
            "gateway_session_id",
            "gateway_request_id",
            "source_session",
            "provenance",
        }
        self.assertFalse(internal_body_fields & upstream_call["json"].keys())

    async def test_internal_identity_headers_are_not_forwarded_to_upstream(self):
        await self.invoke(self.tidal_headers())
        upstream_call = RecordingAsyncClient.calls[-1]
        self.assertNotIn("X-Gateway-Session-ID", upstream_call["headers"])
        self.assertNotIn("X-Gateway-Request-ID", upstream_call["headers"])

    async def test_multimodal_content_survives_gateway_upstream_forwarding(self):
        image_block = {
            "type": "image_url",
            "image_url": {"url": "data:image/png;base64,cG5nZGF0YQ=="},
        }
        body = dict(self.body)
        body["messages"] = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "inspect this image"},
                    image_block,
                ],
            }
        ]

        await self.invoke(self.tidal_headers(), body)

        upstream_call = RecordingAsyncClient.calls[-1]
        user_message = upstream_call["json"]["messages"][-1]
        self.assertEqual("user", user_message["role"])
        self.assertIn(image_block, user_message["content"])
        self.assertNotIn("window_id", upstream_call["json"])
        self.assertNotIn("request_id", upstream_call["json"])


class LongTermMemoryContractTests(unittest.IsolatedAsyncioTestCase):
    def test_memory_storage_accepts_provenance_metadata(self):
        parameters = inspect.signature(database.save_memory).parameters
        self.assertIn("source_session", parameters)
        self.assertIn("provenance", parameters)

    async def test_global_memory_retrieval_does_not_filter_by_window_session(self):
        search = AsyncMock(
            return_value=[
                {
                    "content": "shared stable fact",
                    "importance": 8,
                    "source_session": "another-window",
                    "created_at": None,
                }
            ]
        )
        with (
            patch.object(gateway, "search_memories", search),
            patch.object(gateway, "MAX_MEMORIES_INJECT", 5),
        ):
            result = await gateway.build_memory_text("project fact")

        self.assertIn("shared stable fact", result)
        self.assertNotIn("source_session", search.await_args.kwargs)

    async def test_extracted_memory_records_request_provenance(self):
        session_id = str(uuid.uuid4())
        request_id = str(uuid.uuid4())
        save_memory = AsyncMock()

        with (
            patch.object(gateway, "MEMORY_EXTRACT_ENABLED", True),
            patch.object(gateway, "MEMORY_EXTRACT_INTERVAL", 1),
            patch.object(gateway, "_round_counter", 0),
            patch.object(gateway, "get_last_user_content", AsyncMock(return_value="")),
            patch.object(gateway, "save_message", AsyncMock()),
            patch.object(gateway, "get_recent_memories", AsyncMock(return_value=[])),
            patch.object(
                gateway,
                "extract_memories",
                AsyncMock(
                    return_value=[{"content": "stable fact", "importance": 7}]
                ),
            ),
            patch.object(gateway, "save_memory", save_memory),
            patch.object(gateway, "get_all_memories_count", AsyncMock(return_value=1)),
        ):
            await gateway.process_memories_background(
                session_id,
                "hello",
                "answer",
                "contract-model",
                context_messages=[{"role": "user", "content": "hello"}],
                request_id=request_id,
            )

        kwargs = save_memory.await_args.kwargs
        self.assertEqual(session_id, kwargs["source_session"])
        self.assertEqual("chat_extraction", kwargs["provenance"]["source_type"])
        self.assertEqual(request_id, kwargs["provenance"]["request_id"])


if __name__ == "__main__":
    unittest.main()
