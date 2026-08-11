import unittest
import uuid
from contextlib import ExitStack
from unittest.mock import AsyncMock, patch

import main as gateway
import memory_extractor


class PersistentConversationFake:
    """Minimal PostgreSQL-shaped state used by restart scheduling tests."""

    def __init__(self):
        self.messages = {}
        self.cursors = {}
        self.next_id = 1

    async def ensure_cursor(self, session_id):
        if session_id not in self.cursors:
            existing = self.messages.get(session_id, [])
            self.cursors[session_id] = max(
                (message["id"] for message in existing),
                default=0,
            )
        return self.cursors[session_id]

    async def save_message(self, session_id, role, content, model="", metadata=None):
        message_id = self.next_id
        self.next_id += 1
        self.messages.setdefault(session_id, []).append(
            {
                "id": message_id,
                "role": role,
                "content": content,
                "metadata": metadata,
            }
        )
        return message_id

    async def get_messages(self, session_id, after_id, through_id):
        return [
            dict(message)
            for message in self.messages.get(session_id, [])
            if after_id < message["id"] <= through_id
        ]

    async def save_cursor(self, session_id, message_id):
        self.cursors[session_id] = max(
            self.cursors.get(session_id, 0),
            message_id,
        )


class ExplicitMemoryIntentTests(unittest.TestCase):
    def test_matches_complete_explicit_memory_requests(self):
        samples = (
            "请记住：我的猫叫豆豆。",
            "请把我最喜欢海蓝色这件事保存为长期记忆。",
            "Remember that my emergency contact is Alex.",
            "Please save this as a long-term memory: I am allergic to peanuts.",
        )
        for sample in samples:
            with self.subTest(sample=sample):
                self.assertTrue(memory_extractor.is_explicit_memory_request(sample))

    def test_rejects_questions_negations_and_unrelated_save_commands(self):
        samples = (
            "你还记得我说过什么吗？",
            "你能记住多少内容？",
            "不要记住这句话。",
            "请保存这个文件到桌面。",
            "解释一下长期记忆的工作原理。",
            "Do not remember this sentence.",
        )
        for sample in samples:
            with self.subTest(sample=sample):
                self.assertFalse(memory_extractor.is_explicit_memory_request(sample))


class MemoryExtractionScheduleTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        if hasattr(gateway, "_memory_extraction_pending"):
            gateway._memory_extraction_pending.clear()
        if hasattr(gateway, "_memory_extraction_locks"):
            gateway._memory_extraction_locks.clear()

    def common_patches(self, interval, extracted=None):
        store = PersistentConversationFake()
        extractor = AsyncMock(return_value=list(extracted or []))
        saver = AsyncMock()
        patchers = (
            patch.object(gateway, "MEMORY_EXTRACT_ENABLED", True),
            patch.object(gateway, "MEMORY_EXTRACT_INTERVAL", interval),
            patch.object(gateway, "get_last_user_content", AsyncMock(return_value="")),
            patch.object(gateway, "save_message", store.save_message),
            patch.object(gateway, "ensure_memory_extraction_cursor", store.ensure_cursor),
            patch.object(gateway, "get_memory_extraction_messages", store.get_messages),
            patch.object(gateway, "save_memory_extraction_cursor", store.save_cursor),
            patch.object(gateway, "get_recent_memories", AsyncMock(return_value=[])),
            patch.object(gateway, "extract_memories", extractor),
            patch.object(gateway, "save_memory", saver),
            patch.object(gateway, "get_all_memories_count", AsyncMock(return_value=1)),
        )
        return patchers, extractor, saver, store

    async def run_round(self, session_id, user_text, request_id=None):
        await gateway.process_memories_background(
            session_id,
            user_text,
            "assistant response",
            "test-model",
            context_messages=[{"role": "user", "content": user_text}],
            request_id=request_id or str(uuid.uuid4()),
        )

    def start_patches(self, stack, patchers):
        for patcher in patchers:
            stack.enter_context(patcher)

    async def test_interval_rounds_are_counted_per_session(self):
        session_a = str(uuid.uuid4())
        session_b = str(uuid.uuid4())
        patchers, extractor, _, _ = self.common_patches(interval=2)

        with ExitStack() as stack:
            self.start_patches(stack, patchers)
            await self.run_round(session_a, "A first fact")
            await self.run_round(session_b, "B first fact")
            self.assertEqual(0, extractor.await_count)

            await self.run_round(session_a, "A second fact")
            self.assertEqual(1, extractor.await_count)
            first_batch = extractor.await_args_list[0].args[0]
            first_text = "\n".join(msg["content"] for msg in first_batch)
            self.assertIn("A first fact", first_text)
            self.assertIn("A second fact", first_text)
            self.assertNotIn("B first fact", first_text)

            await self.run_round(session_b, "B second fact")
            self.assertEqual(2, extractor.await_count)
            second_batch = extractor.await_args_list[1].args[0]
            second_text = "\n".join(msg["content"] for msg in second_batch)
            self.assertIn("B first fact", second_text)
            self.assertIn("B second fact", second_text)
            self.assertNotIn("A first fact", second_text)

    async def test_explicit_request_extracts_immediately_and_records_trigger(self):
        session_id = str(uuid.uuid4())
        request_id = str(uuid.uuid4())
        patchers, extractor, saver, _ = self.common_patches(
            interval=5,
            extracted=[{"content": "用户的猫叫豆豆", "importance": 8}],
        )

        with ExitStack() as stack:
            self.start_patches(stack, patchers)
            await self.run_round(session_id, "普通的第一轮")
            self.assertEqual(0, extractor.await_count)

            await self.run_round(
                session_id,
                "请记住：我的猫叫豆豆。",
                request_id=request_id,
            )

        self.assertEqual(1, extractor.await_count)
        extracted_text = "\n".join(
            msg["content"] for msg in extractor.await_args.args[0]
        )
        self.assertIn("普通的第一轮", extracted_text)
        self.assertIn("我的猫叫豆豆", extracted_text)
        provenance = saver.await_args.kwargs["provenance"]
        self.assertEqual("explicit", provenance["trigger"])
        self.assertEqual(request_id, provenance["request_id"])

    async def test_explicitly_processed_round_is_not_reprocessed_by_next_batch(self):
        session_id = str(uuid.uuid4())
        explicit_marker = "EXPLICIT-ROUND-ONLY"
        patchers, extractor, _, _ = self.common_patches(interval=2)

        with ExitStack() as stack:
            self.start_patches(stack, patchers)
            await self.run_round(
                session_id,
                f"请把 {explicit_marker} 保存为长期记忆。",
            )
            self.assertEqual(1, extractor.await_count)

            await self.run_round(session_id, "later round one")
            self.assertEqual(1, extractor.await_count)
            await self.run_round(session_id, "later round two")
            self.assertEqual(2, extractor.await_count)

        next_batch = extractor.await_args_list[1].args[0]
        next_text = "\n".join(msg["content"] for msg in next_batch)
        self.assertNotIn(explicit_marker, next_text)
        self.assertIn("later round one", next_text)
        self.assertIn("later round two", next_text)

    async def test_interval_zero_still_allows_explicit_memory_requests(self):
        session_id = str(uuid.uuid4())
        patchers, extractor, _, _ = self.common_patches(interval=0)

        with ExitStack() as stack:
            self.start_patches(stack, patchers)
            await self.run_round(session_id, "ordinary message")
            self.assertEqual(0, extractor.await_count)
            await self.run_round(session_id, "请记住：我的生日是五月九日。")

        self.assertEqual(1, extractor.await_count)

    async def test_auxiliary_or_tool_only_calls_do_not_advance_interval(self):
        session_id = str(uuid.uuid4())
        patchers, extractor, _, _ = self.common_patches(interval=1)

        with ExitStack() as stack:
            self.start_patches(stack, patchers)
            await gateway.process_memories_background(
                session_id,
                "auxiliary title",
                "title",
                "test-model",
                skip_conversation_log=True,
            )
            await gateway.process_memories_background(
                session_id,
                "",
                "tool result answer",
                "test-model",
                tool_messages=[{"role": "tool", "content": "result"}],
            )

        self.assertEqual(0, extractor.await_count)


class MemoryExtractionRestartTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        if hasattr(gateway, "_memory_extraction_pending"):
            gateway._memory_extraction_pending.clear()
        if hasattr(gateway, "_memory_extraction_locks"):
            gateway._memory_extraction_locks.clear()

    async def test_unfinished_interval_progress_survives_process_restart(self):
        """Deleting all in-process state must not lose two persisted rounds."""
        session_id = str(uuid.uuid4())
        store = PersistentConversationFake()
        extractor = AsyncMock(return_value=[])

        async def run_round(text):
            await gateway.process_memories_background(
                session_id,
                text,
                "assistant response",
                "test-model",
                request_id=str(uuid.uuid4()),
            )

        with (
            patch.object(gateway, "MEMORY_EXTRACT_ENABLED", True),
            patch.object(gateway, "MEMORY_EXTRACT_INTERVAL", 3),
            patch.object(gateway, "get_last_user_content", AsyncMock(return_value="")),
            patch.object(gateway, "save_message", store.save_message),
            patch.object(
                gateway,
                "ensure_memory_extraction_cursor",
                store.ensure_cursor,
                create=True,
            ),
            patch.object(
                gateway,
                "get_memory_extraction_messages",
                store.get_messages,
                create=True,
            ),
            patch.object(
                gateway,
                "save_memory_extraction_cursor",
                store.save_cursor,
                create=True,
            ),
            patch.object(gateway, "get_recent_memories", AsyncMock(return_value=[])),
            patch.object(gateway, "extract_memories", extractor),
            patch.object(gateway, "save_memory", AsyncMock()),
            patch.object(gateway, "get_all_memories_count", AsyncMock(return_value=0)),
        ):
            await run_round("restart fact one")
            await run_round("restart fact two")
            self.assertEqual(0, extractor.await_count)

            # A real restart drops locks and every ordinary Python dictionary.
            if hasattr(gateway, "_memory_extraction_pending"):
                gateway._memory_extraction_pending.clear()
            gateway._memory_extraction_locks.clear()

            await run_round("restart fact three")

        self.assertEqual(1, extractor.await_count)
        batch_text = "\n".join(
            message["content"] for message in extractor.await_args.args[0]
        )
        self.assertIn("restart fact one", batch_text)
        self.assertIn("restart fact two", batch_text)
        self.assertIn("restart fact three", batch_text)
        self.assertEqual(6, store.cursors[session_id])

    async def test_explicit_request_after_restart_consumes_persisted_pending_rounds(self):
        session_id = str(uuid.uuid4())
        store = PersistentConversationFake()
        extractor = AsyncMock(return_value=[])

        async def run_round(text):
            await gateway.process_memories_background(
                session_id,
                text,
                "assistant response",
                "test-model",
                request_id=str(uuid.uuid4()),
            )

        with (
            patch.object(gateway, "MEMORY_EXTRACT_ENABLED", True),
            patch.object(gateway, "MEMORY_EXTRACT_INTERVAL", 5),
            patch.object(gateway, "get_last_user_content", AsyncMock(return_value="")),
            patch.object(gateway, "save_message", store.save_message),
            patch.object(gateway, "ensure_memory_extraction_cursor", store.ensure_cursor),
            patch.object(gateway, "get_memory_extraction_messages", store.get_messages),
            patch.object(gateway, "save_memory_extraction_cursor", store.save_cursor),
            patch.object(gateway, "get_recent_memories", AsyncMock(return_value=[])),
            patch.object(gateway, "extract_memories", extractor),
            patch.object(gateway, "save_memory", AsyncMock()),
            patch.object(gateway, "get_all_memories_count", AsyncMock(return_value=0)),
        ):
            await run_round("ordinary persisted fact")
            self.assertEqual(0, extractor.await_count)

            gateway._memory_extraction_locks.clear()
            await run_round("请记住：显式请求在重启后也要立即处理。")

        self.assertEqual(1, extractor.await_count)
        batch_text = "\n".join(
            message["content"] for message in extractor.await_args.args[0]
        )
        self.assertIn("ordinary persisted fact", batch_text)
        self.assertIn("显式请求在重启后也要立即处理", batch_text)
        self.assertEqual(4, store.cursors[session_id])

    async def test_first_cursor_baselines_legacy_history_without_reextracting_it(self):
        session_id = str(uuid.uuid4())
        store = PersistentConversationFake()
        await store.save_message(session_id, "user", "legacy user fact")
        await store.save_message(session_id, "assistant", "legacy answer")
        extractor = AsyncMock(return_value=[])

        async def run_round(text):
            await gateway.process_memories_background(
                session_id,
                text,
                "assistant response",
                "test-model",
                request_id=str(uuid.uuid4()),
            )

        with (
            patch.object(gateway, "MEMORY_EXTRACT_ENABLED", True),
            patch.object(gateway, "MEMORY_EXTRACT_INTERVAL", 2),
            patch.object(gateway, "get_last_user_content", AsyncMock(return_value="")),
            patch.object(gateway, "save_message", store.save_message),
            patch.object(gateway, "ensure_memory_extraction_cursor", store.ensure_cursor),
            patch.object(gateway, "get_memory_extraction_messages", store.get_messages),
            patch.object(gateway, "save_memory_extraction_cursor", store.save_cursor),
            patch.object(gateway, "get_recent_memories", AsyncMock(return_value=[])),
            patch.object(gateway, "extract_memories", extractor),
            patch.object(gateway, "save_memory", AsyncMock()),
            patch.object(gateway, "get_all_memories_count", AsyncMock(return_value=0)),
        ):
            await run_round("new fact one")
            self.assertEqual(0, extractor.await_count)
            await run_round("new fact two")

        self.assertEqual(1, extractor.await_count)
        batch_text = "\n".join(
            message["content"] for message in extractor.await_args.args[0]
        )
        self.assertNotIn("legacy user fact", batch_text)
        self.assertIn("new fact one", batch_text)
        self.assertIn("new fact two", batch_text)


if __name__ == "__main__":
    unittest.main()
