import unittest
import uuid
from contextlib import ExitStack
from unittest.mock import AsyncMock, patch

import main as gateway
import memory_extractor


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
        extractor = AsyncMock(return_value=list(extracted or []))
        saver = AsyncMock()
        patchers = (
            patch.object(gateway, "MEMORY_EXTRACT_ENABLED", True),
            patch.object(gateway, "MEMORY_EXTRACT_INTERVAL", interval),
            patch.object(gateway, "get_last_user_content", AsyncMock(return_value="")),
            patch.object(gateway, "save_message", AsyncMock()),
            patch.object(gateway, "get_recent_memories", AsyncMock(return_value=[])),
            patch.object(gateway, "extract_memories", extractor),
            patch.object(gateway, "save_memory", saver),
            patch.object(gateway, "get_all_memories_count", AsyncMock(return_value=1)),
        )
        return patchers, extractor, saver

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
        patchers, extractor, _ = self.common_patches(interval=2)

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
        patchers, extractor, saver = self.common_patches(
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
        patchers, extractor, _ = self.common_patches(interval=2)

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
        patchers, extractor, _ = self.common_patches(interval=0)

        with ExitStack() as stack:
            self.start_patches(stack, patchers)
            await self.run_round(session_id, "ordinary message")
            self.assertEqual(0, extractor.await_count)
            await self.run_round(session_id, "请记住：我的生日是五月九日。")

        self.assertEqual(1, extractor.await_count)

    async def test_auxiliary_or_tool_only_calls_do_not_advance_interval(self):
        session_id = str(uuid.uuid4())
        patchers, extractor, _ = self.common_patches(interval=1)

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


if __name__ == "__main__":
    unittest.main()
