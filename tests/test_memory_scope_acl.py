import re
import subprocess
import sys
import unittest
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from unittest.mock import AsyncMock, patch

import database


ROWS = [
    {"id": 1, "content": "secret jiao", "scope": "weiwei-jiao", "confidential": False, "status": "active", "importance": 7, "embedding_json": "[1,0]", "created_at": datetime.now(timezone.utc), "score": 1.0, "hit_count": 1},
    {"id": 2, "content": "secret laoke", "scope": "weiwei-laoke", "confidential": False, "status": "active", "importance": 7, "embedding_json": "[1,0]", "created_at": datetime.now(timezone.utc), "score": 1.0, "hit_count": 1},
    {"id": 3, "content": "secret peers", "scope": "jiao-laoke", "confidential": False, "status": "active", "importance": 7, "embedding_json": "[1,0]", "created_at": datetime.now(timezone.utc), "score": 1.0, "hit_count": 1},
    {"id": 4, "content": "secret group", "scope": "group", "confidential": False, "status": "active", "importance": 7, "embedding_json": "[1,0]", "created_at": datetime.now(timezone.utc), "score": 1.0, "hit_count": 1},
    {"id": 5, "content": "secret confidential", "scope": "weiwei-jiao", "confidential": True, "status": "active", "importance": 9, "embedding_json": "[1,0]", "created_at": datetime.now(timezone.utc), "score": 1.0, "hit_count": 1},
    {"id": 6, "content": "secret legacy", "scope": "legacy_unscoped", "confidential": False, "status": "active", "importance": 8, "embedding_json": "[1,0]", "created_at": datetime.now(timezone.utc), "score": 1.0, "hit_count": 1},
]


class FakeConnection:
    def __init__(self):
        self.calls = []
        self.python_vector_row_ids = set()
        self.pgvector_row_ids = set()

    async def fetch(self, sql, *params):
        self.calls.append((sql, params))
        placeholders = [int(value) for value in re.findall(r"\$(\d+)", sql)]
        if placeholders and max(placeholders) > len(params):
            raise AssertionError(
                f"SQL references ${max(placeholders)} but received {len(params)} parameters"
            )
        if "scope = 'legacy_unscoped'" in sql:
            selected = [row for row in ROWS if row["scope"] == "legacy_unscoped"]
        else:
            scope_args = [set(value) for value in params if isinstance(value, (list, tuple))]
            allowed = scope_args[0] if scope_args else set()
            confidential = scope_args[1] if len(scope_args) > 1 else set()
            selected = [
                row for row in ROWS
                if row["scope"] in allowed
                and row["status"] == "active"
                and (not row["confidential"] or row["scope"] in confidential)
            ]
        if "embedding_json" in sql:
            self.python_vector_row_ids.update(row["id"] for row in selected)
        if "embedding <=>" in sql:
            self.pgvector_row_ids.update(row["id"] for row in selected)
        return [dict(row) for row in selected]

    async def execute(self, *_args):
        return None


class FakePool:
    def __init__(self, connection):
        self.connection = connection

    @asynccontextmanager
    async def acquire(self):
        yield self.connection


class MemoryScopeAclTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.connection = FakeConnection()
        self.pool = FakePool(self.connection)

    async def _search(self, actor, room, present):
        from memory_policy import build_retrieval_policy

        policy = build_retrieval_policy(actor, room, frozenset(present))
        with (
            patch.object(database, "get_pool", AsyncMock(return_value=self.pool)),
            patch.object(database, "MEMORY_VECTOR_ENABLED", False),
            patch.object(database, "MIN_SCORE_THRESHOLD", 0),
        ):
            return await database.search_authorized_memories("secret", policy, limit=20)

    async def test_jiao_group_candidates_exclude_cross_pair_confidential_and_legacy(self):
        result = await self._search(
            "jiao", "room_group_home", {"weiwei", "jiao", "laoke"}
        )
        self.assertEqual({1, 3, 4}, set(result.candidate_ids))
        self.assertFalse({2, 5, 6} & set(result.candidate_ids))
        self.assertEqual({1, 3, 4}, set(result.audit.sql_candidate_ids))
        self.assertEqual({1, 3, 4}, set(result.audit.rerank_candidate_ids))

    async def test_laoke_group_candidates_exclude_weiwei_jiao(self):
        result = await self._search(
            "laoke", "room_group_home", {"weiwei", "jiao", "laoke"}
        )
        self.assertEqual({2, 3, 4}, set(result.candidate_ids))
        self.assertFalse({1, 5, 6} & set(result.candidate_ids))

    async def test_private_room_can_read_matching_confidential_but_not_other_pair(self):
        result = await self._search(
            "jiao", "room_weiwei_jiao", {"weiwei", "jiao"}
        )
        self.assertEqual({1, 4, 5}, set(result.candidate_ids))
        self.assertNotIn(2, result.candidate_ids)

    async def test_legacy_search_is_quarantined_from_new_namespaces(self):
        with (
            patch.object(database, "get_pool", AsyncMock(return_value=self.pool)),
            patch.object(database, "MEMORY_VECTOR_ENABLED", False),
            patch.object(database, "MIN_SCORE_THRESHOLD", 0),
        ):
            rows = await database.search_legacy_memories("secret", limit=20)
        self.assertEqual({6}, {row["id"] for row in rows})

    async def test_python_vector_filters_before_loading_rows(self):
        from memory_policy import build_retrieval_policy

        policy = build_retrieval_policy(
            "jiao", "room_group_home", frozenset({"weiwei", "jiao", "laoke"})
        )
        with (
            patch.object(database, "get_pool", AsyncMock(return_value=self.pool)),
            patch.object(database, "MEMORY_VECTOR_ENABLED", True),
            patch.object(database, "EMBEDDING_API_KEY", "test-only"),
            patch.object(database, "HAS_PGVECTOR", False),
            patch.object(database, "compute_embedding", AsyncMock(return_value=[1.0, 0.0])),
            patch.object(database, "MIN_SCORE_THRESHOLD", 0),
        ):
            result = await database.search_authorized_memories("secret", policy, limit=20)
        self.assertEqual({1, 3, 4}, self.connection.python_vector_row_ids)
        self.assertEqual({1, 3, 4}, set(result.audit.python_vector_loaded_ids))
        self.assertFalse({2, 5, 6} & set(result.candidate_ids))

    async def test_pgvector_filters_before_distance_ordering(self):
        from memory_policy import build_retrieval_policy

        policy = build_retrieval_policy(
            "jiao", "room_group_home", frozenset({"weiwei", "jiao", "laoke"})
        )
        with (
            patch.object(database, "get_pool", AsyncMock(return_value=self.pool)),
            patch.object(database, "MEMORY_VECTOR_ENABLED", True),
            patch.object(database, "EMBEDDING_API_KEY", "test-only"),
            patch.object(database, "HAS_PGVECTOR", True),
            patch.object(database, "compute_embedding", AsyncMock(return_value=[1.0, 0.0])),
            patch.object(database, "MIN_SCORE_THRESHOLD", 0),
        ):
            result = await database.search_authorized_memories("secret", policy, limit=20)
        self.assertEqual({1, 3, 4}, self.connection.pgvector_row_ids)
        self.assertEqual({1, 3, 4}, set(result.audit.vector_candidate_ids))
        self.assertFalse({2, 5, 6} & set(result.audit.rerank_candidate_ids))

    async def test_archive_and_summary_hooks_require_policy_before_scanning(self):
        with self.assertRaises(TypeError):
            await database.search_authorized_archive_candidates("secret", None)
        with self.assertRaises(TypeError):
            await database.search_authorized_summary_candidates("secret", None)

    async def test_authorized_search_requires_a_real_policy(self):
        with self.assertRaises(TypeError):
            await database.search_authorized_memories("secret", None)

    def test_candidate_audit_cli_can_start_from_the_repository_root(self):
        completed = subprocess.run(
            [sys.executable, "scripts/group_acl_candidate_audit.py", "--help"],
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(0, completed.returncode, completed.stderr)
        self.assertIn("--actor-id", completed.stdout)
        self.assertIn("--expected-schema", completed.stdout)

    def test_candidate_audit_requires_explicit_temporary_schema_target(self):
        from scripts.group_acl_candidate_audit import validate_test_schema_target

        with self.assertRaisesRegex(ValueError, "explicit opt-in"):
            validate_test_schema_target(
                "postgresql://test@db.invalid/gateway?options=-csearch_path%3Dgroup_e2e_12345678",
                "group_e2e_12345678",
                opt_in=False,
            )
        with self.assertRaisesRegex(ValueError, "isolated Group test schema"):
            validate_test_schema_target(
                "postgresql://test@db.invalid/gateway?options=-csearch_path%3Dpublic",
                "public",
                opt_in=True,
            )

    def test_candidate_audit_dsn_must_match_exact_expected_schema(self):
        from scripts.group_acl_candidate_audit import validate_test_schema_target

        dsn = (
            "postgresql://test@db.invalid/gateway?"
            "options=-csearch_path%3Dgroup_e2e_12345678"
        )
        self.assertEqual(
            validate_test_schema_target(
                dsn,
                "group_e2e_12345678",
                opt_in=True,
            ),
            "group_e2e_12345678",
        )
        with self.assertRaisesRegex(ValueError, "does not target expected schema"):
            validate_test_schema_target(
                dsn,
                "group_e2e_87654321",
                opt_in=True,
            )


if __name__ == "__main__":
    unittest.main()
