import unittest
from unittest.mock import patch


class RecordingConnection:
    def __init__(self):
        self.statements = []

    async def execute(self, statement):
        self.statements.append(statement)


class ScopedMemorySchemaTests(unittest.IsolatedAsyncioTestCase):
    async def test_existing_memory_is_quarantined_without_inference(self):
        from memory_policy import quarantine_legacy_memory_row

        migrated = quarantine_legacy_memory_row(
            {
                "id": 7,
                "content": "legacy fact",
                "provider": "unknown",
                "model": "unknown",
                "provenance": None,
            }
        )
        self.assertEqual("legacy_unscoped", migrated["scope"])
        self.assertIsNone(migrated["memory_type"])
        self.assertIsNone(migrated["perspective"])
        self.assertIsNone(migrated["source_kind"])
        self.assertEqual({}, migrated["provenance"])

    async def test_additive_schema_separates_identity_and_relationship_memory(self):
        from database import apply_scoped_memory_schema

        connection = RecordingConnection()
        await apply_scoped_memory_schema(connection)
        sql = "\n".join(connection.statements)
        self.assertIn("ADD COLUMN IF NOT EXISTS scope", sql)
        self.assertIn("legacy_unscoped", sql)
        self.assertIn("CREATE TABLE IF NOT EXISTS actor_identity_profiles", sql)
        self.assertNotIn("provider", sql.lower())
        self.assertNotIn("model", sql.lower())

    async def test_scoped_schema_can_run_before_optional_legacy_columns_exist(self):
        from database import apply_scoped_memory_schema

        connection = RecordingConnection()
        await apply_scoped_memory_schema(connection)
        sql = "\n".join(connection.statements)
        self.assertNotIn("WHERE is_active", sql)

    def test_typed_scoped_writes_require_orthogonal_dimensions(self):
        from memory_policy import MemoryScope, MemoryType, MemoryWrite, Perspective, SourceKind

        write = MemoryWrite(
            content="shared observable fact",
            scope=MemoryScope.GROUP,
            memory_type=MemoryType.FACT,
            perspective=Perspective.SHARED,
            confidential=False,
            source_kind=SourceKind.SYNTHETIC_TEST,
        )
        self.assertEqual("group", write.scope.value)

    def test_group_memory_flags_default_closed(self):
        from memory_policy import group_memory_features_from_env

        with patch.dict("os.environ", {}, clear=True):
            self.assertEqual(
                {
                    "group_memory": False,
                    "agent_candidates": False,
                    "burst_extraction": False,
                },
                group_memory_features_from_env(),
            )


if __name__ == "__main__":
    unittest.main()
