import inspect
import unittest
import database


class LongTermMemoryContractTests(unittest.IsolatedAsyncioTestCase):
    def test_memory_storage_accepts_provenance_metadata(self):
        parameters = inspect.signature(database.save_memory).parameters
        self.assertIn("source_session", parameters)
        self.assertIn("provenance", parameters)

if __name__ == "__main__":
    unittest.main()
