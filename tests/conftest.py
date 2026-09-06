"""Opt-in PostgreSQL isolation shared by the stabilization regression tests."""
import os
import uuid

import asyncpg
import pytest


@pytest.fixture
async def isolated_postgres():
    # Reuse the existing explicit database authorization gate. Never read the
    # application's DATABASE_URL or touch its default schema.
    dsn = os.environ.get("GATEWAY_MODEL_SETTINGS_TEST_DSN")
    if not dsn or os.environ.get("GATEWAY_TEST_POSTGRES_APPROVED") != "true":
        pytest.skip("explicit isolated test PostgreSQL authorization/DSN required")
    schema = "group_e2e_" + uuid.uuid4().hex
    admin = await asyncpg.connect(dsn)
    pool = None
    try:
        await admin.execute(f'CREATE SCHEMA "{schema}"')
        pool = await asyncpg.create_pool(
            dsn, min_size=1, max_size=5, server_settings={"search_path": schema})
        async with pool.acquire() as conn:
            assert await conn.fetchval("SELECT current_schema()") == schema
        yield pool
    finally:
        if pool is not None:
            await pool.close()
        await admin.execute(f'DROP SCHEMA "{schema}" CASCADE')
        assert not await admin.fetchval(
            "SELECT EXISTS(SELECT 1 FROM pg_namespace WHERE nspname=$1)", schema)
        await admin.close()
