"""Print non-sensitive authorized candidate IDs/scopes for test-environment audits."""

import argparse
import asyncio
import os
from pathlib import Path
import re
import sys
from urllib.parse import parse_qs, urlsplit

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from database import search_authorized_memories
from memory_policy import build_retrieval_policy


_TEST_SCHEMA_RE = re.compile(r"group_e2e_[0-9a-f]{8,32}")


def validate_test_schema_target(
    dsn: str,
    expected_schema: str,
    *,
    opt_in: bool,
) -> str:
    if not opt_in:
        raise ValueError("candidate audit requires explicit opt-in")
    if not _TEST_SCHEMA_RE.fullmatch(expected_schema):
        raise ValueError("candidate audit requires an isolated Group test schema")
    if not isinstance(dsn, str) or not dsn.strip():
        raise ValueError("candidate audit DATABASE_URL is missing")
    options = parse_qs(urlsplit(dsn).query).get("options", [])
    schemas: list[str] = []
    for option in options:
        schemas.extend(
            re.findall(r"(?:^|\s)-c\s*search_path=([A-Za-z0-9_]+)(?:\s|$)", option)
        )
    if schemas != [expected_schema]:
        raise ValueError("candidate audit DSN does not target expected schema")
    return expected_schema


async def _run(args) -> None:
    policy = build_retrieval_policy(
        args.actor_id,
        args.room_id,
        frozenset(part.strip() for part in args.present.split(",") if part.strip()),
    )
    result = await search_authorized_memories(args.query, policy, args.limit)
    for memory in result.memories:
        print(f"{memory['id']}\t{memory['scope']}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--actor-id", required=True, choices=("jiao", "laoke"))
    parser.add_argument("--room-id", required=True)
    parser.add_argument("--present", required=True)
    parser.add_argument("--query", required=True)
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--expected-schema", required=True)
    args = parser.parse_args()
    try:
        validate_test_schema_target(
            os.getenv("DATABASE_URL", ""),
            args.expected_schema,
            opt_in=os.getenv("GROUP_ACL_AUDIT_ALLOW_TEST_DATABASE", "").lower()
            in {"1", "true", "yes", "on"},
        )
    except ValueError as exc:
        parser.error(str(exc))
    asyncio.run(_run(args))


if __name__ == "__main__":
    main()
