"""Print non-sensitive authorized candidate IDs/scopes for test-environment audits."""

import argparse
import asyncio
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from database import search_authorized_memories
from memory_policy import build_retrieval_policy


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
    asyncio.run(_run(parser.parse_args()))


if __name__ == "__main__":
    main()
