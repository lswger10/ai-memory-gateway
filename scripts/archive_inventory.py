"""Create a no-guess structural inventory for an untrusted export JSON file."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from archive_import import inventory_archive


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True)
    parser.add_argument("--source-system", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    raw = json.loads(Path(args.source).read_text(encoding="utf-8"))
    result = inventory_archive(raw, source_system=args.source_system)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result.to_dict(), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

