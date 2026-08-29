#!/usr/bin/env python3
"""Verify public hashes and, optionally, the accepted source snapshot."""

from __future__ import annotations

import argparse
import csv
import hashlib
from pathlib import Path


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--accepted-source-root", type=Path)
    args = parser.parse_args()
    mapping = args.root / "docs" / "SOURCE_TRANSFORM_MAP.tsv"
    failures: list[str] = []
    with mapping.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream, delimiter="\t"))
    for row in rows:
        relative = Path(row["public_path"])
        if digest(args.root / relative) != row["public_sha256"]:
            failures.append(f"public hash mismatch: {relative}")
        accepted = row["accepted_source_sha256"]
        if args.accepted_source_root is not None and accepted != "NEW":
            source = args.accepted_source_root / relative.relative_to("src/pinn_phase")
            if not source.is_file() or digest(source) != accepted:
                failures.append(f"accepted source mismatch: {relative}")
    if failures:
        print("Source-lineage verification failed:")
        for failure in failures:
            print(f"- {failure}")
        return 1
    mode = "public and accepted source" if args.accepted_source_root else "public"
    print(f"Source-lineage verification passed: {len(rows)} entries ({mode})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
