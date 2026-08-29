#!/usr/bin/env python3
"""Verify the release-candidate manifest."""

from __future__ import annotations

import sys

sys.dont_write_bytecode = True  # importing release_files must not litter the tree

import argparse
import hashlib
from pathlib import Path

from release_files import release_files


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "MANIFEST.sha256"


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--distribution-tree", action="store_true")
    args = parser.parse_args()
    failures: list[str] = []
    listed: set[Path] = set()
    for line in MANIFEST.read_text(encoding="utf-8").splitlines():
        expected, relative_text = line.split("  ", 1)
        relative = Path(relative_text)
        listed.add(relative)
        path = ROOT / relative
        if not path.is_file() or path.is_symlink() or digest(path) != expected:
            failures.append(str(relative))
    actual = release_files(ROOT, manifest_name=MANIFEST.name)
    if args.distribution_tree:
        actual.difference_update({Path("PKG-INFO"), Path("setup.cfg")})
    if actual != listed:
        missing = sorted(str(path) for path in actual - listed)
        extra = sorted(str(path) for path in listed - actual)
        failures.append(f"coverage mismatch: missing={missing}, extra={extra}")
    if failures:
        print("Manifest verification failed:")
        for failure in failures:
            print(f"- {failure}")
        return 1
    print(f"Manifest verification passed: {len(listed)} files")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
