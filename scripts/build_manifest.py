#!/usr/bin/env python3
"""Regenerate the non-self-referential release-candidate manifest."""

from __future__ import annotations

import sys

sys.dont_write_bytecode = True  # importing release_files must not litter the tree

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


paths = sorted(release_files(ROOT, manifest_name=MANIFEST.name))
content = "".join(f"{digest(ROOT / path)}  {path.as_posix()}\n" for path in paths)
MANIFEST.write_text(content, encoding="utf-8", newline="\n")
print(f"Manifest written: {len(paths)} files")
