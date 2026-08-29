"""A digest and a path each have to say which thing they identify.

Two ways of writing a true digest down still mislead a reader:

* quoting the digest of an accepted parent container beside a benchmark, with no
  statement that the parent is not distributed and that what ships is a derived file
  with its own digest -- the reader reasonably tries to find a file with that hash;
* writing a repo-relative path in backticks for a record that lives in some earlier
  tree, which reads as a file the reader can open here.

Both are conflations of identity with availability. The evidence ledger states the
distinction carefully for checkpoints; these tests require the same care for initial
fields and for source-record paths.
"""

from __future__ import annotations

import hashlib
import json
import re
from functools import lru_cache
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
LEDGER_PATH = ROOT / "docs/COMPLETED_EVIDENCE_LEDGER.md"
LEDGER = LEDGER_PATH.read_text(encoding="utf-8")

#: A backticked token containing a directory separator and an extension. This is what
#: a reader sees as "a file in this archive".
PATH_SHAPED = re.compile(r"`([A-Za-z0-9_][A-Za-z0-9_./-]*/[A-Za-z0-9_./-]*\.[A-Za-z0-9]+)`")

#: Initial fields whose accepted parent container is quoted in the ledger.
LEDGER_INITIAL_FIELDS = ("n8_64_cube", "n16_96_cube")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@lru_cache(maxsize=1)
def _manifest_paths() -> frozenset[str]:
    lines = (ROOT / "MANIFEST.sha256").read_text(encoding="utf-8").splitlines()
    return frozenset(line.split("  ", 1)[1] for line in lines if line.strip())


@lru_cache(maxsize=1)
def _distributed_digests() -> frozenset[str]:
    """Every distributed file's own byte digest, hashed once for the whole module."""
    return frozenset(_sha256(ROOT / path) for path in _manifest_paths())


def _bullet_containing(offset: int) -> str:
    """The list item the character at *offset* belongs to."""
    start = LEDGER.rfind("\n- ", 0, offset)
    start = 0 if start < 0 else start + 1
    end = LEDGER.find("\n- ", offset)
    if end < 0:
        end = len(LEDGER)
    return LEDGER[start:end]


def _initial_field_records() -> dict[str, dict]:
    manifest = json.loads(
        (ROOT / "benchmarks/initial_conditions/manifest.json").read_text(encoding="utf-8")
    )
    return {record["name"]: record for record in manifest["artifacts"]}


def test_every_path_shaped_token_in_the_ledger_is_shipped_or_labelled() -> None:
    """A path presented as a file must exist and be manifested, or say it does not."""
    shipped = _manifest_paths()
    problems = []
    for match in PATH_SHAPED.finditer(LEDGER):
        token = match.group(1)
        exists = (ROOT / token).is_file()
        if exists:
            if token not in shipped:
                problems.append(f"{token}: present but absent from MANIFEST.sha256")
            continue
        bullet = _bullet_containing(match.start())
        if "non-distributed" not in bullet.lower():
            problems.append(
                f"{token}: does not exist in this archive and is not labelled "
                "as a non-distributed path"
            )
    assert not problems, "\n".join(problems)


def test_absent_candidate_paths_are_not_links() -> None:
    """A path that cannot be opened here must not be written as a link."""
    for match in PATH_SHAPED.finditer(LEDGER):
        token = match.group(1)
        if (ROOT / token).is_file():
            continue
        assert f"]({token})" not in LEDGER, f"{token} is absent but rendered as a link"
        assert f"<{token}>" not in LEDGER, f"{token} is absent but rendered as a link"


@pytest.mark.parametrize("name", LEDGER_INITIAL_FIELDS)
def test_parent_initial_field_digest_is_labelled_and_names_what_ships(name: str) -> None:
    record = _initial_field_records()[name]
    parent = record["parent_sha256"]
    assert parent in LEDGER, f"{name}: parent digest is not quoted in the ledger"

    bullet = _bullet_containing(LEDGER.index(parent))
    assert "not distributed" in bullet.lower(), (
        f"{name}: the parent container digest is quoted without saying it is not "
        "distributed"
    )

    derived_bullet = _bullet_containing(LEDGER.index(record["derived_sha256"]))
    assert record["derived_path"] in derived_bullet, (
        f"{name}: the derived digest is quoted without naming the distributed file"
    )
    assert record["value_sha256"] in LEDGER, (
        f"{name}: the container-independent value digest is not recorded"
    )


@pytest.mark.parametrize("name", sorted(_initial_field_records()))
def test_parent_derived_and_value_digests_stay_distinct(name: str) -> None:
    """Three different questions, three different answers -- never interchangeable."""
    record = _initial_field_records()[name]
    triple = {
        "parent": record["parent_sha256"],
        "derived": record["derived_sha256"],
        "value": record["value_sha256"],
    }
    assert len(set(triple.values())) == 3, f"{name}: digests collide: {triple}"

    shipped = ROOT / record["derived_path"]
    assert record["derived_path"] in _manifest_paths()
    assert _sha256(shipped) == record["derived_sha256"], (
        f"{name}: derived digest is not the digest of the distributed file"
    )

    assert record["parent_sha256"] not in _distributed_digests(), (
        f"{name}: the accepted parent container is distributed, which contradicts "
        "the ledger"
    )
    assert record["value_sha256"] not in _distributed_digests(), (
        f"{name}: a value digest is being presented as a file digest"
    )
