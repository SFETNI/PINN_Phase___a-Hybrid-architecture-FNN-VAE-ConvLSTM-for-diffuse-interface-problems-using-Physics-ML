"""The derived initial fields must be honest about being derived.

Extracting an initial field and re-serializing it produces a new file with a new
digest. The danger is quoting that derived digest as though it were the identity of
the accepted scientific artifact. These tests check the three things that keep the
distinction sound:

* the derived digest matches the distributed bytes;
* the derived and parent digests are never the same value and never confused;
* the value digest — dtype, shape and contents, independent of container — is
  recorded, so the claim "this is the parent's initial field" is checkable rather
  than asserted.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from pinn_phase.io.artifacts import sha256_file
from pinn_phase.io.phi0 import load_initial_field

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = json.loads(
    (ROOT / "benchmarks/initial_conditions/manifest.json").read_text(encoding="utf-8")
)
ARTIFACTS = MANIFEST["artifacts"]
IDS = [entry["name"] for entry in ARTIFACTS]
SHA256 = __import__("re").compile(r"^[0-9a-f]{64}$")


def _value_digest(array: np.ndarray) -> str:
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode())
    digest.update(b"\0")
    digest.update(str(array.shape).encode())
    digest.update(b"\0")
    digest.update(np.ascontiguousarray(array).tobytes())
    return digest.hexdigest()


@pytest.mark.parametrize("entry", ARTIFACTS, ids=IDS)
def test_derived_digest_matches_distributed_bytes(entry: dict) -> None:
    path = ROOT / entry["derived_path"]
    assert path.is_file(), f"{entry['name']}: not distributed"
    assert sha256_file(path) == entry["derived_sha256"]
    assert path.stat().st_size == entry["derived_bytes"]


@pytest.mark.parametrize("entry", ARTIFACTS, ids=IDS)
def test_value_digest_is_recomputable_from_the_distributed_field(entry: dict) -> None:
    """The claim that this is the parent's initial field is checked, not trusted."""
    field = load_initial_field(ROOT / entry["derived_path"])
    assert _value_digest(field.phi0) == entry["value_sha256"]
    assert str(field.phi0.dtype) == entry["dtype"]
    assert list(field.phi0.shape) == entry["shape"]


@pytest.mark.parametrize("entry", ARTIFACTS, ids=IDS)
def test_derived_and_parent_identities_are_distinct(entry: dict) -> None:
    """A derived file is not the scientific artifact and must never share its digest."""
    assert SHA256.match(entry["derived_sha256"])
    assert SHA256.match(entry["parent_sha256"])
    assert entry["derived_sha256"] != entry["parent_sha256"], (
        f"{entry['name']}: derived and parent digests are equal, which cannot be true "
        "for a re-serialized archive and would mean one is mislabelled"
    )


@pytest.mark.parametrize("entry", ARTIFACTS, ids=IDS)
def test_schema_is_recorded_as_phi0_only(entry: dict) -> None:
    assert entry["schema"] == ["phi0"]
    assert entry["value_exact_against_parent"] is True


@pytest.mark.parametrize("entry", ARTIFACTS, ids=IDS)
def test_distributed_field_is_admissible(entry: dict) -> None:
    assert 0.0 <= entry["phi_min"] and entry["phi_max"] <= 1.0
    assert abs(entry["phase_sum_min"] - 1.0) < 1e-5
    assert abs(entry["phase_sum_max"] - 1.0) < 1e-5


def test_claim_map_records_only_accepted_identities_for_initial_fields() -> None:
    """The claim map has one initial-field slot, and it means the accepted parent.

    There is nowhere in that schema to say "derived", so a derived digest appearing
    in it would be read as the parent's identity. The ban stays absolute here.
    """
    derived = {entry["derived_sha256"] for entry in ARTIFACTS}
    text = (ROOT / "docs/CLAIM_TO_ARTIFACT_MAP.json").read_text(encoding="utf-8")
    leaked = sorted(digest for digest in derived if digest in text)
    assert not leaked, (
        f"docs/CLAIM_TO_ARTIFACT_MAP.json quotes derived initial-field digests "
        f"{leaked}; that record holds accepted scientific artifact identities, and a "
        "packaging digest does not belong there"
    )


def test_ledger_quotes_a_derived_digest_only_as_the_distributed_file() -> None:
    """In prose, a derived digest is allowed -- but only while saying what it is.

    The evidence ledger names the derived public replay weights beside every accepted
    parent checkpoint, so a reader can tell which digest identifies the science and
    which identifies the download. Initial fields are recorded the same way. What must
    never happen is the packaging digest standing alone in the slot where the reader
    looks for the accepted artifact, so each occurrence has to carry the path it names.
    """
    ledger = (ROOT / "docs/COMPLETED_EVIDENCE_LEDGER.md").read_text(encoding="utf-8")
    for entry in ARTIFACTS:
        digest = entry["derived_sha256"]
        start = 0
        while (found := ledger.find(digest, start)) != -1:
            bullet_start = ledger.rfind("\n- ", 0, found)
            bullet_start = 0 if bullet_start < 0 else bullet_start + 1
            bullet_end = ledger.find("\n- ", found)
            bullet = ledger[bullet_start : len(ledger) if bullet_end < 0 else bullet_end]
            assert entry["derived_path"] in bullet, (
                f"{entry['name']}: the derived digest is quoted without naming "
                f"{entry['derived_path']}, so it reads as an artifact identity"
            )
            assert "not the identity of the accepted parent" in bullet, (
                f"{entry['name']}: the derived digest is quoted without saying it is "
                "a packaging digest rather than the accepted parent's identity"
            )
            start = found + len(digest)


def test_every_parent_is_a_distinct_accepted_artifact() -> None:
    parents = [entry["parent_sha256"] for entry in ARTIFACTS]
    assert len(parents) == len(set(parents)), "two derived fields claim the same parent"


def test_manifest_covers_every_distributed_initial_field() -> None:
    on_disk = {path.name for path in (ROOT / "benchmarks/initial_conditions").glob("*.npz")}
    recorded = {Path(entry["derived_path"]).name for entry in ARTIFACTS}
    assert on_disk == recorded, (
        f"unrecorded fields: {sorted(on_disk - recorded)}; "
        f"recorded but absent: {sorted(recorded - on_disk)}"
    )
