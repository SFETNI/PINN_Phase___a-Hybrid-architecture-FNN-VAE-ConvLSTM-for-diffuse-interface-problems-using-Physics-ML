"""The identity ledger must agree with distributed bytes, not with prose.

A digest recorded in a document is only evidence if something recomputes it from
the artifact. These tests hash the staged weight-artifact bytes and compare them
with every place the archive states an identity: the ledger, the lineage record,
the model-reconstruction config, the prose ledger, the ``SHA256SUMS`` file, and the
claim map. A value that is merely self-consistent across the documents, and
disagrees with the bytes, fails here.

They also police the distinction the whole packaging rests on. Two digests exist per
model: the accepted parent checkpoint, which is an identity of record and is not
distributed, and the derived public replay weights, which are. The two must never be
equal and must never be quoted in each other's place.
"""

from __future__ import annotations

import json
from pathlib import Path
import re

import pytest
import yaml

from pinn_phase.io.artifacts import sha256_file

ROOT = Path(__file__).resolve().parents[1]


def _release_payload() -> list[Path]:
    """The manifest-defined archive payload, via the shipped release policy."""
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "public_release_files", ROOT / "scripts/release_files.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.manifest_payload(ROOT)


LEDGER = json.loads((ROOT / "docs/ARTIFACT_IDENTITY_LEDGER.json").read_text(encoding="utf-8"))
ARTIFACTS = LEDGER["artifacts"]
IDS = [entry["name"] for entry in ARTIFACTS]
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


@pytest.fixture(scope="module")
def prose_ledger() -> str:
    return (ROOT / "docs/COMPLETED_EVIDENCE_LEDGER.md").read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def checksums() -> dict[str, str]:
    text = (ROOT / "checkpoints/SHA256SUMS").read_text(encoding="utf-8")
    rows = (line.split() for line in text.splitlines() if line.strip())
    return {name: digest for digest, name in rows}


@pytest.mark.parametrize("entry", ARTIFACTS, ids=IDS)
def test_ledger_digest_matches_distributed_bytes(entry: dict) -> None:
    """The authority is the file, not the record."""
    path = ROOT / entry["public_weights_path"]
    assert path.is_file(), f"{entry['name']}: ledger lists weights that are not distributed"
    assert not path.is_symlink()
    actual = sha256_file(path)
    assert actual == entry["public_weights_sha256"], (
        f"{entry['name']}: ledger records {entry['public_weights_sha256']} but the staged "
        f"bytes hash to {actual}"
    )
    assert path.stat().st_size == entry["public_weights_bytes"]


@pytest.mark.parametrize("entry", ARTIFACTS, ids=IDS)
def test_lineage_record_matches_the_ledger_and_the_bytes(entry: dict) -> None:
    lineage_path = ROOT / entry["weight_lineage_path"]
    assert lineage_path.is_file(), f"{entry['name']}: lineage record is not distributed"
    assert sha256_file(lineage_path) == entry["weight_lineage_sha256"]
    lineage = json.loads(lineage_path.read_text(encoding="utf-8"))
    assert lineage["public_weights"]["sha256"] == sha256_file(ROOT / entry["public_weights_path"])
    assert lineage["accepted_parent_checkpoint"]["sha256"] == entry["parent_checkpoint_sha256"]
    assert lineage["accepted_parent_checkpoint"]["distributed"] is False
    assert lineage["model_state_fingerprint_sha256"] == entry["model_state_fingerprint_sha256"]
    assert lineage["tensor_count"] == entry["tensor_count"]
    assert lineage["parameter_count"] == entry["parameter_count"]


@pytest.mark.parametrize("entry", ARTIFACTS, ids=IDS)
def test_parent_and_derived_identities_are_distinct(entry: dict) -> None:
    """A packaging digest is not the identity of the accepted scientific artifact."""
    parent = entry["parent_checkpoint_sha256"]
    derived = entry["public_weights_sha256"]
    assert SHA256_RE.match(parent) and SHA256_RE.match(derived)
    assert parent != derived, (
        f"{entry['name']}: the parent and derived digests are equal, which cannot be true "
        "for a repackaged artifact and would mean one is mislabelled"
    )
    assert entry["parent_checkpoint_distributed"] is False
    assert entry["public_weights_distributed"] is True


def test_no_accepted_parent_checkpoint_is_distributed() -> None:
    """The archive must not carry a raw parent checkpoint anywhere."""
    parents = {entry["parent_checkpoint_sha256"] for entry in ARTIFACTS}
    # Asked of the release payload, not of the working tree: a local training run
    # writes checkpoints under artifacts/, and those are the reader's files, not ours.
    offenders = [
        relative
        for relative in _release_payload()
        if relative.suffix.lower() in {".pt", ".pth", ".ckpt", ".bin", ".pkl", ".pickle"}
    ]
    assert not offenders, f"serialized model payloads are distributed: {offenders}"
    staged = {sha256_file(path) for path in (ROOT / "checkpoints").iterdir() if path.is_file()}
    assert not (staged & parents), "an accepted parent checkpoint is distributed verbatim"


@pytest.mark.parametrize("entry", ARTIFACTS, ids=IDS)
def test_model_config_digest_matches_distributed_bytes(entry: dict) -> None:
    config = yaml.safe_load(
        (ROOT / entry["model_reconstruction_config"]).read_text(encoding="utf-8")
    )
    assert config["public_weights"] == entry["public_weights_path"]
    assert config["weight_lineage"] == entry["weight_lineage_path"]
    assert config["parent_checkpoint_sha256"] == entry["parent_checkpoint_sha256"]
    assert config["parent_checkpoint_distributed"] is False
    assert config["architecture_family"] == entry["architecture_family"]
    assert config["class_name"] == entry["class_name"]


@pytest.mark.parametrize("entry", ARTIFACTS, ids=IDS)
def test_checksums_file_matches_distributed_bytes(entry: dict, checksums: dict[str, str]) -> None:
    for key in ("public_weights_path", "weight_lineage_path"):
        name = Path(entry[key]).name
        assert name in checksums, f"{name} is absent from checkpoints/SHA256SUMS"
        assert checksums[name] == sha256_file(ROOT / entry[key])


@pytest.mark.parametrize("entry", ARTIFACTS, ids=IDS)
def test_prose_ledger_states_both_identities(entry: dict, prose_ledger: str) -> None:
    """The narrative ledger must quote the digest of the bytes we ship, and the parent's."""
    derived = sha256_file(ROOT / entry["public_weights_path"])
    assert derived in prose_ledger, (
        f"{entry['name']}: COMPLETED_EVIDENCE_LEDGER.md does not state the digest of the "
        f"distributed public replay weights ({derived})"
    )
    assert entry["parent_checkpoint_sha256"] in prose_ledger, (
        f"{entry['name']}: the accepted parent identity is missing from the prose ledger"
    )


@pytest.mark.parametrize("entry", ARTIFACTS, ids=IDS)
def test_distributed_training_config_digest_matches_bytes(entry: dict) -> None:
    """A config declared distributed must be present and hash to its recorded identity."""
    if not entry.get("training_config_distributed"):
        pytest.skip("training configuration is an identity only, not a distributed file")
    matches = [
        path
        for path in (ROOT / "configs").rglob("*.yaml")
        if sha256_file(path) == entry["training_config_sha256"]
    ]
    assert matches, (
        f"{entry['name']}: training_config_distributed is true but no distributed "
        f"configuration hashes to {entry['training_config_sha256']}"
    )


def test_every_distributed_weight_artifact_is_in_the_ledger() -> None:
    """No weight artifact may ride along unrecorded."""
    distributed = {path.name for path in (ROOT / "checkpoints").glob("*.weights.npz")}
    recorded = {Path(entry["public_weights_path"]).name for entry in ARTIFACTS}
    assert distributed == recorded, (
        f"unrecorded weight artifacts: {sorted(distributed - recorded)}; "
        f"ledger entries with no file: {sorted(recorded - distributed)}"
    )


def test_undistributed_configs_are_identities_only() -> None:
    """A digest we cannot check must be declared as such, and must be well formed."""
    for entry in ARTIFACTS:
        digest = entry.get("training_config_sha256")
        if entry.get("training_config_distributed"):
            continue
        if digest is None:
            assert "training_config_note" in entry, (
                f"{entry['name']}: a null configuration digest needs an explanation"
            )
            continue
        assert SHA256_RE.match(digest), f"{entry['name']}: malformed configuration digest"


def test_a_forged_ledger_value_is_rejected(tmp_path: Path) -> None:
    """Guard the guard: a digest that agrees only with prose must fail.

    This mutates a copy of the ledger the way the corrected Arm G defect looked --
    a real digest prefix followed by a fabricated tail -- and asserts the byte
    comparison catches it.
    """
    entry = dict(ARTIFACTS[0])
    real = entry["public_weights_sha256"]
    forged = real[:8] + ("0" * 56 if not real[8:].startswith("0") else "1" * 56)
    assert forged != real and forged[:8] == real[:8]
    entry["public_weights_sha256"] = forged
    actual = sha256_file(ROOT / entry["public_weights_path"])
    assert actual != entry["public_weights_sha256"], (
        "a fabricated digest sharing a real prefix was not distinguished from the bytes"
    )
