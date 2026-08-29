"""The public weight artifacts must be strict, deterministic and honestly labelled.

Three things are checked here.

*Determinism* — writing the same model state twice produces byte-identical files,
and every distributed artifact rebuilds to its own recorded digest, so a reviewer
can regenerate the bytes rather than trust them.

*Refusal* — the loader is handed hostile archives one property at a time: a missing
tensor, an extra one, a renamed one, a re-typed one, a reshaped one, a truncated
one, a non-finite value, an object array, a string array, a re-compressed member, a
tampered lineage. Each must be refused, and the schema refusals must happen before
any array is decoded.

*Labelling* — the derived digest and the parent digest must never be equal, never be
swapped, and no accepted parent checkpoint may be distributed.
"""

from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path
import zipfile

import numpy as np
import pytest
import torch

from pinn_phase.io.artifacts import sha256_file
from pinn_phase.io.public_weights import (
    FIXED_MEMBER_DATE,
    PublicWeightsError,
    as_torch_state_dict,
    load_public_weights,
    load_weight_lineage,
    model_state_fingerprint,
    tensor_records,
    tensor_value_sha256,
    write_public_weights,
)

ROOT = Path(__file__).resolve().parents[1]
LEDGER = json.loads((ROOT / "docs/ARTIFACT_IDENTITY_LEDGER.json").read_text(encoding="utf-8"))
ARTIFACTS = LEDGER["artifacts"]
IDS = [entry["name"] for entry in ARTIFACTS]


def _toy_state() -> dict[str, np.ndarray]:
    return {
        "blend_logit": np.asarray(np.float32(0.125)),          # zero-dimensional on purpose
        "head.weight": np.arange(6, dtype=np.float32).reshape(2, 3),
        "head.bias": np.asarray([0.5, -0.25], dtype=np.float32),
    }


@pytest.fixture()
def toy(tmp_path: Path):
    """A small artifact plus its lineage, written the way the real ones were."""
    state = _toy_state()
    path = tmp_path / "toy.weights.npz"
    digest = write_public_weights(state, path)
    records = tensor_records(state)
    document = {
        "schema": "pinn-phase-public-weight-lineage-v1",
        "benchmark": "toy",
        "artifact": "toy",
        "accepted_parent_checkpoint": {"sha256": "b" * 64, "distributed": False},
        "public_weights": {"path": str(path), "sha256": digest,
                           "bytes": path.stat().st_size},
        "model_state_fingerprint_sha256": model_state_fingerprint(state),
        "tensors": [
            {"name": r.name, "dtype": r.dtype, "shape": list(r.shape),
             "bytes": r.bytes, "value_sha256": r.value_sha256}
            for r in records
        ],
    }
    lineage_path = tmp_path / "toy.lineage.json"
    lineage_path.write_text(json.dumps(document, indent=1), encoding="utf-8")
    return state, path, lineage_path


def _rewrite(path: Path, members: dict[str, bytes]) -> None:
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_STORED) as archive:
        for name, payload in members.items():
            info = zipfile.ZipInfo(name, date_time=FIXED_MEMBER_DATE)
            info.compress_type = zipfile.ZIP_STORED
            archive.writestr(info, payload)


def _members(path: Path) -> dict[str, bytes]:
    with zipfile.ZipFile(path) as archive:
        return {name: archive.read(name) for name in archive.namelist()}


def _encode(array: np.ndarray) -> bytes:
    buffer = io.BytesIO()
    np.lib.format.write_array(buffer, array, allow_pickle=False)
    return buffer.getvalue()


def _reload(path: Path, lineage_path: Path):
    return load_public_weights(path, lineage=load_weight_lineage(lineage_path))


def _repoint(lineage_path: Path, path: Path) -> None:
    """Update only the file-level digest and size in the lineage record.

    The refusal tests below are about the schema and the per-tensor identities, so
    the file-level digest must not be what stops them -- otherwise every mutation
    would be caught by the outermost check and the inner ones would go untested.
    This models the stronger adversary: someone who edited the artifact and then
    updated the digest it is checked against.
    """
    document = json.loads(lineage_path.read_text(encoding="utf-8"))
    document["public_weights"]["sha256"] = sha256_file(path)
    document["public_weights"]["bytes"] = path.stat().st_size
    lineage_path.write_text(json.dumps(document), encoding="utf-8")


# --------------------------------------------------------------------------
# determinism and round-trip fidelity
# --------------------------------------------------------------------------

def test_writing_twice_is_byte_identical(tmp_path: Path) -> None:
    state = _toy_state()
    first = write_public_weights(state, tmp_path / "a" / "w.npz")
    second = write_public_weights(state, tmp_path / "b" / "w.npz")
    assert first == second
    assert (tmp_path / "a/w.npz").read_bytes() == (tmp_path / "b/w.npz").read_bytes()


def test_a_zero_dimensional_tensor_keeps_its_rank(toy) -> None:
    """A scalar parameter must not be silently promoted to shape (1,)."""
    state, path, lineage_path = toy
    restored = _reload(path, lineage_path)
    assert restored["blend_logit"].shape == ()
    assert restored["blend_logit"].dtype == np.float32


def test_round_trip_preserves_every_raw_value(toy) -> None:
    state, path, lineage_path = toy
    restored = _reload(path, lineage_path)
    assert set(restored) == set(state)
    for name, array in state.items():
        assert restored[name].dtype == array.dtype
        assert restored[name].shape == array.shape
        assert restored[name].tobytes() == array.tobytes()
    assert model_state_fingerprint(restored) == model_state_fingerprint(state)


@pytest.mark.parametrize("entry", ARTIFACTS, ids=IDS)
def test_distributed_artifact_rebuilds_to_its_recorded_digest(entry: dict, tmp_path: Path) -> None:
    """Reserializing what we shipped must reproduce the shipped bytes exactly."""
    weights = ROOT / entry["public_weights_path"]
    lineage = load_weight_lineage(ROOT / entry["weight_lineage_path"])
    state = load_public_weights(weights, lineage=lineage)
    rebuilt = write_public_weights(state, tmp_path / "rebuild.npz")
    assert rebuilt == entry["public_weights_sha256"]
    assert (tmp_path / "rebuild.npz").read_bytes() == weights.read_bytes()


@pytest.mark.parametrize("entry", ARTIFACTS, ids=IDS)
def test_distributed_artifact_carries_only_registered_tensors(entry: dict) -> None:
    lineage = load_weight_lineage(ROOT / entry["weight_lineage_path"])
    with zipfile.ZipFile(ROOT / entry["public_weights_path"]) as archive:
        names = archive.namelist()
        infos = archive.infolist()
    assert names == sorted(names), "members are not in canonical sorted order"
    assert [n[: -len(".npy")] for n in names] == list(lineage.tensor_names)
    for info in infos:
        assert info.compress_type == zipfile.ZIP_STORED
        assert info.date_time == FIXED_MEMBER_DATE
        assert info.external_attr >> 16 & 0o022 == 0, "member is group- or world-writable"
        assert info.comment == b""
    state = load_public_weights(ROOT / entry["public_weights_path"], lineage=lineage)
    for name, array in state.items():
        assert array.dtype.kind in "fiub", f"{name}: non-numeric dtype {array.dtype}"
        assert np.isfinite(array).all()


@pytest.mark.parametrize("entry", ARTIFACTS, ids=IDS)
def test_distributed_artifact_strict_loads_into_its_declared_class(entry: dict) -> None:
    import yaml

    from pinn_phase.models import (
        ExplicitMPFHybridRollout,
        PermEquivariantMPFRollout,
        architecture_family_from_config,
    )

    classes = {
        "PermEquivariantMPFRollout": PermEquivariantMPFRollout,
        "ExplicitMPFHybridRollout": ExplicitMPFHybridRollout,
    }
    settings = yaml.safe_load(
        (ROOT / entry["model_reconstruction_config"]).read_text(encoding="utf-8")
    )
    assert str(architecture_family_from_config(settings)) == entry["architecture_family"]
    block = {k: v for k, v in settings["model"].items()
             if k not in {"architecture", "arch_variant"}}
    physics = settings["physics"]
    model = classes[settings["class_name"]](
        model_dt=float(physics["model_dt"]), eta_px=float(physics["eta_px"]),
        mu=float(physics["mu"]), sigma=float(physics["sigma"]), **block,
    )
    lineage = load_weight_lineage(ROOT / entry["weight_lineage_path"])
    state = load_public_weights(ROOT / entry["public_weights_path"], lineage=lineage)
    model.load_state_dict(as_torch_state_dict(state), strict=True)
    assert sum(p.numel() for p in model.parameters()) == entry["parameter_count"]
    assert len(state) == entry["tensor_count"]


# --------------------------------------------------------------------------
# refusal
# --------------------------------------------------------------------------

def test_a_missing_tensor_is_refused_before_any_array_is_read(toy, monkeypatch) -> None:
    state, path, lineage_path = toy
    members = _members(path)
    members.pop("head.bias.npy")
    _rewrite(path, members)
    _repoint(lineage_path, path)

    decoded = False

    def watched(*args, **kwargs):  # noqa: ANN001
        nonlocal decoded
        decoded = True
        raise AssertionError("no array may be decoded from a rejected archive")

    monkeypatch.setattr(np.lib.format, "read_array", watched)
    with pytest.raises(PublicWeightsError, match="registered tensor key set"):
        _reload(path, lineage_path)
    assert decoded is False


def test_an_unregistered_extra_tensor_is_refused(toy) -> None:
    state, path, lineage_path = toy
    members = _members(path)
    members["smuggled.npy"] = _encode(np.zeros(3, dtype=np.float32))
    _rewrite(path, members)
    _repoint(lineage_path, path)
    with pytest.raises(PublicWeightsError, match="unregistered"):
        _reload(path, lineage_path)


def test_a_renamed_tensor_is_refused(toy) -> None:
    state, path, lineage_path = toy
    members = _members(path)
    members["head.WEIGHT.npy"] = members.pop("head.weight.npy")
    _rewrite(path, members)
    _repoint(lineage_path, path)
    with pytest.raises(PublicWeightsError, match="registered tensor key set"):
        _reload(path, lineage_path)


@pytest.mark.parametrize(
    "replacement, match",
    [
        (np.arange(6, dtype=np.float64).reshape(2, 3), "registered"),   # re-typed
        (np.arange(6, dtype=np.float32).reshape(3, 2), "shape"),        # reshaped
        (np.arange(8, dtype=np.float32).reshape(2, 4), "shape|bytes"),  # resized
        (np.full((2, 3), np.nan, dtype=np.float32), "non-finite"),
        (np.full((2, 3), np.inf, dtype=np.float32), "non-finite"),
        (np.array([["a", "b", "c"], ["d", "e", "f"]]), "dtype"),        # string array
    ],
)
def test_a_mutated_tensor_is_refused(toy, replacement, match) -> None:
    state, path, lineage_path = toy
    members = _members(path)
    members["head.weight.npy"] = _encode(replacement)
    _rewrite(path, members)
    _repoint(lineage_path, path)
    with pytest.raises(PublicWeightsError, match=match):
        _reload(path, lineage_path)


def test_a_value_edit_that_preserves_shape_and_dtype_is_refused(toy) -> None:
    """The raw-value digest is what catches a single flipped weight."""
    state, path, lineage_path = toy
    edited = state["head.weight"].copy()
    edited.flat[0] = np.float32(1.0e-7)
    members = _members(path)
    members["head.weight.npy"] = _encode(edited)
    _rewrite(path, members)
    _repoint(lineage_path, path)
    with pytest.raises(PublicWeightsError, match="raw-value digest"):
        _reload(path, lineage_path)


def test_an_object_array_cannot_be_loaded(toy) -> None:
    state, path, lineage_path = toy
    payload = io.BytesIO()
    np.lib.format.write_array(payload, np.array([{"payload": 1}], dtype=object),
                              allow_pickle=True)
    members = _members(path)
    members["head.weight.npy"] = payload.getvalue()
    _rewrite(path, members)
    _repoint(lineage_path, path)
    with pytest.raises(Exception) as info:
        _reload(path, lineage_path)
    assert "pickle" in str(info.value).lower() or isinstance(info.value, PublicWeightsError)


def test_a_digest_mismatch_is_refused_before_the_container_is_opened(toy, monkeypatch) -> None:
    state, path, lineage_path = toy
    lineage = load_weight_lineage(lineage_path)
    path.write_bytes(path.read_bytes() + b"\0")

    def forbidden(*args, **kwargs):  # noqa: ANN001
        raise AssertionError("the container must not be opened after a digest mismatch")

    monkeypatch.setattr(zipfile, "ZipFile", forbidden)
    with pytest.raises(PublicWeightsError, match="SHA-256 mismatch"):
        load_public_weights(path, lineage=lineage)


def test_a_tampered_lineage_digest_does_not_admit_tampered_weights(toy) -> None:
    """Editing both the artifact and its lineage must still fail the ledger digest."""
    state, path, lineage_path = toy
    edited = state["head.bias"].copy()
    edited[0] = np.float32(9.0)
    members = _members(path)
    members["head.bias.npy"] = _encode(edited)
    _rewrite(path, members)

    document = json.loads(lineage_path.read_text(encoding="utf-8"))
    document["public_weights"]["sha256"] = sha256_file(path)
    document["public_weights"]["bytes"] = path.stat().st_size
    for record in document["tensors"]:
        if record["name"] == "head.bias":
            record["value_sha256"] = tensor_value_sha256(edited)
    lineage_path.write_text(json.dumps(document), encoding="utf-8")

    # The artifact now agrees with its own lineage -- and the fingerprint catches it,
    # because a consistent forgery would also have to rewrite the whole-state digest.
    with pytest.raises(PublicWeightsError, match="fingerprint"):
        _reload(path, lineage_path)


def test_a_recompressed_member_changes_the_bytes_and_is_refused(toy) -> None:
    state, path, lineage_path = toy
    lineage = load_weight_lineage(lineage_path)
    members = _members(path)
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, payload in members.items():
            archive.writestr(zipfile.ZipInfo(name, date_time=FIXED_MEMBER_DATE), payload)
    with pytest.raises(PublicWeightsError, match="SHA-256 mismatch"):
        load_public_weights(path, lineage=lineage)


@pytest.mark.parametrize("field", ["comment", "extra"])
def test_zip_metadata_on_a_member_is_refused(toy, field: str) -> None:
    """The lineage says the artifact carries no free-form text; ZIP metadata is text.

    Any such edit also changes the file digest, so the digest gate would stop it
    first in practice. This checks the clause is enforced by the loader in its own
    right, with the digest repointed so the schema layer is the one under test.
    """
    state, path, lineage_path = toy
    members = _members(path)
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_STORED) as archive:
        for name, payload in members.items():
            info = zipfile.ZipInfo(name, date_time=FIXED_MEMBER_DATE)
            info.compress_type = zipfile.ZIP_STORED
            if name == "head.bias.npy":
                if field == "comment":
                    info.comment = b"an internal note"
                else:
                    info.extra = b"\x99\x99\x04\x00note"
            archive.writestr(info, payload)
    _repoint(lineage_path, path)
    with pytest.raises(PublicWeightsError, match="ZIP metadata"):
        _reload(path, lineage_path)


def test_an_archive_comment_is_refused(toy) -> None:
    state, path, lineage_path = toy
    members = _members(path)
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_STORED) as archive:
        for name, payload in members.items():
            info = zipfile.ZipInfo(name, date_time=FIXED_MEMBER_DATE)
            info.compress_type = zipfile.ZIP_STORED
            archive.writestr(info, payload)
        archive.comment = b"an internal note"
    _repoint(lineage_path, path)
    with pytest.raises(PublicWeightsError, match="comment"):
        _reload(path, lineage_path)


@pytest.mark.parametrize("entry", ARTIFACTS, ids=IDS)
def test_the_lineage_states_how_its_digests_are_computed(entry: dict) -> None:
    """A digest you can only reproduce by running our code is weaker evidence."""
    lineage = json.loads((ROOT / entry["weight_lineage_path"]).read_text(encoding="utf-8"))
    recipes = lineage["digest_recipes"]
    for key in ("value_sha256", "model_state_fingerprint_sha256",
                "public_weights_sha256", "accepted_parent_checkpoint.sha256"):
        assert recipes[key].strip(), f"{entry['name']}: no recipe for {key}"
    assert "lexicographic" in recipes["model_state_fingerprint_sha256"]


def test_the_published_fingerprint_recipe_reproduces_the_recorded_value() -> None:
    """Follow the recipe text literally, without the module, and get the same digest."""
    entry = ARTIFACTS[0]
    lineage = load_weight_lineage(ROOT / entry["weight_lineage_path"])
    state = load_public_weights(ROOT / entry["public_weights_path"], lineage=lineage)
    digest = hashlib.sha256()
    for name in sorted(state):
        array = state[name]
        digest.update(name.encode())
        digest.update(b"\0")
        digest.update(str(array.dtype).encode())
        digest.update(b"\0")
        digest.update(str(tuple(array.shape)).encode())
        digest.update(b"\0")
        digest.update(array.tobytes())
        digest.update(b"\0")
    assert digest.hexdigest() == entry["model_state_fingerprint_sha256"]


def test_a_lineage_with_the_wrong_schema_is_refused(toy) -> None:
    state, path, lineage_path = toy
    document = json.loads(lineage_path.read_text(encoding="utf-8"))
    document["schema"] = "something-else-v9"
    lineage_path.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(PublicWeightsError, match="lineage schema"):
        load_weight_lineage(lineage_path)


def test_the_writer_refuses_a_dtype_it_cannot_represent_exactly(tmp_path: Path) -> None:
    with pytest.raises(PublicWeightsError, match="cannot be represented exactly"):
        write_public_weights({"w": np.array(["text"], dtype="U8")}, tmp_path / "w.npz")


def test_torch_tensors_survive_the_round_trip_bitwise(tmp_path: Path) -> None:
    torch.manual_seed(0)
    state = {"a": torch.randn(4, 3), "b": torch.zeros(())}
    path = tmp_path / "t.npz"
    write_public_weights(state, path)
    with np.load(path, allow_pickle=False) as archive:
        restored = {name: archive[name] for name in archive.files}
    for name, tensor in state.items():
        assert restored[name].tobytes() == tensor.numpy().tobytes()
        assert restored[name].shape == tuple(tensor.shape)


@pytest.mark.parametrize("entry", ARTIFACTS, ids=IDS)
def test_lineage_derivation_block_is_self_verifying(entry: dict) -> None:
    """The record names the script and configuration that produced it, by digest.

    Both are distributed, so a reviewer can hash them and re-run the derivation
    against a parent they hold rather than take the lineage on trust.
    """
    lineage = json.loads((ROOT / entry["weight_lineage_path"]).read_text(encoding="utf-8"))
    derivation = lineage["derivation"]
    for path_key, digest_key in (("script", "script_sha256"),
                                 ("model_config", "model_config_sha256")):
        path = ROOT / derivation[path_key]
        assert path.is_file(), f"{entry['name']}: {derivation[path_key]} is not distributed"
        assert sha256_file(path) == derivation[digest_key], (
            f"{entry['name']}: {derivation[path_key]} does not hash to its recorded digest"
        )
    assert derivation["transformation"] == "packaging only"
    assert lineage["accepted_parent_checkpoint"]["distributed"] is False
