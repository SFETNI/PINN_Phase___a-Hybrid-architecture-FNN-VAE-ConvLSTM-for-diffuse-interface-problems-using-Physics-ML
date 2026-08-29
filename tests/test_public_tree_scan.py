from __future__ import annotations

from pathlib import Path
import runpy

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCANNER = runpy.run_path(str(ROOT / "scripts" / "check_public_tree.py"))
scan_tree = SCANNER["scan_tree"]
scan_payload = SCANNER["scan_payload"]


def test_release_scan_detects_private_path_and_unsafe_loading(tmp_path: Path) -> None:
    probe = tmp_path / "probe.py"
    probe.write_text(
        "ROOT = '" + "/HO" + "ME/example/private'\n"
        "payload = torch.load(path)\n",
        encoding="utf-8",
    )
    findings = scan_tree(tmp_path)
    assert any(item.startswith("home-directory path:") for item in findings)
    assert any(item.startswith("unsafe-torch-load:") for item in findings)


def test_release_scan_detects_pickle_and_secret_file(tmp_path: Path) -> None:
    probe = tmp_path / "deserialize.py"
    probe.write_text("value = pickle.loads(blob)\n", encoding="utf-8")
    secret = tmp_path / ".env"
    secret.write_text("TOKEN=" + "gh" + "p_" + "A" * 24, encoding="utf-8")
    findings = scan_tree(tmp_path)
    assert any(item.startswith("unsafe-pickle-loader:") for item in findings)
    assert any(item.startswith("environment-secret-file:") for item in findings)
    assert any(item.startswith("GitHub token:") for item in findings)


def test_release_scan_reads_binary_content_and_nested_scanner_names(tmp_path: Path) -> None:
    image = tmp_path / "cover.png"
    image.write_bytes(b"prefix " + b"/ho" + b"me/example/private suffix")
    nested = tmp_path / "nested"
    nested.mkdir()
    disguised = nested / "check_public_tree.py"
    disguised.write_text("payload = torch.load(path)\n", encoding="utf-8")
    findings = scan_tree(tmp_path)
    assert any(item == "home-directory path:cover.png" for item in findings)
    assert any(item.startswith("unsafe-torch-load:nested/check_public_tree.py") for item in findings)


def test_release_scan_detects_directory_symlinks_and_generated_trees(tmp_path: Path) -> None:
    target = tmp_path / "target"
    target.mkdir()
    try:
        (tmp_path / "linked").symlink_to(target, target_is_directory=True)
    except OSError as exc:
        if getattr(exc, "winerror", None) == 1314:
            pytest.skip(
                "symlink creation requires a privilege this account does not hold "
                "(Windows WinError 1314); the directory-symlink detection path is "
                "not exercised without a symlink to create"
            )
        raise
    (tmp_path / "__pycache__").mkdir()
    findings = scan_tree(tmp_path)
    assert "symlink:linked" in findings
    assert "generated-directory:__pycache__" in findings


def test_release_scan_requires_explicit_numpy_pickle_policy(tmp_path: Path) -> None:
    probe = tmp_path / "arrays.py"
    probe.write_text("arrays = np.load(path)\n", encoding="utf-8")
    assert any(
        item.startswith("numpy-pickle-policy-not-explicit:") for item in scan_tree(tmp_path)
    )


def test_release_scan_resolves_import_aliases(tmp_path: Path) -> None:
    probe = tmp_path / "aliases.py"
    probe.write_text(
        "import torch as t\n"
        "from torch import load as checkpoint_load\n"
        "from pickle import loads as restore\n"
        "import numpy as arrays\n"
        "a = t.load(path)\n"
        "b = checkpoint_load(path)\n"
        "c = restore(blob)\n"
        "d = arrays.load(path)\n",
        encoding="utf-8",
    )
    findings = scan_tree(tmp_path)
    assert sum(item.startswith("unsafe-torch-load:") for item in findings) == 2
    assert sum(item.startswith("unsafe-pickle-loader:") for item in findings) == 1
    assert sum(item.startswith("numpy-pickle-policy-not-explicit:") for item in findings) == 1


def test_release_scan_rejects_native_128_cubed_artifact_paths(tmp_path: Path) -> None:
    path = tmp_path / "data" / "128_cubed_training"
    path.mkdir(parents=True)
    (path / "checkpoint.pt").write_bytes(b"weights")
    assert any(item.startswith("native-128-cubed-path:") for item in scan_tree(tmp_path))


def test_distribution_scan_allows_only_expected_package_metadata(tmp_path: Path) -> None:
    metadata = tmp_path / "src" / "pinn_phase.egg-info"
    metadata.mkdir(parents=True)
    (metadata / "PKG-INFO").write_text("Name: pinn-phase\n", encoding="utf-8")
    assert any(item.startswith("generated-directory:") for item in scan_tree(tmp_path))
    assert scan_tree(tmp_path, allow_packaging_metadata=True) == []


# --------------------------------------------------------------------------
# Binary and container coverage.
#
# The identifiers used below are fabricated. A test that spelled out a real
# forbidden token would put that token in the public tree in order to prove it is
# kept out of it, which is the failure mode these rules exist to avoid. Each probe
# is built from a made-up name that no rule knows about, so what is demonstrated is
# that the detector recognizes the SHAPE of an unrecognized identifier rather than
# a list of words.
# --------------------------------------------------------------------------

def _fabricated() -> str:
    """A run identifier whose non-technical segment is invented for this test."""
    return "PINN-" + "QUARTZWORKS" + "-N16-V2-2031-01-01"


def _write_npz(path: Path, members: dict[str, bytes]) -> None:
    import zipfile

    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, payload in members.items():
            archive.writestr(name, payload)


def _npy(array) -> bytes:
    import io

    import numpy as np

    buffer = io.BytesIO()
    np.lib.format.write_array(buffer, array, allow_pickle=True)
    return buffer.getvalue()


def test_scan_reports_any_serialized_object_payload(tmp_path: Path) -> None:
    """A public tree must not distribute a pickled payload, whatever it contains."""
    (tmp_path / "model.pt").write_bytes(b"not really a checkpoint")
    (tmp_path / "state.pkl").write_bytes(b"nor is this")
    findings = scan_tree(tmp_path)
    assert "serialized-object-payload:model.pt" in findings
    assert "serialized-object-payload:state.pkl" in findings


def test_scan_finds_an_unrecognized_identifier_inside_a_serialized_payload(
    tmp_path: Path,
) -> None:
    torch = __import__("torch")
    path = tmp_path / "payload.pt"
    torch.save({"run_id": _fabricated(), "model_state": {}}, path)
    findings = scan_tree(tmp_path)
    assert any(
        item.startswith("unrecognized-run-identifier-segment:payload.pt") for item in findings
    ), findings
    assert any("QUARTZWORKS" in item for item in findings)


def test_scan_finds_an_unrecognized_identifier_in_an_archive_member_name(
    tmp_path: Path,
) -> None:
    import numpy as np

    _write_npz(tmp_path / "bundle.npz", {f"{_fabricated()}.npy": _npy(np.zeros(2))})
    findings = scan_tree(tmp_path)
    assert any(
        item.startswith("unrecognized-archive-member-identifier-segment:bundle.npz")
        for item in findings
    ), findings


def test_scan_reads_compressed_archive_members(tmp_path: Path) -> None:
    """A private string inside a compressed member must not survive the scan."""
    hidden = b"prefix " + b"/ho" + b"me/example/private suffix"
    _write_npz(tmp_path / "bundle.npz", {"notes.npy": hidden})
    findings = scan_tree(tmp_path)
    assert "home-directory path:bundle.npz:notes.npy" in findings, findings


def test_scan_reports_a_non_numeric_array_member(tmp_path: Path) -> None:
    import numpy as np

    _write_npz(tmp_path / "bundle.npz", {"labels.npy": _npy(np.array(["a", "b"]))})
    assert any(
        item.startswith("non-numeric-array-dtype:bundle.npz:labels.npy")
        for item in scan_tree(tmp_path)
    )


def test_scan_reports_archive_member_comments(tmp_path: Path) -> None:
    import zipfile

    path = tmp_path / "bundle.npz"
    with zipfile.ZipFile(path, "w") as archive:
        info = zipfile.ZipInfo("value.npy")
        info.comment = b"internal note"
        archive.writestr(info, b"\x93NUMPY")
    assert "archive-member-comment:bundle.npz:value.npy" in scan_tree(tmp_path)


def test_scan_finds_an_unrecognized_identifier_in_a_structured_record(
    tmp_path: Path,
) -> None:
    import json

    (tmp_path / "record.json").write_text(
        json.dumps({"run": {"name": _fabricated()}}), encoding="utf-8"
    )
    findings = scan_tree(tmp_path)
    assert any(
        item.startswith("unrecognized-record-identifier-segment:record.json")
        for item in findings
    ), findings


def test_scan_finds_an_unrecognized_identifier_in_a_path(tmp_path: Path) -> None:
    directory = tmp_path / "runs" / _fabricated()
    directory.mkdir(parents=True)
    (directory / "notes.txt").write_text("ordinary text\n", encoding="utf-8")
    assert any(
        item.startswith("unrecognized-path-identifier-segment:") for item in scan_tree(tmp_path)
    )


def test_identifier_detector_accepts_purely_technical_identifiers(tmp_path: Path) -> None:
    """The detector must not fire on identifiers built from technical segments only."""
    import json

    (tmp_path / "record.json").write_text(
        json.dumps({"run": "PINN-MPF-3D-VORONOI-CUBE-N16-H128-TBPTT4-2031-01-01"}),
        encoding="utf-8",
    )
    findings = [item for item in scan_tree(tmp_path) if "identifier" in item]
    assert findings == [], findings


# The tests below describe what was PACKAGED. They ask the manifest, not the
# filesystem, so they give the same answer in a pristine extraction and in a tree
# where someone has installed the package and run the documented commands. Nothing
# here skips: a test that skipped the moment you used the repository would be
# testing nothing at the moment it mattered.

def test_the_distributed_payload_scans_clean() -> None:
    """The shipped archive itself must pass, including inside its binaries."""
    assert scan_payload(ROOT) == []


def _mini_release(root: Path) -> None:
    """A minimal tree with a manifest, standing in for a release."""
    import hashlib

    (root / "docs").mkdir(parents=True)
    (root / "README.md").write_text("# example\n", encoding="utf-8")
    (root / "docs" / "note.md").write_text("public documentation\n", encoding="utf-8")
    listed = ["README.md", "docs/note.md"]
    lines = [
        f"{hashlib.sha256((root / name).read_bytes()).hexdigest()}  {name}"
        for name in listed
    ]
    (root / "MANIFEST.sha256").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _add_local_working_files(root: Path) -> None:
    """Exactly what a reader produces by installing and running the documented commands."""
    (root / "outputs" / "replay").mkdir(parents=True)
    (root / "outputs" / "replay" / "run.json").write_text("{}\n", encoding="utf-8")
    (root / "pinn_phase.egg-info").mkdir()
    (root / "pinn_phase.egg-info" / "PKG-INFO").write_text(
        "Name: example\nHome-page: /ho" + "me/someone/checkout\n", encoding="utf-8")
    (root / "__pycache__").mkdir()
    (root / "__pycache__" / "t.cpython-311.pyc").write_bytes(
        b"\x00" + b"/ho" + b"me/someone/checkout/tests/t.py")


def test_the_payload_scan_is_unaffected_by_local_working_files(tmp_path: Path) -> None:
    """Generated material in the tree must not change the payload verdict.

    This is the property that broke the documented workflow: a reader who installed
    the package and ran the reproductions saw the verifier fail on their own output,
    including on a private path that an interpreter had written into a bytecode
    cache. None of it was ever archive content.
    """
    tree = tmp_path / "tree"
    tree.mkdir()
    _mini_release(tree)
    assert scan_payload(tree) == []
    _add_local_working_files(tree)
    assert scan_payload(tree) == []


def test_staging_mode_still_refuses_the_same_generated_material(tmp_path: Path) -> None:
    """Payload mode ignoring local files must not weaken release construction."""
    tree = tmp_path / "tree"
    tree.mkdir()
    _mini_release(tree)
    assert scan_tree(tree) == []
    _add_local_working_files(tree)
    findings = scan_tree(tree)
    assert "generated-directory:outputs" in findings
    assert "generated-directory:__pycache__" in findings
    assert "generated-directory:pinn_phase.egg-info" in findings
    # The directory is reported and then pruned, so the private path inside it is
    # never read. Refusing the directory is the actionable finding; reporting every
    # byte of a cache we are refusing anyway would be noise.


def test_payload_mode_refuses_a_manifest_that_lists_generated_material(
    tmp_path: Path,
) -> None:
    """Ignoring generated files must not become a way to smuggle one into a release."""
    import hashlib

    tree = tmp_path / "tree"
    tree.mkdir()
    _mini_release(tree)
    (tree / "outputs").mkdir()
    smuggled = tree / "outputs" / "payload.json"
    smuggled.write_text("{}\n", encoding="utf-8")
    manifest = tree / "MANIFEST.sha256"
    manifest.write_text(
        manifest.read_text(encoding="utf-8")
        + f"{hashlib.sha256(smuggled.read_bytes()).hexdigest()}  outputs/payload.json\n",
        encoding="utf-8",
    )
    assert "generated-file-listed-in-manifest:outputs/payload.json" in scan_payload(tree)


def test_payload_mode_refuses_a_manifest_entry_with_no_file(tmp_path: Path) -> None:
    tree = tmp_path / "tree"
    tree.mkdir()
    _mini_release(tree)
    manifest = tree / "MANIFEST.sha256"
    manifest.write_text(manifest.read_text(encoding="utf-8") + f"{'0' * 64}  docs/ghost.md\n",
                        encoding="utf-8")
    assert "manifest-lists-a-missing-file:docs/ghost.md" in scan_payload(tree)


def test_scan_reports_a_run_output_directory(tmp_path: Path) -> None:
    """A tree packaged after a run keeps its outputs/ directory; that must fail."""
    produced = tmp_path / "outputs" / "scalar"
    produced.mkdir(parents=True)
    (produced / "metrics.json").write_text("{}\n", encoding="utf-8")
    assert "generated-directory:outputs" in scan_tree(tmp_path)


def test_no_run_output_is_covered_by_the_manifest() -> None:
    """Whatever a local run produces, none of it is part of what was packaged.

    Stated against the manifest rather than the filesystem, so it holds both in the
    distributed archive and in a working tree someone has actually used.
    """
    manifest = (ROOT / "MANIFEST.sha256").read_text(encoding="utf-8").splitlines()
    listed = [line.split("  ", 1)[1] for line in manifest if line.strip()]
    produced = [path for path in listed
                if path.startswith(("outputs/", "artifacts/", "build/", "dist/"))
                or "__pycache__" in path or ".pytest_cache" in path]
    assert produced == [], f"run output is covered by the manifest: {produced}"


def test_scan_follows_a_container_nested_inside_a_container(tmp_path: Path) -> None:
    """A compressed archive inside a compressed member must not hide its contents.

    Scanning the outer file's bytes cannot see through two layers of compression,
    so nesting is followed rather than trusted.
    """
    import io
    import zipfile

    inner = io.BytesIO()
    with zipfile.ZipFile(inner, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        archive.writestr("note.txt", (b"/ho" + b"me/example/private ") * 200)
    with zipfile.ZipFile(tmp_path / "outer.npz", "w", zipfile.ZIP_DEFLATED,
                         compresslevel=9) as archive:
        archive.writestr("payload.npy", inner.getvalue())
    findings = scan_tree(tmp_path)
    assert "home-directory path:outer.npz:payload.npy:note.txt" in findings, findings


# --------------------------------------------------------------------------
# Carrier coverage. A byte rule written against UTF-8 sees nothing inside a
# compressed stream, a PNG text chunk or a UTF-16 file, and every one of those is
# an ordinary file format rather than an exotic attack. Each probe below plants the
# same fabricated leak in a different carrier and requires the scan to find it.
# --------------------------------------------------------------------------

def _planted_leak() -> bytes:
    """A fabricated leak built from the shapes the rules describe."""
    token = b"AUTH" + b"OR_" + b"GO" + b"=" + b"YES"    # split so this file is clean
    return (b"/ho" + b"me/example/private  someone@" + b"gmail.com  "
            + token + b"  rat" + b"ified at the gate")


def _png(chunks: list[tuple[bytes, bytes]]) -> bytes:
    import struct
    import zlib

    out = b"\x89PNG\r\n\x1a\n"
    for kind, body in chunks:
        out += struct.pack(">I", len(body)) + kind + body
        out += struct.pack(">I", zlib.crc32(kind + body))
    return out + struct.pack(">I", 0) + b"IEND" + struct.pack(">I", zlib.crc32(b"IEND"))


def _rules_hit(findings: list[str], probe: str) -> set[str]:
    return {item.split(":")[0] for item in findings if probe in item}


def test_scan_inflates_a_gzip_member(tmp_path: Path) -> None:
    import gzip

    (tmp_path / "notes.txt.gz").write_bytes(gzip.compress(_planted_leak()))
    assert "home-directory path" in _rules_hit(scan_tree(tmp_path), "notes.txt.gz")


def test_scan_reads_inside_a_compressed_tarball(tmp_path: Path) -> None:
    import gzip
    import io
    import tarfile

    payload = _planted_leak()
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w") as archive:
        info = tarfile.TarInfo("record.md")
        info.size = len(payload)
        archive.addfile(info, io.BytesIO(payload))
    (tmp_path / "bundle.tar.gz").write_bytes(gzip.compress(buffer.getvalue()))
    assert "home-directory path" in _rules_hit(scan_tree(tmp_path), "bundle.tar.gz")


def test_scan_reads_png_text_chunks_including_compressed_ones(tmp_path: Path) -> None:
    import zlib

    payload = _planted_leak()
    (tmp_path / "compressed.png").write_bytes(
        _png([(b"zTXt", b"Comment\x00\x00" + zlib.compress(payload))])
    )
    (tmp_path / "plain.png").write_bytes(_png([(b"tEXt", b"Comment\x00" + payload)]))
    findings = scan_tree(tmp_path)
    assert "home-directory path" in _rules_hit(findings, "compressed.png")
    assert "home-directory path" in _rules_hit(findings, "plain.png")


def test_scan_decodes_utf16_text(tmp_path: Path) -> None:
    (tmp_path / "meta.dat").write_bytes(_planted_leak().decode().encode("utf-16"))
    assert "home-directory path" in _rules_hit(scan_tree(tmp_path), "meta.dat")


def test_scan_reports_an_email_outside_the_contact_metadata_files(tmp_path: Path) -> None:
    (tmp_path / "notes.md").write_text("write to someone@" + "example.org\n",
                                       encoding="utf-8")
    (tmp_path / "CITATION.cff").write_text("email: contact@" + "example.org\n",
                                           encoding="utf-8")
    findings = scan_tree(tmp_path)
    assert any(item.startswith("email-address-outside-contact-metadata:notes.md")
               for item in findings), findings
    assert not any("CITATION.cff" in item for item in findings), findings


def test_email_rule_does_not_fire_on_binary_float_data(tmp_path: Path) -> None:
    """`x@y.zz` occurs by chance in float arrays; a text rule must not chase it."""
    import numpy as np

    rng = np.random.default_rng(0)
    np.savez_compressed(tmp_path / "field.npz", phi0=rng.random((8, 64, 64)))
    findings = [item for item in scan_tree(tmp_path) if "email" in item]
    assert findings == [], findings


def test_scan_reports_a_training_artifacts_directory(tmp_path: Path) -> None:
    """`artifacts/` is where the trainer writes; a packaged tree must not have one."""
    produced = tmp_path / "artifacts" / "benchmarks"
    produced.mkdir(parents=True)
    (produced / "summary.json").write_text("{}\n", encoding="utf-8")
    assert "generated-directory:artifacts" in scan_tree(tmp_path)


# --------------------------------------------------------------------------
# Reader-facing language. The risk here runs both ways: a sentence that refers a
# public reader to an internal decision they cannot resolve, and a rule so eager
# that it fires on ordinary scientific writing. Both directions are tested.
# --------------------------------------------------------------------------

def test_scan_reports_a_reference_to_an_unresolvable_internal_decision(
    tmp_path: Path,
) -> None:
    (tmp_path / "ledger.md").write_text(
        "- Score record: `" + "a" * 64 + "`,\n"
        "  authorized by " + "rul" + "ing `" + "b" * 64 + "`.\n",
        encoding="utf-8",
    )
    findings = scan_tree(tmp_path)
    assert any(item.startswith("unresolvable decision reference:") for item in findings), findings


def test_scan_reports_internal_process_self_reference(tmp_path: Path) -> None:
    # Split so this test file does not itself carry the phrase it hunts for.
    phrase = "internal " + "process " + "language"
    (tmp_path / "disclosure.md").write_text(
        f"Summarised as booleans rather than quoted, because the clause is written in "
        f"{phrase}.\n",
        encoding="utf-8",
    )
    assert any(
        item.startswith("internal process self-reference:") for item in scan_tree(tmp_path)
    )


def test_scan_reports_agent_routing_language(tmp_path: Path) -> None:
    routed = "hand" + "ed off to"
    helper = "sub" + "agent"
    (tmp_path / "notes.md").write_text(
        f"The review was {routed} the second reviewer and a {helper} swept the tree.\n",
        encoding="utf-8",
    )
    assert any(item.startswith("agent routing language:") for item in scan_tree(tmp_path))


def test_scan_reports_an_unpublished_venue_reference(tmp_path: Path) -> None:
    status = "manuscript " + "under review"
    venue = "target " + "venue"
    (tmp_path / "notes.md").write_text(
        f"This result is described in a {status}; the {venue} is not named here.\n",
        encoding="utf-8",
    )
    assert any(item.startswith("unpublished venue reference:") for item in scan_tree(tmp_path))


def test_ordinary_scientific_writing_is_not_flagged(tmp_path: Path) -> None:
    """The words this project legitimately uses must survive the language rules.

    A rule that fired on these would teach the author to weaken accurate wording,
    which is worse than having no rule.
    """
    (tmp_path / "science.md").write_text(
        "The prospective protocol fixed the acceptance criteria before the cohort "
        "existed. Each threshold is frozen, and the campaign of ten cases was scored "
        "once against them. Every case had to pass the interface-agreement gate at "
        "step 4,000, and the extinction-timing gate was not met. The frozen reference "
        "fields and the frozen expected score are identified by digest. This ruling "
        "line of grains coarsens under curvature-driven kinetics.\n",
        encoding="utf-8",
    )
    findings = [
        item for item in scan_tree(tmp_path)
        if any(item.startswith(f"{label}:") for label in (
            "unresolvable decision reference", "internal process self-reference",
            "agent routing language", "unpublished venue reference"))
    ]
    assert findings == [], findings


def test_the_distributed_payload_uses_public_language() -> None:
    """Run over what actually ships, so the rules are not merely unit-tested."""
    findings = [
        item for item in scan_payload(ROOT)
        if any(item.startswith(f"{label}:") for label in (
            "unresolvable decision reference", "internal process self-reference",
            "agent routing language", "unpublished venue reference"))
    ]
    assert findings == [], findings
