from __future__ import annotations

import hashlib
import importlib.util
import json
import re
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from pinn_phase.io.artifacts import load_npz_arrays, load_torch_checkpoint, verify_sha256


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_restricted_checkpoint_loader_verifies_hash_and_schema(tmp_path: Path) -> None:
    path = tmp_path / "checkpoint.pt"
    torch.save({"model_state": {"weight": torch.arange(3)}}, path)
    payload = load_torch_checkpoint(
        path,
        expected_sha256=_sha256(path),
        required_keys=frozenset({"model_state"}),
    )
    assert torch.equal(payload["model_state"]["weight"], torch.arange(3))


def test_checkpoint_loader_rejects_digest_before_deserialization(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "checkpoint.pt"
    path.write_bytes(b"not a checkpoint")
    called = False

    def forbidden_load(*args: object, **kwargs: object) -> object:
        nonlocal called
        called = True
        raise AssertionError("deserializer must not run")

    monkeypatch.setattr(torch, "load", forbidden_load)
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        load_torch_checkpoint(path, expected_sha256="0" * 64)
    assert called is False


def test_npz_loader_disables_pickle_and_checks_required_keys(tmp_path: Path) -> None:
    path = tmp_path / "arrays.npz"
    np.savez(path, phi=np.arange(4, dtype=np.float32))
    arrays = load_npz_arrays(
        path,
        expected_sha256=_sha256(path),
        required_keys=frozenset({"phi"}),
    )
    np.testing.assert_array_equal(arrays["phi"], np.arange(4, dtype=np.float32))
    with pytest.raises(ValueError, match="missing required keys"):
        load_npz_arrays(
            path,
            expected_sha256=_sha256(path),
            required_keys=frozenset({"missing"}),
        )


def test_npz_loader_rejects_unreviewed_members_before_array_access(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "arrays.npz"
    path.write_bytes(b"placeholder")
    accessed: list[str] = []

    class Archive:
        files = ["phi0", "states"]

        def __enter__(self) -> "Archive":
            return self

        def __exit__(self, *args: object) -> None:
            return None

        def __getitem__(self, key: str) -> np.ndarray:
            accessed.append(key)
            raise AssertionError("no array may be read from a rejected schema")

    import pinn_phase.io.artifacts as artifacts

    monkeypatch.setitem(
        artifacts.load_npz_arrays.__globals__,
        "_validate_npz_container",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(np, "load", lambda *args, **kwargs: Archive())
    with pytest.raises(ValueError, match="reviewed schema"):
        load_npz_arrays(
            path,
            expected_sha256=_sha256(path),
            expected_keys=frozenset({"phi0"}),
        )
    assert accessed == []


def test_artifact_loaders_open_each_file_once(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    checkpoint = tmp_path / "checkpoint.pt"
    torch.save({"model_state": {}}, checkpoint)
    archive = tmp_path / "arrays.npz"
    np.savez(archive, phi=np.ones(1, dtype=np.float32))
    checkpoint_sha256 = _sha256(checkpoint)
    archive_sha256 = _sha256(archive)
    counts = {checkpoint: 0, archive: 0}
    original_open = Path.open

    def counting_open(path: Path, *args: object, **kwargs: object):
        if path in counts:
            counts[path] += 1
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", counting_open)
    load_torch_checkpoint(checkpoint, expected_sha256=checkpoint_sha256)
    load_npz_arrays(archive, expected_sha256=archive_sha256)
    assert counts == {checkpoint: 1, archive: 1}


def test_verify_sha256_requires_complete_digest(tmp_path: Path) -> None:
    path = tmp_path / "value.bin"
    path.write_bytes(b"value")
    with pytest.raises(ValueError, match="complete lowercase"):
        verify_sha256(path, "abc")


def test_artifact_loaders_enforce_descriptor_size_limits(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    checkpoint = tmp_path / "checkpoint.pt"
    torch.save({"model_state": {}}, checkpoint)
    archive = tmp_path / "arrays.npz"
    np.savez(archive, phi=np.ones(1, dtype=np.float32))
    import pinn_phase.io.artifacts as artifacts

    monkeypatch.setattr(artifacts, "MAX_CHECKPOINT_BYTES", checkpoint.stat().st_size - 1)
    with pytest.raises(ValueError, match="checkpoint exceeds"):
        load_torch_checkpoint(checkpoint, expected_sha256=_sha256(checkpoint))
    monkeypatch.setattr(artifacts, "MAX_NPZ_FILE_BYTES", archive.stat().st_size - 1)
    with pytest.raises(ValueError, match="file-size limit"):
        load_npz_arrays(archive, expected_sha256=_sha256(archive))


def test_checkpoint_loader_refuses_unpatched_pytorch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    checkpoint = tmp_path / "checkpoint.pt"
    torch.save({"model_state": {}}, checkpoint)
    monkeypatch.setattr(torch, "__version__", "2.5.1")
    with pytest.raises(RuntimeError, match="2.6 or newer"):
        load_torch_checkpoint(checkpoint, expected_sha256=_sha256(checkpoint))


def test_public_media_manifest_binds_every_readme_asset() -> None:
    root = Path(__file__).resolve().parents[1]
    media = root / "media/n25_transfer"
    manifest = json.loads((media / "asset_manifest.json").read_text(encoding="utf-8"))
    assert set(manifest["assets"]) == {
        "terminal_atlas.png",
        "id_02_reference_vs_pinn_phase.gif",
        "id_02_reference_vs_pinn_phase_poster.png",
    }
    for name, record in manifest["assets"].items():
        assert _sha256(media / name) == record["sha256"]
        assert record["alt"]
    present = {path.name for path in media.iterdir() if path.is_file()}
    assert present == set(manifest["assets"]) | {"asset_manifest.json"}


def test_every_rendered_animation_has_a_poster_and_readable_frame_delays() -> None:
    from PIL import Image, ImageSequence

    root = Path(__file__).resolve().parents[1]
    animations = sorted(
        (root / "media/n25_transfer").glob("*.gif")
    ) + sorted((root / "media/current_results").glob("*.gif"))
    assert len(animations) >= 3
    for animation in animations:
        assert animation.with_name(f"{animation.stem}_poster.png").is_file()
        with Image.open(animation) as handle:
            delays = [
                frame.info.get("duration", 0)
                for frame in ImageSequence.Iterator(handle)
            ][1:]
        # A zero delay makes a browser strobe the animation at its own clamp.
        assert delays and all(delay >= 500 for delay in delays)


def test_n25_public_fixture_binds_forty_score_archives() -> None:
    root = Path(__file__).resolve().parents[1]
    benchmark = root / "benchmarks/n25_transfer"
    manifest = json.loads((benchmark / "manifest.json").read_text(encoding="utf-8"))
    artifacts = [
        case[role]
        for case in manifest["cases"].values()
        for role in ("reference", "tail", "primary", "secondary")
    ]
    assert len(artifacts) == 40
    for record in artifacts:
        assert _sha256(benchmark / record["path"]) == record["sha256"]
        assert len(record["source_frozen_sha256"]) == 64


def test_retained_media_manifest_covers_every_legacy_asset() -> None:
    root = Path(__file__).resolve().parents[1]
    media = root / "media"
    manifest = json.loads(
        (media / "legacy_asset_manifest.json").read_text(encoding="utf-8")
    )
    actual = {
        path.name
        for path in media.iterdir()
        if path.is_file() and path.name != "legacy_asset_manifest.json"
    }
    assert set(manifest["assets"]) == actual
    assert manifest["source"] == "earlier_public_candidate"
    for name, record in manifest["assets"].items():
        path = media / name
        assert _sha256(path) == record["sha256"]
        assert path.stat().st_size == record["bytes"]
        assert record["frame_count"] >= 1
        assert len(record["dimensions"]) == 2


def test_retained_media_roles_prevent_cross_release_claim_substitution() -> None:
    root = Path(__file__).resolve().parents[1]
    manifest = json.loads(
        (root / "media/legacy_asset_manifest.json").read_text(encoding="utf-8")
    )["assets"]
    assert manifest["n25_v3_long_reference_vs_pinn_phase.gif"]["status"] == (
        "historical_not_prospective"
    )
    assert manifest["n64_dense_no_graph_reference_vs_pinn_phase.gif"]["status"] == (
        "historical_not_current_n3"
    )
    assert manifest["n8_64_reference_vs_pinn_phase.gif"]["status"] == (
        "current_single_case_evidence"
    )
    assert manifest["n16_96_reference_vs_pinn_phase.gif"]["status"] == (
        "current_single_case_evidence"
    )


def test_readme_exposes_current_and_retained_animations() -> None:
    root = Path(__file__).resolve().parents[1]
    readme = (root / "README.md").read_text(encoding="utf-8")
    for asset in (
        "media/method_loop_hybrid_rollout.gif",
        "media/n25_transfer/id_02_reference_vs_pinn_phase.gif",
        "media/n25_transfer/terminal_atlas.png",
        "media/triple_junction_h1024_reference_vs_pinn_phase.gif",
        "media/multigrain_denser_n44_reference_vs_pinn_phase.gif",
        "media/scalar_3d_spherical_reference_vs_pinn_phase.gif",
        "media/n8_64_reference_vs_pinn_phase.gif",
        "media/n16_96_reference_vs_pinn_phase.gif",
    ):
        assert asset in readme
        assert (root / asset).is_file()
    assert "docs/MEDIA_GALLERY.md" in readme
    assert "superseded" in readme.lower()


def test_readme_legacy_numbers_are_bound_in_the_public_evidence_ledger() -> None:
    root = Path(__file__).resolve().parents[1]
    readme = (root / "README.md").read_text(encoding="utf-8")
    ledger = (root / "docs/COMPLETED_EVIDENCE_LEDGER.md").read_text(
        encoding="utf-8"
    )
    # Values still shown on the front page must be bound in the ledger.
    for value in (
        "3.27%",
        "6.54%",
        "127.8°",
        "115.9°",
        "116.4°",
        "5.5°",
        "0.991",
        "R² 0.9999",
        "94.01%",
    ):
        assert value in readme, value
        assert value in ledger, value
    # Values belonging to superseded model variants are ledger-only.
    for value in ("17.20%", "39.70%", "18.3%", "38.8%"):
        assert value not in readme, value
        assert value in ledger, value
    assert "bound-enforcement" in readme
    assert "bound-enforcement" in ledger
    # The provenance root is the digest of each earlier-public source document,
    # which identifies it independently of any repository or revision.
    for digest in (
        "dfe8a4ab5ef18c84107a407681aee2649046878302397563244d1c12322f6c35",
        "dbf5b81df25eb22df8796eacf87e3484afca892a847970b3e5c6a0cb296cca9f",
        "f0ad608a3f228c70f078c3f00b9fd00be741b6dc810bf223278ce3b51fdfdf1d",
        "c9b8d0d4c7c53bc7cd4f96100b35c70ca265b5e0a0f6b271053743ebd56c7c09",
    ):
        assert digest in ledger


def test_readme_states_the_failed_case_and_refuses_the_post_horizon_rescue() -> None:
    readme = (Path(__file__).resolve().parents[1] / "README.md").read_text(
        encoding="utf-8"
    )
    assert "12,933" in readme
    assert "does not rescue" in readme.lower()
    assert "7 of 8" in readme


CURRENT_RESULT_ASSETS = (
    "n25_cascade_reference_vs_pinn_phase.gif",
    "n25_cascade_reference_vs_pinn_phase_poster.png",
    "n64_dense_reference_vs_pinn_phase.gif",
    "n64_dense_reference_vs_pinn_phase_poster.png",
)


def test_current_result_media_are_present_bound_and_referenced() -> None:
    root = Path(__file__).resolve().parents[1]
    media = root / "media/current_results"
    manifest = json.loads((media / "asset_manifest.json").read_text(encoding="utf-8"))
    readme = (root / "README.md").read_text(encoding="utf-8")
    assert set(manifest["assets"]) == set(CURRENT_RESULT_ASSETS)
    for name in CURRENT_RESULT_ASSETS:
        assert _sha256(media / name) == manifest["assets"][name]["sha256"]
    for name in ("n25_cascade_reference_vs_pinn_phase.gif",
                 "n64_dense_reference_vs_pinn_phase.gif"):
        assert f"media/current_results/{name}" in readme
    present = {path.name for path in media.iterdir() if path.is_file()}
    assert present == set(CURRENT_RESULT_ASSETS) | {"asset_manifest.json"}
    assert manifest["renderer_sha256"] == _sha256(root / "scripts/render_current_results.py")


def test_current_result_media_bind_their_source_artifacts_by_digest() -> None:
    root = Path(__file__).resolve().parents[1]
    manifest = json.loads(
        (root / "media/current_results/asset_manifest.json").read_text(encoding="utf-8")
    )
    ledger = (root / "docs/COMPLETED_EVIDENCE_LEDGER.md").read_text(encoding="utf-8")
    sources = manifest["source_artifacts_sha256"]
    assert set(sources) == {"n25_reference", "n25_model", "n64_reference", "n64_model"}
    for role, digest in sources.items():
        assert len(digest) == 64
        # Every array behind a front-page animation is accounted for in the ledger.
        assert digest in ledger, role
    # The renderer must pin the same digests it recorded.
    renderer = (root / "scripts/render_current_results.py").read_text(encoding="utf-8")
    for digest in sources.values():
        assert digest in renderer


def test_superseded_animations_are_gallery_only() -> None:
    root = Path(__file__).resolve().parents[1]
    readme = (root / "README.md").read_text(encoding="utf-8")
    gallery = (root / "docs/MEDIA_GALLERY.md").read_text(encoding="utf-8")
    for asset in (
        "n25_v3_long_reference_vs_pinn_phase.gif",
        "n64_dense_no_graph_reference_vs_pinn_phase.gif",
        "n25_v2_reference_vs_pinn_phase.gif",
    ):
        assert asset not in readme, asset
        assert asset in gallery, asset
        assert (root / "media" / asset).is_file(), asset
    # Their superseded headline values must not reappear on the front page.
    for value in ("17.20%", "39.70%", "18.3%", "38.8%"):
        assert value not in readme, value
        assert value in (root / "docs/COMPLETED_EVIDENCE_LEDGER.md").read_text(
            encoding="utf-8"
        )


def test_front_page_prose_avoids_superseded_checkpoint_language() -> None:
    readme = (Path(__file__).resolve().parents[1] / "README.md").read_text(
        encoding="utf-8"
    ).lower()
    for phrase in ("earlier checkpoint", "earlier first-generation hybrid checkpoint",
                   "historical case"):
        assert phrase not in readme, phrase


def test_readme_headings_follow_the_required_ladder() -> None:
    readme = (Path(__file__).resolve().parents[1] / "README.md").read_text(
        encoding="utf-8"
    )
    headings = [line[3:].strip() for line in readme.splitlines() if line.startswith("## ")]
    expected = [
        "Rollout, temporal extrapolation, and generalisation",
        "How PINN-Phase advances a phase field",
        "Scalar interface motion and topology change",
        "Multiphase relaxation and two-dimensional coarsening",
        "Dense 64-grain coarsening",
        "Prospective generalization to unseen initial conditions",
        "Three-dimensional progression",
        "Results at a glance",
    ]
    assert headings[: len(expected)] == expected
    assert headings.count("Dense 64-grain coarsening") == 1


def test_primary_and_sensitivity_64_grain_results_cannot_be_conflated() -> None:
    root = Path(__file__).resolve().parents[1]
    readme = (root / "README.md").read_text(encoding="utf-8")
    ledger = (root / "docs/COMPLETED_EVIDENCE_LEDGER.md").read_text(encoding="utf-8")
    for primary in ("1.35%", "3.50%", "6.09%", "69.89%", "0.9767"):
        assert primary in readme, primary
    for sensitivity in ("0.85%", "2.07%", "3.97%", "43 of 43"):
        assert sensitivity in readme, sensitivity
    lowered = readme.lower()
    assert "registered primary" in lowered
    assert "sensitivity" in lowered
    assert "not independent confirmation" in lowered or "not an independent" in lowered
    assert "co-vary" in lowered
    # The two runs are bound to different rollouts in the ledger.
    assert "ad87347c884501e91755ce92b18bb54c7521e15ac05d23453996058813f1de82" in ledger
    assert "ea45d058f68b20daf479cec0c9da14927d6033fbcd16bbbaf4c2a591b6d72d35" in ledger


def test_readme_marks_development_versus_prospective_evidence() -> None:
    readme = (Path(__file__).resolve().parents[1] / "README.md").read_text(
        encoding="utf-8"
    )
    assert "development evidence" in readme.lower()
    assert "timing anchor" in readme.lower()
    for value in ("1.46%", "44.37%", "0.81%", "26.59%", "0.95%", "1.13%", "51.25%"):
        assert value in readme, value


def test_release_summary_distinguishes_the_transfer_cohort_and_stays_timeless() -> None:
    root = Path(__file__).resolve().parents[1]
    readme = (root / "README.md").read_text(encoding="utf-8")
    scope = (root / "docs/RELEASE_SCOPE.md").read_text(encoding="utf-8")
    assert "N25 transfer row comprises eight unseen initial conditions and\ntwo stress cases" in readme
    assert "N16 transfer row comprises six prospectively fixed unseen 96³ microstructures" in readme
    assert "runs from the root of a fresh extraction" in readme
    assert "still running" not in readme.lower()
    assert "unfinished native" not in scope.lower()


def test_the_documented_workflow_declares_no_environment_prerequisites() -> None:
    """The printed commands must be runnable as printed.

    A documented sequence that silently needs an environment variable, a prior
    install or a cleanup step between commands is not a workflow; it is a workflow
    plus folklore. This pins the two things a reader would otherwise have to know.
    """
    root = Path(__file__).resolve().parents[1]
    readme = (root / "README.md").read_text(encoding="utf-8")
    section = readme[readme.index("## Verify this archive"):readme.index("## Installation")]
    flowed = section.replace("\n", " ")
    assert "PYTHONDONTWRITEBYTECODE" not in readme, (
        "the documented workflow must not depend on a bytecode environment setting"
    )
    assert "PYTHONPATH" not in section, (
        "the documented workflow must not require the reader to set PYTHONPATH"
    )
    assert "with no environment variables to set" in flowed
    for command in (
        "python -m pytest -q",
        "python scripts/verify_manifest.py",
        "python scripts/check_public_tree.py",
        "python scripts/verify_checkpoint_identities.py",
        "python scripts/smoke_test.py",
        "python scripts/reproduce_scalar_reference.py",
        "python scripts/reproduce_n25_transfer.py",
    ):
        assert command in section, f"the workflow section does not print: {command}"


def test_every_readme_media_path_exists_and_is_manifested() -> None:
    import re

    root = Path(__file__).resolve().parents[1]
    readme = (root / "README.md").read_text(encoding="utf-8")
    manifest = {
        line.split("  ", 1)[1].strip()
        for line in (root / "MANIFEST.sha256").read_text(encoding="utf-8").splitlines()
        if line.strip()
    }
    referenced = set(re.findall(r'src="([^"]+)"', readme))
    assert referenced
    for target in referenced:
        assert (root / target).is_file(), target
        assert target in manifest, target


def test_release_manifest_policy_excludes_documented_generated_outputs(
    tmp_path: Path,
) -> None:
    root = Path(__file__).resolve().parents[1]
    module_path = root / "scripts/release_files.py"
    spec = importlib.util.spec_from_file_location("public_release_files", module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    (tmp_path / "kept.txt").write_text("release payload", encoding="utf-8")
    for directory in ("outputs", "artifacts", "build", "dist", "__pycache__"):
        generated = tmp_path / directory
        generated.mkdir()
        (generated / "generated.txt").write_text("not payload", encoding="utf-8")
    egg_info = tmp_path / "pinn_phase.egg-info"
    egg_info.mkdir()
    (egg_info / "PKG-INFO").write_text("generated", encoding="utf-8")

    assert module.release_files(tmp_path) == {Path("kept.txt")}


def _release_payload(root: Path) -> list[Path]:
    """The files the manifest defines as the archive, loaded via the shipped policy."""
    module_path = root / "scripts/release_files.py"
    spec = importlib.util.spec_from_file_location("public_release_files", module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.manifest_payload(root)


def test_public_payload_carries_no_private_identifiers() -> None:
    """Checked against the release payload, not against the working tree.

    The distinction matters: an interpreter writes a bytecode cache containing the
    absolute path of the file it compiled, so a checkout that has been used has
    private paths in it that were never part of the archive. Enumerating the
    manifest asks the right question -- what did we ship? -- and gives the same
    answer in a pristine extraction and in a tree someone has run the documented
    commands in.
    """
    root = Path(__file__).resolve().parents[1]
    forbidden = ("/ho" + "me/", "AUTHOR" + "_GO", "8c30" + "75b8")
    payload = _release_payload(root)
    assert len(payload) > 100, "the manifest payload looks implausibly small"
    for relative in payload:
        path = root / relative
        assert path.is_file(), f"{relative}: listed in the manifest but missing"
        data = path.read_bytes()
        for needle in forbidden:
            assert needle.encode() not in data, f"{relative}: {needle}"


def test_ci_requirements_cover_every_import_the_suite_needs() -> None:
    """CI installs only requirements/ci.txt, so anything the suite imports must be pinned there."""
    root = Path(__file__).resolve().parents[1]
    pinned = {
        line.split("==")[0].strip().lower()
        for line in (root / "requirements/ci.txt").read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.startswith("#")
    }
    suite = "\n".join(
        path.read_text(encoding="utf-8") for path in (root / "tests").glob("*.py")
    )
    required = {"pytest", "numpy", "torch"}
    if "from PIL import" in suite or "import PIL" in suite:
        required.add("pillow")
    missing = required - pinned
    assert not missing, f"requirements/ci.txt is missing: {sorted(missing)}"


def test_cpu_environment_matches_the_documented_activation_name() -> None:
    """The README tells a reader to activate this environment by name and then run the scripts."""
    import yaml

    root = Path(__file__).resolve().parents[1]
    environment = yaml.safe_load(
        (root / "environment-cpu.yml").read_text(encoding="utf-8")
    )
    readme = (root / "README.md").read_text(encoding="utf-8")
    assert f"conda activate {environment['name']}" in readme
    flattened = []
    for entry in environment["dependencies"]:
        flattened.extend(entry["pip"] if isinstance(entry, dict) else [entry])
    joined = " ".join(flattened)
    # Without the project itself the documented reproductions cannot import pinn_phase.
    assert "-e ." in flattened
    for declared in ("psutil", "scipy", "numpy", "pyyaml"):
        assert declared in joined.lower(), declared


# --------------------------------------------------------------------------- #
# The 64-grain horizon annotation
#
# The 64-grain animation divides its diagnostic panel at step 8,192 and labels
# both sides inside the panel, so the animation explains itself when it travels
# away from this page. That is easy to draw and easy to get subtly wrong, so the
# properties that keep it honest are pinned here rather than left to review:
#
#   * the boundary is the exact constant, not a nearby saved frame;
#   * the extrapolation interval is stated as 3,808 steps;
#   * both regions carry a persistent label, legible at the README's width;
#   * the whole rollout is described as autonomous on both sides, because it is;
#   * nothing claims that training happened during the displayed rollout, or
#     that reference states were supplied after step 0;
#   * only the rollout advances — the model's own curves stop at the current
#     step, while the evaluation-only reference is fixed and drawn complete;
#   * the reference and the prediction stay separable where they agree;
#   * the marker sits on the step the frame is showing;
#   * crossing the boundary changes nothing about the field panels.
# --------------------------------------------------------------------------- #

HORIZON_STEP = 8192
FINAL_STEP = 12000
EXTRAPOLATION_STEPS = FINAL_STEP - HORIZON_STEP
N64_GIF = "media/current_results/n64_dense_reference_vs_pinn_phase.gif"
N64_POSTER = "media/current_results/n64_dense_reference_vs_pinn_phase_poster.png"

# Geometry of the annotated figure, in figure fractions, as the renderer places it.
AXIS_LEFT, AXIS_WIDTH = 0.055, 0.890
AXIS_BOTTOM, AXIS_HEIGHT = 0.200, 0.235
INK_RGB = (0x3b, 0x40, 0x45)
MODEL_BLUE_RGB = (0x45, 0x75, 0xb4)
DISCREPANCY_RED_RGB = (0xd7, 0x30, 0x27)


def _reference_track(module, background=(1.0, 1.0, 1.0)):
    """The reference as it is actually rendered.

    It is drawn at full opacity in its own ink, so this is INK itself. The
    function is kept because the reference has been restyled twice and every
    pixel check reads its colour from here rather than restating it.
    """
    del module, background
    return INK_RGB


def _renderer():
    """The renderer's declared constants, read without importing it.

    Executing the module would pull matplotlib and imageio into the test suite.
    Those are rendering tools for a maintainer regenerating the media, not
    runtime dependencies of the package, and a test that needed them would turn
    continuous integration red for a reason unrelated to what it checks.
    """
    import ast

    root = Path(__file__).resolve().parents[1]
    source = (root / "scripts/render_current_results.py").read_text(encoding="utf-8")
    wanted = {"HORIZON_ANNOTATION", "ANNOTATION_PT", "STATUS_PT", "REGION_TITLE_PT",
              "BOUNDARY_PT", "EXPECTED_SHA256", "REFERENCE_WIDTH", "REFERENCE_DASHES",
              "MODEL_WIDTH"}
    found = {}
    for node in ast.parse(source).body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            target = node.targets[0]
            if isinstance(target, ast.Name) and target.id in wanted:
                found[target.id] = ast.literal_eval(node.value)
    missing = wanted - set(found)
    assert not missing, f"renderer no longer declares: {sorted(missing)}"
    manifest = json.loads(
        (root / "media/current_results/asset_manifest.json").read_text(encoding="utf-8")
    )
    measured = manifest["measurements"]["n64_dense_reference_vs_pinn_phase"]
    found["MEASURED"] = measured
    found["N64_MILESTONES"] = tuple(measured["recomputed"]["steps"])
    return SimpleNamespace(**found)


def _frames(indices):
    """Selected GIF frames as float RGB arrays.

    ``ImageSequence.Iterator`` yields the same image object reseeked, so each
    frame is converted inside the loop; materialising the iterator first would
    hand back the last frame repeated.
    """
    from PIL import Image, ImageSequence

    root = Path(__file__).resolve().parents[1]
    wanted, out = set(indices), {}
    with Image.open(root / N64_GIF) as animation:
        for index, frame in enumerate(ImageSequence.Iterator(animation)):
            if index in wanted:
                out[index] = np.asarray(frame.convert("RGB"), dtype=np.float64) / 255.0
    assert set(out) == wanted
    return out


def _axis_box(pixels):
    height, width = pixels.shape[:2]
    return {
        "x": lambda step: (AXIS_LEFT + AXIS_WIDTH * step / FINAL_STEP) * width,
        "y": lambda fraction: (1.0 - (AXIS_BOTTOM + fraction * AXIS_HEIGHT)) * height,
    }


def _near(pixels, rgb, tolerance=0.10):
    target = np.array(rgb, dtype=np.float64) / 255.0
    return np.abs(pixels - target).max(axis=-1) < tolerance


def _luminance(pixels):
    return pixels @ np.array([0.2126, 0.7152, 0.0722])


def test_horizon_boundary_is_the_exact_constant_and_not_a_saved_frame() -> None:
    module = _renderer()
    annotation = module.HORIZON_ANNOTATION
    assert annotation["horizon_step"] == HORIZON_STEP
    assert annotation["final_step"] == FINAL_STEP
    assert annotation["boundary_label"] == "H = 8,192"
    milestones = module.N64_MILESTONES
    assert HORIZON_STEP not in milestones
    assert max(s for s in milestones if s < HORIZON_STEP) == 8000
    assert min(s for s in milestones if s > HORIZON_STEP) == 8500

    recorded = module.MEASURED["horizon_annotation"]
    assert recorded["horizon_step"] == HORIZON_STEP
    assert recorded["last_frame_within_horizon"] == 8000
    assert recorded["first_frame_beyond_horizon"] == 8500
    assert recorded["no_restart_at_boundary"] is True


def test_boundary_is_drawn_at_8192_and_not_at_a_neighbouring_frame() -> None:
    """Read the drawn boundary back out of the rendered pixels."""
    from PIL import Image

    root = Path(__file__).resolve().parents[1]
    with Image.open(root / N64_POSTER) as poster:
        pixels = np.asarray(poster.convert("RGB"), dtype=np.float64) / 255.0
    box = _axis_box(pixels)
    row = round(box["y"](0.66))
    left, right = round(box["x"](0)), round(box["x"](FINAL_STEP))
    strip = pixels[row, left + 4:right - 4]
    tint = np.array([0xf3, 0xed, 0xe2], dtype=np.float64) / 255.0
    matched = np.where(np.abs(strip - tint).max(axis=1) < 0.02)[0]
    assert matched.size, "the extrapolation band was not found in the rendered panel"
    recovered = FINAL_STEP * (int(matched[0]) + 4) / (right - left)
    # One pixel is about ten steps, and the detector reads the first fully
    # tinted column of an anti-aliased edge, so a residual of that order is the
    # edge itself rather than a misplaced boundary.
    assert abs(recovered - HORIZON_STEP) < 25, f"band edge recovered at {recovered:.0f}"
    assert abs(recovered - 8000) > 100
    assert abs(recovered - 8500) > 100


def test_extrapolation_interval_is_stated_as_3808_steps_everywhere() -> None:
    root = Path(__file__).resolve().parents[1]
    module = _renderer()
    assert module.MEASURED["horizon_annotation"]["extrapolation_steps"] == 3808
    annotation = module.HORIZON_ANNOTATION
    assert annotation["final_step"] - annotation["horizon_step"] == 3808
    # The region names the interval it covers, on the frame itself.
    beyond = " ".join(module.HORIZON_ANNOTATION["beyond_detail"])
    assert "8,193" in beyond and "12,000" in beyond
    within = " ".join(module.HORIZON_ANNOTATION["within_detail"])
    assert "0–8,192" in within
    readme = (root / "README.md").read_text(encoding="utf-8")
    assert readme.count("3,808") >= 2
    assert "8,192" in readme


def test_both_regions_carry_a_persistent_label_in_every_frame() -> None:
    """The animation must explain its own backgrounds without the page."""
    module = _renderer()
    annotation = module.HORIZON_ANNOTATION
    assert annotation["within_title"] == "Within represented training horizon"
    assert annotation["beyond_title"] == "Autonomous extrapolation"
    # "training zone" would be false: no training happens during this rollout.
    for title in (annotation["within_title"], annotation["beyond_title"]):
        assert "training zone" not in title.lower()

    every = list(range(len(module.N64_MILESTONES)))
    frames = _frames(every)
    for index, pixels in frames.items():
        box = _axis_box(pixels)
        for centre, name in ((HORIZON_STEP / 2, "within"),
                             ((HORIZON_STEP + FINAL_STEP) / 2, "beyond")):
            top, bottom = round(box["y"](0.93)), round(box["y"](0.68))
            half = round(0.11 * pixels.shape[1])
            column = round(box["x"](centre))
            patch = pixels[top:bottom, max(column - half, 0):column + half]
            # Label glyphs are the darkest thing in an otherwise empty band.
            assert _near(patch, INK_RGB, tolerance=0.30).sum() > 200, (
                f"frame {index}: the {name} region label is missing or faint"
            )


def test_region_and_boundary_labels_are_readable_at_the_readme_width() -> None:
    from PIL import Image

    root = Path(__file__).resolve().parents[1]
    module = _renderer()
    readme = (root / "README.md").read_text(encoding="utf-8")
    assert 'n64_dense_reference_vs_pinn_phase.gif" width="900"' in readme
    with Image.open(root / N64_POSTER) as poster:
        native_width = poster.width
    scale = 900 / native_width
    for name, points in (("supporting text", module.ANNOTATION_PT),
                         ("region title", module.REGION_TITLE_PT),
                         ("boundary label", module.BOUNDARY_PT),
                         ("status line", module.STATUS_PT)):
        displayed = points * (110 / 72) * scale
        assert displayed >= 9.0, f"{name} renders at {displayed:.1f} px at 900 px wide"
    # The region titles are the dominant text of the panel.
    assert module.REGION_TITLE_PT > module.ANNOTATION_PT


def test_autonomous_rollout_is_stated_on_both_sides_of_the_boundary() -> None:
    module = _renderer()
    annotation = module.HORIZON_ANNOTATION
    within, beyond = annotation["within_status"], annotation["beyond_status"]
    assert within == "within represented training horizon · autonomous rollout"
    assert beyond == "beyond represented horizon · autonomous extrapolation"
    assert "autonomous" in within and "autonomous" in beyond
    # And in the panel itself, on both sides.
    # Matched case-insensitively: the labels are sentence case, and a check that
    # pinned their capitalisation would be testing the house style rather than
    # the claim that both sides are described as autonomous.
    assert "autonomous rollout" in " ".join(annotation["within_detail"]).lower()
    assert "autonomous" in annotation["beyond_title"].lower()
    assert annotation["model_label"] == "PINN-Phase autonomous rollout"
    readme = (Path(__file__).resolve().parents[1] / "README.md").read_text(
        encoding="utf-8"
    )
    assert "rolled\nautonomously to step 12,000" in readme
    assert "The whole rollout is autonomous" in readme


def test_nothing_claims_training_or_labels_during_the_displayed_rollout() -> None:
    root = Path(__file__).resolve().parents[1]
    module = _renderer()
    annotation = module.HORIZON_ANNOTATION
    frame_text = " ".join(
        [annotation["within_status"], annotation["beyond_status"],
         annotation["boundary_label"], annotation["continuity_note"],
         annotation["within_title"], annotation["beyond_title"],
         annotation["reference_label"], annotation["model_label"],
         *annotation["within_detail"], *annotation["beyond_detail"],
         *annotation["footer"]]
    ).lower()
    for forbidden in ("training data", "training region", "training zone",
                      "seen data", "in-sample", "out-of-sample", "ground truth",
                      "restart", "retrain", "fine-tune", "untrusted", "invalid",
                      "fails beyond", "supervised on the reference",
                      "trained on the reference"):
        assert forbidden not in frame_text, forbidden
    # The label-free statement travels with the animation, not only on the page.
    assert "no reference frames after step 0" in frame_text
    assert "supervised only by physics residuals" in frame_text
    assert "does not mean the model was shown these states" in frame_text
    # The reference appears as a comparator, never as an input. It is drawn
    # complete, so the frame itself has to say that it is a fixed comparator and
    # that only the rollout advances; otherwise a reader could take the full
    # trajectory for something the model was given.
    assert annotation["reference_label"] == "PF reference (evaluation only)"
    assert "the reference is fixed and drawn complete" in frame_text
    assert "only the rollout advances" in frame_text

    # Matched against the flowed text, not the source lines: these sentences are
    # rewrapped whenever a word changes, and a test that pins the line breaks
    # fails for reasons that have nothing to do with what it is checking.
    readme = " ".join((root / "README.md").read_text(encoding="utf-8").split())
    assert "no post-t0 reference states were used" in readme
    assert "held-out comparator used for evaluation only" in readme
    assert "never supplied to the model" in readme


def test_only_the_rollout_advances_and_the_reference_stays_fixed() -> None:
    """The model's own curves stop at the current step; the reference does not move.

    The two halves are one property and are checked together, because each is
    what makes the other readable. The prediction and the differing-pixel curve
    describe the rollout, and drawing either past the marker would show an
    outcome the rollout has not reached. The phase-field reference describes
    nothing the model produced — it is the fixed comparator the rollout is read
    against — so it is drawn complete, and the check proves it is genuinely
    unchanging rather than merely permitted to be.
    """
    module = _renderer()
    annotation = module.HORIZON_ANNOTATION
    milestones = list(module.N64_MILESTONES)
    early, late = milestones.index(2000), milestones.index(8500)
    frames = _frames([early, late])
    pixels = frames[early]
    box = _axis_box(pixels)

    # Everything from just past the playhead to the right edge, including the
    # extrapolation region — the whole plotting band below the label strip, not a
    # convenient window. The boundary rule is drawn full height at 8,192 by
    # design, so its own columns are excluded and nothing else is.
    top, bottom = round(box["y"](0.65)), round(box["y"](0.02))
    rule = round(box["x"](HORIZON_STEP))
    columns = np.r_[round(box["x"](2400)):rule - 30, rule + 30:round(box["x"](FINAL_STEP))]
    ahead = pixels[top:bottom, columns]
    # Tolerances are per ink because the GIF palette quantizes them by different
    # amounts: the red curve lands about 0.17 away from its nominal colour in the
    # blue channel, so a single tight tolerance would silently find nothing and
    # pass.
    rolling = ((DISCREPANCY_RED_RGB, "differing-pixel", 0.22),
               (MODEL_BLUE_RGB, "prediction", 0.14))
    for rgb, name, tolerance in rolling:
        found = int(_near(ahead, rgb, tolerance=tolerance).sum())
        assert found < 40, (
            f"{name} ink appears ahead of the playhead at step 2,000 ({found} px)"
        )
    # The reference is expected in that same region, and in quantity: if it were
    # revealed with the playhead this assertion is what would fail.
    track = _reference_track(module)
    assert int(_near(ahead, track, tolerance=0.10).sum()) > 500, (
        "the reference does not span the axis on an early frame"
    )

    # Fixed, not merely complete. The far right of the panel carries the
    # reference and nothing else until the rollout arrives, so the reference ink
    # there must occupy exactly the same pixels on a frame inside the horizon and
    # on one past the boundary. Compared as a mask rather than byte for byte:
    # the GIF palette is rebuilt per frame from the whole canvas, including the
    # field panels, so the quantized tone of an unchanged line wobbles by up to
    # 0.03. That drift is bounded separately; what must not move is the geometry.
    far = slice(round(box["x"](10600)), round(box["x"](11800)))
    windows = [frames[at][top:bottom, far] for at in (early, late)]
    masks = [_near(window, track, tolerance=0.10) for window in windows]
    # 75 pixels, measured. The count is this small because the reference is a
    # dashed ordinary-weight line rather than the wide track it used to be, so
    # the threshold is set from the measurement; it is still far from vacuous,
    # since a reference revealed with the playhead would put zero here.
    assert masks[0].sum() > 40 and np.array_equal(*masks), (
        "the reference moves between frames; it is not fixed"
    )
    assert np.abs(windows[0] - windows[1]).max() < 0.05, "the reference is restyled mid-animation"
    # Non-vacuous: the same window does change once the rollout reaches it.
    terminal = _frames([len(milestones) - 1])[len(milestones) - 1][top:bottom, far]
    assert np.abs(windows[0] - terminal).max() > 0.3, (
        "the window is insensitive to what is drawn in it"
    )

    # Behind the playhead all three are present, so the test cannot pass merely
    # by looking where nothing is drawn. Each is sought at the height its own
    # recorded value puts it, rather than in one band that might miss a curve
    # hugging the axis.
    recomputed = module.MEASURED["recomputed"]
    at = milestones.index(1000)
    ceiling = max(recomputed["differing_pixels_pct"]) * annotation["disagreement_headroom"]
    grain_ceiling = 64 * annotation["grain_headroom"]
    columns = slice(round(box["x"](800)), round(box["x"](1200)))
    behind_inks = (*rolling, (track, "reference", 0.10))
    heights = (recomputed["differing_pixels_pct"][at] / ceiling,
               recomputed["model_active_grains"][at] / grain_ceiling,
               recomputed["reference_active_grains"][at] / grain_ceiling)
    for (rgb, name, tolerance), fraction in zip(behind_inks, heights):
        row = round(box["y"](fraction))
        behind = pixels[row - 8:row + 9, columns]
        assert _near(behind, rgb, tolerance=tolerance).sum() > 20, f"no {name} ink behind"


def test_marker_sits_on_the_step_the_frame_is_showing() -> None:
    module = _renderer()
    milestones = list(module.N64_MILESTONES)
    model_active = module.MEASURED["recomputed"]["model_active_grains"]
    grain_ceiling = 64 * module.HORIZON_ANNOTATION["grain_headroom"]

    checked = [milestones.index(step) for step in (2500, 6000, 8500, 11000)]
    frames = _frames(checked)
    for index in checked:
        pixels = frames[index]
        box = _axis_box(pixels)

        def density(step_index):
            column = round(box["x"](milestones[step_index]))
            row = round(box["y"](model_active[step_index] / grain_ceiling))
            patch = pixels[row - 9:row + 10, column - 9:column + 10]
            return int(_near(patch, MODEL_BLUE_RGB, tolerance=0.14).sum())

        here, previous = density(index), density(index - 1)
        # The marker is a filled disc; the same window one saved step back holds
        # only the line passing through it, which is a few pixels wide. Measured
        # across every frame of the animation the disc never falls below 145 and
        # the bare line never reaches 80.
        assert here >= 120, f"frame {index}: no marker at step {milestones[index]:,}"
        assert here > 1.6 * previous, (
            f"frame {index}: marker is not distinctly at the current step "
            f"({here} here against {previous} at the previous step)"
        )


def test_reference_and_prediction_stay_separable_where_they_agree() -> None:
    """Two agreeing series must still look like two series.

    Neither is displaced numerically, so the separation has to come from width
    and tone: the reference is laid down wide and pale and the prediction runs
    as a narrower saturated core inside it, leaving a margin on both sides.
    """
    from PIL import Image

    root = Path(__file__).resolve().parents[1]
    module = _renderer()
    recomputed = module.MEASURED["recomputed"]
    milestones = list(module.N64_MILESTONES)
    agreeing = [
        step for step, reference, model in zip(
            milestones, recomputed["reference_active_grains"],
            recomputed["model_active_grains"])
        if reference == model and 1000 < step < 8000
    ]
    assert agreeing, "no coincident stretch to inspect"

    with Image.open(root / N64_POSTER) as poster:
        pixels = np.asarray(poster.convert("RGB"), dtype=np.float64) / 255.0
    box = _axis_box(pixels)
    grain_ceiling = 64 * module.HORIZON_ANNOTATION["grain_headroom"]

    # Both series are ordinary weight now, so the separation is the dash pattern
    # rather than a width difference: the reference must be dashed, and its gaps
    # must be long enough to show the prediction underneath at the README's
    # width. Neither series is displaced numerically.
    offset, pattern = module.REFERENCE_DASHES
    assert offset == 0 and len(pattern) == 2 and min(pattern) >= 2.5, (
        f"the reference is not dashed with legible gaps: {module.REFERENCE_DASHES}"
    )
    assert abs(module.REFERENCE_WIDTH - module.MODEL_WIDTH) < 1.0, (
        "the two series are no longer ordinary comparable weight"
    )
    track = _reference_track(module)
    separable = 0
    for step in agreeing:
        level = recomputed["reference_active_grains"][milestones.index(step)]
        row = round(box["y"](level / grain_ceiling))
        window = pixels[row - 7:row + 8, round(box["x"](step)) - 30:round(box["x"](step)) + 30]
        neutral = _near(window, track, tolerance=0.10).sum()
        blue = _near(window, MODEL_BLUE_RGB, tolerance=0.14).sum()
        if neutral > 25 and blue > 25:
            separable += 1
    assert separable >= max(3, round(0.8 * len(agreeing))), (
        f"only {separable} of {len(agreeing)} coincident steps show both series"
    )
    # A grayscale reader has the dash pattern, which desaturation cannot remove,
    # and a luminance difference between the two inks as well. The margin is
    # smaller than the old wide-track arrangement gave, so the pattern is now the
    # primary cue and this is the secondary one.
    reference_luminance = float(np.dot(np.array(track) / 255.0, [0.2126, 0.7152, 0.0722]))
    model_luminance = float(np.dot(np.array(MODEL_BLUE_RGB) / 255.0, [0.2126, 0.7152, 0.0722]))
    assert abs(reference_luminance - model_luminance) > 0.08, (
        f"only {abs(reference_luminance - model_luminance):.3f} of luminance separates them"
    )


def test_field_panels_do_not_change_because_the_boundary_is_crossed() -> None:
    """The microstructure panels must be untouched by the annotation.

    Frames 16 and 17 straddle step 8,192. The fields themselves evolve between
    them, as the physics requires, but the panel framing must not: no border, no
    tint, no overlay may appear because the rollout left the represented
    horizon. The gaps between and around the panels are where such a change
    would show, and they must be identical.
    """
    module = _renderer()
    milestones = list(module.N64_MILESTONES)
    before, after = milestones.index(8000), milestones.index(8500)
    assert milestones[before] <= HORIZON_STEP < milestones[after]

    frames = _frames([before, after])
    height, width = frames[before].shape[:2]
    top, bottom = round(0.126 * height), round(0.480 * height)
    # The panels are square images letterboxed inside wider axes, so these are
    # the margins and the true gaps between the three drawn fields.
    gaps = np.r_[0:round(0.055 * width),
                 round(0.320 * width):round(0.370 * width),
                 round(0.635 * width):round(0.685 * width),
                 round(0.945 * width):width]
    difference = np.abs(frames[before][top:bottom, gaps]
                        - frames[after][top:bottom, gaps]).max()
    assert difference < 0.02, (
        f"the panel surround changed across the boundary by {difference:.3f}"
    )


def test_two_regions_survive_grayscale_and_colour_vision() -> None:
    from PIL import Image

    root = Path(__file__).resolve().parents[1]
    with Image.open(root / N64_POSTER) as poster:
        pixels = np.asarray(poster.convert("RGB"), dtype=np.float64) / 255.0
    box = _axis_box(pixels)

    def patch(step):
        column = round(box["x"](step))
        row = round(box["y"](0.66))
        return pixels[row - 6:row + 6, column - 14:column + 14].reshape(-1, 3)

    inside, tail = patch(2000), patch(11000)
    # A band carried by hue alone vanishes in a monochrome copy. The rendered
    # step measures 0.066, so this floor is a real bound rather than a formality.
    assert _luminance(inside).mean() - _luminance(tail).mean() > 0.04

    # Dichromacy cannot separate hues the way it separates lightness, so the
    # check that matters is that the distinction is a lightness step at all.
    # Simulated as the loss of one opponent channel: with red and green
    # collapsed the two patches must still differ.
    collapsed_inside = inside[:, [2]].mean() + inside[:, :2].mean()
    collapsed_tail = tail[:, [2]].mean() + tail[:, :2].mean()
    assert abs(collapsed_inside - collapsed_tail) > 0.03


def test_current_64_grain_media_match_their_recorded_digests() -> None:
    root = Path(__file__).resolve().parents[1]
    manifest = json.loads(
        (root / "media/current_results/asset_manifest.json").read_text(encoding="utf-8")
    )
    for name in ("n64_dense_reference_vs_pinn_phase.gif",
                 "n64_dense_reference_vs_pinn_phase_poster.png"):
        recorded = manifest["assets"][name]["sha256"]
        assert len(recorded) == 64
        assert _sha256(root / "media/current_results" / name) == recorded, name


def test_registered_primary_and_horizon_sensitivity_stay_separate() -> None:
    root = Path(__file__).resolve().parents[1]
    module = _renderer()
    readme = (root / "README.md").read_text(encoding="utf-8")
    assert module.HORIZON_ANNOTATION["status_qualifier"] == "not the registered primary result"
    assert readme.index("The registered primary result") < readme.index(
        "A post-evaluation sensitivity run"
    )
    lowered = readme.lower()
    assert "not an independent" in lowered or "not independent confirmation" in lowered
    assert "co-vary" in lowered
    assert module.EXPECTED_SHA256["n64_model"] == (
        "ea45d058f68b20daf479cec0c9da14927d6033fbcd16bbbaf4c2a591b6d72d35"
    )
    ledger = (root / "docs/COMPLETED_EVIDENCE_LEDGER.md").read_text(encoding="utf-8")
    assert "ad87347c884501e91755ce92b18bb54c7521e15ac05d23453996058813f1de82" in ledger


def test_every_documented_script_runs_without_installation() -> None:
    """A script the README tells a reader to run must run in a fresh extraction.

    Three of these needed an editable install before R2.2, which made the documented
    sequence fail from an ordinary extraction. Each now puts this repository's src/
    on the path itself, so the property is structural rather than remembered.
    """
    root = Path(__file__).resolve().parents[1]
    readme = (root / "README.md").read_text(encoding="utf-8")
    section = readme[readme.index("## Verify this archive"):readme.index("## Installation")]
    documented = sorted(set(re.findall(r"python (scripts/[a-z0-9_]+\.py)", section)))
    assert len(documented) >= 6, f"expected the workflow to print several scripts: {documented}"
    for relative in documented:
        source = (root / relative).read_text(encoding="utf-8")
        imports_package = re.search(r"^\s*(?:from|import)\s+pinn_phase\b", source, re.M)
        if not imports_package:
            continue        # a script that does not import the package needs no path
        assert "sys.path.insert" in source, (
            f"{relative} imports pinn_phase but does not put src/ on the path; it "
            "would fail in a fresh extraction unless the reader installed the package"
        )
