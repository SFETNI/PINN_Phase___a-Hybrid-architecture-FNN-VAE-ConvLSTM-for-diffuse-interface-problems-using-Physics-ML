"""Load-bearing public guards for the N16/96 prospective cohort."""
from __future__ import annotations

import hashlib
import inspect
import json
import os
import runpy
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image, ImageSequence


ROOT = Path(__file__).resolve().parents[1]
RECORD_PATH = ROOT / "benchmarks/n16_96_transfer/expected_score.json"
ASSET_PATH = ROOT / "benchmarks/n16_96_transfer/manifest.json"
MEDIA_PATH = ROOT / "media/n16_96_transfer/asset_manifest.json"
CHECKPOINT = "9e15e6b74f4f92f37d5709a242e436f721ec5d1306a30e05e5f39794de4459dd"


def _json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _records():
    return _json(RECORD_PATH)


def _asset_fixture(root: Path) -> Path:
    """Create a complete small 18-asset hierarchy for the real verifier."""
    assets = []
    for ordinal in range(1, 7):
        for filename in ("t0.npz", "model_frames.npz", "ref_frames.npz"):
            payload = f"N16 fixture {ordinal} {filename}".encode("ascii")
            relative = Path("external/n16_96_transfer") / str(ordinal) / filename
            path = root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(payload)
            assets.append({
                "expected_relative_location": relative.as_posix(),
                "filename": filename,
                "bytes": len(payload),
                "sha256": hashlib.sha256(payload).hexdigest(),
            })
    manifest = root / "manifest.json"
    manifest.write_text(json.dumps({"assets": assets}), encoding="utf-8")
    return manifest


def test_terminal_and_monitoring_contract_is_exact() -> None:
    record = _records()
    assert record["saved_cadence_steps"] == 200
    assert record["terminal_scored_step"] == 1600
    assert record["monitored_saved_horizon_step"] == 3200
    assert {(case["terminal_step"], case["monitoring_horizon_step"])
            for case in record["cases"]} == {(1600, 3200)}


def test_topology_identity_and_complete_qualification_are_paired() -> None:
    cases = _records()["cases"]
    assert len(cases) == 6
    assert all(case["terminal_active_set_equal"] for case in cases)
    assert all(case["extinction_identities_equal"] for case in cases)
    assert sum(case["complete_predefined_qualification"] for case in cases) == 5


def test_all_eighteen_event_offsets_have_the_sealed_breakdown() -> None:
    offsets = [event["offset_steps"] for case in _records()["cases"]
               for event in case["event_offsets"]]
    assert len(offsets) == 18
    assert {offset: offsets.count(offset) for offset in (-200, 0, 200)} == {
        -200: 2, 0: 11, 200: 5,
    }
    assert all(abs(offset) <= 200 for offset in offsets)


def test_case_five_has_only_the_two_stated_qualification_failures() -> None:
    case = _records()["cases"][4]
    assert case["public_name"] == "Unseen microstructure 5"
    assert not case["complete_predefined_qualification"]
    assert not case["wave_order_preserved"]
    assert case["minimum_persistence_gain_step"] == 3200
    assert abs(case["minimum_persistence_gain_percentage_points"] + 0.080589) < 1e-6
    assert case["terminal_active_set_equal"]
    assert case["extinction_identities_equal"]
    assert case["tail_active_set_equal"]
    assert all(event["within_plus_minus_200"] for event in case["event_offsets"])
    assert case["unmet_criteria"] == [
        "strict persistence versus static-t0 at every saved post-initial state",
        "strict reference extinction-wave order",
    ]


def test_strict_boundaries_are_not_relaxed_in_the_public_contract() -> None:
    contract = _records()["qualification_contract"]
    assert "< 1e-3 (strict)" in contract["numerical_integrity"]
    assert "strictly greater" in contract["persistence"]
    assert "+/-200 inclusive" in contract["events"]
    assert "strict reference wave order" in contract["events"]
    cases = _records()["cases"]
    assert all(case["maximum_phase_sum_error"] < 1e-3 for case in cases)
    assert all(case["finite_values"] and case["nan_count"] == case["inf_count"] == 0 for case in cases)
    assert all(case["persistence_frames_compared"] == 16 for case in cases)
    assert cases[4]["persistence_frames_strictly_greater"] == 15
    assert all(case["persistence_frames_strictly_greater"] == 16 for case in cases[:4] + cases[5:])


def test_public_names_are_complete_and_internal_ids_are_confined_to_provenance() -> None:
    names = [case["public_name"] for case in _records()["cases"]]
    assert names == [f"Unseen microstructure {index}" for index in range(1, 7)]
    assert _records()["public_case_names_only"] is True
    benchmark = (ROOT / "benchmarks/n16_96_transfer/README.md").read_text(encoding="utf-8")
    assert "C1" not in benchmark and "C6" not in benchmark
    assert {entry["case"] for entry in _json(MEDIA_PATH)["inputs"]} == {
        "C1", "C2", "C3", "C4", "C5", "C6",
    }


def test_readme_and_benchmark_record_are_numerically_synchronized() -> None:
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    benchmark = (ROOT / "benchmarks/n16_96_transfer/README.md").read_text(encoding="utf-8")
    for text in (readme, benchmark):
        assert "step 1600" in text
        assert "3200" in text
        assert "Five of six" in text or "five of six" in text
    assert "95.06" in readme and "95.83" in readme
    assert "exact 13-grain terminal active set" in readme
    assert "all three extinction identities" in readme


def test_same_checkpoint_is_reused_without_duplicate_public_weights() -> None:
    record = _records()
    assert record["checkpoint_sha256"] == CHECKPOINT
    assert record["same_checkpoint_as_development_n16_cube"] is True
    ledger = _json(ROOT / "docs/ARTIFACT_IDENTITY_LEDGER.json")
    entry = next(item for item in ledger["artifacts"] if item["name"] == "n16_96_cube_hybrid")
    assert entry["parent_checkpoint_sha256"] == CHECKPOINT
    assert entry["public_weights_path"] == "checkpoints/n16_96_cube_hybrid.weights.npz"
    assert len(list((ROOT / "checkpoints").glob("*n16*weights.npz"))) == 1


def test_all_six_media_bindings_frame_order_and_outputs() -> None:
    media = _json(MEDIA_PATH)
    assert media["steps"] == list(range(0, 3201, 200))
    assert media["no_interpolation"] is True
    pairs = {(entry["case"], entry["role"]) for entry in media["inputs"]}
    assert pairs == {(f"C{case}", role) for case in range(1, 7)
                     for role in ("model", "reference")}
    assert media["checkpoint_sha256"] == CHECKPOINT
    assert media["cohort_score_sha256"] == _records()["sealed_sources"]["cohort_score_sha256"]
    assert media["cohort_manifest_sha256"] == _records()["sealed_sources"]["cohort_manifest_sha256"]
    environment = media["renderer_environment"]
    assert environment["freetype"] == "2.14.3"
    assert environment["font_file"] == "DejaVuSans.ttf"
    assert len(environment["font_sha256"]) == 64
    gif = MEDIA_PATH.parent / "n16_96_unseen_cohort_reference_vs_pinn_phase.gif"
    poster = MEDIA_PATH.parent / "n16_96_unseen_cohort_reference_vs_pinn_phase_poster.png"
    with Image.open(gif) as image:
        frames = [frame.convert("RGBA").tobytes() for frame in ImageSequence.Iterator(image)]
        assert image.size == (1500, 800)
    assert len(frames) == 17
    assert len(set(frames)) > 1
    with Image.open(poster) as image:
        assert image.size == (1500, 860)
    for name, entry in media["outputs"].items():
        payload = (MEDIA_PATH.parent / name).read_bytes()
        assert len(payload) == entry["bytes"]
        assert hashlib.sha256(payload).hexdigest() == entry["sha256"]
    assert media["outputs"] == {
        "n16_96_unseen_cohort_reference_vs_pinn_phase.gif": {
            "bytes": 745191,
            "sha256": "2334e8f880be27e1f16bef30a793ab3176cee180cd46a3b7d3d3679524cf4feb",
        },
        "n16_96_unseen_cohort_reference_vs_pinn_phase_poster.png": {
            "bytes": 56948,
            "sha256": "d692ad1d439039293769906f5f8f9d1d3ce1eaedc34731edfe4a4e4aaacf59f4",
        },
    }


def test_exact_media_environment_and_non_skipping_ci_job_are_declared() -> None:
    environment = (ROOT / "environment-media.yml").read_text(encoding="utf-8")
    for item in ("python=3.11.15", "numpy=2.2.6", "matplotlib=3.10.7",
                 "freetype=2.14.3", "pillow=12.2.0", "scipy=1.16.2",
                 "imageio==2.37.3", "pytest==8.4.2"):
        assert item in environment
    ci = (ROOT / ".github/workflows/ci.yml").read_text(encoding="utf-8")
    assert "exact-media:" in ci
    assert "environment-media.yml" in ci
    assert "pinn-phase-media" in ci
    assert "-r" + "24" not in ci
    assert "test_n16_96_transfer.py" in ci
    assert "test_renderer_executes_end_to_end_with_synthetic_inputs" in ci

    # The exact-media job is the only non-skipping guarantee this file makes: it
    # runs the selected renderer test for real, on Ubuntu, every time. That
    # guarantee is about the selected renderer test, not the whole module -- an
    # unrelated Windows-only test fixture (the symlink-privilege capability
    # check below) may skip narrowly without weakening it.
    exact_media_start = ci.index("exact-media:")
    assert "runs-on: ubuntu-latest" in ci[exact_media_start:exact_media_start + 400]

    renderer_source = inspect.getsource(test_renderer_executes_end_to_end_with_synthetic_inputs)
    assert "pytest" ".skip" not in renderer_source


def test_whole_tree_renderer_discovery_covers_every_renderer_manifest() -> None:
    manifests = []
    for path in ROOT.rglob("*.json"):
        record = _json(path)
        if isinstance(record, dict) and "renderer_sha256" in record:
            manifests.append(path.relative_to(ROOT).as_posix())
    assert sorted(manifests) == [
        "media/current_results/asset_manifest.json",
        "media/n16_96_interior/asset_manifest.json",
        "media/n16_96_transfer/asset_manifest.json",
        "media/n25_transfer/asset_manifest.json",
    ]


def test_external_asset_total_and_provenance_only_surfaces_agree() -> None:
    assets = _json(ASSET_PATH)["assets"]
    assert len(assets) == 18
    assert sum(asset["bytes"] for asset in assets) == 10_726_256_132
    assert {asset["release_class"] for asset in assets} == {"DOCUMENTED_ONLY"}
    assert _records()["reproduction_level"] == "PROVENANCE_ONLY"
    for path in ("docs/REPRODUCTION_LEVELS.json", "docs/ARTIFACT_IDENTITY_LEDGER.json",
                 "docs/CLAIM_TO_ARTIFACT_MAP.json", "docs/REPLAY_ENTRYPOINTS.json",
                 "docs/TRAINING_PATH_DISCLOSURE.json"):
        assert _json(ROOT / path)["n16_96_prospective_transfer"]["level"] == "PROVENANCE_ONLY"


def test_manifest_covers_release_payload_without_inspecting_mutable_caches() -> None:
    completed = subprocess.run([sys.executable, "-B", str(ROOT / "scripts/verify_manifest.py")],
                               cwd=ROOT, capture_output=True, text=True)
    assert completed.returncode == 0, completed.stderr
    public_tree = subprocess.run([sys.executable, "-B", str(ROOT / "scripts/check_public_tree.py")],
                                 cwd=ROOT, capture_output=True, text=True)
    assert public_tree.returncode == 0, public_tree.stderr
    assert not list(ROOT.rglob("*.tar.gz"))


def test_renderer_camera_metadata_matches_the_implemented_affine_mapping() -> None:
    renderer = runpy.run_path(str(ROOT / "scripts/render_n16_96_transfer.py"))
    assert renderer["camera_metadata"](96) == {
        "projection": "custom affine parallel voxel mapping",
        "image_u": "x - y + (n - 1)",
        "image_v": "integer rasterization of (x + y - z) / 2 + floor(n / 2)",
        "raster_overwrite_order": "ascending x + y + z before raster overwrite",
        "cutaway": "removed octant x,y,z >= 48",
    }
    source = (ROOT / "scripts/render_n16_96_transfer.py").read_text(encoding="utf-8")
    assert "u=((x-y)+(n-1)).astype(np.int16)" in source
    assert "v=((x+y)/2-z/2+n//2).astype(np.int16); depth=x+y+z" in source
    assert "45°" not in source and "30°" not in source and "orthographic" not in source


def test_generated_benchmark_readme_lists_the_six_frozen_conditions() -> None:
    package = runpy.run_path(str(ROOT / "scripts/package_n16_96_transfer.py"))
    rendered = package["benchmark_readme"](_records()["qualification_contract"])
    assert rendered == (ROOT / "benchmarks/n16_96_transfer/README.md").read_text(encoding="utf-8")
    for condition in ("Numerical integrity", "Terminal fidelity", "Persistence",
                      "Terminal topology", "Extinction events", "Tail consistency"):
        assert f"**{condition}.**" in rendered
    for description in _records()["qualification_contract"].values():
        assert description in rendered


def test_asset_root_cli_verifies_all_eighteen_documented_assets(tmp_path: Path) -> None:
    asset_root = tmp_path / "assets"
    manifest = _asset_fixture(asset_root)
    records = tmp_path / "expected_score.json"
    records.write_text(RECORD_PATH.read_text(encoding="utf-8"), encoding="utf-8")
    records.with_name("manifest.json").write_text(manifest.read_text(encoding="utf-8"), encoding="utf-8")
    completed = subprocess.run(
        [sys.executable, "-B", str(ROOT / "scripts/verify_n16_96_transfer.py"),
         "--records", str(records), "--asset-root", str(asset_root)],
        cwd=ROOT, capture_output=True, text=True,
    )
    assert completed.returncode == 0, completed.stderr
    assert "18 external assets" in completed.stdout


def test_asset_verifier_refuses_missing_substituted_truncated_and_wrong_root(tmp_path: Path) -> None:
    verifier = runpy.run_path(str(ROOT / "scripts/verify_n16_96_transfer.py"))["verify_external_assets"]
    asset_root = tmp_path / "assets"
    manifest = _asset_fixture(asset_root)
    first = next((asset_root / "external/n16_96_transfer").rglob("*.npz"))
    first.unlink()
    try:
        verifier(manifest, asset_root)
    except ValueError as exc:
        assert "missing documented external asset" in str(exc)
    else:
        raise AssertionError("missing asset was accepted")
    _asset_fixture(asset_root)
    first = next((asset_root / "external/n16_96_transfer").rglob("*.npz"))
    original = first.read_bytes()
    first.write_bytes(b"x" * len(original))
    try:
        verifier(manifest, asset_root)
    except ValueError as exc:
        assert "SHA-256 mismatch" in str(exc)
    else:
        raise AssertionError("substituted asset was accepted")
    first.write_bytes(b"x")
    try:
        verifier(manifest, asset_root)
    except ValueError as exc:
        assert "size mismatch" in str(exc)
    else:
        raise AssertionError("truncated asset was accepted")
    wrong_root = tmp_path / "wrong-root"
    wrong_root.mkdir()
    try:
        verifier(manifest, wrong_root)
    except ValueError as exc:
        assert "missing documented external asset" in str(exc)
    else:
        raise AssertionError("wrong asset root was accepted")


def test_asset_verifier_refuses_extra_entries_only_inside_the_documented_hierarchy(tmp_path: Path) -> None:
    verifier = runpy.run_path(str(ROOT / "scripts/verify_n16_96_transfer.py"))["verify_external_assets"]
    asset_root = tmp_path / "assets"
    manifest = _asset_fixture(asset_root)
    unrelated = asset_root / "notes.txt"
    unrelated.write_text("permitted outside documented hierarchy", encoding="utf-8")
    verifier(manifest, asset_root)
    documented = asset_root / "external/n16_96_transfer"
    extra = documented / "1" / "unexpected.npz"
    extra.write_bytes(b"unexpected")
    try:
        verifier(manifest, asset_root)
    except ValueError as exc:
        assert "unexpected regular file" in str(exc)
    else:
        raise AssertionError("unexpected regular file was accepted")
    extra.unlink()
    extra_dir = documented / "unexpected-directory"
    extra_dir.mkdir()
    try:
        verifier(manifest, asset_root)
    except ValueError as exc:
        assert "unexpected directory" in str(exc)
    else:
        raise AssertionError("unexpected directory was accepted")
    extra_dir.rmdir()
    extra_link = documented / "1" / "unexpected-link.npz"
    try:
        extra_link.symlink_to(documented / "1" / "t0.npz")
    except OSError as exc:
        if os.name == "nt" and exc.winerror == 1314:
            pytest.skip(
                "symlink creation requires a privilege this account does not hold "
                "(Windows WinError 1314); the unexpected-symlink rejection path is "
                "not exercised without a symlink to create"
            )
        raise
    try:
        verifier(manifest, asset_root)
    except ValueError as exc:
        assert "unexpected symlink" in str(exc)
    else:
        raise AssertionError("unexpected symlink was accepted")


def test_renderer_executes_end_to_end_with_synthetic_inputs(tmp_path: Path) -> None:
    """Exercise the shipped all-six renderer without private arrays or metrics."""
    science = tmp_path / "science"
    for case_index in range(1, 7):
        labels = np.empty((17, 4, 4, 4), dtype=np.int64)
        for frame_index in range(17):
            labels[frame_index] = (case_index + frame_index) % 4
            labels[frame_index, :, :2, :2] = case_index % 4
        frames = np.zeros((17, 4, 4, 4, 4), dtype=np.float32)
        for frame_index in range(17):
            frames[frame_index, labels[frame_index], *np.indices((4, 4, 4))] = 1.0
        for lane, filename in (("references", "ref_frames.npz"), ("rollouts", "model_frames.npz")):
            directory = science / lane / f"C{case_index}"
            directory.mkdir(parents=True, exist_ok=True)
            np.savez(directory / filename, frames=frames)
    (science / "score").mkdir()
    (science / "score/COHORT_SCORE.json").write_text("{}\n", encoding="utf-8")
    (science / "cohort").mkdir()
    (science / "cohort/COHORT_MANIFEST.json").write_text(
        json.dumps({"expected_checkpoint_sha256": CHECKPOINT}) + "\n", encoding="utf-8"
    )
    output = tmp_path / "rendered"
    completed = subprocess.run(
        [sys.executable, "-B", str(ROOT / "scripts/render_n16_96_transfer.py"),
         "--science-root", str(science), "--out", str(output)],
        cwd=ROOT, capture_output=True, text=True,
    )
    assert completed.returncode == 0, completed.stderr
    manifest = _json(output / "asset_manifest.json")
    assert len(manifest["inputs"]) == 12
    assert manifest["steps"] == list(range(0, 3201, 200))
    assert manifest["camera"] == {
        "projection": "custom affine parallel voxel mapping",
        "image_u": "x - y + (n - 1)",
        "image_v": "integer rasterization of (x + y - z) / 2 + floor(n / 2)",
        "raster_overwrite_order": "ascending x + y + z before raster overwrite",
        "cutaway": "removed octant x,y,z >= 2",
    }
    with Image.open(output / "n16_96_unseen_cohort_reference_vs_pinn_phase.gif") as image:
        assert image.size == (1500, 800)
        assert sum(1 for _ in ImageSequence.Iterator(image)) == 17
    with Image.open(output / "n16_96_unseen_cohort_reference_vs_pinn_phase_poster.png") as image:
        assert image.size == (1500, 860)
