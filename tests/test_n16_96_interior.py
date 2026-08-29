"""Binding tests for the N16/96 interior media asset.

Every test reads the shipped artefacts; none of them regenerates the canonical
outputs in place. Re-rendering requires the externally documented sealed cohort
arrays; these tests verify the shipped bytes and bindings.
"""
from __future__ import annotations

import ast
import hashlib
import json
import runpy
from pathlib import Path

import pytest
from PIL import Image, ImageSequence


ROOT = Path(__file__).resolve().parents[1]
MEDIA = ROOT / "media/n16_96_interior"
RENDERER = ROOT / "scripts/render_n16_96_interior.py"
MANIFEST = MEDIA / "asset_manifest.json"

GIF_NAME = "n16_96_unseen_microstructure_3_interior_reference_vs_pinn_phase.gif"
POSTER_NAME = "n16_96_unseen_microstructure_3_interior_reference_vs_pinn_phase_poster.png"
CHECKPOINT_SHA256 = "9e15e6b74f4f92f37d5709a242e436f721ec5d1306a30e05e5f39794de4459dd"
PINS = {"python": "3.11.15", "numpy": "2.2.6", "matplotlib": "3.10.7", "pillow": "12.2.0",
        "scipy": "1.16.2", "imageio": "2.37.3", "freetype": "2.14.3"}
FORBIDDEN = ("terminal 3200", "terminal step 3200", "6/6 success", "all six passed",
             "all 6 passed", "never visible", "far from the boundary")
COHORT_SCORE_SHA256 = "5f6f8643ef77c81c81a5dc5b37109d1be7d2e3212dd2d0f63e6a89ec205a4dd1"
COHORT_MANIFEST_SHA256 = "973357384e676123b16c2cc17dcb5b7f2486dc898116ac062dfffd75af7d5340"
CASE_SCORE_SHA256 = "dcdc25d9aa6b1da4c571f4451ec02a3fa46e0bb5ca0d02676a4975a76086d699"
EXPECTED_SCORE_SHA256 = "9f12027fa05063df7e9dd0f6abcb5135a302fd07fe66b304905cf25fa30aaaed"

#: The README block this asset is published in. The claim locks are applied to it,
#: because a caption is a claim surface exactly as the canvas and the manifest are.
README_MARKER = "**Inside one unseen cube:**"

# Outer-face voxel counts per saved step for the three dying channels — verified
# against the sealed arrays independently three times (analysis, implementation,
# adversarial review). The manifest must reproduce them exactly.
FACE_VECTORS = {
    "reference": {"8": [14, 1] + [0] * 15, "9": [0] * 17,
                  "15": [717, 609, 463, 289, 94, 1] + [0] * 11},
    "model": {"8": [14, 1] + [0] * 15, "9": [0] * 17,
              "15": [717, 621, 496, 322, 146, 4] + [0] * 11},
}


def renderer_namespace() -> dict:
    """Execute the shipped renderer without running its CLI."""
    return runpy.run_path(str(RENDERER))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def manifest() -> dict:
    return json.loads(MANIFEST.read_text(encoding="utf-8"))


def readme_section() -> str:
    """The published README block for this asset, from its lead-in to the next rule."""
    text = (ROOT / "README.md").read_text(encoding="utf-8")
    assert README_MARKER in text, "the interior asset has no README block"
    section = text[text.index(README_MARKER):]
    for stop in ("\n---\n", "\n## "):
        cut = section.find(stop)
        if cut != -1:
            section = section[:cut]
    assert f"media/n16_96_interior/{GIF_NAME}" in section, (
        "the README block does not reference the shipped animation"
    )
    return section


def test_renderer_digest_matches_manifest() -> None:
    assert sha256(RENDERER) == manifest()["renderer_sha256"]


def test_manifest_never_lists_itself() -> None:
    record = manifest()
    assert set(record["outputs"]) == {GIF_NAME, POSTER_NAME}, sorted(record["outputs"])
    assert "asset_manifest.json" not in json.dumps(record["outputs"])
    assert sha256(MANIFEST) not in MANIFEST.read_text(encoding="utf-8"), (
        "the manifest records its own digest"
    )


def test_renderer_carries_no_absolute_paths() -> None:
    """A shipped renderer that names a producer path is not portable evidence.

    The needle is composed rather than written out, because this file is itself
    part of the scanned release payload.
    """
    assert ("/ho" + "me/") not in RENDERER.read_text(encoding="utf-8")


def test_claim_locks() -> None:
    source = RENDERER.read_text(encoding="utf-8")
    tree = ast.parse(source)
    literals = [node.value for node in ast.walk(tree)
                if isinstance(node, ast.Constant) and isinstance(node.value, str)]

    corpus = {"renderer string literals": "\n".join(literals),
              "asset_manifest.json": MANIFEST.read_text(encoding="utf-8"),
              "README.md interior section": readme_section()}

    for doc, text in corpus.items():
        low = text.lower()
        for bad in FORBIDDEN:
            assert bad not in low, f"{doc} contains the forbidden phrase {bad!r}"
        if "6/6" in low or "six of six" in low:
            assert "5/6" in low or "five of six" in low, (
                f"{doc} states 6/6 without the paired 5/6 qualification count")

    for value in literals:
        if "interp" in value.lower():
            assert value == "no_interpolation", f"renderer literal claims resampling: {value!r}"
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute):
            assert "interp" not in node.attr.lower(), f"renderer calls {node.attr}"

    # The ast scan cannot see runtime-composed strings; enumerate every text the
    # renderer can actually place on a canvas and re-apply the locks to that set.
    renderer = renderer_namespace()
    canvas_texts = [renderer["PUBLIC"], renderer["LEGEND_NOTE"], *renderer["STATUS_LINES"]]
    canvas_texts += [renderer["step_title"](index) for index in range(17)]
    composed = "\n".join(canvas_texts).lower()
    for bad in FORBIDDEN:
        assert bad not in composed, f"composed canvas text contains the forbidden phrase {bad!r}"
    if "6/6" in composed or "six of six" in composed:
        assert "5/6" in composed or "five of six" in composed, (
            "composed canvas text states 6/6 without the paired 5/6 count")


def test_manifest_bindings() -> None:
    record = manifest()
    assert record["steps"] == list(range(0, 3201, 200))
    assert record["no_interpolation"] is True
    environment = record["renderer_environment"]
    for key, value in PINS.items():
        assert environment[key] == value, f"{key} pinned to {environment[key]}, expected {value}"
    assert environment["font_file"] == "DejaVuSans.ttf"
    assert len(environment["font_sha256"]) == 64
    assert record["environment_lock_sha256"] == sha256(ROOT / "environment-media.yml")
    assert record["checkpoint_sha256"] == CHECKPOINT_SHA256

    for name, entry in record["outputs"].items():
        path = MEDIA / name
        assert path.is_file(), f"{path} is absent"
        assert path.stat().st_size == entry["bytes"]
        assert sha256(path) == entry["sha256"]

    with Image.open(MEDIA / GIF_NAME) as gif:
        assert gif.n_frames == 17
        frames = {frame.convert("RGB").tobytes() for frame in ImageSequence.Iterator(gif)}
        assert len(frames) > 1, "the animation is a single repeated frame"
        gif_size = gif.size
    with Image.open(MEDIA / POSTER_NAME) as poster:
        assert poster.size[0] == gif_size[0]
        assert poster.size[1] == gif_size[1] + 80

    visibility = record["interior_visibility"]
    assert visibility["zero_at_last_present"] is True
    assert visibility["outer_face_voxels_per_saved_step"] == FACE_VECTORS, (
        "interior_visibility drifted from the independently verified face-count vectors")

    assert record["cohort_score_sha256"] == COHORT_SCORE_SHA256
    assert record["cohort_manifest_sha256"] == COHORT_MANIFEST_SHA256
    assert record["case_score_sha256"] == CASE_SCORE_SHA256
    claims = record["claims_source"]
    assert claims["terminal_agreement_percent"] == 95.831412
    assert claims["displayed_as"] == "95.83%"
    assert claims["source"] == "benchmarks/n16_96_transfer/expected_score.json"
    assert claims["source_sha256"] == EXPECTED_SCORE_SHA256
    assert sha256(ROOT / claims["source"]) == EXPECTED_SCORE_SHA256, (
        "the displayed terminal agreement no longer cites the shipped score record")
    assert record["selection"]["selected_case"] == "C3"
    assert record["selection"]["public_name"] == "Unseen microstructure 3"
    grains = record["dying_grains"]
    assert grains["Grain 1"]["reference_first_absent_step"] == 1000
    assert grains["Grain 2"]["reference_first_absent_step"] == 1400
    assert grains["Grain 3"]["reference_first_absent_step"] == 1400
    assert grains["Grain 1"]["model_first_absent_step"] == 1000
    assert grains["Grain 2"]["model_first_absent_step"] == 1200
    assert grains["Grain 3"]["model_first_absent_step"] == 1400


def test_poster_status_box_has_three_lines() -> None:
    # The status lines are an indivisible pair plus the case line; a silent
    # truncation (e.g. a zip length mismatch) must not survive. Check pixels,
    # not source: each of the three text bands inside the box must hold glyphs.
    # Bands are chosen to exclude the box border rows/columns entirely.
    with Image.open(MEDIA / POSTER_NAME) as handle:
        poster = handle.convert("L")
    for line_no, y in enumerate((947, 969, 991), start=1):
        band = poster.crop((38, y + 6, 780, y + 22))
        dark = sum(1 for pixel in band.tobytes() if pixel < 128)
        assert dark > 300, (
            f"status line {line_no} band (y={y}) holds only {dark} dark pixels; "
            "a status line is missing from the poster")


def test_out_guard_refuses_writing_inside_the_science_root(tmp_path: Path) -> None:
    # Never point the renderer's --out at the sealed science root. Prove the two
    # properties that keep it out with zero write risk:
    # (1) guard_out refuses a destination under the science root — a direct call,
    #     which only inspects paths and exits; it can create nothing;
    # (2) main() calls guard_out before any call that can create or write.
    science = tmp_path / "science_root"
    (science / "references").mkdir(parents=True)
    renderer = renderer_namespace()
    guard_out = renderer["guard_out"]

    refused = science / "media" / "_should_never_exist"
    with pytest.raises(SystemExit) as raised:
        guard_out(refused, science)
    assert raised.value.code == 2
    assert not refused.exists(), f"{refused} was created inside the science root"

    # The guard must not be vacuous: a destination outside the root is allowed.
    guard_out(tmp_path / "out", science)

    main_fn = next(node for node in ast.walk(ast.parse(RENDERER.read_text(encoding="utf-8")))
                   if isinstance(node, ast.FunctionDef) and node.name == "main")
    calls = []
    for node in ast.walk(main_fn):
        if isinstance(node, ast.Call):
            name = node.func.id if isinstance(node.func, ast.Name) else (
                node.func.attr if isinstance(node.func, ast.Attribute) else None)
            if name:
                calls.append((node.lineno, name))
    guard_lines = [line for line, name in calls if name == "guard_out"]
    write_lines = [line for line, name in calls
                   if name in ("mkdir", "save", "write_text", "write_bytes", "open")]
    assert guard_lines, "main() never calls guard_out"
    assert write_lines, "expected main() to contain mkdir/save/write calls"
    assert min(guard_lines) < min(write_lines), (
        "main() must call guard_out before any call that can create or write")
