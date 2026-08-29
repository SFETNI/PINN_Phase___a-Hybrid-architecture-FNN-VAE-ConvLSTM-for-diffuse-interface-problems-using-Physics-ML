"""Every media manifest must agree with the renderer and inputs actually shipped.

A media manifest records the digest of the script that produced it, computed by that
script over its own bytes at render time. That makes the field self-referential: edit
the renderer for any reason -- even a reason that cannot change a pixel -- and the
recorded digest stops describing the distributed file unless the media are rendered
again. Nothing about the images looks wrong when this happens, so only a test that
compares the recorded digest against the shipped renderer's bytes will notice.

The rule is applied to every media manifest that names a renderer, uniformly. A
manifest that is exempted by hand is a manifest that can rot, which is precisely how
the ``media/n25_transfer`` renderer digest came to disagree with its script.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
MEDIA = ROOT / "media"

#: Media manifest -> the script that renders it. ``renderer_sha256`` is written by that
#: script over its own bytes, so this mapping is what turns a self-referential field
#: into a checkable one.
RENDERERS = {
    "media/n25_transfer/asset_manifest.json": "scripts/render_n25_transfer.py",
    "media/current_results/asset_manifest.json": "scripts/render_current_results.py",
    "media/n16_96_transfer/asset_manifest.json": "scripts/render_n16_96_transfer.py",
    "media/n16_96_interior/asset_manifest.json": "scripts/render_n16_96_interior.py",
}

#: Manifests that record rendered files under ``outputs`` rather than ``assets``.
#: The two spellings are historical; both name shipped files and both are checked.
OUTPUT_KEYED_MANIFESTS = frozenset(
    {
        "media/n16_96_transfer/asset_manifest.json",
        "media/n16_96_interior/asset_manifest.json",
    }
)

#: Media manifest -> {manifest field: the distributed file it names}. Only fields that
#: name a *shipped* file belong here; digests of accepted artifacts that are not
#: distributed are covered by the identity ledger, not by this test.
INPUT_MANIFESTS = {
    "media/n25_transfer/asset_manifest.json": {
        "input_manifest_sha256": "benchmarks/n25_transfer/manifest.json",
    },
    "media/current_results/asset_manifest.json": {},
    "media/n16_96_transfer/asset_manifest.json": {},
    "media/n16_96_interior/asset_manifest.json": {},
}

# The N16 renderers bind sealed external JSON identities.  They are not shipped
# as public files, so their equality to the compact public score record is checked
# by test_n16_96_transfer rather than pretending they digest a distributed file.
# The cohort-manifest digest is syntactically an input-manifest digest; the score
# digests are separate sealed-source fields and must not be lost by a suffix-only
# discovery rule. The interior renderer additionally binds the per-case score.
EXTERNAL_MANIFEST_IDENTITIES = {
    "media/n25_transfer/asset_manifest.json": set(),
    "media/current_results/asset_manifest.json": set(),
    "media/n16_96_transfer/asset_manifest.json": {
        "cohort_manifest_sha256",
    },
    "media/n16_96_interior/asset_manifest.json": {
        "cohort_manifest_sha256",
    },
}
EXTERNAL_SEALED_IDENTITIES = {
    "media/n25_transfer/asset_manifest.json": set(),
    "media/current_results/asset_manifest.json": set(),
    "media/n16_96_transfer/asset_manifest.json": {
        "cohort_manifest_sha256", "cohort_score_sha256",
    },
    "media/n16_96_interior/asset_manifest.json": {
        "cohort_manifest_sha256", "cohort_score_sha256", "case_score_sha256",
    },
}

N25_MEDIA = (
    "terminal_atlas.png",
    "id_02_reference_vs_pinn_phase.gif",
    "id_02_reference_vs_pinn_phase_poster.png",
)

#: The rendering stack the distributed media were produced with, established by
#: rerunning the shipped renderer across a matrix of versions.
#:
#: Two different components bind two different properties, and conflating them
#: produces a test that either fails for honest readers or proves nothing:
#:
#: * FreeType rasterizes the glyphs. A different FreeType moves roughly 1% of the
#:   pixels -- axis labels, tick text, titles -- with no change to the fields being
#:   plotted. matplotlib's own wheels vendor FreeType 2.6.1 while conda-forge builds
#:   against 2.14.3, so the same matplotlib version renders differently depending on
#:   where it came from.
#: * Pillow encodes the PNGs. A different Pillow emits a different compressed stream
#:   for pixel-identical images, changing the file digest and nothing a reader sees.
#:
#: So pixel equality is checked whenever the rasterizer matches, and byte equality
#: only when the encoder matches too.
RASTERIZER = {"matplotlib": "3.10.7", "freetype": "2.14.3"}
ENCODER = {"pillow": "12.2.0"}


def _rendering_stack() -> dict[str, str]:
    import matplotlib
    import PIL
    from matplotlib import ft2font

    return {
        "matplotlib": matplotlib.__version__,
        "freetype": ft2font.__freetype_version__,
        "pillow": PIL.__version__,
    }


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _frames(path: Path) -> tuple[bytes, ...]:
    """Decoded RGBA pixels of a still or animation, independent of how it is encoded."""
    from PIL import Image, ImageSequence

    with Image.open(path) as image:
        return tuple(
            frame.convert("RGBA").tobytes()
            for frame in ImageSequence.Iterator(image)
        )


def _manifest_paths() -> set[str]:
    lines = (ROOT / "MANIFEST.sha256").read_text(encoding="utf-8").splitlines()
    return {line.split("  ", 1)[1] for line in lines if line.strip()}


def _media_manifests_naming_a_renderer() -> set[str]:
    found = set()
    for path in sorted(MEDIA.rglob("*.json")):
        record = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(record, dict) and "renderer_sha256" in record:
            found.add(path.relative_to(ROOT).as_posix())
    return found


def test_every_media_manifest_naming_a_renderer_is_covered() -> None:
    """No manifest may name a renderer without being subject to the rule below."""
    assert _media_manifests_naming_a_renderer() == set(RENDERERS)


@pytest.mark.parametrize("manifest_path", sorted(RENDERERS))
def test_recorded_renderer_digest_matches_the_distributed_renderer(
    manifest_path: str,
) -> None:
    record = json.loads((ROOT / manifest_path).read_text(encoding="utf-8"))
    renderer = RENDERERS[manifest_path]
    assert record["renderer_sha256"] == _sha256(ROOT / renderer), (
        f"{manifest_path} records a renderer digest that is not the digest of the "
        f"distributed {renderer}. Re-render the media with the shipped script."
    )
    assert renderer in _manifest_paths(), f"{renderer} is named but not distributed"


@pytest.mark.parametrize("manifest_path", sorted(INPUT_MANIFESTS))
def test_recorded_input_manifest_digests_match_the_distributed_inputs(
    manifest_path: str,
) -> None:
    record = json.loads((ROOT / manifest_path).read_text(encoding="utf-8"))
    bindings = INPUT_MANIFESTS[manifest_path]
    declared = {key for key in record if key.endswith("_manifest_sha256")}
    assert declared == set(bindings) | EXTERNAL_MANIFEST_IDENTITIES[manifest_path], (
        f"{manifest_path} names input manifests {sorted(declared)} but the test binds "
        f"{sorted(bindings)}"
    )
    for field, target in bindings.items():
        assert target in _manifest_paths(), f"{target} is named but not distributed"
        assert record[field] == _sha256(ROOT / target), (
            f"{manifest_path}:{field} disagrees with the distributed {target}"
        )


@pytest.mark.parametrize("manifest_path", sorted(RENDERERS))
def test_external_sealed_identity_fields_are_complete(manifest_path: str) -> None:
    """Known external identities are explicit rather than silently suffix-filtered."""
    record = json.loads((ROOT / manifest_path).read_text(encoding="utf-8"))
    expected = EXTERNAL_SEALED_IDENTITIES[manifest_path]
    actual = {field for field in expected if field in record}
    assert actual == expected


@pytest.mark.parametrize("manifest_path", sorted(RENDERERS))
def test_recorded_asset_digests_match_the_distributed_media(manifest_path: str) -> None:
    record = json.loads((ROOT / manifest_path).read_text(encoding="utf-8"))
    directory = (ROOT / manifest_path).parent
    if manifest_path in OUTPUT_KEYED_MANIFESTS:
        for name, entry in record["outputs"].items():
            asset = directory / name
            assert asset.is_file()
            assert entry["sha256"] == _sha256(asset)
        return
    for name, entry in record["assets"].items():
        asset = directory / name
        assert asset.is_file(), f"{manifest_path} names a missing asset {name}"
        assert entry["sha256"] == _sha256(asset), f"{manifest_path}:{name} digest stale"


def test_n25_transfer_renderer_reproduces_its_media_and_manifest(
    tmp_path: Path,
) -> None:
    """The distributed renderer, rerun, must reproduce what it is recorded as producing.

    This is the property the recorded digest is supposed to stand for: not merely that
    the renderer's digest is written down correctly, but that the shipped renderer is
    in fact the one that produces the shipped images.

    It is checked at whatever strength the reader's rendering stack supports, and the
    skip message says what a stricter check would require. Rendering is an optional
    extra, so this skips entirely where the media dependencies are absent -- the digest
    bindings above run everywhere and do not depend on any of this.
    """
    pytest.importorskip("matplotlib", reason="media extra not installed")
    pytest.importorskip("imageio", reason="media extra not installed")
    pytest.importorskip("PIL", reason="media extra not installed")

    stack = _rendering_stack()
    mismatched = {k: v for k, v in RASTERIZER.items() if stack[k] != v}
    if mismatched:
        pytest.skip(
            "rendering stack differs from the one that produced the distributed "
            f"media: {mismatched} (this reader has "
            f"{ {k: stack[k] for k in mismatched} }). Glyph rasterization differs "
            "across FreeType builds, so re-rendering here would not be expected to "
            "reproduce the shipped pixels."
        )

    output = tmp_path / "media"
    completed = subprocess.run(
        [
            sys.executable,
            "-B",
            str(ROOT / "scripts/render_n25_transfer.py"),
            "--benchmark-root",
            str(ROOT / "benchmarks/n25_transfer"),
            "--output",
            str(output),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr

    shipped = ROOT / "media/n25_transfer"

    # What a reader actually looks at, checked wherever the rasterizer matches.
    for name in N25_MEDIA:
        assert _frames(output / name) == _frames(shipped / name), (
            f"re-rendering changed the pixels of {name}; a changed image is a stop, "
            "not a reason to update a digest"
        )

    if any(stack[key] != value for key, value in ENCODER.items()):
        pytest.skip(
            f"pixels reproduced exactly; byte identity additionally requires {ENCODER} "
            f"(this reader has { {k: stack[k] for k in ENCODER} }), because the PNG "
            "encoder chooses the compressed representation of identical pixels."
        )

    for name in N25_MEDIA:
        assert (output / name).read_bytes() == (shipped / name).read_bytes(), (
            f"re-rendering changed {name}; a changed image byte is a stop, not a "
            "reason to update a digest"
        )
    assert (output / "asset_manifest.json").read_text(encoding="utf-8") == (
        shipped / "asset_manifest.json"
    ).read_text(encoding="utf-8")
