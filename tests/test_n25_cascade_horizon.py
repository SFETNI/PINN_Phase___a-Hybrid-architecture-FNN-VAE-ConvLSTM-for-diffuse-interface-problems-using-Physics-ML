"""The 25-grain cascade animation carries the same horizon annotation as the 64-grain one.

The checks mirror the 64-grain ones in ``test_public_artifacts`` at the 25-grain
animation's own numbers: the represented training horizon of the accepted
checkpoint is 4,096 steps, the rollout runs to 12,000, and the boundary falls
strictly between the 4,000 and 6,000 saved frames. Every claim is read back from
the rendered pixels or from the renderer's declared mapping, never from a document.
"""

from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[1]

HORIZON_STEP = 4096
FINAL_STEP = 12000
EXTRAPOLATION_STEPS = FINAL_STEP - HORIZON_STEP
PHASES = 25
GIF = "media/current_results/n25_cascade_reference_vs_pinn_phase.gif"
POSTER = "media/current_results/n25_cascade_reference_vs_pinn_phase_poster.png"

# Geometry of the annotated figure, in figure fractions, as the renderer places it.
# Identical to the 64-grain layout: the two animations share one frame template.
AXIS_LEFT, AXIS_WIDTH = 0.055, 0.890
AXIS_BOTTOM, AXIS_HEIGHT = 0.200, 0.235
INK_RGB = (0x3b, 0x40, 0x45)
MODEL_BLUE_RGB = (0x45, 0x75, 0xb4)
DISCREPANCY_RED_RGB = (0xd7, 0x30, 0x27)
TAIL_TINT_RGB = (0xf3, 0xed, 0xe2)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _renderer():
    """The renderer's declared constants, read without importing it."""
    source = (ROOT / "scripts/render_current_results.py").read_text(encoding="utf-8")
    wanted = {"HORIZON_ANNOTATION", "N25_HORIZON_ANNOTATION", "N25_MILESTONES",
              "ANNOTATION_PT", "STATUS_PT", "REGION_TITLE_PT", "BOUNDARY_PT",
              "REFERENCE_DASHES", "REFERENCE_WIDTH", "MODEL_WIDTH"}
    found = {}
    for node in ast.parse(source).body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            target = node.targets[0]
            if isinstance(target, ast.Name) and target.id in wanted:
                found[target.id] = ast.literal_eval(node.value)
    missing = wanted - set(found)
    assert not missing, f"renderer no longer declares: {sorted(missing)}"
    manifest = json.loads(
        (ROOT / "media/current_results/asset_manifest.json").read_text(encoding="utf-8")
    )
    found["MEASURED"] = manifest["measurements"]["n25_cascade_reference_vs_pinn_phase"]
    found["MANIFEST"] = manifest
    return SimpleNamespace(**found)


def _frames(indices):
    from PIL import Image, ImageSequence

    wanted, out = set(indices), {}
    with Image.open(ROOT / GIF) as animation:
        for index, frame in enumerate(ImageSequence.Iterator(animation)):
            if index in wanted:
                out[index] = np.asarray(frame.convert("RGB"), dtype=np.float64) / 255.0
    assert set(out) == wanted
    return out


def _poster():
    from PIL import Image

    with Image.open(ROOT / POSTER) as poster:
        return np.asarray(poster.convert("RGB"), dtype=np.float64) / 255.0


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


def test_both_animations_declare_the_same_annotation_shape() -> None:
    """One frame template, two mappings: the drawing code reads only the mapping."""
    module = _renderer()
    n64, n25 = module.HORIZON_ANNOTATION, module.N25_HORIZON_ANNOTATION
    assert set(n64) == set(n25)
    # What is shared is shared verbatim; what differs is exactly the horizon.
    for key in ("within_title", "beyond_title", "reference_label", "model_label",
                "continuity_note", "within_status", "beyond_status", "footer",
                "disagreement_headroom", "grain_headroom", "final_step"):
        assert n64[key] == n25[key], key
    assert n25["horizon_step"] == HORIZON_STEP and n64["horizon_step"] == 8192
    assert n25["boundary_label"] == "H = 4,096"
    assert n25["status_qualifier"] == "development evidence · trained on this initial condition"
    assert n25["boundary_label_side"] == "left" and n64["boundary_label_side"] == "right"


def test_horizon_boundary_is_the_exact_constant_and_not_a_saved_frame() -> None:
    module = _renderer()
    milestones = tuple(module.N25_MILESTONES)
    assert milestones == tuple(module.MEASURED["recomputed"]["steps"])
    assert HORIZON_STEP not in milestones
    assert max(s for s in milestones if s < HORIZON_STEP) == 4000
    assert min(s for s in milestones if s > HORIZON_STEP) == 6000
    recorded = module.MEASURED["horizon_annotation"]
    assert recorded["horizon_step"] == HORIZON_STEP
    assert recorded["final_step"] == FINAL_STEP
    assert recorded["extrapolation_steps"] == EXTRAPOLATION_STEPS == 7904
    assert recorded["last_frame_within_horizon"] == 4000
    assert recorded["first_frame_beyond_horizon"] == 6000
    assert recorded["no_restart_at_boundary"] is True


def test_boundary_is_drawn_at_4096_and_not_at_a_neighbouring_frame() -> None:
    """Read the drawn boundary back out of the rendered pixels."""
    pixels = _poster()
    box = _axis_box(pixels)
    row = round(box["y"](0.66))
    left, right = round(box["x"](0)), round(box["x"](FINAL_STEP))
    strip = pixels[row, left + 4:right - 4]
    tint = np.array(TAIL_TINT_RGB, dtype=np.float64) / 255.0
    matched = np.where(np.abs(strip - tint).max(axis=1) < 0.02)[0]
    assert matched.size, "the extrapolation band was not found in the rendered panel"
    recovered = FINAL_STEP * (int(matched[0]) + 4) / (right - left)
    assert abs(recovered - HORIZON_STEP) < 25, f"band edge recovered at {recovered:.0f}"
    assert abs(recovered - 4000) > 60
    assert abs(recovered - 6000) > 100


def test_extrapolation_interval_is_stated_as_7904_steps_everywhere() -> None:
    module = _renderer()
    annotation = module.N25_HORIZON_ANNOTATION
    assert annotation["final_step"] - annotation["horizon_step"] == 7904
    beyond = " ".join(annotation["beyond_detail"])
    assert "4,097" in beyond and "12,000" in beyond
    assert "0–4,096" in " ".join(annotation["within_detail"])
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    assert readme.count("7,904") >= 2
    assert "4,096-step horizon" in readme


def test_both_regions_carry_a_persistent_label_in_every_frame() -> None:
    """The animation must explain its own backgrounds without the page."""
    module = _renderer()
    frames = _frames(range(len(module.N25_MILESTONES)))
    assert len(frames) == 13
    for index, pixels in frames.items():
        box = _axis_box(pixels)
        for centre, name in ((HORIZON_STEP / 2, "within"),
                             ((HORIZON_STEP + FINAL_STEP) / 2, "beyond")):
            half = round(0.11 * pixels.shape[1])
            column = round(box["x"](centre))
            # The title and the detail lines are checked in their own bands, so
            # blanking either one is caught: a single window over both would be
            # satisfied by the surviving line alone. Measured on the shipped
            # frames the title band never carries fewer than 1,419 ink pixels
            # and the detail band never fewer than 775.
            bands = (("title", 0.93, 0.855, 600), ("detail", 0.85, 0.68, 300))
            for part, upper, lower, floor in bands:
                top, bottom = round(box["y"](upper)), round(box["y"](lower))
                patch = pixels[top:bottom, max(column - half, 0):column + half]
                ink = int(_near(patch, INK_RGB, tolerance=0.30).sum())
                assert ink > floor, (
                    f"frame {index}: the {name} region {part} is missing or faint "
                    f"({ink} ink pixels)"
                )


def test_region_and_boundary_labels_are_readable_at_the_readme_width() -> None:
    from PIL import Image

    module = _renderer()
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    assert 'n25_cascade_reference_vs_pinn_phase.gif" width="900"' in readme
    with Image.open(ROOT / POSTER) as poster:
        native_width = poster.width
    scale = 900 / native_width
    for name, points in (("supporting text", module.ANNOTATION_PT),
                         ("region title", module.REGION_TITLE_PT),
                         ("boundary label", module.BOUNDARY_PT),
                         ("status line", module.STATUS_PT)):
        displayed = points * (110 / 72) * scale
        assert displayed >= 9.0, f"{name} renders at {displayed:.1f} px at 900 px wide"


def test_nothing_claims_training_or_labels_during_the_displayed_rollout() -> None:
    module = _renderer()
    annotation = module.N25_HORIZON_ANNOTATION
    frame_text = " ".join(
        [annotation["within_status"], annotation["beyond_status"],
         annotation["boundary_label"], annotation["continuity_note"],
         annotation["within_title"], annotation["beyond_title"],
         annotation["reference_label"], annotation["model_label"],
         annotation["status_qualifier"],
         *annotation["within_detail"], *annotation["beyond_detail"],
         *annotation["footer"]]
    ).lower()
    for forbidden in ("training data", "training region", "training zone",
                      "seen data", "in-sample", "out-of-sample", "ground truth",
                      "restart", "retrain", "fine-tune", "untrusted", "invalid",
                      "fails beyond", "supervised on the reference",
                      "trained on the reference", "prospective", "unseen"):
        assert forbidden not in frame_text, forbidden
    assert "no reference frames after step 0" in frame_text
    assert "supervised only by physics residuals" in frame_text
    assert "does not mean the model was shown these states" in frame_text
    assert "the reference is fixed and drawn complete" in frame_text
    assert "only the rollout advances" in frame_text
    # This case is development evidence on the training initial condition, and
    # the frame says so rather than leaving it to the page.
    assert "development evidence" in frame_text
    assert "trained on this initial condition" in frame_text

    readme = " ".join((ROOT / "README.md").read_text(encoding="utf-8").split())
    section = readme[readme.index("### The 25-grain cascade"):readme.index("## Dense 64-grain")]
    assert "within the represented training horizon" in section
    assert "autonomous extrapolation" in section
    assert "The whole rollout is autonomous" in section
    assert "does not mean the model was shown these states" in section
    assert "held-out comparator used for evaluation only" in section
    assert "Both cases are development evidence" in section
    # The alt text names what the frame draws.
    alt = section[section.index('alt="') + 5:section.index('">', section.index('alt="'))]
    for phrase in ("H = 4,096", "Within represented training horizon",
                   "Autonomous extrapolation", "4,097 to 12,000", "evaluation only",
                   "Only the rollout advances", "nothing restarts there"):
        assert phrase in alt, phrase


def test_only_the_rollout_advances_and_the_reference_stays_fixed() -> None:
    module = _renderer()
    annotation = module.N25_HORIZON_ANNOTATION
    milestones = list(module.N25_MILESTONES)
    early, late = milestones.index(2000), milestones.index(6000)
    frames = _frames([early, late])
    pixels = frames[early]
    box = _axis_box(pixels)

    top, bottom = round(box["y"](0.65)), round(box["y"](0.02))
    rule = round(box["x"](HORIZON_STEP))
    columns = np.r_[round(box["x"](2400)):rule - 30, rule + 30:round(box["x"](FINAL_STEP))]
    ahead = pixels[top:bottom, columns]
    rolling = ((DISCREPANCY_RED_RGB, "differing-pixel", 0.22),
               (MODEL_BLUE_RGB, "prediction", 0.14))
    for rgb, name, tolerance in rolling:
        found = int(_near(ahead, rgb, tolerance=tolerance).sum())
        assert found < 40, f"{name} ink appears ahead of the playhead at step 2,000 ({found} px)"
    assert int(_near(ahead, INK_RGB, tolerance=0.10).sum()) > 500, (
        "the reference does not span the axis on an early frame"
    )

    far = slice(round(box["x"](10600)), round(box["x"](11800)))
    windows = [frames[at][top:bottom, far] for at in (early, late)]
    masks = [_near(window, INK_RGB, tolerance=0.10) for window in windows]
    assert masks[0].sum() > 40 and np.array_equal(*masks), (
        "the reference moves between frames; it is not fixed"
    )
    assert np.abs(windows[0] - windows[1]).max() < 0.05, "the reference is restyled mid-animation"
    terminal = _frames([len(milestones) - 1])[len(milestones) - 1][top:bottom, far]
    assert np.abs(windows[0] - terminal).max() > 0.3, "the window is insensitive to what is drawn in it"

    recomputed = module.MEASURED["recomputed"]
    at = milestones.index(1000)
    ceiling = max(recomputed["differing_pixels_pct"]) * annotation["disagreement_headroom"]
    grain_ceiling = PHASES * annotation["grain_headroom"]
    columns = slice(round(box["x"](800)), round(box["x"](1200)))
    behind_inks = (*rolling, (INK_RGB, "reference", 0.10))
    heights = (recomputed["differing_pixels_pct"][at] / ceiling,
               recomputed["model_active_grains"][at] / grain_ceiling,
               recomputed["reference_active_grains"][at] / grain_ceiling)
    for (rgb, name, tolerance), fraction in zip(behind_inks, heights):
        row = round(box["y"](fraction))
        behind = pixels[row - 8:row + 9, columns]
        assert _near(behind, rgb, tolerance=tolerance).sum() > 20, f"no {name} ink behind"


def test_marker_sits_on_the_step_the_frame_is_showing() -> None:
    module = _renderer()
    milestones = list(module.N25_MILESTONES)
    model_active = module.MEASURED["recomputed"]["model_active_grains"]
    grain_ceiling = PHASES * module.N25_HORIZON_ANNOTATION["grain_headroom"]
    checked = [milestones.index(step) for step in (2000, 6000, 9000, 11000)]
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
        assert here >= 120, f"frame {index}: no marker at step {milestones[index]:,}"
        assert here > 1.6 * previous, (
            f"frame {index}: marker is not distinctly at the current step "
            f"({here} here against {previous} at the previous step)"
        )


def test_field_panels_do_not_change_because_the_boundary_is_crossed() -> None:
    """Frames at 4,000 and 6,000 straddle step 4,096; the panel surround must not react."""
    module = _renderer()
    milestones = list(module.N25_MILESTONES)
    before, after = milestones.index(4000), milestones.index(6000)
    assert milestones[before] <= HORIZON_STEP < milestones[after]
    frames = _frames([before, after])
    height, width = frames[before].shape[:2]
    top, bottom = round(0.126 * height), round(0.480 * height)
    gaps = np.r_[0:round(0.055 * width),
                 round(0.320 * width):round(0.370 * width),
                 round(0.635 * width):round(0.685 * width),
                 round(0.945 * width):width]
    difference = np.abs(frames[before][top:bottom, gaps]
                        - frames[after][top:bottom, gaps]).max()
    assert difference < 0.02, f"the panel surround changed across the boundary by {difference:.3f}"


def test_two_regions_survive_grayscale_and_colour_vision() -> None:
    pixels = _poster()
    box = _axis_box(pixels)

    def patch(step):
        column = round(box["x"](step))
        row = round(box["y"](0.66))
        return pixels[row - 6:row + 6, column - 14:column + 14].reshape(-1, 3)

    inside, tail = patch(1500), patch(11000)
    assert _luminance(inside).mean() - _luminance(tail).mean() > 0.04
    collapsed_inside = inside[:, [2]].mean() + inside[:, :2].mean()
    collapsed_tail = tail[:, [2]].mean() + tail[:, :2].mean()
    assert abs(collapsed_inside - collapsed_tail) > 0.03


def test_current_25_grain_media_match_their_recorded_digests() -> None:
    module = _renderer()
    for name in ("n25_cascade_reference_vs_pinn_phase.gif",
                 "n25_cascade_reference_vs_pinn_phase_poster.png"):
        recorded = module.MANIFEST["assets"][name]["sha256"]
        assert len(recorded) == 64
        assert _sha256(ROOT / "media/current_results" / name) == recorded, name
    # The annotation did not change what is measured from the arrays.
    recomputed = module.MEASURED["recomputed"]
    assert recomputed["reference_active_grains"][0] == 25
    assert recomputed["reference_active_grains"][-1] == 16
    assert recomputed["model_active_grains"][-1] == 16
    assert abs(recomputed["differing_pixels_pct"][-1] - 1.4648) < 1e-3
