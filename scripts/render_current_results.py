#!/usr/bin/env python3
"""Render the current 25-grain and 64-grain result animations from accepted arrays.

Both animations carry the same represented-horizon annotation: the diagnostic
panel is divided at the checkpoint's represented training horizon and labels
both sides in the frame itself. The 25-grain cascade draws its boundary at
4,096 and the 64-grain sensitivity run at 8,192; the two mappings below are the
only places those numbers live.

The scientific arrays are NOT distributed in this release: they are large frozen
artifacts held outside the public tree. This script therefore takes their
locations as arguments and verifies each one against the SHA-256 recorded below
before drawing anything. A digest identifies an artifact independently of where
it happens to sit on any filesystem, so no location is baked into this file.

Every number printed on a frame is recomputed from the arrays at render time.
None is transcribed from a document.

Usage:

    python scripts/render_current_results.py \
        --n25-reference PATH --n25-model PATH \
        --n64-reference PATH --n64-model PATH \
        --output media/current_results
"""

from __future__ import annotations

import argparse
import colorsys
import hashlib
import io
import json
from pathlib import Path
from typing import Any

import imageio.v2 as imageio
import matplotlib
import matplotlib.legend
import matplotlib.pyplot as plt
import matplotlib.text
from matplotlib.colors import ListedColormap
from matplotlib.transforms import blended_transform_factory
import numpy as np
from PIL import Image


# Accepted source artifacts, bound by digest. Roles are descriptive; identity is
# the digest, never the file name.
EXPECTED_SHA256 = {
    "n25_reference": "351a55961ee187e327ba1a23c0f566350959eed2ebe8d9473945cc47158fd511",
    "n25_model": "02f11f936fac16f206897a4023bc613ccf098834e8cfed39fb0ab18007208b4d",
    "n64_reference": "96d4a5422344a5a458f12e4dae154e9843f1407436b2dc461f7379e25891ffbe",
    "n64_model": "ea45d058f68b20daf479cec0c9da14927d6033fbcd16bbbaf4c2a591b6d72d35",
}

N25_MILESTONES = (0, 250, 500, 1000, 2000, 4000, 6000, 7000, 8000, 9000, 10000, 11000, 12000)
N64_MILESTONES = tuple(range(0, 12001, 500))

DISCREPANCY_RED = "#d73027"
INK = "#3b4045"
MODEL_BLUE = "#4575b4"
TAIL_TINT = "#f3ede2"

# Smallest type used for any label this renderer adds. At 110 dpi a 10 pt label
# is 15.3 px, which survives the README's 900 px display width at roughly 10 px.
# Both grain-count series are ordinary plotting weight. The reference is dashed
# and drawn over the prediction's solid stroke, so where the two coincide the
# prediction shows through the dash gaps; drawing the solid one on top instead
# would hide the dashed one completely wherever they agree. The dash pattern is
# the separation, so it survives desaturation without either series being
# displaced numerically.
REFERENCE_WIDTH = 1.7
REFERENCE_DASHES = (0, (4.4, 3.0))
MODEL_WIDTH = 1.9

ANNOTATION_PT = 10.0
STATUS_PT = 11.0
REGION_TITLE_PT = 11.5
BOUNDARY_PT = 11.0

# The 64-grain animation shows a checkpoint whose represented training horizon
# was extended to 8,192 steps and which was then rolled autonomously to 12,000.
# These are properties of the run, declared here as configuration; every number
# printed from the arrays is still recomputed at render time.
#
# The boundary is a number, not a frame. It falls strictly between the 8,000 and
# 8,500 milestones and is drawn at its true position on the continuous axis. It
# marks a change in the rollout's relation to training and nothing else: there is
# no restart and no physical discontinuity there, which is why nothing about the
# field panels, the curve styling or the frame pacing responds to it.
#
# Every consumer reads the boundary out of this mapping. A second copy of the
# number elsewhere is a copy that can drift from the one that is drawn.
HORIZON_ANNOTATION = {
    "horizon_step": 8192,
    "final_step": 12000,
    "boundary_label": "H = 8,192",
    # The label sits on the side with room for it. The boundary is 68% of the way
    # across this axis, so the label goes right and the continuity note left.
    "boundary_label_side": "right",
    "within_title": "Within represented training horizon",
    "within_detail": ("Autonomous rollout · steps 0–8,192",),
    "beyond_title": "Autonomous extrapolation",
    "beyond_detail": ("Beyond represented horizon", "steps 8,193–12,000"),
    "reference_label": "PF reference (evaluation only)",
    "model_label": "PINN-Phase autonomous rollout",
    "continuity_note": "rollout continuous here — only the relation to training changes",
    "within_status": "within represented training horizon · autonomous rollout",
    "beyond_status": "beyond represented horizon · autonomous extrapolation",
    "status_qualifier": "not the registered primary result",
    # Headroom above the drawn series, so the region labels sit in empty space
    # rather than on top of a curve. A caption laid over a curve is read as a
    # caption about that curve.
    "disagreement_headroom": 1.95,
    "grain_headroom": 1.75,
    "footer": (
        "Training used no reference frames after step 0: the model was supervised only by "
        "physics residuals on its own rollout.",
        "“Within the represented horizon” does not mean the model was shown these states. "
        "The reference is fixed and drawn complete, for evaluation; only the rollout advances.",
    ),
}

# The 25-grain cascade shows the accepted checkpoint trained on this initial
# condition with a represented training horizon of 4,096 steps, rolled
# autonomously to 12,000. As above, the boundary is a number rather than a
# frame: it falls strictly between the 4,000 and 6,000 milestones. Training
# used no reference frames after step 0 for this checkpoint either, so the
# footer is the same statement. Everything that differs between the two
# animations is a value in this mapping; the drawing code reads only the
# mapping and never a literal.
N25_HORIZON_ANNOTATION = {
    "horizon_step": 4096,
    "final_step": 12000,
    "boundary_label": "H = 4,096",
    # The boundary is a third of the way across this axis, so the label goes
    # left, into the represented region, and the continuity note right.
    "boundary_label_side": "left",
    "within_title": "Within represented training horizon",
    "within_detail": ("Autonomous rollout · steps 0–4,096",),
    "beyond_title": "Autonomous extrapolation",
    "beyond_detail": ("Beyond represented horizon", "steps 4,097–12,000"),
    "reference_label": "PF reference (evaluation only)",
    "model_label": "PINN-Phase autonomous rollout",
    "continuity_note": "rollout continuous here — only the relation to training changes",
    "within_status": "within represented training horizon · autonomous rollout",
    "beyond_status": "beyond represented horizon · autonomous extrapolation",
    "status_qualifier": "development evidence · trained on this initial condition",
    "disagreement_headroom": 1.95,
    "grain_headroom": 1.75,
    "footer": (
        "Training used no reference frames after step 0: the model was supervised only by "
        "physics residuals on its own rollout.",
        "“Within the represented horizon” does not mean the model was shown these states. "
        "The reference is fixed and drawn complete, for evaluation; only the rollout advances.",
    ),
}


class SourceConflict(RuntimeError):
    """A source did not match its pinned digest. Building on drift is worse than not building."""


class UnreadableFrame(RuntimeError):
    """A label fell off the canvas or onto another label.

    A truncated or overprinted sentence on a public front page is worse than no
    sentence, and a later wording change is exactly what would cause one.
    """


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify(role: str, path: Path) -> str:
    if not path.is_file():
        raise SourceConflict(f"{role}: source not found at the given location")
    found = sha256_file(path)
    expected = EXPECTED_SHA256[role]
    if found != expected:
        raise SourceConflict(f"{role}: expected {expected[:16]}…, found {found[:16]}…")
    return found


def stable_grain_colors(count: int) -> ListedColormap:
    """One colour per phase index, fixed for all time.

    Recolouring per frame would make a grain change colour when a neighbour dies,
    which reads as a topology change that did not happen. Hues near the
    discrepancy red are skipped so no grain can be mistaken for the overlay.
    """
    colors = []
    index = 0
    while len(colors) < count:
        hue = (index * 0.6180339887498949) % 1.0
        index += 1
        if min(abs(hue), abs(hue - 1.0)) < 0.045:
            continue
        light = 0.58 + 0.16 * (((len(colors) * 7) % 5) / 4.0 - 0.5)
        sat = 0.52 + 0.30 * (((len(colors) * 3) % 4) / 3.0)
        colors.append(colorsys.hls_to_rgb(hue, light, sat))
    return ListedColormap(colors)


def boundaries(labels: np.ndarray) -> np.ndarray:
    """One-pixel grain boundaries, so adjacent grains stay separable in grayscale."""
    edge = np.zeros(labels.shape, dtype=bool)
    edge[:-1, :] |= labels[:-1, :] != labels[1:, :]
    edge[:, :-1] |= labels[:, :-1] != labels[:, 1:]
    return edge


def labels_from_states(states: np.ndarray) -> np.ndarray:
    return states.argmax(axis=1).astype(np.int16)


def draw_field(ax, labels: np.ndarray, cmap: ListedColormap, phases: int) -> None:
    ax.imshow(labels, cmap=cmap, vmin=0, vmax=phases - 1, interpolation="nearest")
    ax.imshow(
        np.ma.masked_where(~boundaries(labels), np.ones_like(labels)),
        cmap=ListedColormap(["#00000030"]),
        vmin=0,
        vmax=1,
        interpolation="nearest",
    )
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def draw_discrepancy(ax, reference: np.ndarray, difference: np.ndarray, phases: int) -> None:
    faint = ListedColormap([(v, v, v) for v in np.linspace(0.74, 0.95, phases)])
    ax.imshow(reference, cmap=faint, vmin=0, vmax=phases - 1, interpolation="nearest")
    ax.imshow(
        np.ma.masked_where(~difference, difference),
        cmap=ListedColormap([DISCREPANCY_RED]),
        vmin=0,
        vmax=1,
        interpolation="nearest",
    )
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def series(reference: dict[str, np.ndarray], model: dict[str, np.ndarray], steps) -> dict[str, Any]:
    disagreement, ref_active, model_active = [], [], []
    for step in steps:
        ref = reference[step]
        pred = ref if step == 0 and step not in model else model[step]
        disagreement.append(100.0 * float((pred != ref).mean()))
        ref_active.append(int(np.unique(ref).size))
        model_active.append(int(np.unique(pred).size))
    return {
        "steps": list(steps),
        "disagreement_pct": disagreement,
        "reference_active": ref_active,
        "model_active": model_active,
    }


def history_axis(ax, data: dict[str, Any], index: int, phases: int, horizon=None) -> None:
    steps = data["steps"]
    upto = slice(0, index + 1)
    if horizon is None:
        ceiling = max(max(data["disagreement_pct"]) * 1.35, 1.0)
    else:
        ceiling = max(max(data["disagreement_pct"]) * horizon["disagreement_headroom"], 1.0)

    if horizon is not None:
        # The tail is tinted and the region inside the horizon is left white, so
        # the distinction survives desaturation as one step in background
        # luminance rather than as two competing hues. A warm neutral, not a
        # warning colour: leaving the represented horizon is not a failure, and
        # the disagreement grows steadily from long before the boundary.
        ax.axvspan(horizon["horizon_step"], horizon["final_step"],
                   facecolor=TAIL_TINT, edgecolor="none", zorder=0)

    # Every series the model produces is drawn only as far as the current step:
    # the disagreement curve and the prediction's grain count both describe the
    # rollout, and drawing them ahead of the marker would show the reader an
    # outcome the rollout has not reached. The phase-field reference is the one
    # exception and is drawn complete — see the grain-count block below.
    stacking = {} if horizon is None else {"zorder": 4}
    ax.plot(steps[upto], data["disagreement_pct"][upto], color=DISCREPANCY_RED,
            marker="o", markersize=3.2, linewidth=1.8,
            label="differing pixels (%)", **stacking)
    ax.set_xlim(0, steps[-1])
    ax.set_ylim(0, ceiling)
    ax.set_xlabel("step", fontsize=9)
    ax.set_ylabel("differing pixels (%)", fontsize=9, color=DISCREPANCY_RED)
    ax.tick_params(axis="both", labelsize=8)
    ax.tick_params(axis="y", labelcolor=DISCREPANCY_RED)
    ax.grid(alpha=0.18, linewidth=0.6)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)

    if horizon is not None:
        ax.set_axisbelow(True)
        boundary = horizon["horizon_step"]
        ax.axvline(boundary, color=INK, linestyle=(0, (4, 3)), linewidth=1.0, zorder=3)
        spanning = blended_transform_factory(ax.transData, ax.transAxes)
        # The boundary names itself, immediately beside the rule, so the reader
        # does not have to trace the line anywhere to find out what it is.
        ax.plot([boundary, boundary], [1.0, 1.04], transform=spanning, color=INK,
                linestyle=(0, (4, 3)), linewidth=1.0, clip_on=False, zorder=5)
        # The label takes the side the mapping names and the continuity note the
        # other, so a boundary early in the axis does not push the note off the
        # left edge of the canvas.
        label_right = horizon["boundary_label_side"] == "right"
        label_x, note_x = ((boundary + 150, boundary - 150) if label_right
                           else (boundary - 150, boundary + 150))
        ax.text(label_x, 1.055, horizon["boundary_label"], transform=spanning,
                fontsize=BOUNDARY_PT, fontweight="bold", color=INK,
                ha="left" if label_right else "right",
                va="bottom", clip_on=False, zorder=6, gid="chk:boundary")
        ax.text(note_x, 1.055, horizon["continuity_note"], transform=spanning,
                fontsize=ANNOTATION_PT, color=INK,
                ha="right" if label_right else "left", va="bottom",
                clip_on=False, zorder=5, gid="chk:continuity")

        # The regions say what they mean inside themselves. A reader meeting this
        # animation on its own, without the surrounding page, has to be able to
        # tell the represented horizon from the extrapolation without being told
        # elsewhere. The headroom above the series exists for these labels.
        regions = (
            (boundary / 2, horizon["within_title"], horizon["within_detail"], "within"),
            ((boundary + horizon["final_step"]) / 2, horizon["beyond_title"],
             horizon["beyond_detail"], "beyond"),
        )
        for centre, title, detail, tag in regions:
            ax.text(centre, 0.915, title, transform=spanning,
                    fontsize=REGION_TITLE_PT, fontweight="bold", color=INK,
                    ha="center", va="top", zorder=6, gid=f"chk:{tag}_title")
            for line, text in enumerate(detail):
                ax.text(centre, 0.795 - 0.085 * line, text, transform=spanning,
                        fontsize=ANNOTATION_PT, color=INK, ha="center", va="top",
                        zorder=6, gid=f"chk:{tag}_detail{line}")

    counts = ax.twinx()
    if horizon is None:
        counts.step(steps[upto], data["reference_active"][upto], where="post",
                    color=INK, linewidth=1.6, label="active grains: reference")
        counts.step(steps[upto], data["model_active"][upto], where="post",
                    color=MODEL_BLUE, linewidth=1.6, linestyle="--",
                    label="active grains: PINN-Phase")
        counts.set_ylim(0, phases + 2)
    else:
        # The reference is drawn once, complete, and does not change from frame to
        # frame. It is the fixed comparator the rollout is read against, so it
        # belongs to the axis rather than to the animation: nothing about it moves,
        # grows or ends, and the only thing travelling across the panel is the
        # prediction. Drawing it in full is safe precisely because it is an
        # evaluation-only trajectory — the model was never given it, on either
        # side of the boundary, and the label and footer say so.
        #
        # The two grain counts agree almost everywhere, so a same-stroke pair
        # would hide one behind the other and invite the reading that only one
        # was drawn. Both are ordinary weight; the reference is dashed and drawn
        # over the prediction's solid stroke, so where they coincide the
        # prediction shows through the dash gaps and the picture says "two series
        # agree" rather than "one series". Neither is displaced numerically.
        counts.step(steps[upto], data["model_active"][upto], where="post",
                    color=MODEL_BLUE, linewidth=MODEL_WIDTH, zorder=3,
                    label=horizon["model_label"])
        counts.step(steps, data["reference_active"], where="post",
                    color=INK, linewidth=REFERENCE_WIDTH, zorder=5,
                    linestyle=REFERENCE_DASHES,
                    label=horizon["reference_label"])
        # The marker is the reader's "you are here". A dark edge keeps it legible
        # over the white region, over the tinted region and over either curve.
        counts.plot([steps[index]], [data["model_active"][index]], marker="o",
                    markersize=11.0, markerfacecolor=MODEL_BLUE, markeredgecolor=INK,
                    markeredgewidth=1.6, zorder=7, clip_on=False, linestyle="none")
        counts.set_ylim(0, phases * horizon["grain_headroom"])
    counts.set_ylabel("active grains", fontsize=9, color=INK)
    counts.tick_params(axis="y", labelsize=8, labelcolor=INK)
    counts.spines["top"].set_visible(False)

    count_handles, count_labels = counts.get_legend_handles_labels()
    if horizon is not None:
        # The prediction is plotted first so the dashed reference can sit over it,
        # but the legend names the reference first, as the comparator the
        # prediction is read against.
        order = [count_labels.index(horizon["reference_label"]),
                 count_labels.index(horizon["model_label"])]
        count_handles = [count_handles[i] for i in order]
        count_labels = [count_labels[i] for i in order]
    handles = ax.get_legend_handles_labels()[0] + count_handles
    labels = ax.get_legend_handles_labels()[1] + count_labels
    anchor = -0.34 if horizon is None else -0.30
    ax.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, anchor),
              ncol=len(labels), fontsize=8.5, frameon=False)


def unreadable(figure) -> str | None:
    """Return a description of the first unreadable label, or None.

    Checked rather than assumed because the labels are prose, and prose is what
    a later revision edits. Overlap is tested only between the annotation labels,
    which carry a ``chk:`` group id; tick labels and axis titles legitimately sit
    close to their own axes.
    """
    figure.canvas.draw()
    renderer = figure.canvas.get_renderer()
    width, height = figure.canvas.get_width_height()
    tagged = []
    for text in figure.findobj(matplotlib.text.Text):
        content = text.get_text()
        if not content.strip() or not text.get_visible():
            continue
        box = text.get_window_extent(renderer=renderer)
        if min(box.x0, box.y0, width - box.x1, height - box.y1) < 0:
            return f"{content.splitlines()[0]!r} falls outside the canvas"
        if (text.get_gid() or "").startswith("chk:"):
            tagged.append((content.splitlines()[0], box))
    for legend in figure.findobj(matplotlib.legend.Legend):
        tagged.append(("the legend", legend.get_window_extent(renderer=renderer)))
    for first in range(len(tagged)):
        for second in range(first + 1, len(tagged)):
            (name_a, box_a), (name_b, box_b) = tagged[first], tagged[second]
            horizontal = max(box_a.x0 - box_b.x1, box_b.x0 - box_a.x1)
            vertical = max(box_a.y0 - box_b.y1, box_b.y0 - box_a.y1)
            if max(horizontal, vertical) < 0:
                return f"{name_a!r} overlaps {name_b!r}"
    return None


def frame(reference, model, data, index, title, phases, cmap, horizon=None) -> Image.Image:
    step = data["steps"][index]
    ref = reference[step]
    pred = ref if step == 0 and step not in model else model[step]
    difference = pred != ref

    if horizon is None:
        figure = plt.figure(figsize=(12.4, 6.9), constrained_layout=True)
        grid = figure.add_gridspec(2, 3, height_ratios=(3.0, 1.35))
        axes = [figure.add_subplot(grid[0, column]) for column in range(3)]
        history = figure.add_subplot(grid[1, :])
    else:
        # Placed by hand rather than by a constrained solver, so the clearances
        # between the title block, the panel captions, the horizon annotation,
        # the legend and the footer are fixed and measurable.
        figure = plt.figure(figsize=(12.4, 8.6), dpi=110)
        left, right, gap = 0.035, 0.965, 0.012
        column_width = (right - left - 2 * gap) / 3
        top, height = 0.874, 334.4 / (8.6 * 110)
        axes = [
            figure.add_axes([left + column * (column_width + gap), top - height,
                             column_width, height])
            for column in range(3)
        ]
        history = figure.add_axes([0.055, 0.200, 0.890, 0.235])

    draw_field(axes[0], ref, cmap, phases)
    axes[0].set_title("Reference", fontsize=12, pad=6)
    draw_field(axes[1], pred, cmap, phases)
    axes[1].set_title("PINN-Phase", fontsize=12, pad=6)
    draw_discrepancy(axes[2], ref, difference, phases)
    caption = ("empty by construction at step 0" if step == 0
               else f"{100 * difference.mean():.2f}% of pixels differ")
    axes[2].set_title(f"Differing pixels\n{caption}", fontsize=12, pad=6)

    history_axis(history, data, index, phases, horizon)

    if horizon is None:
        figure.suptitle(f"{title}  |  step {step:,}  of  {data['steps'][-1]:,}",
                        fontsize=16, fontweight="bold")
    else:
        status = (horizon["within_status"] if step <= horizon["horizon_step"]
                  else horizon["beyond_status"])
        figure.text(0.5, 0.982, title, fontsize=14.5, fontweight="bold",
                    ha="center", va="top", color="black", gid="chk:title")
        figure.text(0.5, 0.955,
                    f"step {step:,} of {data['steps'][-1]:,}  ·  {status}  ·  "
                    f"{horizon['status_qualifier']}",
                    fontsize=STATUS_PT, ha="center", va="top", color=INK,
                    gid="chk:status")
        for offset, line in enumerate(horizon["footer"]):
            figure.text(0.5, 0.062 - 0.021 * offset, line, fontsize=ANNOTATION_PT,
                        ha="center", va="top", color=INK, gid=f"chk:footer{offset}")

    if horizon is not None:
        # Only the annotated figure is checked. The check draws the canvas, and
        # for the unannotated figure that extra draw re-runs the constrained
        # layout solver and moves the result — a check must not change the thing
        # it inspects. That figure adds no prose of its own to truncate.
        problem = unreadable(figure)
        if problem is not None:
            plt.close(figure)
            raise UnreadableFrame(f"step {step:,}: {problem}")

    buffer = io.BytesIO()
    figure.savefig(buffer, format="png", dpi=110, facecolor="white")
    plt.close(figure)
    buffer.seek(0)
    with Image.open(buffer) as image:
        return image.convert("RGB").copy()


def animate(reference, model, steps, title, phases, gif: Path, poster: Path,
            horizon=None) -> dict[str, Any]:
    cmap = stable_grain_colors(phases)
    data = series(reference, model, steps)
    frames = [frame(reference, model, data, i, title, phases, cmap, horizon)
              for i in range(len(steps))]
    delays = [1200.0] * len(frames)
    delays[0] = 1800.0
    delays[-1] = 3600.0
    gif.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(gif, [np.asarray(f) for f in frames], duration=delays, loop=0)
    frames[-1].save(poster, format="PNG", optimize=True)
    measured = {
        "frames": len(frames),
        "size": list(frames[0].size),
        "loop_seconds": round(sum(delays) / 1000.0, 1),
        "recomputed": {
            "steps": data["steps"],
            "differing_pixels_pct": [round(v, 4) for v in data["disagreement_pct"]],
            "reference_active_grains": data["reference_active"],
            "model_active_grains": data["model_active"],
        },
    }
    if horizon is not None:
        boundary = horizon["horizon_step"]
        measured["horizon_annotation"] = {
            "horizon_step": boundary,
            "final_step": horizon["final_step"],
            "extrapolation_steps": horizon["final_step"] - boundary,
            "last_frame_within_horizon": max(s for s in steps if s <= boundary),
            "first_frame_beyond_horizon": min(s for s in steps if s > boundary),
            "within_status": horizon["within_status"],
            "beyond_status": horizon["beyond_status"],
            "region_labels": {
                "within": horizon["within_title"],
                "beyond": horizon["beyond_title"],
            },
            "boundary_label": horizon["boundary_label"],
            "reference_series": horizon["reference_label"],
            "model_series": horizon["model_label"],
            "progressive_reveal": (
                "every series is drawn only as far as the current step; no "
                "reference-derived value is shown ahead of the marker"
            ),
            "no_restart_at_boundary": True,
        }
    return measured


def load_n25(reference_path: Path, model_path: Path):
    reference = np.load(reference_path, allow_pickle=False)
    model = np.load(model_path, allow_pickle=False)
    ref_steps = [int(v) for v in reference["save_steps"].tolist()]
    ref_labels = labels_from_states(reference["states"])
    model_steps = [int(v) for v in model["cadence_steps"].tolist()]
    return (
        {step: ref_labels[i] for i, step in enumerate(ref_steps)},
        {step: model["cadence_labels"][i] for i, step in enumerate(model_steps)},
    )


def load_n64(reference_path: Path, model_path: Path):
    reference = np.load(reference_path, allow_pickle=False)
    model = np.load(model_path, allow_pickle=False)
    ref_steps = [int(v) for v in reference["save_steps"].tolist()]
    model_steps = [int(v) for v in model["save_steps"].tolist()]
    ref_labels = labels_from_states(reference["states"])
    model_labels = labels_from_states(model["states"])
    return (
        {step: ref_labels[i] for i, step in enumerate(ref_steps)},
        {step: model_labels[i] for i, step in enumerate(model_steps)},
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    for role in EXPECTED_SHA256:
        parser.add_argument(f"--{role.replace('_', '-')}", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("outputs/media/current_results"))
    args = parser.parse_args()

    sources = {role: getattr(args, role) for role in EXPECTED_SHA256}
    digests = {role: verify(role, path) for role, path in sources.items()}

    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)

    assets: dict[str, Any] = {}

    n25_reference, n25_model = load_n25(sources["n25_reference"], sources["n25_model"])
    assets["n25_cascade_reference_vs_pinn_phase"] = animate(
        n25_reference, n25_model, N25_MILESTONES,
        "25-grain cascade — training initial condition, "
        "4,096-step training horizon", 25,
        output / "n25_cascade_reference_vs_pinn_phase.gif",
        output / "n25_cascade_reference_vs_pinn_phase_poster.png",
        horizon=N25_HORIZON_ANNOTATION,
    )

    n64_reference, n64_model = load_n64(sources["n64_reference"], sources["n64_model"])
    assets["n64_dense_reference_vs_pinn_phase"] = animate(
        n64_reference, n64_model, N64_MILESTONES,
        "Dense 64-grain coarsening — sensitivity run, "
        "training horizon extended to 8,192 steps", 64,
        output / "n64_dense_reference_vs_pinn_phase.gif",
        output / "n64_dense_reference_vs_pinn_phase_poster.png",
        horizon=HORIZON_ANNOTATION,
    )

    manifest = {
        "schema": "pinn-phase-media-manifest-v1",
        "license": "CC-BY-4.0",
        "renderer_sha256": sha256_file(Path(__file__)),
        "source_artifacts_sha256": digests,
        "assets": {},
        "measurements": assets,
    }
    for stem in assets:
        for name in (f"{stem}.gif", f"{stem}_poster.png"):
            manifest["assets"][name] = {"sha256": sha256_file(output / name)}
    (output / "asset_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    for name in sorted(manifest["assets"]):
        print(f"Rendered {output / name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
