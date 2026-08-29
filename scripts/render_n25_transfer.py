#!/usr/bin/env python3
"""Render public N25 transfer media from the shipped compact fixtures."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
from pathlib import Path
from typing import Any

import imageio.v2 as imageio
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np
from PIL import Image

import sys

# Run from a fresh extraction without installing anything: put this repository's
# src/ at the front of the import path. Prepending rather than appending means the
# code under test is this tree's, not a copy that happens to be installed elsewhere.
sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pinn_phase.io import load_npz_arrays


CASE_LABELS = {
    "id_01": "Held-out case 1",
    "id_02": "Held-out case 2",
    "id_03": "Held-out case 3",
    "id_04": "Held-out case 4",
    "id_05": "Held-out case 5",
    "id_06": "Held-out case 6",
    "id_07": "Held-out case 7",
    "id_08": "Held-out case 8",
    "stress_dense": "Stress: dense",
    "stress_sparse": "Stress: sparse",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def phase_cmap() -> ListedColormap:
    tab20 = list(plt.get_cmap("tab20").colors)
    set3 = list(plt.get_cmap("Set3").colors)
    return ListedColormap((tab20 + set3)[:25])


def muted_cmap() -> ListedColormap:
    """Faint greys that keep the grain structure legible under a red overlay."""
    return ListedColormap([(value, value, value) for value in np.linspace(0.72, 0.95, 25)])


def labels_at(bundle: dict[str, np.ndarray], step: int) -> np.ndarray:
    steps = [int(value) for value in bundle["cadence_steps"].tolist()]
    if step == 0 and step not in steps:
        raise ValueError("step zero fallback must be supplied by the caller")
    return bundle["cadence_labels"][steps.index(step)]


def load_cases(root: Path) -> list[dict[str, Any]]:
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    cases = []
    for case_id, record in sorted(
        manifest["cases"].items(), key=lambda item: item[1]["ordinal"]
    ):
        loaded: dict[str, Any] = {"case_id": case_id, "ordinal": record["ordinal"]}
        for role in ("reference", "primary"):
            entry = record[role]
            loaded[role] = load_npz_arrays(
                root / entry["path"],
                expected_sha256=entry["sha256"],
                required_keys=frozenset({"cadence_steps", "cadence_labels"}),
            )
        cases.append(loaded)
    return cases


def draw_field(ax, labels: np.ndarray, *, difference: np.ndarray | None = None) -> None:
    ax.imshow(labels, cmap=phase_cmap(), vmin=0, vmax=24, interpolation="nearest")
    if difference is not None:
        overlay = np.ma.masked_where(~difference, difference)
        ax.imshow(
            overlay,
            cmap=ListedColormap(["#d73027"]),
            vmin=0,
            vmax=1,
            alpha=0.92,
            interpolation="nearest",
        )
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def terminal_atlas(cases: list[dict[str, Any]], output: Path) -> None:
    figure, axes = plt.subplots(4, 5, figsize=(12, 8.6), constrained_layout=True)
    for index, case in enumerate(cases):
        block = 0 if index < 5 else 2
        column = index % 5
        reference = labels_at(case["reference"], 12000)
        primary = labels_at(case["primary"], 12000)
        disagreement = primary != reference
        draw_field(axes[block, column], reference)
        draw_field(axes[block + 1, column], primary, difference=disagreement)
        compact = CASE_LABELS[case["case_id"]].replace("Held-out case", "Case")
        axes[block, column].set_title(compact, fontsize=12, pad=5)
        axes[block + 1, column].text(
            0.5,
            -0.07,
            f"{100 * disagreement.mean():.2f}% differing pixels",
            transform=axes[block + 1, column].transAxes,
            ha="center",
            va="top",
            fontsize=9,
            color="#3b4045",
        )
    for row, label in (
        (0, "Reference"),
        (1, "PINN-Phase + differences"),
        (2, "Reference"),
        (3, "PINN-Phase + differences"),
    ):
        axes[row, 0].set_ylabel(label, fontsize=10, labelpad=8)
    figure.suptitle(
        "Held-out 25-grain transfer at step 12,000: 7 of 8 held-out and 2 of 2 stress cases pass",
        fontsize=16,
        fontweight="bold",
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=200, facecolor="white")
    plt.close(figure)


def case_series(case: dict[str, Any]) -> dict[str, Any]:
    """Per-case discrepancy and active-grain histories, from the shipped arrays only.

    Label maps exist at the saved cadence, so the disagreement curve is sampled at
    those steps. Grain counts come from the per-step ``areas`` arrays and therefore
    carry full step resolution. At step zero the rollout input is the reference
    initial condition, so the difference panel is empty by construction.
    """
    reference, primary = case["reference"], case["primary"]
    steps = [int(value) for value in reference["cadence_steps"].tolist()]
    disagreement = []
    for step in steps:
        reference_labels = labels_at(reference, step)
        predicted = reference_labels if step == 0 else labels_at(primary, step)
        disagreement.append(100.0 * float((predicted != reference_labels).mean()))
    return {
        "steps": steps,
        "disagreement_pct": disagreement,
        "reference_active": (reference["areas"] > 0).sum(axis=1),
        "model_active": (primary["areas"] > 0).sum(axis=1),
    }


def _history_axis(ax, series: dict[str, Any], index: int) -> None:
    steps = series["steps"]
    step = steps[index]
    ax.plot(
        steps[: index + 1],
        series["disagreement_pct"][: index + 1],
        color="#d73027",
        marker="o",
        markersize=3.4,
        linewidth=1.8,
        label="differing pixels (%)",
    )
    ax.set_xlim(0, steps[-1])
    ax.set_ylim(0, max(max(series["disagreement_pct"]) * 1.35, 1.0))
    ax.set_xlabel("step", fontsize=9)
    ax.set_ylabel("differing pixels (%)", fontsize=9, color="#d73027")
    ax.tick_params(axis="both", labelsize=8)
    ax.tick_params(axis="y", labelcolor="#d73027")
    ax.grid(alpha=0.18, linewidth=0.6)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)

    counts = ax.twinx()
    span = slice(0, step + 1)
    counts.step(
        np.arange(step + 1),
        series["reference_active"][span],
        where="post",
        color="#3b4045",
        linewidth=1.6,
        label="active grains: reference",
    )
    counts.step(
        np.arange(step + 1),
        series["model_active"][span],
        where="post",
        color="#4575b4",
        linewidth=1.6,
        linestyle="--",
        label="active grains: PINN-Phase",
    )
    counts.set_ylim(0, 26)
    counts.set_ylabel("active grains", fontsize=9, color="#3b4045")
    counts.tick_params(axis="y", labelsize=8, labelcolor="#3b4045")
    counts.spines["top"].set_visible(False)

    handles = ax.get_legend_handles_labels()[0] + counts.get_legend_handles_labels()[0]
    labels = ax.get_legend_handles_labels()[1] + counts.get_legend_handles_labels()[1]
    ax.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.34),
        ncol=3,
        fontsize=8.5,
        frameon=False,
    )


def _case_frame(case: dict[str, Any], series: dict[str, Any], index: int) -> Image.Image:
    step = series["steps"][index]
    reference = labels_at(case["reference"], step)
    predicted = reference if step == 0 else labels_at(case["primary"], step)
    difference = predicted != reference

    figure = plt.figure(figsize=(12.4, 6.9), constrained_layout=True)
    grid = figure.add_gridspec(2, 3, height_ratios=(3.0, 1.35))
    fields = [figure.add_subplot(grid[0, column]) for column in range(3)]

    draw_field(fields[0], reference)
    fields[0].set_title("Reference", fontsize=12, pad=6)
    draw_field(fields[1], predicted)
    fields[1].set_title("PINN-Phase", fontsize=12, pad=6)

    fields[2].imshow(reference, cmap=muted_cmap(), vmin=0, vmax=24, interpolation="nearest")
    fields[2].imshow(
        np.ma.masked_where(~difference, difference),
        cmap=ListedColormap(["#d73027"]),
        vmin=0,
        vmax=1,
        interpolation="nearest",
    )
    fields[2].set_xticks([])
    fields[2].set_yticks([])
    for spine in fields[2].spines.values():
        spine.set_visible(False)
    caption = (
        "empty by construction at step 0"
        if step == 0
        else f"{100 * difference.mean():.2f}% of pixels differ"
    )
    fields[2].set_title(f"Discrepancy\n{caption}", fontsize=12, pad=6)

    _history_axis(figure.add_subplot(grid[1, :]), series, index)
    figure.suptitle(
        f"{CASE_LABELS[case['case_id']]}  |  step {step:,}  of  {series['steps'][-1]:,}",
        fontsize=16,
        fontweight="bold",
    )
    buffer = io.BytesIO()
    figure.savefig(buffer, format="png", dpi=110, facecolor="white")
    plt.close(figure)
    buffer.seek(0)
    with Image.open(buffer) as image:
        return image.convert("RGB").copy()


def case_animation(case: dict[str, Any], gif: Path, poster: Path) -> None:
    """Reference, prediction, and discrepancy for one case across the saved cadence."""
    series = case_series(case)
    frames = [_case_frame(case, series, index) for index in range(len(series["steps"]))]
    # Frame delays are milliseconds, matching the retained benchmark animations.
    # Each frame carries three fields plus a history axis, so it is paced slower
    # than those, and the terminal frame holds long enough to be read.
    durations = [1200.0] * len(frames)
    durations[0] = 1800.0
    durations[-1] = 3600.0
    gif.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(
        gif, [np.asarray(frame) for frame in frames], duration=durations, loop=0
    )
    frames[-1].save(poster, format="PNG", optimize=True)


ANIMATED_CASES = ("id_02",)

ANIMATION_ALT = {
    "id_02": (
        "Reference, PINN-Phase prediction, and their pixel differences for a held-out "
        "25-grain case across 12,000 steps, with a differing-pixel curve and an "
        "active-grain staircase that both arrive at thirteen grains."
    ),
}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--benchmark-root", type=Path, default=Path("benchmarks/n25_transfer")
    )
    parser.add_argument("--output", type=Path, default=Path("outputs/media/n25_transfer"))
    args = parser.parse_args()
    benchmark_root = args.benchmark_root.resolve()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    cases = load_cases(benchmark_root)
    by_id = {case["case_id"]: case for case in cases}

    atlas = output / "terminal_atlas.png"
    terminal_atlas(cases, atlas)
    rendered = {
        atlas.name: (
            "Terminal reference and prediction for eight held-out and two stress cases; "
            "differing pixels are marked in red."
        )
    }
    for case_id in ANIMATED_CASES:
        gif = output / f"{case_id}_reference_vs_pinn_phase.gif"
        poster = output / f"{case_id}_reference_vs_pinn_phase_poster.png"
        case_animation(by_id[case_id], gif, poster)
        rendered[gif.name] = ANIMATION_ALT[case_id]
        rendered[poster.name] = (
            f"Final frame of the {CASE_LABELS[case_id]} comparison at step 12,000: "
            "reference, PINN-Phase prediction, and differing pixels."
        )

    assets = {
        "schema": "pinn-phase-media-manifest-v1",
        "license": "CC-BY-4.0",
        "input_manifest_sha256": sha256_file(benchmark_root / "manifest.json"),
        "renderer_sha256": sha256_file(Path(__file__)),
        "assets": {
            name: {"sha256": sha256_file(output / name), "alt": alt}
            for name, alt in sorted(rendered.items())
        },
    }
    (output / "asset_manifest.json").write_text(
        json.dumps(assets, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    for name in sorted(rendered):
        print(f"Rendered {output / name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
