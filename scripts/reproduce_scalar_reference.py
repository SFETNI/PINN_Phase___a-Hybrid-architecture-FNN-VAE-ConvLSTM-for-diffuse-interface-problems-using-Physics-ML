#!/usr/bin/env python3
"""Generate and verify the compact scalar shrinkage reference benchmark."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

import sys

# Run from a fresh extraction without installing anything: put this repository's
# src/ at the front of the import path. Prepending rather than appending means the
# code under test is this tree's, not a copy that happens to be installed elsewhere.
sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pinn_phase.physics.reference_solver import (
    equivalent_radius,
    initial_condition_from_config,
    load_benchmark_config,
    simulate,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def linear_fit(x: np.ndarray, y: np.ndarray) -> dict[str, float]:
    slope, intercept = np.polyfit(x, y, deg=1)
    prediction = slope * x + intercept
    residual = float(np.sum((y - prediction) ** 2))
    total = float(np.sum((y - np.mean(y)) ** 2))
    return {
        "slope": float(slope),
        "intercept": float(intercept),
        "r_squared": 1.0 if total == 0.0 else 1.0 - residual / total,
    }


def matches_reviewed(expected: object, observed: object) -> bool:
    """Compare fit results portably while keeping discrete outcomes exact."""

    if isinstance(expected, dict) and isinstance(observed, dict):
        return expected.keys() == observed.keys() and all(
            matches_reviewed(expected[key], observed[key]) for key in expected
        )
    if isinstance(expected, float) and isinstance(observed, float):
        return math.isclose(expected, observed, rel_tol=1.0e-12, abs_tol=1.0e-14)
    return expected == observed


def reproduce(
    config_path: Path, output_dir: Path, expected_path: Path
) -> dict[str, object]:
    config, initial_spec = load_benchmark_config(config_path)
    initial = initial_condition_from_config(config, initial_spec)
    result = simulate(initial, config)
    radii = np.asarray(
        [equivalent_radius(state, config.spacings) for state in result.states]
    )
    retained = radii >= 0.05 * radii[0]
    fit = linear_fit(result.times[retained], radii[retained] ** 2)
    theoretical_slope = -2.0 * config.mu * config.sigma
    near_extinction = np.flatnonzero(radii <= 0.01 * radii[0])
    metrics: dict[str, object] = {
        "schema": "pinn-phase-scalar-reference-score-v1",
        "benchmark_id": config.benchmark_id,
        "samples": int(result.times.size),
        "initial_radius": float(radii[0]),
        "final_radius": float(radii[-1]),
        "radius_squared_pre_extinction_linear_fit": fit,
        "theoretical_radius_squared_slope": theoretical_slope,
        "relative_slope_error": abs(fit["slope"] - theoretical_slope)
        / abs(theoretical_slope),
        "first_near_extinction_time": (
            float(result.times[near_extinction[0]]) if near_extinction.size else None
        ),
        "energy_nonincrease_fraction": float(
            np.mean(np.diff(result.energies) <= 1.0e-12)
        ),
        "finite": bool(
            np.isfinite(result.states).all()
            and np.isfinite(result.energies).all()
            and np.isfinite(radii).all()
        ),
        "phi_min": float(np.min(result.states)),
        "phi_max": float(np.max(result.states)),
    }
    expected = json.loads(expected_path.read_text(encoding="utf-8"))
    if not matches_reviewed(expected, metrics):
        raise RuntimeError("scalar reference metrics differ from the reviewed expected values")
    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_dir / "reference.npz",
        times=result.times,
        states=result.states,
        energies=result.energies,
        radii=radii,
    )
    (output_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return metrics


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=REPOSITORY_ROOT / "configs/benchmarks/scalar_shrinkage_2d.yaml",
    )
    parser.add_argument(
        "--expected",
        type=Path,
        default=REPOSITORY_ROOT / "benchmarks/scalar_shrinkage_2d/expected_metrics.json",
    )
    parser.add_argument(
        "--output", type=Path, default=Path("outputs/scalar_shrinkage_2d")
    )
    args = parser.parse_args()
    metrics = reproduce(
        args.config.resolve(), args.output.resolve(), args.expected.resolve()
    )
    print(
        "scalar reference: reviewed expected match; "
        f"relative slope error={metrics['relative_slope_error']:.4%}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
