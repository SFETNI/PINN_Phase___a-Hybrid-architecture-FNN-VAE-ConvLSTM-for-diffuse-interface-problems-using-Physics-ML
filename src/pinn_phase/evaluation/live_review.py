"""Review-only physical checks for monitored PINN-Phase rollouts."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def assess_live_radius(
    preview_times: Sequence[float],
    preview_radii: Sequence[float],
    reference_times: Sequence[float],
    reference_radii: Sequence[float],
    *,
    extinction_fraction: float = 0.1,
    retained_reference_fraction: float = 0.5,
    max_radius_abs_error: float = 0.05,
) -> tuple[dict[str, object], np.ndarray]:
    """Compare a PINN preview against PF radii without affecting training.

    The reference trajectory is used only for review diagnostics. It is never
    passed to the optimizer or included in a loss.
    """

    preview_t = np.asarray(preview_times, dtype=np.float64)
    preview_r = np.asarray(preview_radii, dtype=np.float64)
    reference_t = np.asarray(reference_times, dtype=np.float64)
    reference_r = np.asarray(reference_radii, dtype=np.float64)
    if preview_t.ndim != 1 or preview_r.shape != preview_t.shape or preview_t.size < 2:
        raise ValueError("preview times and radii must be matching one-dimensional arrays")
    if reference_t.ndim != 1 or reference_r.shape != reference_t.shape or reference_t.size < 2:
        raise ValueError("reference times and radii must be matching one-dimensional arrays")
    if np.any(np.diff(preview_t) < 0.0) or np.any(np.diff(reference_t) <= 0.0):
        raise ValueError("times must be ordered and reference times must be strictly increasing")
    if not 0.0 < extinction_fraction < 1.0:
        raise ValueError("extinction_fraction must be inside (0, 1)")
    if not 0.0 < retained_reference_fraction < 1.0:
        raise ValueError("retained_reference_fraction must be inside (0, 1)")
    if max_radius_abs_error <= 0.0:
        raise ValueError("max_radius_abs_error must be positive")

    interpolated_reference = np.interp(preview_t, reference_t, reference_r)
    abs_errors = np.abs(preview_r - interpolated_reference)
    initial_radius = float(interpolated_reference[0])
    invalid_extinction = (
        (preview_r <= extinction_fraction * initial_radius)
        & (interpolated_reference >= retained_reference_fraction * initial_radius)
    )
    reasons: list[str] = []
    if bool(np.any(invalid_extinction)):
        first = int(np.flatnonzero(invalid_extinction)[0])
        reasons.append(
            "premature_extinction:"
            f"t={preview_t[first]:.6g},"
            f"pinn_radius={preview_r[first]:.6g},"
            f"pf_radius={interpolated_reference[first]:.6g}"
        )
    if float(abs_errors.max()) > max_radius_abs_error:
        reasons.append(
            "radius_abs_error:"
            f"max={float(abs_errors.max()):.6g},"
            f"limit={max_radius_abs_error:.6g}"
        )
    reference_reaches_extinction = bool(
        interpolated_reference[-1] <= extinction_fraction * initial_radius
    )
    pinn_reaches_extinction = bool(preview_r[-1] <= extinction_fraction * initial_radius)
    if reference_reaches_extinction and not pinn_reaches_extinction:
        reasons.append(
            "missing_extinction:"
            f"t={preview_t[-1]:.6g},"
            f"pinn_radius={preview_r[-1]:.6g},"
            f"pf_radius={interpolated_reference[-1]:.6g}"
        )

    metrics: dict[str, object] = {
        "status": "warning" if reasons else "ok",
        "reasons": reasons,
        "preview_horizon": float(preview_t[-1]),
        "pinn_final_radius": float(preview_r[-1]),
        "pf_final_radius": float(interpolated_reference[-1]),
        "final_radius_abs_error": float(abs_errors[-1]),
        "max_radius_abs_error": float(abs_errors.max()),
        "reference_reaches_extinction": reference_reaches_extinction,
        "pinn_reaches_extinction": pinn_reaches_extinction,
        "uses_pf_reference_for_training": False,
    }
    return metrics, interpolated_reference
