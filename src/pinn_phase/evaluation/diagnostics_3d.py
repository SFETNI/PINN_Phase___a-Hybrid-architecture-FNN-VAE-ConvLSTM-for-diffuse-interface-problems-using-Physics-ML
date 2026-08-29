"""3D phase-field diagnostics for the scalar spherical shrinkage benchmark.

All functions operate on NumPy arrays (CPU) and are audit/report-only.
No PF trajectory enters backpropagation through these functions.

3D expected law:
    R²(t) = R0² − 4 μ σ t
    dR²/dt = −4 μ σ            (NOT −2μσ, which is the 2D circular law)

Key functions
-------------
equivalent_radius_3d_np  : volume-equivalent sphere radius from a numpy phi
r2_slope_3d              : linear fit of R²(t) and slope error vs 3D law
center_of_mass_3d        : volume-weighted CoM coordinates
com_drift_3d             : max displacement of CoM over a rollout (px)
sphericity_proxy_3d      : radial std / mean radius (anisotropy indicator)
extinction_time_3d       : first frame where R < threshold * R0
volume_proxy_3d          : integral of phi over volume (mass proxy)
phi_bounds_ok_3d         : check phi in [0, 1] across all frames
energy_monotone_3d       : check free energy is non-increasing
radius_estimators_3d     : compare hard, soft, radial, and marching-cubes radii
face_shell_error_3d      : face-band vs interior error ratios
residual_components_3d   : final phi-threshold component stats
gate_report_3d           : collect all gate metrics into a dict
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

Array = np.ndarray

# 3D law constant: dR²/dt = -4 * mu * sigma
_3D_LAW_FACTOR = 4.0
_2D_LAW_FACTOR = 2.0  # listed for reference; do NOT use for 3D


def equivalent_radius_3d_np(phi: Array, spacings: Sequence[float]) -> float:
    """Return the volume-equivalent sphere radius of a 3D scalar grain.

    Volume = integral(phi) * cell_volume.
    R_eq   = (3 * Volume / (4 * pi))^(1/3)
    """
    spacing_vals = tuple(float(s) for s in spacings)
    if phi.ndim != 3 or len(spacing_vals) != 3:
        raise ValueError("equivalent_radius_3d_np requires a 3D field and three spacings")
    cell_vol = float(np.prod(spacing_vals))
    volume = float(np.sum(phi)) * cell_vol
    return float((max(volume, 0.0) * 3.0 / (4.0 * np.pi)) ** (1.0 / 3.0))


def hard_radius_3d_np(phi: Array, spacings: Sequence[float], threshold: float = 0.5) -> float:
    """Return volume-equivalent sphere radius from the hard ``phi > threshold`` mask."""

    spacing_vals = tuple(float(s) for s in spacings)
    if phi.ndim != 3 or len(spacing_vals) != 3:
        raise ValueError("hard_radius_3d_np requires a 3D field and three spacings")
    cell_vol = float(np.prod(spacing_vals))
    volume = float(np.count_nonzero(phi > threshold)) * cell_vol
    return float((max(volume, 0.0) * 3.0 / (4.0 * np.pi)) ** (1.0 / 3.0))


def radial_profile_radius_3d_np(
    phi: Array,
    spacings: Sequence[float],
    threshold: float = 0.5,
    bin_width: float | None = None,
) -> float:
    """Estimate the spherical radius from the radial-average ``phi=threshold`` crossing.

    The profile center is the soft volume-weighted center of mass.  This estimator
    is intentionally independent of the hard volume count, so it catches cases where
    a diffuse shell has the right volume but the wrong radial position.
    """

    spacing_vals = tuple(float(s) for s in spacings)
    if phi.ndim != 3 or len(spacing_vals) != 3:
        raise ValueError("radial_profile_radius_3d_np requires a 3D field and three spacings")
    dz, dy, dx = spacing_vals
    zc, yc, xc = center_of_mass_3d(phi, spacing_vals)
    nz, ny, nx = phi.shape
    z = np.arange(nz, dtype=np.float64) * dz
    y = np.arange(ny, dtype=np.float64) * dy
    x = np.arange(nx, dtype=np.float64) * dx
    zz, yy, xx = np.meshgrid(z, y, x, indexing="ij")
    radii = np.sqrt((zz - zc) ** 2 + (yy - yc) ** 2 + (xx - xc) ** 2).ravel()
    values = np.asarray(phi, dtype=np.float64).ravel()
    dr = float(bin_width) if bin_width is not None else 0.5 * min(spacing_vals)
    if dr <= 0.0:
        raise ValueError("bin_width must be positive")
    max_r = float(np.max(radii))
    edges = np.arange(0.0, max_r + 2.0 * dr, dr)
    if len(edges) < 3:
        return float("nan")
    bin_ids = np.clip(np.digitize(radii, edges) - 1, 0, len(edges) - 2)
    counts = np.bincount(bin_ids, minlength=len(edges) - 1).astype(np.float64)
    sums = np.bincount(bin_ids, weights=values, minlength=len(edges) - 1)
    valid = counts > 0
    centers = 0.5 * (edges[:-1] + edges[1:])
    profile = np.full_like(centers, np.nan, dtype=np.float64)
    profile[valid] = sums[valid] / counts[valid]
    valid_idx = np.flatnonzero(np.isfinite(profile))
    if len(valid_idx) < 2:
        return float("nan")
    centers = centers[valid_idx]
    profile = profile[valid_idx]

    # Profile should mostly decrease.  Use the first outward crossing from above.
    above = profile >= threshold
    crossing = np.flatnonzero(above[:-1] & ~above[1:])
    if len(crossing) == 0:
        if np.nanmax(profile) < threshold:
            return 0.0
        return float("nan")
    i = int(crossing[0])
    r0, r1 = float(centers[i]), float(centers[i + 1])
    p0, p1 = float(profile[i]), float(profile[i + 1])
    if abs(p1 - p0) < 1e-12:
        return r0
    frac = (threshold - p0) / (p1 - p0)
    return float(r0 + np.clip(frac, 0.0, 1.0) * (r1 - r0))


def marching_cubes_radius_3d_np(
    phi: Array,
    spacings: Sequence[float],
    level: float = 0.5,
) -> dict[str, float | bool | str | None]:
    """Return a marching-cubes isovolume radius when scikit-image is available."""

    spacing_vals = tuple(float(s) for s in spacings)
    if phi.ndim != 3 or len(spacing_vals) != 3:
        raise ValueError("marching_cubes_radius_3d_np requires a 3D field and three spacings")
    if float(np.nanmin(phi)) > level or float(np.nanmax(phi)) < level:
        return {
            "marching_cubes_available": False,
            "marching_cubes_radius": float("nan"),
            "marching_cubes_surface_area": float("nan"),
            "marching_cubes_volume": float("nan"),
            "marching_cubes_error": "level outside data range",
        }
    try:
        from skimage import measure  # type: ignore
    except Exception as exc:  # pragma: no cover - depends on optional environment
        return {
            "marching_cubes_available": False,
            "marching_cubes_radius": float("nan"),
            "marching_cubes_surface_area": float("nan"),
            "marching_cubes_volume": float("nan"),
            "marching_cubes_error": str(exc),
        }
    try:
        verts, faces, _, _ = measure.marching_cubes(
            np.asarray(phi, dtype=np.float32),
            level=level,
            spacing=spacing_vals,
        )
        tris = verts[faces]
        cross = np.cross(tris[:, 1], tris[:, 2])
        signed_volume = float(np.sum(np.einsum("ij,ij->i", tris[:, 0], cross)) / 6.0)
        volume = abs(signed_volume)
        area = float(measure.mesh_surface_area(verts, faces))
        radius = float((max(volume, 0.0) * 3.0 / (4.0 * np.pi)) ** (1.0 / 3.0))
        return {
            "marching_cubes_available": True,
            "marching_cubes_radius": radius,
            "marching_cubes_surface_area": area,
            "marching_cubes_volume": volume,
            "marching_cubes_error": None,
        }
    except Exception as exc:
        return {
            "marching_cubes_available": False,
            "marching_cubes_radius": float("nan"),
            "marching_cubes_surface_area": float("nan"),
            "marching_cubes_volume": float("nan"),
            "marching_cubes_error": str(exc),
        }


def inertia_sphericity_3d_np(
    phi: Array,
    spacings: Sequence[float],
    threshold: float = 0.5,
) -> dict[str, float]:
    """Return inertia/covariance sanity metrics for the hard grain mask."""

    spacing_vals = tuple(float(s) for s in spacings)
    if phi.ndim != 3 or len(spacing_vals) != 3:
        raise ValueError("inertia_sphericity_3d_np requires a 3D field and three spacings")
    mask = phi > threshold
    if not np.any(mask):
        return {
            "inertia_axis_ratio": float("nan"),
            "inertia_eigen_cv": float("nan"),
            "inertia_points": 0.0,
        }
    dz, dy, dx = spacing_vals
    points = np.argwhere(mask).astype(np.float64)
    points[:, 0] *= dz
    points[:, 1] *= dy
    points[:, 2] *= dx
    centered = points - np.mean(points, axis=0, keepdims=True)
    cov = centered.T @ centered / max(len(points), 1)
    eig = np.linalg.eigvalsh(cov)
    eig = np.clip(eig, 0.0, None)
    if float(np.min(eig)) <= 1e-18:
        axis_ratio = float("nan")
    else:
        axis_ratio = float(np.sqrt(float(np.max(eig)) / float(np.min(eig))))
    mean_eig = float(np.mean(eig))
    eigen_cv = float(np.std(eig) / mean_eig) if mean_eig > 1e-18 else float("nan")
    return {
        "inertia_axis_ratio": axis_ratio,
        "inertia_eigen_cv": eigen_cv,
        "inertia_points": float(len(points)),
    }


def radius_estimators_3d(
    phi: Array,
    spacings: Sequence[float],
    *,
    threshold: float = 0.5,
    include_marching: bool = True,
) -> dict[str, float | bool | str | None]:
    """Return multiple independent 3D radius and sphericity estimators."""

    out: dict[str, float | bool | str | None] = {
        "soft_volume_radius": equivalent_radius_3d_np(phi, spacings),
        "hard_phi05_radius": hard_radius_3d_np(phi, spacings, threshold=threshold),
        "radial_profile_phi05_radius": radial_profile_radius_3d_np(
            phi, spacings, threshold=threshold
        ),
    }
    out.update(inertia_sphericity_3d_np(phi, spacings, threshold=threshold))
    if include_marching:
        out.update(marching_cubes_radius_3d_np(phi, spacings, level=threshold))
    else:
        out.update(
            {
                "marching_cubes_available": False,
                "marching_cubes_radius": float("nan"),
                "marching_cubes_surface_area": float("nan"),
                "marching_cubes_volume": float("nan"),
                "marching_cubes_error": "not evaluated for this frame",
            }
        )
    return out


def radius_estimator_series_3d(
    states: Array,
    spacings: Sequence[float],
    *,
    threshold: float = 0.5,
    marching_stride: int | None = None,
) -> dict[str, Array]:
    """Return estimator time series for ``states`` shaped ``(T,D,H,W)``."""

    if states.ndim != 4:
        raise ValueError("radius_estimator_series_3d requires states shaped (T,D,H,W)")
    stride = None if marching_stride is None else max(int(marching_stride), 1)
    rows: list[dict[str, float | bool | str | None]] = []
    for idx, state in enumerate(states):
        include_mc = stride is None or idx % stride == 0 or idx == len(states) - 1
        rows.append(
            radius_estimators_3d(
                state,
                spacings,
                threshold=threshold,
                include_marching=include_mc,
            )
        )
    numeric_keys = [
        "soft_volume_radius",
        "hard_phi05_radius",
        "radial_profile_phi05_radius",
        "marching_cubes_radius",
        "marching_cubes_surface_area",
        "marching_cubes_volume",
        "inertia_axis_ratio",
        "inertia_eigen_cv",
        "inertia_points",
    ]
    return {
        key: np.asarray([float(row.get(key, float("nan")) or float("nan")) for row in rows])
        for key in numeric_keys
    }


def face_shell_error_3d(
    predicted: Array,
    reference: Array,
    bands: Sequence[tuple[int, int]] = ((0, 1), (2, 3), (4, 7), (8, 999)),
) -> dict[str, float | None]:
    """Return mean absolute error by distance-to-face shell.

    ``predicted`` and ``reference`` may be single fields ``(D,H,W)`` or time
    series ``(T,D,H,W)``.  The final band is treated as the interior denominator
    for ``outer_interior_ratio``.
    """

    pred = np.asarray(predicted, dtype=np.float64)
    ref = np.asarray(reference, dtype=np.float64)
    if pred.shape != ref.shape:
        raise ValueError("predicted and reference must have matching shapes")
    if pred.ndim == 3:
        diff = np.abs(pred[None, ...] - ref[None, ...])
    elif pred.ndim == 4:
        diff = np.abs(pred - ref)
    else:
        raise ValueError("face_shell_error_3d expects (D,H,W) or (T,D,H,W)")

    depth, height, width = diff.shape[-3:]
    zz, yy, xx = np.indices((depth, height, width))
    dist = np.minimum.reduce(
        [zz, yy, xx, depth - 1 - zz, height - 1 - yy, width - 1 - xx]
    )
    out: dict[str, float | None] = {}
    labels = [f"{lo}-{hi}vox" for lo, hi in bands]
    for label, (lo, hi) in zip(labels, bands):
        mask = (dist >= int(lo)) & (dist <= int(hi))
        out[label] = float(diff[:, mask].mean()) if np.any(mask) else None
    outer = out.get(labels[0])
    interior = out.get(labels[-1])
    out["outer_interior_ratio"] = (
        float(outer / interior)
        if outer is not None and interior is not None and interior > 1.0e-12
        else None
    )
    return out


def residual_components_3d(phi: Array, threshold: float = 0.5) -> list[dict[str, Any]]:
    """Return component stats for thresholded residual 3D phase fields."""

    field = np.asarray(phi)
    if field.ndim != 3:
        raise ValueError("residual_components_3d expects a single (D,H,W) field")
    try:
        from scipy.ndimage import label
    except Exception:
        return []
    labeled, n_components = label(field > threshold)
    depth, height, width = field.shape
    components: list[dict[str, Any]] = []
    for component_id in range(1, int(n_components) + 1):
        mask = labeled == component_id
        zz, yy, xx = np.where(mask)
        if zz.size == 0:
            continue
        components.append(
            {
                "component_id": component_id,
                "volume_voxels": int(mask.sum()),
                "bbox_zyx_minmax": [
                    int(zz.min()),
                    int(yy.min()),
                    int(xx.min()),
                    int(zz.max()),
                    int(yy.max()),
                    int(xx.max()),
                ],
                "max_phi": float(field[mask].max()),
                "min_face_distance_voxels": int(
                    min(
                        zz.min(),
                        yy.min(),
                        xx.min(),
                        depth - 1 - zz.max(),
                        height - 1 - yy.max(),
                        width - 1 - xx.max(),
                    )
                ),
            }
        )
    return components


def r2_slope_3d(
    times: Array,
    states: Array,
    spacings: Sequence[float],
    mu: float,
    sigma: float,
    r0_threshold: float = 0.05,
) -> dict[str, float]:
    """Fit R²(t) and compute slope error vs the 3D law dR²/dt = -4μσ.

    Parameters
    ----------
    times:
        1D array of physical times, length T.
    states:
        Array of shape (T, D, H, W) — one scalar phi frame per time step.
    spacings:
        Three positive grid spacings (dz, dy, dx).
    mu, sigma:
        Physical parameters.
    r0_threshold:
        Frames with R < r0_threshold * R0 are excluded from the fit.

    Returns
    -------
    dict with keys:
        r0_px, r0_phys, slope, slope_error_pct, intercept, r_squared,
        theoretical_slope (= -4*mu*sigma), theoretical_t_ext, near_extinction_time.
    """
    spacing_vals = tuple(float(s) for s in spacings)
    dx = spacing_vals[0]
    radii = np.array([equivalent_radius_3d_np(s, spacing_vals) for s in states])
    r2 = radii ** 2
    r0 = radii[0]
    theoretical_slope = -_3D_LAW_FACTOR * mu * sigma
    theoretical_t_ext = r0 ** 2 / (_3D_LAW_FACTOR * mu * sigma)

    alive = radii > r0_threshold * r0
    if alive.sum() < 3:
        return {
            "r0_px": float(r0 / dx), "r0_phys": float(r0),
            "slope": float("nan"), "slope_error_pct": float("nan"),
            "intercept": float("nan"), "r_squared": float("nan"),
            "theoretical_slope": float(theoretical_slope),
            "theoretical_t_ext": float(theoretical_t_ext),
            "near_extinction_time": None,
        }

    slope, intercept = np.polyfit(times[alive], r2[alive], 1)
    pred = slope * times[alive] + intercept
    ss_res = float(np.sum((r2[alive] - pred) ** 2))
    ss_tot = float(np.sum((r2[alive] - np.mean(r2[alive])) ** 2))
    r_sq = 1.0 - ss_res / ss_tot if ss_tot > 0.0 else 1.0
    slope_err = abs(float(slope) - float(theoretical_slope)) / abs(float(theoretical_slope)) * 100.0

    near_ext = np.flatnonzero(radii < 0.01 * r0)
    near_extinction_time = float(times[near_ext[0]]) if len(near_ext) > 0 else None

    return {
        "r0_px": float(r0 / dx),
        "r0_phys": float(r0),
        "slope": float(slope),
        "slope_error_pct": slope_err,
        "intercept": float(intercept),
        "r_squared": float(r_sq),
        "theoretical_slope": float(theoretical_slope),
        "theoretical_t_ext": float(theoretical_t_ext),
        "near_extinction_time": near_extinction_time,
    }


def center_of_mass_3d(phi: Array, spacings: Sequence[float]) -> tuple[float, float, float]:
    """Return (z, y, x) volume-weighted centre-of-mass in physical units."""
    spacing_vals = tuple(float(s) for s in spacings)
    if phi.ndim != 3 or len(spacing_vals) != 3:
        raise ValueError("center_of_mass_3d requires a 3D field and three spacings")
    Nz, Ny, Nx = phi.shape
    dz, dy, dx = spacing_vals
    mass = float(np.sum(phi))
    if mass < 1e-15:
        return (Nz * dz / 2.0, Ny * dy / 2.0, Nx * dx / 2.0)
    z1d = np.arange(Nz, dtype=np.float64) * dz
    y1d = np.arange(Ny, dtype=np.float64) * dy
    x1d = np.arange(Nx, dtype=np.float64) * dx
    cz = float(np.sum(phi * z1d[:, None, None])) / mass
    cy = float(np.sum(phi * y1d[None, :, None])) / mass
    cx = float(np.sum(phi * x1d[None, None, :])) / mass
    return cz, cy, cx


def com_drift_3d(states: Array, spacings: Sequence[float]) -> float:
    """Return maximum CoM displacement over a rollout in pixel units."""
    spacing_vals = tuple(float(s) for s in spacings)
    dx = spacing_vals[0]
    coms = np.array([center_of_mass_3d(s, spacing_vals) for s in states])
    diffs = coms - coms[0]
    displacements = np.sqrt(np.sum(diffs ** 2, axis=1))
    return float(np.max(displacements)) / dx


def sphericity_proxy_3d(
    phi: Array,
    spacings: Sequence[float],
    n_directions: int = 64,
) -> dict[str, float]:
    """Compute a radial anisotropy proxy: radial std / mean radius.

    Samples voxel distances from the CoM for φ>0.5 voxels.
    Returns the coefficient of variation of those distances (std/mean).
    A perfect sphere gives 0; anisotropic shapes give higher values.
    """
    spacing_vals = tuple(float(s) for s in spacings)
    dz, dy, dx = spacing_vals
    Nz, Ny, Nx = phi.shape
    com = center_of_mass_3d(phi, spacing_vals)

    z1d = np.arange(Nz, dtype=np.float64) * dz
    y1d = np.arange(Ny, dtype=np.float64) * dy
    x1d = np.arange(Nx, dtype=np.float64) * dx
    zz, yy, xx = np.meshgrid(z1d, y1d, x1d, indexing="ij")
    dist = np.sqrt((zz - com[0]) ** 2 + (yy - com[1]) ** 2 + (xx - com[2]) ** 2)

    grain_mask = phi > 0.5
    if not np.any(grain_mask):
        return {"anisotropy_cv": float("nan"), "mean_radius_phys": 0.0, "std_radius_phys": 0.0}

    r_grain = dist[grain_mask]
    mean_r = float(np.mean(r_grain))
    std_r = float(np.std(r_grain))
    cv = std_r / mean_r if mean_r > 1e-12 else float("nan")
    return {"anisotropy_cv": cv, "mean_radius_phys": mean_r, "std_radius_phys": std_r}


def extinction_time_3d(
    times: Array,
    states: Array,
    spacings: Sequence[float],
    r0_frac: float = 0.05,
) -> float | None:
    """Return the first time where R < r0_frac * R0, or None if not extinguished."""
    spacing_vals = tuple(float(s) for s in spacings)
    radii = np.array([equivalent_radius_3d_np(s, spacing_vals) for s in states])
    r0 = radii[0]
    near_ext = np.flatnonzero(radii < r0_frac * r0)
    return float(times[near_ext[0]]) if len(near_ext) > 0 else None


def volume_proxy_3d(states: Array, spacings: Sequence[float]) -> Array:
    """Return the discrete volume integral (mass proxy) for each frame."""
    spacing_vals = tuple(float(s) for s in spacings)
    cell_vol = float(np.prod(spacing_vals))
    return np.array([float(np.sum(s)) * cell_vol for s in states])


def phi_bounds_ok_3d(states: Array, tol: float = 1e-6) -> bool:
    """Return True if φ ∈ [−tol, 1+tol] across all frames."""
    return bool(states.min() >= -tol and states.max() <= 1.0 + tol)


def energy_monotone_3d(energies: Array, tol: float = 1e-10) -> bool:
    """Return True if energy is non-increasing across all consecutive frames."""
    return bool(np.all(np.diff(energies) <= tol))


def gate_report_3d(
    times: Array,
    states: Array,
    spacings: Sequence[float],
    energies: Array | None,
    mu: float,
    sigma: float,
    slope_error_gate_pct: float = 1.0,
) -> dict[str, Any]:
    """Collect all 3D gate metrics into a single report dict.

    Parameters
    ----------
    times:
        1D array of physical times, shape (T,).
    states:
        Array of shape (T, D, H, W).
    spacings:
        Three positive grid spacings (dz, dy, dx).
    energies:
        1D array of free energies, shape (T,), or None if not available.
    mu, sigma:
        Physical parameters.
    slope_error_gate_pct:
        Maximum allowed R² slope error (%). Default 1%.

    Returns
    -------
    dict with gate metrics and a ``gate_pass`` bool.
    """
    spacing_vals = tuple(float(s) for s in spacings)
    dx = spacing_vals[0]

    r2_info = r2_slope_3d(times, states, spacing_vals, mu, sigma)
    drift_px = com_drift_3d(states, spacing_vals)
    sph_info = sphericity_proxy_3d(states[0], spacing_vals)
    vol_series = volume_proxy_3d(states, spacing_vals)
    phi_ok = phi_bounds_ok_3d(states)
    e_mono = energy_monotone_3d(energies) if energies is not None else None
    t_ext = extinction_time_3d(times, states, spacing_vals)
    final_r_px = float(equivalent_radius_3d_np(states[-1], spacing_vals) / dx)

    radii = np.array([equivalent_radius_3d_np(s, spacing_vals) for s in states])
    sphere_extinguished = bool(radii[-1] < 0.01 * radii[0])

    slope_err = r2_info["slope_error_pct"]
    slope_gate_pass = not np.isnan(slope_err) and slope_err < slope_error_gate_pct

    gate_pass = (
        slope_gate_pass
        and phi_ok
        and sphere_extinguished
        and (e_mono is None or e_mono)
    )

    return {
        "gate_pass": gate_pass,
        "3d_law": "R²(t) = R0² − 4·μ·σ·t   (dR²/dt = −4μσ, NOT −2μσ)",
        "r2_slope": r2_info,
        "extinction_time": t_ext,
        "final_radius_px": final_r_px,
        "sphere_extinguished": sphere_extinguished,
        "phi_bounds_ok": phi_ok,
        "energy_monotone": e_mono,
        "com_drift_px": drift_px,
        "sphericity_at_t0": sph_info,
        "volume_initial": float(vol_series[0]),
        "volume_final": float(vol_series[-1]),
        "slope_error_gate_pct": slope_error_gate_pct,
    }
