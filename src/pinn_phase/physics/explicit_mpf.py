"""Explicit multichannel MPF physics utilities.

These helpers are intentionally separate from the scalar Allen-Cahn path.  They
operate on explicit phase stacks shaped ``(batch, num_phases, height, width)``
and are safe to use in training because the loader exposes only ``Phi(t=0)``
from PF/MPF reference artifacts.  Saved frames after ``t=0`` remain audit-only.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor
import yaml

from pinn_phase.io.artifacts import load_npz_arrays


@dataclass(frozen=True)
class ExplicitMPFReference:
    """Reference metadata plus the initial explicit phase stack only."""

    benchmark_id: str
    reference_id: str
    phi0: np.ndarray
    num_phases: int
    grid: tuple[int, ...]
    eta_px: float
    sigma: float
    mu: float
    delta_g: float
    model_dt: float
    dt_mu_sigma: float
    horizon_steps: int
    save_steps: tuple[int, ...]
    reference_artifact: str
    reference_config: str
    reference_usage_policy: str = "Phi_ref(t=0) only for training; frames t>0 audit-only"


def _repo_path(value: str | Path, *, base: Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else base / path


def load_explicit_mpf_initial_reference(
    config_path: str | Path,
    *,
    repo_root: str | Path | None = None,
) -> ExplicitMPFReference:
    """Load only ``Phi_ref(t=0)`` and metadata from an explicit-MPF benchmark.

    The returned object deliberately omits reference frames after ``t=0`` so
    callers cannot accidentally use them as training labels, graph features, or
    latent targets.  Audit scripts should load their own reference trajectories
    explicitly and keep that path outside training losses.
    """

    cfg_path = Path(config_path)
    root = Path(repo_root) if repo_root is not None else cfg_path.resolve().parents[3]
    if not cfg_path.is_absolute():
        cfg_path = root / cfg_path
    raw = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("explicit-MPF benchmark config must be a mapping")

    validation = raw.get("validation") or {}
    if validation.get("training_allowed") is not False:
        raise ValueError("explicit-MPF reference must keep validation.training_allowed=false")
    if validation.get("reference_use", "audit_only") != "audit_only":
        raise ValueError("explicit-MPF reference must be marked audit_only")

    physics = raw.get("physics") or {}
    grid = raw.get("grid") or {}
    time = raw.get("time") or {}
    benchmark = raw.get("benchmark") or {}
    artifact_value = validation.get("training_t0_artifact")
    if not artifact_value:
        raise ValueError("validation.training_t0_artifact is required")
    artifact_sha256 = validation.get("training_t0_sha256")
    if not artifact_sha256:
        raise ValueError("validation.training_t0_sha256 is required")
    artifact = _repo_path(str(artifact_value), base=root)
    arrays = load_npz_arrays(
        artifact,
        expected_sha256=str(artifact_sha256),
        expected_keys=frozenset({"phi0"}),
    )
    if frozenset(arrays) != {"phi0"}:
        raise ValueError(
            "training initial-condition archive must contain exactly the 'phi0' member; "
            f"found {sorted(arrays)}"
        )
    phi0 = np.asarray(arrays["phi0"], dtype=np.float32)
    save_steps = (0,)

    num_phases = int(physics["num_phases"])
    # Accept 2D `grid.shape`, 3D `grid.array_shape_zyx` (array layout ZYX),
    # or 3D `grid.shape_xyz` (physical XYZ order; reversed to get array ZYX).
    if "shape" in grid:
        shape = tuple(int(v) for v in grid["shape"])
    elif "array_shape_zyx" in grid:
        shape = tuple(int(v) for v in grid["array_shape_zyx"])
    elif "shape_xyz" in grid:
        shape = tuple(int(v) for v in reversed(grid["shape_xyz"]))
    else:
        raise ValueError("grid must provide 'shape', 'array_shape_zyx', or 'shape_xyz'")
    if phi0.shape != (num_phases,) + shape:
        raise ValueError(f"Phi_ref(t=0) shape {phi0.shape} does not match {(num_phases,) + shape}")

    sigma = float(physics["sigma"])
    mu = float(physics["mu"])
    # ``dt_mu_sigma`` is the reference's *combined* explicit-Euler coefficient that
    # multiplies the unscaled operator ``raw = laplacian(phi) - W'(phi)/eta_px^2``
    # used by the accepted explicit-Euler reference implementation:
    #     phi += dt_mu_sigma * raw   with   dt_mu_sigma = dt_physical * mu * sigma.
    # ``explicit_mpf_rhs`` returns the physically scaled rate ``mu*sigma*raw``, so
    # the integration timestep that reproduces the reference is the PHYSICAL dt,
    # ``model_dt = dt_mu_sigma / (mu*sigma)``, NOT ``dt_mu_sigma`` itself.  Using
    # ``dt_mu_sigma`` directly would apply the ``mu*sigma`` factor twice and shrink
    # every step by ``1/(mu*sigma)``.
    dt_mu_sigma = physics.get("dt_mu_sigma", physics.get("effective_update_coefficient_dt_mu_sigma"))
    dt_mu_sigma = float(dt_mu_sigma if dt_mu_sigma is not None else 0.05)
    if mu * sigma <= 0.0:
        raise ValueError("mu*sigma must be positive to derive model_dt from dt_mu_sigma")
    model_dt = dt_mu_sigma / (mu * sigma)
    return ExplicitMPFReference(
        benchmark_id=str(benchmark["id"]),
        reference_id=str(benchmark["id"]),
        phi0=phi0,
        num_phases=num_phases,
        grid=shape,  # type: ignore[arg-type]
        eta_px=float(physics["eta_px"]),
        sigma=sigma,
        mu=mu,
        delta_g=float(physics.get("delta_g", 0.0)),
        model_dt=model_dt,
        dt_mu_sigma=dt_mu_sigma,
        horizon_steps=int(time.get("steps", 0)),
        save_steps=save_steps,
        reference_artifact=str(artifact_value),
        reference_config=str(cfg_path.relative_to(root) if cfg_path.is_relative_to(root) else cfg_path),
    )


def periodic_laplacian_2d_channels(phi: Tensor) -> Tensor:
    """Return a unit-grid periodic Laplacian for ``(B,N,H,W)`` phase stacks."""

    _validate_phi(phi)
    if phi.ndim != 4:
        raise ValueError("periodic_laplacian_2d_channels requires a 4D tensor [B,N,H,W]")
    return (
        torch.roll(phi, 1, dims=-2)
        + torch.roll(phi, -1, dims=-2)
        + torch.roll(phi, 1, dims=-1)
        + torch.roll(phi, -1, dims=-1)
        - 4.0 * phi
    )


def periodic_laplacian_3d_channels(phi: Tensor) -> Tensor:
    """Return a unit-grid periodic Laplacian for ``(B,N,Z,Y,X)`` phase stacks."""

    _validate_phi(phi)
    if phi.ndim != 5:
        raise ValueError("periodic_laplacian_3d_channels requires a 5D tensor [B,N,Z,Y,X]")
    return (
        torch.roll(phi, 1, dims=-3) + torch.roll(phi, -1, dims=-3)  # z neighbours
        + torch.roll(phi, 1, dims=-2) + torch.roll(phi, -1, dims=-2)  # y neighbours
        + torch.roll(phi, 1, dims=-1) + torch.roll(phi, -1, dims=-1)  # x neighbours
        - 6.0 * phi
    )


def double_well_prime(phi: Tensor) -> Tensor:
    """Derivative of ``W(phi)=phi^2(1-phi)^2`` used by the V2 MPF references."""

    return 2.0 * phi * (1.0 - phi) * (1.0 - 2.0 * phi)


def explicit_mpf_rhs(
    phi: Tensor,
    *,
    eta_px: float,
    mu: float = 1.0,
    sigma: float = 1.0,
    boundary: str = "periodic",
) -> Tensor:
    """Compute the synchronized explicit-MPF RHS over all phase channels.

    ``raw_i = laplacian(phi_i) - W'(phi_i)/eta_px^2`` and the phase-channel
    mean is removed at each pixel so ``sum_i dphi_i/dt = 0``.  All channels are
    evaluated at the same collocation/sample positions.

    TIMESTEP CONVENTION (do not break):
    This returns the PHYSICAL rate ``dphi/dt = mu*sigma*raw``.  The PF/MPF
    reference notebooks integrate the UNSCALED operator with the combined
    coefficient ``dt_mu_sigma``: ``phi_next = phi + dt_mu_sigma * raw`` with
    ``dt_mu_sigma = dt_physical * mu * sigma``.  To reproduce the reference with
    this physical RHS the integration timestep MUST be
    ``model_dt = dt_mu_sigma / (mu*sigma)`` (e.g. ``0.05 / 1.5e-5 = 3333.33``),
    NOT ``model_dt = dt_mu_sigma``.  Setting ``model_dt = dt_mu_sigma`` applies
    ``mu*sigma`` twice and shrinks every step by ``1/(mu*sigma) ~ 66667x``.
    See ``load_explicit_mpf_initial_reference`` (derives model_dt) and the
    regression tests in ``tests/unit/test_explicit_mpf_residual.py``.
    """

    _validate_phi(phi)
    if boundary != "periodic":
        raise NotImplementedError("explicit-MPF dynamics currently support periodic boundaries")
    if eta_px <= 0.0 or mu <= 0.0 or sigma <= 0.0:
        raise ValueError("eta_px, mu, and sigma must be positive")
    if phi.ndim == 5:
        lap = periodic_laplacian_3d_channels(phi)
    else:
        lap = periodic_laplacian_2d_channels(phi)
    raw = lap - double_well_prime(phi) / (eta_px * eta_px)
    raw = raw - raw.mean(dim=1, keepdim=True)
    return float(mu) * float(sigma) * raw


def explicit_mpf_residual(
    phi: Tensor,
    model_dphi_dt: Tensor,
    *,
    eta_px: float,
    mu: float = 1.0,
    sigma: float = 1.0,
    boundary: str = "periodic",
) -> Tensor:
    """Return ``R_i = dphi_i/dt_model - RHS_i(Phi)``."""

    if phi.shape != model_dphi_dt.shape:
        raise ValueError("phi and model_dphi_dt must have the same shape")
    return model_dphi_dt - explicit_mpf_rhs(
        phi,
        eta_px=eta_px,
        mu=mu,
        sigma=sigma,
        boundary=boundary,
    )


def project_simplex(
    phi_raw: Tensor,
    *,
    eps: float = 1.0e-12,
    active_threshold: float = 0.0,
) -> tuple[Tensor, dict[str, Tensor | float]]:
    """Clip and normalize an explicit phase stack onto the simplex.

    Zero-sum pixels are assigned a uniform phase distribution before the final
    normalization.  Diagnostics are tensor-friendly where a caller may want to
    keep gradients out of logging, and scalar values are detached floats.
    """

    _validate_phi(phi_raw)
    if eps <= 0.0:
        raise ValueError("eps must be positive")
    pre_sum_error = torch.abs(phi_raw.sum(dim=1) - 1.0)
    pre_bounds_low = torch.clamp_min(-phi_raw, 0.0)
    pre_bounds_high = torch.clamp_min(phi_raw - 1.0, 0.0)

    clipped = torch.clamp(phi_raw, 0.0, 1.0)
    denom = clipped.sum(dim=1, keepdim=True)
    zero_sum = denom <= eps
    if torch.any(zero_sum):
        clipped = torch.where(zero_sum.expand_as(clipped), torch.ones_like(clipped), clipped)
        denom = clipped.sum(dim=1, keepdim=True)
    phi_next = clipped / denom.clamp_min(eps)

    correction = phi_next - phi_raw
    edited = torch.any(torch.abs(correction) > 1.0e-8, dim=1)
    post_sum_error = torch.abs(phi_next.sum(dim=1) - 1.0)
    post_bounds_low = torch.clamp_min(-phi_next, 0.0)
    post_bounds_high = torch.clamp_min(phi_next - 1.0, 0.0)
    per_phase_area = phi_next.flatten(2).sum(dim=-1)
    labels = torch.argmax(phi_next, dim=1)
    active_counts = []
    for batch_idx in range(phi_next.shape[0]):
        areas = torch.bincount(labels[batch_idx].reshape(-1), minlength=phi_next.shape[1])
        active_counts.append(torch.sum(areas > active_threshold))
    active_phase_count = torch.stack(active_counts).to(device=phi_next.device)

    diagnostics: dict[str, Tensor | float] = {
        "sum_error_pre_projection": _max_float(pre_sum_error),
        "sum_error_post_projection": _max_float(post_sum_error),
        "bounds_violation_pre_projection": _max_float(torch.maximum(pre_bounds_low, pre_bounds_high)),
        "bounds_violation_post_projection": _max_float(torch.maximum(post_bounds_low, post_bounds_high)),
        "projection_correction_l1": float(torch.mean(torch.abs(correction)).detach()),
        "projection_correction_l2": float(torch.sqrt(torch.mean(correction * correction)).detach()),
        "projection_correction_max": _max_float(torch.abs(correction)),
        "projected_pixel_fraction": float(torch.mean(edited.to(phi_raw.dtype)).detach()),
        "per_phase_area": per_phase_area.detach(),
        "active_phase_count": active_phase_count.detach(),
        "zero_sum_pixel_fraction": float(torch.mean(zero_sum.squeeze(1).to(phi_raw.dtype)).detach()),
    }
    return phi_next, diagnostics


def project_simplex_soft_threshold(
    phi_raw: Tensor,
    *,
    threshold_eps: float = 1.0e-3,
    active_threshold: float = 0.0,
) -> tuple[Tensor, dict[str, Tensor | float]]:
    """Soft-threshold projection: clip → zero sub-threshold entries → renormalize.

    Prevents phi_max collapse to 1/N under high-N clip_normalize (confirmed root cause
    for N>=64 Voronoi references, 2026-06-17). Diffuse interfaces have phi >> threshold_eps
    near boundaries, so coarsening dynamics are preserved. Validated at N=64, 128×128.
    """
    _validate_phi(phi_raw)
    pre_sum_error = torch.abs(phi_raw.sum(dim=1) - 1.0)
    clipped = torch.clamp(phi_raw, 0.0, 1.0)
    thresholded = torch.where(clipped < threshold_eps, torch.zeros_like(clipped), clipped)
    denom = thresholded.sum(dim=1, keepdim=True)
    zero_sum = denom <= 1.0e-12
    if torch.any(zero_sum):
        thresholded = torch.where(
            zero_sum.expand_as(thresholded), torch.ones_like(thresholded), thresholded
        )
        denom = thresholded.sum(dim=1, keepdim=True)
    phi_next = thresholded / denom.clamp_min(1.0e-12)

    correction = phi_next - phi_raw
    edited = torch.any(torch.abs(correction) > 1.0e-8, dim=1)
    post_sum_error = torch.abs(phi_next.sum(dim=1) - 1.0)
    post_bounds_low = torch.clamp_min(-phi_next, 0.0)
    post_bounds_high = torch.clamp_min(phi_next - 1.0, 0.0)
    per_phase_area = phi_next.flatten(2).sum(dim=-1)
    labels = torch.argmax(phi_next, dim=1)
    active_counts = []
    for batch_idx in range(phi_next.shape[0]):
        areas = torch.bincount(labels[batch_idx].reshape(-1), minlength=phi_next.shape[1])
        active_counts.append(torch.sum(areas > active_threshold))
    active_phase_count = torch.stack(active_counts).to(device=phi_next.device)

    diagnostics: dict[str, Tensor | float] = {
        "sum_error_pre_projection": _max_float(pre_sum_error),
        "sum_error_post_projection": _max_float(post_sum_error),
        "bounds_violation_pre_projection": _max_float(
            torch.maximum(torch.clamp_min(-phi_raw, 0.0), torch.clamp_min(phi_raw - 1.0, 0.0))
        ),
        "bounds_violation_post_projection": _max_float(torch.maximum(post_bounds_low, post_bounds_high)),
        "projection_correction_l1": float(torch.mean(torch.abs(correction)).detach()),
        "projection_correction_l2": float(torch.sqrt(torch.mean(correction * correction)).detach()),
        "projection_correction_max": _max_float(torch.abs(correction)),
        "projected_pixel_fraction": float(torch.mean(edited.to(phi_raw.dtype)).detach()),
        "per_phase_area": per_phase_area.detach(),
        "active_phase_count": active_phase_count.detach(),
        "zero_sum_pixel_fraction": float(torch.mean(zero_sum.squeeze(1).to(phi_raw.dtype)).detach()),
    }
    return phi_next, diagnostics


def binary_scalar_mpf_rhs(
    phi_one: Tensor,
    *,
    eta_px: float,
    mu: float = 1.0,
    sigma: float = 1.0,
) -> Tensor:
    """Return the channel-0 RHS for the N=2 simplex reduction."""

    if phi_one.ndim != 4 or phi_one.shape[1] != 1:
        raise ValueError("phi_one must have shape (B,1,H,W)")
    stack = torch.cat((phi_one, 1.0 - phi_one), dim=1)
    return explicit_mpf_rhs(stack, eta_px=eta_px, mu=mu, sigma=sigma)[:, :1]


def _validate_phi(phi: Tensor) -> None:
    if phi.ndim not in {4, 5}:
        raise ValueError(
            "explicit-MPF tensors must have shape (B,N,H,W) [4D] or (B,N,Z,H,W) [5D], "
            f"got {phi.ndim}D"
        )
    if phi.shape[1] < 2:
        raise ValueError("explicit-MPF tensors require at least two phase channels")


def _max_float(value: Tensor) -> float:
    return float(torch.max(value).detach()) if value.numel() else 0.0
