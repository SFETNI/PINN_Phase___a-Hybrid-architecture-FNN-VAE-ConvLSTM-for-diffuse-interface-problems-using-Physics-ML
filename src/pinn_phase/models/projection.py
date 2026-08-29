"""Bound-aware projection shared by model rollouts, training, and audits."""

from __future__ import annotations

import torch
from torch import Tensor

# Diagnostic bound-enforcement modes for reduced-projection audits.
#
# These are *diagnostic-only* labels for the scalar rollout's bound-enforcement
# step.  ``project_and_clamp`` reproduces the production default exactly (rate
# projection at ``boundary_eps`` followed by a hard [0, 1] clamp).  The other
# modes allow legacy checkpoints to be evaluated under reduced or absent bound
# enforcement without editing their production configuration.
PROJECT_AND_CLAMP = "project_and_clamp"
CLAMP_ONLY = "clamp_only"
UNBOUNDED = "unbounded"
BOUND_ENFORCEMENT_MODES = (PROJECT_AND_CLAMP, CLAMP_ONLY, UNBOUNDED)

# The production default. Any value other than this triple is a diagnostic mode.
DEFAULT_BOUND_ENFORCEMENT_MODE = PROJECT_AND_CLAMP
DEFAULT_BOUNDARY_EPS = 1.0e-3


def apply_bound_enforcement(
    phase: Tensor,
    derivative: Tensor,
    *,
    dt: float,
    mode: str = DEFAULT_BOUND_ENFORCEMENT_MODE,
    boundary_eps: float = DEFAULT_BOUNDARY_EPS,
) -> Tensor:
    """Advance one explicit step under a selectable bound-enforcement mode.

    Returns ``next_phase`` only.  This centralises the rate-projection + clamp
    logic so the production rollout and the reduced-projection diagnostic share
    one implementation.

    - ``project_and_clamp`` (default): outward-rate projection at ``boundary_eps``
      then a hard [0, 1] clamp.  Byte-for-byte the production behaviour.
    - ``clamp_only``: no rate projection; hard [0, 1] clamp retained.
    - ``unbounded``: neither projection nor clamp (negative control; collapses).
    """

    if mode not in BOUND_ENFORCEMENT_MODES:
        raise ValueError(
            f"mode must be one of {BOUND_ENFORCEMENT_MODES}; got {mode!r}"
        )
    if mode == PROJECT_AND_CLAMP:
        derivative = project_outward_boundary_rates(
            phase, derivative, boundary_eps=boundary_eps
        )
    next_phase = phase + float(dt) * derivative
    if mode in (PROJECT_AND_CLAMP, CLAMP_ONLY):
        next_phase = torch.clamp(next_phase, 0.0, 1.0)
    return next_phase


def project_outward_boundary_rates(
    phase: Tensor,
    derivative: Tensor,
    *,
    boundary_eps: float = 1.0e-3,
) -> Tensor:
    """Suppress rates that push an already bounded phase field outward."""

    if boundary_eps <= 0.0:
        raise ValueError("boundary_eps must be positive")
    if phase.shape != derivative.shape:
        raise ValueError("phase and derivative must have the same shape")
    at_lo = phase <= boundary_eps
    at_hi = phase >= 1.0 - boundary_eps
    derivative = torch.where(at_lo & (derivative < 0.0), torch.zeros_like(derivative), derivative)
    return torch.where(at_hi & (derivative > 0.0), torch.zeros_like(derivative), derivative)


def projection_diagnostic_stats(
    phase: Tensor,
    raw_derivative: Tensor,
    projected_derivative: Tensor,
    *,
    dt: float,
) -> dict[str, float]:
    """Measure how often and how strongly boundary projection edits a rate field.

    ``projected_pixel_fraction`` is intentionally a count-based diagnostic: a
    tiny outward rate in a bulk voxel counts the same as a large correction at
    an interface voxel.  The magnitude fields below record the mean absolute
    update correction, and the region fields split the count/magnitude by the
    model's own phase value before the update.
    """

    if phase.shape != raw_derivative.shape or phase.shape != projected_derivative.shape:
        raise ValueError("phase, raw_derivative, and projected_derivative must match")
    if dt <= 0.0:
        raise ValueError("dt must be positive")

    edited = projected_derivative != raw_derivative
    correction = torch.abs(raw_derivative - projected_derivative) * float(dt)
    total = max(int(phase.numel()), 1)
    edited_count = int(torch.count_nonzero(edited))
    projected_correction = correction[edited]

    out: dict[str, float] = {
        "projected_pixel_fraction": float(edited_count / total),
        "projection_update_abs_mean": float(torch.mean(correction).detach()),
        "projection_update_abs_mean_projected": (
            float(torch.mean(projected_correction).detach()) if edited_count else 0.0
        ),
        "projection_update_abs_max": float(torch.max(correction).detach()) if total else 0.0,
    }

    regions = {
        "bulk_phi_lt_005": phase < 0.05,
        "interface_phi_005_095": (phase >= 0.05) & (phase <= 0.95),
        "grain_phi_gt_095": phase > 0.95,
    }
    for name, mask in regions.items():
        count = int(torch.count_nonzero(mask))
        if count == 0:
            out[f"projected_pixel_fraction_{name}"] = 0.0
            out[f"projection_update_abs_mean_{name}"] = 0.0
            out[f"projection_voxel_fraction_{name}"] = 0.0
            continue
        region_edited = edited & mask
        out[f"projected_pixel_fraction_{name}"] = float(
            int(torch.count_nonzero(region_edited)) / count
        )
        out[f"projection_update_abs_mean_{name}"] = float(
            torch.mean(correction[mask]).detach()
        )
        out[f"projection_voxel_fraction_{name}"] = float(count / total)
    return out
