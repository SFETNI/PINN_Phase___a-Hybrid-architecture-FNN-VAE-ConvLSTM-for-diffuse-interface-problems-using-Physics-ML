"""Explicit-MPF projection and model-construction contract tests."""

from __future__ import annotations

import numpy as np
import torch

from pinn_phase.physics.explicit_mpf import (
    ExplicitMPFReference,
    project_simplex,
    project_simplex_soft_threshold,
)
from pinn_phase.training import build_explicit_mpf_model


def _synthetic_reference() -> ExplicitMPFReference:
    return ExplicitMPFReference(
        benchmark_id="synthetic-model-construction",
        reference_id="synthetic-model-construction",
        phi0=np.full((4, 6, 7), 0.25, dtype=np.float32),
        num_phases=4,
        grid=(6, 7),
        eta_px=3.0,
        sigma=1.0,
        mu=1.0,
        delta_g=0.0,
        model_dt=0.05,
        dt_mu_sigma=0.05,
        horizon_steps=8,
        save_steps=(0,),
        reference_artifact="synthetic",
        reference_config="synthetic",
    )


def _model_settings(*, projection_mode: str | None) -> dict:
    model = {
        "architecture": "explicit_mpf_ann_convlstm_hybrid",
        "hidden_channels": 4,
        "ann_hidden_features": [5],
        "graph": {"enabled": False},
    }
    if projection_mode is not None:
        model["projection_mode"] = projection_mode
    return {"model": model}


def test_project_simplex_enforces_sum_and_bounds() -> None:
    raw = torch.tensor(
        [[
            [[1.2, -0.1], [0.2, 0.3]],
            [[0.4, 0.4], [-0.2, 0.3]],
            [[0.2, 0.7], [0.8, 0.6]],
        ]],
        dtype=torch.float32,
    )

    projected, diag = project_simplex(raw)

    torch.testing.assert_close(projected.sum(dim=1), torch.ones(1, 2, 2))
    assert float(projected.min()) >= 0.0
    assert float(projected.max()) <= 1.0
    assert diag["sum_error_pre_projection"] > 0.0
    assert diag["sum_error_post_projection"] < 1.0e-6
    assert diag["projection_correction_l1"] > 0.0
    assert diag["projection_correction_l2"] > 0.0
    assert diag["projection_correction_max"] > 0.0
    assert diag["projected_pixel_fraction"] > 0.0
    assert diag["per_phase_area"].shape == (1, 3)
    assert diag["active_phase_count"].shape == (1,)


def test_project_simplex_zero_sum_pixels_are_safe_uniform() -> None:
    raw = torch.zeros(1, 4, 3, 3)

    projected, diag = project_simplex(raw)

    torch.testing.assert_close(projected, torch.full_like(projected, 0.25))
    torch.testing.assert_close(projected.sum(dim=1), torch.ones(1, 3, 3))
    assert diag["zero_sum_pixel_fraction"] == 1.0


def test_soft_threshold_kills_background_below_eps_and_preserves_dominant() -> None:
    """Core mechanism that prevents phi_max collapse at high N (no config needed)."""
    N = 64
    H, W = 4, 4
    background_val = 0.05 / (N - 1)
    dominant_val = 1.0 - background_val * (N - 1)
    phi = torch.full((1, N, H, W), background_val)
    phi[:, 0, :, :] = dominant_val
    assert abs(float(phi.sum(dim=1).mean()) - 1.0) < 1e-5

    proj_soft, _ = project_simplex_soft_threshold(phi.clone(), threshold_eps=1.0e-3)
    assert float(proj_soft[:, 0, :, :].max()) > 0.99, "soft_threshold did not preserve dominant channel"
    assert float(proj_soft[:, 1:, :, :].max()) < 1e-6, "soft_threshold did not kill background"

    proj_clip, _ = project_simplex(phi.clone())
    assert float(proj_clip[:, 1:, :, :].max()) > 1e-5, (
        "clip_normalize should preserve non-negative background channels"
    )
    assert float(proj_clip[:, 0, :, :].min()) < 1.0 - 1e-4, (
        "clip_normalize should NOT fully concentrate mass in dominant channel"
    )


def test_build_explicit_mpf_model_forwards_soft_threshold_projection_mode() -> None:
    """build_explicit_mpf_model must pass projection_mode from config to the model.

    Regression check retained from the source suite: the original
    build function once omitted projection_mode, causing the model to always
    use 'simplex_clip_normalize' regardless of config.
    """
    settings = _model_settings(projection_mode="soft_threshold_eps1e3")
    reference = _synthetic_reference()
    model = build_explicit_mpf_model(settings, reference)
    assert model.projection_mode == settings.get("model", {}).get("projection_mode", model.projection_mode)


def test_build_explicit_mpf_model_defaults_to_clip_normalize_when_unset() -> None:
    """Without projection_mode in the model section, build defaults to simplex_clip_normalize."""
    settings = _model_settings(projection_mode=None)
    reference = _synthetic_reference()
    model = build_explicit_mpf_model(settings, reference)
    assert model.projection_mode == "simplex_clip_normalize", (
        f"Expected default simplex_clip_normalize but got {model.projection_mode!r}."
    )
