"""Explicit-MPF 3D tests using compact synthetic contract fixtures."""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

from pinn_phase.models.cells import ConvLSTMCell3d
from pinn_phase.models.explicit_mpf import ExplicitMPFGraphExtractor, ExplicitMPFHybridRollout
from pinn_phase.physics.explicit_mpf import (
    explicit_mpf_rhs,
    load_explicit_mpf_initial_reference,
    periodic_laplacian_2d_channels,
    periodic_laplacian_3d_channels,
    project_simplex,
    project_simplex_soft_threshold,
)

def _simplex_3d(b: int = 1, n: int = 4, d: int = 3, h: int = 4, w: int = 4) -> torch.Tensor:
    torch.manual_seed(99)
    phi = torch.rand(b, n, d, h, w)
    phi = phi / phi.sum(dim=1, keepdim=True)
    return phi


def _load_synthetic_3d_reference(tmp_path: Path):
    phi0 = _simplex_3d(b=1, n=4, d=3, h=4, w=5)[0].numpy()
    artifact = tmp_path / "phi0.npz"
    np.savez(artifact, phi0=phi0)
    digest = hashlib.sha256(artifact.read_bytes()).hexdigest()
    config = {
        "benchmark": {"id": "synthetic-loader-contract"},
        "grid": {"array_shape_zyx": [3, 4, 5]},
        "physics": {
            "num_phases": 4,
            "eta_px": 3.0,
            "sigma": 2.0,
            "mu": 0.5,
            "delta_g": 0.0,
            "dt_mu_sigma": 0.05,
        },
        "time": {"steps": 8},
        "validation": {
            "training_allowed": False,
            "reference_use": "audit_only",
            "training_t0_artifact": artifact.name,
            "training_t0_sha256": digest,
        },
    }
    config_path = tmp_path / "benchmark.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    return load_explicit_mpf_initial_reference(config_path, repo_root=tmp_path)


# ---------------------------------------------------------------------------
# 1. 3D reference loader
# ---------------------------------------------------------------------------

def test_3d_loader_exposes_only_phi0_not_post_t0(tmp_path: Path) -> None:
    ref = _load_synthetic_3d_reference(tmp_path)
    assert ref.phi0.ndim == 4, "phi0 must be (N, Z, Y, X) -- 4D, no time or batch dim"
    assert not hasattr(ref, "states"), "loader must not expose full trajectory"


def test_3d_loader_model_dt_is_physical(tmp_path: Path) -> None:
    ref = _load_synthetic_3d_reference(tmp_path)
    expected = ref.dt_mu_sigma / (ref.mu * ref.sigma)
    assert abs(ref.model_dt - expected) < 1e-6, (
        f"model_dt {ref.model_dt} != dt_mu_sigma/(mu*sigma) {expected}"
    )


# ---------------------------------------------------------------------------
# 2. 3D projection
# ---------------------------------------------------------------------------

def test_3d_project_simplex_shape_sum_bounds() -> None:
    phi = _simplex_3d()
    phi = phi * 1.1 - 0.05

    projected, diag = project_simplex(phi)

    assert projected.shape == phi.shape
    torch.testing.assert_close(projected.sum(dim=1), torch.ones(1, 3, 4, 4), atol=1e-5, rtol=0.0)
    assert float(projected.min()) >= 0.0
    assert float(projected.max()) <= 1.0 + 1e-6
    assert diag["per_phase_area"].shape == (1, 4)
    assert diag["active_phase_count"].shape == (1,)


def test_3d_project_simplex_soft_threshold_shape_sum_bounds() -> None:
    phi = _simplex_3d()
    phi = phi * 1.1 - 0.05

    projected, diag = project_simplex_soft_threshold(phi, threshold_eps=1e-3)

    assert projected.shape == phi.shape
    torch.testing.assert_close(projected.sum(dim=1), torch.ones(1, 3, 4, 4), atol=1e-5, rtol=0.0)
    assert diag["per_phase_area"].shape == (1, 4)


def test_3d_per_phase_area_sums_over_all_spatial_dims() -> None:
    phi = _simplex_3d(b=1, n=4, d=2, h=3, w=3)
    phi, diag = project_simplex(phi)
    total_area = diag["per_phase_area"].sum().item()
    expected = float(2 * 3 * 3)
    assert abs(total_area - expected) < 1e-3


# ---------------------------------------------------------------------------
# 3. 3D Laplacian / RHS
# ---------------------------------------------------------------------------

def test_3d_laplacian_constant_field_is_zero() -> None:
    phi = torch.ones(1, 3, 4, 4, 4) * 0.4
    phi[:, 0] = 0.6

    lap = periodic_laplacian_3d_channels(phi)

    torch.testing.assert_close(lap, torch.zeros_like(lap), atol=1e-6, rtol=0.0)


def test_3d_laplacian_wraps_at_boundaries() -> None:
    field = torch.zeros(1, 2, 4, 4, 4)
    field[:, :, 0, 0, 0] = 1.0

    lap = periodic_laplacian_3d_channels(field)

    assert lap[0, 0, -1, 0, 0] == pytest.approx(1.0)
    assert lap[0, 0, 0, -1, 0] == pytest.approx(1.0)
    assert lap[0, 0, 0, 0, -1] == pytest.approx(1.0)


def test_3d_rhs_shape_matches_input() -> None:
    phi = _simplex_3d(b=2, n=4, d=3, h=5, w=6)

    rhs = explicit_mpf_rhs(phi, eta_px=3.0, mu=1.5e-5, sigma=1.0)

    assert rhs.shape == phi.shape
    assert torch.isfinite(rhs).all()


def test_3d_rhs_phase_sum_near_zero() -> None:
    phi = _simplex_3d(b=1, n=4, d=3, h=5, w=5)

    rhs = explicit_mpf_rhs(phi, eta_px=3.0, mu=1.5e-5, sigma=1.0)

    torch.testing.assert_close(rhs.sum(dim=1), torch.zeros(1, 3, 5, 5), atol=1e-8, rtol=0.0)


def test_3d_laplacian_wrong_rank_raises() -> None:
    with pytest.raises(ValueError, match="5D"):
        periodic_laplacian_3d_channels(torch.ones(1, 4, 6, 6))


def test_2d_laplacian_wrong_rank_raises() -> None:
    with pytest.raises(ValueError, match="4D"):
        periodic_laplacian_2d_channels(torch.ones(1, 4, 4, 4, 4))


# ---------------------------------------------------------------------------
# 4. 3D model build and forward step
# ---------------------------------------------------------------------------

def _build_3d_model(h: int = 8, *, graph_mode: str = "disabled") -> ExplicitMPFHybridRollout:
    return ExplicitMPFHybridRollout(
        num_phases=4,
        model_dt=3333.33,
        eta_px=3.0,
        mu=1.5e-5,
        sigma=1.0,
        spatial_dims=3,
        hidden_channels=h,
        kernel_size=3,
        ann_hidden_features=(8,),
        coordinate_encoding="periodic_3d",
        graph_mode=graph_mode,
        graph_hidden_dim=8,
        projection_mode="soft_threshold_eps1e3",
    )


def test_3d_model_builds_without_error() -> None:
    model = _build_3d_model()
    assert model.spatial_dims == 3
    assert isinstance(model.recurrent_cell, ConvLSTMCell3d)
    import torch.nn as nn
    assert isinstance(model.recurrent_head, nn.Conv3d)


def test_3d_model_initial_state_shapes() -> None:
    model = _build_3d_model()
    phi = _simplex_3d(b=1, n=4, d=3, h=5, w=5)

    state = model.initial_state(phi)

    assert isinstance(state, tuple) and len(state) == 2
    h_state, c_state = state
    assert h_state.shape == (1, 8, 3, 5, 5)
    assert c_state.shape == (1, 8, 3, 5, 5)


def test_3d_model_forward_step_shape_and_simplex() -> None:
    model = _build_3d_model()
    phi = _simplex_3d(b=1, n=4, d=3, h=5, w=5)

    with torch.no_grad():
        next_phi, next_state, rate, diag = model.forward_step(phi)

    assert next_phi.shape == phi.shape
    assert rate.shape == phi.shape
    torch.testing.assert_close(next_phi.sum(dim=1), torch.ones(1, 3, 5, 5), atol=1e-5, rtol=0.0)
    assert torch.isfinite(next_phi).all()
    assert float(next_phi.min()) >= -1e-6
    assert float(next_phi.max()) <= 1.0 + 1e-6


def test_3d_model_rollout_shape() -> None:
    model = _build_3d_model()
    phi = _simplex_3d(b=1, n=4, d=3, h=4, w=4)

    with torch.no_grad():
        states = model.rollout(phi, steps=3)

    assert states.shape == (4, 1, 4, 3, 4, 4)


def test_3d_simple_gnn_model_builds_after_refactor() -> None:
    model = _build_3d_model(h=4, graph_mode="simple_gnn")
    assert model.spatial_dims == 3
    assert model.graph_mode == "simple_gnn"
    assert model.graph_conditioner is not None
    assert isinstance(model.graph_extractor, ExplicitMPFGraphExtractor)


def test_3d_simple_gnn_model_forward_step_no_grad() -> None:
    model = _build_3d_model(h=4, graph_mode="simple_gnn")
    phi = _simplex_3d(b=1, n=4, d=2, h=4, w=5)

    with torch.no_grad():
        next_phi, _, rate, diag = model.forward_step(phi)

    assert next_phi.shape == phi.shape
    assert rate.shape == phi.shape
    assert torch.isfinite(next_phi).all()
    assert torch.isfinite(rate).all()
    assert float(next_phi.min()) >= -1.0e-6
    assert float(next_phi.max()) <= 1.0 + 1.0e-6
    torch.testing.assert_close(next_phi.sum(dim=1), torch.ones(1, 2, 4, 5), atol=1.0e-5, rtol=0.0)
    assert diag["graph_conditioning_stage"] == "pre_bounded_head"


def test_3d_model_wrong_coord_encoding_raises() -> None:
    with pytest.raises(ValueError, match="incompatible"):
        ExplicitMPFHybridRollout(
            num_phases=4,
            model_dt=3333.33,
            eta_px=3.0,
            mu=1.5e-5,
            sigma=1.0,
            spatial_dims=3,
            hidden_channels=4,
            ann_hidden_features=(4,),
            coordinate_encoding="periodic",
        )


# ---------------------------------------------------------------------------
# 5. 2D regression guards
# ---------------------------------------------------------------------------

def test_2d_default_model_still_builds_2d_cells() -> None:
    import torch.nn as nn
    from pinn_phase.models.cells import ConvLSTMCell

    model = ExplicitMPFHybridRollout(
        num_phases=4, model_dt=0.05, eta_px=2.5, hidden_channels=4, ann_hidden_features=(4,),
    )
    assert model.spatial_dims == 2
    assert isinstance(model.recurrent_cell, ConvLSTMCell)
    assert isinstance(model.recurrent_head, nn.Conv2d)


def test_2d_default_model_forward_step_unchanged() -> None:
    torch.manual_seed(7)
    model = ExplicitMPFHybridRollout(
        num_phases=4, model_dt=0.05, eta_px=2.5, hidden_channels=4, ann_hidden_features=(4,),
    )
    phi, _ = project_simplex(torch.rand(1, 4, 8, 8))

    with torch.no_grad():
        next_phi, _, _, diag = model.forward_step(phi)

    assert next_phi.shape == phi.shape
    torch.testing.assert_close(next_phi.sum(dim=1), torch.ones(1, 8, 8), atol=1e-5, rtol=0.0)


def test_2d_projection_per_phase_area_unchanged() -> None:
    phi = torch.rand(2, 3, 6, 8)
    phi = phi / phi.sum(dim=1, keepdim=True)
    phi, diag = project_simplex(phi)
    assert diag["per_phase_area"].shape == (2, 3)
    total = diag["per_phase_area"].sum().item()
    expected = float(2 * 6 * 8)
    assert abs(total - expected) < 1e-2


def test_2d_rhs_phase_sum_still_zero() -> None:
    torch.manual_seed(1)
    phi, _ = project_simplex(torch.rand(2, 4, 6, 7))
    rhs = explicit_mpf_rhs(phi, eta_px=2.5, mu=1.5e-5, sigma=1.0)
    torch.testing.assert_close(rhs.sum(dim=1), torch.zeros(2, 6, 7), atol=1e-10, rtol=0.0)
