"""Physics-only PINN training: no labeled data after t=0.

The model sees only the initial condition phi(t=0).  Every subsequent
time step is predicted auto-regressively.  Training minimises two losses:

  L_pde   = MSE( predicted_derivative ,  Allen-Cahn_RHS(phi) )
  L_energy = MSE( relu( F(phi_{t+1}) - F(phi_t) ) , 0 )
                 (penalises energy increases)

No MSE against labeled reference snapshots is used.  The initial condition
is the only data consumed.

Architecture note:
  The ConvGRU derivative head predicts dphi/dt at each grid point.
  Starting from all-zero weights, the PDE gradient immediately pushes
  the head toward the trusted Allen-Cahn RHS.  The recurrent state
  carries temporal context across the window; the hidden state is
  detached at every TBPTT boundary.

"""

from __future__ import annotations

import time
import math
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Protocol

import psutil
import torch
from torch import Tensor
from torch.optim import Optimizer

from pinn_phase.models import (
    RecurrentState,
    detach_recurrent_state,
    projection_diagnostic_stats,
    project_outward_boundary_rates,
)

from .config import PhysicsConfig
from .losses import (
    algebraic_allen_cahn_rhs,
    ann_algebraic_target_loss,
    mass_rate_loss,
    periodic_laplacian_2d,
    periodic_laplacian_3d,
    projected_scalar_allen_cahn_rhs,
    projected_scalar_allen_cahn_rhs_3d,
    rnn_laplacian_target_loss,
    rnn_laplacian_target_loss_3d,
    scalar_free_energy,
    scalar_free_energy_3d,
)
from .optimizers import GradNormController


def _rss_mib() -> float:
    return psutil.Process().memory_info().rss / 1024.0**2


def _gpu_allocated_mib(device: torch.device) -> float:
    if device.type == "cuda":
        return torch.cuda.memory_allocated(device) / 1024.0**2
    return 0.0


def _gpu_reserved_mib(device: torch.device) -> float:
    if device.type == "cuda":
        return torch.cuda.memory_reserved(device) / 1024.0**2
    return 0.0


def _gpu_peak_allocated_mib(device: torch.device) -> float:
    if device.type == "cuda":
        return torch.cuda.max_memory_allocated(device) / 1024.0**2
    return 0.0


class PhysicsOnlyRollout(Protocol):
    """Minimal model contract required by physics-only training."""

    dt: float
    project_bounds: bool

    def parameters(self, recurse: bool = True): ...

    def train(self, mode: bool = True): ...

    def eval(self): ...

    def initial_state(self, phase: Tensor) -> RecurrentState: ...

    def predict_derivative(
        self,
        phase: Tensor,
        state: RecurrentState,
    ) -> tuple[Tensor, RecurrentState]: ...


@dataclass(frozen=True)
class PINNTrainingConfig:
    """Configuration for physics-only PINN training."""

    epochs: int
    tbptt_window: int
    n_steps: int
    energy_weight: float = 0.05
    learning_rate: float = 0.005
    record_memory: bool = True
    max_grad_norm: float = 1.0
    n_steps_schedule: tuple[tuple[int, int], ...] | None = None
    mass_rate_weight: float = 0.0
    rnn_target_weight: float = 0.0
    ann_target_weight: float = 0.0
    operator_target_warmup_epochs: int = 0
    # Narrow-band PDE residual weighting (unsupervised; uses model's own phi).
    # When > 0, the per-pixel PDE residual is reweighted by
    #   w = 1 + pde_interface_weight * 4*phi*(1-phi)
    # (mean-normalised) so the matrix bulk cannot dominate the residual and
    # reward premature extinction.  0.0 = uniform mean (original behaviour).
    # This is NOT reference-guided: the band weight is built from the model's
    # own phi field, so it preserves the unsupervised claim.
    pde_interface_weight: float = 0.0
    gradnorm_enabled: bool = False
    gradnorm_loss_names: tuple[str, ...] | None = None
    gradnorm_preserve_base_weights: bool = False
    gradnorm_fixed_loss_names: tuple[str, ...] = ()
    gradnorm_min_gradient_norm: float = 0.0
    gradnorm_shared_parameter_scope: str = "trainable_model"
    causal_window_enabled: bool = False
    causal_window_epsilon: float = 5.0
    causal_window_min_weight: float = 0.05
    # Optional anti-collapse penalty on the learned_operator_split scalar blend.
    # DISABLED by default (weight 0.0): we first want to observe whether the
    # learned gamma naturally collapses to one branch or finds a balance.
    # Penalty = weight * (gamma - target_gamma)**2, where gamma is the live
    # learnable sigmoid blend. Reference-free (no PF), uses only the model's own
    # parameter, so it preserves the unsupervised claim when enabled.
    blend_anticollapse_enabled: bool = False
    blend_anticollapse_weight: float = 0.0
    blend_anticollapse_target_gamma: float = 0.5
    # ConvLSTM recurrent hidden-state temporal-smoothness loss (Run 6).
    # L_latent = mean(||h_{t+1} - h_t||^2) / spatial_dim, accumulated per TBPTT step.
    # h_t is the ConvLSTM HIDDEN state (first element of the (hidden, cell) tuple);
    # for ConvGRU h_t is the single hidden tensor.
    # The loss is FIXED at latent_consistency_weight and is NEVER added to the
    # GradNorm adaptive set: it is added directly to total_seg after GradNorm/causal.
    # When weight=0 or enabled=False the contribution is exactly zero.
    # Reference-free: uses only model-internal recurrent states.
    latent_consistency_enabled: bool = False
    latent_consistency_weight: float = 0.0
    # Weak penalty on the raw explicit update before bound projection/clipping.
    # This exposes and discourages rollouts that only look stable because the
    # projection operator repeatedly repairs out-of-bounds updates.
    pre_projection_bounds_weight: float = 0.0

    def __post_init__(self) -> None:
        if self.epochs < 1 or self.tbptt_window < 1 or self.n_steps < 1:
            raise ValueError("epochs, tbptt_window, and n_steps must be positive")
        if self.energy_weight < 0.0:
            raise ValueError("energy_weight must be non-negative")
        if self.pde_interface_weight < 0.0:
            raise ValueError("pde_interface_weight must be non-negative")
        if self.max_grad_norm < 0.0:
            raise ValueError("max_grad_norm must be non-negative (0 disables clipping)")
        if (
            self.mass_rate_weight < 0.0
            or self.rnn_target_weight < 0.0
            or self.ann_target_weight < 0.0
        ):
            raise ValueError("auxiliary loss weights must be non-negative")
        if self.operator_target_warmup_epochs < 0:
            raise ValueError("operator_target_warmup_epochs must be non-negative")
        if self.n_steps_schedule is not None:
            for start, steps in self.n_steps_schedule:
                if start < 1 or steps < 1:
                    raise ValueError(
                        "n_steps_schedule entries must have positive start_epoch and n_steps"
                    )
        if self.gradnorm_enabled:
            if self.causal_window_enabled:
                raise ValueError("GradNorm and causal-window weighting must be separate runs")
            if self.operator_target_warmup_epochs > 0:
                raise ValueError(
                    "gradnorm_enabled requires operator_target_warmup_epochs=0 so all "
                    "balanced losses are active from epoch 1"
                )
            allowed = {
                "pde_residual",
                "energy_monotonicity",
                "rnn_target",
                "ann_target",
                "mass_rate",
            }
            names = self.gradnorm_loss_names or ()
            if not names:
                raise ValueError("gradnorm_loss_names must be provided when GradNorm is enabled")
            if len(set(names)) != len(names):
                raise ValueError("gradnorm_loss_names must be unique")
            unknown = [name for name in names if name not in allowed]
            if unknown:
                raise ValueError(f"unknown GradNorm loss names: {unknown}")
            fixed_unknown = [name for name in self.gradnorm_fixed_loss_names if name not in allowed]
            if fixed_unknown:
                raise ValueError(f"unknown fixed GradNorm loss names: {fixed_unknown}")
            overlap = set(names).intersection(self.gradnorm_fixed_loss_names)
            if overlap:
                raise ValueError(f"GradNorm adaptive and fixed loss names overlap: {sorted(overlap)}")
            if self.gradnorm_min_gradient_norm < 0.0:
                raise ValueError("gradnorm_min_gradient_norm must be non-negative")
            if self.gradnorm_shared_parameter_scope != "trainable_model":
                raise ValueError(
                    "gradnorm_shared_parameter_scope currently supports only 'trainable_model'"
                )
        if self.causal_window_enabled:
            if self.causal_window_epsilon < 0.0:
                raise ValueError("causal_window_epsilon must be non-negative")
            if not 0.0 <= self.causal_window_min_weight <= 1.0:
                raise ValueError("causal_window_min_weight must be in [0, 1]")
        if self.blend_anticollapse_enabled:
            if self.blend_anticollapse_weight < 0.0:
                raise ValueError("blend_anticollapse_weight must be non-negative")
            if not 0.0 < self.blend_anticollapse_target_gamma < 1.0:
                raise ValueError(
                    "blend_anticollapse_target_gamma must be strictly inside (0, 1)"
                )
        if self.latent_consistency_weight < 0.0:
            raise ValueError("latent_consistency_weight must be non-negative")
        if self.pre_projection_bounds_weight < 0.0:
            raise ValueError("pre_projection_bounds_weight must be non-negative")

    def n_steps_at_epoch(self, epoch: int) -> int:
        """Return effective n_steps for a 1-indexed epoch number.

        The schedule is applied in ascending start_epoch order; the last
        entry whose start_epoch <= epoch wins.  Falls back to self.n_steps
        when no schedule is provided or no entry applies.
        """
        if not self.n_steps_schedule:
            return self.n_steps
        current = self.n_steps
        for start_epoch, n in sorted(self.n_steps_schedule):
            if epoch >= start_epoch:
                current = n
        return current


@dataclass
class PINNTrainingHistory:
    """Per-epoch history for physics-only PINN training."""

    pde: list[float] = field(default_factory=list)
    energy: list[float] = field(default_factory=list)
    rnn_target: list[float] = field(default_factory=list)
    ann_target: list[float] = field(default_factory=list)
    mass_rate: list[float] = field(default_factory=list)
    total: list[float] = field(default_factory=list)
    wall_seconds: list[float] = field(default_factory=list)
    rss_mib: list[float] = field(default_factory=list)
    gpu_mib: list[float] = field(default_factory=list)
    gpu_reserved_mib: list[float] = field(default_factory=list)
    gpu_peak_allocated_mib: list[float] = field(default_factory=list)
    blend_gamma: list[float] = field(default_factory=list)
    n_steps_per_epoch: list[int] = field(default_factory=list)
    # Mean signed operator contributions at the interface per epoch.
    ann_interface_rate: list[float] = field(default_factory=list)
    rnn_interface_rate: list[float] = field(default_factory=list)
    algebraic_interface_rate: list[float] = field(default_factory=list)
    laplacian_interface_rate: list[float] = field(default_factory=list)
    rnn_laplacian_relative_error: list[float] = field(default_factory=list)
    rnn_laplacian_norm_ratio: list[float] = field(default_factory=list)
    projected_pixel_fraction: list[float] = field(default_factory=list)
    projection_update_abs_mean: list[float] = field(default_factory=list)
    projection_update_abs_mean_projected: list[float] = field(default_factory=list)
    projection_update_abs_max: list[float] = field(default_factory=list)
    projected_pixel_fraction_bulk_phi_lt_005: list[float] = field(default_factory=list)
    projected_pixel_fraction_interface_phi_005_095: list[float] = field(default_factory=list)
    projected_pixel_fraction_grain_phi_gt_095: list[float] = field(default_factory=list)
    projection_update_abs_mean_bulk_phi_lt_005: list[float] = field(default_factory=list)
    projection_update_abs_mean_interface_phi_005_095: list[float] = field(default_factory=list)
    projection_update_abs_mean_grain_phi_gt_095: list[float] = field(default_factory=list)
    projection_voxel_fraction_bulk_phi_lt_005: list[float] = field(default_factory=list)
    projection_voxel_fraction_interface_phi_005_095: list[float] = field(default_factory=list)
    projection_voxel_fraction_grain_phi_gt_095: list[float] = field(default_factory=list)
    operator_interface_present: list[bool] = field(default_factory=list)
    # learned_operator_split learnable scalar blend diagnostics (per epoch).
    # operator_blend_gamma = sigmoid(raw); raw = logit(gamma). Effective branch
    # magnitudes are the L2 norms of the POST-blend contributions
    # (2*gamma*ann, 2*(1-gamma)*rnn) over interface pixels; ratio = ann/convlstm.
    operator_blend_raw_gamma: list[float] = field(default_factory=list)
    operator_blend_gamma: list[float] = field(default_factory=list)
    ann_effective_magnitude: list[float] = field(default_factory=list)
    convlstm_effective_magnitude: list[float] = field(default_factory=list)
    ann_convlstm_effective_ratio: list[float] = field(default_factory=list)
    gradnorm_objective: list[float] = field(default_factory=list)
    gradnorm_weights: list[dict[str, float]] = field(default_factory=list)
    gradnorm_effective_weights: list[dict[str, float]] = field(default_factory=list)
    gradnorm_gradient_norms: list[dict[str, float]] = field(default_factory=list)
    gradnorm_gradient_norm_ratios: list[dict[str, float]] = field(default_factory=list)
    gradnorm_active_flags: list[dict[str, bool]] = field(default_factory=list)
    gradnorm_raw_losses: list[dict[str, float]] = field(default_factory=list)
    gradnorm_base_coefficients: list[dict[str, float]] = field(default_factory=list)
    gradnorm_base_weighted_losses: list[dict[str, float]] = field(default_factory=list)
    causal_window_mean_weight: list[float] = field(default_factory=list)
    causal_window_final_cumulative_loss: list[float] = field(default_factory=list)
    # Latent-consistency diagnostics (Run 6).
    # latent_consistency_raw_loss: mean ||h_{t+1}-h_t||^2 / spatial_dim per epoch.
    # latent_consistency_weighted_loss: weight * raw_loss per epoch.
    # rnn_hidden_state_norm: mean ||h_t||^2 / spatial_dim (tracks hidden magnitude).
    # rnn_hidden_temporal_variation: same as raw_loss (alias for reporter).
    latent_consistency_raw_loss: list[float] = field(default_factory=list)
    latent_consistency_weighted_loss: list[float] = field(default_factory=list)
    rnn_hidden_state_norm: list[float] = field(default_factory=list)
    rnn_hidden_temporal_variation: list[float] = field(default_factory=list)
    pre_projection_bounds: list[float] = field(default_factory=list)
    pre_projection_min_phi: list[float] = field(default_factory=list)
    pre_projection_max_phi: list[float] = field(default_factory=list)
    pre_projection_violation_mean: list[float] = field(default_factory=list)

    def to_dict(self) -> dict[str, list]:
        return {
            "pde": self.pde,
            "energy": self.energy,
            "rnn_target": self.rnn_target,
            "ann_target": self.ann_target,
            "mass_rate": self.mass_rate,
            "total": self.total,
            "wall_seconds": self.wall_seconds,
            "rss_mib": self.rss_mib,
            "gpu_mib": self.gpu_mib,
            "gpu_reserved_mib": self.gpu_reserved_mib,
            "gpu_peak_allocated_mib": self.gpu_peak_allocated_mib,
            "blend_gamma": self.blend_gamma,
            "n_steps_per_epoch": self.n_steps_per_epoch,
            "ann_interface_rate": self.ann_interface_rate,
            "rnn_interface_rate": self.rnn_interface_rate,
            "algebraic_interface_rate": self.algebraic_interface_rate,
            "laplacian_interface_rate": self.laplacian_interface_rate,
            "rnn_laplacian_relative_error": self.rnn_laplacian_relative_error,
            "rnn_laplacian_norm_ratio": self.rnn_laplacian_norm_ratio,
            "projected_pixel_fraction": self.projected_pixel_fraction,
            "projection_update_abs_mean": self.projection_update_abs_mean,
            "projection_update_abs_mean_projected": self.projection_update_abs_mean_projected,
            "projection_update_abs_max": self.projection_update_abs_max,
            "projected_pixel_fraction_bulk_phi_lt_005": self.projected_pixel_fraction_bulk_phi_lt_005,
            "projected_pixel_fraction_interface_phi_005_095": self.projected_pixel_fraction_interface_phi_005_095,
            "projected_pixel_fraction_grain_phi_gt_095": self.projected_pixel_fraction_grain_phi_gt_095,
            "projection_update_abs_mean_bulk_phi_lt_005": self.projection_update_abs_mean_bulk_phi_lt_005,
            "projection_update_abs_mean_interface_phi_005_095": self.projection_update_abs_mean_interface_phi_005_095,
            "projection_update_abs_mean_grain_phi_gt_095": self.projection_update_abs_mean_grain_phi_gt_095,
            "projection_voxel_fraction_bulk_phi_lt_005": self.projection_voxel_fraction_bulk_phi_lt_005,
            "projection_voxel_fraction_interface_phi_005_095": self.projection_voxel_fraction_interface_phi_005_095,
            "projection_voxel_fraction_grain_phi_gt_095": self.projection_voxel_fraction_grain_phi_gt_095,
            "operator_interface_present": self.operator_interface_present,
            "operator_blend_raw_gamma": self.operator_blend_raw_gamma,
            "operator_blend_gamma": self.operator_blend_gamma,
            "ann_effective_magnitude": self.ann_effective_magnitude,
            "convlstm_effective_magnitude": self.convlstm_effective_magnitude,
            "ann_convlstm_effective_ratio": self.ann_convlstm_effective_ratio,
            "gradnorm_objective": self.gradnorm_objective,
            "gradnorm_weights": self.gradnorm_weights,
            "gradnorm_effective_weights": self.gradnorm_effective_weights,
            "gradnorm_gradient_norms": self.gradnorm_gradient_norms,
            "gradnorm_gradient_norm_ratios": self.gradnorm_gradient_norm_ratios,
            "gradnorm_active_flags": self.gradnorm_active_flags,
            "gradnorm_raw_losses": self.gradnorm_raw_losses,
            "gradnorm_base_coefficients": self.gradnorm_base_coefficients,
            "gradnorm_base_weighted_losses": self.gradnorm_base_weighted_losses,
            "causal_window_mean_weight": self.causal_window_mean_weight,
            "causal_window_final_cumulative_loss": self.causal_window_final_cumulative_loss,
            "latent_consistency_raw_loss": self.latent_consistency_raw_loss,
            "latent_consistency_weighted_loss": self.latent_consistency_weighted_loss,
            "rnn_hidden_state_norm": self.rnn_hidden_state_norm,
            "rnn_hidden_temporal_variation": self.rnn_hidden_temporal_variation,
            "pre_projection_bounds": self.pre_projection_bounds,
            "pre_projection_min_phi": self.pre_projection_min_phi,
            "pre_projection_max_phi": self.pre_projection_max_phi,
            "pre_projection_violation_mean": self.pre_projection_violation_mean,
        }

    @classmethod
    def from_dict(cls, values: dict[str, list]) -> "PINNTrainingHistory":
        """Restore persisted per-epoch metrics for a resumed run."""

        n = len(values["total"])
        return cls(
            pde=list(values.get("pde", [0.0] * n)),
            energy=list(values.get("energy", [0.0] * n)),
            rnn_target=list(values.get("rnn_target", [0.0] * n)),
            ann_target=list(values.get("ann_target", [0.0] * n)),
            mass_rate=list(values.get("mass_rate", [0.0] * n)),
            total=list(values.get("total", [0.0] * n)),
            wall_seconds=list(values.get("wall_seconds", [0.0] * n)),
            rss_mib=list(values.get("rss_mib", [0.0] * n)),
            gpu_mib=list(values.get("gpu_mib", [0.0] * n)),
            gpu_reserved_mib=list(values.get("gpu_reserved_mib", [0.0] * n)),
            gpu_peak_allocated_mib=list(values.get("gpu_peak_allocated_mib", [0.0] * n)),
            blend_gamma=list(values.get("blend_gamma", [0.0] * n)),
            n_steps_per_epoch=[int(v) for v in values.get("n_steps_per_epoch", [0] * n)],
            ann_interface_rate=list(values.get("ann_interface_rate", [0.0] * n)),
            rnn_interface_rate=list(values.get("rnn_interface_rate", [0.0] * n)),
            algebraic_interface_rate=list(values.get("algebraic_interface_rate", [0.0] * n)),
            laplacian_interface_rate=list(values.get("laplacian_interface_rate", [0.0] * n)),
            rnn_laplacian_relative_error=list(
                values.get("rnn_laplacian_relative_error", [0.0] * n)
            ),
            rnn_laplacian_norm_ratio=list(values.get("rnn_laplacian_norm_ratio", [0.0] * n)),
            projected_pixel_fraction=list(values.get("projected_pixel_fraction", [0.0] * n)),
            projection_update_abs_mean=list(
                values.get("projection_update_abs_mean", [0.0] * n)
            ),
            projection_update_abs_mean_projected=list(
                values.get("projection_update_abs_mean_projected", [0.0] * n)
            ),
            projection_update_abs_max=list(values.get("projection_update_abs_max", [0.0] * n)),
            projected_pixel_fraction_bulk_phi_lt_005=list(
                values.get("projected_pixel_fraction_bulk_phi_lt_005", [0.0] * n)
            ),
            projected_pixel_fraction_interface_phi_005_095=list(
                values.get("projected_pixel_fraction_interface_phi_005_095", [0.0] * n)
            ),
            projected_pixel_fraction_grain_phi_gt_095=list(
                values.get("projected_pixel_fraction_grain_phi_gt_095", [0.0] * n)
            ),
            projection_update_abs_mean_bulk_phi_lt_005=list(
                values.get("projection_update_abs_mean_bulk_phi_lt_005", [0.0] * n)
            ),
            projection_update_abs_mean_interface_phi_005_095=list(
                values.get("projection_update_abs_mean_interface_phi_005_095", [0.0] * n)
            ),
            projection_update_abs_mean_grain_phi_gt_095=list(
                values.get("projection_update_abs_mean_grain_phi_gt_095", [0.0] * n)
            ),
            projection_voxel_fraction_bulk_phi_lt_005=list(
                values.get("projection_voxel_fraction_bulk_phi_lt_005", [0.0] * n)
            ),
            projection_voxel_fraction_interface_phi_005_095=list(
                values.get("projection_voxel_fraction_interface_phi_005_095", [0.0] * n)
            ),
            projection_voxel_fraction_grain_phi_gt_095=list(
                values.get("projection_voxel_fraction_grain_phi_gt_095", [0.0] * n)
            ),
            operator_blend_raw_gamma=list(values.get("operator_blend_raw_gamma", [0.0] * n)),
            operator_blend_gamma=list(values.get("operator_blend_gamma", [0.5] * n)),
            ann_effective_magnitude=list(values.get("ann_effective_magnitude", [0.0] * n)),
            convlstm_effective_magnitude=list(
                values.get("convlstm_effective_magnitude", [0.0] * n)
            ),
            ann_convlstm_effective_ratio=list(
                values.get("ann_convlstm_effective_ratio", [0.0] * n)
            ),
            operator_interface_present=list(
                values.get("operator_interface_present", [True] * n)
            ),
            gradnorm_objective=list(values.get("gradnorm_objective", [0.0] * n)),
            gradnorm_weights=list(values.get("gradnorm_weights", [{} for _ in range(n)])),
            gradnorm_effective_weights=list(
                values.get("gradnorm_effective_weights", [{} for _ in range(n)])
            ),
            gradnorm_gradient_norms=list(
                values.get("gradnorm_gradient_norms", [{} for _ in range(n)])
            ),
            gradnorm_gradient_norm_ratios=list(
                values.get("gradnorm_gradient_norm_ratios", [{} for _ in range(n)])
            ),
            gradnorm_active_flags=list(
                values.get("gradnorm_active_flags", [{} for _ in range(n)])
            ),
            gradnorm_raw_losses=list(values.get("gradnorm_raw_losses", [{} for _ in range(n)])),
            gradnorm_base_coefficients=list(
                values.get("gradnorm_base_coefficients", [{} for _ in range(n)])
            ),
            gradnorm_base_weighted_losses=list(
                values.get("gradnorm_base_weighted_losses", [{} for _ in range(n)])
            ),
            causal_window_mean_weight=list(
                values.get("causal_window_mean_weight", [1.0] * n)
            ),
            causal_window_final_cumulative_loss=list(
                values.get("causal_window_final_cumulative_loss", [0.0] * n)
            ),
            latent_consistency_raw_loss=list(
                values.get("latent_consistency_raw_loss", [0.0] * n)
            ),
            latent_consistency_weighted_loss=list(
                values.get("latent_consistency_weighted_loss", [0.0] * n)
            ),
            rnn_hidden_state_norm=list(
                values.get("rnn_hidden_state_norm", [0.0] * n)
            ),
            rnn_hidden_temporal_variation=list(
                values.get("rnn_hidden_temporal_variation", [0.0] * n)
            ),
            pre_projection_bounds=list(values.get("pre_projection_bounds", [0.0] * n)),
            pre_projection_min_phi=list(values.get("pre_projection_min_phi", [0.0] * n)),
            pre_projection_max_phi=list(values.get("pre_projection_max_phi", [1.0] * n)),
            pre_projection_violation_mean=list(
                values.get("pre_projection_violation_mean", [0.0] * n)
            ),
        )


class EarlyStopTriggered(Exception):
    """Raised by an epoch_callback to stop training early.

    The training loop in ``train_pinn_physics_only`` does not catch this; it
    propagates to the caller (``train_pinn_phase.py``), which logs the reason
    and continues to the rollout + metrics phase using the history accumulated
    up to the stopped epoch.

    Attributes
    ----------
    epoch:
        Last completed epoch when the gate triggered.
    reasons:
        Human-readable list of the conditions that satisfied the gate.
    """

    def __init__(self, epoch: int, reasons: list[str]) -> None:
        self.epoch = epoch
        self.reasons = reasons
        super().__init__(f"Early stop at epoch {epoch}: {'; '.join(reasons)}")


EpochCallback = Callable[[int, PINNTrainingHistory], None]


def train_pinn_physics_only(
    model: PhysicsOnlyRollout,
    phi_initial: Tensor,
    *,
    physics: PhysicsConfig,
    config: PINNTrainingConfig,
    optimizer: Optimizer,
    history: PINNTrainingHistory | None = None,
    start_epoch: int = 0,
    epoch_callback: EpochCallback | None = None,
    gradnorm_controller: GradNormController | None = None,
) -> PINNTrainingHistory:
    """Train a ConvGRU surrogate using only physics losses.

    Parameters
    ----------
    model:
        A rollout model that predicts signed derivatives from the current
        phase and recurrent state. Zero-initialised output heads give the
        early PDE gradient an unambiguous direction.
    phi_initial:
        Initial phase field, shape ``(1, 1, H, W)``, transferred to the
        model's device before calling.
    physics:
        Physical parameters.  ``delta_g`` may be zero or nonzero.
    config:
        PINN training hyperparameters.
    optimizer:
        Pre-constructed optimizer over ``model.parameters()``.

    Returns
    -------
    PINNTrainingHistory
        Per-epoch PDE loss, energy loss, total loss, wall time, and memory.
    """

    device = phi_initial.device
    if start_epoch < 0 or start_epoch > config.epochs:
        raise ValueError("start_epoch must be between zero and config.epochs")
    history = PINNTrainingHistory() if history is None else history
    if len(history.total) != start_epoch:
        raise ValueError("history length must equal start_epoch")
    if config.mass_rate_weight > 0.0 and physics.delta_g != 0.0:
        raise ValueError("mass_rate_weight is only valid when delta_g=0")
    branch_mode = getattr(model, "branch_mode", "")
    learned_target_modes = {
        "learned_operator_split",
        "learned_ann_only",
        "learned_convlstm_only",
    }
    rnn_target_modes = {
        "operator_split",
        "learned_operator_split",
        "learned_convlstm_only",
    }
    ann_target_modes = {"learned_operator_split", "learned_ann_only"}
    if config.rnn_target_weight > 0.0 and branch_mode not in rnn_target_modes:
        raise ValueError(
            "rnn_target_weight requires branch_mode='operator_split' or "
            "a learned operator-split mode with an active ConvLSTM branch"
        )
    if config.ann_target_weight > 0.0 and branch_mode not in ann_target_modes:
        raise ValueError(
            "ann_target_weight requires a learned operator-split mode with an active ANN branch"
        )
    if config.operator_target_warmup_epochs > 0:
        if branch_mode not in learned_target_modes:
            raise ValueError(
                "operator_target_warmup_epochs requires a learned operator-split mode"
            )
        if config.ann_target_weight <= 0.0 and config.rnn_target_weight <= 0.0:
            raise ValueError("operator target warmup requires at least one positive target weight")
    if config.gradnorm_enabled:
        if gradnorm_controller is None:
            raise ValueError("gradnorm_enabled requires a GradNormController")
        if tuple(config.gradnorm_loss_names or ()) != gradnorm_controller.loss_names:
            raise ValueError("GradNorm config loss names must match the controller")
    elif gradnorm_controller is not None:
        raise ValueError("GradNormController provided while gradnorm_enabled is false")

    shared_parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]

    def _gradnorm_base_coefficients() -> dict[str, float]:
        return {
            "pde_residual": 1.0,
            "energy_monotonicity": config.energy_weight,
            "rnn_target": config.rnn_target_weight,
            "ann_target": config.ann_target_weight,
            "mass_rate": config.mass_rate_weight,
        }

    def _gradnorm_named_losses(
        *,
        pde: Tensor,
        energy: Tensor,
        rnn: Tensor,
        ann: Tensor,
        mass: Tensor,
    ) -> dict[str, Tensor]:
        return {
            "pde_residual": pde,
            "energy_monotonicity": energy,
            "rnn_target": rnn,
            "ann_target": ann,
            "mass_rate": mass,
        }

    def _gradnorm_adaptive_losses(raw_losses: dict[str, Tensor]) -> dict[str, Tensor]:
        if not config.gradnorm_enabled:
            return {}
        names = config.gradnorm_loss_names or ()
        if config.gradnorm_preserve_base_weights:
            coefficients = _gradnorm_base_coefficients()
            return {name: raw_losses[name] * coefficients[name] for name in names}
        return {name: raw_losses[name] for name in names}

    def _gradnorm_fixed_loss(raw_losses: dict[str, Tensor]) -> Tensor:
        first = next(iter(raw_losses.values()))
        if not config.gradnorm_enabled or not config.gradnorm_fixed_loss_names:
            return first * 0.0
        coefficients = _gradnorm_base_coefficients()
        terms = [
            raw_losses[name] * coefficients[name]
            for name in config.gradnorm_fixed_loss_names
        ]
        return torch.stack(terms).sum() if terms else first * 0.0

    def _gradient_norm_diagnostics(losses: dict[str, Tensor]) -> tuple[dict[str, float], dict[str, float]]:
        norms: dict[str, float] = {}
        for name, loss in losses.items():
            gradients = torch.autograd.grad(
                loss,
                shared_parameters,
                retain_graph=True,
                create_graph=False,
                allow_unused=True,
            )
            pieces = [
                gradient.detach().norm(2)
                for gradient in gradients
                if gradient is not None
            ]
            norm = float(torch.stack(pieces).norm(2)) if pieces else 0.0
            norms[name] = norm
        positive = [value for value in norms.values() if value > 0.0]
        mean_norm = sum(positive) / len(positive) if positive else 0.0
        ratios = {
            name: (value / mean_norm if mean_norm > 0.0 else 0.0)
            for name, value in norms.items()
        }
        return norms, ratios

    for epoch in range(start_epoch, config.epochs):
        effective_n_steps = config.n_steps_at_epoch(epoch + 1)
        epoch_start = time.perf_counter()
        model.train()
        current_phase = phi_initial.clone()
        hidden = model.initial_state(current_phase)
        epoch_pde = 0.0
        epoch_energy = 0.0
        epoch_projected_pixels = 0
        epoch_pixels = 0
        projection_update_abs_sum = 0.0
        projection_update_projected_abs_sum = 0.0
        projection_projected_count = 0.0
        projection_update_abs_max = 0.0
        projection_regions = (
            "bulk_phi_lt_005",
            "interface_phi_005_095",
            "grain_phi_gt_095",
        )
        projection_region_totals = {name: 0.0 for name in projection_regions}
        projection_region_projected = {name: 0.0 for name in projection_regions}
        projection_region_update_abs = {name: 0.0 for name in projection_regions}
        completed = 0

        # 3D dispatch: route physics loss calls to 3D versions when spacings is a 3-tuple.
        _is_3d = len(physics.spacings) == 3
        _projected_rhs_fn = projected_scalar_allen_cahn_rhs_3d if _is_3d else projected_scalar_allen_cahn_rhs
        _free_energy_fn = scalar_free_energy_3d if _is_3d else scalar_free_energy
        _rnn_lap_loss_fn = rnn_laplacian_target_loss_3d if _is_3d else rnn_laplacian_target_loss
        _laplacian_fn = periodic_laplacian_3d if _is_3d else periodic_laplacian_2d

        # Determine once per epoch which auxiliary losses are active.
        _branch_mode = getattr(model, "branch_mode", "")
        _is_op_split = _branch_mode in {
            "operator_split",
            "learned_operator_split",
            "learned_ann_only",
            "learned_convlstm_only",
        }
        _use_rnn_loss = config.rnn_target_weight > 0.0 and _is_op_split
        _use_ann_loss = (
            config.ann_target_weight > 0.0
            and _branch_mode in {"learned_operator_split", "learned_ann_only"}
        )
        _use_mass_loss = config.mass_rate_weight > 0.0
        epoch_rnn = 0.0
        epoch_ann = 0.0
        epoch_mass = 0.0
        epoch_total = 0.0
        epoch_causal_weight_sum = 0.0
        causal_cumulative_loss = torch.zeros((), device=device)
        epoch_gradnorm_objective = 0.0
        epoch_gradnorm_weight_sums: dict[str, float] = {}
        epoch_gradnorm_effective_weight_sums: dict[str, float] = {}
        epoch_gradnorm_norm_sums: dict[str, float] = {}
        epoch_gradnorm_ratio_sums: dict[str, float] = {}
        epoch_gradnorm_active_sums: dict[str, float] = {}
        epoch_gradnorm_raw_loss_sums: dict[str, float] = {}
        epoch_gradnorm_base_coeff_sums: dict[str, float] = {}
        epoch_gradnorm_base_weighted_loss_sums: dict[str, float] = {}
        # Latent-consistency epoch accumulators.
        _use_latent = (
            config.latent_consistency_enabled
            and config.latent_consistency_weight > 0.0
        )
        epoch_latent_raw = 0.0
        epoch_latent_weighted = 0.0
        epoch_hidden_norm = 0.0
        epoch_pre_projection_bounds = 0.0
        epoch_pre_projection_violation = 0.0
        epoch_pre_projection_min = math.inf
        epoch_pre_projection_max = -math.inf

        while completed < effective_n_steps:
            optimizer.zero_grad(set_to_none=True)
            window = min(config.tbptt_window, effective_n_steps - completed)
            segment_pde = torch.zeros((), device=device)
            segment_energy = torch.zeros((), device=device)
            segment_rnn = torch.zeros((), device=device)
            segment_ann = torch.zeros((), device=device)
            segment_mass = torch.zeros((), device=device)
            segment_latent = torch.zeros((), device=device)
            segment_pre_projection_bounds = torch.zeros((), device=device)
            # Track previous hidden state for temporal-variation loss.
            _prev_hidden_tensor: Tensor | None = None

            for _ in range(window):
                phi_before = current_phase  # snapshot for mass-rate loss
                if _use_rnn_loss or _use_ann_loss:
                    raw_derivative, hidden, branches = model.predict_derivative(
                        current_phase, hidden, return_branches=True
                    )
                    rnn_corr = branches["rnn"]
                    ann_local = branches["ann"]
                else:
                    raw_derivative, hidden = model.predict_derivative(current_phase, hidden)
                    rnn_corr = None
                    ann_local = None

                pre_projection_next = current_phase + model.dt * raw_derivative
                pre_projection_violation = torch.relu(-pre_projection_next) + torch.relu(
                    pre_projection_next - 1.0
                )
                pre_projection_bounds = torch.mean(pre_projection_violation**2)
                segment_pre_projection_bounds = (
                    segment_pre_projection_bounds + pre_projection_bounds
                )
                epoch_pre_projection_violation += float(
                    torch.mean(pre_projection_violation).detach()
                )
                epoch_pre_projection_min = min(
                    epoch_pre_projection_min, float(torch.min(pre_projection_next).detach())
                )
                epoch_pre_projection_max = max(
                    epoch_pre_projection_max, float(torch.max(pre_projection_next).detach())
                )

                derivative = raw_derivative
                if model.project_bounds:
                    derivative = project_outward_boundary_rates(current_phase, derivative)
                projection_stats = projection_diagnostic_stats(
                    current_phase,
                    raw_derivative,
                    derivative,
                    dt=float(model.dt),
                )
                step_pixels = derivative.numel()
                step_projected = int(round(projection_stats["projected_pixel_fraction"] * step_pixels))
                epoch_projected_pixels += step_projected
                epoch_pixels += step_pixels
                projection_update_abs_sum += (
                    projection_stats["projection_update_abs_mean"] * step_pixels
                )
                projection_update_projected_abs_sum += (
                    projection_stats["projection_update_abs_mean_projected"]
                    * step_projected
                )
                projection_projected_count += step_projected
                projection_update_abs_max = max(
                    projection_update_abs_max,
                    projection_stats["projection_update_abs_max"],
                )
                for region in projection_regions:
                    region_total = (
                        projection_stats[f"projection_voxel_fraction_{region}"]
                        * step_pixels
                    )
                    projection_region_totals[region] += region_total
                    projection_region_projected[region] += (
                        projection_stats[f"projected_pixel_fraction_{region}"]
                        * region_total
                    )
                    projection_region_update_abs[region] += (
                        projection_stats[f"projection_update_abs_mean_{region}"]
                        * region_total
                    )
                next_phase = current_phase + model.dt * derivative
                if model.project_bounds:
                    next_phase = torch.clamp(next_phase, 0.0, 1.0)

                # L_pde: model derivative must equal the effective Allen-Cahn RHS.
                # The simulator clips phi to [0,1], so the effective dphi/dt at
                # phi=0 with RHS<0 (or phi=1 with RHS>0) is 0, not the raw RHS.
                # The model must learn these clamped bulk dynamics, otherwise the
                # bulk residual dominates and drowns the interface signal.
                effective_rhs = _projected_rhs_fn(
                    current_phase,
                    spacings=physics.spacings,
                    mu=physics.mu,
                    sigma=physics.sigma,
                    eta=physics.eta,
                    delta_g=physics.delta_g,
                )
                pde_sq = (derivative - effective_rhs) ** 2
                if config.pde_interface_weight > 0.0:
                    # Narrow-band reweighting (unsupervised): amplify interface
                    # pixels so the matrix bulk cannot dominate and reward
                    # premature extinction.  Band weight from the model's own
                    # phi: w = 1 + lambda * 4*phi*(1-phi), mean-normalised so the
                    # overall residual scale is preserved.
                    #
                    # The weight is DETACHED: it is a pure spatial mask on the
                    # PDE residual, not a physical term.  Detaching guarantees the
                    # only training signal from L_pde is "make the residual small",
                    # never "move phi so the weight shrinks where the residual is
                    # large".  Detaching does not change the loss VALUE (only the
                    # gradient), so it keeps this an exact spatial-reweighting
                    # ablation of the unchanged scalar-MPF residual.
                    band = 4.0 * current_phase.detach() * (1.0 - current_phase.detach())
                    weights = 1.0 + config.pde_interface_weight * band
                    weights = (weights / weights.mean()).detach()
                    pde = torch.mean(weights * pde_sq)
                else:
                    pde = torch.mean(pde_sq)

                # L_energy: free energy must not increase
                e_cur = _free_energy_fn(
                    current_phase.unsqueeze(0),
                    spacings=physics.spacings,
                    sigma=physics.sigma,
                    eta=physics.eta,
                    delta_g=physics.delta_g,
                )
                e_next = _free_energy_fn(
                    next_phase.unsqueeze(0),
                    spacings=physics.spacings,
                    sigma=physics.sigma,
                    eta=physics.eta,
                    delta_g=physics.delta_g,
                )
                energy = torch.mean(torch.relu(e_next - e_cur) ** 2)

                # L_rnn: push ConvLSTM correction toward mu*sigma*Laplacian(phi)
                if _use_rnn_loss:
                    segment_rnn = segment_rnn + _rnn_lap_loss_fn(
                        rnn_corr,
                        current_phase,
                        mu=physics.mu,
                        sigma=physics.sigma,
                        spacings=physics.spacings,
                        bulk_weight=0.1 if _branch_mode == "learned_operator_split" else 0.0,
                    )

                # L_ann: train the pointwise ANN toward local bulk + driving physics.
                if _use_ann_loss:
                    segment_ann = segment_ann + ann_algebraic_target_loss(
                        ann_local,
                        current_phase,
                        mu=physics.mu,
                        sigma=physics.sigma,
                        eta=physics.eta,
                        delta_g=physics.delta_g,
                        bulk_weight=0.1,
                    )

                # L_mass: enforce d(sum(phi))/dt = -2*pi*mu*sigma/(dx*dy)
                # Only correct for delta_g=0; caller must set mass_rate_weight=0 for delta_g!=0.
                if _use_mass_loss:
                    segment_mass = segment_mass + mass_rate_loss(
                        phi_before,
                        next_phase,
                        dt=model.dt,
                        mu=physics.mu,
                        sigma=physics.sigma,
                        spacings=physics.spacings,
                    )

                segment_pde = segment_pde + pde
                segment_energy = segment_energy + energy

                # L_latent: ConvLSTM hidden-state temporal-smoothness loss.
                # Accumulates ||h_{t+1} - h_t||^2 / spatial_dim.
                # Reference-free: uses only the model's own recurrent state.
                # Fixed weight, never added to GradNorm adaptive set.
                if _use_latent:
                    # Extract the hidden tensor from the current recurrent state.
                    # ConvLSTM: state = (hidden, cell); ConvGRU: state = hidden tensor.
                    if isinstance(hidden, tuple):
                        _cur_hidden_tensor = hidden[0]
                    else:
                        _cur_hidden_tensor = hidden
                    spatial_dim = float(math.prod(_cur_hidden_tensor.shape[2:]))
                    if _prev_hidden_tensor is not None:
                        diff = _cur_hidden_tensor - _prev_hidden_tensor
                        segment_latent = segment_latent + torch.mean(diff ** 2) / spatial_dim
                    _prev_hidden_tensor = _cur_hidden_tensor.detach()

            current_phase = next_phase

            if config.gradnorm_enabled:
                raw_losses = _gradnorm_named_losses(
                    pde=segment_pde / window,
                    energy=segment_energy / window,
                    rnn=segment_rnn / window,
                    ann=segment_ann / window,
                    mass=segment_mass / window,
                )
                coefficients = _gradnorm_base_coefficients()
                losses_for_gradnorm = _gradnorm_adaptive_losses(raw_losses)
                fixed_loss = _gradnorm_fixed_loss(raw_losses)
                assert gradnorm_controller is not None
                norms, ratios = _gradient_norm_diagnostics(losses_for_gradnorm)
                active_loss_names = tuple(
                    name
                    for name in gradnorm_controller.loss_names
                    if norms.get(name, 0.0) > config.gradnorm_min_gradient_norm
                )
                if active_loss_names:
                    gradnorm_objective = gradnorm_controller.step(
                        losses_for_gradnorm,
                        shared_parameters,
                        active_loss_names=active_loss_names,
                    )
                else:
                    gradnorm_objective = torch.zeros((), device=device)
                weights = gradnorm_controller.named_weights()
                effective_weights = gradnorm_controller.named_effective_weights(
                    active_loss_names
                )
                total_seg = (
                    gradnorm_controller.weighted_model_loss(
                        losses_for_gradnorm,
                        active_loss_names=active_loss_names,
                    )
                    + fixed_loss
                )
                epoch_gradnorm_objective += float(gradnorm_objective)
                for name, value in weights.items():
                    epoch_gradnorm_weight_sums[name] = (
                        epoch_gradnorm_weight_sums.get(name, 0.0) + float(value)
                    )
                for name, value in effective_weights.items():
                    epoch_gradnorm_effective_weight_sums[name] = (
                        epoch_gradnorm_effective_weight_sums.get(name, 0.0)
                        + float(value)
                    )
                for name, value in norms.items():
                    epoch_gradnorm_norm_sums[name] = (
                        epoch_gradnorm_norm_sums.get(name, 0.0) + float(value)
                    )
                for name, value in ratios.items():
                    epoch_gradnorm_ratio_sums[name] = (
                        epoch_gradnorm_ratio_sums.get(name, 0.0) + float(value)
                    )
                for name in gradnorm_controller.loss_names:
                    epoch_gradnorm_active_sums[name] = (
                        epoch_gradnorm_active_sums.get(name, 0.0)
                        + (1.0 if name in active_loss_names else 0.0)
                    )
                for name, value in raw_losses.items():
                    epoch_gradnorm_raw_loss_sums[name] = (
                        epoch_gradnorm_raw_loss_sums.get(name, 0.0)
                        + float(value.detach())
                    )
                    epoch_gradnorm_base_coeff_sums[name] = (
                        epoch_gradnorm_base_coeff_sums.get(name, 0.0)
                        + float(coefficients[name])
                    )
                    epoch_gradnorm_base_weighted_loss_sums[name] = (
                        epoch_gradnorm_base_weighted_loss_sums.get(name, 0.0)
                        + float((value * coefficients[name]).detach())
                    )
            elif epoch < config.operator_target_warmup_epochs:
                total_seg = (
                    config.rnn_target_weight * segment_rnn
                    + config.ann_target_weight * segment_ann
                ) / window
            else:
                total_seg = (
                    segment_pde
                    + config.energy_weight * segment_energy
                    + config.rnn_target_weight * segment_rnn
                    + config.ann_target_weight * segment_ann
                    + config.mass_rate_weight * segment_mass
                ) / window
            if config.causal_window_enabled:
                causal_weight = torch.exp(
                    -config.causal_window_epsilon * causal_cumulative_loss
                ).clamp_min(config.causal_window_min_weight)
                causal_cumulative_loss = causal_cumulative_loss + total_seg.detach()
                epoch_causal_weight_sum += float(causal_weight.detach())
                total_seg = causal_weight * total_seg
            else:
                epoch_causal_weight_sum += 1.0
            # Optional anti-collapse penalty on the learnable blend gamma. OFF by
            # default. Applied as a parameter regularizer outside the rollout /
            # causal weighting. Reference-free: uses only the model's own gamma.
            if config.blend_anticollapse_enabled and config.blend_anticollapse_weight > 0.0:
                _ob_gamma = getattr(model, "operator_blend_gamma", None)
                if _ob_gamma is not None and getattr(
                    model, "operator_blend_learnable", False
                ):
                    total_seg = total_seg + config.blend_anticollapse_weight * (
                        _ob_gamma - config.blend_anticollapse_target_gamma
                    ) ** 2
            bounds_raw_seg = segment_pre_projection_bounds / window
            epoch_pre_projection_bounds += float(bounds_raw_seg.detach())
            if config.pre_projection_bounds_weight > 0.0:
                total_seg = total_seg + config.pre_projection_bounds_weight * bounds_raw_seg
            # Fixed latent-consistency contribution (outside GradNorm / causal).
            # Added last so it does not interact with GradNorm weight updates.
            if _use_latent and window > 1:
                latent_raw_seg = segment_latent / (window - 1)
                latent_weighted_seg = config.latent_consistency_weight * latent_raw_seg
                total_seg = total_seg + latent_weighted_seg
                epoch_latent_raw += float(latent_raw_seg.detach())
                epoch_latent_weighted += float(latent_weighted_seg.detach())
                # Track mean hidden-state L2 norm for the last step in this segment.
                if _prev_hidden_tensor is not None:
                    epoch_hidden_norm += float(
                        torch.mean(_prev_hidden_tensor ** 2).detach()
                    )
            total_seg.backward()
            if config.max_grad_norm > 0.0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
            optimizer.step()
            current_phase = current_phase.detach()
            hidden = detach_recurrent_state(hidden)
            epoch_pde += float(segment_pde.detach()) / window
            epoch_energy += float(segment_energy.detach()) / window
            epoch_rnn += float(segment_rnn.detach()) / window
            epoch_ann += float(segment_ann.detach()) / window
            epoch_mass += float(segment_mass.detach()) / window
            epoch_total += float(total_seg.detach())
            completed += window

        n_windows = -(-effective_n_steps // config.tbptt_window)
        mean_pde = epoch_pde / n_windows
        mean_energy = epoch_energy / n_windows
        if config.gradnorm_enabled:
            mean_total = epoch_total / n_windows
        elif epoch < config.operator_target_warmup_epochs:
            mean_total = (
                config.rnn_target_weight * epoch_rnn / n_windows
                + config.ann_target_weight * epoch_ann / n_windows
            )
        else:
            mean_total = (
                mean_pde
                + config.energy_weight * mean_energy
                + config.rnn_target_weight * epoch_rnn / n_windows
                + config.ann_target_weight * epoch_ann / n_windows
                + config.mass_rate_weight * epoch_mass / n_windows
            )
        if (
            config.causal_window_enabled
            or config.pre_projection_bounds_weight > 0.0
            or (
                config.latent_consistency_enabled
                and config.latent_consistency_weight > 0.0
            )
            or (
                config.blend_anticollapse_enabled
                and config.blend_anticollapse_weight > 0.0
            )
        ):
            mean_total = epoch_total / n_windows

        history.pde.append(mean_pde)
        history.energy.append(mean_energy)
        history.rnn_target.append(epoch_rnn / n_windows)
        history.ann_target.append(epoch_ann / n_windows)
        history.mass_rate.append(epoch_mass / n_windows)
        history.total.append(mean_total)
        history.wall_seconds.append(time.perf_counter() - epoch_start)
        history.n_steps_per_epoch.append(effective_n_steps)
        if config.record_memory:
            history.rss_mib.append(_rss_mib())
            history.gpu_mib.append(_gpu_allocated_mib(device))
            history.gpu_reserved_mib.append(_gpu_reserved_mib(device))
            history.gpu_peak_allocated_mib.append(_gpu_peak_allocated_mib(device))
        else:
            history.rss_mib.append(0.0)
            history.gpu_mib.append(0.0)
            history.gpu_reserved_mib.append(0.0)
            history.gpu_peak_allocated_mib.append(0.0)
        gamma = getattr(model, "blend_gamma", None)
        history.blend_gamma.append(float(gamma.detach()) if gamma is not None else 0.0)

        history.projected_pixel_fraction.append(
            epoch_projected_pixels / epoch_pixels if epoch_pixels else 0.0
        )
        history.projection_update_abs_mean.append(
            projection_update_abs_sum / epoch_pixels if epoch_pixels else 0.0
        )
        history.projection_update_abs_mean_projected.append(
            projection_update_projected_abs_sum / projection_projected_count
            if projection_projected_count
            else 0.0
        )
        history.projection_update_abs_max.append(projection_update_abs_max)
        history.projected_pixel_fraction_bulk_phi_lt_005.append(
            projection_region_projected["bulk_phi_lt_005"]
            / projection_region_totals["bulk_phi_lt_005"]
            if projection_region_totals["bulk_phi_lt_005"]
            else 0.0
        )
        history.projected_pixel_fraction_interface_phi_005_095.append(
            projection_region_projected["interface_phi_005_095"]
            / projection_region_totals["interface_phi_005_095"]
            if projection_region_totals["interface_phi_005_095"]
            else 0.0
        )
        history.projected_pixel_fraction_grain_phi_gt_095.append(
            projection_region_projected["grain_phi_gt_095"]
            / projection_region_totals["grain_phi_gt_095"]
            if projection_region_totals["grain_phi_gt_095"]
            else 0.0
        )
        history.projection_update_abs_mean_bulk_phi_lt_005.append(
            projection_region_update_abs["bulk_phi_lt_005"]
            / projection_region_totals["bulk_phi_lt_005"]
            if projection_region_totals["bulk_phi_lt_005"]
            else 0.0
        )
        history.projection_update_abs_mean_interface_phi_005_095.append(
            projection_region_update_abs["interface_phi_005_095"]
            / projection_region_totals["interface_phi_005_095"]
            if projection_region_totals["interface_phi_005_095"]
            else 0.0
        )
        history.projection_update_abs_mean_grain_phi_gt_095.append(
            projection_region_update_abs["grain_phi_gt_095"]
            / projection_region_totals["grain_phi_gt_095"]
            if projection_region_totals["grain_phi_gt_095"]
            else 0.0
        )
        history.projection_voxel_fraction_bulk_phi_lt_005.append(
            projection_region_totals["bulk_phi_lt_005"] / epoch_pixels
            if epoch_pixels
            else 0.0
        )
        history.projection_voxel_fraction_interface_phi_005_095.append(
            projection_region_totals["interface_phi_005_095"] / epoch_pixels
            if epoch_pixels
            else 0.0
        )
        history.projection_voxel_fraction_grain_phi_gt_095.append(
            projection_region_totals["grain_phi_gt_095"] / epoch_pixels
            if epoch_pixels
            else 0.0
        )

        # Record signed operator rates at the interface (phi in [0.2, 0.8]).
        # Uses no_grad and the last current_phase from this epoch.
        branch_fn = getattr(model, "branch_derivatives", None)
        _diagnostic_branch_mode = getattr(model, "branch_mode", "")
        if branch_fn is not None or _diagnostic_branch_mode in {
            "operator_split",
            "exact_laplacian_residual",
            "learned_operator_split",
            "learned_ann_only",
            "learned_convlstm_only",
        }:
            with torch.no_grad():
                model.eval()
                probe_phase = current_phase.detach()
                probe_state = model.initial_state(probe_phase)
                interface_mask = (probe_phase > 0.2) & (probe_phase < 0.8)
                if _diagnostic_branch_mode in {
                    "operator_split",
                    "exact_laplacian_residual",
                }:
                    _, _, branches = model.predict_derivative(
                        probe_phase, probe_state, return_branches=True
                    )
                    algebraic = branches["algebraic"]
                    recurrent = branches["rnn"]
                    ann = model.lambda_correction * branches["ann"]
                    laplacian = branches.get(
                        "laplacian",
                        physics.mu
                        * physics.sigma
                        * _laplacian_fn(probe_phase, physics.spacings),
                    )
                elif _diagnostic_branch_mode in {
                    "learned_operator_split",
                    "learned_ann_only",
                    "learned_convlstm_only",
                }:
                    _, _, branches = model.predict_derivative(
                        probe_phase, probe_state, return_branches=True
                    )
                    ann = branches["ann"]
                    recurrent = branches["rnn"]
                    algebraic = algebraic_allen_cahn_rhs(
                        probe_phase,
                        mu=physics.mu,
                        sigma=physics.sigma,
                        eta=physics.eta,
                        delta_g=physics.delta_g,
                    )
                    laplacian = (
                        physics.mu
                        * physics.sigma
                        * _laplacian_fn(probe_phase, physics.spacings)
                    )
                else:
                    ann_raw, recurrent_raw, _ = branch_fn(probe_phase, probe_state)
                    ann_weight, recurrent_weight = model.branch_weights
                    ann = ann_weight * ann_raw
                    recurrent = recurrent_weight * recurrent_raw
                    algebraic = algebraic_allen_cahn_rhs(
                        probe_phase,
                        mu=physics.mu,
                        sigma=physics.sigma,
                        eta=physics.eta,
                        delta_g=physics.delta_g,
                    )
                    laplacian = (
                        physics.mu
                        * physics.sigma
                        * _laplacian_fn(probe_phase, physics.spacings)
                    )
                interface_present = bool(interface_mask.any())
                if interface_present:
                    ann_value = float(ann[interface_mask].mean())
                    recurrent_value = float(recurrent[interface_mask].mean())
                    algebraic_value = float(algebraic[interface_mask].mean())
                    laplacian_value = float(laplacian[interface_mask].mean())
                    recurrent_view = recurrent[interface_mask]
                    laplacian_view = laplacian[interface_mask]
                else:
                    ann_value = float(ann.mean())
                    recurrent_value = float(recurrent.mean())
                    algebraic_value = float(algebraic.mean())
                    laplacian_value = float(laplacian.mean())
                    recurrent_view = recurrent.flatten()
                    laplacian_view = laplacian.flatten()
                target_norm = torch.linalg.vector_norm(laplacian_view)
                recurrent_norm = torch.linalg.vector_norm(recurrent_view)
                error_norm = torch.linalg.vector_norm(recurrent_view - laplacian_view)
                denom = max(float(target_norm), 1.0e-12)
                history.ann_interface_rate.append(ann_value)
                history.rnn_interface_rate.append(recurrent_value)
                history.algebraic_interface_rate.append(algebraic_value)
                history.laplacian_interface_rate.append(laplacian_value)
                history.rnn_laplacian_relative_error.append(
                    float(error_norm) / denom if interface_present else 0.0
                )
                history.rnn_laplacian_norm_ratio.append(
                    float(recurrent_norm) / denom if interface_present else 0.0
                )
                history.operator_interface_present.append(interface_present)

                # learned_operator_split learnable-blend diagnostics. Effective
                # magnitudes use the POST-blend contributions so we can detect a
                # branch being driven toward zero by the learned gamma. The
                # 'effective' keys exist only for learned_operator_split.
                ob_gamma = getattr(model, "operator_blend_gamma", None)
                if ob_gamma is not None and "ann_effective" in branches:
                    g_val = float(ob_gamma.detach())
                    raw_g = float(
                        torch.logit(
                            torch.clamp(ob_gamma.detach(), 1.0e-6, 1.0 - 1.0e-6)
                        )
                    )
                    ann_eff = branches["ann_effective"]
                    rnn_eff = branches["rnn_effective"]
                    ann_eff_view = ann_eff[interface_mask] if interface_present else ann_eff
                    rnn_eff_view = rnn_eff[interface_mask] if interface_present else rnn_eff
                    ann_eff_mag = float(torch.linalg.vector_norm(ann_eff_view))
                    rnn_eff_mag = float(torch.linalg.vector_norm(rnn_eff_view))
                    history.operator_blend_raw_gamma.append(raw_g)
                    history.operator_blend_gamma.append(g_val)
                    history.ann_effective_magnitude.append(ann_eff_mag)
                    history.convlstm_effective_magnitude.append(rnn_eff_mag)
                    history.ann_convlstm_effective_ratio.append(
                        ann_eff_mag / max(rnn_eff_mag, 1.0e-12)
                    )
                else:
                    history.operator_blend_raw_gamma.append(0.0)
                    history.operator_blend_gamma.append(0.5)
                    history.ann_effective_magnitude.append(0.0)
                    history.convlstm_effective_magnitude.append(0.0)
                    history.ann_convlstm_effective_ratio.append(0.0)
                model.train()
        else:
            history.ann_interface_rate.append(0.0)
            history.rnn_interface_rate.append(0.0)
            history.algebraic_interface_rate.append(0.0)
            history.laplacian_interface_rate.append(0.0)
            history.rnn_laplacian_relative_error.append(0.0)
            history.rnn_laplacian_norm_ratio.append(0.0)
            history.operator_interface_present.append(False)
            history.operator_blend_raw_gamma.append(0.0)
            history.operator_blend_gamma.append(0.5)
            history.ann_effective_magnitude.append(0.0)
            history.convlstm_effective_magnitude.append(0.0)
            history.ann_convlstm_effective_ratio.append(0.0)

        if config.gradnorm_enabled:
            history.gradnorm_objective.append(epoch_gradnorm_objective / n_windows)
            history.gradnorm_weights.append(
                {
                    name: value / n_windows
                    for name, value in epoch_gradnorm_weight_sums.items()
                }
            )
            history.gradnorm_effective_weights.append(
                {
                    name: value / n_windows
                    for name, value in epoch_gradnorm_effective_weight_sums.items()
                }
            )
            history.gradnorm_gradient_norms.append(
                {
                    name: value / n_windows
                    for name, value in epoch_gradnorm_norm_sums.items()
                }
            )
            history.gradnorm_gradient_norm_ratios.append(
                {
                    name: value / n_windows
                    for name, value in epoch_gradnorm_ratio_sums.items()
                }
            )
            history.gradnorm_active_flags.append(
                {
                    name: value > 0.5 * n_windows
                    for name, value in epoch_gradnorm_active_sums.items()
                }
            )
            history.gradnorm_raw_losses.append(
                {
                    name: value / n_windows
                    for name, value in epoch_gradnorm_raw_loss_sums.items()
                }
            )
            history.gradnorm_base_coefficients.append(
                {
                    name: value / n_windows
                    for name, value in epoch_gradnorm_base_coeff_sums.items()
                }
            )
            history.gradnorm_base_weighted_losses.append(
                {
                    name: value / n_windows
                    for name, value in epoch_gradnorm_base_weighted_loss_sums.items()
                }
            )
        else:
            history.gradnorm_objective.append(0.0)
            history.gradnorm_weights.append({})
            history.gradnorm_effective_weights.append({})
            history.gradnorm_gradient_norms.append({})
            history.gradnorm_gradient_norm_ratios.append({})
            history.gradnorm_active_flags.append({})
            history.gradnorm_raw_losses.append({})
            history.gradnorm_base_coefficients.append({})
            history.gradnorm_base_weighted_losses.append({})
        history.causal_window_mean_weight.append(epoch_causal_weight_sum / n_windows)
        history.causal_window_final_cumulative_loss.append(float(causal_cumulative_loss.detach()))
        # Latent-consistency diagnostics per epoch.
        history.latent_consistency_raw_loss.append(epoch_latent_raw / n_windows)
        history.latent_consistency_weighted_loss.append(epoch_latent_weighted / n_windows)
        history.rnn_hidden_state_norm.append(epoch_hidden_norm / n_windows)
        history.rnn_hidden_temporal_variation.append(epoch_latent_raw / n_windows)
        history.pre_projection_bounds.append(epoch_pre_projection_bounds / n_windows)
        history.pre_projection_min_phi.append(
            epoch_pre_projection_min if math.isfinite(epoch_pre_projection_min) else 0.0
        )
        history.pre_projection_max_phi.append(
            epoch_pre_projection_max if math.isfinite(epoch_pre_projection_max) else 1.0
        )
        history.pre_projection_violation_mean.append(
            epoch_pre_projection_violation / max(effective_n_steps, 1)
        )

        if epoch_callback is not None:
            epoch_callback(epoch + 1, history)

    return history


@torch.no_grad()
def rollout_pinn(
    model: PhysicsOnlyRollout,
    phi_initial: Tensor,
    *,
    steps: int,
) -> Tensor:
    """Auto-regressively roll out the trained PINN from phi_initial.

    Uses only the initial condition — no labels.  Returns a tensor of
    shape ``(steps+1, 1, 1, H, W)`` for consistency with training tensors.
    """

    model.eval()
    current_phase = phi_initial.clone()
    hidden = model.initial_state(current_phase)
    states = [current_phase]
    for _ in range(steps):
        derivative, hidden = model.predict_derivative(current_phase, hidden)
        if model.project_bounds:
            derivative = project_outward_boundary_rates(current_phase, derivative)
        next_phase = current_phase + model.dt * derivative
        if model.project_bounds:
            next_phase = torch.clamp(next_phase, 0.0, 1.0)
        states.append(next_phase)
        current_phase = next_phase
    return torch.stack(states, dim=0)


@torch.no_grad()
def rollout_pinn_radii(
    model: PhysicsOnlyRollout,
    phi_initial: Tensor,
    *,
    steps: int,
    spacings: tuple[float, ...],
) -> list[float]:
    """Roll out equivalent radii without retaining every phase field.

    Long live-monitor previews only need ``R(t)``. Keeping full states on the
    accelerator for those previews wastes memory and can make a diagnostic plot
    interrupt otherwise healthy training.
    """

    if steps < 1:
        raise ValueError("steps must be positive")
    spacing_values = tuple(float(value) for value in spacings)
    if len(spacing_values) not in (2, 3):
        raise ValueError("spacings must describe a 2D or 3D scalar field")
    cell_measure = float(math.prod(spacing_values))

    def equivalent_radius(phase: Tensor) -> float:
        mass = max(float(phase.sum()), 0.0) * cell_measure
        if len(spacing_values) == 3:
            return (3.0 * mass / (4.0 * math.pi)) ** (1.0 / 3.0)
        return math.sqrt(mass / math.pi)

    model.eval()
    current_phase = phi_initial.clone()
    hidden = model.initial_state(current_phase)
    radii = [equivalent_radius(current_phase)]
    for _ in range(steps):
        derivative, hidden = model.predict_derivative(current_phase, hidden)
        if model.project_bounds:
            derivative = project_outward_boundary_rates(current_phase, derivative)
        current_phase = current_phase + model.dt * derivative
        if model.project_bounds:
            current_phase = torch.clamp(current_phase, 0.0, 1.0)
        radii.append(equivalent_radius(current_phase))
    return radii
