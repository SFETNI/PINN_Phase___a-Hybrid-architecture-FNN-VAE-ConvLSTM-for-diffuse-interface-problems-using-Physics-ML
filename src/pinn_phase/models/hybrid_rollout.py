"""Minimal monitored PINN-Phase hybrid rollout.

The first article-facing candidate combines:

- a pointwise ANN/FNN branch for signed spatial increments;
- a true ConvLSTM branch for signed recurrent increments;
- a bounded residual-blend coefficient;
- one explicit phase-preserving update.

The reduced local ConvGRU rollout remains available as a control experiment.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
import math
from typing import TypeAlias

import torch
import torch.nn as nn
from torch import Tensor

from .cells import ConvGRUCell, ConvLSTMCell, ConvGRUCell3d, ConvLSTMCell3d
from .projection import (
    DEFAULT_BOUND_ENFORCEMENT_MODE,
    DEFAULT_BOUNDARY_EPS,
    PROJECT_AND_CLAMP,
    apply_bound_enforcement,
    project_outward_boundary_rates,
)
from .residual_blend import ResidualBlend

# Lazy import to avoid module-level circular dependency risk.
# pinn_phase.training.losses does not import from pinn_phase.models.
def _algebraic_rhs_fn(phi, *, mu, sigma, eta, delta_g):
    from pinn_phase.training.losses import algebraic_allen_cahn_rhs  # noqa: PLC0415
    return algebraic_allen_cahn_rhs(phi, mu=mu, sigma=sigma, eta=eta, delta_g=delta_g)


def _laplacian_rhs_fn(phi, *, mu, sigma, spacings):
    from pinn_phase.training.losses import periodic_laplacian_2d  # noqa: PLC0415
    return mu * sigma * periodic_laplacian_2d(phi, spacings)


def _laplacian_rhs_fn_3d(phi, *, mu, sigma, spacings):
    from pinn_phase.training.losses import periodic_laplacian_3d  # noqa: PLC0415
    return mu * sigma * periodic_laplacian_3d(phi, spacings)


RecurrentState: TypeAlias = Tensor | tuple[Tensor, Tensor]


def detach_recurrent_state(state: RecurrentState) -> RecurrentState:
    """Detach a GRU tensor or ConvLSTM ``(hidden, cell)`` state."""

    if isinstance(state, tuple):
        return tuple(value.detach() for value in state)
    return state.detach()


def _step_with_bound_enforcement(model: nn.Module, phase: Tensor, derivative: Tensor) -> Tensor:
    """Apply the explicit update under the model's bound-enforcement setting.

    Production default (``project_bounds=True`` and no diagnostic override) is
    byte-for-byte equivalent to the previous inline ``project + clamp`` logic.

    A diagnostic override may be installed via
    :meth:`set_bound_enforcement_mode`; when present it takes precedence and
    selects ``project_and_clamp`` / ``clamp_only`` / ``unbounded`` and an
    optional ``boundary_eps``.  The override is a plain runtime attribute, so it
    never enters ``state_dict`` and never changes checkpoint loading.
    """

    if not getattr(model, "project_bounds", True):
        # Legacy ``project_bounds=False`` path: raw Euler step, no enforcement.
        return phase + model.dt * derivative

    mode = getattr(model, "_bound_enforcement_mode", DEFAULT_BOUND_ENFORCEMENT_MODE)
    boundary_eps = getattr(model, "_bound_enforcement_boundary_eps", DEFAULT_BOUNDARY_EPS)
    return apply_bound_enforcement(
        phase, derivative, dt=model.dt, mode=mode, boundary_eps=boundary_eps
    )


def _set_bound_enforcement_mode(
    model: nn.Module,
    mode: str,
    *,
    boundary_eps: float,
) -> None:
    """Validate and install the diagnostic bound-enforcement override."""

    from .projection import BOUND_ENFORCEMENT_MODES  # noqa: PLC0415

    if mode not in BOUND_ENFORCEMENT_MODES:
        raise ValueError(
            f"mode must be one of {BOUND_ENFORCEMENT_MODES}; got {mode!r}"
        )
    if boundary_eps <= 0.0:
        raise ValueError("boundary_eps must be positive")
    if mode != PROJECT_AND_CLAMP and boundary_eps != DEFAULT_BOUNDARY_EPS:
        raise ValueError(
            "boundary_eps is only meaningful for mode='project_and_clamp'"
        )
    model._bound_enforcement_mode = mode
    model._bound_enforcement_boundary_eps = float(boundary_eps)


class PointwiseANNIncrement(nn.Module):
    """Predict a signed local increment rate from ``(phi, coord_features)`` per pixel.

    Parameters
    ----------
    hidden_features:
        Widths of hidden layers.
    zero_initialize_head:
        If True, zero-initialize the output head (safe warm-start).
    coordinate_encoding:
        How to encode spatial coordinates for ANN inputs.

        ``"raw"`` (default, backward-compatible):
            Uses raw normalized coordinates ``x, y ∈ [−1, 1]``.  These are
            discontinuous across the periodic wrap and let the ANN memorize the
            specific grain layout.

        ``"periodic"``:
            Replaces raw coordinates with periodic encodings
            ``[sin(π·x), cos(π·x), sin(π·y), cos(π·y)]``.  Continuous across
            the periodic boundary; 4 features instead of 2 so the ANN input
            width is 5 (phi + 4 coord features).

        ``"none"``:
            Drops coordinate inputs entirely; ANN receives only ``phi``.
            Input width is 1.  This eliminates any layout-memorization risk.
    """

    def __init__(
        self,
        hidden_features: Sequence[int] = (32, 32),
        *,
        zero_initialize_head: bool = True,
        coordinate_encoding: str = "raw",
        cache_coordinate_features: bool = False,
    ) -> None:
        super().__init__()
        if coordinate_encoding not in {"raw", "periodic", "none"}:
            raise ValueError(
                f"coordinate_encoding must be 'raw', 'periodic', or 'none'; "
                f"got {coordinate_encoding!r}"
            )
        widths = tuple(int(width) for width in hidden_features)
        if not widths or any(width < 1 for width in widths):
            raise ValueError("hidden_features must contain positive widths")

        self.coordinate_encoding = coordinate_encoding
        self.cache_coordinate_features = bool(cache_coordinate_features)
        self._coordinate_feature_cache: dict[tuple[object, ...], Tensor] = {}
        if coordinate_encoding == "none":
            in_features = 1  # phi only
        elif coordinate_encoding == "periodic":
            in_features = 5  # phi + sin(pi*x) + cos(pi*x) + sin(pi*y) + cos(pi*y)
        else:  # "raw"
            in_features = 3  # phi + x + y

        layers: list[nn.Module] = []
        cur_features = in_features
        for width in widths:
            layers.extend((nn.Linear(cur_features, width), nn.Tanh()))
            cur_features = width
        self.hidden = nn.Sequential(*layers)
        self.head = nn.Linear(cur_features, 1)
        if zero_initialize_head:
            nn.init.zeros_(self.head.weight)
            nn.init.zeros_(self.head.bias)

    def forward(self, phase: Tensor) -> Tensor:
        """Return a signed increment rate with the same shape as ``phase``."""

        if phase.ndim != 4 or phase.shape[1] != 1:
            raise ValueError("phase must have shape (batch, 1, height, width)")
        batch, _, height, width = phase.shape

        if self.coordinate_encoding == "none":
            features = phase.permute(0, 2, 3, 1)  # (B, H, W, 1)
        else:
            coord = self._coordinate_features_2d(phase, height, width)
            features = torch.cat((phase, coord.expand(batch, -1, -1, -1)), dim=1).permute(0, 2, 3, 1)

        rate = self.head(self.hidden(features))
        return rate.permute(0, 3, 1, 2)

    def _coordinate_features_2d(self, phase: Tensor, height: int, width: int) -> Tensor:
        """Return raw or periodic coordinate channels shaped ``(1, C, H, W)``."""

        key = (
            self.coordinate_encoding,
            height,
            width,
            str(phase.device),
            phase.dtype,
        )
        cached = self._coordinate_feature_cache.get(key) if self.cache_coordinate_features else None
        if cached is not None:
            return cached

        y = torch.linspace(-1.0, 1.0, height, device=phase.device, dtype=phase.dtype)
        x = torch.linspace(-1.0, 1.0, width, device=phase.device, dtype=phase.dtype)
        yy, xx = torch.meshgrid(y, x, indexing="ij")
        if self.coordinate_encoding == "periodic":
            coord = torch.stack(
                (
                    torch.sin(math.pi * xx),
                    torch.cos(math.pi * xx),
                    torch.sin(math.pi * yy),
                    torch.cos(math.pi * yy),
                ),
                dim=0,
            ).unsqueeze(0)
        else:  # "raw"
            coord = torch.stack((xx, yy), dim=0).unsqueeze(0)

        if self.cache_coordinate_features:
            self._coordinate_feature_cache.clear()
            self._coordinate_feature_cache[key] = coord
        return coord


class PINNPhaseHybridRollout(nn.Module):
    """Advance a scalar phase field with ANN and ConvRNN increment branches.

    Five branch modes are supported:

    **hybrid**:
        ``increment = (1-gamma) * dt * ann_rate + gamma * dt * recurrent_rate``

    **operator_split**:
        ``∂φ/∂t = F_alg(φ, δg) + F_rnn(φ, h_t) + λ × F_ann(φ, x, y)``
        The algebraic RHS is always at full strength.  The ConvLSTM learns the
        Laplacian/curvature term.  The ANN provides a small bounded correction.
        ``physics_guided_ann=True`` is required for this mode.

    **learned_operator_split**:
        ``∂φ/∂t = F_ann_local(φ, x, y) + F_rnn_spatial(φ, h_t)``.
        The ANN and ConvLSTM learn complementary PDE terms from physics losses.
        No trusted PDE operator is injected into the predicted rollout.

    **exact_laplacian_residual** (calibration control only):
        ``∂φ/∂t = F_alg + μσ∇²φ + F_rnn_residual + λ × F_ann_residual``.
        The known scalar spatial operator stays deterministic. The learned
        branches are zero-initialized residual closures.

    **ann_only / convlstm_only**: historical gamma-blend ablation modes.

    **learned_ann_only / learned_convlstm_only**:
        Matched single-branch controls for ``learned_operator_split``. These
        modes retain the current model-field physics-target training contract
        while disabling the complementary learned branch.

    All modes end with ``phi_next = clip(phi + dt * ∂φ/∂t, 0, 1)``.
    """

    def __init__(
        self,
        *,
        dt: float,
        recurrent_cell: str = "convlstm",
        hidden_channels: int = 8,
        kernel_size: int = 3,
        ann_hidden_features: Sequence[int] = (32, 32),
        blend_gamma: float = 0.5,
        learnable_blend: bool = False,
        branch_mode: str = "hybrid",
        project_bounds: bool = True,
        zero_initialize_heads: bool = True,
        physics_guided_ann: bool = False,
        mu: float | None = None,
        sigma: float | None = None,
        eta: float | None = None,
        delta_g: float = 0.0,
        spacings: Sequence[float] | None = None,
        init_lambda_correction: float = 0.1,
        conv_padding_mode: str = "zeros",
        coordinate_encoding: str = "raw",
        cache_coordinate_features: bool = False,
        recurrent_gate_fusion: str = "separate",
    ) -> None:
        super().__init__()
        if dt <= 0.0:
            raise ValueError("dt must be positive")
        if recurrent_cell not in {"convlstm", "convgru"}:
            raise ValueError("recurrent_cell must be 'convlstm' or 'convgru'")
        _valid_modes = {
            "hybrid",
            "ann_only",
            "convlstm_only",
            "operator_split",
            "learned_operator_split",
            "learned_ann_only",
            "learned_convlstm_only",
            "exact_laplacian_residual",
        }
        if branch_mode not in _valid_modes:
            raise ValueError(f"branch_mode must be one of {_valid_modes}")
        if branch_mode in {"operator_split", "exact_laplacian_residual"} and not physics_guided_ann:
            raise ValueError(f"{branch_mode} requires physics_guided_ann=True")
        if physics_guided_ann and any(v is None for v in [mu, sigma, eta]):
            raise ValueError("mu, sigma, eta are required when physics_guided_ann=True")
        if branch_mode == "exact_laplacian_residual":
            if spacings is None or len(spacings) != 2 or any(float(v) <= 0.0 for v in spacings):
                raise ValueError("exact_laplacian_residual requires two positive spacings")
        if not (0.0 < init_lambda_correction < 1.0):
            raise ValueError("init_lambda_correction must be in (0, 1)")
        if conv_padding_mode not in {"zeros", "circular", "reflect", "replicate"}:
            raise ValueError(
                f"conv_padding_mode must be 'zeros', 'circular', 'reflect', or 'replicate'; "
                f"got {conv_padding_mode!r}"
            )
        if coordinate_encoding not in {"raw", "periodic", "none"}:
            raise ValueError(
                f"coordinate_encoding must be 'raw', 'periodic', or 'none'; "
                f"got {coordinate_encoding!r}"
            )
        if recurrent_gate_fusion not in {"separate", "concat"}:
            raise ValueError(
                f"recurrent_gate_fusion must be 'separate' or 'concat'; "
                f"got {recurrent_gate_fusion!r}"
            )
        if recurrent_cell != "convlstm" and recurrent_gate_fusion != "separate":
            raise ValueError("recurrent_gate_fusion='concat' is supported only for convlstm")

        self.dt = float(dt)
        self.recurrent_cell_name = recurrent_cell
        self.branch_mode = branch_mode
        self.project_bounds = bool(project_bounds)
        self.physics_guided_ann = bool(physics_guided_ann)
        self.conv_padding_mode = conv_padding_mode
        self.coordinate_encoding = coordinate_encoding
        self.cache_coordinate_features = bool(cache_coordinate_features)
        self.recurrent_gate_fusion = recurrent_gate_fusion
        if physics_guided_ann:
            self.register_buffer("_pga_mu", torch.tensor(float(mu)))
            self.register_buffer("_pga_sigma", torch.tensor(float(sigma)))
            self.register_buffer("_pga_eta", torch.tensor(float(eta)))
            self.register_buffer("_pga_delta_g", torch.tensor(float(delta_g)))
            if spacings is not None:
                self.register_buffer("_pga_dx", torch.tensor(float(spacings[0])))
                self.register_buffer("_pga_dy", torch.tensor(float(spacings[1])))
        self.ann = PointwiseANNIncrement(
            ann_hidden_features,
            zero_initialize_head=zero_initialize_heads,
            coordinate_encoding=coordinate_encoding,
            cache_coordinate_features=cache_coordinate_features,
        )
        cell_type = ConvLSTMCell if recurrent_cell == "convlstm" else ConvGRUCell
        self.recurrent_cell = cell_type(
            in_channels=1,
            hidden_channels=hidden_channels,
            kernel_size=kernel_size,
            padding_mode=conv_padding_mode,
            gate_fusion=recurrent_gate_fusion,
        )
        self.recurrent_head = nn.Conv2d(
            hidden_channels,
            1,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            padding_mode=conv_padding_mode,
        )
        if zero_initialize_heads:
            nn.init.zeros_(self.recurrent_head.weight)
            nn.init.zeros_(self.recurrent_head.bias)

        # operator_split: learnable ANN correction weight λ (bounded in (0,1))
        # hybrid/ablation: ResidualBlend with optional learnable gamma
        if branch_mode in {"operator_split", "exact_laplacian_residual"}:
            _lambda_logit = float(torch.logit(torch.tensor(init_lambda_correction)))
            self.lambda_logit = nn.Parameter(torch.tensor(_lambda_logit))
            self.blend = ResidualBlend(dt=self.dt, initial_gamma=0.5, learnable=False)
        else:
            self.lambda_logit = None
            self.blend = ResidualBlend(
                dt=self.dt,
                initial_gamma=blend_gamma,
                learnable=learnable_blend,
            )

        # learned_operator_split learnable scalar blend.
        # Convention: gamma = sigmoid(operator_blend_logit), and
        #   ∂φ/∂t = 2 * (gamma * ann_local + (1 - gamma) * rnn_spatial).
        #   gamma -> 1 : pure ANN local dynamics.
        #   gamma -> 0 : pure ConvLSTM/recurrent spatial dynamics.
        # The factor 2 makes gamma=0.5 reproduce the historical unweighted sum
        # (ann_local + rnn_spatial) EXACTLY at initialization, so a learnable
        # run starts identical to the fixed-blend run. raw_gamma initialises to 0
        # (gamma = 0.5). When learnable_blend is False the logit is a fixed
        # buffer at 0, preserving exact backward compatibility.
        self.operator_blend_learnable = bool(
            learnable_blend and branch_mode == "learned_operator_split"
        )
        if branch_mode == "learned_operator_split":
            _ob_logit = torch.zeros((), dtype=torch.float32)  # sigmoid(0) = 0.5
            if self.operator_blend_learnable:
                self.operator_blend_logit = nn.Parameter(_ob_logit)
            else:
                self.register_buffer("operator_blend_logit", _ob_logit)
        else:
            self.operator_blend_logit = None

        if branch_mode in {"ann_only", "learned_ann_only"}:
            self.recurrent_cell.requires_grad_(False)
            self.recurrent_head.requires_grad_(False)
        elif branch_mode in {"convlstm_only", "learned_convlstm_only"}:
            self.ann.requires_grad_(False)

    @property
    def lambda_correction(self) -> Tensor | None:
        """Return the ANN correction weight λ for operator_split mode."""
        if self.lambda_logit is None:
            return None
        return torch.sigmoid(self.lambda_logit)

    @property
    def operator_blend_gamma(self) -> Tensor | None:
        """Return the learned_operator_split scalar blend γ in (0, 1).

        Convention: ∂φ/∂t = 2 * (γ·ann_local + (1−γ)·rnn_spatial).
        γ→1 weights the ANN local branch; γ→0 weights the ConvLSTM branch.
        γ=0.5 reproduces the historical unweighted sum exactly. Returns None
        for any branch_mode other than learned_operator_split.
        """
        if self.operator_blend_logit is None:
            return None
        return torch.sigmoid(self.operator_blend_logit)

    @property
    def blend_gamma(self) -> Tensor:
        """Return gamma (hybrid) or λ (operator_split) for unified monitoring."""
        if self.branch_mode in {"operator_split", "exact_laplacian_residual"}:
            return self.lambda_correction
        return self.blend.blend_gamma

    @property
    def branch_weights(self) -> tuple[Tensor, Tensor]:
        """Return effective ANN and recurrent weights for the selected ablation."""

        gamma = self.blend.blend_gamma
        if self.branch_mode in {"ann_only", "learned_ann_only"}:
            return torch.ones_like(gamma), torch.zeros_like(gamma)
        if self.branch_mode in {"convlstm_only", "learned_convlstm_only"}:
            return torch.zeros_like(gamma), torch.ones_like(gamma)
        if self.branch_mode == "learned_operator_split":
            ob = self.operator_blend_gamma
            if ob is None:  # defensive; learned_operator_split always sets it
                return torch.ones_like(gamma), torch.ones_like(gamma)
            return 2.0 * ob, 2.0 * (1.0 - ob)  # (ANN weight, ConvLSTM weight)
        if self.branch_mode in {"operator_split", "exact_laplacian_residual"}:
            lam = self.lambda_correction
            return lam, torch.ones_like(lam)  # ANN weight=lambda, ConvLSTM weight=1
        return 1.0 - gamma, gamma

    def initial_state(self, phase: Tensor) -> RecurrentState:
        """Return zero recurrent state matching ``phase``."""

        self._validate_phase(phase)
        batch, _, height, width = phase.shape
        return self.recurrent_cell.initial_state(
            batch,
            height,
            width,
            device=phase.device,
        )

    def _algebraic_rhs(self, phase: Tensor) -> Tensor:
        """Return algebraic Allen-Cahn RHS (bulk + driving, no Laplacian)."""
        return _algebraic_rhs_fn(
            phase,
            mu=self._pga_mu.item(),
            sigma=self._pga_sigma.item(),
            eta=self._pga_eta.item(),
            delta_g=self._pga_delta_g.item(),
        )

    def _laplacian_rhs(self, phase: Tensor) -> Tensor:
        """Return the deterministic scalar Laplacian contribution."""

        return _laplacian_rhs_fn(
            phase,
            mu=self._pga_mu.item(),
            sigma=self._pga_sigma.item(),
            spacings=(self._pga_dx.item(), self._pga_dy.item()),
        )

    def branch_derivatives(
        self,
        phase: Tensor,
        state: RecurrentState,
    ) -> tuple[Tensor, Tensor, RecurrentState]:
        """Return per-branch signed rates and next recurrent state.

        **operator_split mode** returns:
            (rnn_correction, effective_ann_contribution, next_state)
            where effective_ann = lambda * ann_correction.
            The algebraic RHS is NOT included here; it is added in predict_derivative.

        **hybrid mode** returns:
            (ann_rate, recurrent_rate, next_state)
            where ann_rate includes the algebraic anchor if physics_guided_ann=True.

        Both modes have the same signature for unified monitoring in pinn.py.
        """

        self._validate_phase(phase)
        rnn_rate, next_state = self._recurrent_derivative(phase, state)
        ann_corr = self.ann(phase)

        if self.branch_mode in {"operator_split", "exact_laplacian_residual"}:
            # Monitoring: return (rnn_correction, lambda*ann_correction, next_state)
            # The algebraic term is NOT returned here — it is deterministic and added
            # separately in predict_derivative.
            lam = self.lambda_correction
            return rnn_rate, lam * ann_corr, next_state

        # hybrid / legacy modes: return (ann_rate_with_anchor, rnn_rate, next_state)
        if self.physics_guided_ann:
            ann_rate = self._algebraic_rhs(phase) + ann_corr
        else:
            ann_rate = ann_corr
        return ann_rate, rnn_rate, next_state

    def _recurrent_derivative(
        self,
        phase: Tensor,
        state: RecurrentState,
    ) -> tuple[Tensor, RecurrentState]:
        """Return the recurrent rate and next state without evaluating the ANN."""

        if self.recurrent_cell_name == "convlstm":
            if not isinstance(state, tuple):
                raise TypeError("ConvLSTM state must be a (hidden, cell) tuple")
            hidden, cell = state
            next_hidden, next_cell = self.recurrent_cell(phase, hidden, cell)
            next_state: RecurrentState = (next_hidden, next_cell)
        else:
            if isinstance(state, tuple):
                raise TypeError("ConvGRU state must be a hidden-state tensor")
            next_hidden = self.recurrent_cell(phase, state)
            next_state = next_hidden
        recurrent_rate = self.recurrent_head(next_hidden)
        return recurrent_rate, next_state

    def predict_derivative(
        self,
        phase: Tensor,
        state: RecurrentState,
        return_branches: bool = False,
    ) -> tuple[Tensor, RecurrentState] | tuple[Tensor, RecurrentState, dict[str, Tensor]]:
        """Return signed derivative ∂φ/∂t and next recurrent state.

        **operator_split**:
            ∂φ/∂t = F_alg(φ,δg) + F_rnn(φ,h) + λ × F_ann(φ,x,y)
            F_alg is always at full strength.  Zero-initialized heads give
            ∂φ/∂t = F_alg at epoch 0 — a pure algebraic physics solver.

            When ``return_branches=True``, returns a third element — a dict
            with keys ``"algebraic"``, ``"rnn"``, ``"ann"`` — so the training
            loop can compute auxiliary losses (Laplacian-target, etc.) without
            a second forward pass.

        **learned_operator_split**:
            ∂φ/∂t = F_ann_local(φ,x,y) + F_rnn_spatial(φ,h)
            Both branches are learned. Physics operators appear only in losses.

        **hybrid**:
            ∂φ/∂t = ResidualBlend(ann_rate, rnn_rate) / dt
        """

        if self.branch_mode == "operator_split":
            algebraic = self._algebraic_rhs(phase)
            rnn_corr, next_state = self._recurrent_derivative(phase, state)
            ann_corr = self.ann(phase)
            lam = self.lambda_correction
            derivative = algebraic + rnn_corr + lam * ann_corr
            if return_branches:
                return derivative, next_state, {
                    "algebraic": algebraic,
                    "rnn": rnn_corr,
                    "ann": ann_corr,
                }
            return derivative, next_state
        if self.branch_mode == "learned_operator_split":
            rnn_spatial, next_state = self._recurrent_derivative(phase, state)
            ann_local = self.ann(phase)
            # Learnable scalar blend. gamma=0.5 reproduces ann_local + rnn_spatial
            # exactly (factor 2). gamma->1 pure ANN, gamma->0 pure ConvLSTM.
            gamma = self.operator_blend_gamma  # sigmoid(logit), in (0, 1)
            ann_eff = 2.0 * gamma * ann_local
            rnn_eff = 2.0 * (1.0 - gamma) * rnn_spatial
            derivative = ann_eff + rnn_eff
            if return_branches:
                # Expose BOTH the raw per-branch rates and the post-blend
                # effective contributions, so logging can detect branch collapse
                # under the learned gamma without re-deriving the weights.
                return derivative, next_state, {
                    "ann": ann_local,
                    "rnn": rnn_spatial,
                    "ann_effective": ann_eff,
                    "rnn_effective": rnn_eff,
                }
            return derivative, next_state
        if self.branch_mode == "learned_ann_only":
            ann_local = self.ann(phase)
            if return_branches:
                return ann_local, state, {
                    "ann": ann_local,
                    "rnn": torch.zeros_like(ann_local),
                }
            return ann_local, state
        if self.branch_mode == "learned_convlstm_only":
            rnn_spatial, next_state = self._recurrent_derivative(phase, state)
            if return_branches:
                return rnn_spatial, next_state, {
                    "ann": torch.zeros_like(rnn_spatial),
                    "rnn": rnn_spatial,
                }
            return rnn_spatial, next_state
        if self.branch_mode == "exact_laplacian_residual":
            algebraic = self._algebraic_rhs(phase)
            laplacian = self._laplacian_rhs(phase)
            rnn_corr, next_state = self._recurrent_derivative(phase, state)
            ann_corr = self.ann(phase)
            derivative = algebraic + laplacian + rnn_corr + self.lambda_correction * ann_corr
            if return_branches:
                return derivative, next_state, {
                    "algebraic": algebraic,
                    "laplacian": laplacian,
                    "rnn": rnn_corr,
                    "ann": ann_corr,
                }
            return derivative, next_state
        if self.branch_mode == "ann_only":
            if self.physics_guided_ann:
                return self._algebraic_rhs(phase) + self.ann(phase), state
            return self.ann(phase), state
        if self.branch_mode == "convlstm_only":
            return self._recurrent_derivative(phase, state)
        # hybrid
        ann_rate, recurrent_rate, next_state = self.branch_derivatives(phase, state)
        increment = self.blend(self.dt * ann_rate, recurrent_rate)
        return increment / self.dt, next_state

    def set_bound_enforcement_mode(
        self,
        mode: str = DEFAULT_BOUND_ENFORCEMENT_MODE,
        *,
        boundary_eps: float = DEFAULT_BOUNDARY_EPS,
    ) -> None:
        """Install a diagnostic-only bound-enforcement override (default-off).

        DIAGNOSTIC USE ONLY.  ``project_and_clamp`` with the default
        ``boundary_eps`` reproduces the production rollout exactly; any other
        mode or ``boundary_eps`` is a reduced-projection audit setting and must
        never be used as the production baseline.  This sets plain runtime
        attributes only; it does not touch ``state_dict`` or training.
        """

        _set_bound_enforcement_mode(self, mode, boundary_eps=boundary_eps)

    def forward_step(
        self,
        phase: Tensor,
        state: RecurrentState | None = None,
    ) -> tuple[Tensor, RecurrentState, Tensor]:
        """Advance one explicit phase-preserving step."""

        if state is None:
            state = self.initial_state(phase)
        derivative, next_state = self.predict_derivative(phase, state)
        next_phase = _step_with_bound_enforcement(self, phase, derivative)
        return next_phase, next_state, derivative

    def rollout(self, initial_phase: Tensor, *, steps: int) -> Tensor:
        """Return a full inference rollout including the initial state."""

        if steps < 1:
            raise ValueError("steps must be positive")
        phase = initial_phase
        state = self.initial_state(phase)
        states = [phase]
        for _ in range(steps):
            phase, state, _ = self.forward_step(phase, state)
            states.append(phase)
        return torch.stack(states)

    def iter_tbptt_segments(
        self,
        initial_phase: Tensor,
        *,
        steps: int,
        window: int,
    ) -> Iterator[Tensor]:
        """Yield rollout segments with detached recurrent boundaries."""

        if steps < 1 or window < 1:
            raise ValueError("steps and window must be positive")
        phase = initial_phase
        state = self.initial_state(phase)
        completed = 0
        while completed < steps:
            segment_states = [phase]
            for _ in range(min(window, steps - completed)):
                phase, state, _ = self.forward_step(phase, state)
                segment_states.append(phase)
                completed += 1
            yield torch.stack(segment_states)
            phase = phase.detach()
            state = detach_recurrent_state(state)

    @staticmethod
    def _validate_phase(phase: Tensor) -> None:
        if phase.ndim != 4 or phase.shape[1] != 1:
            raise ValueError("phase must have shape (batch, 1, height, width)")


class PointwiseANNIncrement3d(nn.Module):
    """Predict a signed local increment rate from ``phi`` and optional coordinates.

    3D analogue of :class:`PointwiseANNIncrement`. Accepts phase tensors of
    shape ``(batch, 1, D, H, W)`` and returns the same shape.

    The positional coordinates are normalised to ``[-1, 1]`` along each axis.
    Encodings mirror the 2D ANN: ``raw`` uses ``(phi,z,y,x)``, ``periodic``
    uses ``phi`` plus sin/cos per axis, and ``none`` uses only ``phi``.
    """

    def __init__(
        self,
        hidden_features: Sequence[int] = (32, 32),
        *,
        zero_initialize_head: bool = True,
        coordinate_encoding: str = "raw",
        cache_coordinate_features: bool = False,
    ) -> None:
        super().__init__()
        if coordinate_encoding not in {"raw", "periodic", "none"}:
            raise ValueError(
                f"coordinate_encoding must be 'raw', 'periodic', or 'none'; "
                f"got {coordinate_encoding!r}"
            )
        widths = tuple(int(w) for w in hidden_features)
        if not widths or any(w < 1 for w in widths):
            raise ValueError("hidden_features must contain positive widths")

        self.coordinate_encoding = coordinate_encoding
        self.cache_coordinate_features = bool(cache_coordinate_features)
        self._coordinate_feature_cache: dict[tuple[object, ...], Tensor] = {}
        layers: list[nn.Module] = []
        if coordinate_encoding == "none":
            in_features = 1
        elif coordinate_encoding == "periodic":
            in_features = 7  # phi + sin/cos for z, y, x
        else:
            in_features = 4  # phi, z, y, x
        for width in widths:
            layers.extend((nn.Linear(in_features, width), nn.Tanh()))
            in_features = width
        self.hidden = nn.Sequential(*layers)
        self.head = nn.Linear(in_features, 1)
        if zero_initialize_head:
            nn.init.zeros_(self.head.weight)
            nn.init.zeros_(self.head.bias)

    def forward(self, phase: Tensor) -> Tensor:
        """Return a signed increment rate with the same shape as ``phase``.

        Parameters
        ----------
        phase:
            Shape ``(batch, 1, D, H, W)``.
        """
        if phase.ndim != 5 or phase.shape[1] != 1:
            raise ValueError("phase must have shape (batch, 1, depth, height, width)")
        batch, _, depth, height, width = phase.shape
        if self.coordinate_encoding == "none":
            features = phase.permute(0, 2, 3, 4, 1)  # (B,D,H,W,1)
        else:
            coord = self._coordinate_features_3d(phase, depth, height, width)
            features = torch.cat((phase, coord.expand(batch, -1, -1, -1, -1)), dim=1).permute(0, 2, 3, 4, 1)
        rate = self.head(self.hidden(features))
        return rate.permute(0, 4, 1, 2, 3)  # (B,1,D,H,W)

    def _coordinate_features_3d(
        self,
        phase: Tensor,
        depth: int,
        height: int,
        width: int,
    ) -> Tensor:
        """Return raw or periodic coordinate channels shaped ``(1, C, D, H, W)``."""

        key = (
            self.coordinate_encoding,
            depth,
            height,
            width,
            str(phase.device),
            phase.dtype,
        )
        cached = self._coordinate_feature_cache.get(key) if self.cache_coordinate_features else None
        if cached is not None:
            return cached

        zg = torch.linspace(-1.0, 1.0, depth, device=phase.device, dtype=phase.dtype)
        yg = torch.linspace(-1.0, 1.0, height, device=phase.device, dtype=phase.dtype)
        xg = torch.linspace(-1.0, 1.0, width, device=phase.device, dtype=phase.dtype)
        zzz, yyy, xxx = torch.meshgrid(zg, yg, xg, indexing="ij")  # each (D,H,W)
        if self.coordinate_encoding == "periodic":
            coord = torch.stack(
                (
                    torch.sin(math.pi * zzz),
                    torch.cos(math.pi * zzz),
                    torch.sin(math.pi * yyy),
                    torch.cos(math.pi * yyy),
                    torch.sin(math.pi * xxx),
                    torch.cos(math.pi * xxx),
                ),
                dim=0,
            ).unsqueeze(0)
        else:  # "raw"
            coord = torch.stack((zzz, yyy, xxx), dim=0).unsqueeze(0)

        if self.cache_coordinate_features:
            self._coordinate_feature_cache.clear()
            self._coordinate_feature_cache[key] = coord
        return coord


class PINNPhaseHybridRollout3d(nn.Module):
    """3D learned operator-split hybrid rollout for scalar phase-field problems.

    Architecture:
        ``∂φ/∂t = 2·(γ·ANN_local(φ,z,y,x) + (1−γ)·ConvLSTM3d_spatial(φ,h))``

    This is the 3D analogue of :class:`PINNPhaseHybridRollout` with
    ``branch_mode='learned_operator_split'``. The recurrent cell uses
    ``nn.Conv3d`` (via :class:`ConvLSTMCell3d` or :class:`ConvGRUCell3d`).

    Safety contract
    ---------------
    This class ONLY accepts 5D phase tensors ``(batch, 1, D, H, W)``.
    Passing a 4D tensor raises ``ValueError``. The 2D rollout is not changed.

    Parameters
    ----------
    dt:
        Integration time step.
    recurrent_cell:
        ``'convlstm'`` (default) or ``'convgru'``.
    hidden_channels:
        Number of 3D Conv hidden channels.
    kernel_size:
        Odd kernel size for 3D convolutions.
    ann_hidden_features:
        Hidden layer widths for the pointwise ANN branch.
    learnable_blend:
        If True, the ANN/ConvLSTM blend ``γ`` is a learned scalar parameter.
    project_bounds:
        If True, clip φ to [0,1] and project boundary rates.
    zero_initialize_heads:
        If True, zero-initialize ANN and recurrent output heads.
    conv_padding_mode:
        Padding mode for Conv3d recurrent layers and recurrent head.  Use
        ``"circular"`` for periodic-domain scalar 3D diagnostics.
    """

    def __init__(
        self,
        *,
        dt: float,
        recurrent_cell: str = "convlstm",
        hidden_channels: int = 8,
        kernel_size: int = 3,
        ann_hidden_features: Sequence[int] = (32, 32),
        learnable_blend: bool = False,
        project_bounds: bool = True,
        zero_initialize_heads: bool = True,
        conv_padding_mode: str = "zeros",
        coordinate_encoding: str = "raw",
        cache_coordinate_features: bool = False,
        recurrent_gate_fusion: str = "separate",
    ) -> None:
        super().__init__()
        if dt <= 0.0:
            raise ValueError("dt must be positive")
        if recurrent_cell not in {"convlstm", "convgru"}:
            raise ValueError("recurrent_cell must be 'convlstm' or 'convgru'")
        if conv_padding_mode not in {"zeros", "circular", "reflect", "replicate"}:
            raise ValueError(
                f"conv_padding_mode must be 'zeros', 'circular', 'reflect', or 'replicate'; "
                f"got {conv_padding_mode!r}"
            )
        if coordinate_encoding not in {"raw", "periodic", "none"}:
            raise ValueError(
                f"coordinate_encoding must be 'raw', 'periodic', or 'none'; "
                f"got {coordinate_encoding!r}"
            )
        if recurrent_gate_fusion not in {"separate", "concat"}:
            raise ValueError(
                f"recurrent_gate_fusion must be 'separate' or 'concat'; "
                f"got {recurrent_gate_fusion!r}"
            )
        if recurrent_cell != "convlstm" and recurrent_gate_fusion != "separate":
            raise ValueError("recurrent_gate_fusion='concat' is supported only for convlstm")

        self.dt = float(dt)
        self.branch_mode = "learned_operator_split"
        self.recurrent_cell_name = recurrent_cell
        self.project_bounds = bool(project_bounds)
        self.conv_padding_mode = conv_padding_mode
        self.coordinate_encoding = coordinate_encoding
        self.cache_coordinate_features = bool(cache_coordinate_features)
        self.recurrent_gate_fusion = recurrent_gate_fusion

        self.ann = PointwiseANNIncrement3d(
            ann_hidden_features,
            zero_initialize_head=zero_initialize_heads,
            coordinate_encoding=coordinate_encoding,
            cache_coordinate_features=cache_coordinate_features,
        )

        cell_type = ConvLSTMCell3d if recurrent_cell == "convlstm" else ConvGRUCell3d
        self.recurrent_cell = cell_type(
            in_channels=1,
            hidden_channels=hidden_channels,
            kernel_size=kernel_size,
            padding_mode=conv_padding_mode,
            gate_fusion=recurrent_gate_fusion,
        )
        self.recurrent_head = nn.Conv3d(
            hidden_channels,
            1,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            padding_mode=conv_padding_mode,
        )
        if zero_initialize_heads:
            nn.init.zeros_(self.recurrent_head.weight)
            nn.init.zeros_(self.recurrent_head.bias)

        # Learnable scalar blend γ: sigmoid(logit), init 0.5.
        # ∂φ/∂t = 2·(γ·ANN + (1−γ)·ConvLSTM3d)
        _ob_logit = torch.zeros((), dtype=torch.float32)
        if learnable_blend:
            self.operator_blend_logit = nn.Parameter(_ob_logit)
        else:
            self.register_buffer("operator_blend_logit", _ob_logit)
        self.operator_blend_learnable = bool(learnable_blend)

    @property
    def blend_gamma(self) -> Tensor:
        """Return the blend weight γ ∈ (0,1)."""
        return torch.sigmoid(self.operator_blend_logit)

    @property
    def operator_blend_gamma(self) -> Tensor:
        """Return the learned_operator_split scalar blend γ in (0, 1)."""
        return self.blend_gamma

    def initial_state(self, phase: Tensor) -> RecurrentState:
        """Return zero recurrent state matching the spatial shape of ``phase``."""
        self._validate_phase_3d(phase)
        batch, _, depth, height, width = phase.shape
        return self.recurrent_cell.initial_state(
            batch, depth, height, width, device=phase.device
        )

    def _recurrent_derivative(
        self, phase: Tensor, state: RecurrentState
    ) -> tuple[Tensor, RecurrentState]:
        if self.recurrent_cell_name == "convlstm":
            if not isinstance(state, tuple):
                raise TypeError("ConvLSTM3d state must be a (hidden, cell) tuple")
            hidden, cell = state
            next_hidden, next_cell = self.recurrent_cell(phase, hidden, cell)
            next_state: RecurrentState = (next_hidden, next_cell)
        else:
            if isinstance(state, tuple):
                raise TypeError("ConvGRU3d state must be a hidden-state tensor")
            next_hidden = self.recurrent_cell(phase, state)
            next_state = next_hidden
        recurrent_rate = self.recurrent_head(next_hidden)
        return recurrent_rate, next_state

    def predict_derivative(
        self,
        phase: Tensor,
        state: RecurrentState,
        return_branches: bool = False,
    ) -> tuple[Tensor, RecurrentState] | tuple[Tensor, RecurrentState, dict[str, Tensor]]:
        """Return signed derivative ∂φ/∂t and next recurrent state.

        Mode: learned_operator_split (3D).
        ∂φ/∂t = 2·(γ·ANN_local + (1−γ)·ConvLSTM3d_spatial)
        """
        self._validate_phase_3d(phase)
        rnn_spatial, next_state = self._recurrent_derivative(phase, state)
        ann_local = self.ann(phase)
        gamma = self.blend_gamma
        ann_eff = 2.0 * gamma * ann_local
        rnn_eff = 2.0 * (1.0 - gamma) * rnn_spatial
        derivative = ann_eff + rnn_eff
        if return_branches:
            return derivative, next_state, {
                "ann": ann_local,
                "rnn": rnn_spatial,
                "ann_effective": ann_eff,
                "rnn_effective": rnn_eff,
            }
        return derivative, next_state

    def set_bound_enforcement_mode(
        self,
        mode: str = DEFAULT_BOUND_ENFORCEMENT_MODE,
        *,
        boundary_eps: float = DEFAULT_BOUNDARY_EPS,
    ) -> None:
        """Install a diagnostic-only bound-enforcement override (default-off).

        DIAGNOSTIC USE ONLY.  ``project_and_clamp`` with the default
        ``boundary_eps`` reproduces the production rollout exactly; any other
        mode or ``boundary_eps`` is a reduced-projection audit setting and must
        never be used as the production baseline.  This sets plain runtime
        attributes only; it does not touch ``state_dict`` or training.
        """

        _set_bound_enforcement_mode(self, mode, boundary_eps=boundary_eps)

    def forward_step(
        self,
        phase: Tensor,
        state: RecurrentState | None = None,
    ) -> tuple[Tensor, RecurrentState, Tensor]:
        """Advance one explicit phase-preserving step (3D)."""
        if state is None:
            state = self.initial_state(phase)
        derivative, next_state = self.predict_derivative(phase, state)
        next_phase = _step_with_bound_enforcement(self, phase, derivative)
        return next_phase, next_state, derivative

    def rollout(self, initial_phase: Tensor, *, steps: int) -> Tensor:
        """Return a full inference rollout including the initial state (3D)."""
        if steps < 1:
            raise ValueError("steps must be positive")
        phase = initial_phase
        state = self.initial_state(phase)
        states = [phase]
        for _ in range(steps):
            phase, state, _ = self.forward_step(phase, state)
            states.append(phase)
        return torch.stack(states)

    def iter_tbptt_segments(
        self,
        initial_phase: Tensor,
        *,
        steps: int,
        window: int,
    ) -> Iterator[Tensor]:
        """Yield rollout segments with detached recurrent boundaries (3D)."""
        if steps < 1 or window < 1:
            raise ValueError("steps and window must be positive")
        phase = initial_phase
        state = self.initial_state(phase)
        completed = 0
        while completed < steps:
            segment_states = [phase]
            for _ in range(min(window, steps - completed)):
                phase, state, _ = self.forward_step(phase, state)
                segment_states.append(phase)
                completed += 1
            yield torch.stack(segment_states)
            phase = phase.detach()
            state = detach_recurrent_state(state)

    @staticmethod
    def _validate_phase_3d(phase: Tensor) -> None:
        if phase.ndim != 5 or phase.shape[1] != 1:
            raise ValueError("phase must have shape (batch, 1, depth, height, width) for 3D rollout")
