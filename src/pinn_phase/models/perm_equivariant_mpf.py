"""Permutation-equivariant explicit-MPF rollout (arch_variant ``perm_equivariant_v1``).

This module is **additive and default-off**. It is only constructed when a
config sets ``model.arch_variant: perm_equivariant_v1``; when that key is
absent the legacy :class:`ExplicitMPFHybridRollout` is used unchanged, so every
existing behavior is byte-identical (see ``build_explicit_mpf_model``).

Design contract. For ``phi`` shaped
``[B, N, H, W]``, per phase ``i``::

    e_i       = E(phi_i)                       # shared pointwise/circular-conv encoder
    a_mean    = mean_j(e_j)                     # over ALL phases (include self)
    a_max     = max_j(e_j)                      # over ALL phases (symmetric-subgradient amax)
    x_i       = concat(phi_i, e_i, a_mean, a_max)
    (h_i,c_i) = SharedConvLSTM(x_i, h_i, c_i)   # phasewise state [B,N,C_h,H,W]
    r_i^rnn   = SharedConvHead(h_i)
    r_i^ann   = SharedPointwiseANN(phi_i)
    z_i       = gamma*r_i^ann + (1-gamma)*r_i^rnn   # gamma = ONE global scalar
    delta_i   = delta_scale*tanh(z_i)
    delta_i   = delta_i - mean_j(delta_j)       # phase_mean centering
    phi_next  = soft_threshold_eps1e3(phi + delta)

Structural equivariance:

- ``E``, ``SharedConvLSTM``, ``SharedConvHead`` and ``SharedPointwiseANN`` are all
  applied per phase with weights shared across phases and independent of ``N``.
  No learnable tensor carries an ``N``-sized phase dimension, so one ``state_dict``
  is structurally loadable for any ``N``.
- ``a_mean``/``a_max`` are symmetric reductions over ALL phases, so a phase
  permutation ``P`` applied to ``phi`` (and to the phasewise recurrent state
  ``h``/``c``) commutes with every op: ``f(P·x) = P·f(x)``.
- All spatial convolutions use circular padding, so integer periodic translations
  commute with the whole map.
- ``a_max`` uses ``torch.amax`` whose backward distributes gradient equally among
  tied maximal elements, giving a permutation-symmetric subgradient at exact
  ties.

The public API (``initial_state``, ``predict_rate``, ``forward_step``,
``rollout``) matches :class:`ExplicitMPFHybridRollout` so downstream scientific
metric code is unchanged.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn as nn
from torch import Tensor

from pinn_phase.physics.explicit_mpf import (
    project_simplex,
    project_simplex_soft_threshold,
)

from .cells import ConvLSTMCell
from .explicit_mpf import MPFRecurrentState, MultiPhasePointwiseANN, _validate_phi


class SharedPhaseEncoder(nn.Module):
    """Compact shared per-phase encoder ``E``: single-channel -> ``C_e`` channels.

    Applied to each phase independently by reshaping ``[B,N,1,H,W]`` to
    ``[B*N,1,H,W]``; weights are shared across phases and independent of ``N``.
    Circular padding keeps it translation-equivariant.
    """

    def __init__(
        self,
        encoder_channels: int,
        *,
        kernel_size: int = 3,
        padding_mode: str = "circular",
    ) -> None:
        super().__init__()
        if encoder_channels < 1:
            raise ValueError("encoder_channels must be positive")
        if kernel_size % 2 == 0:
            raise ValueError("kernel_size must be odd for same-size padding")
        self.encoder_channels = int(encoder_channels)
        self.conv = nn.Conv2d(
            1,
            encoder_channels,
            kernel_size,
            padding=kernel_size // 2,
            padding_mode=padding_mode,
        )

    def forward(self, phi: Tensor) -> Tensor:
        """``phi`` ``[B,N,H,W]`` -> per-phase encoding ``[B,N,C_e,H,W]``."""

        batch, phases, height, width = phi.shape
        flat = phi.reshape(batch * phases, 1, height, width)
        encoded = torch.tanh(self.conv(flat))
        return encoded.reshape(batch, phases, self.encoder_channels, height, width)


class PermEquivariantMPFRollout(nn.Module):
    """Permutation- and translation-equivariant explicit-MPF hybrid rollout."""

    arch_variant = "perm_equivariant_v1"

    def __init__(
        self,
        *,
        num_phases: int,
        model_dt: float,
        eta_px: float,
        mu: float = 1.0,
        sigma: float = 1.0,
        hidden_channels: int = 8,
        encoder_channels: int = 4,
        kernel_size: int = 3,
        ann_hidden_features: Sequence[int] = (32, 32),
        blend_gamma: float = 0.5,
        learnable_blend: bool = True,
        delta_scale: float = 0.10,
        projection_mode: str = "soft_threshold_eps1e3",
        conv_padding_mode: str = "circular",
        zero_initialize_heads: bool = True,
        validate_num_phases: bool = True,
    ) -> None:
        super().__init__()
        if num_phases < 2:
            raise ValueError("num_phases must be at least 2")
        if model_dt <= 0.0:
            raise ValueError("model_dt must be positive")
        if delta_scale <= 0.0:
            raise ValueError("delta_scale must be positive")
        if hidden_channels < 1:
            raise ValueError("hidden_channels must be positive")
        if projection_mode not in {"simplex_clip_normalize", "soft_threshold_eps1e3"}:
            raise ValueError(
                "projection_mode must be 'simplex_clip_normalize' or 'soft_threshold_eps1e3'"
            )
        self.num_phases = int(num_phases)
        self.model_dt = float(model_dt)
        self.eta_px = float(eta_px)
        self.mu = float(mu)
        self.sigma = float(sigma)
        self.spatial_dims = 2
        self.hidden_channels = int(hidden_channels)
        self.encoder_channels = int(encoder_channels)
        self.delta_scale = float(delta_scale)
        self.projection_mode = projection_mode
        # ``validate_num_phases`` only gates INPUT validation; it never changes any
        # learnable tensor shape, so the same state dict remains loadable for any N.
        self.validate_num_phases = bool(validate_num_phases)
        # Structural invariants (v1 has no graph/coord/phase-id/per-phase-gamma).
        self.graph_enabled = False
        self.graph_mode = "disabled"
        self.per_phase_gamma = False
        self.coordinate_encoding = "none"
        self.phase_id_encoding = "none"

        self.encoder = SharedPhaseEncoder(
            encoder_channels,
            kernel_size=kernel_size,
            padding_mode=conv_padding_mode,
        )
        # x_i = concat(phi_i, e_i, a_mean, a_max): 1 + 3*C_e input channels.
        recurrent_in = 1 + 3 * self.encoder_channels
        self.recurrent_cell = ConvLSTMCell(
            in_channels=recurrent_in,
            hidden_channels=hidden_channels,
            kernel_size=kernel_size,
            padding_mode=conv_padding_mode,
        )
        self.recurrent_head = nn.Conv2d(
            hidden_channels,
            1,
            kernel_size,
            padding=kernel_size // 2,
            padding_mode=conv_padding_mode,
        )
        if zero_initialize_heads:
            nn.init.zeros_(self.recurrent_head.weight)
            nn.init.zeros_(self.recurrent_head.bias)
        # Shared pointwise ANN with NO phase-id and NO coordinate features -> the
        # ANN input is only phi_i (1 feature), so it is permutation-equivariant and
        # N-agnostic.
        self.ann = MultiPhasePointwiseANN(
            num_phases,
            ann_hidden_features,
            zero_initialize_head=zero_initialize_heads,
            coordinate_encoding="none",
            phase_id_encoding="none",
        )
        logit = torch.logit(torch.tensor(float(blend_gamma)))
        if learnable_blend:
            self.blend_logit = nn.Parameter(logit)
        else:
            self.register_buffer("blend_logit", logit)

        # Frozen, non-configurable descriptive attributes. These are not new
        # behavior or constructor parameters; they ensure trainer reads of
        # ``rate_parameterization``/``rate_head_bounded``/``max_delta_phi_per_step``/
        # ``centering_mode`` succeed with semantically TRUTHFUL values instead of
        # receive semantically accurate values rather than defensive defaults.
        #
        # ``rate_parameterization = "scaled_bounded"``: identical parameterization
        # family to the legacy model's ``scaled_bounded`` path -- delta_i =
        # delta_scale*tanh(z_i) -- NOT a new enum value (do not invent
        # "equivariant_tanh"; that label never existed as a real contract value, only
        # as a v1.1 getattr fallback default masking the missing attribute).
        #
        # ``rate_head_bounded = False``: there is no separate bounded-head branch here
        # (the legacy model's ``_scaled_bounded_rate``/``_bound_rate`` gate does not
        # exist in this module); the tanh bound is applied directly in
        # ``predict_rate``.
        #
        # ``max_delta_phi_per_step = float(delta_scale)``: an INACTIVE cap recorded
        # only for trainer-facing cap-saturation diagnostics/summaries. No code path
        # in this module clamps or renormalizes using this value -- ``delta_scale *
        # tanh(z)`` is already bounded in ``(-delta_scale, delta_scale)`` by
        # construction, so an external cap at the same magnitude could never bind.
        #
        # ``centering_mode = "phase_mean"`` documents the
        # ``delta_i -= mean_j(delta_j)`` step already performed in ``predict_rate``.
        self.rate_parameterization = "scaled_bounded"
        self.rate_head_bounded = False
        self.max_delta_phi_per_step = float(self.delta_scale)
        self.centering_mode = "phase_mean"
        # Recurrent-cell summary descriptor for an auditable model summary (the
        # underlying ``recurrent_cell`` module already exists; these two attributes
        # give trainer/report code a clean name+width pair without introspecting the
        # module tree).
        self.recurrent_cell_type = type(self.recurrent_cell).__name__
        self.recurrent_cell_name = "shared_convlstm_cell"
        self._validate_frozen_metadata_contract()

    def _validate_frozen_metadata_contract(self) -> None:
        """Check the frozen trainer-facing metadata contract at construction.

        This is a regression guard, not a configuration switch: none of the five
        values below are constructor parameters, so a future accidental edit that
        changes one of them without updating this check (and the accompanying
        doc comment + the frozen semantic-resolution note) fails fast at
        construction time instead of silently drifting.

        Explicit exceptions keep the contract active under optimized Python.
        """

        if self.rate_parameterization != "scaled_bounded":
            raise ValueError(
                "frozen contract violated: rate_parameterization must stay 'scaled_bounded'"
            )
        if self.rate_head_bounded is not False:
            raise ValueError(
                "frozen contract violated: rate_head_bounded must stay False (no bounded-head branch)"
            )
        if self.max_delta_phi_per_step != float(self.delta_scale):
            raise ValueError(
                "frozen contract violated: max_delta_phi_per_step must mirror delta_scale (inactive cap)"
            )
        if self.centering_mode != "phase_mean":
            raise ValueError(
                "frozen contract violated: centering_mode must stay 'phase_mean'"
            )
        if self.recurrent_cell_type != type(self.recurrent_cell).__name__:
            raise ValueError(
                "frozen contract violated: recurrent_cell_type must match the actual recurrent_cell class"
            )

    # ------------------------------------------------------------------ API --
    @property
    def gamma(self) -> Tensor:
        """Return the single global blend gamma in ``(0,1)``."""

        return torch.sigmoid(self.blend_logit)

    def _check_phi(self, phi: Tensor) -> None:
        _validate_phi(phi, self.num_phases if self.validate_num_phases else None)
        if phi.ndim != 4:
            raise ValueError("perm_equivariant_v1 supports only 2D phi [B,N,H,W]")

    def initial_state(self, phi: Tensor) -> MPFRecurrentState:
        """Return zero phasewise ``(h, c)`` state ``[B,N,C_h,H,W]``."""

        self._check_phi(phi)
        batch, phases, height, width = phi.shape
        zeros = torch.zeros(
            batch,
            phases,
            self.hidden_channels,
            height,
            width,
            device=phi.device,
            dtype=phi.dtype,
        )
        return zeros, zeros.clone()

    def _shared_recurrent(
        self, x: Tensor, state: tuple[Tensor, Tensor]
    ) -> tuple[Tensor, tuple[Tensor, Tensor]]:
        """Run the shared ConvLSTM over the phase axis via a ``[B*N,...]`` reshape."""

        batch, phases, channels, height, width = x.shape
        hidden, cell = state
        x_flat = x.reshape(batch * phases, channels, height, width)
        h_flat = hidden.reshape(batch * phases, self.hidden_channels, height, width)
        c_flat = cell.reshape(batch * phases, self.hidden_channels, height, width)
        h_new, c_new = self.recurrent_cell(x_flat, h_flat, c_flat)
        r_rnn = self.recurrent_head(h_new)  # [B*N,1,H,W]
        r_rnn = r_rnn.reshape(batch, phases, height, width)
        h_new = h_new.reshape(batch, phases, self.hidden_channels, height, width)
        c_new = c_new.reshape(batch, phases, self.hidden_channels, height, width)
        return r_rnn, (h_new, c_new)

    def branch_rates(
        self, phi: Tensor, state: MPFRecurrentState
    ) -> tuple[Tensor, Tensor, MPFRecurrentState]:
        """Return ANN rate, ConvLSTM rate, and next phasewise recurrent state."""

        self._check_phi(phi)
        if not isinstance(state, tuple):
            raise TypeError("perm_equivariant_v1 recurrent state must be an (h, c) tuple")
        batch, phases, height, width = phi.shape
        # Aggregation features over ALL phases (symmetric mean and amax).
        e = self.encoder(phi)  # [B,N,C_e,H,W]
        a_mean = e.mean(dim=1, keepdim=True)  # [B,1,C_e,H,W]
        a_max = e.amax(dim=1, keepdim=True)  # [B,1,C_e,H,W] (symmetric-subgradient)
        a_mean = a_mean.expand(batch, phases, self.encoder_channels, height, width)
        a_max = a_max.expand(batch, phases, self.encoder_channels, height, width)
        phi_c = phi.unsqueeze(2)  # [B,N,1,H,W]
        x = torch.cat((phi_c, e, a_mean, a_max), dim=2)  # [B,N,1+3C_e,H,W]
        rnn_rate, next_state = self._shared_recurrent(x, state)
        ann_rate = self.ann(phi)  # [B,N,H,W]
        return ann_rate, rnn_rate, next_state

    def predict_rate(
        self,
        phi: Tensor,
        state: MPFRecurrentState,
        *,
        previous_phi: Tensor | None = None,  # accepted for API parity; unused (no graph)
    ) -> tuple[Tensor, MPFRecurrentState, dict[str, object]]:
        """Return the blended, phase-mean-centered rate plus diagnostics."""

        ann_rate, rnn_rate, next_state = self.branch_rates(phi, state)
        gamma = self.gamma
        z = gamma * ann_rate + (1.0 - gamma) * rnn_rate
        delta = self.delta_scale * torch.tanh(z)
        delta = delta - delta.mean(dim=1, keepdim=True)  # phase_mean centering
        rate = delta / self.model_dt
        diagnostics: dict[str, object] = {
            "_pre_bound_rate": z,
            "_ann_rate": ann_rate,
            "_rnn_rate": rnn_rate,
            "gamma": float(torch.mean(gamma).detach()),
            "ann_rate_norm": float(torch.linalg.vector_norm(ann_rate).detach()),
            "rnn_rate_norm": float(torch.linalg.vector_norm(rnn_rate).detach()),
            # ANN/RNN contribution ratio and branch-collapse monitoring use the
            # same definitions as the legacy model.
            "ann_rnn_contribution_ratio": float(
                (
                    torch.linalg.vector_norm(gamma * ann_rate)
                    / torch.linalg.vector_norm((1.0 - gamma) * rnn_rate).clamp_min(1.0e-12)
                ).detach()
            ),
            "branch_collapse_min_norm": float(
                torch.minimum(
                    torch.linalg.vector_norm(ann_rate), torch.linalg.vector_norm(rnn_rate)
                ).detach()
            ),
            "arch_variant": self.arch_variant,
            "graph_enabled": 0.0,
            "graph_mode": "disabled",
            "graph_backend": "none",
            "rate_change_due_to_graph": 0.0,
        }
        return rate, next_state, diagnostics

    def mpf_residual(self, phi: Tensor, model_rate: Tensor) -> Tensor:
        """Return the synchronized MPF residual for the model's own field."""

        from pinn_phase.physics.explicit_mpf import explicit_mpf_rhs

        return model_rate - explicit_mpf_rhs(
            phi, eta_px=self.eta_px, mu=self.mu, sigma=self.sigma
        )

    def forward_step(
        self,
        phi: Tensor,
        state: MPFRecurrentState | None = None,
        *,
        previous_phi: Tensor | None = None,
    ) -> tuple[Tensor, MPFRecurrentState, Tensor, dict[str, object]]:
        """Advance one explicit projected MPF step."""

        if state is None:
            state = self.initial_state(phi)
        rate, next_state, diagnostics = self.predict_rate(
            phi, state, previous_phi=previous_phi
        )
        raw_next = phi + self.model_dt * rate
        if self.projection_mode == "soft_threshold_eps1e3":
            next_phi, projection_diagnostics = project_simplex_soft_threshold(
                raw_next, threshold_eps=1.0e-3
            )
        else:
            next_phi, projection_diagnostics = project_simplex(raw_next)
        for key, value in projection_diagnostics.items():
            diagnostics[key] = value
        diagnostics["phase_sum_error_after_step"] = diagnostics.get(
            "sum_error_post_projection", 0.0
        )
        return next_phi, next_state, rate, diagnostics

    def rollout(self, phi0: Tensor, *, steps: int) -> Tensor:
        """Return a rollout stack including the initial state."""

        if steps < 1:
            raise ValueError("steps must be positive")
        phi = phi0
        previous_phi: Tensor | None = None
        state = self.initial_state(phi)
        states = [phi]
        for _ in range(steps):
            next_phi, state, _, _ = self.forward_step(phi, state, previous_phi=previous_phi)
            previous_phi = phi
            phi = next_phi
            states.append(phi)
        return torch.stack(states)
