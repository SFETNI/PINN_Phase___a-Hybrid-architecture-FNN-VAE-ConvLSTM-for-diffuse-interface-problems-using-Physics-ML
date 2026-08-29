"""Explicit multichannel MPF ANN plus ConvLSTM rollout and graph conditioning."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
import math
from time import perf_counter

import torch
import torch.nn as nn
from torch import Tensor

from pinn_phase.physics.explicit_mpf import (
    explicit_mpf_rhs,
    project_simplex,
    project_simplex_soft_threshold,
)

from .cells import ConvGRUCell, ConvGRUCell3d, ConvLSTMCell, ConvLSTMCell3d


MPFRecurrentState = Tensor | tuple[Tensor, Tensor]


@dataclass(frozen=True)
class ExplicitMPFGraph:
    """Batchable dense graph representation extracted from model ``Phi``."""

    node_features: Tensor
    node_mask: Tensor
    edge_features: Tensor
    edge_mask: Tensor
    diagnostics: dict[str, float]


def detach_mpf_recurrent_state(state: MPFRecurrentState) -> MPFRecurrentState:
    """Detach a ConvGRU tensor or ConvLSTM ``(hidden, cell)`` tuple."""

    if isinstance(state, tuple):
        return tuple(value.detach() for value in state)
    return state.detach()


class MultiPhasePointwiseANN(nn.Module):
    """Per-pixel local ANN branch for explicit phase stacks ``[B,N,H,W]``."""

    def __init__(
        self,
        num_phases: int,
        hidden_features: Sequence[int] = (32, 32),
        *,
        zero_initialize_head: bool = True,
        coordinate_encoding: str = "periodic",
        phase_id_encoding: str = "normalized",
    ) -> None:
        super().__init__()
        if num_phases < 2:
            raise ValueError("num_phases must be at least 2")
        if not isinstance(phase_id_encoding, str) or phase_id_encoding not in {"normalized", "none", "zero_masked"}:
            raise ValueError(
                "phase_id_encoding must be 'normalized', 'none', or 'zero_masked'; "
                f"got {phase_id_encoding!r}"
            )
        _COORD_FEATURES = {"none": 0, "raw": 2, "periodic": 4, "raw_3d": 3, "periodic_3d": 6}
        if coordinate_encoding not in _COORD_FEATURES:
            raise ValueError(
                f"coordinate_encoding must be one of {sorted(_COORD_FEATURES)}, "
                f"got {coordinate_encoding!r}"
            )
        widths = tuple(int(width) for width in hidden_features)
        if not widths or any(width < 1 for width in widths):
            raise ValueError("hidden_features must contain positive widths")
        self.num_phases = int(num_phases)
        self.coordinate_encoding = coordinate_encoding
        self.phase_id_encoding = phase_id_encoding
        coord_features = _COORD_FEATURES[coordinate_encoding]
        phase_id_features = 1 if phase_id_encoding in {"normalized", "zero_masked"} else 0
        in_features = 1 + phase_id_features + coord_features  # phi_i, phase_id, coords
        layers: list[nn.Module] = []
        for width in widths:
            layers.extend((nn.Linear(in_features, width), nn.Tanh()))
            in_features = width
        self.hidden = nn.Sequential(*layers)
        self.head = nn.Linear(in_features, 1)
        if zero_initialize_head:
            nn.init.zeros_(self.head.weight)
            nn.init.zeros_(self.head.bias)

    def forward(self, phi: Tensor) -> Tensor:
        """Return a local signed rate shaped like ``phi``."""

        _validate_phi(phi, self.num_phases)
        batch, phases, *spatial = phi.shape
        n_spatial = math.prod(spatial)
        n_ones = len(spatial)
        features = [phi.unsqueeze(-1)]
        if self.phase_id_encoding == "normalized":
            phase_ids = torch.linspace(
                0.0,
                1.0,
                phases,
                device=phi.device,
                dtype=phi.dtype,
            ).view(1, phases, *([1] * n_ones), 1)
            features.append(phase_ids.expand(batch, -1, *spatial, -1))
        elif self.phase_id_encoding == "zero_masked":
            # matched-init mode: keep the phase-id feature COLUMN (identical in_features/
            # RNG draw to 'normalized') but zero its content so channel identity carries
            # no signal -> permutation-equivariant ANN input.
            zeros = phi.new_zeros(1, phases, *([1] * n_ones), 1)
            features.append(zeros.expand(batch, -1, *spatial, -1))
        if self.coordinate_encoding != "none":
            coords = _coordinate_features(phi, spatial, self.coordinate_encoding)
            coords = coords.view(1, 1, *spatial, -1).expand(batch, phases, *spatial, -1)
            features.append(coords)
        flat = torch.cat(features, dim=-1).reshape(batch * phases * n_spatial, -1)
        rate = self.head(self.hidden(flat))
        return rate.reshape(batch, phases, *spatial)


class ExplicitMPFGraphExtractor(nn.Module):
    """Extract grain/phase graph features from model ``Phi`` only."""

    node_feature_dim = 12
    edge_feature_dim = 4

    def __init__(self, *, periodic: bool = True, active_threshold: int = 1) -> None:
        super().__init__()
        self.periodic = bool(periodic)
        self.active_threshold = int(active_threshold)

    def forward(self, phi: Tensor, previous_phi: Tensor | None = None) -> ExplicitMPFGraph:
        """Build dense node/edge features from the current model state."""

        _validate_phi(phi)
        if previous_phi is not None and previous_phi.shape != phi.shape:
            raise ValueError("previous_phi must match phi shape")
        started = perf_counter()
        with torch.no_grad():
            graph = self._extract(phi.detach(), None if previous_phi is None else previous_phi.detach())
        graph.diagnostics["graph_extraction_time"] = perf_counter() - started
        return graph

    def _extract(self, phi: Tensor, previous_phi: Tensor | None) -> ExplicitMPFGraph:
        batch, phases, *spatial = phi.shape
        spatial_dims = len(spatial)
        if spatial_dims not in {2, 3}:
            raise ValueError("ExplicitMPFGraphExtractor supports 2D or 3D phi tensors")
        spatial_volume = math.prod(spatial)
        contact_denominator = max(float(spatial_dims * spatial_volume), 1.0)
        labels = torch.argmax(phi, dim=1)
        prev_centroids = (
            _soft_centroids(previous_phi) if previous_phi is not None else None
        )
        centroids = _soft_centroids(phi)
        node_features = torch.zeros(
            batch,
            phases,
            self.node_feature_dim,
            device=phi.device,
            dtype=phi.dtype,
        )
        edge_features = torch.zeros(
            batch,
            phases,
            phases,
            self.edge_feature_dim,
            device=phi.device,
            dtype=phi.dtype,
        )
        edge_mask = torch.zeros(batch, phases, phases, device=phi.device, dtype=torch.bool)
        node_mask = torch.zeros(batch, phases, device=phi.device, dtype=torch.bool)
        junction_counts = []
        graph_changes = []
        elimination_events = []

        contact_offsets = _axis_contact_offsets(spatial_dims)
        neighborhood_offsets = _neighborhood_offsets(spatial_dims)
        for b in range(batch):
            label = labels[b]
            measure = torch.bincount(label.reshape(-1), minlength=phases).to(phi.dtype)
            active = measure >= self.active_threshold
            node_mask[b] = active

            contact = torch.zeros(phases, phases, device=phi.device, dtype=phi.dtype)
            for offset in contact_offsets:
                shifted = _shift_nd(label, offset, self.periodic)
                mask = label != shifted
                if torch.any(mask):
                    a = label[mask].reshape(-1)
                    c = shifted[mask].reshape(-1)
                    contact.index_put_((a, c), torch.ones_like(a, dtype=phi.dtype), accumulate=True)
                    contact.index_put_((c, a), torch.ones_like(a, dtype=phi.dtype), accumulate=True)
            adjacency = contact > 0
            adjacency.fill_diagonal_(False)
            edge_mask[b] = adjacency

            neighborhood = torch.stack(
                [_shift_nd(label, offset, self.periodic) for offset in neighborhood_offsets],
                dim=0,
            )
            # A pixel/voxel is a diagnostic triple-junction site when its local
            # 3^D label neighborhood contains >=3 distinct phase labels.
            presence = torch.zeros(phases, *spatial, device=phi.device, dtype=torch.bool)
            presence.scatter_(0, neighborhood, torch.ones_like(neighborhood, dtype=torch.bool))
            distinct_per_pixel = presence.sum(dim=0)
            local_unique = distinct_per_pixel >= 3
            junction_count = int(torch.count_nonzero(local_unique).item())
            junction_counts.append(junction_count)

            neighbor_count = adjacency.sum(dim=1).to(phi.dtype)
            contact_measure = contact.sum(dim=1)
            flat_phi = phi[b].flatten(1)
            mean_phi = flat_phi.mean(dim=1)
            max_phi = flat_phi.amax(dim=1)
            motion = torch.zeros(phases, spatial_dims, device=phi.device, dtype=phi.dtype)
            if prev_centroids is not None:
                motion = _periodic_delta(prev_centroids[b], centroids[b], *spatial)
            phase_ids = torch.linspace(0, 1, phases, device=phi.device, dtype=phi.dtype)
            node_features[b, :, 0] = phase_ids
            node_features[b, :, 1] = measure / max(float(spatial_volume), 1.0)
            node_features[b, :, 2] = centroids[b, :, 0] / max(float(spatial[0]), 1.0)
            node_features[b, :, 3] = centroids[b, :, 1] / max(float(spatial[1]), 1.0)
            if spatial_dims == 2:
                node_features[b, :, 4] = contact_measure / contact_denominator
            else:
                node_features[b, :, 4] = centroids[b, :, 2] / max(float(spatial[2]), 1.0)
            node_features[b, :, 5] = active.to(phi.dtype)
            node_features[b, :, 6] = mean_phi
            node_features[b, :, 7] = max_phi
            node_features[b, :, 8] = neighbor_count / max(float(phases - 1), 1.0)
            node_features[b, :, 9] = float(junction_count) / max(float(spatial_volume), 1.0)
            if spatial_dims == 2:
                node_features[b, :, 10:12] = motion
            else:
                node_features[b, :, 10] = contact_measure / contact_denominator
                node_features[b, :, 11] = torch.linalg.vector_norm(motion, dim=1)

            # Vectorized edge features: replaces O(N²) Python loop with batched
            # tensor ops.  Math is identical to the per-pair _periodic_delta calls
            # above — pairwise delta[i,j] = centroids[j] - centroids[i], wrapped
            # and normalized by spatial size exactly as _periodic_delta does.
            _c = centroids[b]  # (N, D)
            _scale = torch.tensor(
                [float(s) for s in spatial], device=phi.device, dtype=phi.dtype
            )
            _half = _scale / 2.0
            _delta_all = _c.unsqueeze(0) - _c.unsqueeze(1)  # (N, N, D): delta[i,j]=c[j]-c[i]
            _delta_all = torch.where(_delta_all > _half, _delta_all - _scale, _delta_all)
            _delta_all = torch.where(_delta_all < -_half, _delta_all + _scale, _delta_all)
            _delta_all = _delta_all / _scale.clamp_min(1.0)  # normalize → [-0.5, 0.5]
            _dist_all = torch.sqrt((_delta_all ** 2).sum(dim=-1))  # (N, N)
            edge_features[b, :, :, 0] = contact / contact_denominator
            edge_features[b, :, :, 1] = _dist_all / math.sqrt(float(spatial_dims))
            edge_features[b, :, :, 2] = _delta_all[..., 0]
            edge_features[b, :, :, 3] = _delta_all[..., 1]

            if previous_phi is not None:
                prev_labels = torch.argmax(previous_phi[b], dim=0)
                prev_measure = torch.bincount(prev_labels.reshape(-1), minlength=phases)
                graph_changes.append(float(torch.count_nonzero((prev_measure > 0) & (~active)).item()))
                elimination_events.append(float(torch.count_nonzero((prev_measure > 0) & (~active)).item()))

        diagnostics = {
            "node_count": float(torch.count_nonzero(node_mask).item()),
            "edge_count": float(torch.count_nonzero(edge_mask).item()),
            "active_phase_count": float(torch.mean(node_mask.sum(dim=1).to(phi.dtype)).item()),
            "junction_count": float(sum(junction_counts) / max(len(junction_counts), 1)),
            "neighbor_graph_changes": float(sum(graph_changes)),
            "grain_elimination_events": float(sum(elimination_events)),
        }
        return ExplicitMPFGraph(
            node_features=node_features,
            node_mask=node_mask,
            edge_features=edge_features,
            edge_mask=edge_mask,
            diagnostics=diagnostics,
        )


class SimpleGraphConditioner(nn.Module):
    """Small torch-only graph message-passing module for phase-wise modulation."""

    def __init__(
        self,
        *,
        node_feature_dim: int,
        edge_feature_dim: int,
        hidden_dim: int = 32,
        layers: int = 1,
        mode: str = "simple_gnn",
        zero_initialize_output: bool = True,
        conditioning_scale: float = 1.0,
        center_graph_bias: bool = False,
    ) -> None:
        super().__init__()
        if hidden_dim < 1 or layers < 1:
            raise ValueError("hidden_dim and layers must be positive")
        if mode != "simple_gnn":
            raise ValueError("graph mode must be simple_gnn")
        if conditioning_scale <= 0.0:
            raise ValueError("conditioning_scale must be positive")
        self.mode = mode
        self.layers = int(layers)
        self.conditioning_scale = float(conditioning_scale)
        self.center_graph_bias = bool(center_graph_bias)
        self.node_proj = nn.Linear(node_feature_dim, hidden_dim)
        self.edge_proj = nn.Linear(edge_feature_dim, hidden_dim)
        self.message = nn.ModuleList(nn.Linear(hidden_dim, hidden_dim) for _ in range(layers))
        self.update = nn.ModuleList(nn.Linear(2 * hidden_dim, hidden_dim) for _ in range(layers))
        self.out = nn.Linear(hidden_dim, 2)
        if zero_initialize_output:
            nn.init.zeros_(self.out.weight)
            nn.init.zeros_(self.out.bias)

    def forward(self, head: Tensor, graph: ExplicitMPFGraph) -> tuple[Tensor, dict[str, float]]:
        """Apply phase-wise graph scale/bias to an unbounded head tensor."""

        started = perf_counter()
        hidden = torch.tanh(self.node_proj(graph.node_features))
        hidden = hidden * graph.node_mask.unsqueeze(-1).to(hidden.dtype)
        edge_hidden = torch.tanh(self.edge_proj(graph.edge_features))
        edge_mask_f = graph.edge_mask.unsqueeze(-1).to(hidden.dtype)
        for layer_idx in range(self.layers):
            src = hidden.unsqueeze(1).expand(-1, hidden.shape[1], -1, -1)
            messages = torch.tanh(self.message[layer_idx](src + edge_hidden)) * edge_mask_f
            degree = graph.edge_mask.sum(dim=2, keepdim=True).clamp_min(1).to(hidden.dtype)
            pooled = messages.sum(dim=2) / degree
            hidden = torch.tanh(self.update[layer_idx](torch.cat((hidden, pooled), dim=-1)))
            hidden = hidden * graph.node_mask.unsqueeze(-1).to(hidden.dtype)
        scale_bias = self.out(hidden)
        spatial_view = (scale_bias.shape[0], scale_bias.shape[1], *([1] * (head.ndim - 2)))
        scale = 0.1 * self.conditioning_scale * torch.tanh(scale_bias[..., 0]).view(spatial_view)
        bias = 0.1 * self.conditioning_scale * torch.tanh(scale_bias[..., 1]).view(spatial_view)
        bias_common_mode = bias.mean(dim=1, keepdim=True)
        bias_centered_abs_mean = torch.mean(torch.abs(bias - bias_common_mode))
        if self.center_graph_bias:
            bias = bias - bias_common_mode
        conditioned = head * (1.0 + scale) + bias
        diagnostics = {
            "graph_hidden_norm": float(torch.linalg.vector_norm(hidden).detach()),
            "graph_head_conditioning_norm": float(torch.linalg.vector_norm(conditioned - head).detach()),
            "head_change_due_to_graph": float(torch.mean(torch.abs(conditioned - head)).detach()),
            "graph_scale_abs_mean": float(torch.mean(torch.abs(scale)).detach()),
            "graph_scale_abs_max": float(torch.amax(torch.abs(scale)).detach()),
            "graph_bias_abs_mean": float(torch.mean(torch.abs(bias)).detach()),
            "graph_bias_abs_max": float(torch.amax(torch.abs(bias)).detach()),
            "graph_bias_common_mode_abs_mean": float(torch.mean(torch.abs(bias_common_mode)).detach()),
            "graph_bias_phase_centered_abs_mean": float(bias_centered_abs_mean.detach()),
            "graph_conditioning_scale": self.conditioning_scale,
            "graph_center_bias": float(self.center_graph_bias),
            "graph_forward_time": perf_counter() - started,
        }
        return conditioned, diagnostics


class ExplicitMPFHybridRollout(nn.Module):
    """ANN local branch + ConvLSTM spatial branch + optional graph conditioning."""

    def __init__(
        self,
        *,
        num_phases: int,
        model_dt: float,
        eta_px: float,
        mu: float = 1.0,
        sigma: float = 1.0,
        spatial_dims: int = 2,
        recurrent_cell: str = "convlstm",
        hidden_channels: int = 32,
        kernel_size: int = 3,
        ann_hidden_features: Sequence[int] = (32, 32),
        learnable_blend: bool = True,
        blend_gamma: float = 0.5,
        per_phase_gamma: bool = False,
        projection_mode: str = "simplex_clip_normalize",
        graph_mode: str = "disabled",
        graph_hidden_dim: int = 32,
        graph_layers: int = 1,
        graph_conditioning_scale: float = 1.0,
        graph_center_bias: bool = False,
        zero_initialize_heads: bool = True,
        coordinate_encoding: str = "periodic",
        phase_id_encoding: str = "normalized",
        conv_padding_mode: str = "circular",
        rate_head_bounded: bool = False,
        max_delta_phi_per_step: float = 0.05,
        rate_parameterization: str = "legacy_tanh_cap",
        delta_scale: float = 0.10,
        centering_mode: str = "phase_mean",
    ) -> None:
        super().__init__()
        if num_phases < 2:
            raise ValueError("num_phases must be at least 2")
        if not isinstance(phase_id_encoding, str) or phase_id_encoding not in {"normalized", "none", "zero_masked"}:
            raise ValueError(
                "phase_id_encoding must be 'normalized', 'none', or 'zero_masked'; "
                f"got {phase_id_encoding!r}"
            )
        if centering_mode not in {"phase_mean", "producer_pays"}:
            raise ValueError(
                "centering_mode must be 'phase_mean' (legacy) or 'producer_pays' (P2-exact); "
                f"got {centering_mode!r}"
            )
        if model_dt <= 0.0:
            raise ValueError("model_dt must be positive")
        if rate_parameterization not in {"legacy_tanh_cap", "scaled_bounded"}:
            raise ValueError("rate_parameterization must be legacy_tanh_cap or scaled_bounded")
        if rate_parameterization == "scaled_bounded" and delta_scale <= 0.0:
            raise ValueError("delta_scale must be positive for scaled_bounded")
        if rate_head_bounded and max_delta_phi_per_step <= 0.0:
            raise ValueError("max_delta_phi_per_step must be positive when rate_head_bounded")
        if recurrent_cell not in {"convlstm", "convgru"}:
            raise ValueError("recurrent_cell must be convlstm or convgru")
        if projection_mode not in {"simplex_clip_normalize", "soft_threshold_eps1e3"}:
            raise ValueError(
                "projection_mode must be 'simplex_clip_normalize' or 'soft_threshold_eps1e3'"
            )
        if graph_mode not in {"disabled", "none", "simple_gnn"}:
            raise ValueError("unsupported graph_mode")
        if graph_conditioning_scale <= 0.0:
            raise ValueError("graph_conditioning_scale must be positive")
        if spatial_dims not in {2, 3}:
            raise ValueError("spatial_dims must be 2 or 3")
        _2D_ENCODINGS = {"none", "raw", "periodic"}
        _3D_ENCODINGS = {"none", "raw_3d", "periodic_3d"}
        _valid_encodings = _2D_ENCODINGS if spatial_dims == 2 else _3D_ENCODINGS
        if coordinate_encoding not in _valid_encodings:
            raise ValueError(
                f"coordinate_encoding {coordinate_encoding!r} is incompatible with "
                f"spatial_dims={spatial_dims}; valid: {sorted(_valid_encodings)}"
            )
        self.num_phases = int(num_phases)
        self.model_dt = float(model_dt)
        self.eta_px = float(eta_px)
        self.mu = float(mu)
        self.sigma = float(sigma)
        self.spatial_dims = int(spatial_dims)
        self.recurrent_cell_name = recurrent_cell
        self.projection_mode = projection_mode
        self.graph_mode = "disabled" if graph_mode == "none" else graph_mode
        self.graph_enabled = self.graph_mode != "disabled"
        self.graph_conditioning_scale = float(graph_conditioning_scale)
        self.graph_center_bias = bool(graph_center_bias)
        self.per_phase_gamma = bool(per_phase_gamma)
        self.rate_head_bounded = bool(rate_head_bounded)
        self.max_delta_phi_per_step = float(max_delta_phi_per_step)
        self.rate_parameterization = str(rate_parameterization)
        self.delta_scale = float(delta_scale)
        self.phase_id_encoding = phase_id_encoding
        self.centering_mode = centering_mode

        self.ann = MultiPhasePointwiseANN(
            num_phases,
            ann_hidden_features,
            zero_initialize_head=zero_initialize_heads,
            coordinate_encoding=coordinate_encoding,
            phase_id_encoding=phase_id_encoding,
        )
        if self.spatial_dims == 3:
            cell_type_3d = ConvLSTMCell3d if recurrent_cell == "convlstm" else ConvGRUCell3d
            self.recurrent_cell = cell_type_3d(
                in_channels=num_phases,
                hidden_channels=hidden_channels,
                kernel_size=kernel_size,
                padding_mode=conv_padding_mode,
            )
            self.recurrent_head = nn.Conv3d(
                hidden_channels,
                num_phases,
                kernel_size=kernel_size,
                padding=kernel_size // 2,
                padding_mode=conv_padding_mode,
            )
        else:
            cell_type_2d = ConvLSTMCell if recurrent_cell == "convlstm" else ConvGRUCell
            self.recurrent_cell = cell_type_2d(
                in_channels=num_phases,
                hidden_channels=hidden_channels,
                kernel_size=kernel_size,
                padding_mode=conv_padding_mode,
            )
            self.recurrent_head = nn.Conv2d(
                hidden_channels,
                num_phases,
                kernel_size=kernel_size,
                padding=kernel_size // 2,
                padding_mode=conv_padding_mode,
            )
        if zero_initialize_heads:
            nn.init.zeros_(self.recurrent_head.weight)
            nn.init.zeros_(self.recurrent_head.bias)

        logit = torch.logit(torch.tensor(float(blend_gamma)))
        if per_phase_gamma:
            init = torch.full((num_phases,), float(logit))
        else:
            init = torch.tensor(float(logit))
        if learnable_blend:
            self.blend_logit = nn.Parameter(init)
        else:
            self.register_buffer("blend_logit", init)

        self.graph_extractor = ExplicitMPFGraphExtractor(periodic=True)
        self.graph_conditioner = (
            SimpleGraphConditioner(
                node_feature_dim=ExplicitMPFGraphExtractor.node_feature_dim,
                edge_feature_dim=ExplicitMPFGraphExtractor.edge_feature_dim,
                hidden_dim=graph_hidden_dim,
                layers=graph_layers,
                mode=self.graph_mode,
                conditioning_scale=self.graph_conditioning_scale,
                center_graph_bias=self.graph_center_bias,
            )
            if self.graph_enabled
            else None
        )

    @property
    def gamma(self) -> Tensor:
        """Return global or per-phase blend gamma in ``(0,1)``."""

        return torch.sigmoid(self.blend_logit)

    def initial_state(self, phi: Tensor) -> MPFRecurrentState:
        """Return zero recurrent state for ``phi``."""

        _validate_phi(phi, self.num_phases)
        batch, _, *spatial = phi.shape
        return self.recurrent_cell.initial_state(batch, *spatial, device=phi.device)

    def _recurrent_rate(self, phi: Tensor, state: MPFRecurrentState) -> tuple[Tensor, MPFRecurrentState]:
        if self.recurrent_cell_name == "convlstm":
            if not isinstance(state, tuple):
                raise TypeError("ConvLSTM state must be a tuple")
            hidden, cell = state
            next_hidden, next_cell = self.recurrent_cell(phi, hidden, cell)
            return self.recurrent_head(next_hidden), (next_hidden, next_cell)
        if isinstance(state, tuple):
            raise TypeError("ConvGRU state must be a tensor")
        next_hidden = self.recurrent_cell(phi, state)
        return self.recurrent_head(next_hidden), next_hidden

    def branch_rates(
        self,
        phi: Tensor,
        state: MPFRecurrentState,
    ) -> tuple[Tensor, Tensor, MPFRecurrentState]:
        """Return ANN and ConvLSTM rates plus next recurrent state."""

        _validate_phi(phi, self.num_phases)
        ann_rate = self.ann(phi)
        rnn_rate, next_state = self._recurrent_rate(phi, state)
        return ann_rate, rnn_rate, next_state

    def _bound_rate(self, rate: Tensor) -> Tensor:
        """Optionally cap the per-step displacement ``model_dt * rate``.

        When ``rate_head_bounded`` is set the blended rate is passed through a
        smooth saturating limiter so the effective per-step update cannot exceed
        ``max_delta_phi_per_step`` in magnitude:

            delta = max_delta_phi_per_step * tanh(model_dt * rate / max_delta_phi_per_step)
            rate  = delta / model_dt

        For small rates ``tanh(x) ~ x`` so this is ~identity near the physical
        scale; for large rates it saturates, preventing raw_next excursions like
        ``[-50, +35]`` that the projection would otherwise mask.  Projection then
        acts purely as a safety/constraint operator, not the dynamics engine.
        """

        if not self.rate_head_bounded:
            return rate
        cap = self.max_delta_phi_per_step
        delta = cap * torch.tanh(self.model_dt * rate / cap)
        return delta / self.model_dt

    def _scaled_bounded_rate(self, head_logits: Tensor) -> Tensor:
        """V5-A physically-scaled bounded-update head.

        The blended head output is treated as pre-squash *logits*.  A single tanh
        at the source maps them into ``(-1, 1)``, scaled by the fixed physical
        ``delta_scale`` to a bounded per-step displacement, mean-removed across the
        phase dim so the simplex sum is preserved (mirrors the physics RHS
        mean-removal), then divided by ``model_dt`` to a rate:

            g     = tanh(head_logits)
            delta = delta_scale * g
            delta = delta - delta.mean(dim=1, keepdim=True)
            rate  = delta / model_dt

        The maximum possible pre-centering per-channel displacement is
        ``delta_scale`` by construction.  After phase-mean centering, the max
        absolute physical displacement is ``2*(N-1)/N*delta_scale`` (1.5x for
        four phases), so a >>cap excursion cannot occur.  Zero-init heads give
        ``tanh(0)=0`` so the step is inert at initialization, as in the legacy
        path.
        """

        g = torch.tanh(head_logits)
        delta = self.delta_scale * g
        if self.centering_mode == "producer_pays":
            # P2-exact (D_PRE_P2_EXACT_ADDENDUM): positive contributors absorb the common
            # mode S; UNIFORM 1/N fallback where every channel is non-positive so Sum(w)=1
            # EXACTLY in both branches (exact zero-sum) and gradients stay finite.
            S = delta.sum(dim=1, keepdim=True)
            pos = torch.relu(delta)
            P = pos.sum(dim=1, keepdim=True)
            Pd = torch.where(P > 0, P, torch.ones_like(P))
            w = torch.where(P > 0, pos / Pd, torch.full_like(pos, 1.0 / self.num_phases))
            delta = delta - w * S
        else:  # "phase_mean" = legacy uniform centering (byte-identical to C0)
            delta = delta - delta.mean(dim=1, keepdim=True)
        return delta / self.model_dt

    def _scaled_bounded_graph_materiality_diagnostics(
        self,
        base_head: Tensor,
        conditioned_head: Tensor,
    ) -> dict[str, float]:
        """Measure graph effect before projection and across rate centering."""

        base_delta = self.delta_scale * torch.tanh(base_head)
        conditioned_delta = self.delta_scale * torch.tanh(conditioned_head)
        uncentered_delta_change = torch.mean(torch.abs(conditioned_delta - base_delta))
        base_centered = base_delta - base_delta.mean(dim=1, keepdim=True)
        conditioned_centered = conditioned_delta - conditioned_delta.mean(dim=1, keepdim=True)
        centered_delta_change = torch.mean(torch.abs(conditioned_centered - base_centered))
        uncentered_rate_change = uncentered_delta_change / self.model_dt
        centered_rate_change = centered_delta_change / self.model_dt
        centering_ratio = centered_delta_change / uncentered_delta_change.clamp_min(1.0e-30)
        return {
            "graph_delta_uncentered_change": float(uncentered_delta_change.detach()),
            "graph_delta_centered_change": float(centered_delta_change.detach()),
            "graph_rate_uncentered_change": float(uncentered_rate_change.detach()),
            "graph_rate_centered_change": float(centered_rate_change.detach()),
            "graph_phase_centering_retained_ratio": float(centering_ratio.detach()),
            "graph_rate_shrinkage_factor": float(self.delta_scale / self.model_dt),
        }

    def predict_rate(
        self,
        phi: Tensor,
        state: MPFRecurrentState,
        *,
        previous_phi: Tensor | None = None,
    ) -> tuple[Tensor, MPFRecurrentState, dict[str, object]]:
        """Return blended signed rate and diagnostics."""

        ann_rate, rnn_rate, next_state = self.branch_rates(phi, state)
        gamma = self.gamma
        gamma_view = (
            gamma.view(1, self.num_phases, *([1] * self.spatial_dims))
            if self.per_phase_gamma
            else gamma
        )
        pre_bound_rate = gamma_view * ann_rate + (1.0 - gamma_view) * rnn_rate
        graph_diag: dict[str, object] | None = None
        graph_base_rate: Tensor | None = None
        graph_base_head: Tensor | None = None
        if self.graph_conditioner is not None:
            if self.rate_parameterization == "scaled_bounded":
                graph_base_rate = self._scaled_bounded_rate(pre_bound_rate)
            else:
                graph_base_rate = self._bound_rate(pre_bound_rate)
            graph_base_head = pre_bound_rate
            graph = self.graph_extractor(phi, previous_phi)
            pre_bound_rate, graph_diag = self.graph_conditioner(pre_bound_rate, graph)
            graph_diag.update(graph.diagnostics)
        if self.rate_parameterization == "scaled_bounded":
            # The blend is the head logits; the structural bound replaces _bound_rate.
            rate = self._scaled_bounded_rate(pre_bound_rate)
        else:
            rate = self._bound_rate(pre_bound_rate)
        diagnostics: dict[str, object] = {
            # Grad-carrying tensors for optional V3 regularizers (NOT serialized;
            # the trainer pops these before writing the JSON review row).
            "_pre_bound_rate": pre_bound_rate,
            "_ann_rate": ann_rate,
            "_rnn_rate": rnn_rate,
            "gamma": float(torch.mean(gamma).detach()),
            "ann_rate_norm": float(torch.linalg.vector_norm(ann_rate).detach()),
            "rnn_rate_norm": float(torch.linalg.vector_norm(rnn_rate).detach()),
            "ann_rnn_contribution_ratio": float(
                (torch.linalg.vector_norm(gamma_view * ann_rate) / torch.linalg.vector_norm((1.0 - gamma_view) * rnn_rate).clamp_min(1.0e-12)).detach()
            ),
            "branch_collapse_min_norm": float(
                torch.minimum(torch.linalg.vector_norm(ann_rate), torch.linalg.vector_norm(rnn_rate)).detach()
            ),
            "graph_enabled": float(self.graph_enabled),
            "graph_mode": self.graph_mode,
            "graph_backend": "torch_only" if self.graph_enabled else "none",
            "graph_mode_simple_gnn": float(self.graph_mode == "simple_gnn"),
        }
        if graph_diag is not None:
            assert graph_base_rate is not None
            assert graph_base_head is not None
            graph_diag["graph_conditioning_norm"] = float(
                torch.linalg.vector_norm(rate - graph_base_rate).detach()
            )
            graph_diag["rate_change_due_to_graph"] = float(
                torch.mean(torch.abs(rate - graph_base_rate)).detach()
            )
            if self.rate_parameterization == "scaled_bounded":
                graph_diag.update(
                    self._scaled_bounded_graph_materiality_diagnostics(
                        graph_base_head,
                        pre_bound_rate,
                    )
                )
            graph_diag["graph_conditioning_stage"] = "pre_bounded_head"
            diagnostics.update(graph_diag)
        else:
            diagnostics.update(
                {
                    "node_count": 0.0,
                    "edge_count": 0.0,
                    "graph_hidden_norm": 0.0,
                    "graph_head_conditioning_norm": 0.0,
                    "head_change_due_to_graph": 0.0,
                    "graph_conditioning_norm": 0.0,
                    "graph_extraction_time": 0.0,
                    "graph_forward_time": 0.0,
                    "rate_change_due_to_graph": 0.0,
                    "graph_conditioning_stage": "disabled",
                    "gamma_with_graph": float(torch.mean(gamma).detach()),
                }
            )
        diagnostics.setdefault("gamma_with_graph", float(torch.mean(gamma).detach()))
        return rate, next_state, diagnostics

    def mpf_residual(self, phi: Tensor, model_rate: Tensor) -> Tensor:
        """Return synchronized MPF residual for the model's own field."""

        return model_rate - explicit_mpf_rhs(
            phi,
            eta_px=self.eta_px,
            mu=self.mu,
            sigma=self.sigma,
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
        rate, next_state, diagnostics = self.predict_rate(phi, state, previous_phi=previous_phi)
        raw_next = phi + self.model_dt * rate
        if self.projection_mode == "soft_threshold_eps1e3":
            next_phi, projection_diagnostics = project_simplex_soft_threshold(
                raw_next, threshold_eps=1.0e-3
            )
        else:
            next_phi, projection_diagnostics = project_simplex(raw_next)
        for key, value in projection_diagnostics.items():
            diagnostics[key] = value
        diagnostics["phase_sum_error_after_step"] = diagnostics.get("sum_error_post_projection", 0.0)
        return next_phi, next_state, rate, diagnostics

    def rollout(self, phi0: Tensor, *, steps: int) -> Tensor:
        """Return a rollout including the initial state."""

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


def _validate_phi(phi: Tensor, num_phases: int | None = None) -> None:
    if phi.ndim not in {4, 5}:
        raise ValueError(
            f"phi must have shape (B,N,H,W) [4D] or (B,N,Z,H,W) [5D], got {phi.ndim}D"
        )
    if phi.shape[1] < 2:
        raise ValueError("explicit-MPF phi requires at least two phases")
    if num_phases is not None and phi.shape[1] != num_phases:
        raise ValueError(f"expected {num_phases} phase channels, got {phi.shape[1]}")


def _coordinate_features(
    phi: Tensor,
    spatial: list[int] | tuple[int, ...],
    mode: str,
) -> Tensor:
    """Return spatial coordinate features of shape ``(*spatial, C)``."""
    if mode in {"periodic_3d", "raw_3d"}:
        depth, height, width = spatial
        z = torch.linspace(-1.0, 1.0, depth, device=phi.device, dtype=phi.dtype)
        y = torch.linspace(-1.0, 1.0, height, device=phi.device, dtype=phi.dtype)
        x = torch.linspace(-1.0, 1.0, width, device=phi.device, dtype=phi.dtype)
        zz, yy, xx = torch.meshgrid(z, y, x, indexing="ij")
        if mode == "periodic_3d":
            return torch.stack(
                (
                    torch.sin(math.pi * zz), torch.cos(math.pi * zz),
                    torch.sin(math.pi * yy), torch.cos(math.pi * yy),
                    torch.sin(math.pi * xx), torch.cos(math.pi * xx),
                ),
                dim=-1,
            )
        return torch.stack((zz, yy, xx), dim=-1)
    height, width = spatial
    y = torch.linspace(-1.0, 1.0, height, device=phi.device, dtype=phi.dtype)
    x = torch.linspace(-1.0, 1.0, width, device=phi.device, dtype=phi.dtype)
    yy, xx = torch.meshgrid(y, x, indexing="ij")
    if mode == "periodic":
        return torch.stack(
            (
                torch.sin(math.pi * yy),
                torch.cos(math.pi * yy),
                torch.sin(math.pi * xx),
                torch.cos(math.pi * xx),
            ),
            dim=-1,
        )
    if mode == "raw":
        return torch.stack((yy, xx), dim=-1)
    raise ValueError(f"coordinate mode {mode!r} has no coordinate features")


def _axis_contact_offsets(spatial_dims: int) -> list[tuple[int, ...]]:
    offsets = []
    for axis in range(spatial_dims):
        offset = [0] * spatial_dims
        offset[axis] = -1
        offsets.append(tuple(offset))
    return offsets


def _neighborhood_offsets(spatial_dims: int) -> list[tuple[int, ...]]:
    if spatial_dims == 2:
        return [(dy, dx) for dy in (-1, 0, 1) for dx in (-1, 0, 1)]
    if spatial_dims == 3:
        return [
            (dz, dy, dx)
            for dz in (-1, 0, 1)
            for dy in (-1, 0, 1)
            for dx in (-1, 0, 1)
        ]
    raise ValueError("spatial_dims must be 2 or 3")


def _shift_nd(labels: Tensor, offset: Sequence[int], periodic: bool) -> Tensor:
    if labels.ndim != len(offset):
        raise ValueError("shift offset rank must match label tensor rank")
    if periodic:
        result = labels
        for axis, delta in enumerate(offset):
            if delta:
                result = torch.roll(result, int(delta), dims=axis)
        return result
    pad = (1, 1) * labels.ndim
    padded = torch.nn.functional.pad(labels.reshape(1, 1, *labels.shape), pad, mode="replicate")
    slices = tuple(
        slice(1 + int(delta), 1 + int(delta) + labels.shape[axis])
        for axis, delta in enumerate(offset)
    )
    return padded[(0, 0, *slices)]


def _shift(labels: Tensor, dy: int, dx: int, periodic: bool) -> Tensor:
    return _shift_nd(labels, (dy, dx), periodic)


def _soft_centroids(phi: Tensor) -> Tensor:
    batch, phases, *spatial = phi.shape
    total = phi.flatten(2).sum(dim=-1).clamp_min(1.0e-12)
    coords = []
    for axis, size in enumerate(spatial):
        axis_values = torch.arange(size, device=phi.device, dtype=phi.dtype)
        angle = 2.0 * math.pi * axis_values / max(float(size), 1.0)
        reduce_dims = tuple(2 + other for other in range(len(spatial)) if other != axis)
        weights = phi.sum(dim=reduce_dims) if reduce_dims else phi
        sin_mean = (weights * torch.sin(angle).view(1, 1, size)).sum(dim=-1) / total
        cos_mean = (weights * torch.cos(angle).view(1, 1, size)).sum(dim=-1) / total
        coord = torch.remainder(torch.atan2(sin_mean, cos_mean), 2.0 * math.pi)
        coords.append(coord * size / (2.0 * math.pi))
    return torch.stack(coords, dim=-1)


def _periodic_delta(a: Tensor, b: Tensor, *spatial: int) -> Tensor:
    delta = b - a
    scale = torch.tensor([float(size) for size in spatial], device=delta.device, dtype=delta.dtype)
    half = scale / 2.0
    delta = torch.where(delta > half, delta - scale, delta)
    delta = torch.where(delta < -half, delta + scale, delta)
    return delta / scale.clamp_min(1.0)
