"""Minimal local ConvGRU rollout used to validate clean training plumbing."""

from __future__ import annotations

from collections.abc import Callable, Iterator

import torch
import torch.nn as nn
from torch import Tensor

from .cells import ConvGRUCell


RHSOperator = Callable[[Tensor], Tensor]


class LocalConvGRURollout(nn.Module):
    """Advance a scalar phase field with a local ConvGRU correction.

    An optional trusted local RHS operator can anchor the derivative during the
    first smoke baseline. The trainable recurrent correction remains explicit,
    local, and signed. Later ablations can remove the anchor.
    """

    def __init__(
        self,
        *,
        dt: float,
        hidden_channels: int = 8,
        kernel_size: int = 3,
        rhs_operator: RHSOperator | None = None,
        project_bounds: bool = True,
        zero_initialize_head: bool = True,
    ) -> None:
        super().__init__()
        if dt <= 0.0:
            raise ValueError("dt must be positive")
        self.dt = float(dt)
        self.project_bounds = bool(project_bounds)
        self.rhs_operator = rhs_operator
        self.cell = ConvGRUCell(
            in_channels=1,
            hidden_channels=hidden_channels,
            kernel_size=kernel_size,
        )
        padding = kernel_size // 2
        self.correction_head = nn.Conv2d(
            hidden_channels,
            1,
            kernel_size=kernel_size,
            padding=padding,
        )
        if zero_initialize_head:
            nn.init.zeros_(self.correction_head.weight)
            nn.init.zeros_(self.correction_head.bias)

    def initial_state(self, phase: Tensor) -> Tensor:
        """Return a zero hidden state matching one scalar phase-field batch."""

        self._validate_phase(phase)
        batch, _, height, width = phase.shape
        return self.cell.initial_state(batch, height, width, device=phase.device)

    def predict_derivative(self, phase: Tensor, hidden: Tensor) -> tuple[Tensor, Tensor]:
        """Return a signed derivative and the next recurrent hidden state."""

        self._validate_phase(phase)
        next_hidden = self.cell(phase, hidden)
        derivative = self.correction_head(next_hidden)
        if self.rhs_operator is not None:
            derivative = self.rhs_operator(phase) + derivative
        return derivative, next_hidden

    def forward_step(
        self,
        phase: Tensor,
        hidden: Tensor | None = None,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Advance one explicit-Euler step."""

        if hidden is None:
            hidden = self.initial_state(phase)
        derivative, next_hidden = self.predict_derivative(phase, hidden)
        next_phase = phase + self.dt * derivative
        if self.project_bounds:
            next_phase = torch.clamp(next_phase, 0.0, 1.0)
        return next_phase, next_hidden, derivative

    def rollout(
        self,
        initial_phase: Tensor,
        *,
        steps: int,
    ) -> Tensor:
        """Return a full inference rollout including the initial state."""

        if steps < 1:
            raise ValueError("steps must be positive")
        phase = initial_phase
        hidden = self.initial_state(phase)
        states = [phase]
        for _ in range(steps):
            phase, hidden, _ = self.forward_step(phase, hidden)
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
        hidden = self.initial_state(phase)
        completed = 0
        while completed < steps:
            segment_states = [phase]
            for _ in range(min(window, steps - completed)):
                phase, hidden, _ = self.forward_step(phase, hidden)
                segment_states.append(phase)
                completed += 1
            yield torch.stack(segment_states)
            phase = phase.detach()
            hidden = hidden.detach()

    @staticmethod
    def _validate_phase(phase: Tensor) -> None:
        if phase.ndim != 4 or phase.shape[1] != 1:
            raise ValueError("phase must have shape (batch, 1, height, width)")
