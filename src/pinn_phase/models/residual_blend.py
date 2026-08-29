"""Residual blend module: combines ANN and recurrent branch outputs.

Convention (preprint Eq. 1):

    x_next = (1 - blend_gamma) * x_ann
             + blend_gamma * dt * delta_x_rnn

where:
  x_ann       — ANN spatial prediction at the current time step
  delta_x_rnn — signed recurrent correction (NOT relu-ed; negative values
                are essential for shrinking interfaces)
  blend_gamma — scalar in (0, 1), optionally learnable via sigmoid(logit)
  dt          — physical time step size (scalar float)

Design decisions:
  R1. Signed recurrent corrections: no ReLU is applied to delta_x_rnn.
      This allows negative corrections needed for grain shrinkage.
  R2. One named convention used everywhere: the gamma convention from the
      preprint (gamma=0 → pure ANN, gamma=1 → pure recurrent).
  learned mode: gamma = sigmoid(blend_gamma_logit), constrained to (0, 1).
  fixed mode:   gamma is a plain float buffer, not a learnable parameter.

The legacy prototype reverses the convention (gamma=0.9 ≈ 90% ANN in code,
but the preprint means 90% recurrent).  This module uses the preprint
convention explicitly.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor


class ResidualBlend(nn.Module):
    """Combine ANN and recurrent predictions with a bounded blend coefficient.

    Parameters
    ----------
    dt:
        Physical time step used to scale the recurrent increment.
    initial_gamma:
        Starting value for blend_gamma in (0, 1).  If ``learnable`` is True
        this is converted to the corresponding logit.
    learnable:
        When True, blend_gamma is optimized via a sigmoid-constrained scalar
        logit.  When False, gamma is a fixed buffer.

    Examples
    --------
    >>> blend = ResidualBlend(dt=1.0, initial_gamma=0.5)
    >>> x_next = blend(x_ann, delta_x_rnn)
    """

    def __init__(
        self,
        dt: float,
        initial_gamma: float = 0.5,
        learnable: bool = True,
    ) -> None:
        super().__init__()
        if not 0.0 < initial_gamma < 1.0:
            raise ValueError("initial_gamma must be strictly inside (0, 1)")
        if dt <= 0.0:
            raise ValueError("dt must be positive")

        self.dt = float(dt)
        self._learnable = learnable

        logit_value = torch.tensor(
            float(torch.logit(torch.tensor(initial_gamma))),
            dtype=torch.float32,
        )
        if learnable:
            self.blend_gamma_logit = nn.Parameter(logit_value)
        else:
            self.register_buffer("blend_gamma_logit", logit_value)

    @property
    def blend_gamma(self) -> Tensor:
        """Return the current blend coefficient, guaranteed in (0, 1)."""
        return torch.sigmoid(self.blend_gamma_logit)

    def forward(self, x_ann: Tensor, delta_x_rnn: Tensor) -> Tensor:
        """Compute the blended next-step prediction.

        Parameters
        ----------
        x_ann:
            ANN spatial prediction at the current time step.
            Shape: ``(batch, channels, H, W)`` or any broadcastable shape.
        delta_x_rnn:
            Signed recurrent time-derivative prediction.  Negative values
            are preserved; no ReLU is applied.
            Shape: broadcastable with ``x_ann``.

        Returns
        -------
        x_next:
            Blended prediction with the same shape as ``x_ann``.
        """

        gamma = self.blend_gamma
        return (1.0 - gamma) * x_ann + gamma * self.dt * delta_x_rnn

    def extra_repr(self) -> str:
        return (
            f"dt={self.dt}, learnable={self._learnable}, "
            f"blend_gamma={self.blend_gamma.item():.4f}"
        )
