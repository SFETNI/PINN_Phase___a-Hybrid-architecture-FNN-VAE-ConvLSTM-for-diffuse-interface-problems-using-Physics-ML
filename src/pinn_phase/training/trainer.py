"""Small TBPTT trainer used by the clean local-rollout baseline."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import torch
from torch import Tensor
from torch.optim import Optimizer

from pinn_phase.models import LocalConvGRURollout


SegmentLoss = Callable[[Tensor], Tensor]


@dataclass(frozen=True)
class TBPTTEpochResult:
    """Summary of one truncated-backpropagation epoch."""

    segment_losses: tuple[float, ...]

    @property
    def mean_loss(self) -> float:
        return sum(self.segment_losses) / len(self.segment_losses)


def train_tbptt_epoch(
    model: LocalConvGRURollout,
    initial_phase: Tensor,
    *,
    steps: int,
    window: int,
    optimizer: Optimizer,
    segment_loss: SegmentLoss,
) -> TBPTTEpochResult:
    """Train one rollout epoch while detaching state at each TBPTT boundary."""

    losses = []
    for states in model.iter_tbptt_segments(initial_phase, steps=steps, window=window):
        optimizer.zero_grad(set_to_none=True)
        loss = segment_loss(states)
        if loss.ndim != 0:
            raise ValueError("segment_loss must return a scalar tensor")
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach()))
    return TBPTTEpochResult(segment_losses=tuple(losses))
