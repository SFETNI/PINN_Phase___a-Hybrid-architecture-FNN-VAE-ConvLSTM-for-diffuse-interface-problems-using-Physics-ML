"""Explicit training modes with lazy ground-truth access."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum

import torch
from torch import Tensor


class TrainingMode(str, Enum):
    """Supported clean-framework supervision modes."""

    UNSUPERVISED = "unsupervised"
    WEAKLY_SUPERVISED = "weakly_supervised"
    SUPERVISED = "supervised"


@dataclass(frozen=True)
class TrainingModeConfig:
    """Configure an explicit supervision contract.

    Weak supervision is defined by concrete rollout time indices. This avoids
    interpreting a percentage as time steps, trajectories, pixels, or epochs.
    """

    mode: TrainingMode
    physics_weight: float = 1.0
    supervision_weight: float = 1.0
    supervised_time_indices: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        if self.physics_weight < 0.0 or self.supervision_weight < 0.0:
            raise ValueError("loss weights must be non-negative")
        if self.physics_weight == 0.0 and self.supervision_weight == 0.0:
            raise ValueError("at least one loss weight must be positive")
        if len(set(self.supervised_time_indices)) != len(self.supervised_time_indices):
            raise ValueError("supervised_time_indices must be unique")
        if any(index < 0 for index in self.supervised_time_indices):
            raise ValueError("supervised_time_indices must be non-negative")
        if self.mode is TrainingMode.UNSUPERVISED and self.supervised_time_indices:
            raise ValueError("unsupervised mode cannot declare supervised time indices")
        if self.mode is TrainingMode.WEAKLY_SUPERVISED and not self.supervised_time_indices:
            raise ValueError("weakly supervised mode requires explicit time indices")
        if self.mode is TrainingMode.SUPERVISED and self.supervised_time_indices:
            raise ValueError("supervised mode uses every time step")


@dataclass(frozen=True)
class ModeLosses:
    """Loss terms assembled under one explicit supervision mode."""

    total: Tensor
    physics: Tensor
    supervision: Tensor | None


GroundTruthLoader = Callable[[], Tensor]


def compute_mode_losses(
    predicted_states: Tensor,
    physics_loss: Tensor,
    config: TrainingModeConfig,
    *,
    ground_truth_loader: GroundTruthLoader | None = None,
) -> ModeLosses:
    """Assemble mode-specific losses while keeping physics-only access lazy."""

    if physics_loss.ndim != 0:
        raise ValueError("physics_loss must be a scalar tensor")
    if config.mode is TrainingMode.UNSUPERVISED:
        return ModeLosses(
            total=config.physics_weight * physics_loss,
            physics=physics_loss,
            supervision=None,
        )
    if ground_truth_loader is None:
        raise ValueError(f"{config.mode.value} mode requires a ground-truth loader")

    ground_truth = ground_truth_loader()
    if ground_truth.shape != predicted_states.shape:
        raise ValueError("ground truth and predicted states must have matching shapes")
    if config.mode is TrainingMode.WEAKLY_SUPERVISED:
        if max(config.supervised_time_indices) >= predicted_states.shape[0]:
            raise ValueError("supervised time index exceeds rollout length")
        indices = torch.as_tensor(
            config.supervised_time_indices,
            device=predicted_states.device,
        )
        predicted = predicted_states.index_select(0, indices)
        target = ground_truth.index_select(0, indices)
    else:
        predicted = predicted_states
        target = ground_truth

    supervision = torch.mean((predicted - target) ** 2)
    total = config.physics_weight * physics_loss + config.supervision_weight * supervision
    return ModeLosses(total=total, physics=physics_loss, supervision=supervision)
