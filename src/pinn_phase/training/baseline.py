"""Short-window autoregressive training for the first clean ConvGRU surrogate."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor
from torch.optim import Optimizer

from pinn_phase.models import LocalConvGRURollout

from .config import PhysicsConfig
from .losses import scalar_allen_cahn_rhs, scalar_free_energy


@dataclass(frozen=True)
class SurrogateTrainingConfig:
    """Short-baseline training settings."""

    epochs: int
    tbptt_window: int
    physics_weight: float = 0.0
    energy_weight: float = 0.0

    def __post_init__(self) -> None:
        if self.epochs < 1 or self.tbptt_window < 1:
            raise ValueError("epochs and tbptt_window must be positive")
        if self.physics_weight < 0.0 or self.energy_weight < 0.0:
            raise ValueError("loss weights must be non-negative")


@dataclass(frozen=True)
class SurrogateTrainingHistory:
    """Per-epoch scalar loss history."""

    total: tuple[float, ...]
    supervised: tuple[float, ...]
    physics: tuple[float, ...]
    energy: tuple[float, ...]

    def to_dict(self) -> dict[str, list[float]]:
        """Return a JSON-serializable history."""

        return {
            "total": list(self.total),
            "supervised": list(self.supervised),
            "physics": list(self.physics),
            "energy": list(self.energy),
        }


def train_autoregressive_surrogate(
    model: LocalConvGRURollout,
    reference_states: Tensor,
    *,
    physics: PhysicsConfig,
    optimizer: Optimizer,
    config: SurrogateTrainingConfig,
) -> SurrogateTrainingHistory:
    """Fit a local neural derivative using PF targets and autoregressive TBPTT."""

    if model.rhs_operator is not None:
        raise ValueError("trained surrogate must not use a trusted RHS anchor")
    if reference_states.ndim != 5 or reference_states.shape[2] != 1:
        raise ValueError("reference_states must have shape (time, batch, 1, height, width)")
    if reference_states.shape[0] < 2:
        raise ValueError("reference_states must contain at least two time steps")
    if abs(model.dt - physics.dt) > 1.0e-12:
        raise ValueError("model and physics dt must match")

    total_history = []
    supervised_history = []
    physics_history = []
    energy_history = []
    transitions = reference_states.shape[0] - 1
    for _ in range(config.epochs):
        hidden = model.initial_state(reference_states[0])
        epoch_totals = torch.zeros(4, device=reference_states.device)
        completed = 0
        while completed < transitions:
            optimizer.zero_grad(set_to_none=True)
            segment_total = torch.zeros((), device=reference_states.device)
            segment_count = min(config.tbptt_window, transitions - completed)
            phase = reference_states[completed]
            for offset in range(segment_count):
                index = completed + offset
                target = reference_states[index + 1]
                derivative, hidden = model.predict_derivative(phase, hidden)
                prediction = phase + model.dt * derivative
                if model.project_bounds:
                    prediction = torch.clamp(prediction, 0.0, 1.0)
                supervised = torch.mean((prediction - target) ** 2)
                trusted_rhs = scalar_allen_cahn_rhs(
                    phase,
                    spacings=physics.spacings,
                    mu=physics.mu,
                    sigma=physics.sigma,
                    eta=physics.eta,
                    delta_g=physics.delta_g,
                )
                pde = torch.mean((derivative - trusted_rhs) ** 2)
                energies = scalar_free_energy(
                    torch.stack((phase, prediction)),
                    spacings=physics.spacings,
                    sigma=physics.sigma,
                    eta=physics.eta,
                    delta_g=physics.delta_g,
                )
                increase = torch.relu(energies[1] - energies[0])
                energy = torch.mean(increase**2)
                loss = supervised + config.physics_weight * pde + config.energy_weight * energy
                segment_total = segment_total + loss
                epoch_totals = epoch_totals + torch.stack(
                    (loss.detach(), supervised.detach(), pde.detach(), energy.detach())
                )
                phase = prediction
            (segment_total / segment_count).backward()
            optimizer.step()
            hidden = hidden.detach()
            completed += segment_count

        means = (epoch_totals / transitions).cpu()
        total_history.append(float(means[0]))
        supervised_history.append(float(means[1]))
        physics_history.append(float(means[2]))
        energy_history.append(float(means[3]))

    return SurrogateTrainingHistory(
        total=tuple(total_history),
        supervised=tuple(supervised_history),
        physics=tuple(physics_history),
        energy=tuple(energy_history),
    )
