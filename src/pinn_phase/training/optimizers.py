"""Optimizer ownership and persistent GradNorm utilities for the clean framework."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.optim import Adam, Optimizer


class OptimizerOwnershipError(ValueError):
    """Raised when one trainable parameter is assigned to multiple optimizers."""


@dataclass(frozen=True)
class OptimizerGroup:
    """Named trainable parameter group with its own optimizer settings."""

    name: str
    parameters: Iterable[nn.Parameter]
    learning_rate: float
    weight_decay: float = 0.0


@dataclass(frozen=True)
class OptimizerBundle:
    """Optimizers and immutable parameter-owner lookup."""

    optimizers: dict[str, Optimizer]
    parameter_owners: dict[int, str]

    def zero_grad(self, *, set_to_none: bool = True) -> None:
        for optimizer in self.optimizers.values():
            optimizer.zero_grad(set_to_none=set_to_none)

    def step(self) -> None:
        for optimizer in self.optimizers.values():
            optimizer.step()


def build_adam_optimizers(groups: Sequence[OptimizerGroup]) -> OptimizerBundle:
    """Build non-overlapping Adam optimizers from named trainable groups."""

    if not groups:
        raise ValueError("at least one optimizer group is required")

    optimizers: dict[str, Optimizer] = {}
    parameter_owners: dict[int, str] = {}
    for group in groups:
        if not group.name:
            raise ValueError("optimizer group names must be non-empty")
        if group.name in optimizers:
            raise ValueError(f"duplicate optimizer group name: {group.name}")
        if group.learning_rate <= 0.0:
            raise ValueError(f"learning_rate must be positive for group {group.name}")
        if group.weight_decay < 0.0:
            raise ValueError(f"weight_decay must be non-negative for group {group.name}")

        parameters = [parameter for parameter in group.parameters if parameter.requires_grad]
        if not parameters:
            raise ValueError(f"optimizer group {group.name} has no trainable parameters")
        for parameter in parameters:
            identifier = id(parameter)
            previous_owner = parameter_owners.get(identifier)
            if previous_owner is not None:
                raise OptimizerOwnershipError(
                    f"parameter appears in both optimizer groups: {previous_owner}, {group.name}"
                )
            parameter_owners[identifier] = group.name

        optimizers[group.name] = Adam(
            parameters,
            lr=group.learning_rate,
            weight_decay=group.weight_decay,
        )

    return OptimizerBundle(optimizers=optimizers, parameter_owners=parameter_owners)


class GradNormController(nn.Module):
    """Persistent GradNorm loss-weight controller.

    The controller owns only its loss-weight logits. Model optimizers remain
    separate, so model parameters cannot be double-stepped by GradNorm.
    """

    def __init__(
        self,
        loss_names: Sequence[str],
        *,
        alpha: float = 0.5,
        learning_rate: float = 1.0e-4,
        initial_weight: float = 1.0,
        epsilon: float = 1.0e-8,
    ) -> None:
        super().__init__()
        if not loss_names:
            raise ValueError("loss_names must not be empty")
        if len(set(loss_names)) != len(loss_names):
            raise ValueError("loss_names must be unique")
        if alpha < 0.0:
            raise ValueError("alpha must be non-negative")
        if learning_rate <= 0.0 or initial_weight <= 0.0 or epsilon <= 0.0:
            raise ValueError("learning_rate, initial_weight, and epsilon must be positive")

        self.loss_names = tuple(loss_names)
        self.alpha = float(alpha)
        self.epsilon = float(epsilon)
        self.register_buffer("target_weight_sum", torch.tensor(float(len(loss_names))))
        initial_raw = torch.log(torch.expm1(torch.tensor(float(initial_weight))))
        self.raw_weights = nn.Parameter(initial_raw.repeat(len(loss_names)))
        self.optimizer = Adam([self.raw_weights], lr=learning_rate)
        self.register_buffer("initial_losses", torch.empty(0))

    @property
    def weights(self) -> Tensor:
        """Return positive weights normalized to a fixed sum."""

        positive = F.softplus(self.raw_weights) + self.epsilon
        return positive / positive.sum() * self.target_weight_sum

    def effective_weights(self, active_loss_names: Sequence[str] | None = None) -> Tensor:
        """Return weights after optional active-loss-only renormalization."""

        weights = self.weights
        if active_loss_names is None:
            return weights
        active = set(active_loss_names)
        unknown = active.difference(self.loss_names)
        if unknown:
            raise ValueError(f"unknown active GradNorm loss names: {sorted(unknown)}")
        mask = torch.tensor(
            [name in active for name in self.loss_names],
            device=weights.device,
            dtype=weights.dtype,
        )
        active_count = mask.sum()
        if active_count <= 0:
            return torch.zeros_like(weights)
        selected = weights * mask
        return selected / selected.sum().clamp_min(self.epsilon) * active_count

    def weighted_model_loss(
        self,
        losses: Mapping[str, Tensor],
        *,
        active_loss_names: Sequence[str] | None = None,
    ) -> Tensor:
        """Return weighted model loss while holding GradNorm weights fixed."""

        loss_values = torch.stack(self._ordered_losses(losses))
        return torch.sum(self.effective_weights(active_loss_names).detach() * loss_values)

    def initialize(self, losses: Mapping[str, Tensor]) -> None:
        """Record initial loss magnitudes once for relative training rates."""

        if self.initial_losses.numel() != 0:
            return
        initial = torch.stack(self._ordered_losses(losses)).detach().clamp_min(self.epsilon)
        self.initial_losses = initial

    def gradnorm_loss(
        self,
        losses: Mapping[str, Tensor],
        shared_parameters: Iterable[nn.Parameter],
        *,
        active_loss_names: Sequence[str] | None = None,
    ) -> Tensor:
        """Return GradNorm objective for the persistent weight optimizer."""

        self.initialize(losses)
        loss_values = self._ordered_losses(losses)
        active_indices = self._active_indices(active_loss_names)
        if not active_indices:
            return self.raw_weights.sum() * 0.0
        parameters = [parameter for parameter in shared_parameters if parameter.requires_grad]
        if not parameters:
            raise ValueError("shared_parameters must contain trainable parameters")

        weights = self.effective_weights(active_loss_names)
        gradient_norms = []
        active_loss_values = [loss_values[index] for index in active_indices]
        active_initial_losses = self.initial_losses[active_indices]
        for index, loss in zip(active_indices, active_loss_values):
            weight = weights[index]
            gradients = torch.autograd.grad(
                loss,
                parameters,
                retain_graph=True,
                create_graph=False,
                allow_unused=True,
            )
            norm_terms = [
                gradient.detach().norm(2)
                for gradient in gradients
                if gradient is not None
            ]
            if not norm_terms:
                raise ValueError("a configured GradNorm loss does not reach shared parameters")
            base_norm = torch.stack(norm_terms).norm(2)
            gradient_norms.append(weight * base_norm)

        norms = torch.stack(gradient_norms)
        relative_losses = (
            torch.stack(active_loss_values).detach().clamp_min(self.epsilon)
            / active_initial_losses
        )
        inverse_training_rates = relative_losses / relative_losses.mean()
        targets = norms.detach().mean() * inverse_training_rates.pow(self.alpha)
        return F.l1_loss(norms, targets.detach(), reduction="sum")

    def step(
        self,
        losses: Mapping[str, Tensor],
        shared_parameters: Iterable[nn.Parameter],
        *,
        active_loss_names: Sequence[str] | None = None,
    ) -> Tensor:
        """Update GradNorm weights with the controller's persistent optimizer."""

        self.optimizer.zero_grad(set_to_none=True)
        objective = self.gradnorm_loss(
            losses,
            shared_parameters,
            active_loss_names=active_loss_names,
        )
        objective.backward()
        self.optimizer.step()
        return objective.detach()

    def named_weights(self) -> dict[str, float]:
        """Return detached normalized weights for logs and artifacts."""

        return {
            name: float(value)
            for name, value in zip(self.loss_names, self.weights.detach().cpu())
        }

    def named_effective_weights(
        self,
        active_loss_names: Sequence[str] | None = None,
    ) -> dict[str, float]:
        """Return detached effective weights after active-loss renormalization."""

        return {
            name: float(value)
            for name, value in zip(
                self.loss_names,
                self.effective_weights(active_loss_names).detach().cpu(),
            )
        }

    def _active_indices(self, active_loss_names: Sequence[str] | None) -> list[int]:
        if active_loss_names is None:
            return list(range(len(self.loss_names)))
        active = set(active_loss_names)
        unknown = active.difference(self.loss_names)
        if unknown:
            raise ValueError(f"unknown active GradNorm loss names: {sorted(unknown)}")
        return [index for index, name in enumerate(self.loss_names) if name in active]

    def _ordered_losses(self, losses: Mapping[str, Tensor]) -> tuple[Tensor, ...]:
        missing = [name for name in self.loss_names if name not in losses]
        extra = [name for name in losses if name not in self.loss_names]
        if missing or extra:
            raise ValueError(f"loss names differ: missing={missing}, extra={extra}")
        values = [losses[name] for name in self.loss_names]
        if any(value.ndim != 0 for value in values):
            raise ValueError("GradNorm losses must be scalar tensors")
        return tuple(values)
