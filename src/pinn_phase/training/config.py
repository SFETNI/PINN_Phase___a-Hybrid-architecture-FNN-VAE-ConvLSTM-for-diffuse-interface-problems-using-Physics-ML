"""Typed configuration for the clean PINN-Phase training framework."""

from __future__ import annotations

from dataclasses import dataclass

from pinn_phase.models import ResidualBlend

from .optimizers import GradNormController


@dataclass(frozen=True)
class PhysicsConfig:
    """Scalar Allen-Cahn coefficients used by differentiable physics losses."""

    dt: float
    spacings: tuple[float, ...]
    mu: float
    sigma: float
    eta: float
    delta_g: float = 0.0
    time_scale: float = 1.0

    def __post_init__(self) -> None:
        if self.dt <= 0.0:
            raise ValueError("dt must be positive")
        if self.time_scale <= 0.0:
            raise ValueError("time_scale must be positive")
        if len(self.spacings) not in (2, 3) or any(value <= 0.0 for value in self.spacings):
            raise ValueError("spacings must contain two (2D) or three (3D) positive values")
        if self.mu <= 0.0 or self.sigma <= 0.0 or self.eta <= 0.0:
            raise ValueError("mu, sigma, and eta must be positive")

    @property
    def physical_dt(self) -> float:
        """Return the PF-reference time increment represented by one model step."""

        return self.dt * self.time_scale

    def loss_kwargs(self) -> dict[str, object]:
        """Return keyword arguments for scalar physical losses."""

        return {
            "dt": self.dt,
            "spacings": self.spacings,
            "mu": self.mu,
            "sigma": self.sigma,
            "eta": self.eta,
            "delta_g": self.delta_g,
        }


@dataclass(frozen=True)
class ModelConfig:
    """Local neural model settings."""

    hidden_channels: int = 8
    kernel_size: int = 3
    blend_gamma: float = 0.5
    learnable_blend: bool = False

    def __post_init__(self) -> None:
        if self.hidden_channels < 1:
            raise ValueError("hidden_channels must be positive")
        if self.kernel_size < 1 or self.kernel_size % 2 == 0:
            raise ValueError("kernel_size must be a positive odd integer")
        if not 0.0 < self.blend_gamma < 1.0:
            raise ValueError("blend_gamma must be strictly inside (0, 1)")


@dataclass(frozen=True)
class GradNormConfig:
    """Adaptive loss-balancing settings."""

    alpha: float = 0.5
    learning_rate: float = 1.0e-4

    def __post_init__(self) -> None:
        if self.alpha < 0.0:
            raise ValueError("alpha must be non-negative")
        if self.learning_rate <= 0.0:
            raise ValueError("learning_rate must be positive")


def build_residual_blend(physics: PhysicsConfig, model: ModelConfig) -> ResidualBlend:
    """Build a residual blend without conflating physical and model coefficients."""

    return ResidualBlend(
        dt=physics.dt,
        initial_gamma=model.blend_gamma,
        learnable=model.learnable_blend,
    )


def build_gradnorm_controller(
    loss_names: tuple[str, ...],
    config: GradNormConfig,
) -> GradNormController:
    """Build a persistent GradNorm controller from its independent config."""

    return GradNormController(
        loss_names,
        alpha=config.alpha,
        learning_rate=config.learning_rate,
    )
