"""Differentiable scalar phase-field losses for the clean PINN framework.

The PDE is evaluated in physical phase space. Latent states are decoded before
physics losses are applied because arbitrary encoder channels are not physical
order parameters unless an experiment establishes that stronger assumption.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn as nn
from torch import Tensor

from pinn_phase.models.projection import project_outward_boundary_rates


def _validate_spacings(spacings: Sequence[float]) -> tuple[float, float]:
    values = tuple(float(value) for value in spacings)
    if len(values) != 2 or any(value <= 0.0 for value in values):
        raise ValueError("spacings must contain two positive values")
    return values


def _validate_spacings_3d(spacings: Sequence[float]) -> tuple[float, float, float]:
    values = tuple(float(value) for value in spacings)
    if len(values) != 3 or any(value <= 0.0 for value in values):
        raise ValueError("spacings_3d must contain three positive values")
    return values  # type: ignore[return-value]


def _validate_scalar_states(states: Tensor) -> None:
    if states.ndim < 4:
        raise ValueError("states must have shape (time, ..., channels, height, width)")
    if states.shape[0] < 2:
        raise ValueError("states must contain at least two time steps")
    if states.shape[-3] != 1:
        raise ValueError("scalar phase-field losses require exactly one channel")


def _validate_scalar_states_3d(states: Tensor) -> None:
    # 3D states: (time, batch, 1, D, H, W) — ndim=6
    if states.ndim < 5:
        raise ValueError("3D states must have shape (time, ..., 1, depth, height, width)")
    if states.shape[0] < 2:
        raise ValueError("states must contain at least two time steps")
    if states.shape[-4] != 1:
        raise ValueError("scalar 3D losses require exactly one channel (states[..., -4])")


def periodic_laplacian_2d(field: Tensor, spacings: Sequence[float]) -> Tensor:
    """Return the second-order periodic finite-difference Laplacian (2D)."""

    dx, dy = _validate_spacings(spacings)
    return (
        (torch.roll(field, 1, dims=-2) + torch.roll(field, -1, dims=-2) - 2.0 * field)
        / dx**2
        + (torch.roll(field, 1, dims=-1) + torch.roll(field, -1, dims=-1) - 2.0 * field)
        / dy**2
    )


def periodic_laplacian_3d(field: Tensor, spacings: Sequence[float]) -> Tensor:
    """Return the second-order periodic finite-difference Laplacian (3D).

    Parameters
    ----------
    field:
        Tensor of shape ``(..., D, H, W)`` — the Z/depth axis is ``-3``,
        Y/height is ``-2``, X/width is ``-1``.
    spacings:
        Three positive grid spacings ``(dz, dy, dx)`` matching the last three dims.

    Returns
    -------
    Tensor of same shape as ``field``.
    """
    dz, dy, dx = _validate_spacings_3d(spacings)
    return (
        (torch.roll(field, 1, dims=-3) + torch.roll(field, -1, dims=-3) - 2.0 * field) / dz**2
        + (torch.roll(field, 1, dims=-2) + torch.roll(field, -1, dims=-2) - 2.0 * field) / dy**2
        + (torch.roll(field, 1, dims=-1) + torch.roll(field, -1, dims=-1) - 2.0 * field) / dx**2
    )


def driving_shape(phi: Tensor, eta: float) -> Tensor:
    """Return the scalar external driving-force multiplier."""

    if eta <= 0.0:
        raise ValueError("eta must be positive")
    bounded_phi = torch.clamp(phi, 0.0, 1.0)
    # Use a small epsilon so that subnormal-float bulk pixels (phi ≈ 0 or 1
    # in float32) are treated as boundary, not interior.  Without this,
    # next_phase = 5e-8 rounds to u = -1.0 in float32 and the sqrt gradient
    # 1/(2*sqrt(phi*(1-phi))) blows up.
    _EPS = 1e-4
    interior = (bounded_phi > _EPS) & (bounded_phi < 1.0 - _EPS)
    safe_phi = torch.where(interior, bounded_phi, torch.full_like(bounded_phi, 0.5))
    shape = torch.sqrt(safe_phi * (1.0 - safe_phi))
    return torch.pi / eta * torch.where(interior, shape, torch.zeros_like(shape))


def driving_potential(phi: Tensor, eta: float) -> Tensor:
    """Return a primitive of :func:`driving_shape` for free-energy diagnostics."""

    if eta <= 0.0:
        raise ValueError("eta must be positive")
    bounded_phi = torch.clamp(phi, 0.0, 1.0)
    # Same epsilon guard as driving_shape: subnormal bulk phi rounds u to ±1
    # in float32, giving arcsin'(±1) = ∞ in the backward pass.
    _EPS = 1e-4
    interior = (bounded_phi > _EPS) & (bounded_phi < 1.0 - _EPS)
    safe_phi = torch.where(interior, bounded_phi, torch.full_like(bounded_phi, 0.5))
    root = torch.sqrt(safe_phi * (1.0 - safe_phi))
    # Clamp arcsin argument strictly inside (-1, 1) to prevent inf gradient.
    u = torch.clamp(2.0 * safe_phi - 1.0, -1.0 + _EPS, 1.0 - _EPS)
    integral = 0.25 * u * root + 0.125 * torch.arcsin(u)
    lower = torch.full_like(integral, -torch.pi / 16.0)
    upper = torch.full_like(integral, torch.pi / 16.0)
    integral = torch.where(bounded_phi <= _EPS, lower, integral)
    integral = torch.where(bounded_phi >= 1.0 - _EPS, upper, integral)
    return torch.pi / eta * integral


def algebraic_allen_cahn_rhs(
    phi: Tensor,
    *,
    mu: float,
    sigma: float,
    eta: float,
    delta_g: float = 0.0,
) -> Tensor:
    """Return the algebraic (non-Laplacian) Allen-Cahn RHS per pixel.

    This is the bulk + external driving contribution only — no spatial coupling.
    It is computable pointwise from phi alone, making it the natural anchor for
    a physics-guided FNN/ANN branch that has no spatial context.

    ``scalar_allen_cahn_rhs`` = algebraic_allen_cahn_rhs + mu*sigma*Laplacian(phi)
    """

    if mu <= 0.0 or sigma <= 0.0 or eta <= 0.0:
        raise ValueError("mu, sigma, and eta must be positive")
    bulk = torch.pi**2 / (2.0 * eta**2) * (2.0 * phi - 1.0)
    rhs = mu * sigma * bulk
    if delta_g != 0.0:
        rhs = rhs + mu * driving_shape(phi, eta) * delta_g
    return rhs


def mass_rate_loss(
    phi_t: Tensor,
    phi_t1: Tensor,
    *,
    dt: float,
    mu: float,
    sigma: float,
    spacings: Sequence[float],
) -> Tensor:
    """Penalise deviation from the theoretical mass change rate.

    **WARNING — geometry-specific constraint; NOT for the primary unsupervised claim.**

    For a single CIRCULAR grain with delta_g=0 the Allen-Cahn theory gives the
    exact law R^2(t) = R0^2 - 2*mu*sigma*t, so dM/dt = -2*pi*mu*sigma (constant).
    This translates to a constant expected change in sum(phi) per time step:

        d(sum(phi))/dt = -2*pi*mu*sigma / (dx*dy)

    This is geometry-specific: it fails for non-circular grains, multi-grain
    microstructures, or delta_g != 0.  Using it makes the training method
    geometry-aware and undermines the general unsupervised claim.

    Use case: ablation studies to quantify the benefit of this additional
    physics constraint vs the general rnn_laplacian_target_loss.

    Set mass_rate_weight=0.0 (default) in production unsupervised configs.
    """
    if mu <= 0.0 or sigma <= 0.0 or dt <= 0.0:
        raise ValueError("mu, sigma, and dt must be positive")
    dx, dy = _validate_spacings(spacings)
    actual_rate = (phi_t1.sum() - phi_t.sum()) / dt
    target_rate = torch.tensor(
        -2.0 * float(torch.pi) * mu * sigma / (dx * dy),
        dtype=phi_t.dtype,
        device=phi_t.device,
    )
    return (actual_rate - target_rate) ** 2


def rnn_laplacian_target_loss(
    rnn_correction: Tensor,
    phi: Tensor,
    *,
    mu: float,
    sigma: float,
    spacings: Sequence[float],
    detach_target: bool = True,
    interface_only: bool = True,
    interface_eps: float = 1.0e-3,
    bulk_weight: float = 0.0,
) -> Tensor:
    """Penalise ConvLSTM correction deviating from the trusted Laplacian term.

    In operator-split mode the ConvLSTM branch F_rnn should approximate the
    spatial term mu*sigma*Laplacian(phi).  This loss prevents F_rnn from
    amplifying beyond the correct Laplacian magnitude, which would trigger a
    positive-feedback over-shrinkage loop.

    The target is computed from the model's own phi field (unsupervised).
    """
    if mu <= 0.0 or sigma <= 0.0:
        raise ValueError("mu and sigma must be positive")
    if interface_eps <= 0.0:
        raise ValueError("interface_eps must be positive")
    if bulk_weight < 0.0:
        raise ValueError("bulk_weight must be non-negative")
    target = mu * sigma * periodic_laplacian_2d(phi, spacings)
    if detach_target:
        target = target.detach()
    error_squared = (rnn_correction - target) ** 2
    all_field_loss = torch.mean(error_squared)
    if not interface_only:
        return all_field_loss
    interface = (phi > interface_eps) & (phi < 1.0 - interface_eps)
    if not torch.any(interface):
        return torch.sum(error_squared) * 0.0
    return torch.mean(error_squared[interface]) + bulk_weight * all_field_loss


def ann_algebraic_target_loss(
    ann_local: Tensor,
    phi: Tensor,
    *,
    mu: float,
    sigma: float,
    eta: float,
    delta_g: float = 0.0,
    detach_target: bool = True,
    interface_only: bool = True,
    interface_eps: float = 1.0e-3,
    bulk_weight: float = 0.0,
) -> Tensor:
    """Train the ANN toward the local bulk and driving contribution.

    The target is evaluated from the model's own ``phi`` field. It is a
    physics-only operator target, not a labeled trajectory or a
    geometry-specific answer constraint.
    """

    if interface_eps <= 0.0:
        raise ValueError("interface_eps must be positive")
    if bulk_weight < 0.0:
        raise ValueError("bulk_weight must be non-negative")
    target = algebraic_allen_cahn_rhs(
        phi,
        mu=mu,
        sigma=sigma,
        eta=eta,
        delta_g=delta_g,
    )
    if detach_target:
        target = target.detach()
    error_squared = (ann_local - target) ** 2
    all_field_loss = torch.mean(error_squared)
    if not interface_only:
        return all_field_loss
    interface = (phi > interface_eps) & (phi < 1.0 - interface_eps)
    if not torch.any(interface):
        return torch.sum(error_squared) * 0.0
    return torch.mean(error_squared[interface]) + bulk_weight * all_field_loss


def scalar_allen_cahn_rhs(
    phi: Tensor,
    *,
    spacings: Sequence[float],
    mu: float,
    sigma: float,
    eta: float,
    delta_g: float = 0.0,
) -> Tensor:
    """Evaluate the trusted scalar Allen-Cahn RHS with periodic boundaries (2D)."""

    if mu <= 0.0 or sigma <= 0.0 or eta <= 0.0:
        raise ValueError("mu, sigma, and eta must be positive")
    laplacian = periodic_laplacian_2d(phi, spacings)
    bulk = torch.pi**2 / (2.0 * eta**2) * (2.0 * phi - 1.0)
    rhs = sigma * (laplacian + bulk)
    if delta_g != 0.0:
        rhs = rhs + driving_shape(phi, eta) * delta_g
    return mu * rhs


def scalar_allen_cahn_rhs_3d(
    phi: Tensor,
    *,
    spacings: Sequence[float],
    mu: float,
    sigma: float,
    eta: float,
    delta_g: float = 0.0,
) -> Tensor:
    """Evaluate the trusted scalar Allen-Cahn RHS with periodic boundaries (3D).

    3D sharp-interface law for a sphere: dR²/dt = −4μσ  (NOT −2μσ as in 2D).
    The factor of 4 arises because the mean curvature of a sphere is 2/R vs 1/R for a circle.

    Parameters
    ----------
    phi:
        Phase field tensor of shape ``(batch, 1, D, H, W)``.
    spacings:
        Three positive grid spacings ``(dz, dy, dx)``.
    """
    if mu <= 0.0 or sigma <= 0.0 or eta <= 0.0:
        raise ValueError("mu, sigma, and eta must be positive")
    laplacian = periodic_laplacian_3d(phi, spacings)
    bulk = torch.pi**2 / (2.0 * eta**2) * (2.0 * phi - 1.0)
    rhs = sigma * (laplacian + bulk)
    if delta_g != 0.0:
        rhs = rhs + driving_shape(phi, eta) * delta_g
    return mu * rhs


def projected_scalar_allen_cahn_rhs(
    phi: Tensor,
    *,
    boundary_eps: float = 1.0e-3,
    **kwargs: object,
) -> Tensor:
    """Return the bound-aware RHS used by projected explicit rollouts."""

    if boundary_eps <= 0.0:
        raise ValueError("boundary_eps must be positive")
    rhs = scalar_allen_cahn_rhs(phi, **kwargs)
    return project_outward_boundary_rates(phi, rhs, boundary_eps=boundary_eps)


def projected_scalar_allen_cahn_rhs_3d(
    phi: Tensor,
    *,
    boundary_eps: float = 1.0e-3,
    **kwargs: object,
) -> Tensor:
    """3D analogue of projected_scalar_allen_cahn_rhs for (B,1,D,H,W) tensors."""

    if boundary_eps <= 0.0:
        raise ValueError("boundary_eps must be positive")
    rhs = scalar_allen_cahn_rhs_3d(phi, **kwargs)
    return project_outward_boundary_rates(phi, rhs, boundary_eps=boundary_eps)


def scalar_pde_residual(
    states: Tensor,
    *,
    dt: float,
    spacings: Sequence[float],
    mu: float,
    sigma: float,
    eta: float,
    delta_g: float = 0.0,
) -> Tensor:
    """Return forward-Euler PDE residuals for a scalar physical rollout."""

    _validate_scalar_states(states)
    if dt <= 0.0:
        raise ValueError("dt must be positive")
    phi = states[:-1]
    phi_t = (states[1:] - phi) / dt
    rhs = scalar_allen_cahn_rhs(
        phi,
        spacings=spacings,
        mu=mu,
        sigma=sigma,
        eta=eta,
        delta_g=delta_g,
    )
    return phi_t - rhs


def scalar_pde_residual_loss(states: Tensor, **kwargs: object) -> Tensor:
    """Return mean-squared scalar PDE residual loss."""

    return torch.mean(scalar_pde_residual(states, **kwargs) ** 2)


def scalar_free_energy(
    states: Tensor,
    *,
    spacings: Sequence[float],
    sigma: float,
    eta: float,
    delta_g: float = 0.0,
) -> Tensor:
    """Return discrete free energy for each leading state entry."""

    if states.ndim < 3:
        raise ValueError("states must include channels, height, and width dimensions")
    if states.shape[-3] != 1:
        raise ValueError("scalar free energy requires exactly one channel")
    if sigma <= 0.0 or eta <= 0.0:
        raise ValueError("sigma and eta must be positive")
    dx, dy = _validate_spacings(spacings)
    gradient_x = (torch.roll(states, -1, dims=-2) - states) / dx
    gradient_y = (torch.roll(states, -1, dims=-1) - states) / dy
    gradient_squared = gradient_x**2 + gradient_y**2
    interfacial = sigma * (
        0.5 * gradient_squared
        + torch.pi**2 / (2.0 * eta**2) * states * (1.0 - states)
    )
    if delta_g == 0.0:
        chemical = torch.zeros_like(states)
    else:
        chemical = -delta_g * driving_potential(states, eta)
    return torch.sum(interfacial + chemical, dim=(-3, -2, -1)) * dx * dy


def energy_monotonicity_loss(
    states: Tensor,
    *,
    spacings: Sequence[float],
    sigma: float,
    eta: float,
    delta_g: float = 0.0,
) -> Tensor:
    """Penalize increases in scalar free energy across a rollout (2D)."""

    _validate_scalar_states(states)
    energies = scalar_free_energy(
        states,
        spacings=spacings,
        sigma=sigma,
        eta=eta,
        delta_g=delta_g,
    )
    increases = torch.relu(energies[1:] - energies[:-1])
    return torch.mean(increases**2)


def scalar_free_energy_3d(
    states: Tensor,
    *,
    spacings: Sequence[float],
    sigma: float,
    eta: float,
    delta_g: float = 0.0,
) -> Tensor:
    """Return discrete free energy for each leading state entry (3D).

    Parameters
    ----------
    states:
        Tensor of shape ``(time, batch, 1, D, H, W)`` or ``(batch, 1, D, H, W)``.
    spacings:
        Three positive grid spacings ``(dz, dy, dx)``.
    """
    if states.ndim < 5:
        raise ValueError("3D states must include channels, depth, height, width dimensions")
    if states.shape[-4] != 1:
        raise ValueError("scalar 3D free energy requires exactly one channel")
    if sigma <= 0.0 or eta <= 0.0:
        raise ValueError("sigma and eta must be positive")
    dz, dy, dx = _validate_spacings_3d(spacings)
    cell_vol = dz * dy * dx
    gradient_z = (torch.roll(states, -1, dims=-3) - states) / dz
    gradient_y = (torch.roll(states, -1, dims=-2) - states) / dy
    gradient_x = (torch.roll(states, -1, dims=-1) - states) / dx
    gradient_squared = gradient_z**2 + gradient_y**2 + gradient_x**2
    interfacial = sigma * (
        0.5 * gradient_squared
        + torch.pi**2 / (2.0 * eta**2) * states * (1.0 - states)
    )
    if delta_g == 0.0:
        chemical = torch.zeros_like(states)
    else:
        chemical = -delta_g * driving_potential(states, eta)
    return torch.sum(interfacial + chemical, dim=(-4, -3, -2, -1)) * cell_vol


def energy_monotonicity_loss_3d(
    states: Tensor,
    *,
    spacings: Sequence[float],
    sigma: float,
    eta: float,
    delta_g: float = 0.0,
) -> Tensor:
    """Penalize increases in scalar free energy across a 3D rollout."""

    _validate_scalar_states_3d(states)
    energies = scalar_free_energy_3d(
        states,
        spacings=spacings,
        sigma=sigma,
        eta=eta,
        delta_g=delta_g,
    )
    increases = torch.relu(energies[1:] - energies[:-1])
    return torch.mean(increases**2)


def scalar_pde_residual_3d(
    states: Tensor,
    *,
    dt: float,
    spacings: Sequence[float],
    mu: float,
    sigma: float,
    eta: float,
    delta_g: float = 0.0,
) -> Tensor:
    """Return forward-Euler PDE residuals for a 3D scalar physical rollout.

    Parameters
    ----------
    states:
        Tensor of shape ``(time, batch, 1, D, H, W)``.
    """
    _validate_scalar_states_3d(states)
    if dt <= 0.0:
        raise ValueError("dt must be positive")
    phi = states[:-1]
    phi_t = (states[1:] - phi) / dt
    rhs = scalar_allen_cahn_rhs_3d(
        phi,
        spacings=spacings,
        mu=mu,
        sigma=sigma,
        eta=eta,
        delta_g=delta_g,
    )
    return phi_t - rhs


def scalar_pde_residual_loss_3d(states: Tensor, **kwargs: object) -> Tensor:
    """Return mean-squared scalar PDE residual loss (3D)."""
    return torch.mean(scalar_pde_residual_3d(states, **kwargs) ** 2)


def rnn_laplacian_target_loss_3d(
    rnn_correction: Tensor,
    phi: Tensor,
    *,
    mu: float,
    sigma: float,
    spacings: Sequence[float],
    detach_target: bool = True,
    interface_only: bool = True,
    interface_eps: float = 1.0e-3,
    bulk_weight: float = 0.0,
) -> Tensor:
    """Penalise ConvLSTM3d correction deviating from the 3D Laplacian term.

    3D analogue of rnn_laplacian_target_loss. Target is mu*sigma*Laplacian3d(phi).
    """
    if mu <= 0.0 or sigma <= 0.0:
        raise ValueError("mu and sigma must be positive")
    if interface_eps <= 0.0:
        raise ValueError("interface_eps must be positive")
    target = mu * sigma * periodic_laplacian_3d(phi, spacings)
    if detach_target:
        target = target.detach()
    error_squared = (rnn_correction - target) ** 2
    all_field_loss = torch.mean(error_squared)
    if not interface_only:
        return all_field_loss
    interface = (phi > interface_eps) & (phi < 1.0 - interface_eps)
    if not torch.any(interface):
        return torch.sum(error_squared) * 0.0
    return torch.mean(error_squared[interface]) + bulk_weight * all_field_loss


def decode_latent_rollout(latent_states: Tensor, decoder: nn.Module) -> Tensor:
    """Decode a ``(time, batch, channels, height, width)`` latent rollout."""

    if latent_states.ndim != 5:
        raise ValueError("latent_states must have shape (time, batch, channels, height, width)")
    time_steps, batch = latent_states.shape[:2]
    flattened = latent_states.reshape(time_steps * batch, *latent_states.shape[2:])
    decoded = decoder(flattened)
    if decoded.ndim != 4:
        raise ValueError("decoder must return (batch, channels, height, width)")
    return decoded.reshape(time_steps, batch, *decoded.shape[1:])


def decoded_latent_physics_loss(
    latent_states: Tensor,
    decoder: nn.Module,
    **kwargs: object,
) -> Tensor:
    """Decode latent states and evaluate the trusted physical-space PDE loss."""

    decoded_states = decode_latent_rollout(latent_states, decoder)
    return scalar_pde_residual_loss(decoded_states, **kwargs)
