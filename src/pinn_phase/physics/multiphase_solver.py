"""Multi-phase-field reference solver.

Governing equation (preprint Eq. 2-3, notebook cell 65):

    dphi_alpha/dt = (1 / N) * sum_{beta != alpha} mu_ab *
                    sigma_ab * (I_alpha - I_beta)

    I_alpha = laplacian(phi_alpha) + (pi^2 / eta^2) * phi_alpha

Phase-sum constraint sum_alpha phi_alpha = 1 is enforced after each step
by renormalizing the updated fields.

For an isotropic two-grain system (all sigma_ab equal, Δg = 0) this reduces
algebraically to the scalar Allen-Cahn equation validated in reference_solver.

Non-zero Δg for the general MPF case is not implemented because the
preprint (Eq. 2) uses a constant (π²/4η)·Δg_αβ factor whereas the notebook
prototype (cell 65, commented out) uses (π/η)·√(φ_α·φ_β)·Δg. The
MultiphasePFConfig constructor raises NotImplementedError for delta_g != 0
until that convention is resolved.

Sources:
  preprint PINN-Phase ssrn-5207041.pdf §3.1 Eqs. (2)-(4)
  Phase_Field/dual_phases_64_ref.ipynb cell 65
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from pinn_phase.physics.reference_solver import periodic_laplacian

Array = np.ndarray


@dataclass(frozen=True)
class MultiphasePFConfig:
    """Physical and numerical parameters for a periodic MPF benchmark."""

    benchmark_id: str
    num_phases: int
    shape: tuple[int, int]
    domain_lengths: tuple[float, float]
    dt: float
    steps: int
    sigma: float
    mu: float
    eta: float
    delta_g: float = 0.0
    sample_every: int = 1
    clip: bool = True

    def __post_init__(self) -> None:
        if self.num_phases < 2:
            raise ValueError("at least 2 phases required")
        if len(self.shape) != 2 or len(self.domain_lengths) != 2:
            raise ValueError("multiphase solver requires a 2D shape")
        if min(self.shape) < 3:
            raise ValueError("each grid axis needs at least 3 points")
        if any(length <= 0 for length in self.domain_lengths):
            raise ValueError("domain lengths must be positive")
        if self.dt <= 0 or self.steps < 1 or self.sample_every < 1:
            raise ValueError("dt, steps, and sample_every must be positive")
        if self.sigma <= 0 or self.mu <= 0 or self.eta <= 0:
            raise ValueError("sigma, mu, and eta must be positive")
        if self.delta_g != 0.0:
            raise NotImplementedError(
                "non-zero delta_g is unsupported because the available formulations "
                "use unresolved driving-force conventions"
            )

    @property
    def spacings(self) -> tuple[float, float]:
        return tuple(length / points for length, points in zip(self.domain_lengths, self.shape))


@dataclass(frozen=True)
class MultiphaseResult:
    """Sampled MPF rollout with diagnostics."""

    times: Array
    states: Array
    phase_sums: Array
    free_energies: Array
    clipped_counts: Array


def load_multiphase_config(path: str | Path) -> tuple[MultiphasePFConfig, dict[str, Any]]:
    """Load a multiphase benchmark YAML and return config with initial-condition spec."""

    with Path(path).open(encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)

    grid = raw["grid"]
    physics = raw["physics"]
    time = raw["time"]

    eta = physics.get("eta")
    if eta is None:
        eta = float(physics["eta_grid_points"]) * (
            float(grid["domain_lengths"][0]) / int(grid["shape"][0])
        )

    config = MultiphasePFConfig(
        benchmark_id=str(raw["benchmark"]["id"]),
        num_phases=int(raw["benchmark"]["num_phases"]),
        shape=tuple(int(v) for v in grid["shape"]),
        domain_lengths=tuple(float(v) for v in grid["domain_lengths"]),
        dt=float(time["dt"]),
        steps=int(time["steps"]),
        sigma=float(physics["sigma"]),
        mu=float(physics["mu"]),
        eta=float(eta),
        delta_g=float(physics.get("delta_g", 0.0)),
        sample_every=int(time.get("sample_every", 1)),
        clip=bool(time.get("clip", True)),
    )
    return config, raw["initial_condition"]


def multiphase_I(phi_alpha: Array, config: MultiphasePFConfig) -> Array:
    """Return the interfacial operator I_alpha = laplacian(phi) + (pi^2/eta^2)*phi."""

    phi_alpha = np.asarray(phi_alpha, dtype=np.float64)
    lap = periodic_laplacian(phi_alpha, config.spacings)
    return lap + (np.pi**2 / config.eta**2) * phi_alpha


def multiphase_rhs(phi: Array, config: MultiphasePFConfig) -> Array:
    """Return dphi/dt for all phases.

    phi has shape (num_phases, Nx, Ny).  The returned array has the same shape.
    """

    phi = np.asarray(phi, dtype=np.float64)
    if phi.shape != (config.num_phases,) + config.shape:
        raise ValueError(
            f"expected phi shape {(config.num_phases,) + config.shape}, got {phi.shape}"
        )

    num_phases = config.num_phases
    I = np.stack([multiphase_I(phi[a], config) for a in range(num_phases)], axis=0)
    I_mean = I.mean(axis=0)  # mean over phases at each grid cell

    # dphi_alpha/dt = (mu * sigma) * (I_alpha - I_mean)
    # This is algebraically equivalent to the pairwise sum form in the preprint.
    rhs = config.mu * config.sigma * (I - I_mean)
    return rhs


def multiphase_free_energy(phi: Array, config: MultiphasePFConfig) -> float:
    """MPF free energy over all phase pairs (preprint §3.1, notebook cell 7).

    F = sum_{alpha > beta} (4 sigma / eta) *
        integral [ phi_alpha * phi_beta
                   - (eta^2 / pi^2) * grad(phi_alpha) . grad(phi_beta) ] dV
    """

    phi = np.asarray(phi, dtype=np.float64)
    dx, dy = config.spacings
    num_phases = config.num_phases
    F = 0.0
    for a in range(num_phases):
        gax = (np.roll(phi[a], -1, axis=0) - phi[a]) / dx
        gay = (np.roll(phi[a], -1, axis=1) - phi[a]) / dy
        for b in range(a + 1, num_phases):
            gbx = (np.roll(phi[b], -1, axis=0) - phi[b]) / dx
            gby = (np.roll(phi[b], -1, axis=1) - phi[b]) / dy
            bulk = phi[a] * phi[b]
            grad_dot = gax * gbx + gay * gby
            integrand = bulk - (config.eta**2 / np.pi**2) * grad_dot
            F += (4.0 * config.sigma / config.eta) * float(np.sum(integrand)) * dx * dy
    return F


def multiphase_step(
    phi: Array, config: MultiphasePFConfig
) -> tuple[Array, Array, int]:
    """Advance one explicit Euler step, clip, and renormalize.

    Returns (phi_new, rhs, clipped_count).
    """

    rhs = multiphase_rhs(phi, config)
    phi_new = np.asarray(phi, dtype=np.float64) + config.dt * rhs

    clipped_count = int(np.count_nonzero((phi_new < 0.0) | (phi_new > 1.0)))
    if config.clip:
        phi_new = np.clip(phi_new, 0.0, 1.0)

    # Renormalize so that sum_alpha phi_alpha = 1 at each grid cell.
    phi_sum = phi_new.sum(axis=0)
    phi_sum = np.where(phi_sum < 1e-12, 1.0, phi_sum)
    phi_new = phi_new / phi_sum[np.newaxis, :, :]

    return phi_new, rhs, clipped_count


def simulate_multiphase(
    phi_initial: Array, config: MultiphasePFConfig
) -> MultiphaseResult:
    """Roll out the MPF solver and sample diagnostics."""

    phi = np.asarray(phi_initial, dtype=np.float64).copy()
    expected_shape = (config.num_phases,) + config.shape
    if phi.shape != expected_shape:
        raise ValueError(f"expected initial shape {expected_shape}, got {phi.shape}")

    states = [phi.copy()]
    times = [0.0]
    phase_sums = [phi.sum(axis=0).copy()]
    free_energies = [multiphase_free_energy(phi, config)]
    clipped_counts = [0]

    for step in range(1, config.steps + 1):
        phi, _, clipped_count = multiphase_step(phi, config)
        if step % config.sample_every == 0 or step == config.steps:
            states.append(phi.copy())
            times.append(step * config.dt)
            phase_sums.append(phi.sum(axis=0).copy())
            free_energies.append(multiphase_free_energy(phi, config))
            clipped_counts.append(clipped_count)

    return MultiphaseResult(
        times=np.asarray(times, dtype=np.float64),
        states=np.asarray(states, dtype=np.float64),
        phase_sums=np.asarray(phase_sums, dtype=np.float64),
        free_energies=np.asarray(free_energies, dtype=np.float64),
        clipped_counts=np.asarray(clipped_counts, dtype=np.int64),
    )


def two_grain_circular(
    config: MultiphasePFConfig,
    *,
    center: tuple[float, float],
    radius: float,
) -> Array:
    """Build a two-phase field with a circular grain-1 embedded in a grain-2 matrix."""

    if config.num_phases != 2:
        raise ValueError("two_grain_circular requires exactly 2 phases")

    lx, ly = config.domain_lengths
    dx, dy = config.spacings
    nx, ny = config.shape

    xs = np.arange(nx, dtype=np.float64) * dx
    ys = np.arange(ny, dtype=np.float64) * dy
    x, y = np.meshgrid(xs, ys, indexing="ij")

    cx, cy = center
    rx = (x - cx + lx / 2.0) % lx - lx / 2.0
    ry = (y - cy + ly / 2.0) % ly - ly / 2.0
    dist = np.sqrt(rx**2 + ry**2) - radius

    inside = dist < -config.eta / 2.0
    outside = dist > config.eta / 2.0
    interface = ~inside & ~outside

    phi1 = np.empty_like(dist)
    phi1[inside] = 1.0
    phi1[outside] = 0.0
    phi1[interface] = 0.5 - 0.5 * np.sin(np.pi * dist[interface] / config.eta)

    return np.stack([phi1, 1.0 - phi1], axis=0)


def multigrain(
    config: MultiphasePFConfig,
    *,
    centers: list[tuple[float, float]],
    radii: list[float],
    rng: np.random.Generator | None = None,
) -> Array:
    """Build N-phase field from circular grains with Voronoi fill.

    Each grain alpha uses the diffuse sinusoidal interface profile within
    eta/2 of its boundary.  Cells where no grain has significant weight are
    assigned to the nearest grain center (Voronoi fill) so that the
    phase-sum constraint sum_alpha phi_alpha = 1 is satisfied everywhere.
    """

    if len(centers) != config.num_phases:
        raise ValueError("centers must have one entry per phase")
    if len(radii) != config.num_phases:
        raise ValueError("radii must have one entry per phase")

    lx, ly = config.domain_lengths
    dx, dy = config.spacings
    nx, ny = config.shape

    xs = np.arange(nx, dtype=np.float64) * dx
    ys = np.arange(ny, dtype=np.float64) * dy
    x, y = np.meshgrid(xs, ys, indexing="ij")

    # Signed distances from each grain boundary (positive = outside grain).
    signed_dist = np.empty((config.num_phases, nx, ny), dtype=np.float64)
    raw = np.zeros((config.num_phases, nx, ny), dtype=np.float64)

    for a, (center, radius) in enumerate(zip(centers, radii)):
        cx, cy = center
        rx = (x - cx + lx / 2.0) % lx - lx / 2.0
        ry = (y - cy + ly / 2.0) % ly - ly / 2.0
        signed_dist[a] = np.sqrt(rx**2 + ry**2) - radius

        inside = signed_dist[a] < -config.eta / 2.0
        interface = np.abs(signed_dist[a]) <= config.eta / 2.0

        raw[a, inside] = 1.0
        raw[a, interface] = (
            0.5 - 0.5 * np.sin(np.pi * signed_dist[a, interface] / config.eta)
        )

    phi_sum = raw.sum(axis=0)

    # Cells not covered by any grain: assign to the nearest grain center.
    void_mask = phi_sum < 1e-12
    if np.any(void_mask):
        nearest = np.argmin(signed_dist, axis=0)
        for a in range(config.num_phases):
            raw[a, void_mask & (nearest == a)] = 1.0
        phi_sum = raw.sum(axis=0)

    phi_sum = np.where(phi_sum < 1e-12, 1.0, phi_sum)
    return raw / phi_sum[np.newaxis, :, :]


def initial_multiphase_condition(
    config: MultiphasePFConfig, spec: dict
) -> Array:
    """Create an initial condition from a benchmark YAML spec."""

    kind = spec["type"]
    if kind == "two_grain_circular":
        center = tuple(float(v) for v in spec["center"])
        radius = float(spec["radius"])
        return two_grain_circular(config, center=center, radius=radius)
    if kind == "multigrain":
        centers = [tuple(float(v) for v in c) for c in spec["centers"]]
        radii = [float(r) for r in spec["radii"]]
        return multigrain(config, centers=centers, radii=radii)
    raise ValueError(f"unsupported multiphase initial condition: {kind!r}")
