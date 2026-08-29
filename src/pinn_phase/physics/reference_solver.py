"""Deterministic scalar Allen-Cahn reference solver.

The equation follows ``Phase_Field/dual_phases_64_ref.ipynb``:

    dphi/dt = mu * (
        sigma * (laplacian(phi) + pi**2 / (2 * eta**2) * (2 * phi - 1))
        + pi / eta * sqrt(phi * (1 - phi)) * delta_g
    )

The solver is intentionally small. It establishes trusted scalar benchmarks
before the recovery project introduces neural models or multiphase coupling.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import yaml


Array = np.ndarray


@dataclass(frozen=True)
class ScalarAllenCahnConfig:
    """Physical and numerical parameters for a periodic scalar benchmark."""

    benchmark_id: str
    shape: tuple[int, ...]
    domain_lengths: tuple[float, ...]
    dt: float
    steps: int
    sigma: float
    mu: float
    eta: float
    delta_g: float = 0.0
    sample_every: int = 1
    clip: bool = True

    def __post_init__(self) -> None:
        if len(self.shape) not in (1, 2, 3):
            raise ValueError("scalar reference solver supports 1D, 2D, and 3D grids")
        if len(self.domain_lengths) != len(self.shape):
            raise ValueError("domain_lengths must match shape dimensionality")
        if min(self.shape) < 3:
            raise ValueError("each grid axis needs at least three points")
        if any(length <= 0 for length in self.domain_lengths):
            raise ValueError("domain lengths must be positive")
        if self.dt <= 0 or self.steps < 1 or self.sample_every < 1:
            raise ValueError("dt, steps, and sample_every must be positive")
        if self.sigma <= 0 or self.mu <= 0 or self.eta <= 0:
            raise ValueError("sigma, mu, and eta must be positive")

    @property
    def spacings(self) -> tuple[float, ...]:
        """Return periodic grid spacings for endpoint-exclusive coordinates."""

        return tuple(length / points for length, points in zip(self.domain_lengths, self.shape))

    def with_delta_g(self, delta_g: float) -> "ScalarAllenCahnConfig":
        return replace(self, delta_g=float(delta_g))


@dataclass(frozen=True)
class SimulationResult:
    """Sampled scalar rollout and diagnostics."""

    times: Array
    states: Array
    energies: Array
    clipped_counts: Array


def load_benchmark_config(path: str | Path) -> tuple[ScalarAllenCahnConfig, dict[str, Any]]:
    """Load a benchmark YAML file and retain its initialization settings."""

    config_path = Path(path)
    with config_path.open(encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)

    grid = raw["grid"]
    physics = raw["physics"]
    time = raw["time"]
    eta = physics.get("eta")
    if eta is None:
        eta = float(physics["eta_grid_points"]) * (
            float(grid["domain_lengths"][0]) / int(grid["shape"][0])
        )

    config = ScalarAllenCahnConfig(
        benchmark_id=str(raw["benchmark"]["id"]),
        shape=tuple(int(value) for value in grid["shape"]),
        domain_lengths=tuple(float(value) for value in grid["domain_lengths"]),
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


def periodic_laplacian(phi: Array, spacings: Iterable[float]) -> Array:
    """Compute an N-dimensional second-order periodic finite-difference Laplacian."""

    phi = np.asarray(phi, dtype=np.float64)
    spacing_values = tuple(float(value) for value in spacings)
    if phi.ndim != len(spacing_values):
        raise ValueError("number of spacings must match phi dimensionality")

    laplacian = np.zeros_like(phi)
    for axis, spacing in enumerate(spacing_values):
        laplacian += (
            np.roll(phi, shift=1, axis=axis)
            + np.roll(phi, shift=-1, axis=axis)
            - 2.0 * phi
        ) / spacing**2
    return laplacian


def driving_shape(phi: Array, eta: float) -> Array:
    """Return the benchmark-specific external driving-force multiplier."""

    bounded_phi = np.clip(np.asarray(phi, dtype=np.float64), 0.0, 1.0)
    return np.pi / eta * np.sqrt(bounded_phi * (1.0 - bounded_phi))


def driving_potential(phi: Array, eta: float) -> Array:
    """Return a primitive of ``driving_shape`` for energy diagnostics."""

    bounded_phi = np.clip(np.asarray(phi, dtype=np.float64), 0.0, 1.0)
    root = np.sqrt(bounded_phi * (1.0 - bounded_phi))
    u = 2.0 * bounded_phi - 1.0
    integral = 0.25 * u * root + 0.125 * np.arcsin(u)
    return np.pi / eta * integral


def scalar_rhs(phi: Array, config: ScalarAllenCahnConfig) -> Array:
    """Evaluate the unconstrained scalar Allen-Cahn right-hand side."""

    phi = np.asarray(phi, dtype=np.float64)
    if phi.shape != config.shape:
        raise ValueError(f"expected phi shape {config.shape}, received {phi.shape}")

    laplacian = periodic_laplacian(phi, config.spacings)
    bulk = np.pi**2 / (2.0 * config.eta**2) * (2.0 * phi - 1.0)
    return config.mu * (
        config.sigma * (laplacian + bulk)
        + driving_shape(phi, config.eta) * config.delta_g
    )


def explicit_euler_step(phi: Array, config: ScalarAllenCahnConfig) -> tuple[Array, Array, int]:
    """Advance one step and report the unconstrained RHS and clip count."""

    rhs = scalar_rhs(phi, config)
    unconstrained = np.asarray(phi, dtype=np.float64) + config.dt * rhs
    clipped_count = int(np.count_nonzero((unconstrained < 0.0) | (unconstrained > 1.0)))
    if config.clip:
        return np.clip(unconstrained, 0.0, 1.0), rhs, clipped_count
    return unconstrained, rhs, clipped_count


def free_energy(phi: Array, config: ScalarAllenCahnConfig) -> float:
    """Compute the discrete free energy consistent with ``scalar_rhs``."""

    phi = np.asarray(phi, dtype=np.float64)
    gradients = [
        (np.roll(phi, shift=-1, axis=axis) - phi) / spacing
        for axis, spacing in enumerate(config.spacings)
    ]
    gradient_squared = sum(gradient**2 for gradient in gradients)
    interfacial = config.sigma * (
        0.5 * gradient_squared
        + np.pi**2 / (2.0 * config.eta**2) * phi * (1.0 - phi)
    )
    chemical = -config.delta_g * driving_potential(phi, config.eta)
    cell_volume = float(np.prod(config.spacings))
    return float(np.sum(interfacial + chemical) * cell_volume)


def simulate(phi_initial: Array, config: ScalarAllenCahnConfig) -> SimulationResult:
    """Roll out explicit Euler updates and sample deterministic diagnostics."""

    phi = np.asarray(phi_initial, dtype=np.float64).copy()
    if phi.shape != config.shape:
        raise ValueError(f"expected initial shape {config.shape}, received {phi.shape}")

    states = [phi.copy()]
    times = [0.0]
    energies = [free_energy(phi, config)]
    clipped_counts = [0]
    for step in range(1, config.steps + 1):
        phi, _, clipped_count = explicit_euler_step(phi, config)
        if step % config.sample_every == 0 or step == config.steps:
            states.append(phi.copy())
            times.append(step * config.dt)
            energies.append(free_energy(phi, config))
            clipped_counts.append(clipped_count)

    return SimulationResult(
        times=np.asarray(times, dtype=np.float64),
        states=np.asarray(states, dtype=np.float64),
        energies=np.asarray(energies, dtype=np.float64),
        clipped_counts=np.asarray(clipped_counts, dtype=np.int64),
    )


def _diffuse_inside_profile(signed_distance: Array, eta: float) -> Array:
    """Build the notebook's piecewise-sine diffuse interface profile."""

    signed_distance = np.asarray(signed_distance, dtype=np.float64)
    profile = np.empty_like(signed_distance)
    profile[signed_distance < -eta / 2.0] = 1.0
    profile[signed_distance > eta / 2.0] = 0.0
    interface = np.abs(signed_distance) <= eta / 2.0
    profile[interface] = 0.5 - 0.5 * np.sin(np.pi * signed_distance[interface] / eta)
    return profile


def planar_slab(config: ScalarAllenCahnConfig, *, center: float, half_width: float) -> Array:
    """Create a periodic 1D phase-one slab with two flat diffuse interfaces."""

    if len(config.shape) != 1:
        raise ValueError("planar_slab requires a 1D config")
    (length,) = config.domain_lengths
    coordinates = np.arange(config.shape[0], dtype=np.float64) * config.spacings[0]
    periodic_distance = np.abs((coordinates - center + length / 2.0) % length - length / 2.0)
    return _diffuse_inside_profile(periodic_distance - half_width, config.eta)


def circular_grain(
    config: ScalarAllenCahnConfig, *, center: tuple[float, float], radius: float
) -> Array:
    """Create a periodic 2D circular grain using the notebook's interface profile."""

    if len(config.shape) != 2:
        raise ValueError("circular_grain requires a 2D config")
    axes = [
        np.arange(points, dtype=np.float64) * spacing
        for points, spacing in zip(config.shape, config.spacings)
    ]
    x, y = np.meshgrid(*axes, indexing="ij")
    lx, ly = config.domain_lengths
    dx = (x - center[0] + lx / 2.0) % lx - lx / 2.0
    dy = (y - center[1] + ly / 2.0) % ly - ly / 2.0
    return _diffuse_inside_profile(np.sqrt(dx**2 + dy**2) - radius, config.eta)


def multi_circular_scalar(
    config: ScalarAllenCahnConfig,
    *,
    centers: list[tuple[float, float]],
    radii: list[float],
) -> Array:
    """Combine multiple circular scalar grains via element-wise max.

    phi = max(phi_grain_i) preserves each grain's sinusoidal diffuse-interface
    profile without creating a spurious merged bulk when grains are close but
    not touching.  Requires at least two grains and a 2D config.
    """

    if len(config.shape) != 2:
        raise ValueError("multi_circular_scalar requires a 2D config")
    if len(centers) < 2:
        raise ValueError("multi_circular_scalar requires at least two grains")
    if len(centers) != len(radii):
        raise ValueError("centers and radii must have the same length")

    phi = np.zeros(config.shape, dtype=np.float64)
    for center, radius in zip(centers, radii):
        phi_grain = circular_grain(
            config,
            center=tuple(float(v) for v in center),
            radius=float(radius),
        )
        phi = np.maximum(phi, phi_grain)
    return phi


def equivalent_radius(phi: Array, spacings: Iterable[float]) -> float:
    """Return the mass-equivalent radius of a 2D scalar grain."""

    spacing_values = tuple(float(value) for value in spacings)
    if np.asarray(phi).ndim != 2 or len(spacing_values) != 2:
        raise ValueError("equivalent_radius requires a 2D field and two spacings")
    area = float(np.sum(phi) * np.prod(spacing_values))
    return float(np.sqrt(max(area, 0.0) / np.pi))


def equivalent_radius_3d(phi: Array, spacings: Iterable[float]) -> float:
    """Return the volume-equivalent sphere radius of a 3D scalar grain."""

    spacing_values = tuple(float(value) for value in spacings)
    if np.asarray(phi).ndim != 3 or len(spacing_values) != 3:
        raise ValueError("equivalent_radius_3d requires a 3D field and three spacings")
    volume = float(np.sum(phi) * np.prod(spacing_values))
    return float((max(volume, 0.0) * 3.0 / (4.0 * np.pi)) ** (1.0 / 3.0))


def spherical_grain(
    config: ScalarAllenCahnConfig,
    *,
    center: tuple[float, float, float],
    radius: float,
) -> Array:
    """Create a periodic 3D spherical grain using the notebook's interface profile."""

    if len(config.shape) != 3:
        raise ValueError("spherical_grain requires a 3D config")
    axes = [
        np.arange(points, dtype=np.float64) * spacing
        for points, spacing in zip(config.shape, config.spacings)
    ]
    x, y, z = np.meshgrid(*axes, indexing="ij")
    lx, ly, lz = config.domain_lengths
    dx = (x - center[0] + lx / 2.0) % lx - lx / 2.0
    dy = (y - center[1] + ly / 2.0) % ly - ly / 2.0
    dz = (z - center[2] + lz / 2.0) % lz - lz / 2.0
    return _diffuse_inside_profile(np.sqrt(dx**2 + dy**2 + dz**2) - radius, config.eta)


def generate_multigrain_placement(
    config: ScalarAllenCahnConfig,
    *,
    seed: int = 42,
    num_grains: int = 35,
    mean_radius_px: float = 12.0,
    std_radius_px: float = 2.0,
    min_gap_px: float = 5.0,
    max_attempts: int = 10000,
) -> tuple[list[tuple[float, float]], list[float]]:
    """Return (centers_phys, radii_phys) for a deterministic random multigrain layout.

    Uses rejection sampling with the periodic minimum centre-to-centre distance:
        dist_px >= R_i_px + R_j_px + min_gap_px

    Grains are sorted largest-first before placement so the tightest grains
    claim space early and smaller ones fill in.  The number actually placed may
    be less than ``num_grains`` if the domain is too full.  Check
    ``len(centers_phys)`` in the caller to know how many were placed.
    """

    if len(config.shape) != 2:
        raise ValueError("generate_multigrain_placement requires a 2D config")

    rng = np.random.default_rng(seed)
    Nx, Ny = config.shape
    Lx, Ly = config.domain_lengths
    dx = config.spacings[0]

    min_radius_px = max(3.0, std_radius_px)
    max_radius_px = min(Nx, Ny) / 4.0

    # Collect target number of valid radii from truncated normal
    radii_px: list[float] = []
    for _ in range(num_grains * 200):
        if len(radii_px) >= num_grains:
            break
        r = rng.normal(mean_radius_px, std_radius_px)
        if min_radius_px <= r <= max_radius_px:
            radii_px.append(float(r))

    # Largest grains first — easier to place, reduces fragmentation
    radii_px.sort(reverse=True)

    centers_px: list[tuple[float, float]] = []
    placed_radii_px: list[float] = []

    for r_px in radii_px:
        for _ in range(max_attempts):
            cx = float(rng.uniform(0.0, float(Nx)))
            cy = float(rng.uniform(0.0, float(Ny)))
            ok = True
            for (pcx, pcy), pr_px in zip(centers_px, placed_radii_px):
                ddx = ((cx - pcx + Nx / 2.0) % Nx) - Nx / 2.0
                ddy = ((cy - pcy + Ny / 2.0) % Ny) - Ny / 2.0
                if (ddx * ddx + ddy * ddy) < (r_px + pr_px + min_gap_px) ** 2:
                    ok = False
                    break
            if ok:
                centers_px.append((cx, cy))
                placed_radii_px.append(r_px)
                break

    centers_phys = [(cx * dx, cy * dx) for cx, cy in centers_px]
    radii_phys = [r * dx for r in placed_radii_px]
    return centers_phys, radii_phys


def scalar_multigrain_random(
    config: ScalarAllenCahnConfig,
    *,
    seed: int = 42,
    num_grains: int = 35,
    mean_radius_px: float = 12.0,
    std_radius_px: float = 2.0,
    min_gap_px: float = 5.0,
    max_attempts: int = 10000,
) -> Array:
    """Build a random multi-grain scalar φ field via element-wise max composition.

    Placement is deterministic given the ``seed``.  The composition rule
    ``phi = max(phi_i)`` preserves each grain's sinusoidal diffuse-interface
    profile without creating spurious merged bulk in close but non-overlapping
    gaps.  Requires a 2D config.
    """

    centers_phys, radii_phys = generate_multigrain_placement(
        config,
        seed=seed,
        num_grains=num_grains,
        mean_radius_px=mean_radius_px,
        std_radius_px=std_radius_px,
        min_gap_px=min_gap_px,
        max_attempts=max_attempts,
    )

    if len(centers_phys) == 0:
        raise RuntimeError(
            "scalar_multigrain_random: could not place any grains with the given parameters"
        )

    phi = np.zeros(config.shape, dtype=np.float64)
    for center, radius in zip(centers_phys, radii_phys):
        phi_grain = circular_grain(config, center=center, radius=radius)
        phi = np.maximum(phi, phi_grain)
    return phi


def initial_condition_from_config(
    config: ScalarAllenCahnConfig, initial_condition: dict[str, Any]
) -> Array:
    """Create a supported scalar initial condition from benchmark YAML."""

    kind = initial_condition["type"]
    if kind == "planar_slab":
        return planar_slab(
            config,
            center=float(initial_condition["center"]),
            half_width=float(initial_condition["half_width"]),
        )
    if kind == "circular_grain":
        center = tuple(float(value) for value in initial_condition["center"])
        return circular_grain(config, center=center, radius=float(initial_condition["radius"]))
    if kind == "multi_circular":
        centers = [tuple(float(v) for v in c) for c in initial_condition["centers"]]
        radii = [float(r) for r in initial_condition["radii"]]
        return multi_circular_scalar(config, centers=centers, radii=radii)
    if kind == "scalar_multigrain_random":
        return scalar_multigrain_random(
            config,
            seed=int(initial_condition.get("seed", 42)),
            num_grains=int(initial_condition.get("num_grains", 35)),
            mean_radius_px=float(initial_condition.get("mean_radius_px", 12.0)),
            std_radius_px=float(initial_condition.get("std_radius_px", 2.0)),
            min_gap_px=float(initial_condition.get("min_gap_px", 5.0)),
            max_attempts=int(initial_condition.get("max_attempts", 10000)),
        )
    if kind == "spherical_grain":
        center = tuple(float(value) for value in initial_condition["center"])
        return spherical_grain(config, center=center, radius=float(initial_condition["radius"]))
    raise ValueError(f"unsupported scalar initial condition: {kind}")
