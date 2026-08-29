"""Short-run memory and runtime profiling for clean rollout models."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import gc
import os
from time import perf_counter
import warnings

import torch
from torch import Tensor

from pinn_phase.models import LocalConvGRURollout

try:
    import resource
except ImportError:  # pragma: no cover - exercised only on Windows
    resource = None


MIB = 1024.0**2


@dataclass(frozen=True)
class RolloutProfile:
    """Measured resource envelope for one inference rollout."""

    label: str
    device: str
    grid_height: int
    grid_width: int
    steps: int
    hidden_channels: int
    wall_seconds: float
    rss_before_mib: float
    rss_with_output_mib: float
    rss_after_cleanup_mib: float
    process_peak_rss_mib: float
    live_tensor_before_mib: float
    live_tensor_with_output_mib: float
    live_tensor_after_cleanup_mib: float
    cuda_peak_allocated_mib: float
    cuda_peak_reserved_mib: float

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serializable representation."""

        return asdict(self)


def current_rss_mib() -> float:
    """Return current process RSS in MiB.

    Reads `/proc/self/statm` on Linux. Where that is unavailable and the
    POSIX `resource` module is present (e.g. macOS), falls back to the
    process high-water mark, matching prior behavior on those platforms.
    Where `resource` is unavailable (Windows), reports the current working
    set size via `psutil` instead of substituting the peak value.
    """

    try:
        resident_pages = int(open("/proc/self/statm", encoding="utf-8").read().split()[1])
        return resident_pages * os.sysconf("SC_PAGE_SIZE") / MIB
    except (FileNotFoundError, IndexError, OSError, ValueError):
        if resource is not None:
            return peak_rss_mib()
        return _windows_current_rss_mib()


def peak_rss_mib() -> float:
    """Return process high-water RSS in MiB.

    Uses POSIX `getrusage` where the `resource` module is available;
    uses the Windows peak working-set size via `psutil` otherwise.
    """

    if resource is not None:
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
    return _windows_peak_rss_mib()


def _windows_current_rss_mib() -> float:
    """Return the current Windows working-set size in MiB via `psutil`."""

    import psutil

    return psutil.Process(os.getpid()).memory_info().rss / MIB


def _windows_peak_rss_mib() -> float:
    """Return the peak Windows working-set size in MiB via `psutil`."""

    import psutil

    memory_info = psutil.Process(os.getpid()).memory_info()
    return float(getattr(memory_info, "peak_wset", memory_info.rss)) / MIB


def live_tensor_mib() -> float:
    """Return storage size for tensors currently visible to Python GC."""

    total_bytes = 0
    for candidate in gc.get_objects():
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", FutureWarning)
                if isinstance(candidate, Tensor):
                    total_bytes += candidate.numel() * candidate.element_size()
        except RuntimeError:
            continue
    return total_bytes / MIB


def profile_rollout(
    model: LocalConvGRURollout,
    initial_phase: Tensor,
    *,
    label: str,
    steps: int,
    hidden_channels: int,
) -> RolloutProfile:
    """Measure one no-gradient rollout and clean its output afterward."""

    if steps < 1:
        raise ValueError("steps must be positive")
    device = initial_phase.device
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)

    rss_before = current_rss_mib()
    tensor_before = live_tensor_mib()
    started = perf_counter()
    with torch.no_grad():
        output = model.rollout(initial_phase, steps=steps)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    wall_seconds = perf_counter() - started
    rss_with_output = current_rss_mib()
    tensor_with_output = live_tensor_mib()
    cuda_allocated = (
        torch.cuda.max_memory_allocated(device) / MIB if device.type == "cuda" else 0.0
    )
    cuda_reserved = (
        torch.cuda.max_memory_reserved(device) / MIB if device.type == "cuda" else 0.0
    )

    height, width = initial_phase.shape[-2:]
    del output
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return RolloutProfile(
        label=label,
        device=str(device),
        grid_height=height,
        grid_width=width,
        steps=steps,
        hidden_channels=hidden_channels,
        wall_seconds=wall_seconds,
        rss_before_mib=rss_before,
        rss_with_output_mib=rss_with_output,
        rss_after_cleanup_mib=current_rss_mib(),
        process_peak_rss_mib=peak_rss_mib(),
        live_tensor_before_mib=tensor_before,
        live_tensor_with_output_mib=tensor_with_output,
        live_tensor_after_cleanup_mib=live_tensor_mib(),
        cuda_peak_allocated_mib=cuda_allocated,
        cuda_peak_reserved_mib=cuda_reserved,
    )
