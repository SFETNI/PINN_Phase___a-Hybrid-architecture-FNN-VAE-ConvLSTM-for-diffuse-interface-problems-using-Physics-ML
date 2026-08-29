"""Strict initial-field loading for the public replay path.

The replay path accepts an initial-condition archive **only** if its member set is
exactly ``{"phi0"}``. Anything else is refused before a single array is read: a
post-t0 frame, a stored trajectory, a label map, a target, a per-phase weight, or
any member that has not been reviewed.

That strictness is the point. A replay entry point that tolerated a ``labels`` or
``targets`` member could be handed a supervision-shaped archive and would not
notice. Refusing the schema outright means the public replay path cannot be fed
anything but an initial field, whatever a caller intends.

This module is deliberately narrow and additive. It does not change the historical
training loaders, which have their own reviewed schemas.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import io
import os
from pathlib import Path
import zipfile

import numpy as np

from .artifacts import _normalized_sha256, _validate_npz_container, sha256_file

#: The only member a public initial-condition archive may contain.
REQUIRED_MEMBER = "phi0"

#: Members whose presence means the archive is not an initial field. Named
#: explicitly so a refusal message can say what was wrong rather than "bad schema".
SUPERVISION_SHAPED_MEMBERS = frozenset(
    {"labels", "targets", "weights", "seeds", "phase_permutation"}
)
TRAJECTORY_SHAPED_MEMBERS = frozenset(
    {"states", "trajectory", "frames", "reference_states", "save_steps", "reference_tail"}
)

MAX_PHI0_BYTES = 1024**3


class InitialFieldSchemaError(ValueError):
    """Raised when an archive is not an exact ``{phi0}`` initial field."""


@dataclass(frozen=True)
class InitialField:
    """An initial field and nothing else.

    There is no trajectory attribute, and no attribute that could hold one. A
    caller cannot reach post-t0 state through this object because none is stored.
    """

    phi0: np.ndarray
    sha256: str
    value_sha256: str
    num_phases: int
    spatial_shape: tuple[int, ...]


def _value_digest(array: np.ndarray) -> str:
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode())
    digest.update(b"\0")
    digest.update(str(array.shape).encode())
    digest.update(b"\0")
    digest.update(np.ascontiguousarray(array).tobytes())
    return digest.hexdigest()


def _reject(path: Path, members: set[str]) -> None:
    """Explain the refusal in terms of what the archive actually carries."""
    supervision = sorted(members & SUPERVISION_SHAPED_MEMBERS)
    trajectory = sorted(members & TRAJECTORY_SHAPED_MEMBERS)
    if supervision:
        raise InitialFieldSchemaError(
            f"{path}: refused before reading any array. This archive carries "
            f"supervision-shaped members {supervision}; the public replay path accepts "
            f"an initial field only, whose member set must be exactly ['{REQUIRED_MEMBER}']."
        )
    if trajectory:
        raise InitialFieldSchemaError(
            f"{path}: refused before reading any array. This archive carries "
            f"trajectory-shaped members {trajectory}, so it is a reference or rollout "
            f"rather than an initial field. The public replay path accepts exactly "
            f"['{REQUIRED_MEMBER}']."
        )
    if REQUIRED_MEMBER not in members:
        raise InitialFieldSchemaError(
            f"{path}: refused before reading any array. No '{REQUIRED_MEMBER}' member is "
            f"present; found {sorted(members)}."
        )
    raise InitialFieldSchemaError(
        f"{path}: refused before reading any array. Unreviewed members "
        f"{sorted(members - {REQUIRED_MEMBER})} accompany '{REQUIRED_MEMBER}'; the member "
        f"set must be exactly ['{REQUIRED_MEMBER}']."
    )


def archive_members(path: str | Path) -> set[str]:
    """Member names of an archive, read from the container without decoding arrays."""
    with zipfile.ZipFile(Path(path)) as archive:
        return {Path(name).stem for name in archive.namelist()}


def load_initial_field(
    path: str | Path,
    *,
    expected_sha256: str | None = None,
    expected_num_phases: int | None = None,
) -> InitialField:
    """Load an initial field, refusing anything that is not exactly ``{phi0}``.

    The digest, when supplied, is checked before deserialization. The member set is
    checked before any array is decoded.
    """

    archive_path = Path(path)
    if archive_path.stat().st_size > MAX_PHI0_BYTES:
        raise InitialFieldSchemaError(f"initial field exceeds the size limit: {archive_path}")

    digest = sha256_file(archive_path)
    if expected_sha256 is not None:
        expected = _normalized_sha256(expected_sha256)
        if digest != expected:
            raise ValueError(
                f"SHA-256 mismatch for {archive_path}: expected {expected}, got {digest}"
            )

    members = archive_members(archive_path)
    if members != {REQUIRED_MEMBER}:
        _reject(archive_path, members)

    with archive_path.open("rb") as stream:
        _validate_npz_container(stream, path=archive_path)
        with np.load(stream, allow_pickle=False) as loaded:
            if set(loaded.files) != {REQUIRED_MEMBER}:  # pragma: no cover - re-checked
                _reject(archive_path, set(loaded.files))
            phi0 = np.asarray(loaded[REQUIRED_MEMBER]).copy()

    if phi0.ndim not in (3, 4):
        raise InitialFieldSchemaError(
            f"{archive_path}: phi0 must be [N,H,W] or [N,D,H,W]; got shape {phi0.shape}"
        )
    if expected_num_phases is not None and phi0.shape[0] != expected_num_phases:
        raise InitialFieldSchemaError(
            f"{archive_path}: phi0 carries {phi0.shape[0]} phases, "
            f"expected {expected_num_phases}"
        )
    return InitialField(
        phi0=phi0,
        sha256=digest,
        value_sha256=_value_digest(phi0),
        num_phases=int(phi0.shape[0]),
        spatial_shape=tuple(int(size) for size in phi0.shape[1:]),
    )


def admissibility(phi0: np.ndarray) -> dict[str, float]:
    """Bounds and phase-sum report for an initial field. Diagnostic only."""
    sums = phi0.sum(axis=0)
    return {
        "phi_min": float(phi0.min()),
        "phi_max": float(phi0.max()),
        "phase_sum_min": float(sums.min()),
        "phase_sum_max": float(sums.max()),
        "max_abs_phase_sum_error": float(np.abs(sums - 1.0).max()),
    }
