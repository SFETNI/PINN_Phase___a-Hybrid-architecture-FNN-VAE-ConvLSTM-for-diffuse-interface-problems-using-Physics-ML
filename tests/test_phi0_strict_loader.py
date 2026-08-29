"""The public replay path must accept exactly ``{phi0}`` and refuse everything else.

Each refusal test asserts two things: that the archive is rejected, and that it is
rejected *before any array is read*. The second matters more than the first. A
loader that decodes a ``labels`` array and then complains has already put
supervision-shaped data in memory on a replay path; one that refuses on the member
set never touches it.

The archives built here are real ``.npz`` files, not mocks, so the guard is
exercised through the same container code the shipped artifacts go through.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from pinn_phase.io.phi0 import (
    InitialField,
    InitialFieldSchemaError,
    admissibility,
    archive_members,
    load_initial_field,
)

ROOT = Path(__file__).resolve().parents[1]
DISTRIBUTED = sorted((ROOT / "benchmarks/initial_conditions").glob("*.npz"))
DISTRIBUTED_IDS = [path.stem for path in DISTRIBUTED]


@pytest.fixture
def phi0() -> np.ndarray:
    field = np.zeros((3, 8, 8), dtype=np.float32)
    field[0] = 1.0
    return field


def _write(path: Path, **members: np.ndarray) -> Path:
    np.savez_compressed(path, **members)
    return path


# --------------------------------------------------------------------------
# Acceptance
# --------------------------------------------------------------------------


def test_exact_phi0_schema_is_accepted(tmp_path: Path, phi0: np.ndarray) -> None:
    path = _write(tmp_path / "ok.npz", phi0=phi0)
    field = load_initial_field(path)
    assert isinstance(field, InitialField)
    assert field.num_phases == 3
    assert field.spatial_shape == (8, 8)
    np.testing.assert_array_equal(field.phi0, phi0)


def test_initial_field_object_cannot_carry_a_trajectory() -> None:
    """Structural guard: there is no attribute a post-t0 frame could occupy."""
    fields = set(InitialField.__dataclass_fields__)
    assert "phi0" in fields
    assert not (fields & {"states", "trajectory", "frames", "reference_states", "labels"})


# --------------------------------------------------------------------------
# Refusals -- one per forbidden shape named in the review brief
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "extra",
    ["labels", "targets", "weights", "seeds", "phase_permutation"],
    ids=lambda name: f"supervision_{name}",
)
def test_supervision_shaped_member_is_refused(tmp_path: Path, phi0: np.ndarray, extra: str) -> None:
    path = _write(tmp_path / f"{extra}.npz", phi0=phi0, **{extra: np.zeros((3,), dtype=np.int64)})
    with pytest.raises(InitialFieldSchemaError, match="supervision-shaped"):
        load_initial_field(path)


@pytest.mark.parametrize(
    "extra",
    ["states", "trajectory", "frames", "reference_states", "save_steps", "reference_tail"],
    ids=lambda name: f"trajectory_{name}",
)
def test_trajectory_shaped_member_is_refused(tmp_path: Path, phi0: np.ndarray, extra: str) -> None:
    path = _write(tmp_path / f"{extra}.npz", phi0=phi0, **{extra: np.zeros((2,), dtype=np.int64)})
    with pytest.raises(InitialFieldSchemaError, match="trajectory-shaped"):
        load_initial_field(path)


def test_post_t0_frames_are_refused(tmp_path: Path, phi0: np.ndarray) -> None:
    """A stored trajectory beside the initial field is the exact leak to prevent."""
    path = _write(tmp_path / "traj.npz", phi0=phi0, states=np.zeros((13, 3, 8, 8), dtype=np.float32))
    with pytest.raises(InitialFieldSchemaError, match="reference or rollout"):
        load_initial_field(path)


def test_missing_phi0_is_refused(tmp_path: Path) -> None:
    path = _write(tmp_path / "nophi0.npz", fields=np.zeros((3, 8, 8), dtype=np.float32))
    with pytest.raises(InitialFieldSchemaError, match="No 'phi0' member"):
        load_initial_field(path)


def test_unreviewed_extra_member_is_refused(tmp_path: Path, phi0: np.ndarray) -> None:
    path = _write(tmp_path / "extra.npz", phi0=phi0, provenance_json=np.array(["x"]))
    with pytest.raises(InitialFieldSchemaError, match="Unreviewed members"):
        load_initial_field(path)


def test_the_historical_cohort_schema_is_refused(tmp_path: Path, phi0: np.ndarray) -> None:
    """The real prospective-cohort t0 schema must not be loadable by the replay path.

    Those archives carry fields/labels/seeds/targets/weights/phase_permutation. The
    distributed initial fields were derived from them by extracting only the field.
    """
    path = _write(
        tmp_path / "cohort.npz",
        fields=phi0,
        labels=np.zeros((8, 8), dtype=np.int64),
        seeds=np.zeros((3, 2), dtype=np.int64),
        targets=np.zeros((3,), dtype=np.float64),
        weights=np.zeros((3,), dtype=np.float64),
        phase_permutation=np.arange(3, dtype=np.int16),
    )
    with pytest.raises(InitialFieldSchemaError):
        load_initial_field(path)


def test_refusal_happens_before_any_array_is_read(tmp_path: Path, phi0: np.ndarray,
                                                  monkeypatch: pytest.MonkeyPatch) -> None:
    """If np.load is ever called on a bad schema, this test fails."""
    path = _write(tmp_path / "bad.npz", phi0=phi0, labels=np.zeros((8, 8), dtype=np.int64))

    def explode(*args: object, **kwargs: object) -> None:
        raise AssertionError("np.load was reached on an archive that must be refused first")

    monkeypatch.setattr(np, "load", explode)
    with pytest.raises(InitialFieldSchemaError):
        load_initial_field(path)


def test_digest_mismatch_is_refused_before_deserialization(tmp_path: Path, phi0: np.ndarray) -> None:
    path = _write(tmp_path / "ok.npz", phi0=phi0)
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        load_initial_field(path, expected_sha256="0" * 64)


def test_phase_count_mismatch_is_refused(tmp_path: Path, phi0: np.ndarray) -> None:
    path = _write(tmp_path / "ok.npz", phi0=phi0)
    with pytest.raises(InitialFieldSchemaError, match="expected 25"):
        load_initial_field(path, expected_num_phases=25)


# --------------------------------------------------------------------------
# The distributed artifacts themselves
# --------------------------------------------------------------------------


def test_initial_fields_are_distributed() -> None:
    assert DISTRIBUTED, "no initial fields are distributed"


@pytest.mark.parametrize("path", DISTRIBUTED, ids=DISTRIBUTED_IDS)
def test_distributed_initial_field_has_exactly_the_phi0_schema(path: Path) -> None:
    assert archive_members(path) == {"phi0"}, (
        f"{path.name} carries {sorted(archive_members(path))}, not exactly ['phi0']"
    )


@pytest.mark.parametrize("path", DISTRIBUTED, ids=DISTRIBUTED_IDS)
def test_distributed_initial_field_loads_and_is_admissible(path: Path) -> None:
    field = load_initial_field(path)
    report = admissibility(field.phi0)
    assert report["phi_min"] >= 0.0
    assert report["phi_max"] <= 1.0
    assert report["max_abs_phase_sum_error"] < 1e-5, (
        f"{path.name}: phase sum deviates by {report['max_abs_phase_sum_error']}"
    )
