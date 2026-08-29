"""No-reference-leakage guarantee, checked against the shipped public API.

An earlier source variant invoked a config-linting script that is
not shipped here. Rather than drop the concept entirely, this verifies the
same reference-leakage guarantee directly against the shipped public API:

  1. ``pinn_phase.training.assert_no_reference_leakage`` /
     ``ReferenceLeakageError`` (also exercised in test_explicit_mpf_trainer.py)
     reject a config that requests reference frames after t=0.
  2. ``pinn_phase.physics.explicit_mpf.load_explicit_mpf_initial_reference``
     structurally can only ever expose ``phi0`` (a single frame) -- it has no
     ``states``/trajectory attribute at all, so no caller of the public
     loader API can accidentally receive post-t0 reference frames.
"""

from __future__ import annotations

import copy
import hashlib
from pathlib import Path

import numpy as np
import pytest

from pinn_phase.training import ReferenceLeakageError, assert_no_reference_leakage
from pinn_phase.physics.explicit_mpf import (
    ExplicitMPFReference,
    load_explicit_mpf_initial_reference,
)


def _clean_policy_settings() -> dict:
    """A configuration that declares the complete reviewed reference-usage contract.

    The guard requires the full key set: a policy that is silent about a channel has
    not declared that channel closed. This mirrors the accepted explicit-MPF
    training configurations recorded in docs/TRAINING_PATH_DISCLOSURE.json.
    """
    return {
        "reference_usage_policy": {
            "training_initial_condition": "Phi_ref_t0_only",
            "training_uses_reference_frames_after_t0": False,
            "reference_after_t0_use": "audit_only",
            "audit_uses_reference_frames_after_t0": True,
            "graph_features_from": "model_phi_only",
            "latent_targets_from_reference_after_t0": False,
            "sampler_targets_from_reference_after_t0": False,
        },
        "training_policy": {
            "supervision": "initial_condition_only",
            "no_training_labels_after_t0": True,
            "no_reference_graph_after_t0": True,
            "no_reference_latent_targets_after_t0": True,
        },
    }


def test_no_reference_leakage_passes_clean_policy() -> None:
    assert_no_reference_leakage(_clean_policy_settings())  # must not raise


def test_no_reference_leakage_rejects_training_reference_frames_after_t0() -> None:
    bad = copy.deepcopy(_clean_policy_settings())
    bad["reference_usage_policy"]["training_uses_reference_frames_after_t0"] = True
    with pytest.raises(ReferenceLeakageError):
        assert_no_reference_leakage(bad)


def test_explicit_mpf_reference_dataclass_exposes_only_phi0() -> None:
    """Structural guard: the reference object returned by the public loader
    API has no attribute that could carry a post-t0 trajectory."""
    fields = {f for f in ExplicitMPFReference.__dataclass_fields__}
    assert "phi0" in fields
    forbidden_names = {"states", "trajectory", "frames", "reference_states"}
    assert not (fields & forbidden_names), (
        f"ExplicitMPFReference unexpectedly exposes trajectory-like field(s): {fields & forbidden_names}"
    )


def _t0_config(path: Path) -> dict:
    return {
        "benchmark": {"id": "t0-contract-test"},
        "validation": {
            "training_allowed": False,
            "reference_use": "audit_only",
            "training_t0_artifact": str(path),
            "training_t0_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        },
        "physics": {
            "num_phases": 3,
            "eta_px": 4.0,
            "sigma": 1.0,
            "mu": 1.0,
            "dt_mu_sigma": 0.1,
        },
        "grid": {"shape": [4, 5]},
        "time": {"steps": 12},
    }


def test_initial_loader_reads_only_the_phi0_member(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    accesses: list[str] = []
    phi0 = np.full((3, 4, 5), 1.0 / 3.0, dtype=np.float32)
    artifact = tmp_path / "t0.npz"
    artifact.write_bytes(b"placeholder")

    class InitialOnlyArchive:
        files = ["phi0"]

        def __enter__(self) -> "InitialOnlyArchive":
            return self

        def __exit__(self, *args: object) -> None:
            return None

        def __getitem__(self, key: str) -> np.ndarray:
            accesses.append(key)
            if key != "phi0":
                raise AssertionError(f"unexpected archive member access: {key}")
            return phi0

    import pinn_phase.io.artifacts as artifacts

    monkeypatch.setitem(
        artifacts.load_npz_arrays.__globals__,
        "_validate_npz_container",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(np, "load", lambda *args, **kwargs: InitialOnlyArchive())
    cfg_path = tmp_path / "config.yaml"
    import yaml

    cfg_path.write_text(yaml.safe_dump(_t0_config(artifact)), encoding="utf-8")
    loaded = load_explicit_mpf_initial_reference(cfg_path, repo_root=tmp_path)
    assert accesses == ["phi0"]
    np.testing.assert_array_equal(loaded.phi0, phi0)


def test_initial_loader_rejects_trajectory_schema_before_array_access(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    artifact = tmp_path / "trajectory.npz"
    artifact.write_bytes(b"placeholder")

    class TrajectoryArchive:
        files = ["phi0", "states"]

        def __enter__(self) -> "TrajectoryArchive":
            return self

        def __exit__(self, *args: object) -> None:
            return None

        def __getitem__(self, key: str) -> np.ndarray:
            raise AssertionError(f"no member may be read from rejected schema: {key}")

    import pinn_phase.io.artifacts as artifacts

    monkeypatch.setitem(
        artifacts.load_npz_arrays.__globals__,
        "_validate_npz_container",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(np, "load", lambda *args, **kwargs: TrajectoryArchive())
    cfg_path = tmp_path / "config.yaml"
    import yaml

    cfg_path.write_text(yaml.safe_dump(_t0_config(artifact)), encoding="utf-8")
    with pytest.raises(ValueError, match="reviewed schema"):
        load_explicit_mpf_initial_reference(cfg_path, repo_root=tmp_path)
