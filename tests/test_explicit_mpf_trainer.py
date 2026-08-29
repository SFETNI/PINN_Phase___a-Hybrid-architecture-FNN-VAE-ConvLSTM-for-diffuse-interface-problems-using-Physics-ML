"""Explicit-MPF trainer guard tests.

An earlier source variant exercised the trainer against a
``triple_junction_128x128_*.yaml`` config family, which is not shipped in
this repository. ``assert_no_reference_leakage`` and ``ReferenceLeakageError``
operate purely on an in-memory settings mapping, so this adaptation builds a
minimal synthetic settings dict instead of loading an unshipped config file
-- this is fully self-contained and exercises the identical guard logic.
``build_explicit_mpf_model`` model-construction behavior (no config
dependency) is also covered directly. Trainer-loop / monitor / dispatch
tests that required unshipped command-line and monitoring scripts
(not shipped in this repository) were dropped.
"""

from __future__ import annotations

import copy

import pytest
import torch

from pinn_phase.physics.explicit_mpf import project_simplex
from pinn_phase.training import (
    ReferenceLeakageError,
    assert_no_reference_leakage,
    build_explicit_mpf_model,
)
from pinn_phase.training.explicit_mpf_trainer import maybe_load_explicit_mpf_warmstart


def _clean_settings() -> dict:
    return {
        "model": {
            "architecture": "explicit_mpf_ann_convlstm_hybrid",
            "num_phases": 4,
            "eta_px": 2.5,
            "hidden_channels": 4,
            "ann_hidden_features": [4],
        },
        "reference": {},
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


def test_assert_no_reference_leakage_passes_clean_config() -> None:
    assert_no_reference_leakage(_clean_settings())  # must not raise


def test_trainer_rejects_reference_leakage_via_usage_flag() -> None:
    bad = copy.deepcopy(_clean_settings())
    bad["reference_usage_policy"]["training_uses_reference_frames_after_t0"] = True
    with pytest.raises(ReferenceLeakageError):
        assert_no_reference_leakage(bad)


def test_trainer_rejects_reference_leakage_via_graph_features_from() -> None:
    bad = copy.deepcopy(_clean_settings())
    bad["reference_usage_policy"]["graph_features_from"] = "reference_states"
    with pytest.raises(ReferenceLeakageError):
        assert_no_reference_leakage(bad)


def test_trainer_rejects_reference_leakage_via_training_initial_condition() -> None:
    bad = copy.deepcopy(_clean_settings())
    bad["reference_usage_policy"]["training_initial_condition"] = "Phi_ref_full_trajectory"
    with pytest.raises(ReferenceLeakageError):
        assert_no_reference_leakage(bad)


class _SyntheticReference:
    """Minimal stand-in for pinn_phase.physics.explicit_mpf.ExplicitMPFReference.

    build_explicit_mpf_model only reads a handful of attributes off the
    reference object; this avoids depending on any config/artifact file.
    """

    def __init__(self) -> None:
        self.num_phases = 4
        self.eta_px = 2.5
        self.sigma = 1.0
        self.mu = 1.5e-5
        self.model_dt = 0.05
        self.grid = (8, 8)


def test_build_explicit_mpf_model_no_config_dependency() -> None:
    settings = _clean_settings()
    reference = _SyntheticReference()
    model = build_explicit_mpf_model(settings, reference)
    assert model.num_phases == 4
    assert model.graph_mode == "disabled"
    assert abs(model.model_dt - reference.model_dt) < 1e-9


def test_explicit_mpf_warmstart_loads_weights_and_checks_sha(tmp_path) -> None:
    settings = _clean_settings()
    reference = _SyntheticReference()
    source = build_explicit_mpf_model(settings, reference)
    with torch.no_grad():
        next(source.parameters()).fill_(0.125)
    ckpt = tmp_path / "warm.pt"
    torch.save({"model_state": source.state_dict()}, ckpt)

    import hashlib
    digest = hashlib.sha256(ckpt.read_bytes()).hexdigest()
    target = build_explicit_mpf_model(settings, reference)
    before = next(target.parameters()).detach().clone()
    cfg = copy.deepcopy(settings)
    cfg["warmstart"] = {"enabled": True, "checkpoint": str(ckpt), "checkpoint_sha256": digest}
    meta = maybe_load_explicit_mpf_warmstart(target, cfg, repo_root=tmp_path, device=torch.device("cpu"))

    assert meta is not None
    assert meta["checkpoint_sha256"] == digest
    assert not torch.allclose(before, next(target.parameters()).detach())
    assert torch.allclose(next(target.parameters()).detach(), torch.full_like(next(target.parameters()), 0.125))


def test_explicit_mpf_warmstart_rejects_sha_mismatch(tmp_path) -> None:
    settings = _clean_settings()
    reference = _SyntheticReference()
    source = build_explicit_mpf_model(settings, reference)
    ckpt = tmp_path / "warm.pt"
    torch.save({"model_state": source.state_dict()}, ckpt)

    target = build_explicit_mpf_model(settings, reference)
    cfg = copy.deepcopy(settings)
    cfg["warmstart"] = {
        "enabled": True,
        "checkpoint": str(ckpt),
        "checkpoint_sha256": "0" * 64,  # deliberately wrong
    }
    with pytest.raises(Exception):  # noqa: B017 - the trainer raises a checksum-mismatch error
        maybe_load_explicit_mpf_warmstart(target, cfg, repo_root=tmp_path, device=torch.device("cpu"))
