"""Every advertised replay must be executable, and must stay model-only.

These tests bind the replay registry to distributed bytes, then actually run the
model for a few steps for each architecture family. The last test is the one that
matters most: it runs a replay with the filesystem instrumented and asserts that no
reference or score fixture is opened while the model is running.
"""

from __future__ import annotations

import builtins
import json
from pathlib import Path
import runpy
import sys

import numpy as np
import pytest
import torch
import yaml

from pinn_phase.io.artifacts import sha256_file
from pinn_phase.io.phi0 import load_initial_field
from pinn_phase.io.public_weights import (
    as_torch_state_dict,
    load_public_weights,
    load_weight_lineage,
)
from pinn_phase.models import (
    ExplicitMPFHybridRollout,
    PermEquivariantMPFRollout,
    architecture_family_from_config,
)

ROOT = Path(__file__).resolve().parents[1]
REGISTRY = json.loads((ROOT / "docs/REPLAY_ENTRYPOINTS.json").read_text(encoding="utf-8"))
ENTRIES = REGISTRY["entrypoints"]
NAMES = sorted(ENTRIES)
CLASSES = {
    "PermEquivariantMPFRollout": PermEquivariantMPFRollout,
    "ExplicitMPFHybridRollout": ExplicitMPFHybridRollout,
}
NON_CONSTRUCTOR = frozenset({"architecture", "arch_variant"})
# One representative per architecture family, kept small so the suite stays quick.
REPRESENTATIVES = ["n25_cascade_development", "n8_64_cube"]


@pytest.mark.parametrize("name", NAMES)
def test_registry_digests_match_distributed_bytes(name: str) -> None:
    entry = ENTRIES[name]
    for key, digest_key in (
        ("public_weights", "public_weights_sha256"),
        ("weight_lineage", "weight_lineage_sha256"),
        ("model_config", "model_config_sha256"),
        ("initial_field", "initial_field_sha256"),
    ):
        path = ROOT / entry[key]
        assert path.is_file(), f"{name}: {entry[key]} is not distributed"
        assert sha256_file(path) == entry[digest_key], (
            f"{name}: {key} digest disagrees with the distributed bytes"
        )


@pytest.mark.parametrize("name", NAMES)
def test_initial_field_phase_count_matches_the_model(name: str) -> None:
    entry = ENTRIES[name]
    settings = yaml.safe_load((ROOT / entry["model_config"]).read_text(encoding="utf-8"))
    field = load_initial_field(ROOT / entry["initial_field"])
    assert field.num_phases == int(settings["model"]["num_phases"])


@pytest.mark.parametrize("name", NAMES)
def test_declared_family_resolves_from_the_configuration(name: str) -> None:
    entry = ENTRIES[name]
    settings = yaml.safe_load((ROOT / entry["model_config"]).read_text(encoding="utf-8"))
    assert str(architecture_family_from_config(settings)) == entry["architecture_family"]


@pytest.mark.parametrize("name", REPRESENTATIVES)
def test_bounded_replay_runs_for_each_architecture_family(name: str) -> None:
    """Construct, strict-load, and advance the real model a few steps."""
    entry = ENTRIES[name]
    settings = yaml.safe_load((ROOT / entry["model_config"]).read_text(encoding="utf-8"))
    field = load_initial_field(ROOT / entry["initial_field"],
                              expected_sha256=entry["initial_field_sha256"])
    lineage = load_weight_lineage(ROOT / entry["weight_lineage"])
    state = load_public_weights(ROOT / entry["public_weights"], lineage=lineage)
    block = {k: v for k, v in settings["model"].items() if k not in NON_CONSTRUCTOR}
    physics = settings["physics"]
    model = CLASSES[entry["class_name"]](
        model_dt=float(physics["model_dt"]), eta_px=float(physics["eta_px"]),
        mu=float(physics["mu"]), sigma=float(physics["sigma"]), **block,
    )
    model.load_state_dict(as_torch_state_dict(state), strict=True)
    model.eval()

    phi = torch.from_numpy(np.ascontiguousarray(field.phi0)).to(torch.float32).unsqueeze(0)
    with torch.no_grad():
        recurrent = model.initial_state(phi)
        for _ in range(2):
            phi, recurrent = model.forward_step(phi, recurrent)[:2]
    assert torch.isfinite(phi).all()
    assert float((phi.sum(dim=1) - 1.0).abs().max()) < 1e-4
    assert float(phi.min()) >= -1e-6 and float(phi.max()) <= 1.0 + 1e-6


def test_replay_opens_no_reference_or_score_fixture(tmp_path: Path,
                                                    monkeypatch: pytest.MonkeyPatch) -> None:
    """Run the shipped replay script with every file open recorded.

    A replay that touched ``benchmarks/n25_transfer`` or any reference archive would
    show up here. This is the property the whole separation exists to guarantee, so
    it is checked by observation rather than by reading the source.
    """
    opened: list[str] = []
    real_open = builtins.open
    real_np_load = np.load

    def watched_open(file, *args, **kwargs):  # noqa: ANN001
        opened.append(str(file))
        return real_open(file, *args, **kwargs)

    def watched_np_load(file, *args, **kwargs):  # noqa: ANN001
        opened.append(str(getattr(file, "name", file)))
        return real_np_load(file, *args, **kwargs)

    monkeypatch.setattr(builtins, "open", watched_open)
    monkeypatch.setattr(np, "load", watched_np_load)
    monkeypatch.setattr(
        sys, "argv",
        ["replay_rollout.py", "--benchmark", "n25_cascade_development", "--smoke",
         "--output-dir", str(tmp_path)],
    )
    with pytest.raises(SystemExit) as exit_info:
        runpy.run_path(str(ROOT / "scripts/replay_rollout.py"), run_name="__main__")
    assert exit_info.value.code == 0

    forbidden = [
        path for path in opened
        if "n25_transfer" in path or "reference" in path.lower() or "expected_score" in path
    ]
    assert not forbidden, f"model-only replay opened reference or score material: {forbidden}"
    report = json.loads((tmp_path / "n25_cascade_development_replay.json").read_text())
    assert report["reference_opened"] is False
    assert report["scored"] is False


def test_every_replay_capable_benchmark_is_classified_as_replayable() -> None:
    levels = json.loads((ROOT / "docs/REPRODUCTION_LEVELS.json").read_text(encoding="utf-8"))
    by_id = {row["id"]: row for row in levels["benchmarks"]}
    for benchmark in ("n25_cascade_development", "n64_dense_primary", "n64_dense_sensitivity",
                      "n8_64_cube", "n16_96_cube"):
        row = by_id[benchmark]
        assert row["level"] == "CHECKPOINT_AND_CODE_REPLAY", (
            f"{benchmark} is replayable but classified {row['level']}"
        )
        assert any("initial_conditions/" in item for item in row["artifacts_present"]), (
            f"{benchmark} claims replay but lists no initial field"
        )
        assert "replay_command" in row


def test_no_benchmark_claims_replay_without_a_distributed_initial_field() -> None:
    levels = json.loads((ROOT / "docs/REPRODUCTION_LEVELS.json").read_text(encoding="utf-8"))
    for row in levels["benchmarks"]:
        if row["level"] != "CHECKPOINT_AND_CODE_REPLAY":
            continue
        fields = [item for item in row["artifacts_present"] if "initial_conditions/" in item]
        assert fields, f"{row['id']}: claims replay with no initial field"
        for item in fields:
            assert (ROOT / item).is_file(), f"{row['id']}: initial field {item} is missing"
