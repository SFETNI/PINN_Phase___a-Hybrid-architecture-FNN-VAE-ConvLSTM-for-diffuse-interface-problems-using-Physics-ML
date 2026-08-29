"""The disclosed training-path policies must survive the shipped fail-closed guard.

``docs/TRAINING_PATH_DISCLOSURE.json`` reproduces the reference-usage policy of
each distributed model's training run. These tests replay every disclosed
policy through ``assert_no_reference_leakage`` -- the same guard the trainer runs
-- and then mutate each policy to check the guard actually bites. A disclosure
that merely reads as compliant would pass the first test and fail the second.

The reachability tests below check the property directly in the shipped source
rather than trusting the disclosure: no reference metric may reach checkpoint
selection, and the one reference-comparing early-stop branch must stay
unreachable.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from pinn_phase.training import ReferenceLeakageError, assert_no_reference_leakage

ROOT = Path(__file__).resolve().parents[1]
DISCLOSURE = json.loads((ROOT / "docs/TRAINING_PATH_DISCLOSURE.json").read_text(encoding="utf-8"))
RUNS = [run for run in DISCLOSURE["runs"] if run["reference_usage_policy"]]
RUN_IDS = [run["name"] for run in RUNS]
TRAINING_SRC = ROOT / "src/pinn_phase/training"


def _settings(run: dict) -> dict:
    settings: dict = {
        "reference_usage_policy": copy.deepcopy(run["reference_usage_policy"]),
        "training_policy": copy.deepcopy(run["training_policy"]),
    }
    digests = run.get("training_initial_condition_digests")
    if digests:
        settings["training"] = {
            "multi_ic_batch": {"entries": [{"sha256": item["t0_sha256"]} for item in digests]}
        }
    return settings


@pytest.mark.parametrize("run", RUNS, ids=RUN_IDS)
def test_disclosed_policy_passes_the_shipped_guard(run: dict) -> None:
    assert_no_reference_leakage(_settings(run))  # must not raise


@pytest.mark.parametrize("run", RUNS, ids=RUN_IDS)
def test_disclosed_policy_declares_no_post_t0_training_use(run: dict) -> None:
    policy = run["reference_usage_policy"]
    assert policy["training_uses_reference_frames_after_t0"] is False
    assert policy["latent_targets_from_reference_after_t0"] is False
    assert policy["sampler_targets_from_reference_after_t0"] is False
    assert policy["graph_features_from"] == "model_phi_only"


@pytest.mark.parametrize(
    "key",
    [
        "training_uses_reference_frames_after_t0",
        "latent_targets_from_reference_after_t0",
        "sampler_targets_from_reference_after_t0",
    ],
)
@pytest.mark.parametrize("run", RUNS, ids=RUN_IDS)
def test_guard_rejects_a_mutated_policy(run: dict, key: str) -> None:
    """Flipping any one disclosed flag must be refused, so the pass above means something."""
    settings = _settings(run)
    settings["reference_usage_policy"][key] = True
    with pytest.raises(ReferenceLeakageError):
        assert_no_reference_leakage(settings)


@pytest.mark.parametrize("run", RUNS, ids=RUN_IDS)
def test_guard_rejects_reference_derived_graph_features(run: dict) -> None:
    settings = _settings(run)
    settings["reference_usage_policy"]["graph_features_from"] = "reference_phi"
    with pytest.raises(ReferenceLeakageError):
        assert_no_reference_leakage(settings)


def test_six_ic_policy_requires_exactly_six_initial_conditions() -> None:
    """The six-field firewall is bidirectional; a short batch must be refused."""
    six = [run for run in RUNS if run.get("training_initial_condition_digests")]
    assert six, "no six-initial-condition run is disclosed"
    for run in six:
        digests = run["training_initial_condition_digests"]
        assert len(digests) == 6
        assert len({item["t0_sha256"] for item in digests}) == 6
        settings = _settings(run)
        settings["training"]["multi_ic_batch"]["entries"].pop()
        with pytest.raises(ReferenceLeakageError):
            assert_no_reference_leakage(settings)


def test_no_checkpoint_selection_machinery_in_the_training_package() -> None:
    """No reference-scored checkpoint selection may exist in the shipped training path."""
    offenders = [
        f"{path.relative_to(ROOT)}:{number}"
        for path in sorted(TRAINING_SRC.rglob("*.py"))
        for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1)
        if "best_physical" in line or "lower_weighted_physical_audit_score" in line
    ]
    assert not offenders, f"reference-scored checkpoint selection reachable at: {offenders}"


def test_reference_comparing_early_stop_branch_is_unreachable() -> None:
    """``zero_rate_baseline`` is the only reference-comparing early-stop input.

    It must stay unreachable two ways: no call site supplies it, and no code
    produces the review-row field it reads.
    """
    trainer = (TRAINING_SRC / "explicit_mpf_trainer.py").read_text(encoding="utf-8")
    call_sites = [
        line
        for line in trainer.splitlines()
        if "evaluate_early_stop_gates(" in line and "def " not in line
    ]
    assert call_sites, "the early-stop gate is never called; this test would be vacuous"
    body = trainer[trainer.index("evaluate_early_stop_gates(", trainer.index("should_stop")) :]
    assert "zero_rate_baseline" not in body.split(")")[0], (
        "a call site now supplies zero_rate_baseline, enabling the reference-comparing branch"
    )
    producers = [
        line
        for line in trainer.splitlines()
        if "learned_vs_zero_rate_disagree" in line and "row.get" not in line
    ]
    assert not producers, f"the reference-comparison field now has a producer: {producers}"


def test_training_package_cannot_reach_the_evaluation_fixtures() -> None:
    """Reference arrays must not be reachable from the training package.

    The trainer does name ``artifacts/benchmarks/explicit_mpf/<run_id>``: that is
    the write-side output directory for a run, under the ignored ``artifacts/``
    root, and it is not the distributed fixture tree. Only reads of the shipped
    ``benchmarks/`` fixtures are forbidden here.
    """
    forbidden = ("n25_transfer", "reference_tail", "reference.npz", "expected_score")
    offenders: list[str] = []
    for path in sorted(TRAINING_SRC.rglob("*.py")):
        text = path.read_text(encoding="utf-8")
        offenders += [f"{path.relative_to(ROOT)}: {t}" for t in forbidden if t in text]
        for number, line in enumerate(text.splitlines(), 1):
            if "benchmarks/" in line and "artifacts/benchmarks/" not in line:
                offenders.append(f"{path.relative_to(ROOT)}:{number}: fixture-tree reference")
    assert not offenders, f"training package references evaluation fixtures: {offenders}"


def test_every_distributed_checkpoint_has_a_disclosed_training_path() -> None:
    ledger = json.loads((ROOT / "docs/ARTIFACT_IDENTITY_LEDGER.json").read_text(encoding="utf-8"))
    disclosed = {run["name"] for run in DISCLOSURE["runs"]}
    disclosed |= {row["name"] for row in DISCLOSURE["benchmark_adapter_policies"]["adapters"]}
    missing = {entry["name"] for entry in ledger["artifacts"]} - disclosed
    assert not missing, f"checkpoints distributed with no disclosed training path: {sorted(missing)}"


# --------------------------------------------------------------------------
# Disclosures about the shipped source must stay true OF the shipped source. Each
# test below re-derives the fact from the code rather than trusting the sentence.
# --------------------------------------------------------------------------

def test_no_training_command_ships() -> None:
    """The absence claim is checked: no console script, no __main__, no caller."""
    import tomllib

    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    assert "scripts" not in pyproject.get("project", {}), (
        "a console entry point is declared; the disclosure says none ships"
    )
    with_main = [
        path.relative_to(ROOT)
        for path in (ROOT / "src").rglob("*.py")
        if '__name__ == "__main__"' in path.read_text(encoding="utf-8")
    ]
    assert not with_main, f"package modules are executable: {with_main}"
    trainers = ("train_explicit_mpf", "train_autoregressive_surrogate",
                "train_pinn_physics_only", "train_tbptt_epoch")
    callers = [
        path.relative_to(ROOT)
        for path in (ROOT / "scripts").glob("*.py")
        if any(name + "(" in path.read_text(encoding="utf-8") for name in trainers)
    ]
    assert not callers, f"a shipped script starts training: {callers}"


def test_the_disclosure_does_not_claim_training_cannot_run() -> None:
    """Training functions are importable and will run; the wording must not deny it."""
    entry = DISCLOSURE["no_training_entry_point"]
    assert isinstance(entry, dict), "the entry-point disclosure must be structured"
    # The impossibility phrasing must not be asserted anywhere. It may be quoted in
    # order to be rejected, which is what what_is_present does, so the check is run
    # against the fields that make claims rather than against the whole record.
    for field in ("what_is_absent", "what_cannot_be_reproduced",
                  "consequence_for_the_claims_here"):
        assert "cannot be executed" not in entry[field].lower(), (
            f"{field} says training cannot be executed; the shipped functions do run "
            "when called, and only the recorded RUNS cannot be reproduced"
        )
    assert "will run" in entry["what_is_present"].lower(), (
        "the disclosure must state plainly that the shipped training functions execute"
    )
    assert entry["what_cannot_be_reproduced"], "the real limitation must be named"


def test_supervised_modules_are_disclosed_and_still_exist() -> None:
    """Modules that put a post-t0 frame into a loss must be named where they are."""
    block = DISCLOSURE["supervised_modules_present"]
    disclosed = {entry["path"] for entry in block["modules"]}
    found = set()
    for path in (ROOT / "src/pinn_phase/training").glob("*.py"):
        source = path.read_text(encoding="utf-8")
        supervises = ("reference_states[" in source or "ground_truth_loader" in source)
        if supervises and "(prediction - target)" in source.replace("predicted", "prediction"):
            found.add(path.relative_to(ROOT).as_posix())
    assert found <= disclosed, f"undisclosed supervised training module(s): {found - disclosed}"
    for entry in block["modules"]:
        assert (ROOT / entry["path"]).is_file(), f"{entry['path']} is disclosed but absent"


def test_guard_scope_states_what_it_proves_and_what_it_does_not() -> None:
    """The guard validates a declaration. The disclosure must claim exactly that."""
    scope = DISCLOSURE["guard_scope"]
    assert "declar" in scope["what_the_guard_is"].lower()
    assert scope["what_it_does_not_establish"], (
        "the disclosure must say what configuration validation cannot show"
    )
    text = json.dumps(scope).lower()
    for overclaim in ("inspects tensors", "inspects the data", "observes file access",
                      "proves no reference was read"):
        assert overclaim not in text, f"guard_scope overclaims: {overclaim!r}"

    from pinn_phase.training import ReferenceLeakageError, assert_no_reference_leakage

    # The contract is fail-closed, and that is checked here rather than described.
    # The exhaustive mutation matrix lives in tests/test_reference_policy_guard.py.
    for undeclared in ({}, {"reference_usage_policy": {}}, {"model": {}}):
        with pytest.raises(ReferenceLeakageError):
            assert_no_reference_leakage(undeclared)
