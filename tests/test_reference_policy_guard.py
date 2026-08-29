"""Adversarial mutation matrix for the reference-usage guard.

``assert_no_reference_leakage`` runs at entry to explicit-MPF training and decides
whether a run may start. Its contract is fail-closed: it accepts only a
configuration that declares the complete reviewed reference-usage policy, and it
refuses everything else — including the cases that are easiest to get wrong, where a
declaration is absent, empty, misspelled, or expressed in the wrong type.

The matrix below mutates one property at a time from a known-good configuration and
requires a refusal for each. Two properties are checked throughout:

* the refusal is a ``ReferenceLeakageError``. A ``TypeError`` or ``AttributeError``
  would be a crash, not a refusal, and at a call site the two are not the same
  thing;
* the four policies recorded in ``docs/TRAINING_PATH_DISCLOSURE.json`` — the
  policies of the runs that actually produced the distributed models — pass
  unmodified. A guard that refused those would be wrong in the other direction.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import pytest

from pinn_phase.training import ReferenceLeakageError, assert_no_reference_leakage
from pinn_phase.training.explicit_mpf_trainer import (
    MULTI_IC_ENTRY_COUNT,
    MULTI_IC_MODE,
    REFERENCE_POLICY_KEYS,
    REFERENCE_POLICY_MUST_BE_FALSE,
    SINGLE_IC_MODE,
    TRAINING_POLICY_MUST_BE_TRUE,
)

ROOT = Path(__file__).resolve().parents[1]
DISCLOSURE = json.loads(
    (ROOT / "docs/TRAINING_PATH_DISCLOSURE.json").read_text(encoding="utf-8")
)
ACCEPTED_RUNS = [run for run in DISCLOSURE["runs"] if run["reference_usage_policy"]]

#: Values that are not the literal ``False`` but are commonly written as if they
#: were, plus values that are outright wrong. Every one must be refused.
NOT_LITERAL_FALSE: list[Any] = [
    True, 1, 0, 0.0, "false", "False", "no", "", None, [], {}, "0",
]
NOT_LITERAL_TRUE: list[Any] = [
    False, 0, 1, "true", "True", "yes", "", None, [], {}, "1",
]


def _policy() -> dict:
    return {
        "training_initial_condition": SINGLE_IC_MODE,
        "training_uses_reference_frames_after_t0": False,
        "reference_after_t0_use": "audit_only",
        "audit_uses_reference_frames_after_t0": True,
        "graph_features_from": "model_phi_only",
        "latent_targets_from_reference_after_t0": False,
        "sampler_targets_from_reference_after_t0": False,
    }


def _training_policy() -> dict:
    return {
        "supervision": "initial_condition_only",
        "no_training_labels_after_t0": True,
        "no_reference_graph_after_t0": True,
        "no_reference_latent_targets_after_t0": True,
    }


def _settings() -> dict:
    return {"reference_usage_policy": _policy(), "training_policy": _training_policy()}


def _multi_ic_settings(entries: int = MULTI_IC_ENTRY_COUNT) -> dict:
    settings = _settings()
    settings["reference_usage_policy"]["training_initial_condition"] = MULTI_IC_MODE
    settings["training"] = {
        "multi_ic_batch": {"entries": [{"sha256": f"{index:064d}"} for index in range(entries)]}
    }
    return settings


# --------------------------------------------------------------------------
# the contract holds in the accepting direction
# --------------------------------------------------------------------------

def test_a_complete_declaration_is_accepted() -> None:
    assert_no_reference_leakage(_settings())


def test_the_multi_initial_condition_mode_is_accepted_with_six_entries() -> None:
    assert_no_reference_leakage(_multi_ic_settings())


@pytest.mark.parametrize("run", ACCEPTED_RUNS, ids=[run["name"] for run in ACCEPTED_RUNS])
def test_every_accepted_run_policy_passes_unmodified(run: dict) -> None:
    """The runs that produced the distributed models must not be refused."""
    settings = {
        "reference_usage_policy": copy.deepcopy(run["reference_usage_policy"]),
        "training_policy": copy.deepcopy(run["training_policy"]),
    }
    if settings["reference_usage_policy"]["training_initial_condition"] == MULTI_IC_MODE:
        settings["training"] = _multi_ic_settings()["training"]
    assert_no_reference_leakage(settings)


# --------------------------------------------------------------------------
# malformed and absent declarations
# --------------------------------------------------------------------------

@pytest.mark.parametrize(
    "settings",
    [None, [], "policy", 0, True, set()],
    ids=["none", "list", "str", "int", "bool", "set"],
)
def test_a_non_mapping_configuration_is_refused(settings: Any) -> None:
    with pytest.raises(ReferenceLeakageError):
        assert_no_reference_leakage(settings)


@pytest.mark.parametrize(
    "settings",
    [{}, {"model": {}}, {"training_policy": {}}, {"reference_usage_policy": None},
     {"reference_usage_policy": []}, {"reference_usage_policy": {}}],
    ids=["empty", "other-keys-only", "training-policy-only", "policy-none",
         "policy-list", "policy-empty"],
)
def test_an_absent_or_empty_policy_is_refused(settings: dict) -> None:
    """Silence is not consent: an undeclared channel is not a closed one."""
    with pytest.raises(ReferenceLeakageError):
        assert_no_reference_leakage(settings)


@pytest.mark.parametrize("key", sorted(REFERENCE_POLICY_KEYS))
def test_every_reviewed_policy_key_is_required(key: str) -> None:
    settings = _settings()
    del settings["reference_usage_policy"][key]
    with pytest.raises(ReferenceLeakageError, match="missing reviewed key"):
        assert_no_reference_leakage(settings)


@pytest.mark.parametrize("key", sorted(REFERENCE_POLICY_KEYS))
def test_a_misspelled_policy_key_is_refused(key: str) -> None:
    """A near miss must not read as the reviewed key, and must not slip in as an extra."""
    settings = _settings()
    value = settings["reference_usage_policy"].pop(key)
    settings["reference_usage_policy"][key + "_"] = value
    with pytest.raises(ReferenceLeakageError):
        assert_no_reference_leakage(settings)


def test_an_unreviewed_extra_policy_key_is_refused() -> None:
    settings = _settings()
    settings["reference_usage_policy"]["latent_targets_from_reference_before_t0"] = False
    with pytest.raises(ReferenceLeakageError, match="unreviewed key"):
        assert_no_reference_leakage(settings)


# --------------------------------------------------------------------------
# value substitutions
# --------------------------------------------------------------------------

@pytest.mark.parametrize("key", REFERENCE_POLICY_MUST_BE_FALSE)
@pytest.mark.parametrize("value", NOT_LITERAL_FALSE, ids=[repr(v) for v in NOT_LITERAL_FALSE])
def test_a_post_t0_reference_flag_must_be_literal_false(key: str, value: Any) -> None:
    settings = _settings()
    settings["reference_usage_policy"][key] = value
    with pytest.raises(ReferenceLeakageError):
        assert_no_reference_leakage(settings)


@pytest.mark.parametrize("key", TRAINING_POLICY_MUST_BE_TRUE)
@pytest.mark.parametrize("value", NOT_LITERAL_TRUE, ids=[repr(v) for v in NOT_LITERAL_TRUE])
def test_a_training_policy_flag_must_be_literal_true(key: str, value: Any) -> None:
    settings = _settings()
    settings["training_policy"][key] = value
    with pytest.raises(ReferenceLeakageError):
        assert_no_reference_leakage(settings)


@pytest.mark.parametrize(
    "value", [True, False, 1, None, [], {}], ids=["true", "false", "int", "none", "list", "dict"]
)
def test_a_string_valued_policy_key_must_be_a_string(value: Any) -> None:
    settings = _settings()
    settings["reference_usage_policy"]["reference_after_t0_use"] = value
    with pytest.raises(ReferenceLeakageError):
        assert_no_reference_leakage(settings)


@pytest.mark.parametrize(
    "value", [1, 0, "true", None, [], {}], ids=["one", "zero", "str", "none", "list", "dict"]
)
def test_the_audit_flag_must_be_a_literal_boolean(value: Any) -> None:
    settings = _settings()
    settings["reference_usage_policy"]["audit_uses_reference_frames_after_t0"] = value
    with pytest.raises(ReferenceLeakageError):
        assert_no_reference_leakage(settings)


@pytest.mark.parametrize(
    "value",
    ["reference_states", "model_phi", "MODEL_PHI_ONLY", "", None, True],
    ids=["reference", "near-miss", "case", "empty", "none", "bool"],
)
def test_graph_features_must_come_from_the_model_field(value: Any) -> None:
    settings = _settings()
    settings["reference_usage_policy"]["graph_features_from"] = value
    with pytest.raises(ReferenceLeakageError):
        assert_no_reference_leakage(settings)


@pytest.mark.parametrize(
    "value",
    ["Phi_ref_full_trajectory", "phi_ref_t0_only", "", None, True,
     "six_frozen_foundry_t0_fields", SINGLE_IC_MODE + " "],
    ids=["trajectory", "case", "empty", "none", "bool", "near-miss-multi", "trailing-space"],
)
def test_the_initial_condition_mode_must_be_a_reviewed_value(value: Any) -> None:
    settings = _settings()
    settings["reference_usage_policy"]["training_initial_condition"] = value
    with pytest.raises(ReferenceLeakageError):
        assert_no_reference_leakage(settings)


# --------------------------------------------------------------------------
# the bidirectional multi-initial-condition contract
# --------------------------------------------------------------------------

@pytest.mark.parametrize("count", [0, 1, 5, 7, 12])
def test_the_multi_initial_condition_mode_requires_exactly_six_entries(count: int) -> None:
    with pytest.raises(ReferenceLeakageError, match="exactly"):
        assert_no_reference_leakage(_multi_ic_settings(count))


@pytest.mark.parametrize(
    "batch", [None, {}, {"entries": None}, {"entries": {}}, {"entries": "six"}, "batch"],
    ids=["absent", "empty", "entries-none", "entries-dict", "entries-str", "not-a-mapping"],
)
def test_the_multi_initial_condition_mode_requires_a_well_formed_batch(batch: Any) -> None:
    settings = _multi_ic_settings()
    if batch is None:
        del settings["training"]["multi_ic_batch"]
    else:
        settings["training"]["multi_ic_batch"] = batch
    with pytest.raises(ReferenceLeakageError):
        assert_no_reference_leakage(settings)


def test_a_malformed_multi_initial_condition_entry_is_refused() -> None:
    settings = _multi_ic_settings()
    settings["training"]["multi_ic_batch"]["entries"][3] = "not-a-mapping"
    with pytest.raises(ReferenceLeakageError, match="entries\\[3\\]"):
        assert_no_reference_leakage(settings)


def test_a_multi_initial_condition_batch_may_not_run_under_the_single_mode() -> None:
    """The firewall is bidirectional: the batch requires the mode, not just vice versa."""
    settings = _settings()
    settings["training"] = {
        "multi_ic_batch": {"entries": [{"sha256": f"{i:064d}"} for i in range(6)]}
    }
    with pytest.raises(ReferenceLeakageError, match=MULTI_IC_MODE):
        assert_no_reference_leakage(settings)


def test_a_non_mapping_training_block_is_refused() -> None:
    settings = _settings()
    settings["training"] = ["multi_ic_batch"]
    with pytest.raises(ReferenceLeakageError):
        assert_no_reference_leakage(settings)


# --------------------------------------------------------------------------
# training_policy
# --------------------------------------------------------------------------

@pytest.mark.parametrize(
    "block", [None, {}, [], "policy"], ids=["absent", "empty", "list", "str"]
)
def test_training_policy_is_mandatory_and_must_be_a_mapping(block: Any) -> None:
    settings = _settings()
    if block is None:
        del settings["training_policy"]
    else:
        settings["training_policy"] = block
    with pytest.raises(ReferenceLeakageError):
        assert_no_reference_leakage(settings)


@pytest.mark.parametrize(
    "value", ["supervised", "initial_condition", "", None, True],
    ids=["supervised", "near-miss", "empty", "none", "bool"],
)
def test_supervision_must_be_declared_initial_condition_only(value: Any) -> None:
    settings = _settings()
    settings["training_policy"]["supervision"] = value
    with pytest.raises(ReferenceLeakageError, match="supervision"):
        assert_no_reference_leakage(settings)


def test_an_extra_training_policy_key_is_tolerated() -> None:
    """Unlike the reference policy, this block is allowed to carry run bookkeeping.

    The accepted 64-grain configurations record `single_attempt` here. The keys that
    matter are required and checked; an additional descriptive key is not a channel.
    """
    settings = _settings()
    settings["training_policy"]["single_attempt"] = True
    assert_no_reference_leakage(settings)


# --------------------------------------------------------------------------
# no refusal may arrive as a crash
# --------------------------------------------------------------------------

@pytest.mark.parametrize(
    "settings",
    [None, [], {}, {"reference_usage_policy": None}, {"reference_usage_policy": 7},
     {"reference_usage_policy": {}, "training_policy": None},
     {"reference_usage_policy": {}, "training": 3}],
    ids=["none", "list", "empty", "policy-none", "policy-int", "training-policy-none",
         "training-int"],
)
def test_every_refusal_is_a_reference_leakage_error(settings: Any) -> None:
    """A crash is not a refusal. The caller must be able to tell the two apart."""
    with pytest.raises(ReferenceLeakageError):
        try:
            assert_no_reference_leakage(settings)
        except (AttributeError, TypeError, KeyError) as exc:  # pragma: no cover
            pytest.fail(f"guard raised {type(exc).__name__} instead of refusing: {exc}")
