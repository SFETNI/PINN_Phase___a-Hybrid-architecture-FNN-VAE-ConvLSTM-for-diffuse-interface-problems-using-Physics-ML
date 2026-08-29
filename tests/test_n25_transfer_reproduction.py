from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from pinn_phase.evaluation.n25_transfer import (
    THRESHOLDS,
    campaign_disposition,
    false_death_tail_steps,
    per_case_transfer,
)
import runpy


ROOT = Path(__file__).resolve().parents[1]


def _passing_case(case_id: str) -> dict:
    return {
        "case_id": case_id,
        "finite": True,
        "max_abs_sum_error": 0.0,
        "phi_min": 0.0,
        "phi_max": 1.0,
        "margin_4000": 0.7,
        "reappearances": [],
        "D_persist_4000_pct": 20.0,
        "D_persist_12000_pct": 40.0,
        "D_model_4000_pct": 1.0,
        "D_terminal_pct": 2.0,
        "G_4000": 0.95,
        "G_12000": 0.95,
        "L_int_4000": 1.0,
        "terminal_f1": 1.0,
        "false_deaths": [],
    }


def test_strict_false_death_gate_creates_boundary_pass() -> None:
    cases = [_passing_case(f"id_{index:02d}") for index in range(1, 9)]
    cases[-1] = dict(cases[-1], false_deaths=[19], terminal_f1=0.967741935483871)
    stress = [_passing_case("stress_dense"), _passing_case("stress_sparse")]
    result = campaign_disposition(cases, stress, arm="Z")
    assert result["id_pass_count"] == 7
    assert result["stress_pass_count"] == 2
    assert result["disposition"].startswith("IC_GENERALIZATION_ESTABLISHED_Z")
    assert per_case_transfer(cases[-1])["no_false_death"] is False


def test_tail_context_is_descriptive_for_false_deaths() -> None:
    steps = np.arange(12001, 13001)
    areas = np.ones((1000, 25), dtype=np.int64)
    areas[932:, 19] = 0
    tail = {
        "tail_steps": steps,
        "tail_areas": areas,
        "tail_active_mask": areas > 0,
        "tail_terminal_active_ids": np.flatnonzero(areas[-1] > 0),
    }
    assert false_death_tail_steps({"false_deaths": [19]}, tail) == {"19": 12933}


def test_shipped_expected_score_records_the_reviewed_boundary() -> None:
    score = json.loads(
        (ROOT / "benchmarks/n25_transfer/expected_score.json").read_text(encoding="utf-8")
    )
    primary = score["arms"]["Z"]
    secondary = score["arms"]["G"]
    assert primary["campaign"]["id_pass_count"] == 7
    assert primary["campaign"]["stress_pass_count"] == 2
    assert secondary["campaign"]["id_pass_count"] == 7
    assert secondary["campaign"]["stress_pass_count"] == 2
    assert secondary["campaign"]["arm"] == "G_SECONDARY"
    assert secondary["campaign"]["disposition"] == (
        "IC_GENERALIZATION_ESTABLISHED_G_SECONDARY_WITHIN_FROZEN_N25_CASCADE_FAMILY"
    )
    assert primary["id_cases"][6]["false_deaths"] == [19]
    assert primary["post_horizon_false_death_context"] == {"id_07": {"19": 12933}}
    assert primary["id_cases"][0]["timing_symmetric_diagnostic"][
        "symmetric_bias_median"
    ] == -58.0
    assert secondary["id_cases"][0]["timing_symmetric_diagnostic"][
        "symmetric_bias_median"
    ] == -66.0


def test_frozen_score_provenance_and_thresholds_are_independently_pinned() -> None:
    manifest = json.loads(
        (ROOT / "benchmarks/n25_transfer/manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["frozen_score_sha256"] == (
        "26fbfdefed5278106a16cedf4be2a1db1ed010760cabf54336f6ee6a378e7202"
    )
    assert manifest["frozen_scorer_sha256"] == (
        "3a278860a5c35b5e7acf16db9aa146b44f71d1aba2fb4170ecbaab1e8f8f8a96"
    )
    assert THRESHOLDS == {
        "difficulty_persist_4000_min_pct": 10.0,
        "difficulty_persist_12000_min_pct": 30.0,
        "max_phase_sum_error": 1.0e-5,
        "phi_min": -1.0e-6,
        "phi_max": 1.0 + 1.0e-6,
        "margin_4000_min": 0.30,
        "G_4000_min": 0.70,
        "L_int_4000_min": 0.70,
        "D_terminal_max_pct": 15.0,
        "G_12000_min": 0.60,
        "terminal_f1_min": 0.90,
        "id_pass_count_min": 7,
        "id_median_G4000_min": 0.70,
        "id_worst_G4000_min_exclusive": 0.0,
        "id_worst_terminal_f1_min": 0.80,
    }


def test_all_shipped_n25_arrays_recompute_the_exact_expected_score() -> None:
    namespace = runpy.run_path(str(ROOT / "scripts/reproduce_n25_transfer.py"))
    observed = namespace["score"]((ROOT / "benchmarks/n25_transfer").resolve())
    expected = json.loads(
        (ROOT / "benchmarks/n25_transfer/expected_score.json").read_text(encoding="utf-8")
    )
    assert observed == expected
