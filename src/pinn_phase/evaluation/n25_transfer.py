"""Metrics and frozen decision rules for the prospective N25 transfer benchmark."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import hashlib
import math
from typing import Any

import numpy as np


N_PHASES = 25
TERMINAL_STEP = 12_000
ID_CASES = 8
STRESS_CASES = 2

MODEL_KEYS = frozenset(
    {
        "areas",
        "cadence_labels",
        "cadence_margin_mean",
        "cadence_steps",
        "phi_max_per_step",
        "phi_min_per_step",
        "sumerr_per_step",
    }
)
REFERENCE_KEYS = frozenset({"areas", "cadence_labels", "cadence_steps"})
TAIL_KEYS = frozenset(
    {"tail_active_mask", "tail_areas", "tail_steps", "tail_terminal_active_ids"}
)

# Frozen before the prospective confirmation cohort was scored.
THRESHOLDS = {
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


def array_sha256(array: np.ndarray) -> str:
    """Hash an array's dtype, shape, and contiguous value bytes."""

    value = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(value.dtype.str.encode("ascii"))
    digest.update(b"\0")
    digest.update(",".join(str(size) for size in value.shape).encode("ascii"))
    digest.update(b"\0")
    digest.update(value.tobytes(order="C"))
    return digest.hexdigest()


def _labels_at(bundle: Mapping[str, np.ndarray], step: int) -> np.ndarray:
    steps = [int(value) for value in bundle["cadence_steps"].tolist()]
    if step not in steps:
        raise ValueError(f"required cadence step {step} is absent")
    return bundle["cadence_labels"][steps.index(step)]


def _terminal_zero_run_start(area: np.ndarray) -> int | None:
    if area[-1] != 0:
        return None
    nonzero = np.flatnonzero(area)
    return 0 if nonzero.size == 0 else int(nonzero[-1] + 1)


def argmax_disagreement_pct(labels_a: np.ndarray, labels_b: np.ndarray) -> float:
    return 100.0 * float(np.mean(labels_a != labels_b))


def gain_over_persistence(d_model_pct: float, d_persist_pct: float) -> float | None:
    if d_persist_pct == 0.0:
        return None
    return 1.0 - d_model_pct / d_persist_pct


def _boundary(labels: np.ndarray) -> np.ndarray:
    edge = np.zeros(labels.shape, dtype=bool)
    for axis, shift in ((0, 1), (0, -1), (1, 1), (1, -1)):
        edge |= labels != np.roll(labels, shift, axis=axis)
    return edge


def _dilate_radius_two(mask: np.ndarray) -> np.ndarray:
    result = np.zeros_like(mask)
    for dx in range(-2, 3):
        for dy in range(-2, 3):
            if dx * dx + dy * dy <= 4:
                result |= np.roll(np.roll(mask, dx, axis=0), dy, axis=1)
    return result


def interface_localization(reference: np.ndarray, model: np.ndarray) -> float:
    disagreement = reference != model
    if not disagreement.any():
        return 1.0
    band = _dilate_radius_two(_boundary(reference) | _boundary(model))
    return float(np.sum(disagreement & band) / np.sum(disagreement))


def survivor_f1(model_survivors: set[int], reference_survivors: set[int]) -> float:
    true_positive = len(model_survivors & reference_survivors)
    precision = true_positive / max(len(model_survivors), 1)
    recall = true_positive / max(len(reference_survivors), 1)
    return 2.0 * precision * recall / max(precision + recall, 1.0e-12)


def wilson_interval_95(successes: int, trials: int) -> tuple[float, float]:
    if trials <= 0:
        raise ValueError("trials must be positive")
    z = 1.959963984540054
    estimate = successes / trials
    denominator = 1.0 + z * z / trials
    center = (estimate + z * z / (2.0 * trials)) / denominator
    half_width = (z / denominator) * math.sqrt(
        estimate * (1.0 - estimate) / trials + z * z / (4.0 * trials * trials)
    )
    return max(0.0, center - half_width), min(1.0, center + half_width)


def timing_symmetric_diagnostic(
    reference_by_phase: Mapping[int, int], model_by_phase: Mapping[int, int]
) -> dict[str, Any]:
    """Report cadence-symmetric timing without using it as a gate."""

    raw: dict[int, int] = {}
    symmetric: dict[int, int] = {}
    censored: list[int] = []
    for phase, reference_step in sorted(reference_by_phase.items()):
        model_step = model_by_phase.get(phase)
        if model_step is None:
            censored.append(phase)
            continue
        raw[phase] = int(model_step) - int(reference_step)
        symmetric[phase] = 50 * math.ceil(int(model_step) / 50) - int(reference_step)

    def median(values: Sequence[int]) -> float | None:
        ordered = sorted(values)
        count = len(ordered)
        if count == 0:
            return None
        if count % 2:
            return float(ordered[count // 2])
        return float((ordered[count // 2 - 1] + ordered[count // 2]) / 2)

    return {
        "convention": (
            "e_sym = 50*ceil(t_model/50) - t_ref; raw co-reported; "
            "DIAGNOSTIC ONLY (never a gate; never replaces no_false_death)"
        ),
        "e_raw_by_phase": {str(phase): raw[phase] for phase in sorted(raw)},
        "e_symmetric_by_phase": {
            str(phase): symmetric[phase] for phase in sorted(symmetric)
        },
        "censored_phases": censored,
        "raw_bias_median": median(list(raw.values())),
        "raw_medae": median([abs(value) for value in raw.values()]),
        "symmetric_bias_median": median(list(symmetric.values())),
        "symmetric_medae": median([abs(value) for value in symmetric.values()]),
    }


def extract_case_metrics(
    reference: Mapping[str, np.ndarray],
    model: Mapping[str, np.ndarray],
    *,
    case_id: str,
) -> dict[str, Any]:
    """Recompute every gated metric from one reference/model pair."""

    reference_steps = np.asarray(reference["cadence_steps"])
    model_steps = np.asarray(model["cadence_steps"])
    if reference_steps.ndim != 1 or model_steps.ndim != 1:
        raise ValueError("cadence_steps must be one-dimensional")
    if reference["areas"].shape != (TERMINAL_STEP + 1, N_PHASES):
        raise ValueError("reference areas have an unexpected shape")
    if model["areas"].shape != (TERMINAL_STEP + 1, N_PHASES):
        raise ValueError("model areas have an unexpected shape")

    t0 = _labels_at(reference, 0)
    reference_4000 = _labels_at(reference, 4000)
    reference_12000 = _labels_at(reference, TERMINAL_STEP)
    model_4000 = _labels_at(model, 4000)
    model_12000 = _labels_at(model, TERMINAL_STEP)

    persistence_4000 = argmax_disagreement_pct(t0, reference_4000)
    persistence_12000 = argmax_disagreement_pct(t0, reference_12000)
    disagreement_4000 = argmax_disagreement_pct(model_4000, reference_4000)
    disagreement_12000 = argmax_disagreement_pct(model_12000, reference_12000)

    reference_areas = reference["areas"]
    model_areas = model["areas"]
    reference_survivors = {
        phase for phase in range(N_PHASES) if reference_areas[TERMINAL_STEP, phase] > 0
    }
    model_survivors = {
        phase for phase in range(N_PHASES) if model_areas[TERMINAL_STEP, phase] > 0
    }

    model_cadence_areas = np.stack(
        [
            np.bincount(labels.ravel(), minlength=N_PHASES)
            for labels in model["cadence_labels"]
        ]
    )
    reappearances: list[int] = []
    for phase in range(N_PHASES):
        values = model_cadence_areas[:, phase]
        zeros = np.flatnonzero(values == 0)
        if zeros.size and np.any(values[zeros[0] :] > 0):
            reappearances.append(phase)

    reference_extinction = {
        phase: _terminal_zero_run_start(reference_areas[:, phase])
        for phase in range(N_PHASES)
    }
    model_extinction = {
        phase: _terminal_zero_run_start(model_areas[:, phase])
        for phase in range(N_PHASES)
    }
    timing = {
        phase: int(model_extinction[phase] - reference_extinction[phase])
        for phase in range(N_PHASES)
        if reference_extinction[phase] is not None and model_extinction[phase] is not None
    }
    model_step_values = [int(value) for value in model_steps.tolist()]
    margin_4000 = float(model["cadence_margin_mean"][model_step_values.index(4000)])

    gain_4000 = gain_over_persistence(disagreement_4000, persistence_4000)
    gain_12000 = gain_over_persistence(disagreement_12000, persistence_12000)
    health_arrays = (
        model["sumerr_per_step"],
        model["phi_min_per_step"],
        model["phi_max_per_step"],
    )
    return {
        "case_id": case_id,
        "finite": bool(all(np.isfinite(array).all() for array in health_arrays)),
        "max_abs_sum_error": float(np.max(model["sumerr_per_step"])),
        "phi_min": float(np.min(model["phi_min_per_step"])),
        "phi_max": float(np.max(model["phi_max_per_step"])),
        "margin_4000": margin_4000,
        "reappearances": reappearances,
        "D_persist_4000_pct": persistence_4000,
        "D_persist_12000_pct": persistence_12000,
        "D_model_4000_pct": disagreement_4000,
        "D_terminal_pct": disagreement_12000,
        "G_4000": gain_4000,
        "G_12000": gain_12000,
        "L_int_4000": interface_localization(reference_4000, model_4000),
        "terminal_f1": survivor_f1(model_survivors, reference_survivors),
        "ref_terminal_survivors": sorted(reference_survivors),
        "model_terminal_survivors": sorted(model_survivors),
        "false_deaths": sorted(reference_survivors - model_survivors),
        "ref_extinction_count": sum(value is not None for value in reference_extinction.values()),
        "detected_extinction_count": len(timing),
        "timing_residuals_diagnostic": {
            str(phase): residual for phase, residual in sorted(timing.items())
        },
        "timing_median_diagnostic": float(np.median(list(timing.values()))) if timing else None,
        "timing_symmetric_diagnostic": timing_symmetric_diagnostic(
            {
                phase: value
                for phase, value in reference_extinction.items()
                if value is not None
            },
            {
                phase: value
                for phase, value in model_extinction.items()
                if value is not None
            },
        ),
    }


def per_case_structural(metrics: Mapping[str, Any]) -> dict[str, bool]:
    threshold = THRESHOLDS
    return {
        "finite": bool(metrics["finite"]),
        "phase_sum": metrics["max_abs_sum_error"] <= threshold["max_phase_sum_error"],
        "phi_bounds": (
            metrics["phi_min"] >= threshold["phi_min"]
            and metrics["phi_max"] <= threshold["phi_max"]
        ),
        "margin_4000": metrics["margin_4000"] >= threshold["margin_4000_min"],
        "no_reappearance": not metrics["reappearances"],
    }


def per_case_difficulty(metrics: Mapping[str, Any]) -> bool:
    threshold = THRESHOLDS
    return (
        metrics["D_persist_4000_pct"] >= threshold["difficulty_persist_4000_min_pct"]
        and metrics["D_persist_12000_pct"]
        >= threshold["difficulty_persist_12000_min_pct"]
    )


def per_case_transfer(metrics: Mapping[str, Any]) -> dict[str, bool]:
    threshold = THRESHOLDS
    return {
        "G_4000": metrics["G_4000"] is not None
        and metrics["G_4000"] >= threshold["G_4000_min"],
        "L_int_4000": metrics["L_int_4000"] >= threshold["L_int_4000_min"],
        "D_terminal": metrics["D_terminal_pct"] <= threshold["D_terminal_max_pct"],
        "G_12000": metrics["G_12000"] is not None
        and metrics["G_12000"] >= threshold["G_12000_min"],
        "terminal_f1": metrics["terminal_f1"] >= threshold["terminal_f1_min"],
        "no_false_death": not metrics["false_deaths"],
        "structural_all": all(per_case_structural(metrics).values()),
    }


def campaign_disposition(
    id_cases: Sequence[Mapping[str, Any]],
    stress_cases: Sequence[Mapping[str, Any]],
    *,
    arm: str,
) -> dict[str, Any]:
    """Apply the prospectively frozen ID and stress aggregation rules."""

    if len(id_cases) != ID_CASES or len(stress_cases) != STRESS_CASES:
        raise ValueError("expected exactly eight ID cases and two stress cases")
    threshold = THRESHOLDS
    structural = [all(per_case_structural(case).values()) for case in id_cases]
    difficulty = [per_case_difficulty(case) for case in id_cases]
    transfer = [all(per_case_transfer(case).values()) for case in id_cases]
    gain_4000 = [case["G_4000"] for case in id_cases]
    structural_all = all(structural)
    difficulty_all = all(difficulty)
    pass_count = sum(transfer)
    primary_all = (
        pass_count >= threshold["id_pass_count_min"]
        and all(value is not None for value in gain_4000)
        and float(np.median(gain_4000)) >= threshold["id_median_G4000_min"]
        and min(gain_4000) > threshold["id_worst_G4000_min_exclusive"]
        and min(case["terminal_f1"] for case in id_cases)
        >= threshold["id_worst_terminal_f1_min"]
    )
    if not structural_all:
        disposition = f"GENERALIZATION_NOT_ESTABLISHED_{arm}:STRUCTURAL_FAILURE"
    elif not difficulty_all:
        disposition = f"GENERALIZATION_NOT_ESTABLISHED_{arm}:CHALLENGE_ADEQUACY_FAILURE"
    elif primary_all:
        disposition = f"IC_GENERALIZATION_ESTABLISHED_{arm}_WITHIN_FROZEN_N25_CASCADE_FAMILY"
    else:
        disposition = f"PARTIAL_IC_TRANSFER_{arm}"

    stress_structural = [all(per_case_structural(case).values()) for case in stress_cases]
    stress_full = [
        structurally_valid
        and per_case_difficulty(case)
        and all(per_case_transfer(case).values())
        for structurally_valid, case in zip(stress_structural, stress_cases, strict=True)
    ]
    if not all(stress_structural):
        stress_disposition = f"STRESS_ROBUSTNESS_NOT_ESTABLISHED_{arm}"
    elif sum(stress_full) == 2:
        stress_disposition = f"STRESS_ROBUSTNESS_PASS_{arm}"
    elif sum(stress_full) == 1:
        stress_disposition = f"STRESS_ROBUSTNESS_PARTIAL_{arm}"
    else:
        stress_disposition = f"STRESS_ROBUSTNESS_NOT_ESTABLISHED_{arm}"
    return {
        "arm": arm,
        "GEN_ID_S": structural_all,
        "GEN_ID_D": difficulty_all,
        "GEN_ID_P": primary_all,
        "id_pass_count": pass_count,
        "id_wilson_95": list(wilson_interval_95(pass_count, ID_CASES)),
        "disposition": disposition,
        "stress_disposition": stress_disposition,
        "stress_pass_count": sum(stress_full),
    }


def false_death_tail_steps(
    metrics: Mapping[str, Any], tail: Mapping[str, np.ndarray]
) -> dict[str, int | None]:
    """Describe post-horizon reference outcomes without changing the strict verdict."""

    steps = np.asarray(tail["tail_steps"])
    areas = np.asarray(tail["tail_areas"])
    if areas.shape != (steps.size, N_PHASES):
        raise ValueError("tail areas have an unexpected shape")
    result: dict[str, int | None] = {}
    for phase in metrics["false_deaths"]:
        zeros = np.flatnonzero(areas[:, phase] == 0)
        result[str(phase)] = int(steps[zeros[0]]) if zeros.size else None
    return result
