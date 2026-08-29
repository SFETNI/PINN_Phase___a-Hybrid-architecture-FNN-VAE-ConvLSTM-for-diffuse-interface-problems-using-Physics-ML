#!/usr/bin/env python3
"""Recompute the public N25 prospective transfer result from shipped arrays."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import sys

# Run from a fresh extraction without installing anything: put this repository's
# src/ at the front of the import path. Prepending rather than appending means the
# code under test is this tree's, not a copy that happens to be installed elsewhere.
sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pinn_phase.evaluation.n25_transfer import (
    MODEL_KEYS,
    REFERENCE_KEYS,
    TAIL_KEYS,
    array_sha256,
    campaign_disposition,
    extract_case_metrics,
    false_death_tail_steps,
)
from pinn_phase.io import load_npz_arrays


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def load_artifact(root: Path, entry: dict[str, Any], keys: frozenset[str]):
    arrays = load_npz_arrays(
        root / entry["path"],
        expected_sha256=entry["sha256"],
        expected_keys=keys,
    )
    for key, expected in entry["arrays"].items():
        array = arrays[key]
        if array.dtype.str != expected["dtype"] or list(array.shape) != expected["shape"]:
            raise ValueError(f"array schema mismatch for {entry['path']}:{key}")
        if array_sha256(array) != expected["sha256"]:
            raise ValueError(f"array digest mismatch for {entry['path']}:{key}")
    return arrays


def score(root: Path) -> dict[str, Any]:
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    cases = manifest["cases"]
    arms: dict[str, Any] = {}
    for arm_key, result_key, disposition_arm in (
        ("primary", "Z", "Z"),
        ("secondary", "G", "G_SECONDARY"),
    ):
        id_metrics: list[dict[str, Any]] = []
        stress_metrics: list[dict[str, Any]] = []
        tail_context: dict[str, Any] = {}
        for case_id, case in sorted(cases.items(), key=lambda item: item[1]["ordinal"]):
            reference = load_artifact(root, case["reference"], REFERENCE_KEYS)
            model = load_artifact(root, case[arm_key], MODEL_KEYS)
            tail = load_artifact(root, case["tail"], TAIL_KEYS)
            metrics = extract_case_metrics(reference, model, case_id=case_id)
            target = id_metrics if case["stratum"] == "id" else stress_metrics
            target.append(metrics)
            if metrics["false_deaths"]:
                tail_context[case_id] = false_death_tail_steps(metrics, tail)
        arms[result_key] = {
            "id_cases": id_metrics,
            "stress_cases": stress_metrics,
            "campaign": campaign_disposition(
                id_metrics, stress_metrics, arm=disposition_arm
            ),
            "post_horizon_false_death_context": tail_context,
        }
    return {
        "schema": "pinn-phase-n25-transfer-score-v1",
        "strict_horizon": manifest["strict_horizon"],
        "tail_is_descriptive_and_non_rescuing": True,
        "arms": arms,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--benchmark-root",
        type=Path,
        default=REPOSITORY_ROOT / "benchmarks/n25_transfer",
    )
    parser.add_argument("--output", type=Path, default=Path("outputs/n25_transfer_score.json"))
    args = parser.parse_args()
    benchmark_root = args.benchmark_root.resolve()
    result = score(benchmark_root)
    expected = json.loads(
        (benchmark_root / "expected_score.json").read_text(encoding="utf-8")
    )
    if result != expected:
        raise RuntimeError("recomputed N25 score differs from the reviewed expected score")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    for label in ("Z", "G"):
        campaign = result["arms"][label]["campaign"]
        print(
            f"{label}: {campaign['id_pass_count']}/8 ID, "
            f"{campaign['stress_pass_count']}/2 stress, {campaign['disposition']}"
        )
    print("Reviewed expected score: exact match")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
