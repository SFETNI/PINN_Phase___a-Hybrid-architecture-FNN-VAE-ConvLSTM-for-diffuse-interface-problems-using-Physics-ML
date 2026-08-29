#!/usr/bin/env python3
"""Run a deterministic CPU smoke test against the installed package."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

import sys

# Run from a fresh extraction without installing anything: put this repository's
# src/ at the front of the import path. Prepending rather than appending means the
# code under test is this tree's, not a copy that happens to be installed elsewhere.
sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pinn_phase.models.perm_equivariant_mpf import PermEquivariantMPFRollout
from pinn_phase.physics.explicit_mpf import explicit_mpf_rhs, project_simplex


def run_smoke() -> dict[str, object]:
    torch.manual_seed(2026)
    raw = torch.rand(1, 4, 9, 11)
    phi, _ = project_simplex(raw)
    model = PermEquivariantMPFRollout(
        num_phases=4,
        model_dt=0.05,
        eta_px=3.0,
        hidden_channels=3,
        encoder_channels=2,
        ann_hidden_features=(5,),
        zero_initialize_heads=False,
    ).eval()
    permutation = torch.tensor([2, 0, 3, 1])
    with torch.no_grad():
        rate, _, _ = model.predict_rate(phi, model.initial_state(phi))
        permuted = phi[:, permutation]
        permuted_rate, _, _ = model.predict_rate(
            permuted, model.initial_state(permuted)
        )
        translated = torch.roll(phi, shifts=(2, -3), dims=(-2, -1))
        translated_rate, _, _ = model.predict_rate(
            translated, model.initial_state(translated)
        )
        next_phi, _, _, _ = model.forward_step(phi)
        rhs = explicit_mpf_rhs(phi, eta_px=3.0, mu=1.0, sigma=1.0)
    result = {
        "schema": "pinn-phase-cpu-smoke-v1",
        "finite": bool(torch.isfinite(next_phi).all() and torch.isfinite(rhs).all()),
        "phase_sum_error": float((next_phi.sum(dim=1) - 1.0).abs().max()),
        "phi_min": float(next_phi.min()),
        "phi_max": float(next_phi.max()),
        "rhs_phase_sum_error": float(rhs.sum(dim=1).abs().max()),
        "permutation_max_error": float(
            (permuted_rate - rate[:, permutation]).abs().max()
        ),
        "translation_max_error": float(
            (
                translated_rate
                - torch.roll(rate, shifts=(2, -3), dims=(-2, -1))
            )
            .abs()
            .max()
        ),
    }
    result["passed"] = bool(
        result["finite"]
        and result["phase_sum_error"] <= 1.0e-6
        and result["phi_min"] >= -1.0e-7
        and result["phi_max"] <= 1.0 + 1.0e-7
        and result["rhs_phase_sum_error"] <= 1.0e-6
        and result["permutation_max_error"] <= 1.0e-6
        and result["translation_max_error"] <= 1.0e-6
    )
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("outputs/smoke/report.json"))
    args = parser.parse_args()
    result = run_smoke()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print("CPU smoke:", "PASS" if result["passed"] else "FAIL")
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
