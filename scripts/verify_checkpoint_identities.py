#!/usr/bin/env python3
"""Verify every distributed weight artifact: identity, strict load, one real step.

The archive distributes derived public replay weights, not accepted training
checkpoints. For each public model-reconstruction config in ``configs/models`` this
script

1. hashes the lineage record and the weight artifact, and compares them with the
   digests written in ``docs/ARTIFACT_IDENTITY_LEDGER.json`` -- digests are checked
   *before* deserialization;
2. confirms the lineage record names the same accepted parent checkpoint as the
   configuration and the ledger, so the derived artifact cannot be quietly
   re-pointed at a different parent;
3. loads the weights through the strict public loader, which requires exactly the
   registered tensor key set and verifies every tensor's dtype, shape, byte count,
   raw-value digest and finiteness, then the whole model-state fingerprint;
4. resolves the architecture family from the config and constructs the model class
   it names;
5. loads the state with ``strict=True``, so a config that maps to the wrong
   architecture fails rather than silently loading a subset;
6. advances one admissible step on a small periodic field, proving the pair is
   executable and not merely loadable.

Nothing here reads a reference trajectory, and nothing here loads a pickle.
Everything runs on CPU.
"""

from __future__ import annotations

import sys

sys.dont_write_bytecode = True

import argparse
import json
from pathlib import Path

import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pinn_phase.io.artifacts import sha256_file  # noqa: E402
from pinn_phase.io.public_weights import (  # noqa: E402
    as_torch_state_dict,
    load_public_weights,
    load_weight_lineage,
)
from pinn_phase.models import (  # noqa: E402
    ExplicitMPFHybridRollout,
    PermEquivariantMPFRollout,
    architecture_family_from_config,
)

CLASSES = {
    "PermEquivariantMPFRollout": PermEquivariantMPFRollout,
    "ExplicitMPFHybridRollout": ExplicitMPFHybridRollout,
}
# Constructor keys that are documentation rather than arguments.
NON_CONSTRUCTOR = frozenset({"architecture", "arch_variant"})


def _build(settings: dict):
    class_name = settings["class_name"]
    if class_name not in CLASSES:
        raise ValueError(f"unknown public model class: {class_name!r}")
    family = architecture_family_from_config(settings)
    if str(family) != settings["architecture_family"]:
        raise ValueError(
            f"architecture family mismatch: config declares "
            f"{settings['architecture_family']!r}, resolver returns {str(family)!r}"
        )
    model_block = {k: v for k, v in settings["model"].items() if k not in NON_CONSTRUCTOR}
    physics = settings["physics"]
    kwargs = dict(
        model_dt=float(physics["model_dt"]),
        eta_px=float(physics["eta_px"]),
        mu=float(physics["mu"]),
        sigma=float(physics["sigma"]),
        **model_block,
    )
    return CLASSES[class_name](**kwargs), family


def _one_step(model, settings: dict) -> dict:
    """Advance one admissible step on a small periodic field."""
    torch.manual_seed(0)
    phases = int(settings["model"]["num_phases"])
    dims = int(settings["model"].get("spatial_dims", 2))
    extent = (8,) * dims
    phi = torch.full((1, phases, *extent), 1.0 / phases, dtype=torch.float32)
    with torch.no_grad():
        state = model.initial_state(phi)
        phi_next = model.forward_step(phi, state)[0]
    sum_error = float((phi_next.sum(dim=1) - 1.0).abs().max())
    return {
        "step_shape": list(phi_next.shape),
        "max_abs_phase_sum_error": sum_error,
        "phi_min": float(phi_next.min()),
        "phi_max": float(phi_next.max()),
        "finite": bool(torch.isfinite(phi_next).all()),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", type=Path, default=None, help="write a machine-readable report")
    args = parser.parse_args()

    ledger = json.loads((ROOT / "docs/ARTIFACT_IDENTITY_LEDGER.json").read_text(encoding="utf-8"))
    by_name = {entry["name"]: entry for entry in ledger["artifacts"]}

    report: list[dict] = []
    failures: list[str] = []
    for config_path in sorted((ROOT / "configs/models").glob("*.yaml")):
        name = config_path.stem
        settings = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        weights_path = ROOT / settings["public_weights"]
        lineage_path = ROOT / settings["weight_lineage"]
        row: dict = {"name": name, "config": str(config_path.relative_to(ROOT))}
        try:
            entry = by_name.get(name)
            if entry is None:
                raise ValueError("artifact is absent from ARTIFACT_IDENTITY_LEDGER.json")

            lineage_digest = sha256_file(lineage_path)
            if entry["weight_lineage_sha256"] != lineage_digest:
                raise ValueError(
                    f"ledger lineage digest {entry['weight_lineage_sha256']} "
                    f"!= staged bytes {lineage_digest}"
                )
            lineage = load_weight_lineage(lineage_path)
            weights_digest = sha256_file(weights_path)
            if lineage.public_weights_sha256 != weights_digest:
                raise ValueError(
                    f"lineage digest {lineage.public_weights_sha256} "
                    f"!= staged bytes {weights_digest}"
                )
            if entry["public_weights_sha256"] != weights_digest:
                raise ValueError(
                    f"ledger digest {entry['public_weights_sha256']} "
                    f"!= staged bytes {weights_digest}"
                )
            declared_parent = settings["parent_checkpoint_sha256"]
            if lineage.parent_checkpoint_sha256 != declared_parent:
                raise ValueError(
                    "lineage names a different accepted parent checkpoint than the config"
                )
            if entry["parent_checkpoint_sha256"] != declared_parent:
                raise ValueError(
                    "ledger names a different accepted parent checkpoint than the config"
                )

            state = load_public_weights(weights_path, lineage=lineage)
            model, family = _build(settings)
            missing, unexpected = model.load_state_dict(as_torch_state_dict(state), strict=True)
            if missing or unexpected:  # pragma: no cover - strict=True raises first
                raise ValueError(f"strict load reported missing={missing} unexpected={unexpected}")
            model.eval()
            row.update(
                public_weights=str(weights_path.relative_to(ROOT)),
                public_weights_sha256=weights_digest,
                weight_lineage_sha256=lineage_digest,
                parent_checkpoint_sha256=lineage.parent_checkpoint_sha256,
                parent_checkpoint_distributed=False,
                model_state_fingerprint_sha256=lineage.model_state_fingerprint_sha256,
                registered_tensors=len(state),
                architecture_family=str(family),
                class_name=settings["class_name"],
                trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
                strict_load="ok",
                **_one_step(model, settings),
            )
            row["status"] = "PASS"
        except Exception as exc:  # noqa: BLE001 - report, do not abort the sweep
            row["status"] = "FAIL"
            row["error"] = f"{type(exc).__name__}: {exc}"
            failures.append(f"{name}: {row['error']}")
        report.append(row)

    for row in report:
        if row["status"] == "PASS":
            print(
                f"PASS {row['name']}: {row['class_name']} "
                f"({row['trainable_parameters']} parameters in {row['registered_tensors']} "
                f"registered tensors), strict load ok, "
                f"one step |sum-1|max={row['max_abs_phase_sum_error']:.3e}"
            )
        else:
            print(f"FAIL {row['name']}: {row['error']}")

    if args.json is not None:
        args.json.write_text(json.dumps(report, indent=1) + "\n", encoding="utf-8")

    if failures:
        print(f"\n{len(failures)} weight-artifact identity check(s) failed.")
        return 1
    print(f"\nAll {len(report)} public weight artifacts verified, constructed and stepped.")
    print("No accepted parent checkpoint was loaded; none is distributed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
