#!/usr/bin/env python3
"""Model-only autonomous replay from public replay weights and an initial field.

This command runs the model and nothing else. It never opens a reference
trajectory, never computes a score, and never selects a checkpoint. Scoring is a
separate command, ``scripts/evaluate_rollout.py``, which is the only place a
reference may be supplied, and it runs after a rollout has already been written.

What gets loaded is a **derived public replay-weight artifact**: model-state
tensors, no pickle, no optimizer state, no run metadata. The accepted parent
scientific checkpoint is identified by digest in the lineage record beside it and
is not distributed; no public command here loads one.

Before anything is loaded, four identities are verified against the bytes on disk:
the weight artifact, its lineage record, the model-reconstruction configuration and
the initial field. The weights are then read through the strict public loader,
which requires exactly the registered tensor key set and checks every tensor's
dtype, shape, byte count and raw-value digest before returning it. The exact
architecture is constructed from the configuration and every tensor is
strict-loaded. The initial field is read through the strict ``{phi0}`` loader,
which refuses any archive carrying a label, a target, a weight, or a stored
trajectory.

Examples
--------
Bounded CPU smoke, a few steps, enough to prove the pair is executable::

    python scripts/replay_rollout.py --benchmark n25_cascade_development \\
        --smoke --output-dir outputs/replay

The documented full rollout, run separately and deliberately::

    python scripts/replay_rollout.py --benchmark n25_cascade_development \\
        --steps 12000 --output-dir outputs/replay
"""

from __future__ import annotations

import sys

sys.dont_write_bytecode = True

import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pinn_phase.io.artifacts import sha256_file  # noqa: E402
from pinn_phase.io.phi0 import admissibility, load_initial_field  # noqa: E402
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
NON_CONSTRUCTOR = frozenset({"architecture", "arch_variant"})
SMOKE_STEPS = 4


def _registry() -> dict:
    return json.loads((ROOT / "docs/REPLAY_ENTRYPOINTS.json").read_text(encoding="utf-8"))


def _build(settings: dict, num_phases: int):
    family = architecture_family_from_config(settings)
    if str(family) != settings["architecture_family"]:
        raise ValueError(
            f"architecture family mismatch: configuration declares "
            f"{settings['architecture_family']!r}, resolver returns {str(family)!r}"
        )
    model_block = {k: v for k, v in settings["model"].items() if k not in NON_CONSTRUCTOR}
    if int(model_block["num_phases"]) != num_phases:
        raise ValueError(
            f"initial field carries {num_phases} phases but the configuration declares "
            f"{model_block['num_phases']}"
        )
    physics = settings["physics"]
    model = CLASSES[settings["class_name"]](
        model_dt=float(physics["model_dt"]),
        eta_px=float(physics["eta_px"]),
        mu=float(physics["mu"]),
        sigma=float(physics["sigma"]),
        **model_block,
    )
    return model, family


def main() -> int:
    registry = _registry()
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--benchmark", required=True, choices=sorted(registry["entrypoints"]))
    parser.add_argument("--steps", type=int, default=None,
                        help="number of autonomous steps; defaults to the documented horizon")
    parser.add_argument("--smoke", action="store_true",
                        help=f"bounded CPU run of {SMOKE_STEPS} steps, for executability only")
    parser.add_argument("--save-every", type=int, default=0,
                        help="also save intermediate frames every N steps (0 = terminal only)")
    parser.add_argument("--output-dir", type=Path, required=True,
                        help="directory to write into; nothing is written anywhere else")
    args = parser.parse_args()

    entry = registry["entrypoints"][args.benchmark]
    weights_path = ROOT / entry["public_weights"]
    lineage_path = ROOT / entry["weight_lineage"]
    config_path = ROOT / entry["model_config"]
    phi0_path = ROOT / entry["initial_field"]

    print(f"benchmark        : {args.benchmark}")
    print(f"architecture     : {entry['class_name']} ({entry['architecture_family']})")

    # --- identities first, before anything is deserialized -------------------
    for label, path, expected in (
        ("public weights", weights_path, entry["public_weights_sha256"]),
        ("weight lineage", lineage_path, entry["weight_lineage_sha256"]),
        ("model config", config_path, entry["model_config_sha256"]),
        ("initial field", phi0_path, entry["initial_field_sha256"]),
    ):
        actual = sha256_file(path)
        if actual != expected:
            print(f"FAIL {label} digest mismatch: expected {expected}, got {actual}")
            return 1
        print(f"{label:17s}: {actual}  verified")

    lineage = load_weight_lineage(lineage_path)
    if lineage.parent_checkpoint_sha256 != entry["parent_checkpoint_sha256"]:
        print("FAIL lineage names a different accepted parent checkpoint than the registry")
        return 1
    print(f"parent checkpoint: {lineage.parent_checkpoint_sha256}  "
          "(accepted scientific artifact, identified by digest, not distributed)")

    settings = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    field = load_initial_field(
        phi0_path,
        expected_sha256=entry["initial_field_sha256"],
        expected_num_phases=int(settings["model"]["num_phases"]),
    )
    print(f"initial field    : {field.phi0.dtype} {tuple(field.phi0.shape)}")
    print(f"  admissibility  : {admissibility(field.phi0)}")

    state = load_public_weights(weights_path, lineage=lineage)
    model, family = _build(settings, field.num_phases)
    model.load_state_dict(as_torch_state_dict(state), strict=True)
    model.eval()
    parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"strict load      : ok, {len(state)} registered tensors, "
          f"{parameters:,} trainable parameters")
    print(f"  fingerprint    : {lineage.model_state_fingerprint_sha256}")

    steps = SMOKE_STEPS if args.smoke else (args.steps or int(entry["documented_horizon"]))
    print(f"rollout          : {steps} autonomous steps on CPU, model only, no reference")

    phi = torch.from_numpy(np.ascontiguousarray(field.phi0)).to(torch.float32).unsqueeze(0)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    started = perf_counter()
    saved: list[int] = []
    frames: list[np.ndarray] = []
    with torch.no_grad():
        recurrent = model.initial_state(phi)
        for step in range(1, steps + 1):
            phi, recurrent = model.forward_step(phi, recurrent)[:2]
            if args.save_every and step % args.save_every == 0:
                saved.append(step)
                frames.append(phi.squeeze(0).numpy().copy())
            if steps >= 100 and step % max(1, steps // 10) == 0:
                print(f"  step {step}/{steps}")
    elapsed = perf_counter() - started

    terminal = phi.squeeze(0).numpy()
    sums = terminal.sum(axis=0)
    report = {
        "benchmark": args.benchmark,
        "architecture_family": str(family),
        "class_name": entry["class_name"],
        "public_weights_sha256": entry["public_weights_sha256"],
        "weight_lineage_sha256": entry["weight_lineage_sha256"],
        "model_state_fingerprint_sha256": lineage.model_state_fingerprint_sha256,
        "parent_checkpoint_sha256": lineage.parent_checkpoint_sha256,
        "parent_checkpoint_distributed": False,
        "model_config_sha256": entry["model_config_sha256"],
        "initial_field_sha256": entry["initial_field_sha256"],
        "initial_field_value_sha256": field.value_sha256,
        "registered_tensors": len(state),
        "trainable_parameters": parameters,
        "steps": steps,
        "smoke": bool(args.smoke),
        "save_every": args.save_every,
        "saved_steps": saved,
        "device": "cpu",
        "wall_seconds": elapsed,
        "terminal_phi_min": float(terminal.min()),
        "terminal_phi_max": float(terminal.max()),
        "terminal_max_abs_phase_sum_error": float(np.abs(sums - 1.0).max()),
        "terminal_active_phases": int((terminal.max(axis=tuple(range(1, terminal.ndim))) > 0.5).sum()),
        "reference_opened": False,
        "scored": False,
        "note": ("Model-only replay. No reference trajectory was opened and no score was "
                 "computed. Scoring is scripts/evaluate_rollout.py, run separately. The "
                 "reference_opened and scored fields above are statements of what this "
                 "command does, not measurements; tests/test_replay_entrypoints.py checks "
                 "the property by instrumenting file opens during an actual replay."),
    }
    out = args.output_dir / f"{args.benchmark}_replay.npz"
    np.savez_compressed(
        out,
        terminal=terminal,
        **({"frames": np.stack(frames), "saved_steps": np.asarray(saved)} if frames else {}),
    )
    (args.output_dir / f"{args.benchmark}_replay.json").write_text(
        json.dumps(report, indent=1) + "\n", encoding="utf-8"
    )
    print(f"wrote            : {out}")
    print(f"  |sum-1|max     : {report['terminal_max_abs_phase_sum_error']:.3e}")
    print(f"  active phases  : {report['terminal_active_phases']} of {field.num_phases}")
    print(f"  wall           : {elapsed:.2f} s")
    print("\nModel-only replay complete. No reference was read; nothing was scored.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
