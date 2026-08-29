#!/usr/bin/env python3
"""Derive public replay weights from an accepted parent checkpoint.

This is the script that produced every ``checkpoints/*.weights.npz`` in this
archive, and its digest is recorded in each lineage record. It takes the parent
checkpoint as an argument: no location is written into it, and none is written
into anything it produces.

What it does, in order:

1. hashes the parent checkpoint and, if a digest is supplied, refuses to proceed
   unless the bytes are the accepted ones;
2. reads the parent through PyTorch's restricted, weights-only loader and takes
   the model state -- only the model state. Optimizer state, schedule state and
   run metadata are left behind, unread;
3. refuses any tensor whose dtype cannot be represented exactly, rather than
   casting, reshaping, renaming or normalizing it;
4. writes the tensors to a deterministic archive: sorted names, stored members,
   fixed timestamps and permissions, no pickle;
5. builds the artifact a second time in a separate directory and requires the two
   to be byte-identical;
6. loads the result back through the strict public loader and requires the
   model-state fingerprint to equal the parent's;
7. writes a neutral lineage record: digests, tensor identities, and no free-form
   metadata copied from the parent.

The derivation is a packaging transformation. It does not change a model
parameter, and the artifact it writes is not the accepted scientific checkpoint.

Example::

    python scripts/derive_public_weights.py \\
        --checkpoint /path/to/accepted/checkpoint.pt \\
        --parent-sha256 <digest of the accepted checkpoint> \\
        --artifact n25_cascade_primary_permequiv \\
        --benchmark n25_cascade_development \\
        --model-config configs/models/n25_cascade_primary_permequiv.yaml \\
        --output-dir checkpoints
"""

from __future__ import annotations

import sys

sys.dont_write_bytecode = True

import argparse
import json
from pathlib import Path
import tempfile

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pinn_phase.io.artifacts import load_torch_checkpoint, sha256_file  # noqa: E402
from pinn_phase.io.public_weights import (  # noqa: E402
    PublicWeightsError,
    lineage_document,
    load_public_weights,
    load_weight_lineage,
    model_state_fingerprint,
    tensor_records,
    write_public_weights,
)

STATE_KEYS = ("model_state", "model_state_dict")


def _model_state(payload) -> dict:
    for key in STATE_KEYS:
        if key in payload:
            state = payload[key]
            if not isinstance(state, dict):
                raise PublicWeightsError(f"{key} is not a mapping")
            return dict(state)
    raise PublicWeightsError(f"parent carries no model state under any of {STATE_KEYS}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--checkpoint", type=Path, required=True,
                        help="path to the accepted parent checkpoint")
    parser.add_argument("--parent-sha256", default=None,
                        help="accepted parent digest; refuses to derive from other bytes")
    parser.add_argument("--artifact", required=True, help="public artifact name")
    parser.add_argument("--benchmark", required=True, help="public benchmark identifier")
    parser.add_argument("--model-config", type=Path, required=True,
                        help="the public model-reconstruction configuration, repository-relative")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    parent_digest = sha256_file(args.checkpoint)
    if args.parent_sha256 is not None and parent_digest != args.parent_sha256.strip().lower():
        print(f"FAIL parent digest mismatch: accepted {args.parent_sha256}, bytes {parent_digest}")
        return 1

    payload = load_torch_checkpoint(args.checkpoint, expected_sha256=parent_digest)
    state = _model_state(payload)
    records = tensor_records(state)
    fingerprint = model_state_fingerprint(state)

    weights_path = args.output_dir / f"{args.artifact}.weights.npz"
    digest = write_public_weights(state, weights_path)

    # Determinism: build again, elsewhere, and require the same bytes.
    with tempfile.TemporaryDirectory() as first, tempfile.TemporaryDirectory() as second:
        digests = {
            write_public_weights(state, Path(directory) / "probe.npz")
            for directory in (first, second)
        }
    if digests != {digest}:
        print(f"FAIL rebuild is not deterministic: {sorted(digests | {digest})}")
        return 1

    config_relative = args.model_config
    if config_relative.is_absolute():
        config_relative = config_relative.relative_to(ROOT)
    document = lineage_document(
        benchmark=args.benchmark,
        artifact=args.artifact,
        public_weights_path=str(weights_path.relative_to(ROOT))
        if weights_path.is_absolute()
        else str(weights_path),
        public_weights_sha256=digest,
        public_weights_bytes=weights_path.stat().st_size,
        parent_checkpoint_sha256=parent_digest,
        model_state_fingerprint=fingerprint,
        records=records,
        derivation_script="scripts/derive_public_weights.py",
        derivation_script_sha256=sha256_file(Path(__file__)),
        model_config=str(config_relative),
        model_config_sha256=sha256_file(ROOT / config_relative),
    )
    lineage_path = args.output_dir / f"{args.artifact}.lineage.json"
    lineage_path.write_text(json.dumps(document, indent=1) + "\n", encoding="utf-8")

    # Read the artifact back the way the public replay path will read it.
    restored = load_public_weights(weights_path, lineage=load_weight_lineage(lineage_path))
    if model_state_fingerprint(restored) != fingerprint:
        print("FAIL round-trip fingerprint differs from the parent model state")
        return 1
    for name, array in restored.items():
        original = state[name].detach().cpu().numpy() if isinstance(state[name], torch.Tensor) else state[name]
        if array.dtype != original.dtype or array.shape != original.shape:
            print(f"FAIL {name}: dtype or shape changed in the round trip")
            return 1
        if array.tobytes() != original.tobytes():
            print(f"FAIL {name}: raw bytes changed in the round trip")
            return 1

    print(f"artifact          : {args.artifact}")
    print(f"parent sha256     : {parent_digest}  (accepted checkpoint, not distributed)")
    print(f"public weights    : {weights_path}")
    print(f"  sha256          : {digest}")
    print(f"  bytes           : {weights_path.stat().st_size:,}")
    print(f"  tensors         : {len(records)}, {document['parameter_count']:,} parameters")
    print(f"  fingerprint     : {fingerprint}")
    print(f"lineage           : {lineage_path}")
    print("packaging only: no model parameter was changed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
