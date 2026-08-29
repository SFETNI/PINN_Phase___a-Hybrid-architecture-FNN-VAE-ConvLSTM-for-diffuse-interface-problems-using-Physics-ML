#!/usr/bin/env python3
"""Score a rollout that has already been produced, against a reference you supply.

This is the only public command that accepts a reference trajectory, and it is
deliberately separate from ``scripts/replay_rollout.py``. Replay runs the model and
writes a rollout; scoring happens afterwards, on a file, by a different process. A
reference cannot reach the model through this command because the model is never
loaded here.

The archive does not distribute the large reference fields. Supply your own, or
obtain the accepted one by the digest recorded in ``docs/REPRODUCTION_LEVELS.json``.

Example::

    python scripts/evaluate_rollout.py \\
        --rollout outputs/replay/n64_dense_primary_replay.npz \\
        --reference /path/to/reference.npz --reference-member states \\
        --reference-index -1 --output-dir outputs/scores
"""

from __future__ import annotations

import sys

sys.dont_write_bytecode = True

import argparse
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pinn_phase.io.artifacts import sha256_file  # noqa: E402


def _labels(field: np.ndarray) -> np.ndarray:
    return np.argmax(field, axis=0)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--rollout", type=Path, required=True,
                        help="an .npz written by scripts/replay_rollout.py")
    parser.add_argument("--reference", type=Path, required=True,
                        help="a reference field archive that you supply")
    parser.add_argument("--reference-member", default="states")
    parser.add_argument("--reference-index", type=int, default=-1,
                        help="frame index to compare against; -1 is the last frame")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    with np.load(args.rollout, allow_pickle=False) as archive:
        if "terminal" not in archive.files:
            print(f"FAIL {args.rollout} has no 'terminal' member; is it a replay output?")
            return 1
        prediction = np.asarray(archive["terminal"])

    with np.load(args.reference, allow_pickle=False) as archive:
        if args.reference_member not in archive.files:
            print(f"FAIL reference has no member {args.reference_member!r}; "
                  f"found {sorted(archive.files)}")
            return 1
        block = np.asarray(archive[args.reference_member])
        reference = block[args.reference_index] if block.ndim == prediction.ndim + 1 else block

    if reference.shape != prediction.shape:
        print(f"FAIL shape mismatch: rollout {prediction.shape}, reference {reference.shape}")
        return 1

    predicted_labels, reference_labels = _labels(prediction), _labels(reference)
    differing = float((predicted_labels != reference_labels).mean() * 100.0)
    predicted_survivors = sorted(int(i) for i in np.unique(predicted_labels))
    reference_survivors = sorted(int(i) for i in np.unique(reference_labels))
    shared = set(predicted_survivors) & set(reference_survivors)
    precision = len(shared) / len(predicted_survivors) if predicted_survivors else 0.0
    recall = len(shared) / len(reference_survivors) if reference_survivors else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0

    report = {
        "rollout": str(args.rollout),
        "rollout_sha256": sha256_file(args.rollout),
        "reference_sha256": sha256_file(args.reference),
        "reference_member": args.reference_member,
        "reference_index": args.reference_index,
        "differing_pixels_pct": differing,
        "predicted_survivors": predicted_survivors,
        "reference_survivors": reference_survivors,
        "false_deaths": sorted(set(reference_survivors) - set(predicted_survivors)),
        "extra_survivors": sorted(set(predicted_survivors) - set(reference_survivors)),
        "terminal_survivor_f1": f1,
        "note": ("Computed after the fact from two files. No model was loaded and no "
                 "checkpoint was selected. This command cannot influence training or "
                 "replay in any way."),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    out = args.output_dir / f"{args.rollout.stem}_score.json"
    out.write_text(json.dumps(report, indent=1) + "\n", encoding="utf-8")
    print(f"differing pixels : {differing:.4f}%")
    print(f"survivors        : model {len(predicted_survivors)}, reference {len(reference_survivors)}")
    print(f"terminal F1      : {f1:.4f}")
    print(f"false deaths     : {report['false_deaths'] or 'none'}")
    print(f"wrote            : {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
