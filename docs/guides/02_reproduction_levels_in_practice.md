# Reproduction Levels in Practice

PINN-Phase records one reproduction level for each benchmark in
[`docs/REPRODUCTION_LEVELS.json`](../REPRODUCTION_LEVELS.json). The level states
what can be checked from distributed files; it does not turn an unavailable
artifact into a replay or a score.

| Level | Meaning in this repository | Example |
|---|---|---|
| `FULL_RECOMPUTATION` | Everything needed to regenerate the reported metric is distributed. | The scalar 128^2 curvature-driven shrinkage reference. |
| `SCORE_RECOMPUTATION` | Frozen arrays regenerate the reported score, without rerunning model inference. | The N25 unseen-initial-condition transfer score. |
| `CHECKPOINT_AND_CODE_REPLAY` | Derived public replay weights, lineage, a model configuration, an initial field, and rollout code can rerun a model; a missing large reference may still prevent score regeneration. | The N25 cascade development replay. |
| `PROVENANCE_ONLY` | Immutable identities and records are distributed, but the result is not independently recomputable from this tree. | The N16 96^3 prospective transfer record. |

The current replay registry is
[`docs/REPLAY_ENTRYPOINTS.json`](../REPLAY_ENTRYPOINTS.json). It names the model,
lineage, configuration, and initial field used by each supported replay.

| Example | Distributed here | External | Can be regenerated here | Cannot be obtained from this tree alone |
|---|---|---|---|---|
| Scalar shrinkage | Benchmark configuration, expected metrics, and reference solver | Nothing | Reference fields and reviewed scalar metrics | A PINN rollout, which is not part of this reference-solver benchmark |
| N25 prospective score | 40 compact arrays, manifest, expected score, and public replay weights | Full 12,000-step model and reference fields | The reviewed score from the compact arrays | The original full trajectories through this score command |
| N25 development replay | Derived public weights, lineage, model configuration, initial field, and rollout code | Accepted reference trajectory | An autonomous model replay and its admissibility checks | The reported comparison percentage without the reference |
| N16 prospective record | Compact score record, external-asset manifest, benchmark description, and media | Six sets of t0, model, and reference arrays | The public record and criteria checks | The six-case campaign and array-level verification when those assets are absent |

## Checkpoint and code replay

This bounded replay verifies the public weight, lineage, configuration, and
initial-field bytes before strict loading. It writes an unscored terminal field;
it does not open a reference trajectory or compute a score.

**Classification:** `EXECUTED_AND_PASSING`

```bash
python scripts/replay_rollout.py --benchmark n25_cascade_development --smoke --output-dir <outside-tree>/replay
```

This is `CHECKPOINT_AND_CODE_REPLAY`, not score recomputation. Its reference
trajectory is not distributed, so the reported percentage cannot be regenerated
from the public tree alone.

## Score recomputation

The following command recomputes the N25 prospective score from shipped compact
arrays and checks it against the reviewed record. It does not rerun inference.

**Classification:** `EXECUTED_AND_PASSING`

```bash
python scripts/reproduce_n25_transfer.py --output <outside-tree>/n25_transfer_score.json
```

This recomputes the campaign metrics from the shipped compact arrays and
compares them, field by field, against the reviewed `expected_score.json`.
It runs on Windows as well as POSIX platforms and raises if the recomputed
score differs from the reviewed record in any way.

## Full recomputation

The scalar reference solver regenerates its compact reference field and verified
metrics from the distributed configuration.

**Classification:** `EXECUTED_AND_PASSING`

```bash
python scripts/reproduce_scalar_reference.py --output <outside-tree>/scalar_reference
```

This is `FULL_RECOMPUTATION`: it is a reference-solver benchmark and does not
train or evaluate a PINN rollout.

## Provenance only

The N16 six-condition prospective result is `PROVENANCE_ONLY`. Its compact public
record can be inspected and verified, while the documented t0, model, and
reference arrays remain external. The next tutorial explains that distinction
without representing record verification as a campaign rerun.
