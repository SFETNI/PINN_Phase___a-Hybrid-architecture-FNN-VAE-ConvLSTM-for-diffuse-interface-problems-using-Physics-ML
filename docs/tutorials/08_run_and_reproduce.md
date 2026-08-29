# Run, Replay, and Reproduce

## What you will learn

- where code, configurations, checkpoints, benchmarks, and provenance records live;
- how to run the lightweight CPU checks and a bounded model replay;
- how the four reproduction levels differ;
- how to verify archive integrity without optional large external assets.

## Prerequisites

Use Python 3.11 with the dependencies in `pyproject.toml`, or create the pinned
environment described by [`environment-cpu.yml`](../../environment-cpu.yml). No external
asset package is required for this tutorial.

## Repository map

```text
src/pinn_phase/        model, physics, training, evaluation, and I/O code
configs/               benchmark and replay configurations
checkpoints/           derived public replay-weight archives and lineage
benchmarks/            benchmark-local data, expected scores, and instructions
scripts/               smoke, replay, reproduction, and verification entry points
docs/                  tutorials, guides, method reference, and evidence ledgers
media/                 accepted figures and animations
tests/                 executable public contracts
```

## 1. Run the lightweight CPU check

From the repository root:

```bash
python scripts/smoke_test.py
```

**Classification:** `EXECUTED_AND_PASSING`

This checks periodic explicit-MPF physics, admissibility, a forward step, and the
equivariance contracts exercised by the smoke path.

## 2. Replay a distributed model

A bounded replay proves that a shipped configuration, initial field, public
weights, and executable model code work together:

```bash
python scripts/replay_rollout.py --benchmark n25_cascade_development --smoke --output-dir outputs/replay
```

**Classification:** `EXECUTED_AND_PASSING`

The smoke flag keeps the run short. Removing it requests the documented rollout
and should be treated as a deliberate, longer computation. Replay is not the same
as score recomputation when a large reference trajectory is external.

## 3. Read the reproduction level first

[`REPRODUCTION_LEVELS.json`](../REPRODUCTION_LEVELS.json) assigns one of four
levels to each public result:

| Level | Meaning |
|---|---|
| `FULL_RECOMPUTATION` | The archive contains what is needed to regenerate the reported metric. |
| `SCORE_RECOMPUTATION` | Frozen compact arrays reproduce the score; model inference is separate. |
| `CHECKPOINT_AND_CODE_REPLAY` | Weights, configuration, initial field, and code replay the model; an external reference may prevent local score regeneration. |
| `PROVENANCE_ONLY` | Immutable identities and records are present, without a claim of local recomputation. |

This vocabulary prevents a digest-only record from being called reproducible.
The practical guide is [Reproduction Levels in Practice](../guides/02_reproduction_levels_in_practice.md).

## 4. Recompute the compact benchmark scores

The prospective N25 score is recomputed from shipped, hash-bound arrays:

```bash
python scripts/reproduce_n25_transfer.py
```

**Classification:** `EXECUTED_AND_PASSING`

The compact scalar reference benchmark regenerates its field and metrics:

```bash
python scripts/reproduce_scalar_reference.py
```

**Classification:** `EXECUTED_AND_PASSING`

For N16/96, the public record and optional external-asset contract are verified
with:

```bash
python scripts/verify_n16_96_transfer.py --records benchmarks/n16_96_transfer/expected_score.json
```

**Classification:** `EXECUTED_AND_PASSING`

The approximately 10.7 GB external N16 package is optional. It is not required
to learn the method, run the lightweight checks, verify the public record, or
replay the distributed lightweight paths. No download URL is assumed here.

**Classification:** `OPTIONAL_EXTERNAL_ASSET`

## 5. Verify integrity and provenance

Run the archive-level checks:

```bash
python scripts/verify_manifest.py
python scripts/verify_source_lineage.py
python scripts/verify_checkpoint_identities.py
python -m pytest -q tests/test_initial_field_provenance.py
python scripts/check_public_tree.py --mode payload
```

**Classification:** `EXECUTED_AND_PASSING`

The manifest verifies the public path/byte identities; source lineage binds
public modules to accepted origins; checkpoint identity verifies derived replay
weights tensor by tensor; initial-field provenance binds replay inputs; and the
tree scan rejects private paths, secrets, and release residue. See
[Integrity and Provenance](../guides/03_integrity_and_provenance.md) for the
meaning of each layer.

## Command-line notes

The commands above are shell-neutral because each is a single Python invocation.
On Windows PowerShell and POSIX shells, run them from the repository root. Paths
written by replay commands use the platform's native filesystem conventions;
machine-readable public manifests use forward-slash repository paths.

## What to remember

- Check the reproduction level before interpreting a command's result.
- Score recomputation, checkpoint replay, and provenance are distinct claims.
- Lightweight learning and verification do not require the optional N16 assets.
- The public machine records, not undocumented history, define artifact identity.

## Next

Return to the [documentation map](../README.md), or use the
[CPU getting-started guide](../guides/01_getting_started_cpu.md) for a compact
hands-on walkthrough.
