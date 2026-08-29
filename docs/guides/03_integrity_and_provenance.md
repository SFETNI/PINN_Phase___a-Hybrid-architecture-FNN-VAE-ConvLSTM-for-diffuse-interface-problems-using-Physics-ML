# Verifying Integrity and Provenance

PINN-Phase exposes several complementary checks. They are deliberately
different: a matching SHA-256 establishes file identity, not scientific validity
on its own.

| Surface | What it establishes | What it does not establish |
|---|---|---|
| File integrity | A distributed file matches its recorded SHA-256. | That the file alone supports a scientific claim. |
| Scientific lineage | A public source module matches its recorded transformation record. | That a training run can be recreated. |
| Checkpoint identity | Derived public replay weights, lineage, configurations, and model construction agree. | That an undistributed accepted parent checkpoint is present. |
| External-asset identity | A supplied N16 asset hierarchy matches documented location, byte count, and SHA-256. | That those assets are distributed here. |
| Reproduction level | Which checks or reruns distributed evidence supports. | That every benchmark has the same availability. |

## Check file integrity

[`MANIFEST.sha256`](../../MANIFEST.sha256) is the file-integrity ledger for the
public tree.

**Classification:** `EXECUTED_AND_PASSING`

```bash
python scripts/verify_manifest.py
```

Run this command against a tree whose manifest includes the files being checked.
It hashes every listed file and confirms that the manifest covers the payload.

## Check source lineage

[`docs/SOURCE_TRANSFORM_MAP.tsv`](../SOURCE_TRANSFORM_MAP.tsv) records public
source transformation hashes.

**Classification:** `EXECUTED_AND_PASSING`

```bash
python scripts/verify_source_lineage.py
```

This verifies the distributed public source hashes against that map. It does not
load a model or establish a benchmark score.

## Check derived replay weights

[`docs/ARTIFACT_IDENTITY_LEDGER.json`](../ARTIFACT_IDENTITY_LEDGER.json) records
the derived public replay weights and the identity of each accepted parent
checkpoint.

**Classification:** `EXECUTED_AND_PASSING`

```bash
python scripts/verify_checkpoint_identities.py
```

The verifier checks each derived artifact before deserialization, strict-loads
the declared model, and advances one CPU step. The accepted parent checkpoints
are identified by digest and are not distributed.

## Check initial-field provenance

[`benchmarks/initial_conditions/manifest.json`](../../benchmarks/initial_conditions/manifest.json)
records each distributed t0-only field, its derived-file identity, its parent
identity, and a container-independent value digest. The focused public test
recomputes those relationships without opening a post-t0 trajectory.

**Classification:** `EXECUTED_AND_PASSING`

```bash
python -m pytest -q tests/test_initial_field_provenance.py --basetemp <outside-tree>/pytest-initial-fields
```

This verifies initial-field provenance. It is separate from the checkpoint
identity command above.

## Scan the public tree

**Classification:** `EXECUTED_AND_PASSING`

```bash
python scripts/check_public_tree.py
```

This scans the manifest-listed public payload for the repository content rules.
Run it with the manifest check so the payload and its file list are evaluated
together.

For N16 record and optional external-asset checks, use
[the N16 prospective-evidence case study](../case-studies/n16_prospective_evidence.md).
