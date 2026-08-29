# Derived public replay weights

This directory does **not** contain training checkpoints. It contains six derived
public replay-weight artifacts, one per model, each with a lineage record beside it.

The distinction is load-bearing and is kept everywhere in this archive:

| | accepted parent checkpoint | derived public replay weights |
|---|---|---|
| what it is | the scientific training payload | a repackaging of that payload's model state |
| carries | model state, and for some runs optimizer state, schedule state and free-form run metadata written at training time | model-state tensors, and nothing else |
| format | a pickled PyTorch payload | a NumPy archive; no pickle anywhere in it |
| distributed here | **no** — identified by SHA-256 only | yes |
| identity | the digest of record for the scientific artifact | a *packaging* digest; never the identity of the parent |

Deriving these artifacts changed packaging only. Every tensor name, dtype, shape
and raw value is the parent's, unchanged, and no model parameter differs. That is a
claim you can check rather than take on faith: each `*.lineage.json` records the
parent's digest, the derived digest, a fingerprint over the whole model state, and
the dtype, shape, byte count and raw-value digest of every individual tensor.

## Files

- `<name>.weights.npz` — the weights. Members are one `.npy` per tensor, stored
  uncompressed, in sorted name order, with fixed timestamps and permissions, so
  rebuilding from the same parent produces byte-identical output. Loading uses
  `allow_pickle=False`.
- `<name>.lineage.json` — the registered identity of that artifact.
- `SHA256SUMS` — digests of every file here.

## Reading them

```python
from pinn_phase.io.public_weights import (
    load_public_weights, load_weight_lineage, as_torch_state_dict,
)

lineage = load_weight_lineage("checkpoints/n25_cascade_primary_permequiv.lineage.json")
state = load_public_weights("checkpoints/n25_cascade_primary_permequiv.weights.npz",
                            lineage=lineage)
model.load_state_dict(as_torch_state_dict(state), strict=True)
```

The loader hashes the file before opening it, requires the member set to be exactly
the registered tensor keys — checked before any array is decoded — and verifies
each tensor's dtype, shape, byte count, raw-value digest and finiteness, then the
fingerprint of the whole state. An artifact that has been edited, truncated,
extended, re-typed or re-ordered does not load.

Verify all six, end to end, with:

```bash
python scripts/verify_checkpoint_identities.py
```

## Re-deriving

`scripts/derive_public_weights.py` is the script that produced these files, and its
digest is recorded in every lineage record. If you hold an accepted parent
checkpoint, you can re-run it and compare:

```bash
python scripts/derive_public_weights.py \
    --checkpoint /path/to/accepted/checkpoint.pt \
    --artifact <name> --benchmark <benchmark> \
    --model-config configs/models/<name>.yaml \
    --output-dir /tmp/derived
```

It takes the parent as an argument, writes no location into anything it produces,
builds the artifact twice in separate directories and requires the two to be
byte-identical, and reads the result back through the strict loader before writing
a lineage record.

## What this does not give you

Replaying a model is not reproducing a published number. These weights let you rerun
the models; the large accepted reference trajectories that the published percentages
are measured against are not distributed here. See
[`../docs/REPRODUCTION_LEVELS.json`](../docs/REPRODUCTION_LEVELS.json), which says
per benchmark which of the two you can do.

No training command ships either, and the training configurations, training initial
conditions and accepted parent checkpoints are not distributed, so the runs that
produced these weights cannot be re-executed here. This archive supports inference
replay and code audit, not training reproduction.
