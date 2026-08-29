# Initial fields

Each archive here holds one initial field and nothing else. The member set is
exactly `phi0`; there is no label map, no target, no per-phase weight, and no
stored trajectory. `manifest.json` records, for every file, the digest of the
distributed bytes, the digest of the accepted scientific artifact it was derived
from, and a value digest over dtype, shape and contents.

## These are derived files

Extracting an initial field and re-serializing it produces a **new file with a new
digest**. `derived_sha256` identifies what is distributed here. `parent_sha256`
identifies the accepted scientific artifact. They are different files and the
derived digest must never be quoted as the identity of the original.

What ties them together is `value_sha256`: a digest of dtype, shape and raw values,
independent of container framing. It is identical for the derived field and for the
initial frame of its parent, which is what makes the derivation checkable rather
than merely asserted. No numerical transformation was applied — dtype, shape,
values and phase ordering are preserved bit-exactly.

The derivation is a packaging step. It is not new scientific evidence, and it does
not upgrade any result on its own.

## Loading

The public replay path loads these through `pinn_phase.io.load_initial_field`,
which refuses any archive whose member set is not exactly `{phi0}` — before it
reads a single array. Handing it a supervision-shaped or trajectory-shaped archive
raises `InitialFieldSchemaError`; `tests/test_phi0_strict_loader.py` proves each
refusal, including for the real historical cohort schema these fields were derived
from.

## What they enable

With an initial field present, a checkpoint, its model configuration and the rollout
code, a user can replay the model autonomously:

```bash
python scripts/replay_rollout.py --benchmark n64_dense_primary --smoke --output-dir outputs/replay
```

That is *replay*, not *recomputation of the published metric*. The large reference
trajectories are not distributed, so scoring a replay against the accepted reference
requires obtaining it separately — see `docs/REPRODUCTION_LEVELS.json`. Scoring is a
separate command, `scripts/evaluate_rollout.py`, which never loads a model.
