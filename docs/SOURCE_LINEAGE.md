# Scientific source lineage

The public foundation was composed from two read-only sources rather than by
publishing a mutable research working tree.

## Accepted permutation-equivariant training lineage

| Public module | Accepted source SHA-256 |
|---|---|
| `models/perm_equivariant_mpf.py` | `26dfc5cc61fb5e4a60b8f0f0e4dc66e93a0db106ec371eea4d541e415a5369cc` |
| `training/explicit_mpf_trainer.py` | `45b1836aaf79f6e54905c21aaa661cfb0ff5cc8ca8491294a6dc1f2cd6d3481a` |
| `training/multi_ic_t0_batch.py` | `3b9ab21d9c3b9a2464662539fdadf416ec6f36c3f1e9fe1cd5f5d775c85e66c4` |

The remaining generic modules came from the same accepted six-initial-condition
training source snapshot.
The earlier public candidate supplied compatibility tests and the retained
public media listed in `media/legacy_asset_manifest.json`. Scientific source
modules still follow the accepted-source lineage above.

[`SOURCE_TRANSFORM_MAP.tsv`](SOURCE_TRANSFORM_MAP.tsv) binds every public Python
module to both its accepted-source digest and its current public digest. Run
`python scripts/verify_source_lineage.py` to verify the public side. A custodian
of the accepted snapshot can additionally pass `--accepted-source-root` to
verify both sides without embedding a private filesystem location here.

## Public adaptations

Public changes are limited to:

- removing development-only prose and environment assumptions;
- restricted, hash-first checkpoint loading;
- pickle-disabled NumPy loading;
- architecture-family and checkpoint-identity helpers;
- public packaging, tests, documentation, and release controls.

The release adds the public transfer scorer as a reviewed extraction of the frozen
metric and gate definitions. Compact fixture archives retain only arrays needed
for scoring; their manifest binds each frozen source artifact to the public
archive and every retained array value.

Deterministic rate equivalence, phase-permutation equivariance, periodic
translation equivariance, and phase-count-independent state loading are tested
at the model boundary. The shipped N25 arrays additionally reproduce the full
reviewed confirmation score without private execution metadata.

## Front-page result media

`scripts/render_current_results.py` renders the 25-grain cascade and 64-grain
animations from frozen research arrays that are not distributed in this release.
The script holds the SHA-256 of every source it accepts and refuses to draw
anything if a source does not match, so an animation cannot be built from a
drifted or substituted array. Source locations are supplied as arguments rather
than written into the file: a digest identifies an artifact independently of
where it sits, and no private path belongs in a public release.

Every number drawn on those frames — differing-pixel percentages and active-grain
counts — is recomputed from the arrays at render time and written into
`media/current_results/asset_manifest.json` alongside the source digests. None is
transcribed from a document. The corresponding accepted values, and the boundary
that separates the registered primary 64-grain result from the post-evaluation
sensitivity run, are recorded in
[`COMPLETED_EVIDENCE_LEDGER.md`](COMPLETED_EVIDENCE_LEDGER.md).


## N16/96 prospective transfer

Six unseen fixed-family initial conditions use the existing N16 checkpoint; no weight is duplicated and no training occurs. Public score records are PROVENANCE_ONLY; external arrays and the all-six renderer are hash-bound in `benchmarks/n16_96_transfer/manifest.json` and `media/n16_96_transfer/asset_manifest.json`.
