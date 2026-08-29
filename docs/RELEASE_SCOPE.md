# Release scope

## Included in v0.1

- reviewed generic model, physics, evaluation, and training modules;
- the permutation-equivariant multiphase-field implementation;
- the accepted six-initial-condition training loader;
- safe NumPy archive loading, including the strict public replay-weight loader;
- hash-verified method and benchmark media;
- completed benchmark packages distributed in this release;
- six derived public replay-weight artifacts under `checkpoints/`, each carrying
  the model state of an accepted parent checkpoint tensor-for-tensor, with a lineage
  record binding it to that parent by digest and a model-reconstruction
  configuration sufficient to rebuild its class and strict-load it;
- one reproduction-level classification per benchmark, and a machine-readable
  map from every front-page claim to the artifact that supports it.

## Excluded from v0.1

- native-`128^3` campaign artifacts and checkpoints;
- cloud-provider launch scripts and billing controls;
- private orchestration, execution, and billing records;
- private paths, coordination records, and unpublished review material;
- graph conditioning as evidence for the equivariance claim;
- the training configurations themselves, which are recorded as digests rather
  than distributed, with one exception noted in
  [`ARTIFACT_IDENTITY_LEDGER.json`](ARTIFACT_IDENTITY_LEDGER.json);
- large reference and initial-condition fields for the 25-grain, 64-grain and
  3D cube benchmarks, which are identified by digest and byte size in
  [`REPRODUCTION_LEVELS.json`](REPRODUCTION_LEVELS.json) rather than carried;
- the accepted parent checkpoints themselves, which are identified by digest in
  [`ARTIFACT_IDENTITY_LEDGER.json`](ARTIFACT_IDENTITY_LEDGER.json) rather than
  distributed; what ships is the derived public replay weights above;
- any training command: no console entry point, no `__main__`, and no script that
  starts a run. The training functions themselves ship, for audit, and will execute if
  imported and called with data of your own; what is absent is the data and identities
  needed to re-execute the runs behind the distributed models, so this release supports
  inference replay and code audit, not training reproduction;
- the scalar `64^3` spherical model, whose training path used a different
  contract from every model distributed here; the reason is recorded under
  `scalar_3d_spherical` in [`REPRODUCTION_LEVELS.json`](REPRODUCTION_LEVELS.json).

Exclusion is deliberate: active experiments must freeze independently before
their source or results can enter a public release.


## N16/96 prospective transfer

See [`SOURCE_LINEAGE.md`](SOURCE_LINEAGE.md#n1696-prospective-transfer) for
the N16/96 prospective-transfer provenance summary — kept in one place to
avoid the two files drifting apart.
