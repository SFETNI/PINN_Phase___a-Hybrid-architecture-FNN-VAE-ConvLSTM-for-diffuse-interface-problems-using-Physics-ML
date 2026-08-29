# PINN-Phase documentation

Choose the path that matches what you want to do. The project
[`README.md`](../README.md) is the results-focused landing page; this page maps
the scientific explanation, runnable guides, benchmarks, and provenance records.

## Learn the method

Start with the [eight-part physics-first tutorial curriculum](tutorials/README.md):

1. phase-field foundations and the admissible state;
2. the projected explicit MPF operator;
3. local and recurrent neural dynamics;
4. one bounded, admissible model step;
5. physical-target training and autonomous inference;
6. symmetry and model-family lineage;
7. long-horizon field and topology metrics;
8. replay, reproduction levels, and provenance.

Use [`METHOD_OVERVIEW.md`](METHOD_OVERVIEW.md) as the compact reference after
the tutorials.

## Run a lightweight example

- [Getting started: a CPU step through PINN-Phase](guides/01_getting_started_cpu.md)
- [Tutorial 08: run, replay, and reproduce](tutorials/08_run_and_reproduce.md)

## Explore benchmark evidence

- the root [`README.md`](../README.md), including the benchmark ladder and
  results-at-a-glance table;
- [`MEDIA_GALLERY.md`](MEDIA_GALLERY.md), the complete figure and animation map;
- [`benchmarks/n25_transfer/README.md`](../benchmarks/n25_transfer/README.md),
  [`benchmarks/n16_96_transfer/README.md`](../benchmarks/n16_96_transfer/README.md),
  [`benchmarks/initial_conditions/README.md`](../benchmarks/initial_conditions/README.md),
  and [`benchmarks/scalar_shrinkage_2d/README.md`](../benchmarks/scalar_shrinkage_2d/README.md);
- [N16 96^3 prospective evidence](case-studies/n16_prospective_evidence.md), a
  case study in reading field, topology, and qualification claims together.

## Reproduce and verify

- [Reproduction levels in practice](guides/02_reproduction_levels_in_practice.md)
- [Integrity and provenance](guides/03_integrity_and_provenance.md)
- [`REPRODUCTION_LEVELS.json`](REPRODUCTION_LEVELS.json) and
  [`REPLAY_ENTRYPOINTS.json`](REPLAY_ENTRYPOINTS.json), the machine-readable
  contracts behind public replay and recomputation claims

## Audit scientific identity

These records are intentionally later in the reader journey. They make the
claims auditable without being prerequisites for learning the method:

- [`RELEASE_SCOPE.md`](RELEASE_SCOPE.md)
- [`SOURCE_LINEAGE.md`](SOURCE_LINEAGE.md)
- [`COMPLETED_EVIDENCE_LEDGER.md`](COMPLETED_EVIDENCE_LEDGER.md)
- [`ARTIFACT_IDENTITY_LEDGER.json`](ARTIFACT_IDENTITY_LEDGER.json)
- [`CLAIM_TO_ARTIFACT_MAP.json`](CLAIM_TO_ARTIFACT_MAP.json)
- [`TRAINING_PATH_DISCLOSURE.json`](TRAINING_PATH_DISCLOSURE.json)
- [`SOURCE_TRANSFORM_MAP.tsv`](SOURCE_TRANSFORM_MAP.tsv)
