# Changelog

All notable changes to this project are documented here. This project follows
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## Unreleased

### Added

- Permutation-equivariant multiphase-field model (`PermEquivariantMPFRollout`),
  a 9,605-parameter operator whose predictions are equivariant under relabelling
  of the phase channels.
- Prospective transfer benchmark: a ten-case cohort — eight unseen initial
  conditions and two stress cases — scored once against criteria fixed before the
  cases existed. The complete score recomputes offline from 2.6 MB of shipped,
  hash-bound arrays via `scripts/reproduce_n25_transfer.py`.
- Compact scalar shrinkage benchmark that regenerates locally and checks the
  radius-squared law and monotone sampled energy.
- CPU smoke check covering periodic multiphase-field physics, the admissibility
  map, a forward pass, and both phase-permutation and translation
  equivariance.
- Hash-first artifact loading: model weights ship as derived public replay-weight
  archives carrying model-state tensors only, verified by SHA-256 before the
  container is opened, required to carry exactly their registered tensor keys, and
  checked tensor by tensor against a recorded dtype, shape, byte count and
  raw-value digest; every NumPy archive is hash-verified, container-validated, and
  loaded with pickle disabled. No pickle is distributed.
- Evidence ledger binding every published result to accepted artifacts by
  SHA-256, and a source-lineage map binding every public module to its accepted
  source digest.
- Animations for the 25-grain cascade and the dense 64-grain case, rendered by
  `scripts/render_current_results.py` from digest-pinned arrays, with every
  displayed number recomputed at render time.
- Interior-evolution animation for Unseen microstructure 3
  (`media/n16_96_interior/`), with its deterministic renderer, binding tests,
  and source-lineage row; no science, weights, scores, or arrays changed.
- Release verification tooling: manifest build and verification, public-tree
  scanning, and source-lineage verification.

### Changed

- Documentation now names model families in plain language and states, for each
  result, whether it is development evidence or a prospectively fixed evaluation.
- The dense 64-grain section reports the registered primary result first and
  labels the horizon-extension run as a post-evaluation sensitivity study.
- The 25-grain cascade animation carries the same represented-horizon annotation
  as the 64-grain one: its diagnostic panel is divided at the 4,096-step training
  horizon and labels both sides in the frame itself.
- The reference-usage guard is fail-closed: explicit-MPF training requires the
  complete reviewed reference-usage declaration, with literal boolean values and a
  mandatory supervision policy, and refuses a missing, empty, misspelled or
  wrongly-typed declaration rather than passing it.
- One verification workflow that runs from a fresh extraction. The content scanner
  checks the manifest-defined payload by default, so a reader's own outputs and
  caches no longer make it fail, and `--mode staging` retains the strict whole-tree
  walk used when packaging.
- Model weights are distributed as derived public replay-weight artifacts rather
  than as training checkpoints. Each carries the model state of its accepted parent
  tensor-for-tensor, with a lineage record binding it to that parent by digest, a
  model-state fingerprint, and a per-tensor identity; the parents themselves are
  identified by digest and are not distributed.
- Documentation reorganized. A new eight-part physics-first tutorial curriculum
  (`docs/tutorials/`) explains diffuse-interface foundations, the explicit MPF
  operator, the model, one constrained step, physics-informed training,
  symmetry, long-horizon topology, and reproduction in sequence; the four
  original tutorials are retained with a changed role
  as how-to guides (`docs/guides/`) and one benchmark case study
  (`docs/case-studies/`); `docs/README.md` is a new reader map across the
  whole documentation tree; `docs/ARCHITECTURE_FAMILIES.md` is merged into
  `docs/METHOD_OVERVIEW.md`; `docs/benchmarks/n25-generalization.md` is
  merged into `benchmarks/n25_transfer/README.md`. No science, weights,
  scores, benchmarks, or configurations changed.

### Known limitations

- Results come from designed benchmarks. The project makes no claim of
  statistical grain growth, universal kinetics, or scaling.
- The pre-registered extinction-timing anchor on the 25-grain cascade is not met;
  extinction events are predominantly premature.
- The prospective cohort passes 7 of 8 unseen cases; the single strict failure is
  reported and counted.
- Extinction timing is claimed only at saved-frame resolution.
- Large reference and rollout arrays for the 64-grain, 64³, and 96³ cases are not
  distributed in this release. Their digests are recorded in the evidence ledger.
- No public training entry point is provided. The release supports inference replay
  and code audit, not training reproduction.
