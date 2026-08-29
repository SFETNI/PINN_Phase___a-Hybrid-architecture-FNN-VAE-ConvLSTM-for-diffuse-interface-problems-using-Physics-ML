# Method overview

PINN-Phase is a family of neural time integrators for curvature-driven
phase-field evolution. For the reported explicit-MPF path, training evaluates a
physical target on the model's own evolving state. Inference then advances that
state autonomously with a learned bounded update and a configured admissibility
map.

For a derivation and worked examples, start with the
[physics-first tutorials](tutorials/README.md). This page is the compact reference.

## Shared rollout pattern

Each accepted explicit-MPF step has four conceptual operations:

1. local and spatial neural branches produce a response;
2. the accepted scaled-bounded path forms a bounded, zero-sum increment;
3. the increment advances the current phase field;
4. a strict or thresholded state map restores channel bounds and unit sum.

The explicit MPF right-hand side supplies the physical target during training.
It is not evaluated during ordinary autonomous inference. Tutorial
[04](tutorials/04_one_pinn_phase_step.md) distinguishes rate projection from
state admissibility; Tutorial [05](tutorials/05_physics_informed_training.md)
shows the training and inference data flows.

The animation in `media/method_loop_hybrid_rollout.gif` illustrates the
first-generation hybrid realization with hard clip-and-renormalize. It is a
method schematic, not a benchmark rendering or a diagram of the
permutation-equivariant encoder.

## Model families

### Permutation-equivariant explicit MPF

`PermEquivariantMPFRollout` applies shared per-phase encoders and recurrent
transformations with symmetric mean/maximum cross-phase aggregation. Reordering
phase channels therefore reorders the output. This family uses circular spatial
convolution, no phase-index input, and no absolute coordinate identity input. It
supports the 25-grain and dense 64-grain results. Its machine identifier is
`perm_equivariant_mpf`.

### First-generation hybrid explicit MPF

`ExplicitMPFHybridRollout` combines a pointwise branch with a shared
convolutional recurrent branch. Accepted configurations can include phase-index
and periodic coordinate encodings, so this family is not claimed to be
phase-permutation equivariant. It nevertheless uses circular spatial treatment
where configured. It supports the retained 8-grain 64^3 and N16 96^3 evidence.
Its machine identifier is `legacy_pa_hybrid_mpf`.

### Scalar family

The scalar family provides Allen-Cahn-style reference and neural rollout
utilities with a distinct scalar state and loss contract. Its compact benchmark
regenerates a shrinking interface and checks sampled energy together with the
radius-squared law. It is not the one-channel reduction of the explicit MPF
simplex state.

## State-map conventions

Accepted configurations identify either strict clip-and-renormalize
(`simplex_clip_normalize`) or thresholded pruning plus renormalization
(`soft_threshold_eps1e3`). The latter string is a historical code identifier,
not a claim that the operation is an unspecified soft map. An optional pre-map
phase-sum penalty is a separate training term and does not replace either hard
state operation.

## Training boundary

For the reported explicit-MPF models, the physical target is evaluated from the
current model field. Post-initial reference states do not create the target,
enter the loss, select checkpoints, or control early stopping. They are reserved
for later evaluation. Exact path-by-path scope is recorded in
[`TRAINING_PATH_DISCLOSURE.json`](TRAINING_PATH_DISCLOSURE.json).

## Machine identity

Public weight records bind the architecture family, class, configuration digest,
derived replay-weight digest, accepted parent-checkpoint digest, and tensor-level
identity. The derived public artifact and accepted parent are distinct objects
and their digests are not interchangeable. See
[`ARTIFACT_IDENTITY_LEDGER.json`](ARTIFACT_IDENTITY_LEDGER.json).

## Source map

| Component | Public path |
|---|---|
| Permutation-equivariant model | `src/pinn_phase/models/perm_equivariant_mpf.py` |
| First-generation hybrid model | `src/pinn_phase/models/explicit_mpf.py` |
| Recurrent cells | `src/pinn_phase/models/cells.py` |
| Explicit MPF physics | `src/pinn_phase/physics/explicit_mpf.py` |
| Scalar reference solver | `src/pinn_phase/physics/reference_solver.py` |
| Explicit-MPF trainer | `src/pinn_phase/training/explicit_mpf_trainer.py` |
| Initial-condition-only multi-IC loader | `src/pinn_phase/training/multi_ic_t0_batch.py` |
| Held-out transfer scorer | `src/pinn_phase/evaluation/n25_transfer.py` |
