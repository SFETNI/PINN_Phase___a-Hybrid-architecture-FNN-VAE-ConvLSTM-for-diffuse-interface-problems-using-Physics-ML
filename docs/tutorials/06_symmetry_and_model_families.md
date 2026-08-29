# Symmetry, Phase Labels, and PINN-Phase Model Families

## What you will learn

- what phase-permutation equivariance means physically and algebraically;
- how the later architecture enforces it;
- why periodic convolution and translation equivariance are separate claims;
- which public results belong to each model family.

## Prerequisites

Read [Inside PINN-Phase](03_inside_pinn_phase.md) and
[One Complete Time Step](04_one_pinn_phase_step.md).

## Relabeling should relabel the answer

The numerical channel index assigned to a grain is not a physical property. If
\(P\) permutes phase channels, an equivariant model satisfies

$$
f(P\phi)=P f(\phi).
$$

In plain language: renaming grains should rename the predicted channels and
change nothing else about the physical evolution.

**Illustrative example - not a benchmark output.** Suppose channels
\((1,2,3)\) are reordered to \((3,1,2)\). Applying the model after reordering
must give the original output reordered in exactly the same way. It must not
make grain 3 behave like grain 1 merely because it now occupies the first tensor
channel.

You can execute this contract with the CPU smoke check:

```bash
python scripts/smoke_test.py
```

**Classification:** `EXECUTED_AND_PASSING`

## How the equivariant family is built

[`PermEquivariantMPFRollout`](../../src/pinn_phase/models/perm_equivariant_mpf.py)
uses structural choices that commute with phase relabeling:

- no phase-index input;
- no absolute coordinate identity input;
- one shared phase encoder and one shared pointwise transformation;
- symmetric mean and maximum aggregation across all phases;
- one shared phase-wise ConvLSTM transformation;
- one global branch-blend scalar;
- phase-mean centering of the bounded increment.

For each phase \(i\), the encoder produces \(e_i\). The recurrent branch receives
the phase's own field and encoding together with the same symmetric summaries
\(\operatorname{mean}_j e_j\) and \(\operatorname{max}_j e_j\). Reordering the
phase axis therefore reorders the per-phase computations while leaving the
shared context unchanged.

All spatial convolutions in this family use circular padding. Integer shifts on
the periodic grid are tested separately as translation equivariance. That is a
spatial symmetry; it is not the same as relabeling phase channels.

## First-generation hybrid family

[`ExplicitMPFHybridRollout`](../../src/pinn_phase/models/explicit_mpf.py) also
uses circular convolution by default and can use periodic coordinate features.
It combines a pointwise branch with a shared spatial recurrent branch, as shown
in Tutorial 03. However, accepted first-generation configurations may include
phase-index and coordinate encodings, and the architecture is not claimed to be
equivariant under phase relabeling.

This precision matters: saying that the first-generation family is not
phase-permutation equivariant does **not** mean that it lacks periodic spatial
treatment.

The generic first-generation class also exposes optional graph conditioning.
That option is a configuration-specific extension, not a property of every
reported model and not part of the accepted equivariant family.

## Public result lineage

| Model family | Structural summary | Public result lineage |
|---|---|---|
| First-generation hybrid | Pointwise branch + shared ConvLSTM/ConvGRU branch; configuration-specific phase/coordinate inputs; circular convolution | Retained 8-grain 64^3 and N16 96^3 demonstrations |
| Permutation-equivariant explicit MPF | Shared per-phase transformations; symmetric cross-phase aggregation; no phase ID or absolute coordinates; circular convolution | 25-grain cascade/transfer and dense 64-grain results |
| Scalar family | Separate scalar state and loss contract | Curvature-driven scalar shrinkage and scalar development records |

The N16 prospective evidence is scientifically valuable but does not demonstrate
permutation equivariance: it uses the first-generation hybrid architecture. The
family identity for each replayable result is bound in checkpoint lineage and
artifact records, not inferred from a figure.

## What to remember

- Phase labels are arbitrary, so equivariance is the correct relabeling behavior.
- Shared transformations and symmetric aggregation enforce that behavior.
- Circular padding addresses periodic space; it is not phase-label symmetry.
- N16/96 is first-generation evidence, while N25 and dense N64 use the
  permutation-equivariant family.

## Next

Continue to [Long-Horizon Evolution and Topology](07_long_horizon_and_topology.md).
