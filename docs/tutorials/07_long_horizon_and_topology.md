# Long-Horizon Evolution and Topology

## What you will learn

- why autoregressive errors accumulate beyond the represented training horizon;
- how argmax identity fields turn diffuse states into topology diagnostics;
- how survivors, extinctions, reappearances, and persistence are defined;
- why field agreement and event timing answer different questions.

## Prerequisites

Read Tutorials [01](01_phase_field_foundations.md),
[05](05_physics_informed_training.md), and
[06](06_symmetry_and_model_families.md).

## From one step to thousands

Autonomous rollout repeatedly feeds \(\phi_{t+1}\) back as the next input. Even
when one-step updates are close to the physical target, small interface-position
errors alter later neighborhoods and can accumulate. Evaluation therefore
distinguishes:

- the **represented training horizon**, over which physical losses were applied;
- the longer **extrapolation horizon**, over which the trained model is rolled out
  without optimization.

Crossing the represented horizon is not automatically failure. It is the point
where the test becomes a long-horizon generalization question.

## Identity fields and active sets

For an explicit MPF state, a crisp diagnostic label at grid point \(x\) is

$$
g_t(x)=\operatorname*{argmax}_k\phi_k(t,x).
$$

The active set is the set of labels occupying at least one grid point:

$$
A_t=\{k:\exists x,\ g_t(x)=k\}.
$$

These definitions do not replace the diffuse field. They provide discrete
questions that a field average cannot answer: which grains survive, which vanish,
and whether a vanished identity returns.

## Extinction, survivors, and reappearance

For the N25 scorer, a phase's area is counted from the argmax labels at every
step. Its extinction step is the first step of the terminal run of zero area: the
phase is absent from that point through the terminal state. Terminal survivors
are the labels with nonzero terminal area.

A reappearance diagnostic detects a phase that is absent at a saved cadence
frame and then appears at a later saved frame. Saved-frame cadence therefore
limits the precision of any event-time claim. An extinction reported at cadence
50 is not a continuously resolved event time.

**Illustrative sequence - not a benchmark output.** If one phase has saved-frame
areas

```text
step:  0  50 100 150 200
area: 12   6   0   0   0
```

its saved-frame extinction is step 100. The sequence `12, 0, 2, 0, 0` instead
contains a reappearance at step 100 and has a terminal zero run beginning at 150.

## Field agreement and persistence

Argmax disagreement between label fields \(a\) and \(b\) is

$$
D(a,b)=100\left\langle\mathbf{1}[a(x)\neq b(x)]\right\rangle_x.
$$

The model/reference disagreement asks how well the rollout matches the physical
reference. The static-$t_0$ persistence baseline compares the unchanged initial
label field with the evolved reference. Beating persistence shows that the model
captures evolution rather than merely preserving the initial segmentation.

Boundary localization asks how much disagreement lies near either model or
reference interfaces. Survivor-set F1, false deaths, extinction identities, and
event timing complement these field metrics. A low field disagreement can coexist
with a wrong extinction identity; correct topology can coexist with a shifted
interface. Both views are needed.

## Reading the public evidence conservatively

The archive separates represented-horizon evaluation, later extrapolation, and
post-evaluation sensitivity studies. It also reports failed anchors rather than
silently replacing them. In particular, boundary placement and extinction timing
are different error modes, and a timing anchor can fail even when terminal
topology and field agreement are strong.

For a concrete prospective case, continue with the
[N16 96^3 case study](../case-studies/n16_prospective_evidence.md). That result
uses a first-generation model and must not be read as evidence for phase-label
equivariance.

## What to remember

- Long rollout evaluates accumulated dynamics, not only one-step accuracy.
- Argmax labels define active identities but do not replace the diffuse field.
- Extinction precision is limited by the saved or evaluated cadence.
- Field, topology, persistence, and timing metrics are complementary.

## Next

Continue to [Run, Replay, and Reproduce](08_run_and_reproduce.md).
