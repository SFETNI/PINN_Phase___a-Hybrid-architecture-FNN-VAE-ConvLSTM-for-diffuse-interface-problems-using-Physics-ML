# Inside PINN-Phase: Local and Spatial Dynamics

## What you will learn

- how the first-generation hybrid model combines local and spatial branches;
- what recurrent hidden and cell states retain across steps;
- how tensor shapes pass through the two branches;
- which statements belong only to the later permutation-equivariant family.

## Prerequisites

Read Tutorials [01](01_phase_field_foundations.md) and
[02](02_explicit_mpf_operator.md). Familiarity with tensors and convolution is
helpful but not required.

## The first-generation hybrid

For the first-generation model, the two neural branches produce channel-wise
responses that are combined as

$$
z_t=f_\theta(\phi_t)
=\gamma z_t^{\mathrm{loc}}+(1-\gamma)z_t^{\mathrm{sp}},
\qquad 0<\gamma<1.
$$

The implementation stores an unconstrained blend logit and applies a sigmoid,
so the learned global blend remains between zero and one. Some generic options
also support a per-phase blend, but that is not a property to assume for every
distributed model.

### Site-local branch

The pointwise MLP applies the same stack at every grid site, once per phase, and
returns one response per phase. In the first-generation class each evaluation
takes that single phase's own local fraction and, depending on configuration, a
phase-index encoding and periodic coordinate features. The phase-index and
coordinate inputs are shortcuts available to that family; they must not be
attributed to the later equivariant model.

### Spatial recurrent branch

The ConvLSTM branch sees spatial neighborhoods and carries a hidden state and a
cell state from one autoregressive step to the next. Circular padding is the
default for its convolutions, so spatial neighborhoods wrap across periodic
boundaries. A convolutional head maps the recurrent features back to one
response channel per phase.

The output heads are zero-initialized when the verified constructor option is
enabled. That makes the initial response inert before training, but it does not
mean a trained checkpoint continues to output zero.

## Shape walkthrough

**Illustrative example - not a benchmark configuration.** Let

```text
batch B = 1
phases N = 3
grid H = W = 16
field phi_t shape = [1, 3, 16, 16]
```

The pointwise branch is evaluated once per phase per site. For phase \(k\) at a
given site it reads that phase's own scalar fraction \(\phi_k\), augmented with
the configured phase-index and coordinate features, and returns a single scalar
response. It does not receive the other phases' local values, so there is no
cross-phase coupling inside this branch. Vectorizing those per-phase
evaluations over the batch, phase, and spatial axes gives

```text
local response shape = [1, 3, 16, 16]
```

The ConvLSTM receives the complete field and a recurrent state such as

```text
hidden shape = [1, C_h, 16, 16]
cell shape   = [1, C_h, 16, 16]
```

and its convolutional head returns

```text
spatial response shape = [1, 3, 16, 16]
```

The scalar \(\gamma\) broadcasts over batch, phase, and space to blend the two
responses without changing shape. The result then enters the bounded-update and
state-map operations in Tutorial 04.

## What this diagram does not claim

The first-generation class also exposes optional graph conditioning and generic
rate parameterizations. Those API options are not universal properties of every
reported checkpoint. The accepted architecture family and configuration record
remain the authority for a particular result.

The later permutation-equivariant family shares transformations across phases,
uses symmetric cross-phase aggregation, and removes phase-index and absolute
coordinate identity inputs. Its recurrent state is phase-wise rather than the
single shared state sketched above. That family is taught in Tutorial 06.

## What to remember

- The first-generation hybrid blends a site-local response with a spatial,
  recurrent response.
- Circular convolution handles periodic neighborhoods in this family.
- Recurrent state carries spatial memory across autonomous steps.
- Input features and optional components are configuration-specific.
- Periodic spatial treatment alone does not imply phase-permutation equivariance.

## Next

Continue to [One Complete PINN-Phase Time Step](04_one_pinn_phase_step.md).
