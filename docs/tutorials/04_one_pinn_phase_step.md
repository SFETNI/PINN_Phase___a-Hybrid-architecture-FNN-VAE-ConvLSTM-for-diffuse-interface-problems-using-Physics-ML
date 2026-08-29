# One Complete PINN-Phase Time Step

## What you will learn

- how a neural response becomes a bounded, zero-sum increment;
- how the strict state map restores bounds and unit sum;
- how the thresholded state map differs from strict clip-and-renormalize;
- why an auxiliary phase-sum loss is not a replacement for either operation.

## Prerequisites

Read [Inside PINN-Phase](03_inside_pinn_phase.md) and the zero-sum construction
in [The Multiphase-Field Operator](02_explicit_mpf_operator.md).

## 1. Bound and center the proposed update

For the accepted scaled-bounded parameterization, a neural response \(z_t\)
becomes

$$
\Delta\phi_t=\Delta\phi_{\max}P_N[\tanh(z_t)].
$$

The three parts have separate roles:

- \(\tanh\) bounds each pre-centered response;
- \(\Delta\phi_{\max}\) sets the displacement scale;
- \(P_N\), implemented as phase-mean removal for this path, makes the proposed
  increment sum to zero across channels.

The provisional next state is

$$
\psi_{t+1}=\phi_t+\Delta\phi_t.
$$

Zero channel sum preserves the total fraction algebraically, but a finite
increment can still push an individual component below zero or above one.

## 2. Apply the strict hard state map

For a raw state \(\psi\), first clip each channel:

$$
\bar\psi_k=\min(1,\max(0,\psi_k)),
\qquad
s=\sum_j\bar\psi_j.
$$

For a normal nonzero denominator,

$$
\Pi_k(\psi)=\frac{\bar\psi_k}{s},
\qquad
\phi_{t+1}=\Pi(\psi_{t+1}).
$$

The implementation guards the near-zero case: if clipping leaves no positive
mass, it substitutes equal channel values before normalization. This is a
deterministic clip-and-renormalize **state map**, not a Euclidean simplex
projection.

**Illustrative example - not a benchmark output.** For

$$
\psi=(0.80,0.25,-0.05),
$$

clipping gives \((0.80,0.25,0)\), so

$$
\Pi(\psi)=
\left(\frac{0.80}{1.05},\frac{0.25}{1.05},0\right)
\approx(0.762,0.238,0).
$$

## 3. Distinguish the thresholded variant

The configuration identifier `soft_threshold_eps1e3` selects an explicit
thresholded state map. Despite the historical identifier, the operation is not
an unspecified "soft map":

1. clip to \([0,1]\);
2. set components below \(10^{-3}\) to zero;
3. use the same near-zero denominator guard;
4. renormalize the remaining components.

**Illustrative example - not a benchmark output.** Starting from
\((0.9995,0.0004,0.0001)\), the threshold removes the two sub-threshold channels
and renormalizes the state to \((1,0,0)\). This pruning changes small phase
fractions deliberately, so captions and benchmark manifests identify which map
applies.

## 4. Keep the auxiliary penalty separate

Training can also include a pre-map phase-sum penalty of the form

$$
L_{\mathrm{sum}}=
\left\langle\left(\sum_k\psi_k-1\right)^2\right\rangle.
$$

This is a soft optimization signal on the provisional state. It neither bounds
the update nor guarantees an admissible state. In the accepted trainer it is
optional and configuration-controlled.

## Rate constraint versus state constraint

The distinction is central:

```text
phase-mean removal: constrains the UPDATE to have zero channel sum
hard/thresholded map: constrains the resulting STATE
auxiliary penalty: encourages a pre-map property during training
```

These operations are complementary. A hard state map acts on every rollout
step, including inference, whereas a loss penalty exists only while optimizing.

## What to remember

- The neural response is bounded before it becomes a phase increment.
- Zero-sum rate projection and state admissibility solve different problems.
- Strict and thresholded state maps are explicit, configuration-specific maps.
- A training penalty cannot replace a hard inference-time constraint.

## Next

Continue to [How PINN-Phase Learns from the Physics](05_physics_informed_training.md).
