# The Multiphase-Field Operator

## What you will learn

- how the double-well and Laplacian terms form the explicit MPF rate;
- why the physical rate is projected to have zero channel sum;
- how periodic finite differences enter the implementation;
- how to calculate a three-channel rate projection by hand.

## Prerequisites

Read [Phase-Field Foundations](01_phase_field_foundations.md). Basic derivatives
and finite differences are sufficient.

## Local potential and interfacial smoothing

PINN-Phase uses the equal-parameter explicit MPF operator implemented in
[`explicit_mpf.py`](../../src/pinn_phase/physics/explicit_mpf.py). Its double-well
potential is

$$
W(\phi)=\phi^2(1-\phi)^2,
$$

with derivative

$$
W'(\phi)=2\phi(1-\phi)(1-2\phi).
$$

The potential favors values near zero or one. The Laplacian term penalizes
abrupt spatial variation, so the two terms together form a diffuse interface of
finite width.

## A tangent physical rate

For a channel vector \(v\), define the zero-sum projection

$$
P_N(v)_k=v_k-\frac{1}{N}\sum_{j=1}^{N}v_j.
$$

Summing over channels gives

$$
\sum_k P_N(v)_k
=\sum_k v_k-N\left(\frac{1}{N}\sum_jv_j\right)=0.
$$

The equal-parameter projected MPF rate is

$$
R(\phi)=\mu\sigma P_N\left[
\nabla^2\phi-\frac{1}{\eta^2}W'(\phi)
\right].
$$

Here \(\mu\) is mobility, \(\sigma\) is interfacial energy, and \(\eta\)
sets the diffuse-interface width. The Laplacian \(\nabla^2\phi\) couples each
point to its spatial neighbors. Because \(P_N\) removes the common channel
mode, \(R(\phi)\) is tangent to the unit-sum manifold: an infinitesimal physical
update does not change the sum of the phase fractions.

## Periodic finite differences

On a regular grid with spacing \(\Delta x\), the conceptual central-difference
Laplacian at point \(i\) is

$$
(\nabla_h^2\phi)_i=
\frac{1}{\Delta x^2}\sum_{m=1}^{d}
\left(\phi_{i+e_m}-2\phi_i+\phi_{i-e_m}\right).
$$

The neighbor indices wrap at every domain face. The accepted implementation has
separate 2D and 3D unit-grid periodic Laplacians and validates tensor rank and
phase-stack shape before evaluating the rate.

## Worked zero-sum projection

**Illustrative example - not a benchmark output.** Suppose a local raw rate is

$$
v=(0.06,0.02,-0.01).
$$

Its mean is

$$
\bar v=\frac{0.06+0.02-0.01}{3}=0.023333\ldots
$$

and the projected rate is

$$
P_3(v)\approx(0.036667,-0.003333,-0.033333).
$$

The projected components sum to zero. Notice that this operation constrains a
**rate**. It does not clip a finite updated state to \([0,1]\); that is a separate
operation developed in Tutorial 04.

## Physical target, not an inference subroutine

During the reported explicit-MPF training path, the operator is evaluated on
the model's current evolving field to form a physical target. Ordinary
autonomous inference applies only the learned model and state map; it does not
evaluate \(R(\phi)\) at every prediction step.

## What to remember

- The double well and periodic Laplacian define local interfacial dynamics.
- \(P_N\) removes the channel mean, making the physical rate zero-sum.
- Rate projection is not the same as mapping a finite state back to admissibility.
- The MPF operator supplies the explicit-MPF training target.

## Next

Continue to [Inside PINN-Phase](03_inside_pinn_phase.md).
