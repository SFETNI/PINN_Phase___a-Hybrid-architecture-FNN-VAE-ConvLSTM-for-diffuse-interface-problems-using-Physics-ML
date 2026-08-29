# Phase-Field Foundations for PINN-Phase

## What you will learn

- how a diffuse field represents an interface without tracking a sharp surface;
- how scalar and explicit multiphase-field states differ;
- what makes a local multiphase state admissible;
- why curvature, topology change, and long rollouts form a demanding test.

## Prerequisites

Familiarity with fields on a numerical grid is helpful. No machine-learning
background is required.

## From sharp boundaries to diffuse interfaces

A phase-field model replaces a geometrically sharp boundary by a narrow layer in
which one or more continuous fields change smoothly. The interface is therefore
part of the state on the grid: it can move, bend, meet other interfaces, and
disappear without an explicit mesh of the boundary.

The grids used here are periodic. Crossing the right face re-enters through the
left face, and the same rule holds in every spatial direction. Periodicity avoids
introducing a special wall into the designed coarsening problems.

Surface energy drives a curved interface toward configurations of lower area.
Small convex grains tend to shrink; interfaces rearrange at junctions; domains
can disappear; and neighboring regions with the same identity can coalesce.
These are changes of topology, not merely smooth translations of a fixed shape.

## Two state descriptions

The scalar benchmark family evolves one scalar order parameter. It is useful for
isolating shrinking and coalescing interfaces, but it is **not** the \(N=1\)
reduction of the explicit phase-fraction formulation. A one-channel explicit
state constrained to sum to one would be identically one and could not carry an
interface.

The explicit multiphase-field state has one channel per phase or grain:

$$
\phi=(\phi_1,\ldots,\phi_N).
$$

At every grid point an admissible state satisfies

$$
0\leq \phi_k\leq 1,
\qquad
\sum_{k=1}^{N}\phi_k=1.
$$

In a grain interior one channel is close to one and the others close to zero. At
an interface, two or more channels share the local fraction.

**Illustrative example - not a benchmark output.** The three-phase state

$$
(0.70,0.20,0.10)
$$

is admissible: every component lies in \([0,1]\), and the components sum to one.
The vector \((0.70,0.40,-0.10)\) is not admissible even though it also sums to
one, because one component is negative.

## A scalar physical check

For an isolated circular interface in the scalar lineage, curvature-driven
motion gives the radius-squared relation

$$
R^2(t)=R_0^2-2\mu\sigma t,
$$

where \(\mu\) is mobility and \(\sigma\) is interfacial energy. This compact
observable is useful because a field-level evolution can be checked against a
direct physical trend. The scalar benchmark in this repository checks both this
law and sampled energy monotonicity.

## Why long neural rollouts are difficult

PINN-Phase applies a learned update repeatedly. A small local error can change
an interface position; that position changes the next input; and the difference
can accumulate over thousands of steps. Near an extinction or a junction
rearrangement, a small timing error can also change a discrete event. Long-range
evaluation therefore needs both field metrics and topology-aware metrics.

The method does not remove the physical constraints. Instead, it combines a
learned time integrator with structural operations that preserve the meaning of
the multiphase state. The next tutorial develops the physical operator that
supplies its training target.

## What to remember

- A diffuse interface is represented directly by continuous grid fields.
- Explicit MPF channels are local fractions constrained to \([0,1]\) and unit sum.
- The scalar family is a separate benchmark lineage, not a one-phase MPF state.
- Curvature-driven motion naturally includes extinction and topology change.

## Next

Continue to [The Multiphase-Field Operator](02_explicit_mpf_operator.md).
