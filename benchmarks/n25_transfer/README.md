# Held-out 25-grain initial-condition transfer

This package is a compact, offline reproduction of the frozen confirmation
score. It contains the exact arrays needed by the reviewed metric and gate
definitions, copied without numerical transformation from ten reference
trajectories and two autonomous model arms.

## Design

The primary permutation-equivariant checkpoint was trained from **one**
development initial condition. The secondary checkpoint was trained from six.
Both were frozen before the ten-member confirmation cohort was generated and
scored, and the cohort was scored once. The primary arm determines the
headline result; the secondary arm can neither rescue nor veto it.

The cohort contains eight held-out members drawn from the same generator as
training, plus two structurally harder members — one with a denser and one
with a sparser initial grain structure. The strict evaluation horizon is
step 12,000.

## Result

| Arm | Training support | Held-out | Harder cases | Disposition |
|---|---:|---:|---:|---|
| Primary | 1 initial condition | **7/8** | **2/2** | Transfer established within the frozen 25-grain family |
| Secondary | 6 initial conditions | **7/8** | **2/2** | Same count, reported as secondary evidence |

The primary 7-of-8 proportion has a Wilson 95% interval of `[0.5291, 0.9776]`.
This is a pass at the threshold registered in advance, not evidence of
statistical headroom.

The sole strict failure is held-out case `id_07`: its terminal survivor F1 is
0.9677 and 96.29% of its pixels match the reference, but the strict
no-false-death gate fails — phase 19 is absent from the model and still
present in the step-12,000 reference. In the separately registered
descriptive tail, the reference removes phase 19 at step 12,933. That
post-horizon observation does not alter or rescue the strict 7-of-8 result,
because information after the pre-registered horizon cannot change a strict
verdict.

## Frozen criteria

Each member must pass finite-state, phase-sum, field-bound,
confidence-margin, no-reappearance, challenge-difficulty,
gain-over-persistence, interface localization, terminal disagreement,
terminal F1, and strict no-false-death checks. Aggregation over the held-out
members additionally requires at least 7 of 8 passing cases, adequate median
and worst-member gain over persistence, and a worst-member F1 floor. The two
harder members are aggregated separately.

The implementation is in
[`src/pinn_phase/evaluation/n25_transfer.py`](../../src/pinn_phase/evaluation/n25_transfer.py).

## Reproduce

From an installed wheel and the repository root:

```bash
python scripts/reproduce_n25_transfer.py
```

The command:

1. verifies all 40 public archives by SHA-256;
2. verifies each retained array's dtype, shape, and value digest;
3. recomputes all per-case metrics for both arms;
4. applies the frozen gate and aggregation logic;
5. compares the complete result with the reviewed expected score;
6. writes a new report to the ignored `outputs/` directory.

The public fixture is compact because soft cadence fields and embedded
execution metadata are not needed for these gates; the argmax label maps at
the saved cadence and the per-step area histories are retained in full. No
retained numerical array is transformed.

## Public adaptation

`manifest.json` records, for every public archive:

- the SHA-256 of the frozen source artifact;
- the SHA-256 of the compact public archive;
- the dtype, shape, and value digest of every retained array.

Embedded execution metadata and unused soft-field snapshots are omitted. The
retained labels, areas, health series, and margin series are sufficient to
recompute all strict gates and the descriptive false-death tail context.
`manifest.json` binds the frozen source artifact, the public archive, and
each retained array.

## Scope

The result supports initial-condition transfer within one fixed family of
25-grain coarsening problems — the same generator, resolution, and physics.
It does not establish transfer across materials, operators, phase counts,
spatial resolutions, or dimensions. An earlier 6-of-8 development campaign
is historical evidence and is never pooled with this confirmation cohort.
