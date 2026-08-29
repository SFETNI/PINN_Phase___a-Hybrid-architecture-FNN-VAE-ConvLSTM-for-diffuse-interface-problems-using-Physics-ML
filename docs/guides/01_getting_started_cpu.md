# Getting Started: A CPU Step Through PINN-Phase

This tutorial takes one small, CPU-only step through the public model surface.
It does not train a model, regenerate a benchmark, or require an external asset
package. The CPU environment specification is
[`environment-cpu.yml`](../../environment-cpu.yml).

## Run the smoke check

Run this command from the repository root. Replace `<outside-tree>` with a
writable directory outside the repository.

**Classification:** `EXECUTED_AND_PASSING`

```bash
python scripts/smoke_test.py --output <outside-tree>/smoke_report.json
```

The command writes a JSON report and prints `CPU smoke: PASS` when its checks
hold. It constructs a small periodic multiphase field, evaluates the explicit
multiphase-field right-hand side, projects the field back to the admissible
simplex, and checks finite values together with phase-permutation and translation
equivariance. These are lightweight implementation checks, not benchmark scores.

## Read one model update

The smoke check follows the same high-level update used by the rollout models:

1. A neural branch proposes a rate from the present phase field.
2. An explicit Euler update advances the field using that learned rate.
3. The admissibility map returns a bounded field whose phases sum to one.
4. The returned field can become the next autonomous input.

During training, the explicit multiphase-field right-hand side supplies the
physical rate target used in the residual. The autonomous inference update does
not call that right-hand side; it advances the learned rate.

The current model families and physics terms are described in
[`docs/METHOD_OVERVIEW.md`](../METHOD_OVERVIEW.md). Larger workflows begin with
the benchmark-specific scripts in the next tutorials; keep their output outside
the repository as well.
