# Scalar circular-grain shrinkage

This deterministic CPU benchmark integrates a periodic `128 x 128` scalar
Allen-Cahn reference from one diffuse circular grain. Before extinction, the
equivalent radius follows the expected linear law in `R^2(t)` with a 0.997%
relative slope error and `R^2 = 0.999996`.

```bash
python scripts/reproduce_scalar_reference.py
```

The command regenerates the trajectory, verifies exact expected metrics, and
writes the reference and score under `outputs/scalar_shrinkage_2d/`.
