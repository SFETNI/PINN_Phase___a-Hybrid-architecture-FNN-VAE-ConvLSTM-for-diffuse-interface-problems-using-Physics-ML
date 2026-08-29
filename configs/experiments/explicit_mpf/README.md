# Retained experiment configuration

`voronoi_3d_cube_64x64x64_n8_h256_tbptt16_c2_launch.yaml` is carried here
byte-for-byte from the earlier public candidate so that the configuration digest
recorded for the 64³ eight-phase cube in
[`../../docs/ARTIFACT_IDENTITY_LEDGER.json`](../../../docs/ARTIFACT_IDENTITY_LEDGER.json)
can be verified against distributed bytes rather than asserted. It is the only
training-side configuration in this archive whose digest is checkable from the
archive itself; every other training configuration is an identity only.

Its header names three things from the earlier candidate that this release does not
carry — a `scripts/rollout_n8_64.py` rollout entry point, a
`tests/test_n8_config_strict_load.py` compatibility test, and a
`checkpoints/n8_64_c2.pt` training checkpoint. Their function is served here by
[`../models/n8_64_cube_hybrid.yaml`](../../models/n8_64_cube_hybrid.yaml) together
with `scripts/verify_checkpoint_identities.py`, which performs the same strict load
against the derived public replay weights for the same model.

The digest the header quotes is the **accepted parent checkpoint's**, which this
archive identifies but does not distribute; the distributed artifact is
`checkpoints/n8_64_cube_hybrid.weights.npz` and has a different digest of its own.
The file is not edited, because editing it would change the digest that makes it
evidence — so its header is read as a historical record of the run, not as a
description of what ships here.
