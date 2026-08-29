# Completed evidence ledger

This ledger separates score-complete public packages from completed results
whose large fields are not distributed in v0.1. Section headings name a case by
its phase count and grid — N64 is a 64-grain field, N16 a sixteen-phase 96³
cube — while the model families are named in
[`METHOD_OVERVIEW.md`](METHOD_OVERVIEW.md). A digest identifies an
accepted artifact but is not a substitute for publishing it.

The method and benchmark animations from the earlier public candidate remain
available in `media/`. Their byte identities and continuity roles are recorded
in `media/legacy_asset_manifest.json`; historical N25 and N64 visuals are not
used as substitutes for the newer permutation-equivariant results.

## Current results shown on the front page

Each value on the front page traces to an accepted artifact by SHA-256. The
arrays themselves are frozen research artifacts held outside this release; the
digests identify them independently of any location. A score-record digest is the
digest of the record file itself — the frozen file in which that benchmark's
metrics were computed and fixed — and is distinct from the digests of the arrays
it was computed from.

### 25-grain cascade (development evidence)

- Architecture: permutation-equivariant multiphase-field model, 9,605 parameters,
  trained at a 4,096-step horizon.
- Accepted parent checkpoint: `6673d46000b00867e3b74d332342bea51050398ac873376b5fc1ba65a99556f3` — the identity of
  record. **Not distributed.**
- Derived public replay weights: `e160ddbaa9e7837cb5e7b4be134c2614c14fae8376257a2331cac8cb4b8d499e`,
  distributed as `checkpoints/n25_cascade_primary_permequiv.weights.npz`. Model-state tensors only;
  a packaging digest, not the identity of the accepted parent.
- Training-case model rollout:
  `02f11f936fac16f206897a4023bc613ccf098834e8cfed39fb0ab18007208b4d`.
- Training-case reference fields:
  `351a55961ee187e327ba1a23c0f566350959eed2ebe8d9473945cc47158fd511`.
- Second development case, model rollout:
  `0b652628814b1adc707ab94d112f3be74b7c64fd981b7ead53bf171dc1d3709f`.
- Second development case, reference fields:
  `ef2f13dbd615b21fe515e24d2e31d4f52cd24089d229ef4027348dc5c4b65490`.
- Verification record binding both cases:
  `6cdcea1dba38b916c2e4a0428cb43b4327e289b909c1de96815dc72cf28332d3`.
- Result, training case: 0.81% differing pixels at step 4,000 and 1.46% at step
  12,000, against a persistence floor of 26.59% and 44.37%; exact 16-grain
  terminal survivor set, survivor F1 1.0, no false deaths and no extra survivors.
- Result, second development case: 0.95% at step 4,000 and 1.13% at step 12,000,
  against 28.97% and 51.25%; exact 12-grain terminal survivor set; 13 reference
  extinctions through step 12,000, all reproduced.
- Disclosed failure: the pre-registered timing anchor is not met. Extinction
  residuals are predominantly premature, with medians of −53 steps (training
  case) and −85 steps (second case). Source record:
  `669896603f10d9505ce3cc285340b1f4848035bf8ea9e8eb1732efa6a89a5215`.
- Boundary: both cases are **development evidence** — the model was trained on
  the first and the second is drawn from the same development suite. Neither is
  prospective generalization evidence.

### Dense 64-grain, registered primary result

- Architecture: permutation-equivariant multiphase-field model (internal
  designation N3), 9,605 parameters, trained at a 4,096-step horizon.
- Config: `ccb87eb42322849174e8c9bed1d69d252369ce2a0ead4a5f33ebfaf19a1a7749`.
- Accepted parent epoch-99 checkpoint: `301d2431fec9fd75171888ce9635f5dcef1c200f9a7121f3cff3a29976785813` — the identity of
  record. **Not distributed.**
- Derived public replay weights: `44703ecfe7edc65316835976e218857d800d229199465b4f122e276fb30c262f`,
  distributed as `checkpoints/n64_dense_primary_permequiv_ep0099.weights.npz`. Model-state tensors only;
  a packaging digest, not the identity of the accepted parent.
- Autonomous rollout: `ad87347c884501e91755ce92b18bb54c7521e15ac05d23453996058813f1de82`.
- Reference: `96d4a5422344a5a458f12e4dae154e9843f1407436b2dc461f7379e25891ffbe`.
- Score record: `e55a902eb3b18b12698321f39e301aa2a155d21e9c9bd51aa4d388ddbc116d43`.
  This benchmark was scored once, against acceptance criteria fixed before the
  scoring run; the digest identifies the frozen record of that single evaluation.
- Result: 1.35%, 3.50%, and 6.09% differing pixels at steps 4,000, 8,000, and
  12,000; persistence 69.89% at step 12,000; 21 reference survivors, all retained,
  plus one additional grain; terminal survivor F1 0.9767; no false deaths and no
  reappearances.
- Boundary: one initial condition and one seed. This is the result that was
  scored once against criteria fixed beforehand.

### Dense 64-grain, horizon-extension sensitivity run

- Same model continued for 25 further epochs at an 8,192-step horizon
  (125 cumulative epochs).
- Accepted parent epoch-124 checkpoint: `d1fbef94e363b8fff49416f996854849c8d443c4f94a0fee4c34634752e597dd` — the identity of
  record. **Not distributed.**
- Derived public replay weights: `9b199b1d53eea17084c179ec6af989b5be2949432791f8bd8abf20340401a468`,
  distributed as `checkpoints/n64_dense_sensitivity_h8192_ep0124.weights.npz`. Model-state tensors only;
  a packaging digest, not the identity of the accepted parent.
- Config: `44d8fb5f74318686dd60fe90c6a77aac31c48597136adffc971afa8a135ad942`.
- Continuation rollout: `ea45d058f68b20daf479cec0c9da14927d6033fbcd16bbbaf4c2a591b6d72d35`.
- Reference: as above.
- Score record: `1e13d88cb9d615923c32ec47721c2f28b7db5653d22aa8ddf63fddd2c4c95e9c`.
  This is an exploratory evaluation, not a second scoring of the registered primary
  result; the distinction is stated in the boundary note below.
- Result: 0.85%, 2.07%, and 3.97% at the same three milestones; exact 21-grain
  terminal survivor set; 43 of 43 reference extinctions matched by identity.
- Boundary: this is a **post-evaluation sensitivity study of one design choice**.
  It is not independent confirmation, it does not replace or re-score the
  registered primary result above, and horizon and additional training effort
  co-vary so the improvement cannot be attributed to the horizon alone.

## Retained quantitative records

The values in this section are bound to the source documents and SHA-256
identities below. Those digests are the provenance root: they identify the exact
records independently of a repository revision, and the archive is self-contained
for the stated evidence level.

The four names below are historical record paths. None is distributed here, and
none exists in this archive; they are written as plain text rather than links.
Each digest, not its path, is the provenance identity — that is what makes the
record locatable without the candidate tree.

- non-distributed path in the earlier public candidate, `docs/BENCHMARK_CATALOG.md`:
  `dfe8a4ab5ef18c84107a407681aee2649046878302397563244d1c12322f6c35`
- non-distributed path in the earlier public candidate, `docs/BENCHMARKS.md`:
  `dbf5b81df25eb22df8796eacf87e3484afca892a847970b3e5c6a0cb296cca9f`
- non-distributed path in the earlier public candidate, `examples/n8_64/EXPECTED_METRICS.yaml`:
  `f0ad608a3f228c70f078c3f00b9fd00be741b6dc810bf223278ce3b51fdfdf1d`
- non-distributed path in the earlier public candidate,
  `examples/scalar_3d_spherical/EXPECTED_METRICS.yaml`:
  `c9b8d0d4c7c53bc7cd4f96100b35c70ca265b5e0a0f6b271053743ebd56c7c09`

The retained values used by the README are:

- four-phase junction: **3.27%** terminal disagreement versus **6.54%**
  persistence (precise records 3.272% and 6.543%); junction angles
  **127.8°**, **115.9°**, and **116.4°**, with approximately **5.5°** RMS
  deviation from 120°;
- historical long-horizon 25-grain case: **17.20%** terminal disagreement
  versus **39.70%** persistence, with 17 model-active phases versus 14 in the
  reference;
- historical dense 64-grain case: **18.3%** terminal disagreement versus
  **38.8%** persistence (precise records 18.26% and 38.81%), with 54
  model-active phases versus 49 in the reference;
- scalar 3D spherical case: radius-law slope ratio **0.991** (0.990977) and
  fit **R² 0.9999** (0.999937). Its reported production result is explicitly
  bound-enforcement dependent; the unbounded variant fails and the clamp-only
  result is retained only as a diagnostic;
- N8 `64^3`: **99.533%** terminal agreement versus a **94.01%** persistence
  baseline (precise records 0.9953346 and 0.9400787).

These records preserve previously public evidence. They do not make the old
large fields score-complete in this lightweight release. The 25-grain and
64-grain values in this section belong to **superseded model variants**: they are
gallery-only, they do not appear on the front page, and they are never mixed
with the current results above.

## Score-complete in v0.1

### Held-out 25-grain transfer

- Architecture: permutation-equivariant multiphase-field model (internal
  designation N3), 9,605 parameters.
- Accepted parent primary checkpoint: `6673d46000b00867e3b74d332342bea51050398ac873376b5fc1ba65a99556f3` — the identity of
  record. **Not distributed.**
- Derived public replay weights: `e160ddbaa9e7837cb5e7b4be134c2614c14fae8376257a2331cac8cb4b8d499e`,
  distributed as `checkpoints/n25_cascade_primary_permequiv.weights.npz`. Model-state tensors only;
  a packaging digest, not the identity of the accepted parent.
- Primary config: `324d4b50678c93a2999fef500e71ba02e7d4d39480c0519e26983b20d91a487a`.
- Accepted parent secondary checkpoint: `85e920603e69a9c035a585703ca6f12d27fc9050b8332939e38f11e2e6899c7b` — the identity of
  record. **Not distributed.**
- Derived public replay weights: `8fd13deb8d72a36d78796d8d26d3164dcdbe9e8ae116de5960e80efdbfbc7e1e`,
  distributed as `checkpoints/n25_cascade_secondary_six_ic_permequiv.weights.npz`. Model-state tensors only;
  a packaging digest, not the identity of the accepted parent.
- Secondary config: `03c5cfb54707ceefc1ee5d1da155ab2d3719757d230549423346ca18743438d0`.
- Frozen strict score: `26fbfdefed5278106a16cedf4be2a1db1ed010760cabf54336f6ee6a378e7202`.
- Public evidence: `benchmarks/n25_transfer/manifest.json` binds all forty
  compact archives to their frozen-source digests and retained array values.
- Result: primary and secondary each pass 7/8 ID and 2/2 stress.

### Scalar `128^2` shrinkage

- Public config: `configs/benchmarks/scalar_shrinkage_2d.yaml`.
- Public expected result: `benchmarks/scalar_shrinkage_2d/expected_metrics.json`.
- Result: 0.997% relative error in the pre-extinction radius-squared slope,
  with monotone sampled energy.

## Completed, large fields not distributed in v0.1

The 64-grain case is recorded once, above, under the current front-page results.

### N16 `96^3`

- Architecture: first-generation hybrid multiphase-field model.
- Accepted parent checkpoint: `9e15e6b74f4f92f37d5709a242e436f721ec5d1306a30e05e5f39794de4459dd` — the identity of
  record. **Not distributed.**
- Derived public replay weights: `690fd690e8af107870743dcd2f43b95e0c05c46ca4dd4b9d11f64e7994fb93c9`,
  distributed as `checkpoints/n16_96_cube_hybrid.weights.npz`. Model-state tensors only;
  a packaging digest, not the identity of the accepted parent.
- Initial condition, accepted parent container:
  `d6a5e7f28b85d14c1610eecabb017107f3caa48e73fe78c3c3270b75be389e3e` — the identity of
  record. **Not distributed.**
- Derived public initial field: `81eef72fd4d8ee2e9686b34d38fa6f9d02aa1fee99f0da900ea4bea9368229b5`,
  distributed as `benchmarks/initial_conditions/n16_96_cube.npz`. Member set exactly
  `{phi0}`; a packaging digest, not the identity of the accepted parent. Its
  container-independent value digest is
  `caba3a1c375714d7bbf371e7268d6680b8bf0d810fac839ad2f9d29058095c1e`, and that value is
  exact against the parent.
- Reference: `4fe2fbfffe1f54e4b2a36b5ff6464176fb0cc3d311f569b3ddd3b7f839eb763f`.
- Result: 99.668% agreement at evaluation step 1,600, exact 13-grain
  survivor set, and three extinctions matched at the 200-step saved-frame
  resolution.
- Boundary: one designed cube; no statistical 3D-transfer claim.

### N8 `64^3`

- Architecture: first-generation hybrid multiphase-field model.
- Accepted parent checkpoint: `2482f3bf315f61be82492aa972bacb2e8895942509c993f6c017d72fc46e5441` — the identity of
  record. **Not distributed.**
- Derived public replay weights: `327e420f4b25b9c9dae10514e4f672e1057d9d1728ed839e5f5b535b4898ef60`,
  distributed as `checkpoints/n8_64_cube_hybrid.weights.npz`. Model-state tensors only;
  a packaging digest, not the identity of the accepted parent.
- Config: `681ac195f53743c5a0d506ac40316544b62182834fa58eacbb15eadf4b87fd30`,
  distributed as
  `configs/experiments/explicit_mpf/voronoi_3d_cube_64x64x64_n8_h256_tbptt16_c2_launch.yaml`.
- Initial condition, accepted parent container:
  `ae0e698a23c4ee1017b328d31333fdbd39452677548f0fa70a8afabe870a186d` — the identity of
  record. **Not distributed.**
- Derived public initial field: `819b9196b3901827b69fc08e4f872bd3e72aa055719cd700f374ac1dae7cb6cb`,
  distributed as `benchmarks/initial_conditions/n8_64_cube.npz`. Member set exactly
  `{phi0}`; a packaging digest, not the identity of the accepted parent. Its
  container-independent value digest is
  `deaedaeef36549effc4b8ee6c9a875fb9ab812b7d3417533636431f922d97101`, and that value is
  exact against the parent.
- Reference: `d273d467915f2a892302e042da76b1bf102a2cc46f5375f42bfe5685135e0a83`.
- Result: 99.533% agreement at step 2,000 and exact 7-grain survivor set.
- Boundary: one designed cube with one extinction; no statistical 3D-transfer
  claim.

No download URL is advertised for the omitted fields until an immutable,
licensed public evidence package exists and its digest is verified here.


## N16/96 prospective transfer

Six unseen fixed-family initial conditions use the existing N16 checkpoint; no weight is duplicated and no training occurs. Public score records are PROVENANCE_ONLY; external arrays and the all-six renderer are hash-bound in `benchmarks/n16_96_transfer/manifest.json` and `media/n16_96_transfer/asset_manifest.json`.
