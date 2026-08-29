# Media and benchmark gallery

This release keeps the public visual record from the earlier candidate and adds
the figures for the held-out 25-grain transfer study. Every asset is covered by
a SHA-256 manifest. Media visualize completed runs or the method architecture;
they do not replace the quantitative records in
[`COMPLETED_EVIDENCE_LEDGER.md`](COMPLETED_EVIDENCE_LEDGER.md).

## Current results

| Asset | Role |
|---|---|
| `media/current_results/n25_cascade_reference_vs_pinn_phase.gif` | 25-grain cascade on the training initial condition: reference, prediction, differing pixels; diagnostic panel divided at the 4,096-step represented training horizon |
| `media/current_results/n64_dense_reference_vs_pinn_phase.gif` | Dense 64-grain coarsening, horizon-extension sensitivity run |

The two `current_results` animations are rendered by
`scripts/render_current_results.py` from frozen arrays that are too large to
distribute in this release. Their `asset_manifest.json` records the SHA-256 of
every source artifact, the renderer digest, and every number drawn on the frames,
each recomputed at render time rather than transcribed. Grain colours are fixed
per phase index for all time, so a grain never changes colour when a neighbour
dies, and hues near the discrepancy red are excluded.

The 64-grain animation shows the **post-evaluation horizon-extension sensitivity
run**, not the registered primary result. Both are reported in the README, and
the ledger records the arrays behind each.

Its diagnostic axis marks step **8,192**, the represented training horizon, and
divides the axis into the represented horizon and the **3,808** steps of
autonomous extrapolation beyond it. Read that mark correctly:

- It is a number, not a saved frame. It falls between the 8,000 and 8,500
  milestones and is drawn at its true position on the continuous axis.
- **Nothing happens to the microstructure there.** There is no restart and no
  physical discontinuity, which is why the field panels, the curve styling and
  the frame pacing are identical on both sides of it.
- **Within the represented horizon does not mean the model saw reference
  states.** For the reported explicit-MPF path, training used no post-initial
  reference state as a target: the model was
  unrolled on its own states and supervised by physics residuals. The status
  strip says *autonomous rollout* on both sides of the boundary for that reason.
- **Both regions are labelled inside the panel**, so the animation carries its
  own explanation when it travels away from this page.
- **Only the rollout advances.** The reference grain count is drawn **complete
  and fixed** across the whole axis — it is the backdrop the prediction is read
  against, not part of the animation, and it spans the boundary as a **held-out
  comparator, used for evaluation only** that was never supplied to the model on
  either side. What moves is the model's own trajectory: the prediction and the
  differing-pixel curve are drawn as far as the current step and no further.
- Both grain counts are drawn at ordinary weight. The reference is **dashed** and
  laid over the prediction's **solid** stroke, so where the two agree the
  prediction shows through the dash gaps and the picture reads as two series
  rather than one. The dash pattern is the separation, so it survives
  desaturation. The moving marker belongs to the prediction. Neither series is
  displaced numerically.
- The tail tint is a warm neutral, not a warning. Disagreement grows steadily
  from well before the boundary, so the tint marks a relation to training, not a
  cause of error.

## Held-out 25-grain transfer

| Asset | Role |
|---|---|
| `media/n25_transfer/id_02_reference_vs_pinn_phase.gif` | A representative unseen case: reference, prediction, and discrepancy across the saved cadence |
| `media/n25_transfer/terminal_atlas.png` | Terminal state of all ten cases at step 12,000, including the one strict failure |

These are rendered by `scripts/render_n25_transfer.py` from the compact arrays
under `benchmarks/n25_transfer/`, and bound by
`media/n25_transfer/asset_manifest.json`, which pins the renderer digest and the
input manifest digest alongside each asset.

Read the animations with two facts in mind. The field panels are argmax label
maps at the saved cadence — 13 non-uniformly spaced steps between 0 and 12,000 —
not a continuous rollout, and the step number is printed on every frame. The
active-grain staircases underneath are computed at full per-step resolution. The
discrepancy panel is empty at step 0 because that frame is the initial condition
supplied to both the reference and the model.

## Retained method and benchmark media

| Asset | Role | Status in this release |
|---|---|---|
| `media/method_loop_hybrid_rollout.gif` | Autoregressive update loop of the first-generation hybrid model | Architecture context; not a diagram of the permutation-equivariant encoder |
| `media/framework_hybrid_rollout.png` | Static schematic of the same family | Retained architecture context |
| `media/n8_64_reference_vs_pinn_phase.gif` | 64³ eight-phase cube, one designed extinction | Current completed single-case evidence |
| `media/n16_96_reference_vs_pinn_phase.gif` | 96³ sixteen-phase cube, three designed extinctions | Current completed single-case evidence |
| `media/triple_junction_h1024_reference_vs_pinn_phase.gif` | Four-phase junction relaxation | Retained completed demonstration |
| `media/scalar_3d_spherical_reference_vs_pinn_phase.gif` | Spherical-grain shrinkage in 3D | Retained completed demonstration |
| `media/scalar_shrinkage_reference.gif` | Scalar reference evolution | Reference-only onboarding visual |
| `media/multigrain_denser_n44_reference_vs_pinn_phase.gif` | Scalar many-domain coarsening | Retained historical demonstration |

Each animation has a static poster with the same stem. The retained files are
licensed under the media terms in [`ASSET_LICENSE.md`](../ASSET_LICENSE.md), and
`media/legacy_asset_manifest.json` records byte identities, dimensions, frame
counts, and continuity roles for the assets carried over from the earlier
candidate.

## Interpretation rules

- The prospective transfer disposition comes only from the compact confirmation
  package and its scorer, never from any superseded 25-grain animation.
- The 64-grain animation shows the post-evaluation sensitivity run. The
  registered primary 64-grain result is a separate run and is not animated here;
  the two must never be conflated or averaged.
- The 25-grain cascade animation is development evidence: the model was trained
  on that initial condition. Only the transfer cohort was prospectively fixed.
- The 64³ and 96³ cubes are individual designed cases, not evidence of
  statistical 3D transfer.
- Saved-frame extinction agreement has the resolution of the saved cadence; it
  is not a continuous-time equality claim.
- The method animation is explanatory artwork and carries no benchmark metric.

## Superseded model variants (gallery only)

These animations come from earlier model variants of the earlier public
candidate. They are retained so the visual record stays complete. **They are not
the present model of record**, they do not appear on the front page, and their
values do not appear in "Results at a glance".

| Asset | What it shows | Why it is not current |
|---|---|---|
| `media/n25_v2_reference_vs_pinn_phase.gif` | Compact earlier 25-grain case | Earlier model variant; superseded by the 25-grain cascade result |
| `media/n25_v3_long_reference_vs_pinn_phase.gif` | Earlier long-horizon 25-grain case | Earlier model variant; superseded by the 25-grain cascade result |
| `media/n64_dense_no_graph_reference_vs_pinn_phase.gif` | Earlier dense 64-grain case | Earlier model variant on a shorter horizon; superseded by the 64-grain result |

Their previously published quantitative values are recorded in
[`COMPLETED_EVIDENCE_LEDGER.md`](COMPLETED_EVIDENCE_LEDGER.md) under the retained
earlier-public records, and are never mixed with the current results.
