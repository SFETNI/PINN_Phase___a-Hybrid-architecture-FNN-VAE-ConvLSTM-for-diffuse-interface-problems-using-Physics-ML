# Inspecting the N16 96^3 Prospective Evidence

This tutorial inspects the public record for six prospectively fixed unseen N=16,
96^3 initial microstructures. It does not rerun the six-case research campaign.
The scope is within-family prospective initial-condition transfer at fixed N = 16
and 96^3. It does not establish resolution transfer, phase-count transfer,
material transfer, population success rate, or statistical grain-growth
validation.

## Read the public record

Start with these public files:

- [`benchmark README`](../../benchmarks/n16_96_transfer/README.md) defines the
  complete predefined qualification.
- [`expected_score.json`](../../benchmarks/n16_96_transfer/expected_score.json)
  records the six-case result.
- [`manifest.json`](../../benchmarks/n16_96_transfer/manifest.json) lists the
  documented external assets.
- [`verify_n16_96_transfer.py`](../../scripts/verify_n16_96_transfer.py) checks
  the record and, when supplied, the external assets.
- [`cohort animation`](../../media/n16_96_transfer/n16_96_unseen_cohort_reference_vs_pinn_phase.gif)
  shows the saved-state progression.

The same fixed development model was used for all six microstructures. The scored
terminal state is step 1600; saved-state monitoring continues through step 3200.

| Contract item | Public record |
|---|---|
| Terminal agreement | 95.06%-95.83% |
| Static-t0 agreement | 93.08%-93.78% |
| Terminal gain | +1.71 to +2.47 percentage points |
| Exact terminal 13-grain active set | 6/6 |
| All three extinction identities | 6/6 |
| Complete predefined qualification | 5/6 |

The 6/6 topology and extinction identities result is not the same as 6/6 complete
qualification. The latter includes numerical integrity, terminal fidelity,
persistence, topology, extinction events, and tail consistency.

## Verify the compact record

**Classification:** `EXECUTED_AND_PASSING`

```bash
python scripts/verify_n16_96_transfer.py --records benchmarks/n16_96_transfer/expected_score.json
```

This validates the compact public record and its criteria without requiring the
external arrays. It reports `N16 prospective public record: PASS` when the record
is consistent.

## Read the one incomplete qualification

Unseen microstructure 5 has the following status:

- Terminal topology: PASS.
- Extinction identities: PASS.
- Individual +/-200-step timing: PASS.
- Tail active-set identity: PASS.
- Wave order: FAIL.
- Strict persistence: FAIL.
- Minimum persistence gain: -0.080589 percentage points at step 3200
  (approximately -0.081 pp).

The terminal topology and extinction identities remain correct for this case; the
two failed criteria are why the complete predefined qualification remains 5/6.

## Optional external asset check

The normal record inspection needs no external asset. When the documented asset
hierarchy is already available, the optional verifier checks all 18 documented
assets, totaling 10,726,256,132 bytes, by location, size, and SHA-256.

**Classification:** `OPTIONAL_EXTERNAL_ASSET`

```bash
python scripts/verify_n16_96_transfer.py --records benchmarks/n16_96_transfer/expected_score.json --asset-root <asset-root>
```

`<asset-root>` must contain `external/n16_96_transfer/`. No download location is
required for the record, criteria, manifest, or media interpretation.
