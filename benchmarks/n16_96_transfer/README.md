# N16/96 prospective initial-condition transfer

This package records six prospectively fixed unseen initial microstructures evaluated with the same fixed development checkpoint. The terminal scored state is step 1600; saved states are monitored through step 3200.

The public reproduction level is **PROVENANCE_ONLY** because frozen t0, model and reference arrays are documented external assets, not distributed here. `expected_score.json` is a compact, sanitized record derived from sealed score records; it is not a replacement for the arrays.

## Complete predefined qualification

A case meets the complete predefined qualification only when all six conditions hold:

1. **Numerical integrity.** finite, exact cadence/horizon, max phase-sum error < 1e-3 (strict)
2. **Terminal fidelity.** agreement >= max(94.3%, static-t0 + 1.0 percentage point)
3. **Persistence.** strictly greater than static-t0 at every saved step 200..3200
4. **Terminal topology.** exact active-label identity at 1600
5. **Extinction events.** same extinct labels/count, each within +/-200 inclusive, strict reference wave order, no reappearance
6. **Tail consistency.** exact active-label identity at each saved step 1800..3200, no reappearance

All six cases recover the exact terminal 13-grain active set and extinction identities. Five of six meet the complete predefined qualification. Unseen microstructure 5 remains visible: it loses the strict reference wave order and is below static-t0 by 0.081 percentage points at step 3200, while its topology, extinction identities, individual +/-200-step timing and tail active set are correct.

## Verify documented external assets

The compact public score record is checked with:

```bash
python scripts/verify_n16_96_transfer.py --records benchmarks/n16_96_transfer/expected_score.json
```

When the separately held arrays are available beneath a directory that contains `external/n16_96_transfer/`, verify all 18 documented assets before using them:

```bash
python scripts/verify_n16_96_transfer.py --records benchmarks/n16_96_transfer/expected_score.json --asset-root /path/to/asset-root
```
