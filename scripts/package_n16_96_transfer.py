#!/usr/bin/env python3
"""Build the public N16/96 prospective-cohort records from sealed score JSON.

This packs already accepted measurements; it never evaluates a model or reference.
"""
from __future__ import annotations

import argparse, hashlib, json
from pathlib import Path

def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def benchmark_readme(contract: dict[str, str]) -> str:
    """Render the public benchmark README from the same six-condition contract."""
    conditions = (
        ('Numerical integrity', contract['numerical_integrity']),
        ('Terminal fidelity', contract['terminal_fidelity']),
        ('Persistence', contract['persistence']),
        ('Terminal topology', contract['terminal_topology']),
        ('Extinction events', contract['events']),
        ('Tail consistency', contract['tail']),
    )
    rendered_conditions = '\n'.join(
        f'{number}. **{name}.** {description}'
        for number, (name, description) in enumerate(conditions, 1)
    )
    return f'''# N16/96 prospective initial-condition transfer

This package records six prospectively fixed unseen initial microstructures evaluated with the same fixed development checkpoint. The terminal scored state is step 1600; saved states are monitored through step 3200.

The public reproduction level is **PROVENANCE_ONLY** because frozen t0, model and reference arrays are documented external assets, not distributed here. `expected_score.json` is a compact, sanitized record derived from sealed score records; it is not a replacement for the arrays.

## Complete predefined qualification

A case meets the complete predefined qualification only when all six conditions hold:

{rendered_conditions}

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
'''


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--science-root', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    a = ap.parse_args(); out = a.out; out.mkdir(parents=True, exist_ok=True)
    score = json.loads((a.science_root / 'score/COHORT_SCORE.json').read_text())
    manifest = json.loads((a.science_root / 'cohort/COHORT_MANIFEST.json').read_text())
    cases = []
    for ordinal, cid in enumerate(('C1','C2','C3','C4','C5','C6'), 1):
        x = json.loads((a.science_root / 'score/per_case' / f'{cid}.json').read_text())
        g1, g2, g3, g4, g5, g6 = (x[k] for k in ('G1_integrity','G2_field_fidelity','G3_persistence','G4_topology','G5_events','G6_tail'))
        events = [{'offset_steps': e['residual'], 'within_plus_minus_200': e['within_tolerance']} for e in g5['step3_timing']]
        unmet = []
        if not g3['pass']: unmet.append('strict persistence versus static-t0 at every saved post-initial state')
        if not g5['pass']: unmet.append('strict reference extinction-wave order')
        cases.append({'public_name': f'Unseen microstructure {ordinal}', 'terminal_step': x['terminal_step'], 'monitoring_horizon_step': 3200,
          'finite_values': g1['nan'] == 0 and g1['inf'] == 0, 'nan_count': g1['nan'], 'inf_count': g1['inf'],
          'saved_cadence_exact': g1['cadence_exact'], 'monitoring_horizon_exact': g1['horizon_exact'],
          'maximum_phase_sum_error': float(g1['max_abs_sum_error']),
          'terminal_agreement_percent': round(float(g2['terminal_agreement'])*100, 6), 'terminal_disagreement_percent': round((1-float(g2['terminal_agreement']))*100, 6),
          'static_t0_agreement_percent': round(float(g2['static_t0_terminal'])*100, 6), 'terminal_gain_percentage_points': round((float(g2['terminal_agreement'])-float(g2['static_t0_terminal']))*100, 6),
          'minimum_persistence_gain_percentage_points': round(float(g3['min_margin_pp']), 6), 'minimum_persistence_gain_step': g3['min_margin_step'],
          'persistence_frames_compared': g3['frames_compared'], 'persistence_frames_strictly_greater': g3['frames_strictly_greater'],
          'persistence_all_post_initial_strictly_above_static_t0': g3['frames_compared'] == g3['frames_strictly_greater'],
          'reference_terminal_active_count': g4['reference_active_count'], 'model_terminal_active_count': g4['model_active_count'], 'terminal_active_set_equal': g4['pass'],
          'extinction_identities_equal': g5['step2_set_equality'], 'event_offsets': events, 'wave_order_preserved': g5['step5_strict_group_order'],
          'tail_active_set_equal': g6['active_set_matches_at_every_frame'], 'complete_predefined_qualification': x['case_verdict']=='PASS', 'unmet_criteria': unmet})
    qualification_contract = {'numerical_integrity': 'finite, exact cadence/horizon, max phase-sum error < 1e-3 (strict)', 'terminal_fidelity': 'agreement >= max(94.3%, static-t0 + 1.0 percentage point)', 'persistence': 'strictly greater than static-t0 at every saved step 200..3200', 'terminal_topology': 'exact active-label identity at 1600', 'events': 'same extinct labels/count, each within +/-200 inclusive, strict reference wave order, no reappearance', 'tail': 'exact active-label identity at each saved step 1800..3200, no reappearance'}
    result = {'schema': 'pinn-phase-n16-96-transfer-v1', 'reproduction_level': 'PROVENANCE_ONLY', 'public_case_names_only': True,
      'checkpoint_sha256': manifest['expected_checkpoint_sha256'], 'same_checkpoint_as_development_n16_cube': True,
      'terminal_scored_step': 1600, 'monitored_saved_horizon_step': 3200, 'saved_cadence_steps': 200,
      'qualification_contract': qualification_contract,
      'headline': {'terminal_agreement_percent_range': [95.062595,95.831412], 'static_t0_agreement_percent_range': [93.077709,93.777014], 'terminal_gain_percentage_points_range': [1.70808,2.46582], 'terminal_topology_and_extinction_identities': '6/6', 'complete_predefined_qualification': '5/6', 'event_offsets': {'total':18,'exact':11,'early_by_200':2,'late_by_200':5}}, 'cases': cases,
      'sealed_sources': {'cohort_score_sha256': sha(a.science_root/'score/COHORT_SCORE.json'), 'cohort_manifest_sha256': sha(a.science_root/'cohort/COHORT_MANIFEST.json')}}
    (out/'expected_score.json').write_text(json.dumps(result, indent=2, sort_keys=True)+'\n')
    assets=[]
    for ordinal, c in enumerate(manifest['cases'], 1):
        for role, p in [('initial_t0', a.science_root/f"cohort/cases/{c['case_id']}/t0.npz"), ('model_trajectory', a.science_root/f"rollouts/{c['case_id']}/model_frames.npz"), ('reference_trajectory', a.science_root/f"references/{c['case_id']}/ref_frames.npz")]:
            assets.append({'public_case':f'Unseen microstructure {ordinal}','role':role,'filename':p.name,'bytes':p.stat().st_size,'sha256':sha(p),'release_class':'DOCUMENTED_ONLY','expected_relative_location':f'external/n16_96_transfer/{ordinal}/{p.name}'})
    (out/'manifest.json').write_text(json.dumps({'schema':'pinn-phase-n16-96-external-assets-v1','assets':assets,'consumer_command':'python scripts/verify_n16_96_transfer.py --records benchmarks/n16_96_transfer/expected_score.json','asset_verification_command':'python scripts/verify_n16_96_transfer.py --records benchmarks/n16_96_transfer/expected_score.json --asset-root /path/to/asset-root'},indent=2,sort_keys=True)+'\n')
    (out/'README.md').write_text(benchmark_readme(qualification_contract), encoding='utf-8')
if __name__ == '__main__': main()
