"""Six-field simultaneous initial-condition batch loader.

Every entry is verified with explicit exceptions: file SHA-256, initial-only
key schema, shape, dtype, and admitted-ordinal order. The batch order is the
admitted-ordinal order and is not configurable.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from pinn_phase.io.artifacts import load_npz_arrays

ALLOWED_T0_KEYS = frozenset(
    {"fields", "labels", "seeds", "targets", "weights", "phase_permutation",
     "provenance_json"}
)


class MultiICBatchError(RuntimeError):
    """Fail-closed loader error; never continue past a failed verification."""


def _require(condition: bool, code: str) -> None:
    if not condition:
        raise MultiICBatchError(code)


def load_multi_ic_t0_batch(
    cfg: Mapping[str, Any],
    *,
    num_phases: int,
    spatial_shape: tuple[int, ...],
    repo_root: Path,
) -> np.ndarray:
    """Load and verify the fixed multi-IC t0 batch; return float32 [B, N, H, W].

    ``cfg`` is the ``training.multi_ic_batch`` config block::

        multi_ic_batch:
          entries:
            - {t0_npz: <repo-relative path>, sha256: <64-hex>,
               candidate_index: <int>, admitted_ordinal: <1-based int>}
            - ...

    Entries must be listed in admitted-ordinal order 1..B with strictly
    increasing candidate indices (the foundry admission order). Any hash,
    schema, shape, dtype, or ordering mismatch raises ``MultiICBatchError``.
    """
    entries: Sequence[Mapping[str, Any]] = cfg.get("entries") or ()
    # The prospective multi-initial-condition contract is exactly six fields.
    _require(len(entries) == 6, f"MULTI_IC_BATCH_REQUIRES_EXACTLY_SIX_ENTRIES:{len(entries)}")
    fields_list: list[np.ndarray] = []
    prev_candidate = -1
    for position, entry in enumerate(entries):
        unknown = set(entry.keys()) - {"t0_npz", "sha256", "candidate_index",
                                       "admitted_ordinal"}
        _require(not unknown, f"MULTI_IC_BATCH_UNKNOWN_ENTRY_FIELDS:{sorted(unknown)}")
        ordinal = int(entry["admitted_ordinal"])
        _require(ordinal == position + 1,
                 f"MULTI_IC_BATCH_ORDINAL_ORDER_VIOLATION:pos{position}:ord{ordinal}")
        candidate = int(entry["candidate_index"])
        _require(candidate > prev_candidate,
                 f"MULTI_IC_BATCH_CANDIDATE_ORDER_VIOLATION:{candidate}")
        prev_candidate = candidate

        path = Path(str(entry["t0_npz"]))
        if not path.is_absolute():
            path = repo_root / path
        _require(path.is_file(), f"MULTI_IC_BATCH_T0_MISSING:{path}")
        want = str(entry["sha256"]).lower()
        _require(len(want) == 64, f"MULTI_IC_BATCH_SHA_NOT_FULL_LENGTH:{want}")
        try:
            arrays = load_npz_arrays(
                path,
                expected_sha256=want,
                expected_keys=ALLOWED_T0_KEYS,
            )
        except (OSError, TypeError, ValueError) as exc:
            raise MultiICBatchError(f"MULTI_IC_BATCH_ARTIFACT_REFUSED:{path.name}:{exc}") from exc
        keys = set(arrays)
        _require(keys == set(ALLOWED_T0_KEYS),
                 f"MULTI_IC_BATCH_T0_KEY_DRIFT:{sorted(keys ^ set(ALLOWED_T0_KEYS))}")
        fields = np.asarray(arrays["fields"], dtype=np.float32)
        _require(fields.shape == (num_phases,) + tuple(spatial_shape),
                 f"MULTI_IC_BATCH_FIELD_SHAPE:{fields.shape}")
        _require(bool(np.isfinite(fields).all()), "MULTI_IC_BATCH_NONFINITE_FIELDS")
        fields_list.append(fields)

    batch = np.stack(fields_list, axis=0)
    _require(batch.shape[0] == len(entries), "MULTI_IC_BATCH_STACK_SIZE_MISMATCH")
    return batch
