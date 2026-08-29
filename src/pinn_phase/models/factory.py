"""Public architecture-family identification and checkpoint identity helpers."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any

from pinn_phase.io.artifacts import sha256_file


class ArchitectureFamily(StrEnum):
    N3_EQUIVARIANT_MPF = "n3_permutation_equivariant_mpf"
    LEGACY_PA_HYBRID_MPF = "legacy_pa_hybrid_mpf"
    SCALAR = "scalar"


def architecture_family_from_config(settings: dict[str, Any]) -> ArchitectureFamily:
    """Resolve a public architecture family from a parsed configuration."""

    model = settings.get("model") or {}
    architecture = str(model.get("architecture", ""))
    if model.get("arch_variant") == "perm_equivariant_v1":
        if architecture != "explicit_mpf_ann_convlstm_hybrid":
            raise ValueError(
                "perm_equivariant_v1 requires architecture "
                "'explicit_mpf_ann_convlstm_hybrid'"
            )
        return ArchitectureFamily.N3_EQUIVARIANT_MPF
    if architecture == "explicit_mpf_ann_convlstm_hybrid":
        return ArchitectureFamily.LEGACY_PA_HYBRID_MPF
    if architecture.startswith("scalar_") or architecture in {
        "pinn_phase_hybrid",
        "local_convgru_rollout",
    }:
        return ArchitectureFamily.SCALAR
    raise ValueError(f"unrecognized PINN-Phase architecture: {architecture!r}")


@dataclass(frozen=True)
class CheckpointIdentity:
    """Minimal public identity required before loading a checkpoint."""

    architecture_family: ArchitectureFamily
    class_name: str
    checkpoint_sha256: str
    config_sha256: str

    @classmethod
    def from_paths(
        cls,
        architecture_family: ArchitectureFamily,
        class_name: str,
        checkpoint_path: str | Path,
        config_path: str | Path,
    ) -> "CheckpointIdentity":
        return cls(
            architecture_family=architecture_family,
            class_name=class_name,
            checkpoint_sha256=sha256_file(checkpoint_path),
            config_sha256=sha256_file(config_path),
        )
