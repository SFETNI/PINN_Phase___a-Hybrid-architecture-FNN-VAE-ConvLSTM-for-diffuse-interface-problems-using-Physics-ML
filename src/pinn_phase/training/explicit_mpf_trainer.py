"""Explicit-MPF trainer using initial conditions and physics residuals only.

This trainer is deliberately separate from the scalar training path. It builds
:class:`ExplicitMPFHybridRollout`, integrates the *learned*
rate with the corrected physical timestep ``model_dt = dt_mu_sigma/(mu*sigma)``,
enforces the simplex via hard projection after each step, and trains only from:

* the initial condition ``Phi_ref(t=0)`` (consistency at ``t=0``), and
* the synchronized MPF residual ``|| rate - explicit_mpf_rhs(Phi) ||^2`` computed
  on the model's own rollout states.

PF/MPF reference frames after ``t=0`` are NEVER read here.  The reference loader
exposes only ``Phi_ref(t=0)`` by construction.  Audit/scoring against reference
``t>0`` belongs in separate offline evaluation, not in this training loss.

Safety: nothing in this module trains unless the caller passes ``launch=True``.
``mode="smoke"`` builds the model and runs a couple of steps with no checkpoint
and no training, for tests and dry verification.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import math
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import torch
from torch import Tensor

from pinn_phase.io.artifacts import load_torch_checkpoint
from pinn_phase.models.explicit_mpf import ExplicitMPFHybridRollout
from pinn_phase.models.perm_equivariant_mpf import PermEquivariantMPFRollout
from pinn_phase.physics.explicit_mpf import (
    ExplicitMPFReference,
    explicit_mpf_rhs,
    load_explicit_mpf_initial_reference,
    project_simplex,
    project_simplex_soft_threshold,
)


FORBIDDEN_REFERENCE_USAGE = (
    ("reference_usage_policy", "training_uses_reference_frames_after_t0", True),
    ("reference_usage_policy", "latent_targets_from_reference_after_t0", True),
    ("reference_usage_policy", "sampler_targets_from_reference_after_t0", True),
    ("training_policy", "no_training_labels_after_t0", False),
    ("training_policy", "no_reference_graph_after_t0", False),
)

# --- the reviewed reference-usage contract -------------------------------------
#
# Every key below appears in all four accepted explicit-MPF training configurations.
# The guard requires the complete set, with these exact types, because a policy that
# is silent about a channel has not declared that channel closed -- and an absent
# declaration must not read as a safe one.

#: Flags that must be the literal boolean False. Not 0, not "false", not None, not
#: absent. A configuration expresses this as a YAML boolean; anything else is a
#: malformed declaration and is refused rather than interpreted.
REFERENCE_POLICY_MUST_BE_FALSE = (
    "training_uses_reference_frames_after_t0",
    "latent_targets_from_reference_after_t0",
    "sampler_targets_from_reference_after_t0",
)

#: Remaining reviewed keys and their required types.
REFERENCE_POLICY_TYPES = {
    "training_initial_condition": str,
    "reference_after_t0_use": str,
    "audit_uses_reference_frames_after_t0": bool,
    "graph_features_from": str,
}

#: The complete reviewed key set. Extra keys are refused: an unreviewed key in a
#: policy block is an undeclared channel, and this guard cannot vouch for it.
REFERENCE_POLICY_KEYS = frozenset(REFERENCE_POLICY_MUST_BE_FALSE) | frozenset(
    REFERENCE_POLICY_TYPES
)

#: The only source a graph feature may be computed from.
REQUIRED_GRAPH_FEATURE_SOURCE = "model_phi_only"

#: The reviewed initial-condition modes.
SINGLE_IC_MODE = "Phi_ref_t0_only"
MULTI_IC_MODE = "six_frozen_foundry_t0_fields_only"
MULTI_IC_ENTRY_COUNT = 6

#: training_policy is REQUIRED, not optional. All four accepted explicit-MPF
#: training configurations carry it in full; the only disclosure record without one
#: is a retained experiment/launch configuration whose policy lives in a benchmark
#: adapter and which is not a training configuration. Treating the block as optional
#: would mean a configuration could omit its supervision declaration entirely and
#: still pass, which is the failure this guard exists to prevent.
TRAINING_POLICY_MUST_BE_TRUE = (
    "no_training_labels_after_t0",
    "no_reference_graph_after_t0",
    "no_reference_latent_targets_after_t0",
)
TRAINING_POLICY_REQUIRED_SUPERVISION = "initial_condition_only"


class ReferenceLeakageError(RuntimeError):
    """Raised when a config requests forbidden reference use after t=0."""


@dataclass
class ExplicitMPFTrainResult:
    run_id: str
    architecture: str
    model_dt: float
    dt_mu_sigma: float
    num_phases: int
    epochs_run: int
    diagnostics_path: str | None
    checkpoint_path: str | None
    summary_path: str | None
    final_diagnostics: dict[str, Any] = field(default_factory=dict)
    launched: bool = False


def _bool_setting(value: Any, *, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


def maybe_load_explicit_mpf_warmstart(
    model: ExplicitMPFHybridRollout,
    settings: dict[str, Any],
    *,
    repo_root: Path,
    device: torch.device,
) -> dict[str, Any] | None:
    """Load an explicit-MPF warm-start checkpoint when requested.

    Supports ``warmstart`` and the ``warm_start`` alias. This loads weights only;
    optimizer and curriculum state are
    intentionally fresh.
    """

    warm = settings.get("warmstart") or settings.get("warm_start") or {}
    if not bool(warm.get("enabled", False)):
        return None

    checkpoint_value = warm.get("checkpoint") or warm.get("checkpoint_path")
    if not checkpoint_value:
        raise ValueError("warmstart.checkpoint is required when warmstart.enabled=true")
    checkpoint = Path(str(checkpoint_value))
    if not checkpoint.is_absolute():
        checkpoint = repo_root / checkpoint
    if not checkpoint.exists():
        raise FileNotFoundError(f"required explicit-MPF warm-start checkpoint not found: {checkpoint}")

    expected_sha = warm.get("checkpoint_sha256")
    if not expected_sha:
        raise ValueError("warmstart.checkpoint_sha256 is required for public checkpoint loading")
    payload = load_torch_checkpoint(
        checkpoint,
        expected_sha256=str(expected_sha),
        map_location=device,
    )
    actual_sha = str(expected_sha)
    state = payload.get("model_state") if isinstance(payload, dict) else None
    if state is None:
        state = payload
    strict = bool(warm.get("strict", True))
    load_result = model.load_state_dict(state, strict=strict)
    missing = list(getattr(load_result, "missing_keys", []))
    unexpected = list(getattr(load_result, "unexpected_keys", []))
    if strict and (missing or unexpected):
        raise RuntimeError(
            "strict explicit-MPF warm-start load reported incompatible keys: "
            f"missing={missing}, unexpected={unexpected}"
        )

    metadata = {
        "enabled": True,
        "checkpoint": str(checkpoint.resolve().relative_to(repo_root.resolve()))
        if checkpoint.resolve().is_relative_to(repo_root.resolve())
        else str(checkpoint),
        "checkpoint_sha256": actual_sha,
        "strict": strict,
        "missing_keys": missing,
        "unexpected_keys": unexpected,
        "init_from": warm.get("init_from", "weights_only"),
    }
    print(
        "[explicit-mpf] WARMSTART loaded "
        f"checkpoint={metadata['checkpoint']} sha256={actual_sha} strict={strict}"
    )
    return metadata


def _require_mapping(value: Any, what: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ReferenceLeakageError(
            f"{what} must be a mapping; got {type(value).__name__}. A configuration "
            "that does not declare its reference-usage policy is refused."
        )
    return value


def assert_no_reference_leakage(settings: Any) -> None:
    """Fail closed unless the configuration declares the full reviewed policy.

    This runs at entry to explicit-MPF training, before any data, model, checkpoint
    or output is touched, and it refuses anything it cannot positively verify:

    * ``settings`` and ``reference_usage_policy`` must be mappings;
    * the reviewed policy key set must be present exactly -- no missing key, no
      unreviewed extra key, and each value of the reviewed type;
    * the three post-``t0`` reference flags must be the literal ``False``;
    * graph features must come from the model's own field;
    * the initial-condition mode must be one of the two reviewed values, and the
      multi-initial-condition contract must hold in both directions;
    * ``training_policy`` must be present and declare initial-condition-only
      supervision with its three no-reference flags literally ``True``.

    Every refusal raises :class:`ReferenceLeakageError`. A malformed configuration
    must never surface as an ``AttributeError`` or a ``TypeError``, because a crash
    is not a refusal and would be indistinguishable from a bug at the call site.

    What this proves, and what it does not: it proves that a run's declared policy
    matches the reviewed contract exactly. It is configuration validation. It does
    not inspect a tensor, open a file, or observe what the training loop does with
    what it loads -- the evidence for actual reference isolation is the data-flow
    audit of this module, not this function.
    """

    settings = _require_mapping(settings, "settings")

    if "reference_usage_policy" not in settings:
        raise ReferenceLeakageError(
            "reference_usage_policy is missing. Explicit-MPF training requires an "
            "explicit reference-usage declaration; silence is not consent."
        )
    policy = _require_mapping(settings["reference_usage_policy"], "reference_usage_policy")

    present = frozenset(policy)
    missing = sorted(REFERENCE_POLICY_KEYS - present)
    if missing:
        raise ReferenceLeakageError(
            f"reference_usage_policy is missing reviewed key(s) {missing}. Each key "
            "declares one channel; an undeclared channel is not a closed channel."
        )
    unreviewed = sorted(present - REFERENCE_POLICY_KEYS)
    if unreviewed:
        raise ReferenceLeakageError(
            f"reference_usage_policy carries unreviewed key(s) {unreviewed}. This "
            "guard can only vouch for the reviewed contract."
        )

    for key in REFERENCE_POLICY_MUST_BE_FALSE:
        value = policy[key]
        if value is not False:
            raise ReferenceLeakageError(
                f"reference_usage_policy.{key} must be the literal False; got "
                f"{value!r} ({type(value).__name__}). A truthy, string, numeric or "
                "null value is a malformed declaration, not a negative one."
            )

    for key, expected_type in REFERENCE_POLICY_TYPES.items():
        value = policy[key]
        if expected_type is bool:
            if value is not True and value is not False:
                raise ReferenceLeakageError(
                    f"reference_usage_policy.{key} must be a literal boolean; got "
                    f"{value!r} ({type(value).__name__})"
                )
        elif not isinstance(value, expected_type) or isinstance(value, bool):
            raise ReferenceLeakageError(
                f"reference_usage_policy.{key} must be {expected_type.__name__}; got "
                f"{value!r} ({type(value).__name__})"
            )

    if policy["graph_features_from"] != REQUIRED_GRAPH_FEATURE_SOURCE:
        raise ReferenceLeakageError(
            f"reference_usage_policy.graph_features_from must be "
            f"{REQUIRED_GRAPH_FEATURE_SOURCE!r}; got {policy['graph_features_from']!r}"
        )

    # Retained for configurations that also carry the older flag spellings: if a
    # forbidden value is declared anywhere, refuse it on its own terms.
    for section, key, forbidden in FORBIDDEN_REFERENCE_USAGE:
        block = settings.get(section)
        if isinstance(block, dict) and block.get(key) is forbidden:
            raise ReferenceLeakageError(
                f"{section}.{key} == {forbidden!r} is forbidden for explicit-MPF training"
            )

    training = settings.get("training")
    if training is not None:
        training = _require_mapping(training, "training")
    multi_ic = training.get("multi_ic_batch") if training else None

    # The six-field firewall is bidirectional. The multi-initial-condition mode
    # requires an exactly-six-entry batch, and any multi_ic_batch block requires
    # exactly that mode -- a multi-IC batch may never execute under the
    # single-initial-condition policy. Supervision stays initial-condition-only in
    # both directions.
    mode = policy["training_initial_condition"]
    if mode == MULTI_IC_MODE:
        if not isinstance(multi_ic, dict):
            raise ReferenceLeakageError(
                f"{MULTI_IC_MODE} requires training.multi_ic_batch to be a mapping"
            )
        entries = multi_ic.get("entries")
        if not isinstance(entries, list) or len(entries) != MULTI_IC_ENTRY_COUNT:
            raise ReferenceLeakageError(
                f"{MULTI_IC_MODE} requires training.multi_ic_batch with exactly "
                f"{MULTI_IC_ENTRY_COUNT} entries; got "
                f"{len(entries) if isinstance(entries, list) else type(entries).__name__}"
            )
        for index, entry in enumerate(entries):
            if not isinstance(entry, dict):
                raise ReferenceLeakageError(
                    f"training.multi_ic_batch.entries[{index}] must be a mapping; got "
                    f"{type(entry).__name__}"
                )
    elif mode == SINGLE_IC_MODE:
        if multi_ic:
            raise ReferenceLeakageError(
                f"training.multi_ic_batch requires training_initial_condition == "
                f"{MULTI_IC_MODE} ({SINGLE_IC_MODE} refused)"
            )
    else:
        raise ReferenceLeakageError(
            f"reference_usage_policy.training_initial_condition must be "
            f"{SINGLE_IC_MODE!r} or {MULTI_IC_MODE!r}; got {mode!r}"
        )

    if "training_policy" not in settings:
        raise ReferenceLeakageError(
            "training_policy is missing. Every accepted explicit-MPF training "
            "configuration declares it; a run that does not is refused."
        )
    training_policy = _require_mapping(settings["training_policy"], "training_policy")
    supervision = training_policy.get("supervision")
    if supervision != TRAINING_POLICY_REQUIRED_SUPERVISION:
        raise ReferenceLeakageError(
            f"training_policy.supervision must be "
            f"{TRAINING_POLICY_REQUIRED_SUPERVISION!r}; got {supervision!r}"
        )
    for key in TRAINING_POLICY_MUST_BE_TRUE:
        if key not in training_policy:
            raise ReferenceLeakageError(f"training_policy.{key} is missing")
        value = training_policy[key]
        if value is not True:
            raise ReferenceLeakageError(
                f"training_policy.{key} must be the literal True; got {value!r} "
                f"({type(value).__name__})"
            )


def build_explicit_mpf_model(
    settings: dict[str, Any],
    reference: ExplicitMPFReference,
) -> ExplicitMPFHybridRollout:
    """Construct the hybrid rollout from a config and loaded initial state.

    The physical timestep is taken from ``reference.model_dt`` (already derived
    as ``dt_mu_sigma/(mu*sigma)`` by the loader); the config ``model_dt`` is only
    used to cross-check and warn on disagreement.  ``mu``/``sigma``/``eta_px``
    come from the reference physics so the residual matches the reference.
    """

    model_cfg = settings.get("model") or {}
    if str(model_cfg.get("architecture")) != "explicit_mpf_ann_convlstm_hybrid":
        raise ValueError("explicit-MPF trainer requires architecture explicit_mpf_ann_convlstm_hybrid")

    # ADDITIVE, DEFAULT-OFF: when ``arch_variant`` is absent (or any value other
    # than ``perm_equivariant_v1``) the construction below is byte-identical to
    # the legacy path.  Only ``perm_equivariant_v1`` diverts to the equivariant
    # rollout; every existing config therefore behaves exactly as before.
    if model_cfg.get("arch_variant") == "perm_equivariant_v1":
        return _build_perm_equivariant_v1(model_cfg, reference)

    graph_cfg = model_cfg.get("graph") or {}
    graph_mode = str(graph_cfg.get("mode", "disabled")) if graph_cfg.get("enabled", False) else "disabled"
    phase_id_encoding = model_cfg.get("phase_id_encoding", "normalized")
    if not isinstance(phase_id_encoding, str) or phase_id_encoding not in {"normalized", "none", "zero_masked"}:
        raise ValueError(
            "phase_id_encoding must be 'normalized', 'none', or 'zero_masked'; "
            f"got {phase_id_encoding!r}"
        )
    if phase_id_encoding == "none" and graph_mode != "disabled":
        raise ValueError(
            "phase_id_encoding='none' requires graph mode disabled because the current "
            "graph extractor still includes phase identity and centroid features"
        )

    cfg_model_dt = settings.get("model_dt", settings.get("training", {}).get("model_dt"))
    if cfg_model_dt is not None and abs(float(cfg_model_dt) - reference.model_dt) > 1e-6 * max(1.0, reference.model_dt):
        print(
            f"[explicit-mpf] WARNING: config model_dt={float(cfg_model_dt):.6g} "
            f"!= reference dt_mu_sigma/(mu*sigma)={reference.model_dt:.6g}; using the reference value."
        )

    spatial_dims = int(model_cfg.get("spatial_dims", 2))
    return ExplicitMPFHybridRollout(
        num_phases=reference.num_phases,
        model_dt=reference.model_dt,
        eta_px=reference.eta_px,
        mu=reference.mu,
        sigma=reference.sigma,
        spatial_dims=spatial_dims,
        hidden_channels=int(model_cfg.get("hidden_channels", 32)),
        kernel_size=int(model_cfg.get("kernel_size", 3)),
        ann_hidden_features=tuple(int(w) for w in model_cfg.get("ann_hidden_features", (32, 32))),
        learnable_blend=bool(model_cfg.get("learnable_blend", True)),
        blend_gamma=float(model_cfg.get("blend_gamma", 0.5)),
        per_phase_gamma=bool(model_cfg.get("per_phase_gamma", False)),
        projection_mode=str(model_cfg.get("projection_mode", "simplex_clip_normalize")),
        graph_mode=graph_mode,
        graph_hidden_dim=int(graph_cfg.get("hidden_dim", 32) or 32),
        graph_layers=int(graph_cfg.get("message_passing_layers", 1) or 1),
        graph_conditioning_scale=float(graph_cfg.get("conditioning_scale", 1.0) or 1.0),
        graph_center_bias=_bool_setting(graph_cfg.get("center_bias", False)),
        coordinate_encoding=str(model_cfg.get("coordinate_encoding", "periodic")),
        phase_id_encoding=phase_id_encoding,
        conv_padding_mode=str(model_cfg.get("conv_padding_mode", "circular")),
        rate_head_bounded=bool(model_cfg.get("rate_head_bounded", False)),
        max_delta_phi_per_step=float(model_cfg.get("max_delta_phi_per_step", 0.05)),
        rate_parameterization=str(model_cfg.get("rate_parameterization", "legacy_tanh_cap")),
        delta_scale=float(model_cfg.get("delta_scale", 0.10)),
        centering_mode=str(model_cfg.get("centering_mode", "phase_mean")),
    )


def _build_perm_equivariant_v1(
    model_cfg: dict[str, Any],
    reference: ExplicitMPFReference,
) -> PermEquivariantMPFRollout:
    """Construct the additive ``perm_equivariant_v1`` rollout with frozen constraints.

    The structural constraints are enforced here:
    no coordinate encoding, no phase-id encoding, no per-phase gamma, no graph.
    """

    if str(model_cfg.get("coordinate_encoding", "none")) != "none":
        raise ValueError("perm_equivariant_v1 requires model.coordinate_encoding: none")
    if str(model_cfg.get("phase_id_encoding", "none")) != "none":
        raise ValueError("perm_equivariant_v1 requires model.phase_id_encoding: none")
    if bool(model_cfg.get("per_phase_gamma", False)):
        raise ValueError("perm_equivariant_v1 requires model.per_phase_gamma: false")
    graph_cfg = model_cfg.get("graph") or {}
    if bool(graph_cfg.get("enabled", False)):
        raise ValueError("perm_equivariant_v1 requires model.graph.enabled: false")
    if int(model_cfg.get("spatial_dims", 2)) != 2:
        raise ValueError("perm_equivariant_v1 supports only spatial_dims: 2")
    projection_mode = str(model_cfg.get("projection_mode", "soft_threshold_eps1e3"))
    if projection_mode != "soft_threshold_eps1e3":
        raise ValueError(
            "perm_equivariant_v1 requires model.projection_mode: soft_threshold_eps1e3"
        )
    # These three keys are frozen and non-configurable on the model (see
    # PermEquivariantMPFRollout._validate_frozen_metadata_contract). If the config
    # states them explicitly (as the corrected frozen config now does), cross-check
    # rather than silently accept a drifted value -- fail fast instead of masking a
    # future config/model divergence.
    delta_scale = float(model_cfg.get("delta_scale", 0.10))
    if "rate_parameterization" in model_cfg and str(model_cfg["rate_parameterization"]) != "scaled_bounded":
        raise ValueError(
            "perm_equivariant_v1 requires model.rate_parameterization: scaled_bounded "
            f"(config states {model_cfg['rate_parameterization']!r})"
        )
    if "rate_head_bounded" in model_cfg and bool(model_cfg["rate_head_bounded"]) is not False:
        raise ValueError("perm_equivariant_v1 requires model.rate_head_bounded: false")
    if "max_delta_phi_per_step" in model_cfg:
        cfg_cap = float(model_cfg["max_delta_phi_per_step"])
        if abs(cfg_cap - delta_scale) > 1.0e-9:
            raise ValueError(
                "perm_equivariant_v1 requires model.max_delta_phi_per_step == model.delta_scale "
                f"(inactive cap; config has {cfg_cap!r} != delta_scale {delta_scale!r})"
            )
    return PermEquivariantMPFRollout(
        num_phases=reference.num_phases,
        model_dt=reference.model_dt,
        eta_px=reference.eta_px,
        mu=reference.mu,
        sigma=reference.sigma,
        hidden_channels=int(model_cfg.get("hidden_channels", 8)),
        encoder_channels=int(model_cfg.get("encoder_channels", 4)),
        kernel_size=int(model_cfg.get("kernel_size", 3)),
        ann_hidden_features=tuple(int(w) for w in model_cfg.get("ann_hidden_features", (32, 32))),
        blend_gamma=float(model_cfg.get("blend_gamma", 0.5)),
        learnable_blend=bool(model_cfg.get("learnable_blend", True)),
        delta_scale=delta_scale,
        projection_mode=projection_mode,
        conv_padding_mode=str(model_cfg.get("conv_padding_mode", "circular")),
    )


def _phase_sum_consistency(raw_next: Tensor) -> Tensor:
    """Mean squared pre-projection phase-sum error (PINNs-MPF L_sum analogue)."""

    return torch.mean((raw_next.sum(dim=1) - 1.0) ** 2)


def mpf_residual_loss(
    rate: Tensor,
    target: Tensor,
    *,
    model_dt: float,
    normalization: str = "raw",
    eps: float = 1.0e-8,
    abs_scale: float = 0.05,
    mixed_w_rel: float = 1.0,
    mixed_w_abs: float = 1.0,
) -> Tensor:
    """Scale-aware MPF residual loss.

    ``raw`` (legacy) penalises ``mean((rate - target)^2)`` in raw physical-rate
    units.  Because the physical target rate is ~1e-5, this loss is microscopic
    for *any* rate and gives the optimizer almost no gradient signal to keep the
    learned rate at the physical scale — the failure mode that collapsed the
    first 300-epoch run (learned rate ~1e-2, ~690x too large, masked by projection).

    ``displacement_relative`` compares the physically meaningful per-step
    *displacement* ``model_dt * rate`` against ``model_dt * target`` and divides by
    the target displacement energy, yielding an O(1) relative loss:

        || dt*rate - dt*target ||^2 / (|| dt*target ||^2 + eps)

    This restores a usable gradient but it RE-NORMALISES per step, so a near-static
    wrong late state (small target energy) is under-penalised relative to the early
    transient — the failure mode of the STABILIZED run (relative residual collapsed
    ~217 -> ~1.65 from early to late while the state stayed wrong).

    ``mixed`` (stabilized-v2) keeps the relative term for scale-invariant gradient
    AND adds an *absolute* displacement term normalised by a FIXED ``abs_scale``
    (not the per-step target), so late wrong states stay penalised:

        mixed_w_rel * relative + mixed_w_abs * ||dt*rate - dt*target||^2 / abs_scale^2

    ``rate_relative`` is the displacement-relative idea applied directly to rates
    (equivalent up to the constant ``model_dt^2`` that cancels in the ratio).
    """

    if normalization == "raw":
        return torch.mean((rate - target) ** 2)
    if normalization == "rate_relative":
        num = torch.mean((rate - target) ** 2)
        den = torch.mean(target ** 2) + eps
        return num / den
    if normalization == "displacement_relative":
        d_pred = model_dt * rate
        d_target = model_dt * target
        num = torch.mean((d_pred - d_target) ** 2)
        den = torch.mean(d_target ** 2) + eps
        return num / den
    if normalization == "mixed":
        d_pred = model_dt * rate
        d_target = model_dt * target
        rel = torch.mean((d_pred - d_target) ** 2) / (torch.mean(d_target ** 2) + eps)
        abs_term = torch.mean((d_pred - d_target) ** 2) / (abs_scale ** 2 + eps)
        return mixed_w_rel * rel + mixed_w_abs * abs_term
    raise ValueError(f"unknown mpf_residual_normalization: {normalization!r}")


def interface_local_weight_mask(
    phi: Tensor,
    *,
    interface_eps: float = 0.05,
    interface_weight: float = 2.0,
) -> Tensor:
    """Return a per-pixel weight that emphasizes interfaces and junctions.

    The mask is computed from the model's OWN current ``phi`` ([B,N,H,W]) — never
    from any reference frame after t=0.  A pixel is "interface" when at least two
    phases have non-trivial support, detected via the SECOND-LARGEST phase value
    exceeding ``interface_eps`` (bulk pixels have one phase ~1 and the rest ~0, so
    their second-largest is ~0 and they get the base weight 1.0).

    Returns a [B,1,H,W] weight = 1.0 in the bulk and ``interface_weight`` at
    interfaces.  The mask is DETACHED: it modulates *where* the existing residual
    is emphasised, it is not itself a gradient path (so it cannot be gamed and
    keeps the residual's own differentiability intact).  This is a weighting, not
    a new physical loss term.
    """

    with torch.no_grad():
        # second-largest phase value per pixel (top-2 along the phase channel)
        top2 = torch.topk(phi, k=min(2, phi.shape[1]), dim=1).values  # [B,k,H,W]
        second = top2[:, 1:2] if top2.shape[1] > 1 else torch.zeros_like(top2[:, :1])
        is_interface = (second > interface_eps).to(phi.dtype)  # [B,1,H,W]
        weight = 1.0 + (interface_weight - 1.0) * is_interface
    return weight  # detached by construction


def interface_local_residual_loss(
    rate: Tensor,
    target: Tensor,
    phi: Tensor,
    *,
    model_dt: float,
    weight_mask: Tensor,
    eps: float = 1.0e-8,
    abs_scale: float = 0.05,
    mixed_w_rel: float = 1.0,
    mixed_w_abs: float = 1.0,
) -> Tensor:
    """Return the mixed displacement residual with an interface-weighted mean.

    Identical mathematics to ``mpf_residual_loss(normalization="mixed")`` except the
    pixel-wise squared displacement error is averaged with ``weight_mask`` (detached,
    from ``interface_local_weight_mask``) instead of a uniform mean.  When the mask is
    all-ones this reduces EXACTLY to the uniform mixed residual, so the term is a strict
    generalisation and the default-disabled path is unchanged.

    The relative-normalisation denominator and the abs_scale floor are UNCHANGED
    (uniform), so only the *spatial emphasis* of the numerator changes — keeping the
    loss scale comparable to the unweighted mixed residual.
    """

    d_pred = model_dt * rate
    d_target = model_dt * target
    sq = (d_pred - d_target) ** 2  # [B,N,H,W]
    w = weight_mask  # [B,1,H,W], broadcast over phases; detached
    wmean = (w * sq).sum() / (w.expand_as(sq).sum() + eps)
    rel = wmean / (torch.mean(d_target ** 2) + eps)
    abs_term = wmean / (abs_scale ** 2 + eps)
    return mixed_w_rel * rel + mixed_w_abs * abs_term


def mpf_pairwise_energy_density(
    phi: Tensor,
    *,
    eta_px: float,
    sigma: float,
) -> Tensor:
    """Return local MPF energy density from model ``phi`` only.

    This is the MPF Energy-LH2 density, not the scalar Allen-Cahn energy:

        sigma * [sum_i 0.5 |grad phi_i|^2
                 + pi^2/(2 eta_px^2) * sum_{i<j} phi_i^2 phi_j^2]

    Gradients use the same unit-grid periodic forward-difference convention as
    the explicit-MPF RHS helpers.
    """

    if phi.ndim != 4 or phi.shape[1] < 2:
        raise ValueError("phi must have shape [B,N,H,W] with at least two phases")
    if eta_px <= 0.0 or sigma <= 0.0:
        raise ValueError("eta_px and sigma must be positive")
    grad_y = torch.roll(phi, -1, dims=-2) - phi
    grad_x = torch.roll(phi, -1, dims=-1) - phi
    gradient_density = 0.5 * (grad_y.square() + grad_x.square()).sum(dim=1)
    phi2 = phi.square()
    pairwise_well = 0.5 * (phi2.sum(dim=1).square() - phi2.square().sum(dim=1))
    well_density = (math.pi**2 / (2.0 * eta_px * eta_px)) * pairwise_well
    return float(sigma) * (gradient_density + well_density)


def energy_lh2_interface_mask(phi: Tensor, *, interface_eps: float) -> Tensor:
    """Return Energy-LH2 model-phi interface mask [B,H,W].

    A pixel is an interface when the second-largest phase value exceeds
    ``interface_eps``.  No reference frame or detached reference mask is involved.
    """

    if phi.ndim != 4 or phi.shape[1] < 2:
        raise ValueError("phi must have shape [B,N,H,W] with at least two phases")
    top2 = torch.topk(phi, k=2, dim=1).values
    return top2[:, 1] > float(interface_eps)


def energy_lh2_density_loss(
    phi: Tensor,
    phi_next: Tensor,
    *,
    eta_px: float,
    sigma: float,
    lambda_energy_density: float,
    interface_eps: float = 0.05,
    tol: float = 0.0,
    normalization: dict | None = None,
) -> tuple[Tensor, Tensor, dict[str, float]]:
    """Return raw and weighted Energy-LH2 local/interface residual.

    The loss is

        mean_interface relu(e(phi_next) - e(phi) + tol)^2

    with ``e`` from :func:`mpf_pairwise_energy_density`.  If the model-phi
    interface mask is empty, the raw loss is a zero scalar that remains connected
    to the energy computation graph.

    Optional ``normalization`` dict (``mode: fixed_raw_scale``) divides ``raw``
    by a pre-calibrated scale before weighting, so that ``lambda_energy_density``
    can be tuned in a normal range (e.g. 0.001–0.01) regardless of the physical
    energy units.  ``raw`` is always the un-normalised physical value.
    """

    if phi.shape != phi_next.shape:
        raise ValueError("phi and phi_next must have the same shape")
    norm_cfg = normalization or {}
    norm_enabled = bool(norm_cfg.get("enabled", False))
    if norm_enabled:
        norm_mode = str(norm_cfg.get("mode", "fixed_raw_scale"))
        if norm_mode != "fixed_raw_scale":
            raise ValueError(
                f"energy_lh2 normalization.mode must be fixed_raw_scale, got {norm_mode!r}"
            )
        raw_scale = float(norm_cfg.get("raw_scale", 1.0))
        norm_eps = float(norm_cfg.get("eps", 1.0e-12))
        if raw_scale <= 0.0:
            raise ValueError("energy_lh2 normalization.raw_scale must be positive")
        effective_scale = max(raw_scale, norm_eps)
    energy_cur = mpf_pairwise_energy_density(phi, eta_px=eta_px, sigma=sigma)
    energy_next = mpf_pairwise_energy_density(phi_next, eta_px=eta_px, sigma=sigma)
    delta = energy_next - energy_cur
    mask = energy_lh2_interface_mask(phi, interface_eps=interface_eps)
    mask_f = mask.to(dtype=delta.dtype)
    count = mask_f.sum()
    denom = count.clamp_min(1.0)
    positive = torch.relu(delta + float(tol))
    raw = (positive.square() * mask_f).sum() / denom
    if norm_enabled:
        normalized = raw / effective_scale
        weighted = float(lambda_energy_density) * normalized
    else:
        normalized = None
        weighted = float(lambda_energy_density) * raw
    with torch.no_grad():
        positive_count = ((positive > 0.0) & mask).to(delta.dtype).sum()
        diagnostics: dict[str, float | str] = {
            "energy_lh2_mean_delta": ((delta * mask_f).sum() / denom).detach().item(),
            "energy_lh2_positive_fraction": (
                positive_count / denom
            ).detach().item(),
            "energy_lh2_interface_fraction": mask_f.mean().detach().item(),
            "energy_lh2_lambda": float(lambda_energy_density),
        }
        if norm_enabled and normalized is not None:
            diagnostics["energy_lh2_loss_normalized"] = normalized.detach().item()
            diagnostics["energy_lh2_norm_mode"] = "fixed_raw_scale"
            diagnostics["energy_lh2_raw_scale"] = effective_scale
    return raw, weighted, diagnostics


def surface_grad_energy(phi: Tensor) -> Tensor:
    """Dirichlet/gradient interface energy  S_grad = sum_i |grad phi_i|^2.

    Periodic forward differences over the spatial dims (2..end); summed over phase
    channels + space, returned per-sample (shape ``(B,)``).  Works for 2D
    ``(B,N,H,W)`` and 3D ``(B,N,Z,Y,X)``.  This is the sharpen-safe, monotone-under-
    coarsening surface proxy used by the reviewed surface-descent check. The
    bare ``phi(1-phi)`` alternative is non-monotone and sharpen-gameable.
    """
    spatial = tuple(range(2, phi.dim()))
    g = torch.zeros_like(phi)
    for ax in spatial:
        d = torch.roll(phi, -1, dims=ax) - phi
        g = g + d * d
    return g.sum(dim=tuple(range(1, phi.dim())))


def _surface_huber(x: Tensor, delta: float) -> Tensor:
    a = x.abs()
    return torch.where(a < delta, 0.5 * x * x, delta * (a - 0.5 * delta))


def surface_descent_consistency_loss(
    window_start_phi: Tensor,
    model_end_phi: Tensor,
    n_steps: int,
    *,
    model_dt: float,
    eta_px: float,
    mu: float,
    sigma: float,
    projection_mode: str,
    huber_delta: float = 0.02,
    eps: float = 1.0e-8,
) -> tuple[Tensor, dict[str, float]]:
    """Windowed, SYMMETRIC surface descent-consistency loss (model-phi only).

    Anchors the model's relative ``S_grad`` reduction across a TBPTT window to the
    explicit-MPF model-field RHS rolled the same number of steps from the SAME
    window-start state.  NO post-t0 reference frame is used: the only physics
    signal is ``explicit_mpf_rhs(window_start_phi)``.  The oracle rollout is
    no-grad/detached, so the gradient flows only through ``model_end_phi`` (the
    model's window-end state) into the rate head.

        r_pred = (S_grad(model_end_phi) - S_grad(phi0)) / (S_grad(phi0) + eps)
        r_rhs  = (S_grad(oracle_k(phi0)) - S_grad(phi0)) / (S_grad(phi0) + eps)   [detached]
        L      = Huber(r_pred - r_rhs)                                            [symmetric]

    The symmetric (not one-sided) form penalises both under-coarsening (sharpen
    adversary) and over-coarsening (blur adversary), because either deviates from
    the oracle's specific decrement.
    """
    sg0 = surface_grad_energy(window_start_phi).detach()
    denom = sg0 + eps
    r_pred = (surface_grad_energy(model_end_phi) - sg0) / denom
    with torch.no_grad():
        po = window_start_phi.detach()
        for _ in range(max(int(n_steps), 1)):
            tg = explicit_mpf_rhs(po, eta_px=eta_px, mu=mu, sigma=sigma)
            raw = po + model_dt * tg
            if projection_mode == "soft_threshold_eps1e3":
                po, _ = project_simplex_soft_threshold(raw, threshold_eps=1.0e-3)
            else:
                po, _ = project_simplex(raw)
        r_rhs = (surface_grad_energy(po) - sg0) / denom
    loss = _surface_huber(r_pred - r_rhs, huber_delta).mean()
    diag = {
        "surface_r_pred": float(r_pred.mean().detach()),
        "surface_r_rhs": float(r_rhs.mean().detach()),
        "surface_loss": float(loss.detach()),
    }
    return loss, diag


def evaluate_early_stop_gates(
    row: dict[str, Any],
    *,
    epoch: int,
    num_phases: int,
    cfg: dict[str, Any],
    zero_rate_baseline: float | None = None,
) -> tuple[bool, list[str]]:
    """Evaluate configured early-stop checks against a review row.

    Returns ``(should_stop, reasons)``.  Gates are config-driven under
    ``training.early_stop`` and are OFF unless ``enabled: true``.  These encode the
    configured stop conditions:

    * active phases drop below ``min_active_phases`` (topology collapse);
    * cap saturation stays near 1.0 after the early curriculum phase;
    * projection correction becomes the main dynamics (L1 too high);
    * learned rollout stays worse than zero-rate at a review point (if provided);
    * pre-bound delta ratio stays enormous past the warm-up.
    """

    es = cfg or {}
    if not es.get("enabled", False):
        return False, []
    reasons: list[str] = []
    grace = int(es.get("grace_epochs", 0))
    if epoch < grace:
        return False, []

    min_active = int(es.get("min_active_phases", num_phases))
    if int(row.get("active_phase_count", num_phases)) < min_active:
        reasons.append(
            f"active_phase_count {row.get('active_phase_count')} < {min_active} (topology collapse)"
        )

    cap_sat_max = float(es.get("max_cap_saturation_after_grace", 1.01))
    # delta_to_target_ratio is a proxy: if the bounded delta is pinned at the cap
    # while target is small, ratio stays high; combined with proj reliance it flags
    # cap-as-dynamics.  We use projection_correction_l1 as the primary saturation
    # proxy available in the review row.
    proj_l1_max = float(es.get("max_projection_l1", 1.0))
    if float(row.get("projection_correction_l1", 0.0)) > proj_l1_max:
        reasons.append(
            f"projection_correction_l1 {row.get('projection_correction_l1'):.3f} > {proj_l1_max} "
            "(projection becoming the main dynamics)"
        )

    if zero_rate_baseline is not None:
        margin = float(es.get("worse_than_zero_rate_margin", 0.0))
        learned = float(row.get("learned_vs_zero_rate_disagree", float("nan")))
        if learned == learned and learned > zero_rate_baseline + margin:  # not NaN
            reasons.append(
                f"learned argmax disagree {learned:.3f} > zero-rate {zero_rate_baseline:.3f}+{margin} "
                "(worse than zero-rate)"
            )

    # cap_sat_max gate uses an explicit cap-saturation field if present.
    cap_sat = row.get("cap_saturation_fraction")
    if cap_sat is not None and float(cap_sat) > cap_sat_max:
        reasons.append(
            f"cap_saturation_fraction {float(cap_sat):.3f} > {cap_sat_max} after grace "
            "(cap is the dynamics engine)"
        )

    return (len(reasons) > 0), reasons


# ---------------------------------------------------------------------------
# Two-signal model-only health check. No post-t0 reference is read anywhere in
# this section: every metric is computed from the model's own rollout of its
# own initial condition.
# ---------------------------------------------------------------------------

#: Frozen deterministic connectivity rule for the fragmentation metric.
#: "4" = 4-connectivity (no
#: diagonal neighbors, i.e. ``scipy.ndimage.generate_binary_structure(2, 1)``,
#: the plus/cross-shaped structuring element). This choice is FROZEN: it must
#: not be changed silently because configured thresholds depend on this rule.
HEALTH_GATE_CONNECTIVITY_CHOICES = ("4", "8")


def _health_gate_connectivity_structure(connectivity: str):
    """Return the frozen ``scipy.ndimage`` structuring element for ``connectivity``."""

    from scipy.ndimage import generate_binary_structure

    if connectivity == "4":
        return generate_binary_structure(2, 1)
    if connectivity == "8":
        return generate_binary_structure(2, 2)
    raise ValueError(
        f"unknown health_gate connectivity {connectivity!r}; must be one of "
        f"{HEALTH_GATE_CONNECTIVITY_CHOICES}"
    )


def fragmentation_component_count(labels: np.ndarray, num_phases: int, *, connectivity: str = "4") -> int:
    """Return the total connected-component count summed over ALL phase labels.

    ``labels`` is a single 2D integer argmax-label field (e.g. ``[H, W]``). For
    each phase value ``0..num_phases-1`` the binary mask ``labels == phase`` is
    labeled with the FROZEN deterministic connectivity rule (see
    ``_health_gate_connectivity_structure``) and its component count is added to
    the running total. A phase with zero pixels contributes 0 components. This
    is the model-only fragmentation signal used by the health check.
    """

    from scipy.ndimage import label as ndimage_label

    structure = _health_gate_connectivity_structure(connectivity)
    total = 0
    for phase in range(num_phases):
        mask = labels == phase
        if not mask.any():
            continue
        _, num_components = ndimage_label(mask, structure=structure)
        total += int(num_components)
    return total


@torch.no_grad()
def compute_health_gate_metrics(
    model: ExplicitMPFHybridRollout | PermEquivariantMPFRollout,
    phi0: Tensor,
    *,
    review_step: int,
    connectivity: str = "4",
) -> dict[str, Any]:
    """Roll the model's OWN dynamics to ``review_step`` and return the three health-gate metrics.

    No-grad, model-only (only ``phi0`` -- the model's own ``t=0`` field -- and the
    model's own forward/projection are used; no reference frame after ``t=0`` is
    ever read here). Runs step-by-step (never stacking a full trajectory) so
    memory stays bounded regardless of ``review_step``.

    Returns
    -------
    dict with keys ``margin_mean`` (mean winner-runner-up simplex margin over all
    pixels), ``largest_phase_area_fraction`` (argmax pixel-count share of the
    single largest phase), ``fragmentation`` (total connected-component count
    over all phase labels under the frozen connectivity rule), and
    ``health_gate_review_step`` (the step actually reached, for audit).
    """

    if review_step < 1:
        raise ValueError("review_step must be positive")
    phi = phi0.detach()
    state = model.initial_state(phi)
    previous_phi: Tensor | None = None
    for _ in range(review_step):
        rate, state, _ = model.predict_rate(phi, state, previous_phi=previous_phi)
        raw_next = phi + model.model_dt * rate
        if model.projection_mode == "soft_threshold_eps1e3":
            next_phi, _ = project_simplex_soft_threshold(raw_next, threshold_eps=1.0e-3)
        else:
            next_phi, _ = project_simplex(raw_next)
        previous_phi = phi
        phi = next_phi

    num_phases = phi.shape[1]
    sorted_vals, _ = torch.sort(phi, dim=1, descending=True)
    winner = sorted_vals[:, 0]
    runner_up = sorted_vals[:, 1] if num_phases > 1 else torch.zeros_like(winner)
    margin_per_element = (winner - runner_up).mean(dim=(1, 2))  # [B]

    labels = phi.argmax(dim=1)  # [B, H, W]
    per_element: list[dict[str, Any]] = []
    for b in range(labels.shape[0]):
        labels_np = labels[b].detach().cpu().numpy()
        total_pixels = int(labels_np.size)
        counts = np.bincount(labels_np.reshape(-1), minlength=num_phases)
        per_element.append({
            "batch_index": b,
            "margin_mean": float(margin_per_element[b].item()),
            "largest_phase_area_fraction": (
                float(counts.max() / total_pixels) if total_pixels > 0 else 0.0
            ),
            "fragmentation": fragmentation_component_count(
                labels_np, num_phases, connectivity=connectivity
            ),
            "active_phase_count": int((counts > 0).sum()),
        })

    # Aggregate = WORST batch member for every signal, so the frozen conjunctive
    # gate (evaluate_two_signal_health_gate, unchanged) aborts when ANY batch
    # member trips. With batch size 1 these aggregates
    # equal the single element's values (N3 behaviour byte-identical).
    return {
        "margin_mean": min(e["margin_mean"] for e in per_element),
        "largest_phase_area_fraction": max(
            e["largest_phase_area_fraction"] for e in per_element
        ),
        "fragmentation": max(e["fragmentation"] for e in per_element),
        "health_gate_review_step": review_step,
        "per_element": per_element,
    }


def evaluate_two_signal_health_gate(
    *,
    epoch: int,
    margin_mean: float,
    largest_phase_area_fraction: float,
    fragmentation: int,
    cfg: dict[str, Any],
) -> tuple[bool, list[str]]:
    """Evaluate the frozen conjunctive model-only health check.

    From ``gate_active_from_epoch`` onward, the run may stop ONLY when
    ``margin_mean < margin_mean_threshold`` AND (
    ``largest_phase_area_fraction > largest_phase_area_fraction_threshold`` OR
    ``fragmentation >= fragmentation_threshold``). Margin alone is advisory and
    can never stop the run by itself; area/fragmentation alone (with a high
    margin) can never stop the run either -- both signals must corroborate.

    Returns ``(should_stop, reasons)``; empty reasons when the gate is disabled,
    not yet active, or not triggered.
    """

    hg = cfg or {}
    if not bool(hg.get("enabled", False)):
        return False, []
    gate_active_from_epoch = int(hg.get("gate_active_from_epoch", 50))
    if epoch < gate_active_from_epoch:
        return False, []

    margin_threshold = float(hg.get("margin_mean_threshold", 0.10))
    area_threshold = float(hg.get("largest_phase_area_fraction_threshold", 0.50))
    fragmentation_threshold = int(hg.get("fragmentation_threshold", 1925))

    margin_low = margin_mean < margin_threshold
    area_collapse = largest_phase_area_fraction > area_threshold
    fragmentation_collapse = fragmentation >= fragmentation_threshold

    if margin_low and (area_collapse or fragmentation_collapse):
        reasons = [
            f"health_gate: margin_mean {margin_mean:.4f} < {margin_threshold} "
            "(model-only two-signal health gate)"
        ]
        if area_collapse:
            reasons.append(
                f"health_gate: largest_phase_area_fraction {largest_phase_area_fraction:.4f} "
                f"> {area_threshold} (corroborating collapse signal)"
            )
        if fragmentation_collapse:
            reasons.append(
                f"health_gate: fragmentation {fragmentation} >= {fragmentation_threshold} "
                "(corroborating collapse signal)"
            )
        return True, reasons
    return False, []


def evaluate_health_gate_review(
    model: ExplicitMPFHybridRollout | PermEquivariantMPFRollout,
    phi0: Tensor,
    *,
    epoch: int,
    enabled: bool,
    review_step: int,
    connectivity: str,
    cfg: dict[str, Any],
) -> tuple[bool, list[str], dict[str, Any]]:
    """Compute and fail-closed-classify one review-epoch health-gate evaluation.

    This function is separate from the training loop so it can be tested without
    running an optimizer step. It returns ``(health_should_stop, health_stop_reasons,
    row_updates)`` where ``row_updates`` is the dict the caller must merge into
    the diagnostics review row BEFORE it is written to ``diagnostics.jsonl``.

    - ``enabled=False``: returns ``(False, [], {})`` -- a disabled gate can
      neither stop the run nor report an error; no key is added.
    - ``enabled=True`` and computation succeeds: ``row_updates`` carries the raw
      metrics plus ``health_gate_status`` in ``{"ok", "stop"}``,
      ``health_gate_should_stop``, and ``health_gate_stop_reasons``.
    - ``enabled=True`` and computation raises: FAIL CLOSED, never fail open.
      ``health_should_stop`` is forced ``True`` regardless of what the exception
      was; ``row_updates`` carries ``health_gate_status="error"``,
      ``health_gate_should_stop=True``, ``health_gate_stop_reasons`` naming the
      failure, and ``health_gate_error`` with the preserved exception text. The
      caller folds the forced stop into the SAME should_stop/early_stop
      mechanism a triggered gate uses (no new promotion path), so the resulting
      checkpoint is classified an aborted diagnostic, never a promotable final.
    """

    if not enabled:
        return False, [], {}
    try:
        health_metrics = compute_health_gate_metrics(
            model, phi0, review_step=review_step, connectivity=connectivity
        )
        # Evaluate the frozen conjunctive check separately for every batch
        # member and combine the
        # per-member stop decisions. Mixed cross-member extrema must never
        # trigger the gate (margin_low from IC A + area_high from IC B is not
        # a stop); the extrema keys in health_metrics are descriptive only.
        member_decisions: list[dict[str, Any]] = []
        health_should_stop = False
        health_stop_reasons: list[str] = []
        for element in health_metrics["per_element"]:
            member_stop, member_reasons = evaluate_two_signal_health_gate(
                epoch=epoch,
                margin_mean=element["margin_mean"],
                largest_phase_area_fraction=element["largest_phase_area_fraction"],
                fragmentation=element["fragmentation"],
                cfg=cfg,
            )
            member_decisions.append({
                **element,
                "should_stop": bool(member_stop),
                "stop_reasons": list(member_reasons),
            })
            if member_stop:
                health_should_stop = True
                health_stop_reasons.extend(
                    f"batch_member_{element['batch_index']}: {reason}"
                    for reason in member_reasons
                )
        row_updates: dict[str, Any] = {
            "health_gate_margin_mean": health_metrics["margin_mean"],
            "health_gate_largest_phase_area_fraction": health_metrics[
                "largest_phase_area_fraction"
            ],
            "health_gate_fragmentation": health_metrics["fragmentation"],
            "health_gate_review_step": health_metrics["health_gate_review_step"],
            "health_gate_status": "stop" if health_should_stop else "ok",
            "health_gate_should_stop": bool(health_should_stop),
            "health_gate_stop_reasons": list(health_stop_reasons),
            "health_gate_per_ic": member_decisions,
            "health_gate_aggregation": "per_member_or_extrema_descriptive_only",
        }
        return health_should_stop, health_stop_reasons, row_updates
    except Exception as exc:  # FAIL CLOSED: hard-stop, never silently continue.
        health_gate_error = str(exc)
        health_stop_reasons = [f"health_gate_computation_error: {health_gate_error}"]
        row_updates = {
            "health_gate_status": "error",
            "health_gate_should_stop": True,
            "health_gate_stop_reasons": list(health_stop_reasons),
            "health_gate_error": health_gate_error,
        }
        return True, health_stop_reasons, row_updates


def resolve_curriculum_horizon(curriculum: list, epoch: int, default_horizon: int) -> int:
    """Return the active horizon for ``epoch`` given a curriculum schedule.

    ``curriculum`` is a list of ``[epoch_start, horizon]`` pairs (sorted or not).
    The active horizon is the one with the largest ``epoch_start <= epoch``.  Empty
    curriculum returns ``default_horizon``.  Resolved horizon is capped at
    ``default_horizon`` so the curriculum can only ramp UP to the configured max.
    """

    if not curriculum:
        return default_horizon
    active = default_horizon
    chosen_start = -1
    for pair in curriculum:
        start, h = int(pair[0]), int(pair[1])
        if start <= epoch and start > chosen_start:
            chosen_start = start
            active = min(int(h), default_horizon)
    return active if chosen_start >= 0 else min(int(curriculum[0][1]), default_horizon)


def prebound_magnitude_penalty(pre_bound_rate: Tensor, model_dt: float, target_delta: float) -> Tensor:
    """Hinge penalty on pre-bound per-step displacement above ``target_delta``.

    The V2 blocker is that pre-bound deltas sit ~120x above the cap (~12 vs 0.10),
    so the tanh cap is the dynamics engine.  This penalises only the EXCESS over
    ``target_delta`` (a one-sided hinge), so physically-scaled rates incur no cost
    while runaway pre-bound output is pushed down toward the physical scale.  It
    does not fight the residual within the physical band.
    """

    disp = (model_dt * pre_bound_rate).abs()
    excess = torch.clamp(disp - target_delta, min=0.0)
    return torch.mean(excess ** 2)


def _area_balance_penalty(phi: Tensor) -> Tensor:
    """Variance of per-phase area/volume fractions (optional anti-collapse penalty).

    Penalises one phase eating the domain by pushing area fractions toward
    uniform.  Conservative and OFF by default: too strong a weight biases
    legitimate coarsening, so this is a safety net, not the primary fix.
    """

    area = phi.flatten(2).sum(dim=-1)  # (B, N) — rank-agnostic
    frac = area / area.sum(dim=1, keepdim=True).clamp_min(1.0e-12)
    return torch.mean(frac.var(dim=1))


@torch.no_grad()
def write_rollout_snapshot(
    model: ExplicitMPFHybridRollout,
    phi0: Tensor,
    *,
    epoch: int,
    steps: int,
    out_dir: Path,
    keep_full_argmax: bool = True,
) -> Path:
    """Write a cheap eval-mode rollout snapshot for the out-of-process watcher.

    This runs ONE no-grad eval rollout (no backward, no optimizer) and stores a
    compact ``.npz``: per-step argmax labels (uint8) plus per-step scalar
    trajectories (per-phase area, bounds violation, sum error, projected-pixel
    fraction).  It is intentionally lightweight so live monitoring never reads
    the training graph or slows the optimizer.  Uses the model's OWN rollout
    only; no reference frames after ``t=0`` are touched here.
    """

    was_training = model.training
    model.eval()
    states = model.rollout(phi0, steps=max(steps, 1))  # (T+1, 1, N, *spatial)
    arr = states[:, 0].detach().cpu().numpy()  # (T+1, N, *spatial)
    if was_training:
        model.train()

    t_plus_1, n_phases, *spatial = arr.shape
    labels = arr.argmax(axis=1).astype(np.uint8)  # (T+1, *spatial)
    labels_flat = labels.reshape(t_plus_1, -1)  # (T+1, n_voxels)
    per_phase_area = (labels_flat[:, None, :] == np.arange(n_phases)[None, :, None]).sum(axis=2)
    sums = arr.sum(axis=1)  # (T+1, *spatial)
    sum_err = np.abs(sums - 1.0).reshape(t_plus_1, -1).max(axis=1)
    arr_flat = arr.reshape(t_plus_1, n_phases, -1)
    bounds = np.maximum(arr_flat.max(axis=(1, 2)) - 1.0, -arr_flat.min(axis=(1, 2)))
    active = (per_phase_area > 0).sum(axis=1)

    snap_dir = out_dir / "snapshots"
    snap_dir.mkdir(parents=True, exist_ok=True)
    path = snap_dir / f"rollout_ep{epoch:04d}.npz"
    payload: dict[str, Any] = {
        "epoch": np.int64(epoch),
        "steps": np.int64(steps),
        "num_phases": np.int64(n_phases),
        "per_phase_area": per_phase_area.astype(np.float32),
        "sum_error_per_step": sum_err.astype(np.float32),
        "bounds_per_step": bounds.astype(np.float32),
        "active_phase_count_per_step": active.astype(np.int32),
    }
    if keep_full_argmax:
        payload["argmax_labels"] = labels  # (T+1, H, W) uint8 — cheap
    np.savez_compressed(path, **payload)
    # Stable pointer to the newest snapshot for the watcher.
    (snap_dir / "latest.txt").write_text(path.name, encoding="utf-8")
    return path


def _jsonable(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    return value


def train_explicit_mpf(
    settings: dict[str, Any],
    *,
    repo_root: Path,
    launch: bool = False,
    mode: str = "full",
    device: str | torch.device = "cpu",
) -> ExplicitMPFTrainResult:
    """Train (or smoke-build) the explicit-MPF hybrid.

    Parameters
    ----------
    launch:
        Must be ``True`` to actually train and write checkpoints.  When ``False``
        the function only builds the model and validates the config (``mode`` is
        forced to ``"build"``), so no run can start by accident.
    mode:
        ``"full"`` trains for the configured epochs; ``"smoke"`` runs a couple of
        epochs at a tiny horizon and writes diagnostics but no checkpoint;
        ``"build"`` only constructs the model and returns.
    """

    assert_no_reference_leakage(settings)
    training_policy = settings.get("training_policy") or {}
    if launch and _bool_setting(training_policy.get("materiality_diagnostic_only", False)):
        raise ValueError(
            "materiality diagnostic configs are check-build/no-grad only; "
            "do not use launch, smoke, or training modes"
        )

    reference_cfg = (settings.get("reference") or {}).get("config")
    if not reference_cfg:
        raise ValueError("reference.config (path to the benchmark YAML) is required")
    reference = load_explicit_mpf_initial_reference(reference_cfg, repo_root=repo_root)

    dev = torch.device(device)
    model = build_explicit_mpf_model(settings, reference).to(dev)
    _cfg_proj = str((settings.get("model") or {}).get("projection_mode", "simplex_clip_normalize"))
    if model.projection_mode != _cfg_proj:
        raise RuntimeError(
            f"[explicit-mpf] PROJECTION MISMATCH: model.projection_mode={model.projection_mode!r} "
            f"!= config model.projection_mode={_cfg_proj!r}. "
            "This indicates a build bug in build_explicit_mpf_model."
        )
    run_id = str(settings.get("run_id") or settings.get("benchmark_id") or "explicit-mpf-run")
    architecture = str((settings.get("model") or {}).get("architecture"))
    warmstart_metadata = maybe_load_explicit_mpf_warmstart(
        model, settings, repo_root=repo_root, device=dev
    )
    print(
        "[explicit-mpf] MODEL "
        # `model.rate_parameterization` is a required trainer-facing metadata
        # contract attribute guaranteed present on every dispatched
        # model class -- read it directly instead of `getattr(..., "equivariant_tanh")`,
        # which was never a real contract value, only an invented defensive default.
        f"run_id={run_id} rate_parameterization={model.rate_parameterization} "
        f"delta_scale={model.delta_scale:.6g} graph_mode={model.graph_mode} "
        f"per_phase_gamma={model.per_phase_gamma} projection_mode={model.projection_mode}"
    )

    if not launch:
        mode = "build"

    if mode == "build":
        return ExplicitMPFTrainResult(
            run_id=run_id,
            architecture=architecture,
            model_dt=reference.model_dt,
            dt_mu_sigma=reference.dt_mu_sigma,
            num_phases=reference.num_phases,
            epochs_run=0,
            diagnostics_path=None,
            checkpoint_path=None,
            summary_path=None,
            final_diagnostics={"warmstart": warmstart_metadata} if warmstart_metadata else {},
            launched=False,
        )

    tr = settings.get("training") or {}
    epochs = int(tr.get("epochs", 50))
    horizon = int(settings.get("horizon", tr.get("horizon", 64)))
    tbptt_window = int(tr.get("tbptt_window", 8))
    lr = float(tr.get("learning_rate", 5.0e-5))
    review_every = int(tr.get("review_every", 25))
    lambda_ic = float(tr.get("lambda_ic", 1.0))
    lambda_mpf = float(tr.get("lambda_mpf", 1.0))
    lambda_proj = float(tr.get("lambda_projection", 0.0))  # default OFF (diagnostic)
    # Residual scaling + update-stability controls (the stabilization fix).
    mpf_norm = str(tr.get("mpf_residual_normalization", "raw"))
    mpf_eps = float(tr.get("mpf_residual_eps", 1.0e-8))
    grad_clip_norm = float(tr.get("grad_clip_norm", 0.0))  # 0 disables clipping
    lambda_area_balance = float(tr.get("lambda_area_balance", 0.0))  # OFF by default
    # V3 pre-bound scale controls (all OFF by default; existing configs unchanged).
    # lambda_prebound: penalises pre-bound per-step displacement that EXCEEDS the
    # cap (hinge on |model_dt*pre_bound_rate| - cap), pushing the model to emit
    # physically-scaled rates instead of relying on the tanh cap (the V2 blocker).
    lambda_prebound = float(tr.get("lambda_prebound_magnitude", 0.0))
    prebound_target = float(tr.get("prebound_target_delta", 0.10))  # cap by default
    # lambda_convlstm: L2 penalty on the ConvLSTM branch rate (it dominated ANN
    # ~13x at V2 ep125 and is the source of the large pre-bound output).
    lambda_convlstm = float(tr.get("lambda_convlstm_magnitude", 0.0))
    # AdamW-style weight decay applied only to the ANN/ConvLSTM rate-head
    # weights (ann.head.weight, recurrent_head.weight) -- not biases, not the
    # ConvLSTM cell -- to shrink the head magnitude that drives the large pre-bound
    # output.  0 disables (plain Adam over all params).
    rate_head_weight_decay = float(tr.get("rate_head_weight_decay", 0.0))
    # Mixed-residual parameters include an absolute displacement floor so late
    # wrong near-static states stay penalised.  Only used when mpf_norm == "mixed".
    mpf_abs_scale = float(tr.get("mpf_abs_scale", 0.05))
    mpf_mixed_w_rel = float(tr.get("mpf_mixed_w_rel", 1.0))
    mpf_mixed_w_abs = float(tr.get("mpf_mixed_w_abs", 1.0))
    # Optional interface weighting of the mixed residual. When disabled, behavior
    # is identical to the unweighted mixed residual.
    # Only active when enabled AND mpf_norm == "mixed".  Config:
    #   interface_local_weighting: {enabled, source: model_phi_interface_mask,
    #                               interface_eps, weight}
    ilw_cfg = tr.get("interface_local_weighting") or {}
    ilw_enabled = bool(ilw_cfg.get("enabled", False))
    ilw_eps = float(ilw_cfg.get("interface_eps", 0.05))
    ilw_weight = float(ilw_cfg.get("weight", 2.0))
    ilw_source = str(ilw_cfg.get("source", "model_phi_interface_mask"))
    if ilw_enabled:
        if ilw_source != "model_phi_interface_mask":
            raise ValueError(
                "interface_local_weighting.source must be 'model_phi_interface_mask' "
                "(model-phi only; no post-t0 reference frames)"
            )
        if mpf_norm != "mixed":
            raise ValueError(
                "interface_local_weighting requires mpf_residual_normalization: mixed"
            )
    # Energy-LH2: opt-in local/interface MPF energy-density residual.  OFF by
    # default; when disabled or lambda=0 the term is never constructed.
    energy_lh2_cfg = tr.get("energy_lh2") or {}
    energy_lh2_enabled = bool(energy_lh2_cfg.get("enabled", False))
    energy_lh2_lambda = float(energy_lh2_cfg.get("lambda_energy_density", 0.0))
    energy_lh2_eps = float(energy_lh2_cfg.get("interface_eps", 0.05))
    energy_lh2_tol = float(energy_lh2_cfg.get("tol", 0.0))
    energy_lh2_source = str(energy_lh2_cfg.get("source", "model_phi_only"))
    energy_lh2_norm_cfg = energy_lh2_cfg.get("normalization") or {}
    energy_lh2_norm_enabled = bool(energy_lh2_norm_cfg.get("enabled", False))
    energy_lh2_raw_scale = float(energy_lh2_norm_cfg.get("raw_scale", 1.0))
    energy_lh2_norm_eps = float(energy_lh2_norm_cfg.get("eps", 1.0e-12))
    if energy_lh2_enabled:
        if energy_lh2_source != "model_phi_only":
            raise ValueError("training.energy_lh2.source must be model_phi_only")
        if energy_lh2_lambda < 0.0:
            raise ValueError("training.energy_lh2.lambda_energy_density must be non-negative")
        if energy_lh2_eps < 0.0:
            raise ValueError("training.energy_lh2.interface_eps must be non-negative")
        if energy_lh2_norm_enabled:
            norm_mode_check = str(energy_lh2_norm_cfg.get("mode", "fixed_raw_scale"))
            if norm_mode_check != "fixed_raw_scale":
                raise ValueError(
                    "training.energy_lh2.normalization.mode must be fixed_raw_scale"
                )
            if energy_lh2_raw_scale <= 0.0:
                raise ValueError(
                    "training.energy_lh2.normalization.raw_scale must be positive"
                )
            if energy_lh2_norm_eps <= 0.0:
                raise ValueError(
                    "training.energy_lh2.normalization.eps must be positive"
                )
    energy_lh2_active = energy_lh2_enabled and energy_lh2_lambda > 0.0
    # Surface descent-consistency (windowed, symmetric S_grad).  OFF by default;
    # when disabled or lambda=0 the term is never constructed -> byte-identical to
    # prior behaviour.  Anchors the model's relative S_grad reduction across each
    # TBPTT window to the explicit-MPF RHS rolled the same #steps from the window
    # start (model-phi only; oracle rollout detached; no post-t0 reference).  A
    # runtime cap clamps lambda*L_surface to <= surface_cap_frac * window-mean MPF
    # residual so the gentle surface term can never dominate the base physics loss.
    surface_cfg = tr.get("surface_descent") or {}
    surface_enabled = bool(surface_cfg.get("enabled", False))
    surface_lambda = float(surface_cfg.get("lambda_surface", 0.0))
    surface_huber_delta = float(surface_cfg.get("huber_delta", 0.02))
    surface_cap_frac = float(surface_cfg.get("cap_fraction", 0.15))
    surface_warmup_epochs = int(surface_cfg.get("warmup_epochs", 0))
    surface_source = str(surface_cfg.get("source", "model_phi_only"))
    if surface_enabled:
        if surface_source != "model_phi_only":
            raise ValueError("training.surface_descent.source must be model_phi_only")
        if surface_lambda < 0.0:
            raise ValueError("training.surface_descent.lambda_surface must be non-negative")
        if not (0.0 < surface_cap_frac <= 1.0):
            raise ValueError("training.surface_descent.cap_fraction must be in (0, 1]")
        if surface_huber_delta <= 0.0:
            raise ValueError("training.surface_descent.huber_delta must be positive")
    surface_active = surface_enabled and surface_lambda > 0.0
    # Optional horizon curriculum (stabilized-v2): grow the rollout horizon over
    # epoch milestones so the optimizer first fixes the steps 3-10 jump at short
    # horizon.  Format: list of [epoch_start, horizon] pairs; default = none.
    horizon_curriculum = tr.get("horizon_curriculum") or []
    # Optional LR warm-up over the first N epochs (linear from 0 -> lr).
    lr_warmup_epochs = int(tr.get("lr_warmup_epochs", 0))
    # Optional post-warmup LR schedule (cosine decay) to damp the grad-norm
    # spikes seen when the rollout horizon lengthens.  Config:
    #   lr_schedule: {type: cosine, start_after_epoch: N, lr_min: x}
    # OFF unless `type == cosine`; the schedule applies only to epochs at or after
    # start_after_epoch, decaying from `lr` to `lr_min` over the remaining epochs.
    lr_schedule_cfg = tr.get("lr_schedule") or {}
    lr_schedule_type = str(lr_schedule_cfg.get("type", "none"))
    lr_schedule_start = int(lr_schedule_cfg.get("start_after_epoch", lr_warmup_epochs))
    lr_schedule_min = float(lr_schedule_cfg.get("lr_min", lr))
    # Hold the curriculum horizon while the latest cap saturation exceeds the
    # latest cap_saturation_fraction exceeds this threshold (0 disables the gate).
    curriculum_cap_gate = float(tr.get("curriculum_cap_saturation_gate", 0.0))
    # Save a checkpoint at every review point when requested, so a
    # later best-physical selector can pick the best rollout rather than the last.
    checkpoint_every_review = bool(tr.get("checkpoint_every_review", False))
    keep_all_review_checkpoints = bool(tr.get("keep_all_review_checkpoints", checkpoint_every_review))
    # Warn without stopping when projection_correction_l1 crosses this threshold,
    # well below the early-stop kill so projection-reliance creep is visible early.
    projection_warn_l1 = float(tr.get("projection_warn_threshold_l1", 0.0))
    # Optional automated early-stop gates (config-driven, OFF unless enabled).
    early_stop_cfg = tr.get("early_stop") or {}
    # The two-signal model-only health check is disabled unless
    # training.health_gate.enabled=true. All
    # thresholds/connectivity/review-step/gate-active-epoch live in the config; see
    # evaluate_two_signal_health_gate / compute_health_gate_metrics.
    health_gate_cfg = tr.get("health_gate") or {}
    health_gate_enabled = bool(health_gate_cfg.get("enabled", False))
    health_gate_review_step = int(health_gate_cfg.get("review_step", 4000))
    health_gate_connectivity = str(health_gate_cfg.get("connectivity", "4"))
    # Live-monitoring snapshots: cheap eval rollout written at review cadence for
    # the out-of-process watcher.  snapshot_steps may extend past the training
    # horizon to expose extrapolation-toward-end-of-domain behaviour.
    snapshots_enabled = bool(tr.get("monitor_snapshots", True))
    snapshot_every = int(tr.get("snapshot_every", review_every))
    snapshot_steps = int(tr.get("snapshot_steps", max(horizon * 2, horizon)))
    # A longer rollout only at the final epoch so periodic snapshots stay cheap
    # but the definitive pred-vs-reference comparison spans the full trajectory.
    snapshot_steps_final = int(tr.get("snapshot_steps_final", snapshot_steps))
    # Optional phase-permutation augmentation (default off).
    # Uses a SEPARATE, hash-frozen permutation ledger (epoch -> perm) generated BEFORE the
    # run; the trainer only READS it, so it cannot perturb model-init or any other RNG stream.
    # Sound only with phase_id_encoding=zero_masked (permutation-equivariant ANN input).
    aug_cfg = tr.get("phase_permutation_augmentation") or {}
    aug_enabled = bool(aug_cfg.get("enabled", False))
    aug_perms: dict[int, Tensor] = {}
    if aug_enabled:
        import hashlib as _hashlib
        _ledger = Path(str(aug_cfg["ledger_path"]))
        if not _ledger.is_absolute():
            _ledger = repo_root / _ledger
        _raw = _ledger.read_bytes()
        _want_sha = aug_cfg.get("ledger_sha256")
        if _want_sha and _hashlib.sha256(_raw).hexdigest() != str(_want_sha):
            raise ValueError("phase_permutation_augmentation ledger SHA256 mismatch")
        for _line in _raw.decode("utf-8").splitlines():
            if _line.strip():
                _rec = json.loads(_line)
                aug_perms[int(_rec["epoch"])] = torch.tensor(_rec["perm"], dtype=torch.long)
    # Optional frozen-parameter deployment evaluation (default off, no gradients).
    fta_cfg = tr.get("frozen_theta_audit") or {}
    fta_enabled = bool(fta_cfg.get("enabled", False))
    fta_every = int(fta_cfg.get("every", review_every))
    fta_steps = int(fta_cfg.get("steps", horizon))
    # Optional optimizer/scheduler state in review checkpoints. Additive I/O only; never touches
    # the training/update math or any diagnostic quantity.
    save_optimizer_state_in_review_ckpts = bool(
        tr.get("save_optimizer_state_in_review_ckpts", False)
    )
    # Optional per-TBPTT-window convergence instrumentation. When disabled, no
    # extra tensors are read and no extra file is written. It collects, for each
    # TBPTT window WITHIN A REVIEW EPOCH ONLY (to bound overhead), the pre-clip grad-norm
    # (the value `torch.nn.utils.clip_grad_norm_` already returns BEFORE it clips -- reused,
    # not recomputed), whether that window's grad was actually clipped, and the post-step
    # parameter-update L2 norm (read-only: params are cloned before `optimizer.step()` and
    # diffed after -- this never mutates the optimizer's own update). Quantiles + clip
    # fraction + LR are written once per review epoch to `convergence_instrumentation.jsonl`.
    conv_instr_cfg = tr.get("convergence_instrumentation") or {}
    conv_instr_enabled = bool(conv_instr_cfg.get("enabled", False))
    # Optional convergence-bank label-free residual (default off).
    # Loads + SHA256-verifies a frozen bank of phase-permutation transforms (generated and
    # hashed BEFORE this run) and, at each review epoch, applies each bank permutation to the
    # model's OWN current-rollout anchor state (the window-start state reached by that epoch's
    # own training rollout -- never a post-t0 reference frame), takes one no-grad model step,
    # and measures the same mixed MPF residual used in training. Purely diagnostic: no-grad,
    # read-only, cannot perturb training; refuses to proceed on a SHA mismatch (same
    # fail-closed pattern as the phase-permutation augmentation ledger above).
    cbank_cfg = tr.get("convergence_bank") or {}
    cbank_enabled = bool(cbank_cfg.get("enabled", False))
    cbank_permutations: list[list[int]] = []
    if cbank_enabled:
        import hashlib as _hashlib_cbank

        _bank_path = Path(str(cbank_cfg["path"]))
        if not _bank_path.is_absolute():
            _bank_path = repo_root / _bank_path
        _bank_raw = _bank_path.read_bytes()
        _bank_want_sha = cbank_cfg.get("sha256")
        if _bank_want_sha and _hashlib_cbank.sha256(_bank_raw).hexdigest() != str(_bank_want_sha):
            raise ValueError("convergence_bank SHA256 mismatch")
        _bank_payload = json.loads(_bank_raw.decode("utf-8"))
        if int(_bank_payload.get("num_phases", -1)) != reference.num_phases:
            raise ValueError(
                "convergence_bank num_phases mismatch: "
                f"bank={_bank_payload.get('num_phases')} reference={reference.num_phases}"
            )
        cbank_permutations = [list(p) for p in _bank_payload["permutations"]]
    if mode == "smoke":
        epochs = min(epochs, 2)
        horizon = min(horizon, 4)
        tbptt_window = min(tbptt_window, 2)
        review_every = 1
        snapshot_every = 1
        snapshot_steps = min(snapshot_steps, 6)
        snapshot_steps_final = min(snapshot_steps_final, 6)
        # Keep the health-check rollout cheap in smoke mode. Its fixed review
        # step needs a separate clamp from the training horizon.
        health_gate_review_step = min(health_gate_review_step, 6)

    out_dir = repo_root / Path(settings.get("artifacts", {}).get("output_directory", f"artifacts/benchmarks/explicit_mpf/{run_id}"))
    out_dir.mkdir(parents=True, exist_ok=True)
    diagnostics_path = out_dir / "diagnostics.jsonl"
    summary_path = out_dir / "summary.json"
    checkpoint_path = out_dir / "checkpoint.pt"
    conv_instr_path = (
        out_dir / str(conv_instr_cfg.get("jsonl_name", "convergence_instrumentation.jsonl"))
        if conv_instr_enabled
        else None
    )
    cbank_path = (
        out_dir / str(cbank_cfg.get("jsonl_name", "convergence_bank_residual.jsonl"))
        if cbank_enabled
        else None
    )

    multi_ic_cfg = tr.get("multi_ic_batch") or None
    if multi_ic_cfg:
        # Replace the single initial field with a fixed simultaneous batch of
        # hash-verified frozen initial
        # fields. The reference adapter above is then used for geometry/N
        # metadata only; its field never enters training.
        from pinn_phase.training.multi_ic_t0_batch import load_multi_ic_t0_batch

        _batch_np = load_multi_ic_t0_batch(
            multi_ic_cfg,
            num_phases=reference.num_phases,
            spatial_shape=tuple(reference.phi0.shape[1:]),
            repo_root=repo_root,
        )
        phi0 = torch.tensor(_batch_np, dtype=torch.float32, device=dev)
    else:
        phi0 = torch.tensor(reference.phi0, dtype=torch.float32, device=dev).unsqueeze(0)
    if model.projection_mode == "soft_threshold_eps1e3":
        phi0, _ = project_simplex_soft_threshold(phi0, threshold_eps=1.0e-3)
    else:
        phi0, _ = project_simplex(phi0)  # ensure exact simplex start

    if rate_head_weight_decay > 0.0:
        # Decay ONLY the rate-head weight matrices; everything else (biases, the
        # ConvLSTM cell, blend logit) stays decay-free.  AdamW applies decoupled
        # weight decay, so this directly shrinks the head magnitude.
        head_weight_names = {"ann.head.weight", "recurrent_head.weight"}
        decay_params, other_params = [], []
        for name, param in model.named_parameters():
            (decay_params if name in head_weight_names else other_params).append(param)
        optimizer = torch.optim.AdamW(
            [
                {"params": decay_params, "weight_decay": rate_head_weight_decay},
                {"params": other_params, "weight_decay": 0.0},
            ],
            lr=lr,
        )
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    # This trainer schedules the LR manually per-epoch (see `epoch_lr` below) rather than via a
    # torch.optim.lr_scheduler object; `scheduler` stays None so the optimizer/scheduler
    # checkpoint block below can check "if a scheduler exists" generically without assuming one.
    scheduler = None

    diag_rows: list[dict[str, Any]] = []
    if diagnostics_path.exists():
        diagnostics_path.unlink()

    # State for the cap-gated curriculum: the highest horizon reached so far and
    # the most recent observed cap saturation.  The gate prevents advancing the
    # curriculum while saturation is still high.
    curriculum_horizon_cap = 0
    last_cap_saturation = 0.0

    last_diag: dict[str, Any] = {}
    for epoch in range(epochs):
        model.train()
        # Optional LR warm-up (linear 0 -> lr over the first lr_warmup_epochs),
        # then an optional cosine decay from lr -> lr_min over the epochs at
        # or after lr_schedule_start.  Warm-up always takes precedence while active.
        if lr_warmup_epochs > 0 and epoch < lr_warmup_epochs:
            epoch_lr = lr * float(epoch + 1) / float(lr_warmup_epochs)
        elif lr_schedule_type == "cosine" and epoch >= lr_schedule_start:
            span = max(epochs - 1 - lr_schedule_start, 1)
            progress = min(max(epoch - lr_schedule_start, 0) / span, 1.0)
            epoch_lr = lr_schedule_min + 0.5 * (lr - lr_schedule_min) * (
                1.0 + math.cos(math.pi * progress)
            )
        else:
            epoch_lr = lr
        for pg in optimizer.param_groups:
            pg["lr"] = epoch_lr
        # Optional horizon curriculum: shorter rollout early so the optimizer first
        # fixes the steps 3-10 jump before training the long-horizon trajectory.
        epoch_horizon = resolve_curriculum_horizon(horizon_curriculum, epoch, horizon)
        # Cap-saturation gate: don't advance past the highest horizon reached while
        # saturation is high; hold the previous horizon until saturation drops.
        if curriculum_cap_gate > 0.0 and horizon_curriculum:
            if epoch_horizon > curriculum_horizon_cap and last_cap_saturation > curriculum_cap_gate:
                epoch_horizon = max(curriculum_horizon_cap, 1)
            else:
                curriculum_horizon_cap = max(curriculum_horizon_cap, epoch_horizon)
        # IC consistency loss: at t=0 the model's first rate should match the
        # MPF physics target (initial-condition-anchored residual).
        if aug_enabled and epoch in aug_perms:
            phi = phi0[:, aug_perms[epoch].to(phi0.device)].detach()
        else:
            phi = phi0.detach()
        state = model.initial_state(phi)
        previous_phi: Tensor | None = None
        window_start_phi = phi  # detached state at the start of the current TBPTT window
        # Collect per-window instrumentation only during review epochs to bound the
        # read-only param-clone overhead to review points); no-op / empty lists when disabled.
        is_review_epoch = (epoch % review_every == 0) or (epoch == epochs - 1)
        epoch_conv_pre_clip_norms: list[float] = []
        epoch_conv_update_norms: list[float] = []

        epoch_mpf = 0.0
        epoch_proj = 0.0
        epoch_energy_lh2_raw = 0.0
        epoch_energy_lh2_weighted = 0.0
        epoch_energy_lh2_normalized = 0.0
        epoch_energy_lh2_mean_delta = 0.0
        epoch_energy_lh2_positive_fraction = 0.0
        epoch_energy_lh2_interface_fraction = 0.0
        window_mpf = 0.0  # weighted MPF residual accumulated over the current window
        epoch_surface_loss = 0.0
        epoch_surface_contrib = 0.0
        epoch_surface_windows = 0
        window_loss = torch.zeros((), device=dev)
        n_window = 0
        step_diag: dict[str, Any] = {}
        last_grad_norm = 0.0
        optimizer.zero_grad(set_to_none=True)
        for step in range(epoch_horizon):
            rate, state, step_diag = model.predict_rate(phi, state, previous_phi=previous_phi)
            target = explicit_mpf_rhs(phi, eta_px=model.eta_px, mu=model.mu, sigma=model.sigma)
            if ilw_enabled:
                # Interface-weighted mixed residual. The mask is detached and
                # computed from the current phi only (no reference).  Reduces to the
                # uniform mixed residual when the mask is all-ones.
                ilw_mask = interface_local_weight_mask(
                    phi, interface_eps=ilw_eps, interface_weight=ilw_weight
                )
                mpf_loss = interface_local_residual_loss(
                    rate, target, phi, model_dt=model.model_dt, weight_mask=ilw_mask,
                    eps=mpf_eps, abs_scale=mpf_abs_scale,
                    mixed_w_rel=mpf_mixed_w_rel, mixed_w_abs=mpf_mixed_w_abs,
                )
            else:
                mpf_loss = mpf_residual_loss(
                    rate, target, model_dt=model.model_dt, normalization=mpf_norm, eps=mpf_eps,
                    abs_scale=mpf_abs_scale, mixed_w_rel=mpf_mixed_w_rel, mixed_w_abs=mpf_mixed_w_abs,
                )
            raw_next = phi + model.model_dt * rate
            proj_loss = _phase_sum_consistency(raw_next)
            if model.projection_mode == "soft_threshold_eps1e3":
                next_phi, proj_d = project_simplex_soft_threshold(raw_next, threshold_eps=1.0e-3)
            else:
                next_phi, proj_d = project_simplex(raw_next)
            ic_weight = lambda_ic if step == 0 else 0.0
            step_loss = lambda_mpf * mpf_loss + ic_weight * mpf_loss + lambda_proj * proj_loss
            if lambda_area_balance > 0.0:
                step_loss = step_loss + lambda_area_balance * _area_balance_penalty(raw_next)
            if energy_lh2_active:
                energy_raw, energy_weighted, energy_diag = energy_lh2_density_loss(
                    phi,
                    next_phi,
                    eta_px=model.eta_px,
                    sigma=model.sigma,
                    lambda_energy_density=energy_lh2_lambda,
                    interface_eps=energy_lh2_eps,
                    tol=energy_lh2_tol,
                    normalization=energy_lh2_norm_cfg if energy_lh2_norm_enabled else None,
                )
                step_loss = step_loss + energy_weighted
                epoch_energy_lh2_raw += float(energy_raw.detach())
                epoch_energy_lh2_weighted += float(energy_weighted.detach())
                if "energy_lh2_loss_normalized" in energy_diag:
                    epoch_energy_lh2_normalized += float(
                        energy_diag["energy_lh2_loss_normalized"]
                    )
                epoch_energy_lh2_mean_delta += energy_diag["energy_lh2_mean_delta"]
                epoch_energy_lh2_positive_fraction += energy_diag["energy_lh2_positive_fraction"]
                epoch_energy_lh2_interface_fraction += energy_diag["energy_lh2_interface_fraction"]
                step_diag.update(energy_diag)
            # V3 pre-bound scale regularizers (use the grad-carrying tensors that
            # predict_rate stashed under _-prefixed keys).
            if lambda_prebound > 0.0 and "_pre_bound_rate" in step_diag:
                step_loss = step_loss + lambda_prebound * prebound_magnitude_penalty(
                    step_diag["_pre_bound_rate"], model.model_dt, prebound_target
                )
            if lambda_convlstm > 0.0 and "_rnn_rate" in step_diag:
                step_loss = step_loss + lambda_convlstm * torch.mean(step_diag["_rnn_rate"] ** 2)
            window_loss = window_loss + step_loss
            n_window += 1
            epoch_mpf += float(mpf_loss.detach())
            window_mpf += float((lambda_mpf * mpf_loss).detach())
            epoch_proj += float(proj_loss.detach())

            # Scale diagnostics that exposed the failure (logged at review points).
            with torch.no_grad():
                step_diag["target_rate_abs_max"] = float(target.abs().max())
                step_diag["pred_rate_abs_max"] = float(rate.abs().max())
                step_diag["rate_to_target_ratio"] = float(
                    rate.abs().max() / target.abs().max().clamp_min(1.0e-30)
                )
                d_target = model.model_dt * target
                d_pred = model.model_dt * rate
                step_diag["target_delta_abs_max"] = float(d_target.abs().max())
                step_diag["pred_delta_abs_max"] = float(d_pred.abs().max())
                step_diag["delta_to_target_ratio"] = float(
                    d_pred.abs().max() / d_target.abs().max().clamp_min(1.0e-30)
                )
                step_diag["raw_next_min"] = float(raw_next.min())
                step_diag["raw_next_max"] = float(raw_next.max())
                # Displacement-magnitude distribution (learned vs physical target) and
                # direction agreement.  These are the recalibrated anti-suppression
                # signals: argmax-change is near-zero at short horizons even for the
                # ground-truth oracle, so under-/over-driving must be judged on the
                # rate magnitude + direction, not on argmax motion.
                dp = d_pred.abs().flatten()
                dtg = d_target.abs().flatten()
                step_diag["pred_delta_mean"] = float(dp.mean())
                step_diag["pred_delta_p50"] = float(dp.median())
                step_diag["pred_delta_p95"] = float(torch.quantile(dp, 0.95))
                step_diag["pred_delta_p99"] = float(torch.quantile(dp, 0.99))
                step_diag["target_delta_mean"] = float(dtg.mean())
                step_diag["target_delta_p50"] = float(dtg.median())
                step_diag["target_delta_p95"] = float(torch.quantile(dtg, 0.95))
                step_diag["target_delta_p99"] = float(torch.quantile(dtg, 0.99))
                step_diag["learned_target_magnitude_ratio"] = float(
                    dp.mean() / dtg.mean().clamp_min(1.0e-30)
                )
                step_diag["displacement_cosine_to_target"] = float(
                    torch.nn.functional.cosine_similarity(
                        d_pred.flatten(), d_target.flatten(), dim=0
                    )
                )
                step_diag["displacement_relative_residual"] = float(
                    ((d_pred - d_target) ** 2).mean() / ((d_target ** 2).mean() + 1.0e-30)
                )
                step_diag["displacement_sign_agreement"] = float(
                    (torch.sign(d_pred) == torch.sign(d_target)).float().mean()
                )
                # Locality diagnostics: localized max-delta ratio restricted to
                # INTERFACE pixels (the confirmed deficit is "mean OK, localized max too
                # small").  Computed from the model's own phi mask; logged for every run
                # (mask threshold uses ilw_eps; weight is irrelevant here).  This lets us
                # verify whether the intervention raises the interface localized max-delta
                # WITHOUT raising cap saturation / projection reliance.
                imask = interface_local_weight_mask(
                    phi, interface_eps=ilw_eps, interface_weight=2.0
                ) > 1.5  # bool [B,1,H,W]: interface pixels
                imask_b = imask.expand_as(d_pred)
                if imask_b.any():
                    step_diag["interface_pred_delta_max"] = float(d_pred.abs()[imask_b].max())
                    step_diag["interface_target_delta_max"] = float(d_target.abs()[imask_b].max())
                    step_diag["interface_localized_maxdelta_ratio"] = float(
                        d_pred.abs()[imask_b].max()
                        / d_target.abs()[imask_b].max().clamp_min(1.0e-30)
                    )
                    step_diag["interface_pixel_fraction"] = float(imask.float().mean())
                else:
                    step_diag["interface_localized_maxdelta_ratio"] = 0.0
                    step_diag["interface_pixel_fraction"] = 0.0
                step_diag["interface_local_weighting_enabled"] = bool(ilw_enabled)
                # Cap-saturation proxy: fraction of bounded deltas pinned near the
                # cap (only meaningful when the head is bounded).  High + sustained
                # => the cap is acting as the dynamics engine.
                if getattr(model, "rate_head_bounded", False):
                    cap = model.max_delta_phi_per_step
                    sat = (d_pred.abs() > 0.95 * cap).float().mean()
                    step_diag["cap_saturation_fraction"] = float(sat)

            for k, v in proj_d.items():
                step_diag[k] = v

            # Anti-suppression diagnostic (V5): fraction of pixels whose argmax phase
            # differs from the t=0 field.  ~0 means the field is frozen / zero-rate
            # collapsed (V5's characteristic failure if the bound is too tight).
            with torch.no_grad():
                step_diag["argmax_change_fraction_vs_t0"] = float(
                    (next_phi.argmax(dim=1) != phi0.argmax(dim=1)).float().mean()
                )

            if n_window >= tbptt_window or step == epoch_horizon - 1:
                backward_loss = window_loss / max(n_window, 1)
                # Windowed symmetric S_grad surface descent-consistency term
                # (default-off; model-phi only; oracle rollout detached).  Runtime
                # cap clamps lambda*L_surface to <= surface_cap_frac * window-mean
                # MPF residual so the gentle surface term can never dominate.
                if surface_active and epoch >= surface_warmup_epochs:
                    surf_loss, surf_diag = surface_descent_consistency_loss(
                        window_start_phi,
                        next_phi,
                        n_window,
                        model_dt=model.model_dt,
                        eta_px=model.eta_px,
                        mu=model.mu,
                        sigma=model.sigma,
                        projection_mode=model.projection_mode,
                        huber_delta=surface_huber_delta,
                    )
                    surf_contrib = surface_lambda * surf_loss
                    mpf_window_mean = window_mpf / max(n_window, 1)
                    cap_val = surface_cap_frac * mpf_window_mean
                    contrib_val = float(surf_contrib.detach())
                    capped = False
                    if contrib_val > cap_val and contrib_val > 0.0:
                        surf_contrib = surf_contrib * (cap_val / contrib_val)
                        capped = True
                    backward_loss = backward_loss + surf_contrib
                    epoch_surface_loss += surf_diag["surface_loss"]
                    epoch_surface_contrib += float(surf_contrib.detach())
                    epoch_surface_windows += 1
                    surf_diag["surface_contrib"] = float(surf_contrib.detach())
                    surf_diag["surface_mpf_window_mean"] = float(mpf_window_mean)
                    surf_diag["surface_contrib_over_mpf"] = float(surf_contrib.detach()) / (
                        float(mpf_window_mean) + 1.0e-30
                    )
                    surf_diag["surface_capped"] = capped
                    step_diag.update(surf_diag)
                backward_loss.backward()
                if grad_clip_norm > 0.0:
                    last_grad_norm = float(
                        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
                    )
                else:
                    last_grad_norm = float(
                        torch.nn.utils.clip_grad_norm_(model.parameters(), float("inf"))
                    )
                # Per-window convergence instrumentation (default off).
                # `last_grad_norm` above is already the PRE-CLIP total norm --
                # `clip_grad_norm_` computes it before clipping in place, so this is a
                # pure read of an already-computed value, not a new computation. The
                # param snapshot is a read-only clone taken before `optimizer.step()`.
                _conv_instr_active = conv_instr_enabled and is_review_epoch
                if _conv_instr_active:
                    epoch_conv_pre_clip_norms.append(last_grad_norm)
                    _pre_step_snapshot = [p.detach().clone() for p in model.parameters()]
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                if _conv_instr_active:
                    _update_sq = 0.0
                    for _p, _pre in zip(model.parameters(), _pre_step_snapshot):
                        _update_sq += float(((_p.detach() - _pre) ** 2).sum())
                    epoch_conv_update_norms.append(_update_sq ** 0.5)
                    del _pre_step_snapshot
                window_loss = torch.zeros((), device=dev)
                n_window = 0
                window_mpf = 0.0
                state = _detach_state(state)
                next_phi = next_phi.detach()
                window_start_phi = next_phi  # next window starts from this detached state

            previous_phi = phi.detach()
            phi = next_phi
        if energy_lh2_active:
            energy_steps = max(epoch_horizon, 1)
            step_diag["energy_lh2_loss_raw"] = epoch_energy_lh2_raw / energy_steps
            step_diag["energy_lh2_loss_weighted"] = epoch_energy_lh2_weighted / energy_steps
            step_diag["energy_lh2_mean_delta"] = epoch_energy_lh2_mean_delta / energy_steps
            step_diag["energy_lh2_positive_fraction"] = (
                epoch_energy_lh2_positive_fraction / energy_steps
            )
            step_diag["energy_lh2_interface_fraction"] = (
                epoch_energy_lh2_interface_fraction / energy_steps
            )
            step_diag["energy_lh2_lambda"] = energy_lh2_lambda
            if energy_lh2_norm_enabled:
                step_diag["energy_lh2_loss_normalized"] = (
                    epoch_energy_lh2_normalized / energy_steps
                )
                step_diag["energy_lh2_norm_mode"] = "fixed_raw_scale"
                step_diag["energy_lh2_raw_scale"] = max(
                    energy_lh2_raw_scale, energy_lh2_norm_eps
                )
        step_diag["energy_lh2_enabled"] = bool(energy_lh2_enabled)
        step_diag["surface_descent_enabled"] = bool(surface_active)
        if surface_active and epoch_surface_windows > 0:
            step_diag["surface_loss_epoch_mean"] = epoch_surface_loss / epoch_surface_windows
            step_diag["surface_contrib_epoch_mean"] = epoch_surface_contrib / epoch_surface_windows
            step_diag["surface_lambda"] = surface_lambda
            step_diag["surface_cap_fraction"] = surface_cap_frac
        step_diag["no_reference_frames_after_t0_used"] = True
        step_diag["grad_norm"] = last_grad_norm
        last_cap_saturation = float(step_diag.get("cap_saturation_fraction", last_cap_saturation))

        if (epoch % review_every == 0) or (epoch == epochs - 1):
            row = _review_row(epoch, epoch_horizon, epoch_mpf, epoch_proj, step_diag, model)

            # The two-signal model-only health check fails closed. See
            # `evaluate_health_gate_review`. It is computed before the row is
            # persisted so the metrics/status land in diagnostics.jsonl for this
            # review epoch. ZERO reference frames after t=0 are read -- only the
            # model's own phi0 and its own forward/projection.
            health_should_stop, health_stop_reasons, health_row_updates = evaluate_health_gate_review(
                model,
                phi0,
                epoch=epoch,
                enabled=health_gate_enabled,
                review_step=health_gate_review_step,
                connectivity=health_gate_connectivity,
                cfg=health_gate_cfg,
            )
            row.update(health_row_updates)
            if health_row_updates.get("health_gate_status") == "error":
                print(
                    f"[explicit-mpf] HARD STOP: enabled health gate computation at "
                    f"epoch {epoch} raised an exception -- failing CLOSED (never open): "
                    f"{health_row_updates.get('health_gate_error')}"
                )

            diag_rows.append(row)
            last_diag = row
            with diagnostics_path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps({k: _jsonable(v) for k, v in row.items()}) + "\n")
            print(
                f"[explicit-mpf] epoch {epoch+1}/{epochs} "
                f"loss_mpf={row['loss_mpf_residual']:.4e} gamma={row['gamma']:.3f} "
                f"pred_delta_max={row['pred_delta_abs_max']:.3f} "
                f"delta_ratio={row['delta_to_target_ratio']:.2f} "
                f"raw_next=[{row['raw_next_min']:.2f},{row['raw_next_max']:.2f}] "
                f"grad_norm={row['grad_norm']:.2e} "
                f"sum_err_post={row['sum_error_post_projection']:.2e} "
                f"active_phases={row['active_phase_count']}"
            )

            # Thresholded projection-reliance warning (advisory; does not stop).
            if projection_warn_l1 > 0.0:
                proj_l1 = float(row.get("projection_correction_l1", 0.0))
                if proj_l1 > projection_warn_l1:
                    print(
                        f"[explicit-mpf] PROJECTION WARNING epoch {epoch+1}: "
                        f"projection_correction_l1={proj_l1:.3e} > "
                        f"threshold {projection_warn_l1:.3e} (reliance creep)"
                    )

            # Per-review checkpoint so a selector can later choose
            # the best rollout instead of assuming the final epoch is best.
            if mode != "smoke" and checkpoint_every_review:
                stage_dir = out_dir / "review_checkpoints"
                stage_dir.mkdir(parents=True, exist_ok=True)
                stage_ckpt = stage_dir / f"checkpoint_ep{epoch:04d}.pt"
                stage_ckpt_payload: dict[str, Any] = {
                    "model_state": model.state_dict(),
                    "run_id": run_id,
                    "epoch": epoch,
                    "horizon": epoch_horizon,
                    "lr": epoch_lr,
                    "model_dt": reference.model_dt,
                    "dt_mu_sigma": reference.dt_mu_sigma,
                    "num_phases": reference.num_phases,
                }
                # Optionally add optimizer/scheduler state. When disabled, the
                # payload above is exactly what C1 saved, key-for-key, in the same order).
                if save_optimizer_state_in_review_ckpts:
                    stage_ckpt_payload["optimizer_state"] = optimizer.state_dict()
                    if scheduler is not None:
                        stage_ckpt_payload["scheduler_state"] = scheduler.state_dict()
                torch.save(stage_ckpt_payload, stage_ckpt)
                if not keep_all_review_checkpoints:
                    for old in sorted(stage_dir.glob("checkpoint_ep*.pt"))[:-1]:
                        old.unlink()

            # Per-review-epoch convergence-instrumentation summary. When disabled,
            # conv_instr_path is None and this block is skipped entirely).
            if conv_instr_enabled and conv_instr_path is not None:
                _gn = np.array(epoch_conv_pre_clip_norms, dtype=np.float64)
                _un = np.array(epoch_conv_update_norms, dtype=np.float64)
                _n_windows = int(_gn.size)
                if _n_windows > 0:
                    _clip_fraction = (
                        float(np.mean(_gn > grad_clip_norm)) if grad_clip_norm > 0.0 else 0.0
                    )

                    def _q(arr: np.ndarray, q: float) -> float:
                        return float(np.quantile(arr, q))

                    _conv_row = {
                        "epoch": epoch,
                        "lr": epoch_lr,
                        "grad_clip_norm": grad_clip_norm,
                        "n_windows": _n_windows,
                        "clip_fraction": _clip_fraction,
                        "pre_clip_grad_norm_p05": _q(_gn, 0.05),
                        "pre_clip_grad_norm_p25": _q(_gn, 0.25),
                        "pre_clip_grad_norm_p50": _q(_gn, 0.50),
                        "pre_clip_grad_norm_p75": _q(_gn, 0.75),
                        "pre_clip_grad_norm_p95": _q(_gn, 0.95),
                        "pre_clip_grad_norm_mean": float(np.mean(_gn)),
                        "pre_clip_grad_norm_max": float(np.max(_gn)),
                        "update_norm_p05": _q(_un, 0.05),
                        "update_norm_p25": _q(_un, 0.25),
                        "update_norm_p50": _q(_un, 0.50),
                        "update_norm_p75": _q(_un, 0.75),
                        "update_norm_p95": _q(_un, 0.95),
                        "update_norm_mean": float(np.mean(_un)),
                        "update_norm_max": float(np.max(_un)),
                    }
                    try:
                        with conv_instr_path.open("a", encoding="utf-8") as _cifh:
                            _cifh.write(json.dumps(_conv_row) + "\n")
                    except Exception as exc:  # instrumentation is advisory; never break training
                        print(
                            f"[explicit-mpf] WARNING: convergence instrumentation write at "
                            f"epoch {epoch} failed: {exc}"
                        )

            should_stop, stop_reasons = evaluate_early_stop_gates(
                row, epoch=epoch, num_phases=reference.num_phases, cfg=early_stop_cfg
            )
            if health_should_stop:
                should_stop = True
                stop_reasons = list(stop_reasons) + health_stop_reasons

            if should_stop:
                print(f"[explicit-mpf] EARLY STOP at epoch {epoch+1}: {'; '.join(stop_reasons)}")
                last_diag = dict(row)
                last_diag["early_stopped"] = True
                last_diag["early_stop_reasons"] = stop_reasons
                break

        if fta_enabled and ((epoch % fta_every == 0) or (epoch == epochs - 1)):
            # No-gradient frozen-parameter deployment rollout evaluation,
            # never a gate quantity). Reads the deployed dynamics, not the stitched trace.
            try:
                with torch.no_grad():
                    _fp = phi0.detach()
                    _fs = model.initial_state(_fp)
                    _fprev = None
                    for _s in range(fta_steps):
                        _r, _fs, _ = model.predict_rate(_fp, _fs, previous_phi=_fprev)
                        _rn = _fp + model.model_dt * _r
                        if model.projection_mode == "soft_threshold_eps1e3":
                            _fp, _ = project_simplex_soft_threshold(_rn, threshold_eps=1.0e-3)
                        else:
                            _fp, _ = project_simplex(_rn)
                        _fprev = _fp
                    _fa = torch.bincount(_fp.argmax(dim=1).reshape(-1), minlength=model.num_phases)
                    _active = int((_fa[: model.num_phases] > 0).sum().item())
                with (out_dir / "frozen_theta_audit.jsonl").open("a", encoding="utf-8") as _fh:
                    _fh.write(json.dumps({"epoch": epoch, "steps": fta_steps,
                                          "frozen_theta_active_phases": _active}) + "\n")
            except Exception as exc:  # audit is advisory; never break training
                print(f"[explicit-mpf] WARNING: frozen-theta audit at epoch {epoch} failed: {exc}")

        if cbank_enabled and ((epoch % review_every == 0) or (epoch == epochs - 1)):
            # Label-free one-step residual under the frozen convergence-
            # bank permutations, applied to the model's OWN current-rollout anchor state
            # (`window_start_phi`, the detached window-boundary state this epoch's own
            # training rollout already reached -- never a post-t0 reference frame). No-grad,
            # model-only, advisory; never breaks training.
            try:
                _anchor = window_start_phi.detach()
                _perm_residuals: list[float] = []
                with torch.no_grad():
                    for _perm in cbank_permutations:
                        _idx = torch.tensor(_perm, dtype=torch.long, device=_anchor.device)
                        _phi_perm = _anchor.index_select(1, _idx)
                        _state_perm = model.initial_state(_phi_perm)
                        _rate_perm, _, _ = model.predict_rate(
                            _phi_perm, _state_perm, previous_phi=None
                        )
                        _target_perm = explicit_mpf_rhs(
                            _phi_perm, eta_px=model.eta_px, mu=model.mu, sigma=model.sigma
                        )
                        _res = mpf_residual_loss(
                            _rate_perm,
                            _target_perm,
                            model_dt=model.model_dt,
                            normalization=mpf_norm,
                            eps=mpf_eps,
                            abs_scale=mpf_abs_scale,
                            mixed_w_rel=mpf_mixed_w_rel,
                            mixed_w_abs=mpf_mixed_w_abs,
                        )
                        _perm_residuals.append(float(_res.detach()))
                _cbank_row = {
                    "epoch": epoch,
                    "n_permutations": len(cbank_permutations),
                    "per_permutation_residual": _perm_residuals,
                    "mean_residual": (
                        sum(_perm_residuals) / len(_perm_residuals) if _perm_residuals else 0.0
                    ),
                    "anchor_source": "epoch_final_window_start_phi",
                }
                assert cbank_path is not None
                with cbank_path.open("a", encoding="utf-8") as _cbfh:
                    _cbfh.write(json.dumps(_cbank_row) + "\n")
            except Exception as exc:  # diagnostic is advisory; never break training
                print(
                    f"[explicit-mpf] WARNING: convergence-bank residual at epoch {epoch} "
                    f"failed: {exc}"
                )

        is_final_epoch = epoch == epochs - 1
        if snapshots_enabled and ((epoch % snapshot_every == 0) or is_final_epoch):
            this_steps = snapshot_steps_final if is_final_epoch else snapshot_steps
            try:
                write_rollout_snapshot(
                    model, phi0, epoch=epoch, steps=this_steps, out_dir=out_dir
                )
            except Exception as exc:  # snapshots are advisory; never break training
                print(f"[explicit-mpf] WARNING: snapshot at epoch {epoch} failed: {exc}")

    if mode != "smoke":
        torch.save(
            {
                "model_state": model.state_dict(),
                "run_id": run_id,
                "model_dt": reference.model_dt,
                "dt_mu_sigma": reference.dt_mu_sigma,
                "num_phases": reference.num_phases,
            },
            checkpoint_path,
        )
        ckpt_out: str | None = _rel(checkpoint_path, repo_root)
    else:
        ckpt_out = None

    summary = {
        "run_id": run_id,
        "architecture": architecture,
        "model_dt": reference.model_dt,
        "dt_mu_sigma": reference.dt_mu_sigma,
        "num_phases": reference.num_phases,
        "epochs_run": epochs,
        "horizon": horizon,
        "graph_mode": model.graph_mode,
        "final": {k: _jsonable(v) for k, v in last_diag.items()},
        "no_reference_frames_after_t0_used": True,
    }
    if warmstart_metadata is not None:
        summary["warmstart"] = warmstart_metadata
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    return ExplicitMPFTrainResult(
        run_id=run_id,
        architecture=architecture,
        model_dt=reference.model_dt,
        dt_mu_sigma=reference.dt_mu_sigma,
        num_phases=reference.num_phases,
        epochs_run=epochs,
        diagnostics_path=_rel(diagnostics_path, repo_root),
        checkpoint_path=ckpt_out,
        summary_path=_rel(summary_path, repo_root),
        final_diagnostics={
            **({"warmstart": warmstart_metadata} if warmstart_metadata else {}),
            **{k: _jsonable(v) for k, v in last_diag.items()},
        },
        launched=True,
    )


def _rel(path: Path, repo_root: Path) -> str:
    """Return path relative to repo_root when possible, else the path itself."""

    try:
        return str(path.resolve().relative_to(repo_root.resolve()))
    except ValueError:
        return str(path)


def _detach_state(state: Any) -> Any:
    if isinstance(state, tuple):
        return tuple(s.detach() for s in state)
    return state.detach()


def _review_row(
    epoch: int,
    horizon: int,
    epoch_mpf: float,
    epoch_proj: float,
    step_diag: dict[str, Any],
    model: ExplicitMPFHybridRollout,
) -> dict[str, Any]:
    steps = max(horizon, 1)
    per_phase_area = step_diag.get("per_phase_area")
    active = step_diag.get("active_phase_count")
    row = {
        "epoch": epoch,
        "step": horizon,
        "loss_total": epoch_mpf / steps,
        "loss_ic": step_diag.get("ic_loss", 0.0),
        "loss_mpf_residual": epoch_mpf / steps,
        "loss_projection": epoch_proj / steps,
        "gamma": float(step_diag.get("gamma", 0.0)),
        "ann_norm": float(step_diag.get("ann_rate_norm", 0.0)),
        "convlstm_norm": float(step_diag.get("rnn_rate_norm", 0.0)),
        "ann_convlstm_ratio": float(step_diag.get("ann_rnn_contribution_ratio", 0.0)),
        "branch_collapse_min_norm": float(step_diag.get("branch_collapse_min_norm", 0.0)),
        "branch_collapse_warning": bool(float(step_diag.get("branch_collapse_min_norm", 1.0)) < 1.0e-8),
        "sum_error_pre_projection": float(step_diag.get("sum_error_pre_projection", 0.0)),
        "sum_error_post_projection": float(step_diag.get("sum_error_post_projection", 0.0)),
        "projection_correction_l1": float(step_diag.get("projection_correction_l1", 0.0)),
        "projection_correction_l2": float(step_diag.get("projection_correction_l2", 0.0)),
        "projection_correction_max": float(step_diag.get("projection_correction_max", 0.0)),
        "projected_pixel_fraction": float(step_diag.get("projected_pixel_fraction", 0.0)),
        "bounds_pre_projection": float(step_diag.get("bounds_violation_pre_projection", 0.0)),
        "bounds_post_projection": float(step_diag.get("bounds_violation_post_projection", 0.0)),
        "per_phase_area": _jsonable(per_phase_area) if per_phase_area is not None else [],
        "active_phase_count": int(_max_int(active)),
        "graph_mode": str(step_diag.get("graph_mode", model.graph_mode)),
        "node_count": float(step_diag.get("node_count", 0.0)),
        "edge_count": float(step_diag.get("edge_count", 0.0)),
        "junction_count": float(step_diag.get("junction_count", 0.0)),
        "rate_change_due_to_graph": float(step_diag.get("rate_change_due_to_graph", 0.0)),
        "graph_rate_centered_change": float(step_diag.get("graph_rate_centered_change", 0.0)),
        "graph_rate_uncentered_change": float(step_diag.get("graph_rate_uncentered_change", 0.0)),
        "graph_delta_centered_change": float(step_diag.get("graph_delta_centered_change", 0.0)),
        "graph_delta_uncentered_change": float(step_diag.get("graph_delta_uncentered_change", 0.0)),
        "graph_phase_centering_retained_ratio": float(
            step_diag.get("graph_phase_centering_retained_ratio", 0.0)
        ),
        "graph_rate_shrinkage_factor": float(step_diag.get("graph_rate_shrinkage_factor", 0.0)),
        "graph_scale_abs_mean": float(step_diag.get("graph_scale_abs_mean", 0.0)),
        "graph_scale_abs_max": float(step_diag.get("graph_scale_abs_max", 0.0)),
        "graph_bias_abs_mean": float(step_diag.get("graph_bias_abs_mean", 0.0)),
        "graph_bias_abs_max": float(step_diag.get("graph_bias_abs_max", 0.0)),
        "graph_bias_common_mode_abs_mean": float(
            step_diag.get("graph_bias_common_mode_abs_mean", 0.0)
        ),
        "graph_bias_phase_centered_abs_mean": float(
            step_diag.get("graph_bias_phase_centered_abs_mean", 0.0)
        ),
        "graph_conditioning_norm": float(step_diag.get("graph_conditioning_norm", 0.0)),
        "graph_head_conditioning_norm": float(
            step_diag.get("graph_head_conditioning_norm", 0.0)
        ),
        "head_change_due_to_graph": float(step_diag.get("head_change_due_to_graph", 0.0)),
        "graph_hidden_norm": float(step_diag.get("graph_hidden_norm", 0.0)),
        "graph_scale": float(step_diag.get("graph_conditioning_scale", 1.0)),
        "graph_center_bias": bool(float(step_diag.get("graph_center_bias", 0.0))),
        # Residual-scale / update-stability diagnostics (the stabilization fix).
        "target_rate_abs_max": float(step_diag.get("target_rate_abs_max", 0.0)),
        "pred_rate_abs_max": float(step_diag.get("pred_rate_abs_max", 0.0)),
        "rate_to_target_ratio": float(step_diag.get("rate_to_target_ratio", 0.0)),
        "target_delta_abs_max": float(step_diag.get("target_delta_abs_max", 0.0)),
        "pred_delta_abs_max": float(step_diag.get("pred_delta_abs_max", 0.0)),
        "delta_to_target_ratio": float(step_diag.get("delta_to_target_ratio", 0.0)),
        "raw_next_min": float(step_diag.get("raw_next_min", 0.0)),
        "raw_next_max": float(step_diag.get("raw_next_max", 0.0)),
        "grad_norm": float(step_diag.get("grad_norm", 0.0)),
        "cap_saturation_fraction": float(step_diag.get("cap_saturation_fraction", 0.0)),
        # Locality diagnostics logged for every run.
        "interface_localized_maxdelta_ratio": float(step_diag.get("interface_localized_maxdelta_ratio", 0.0)),
        "interface_pred_delta_max": float(step_diag.get("interface_pred_delta_max", 0.0)),
        "interface_target_delta_max": float(step_diag.get("interface_target_delta_max", 0.0)),
        "interface_pixel_fraction": float(step_diag.get("interface_pixel_fraction", 0.0)),
        "interface_local_weighting_enabled": bool(step_diag.get("interface_local_weighting_enabled", False)),
        # Long-horizon-only diagnostic (NOT a short-smoke gate): even the oracle
        # has ~0 argmax change at smoke horizons because the reference moves
        # sub-pixel for ~100 steps.
        "argmax_change_fraction_vs_t0": float(step_diag.get("argmax_change_fraction_vs_t0", 0.0)),
        # Recalibrated anti-suppression / scaling diagnostics (the short-smoke gate).
        "pred_delta_mean": float(step_diag.get("pred_delta_mean", 0.0)),
        "pred_delta_p50": float(step_diag.get("pred_delta_p50", 0.0)),
        "pred_delta_p95": float(step_diag.get("pred_delta_p95", 0.0)),
        "pred_delta_p99": float(step_diag.get("pred_delta_p99", 0.0)),
        "target_delta_mean": float(step_diag.get("target_delta_mean", 0.0)),
        "target_delta_p50": float(step_diag.get("target_delta_p50", 0.0)),
        "target_delta_p95": float(step_diag.get("target_delta_p95", 0.0)),
        "target_delta_p99": float(step_diag.get("target_delta_p99", 0.0)),
        "learned_target_magnitude_ratio": float(step_diag.get("learned_target_magnitude_ratio", 0.0)),
        "displacement_cosine_to_target": float(step_diag.get("displacement_cosine_to_target", 0.0)),
        "displacement_relative_residual": float(step_diag.get("displacement_relative_residual", 0.0)),
        "displacement_sign_agreement": float(step_diag.get("displacement_sign_agreement", 0.0)),
        "rate_head_bounded": bool(getattr(model, "rate_head_bounded", False)),
        "rate_parameterization": str(getattr(model, "rate_parameterization", "legacy_tanh_cap")),
        "delta_scale": float(getattr(model, "delta_scale", 0.0)),
        "max_delta_phi_per_step": float(getattr(model, "max_delta_phi_per_step", 0.0)),
        "graph_contribution_to_pred_rate_abs_max": float(
            step_diag.get("rate_change_due_to_graph", 0.0)
        )
        / max(float(step_diag.get("pred_rate_abs_max", 0.0)), 1.0e-30),
        "energy_lh2_enabled": bool(step_diag.get("energy_lh2_enabled", False)),
        "surface_descent_enabled": bool(step_diag.get("surface_descent_enabled", False)),
        "no_reference_frames_after_t0_used": bool(
            step_diag.get("no_reference_frames_after_t0_used", True)
        ),
    }
    if step_diag.get("surface_descent_enabled"):
        row.update(
            {
                "surface_loss": float(step_diag.get("surface_loss", 0.0)),
                "surface_r_pred": float(step_diag.get("surface_r_pred", 0.0)),
                "surface_r_rhs": float(step_diag.get("surface_r_rhs", 0.0)),
                "surface_contrib": float(step_diag.get("surface_contrib", 0.0)),
                "surface_contrib_over_mpf": float(step_diag.get("surface_contrib_over_mpf", 0.0)),
                "surface_capped": bool(step_diag.get("surface_capped", False)),
                "surface_loss_epoch_mean": float(step_diag.get("surface_loss_epoch_mean", 0.0)),
                "surface_contrib_epoch_mean": float(step_diag.get("surface_contrib_epoch_mean", 0.0)),
                "surface_lambda": float(step_diag.get("surface_lambda", 0.0)),
                "surface_cap_fraction": float(step_diag.get("surface_cap_fraction", 0.0)),
            }
        )
    if "energy_lh2_loss_raw" in step_diag:
        row.update(
            {
                "energy_lh2_loss_raw": float(step_diag.get("energy_lh2_loss_raw", 0.0)),
                "energy_lh2_loss_weighted": float(
                    step_diag.get("energy_lh2_loss_weighted", 0.0)
                ),
                "energy_lh2_mean_delta": float(step_diag.get("energy_lh2_mean_delta", 0.0)),
                "energy_lh2_positive_fraction": float(
                    step_diag.get("energy_lh2_positive_fraction", 0.0)
                ),
                "energy_lh2_interface_fraction": float(
                    step_diag.get("energy_lh2_interface_fraction", 0.0)
                ),
                "energy_lh2_lambda": float(step_diag.get("energy_lh2_lambda", 0.0)),
            }
        )
        if "energy_lh2_loss_normalized" in step_diag:
            row["energy_lh2_loss_normalized"] = float(
                step_diag.get("energy_lh2_loss_normalized", 0.0)
            )
            row["energy_lh2_norm_mode"] = str(step_diag.get("energy_lh2_norm_mode", ""))
            row["energy_lh2_raw_scale"] = float(step_diag.get("energy_lh2_raw_scale", 0.0))
        row["loss_total"] = row["loss_total"] + row["energy_lh2_loss_weighted"]
    return row


def _max_int(value: Any) -> int:
    if isinstance(value, torch.Tensor):
        return int(value.max().item()) if value.numel() else 0
    if isinstance(value, (list, tuple)):
        return int(max(value)) if value else 0
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0
