from __future__ import annotations

import torch

from pinn_phase.models.factory import ArchitectureFamily, architecture_family_from_config
from pinn_phase.models.perm_equivariant_mpf import PermEquivariantMPFRollout


def test_architecture_family_is_explicit() -> None:
    assert architecture_family_from_config(
        {
            "model": {
                "architecture": "explicit_mpf_ann_convlstm_hybrid",
                "arch_variant": "perm_equivariant_v1",
            }
        }
    ) is ArchitectureFamily.N3_EQUIVARIANT_MPF
    assert architecture_family_from_config(
        {"model": {"architecture": "explicit_mpf_ann_convlstm_hybrid"}}
    ) is ArchitectureFamily.LEGACY_PA_HYBRID_MPF


def test_n3_variant_rejects_an_incompatible_base_architecture() -> None:
    try:
        architecture_family_from_config(
            {"model": {"architecture": "unrelated", "arch_variant": "perm_equivariant_v1"}}
        )
    except ValueError as exc:
        assert "requires architecture" in str(exc)
    else:
        raise AssertionError("invalid N3 base architecture was accepted")


def test_n3_rate_commutes_with_phase_permutation() -> None:
    torch.manual_seed(17)
    model = PermEquivariantMPFRollout(
        num_phases=4,
        model_dt=0.1,
        eta_px=4.0,
        hidden_channels=3,
        encoder_channels=2,
        ann_hidden_features=(5,),
        zero_initialize_heads=False,
    ).eval()
    phi = torch.softmax(torch.randn(1, 4, 9, 11), dim=1)
    permutation = torch.tensor([2, 0, 3, 1])

    rate, _, _ = model.predict_rate(phi, model.initial_state(phi))
    permuted_phi = phi[:, permutation]
    permuted_rate, _, _ = model.predict_rate(
        permuted_phi,
        model.initial_state(permuted_phi),
    )

    torch.testing.assert_close(permuted_rate, rate[:, permutation], rtol=1e-6, atol=1e-7)


def test_n3_state_dict_is_independent_of_phase_count() -> None:
    model_four = PermEquivariantMPFRollout(
        num_phases=4,
        model_dt=0.1,
        eta_px=4.0,
        hidden_channels=3,
        encoder_channels=2,
        ann_hidden_features=(5,),
    )
    model_seven = PermEquivariantMPFRollout(
        num_phases=7,
        model_dt=0.1,
        eta_px=4.0,
        hidden_channels=3,
        encoder_channels=2,
        ann_hidden_features=(5,),
    )
    result = model_seven.load_state_dict(model_four.state_dict(), strict=True)
    assert result.missing_keys == []
    assert result.unexpected_keys == []


def test_n3_rate_commutes_with_periodic_translation() -> None:
    torch.manual_seed(23)
    model = PermEquivariantMPFRollout(
        num_phases=4,
        model_dt=0.1,
        eta_px=4.0,
        hidden_channels=3,
        encoder_channels=2,
        ann_hidden_features=(5,),
        zero_initialize_heads=False,
    ).eval()
    phi = torch.softmax(torch.randn(1, 4, 9, 11), dim=1)
    rate, _, _ = model.predict_rate(phi, model.initial_state(phi))

    translated_phi = torch.roll(phi, shifts=(2, -3), dims=(-2, -1))
    translated_rate, _, _ = model.predict_rate(
        translated_phi,
        model.initial_state(translated_phi),
    )
    expected = torch.roll(rate, shifts=(2, -3), dims=(-2, -1))
    torch.testing.assert_close(translated_rate, expected, rtol=1e-6, atol=1e-7)
