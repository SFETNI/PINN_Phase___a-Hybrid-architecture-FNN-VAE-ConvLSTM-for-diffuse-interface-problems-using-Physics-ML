"""Keep public tutorial paths, commands, and evidence claims synchronized."""
from __future__ import annotations

import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TUTORIALS = ROOT / "docs/tutorials"
GUIDES = ROOT / "docs/guides"
CASE_STUDIES = ROOT / "docs/case-studies"
TUTORIAL_PATHS = (
    TUTORIALS / "README.md",
    TUTORIALS / "01_phase_field_foundations.md",
    TUTORIALS / "02_explicit_mpf_operator.md",
    TUTORIALS / "03_inside_pinn_phase.md",
    TUTORIALS / "04_one_pinn_phase_step.md",
    TUTORIALS / "05_physics_informed_training.md",
    TUTORIALS / "06_symmetry_and_model_families.md",
    TUTORIALS / "07_long_horizon_and_topology.md",
    TUTORIALS / "08_run_and_reproduce.md",
    GUIDES / "01_getting_started_cpu.md",
    GUIDES / "02_reproduction_levels_in_practice.md",
    GUIDES / "03_integrity_and_provenance.md",
    CASE_STUDIES / "n16_prospective_evidence.md",
)
MARKDOWN_LINK = re.compile(r"(?<!!)\[[^\]]+\]\(([^)]+)\)")


def _text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def test_tutorial_files_and_local_links_exist() -> None:
    for path in TUTORIAL_PATHS:
        assert path.is_file(), f"missing tutorial file: {path.relative_to(ROOT)}"
        for target in MARKDOWN_LINK.findall(_text(path)):
            relative = target.split("#", 1)[0].strip().strip("<>")
            if relative:
                assert (path.parent / relative).is_file(), (
                    f"broken local tutorial link in {path.relative_to(ROOT)}: {target}"
                )
    assert "docs/tutorials/README.md" in _text(ROOT / "README.md")


def test_physics_first_curriculum_structure_and_core_terms() -> None:
    pages = TUTORIAL_PATHS[1:9]
    assert len(pages) == 8
    for path in pages:
        content = _text(path)
        assert content.startswith("# ")
        assert "## What you will learn" in content
        assert "## Prerequisites" in content
        assert "## What to remember" in content
        assert "## Next" in content
        assert content.index("## What you will learn") < content.index("## Prerequisites")
        assert content.index("## What to remember") < content.index("## Next")

    combined = "\n".join(_text(path) for path in pages)
    required = (
        "W'(\\phi)", "P_N(v)_k", "R(\\phi)", "\\nabla_h^2",
        "\\Delta\\phi_t", "\\Pi_k(\\psi)", "q_t=", "p_t=",
        "L_{\\mathrm{res}}", "f(P\\phi)=P f(\\phi)", "TBPTT",
        "static-$t_0$ persistence baseline", "CHECKPOINT_AND_CODE_REPLAY",
    )
    for term in required:
        assert term in combined, f"missing required tutorial concept: {term}"


def test_documented_commands_map_to_current_public_scripts() -> None:
    expected_scripts = {
        "scripts/smoke_test.py", "scripts/replay_rollout.py",
        "scripts/reproduce_n25_transfer.py", "scripts/reproduce_scalar_reference.py",
        "scripts/verify_n16_96_transfer.py", "scripts/verify_manifest.py",
        "scripts/verify_source_lineage.py", "scripts/verify_checkpoint_identities.py",
        "scripts/check_public_tree.py",
    }
    tutorial_text = "\n".join(_text(path) for path in TUTORIAL_PATHS)
    for script in expected_scripts:
        assert (ROOT / script).is_file()
        assert script in tutorial_text
    initial_field_test = "tests/test_initial_field_provenance.py"
    assert (ROOT / initial_field_test).is_file()
    assert initial_field_test in tutorial_text
    allowed = {
        "EXECUTED_AND_PASSING",
        "DOCUMENTATION_ONLY", "OPTIONAL_EXTERNAL_ASSET",
    }
    used = set(re.findall(r"\*\*Classification:\*\* `([A-Z_]+)`", tutorial_text))
    assert used <= allowed
    assert {
        "EXECUTED_AND_PASSING",
        "OPTIONAL_EXTERNAL_ASSET",
    } <= used
    assert "EXECUTED_PLATFORM_LIMITATION" not in tutorial_text


def test_reproduction_levels_match_the_machine_readable_contract() -> None:
    levels = json.loads((ROOT / "docs/REPRODUCTION_LEVELS.json").read_text(encoding="utf-8"))
    tutorial = _text(GUIDES / "02_reproduction_levels_in_practice.md")
    for level in (
        "FULL_RECOMPUTATION", "SCORE_RECOMPUTATION",
        "CHECKPOINT_AND_CODE_REPLAY", "PROVENANCE_ONLY",
    ):
        assert level in levels["levels"]
        assert level in tutorial
    assert "CURRENT_WINDOWS_IMPLEMENTATION_LIMITATION" not in tutorial


def test_n16_tutorial_claims_match_the_current_score_record() -> None:
    record = json.loads(
        (ROOT / "benchmarks/n16_96_transfer/expected_score.json").read_text(encoding="utf-8")
    )
    tutorial = _text(CASE_STUDIES / "n16_prospective_evidence.md")
    headline = record["headline"]
    terminal = headline["terminal_agreement_percent_range"]
    static_t0 = headline["static_t0_agreement_percent_range"]
    gain = headline["terminal_gain_percentage_points_range"]
    assert f"{terminal[0]:.2f}%-{terminal[1]:.2f}%" in tutorial
    assert f"{static_t0[0]:.2f}%-{static_t0[1]:.2f}%" in tutorial
    assert f"+{gain[0]:.2f} to +{gain[1]:.2f} percentage points" in tutorial
    assert record["terminal_scored_step"] == 1600
    assert record["monitored_saved_horizon_step"] == 3200
    assert "step 1600" in tutorial and "step 3200" in tutorial
    assert headline["terminal_topology_and_extinction_identities"] == "6/6"
    assert "6/6 topology and extinction identities" in tutorial
    assert headline["complete_predefined_qualification"] == "5/6"
    assert "5/6" in tutorial
    case_five = record["cases"][4]
    assert case_five["public_name"] == "Unseen microstructure 5"
    assert case_five["minimum_persistence_gain_step"] == 3200
    assert f"{case_five['minimum_persistence_gain_percentage_points']:.6f}" in tutorial
    assert not case_five["wave_order_preserved"]
    assert not case_five["persistence_all_post_initial_strictly_above_static_t0"]
    assert case_five["tail_active_set_equal"]


def test_tutorials_have_no_private_paths_or_stale_operational_language() -> None:
    tutorial_text = "\n".join(_text(path) for path in TUTORIAL_PATHS)
    assert not re.search(r"\b[A-Za-z]:[\\/][A-Za-z0-9_. -]+[\\/]", tutorial_text)
    forbidden = (
        "CloudPC", "OneDrive", "Codex", "ChatGPT", "AI-generated",
        "reviewer", "submission", "manuscript", "paper Figure", "paper Table", "VAE",
    )
    for value in forbidden:
        assert value not in tutorial_text
    assert not re.search(r"\bagent\b", tutorial_text, flags=re.IGNORECASE)


def test_n16_external_asset_count_and_total_are_synchronized() -> None:
    manifest = json.loads((ROOT / "benchmarks/n16_96_transfer/manifest.json").read_text(encoding="utf-8"))
    assets = manifest["assets"]
    tutorial = _text(CASE_STUDIES / "n16_prospective_evidence.md")
    assert len(assets) == 18
    assert sum(asset["bytes"] for asset in assets) == 10_726_256_132
    assert "18 documented" in tutorial
    assert "10,726,256,132 bytes" in tutorial
