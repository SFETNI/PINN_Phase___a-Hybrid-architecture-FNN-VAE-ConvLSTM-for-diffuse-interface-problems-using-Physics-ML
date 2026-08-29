"""Every front-page claim must point at something the archive actually contains.

The claim map binds each quantitative README statement to supporting artifacts and
to the reproduction level at which this archive supports it. These tests refuse a
dangling path, a digest that disagrees with distributed bytes, and a claim whose
declared level is stronger than the level its benchmark is classified at.
"""

from __future__ import annotations

import json
from pathlib import Path
import re

import pytest

from pinn_phase.io.artifacts import sha256_file

ROOT = Path(__file__).resolve().parents[1]
CLAIMS = json.loads((ROOT / "docs/CLAIM_TO_ARTIFACT_MAP.json").read_text(encoding="utf-8"))["claims"]
LEVELS = json.loads((ROOT / "docs/REPRODUCTION_LEVELS.json").read_text(encoding="utf-8"))
CLAIM_IDS = [claim["id"] for claim in CLAIMS]
VALID_LEVELS = set(LEVELS["levels"])
BY_BENCHMARK = {row["id"]: row for row in LEVELS["benchmarks"]}
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
# A claim may be supported more strongly than its benchmark only when it is a
# property of the shipped code rather than of the benchmark's fields.
CODE_LEVEL_CLAIMS = {"parameter_count", "reference_isolation"}
STRENGTH = {
    "PROVENANCE_ONLY": 0,
    "CHECKPOINT_AND_CODE_REPLAY": 1,
    "SCORE_RECOMPUTATION": 2,
    "FULL_RECOMPUTATION": 3,
}


@pytest.mark.parametrize("claim", CLAIMS, ids=CLAIM_IDS)
def test_supporting_paths_exist(claim: dict) -> None:
    for entry in claim["supporting_paths"]:
        path = ROOT / entry.split(" ")[0]
        assert path.exists(), f"{claim['id']}: supporting path does not exist: {entry}"


@pytest.mark.parametrize("claim", CLAIMS, ids=CLAIM_IDS)
def test_declared_level_is_valid(claim: dict) -> None:
    assert claim["level"] in VALID_LEVELS, f"{claim['id']}: unknown level {claim['level']!r}"


@pytest.mark.parametrize("claim", CLAIMS, ids=CLAIM_IDS)
def test_claim_names_a_real_benchmark(claim: dict) -> None:
    """Checked before, and independently of, the level comparison.

    Folding this into the level test let a dangling benchmark id ride through
    whenever the claim was one of the code-level exemptions, because the
    exemption returned before the existence assertion was ever reached.
    """
    if claim["benchmark"] == "code_property":
        assert "benchmark_note" in claim, (
            f"{claim['id']}: a claim declared code_property must say why it is not a benchmark"
        )
        return
    assert claim["benchmark"] in BY_BENCHMARK, (
        f"{claim['id']}: names benchmark {claim['benchmark']!r}, which is absent from "
        "REPRODUCTION_LEVELS.json"
    )


@pytest.mark.parametrize("claim", CLAIMS, ids=CLAIM_IDS)
def test_claim_does_not_outrank_its_benchmark(claim: dict) -> None:
    if claim["id"] in CODE_LEVEL_CLAIMS or claim["benchmark"] == "code_property":
        return
    benchmark = BY_BENCHMARK[claim["benchmark"]]
    assert STRENGTH[claim["level"]] <= STRENGTH[benchmark["level"]], (
        f"{claim['id']}: claimed at {claim['level']} but benchmark {claim['benchmark']} "
        f"is only {benchmark['level']}"
    )


def test_no_benchmark_carries_a_level_whose_definition_it_fails() -> None:
    """CHECKPOINT_AND_CODE_REPLAY requires an initial field. None is distributed."""
    initial_fields = [
        path
        for path in sorted((ROOT / "benchmarks/initial_conditions").glob("*.npz"))
        if _npz_members(path) & {"phi0", "fields", "phi", "states"}
    ]
    replay = [row["id"] for row in LEVELS["benchmarks"] if row["level"] == "CHECKPOINT_AND_CODE_REPLAY"]
    if not initial_fields:
        assert not replay, (
            "benchmarks claim CHECKPOINT_AND_CODE_REPLAY, whose definition requires the "
            f"initial field, but no initial field is distributed: {replay}"
        )


def _npz_members(path: Path) -> set[str]:
    import zipfile

    try:
        with zipfile.ZipFile(path) as archive:
            return {Path(name).stem for name in archive.namelist()}
    except zipfile.BadZipFile:  # pragma: no cover - fixtures are valid archives
        return set()


@pytest.mark.parametrize("claim", CLAIMS, ids=CLAIM_IDS)
def test_named_digests_match_bytes_where_the_artifact_is_distributed(claim: dict) -> None:
    """A digest naming a distributed file must be the digest of that file."""
    distributed = {
        sha256_file(path): path for path in (ROOT / "checkpoints").glob("*.weights.npz")
    }
    distributed.update(
        {sha256_file(path): path for path in (ROOT / "configs").rglob("*.yaml")}
    )
    parents = {
        entry["parent_checkpoint_sha256"]
        for entry in json.loads(
            (ROOT / "docs/ARTIFACT_IDENTITY_LEDGER.json").read_text(encoding="utf-8")
        )["artifacts"]
    }
    for role, digest in claim["digests"].items():
        assert SHA256_RE.match(digest), f"{claim['id']}/{role}: malformed digest"
        if role.startswith("accepted_parent_checkpoint"):
            # A parent identity must be labelled as one, must be a known parent, and
            # must not be quoted as though it were a distributed file.
            assert digest in parents, (
                f"{claim['id']}/{role}: names a parent checkpoint the ledger does not record"
            )
            assert digest not in distributed, (
                f"{claim['id']}/{role}: an accepted parent digest matches a distributed file"
            )
            continue
        if digest in distributed:
            continue  # verified against real bytes
        # Otherwise the digest must name an artifact the map does not claim to ship.
        assert claim["level"] != "FULL_RECOMPUTATION" or claim["id"] in CODE_LEVEL_CLAIMS, (
            f"{claim['id']}/{role}: claimed FULL_RECOMPUTATION but {digest} is not distributed"
        )


def test_provenance_only_claims_do_not_promise_reproduction(claim_map: None = None) -> None:
    for claim in CLAIMS:
        if claim["level"] != "PROVENANCE_ONLY":
            continue
        assert "not_supported_here" in claim or "note" in BY_BENCHMARK.get(claim["benchmark"], {}), (
            f"{claim['id']}: a provenance-only claim must say what the archive cannot do"
        )


def test_every_benchmark_has_at_least_one_claim() -> None:
    claimed = {claim["benchmark"] for claim in CLAIMS}
    unclaimed = set(BY_BENCHMARK) - claimed
    assert not unclaimed, f"benchmarks classified but never claimed: {sorted(unclaimed)}"


def test_absent_artifacts_are_named_with_a_requirement() -> None:
    """An omitted artifact must be identified, not merely mentioned."""
    for row in LEVELS["benchmarks"]:
        for absent in row["artifacts_absent"]:
            if isinstance(absent, str):
                continue
            assert absent.get("what"), f"{row['id']}: an absent artifact has no description"
            assert absent.get("requirement"), (
                f"{row['id']}: absent artifact {absent['what']!r} has no retrieval or "
                "regeneration requirement"
            )
            digest = absent.get("sha256")
            if digest is not None:
                assert SHA256_RE.match(digest), f"{row['id']}: malformed absent-artifact digest"


def test_no_benchmark_claims_full_recomputation_without_shipping_its_inputs() -> None:
    for row in LEVELS["benchmarks"]:
        if row["level"] != "FULL_RECOMPUTATION":
            continue
        assert not row["artifacts_absent"], (
            f"{row['id']}: FULL_RECOMPUTATION with absent artifacts {row['artifacts_absent']}"
        )
        assert row.get("command"), f"{row['id']}: FULL_RECOMPUTATION with no command to run"
