#!/usr/bin/env python3
"""Verify the public N16/96 score record and, optionally, documented assets."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_score_record(records: Path) -> None:
    x = json.loads(records.read_text(encoding="utf-8"))
    c = x["cases"]
    assert len(c) == 6 and sum(v["complete_predefined_qualification"] for v in c) == 5
    assert x["terminal_scored_step"] == 1600 and x["monitored_saved_horizon_step"] == 3200
    assert all(v["terminal_step"] == 1600 and v["monitoring_horizon_step"] == 3200 for v in c)
    assert all(v["finite_values"] and v["nan_count"] == 0 and v["inf_count"] == 0 for v in c)
    assert all(v["saved_cadence_exact"] and v["monitoring_horizon_exact"] for v in c)
    assert all(v["maximum_phase_sum_error"] < 1e-3 for v in c)
    assert all(v["persistence_frames_compared"] == 16 for v in c)
    assert all(v["persistence_frames_strictly_greater"] in (15, 16) for v in c)
    assert all(v["persistence_all_post_initial_strictly_above_static_t0"] == (v["persistence_frames_strictly_greater"] == 16) for v in c)
    assert all(v["terminal_active_set_equal"] and v["extinction_identities_equal"] and v["tail_active_set_equal"] for v in c)
    offsets = [event["offset_steps"] for value in c for event in value["event_offsets"]]
    assert len(offsets) == 18 and offsets.count(0) == 11 and offsets.count(-200) == 2 and offsets.count(200) == 5
    assert x["headline"]["event_offsets"] == {"total": 18, "exact": 11, "early_by_200": 2, "late_by_200": 5}
    five = c[4]
    assert not five["complete_predefined_qualification"] and five["minimum_persistence_gain_step"] == 3200
    assert abs(five["minimum_persistence_gain_percentage_points"] + 0.080589) < 1e-6
    assert not five["wave_order_preserved"] and five["persistence_frames_strictly_greater"] == 15
    for value in c:
        qualifies = (
            value["finite_values"]
            and value["saved_cadence_exact"]
            and value["monitoring_horizon_exact"]
            and value["maximum_phase_sum_error"] < 1e-3
            and value["terminal_agreement_percent"] >= max(94.3, value["static_t0_agreement_percent"] + 1.0)
            and value["persistence_all_post_initial_strictly_above_static_t0"]
            and value["terminal_active_set_equal"]
            and value["extinction_identities_equal"]
            and all(event["within_plus_minus_200"] for event in value["event_offsets"])
            and value["wave_order_preserved"]
            and value["tail_active_set_equal"]
        )
        assert value["complete_predefined_qualification"] == qualifies


def verify_external_assets(asset_manifest: Path, asset_root: Path) -> None:
    """Fail closed unless every documented external payload matches its record."""
    document = json.loads(asset_manifest.read_text(encoding="utf-8"))
    assets = document.get("assets")
    if not isinstance(assets, list) or len(assets) != 18:
        raise ValueError("asset manifest must declare exactly 18 external assets")
    if not asset_root.is_dir():
        raise ValueError(f"asset root is not a directory: {asset_root}")
    seen: set[str] = set()
    expected_directories = {"external/n16_96_transfer"}
    for asset in assets:
        relative = asset.get("expected_relative_location")
        filename = asset.get("filename")
        expected_bytes = asset.get("bytes")
        expected_sha256 = asset.get("sha256")
        if not isinstance(relative, str) or not isinstance(filename, str):
            raise ValueError("asset manifest has a non-string path or filename")
        if not isinstance(expected_bytes, int) or not isinstance(expected_sha256, str):
            raise ValueError(f"asset manifest has invalid size or SHA-256 for {relative}")
        relative_path = Path(relative)
        if relative_path.is_absolute() or ".." in relative_path.parts or relative_path.name != filename:
            raise ValueError(f"asset manifest has unsafe or mismatched path: {relative}")
        if relative in seen:
            raise ValueError(f"asset manifest repeats an external asset path: {relative}")
        seen.add(relative)
        expected_directories.add(relative_path.parent.as_posix())
        candidate = asset_root / relative_path
        if candidate.is_symlink():
            raise ValueError(f"documented external asset must not be a symlink: {relative}")
        if not candidate.is_file():
            raise ValueError(f"missing documented external asset: {relative}")
        if candidate.stat().st_size != expected_bytes:
            raise ValueError(
                f"size mismatch for {relative}: expected {expected_bytes}, got {candidate.stat().st_size}"
            )
        actual_sha256 = sha256(candidate)
        if actual_sha256 != expected_sha256:
            raise ValueError(f"SHA-256 mismatch for {relative}: expected {expected_sha256}, got {actual_sha256}")
    documented_root = asset_root / "external/n16_96_transfer"
    if documented_root.is_symlink():
        raise ValueError("documented external asset hierarchy must not be a symlink")
    if not documented_root.is_dir():
        raise ValueError("documented external asset hierarchy is missing")
    for candidate in sorted(documented_root.rglob("*")):
        relative = candidate.relative_to(asset_root).as_posix()
        if candidate.is_symlink():
            raise ValueError(f"unexpected symlink in documented external assets: {relative}")
        if candidate.is_file() and relative not in seen:
            raise ValueError(f"unexpected regular file in documented external assets: {relative}")
        if candidate.is_dir() and relative not in expected_directories:
            raise ValueError(f"unexpected directory in documented external assets: {relative}")
        if not candidate.is_file() and not candidate.is_dir():
            raise ValueError(f"unexpected non-regular external asset entry: {relative}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--records", type=Path, required=True)
    parser.add_argument(
        "--asset-root",
        type=Path,
        help="directory containing the documented external/n16_96_transfer/ hierarchy",
    )
    args = parser.parse_args()
    try:
        verify_score_record(args.records)
        if args.asset_root is not None:
            verify_external_assets(args.records.parent / "manifest.json", args.asset_root)
    except (AssertionError, OSError, ValueError, json.JSONDecodeError) as exc:
        parser.error(str(exc) or "N16 prospective record verification failed")
    suffix = " and 18 external assets" if args.asset_root is not None else ""
    print(f"N16 prospective public record: PASS{suffix}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
