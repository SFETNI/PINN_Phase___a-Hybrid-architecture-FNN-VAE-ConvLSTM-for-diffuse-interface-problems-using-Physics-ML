from __future__ import annotations

from pathlib import Path
import runpy


ROOT = Path(__file__).resolve().parents[1]


def test_cpu_smoke_contract() -> None:
    namespace = runpy.run_path(str(ROOT / "scripts/smoke_test.py"))
    report = namespace["run_smoke"]()
    assert report["passed"] is True


def test_scalar_reference_reproduces_expected_metrics(tmp_path: Path) -> None:
    namespace = runpy.run_path(str(ROOT / "scripts/reproduce_scalar_reference.py"))
    metrics = namespace["reproduce"](
        (ROOT / "configs/benchmarks/scalar_shrinkage_2d.yaml").resolve(),
        tmp_path / "scalar",
        (ROOT / "benchmarks/scalar_shrinkage_2d/expected_metrics.json").resolve(),
    )
    assert metrics["energy_nonincrease_fraction"] == 1.0
    assert metrics["relative_slope_error"] < 0.01
