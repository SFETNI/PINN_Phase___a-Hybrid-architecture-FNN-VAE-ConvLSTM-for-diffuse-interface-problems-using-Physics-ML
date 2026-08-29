"""Portability tests for pinn_phase.evaluation.profiling across platforms."""

from __future__ import annotations

from dataclasses import fields
import importlib
import sys
import types

import pytest


MODULE_NAME = "pinn_phase.evaluation.profiling"


def _reload_profiling():
    sys.modules.pop(MODULE_NAME, None)
    return importlib.import_module(MODULE_NAME)


@pytest.fixture(autouse=True)
def _restore_real_module():
    yield
    _reload_profiling()


def test_module_imports_without_resource(monkeypatch):
    """The module must import cleanly when `resource` is absent (Windows)."""

    monkeypatch.setitem(sys.modules, "resource", None)
    module = _reload_profiling()
    assert module.resource is None
    assert callable(module.current_rss_mib)
    assert callable(module.peak_rss_mib)


def test_windows_fallback_values_and_types(monkeypatch):
    """Without `resource`, both RSS readers must return non-negative floats."""

    monkeypatch.setitem(sys.modules, "resource", None)
    module = _reload_profiling()
    current = module.current_rss_mib()
    peak = module.peak_rss_mib()
    assert isinstance(current, float)
    assert isinstance(peak, float)
    assert current >= 0.0
    assert peak >= 0.0


def test_posix_resource_branch_uses_getrusage(monkeypatch):
    """When `resource` is present, peak_rss_mib must call it, unchanged."""

    calls = []

    class _FakeUsage:
        ru_maxrss = 204800  # KiB, as returned on Linux

    fake_resource = types.ModuleType("resource")
    fake_resource.RUSAGE_SELF = 0

    def _fake_getrusage(who):
        calls.append(who)
        return _FakeUsage()

    fake_resource.getrusage = _fake_getrusage

    monkeypatch.setitem(sys.modules, "resource", fake_resource)
    module = _reload_profiling()
    assert module.resource is fake_resource

    peak = module.peak_rss_mib()
    assert peak == pytest.approx(200.0)
    assert calls == [fake_resource.RUSAGE_SELF]


def test_public_api_preserved_across_platforms():
    """Public names, RolloutProfile fields, and units must not change."""

    module = _reload_profiling()
    for name in ("RolloutProfile", "current_rss_mib", "peak_rss_mib", "live_tensor_mib", "profile_rollout"):
        assert hasattr(module, name)

    field_names = {f.name for f in fields(module.RolloutProfile)}
    assert field_names == {
        "label",
        "device",
        "grid_height",
        "grid_width",
        "steps",
        "hidden_channels",
        "wall_seconds",
        "rss_before_mib",
        "rss_with_output_mib",
        "rss_after_cleanup_mib",
        "process_peak_rss_mib",
        "live_tensor_before_mib",
        "live_tensor_with_output_mib",
        "live_tensor_after_cleanup_mib",
        "cuda_peak_allocated_mib",
        "cuda_peak_reserved_mib",
    }


def test_evaluation_package_imports_on_windows():
    """The evaluation package and its N25 consumer must import on Windows."""

    import pinn_phase.evaluation  # noqa: F401
    from pinn_phase.evaluation import n25_transfer  # noqa: F401
