"""Import-closure tests for modules outside the public physics-only scope.

The repository deliberately excludes six supervised modules that are out of
scope for this public reproducibility package: ``survival_gate``,
``event_conditioning``, ``event_positive_weighting``,
``supervised_sentinel_hook``, ``survival_stage_gate``,
``temporal_window_sampler``. This test applies two complementary checks:

1. (static) none of the six module files exist anywhere under ``src/``.
2. (live import) importing ``pinn_phase`` and each of its public submodules
   must not raise ``ModuleNotFoundError`` for any of the six excluded names.

The public API must remain importable without any of these modules.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

CANDIDATE_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = CANDIDATE_ROOT / "src"

EXCLUDED_MODULE_NAMES = [
    "survival_gate",
    "event_conditioning",
    "event_positive_weighting",
    "supervised_sentinel_hook",
    "survival_stage_gate",
    "temporal_window_sampler",
]

PUBLIC_SUBMODULES = [
    "pinn_phase",
    "pinn_phase.physics",
    "pinn_phase.models",
    "pinn_phase.training",
    "pinn_phase.evaluation",
    "pinn_phase.io",
]


def test_excluded_module_files_are_absent_from_src() -> None:
    """Static check: none of the 6 excluded module filenames exist anywhere under src/."""
    found: list[str] = []
    for name in EXCLUDED_MODULE_NAMES:
        matches = list(SRC_ROOT.rglob(f"{name}.py"))
        if matches:
            found.extend(str(m.relative_to(CANDIDATE_ROOT)) for m in matches)
    assert not found, f"excluded supervised module file(s) found in repository src/: {found}"


def test_public_submodules_import_without_referencing_excluded_modules() -> None:
    """Live-import check: importing the public API must not fail due to an excluded module.

    This intentionally does NOT swallow a ModuleNotFoundError that names one
    of the six excluded modules -- if a shipped module ever imports an excluded
    module unguarded, this test must fail loudly (not silently pass or skip).
    """
    program = (
        "import importlib, json, sys\n"
        f"names = {PUBLIC_SUBMODULES!r}\n"
        "failures = {}\n"
        "origin = None\n"
        "for name in names:\n"
        "    try:\n"
        "        module = importlib.import_module(name)\n"
        "        origin = origin or getattr(module, '__file__', None)\n"
        "    except ModuleNotFoundError as exc:\n"
        "        failures[name] = str(exc)\n"
        "print(json.dumps({'failures': failures, 'origin': origin}))\n"
    )
    # The child gets this repository's src/ explicitly, at the FRONT of the path.
    #
    # Two things depend on that. First, pytest's `pythonpath` setting applies to the
    # parent interpreter only, so without this the child cannot import the package at
    # all unless it happens to be installed -- which made this test pass or fail
    # according to the state of the developer's environment rather than the state of
    # the tree. Second, putting src/ first means an unrelated installed copy cannot
    # shadow the code under test: the assertion below checks which one was imported.
    environment = dict(os.environ)
    existing = environment.get("PYTHONPATH")
    environment["PYTHONPATH"] = os.pathsep.join(
        [str(SRC_ROOT)] + ([existing] if existing else [])
    )
    environment["PYTHONDONTWRITEBYTECODE"] = "1"   # leave no cache in the tree
    result = subprocess.run(
        [sys.executable, "-B", "-c", program],
        cwd=CANDIDATE_ROOT,
        check=True,
        capture_output=True,
        text=True,
        env=environment,
    )
    payload = json.loads(result.stdout)
    failures: dict[str, str] = payload["failures"]
    origin = payload["origin"]

    assert origin is not None, "the child interpreter imported nothing at all"
    assert Path(origin).resolve().is_relative_to(SRC_ROOT.resolve()), (
        f"the child imported pinn_phase from {origin}, not from this repository's "
        f"{SRC_ROOT}. This test must exercise the shipped source, not an installed copy."
    )

    excluded_related_failures = {
        mod: err for mod, err in failures.items()
        if any(excluded in err for excluded in EXCLUDED_MODULE_NAMES)
    }
    other_failures = {mod: err for mod, err in failures.items() if mod not in excluded_related_failures}

    assert not other_failures, f"unexpected import failures unrelated to the excluded modules: {other_failures}"

    assert not excluded_related_failures, (
        "public modules reference excluded supervised modules: "
        f"{excluded_related_failures}"
    )
