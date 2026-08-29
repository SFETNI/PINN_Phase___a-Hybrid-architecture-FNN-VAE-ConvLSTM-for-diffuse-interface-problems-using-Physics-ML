"""Pytest configuration for the PINN-Phase public repository test suite.

Adds ``<candidate_root>/src`` to ``sys.path`` if the ``pinn_phase`` package is
not already importable, so the bounded test suite can run without requiring
``pip install -e .`` first. This mirrors the documented
``PYTHONPATH=<repo>/src`` convention; it is a test-harness convenience,
not a change to how any of the shipped public scripts import the package
(the scripts under ``scripts/`` do not perform this sys.path insertion
themselves).
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

CANDIDATE_ROOT = Path(__file__).resolve().parents[1]
SRC = CANDIDATE_ROOT / "src"
SCRIPTS = CANDIDATE_ROOT / "scripts"

try:
    importlib.import_module("pinn_phase")
except ModuleNotFoundError:
    if str(SRC) not in sys.path:
        sys.path.insert(0, str(SRC))

if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))
