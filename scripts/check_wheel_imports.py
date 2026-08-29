#!/usr/bin/env python3
"""Import every public package from an installed wheel."""

from __future__ import annotations

import importlib


MODULES = (
    "pinn_phase",
    "pinn_phase.evaluation",
    "pinn_phase.io",
    "pinn_phase.models",
    "pinn_phase.physics",
    "pinn_phase.training",
)


for module in MODULES:
    importlib.import_module(module)
print(f"wheel imports passed: {len(MODULES)} public packages")
