#!/usr/bin/env python3
"""Require a fully passing public test run with no hidden skips."""

from __future__ import annotations

import sys
from pathlib import Path
import xml.etree.ElementTree as ET


EXPECTED_SKIPS: set[str] = set()


def main() -> int:
    path = Path(sys.argv[1] if len(sys.argv) > 1 else "pytest.xml")
    root = ET.parse(path).getroot()
    cases = root.findall(".//testcase")
    failed = [case for case in cases if case.find("failure") is not None or case.find("error") is not None]
    skipped = {
        f"{case.attrib.get('classname', '')}::{case.attrib.get('name', '').split('[')[0]}"
        for case in cases
        if case.find("skipped") is not None
    }
    if failed:
        print(f"test outcome refused: {len(failed)} failed or errored cases")
        return 1
    if skipped != EXPECTED_SKIPS:
        print(f"test outcome refused: expected skips {sorted(EXPECTED_SKIPS)}, got {sorted(skipped)}")
        return 1
    print(f"test outcome accepted: {len(cases)} passed, no skips")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
