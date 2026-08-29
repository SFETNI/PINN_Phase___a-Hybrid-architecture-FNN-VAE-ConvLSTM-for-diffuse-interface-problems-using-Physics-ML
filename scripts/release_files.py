"""Shared file-selection policy for the release manifest and the public verifier.

Two different questions get asked of a checkout, and conflating them is what made
the documented workflow fail:

``release_files``
    *What belongs in a release?* Walks the tree and applies the exclusion policy.
    This is the constructor's question, asked when a manifest, sdist, wheel or
    external archive is built. Generated material is excluded here, and this is the
    only place that decides what may ship.

``manifest_payload``
    *What did ship?* Reads ``MANIFEST.sha256`` and returns exactly those paths. This
    is the reader's question. It is deliberately immune to whatever the reader's own
    commands have since produced in the working tree -- an editable install, a
    reproduction run, a pytest cache -- because none of that is archive content.

A file ignored by ``manifest_payload`` cannot thereby enter a release: it would have
to pass ``release_files`` first, and that function has no knowledge of the manifest.
"""

from __future__ import annotations

from pathlib import Path


EXCLUDED_DIRECTORY_NAMES = frozenset(
    {
        ".git",
        ".pytest_cache",
        ".ruff_cache",
        "__pycache__",
        "artifacts",
        "build",
        "dist",
        "outputs",
    }
)

#: Directories and files a reader legitimately creates by following the documented
#: workflow. They are named here so the verifier can say *why* it ignored something
#: rather than ignoring it silently.
LOCAL_WORKING_ARTEFACTS = {
    "outputs": "output of the documented reproduction and replay commands",
    "artifacts": "output of a training run, if you start one yourself",
    "__pycache__": "interpreter bytecode cache",
    ".pytest_cache": "pytest cache",
    ".ruff_cache": "linter cache",
    "build": "build backend output",
    "dist": "build backend output",
    "*.egg-info": "editable-install metadata",
}

MANIFEST_NAME = "MANIFEST.sha256"


def is_generated(relative: Path) -> bool:
    """True when a path is generated material rather than authored release content."""
    return any(
        part in EXCLUDED_DIRECTORY_NAMES or part.endswith(".egg-info")
        for part in relative.parts
    )


def release_files(root: Path, *, manifest_name: str = MANIFEST_NAME) -> set[Path]:
    """Return regular, non-generated files relative to ``root``."""
    return {
        path.relative_to(root)
        for path in root.rglob("*")
        if path.is_file()
        and not path.is_symlink()
        and path.name != manifest_name
        and not is_generated(path.relative_to(root))
    }


def manifest_payload(root: Path, *, manifest_name: str = MANIFEST_NAME) -> list[Path]:
    """Return the release payload as the manifest defines it, in manifest order.

    The manifest does not list itself -- a file cannot contain its own digest -- so
    it is added here, since it is unambiguously archive content.
    """
    manifest = root / manifest_name
    if not manifest.is_file():
        raise FileNotFoundError(f"{manifest_name} is missing; this is not a release tree")
    paths = [
        Path(line.split("  ", 1)[1])
        for line in manifest.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    return [Path(manifest_name), *paths]
