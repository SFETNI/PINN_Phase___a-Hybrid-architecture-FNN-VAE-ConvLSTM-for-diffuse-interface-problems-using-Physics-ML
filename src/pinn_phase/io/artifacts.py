"""Safe, hash-verified loading for public scientific artifacts."""

from __future__ import annotations

from collections.abc import Mapping
import hashlib
import io
import os
from pathlib import Path
import re
from typing import Any
import zipfile

import numpy as np
import torch


MAX_CHECKPOINT_BYTES = 2 * 1024**3
MAX_NPZ_FILE_BYTES = 2 * 1024**3
MAX_NPZ_MEMBERS = 512
MAX_NPZ_MEMBER_BYTES = 1024**3
MAX_NPZ_UNCOMPRESSED_BYTES = 2 * 1024**3
MAX_NPZ_COMPRESSION_RATIO = 10_000


def _normalized_sha256(expected_sha256: str) -> str:
    expected = expected_sha256.strip().lower()
    if len(expected) != 64 or any(char not in "0123456789abcdef" for char in expected):
        raise ValueError("expected_sha256 must be a complete lowercase SHA-256 digest")
    return expected


def _hash_stream(stream: io.BufferedReader) -> str:
    digest = hashlib.sha256()
    for chunk in iter(lambda: stream.read(1024 * 1024), b""):
        digest.update(chunk)
    return digest.hexdigest()


def _verify_open_stream(
    stream: io.BufferedReader,
    *,
    path: Path,
    expected_sha256: str,
) -> str:
    expected = _normalized_sha256(expected_sha256)
    stream.seek(0)
    actual = _hash_stream(stream)
    if actual != expected:
        raise ValueError(f"SHA-256 mismatch for {path}: expected {expected}, got {actual}")
    stream.seek(0)
    return actual


def _require_safe_torch_version() -> None:
    match = re.match(r"^(\d+)\.(\d+)", torch.__version__)
    if match is None or tuple(map(int, match.groups())) < (2, 6):
        raise RuntimeError("checkpoint loading requires PyTorch 2.6 or newer")


def _validate_npz_container(stream: io.BufferedReader, *, path: Path) -> None:
    stream.seek(0)
    try:
        with zipfile.ZipFile(stream) as archive:
            members = archive.infolist()
            if len(members) > MAX_NPZ_MEMBERS:
                raise ValueError(f"NumPy archive has too many members: {path}")
            names = [member.filename for member in members]
            if len(names) != len(set(names)):
                raise ValueError(f"NumPy archive has duplicate members: {path}")
            total_uncompressed = 0
            for member in members:
                name = member.filename
                if (
                    not name.endswith(".npy")
                    or Path(name).name != name
                    or "/" in name
                    or "\\" in name
                ):
                    raise ValueError(f"NumPy archive contains an invalid member name: {name!r}")
                if member.file_size > MAX_NPZ_MEMBER_BYTES:
                    raise ValueError(f"NumPy archive member exceeds the size limit: {name}")
                total_uncompressed += member.file_size
                if member.file_size and member.compress_size == 0:
                    raise ValueError(f"NumPy archive member has an invalid compressed size: {name}")
                if (
                    member.compress_size
                    and member.file_size / member.compress_size > MAX_NPZ_COMPRESSION_RATIO
                ):
                    raise ValueError(f"NumPy archive member exceeds the compression-ratio limit: {name}")
            if total_uncompressed > MAX_NPZ_UNCOMPRESSED_BYTES:
                raise ValueError(f"NumPy archive exceeds the uncompressed-size limit: {path}")
    except zipfile.BadZipFile as exc:
        raise ValueError(f"invalid NumPy archive: {path}") from exc
    finally:
        stream.seek(0)


def sha256_file(path: str | Path) -> str:
    """Return the SHA-256 digest of *path* without loading it into memory."""

    with Path(path).open("rb") as stream:
        return _hash_stream(stream)


def verify_sha256(path: str | Path, expected_sha256: str) -> str:
    """Verify an artifact digest and return the normalized digest."""

    expected = _normalized_sha256(expected_sha256)
    actual = sha256_file(path)
    if actual != expected:
        raise ValueError(f"SHA-256 mismatch for {Path(path)}: expected {expected}, got {actual}")
    return actual


def load_torch_checkpoint(
    path: str | Path,
    *,
    expected_sha256: str,
    map_location: str | torch.device = "cpu",
    required_keys: frozenset[str] | None = None,
) -> Mapping[str, Any]:
    """Load a hash-pinned checkpoint through PyTorch's restricted loader."""

    checkpoint = Path(path)
    _require_safe_torch_version()
    with checkpoint.open("rb") as stream:
        if os.fstat(stream.fileno()).st_size > MAX_CHECKPOINT_BYTES:
            raise ValueError(f"checkpoint exceeds the size limit: {checkpoint}")
        _verify_open_stream(stream, path=checkpoint, expected_sha256=expected_sha256)
        payload = torch.load(stream, map_location=map_location, weights_only=True)
    if not isinstance(payload, Mapping):
        raise TypeError("checkpoint payload must be a mapping")
    if required_keys is not None:
        missing = required_keys.difference(payload)
        if missing:
            raise ValueError(f"checkpoint is missing required keys: {sorted(missing)}")
    return payload


def load_npz_arrays(
    path: str | Path,
    *,
    expected_sha256: str,
    required_keys: frozenset[str] | None = None,
    expected_keys: frozenset[str] | None = None,
) -> dict[str, np.ndarray]:
    """Load a NumPy archive with pickle disabled and copy arrays out safely."""

    archive_path = Path(path)
    with archive_path.open("rb") as stream:
        if os.fstat(stream.fileno()).st_size > MAX_NPZ_FILE_BYTES:
            raise ValueError(f"NumPy archive exceeds the file-size limit: {archive_path}")
        _verify_open_stream(stream, path=archive_path, expected_sha256=expected_sha256)
        _validate_npz_container(stream, path=archive_path)
        with np.load(stream, allow_pickle=False) as archive:
            keys = frozenset(archive.files)
            if expected_keys is not None and keys != expected_keys:
                raise ValueError(
                    "NumPy archive key set differs from the reviewed schema: "
                    f"expected {sorted(expected_keys)}, got {sorted(keys)}"
                )
            if required_keys is not None:
                missing = required_keys.difference(keys)
                if missing:
                    raise ValueError(f"NumPy archive is missing required keys: {sorted(missing)}")
            return {name: np.asarray(archive[name]).copy() for name in archive.files}
