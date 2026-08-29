"""Derived public replay weights: deterministic writing, strict loading.

The archive distributes **derived public replay weights**, not the accepted
training checkpoints. The two are different objects and the difference matters:

* an accepted parent scientific checkpoint is a serialized training payload. It
  carries model state, and it may also carry optimizer state, schedule state and
  free-form run metadata written at training time;
* a derived public replay-weight artifact carries model-state tensors and nothing
  else. It has no pickle, no optimizer state, no run identifier, no path, no
  timestamp and no prose. It is a repackaging of the parent's model state, with
  every tensor name, dtype, shape and raw value preserved exactly.

Deriving these artifacts changes packaging only. It does not change a single model
parameter, and it is not new scientific evidence. The parent's identity is recorded
as a digest in the lineage record beside each artifact, so the derivation stays
checkable without the parent being distributed.

Determinism
-----------
Members are written in sorted order with a fixed timestamp, fixed permissions and
no compression. Stored entries make the file a pure function of the tensor data and
the fixed header fields, so a rebuild is byte-identical and does not depend on the
compression library a reviewer happens to have. The archive's outer transport is
compressed instead.

Strictness
----------
:func:`load_public_weights` hashes the file before opening it, opens it with
pickling disabled, requires exactly the registered tensor keys, and checks every
tensor's dtype, shape, byte count, raw-value digest and finiteness before it hands
anything back. A file that has been edited, truncated, extended, re-typed or
re-ordered does not load.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
import io
import json
from pathlib import Path
from typing import Any
import zipfile

import numpy as np
import torch

from .artifacts import _normalized_sha256, _validate_npz_container, sha256_file

#: Suffix of a member inside the archive. One member per tensor.
MEMBER_SUFFIX = ".npy"

#: Fixed ZIP member timestamp. The earliest value the format can represent.
FIXED_MEMBER_DATE = (1980, 1, 1, 0, 0, 0)

#: Fixed member permissions: readable, not writable by group or world.
FIXED_MEMBER_MODE = 0o644

#: Numeric kinds a model-state tensor may have. Object ('O'), string ('U', 'S')
#: and void ('V') kinds are refused: they are how non-numeric payloads travel.
ALLOWED_DTYPE_KINDS = frozenset({"f", "i", "u", "b"})

#: Value dtypes a public weight artifact may declare.
ALLOWED_DTYPES = frozenset({"float32", "float64", "float16", "int64", "int32", "uint8", "bool"})

MAX_WEIGHTS_BYTES = 512 * 1024**2
MAX_TENSORS = 4096

SCHEMA = "pinn-phase-public-replay-weights-v1"
LINEAGE_SCHEMA = "pinn-phase-public-weight-lineage-v1"


class PublicWeightsError(ValueError):
    """Raised when an artifact is not exactly the registered public weight set."""


@dataclass(frozen=True)
class TensorRecord:
    """The registered identity of one model-state tensor."""

    name: str
    dtype: str
    shape: tuple[int, ...]
    bytes: int
    value_sha256: str


@dataclass(frozen=True)
class WeightLineage:
    """A public weight artifact's registered identity and its parent's digest."""

    benchmark: str
    artifact: str
    public_weights_path: str
    public_weights_sha256: str
    public_weights_bytes: int
    parent_checkpoint_sha256: str
    model_state_fingerprint_sha256: str
    tensors: tuple[TensorRecord, ...]

    @property
    def tensor_names(self) -> tuple[str, ...]:
        return tuple(record.name for record in self.tensors)

    @property
    def parameter_count(self) -> int:
        return sum(int(np.prod(record.shape)) if record.shape else 1 for record in self.tensors)


def _contiguous(array: np.ndarray) -> np.ndarray:
    """C-contiguous view of *array*, preserving rank.

    ``np.ascontiguousarray`` promotes a zero-dimensional array to shape ``(1,)``,
    which would silently reshape a scalar parameter. A zero-dimensional array is
    already contiguous, so the promotion is never needed; this checks the flag
    instead of forcing the conversion.
    """

    return array if array.flags["C_CONTIGUOUS"] else np.ascontiguousarray(array)


def tensor_value_sha256(array: np.ndarray) -> str:
    """Digest of dtype, shape and raw values, independent of any container."""

    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode())
    digest.update(b"\0")
    digest.update(str(tuple(array.shape)).encode())
    digest.update(b"\0")
    digest.update(_contiguous(array).tobytes())
    return digest.hexdigest()


def _as_numpy(value: Any) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        if value.requires_grad:  # pragma: no cover - eval-mode payloads are detached
            value = value.detach()
        return _contiguous(value.cpu().numpy())
    return _contiguous(np.asarray(value))


def model_state_fingerprint(state: Mapping[str, Any]) -> str:
    """Digest of an entire model state: names, dtypes, shapes and raw values.

    Computed identically from a parent checkpoint's tensors and from a derived
    artifact's arrays, so equality of the two is the statement that the derivation
    preserved the model state exactly.
    """

    digest = hashlib.sha256()
    for name in sorted(state):
        array = _as_numpy(state[name])
        digest.update(name.encode())
        digest.update(b"\0")
        digest.update(str(array.dtype).encode())
        digest.update(b"\0")
        digest.update(str(tuple(array.shape)).encode())
        digest.update(b"\0")
        digest.update(array.tobytes())
        digest.update(b"\0")
    return digest.hexdigest()


def tensor_records(state: Mapping[str, Any]) -> tuple[TensorRecord, ...]:
    """Registered per-tensor identities, in the canonical sorted key order."""

    records = []
    for name in sorted(state):
        array = _as_numpy(state[name])
        records.append(
            TensorRecord(
                name=name,
                dtype=str(array.dtype),
                shape=tuple(int(size) for size in array.shape),
                bytes=int(array.nbytes),
                value_sha256=tensor_value_sha256(array),
            )
        )
    return tuple(records)


def write_public_weights(state: Mapping[str, Any], path: str | Path) -> str:
    """Write derived public replay weights deterministically; return the digest.

    Tensor names are sorted, members are stored uncompressed with a fixed
    timestamp and fixed permissions, and every value is written through the NumPy
    array format with pickling disabled. Nothing but the tensors is written: the
    file has no place to carry a run identifier, a path or a comment.
    """

    out = Path(path)
    if not state:
        raise PublicWeightsError("refusing to write an empty model state")
    if len(state) > MAX_TENSORS:
        raise PublicWeightsError(f"model state has too many tensors: {len(state)}")

    payloads: list[tuple[str, bytes]] = []
    for name in sorted(state):
        array = _as_numpy(state[name])
        if array.dtype.kind not in ALLOWED_DTYPE_KINDS or str(array.dtype) not in ALLOWED_DTYPES:
            raise PublicWeightsError(
                f"{name}: dtype {array.dtype} cannot be represented exactly in a public "
                "weight artifact. Refusing to cast, reshape or rename it."
            )
        if MEMBER_SUFFIX in name or "/" in name or "\\" in name:
            raise PublicWeightsError(f"unsupported tensor name for a member name: {name!r}")
        buffer = io.BytesIO()
        np.lib.format.write_array(buffer, array, allow_pickle=False)
        payloads.append((name + MEMBER_SUFFIX, buffer.getvalue()))

    out.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(out, "w", compression=zipfile.ZIP_STORED) as archive:
        for member_name, payload in payloads:
            info = zipfile.ZipInfo(member_name, date_time=FIXED_MEMBER_DATE)
            info.compress_type = zipfile.ZIP_STORED
            info.external_attr = FIXED_MEMBER_MODE << 16
            info.create_system = 3
            archive.writestr(info, payload)
    return sha256_file(out)


def load_weight_lineage(path: str | Path) -> WeightLineage:
    """Read a lineage record. The record registers what the artifact must contain."""

    record = json.loads(Path(path).read_text(encoding="utf-8"))
    if record.get("schema") != LINEAGE_SCHEMA:
        raise PublicWeightsError(
            f"{path}: unexpected lineage schema {record.get('schema')!r}"
        )
    tensors = tuple(
        TensorRecord(
            name=entry["name"],
            dtype=entry["dtype"],
            shape=tuple(int(size) for size in entry["shape"]),
            bytes=int(entry["bytes"]),
            value_sha256=_normalized_sha256(entry["value_sha256"]),
        )
        for entry in record["tensors"]
    )
    return WeightLineage(
        benchmark=record["benchmark"],
        artifact=record["artifact"],
        public_weights_path=record["public_weights"]["path"],
        public_weights_sha256=_normalized_sha256(record["public_weights"]["sha256"]),
        public_weights_bytes=int(record["public_weights"]["bytes"]),
        parent_checkpoint_sha256=_normalized_sha256(
            record["accepted_parent_checkpoint"]["sha256"]
        ),
        model_state_fingerprint_sha256=_normalized_sha256(
            record["model_state_fingerprint_sha256"]
        ),
        tensors=tensors,
    )


def _member_names(path: Path) -> list[str]:
    """Member names read from the container, without decoding any array.

    Member comments and extra fields are refused here rather than ignored. The
    lineage record states that a weight artifact carries "free-form text of any
    kind" nowhere; ZIP metadata is free-form text, so the loader has to enforce that
    clause rather than leave it to a separate scanner.
    """

    try:
        with zipfile.ZipFile(path) as archive:
            if archive.comment:
                raise PublicWeightsError(
                    f"{path}: the archive carries a comment; a weight artifact carries "
                    "tensors and no free-form text"
                )
            for info in archive.infolist():
                if info.comment or info.extra:
                    raise PublicWeightsError(
                        f"{path}: member {info.filename!r} carries ZIP metadata "
                        f"({'comment' if info.comment else 'extra field'}); a weight "
                        "artifact carries tensors and no free-form text"
                    )
            return list(archive.namelist())
    except zipfile.BadZipFile as exc:
        raise PublicWeightsError(f"{path}: not a readable weight archive") from exc


def load_public_weights(
    path: str | Path,
    *,
    lineage: WeightLineage,
    expected_sha256: str | None = None,
) -> dict[str, np.ndarray]:
    """Load derived public replay weights, refusing anything unregistered.

    The order of operations is the point:

    1. the file size is bounded and the bytes are hashed **before** the container
       is opened, against the digest the lineage record registers;
    2. member names are read from the container and must be exactly the registered
       tensor key set, so an extra or missing member is refused before any array
       is decoded;
    3. arrays are decoded with pickling disabled;
    4. each tensor's dtype, shape, byte count and raw-value digest must match its
       registered identity, and every value must be finite;
    5. the model-state fingerprint of the whole set must match.

    Nothing is cast, reshaped, renamed or normalized anywhere in this path.
    """

    archive_path = Path(path)
    size = archive_path.stat().st_size
    if size > MAX_WEIGHTS_BYTES:
        raise PublicWeightsError(f"weight artifact exceeds the size limit: {archive_path}")

    digest = sha256_file(archive_path)
    expected = _normalized_sha256(expected_sha256 or lineage.public_weights_sha256)
    if digest != expected:
        raise PublicWeightsError(
            f"SHA-256 mismatch for {archive_path}: expected {expected}, got {digest}"
        )
    if size != lineage.public_weights_bytes:
        raise PublicWeightsError(
            f"{archive_path}: {size} bytes on disk, {lineage.public_weights_bytes} registered"
        )

    registered = {record.name: record for record in lineage.tensors}
    present = _member_names(archive_path)
    if len(present) != len(set(present)):
        raise PublicWeightsError(f"{archive_path}: duplicate members in the weight archive")
    stems = []
    for member in present:
        if not member.endswith(MEMBER_SUFFIX) or Path(member).name != member:
            raise PublicWeightsError(f"{archive_path}: invalid member name {member!r}")
        stems.append(member[: -len(MEMBER_SUFFIX)])
    if set(stems) != set(registered):
        missing = sorted(set(registered) - set(stems))
        extra = sorted(set(stems) - set(registered))
        raise PublicWeightsError(
            f"{archive_path}: refused before reading any array. The member set is not the "
            f"registered tensor key set (missing {missing}, unregistered {extra})."
        )

    with archive_path.open("rb") as stream:
        _validate_npz_container(stream, path=archive_path)
        with np.load(stream, allow_pickle=False) as loaded:
            names = list(loaded.files)
            if set(names) != set(registered):  # pragma: no cover - re-checked
                raise PublicWeightsError(f"{archive_path}: decoded member set changed")
            arrays = {name: np.asarray(loaded[name]).copy() for name in names}

    state: dict[str, np.ndarray] = {}
    for record in lineage.tensors:
        array = arrays[record.name]
        if array.dtype.kind not in ALLOWED_DTYPE_KINDS:
            raise PublicWeightsError(
                f"{archive_path}: {record.name} has non-numeric dtype {array.dtype}"
            )
        if str(array.dtype) != record.dtype:
            raise PublicWeightsError(
                f"{archive_path}: {record.name} is {array.dtype}, registered {record.dtype}"
            )
        if tuple(array.shape) != record.shape:
            raise PublicWeightsError(
                f"{archive_path}: {record.name} has shape {tuple(array.shape)}, "
                f"registered {record.shape}"
            )
        if int(array.nbytes) != record.bytes:
            raise PublicWeightsError(
                f"{archive_path}: {record.name} is {array.nbytes} bytes, "
                f"registered {record.bytes}"
            )
        if array.dtype.kind == "f" and not bool(np.isfinite(array).all()):
            raise PublicWeightsError(
                f"{archive_path}: {record.name} carries a non-finite value"
            )
        if tensor_value_sha256(array) != record.value_sha256:
            raise PublicWeightsError(
                f"{archive_path}: {record.name} does not match its registered raw-value digest"
            )
        state[record.name] = array

    fingerprint = model_state_fingerprint(state)
    if fingerprint != lineage.model_state_fingerprint_sha256:
        raise PublicWeightsError(
            f"{archive_path}: model-state fingerprint {fingerprint} does not match the "
            f"registered {lineage.model_state_fingerprint_sha256}"
        )
    return state


def as_torch_state_dict(state: Mapping[str, np.ndarray]) -> dict[str, torch.Tensor]:
    """Wrap validated arrays as tensors. No cast, no reshape, no renaming."""

    return {name: torch.from_numpy(_contiguous(array)) for name, array in state.items()}


def load_public_state_dict(
    weights_path: str | Path,
    lineage_path: str | Path,
    *,
    expected_lineage_sha256: str | None = None,
) -> dict[str, torch.Tensor]:
    """Load a public weight artifact straight into a strict-loadable state dict."""

    if expected_lineage_sha256 is not None:
        actual = sha256_file(lineage_path)
        expected = _normalized_sha256(expected_lineage_sha256)
        if actual != expected:
            raise PublicWeightsError(
                f"SHA-256 mismatch for {lineage_path}: expected {expected}, got {actual}"
            )
    lineage = load_weight_lineage(lineage_path)
    return as_torch_state_dict(load_public_weights(weights_path, lineage=lineage))


def lineage_document(
    *,
    benchmark: str,
    artifact: str,
    public_weights_path: str,
    public_weights_sha256: str,
    public_weights_bytes: int,
    parent_checkpoint_sha256: str,
    model_state_fingerprint: str,
    records: Sequence[TensorRecord],
    derivation_script: str,
    derivation_script_sha256: str,
    model_config: str,
    model_config_sha256: str,
) -> dict[str, Any]:
    """Build the neutral lineage record that ships beside a weight artifact."""

    return {
        "schema": LINEAGE_SCHEMA,
        "benchmark": benchmark,
        "artifact": artifact,
        "statement": (
            "This artifact carries derived public replay weights. It is a repackaging of "
            "the model state of the accepted parent scientific checkpoint: every tensor "
            "name, dtype, shape and raw value is preserved exactly. The derivation "
            "changes packaging only, not model parameters, and is not new scientific "
            "evidence. The parent checkpoint is identified below by digest; it is not "
            "distributed in this archive."
        ),
        "accepted_parent_checkpoint": {
            "sha256": parent_checkpoint_sha256,
            "distributed": False,
            "role": "accepted scientific checkpoint; identity of record",
        },
        "public_weights": {
            "path": public_weights_path,
            "sha256": public_weights_sha256,
            "bytes": public_weights_bytes,
            "format": SCHEMA,
            "container": "zip, stored members, fixed timestamps and permissions, no pickle",
            "carries": "model-state tensors only",
            "excludes": [
                "optimizer state",
                "scheduler or other training state",
                "run identifiers",
                "paths, usernames, timestamps",
                "object or string arrays",
                "free-form text of any kind",
            ],
        },
        "model_state_fingerprint_sha256": model_state_fingerprint,
        "digest_recipes": {
            "note": (
                "Stated here so that the identities below can be recomputed by anyone "
                "holding the parent checkpoint and this record, without reading the "
                "archive's source. A digest you can only obtain by running our code is "
                "weaker evidence than one you can derive from a specification."
            ),
            "value_sha256": (
                "SHA-256 over: str(dtype) + b'\\0' + str(tuple(shape)) + b'\\0' + the "
                "array's C-contiguous raw bytes. dtype and shape are rendered exactly as "
                "NumPy's str() produces them, e.g. 'float32' and '(32, 1)'; a "
                "zero-dimensional array renders as '()' and contributes its scalar bytes."
            ),
            "model_state_fingerprint_sha256": (
                "SHA-256 over the concatenation, for each tensor name in ascending "
                "lexicographic order, of: name + b'\\0' + str(dtype) + b'\\0' + "
                "str(tuple(shape)) + b'\\0' + raw bytes + b'\\0'. Note the trailing "
                "separator after every tensor, including the last."
            ),
            "public_weights_sha256": "SHA-256 over the artifact file's bytes.",
            "accepted_parent_checkpoint.sha256": "SHA-256 over the parent file's bytes.",
        },
        "tensor_count": len(records),
        "parameter_count": sum(
            int(np.prod(record.shape)) if record.shape else 1 for record in records
        ),
        "tensor_key_order": [record.name for record in records],
        "tensors": [
            {
                "name": record.name,
                "dtype": record.dtype,
                "shape": list(record.shape),
                "bytes": record.bytes,
                "value_sha256": record.value_sha256,
            }
            for record in records
        ],
        "derivation": {
            "script": derivation_script,
            "script_sha256": derivation_script_sha256,
            "model_config": model_config,
            "model_config_sha256": model_config_sha256,
            "transformation": "packaging only",
            "verification": (
                "Tensor keys, dtypes, shapes, raw values and the model-state fingerprint "
                "were compared against the accepted parent checkpoint before this record "
                "was written, and the artifact was built twice in separate directories "
                "and required to be byte-identical."
            ),
        },
    }
