#!/usr/bin/env python3
"""Fail when a release tree contains private, unsafe, or generated material.

The scan covers bytes, not just text. Every file is read whole and matched against
the rules below; container files are additionally opened so that member names,
member metadata and **decompressed** member contents are scanned too. A compressed
member is exactly where a private string survives a repository-wide text search, so
it is opened rather than trusted.

Rules describe forbidden CLASSES, never named parties. A scanner that spells out the
internal names it hunts for puts those names in the public tree in order to keep them
out of it, and it fails the moment a name changes.
"""

from __future__ import annotations

import sys

# Importing the shared release policy must not leave a bytecode cache behind: this
# script is run during release construction, where a stray __pycache__ would be a
# finding against the tree it is checking.
sys.dont_write_bytecode = True

import argparse
import ast
import gzip
import io
import json
import os
from pathlib import Path
import re
import tarfile
import zipfile
import zlib

sys.path.insert(0, str(Path(__file__).resolve().parent))

from release_files import (  # noqa: E402
    LOCAL_WORKING_ARTEFACTS,
    is_generated,
    manifest_payload,
)


# Directories that are produced by running something, never authored. "outputs" is
# where every shipped script writes, so a tree that still has one was packaged
# after a run rather than from a clean checkout.
GENERATED_NAMES = {".pytest_cache", "__pycache__", ".ruff_cache", "build", "dist",
                   "outputs", "artifacts", ".coverage", ".ipynb_checkpoints"}
# Literals split so this file does not match its own rules when scanned.
SECRET_FILENAMES = {".netrc", ".npmrc", ".bash_history", ".python_history",
                    "id" "_rsa", ".DS_Store", "credentials"}
MAX_FILE_BYTES = 90 * 1024 * 1024
NATIVE_128_PATH = re.compile(
    r"(?:128cubed|128_cubed|128x128x128|128\^3|128³)", re.IGNORECASE
)

# These rules describe forbidden CLASSES, not a list of names.
#
# An earlier version of this scanner spelled out the internal tool names it was
# looking for, which put those names in the public tree in order to keep them out
# of it. That is self-defeating, and it also fails the moment a name changes. Each
# rule below matches the shape of the thing instead: an authorization token, an
# inter-party record filename, a governance decision. Nothing here needs updating
# when a tool is renamed.
#
# Literals are still split where a rule would otherwise match this file's own
# source, so the scanner can be run over itself.
BYTE_RULES = {
    "home-directory path": re.compile(b"/ho" + b"me/", re.IGNORECASE),
    "other-platform home path": re.compile(b"/Us" + b"ers/|C:\\\\Us" + b"ers\\\\", re.IGNORECASE),
    "overstated generalization wording": re.compile(b"zero" + br"[- ]shot", re.IGNORECASE),
    "private root path": re.compile(br"/work" + b"space/|/ro" + br"ot/[A-Za-z]"),
    # A capitalized token ending in a GO/APPROVE/AUTHORIZE verb, optionally asserted
    # with =YES. Matches an authorization literal of any name.
    "execution authorization token": re.compile(
        br"\b[A-Z][A-Z0-9]*(?:_[A-Z0-9]+)*_(?:GO|APPROVE[DR]?|AUTHORIZE[DR]?)"
        br"(?:_[A-Z0-9]+)*\s*=\s*(?:YES|TRUE|1)\b"
    ),
    "execution authorization prefix": re.compile(
        b"AUTH" + br"OR_GO|GRA" + br"NT_TOKEN", re.IGNORECASE
    ),
    "provider-specific launcher": re.compile(b"Run" + b"Pod", re.IGNORECASE),
    # Coordination and governance record shapes, named by role rather than by author.
    # A record whose NAME or heading marks it as an inter-party coordination
    # document. Prose that merely mentions coordination is not a coordination record,
    # so this matches the identifier shape, not the English word.
    "coordination record": re.compile(
        br"\bHAND" + br"OFF_[A-Z0-9_]{3,}\b|\bCOORD" + br"INATION_[A-Z0-9_]{3,}\b"
        br"|\bSTA" + br"TUS_REPORT_[A-Z0-9_]{3,}\b"
    ),
    "governance record": re.compile(
        br"\brat" + br"ifi(?:ed|cation)\b|\badjud" + br"icat(?:ed|ion)\b"
        br"|\bconsu" + br"med[- ]token\b|\bgover" + br"nance gate\b|\bgate" + br" review\b",
        re.IGNORECASE,
    ),
    "assistant attribution": re.compile(
        br"\bCo-" + br"Authored-By:\s*\S+\s+(?:AI|Assistant|Agent)\b"
        br"|\bgenerated (?:with|by) an? (?:AI|assistant|agent|language model)\b",
        re.IGNORECASE,
    ),
    "private key material": re.compile(
        b"BEGIN " + br"(?:RSA |OPENSSH )?PRIVATE KEY", re.IGNORECASE
    ),
    "AWS access key": re.compile(br"AKIA[0-9A-Z]{16}"),
    "GitHub token": re.compile(br"gh[pousr]_[A-Za-z0-9]{20,}"),
    "OpenAI-style token": re.compile(b"s" + br"k-[A-Za-z0-9_-]{20,}"),
    # A consumer mailbox is a personal address wherever it appears. Institutional
    # contact addresses are a different class and are handled by EMAIL_ADDRESS below,
    # which allows them only in the files a public repository publishes them from.
    "personal mailbox": re.compile(
        br"[A-Za-z0-9._%+-]+@(?:gmail|googlemail|outlook|hotmail|live|yahoo|proton"
        br"|protonmail|icloud|gmx|web|mail)\.[A-Za-z.]{2,}",
        re.IGNORECASE,
    ),
}

#: Any address at all. An email is publishable metadata in the three files a public
#: repository conventionally carries one in, and a privacy finding everywhere else,
#: so the rule is about location rather than about a particular address.
#
# Unlike the rules above, this one is short enough to collide with random float
# bytes -- a `.npz` of physical fields contains sequences like ``4O@m.bH`` by
# chance -- so it is applied to decodable text only. Binary payloads remain covered
# by the "personal mailbox" rule, whose literal domain names do not collide.
EMAIL_ADDRESS = re.compile(
    br"[A-Za-z0-9._%+-]{2,}@[A-Za-z0-9-]{2,}(?:\.[A-Za-z0-9-]{2,})*\.[A-Za-z]{2,24}"
)
EMAIL_BEARING_FILES = frozenset({"CITATION.cff", "pyproject.toml", "SECURITY.md"})

# Reader-facing language rules.
#
# These run over decodable text only, and they describe a different failure from the
# byte rules above: not a secret, but a sentence that refers a public reader to an
# internal decision they cannot resolve -- a sentence of the form "authorized by
# <internal-decision-word> <digest>". It publishes an opaque identifier and asks the
# reader to accept that something was decided somewhere they cannot look.
#
# The hard part is not catching those; it is not catching ordinary science. This
# archive legitimately says frozen, prospective, protocol, gate and campaign, and a
# rule that fired on those would be worse than no rule, because it would train the
# author to weaken accurate scientific wording. Each pattern below therefore matches
# a phrase shape, not a word.
PUBLIC_TEXT_RULES = {
    "unresolvable decision reference": re.compile(
        br"\b(?:authoriz(?:ed|ation)|approved|sanctioned)\s+by\s+"
        br"(?:the\s+)?(?:rul" + br"ing|decision|gate|token|authority)\b"
        br"|\brul" + br"ing\s+(?:digest|record|sha)\b"
        br"|\bper\s+the\s+rul" + br"ing\b",
        re.IGNORECASE,
    ),
    "internal process self-reference": re.compile(
        br"\binternal\s+process\s+language\b"
        br"|\bprohibited-actions\s+clause\b"
        br"|\binternal\s+(?:governance|authorization|decision)\s+(?:chain|record|process)\b",
        re.IGNORECASE,
    ),
    "agent routing language": re.compile(
        br"\bsub-?agent\b|\bagent\s+hand" + br"off\b|\bhand" + br"ed\s+off\s+to\b"
        br"|\bdelegat(?:ed|ion)\s+to\s+(?:the\s+)?(?:agent|assistant|model)\b"
        br"|\borchestrat(?:or|ion)\s+(?:agent|assistant)\b",
        re.IGNORECASE,
    ),
    "unpublished venue reference": re.compile(
        br"\b(?:target|intended|submission)\s+venue\b"
        br"|\bunder\s+review\s+at\b|\bsubmitted\s+to\s+(?:the\s+)?journal\b"
        br"|\bmanuscript\s+(?:is\s+)?under\s+review\b",
        re.IGNORECASE,
    ),
}


# Capitalized top-level Markdown and YAML files are how coordination and status
# documents usually arrive. Rather than enumerate the ones we do not want, allow the
# small set a public repository legitimately has and flag anything else of that shape.
PUBLIC_CAPITALIZED_FILES = frozenset(
    {
        "README.md", "LICENSE", "LICENSE.md", "CHANGELOG.md", "CONTRIBUTING.md",
        "SECURITY.md", "CITATION.cff", "ASSET_LICENSE.md", "CODE_OF_CONDUCT.md",
        "MANIFEST.in", "MANIFEST.sha256", "NOTICE", "AUTHORS", "SHA256SUMS",
    }
)
CAPITALIZED_RECORD = re.compile(r"^[A-Z][A-Z0-9_]{2,}\.(?:md|ya?ml|txt)$")

# Checkpoint run identifiers travel inside the serialized payload, where a text
# scan of the repository will not reach them and a reader will never look. They are
# free-form strings written at training time, so they are exactly where an internal
# project, programme or venue name leaks into a public binary.
#
# Rather than enumerate forbidden words -- which would put those words in this file
# and would miss the next one -- a run identifier must be built ONLY from segments
# recognized as technical. Anything else is reported for a human to look at.
RUN_ID_DATE = re.compile(r"[0-9]{4}-[0-9]{2}-[0-9]{2}|[0-9]{8}")
RUN_ID_SEGMENT = re.compile(
    r"""^(?:
        PINN | MPF | PF | GEN | ARM[A-Z] | CASE | NO | GNN | V[0-9]+[A-Z]? | R[0-9]+
      | E[0-9]+ | N[0-9]+ | [0-9]+X[0-9]+(?:X[0-9]+)? | [0-9]+D | CUBE | VORONOI
      | SCALAR | EQUIVARIANT | PERMEQUIV | HYBRID | CONT | CURRICULUM | HORIZON
      | CONSISTENT | CONTINUATION | HARD | SOFT | DENSE | SPARSE | STRESS | BLIND
      | ID | TAIL | SIX | IC | H[0-9]+ | TBPTT[0-9]+ | EP[0-9]+ | C[0-9]+ | ORD[0-9]+
      | [0-9]+ | [A-Z]
    )$""",
    re.VERBOSE,
)


#: The shape of a run identifier wherever one may appear: an all-caps, hyphenated
#: token long enough not to be an ordinary abbreviation.
IDENTIFIER_SHAPED = re.compile(r"^[A-Z0-9]+(?:-[A-Z0-9]+){2,}$")

#: Serialized payload formats that may carry arbitrary objects, or that would mean a
#: raw training checkpoint is being distributed.
SERIALIZED_PAYLOAD_SUFFIXES = {".pt", ".pth", ".ckpt", ".pkl", ".pickle", ".joblib", ".bin"}

#: Bound on how much of a decompressed member is scanned, so a hostile archive
#: cannot turn the scanner into a decompression bomb.
MAX_MEMBER_SCAN_BYTES = 256 * 1024 * 1024

#: NumPy dtype kinds that mean a member is not plain numeric data.
NON_NUMERIC_NPY_KINDS = ("O", "U", "S", "V")


def _unknown_identifier_segments(value: str) -> list[str]:
    """Segments of an identifier that are not recognized technical tokens."""
    # Collapse dates first: splitting on "-" would shred them into bare numbers.
    stripped = RUN_ID_DATE.sub("", value)
    return sorted({part for part in stripped.split("-")
                   if part and not RUN_ID_SEGMENT.match(part)})


def _scan_identifier(kind: str, relative: Path, where: str, value: str) -> list[str]:
    unknown = _unknown_identifier_segments(value)
    if not unknown:
        return []
    return [f"unrecognized-{kind}-segment:{relative.as_posix()}{where}:{','.join(unknown)}"]


def _scan_serialized_payload(relative: Path, path: Path) -> list[str]:
    """Report identifier leakage inside a serialized payload, if one is present.

    Nothing in a public tree should be a pickled payload, and its presence is a
    finding of its own. This looks inside anyway, because a payload that is present
    is a payload whose free-form metadata can leak.
    """
    try:
        import torch  # imported lazily; the scanner must run without a model stack
    except ImportError:  # pragma: no cover - torch is a declared dependency
        return []
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
    except Exception:  # noqa: BLE001 - a payload we cannot read is not our finding here
        return []
    if not isinstance(payload, dict):
        return []
    findings: list[str] = []
    for key, value in payload.items():
        if isinstance(value, str):
            findings.extend(_scan_identifier("run-identifier", relative, f":{key}", value))
    return findings


def _npy_header_text(data: bytes) -> str:
    """The header dictionary of a .npy payload, as text, without decoding the array."""
    if len(data) < 10:
        raise ValueError("truncated array")
    if data[6] == 1:
        length = int.from_bytes(data[8:10], "little")
        start = 10
    else:
        length = int.from_bytes(data[8:12], "little")
        start = 12
    if len(data) < start + length:
        raise ValueError("truncated array header")
    return data[start:start + length].decode("latin-1")


def _npy_header_findings(relative: Path, member: str, data: bytes) -> list[str]:
    if not data.startswith(b"\x93NUMPY"):
        return []
    try:
        header = _npy_header_text(data)
    except ValueError:
        return [f"unreadable-array-header:{relative.as_posix()}:{member}"]
    findings: list[str] = []
    descr = re.search(r"'descr':\s*(?P<quote>['\"])(?P<value>[^'\"]*)(?P=quote)", header)
    if descr is None:
        findings.append(f"structured-or-missing-array-dtype:{relative.as_posix()}:{member}")
    elif any(kind in descr.group("value") for kind in NON_NUMERIC_NPY_KINDS):
        findings.append(
            f"non-numeric-array-dtype:{relative.as_posix()}:{member}:{descr.group('value')}"
        )
    return findings


#: How deep container nesting is followed. A compressed archive stored inside a
#: compressed member hides its contents from a scan of the outer file's bytes, so
#: nesting is followed rather than trusted. The bound stops a hostile archive from
#: nesting indefinitely.
MAX_CONTAINER_DEPTH = 4


def _scan_container(relative: Path, path: Path | None, *, blob: bytes | None = None,
                    depth: int = 0) -> list[str]:
    """Open a ZIP-shaped file or blob: member names, metadata, decompressed contents."""
    findings: list[str] = []
    if depth > MAX_CONTAINER_DEPTH:
        return [f"container-nested-too-deeply:{relative.as_posix()}"]
    source: Path | io.BytesIO = io.BytesIO(blob) if blob is not None else path
    try:
        with zipfile.ZipFile(source) as archive:
            if archive.comment:
                findings.append(f"archive-comment:{relative.as_posix()}")
            for info in archive.infolist():
                member = info.filename
                if info.comment:
                    findings.append(f"archive-member-comment:{relative.as_posix()}:{member}")
                if info.extra:
                    findings.append(f"archive-member-extra-field:{relative.as_posix()}:{member}")
                encoded = member.encode("utf-8", "ignore")
                for label, pattern in BYTE_RULES.items():
                    if pattern.search(encoded):
                        findings.append(f"{label}:{relative.as_posix()}:{member}")
                stem = member[:-4] if member.endswith(".npy") else member
                for token in re.split(r"[/_.]", stem):
                    if IDENTIFIER_SHAPED.match(token):
                        findings.extend(
                            _scan_identifier("archive-member-identifier", relative,
                                             f":{member}", token)
                        )
                if info.file_size > MAX_MEMBER_SCAN_BYTES:
                    findings.append(f"member-too-large-to-scan:{relative.as_posix()}:{member}")
                    continue
                try:
                    data = archive.read(member)
                except Exception:  # noqa: BLE001 - unreadable member, reported as such
                    findings.append(f"unreadable-archive-member:{relative.as_posix()}:{member}")
                    continue
                findings.extend(_npy_header_findings(relative, member, data))
                for label, pattern in BYTE_RULES.items():
                    if pattern.search(data):
                        findings.append(f"{label}:{relative.as_posix()}:{member}")
                if data[:4] == b"PK\x03\x04":
                    findings.extend(
                        _scan_container(Path(f"{relative.as_posix()}:{member}"), None,
                                        blob=data, depth=depth + 1)
                    )
    except zipfile.BadZipFile:
        findings.append(f"unreadable-archive:{relative.as_posix()}")
    return findings


def _decoded_views(data: bytes) -> list[tuple[str, bytes]]:
    """Alternative byte views of the same content, so a re-encoding cannot hide text.

    A byte rule is written against UTF-8/ASCII. The same sentence stored as UTF-16
    matches nothing, and so does the same sentence stored gzip-compressed. Both are
    ordinary file formats rather than exotic attacks, so both are decoded and scanned
    as well as the raw bytes. Content-level encodings that are not file formats --
    base64, hex, homoglyphs -- are out of reach of any byte scanner and are declared
    as a limitation rather than pretended away.
    """

    views: list[tuple[str, bytes]] = []
    if data[:2] == b"\x1f\x8b":                      # gzip, including .tar.gz
        try:
            views.append(("gzip", gzip.decompress(data)[:MAX_MEMBER_SCAN_BYTES]))
        except (OSError, EOFError, zlib.error):
            views.append(("gzip", b""))
    if data[:8] == b"\x89PNG\r\n\x1a\n":
        views.extend(_png_text_chunks(data))
    for label, encoding in (("utf-16", "utf-16"), ("utf-16-be", "utf-16-be")):
        if b"\x00" not in data[:4096]:
            break
        try:
            views.append((label, data.decode(encoding).encode("utf-8", "ignore")))
        except (UnicodeDecodeError, UnicodeEncodeError):
            continue
    return views


def _png_text_chunks(data: bytes) -> list[tuple[str, bytes]]:
    """Text chunks of a PNG, including the compressed ones.

    ``tEXt`` is plain, ``zTXt`` is zlib-compressed and ``iTXt`` may be either. All
    three are where authoring tools write paths, usernames and timestamps.
    """

    views: list[tuple[str, bytes]] = []
    offset = 8
    while offset + 8 <= len(data):
        length = int.from_bytes(data[offset:offset + 4], "big")
        kind = data[offset + 4:offset + 8]
        body = data[offset + 8:offset + 8 + length]
        offset += 12 + length
        if kind == b"IEND" or length > len(data):
            break
        if kind == b"tEXt":
            views.append(("png-tEXt", body))
        elif kind == b"zTXt":
            keyword, _, rest = body.partition(b"\0")
            try:
                views.append(("png-zTXt", keyword + b" " + zlib.decompress(rest[1:])))
            except zlib.error:
                views.append(("png-zTXt", keyword))
        elif kind == b"iTXt":
            parts = body.split(b"\0", 5)
            if len(parts) >= 6 and parts[1:2] == [b"\x01"]:
                try:
                    views.append(("png-iTXt", parts[0] + b" " + zlib.decompress(parts[5])))
                except zlib.error:
                    views.append(("png-iTXt", parts[0]))
            else:
                views.append(("png-iTXt", body.replace(b"\0", b" ")))
    return views


def _scan_tar(relative: Path, view_label: str, data: bytes) -> list[str]:
    """Member names and contents of a tar, once something has been decompressed."""
    if len(data) < 512 or data[257:262] != b"ustar":
        return []
    findings: list[str] = []
    try:
        with tarfile.open(fileobj=io.BytesIO(data)) as archive:
            for member in archive.getmembers():
                where = f"{relative.as_posix()}:{view_label}:{member.name}"
                if not member.isfile():
                    findings.append(f"tar-non-regular-member:{where}")
                    continue
                if member.size > MAX_MEMBER_SCAN_BYTES:
                    findings.append(f"member-too-large-to-scan:{where}")
                    continue
                handle = archive.extractfile(member)
                payload = member.name.encode() + b"\n" + (handle.read() if handle else b"")
                for label, pattern in BYTE_RULES.items():
                    if pattern.search(payload):
                        findings.append(f"{label}:{where}")
                findings.extend(_scan_email(Path(where), payload))
    except (tarfile.TarError, EOFError):
        findings.append(f"unreadable-tar:{relative.as_posix()}:{view_label}")
    return findings


def _scan_public_language(relative: Path, data: bytes) -> list[str]:
    """Reader-facing language rules, over decodable text only.

    Applied to text because these are sentences, not secrets: a rule this loose run
    over compressed bytes would produce collisions, and the thing being looked for
    cannot hide in a float array anyway.
    """
    try:
        data.decode("utf-8")
    except UnicodeDecodeError:
        return []
    findings = []
    for label, pattern in PUBLIC_TEXT_RULES.items():
        match = pattern.search(data)
        if match:
            phrase = match.group(0).decode("utf-8", "replace")
            findings.append(f"{label}:{relative.as_posix()}:{phrase}")
    return findings


def _scan_email(relative: Path, data: bytes) -> list[str]:
    """An address outside the files a public repository publishes contact details in.

    Text only: see the note on ``EMAIL_ADDRESS``. A binary payload is covered by the
    "personal mailbox" byte rule instead, which cannot collide.
    """
    if relative.name in EMAIL_BEARING_FILES:
        return []
    try:
        data.decode("utf-8")
    except UnicodeDecodeError:
        return []
    found = sorted({match.group(0).decode("utf-8", "replace")
                    for match in EMAIL_ADDRESS.finditer(data)})
    return [f"email-address-outside-contact-metadata:{relative.as_posix()}:{address}"
            for address in found]


def _scan_structured_record(relative: Path, data: bytes) -> list[str]:
    """Scan decoded JSON: every string value, for identifier-shaped tokens."""
    try:
        document = json.loads(data.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return [f"unparsable-json:{relative.as_posix()}"]
    findings: list[str] = []

    def walk(node: object, where: str) -> None:
        if isinstance(node, dict):
            for key, value in node.items():
                walk(key, where)
                walk(value, f"{where}.{key}" if isinstance(key, str) else where)
        elif isinstance(node, list):
            for item in node:
                walk(item, where)
        elif isinstance(node, str):
            for token in re.split(r"[\s/_,;:()\[\]]+", node):
                if IDENTIFIER_SHAPED.match(token):
                    findings.extend(
                        _scan_identifier("record-identifier", relative, where, token)
                    )

    walk(document, "")
    return findings
# Role words that mark a coordination or governance document wherever it sits.
ROLE_MARKED_FILENAME = re.compile(
    r"HAND" r"OFF|STA" r"TUS|COORD" r"INATION|RA" r"TIFICATION|ADJU" r"DICATION"
    r"|GATE_" r"REVIEW|REVIEW_" r"INTEGRATION",
    re.IGNORECASE,
)


def _dotted_name(node: ast.AST) -> str | None:
    parts: list[str] = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        parts.append(node.id)
        return ".".join(reversed(parts))
    return None


def _keyword_is_literal(call: ast.Call, name: str, expected: bool) -> bool:
    for keyword in call.keywords:
        if keyword.arg == name:
            return isinstance(keyword.value, ast.Constant) and keyword.value.value is expected
    return False


def _scan_python(relative: Path, data: bytes) -> list[str]:
    try:
        tree = ast.parse(data.decode("utf-8"), filename=str(relative))
    except (SyntaxError, UnicodeDecodeError) as exc:
        return [f"invalid-python:{relative.as_posix()}:{exc.__class__.__name__}"]
    aliases: dict[str, str] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for entry in node.names:
                aliases[entry.asname or entry.name.split(".")[0]] = entry.name
        elif isinstance(node, ast.ImportFrom) and node.module:
            for entry in node.names:
                aliases[entry.asname or entry.name] = f"{node.module}.{entry.name}"

    def resolved_name(node: ast.AST) -> str | None:
        name = _dotted_name(node)
        if name is None:
            return None
        head, separator, tail = name.partition(".")
        replacement = aliases.get(head, head)
        return replacement + (separator + tail if separator else "")

    findings: list[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        name = resolved_name(node.func)
        if name == "torch.load" and not _keyword_is_literal(node, "weights_only", True):
            findings.append(f"unsafe-torch-load:{relative.as_posix()}:{node.lineno}")
        if name in {"pickle.load", "pickle.loads", "joblib.load"}:
            findings.append(f"unsafe-pickle-loader:{relative.as_posix()}:{node.lineno}")
        if name in {"np.load", "numpy.load"} and not _keyword_is_literal(
            node, "allow_pickle", False
        ):
            findings.append(f"numpy-pickle-policy-not-explicit:{relative.as_posix()}:{node.lineno}")
    return findings


def scan_file(root: Path, relative: Path) -> list[str]:
    """Every content rule, applied to one file. Shared by both scan modes."""
    path = root / relative
    findings: list[str] = []
    name = path.name
    if path.is_symlink():
        return [f"symlink:{relative.as_posix()}"]
    if name == ".env" or name.startswith(".env."):
        findings.append(f"environment-secret-file:{relative.as_posix()}")
    if name in SECRET_FILENAMES or name.endswith((".pem", ".key")):
        findings.append(f"secret-or-editor-file:{relative.as_posix()}")
    # An all-caps record at the tree root is where coordination documents land;
    # inside docs/ the same shape is ordinary documentation.
    at_root = relative.parent == Path(".")
    if (
        at_root
        and CAPITALIZED_RECORD.match(name)
        and name not in PUBLIC_CAPITALIZED_FILES
    ) or ROLE_MARKED_FILENAME.search(name):
        findings.append(f"coordination-record-filename:{relative.as_posix()}")
    if NATIVE_128_PATH.search(relative.as_posix()):
        findings.append(f"native-128-cubed-path:{relative.as_posix()}")
    for token in re.split(r"[/_.]", relative.as_posix()):
        if IDENTIFIER_SHAPED.match(token):
            findings.extend(_scan_identifier("path-identifier", relative, "", token))
    if path.stat().st_size > MAX_FILE_BYTES:
        findings.append(f"file-over-90MiB:{relative.as_posix()}")
    suffix = path.suffix.lower()
    if suffix in SERIALIZED_PAYLOAD_SUFFIXES:
        # A public tree must not distribute a serialized object payload at all; a
        # training checkpoint is the usual way one arrives.
        findings.append(f"serialized-object-payload:{relative.as_posix()}")
        findings.extend(_scan_serialized_payload(relative, path))
    data = path.read_bytes()
    for label, pattern in BYTE_RULES.items():
        if pattern.search(data):
            findings.append(f"{label}:{relative.as_posix()}")
    findings.extend(_scan_email(relative, data))
    findings.extend(_scan_public_language(relative, data))
    for view_label, view in _decoded_views(data):
        for label, pattern in BYTE_RULES.items():
            if pattern.search(view):
                findings.append(f"{label}:{relative.as_posix()}:{view_label}")
        findings.extend(f"{item}:{view_label}" for item in _scan_email(relative, view))
        if view[:4] == b"PK\x03\x04":
            findings.extend(
                _scan_container(Path(f"{relative.as_posix()}:{view_label}"), None, blob=view)
            )
        findings.extend(_scan_tar(relative, view_label, view))
    if data[:4] == b"PK\x03\x04":
        findings.extend(_scan_container(relative, path))
    if suffix == ".json":
        findings.extend(_scan_structured_record(relative, data))
    if suffix == ".py":
        findings.extend(_scan_python(relative, data))
    return findings


def scan_payload(root: Path) -> list[str]:
    """Scan exactly the files the manifest defines as the release payload.

    This is the reader-facing mode. It describes the archive, so it is unaffected by
    anything the reader's own commands have produced in the working tree: an
    editable install, a reproduction run, a bytecode cache. Those are not archive
    members and never were.

    Ignoring them here cannot let them into a release. What ships is decided by
    ``release_files``, which walks the tree and excludes generated material, and the
    manifest is built from that. As a backstop this mode also refuses a manifest that
    lists generated material at all.
    """
    root = root.resolve()
    findings: list[str] = []
    payload = manifest_payload(root)
    for relative in payload:
        if is_generated(relative):
            findings.append(f"generated-file-listed-in-manifest:{relative.as_posix()}")
            continue
        path = root / relative
        if not path.exists():
            findings.append(f"manifest-lists-a-missing-file:{relative.as_posix()}")
            continue
        findings.extend(scan_file(root, relative))
    return sorted(set(findings))


def scan_tree(root: Path, *, allow_packaging_metadata: bool = False) -> list[str]:
    """Walk the whole tree and refuse anything that does not belong in a release.

    This is the construction-time mode: it is what a staging tree must pass before
    it is packaged, and it treats generated material as a finding rather than
    ignoring it.
    """
    findings: list[str] = []
    root = root.resolve()
    for current, directories, files in os.walk(root, followlinks=False):
        current_path = Path(current)
        kept_directories: list[str] = []
        for name in sorted(directories):
            path = current_path / name
            relative = path.relative_to(root)
            if name == ".git":
                continue
            if path.is_symlink():
                findings.append(f"symlink:{relative.as_posix()}")
                continue
            if name.endswith(".egg-info") and allow_packaging_metadata:
                continue
            if name in GENERATED_NAMES or name.endswith(".egg-info"):
                findings.append(f"generated-directory:{relative.as_posix()}")
                continue
            if NATIVE_128_PATH.search(relative.as_posix()):
                findings.append(f"native-128-cubed-path:{relative.as_posix()}")
            kept_directories.append(name)
        directories[:] = kept_directories

        for name in sorted(files):
            relative = (current_path / name).relative_to(root)
            findings.extend(scan_file(root, relative))
    return sorted(set(findings))


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Scan a PINN-Phase release for private, unsafe or generated material."
    )
    parser.add_argument("root", nargs="?", type=Path, default=Path.cwd())
    parser.add_argument(
        "--mode",
        choices=("payload", "staging"),
        default="payload",
        help="payload (default): scan exactly the files MANIFEST.sha256 defines, which "
             "is what the archive contains. staging: walk the whole tree and refuse "
             "generated material; use this when constructing a release.",
    )
    parser.add_argument("--allow-packaging-metadata", action="store_true",
                        help="staging mode only: tolerate *.egg-info from a local install")
    args = parser.parse_args()

    if args.mode == "payload":
        try:
            findings = scan_payload(args.root)
        except FileNotFoundError as exc:
            print(f"Public payload scan failed: {exc}")
            return 1
        label, scope = "payload", "the files MANIFEST.sha256 defines"
    else:
        findings = scan_tree(args.root, allow_packaging_metadata=args.allow_packaging_metadata)
        label, scope = "staging", "every file in the tree"

    if findings:
        print(f"Public-tree scan failed ({label} mode, {scope}):")
        for finding in findings:
            print(f"- {finding}")
        return 1
    print(f"Public-tree scan passed ({label} mode, {scope})")
    if args.mode == "payload":
        print("Local working files are not archive members and were not scanned:")
        for name, why in sorted(LOCAL_WORKING_ARTEFACTS.items()):
            print(f"  {name:16s} {why}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
