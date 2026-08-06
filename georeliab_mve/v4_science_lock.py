"""Immutable governance and admission boundary for GeoReliab v4.

Task 1 of the v4 protocol intentionally does not implement experiments,
metrics, or GPU execution.  It freezes the claim and rejects scientific
evidence that is not explicitly bound to the v4 protocol hash.
"""

from __future__ import annotations

import math
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from . import toml_compat as tomllib
from .science_lock import sha256_canonical_text


V4_BASE_COMMIT = "e6c7c8ec22f3941aa18f50a667b026ba0718bf80"
V4_PROTOCOL_ID = "georeliab-v4-ranking-warning"
V4_PROTOCOL_VERSION = "4.0"
V4_PROTOCOL_SCHEMA_VERSION = "georeliab-v4-protocol-1.0"
V4_PROTOCOL_PROVENANCE_SCHEMA_VERSION = (
    "georeliab-v4-protocol-provenance-1.0"
)
V4_SCIENTIFIC_BUNDLE_SCHEMA_VERSION = "georeliab-v4-scientific-bundle-1.0"
V4_ARTIFACT_RECORD_SCHEMA_VERSION = "georeliab-v4-artifact-record-1.0"
V4_RECORD_ORIGIN_SCHEMA_VERSION = "georeliab-v4-record-origin-1.0"
V4_HASH_ALGORITHM = "sha256-canonical-lf-v1"

GEORELIAB_V4_PROTOCOL_READY = "GEORELIAB_V4_PROTOCOL_READY"
GPU_SELECTION_REQUIRED = "GPU_SELECTION_REQUIRED"
NO_SCIENTIFIC_RESULT = "NO_SCIENTIFIC_RESULT"
SUPERSEDED_BY_PRIOR_ART_CHANGE = "SUPERSEDED_BY_PRIOR_ART_CHANGE"

V4_PROTOCOL_RELATIVE_PATH = "configs/georeliab_v4_protocol.toml"
V4_PROTOCOL_SHA256 = (
    "3765955b255e12fb91eaabf7b91d8a5cb975cbc3afccdd5d20f8038594b74fcd"
)
V4_LOCKED_SCIENCE_FILES: Mapping[str, str] = {
    V4_PROTOCOL_RELATIVE_PATH: V4_PROTOCOL_SHA256,
    "docs/GEORELIAB_V4_PROTOCOL.md": (
        "9572c0ae926bf6370b1ab514fa6ae497e72cbc3f0d93f4f9f02300231ae99095"
    ),
}

# These are the canonical-LF digests of the unchanged v1 inputs at the v4 base.
# The Task 1 regression suite also compares their Git blobs byte-for-byte.
V1_IMMUTABLE_SCIENCE_FILES: Mapping[str, str] = {
    "configs/dual_mve_protocol.toml": (
        "91eddc87c3bfdf2a6de413174dc758163c11d7c37e224b1deb4ac23288fafb6c"
    ),
    "configs/a100_real_mve_overlay.toml": (
        "c65ef97684adeda6b1bba8d8c152eb559df44367bc35973f549d7e8049136011"
    ),
    "georeliab_mve/science_lock.py": (
        "ff7ea5dca83a6f8373f3585cf62e037c9b31f6a64c4637acb4056ccbf0d99667"
    ),
    "georeliab_mve/splits.py": (
        "a270f114a649941abed5123a918e238273b8bc63ee9b9492509d428a6081a213"
    ),
    "georeliab_mve/preparation_round2.py": (
        "304e60b50368354fa9623f7145832c140fb45a2d663a26c19f638c22119d3359"
    ),
    "georeliab_mve/gates.py": (
        "cb77a1c37f30e813d86097722874cda118b34aa9c9d0bfa084d0950d752d5c51"
    ),
}

_EXPECTED_STATUS = {
    "protocol_status": GEORELIAB_V4_PROTOCOL_READY,
    "execution_status": GPU_SELECTION_REQUIRED,
    "scientific_result_status": NO_SCIENTIFIC_RESULT,
}
_EXPECTED_ROUTE = {
    "v4": "SOLE_CVPR_2027_MAIN_LINE",
    "v1": SUPERSEDED_BY_PRIOR_ART_CHANGE,
    "v1_scientific_result": NO_SCIENTIFIC_RESULT,
    "v1_fail_classification_forbidden": True,
    "geometry_causal_audit": "INDEPENDENT_BACKLOG_NON_BLOCKING",
    "deformable_world": "INDEPENDENT_BACKLOG_NON_BLOCKING",
}
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_SCHEMA_MARKER_KEYS = (
    "schema_version",
    "artifact_schema_version",
    "evidence_schema_version",
)
_PREDICTION_ARTIFACT_KEYS = frozenset(
    {
        "run_id",
        "sample_key",
        "geometry_prediction_uri",
        "native_confidence_uri",
        "valid_mask_uri",
    }
)
_AUDIT_RECORD_KEYS = frozenset(
    {
        "run_id",
        "sample_key",
        "failure_label",
        "selection_score",
        "coverage",
        "accepted",
        "downstream_outcome",
    }
)
_STAGE_EVIDENCE_KEYS = frozenset({"scientific_validity", "bundle_index"})
_V4_MARKERS: Mapping[str, object] = {
    "project_line": "v4",
    "main_line": "v4",
    "protocol_id": V4_PROTOCOL_ID,
    "protocol_version": V4_PROTOCOL_VERSION,
    "protocol_sha256": V4_PROTOCOL_SHA256,
}
_V4_PROJECT_ROUTES = frozenset({"v4", "SOLE_CVPR_2027_MAIN_LINE"})
_V4_BUNDLE_KEYS = frozenset(
    {
        "schema_version",
        "project_line",
        "scientific_validity",
        "protocol_provenance",
        "artifacts",
    }
)
_V4_ENVELOPE_KEYS = frozenset(
    {"project_line", "protocol_provenance", "artifact"}
)
_V4_RECORD_REQUIRED_KEYS = frozenset(
    {
        "schema_version",
        "record_kind",
        "origin",
        "source_uri",
        "source_sha256",
        "data",
    }
)
_V4_RECORD_OPTIONAL_KEYS = frozenset({"source_schema_version"})


class V4ScienceLockError(RuntimeError):
    """Raised when v4 governance or evidence provenance fails closed."""


def _resolve_locked_file(source_root: Path, relative: str) -> Path:
    root = source_root.resolve()
    path = (root / relative).resolve()
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise V4ScienceLockError(
            f"v4 science lock escaped source root: {relative}"
        ) from exc
    if not path.is_file():
        raise V4ScienceLockError(f"missing science-locked file: {relative}")
    return path


def _validate_locked_files(
    source_root: Path,
    locked_files: Mapping[str, str],
    *,
    label: str,
) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for relative, expected in locked_files.items():
        path = _resolve_locked_file(source_root, relative)
        actual = sha256_canonical_text(path)
        rows.append(
            {
                "path": relative,
                "sha256": actual,
                "expected_sha256": expected,
            }
        )
        if actual != expected:
            raise V4ScienceLockError(
                f"{label} mismatch for {relative}: {actual} != {expected}"
            )
    return rows


def _load_protocol(source_root: Path) -> dict[str, Any]:
    path = _resolve_locked_file(source_root, V4_PROTOCOL_RELATIVE_PATH)
    try:
        raw = tomllib.loads(path.read_text(encoding="utf-8"))
    except (OSError, tomllib.TOMLDecodeError) as exc:
        raise V4ScienceLockError(f"cannot load v4 protocol: {exc}") from exc
    if not isinstance(raw, dict):
        raise V4ScienceLockError("v4 protocol root must be a table")
    return raw


def _validate_protocol_semantics(protocol: Mapping[str, Any]) -> None:
    expected = {
        "schema_version": V4_PROTOCOL_SCHEMA_VERSION,
        "protocol_id": V4_PROTOCOL_ID,
        "protocol_version": V4_PROTOCOL_VERSION,
        "main_line": "v4",
        "sole_main_line": True,
        "official_deadline": "TBA",
        **_EXPECTED_STATUS,
    }
    drift = [
        f"{key}: configured={protocol.get(key)!r}, frozen={value!r}"
        for key, value in expected.items()
        if protocol.get(key) != value
    ]
    if protocol.get("route") != _EXPECTED_ROUTE:
        drift.append("route semantics changed")
    mve = protocol.get("mve")
    if not isinstance(mve, Mapping):
        drift.append("mve table missing")
    else:
        if mve.get("scientific_unit_count") != 400:
            drift.append("scientific unit count changed")
        if mve.get("lighting_axis") != "UNORDERED_DISCRETE":
            drift.append("DTU lighting ordering changed")
        if mve.get("boundary_lag_legal_sequence") != [
            "L3",
            "fog-s1",
            "fog-s2",
            "fog-s3",
        ]:
            drift.append("Boundary Lag sequence changed")
    if drift:
        raise V4ScienceLockError("v4 protocol semantic drift: " + "; ".join(drift))


def load_v4_protocol(source_root: Path) -> dict[str, Any]:
    """Load the protocol only after its immutable lock has passed."""

    validate_v4_science_lock(source_root)
    return _load_protocol(source_root)


def validate_v4_science_lock(source_root: Path) -> dict[str, object]:
    """Validate v4 files, unchanged v1 inputs, and frozen status semantics."""

    v4_rows = _validate_locked_files(
        source_root, V4_LOCKED_SCIENCE_FILES, label="v4 science lock"
    )
    v1_rows = _validate_locked_files(
        source_root, V1_IMMUTABLE_SCIENCE_FILES, label="v1 immutability lock"
    )
    protocol = _load_protocol(source_root)
    _validate_protocol_semantics(protocol)
    return {
        "schema_version": "georeliab-v4-science-lock-1.0",
        "base_commit": V4_BASE_COMMIT,
        "protocol_id": V4_PROTOCOL_ID,
        "protocol_version": V4_PROTOCOL_VERSION,
        "protocol_sha256": V4_PROTOCOL_SHA256,
        "hash_algorithm": V4_HASH_ALGORITHM,
        "status": GEORELIAB_V4_PROTOCOL_READY,
        "execution_status": GPU_SELECTION_REQUIRED,
        "scientific_result_status": NO_SCIENTIFIC_RESULT,
        "v1_status": SUPERSEDED_BY_PRIOR_ART_CHANGE,
        "v1_scientific_result_status": NO_SCIENTIFIC_RESULT,
        "files": v4_rows,
        "unchanged_v1_files": v1_rows,
    }


def v4_protocol_provenance(source_root: Path) -> dict[str, str]:
    """Return the exact provenance every v4 scientific envelope must carry."""

    validate_v4_science_lock(source_root)
    return {
        "schema_version": V4_PROTOCOL_PROVENANCE_SCHEMA_VERSION,
        "protocol_id": V4_PROTOCOL_ID,
        "protocol_version": V4_PROTOCOL_VERSION,
        "protocol_sha256": V4_PROTOCOL_SHA256,
    }


def v4_record_origin(source_root: Path) -> dict[str, object]:
    """Return the exact origin every admitted v4 record must carry."""

    return {
        "schema_version": V4_RECORD_ORIGIN_SCHEMA_VERSION,
        "project_line": "v4",
        "protocol_provenance": v4_protocol_provenance(source_root),
    }


def _require_v4_provenance(
    value: object,
    *,
    expected: Mapping[str, str],
    location: str,
) -> None:
    if not isinstance(value, Mapping) or dict(value) != dict(expected):
        raise V4ScienceLockError(
            f"{location} is not bound to the immutable v4 protocol"
        )


def _require_v4_record_origin(
    value: object,
    *,
    expected: Mapping[str, object],
    location: str,
) -> None:
    if not isinstance(value, Mapping) or dict(value) != dict(expected):
        raise V4ScienceLockError(
            f"{location} is not bound to the exact v4 record origin"
        )


def _require_closed_schema(
    value: Mapping[object, object],
    *,
    required: frozenset[str],
    location: str,
    optional: frozenset[str] = frozenset(),
) -> None:
    actual = set(value)
    non_string = {key for key in actual if not isinstance(key, str)}
    string_keys = {key for key in actual if isinstance(key, str)}
    missing = required - string_keys
    unexpected = string_keys - required - optional
    if not missing and not unexpected and not non_string:
        return

    reasons: list[str] = []
    if missing:
        reasons.append(
            "missing keys " + ", ".join(sorted(repr(key) for key in missing))
        )
    if unexpected:
        reasons.append(
            "unexpected keys "
            + ", ".join(sorted(repr(key) for key in unexpected))
        )
    if non_string:
        reasons.append(
            "non-JSON keys "
            + ", ".join(sorted(repr(key) for key in non_string))
        )
    raise V4ScienceLockError(
        f"{location} closed schema violation: {'; '.join(reasons)}"
    )


def _path_digest_pairs(payload: Mapping[object, object]) -> set[tuple[str, str]]:
    keys = {key for key in payload if isinstance(key, str)}
    pairs: set[tuple[str, str]] = set()
    for location_key in ("path", "uri"):
        for digest_key in ("sha256", "digest"):
            if {location_key, digest_key} <= keys:
                pairs.add((location_key, digest_key))
    for key in keys:
        for suffix in ("_path", "_uri"):
            if key.endswith(suffix):
                for digest_suffix in ("_sha256", "_digest"):
                    digest_key = f"{key[:-len(suffix)]}{digest_suffix}"
                    if digest_key in keys:
                        pairs.add((key, digest_key))
    return pairs


def _reject_non_v4_nested_content(
    value: object,
    *,
    location: str,
    allow_source_binding: bool = False,
    active_ids: set[int] | None = None,
) -> None:
    """Recursively reject raw historical objects and non-v4 markers."""

    if isinstance(value, Mapping):
        if active_ids is None:
            active_ids = set()
        value_id = id(value)
        if value_id in active_ids:
            raise V4ScienceLockError(f"{location} contains a cyclic mapping")
        active_ids.add(value_id)
        try:
            non_string_keys = [
                key for key in value if not isinstance(key, str)
            ]
            if non_string_keys:
                names = ", ".join(
                    sorted(repr(key) for key in non_string_keys)
                )
                raise V4ScienceLockError(
                    f"{location} contains non-JSON mapping keys: {names}"
                )

            for schema_marker in _SCHEMA_MARKER_KEYS:
                schema = value.get(schema_marker)
                if schema is not None and (
                    not isinstance(schema, str)
                    or not schema.startswith("georeliab-v4-")
                ):
                    raise V4ScienceLockError(
                        f"{location} contains legacy schema marker "
                        f"{schema!r} in {schema_marker}; raw historical "
                        "runtime objects are forbidden"
                    )

            for marker, expected in _V4_MARKERS.items():
                configured = value.get(marker)
                if configured is not None and configured != expected:
                    raise V4ScienceLockError(
                        f"{location} contains non-v4 {marker} {configured!r}"
                    )
            for route_marker in ("project_route", "route"):
                project_route = value.get(route_marker)
                if (
                    project_route is not None
                    and (
                        not isinstance(project_route, str)
                        or project_route not in _V4_PROJECT_ROUTES
                    )
                ):
                    raise V4ScienceLockError(
                        f"{location} contains non-v4 {route_marker} "
                        f"{project_route!r}"
                    )

            keys = frozenset(
                key for key in value if isinstance(key, str)
            )
            legacy_kind = None
            if _PREDICTION_ARTIFACT_KEYS <= keys:
                legacy_kind = "PredictionArtifact"
            elif _AUDIT_RECORD_KEYS <= keys:
                legacy_kind = "AuditRecord"
            elif _STAGE_EVIDENCE_KEYS <= keys:
                legacy_kind = "stage evidence"
            if legacy_kind is not None:
                raise V4ScienceLockError(
                    f"{location} contains raw legacy {legacy_kind}; "
                    "a new v4 artifact/evidence record is required"
                )

            path_pairs = _path_digest_pairs(value)
            allowed_pairs = (
                {("source_uri", "source_sha256")}
                if allow_source_binding
                else set()
            )
            unexpected_pairs = path_pairs - allowed_pairs
            if unexpected_pairs:
                names = ", ".join(
                    f"{path_key}/{digest_key}"
                    for path_key, digest_key in sorted(unexpected_pairs)
                )
                raise V4ScienceLockError(
                    f"{location} contains legacy path/digest wrapper "
                    f"({names}); raw evidence indexes are forbidden"
                )

            for key, nested in value.items():
                _reject_non_v4_nested_content(
                    nested,
                    location=f"{location}.{key}",
                    active_ids=active_ids,
                )
        finally:
            active_ids.remove(value_id)
        return

    if isinstance(value, Sequence) and not isinstance(
        value, (str, bytes, bytearray)
    ):
        if not isinstance(value, list):
            raise V4ScienceLockError(
                f"{location} contains non-JSON sequence type "
                f"{type(value).__name__}"
            )
        if active_ids is None:
            active_ids = set()
        value_id = id(value)
        if value_id in active_ids:
            raise V4ScienceLockError(f"{location} contains a cyclic sequence")
        active_ids.add(value_id)
        try:
            for index, nested in enumerate(value):
                _reject_non_v4_nested_content(
                    nested,
                    location=f"{location}[{index}]",
                    active_ids=active_ids,
                )
        finally:
            active_ids.remove(value_id)
        return

    if value is None or isinstance(value, (str, bool, int)):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise V4ScienceLockError(
                f"{location} contains non-JSON non-finite number {value!r}"
            )
        return
    raise V4ScienceLockError(
        f"{location} contains non-JSON value of type "
        f"{type(value).__name__}"
    )


def _validate_v4_artifact_record(
    value: object,
    *,
    expected_origin: Mapping[str, object],
    location: str,
) -> None:
    if not isinstance(value, Mapping):
        raise V4ScienceLockError(
            f"{location} requires a new v4 artifact/evidence record"
        )
    _require_closed_schema(
        value,
        required=_V4_RECORD_REQUIRED_KEYS,
        optional=_V4_RECORD_OPTIONAL_KEYS,
        location=f"{location} artifact record",
    )
    if value.get("schema_version") != V4_ARTIFACT_RECORD_SCHEMA_VERSION:
        raise V4ScienceLockError(
            f"{location} requires a new v4 artifact/evidence record"
        )
    record_kind = value.get("record_kind")
    if (
        not isinstance(record_kind, str)
        or record_kind not in {"artifact", "evidence"}
    ):
        raise V4ScienceLockError(
            f"{location} record_kind must be 'artifact' or 'evidence'"
        )
    _require_v4_record_origin(
        value.get("origin"),
        expected=expected_origin,
        location=f"{location} origin",
    )
    source_uri = value.get("source_uri")
    if not isinstance(source_uri, str) or not source_uri.strip():
        raise V4ScienceLockError(
            f"{location} requires a non-empty source_uri"
        )
    source_sha256 = value.get("source_sha256")
    if (
        not isinstance(source_sha256, str)
        or _SHA256_RE.fullmatch(source_sha256) is None
    ):
        raise V4ScienceLockError(
            f"{location} requires a lowercase source_sha256"
        )
    source_schema = value.get("source_schema_version")
    if source_schema is not None and (
        not isinstance(source_schema, str) or not source_schema.strip()
    ):
        raise V4ScienceLockError(
            f"{location} source_schema_version must be a non-empty string"
        )
    data = value.get("data")
    if not isinstance(data, (Mapping, Sequence)) or isinstance(
        data, (str, bytes, bytearray)
    ):
        raise V4ScienceLockError(
            f"{location} data must be a mapping or sequence"
        )
    _reject_non_v4_nested_content(
        value,
        location=location,
        allow_source_binding=True,
    )


def validate_v4_scientific_bundle_structure(
    source_root: Path,
    bundle: Mapping[str, Any],
) -> dict[str, object]:
    """Validate v4 envelope structure without granting scientific admission.

    Existing v1 artifact schemas remain readable by the historical code, but a
    raw v1 artifact/evidence object cannot pass this boundary.  Historical
    source bytes may be reused only through a new v4 record that binds their
    URI and digest to the exact v4 record origin and protocol hash. A
    structural success is never a scientific PASS; only the canonical v4
    finalizer may admit an exact 400-record bundle.
    """

    if not isinstance(bundle, Mapping):
        raise V4ScienceLockError("v4 scientific bundle must be a mapping")
    _require_closed_schema(
        bundle,
        required=_V4_BUNDLE_KEYS,
        location="v4 scientific bundle",
    )
    expected = v4_protocol_provenance(source_root)
    expected_origin = {
        "schema_version": V4_RECORD_ORIGIN_SCHEMA_VERSION,
        "project_line": "v4",
        "protocol_provenance": expected,
    }
    if bundle.get("schema_version") != V4_SCIENTIFIC_BUNDLE_SCHEMA_VERSION:
        raise V4ScienceLockError(
            "v4 scientific bundle schema missing or mismatched; "
            "superseded v1 evidence is not admissible"
        )
    if bundle.get("project_line") != "v4":
        raise V4ScienceLockError(
            "v4 scientific bundle project_line must be exactly 'v4'"
        )
    if bundle.get("scientific_validity") != "SCIENTIFIC":
        raise V4ScienceLockError(
            "v4 scientific bundle must be marked SCIENTIFIC"
        )
    _require_v4_provenance(
        bundle.get("protocol_provenance"),
        expected=expected,
        location="bundle protocol provenance",
    )
    artifacts = bundle.get("artifacts")
    if not isinstance(artifacts, list) or not artifacts:
        raise V4ScienceLockError(
            "v4 scientific bundle artifacts must be a non-empty JSON list "
            "of artifact envelopes"
        )
    for index, envelope in enumerate(artifacts):
        location = f"artifact envelope {index}"
        if not isinstance(envelope, Mapping):
            raise V4ScienceLockError(f"{location} must be a mapping")
        _require_closed_schema(
            envelope,
            required=_V4_ENVELOPE_KEYS,
            location=location,
        )
        if envelope.get("project_line") != "v4":
            raise V4ScienceLockError(
                f"{location} project_line must be exactly 'v4'"
            )
        _require_v4_provenance(
            envelope.get("protocol_provenance"),
            expected=expected,
            location=f"{location} protocol provenance",
        )
        payload = envelope.get("artifact")
        _validate_v4_artifact_record(
            payload,
            expected_origin=expected_origin,
            location=location,
        )
    return {
        "status": "V4_BUNDLE_STRUCTURE_VALID_ONLY",
        "protocol_id": V4_PROTOCOL_ID,
        "protocol_sha256": V4_PROTOCOL_SHA256,
        "artifact_count": len(artifacts),
        "scientific_admission": False,
    }


def validate_v4_scientific_bundle(
    source_root: Path,
    bundle: Mapping[str, Any],
) -> dict[str, object]:
    """Reject direct scientific admission outside the canonical finalizer.

    The public legacy name remains fail-closed so callers cannot turn a
    structurally valid, undersized bundle into a scientific PASS. The
    canonical finalizer validates structure, verifies source files, rebuilds
    evidence from exactly 400 TaskAuditRecords, evaluates the frozen gate, and
    publishes atomically.
    """

    validate_v4_scientific_bundle_structure(source_root, bundle)
    raise V4ScienceLockError(
        "scientific admission requires the canonical finalizer and its exact 400-record recomputation"
    )
