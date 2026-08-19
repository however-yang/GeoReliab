"""Test-only Pilot partition-freeze and authorization audit harness.

This module is intentionally outside :mod:`georeliab_mve`.  It consumes only
already-recorded admission, protocol, and resource-candidate evidence.  It can
seal a deterministic Pilot partition and a bounded authorization record, but
it cannot dispatch a GPU, materialize inputs, or start a Pilot unit.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import hashlib
import json
from pathlib import Path, PurePosixPath
import re

from georeliab_mve.v4_counterfactuals import (
    SCIENTIFIC_MODELS,
    SCIENTIFIC_STATES,
    TEST_SCENE_IDS,
)
from georeliab_mve.v4_scoped import SELECTOR_VERSION, build_pilot_partition


SCHEMA_VERSION = "georeliab-v4-pilot-freeze-authorization-audit-1.0"
READY_STATUS = "V4_PILOT_PARTITION_AND_AUTHORIZATION_READY"
DEVELOPMENT_EVIDENCE_ONLY = "DEVELOPMENT_EVIDENCE_ONLY"
NO_SCIENTIFIC_RESULT = "NO_SCIENTIFIC_RESULT"

ADMISSION_SCHEMA_VERSION = "georeliab-v4-gate2-pilot-readiness-audit-1.0"
ADMISSION_READY_STATUS = "V4_PILOT_ADMISSION_READY_FOR_EXPLICIT_AUTHORIZATION"
ADMISSION_VALIDATION_CLASS = "GATE2_TO_PILOT_READINESS_AUDIT"
ADMISSION_NEXT_ACTION = (
    "OBTAIN_EXPLICIT_PILOT_GPU_BUDGET_AUTHORIZATION_THEN_FREEZE_PARTITION"
)
REQUEST_SCHEMA_VERSION = "georeliab-v4-pilot-authorization-request-1.0"
REQUEST_STATUS = "USER_APPROVED_PILOT_GPU_BUDGET_REQUEST"
RESOURCE_SCHEMA_VERSION = "georeliab-v4-pilot-resource-candidate-1.0"
PARTITION_SCHEMA_VERSION = "georeliab-v4-pilot-partition-freeze-1.0"
PARTITION_STATUS = "V4_PILOT_PARTITION_FROZEN"
AUTHORIZATION_SCHEMA_VERSION = "georeliab-v4-pilot-authorization-1.0"
AUTHORIZATION_STATUS = "V4_PILOT_EXPLICIT_AUTHORIZATION_RECORDED"
REQUESTED_SCOPE = "PRIMARY_3_SCENE_60_UNIT_PILOT_ONLY"
NEXT_ACTION = "MATERIALIZE_AND_AUDIT_FRESH_PILOT_INPUTS"

MAX_GPU_SECONDS = 21_600
MAX_WALL_SECONDS = 43_200
MAX_STORAGE_BYTES = 25 * 1024**3

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_GIT_OID_RE = re.compile(r"^(?:[0-9a-f]{40}|[0-9a-f]{64})$")
_GPU_UUID_RE = re.compile(r"^GPU-[0-9A-Za-z-]+$")
_MANIFEST_RE = re.compile(r"^([0-9a-f]{64})  ([^\r\n]+)$")

_REQUEST_KEYS = frozenset(
    {
        "schema_version",
        "status",
        "validation_class",
        "scientific_result",
        "user_approved",
        "authorization_note",
        "requested_scope",
        "admission_report_path",
        "admission_report_sha256",
        "resource_manifest_path",
        "resource_manifest_sha256",
        "schedule_identity_sha256",
        "protocol_path",
        "protocol_sha256",
        "production_source_commit",
        "production_source_tree",
        "model_bindings_sha256",
        "gpu_uuid",
        "physical_gpu_index",
        "physical_gpu_count",
        "model_order",
        "sequential_model_execution",
        "sequential_unit_execution",
        "fallback_allowed",
        "auto_retry_allowed",
        "device_switch_allowed",
        "grid_reduction_allowed",
        "downstream_advance_allowed",
        "extension_authorized",
        "confirmation_authorized",
        "max_gpu_seconds",
        "max_wall_seconds",
        "max_storage_bytes",
        "run_root",
        "forbidden_provenance",
        "pilot_inputs_materialized",
        "pilot_started",
    }
)

_PARTITION_KEYS = frozenset(
    {
        "schema_version",
        "status",
        "validation_class",
        "scientific_result",
        "selector_version",
        "schedule_identity_sha256",
        "protocol_sha256",
        "production_source_commit",
        "production_source_tree",
        "model_bindings_sha256",
        "model_order",
        "state_ids",
        "primary_scene_ids",
        "extension_scene_ids",
        "core_scene_ids",
        "primary_unit_ids",
        "extension_unit_ids",
        "confirmation_15_unit_ids",
        "confirmation_17_unit_ids",
        "selector_payload_sha256",
        "selector_partition_sha256",
        "primary_scope",
        "extension_scope",
        "confirmation_15_scope",
        "confirmation_17_scope",
        "disjointness_proof",
        "attempt05_predictions_read",
        "gate2_predictions_read",
        "pilot_inputs_materialized",
        "pilot_started",
        "partition_sha256",
    }
)

_AUTHORIZATION_KEYS = frozenset(
    {
        "schema_version",
        "status",
        "validation_class",
        "scientific_result",
        "authorization_scope",
        "user_approved",
        "authorization_note",
        "approval_request_path",
        "approval_request_sha256",
        "admission_report_path",
        "admission_report_sha256",
        "resource_manifest_path",
        "resource_manifest_sha256",
        "protocol_path",
        "protocol_sha256",
        "schedule_identity_sha256",
        "production_source_commit",
        "production_source_tree",
        "model_bindings_sha256",
        "partition_manifest_path",
        "partition_manifest_sha256",
        "partition_sha256",
        "unit_inventory_sha256",
        "gpu_uuid",
        "physical_gpu_index",
        "physical_gpu_count",
        "model_order",
        "sequential_model_execution",
        "sequential_unit_execution",
        "fallback_allowed",
        "auto_retry_allowed",
        "device_switch_allowed",
        "grid_reduction_allowed",
        "downstream_advance_allowed",
        "extension_authorized",
        "confirmation_authorized",
        "max_gpu_seconds",
        "max_wall_seconds",
        "max_storage_bytes",
        "run_root",
        "pilot_execution_authorized",
        "pilot_partition_frozen",
        "pilot_inputs_materialized",
        "pilot_started",
        "automatic_progression_allowed",
        "next_action",
        "authorization_sha256",
    }
)

_BUNDLE_FILES = frozenset(
    {
        "manifests/pilot-partition-manifest.json",
        "manifests/pilot-authorization.json",
        "pilot-freeze-audit.json",
    }
)


class PilotFreezeAuthorizationError(ValueError):
    """Raised when the test-only Pilot preparation chain fails closed."""


def _fail(reason: str) -> None:
    raise PilotFreezeAuthorizationError(reason)


def _canonical_bytes(value: object) -> bytes:
    try:
        text = json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
        )
    except (TypeError, ValueError) as exc:
        raise PilotFreezeAuthorizationError(
            f"CANONICAL_JSON_INVALID:{type(exc).__name__}:{exc}"
        ) from exc
    return (text + "\n").encode("utf-8")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
    except OSError as exc:
        raise PilotFreezeAuthorizationError(
            f"FILE_DIGEST_UNREADABLE:{path}:{exc}"
        ) from exc
    return digest.hexdigest()


def _sha256_value(value: object) -> str:
    return _sha256_bytes(_canonical_bytes(value))


def _domain_sha256(domain: str, value: object) -> str:
    return hashlib.sha256(
        domain.encode("ascii") + b"\0" + _canonical_bytes(value)
    ).hexdigest()


def _require_sha256(value: object, reason: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        _fail(reason)
    return value


def _require_git_oid(value: object, reason: str) -> str:
    if not isinstance(value, str) or _GIT_OID_RE.fullmatch(value) is None:
        _fail(reason)
    return value


def _read_json(path: Path, reason: str) -> Mapping[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise PilotFreezeAuthorizationError(
            f"{reason}:{type(exc).__name__}:{exc}"
        ) from exc
    if not isinstance(value, Mapping):
        _fail(f"{reason}:JSON_OBJECT_REQUIRED")
    return value


def _require_real_home(home_root: Path) -> Path:
    home = Path(home_root)
    if (
        not home.is_absolute()
        or not home.is_dir()
        or home.is_symlink()
    ):
        _fail("HOME_ROOT_INVALID")
    return home.resolve(strict=True)


def _inside(path: Path, root: Path) -> bool:
    return path == root or root in path.parents


def _require_home_file(path: Path, home: Path, reason: str) -> Path:
    raw = Path(path)
    if not raw.is_absolute() or not raw.is_file() or raw.is_symlink():
        _fail(f"{reason}_FILE_INVALID")
    try:
        resolved = raw.resolve(strict=True)
    except OSError as exc:
        raise PilotFreezeAuthorizationError(f"{reason}_FILE_INVALID:{exc}") from exc
    if not _inside(resolved, home):
        _fail(f"{reason}_ROOT_INVALID")
    return resolved


def _require_fresh_output_root(path: Path, home: Path) -> Path:
    raw = Path(path)
    if not raw.is_absolute():
        _fail("OUTPUT_ROOT_INVALID")
    resolved = raw.resolve(strict=False)
    expected_parent = (home / "georeliab-v4-pilot" / "readiness").resolve(
        strict=False
    )
    if expected_parent not in resolved.parents:
        _fail("OUTPUT_ROOT_SCOPE_INVALID")
    if raw.exists() or resolved.exists():
        _fail("OUTPUT_ROOT_EXISTS")
    return resolved


def _require_fresh_run_root(value: object, home: Path) -> Path:
    if not isinstance(value, str) or not value:
        _fail("RUN_ROOT_INVALID")
    raw = Path(value)
    if not raw.is_absolute():
        _fail("RUN_ROOT_INVALID")
    resolved = raw.resolve(strict=False)
    expected_parent = (home / "georeliab-v4-pilot" / "runs").resolve(
        strict=False
    )
    if expected_parent not in resolved.parents:
        _fail("RUN_ROOT_SCOPE_INVALID")
    if raw.exists() or resolved.exists():
        _fail("RUN_ROOT_NOT_FRESH")
    return resolved


def _require_false(payload: Mapping[str, object], field: str, reason: str) -> None:
    if payload.get(field) is not False:
        _fail(reason)


def _require_true(payload: Mapping[str, object], field: str, reason: str) -> None:
    if payload.get(field) is not True:
        _fail(reason)


def _validate_admission(
    payload: Mapping[str, object],
    *,
    home: Path,
) -> dict[str, object]:
    if payload.get("schema_version") != ADMISSION_SCHEMA_VERSION:
        _fail("ADMISSION_SCHEMA_INVALID")
    if payload.get("status") != ADMISSION_READY_STATUS:
        _fail("ADMISSION_STATUS_NOT_READY")
    if payload.get("validation_class") != ADMISSION_VALIDATION_CLASS:
        _fail("ADMISSION_CLASS_INVALID")
    if payload.get("scientific_result") != NO_SCIENTIFIC_RESULT:
        _fail("ADMISSION_SCIENTIFIC_RESULT_FORBIDDEN")
    blockers = payload.get("blockers")
    if not isinstance(blockers, Sequence) or isinstance(
        blockers, (str, bytes, bytearray)
    ) or list(blockers):
        _fail("ADMISSION_BLOCKERS_PRESENT")
    _require_true(
        payload,
        "can_request_pilot_execution_authorization",
        "ADMISSION_AUTHORIZATION_REQUEST_FORBIDDEN",
    )
    for field in (
        "pilot_execution_authorized",
        "pilot_partition_frozen",
        "pilot_started",
        "confirmation_started",
        "automatic_progression_allowed",
    ):
        _require_false(payload, field, f"ADMISSION_STATE_INVALID:{field}")
    if payload.get("next_action") != ADMISSION_NEXT_ACTION:
        _fail("ADMISSION_NEXT_ACTION_INVALID")

    gates = payload.get("gates")
    if not isinstance(gates, Mapping):
        _fail("ADMISSION_GATES_INVALID")
    g0 = gates.get("g0")
    g1 = gates.get("g1")
    g2 = gates.get("g2")
    if not all(isinstance(item, Mapping) for item in (g0, g1, g2)):
        _fail("ADMISSION_GATES_INVALID")
    assert isinstance(g0, Mapping)
    assert isinstance(g1, Mapping)
    assert isinstance(g2, Mapping)
    if (
        g0.get("status") != "G0_SOURCE_TOOLCHAIN_PASS"
        or g0.get("worktree_clean") is not True
        or g0.get("scientific_result") != NO_SCIENTIFIC_RESULT
    ):
        _fail("ADMISSION_G0_INVALID")
    if (
        g1.get("status") != "G1_CPU_FAULT_MATRIX_PASS"
        or g1.get("scientific_result") != NO_SCIENTIFIC_RESULT
    ):
        _fail("ADMISSION_G1_INVALID")
    if (
        g2.get("status") != "G2_FORMAL_GPU_SMOKE_PASS"
        or g2.get("formal_gate2_equivalent") is not True
        or g2.get("scientific_result") != NO_SCIENTIFIC_RESULT
    ):
        _fail("ADMISSION_G2_INVALID")
    source_commit = _require_git_oid(
        g0.get("source_commit"), "ADMISSION_SOURCE_COMMIT_INVALID"
    )
    source_tree = _require_git_oid(
        g0.get("source_tree"), "ADMISSION_SOURCE_TREE_INVALID"
    )

    auditor = payload.get("auditor")
    if not isinstance(auditor, Mapping):
        _fail("ADMISSION_AUDITOR_INVALID")
    if (
        auditor.get("source_path") != "tests/gate2_pilot_readiness_audit.py"
        or auditor.get("source_tracked") is not True
        or auditor.get("worktree_clean") is not True
    ):
        _fail("ADMISSION_AUDITOR_LINEAGE_INVALID")
    _require_sha256(
        auditor.get("source_sha256"), "ADMISSION_AUDITOR_DIGEST_INVALID"
    )
    if (
        auditor.get("source_commit") != source_commit
        or auditor.get("source_tree") != source_tree
    ):
        _fail("ADMISSION_AUDITOR_SOURCE_MISMATCH")

    binding = g2.get("input_binding")
    if not isinstance(binding, Mapping):
        _fail("ADMISSION_G2_INPUT_BINDING_INVALID")
    if (
        binding.get("topology") != "FORMAL_HOME_DUAL_ROOT"
        or binding.get("attempt05_predictions_read") is not False
        or binding.get("prediction_outputs_reused") is not False
    ):
        _fail("ADMISSION_G2_INPUT_BINDING_INVALID")
    closure_text = binding.get("formal_closure_path")
    if not isinstance(closure_text, str):
        _fail("ADMISSION_FORMAL_CLOSURE_PATH_INVALID")
    closure_path = _require_home_file(
        Path(closure_text), home, "ADMISSION_FORMAL_CLOSURE"
    )
    expected_closure_sha = _require_sha256(
        binding.get("formal_closure_sha256"),
        "ADMISSION_FORMAL_CLOSURE_DIGEST_INVALID",
    )
    if _sha256_file(closure_path) != expected_closure_sha:
        _fail("ADMISSION_FORMAL_CLOSURE_DIGEST_MISMATCH")
    closure = _read_json(closure_path, "ADMISSION_FORMAL_CLOSURE_INVALID")
    schedule_sha = _require_sha256(
        closure.get("schedule_identity_sha256"),
        "ADMISSION_SCHEDULE_IDENTITY_INVALID",
    )
    if (
        closure.get("attempt05_predictions_read") is not False
        or closure.get("prediction_outputs_reused") is not False
        or closure.get("scientific_result") != NO_SCIENTIFIC_RESULT
    ):
        _fail("ADMISSION_FORMAL_CLOSURE_UNSAFE")
    return {
        "source_commit": source_commit,
        "source_tree": source_tree,
        "schedule_identity_sha256": schedule_sha,
        "formal_closure_path": str(closure_path),
        "formal_closure_sha256": expected_closure_sha,
    }


def _validate_resource_candidate(
    payload: Mapping[str, object],
) -> dict[str, object]:
    if payload.get("schema_version") != RESOURCE_SCHEMA_VERSION:
        _fail("RESOURCE_SCHEMA_INVALID")
    if payload.get("validation_class") != DEVELOPMENT_EVIDENCE_ONLY:
        _fail("RESOURCE_CLASS_INVALID")
    if payload.get("scientific_result") != NO_SCIENTIFIC_RESULT:
        _fail("RESOURCE_SCIENTIFIC_RESULT_FORBIDDEN")
    if tuple(payload.get("model_order", ())) != tuple(SCIENTIFIC_MODELS):
        _fail("RESOURCE_MODEL_ORDER_INVALID")
    _require_false(
        payload, "pilot_inputs_materialized", "RESOURCE_INPUTS_ALREADY_MATERIALIZED"
    )
    _require_false(payload, "pilot_started", "RESOURCE_PILOT_ALREADY_STARTED")
    models = payload.get("models")
    if not isinstance(models, Sequence) or isinstance(
        models, (str, bytes, bytearray)
    ):
        _fail("RESOURCE_MODEL_BINDINGS_INVALID")
    rows = list(models)
    if len(rows) != 2 or any(not isinstance(row, Mapping) for row in rows):
        _fail("RESOURCE_MODEL_BINDINGS_INVALID")
    for expected_model, row in zip(SCIENTIFIC_MODELS, rows, strict=True):
        assert isinstance(row, Mapping)
        if row.get("model_id") != expected_model:
            _fail("RESOURCE_MODEL_IDENTITY_INVALID")
        _require_git_oid(
            row.get("source_commit"), "RESOURCE_MODEL_SOURCE_COMMIT_INVALID"
        )
        _require_sha256(
            row.get("checkpoint_sha256"), "RESOURCE_CHECKPOINT_DIGEST_INVALID"
        )
        if not isinstance(row.get("adapter_id"), str) or not row.get("adapter_id"):
            _fail("RESOURCE_ADAPTER_ID_INVALID")
        _require_sha256(
            row.get("adapter_sha256"), "RESOURCE_ADAPTER_DIGEST_INVALID"
        )
    expected_binding_sha = _sha256_value(rows)
    if payload.get("model_bindings_sha256") != expected_binding_sha:
        _fail("RESOURCE_MODEL_BINDINGS_DIGEST_MISMATCH")
    return {
        "models": rows,
        "model_bindings_sha256": expected_binding_sha,
    }


def _validate_budget(payload: Mapping[str, object]) -> None:
    limits = {
        "max_gpu_seconds": MAX_GPU_SECONDS,
        "max_wall_seconds": MAX_WALL_SECONDS,
        "max_storage_bytes": MAX_STORAGE_BYTES,
    }
    for field, ceiling in limits.items():
        value = payload.get(field)
        if type(value) is not int or value <= 0 or value > ceiling:
            _fail(f"BUDGET_INVALID:{field}")


def _validate_approval_request(
    payload: Mapping[str, object],
    *,
    home: Path,
    request_path: Path,
    admission_path: Path,
    admission_sha256: str,
    admission: Mapping[str, object],
) -> dict[str, object]:
    if set(payload) != _REQUEST_KEYS:
        _fail("AUTHORIZATION_REQUEST_SCHEMA_INVALID")
    if (
        payload.get("schema_version") != REQUEST_SCHEMA_VERSION
        or payload.get("status") != REQUEST_STATUS
        or payload.get("validation_class") != DEVELOPMENT_EVIDENCE_ONLY
        or payload.get("scientific_result") != NO_SCIENTIFIC_RESULT
    ):
        _fail("AUTHORIZATION_REQUEST_HEADER_INVALID")
    if payload.get("user_approved") is not True:
        _fail("AUTHORIZATION_USER_APPROVAL_MISSING")
    note = payload.get("authorization_note")
    if not isinstance(note, str) or not note.strip():
        _fail("AUTHORIZATION_NOTE_MISSING")
    if payload.get("requested_scope") != REQUESTED_SCOPE:
        _fail("SCOPE_AUTHORIZATION_INVALID")
    if (
        payload.get("admission_report_path") != str(admission_path)
        or payload.get("admission_report_sha256") != admission_sha256
    ):
        _fail("ADMISSION_REQUEST_BINDING_MISMATCH")
    if (
        payload.get("schedule_identity_sha256")
        != admission.get("schedule_identity_sha256")
    ):
        _fail("AUTHORIZATION_SCHEDULE_IDENTITY_MISMATCH")
    if payload.get("production_source_commit") != admission.get("source_commit"):
        _fail("AUTHORIZATION_SOURCE_COMMIT_MISMATCH")
    if payload.get("production_source_tree") != admission.get("source_tree"):
        _fail("AUTHORIZATION_SOURCE_TREE_MISMATCH")

    protocol_text = payload.get("protocol_path")
    if not isinstance(protocol_text, str):
        _fail("PROTOCOL_PATH_INVALID")
    protocol_path = _require_home_file(Path(protocol_text), home, "PROTOCOL")
    protocol_sha = _require_sha256(
        payload.get("protocol_sha256"), "PROTOCOL_DIGEST_INVALID"
    )
    if _sha256_file(protocol_path) != protocol_sha:
        _fail("PROTOCOL_DIGEST_MISMATCH")

    resource_text = payload.get("resource_manifest_path")
    if not isinstance(resource_text, str):
        _fail("RESOURCE_PATH_INVALID")
    resource_path = _require_home_file(Path(resource_text), home, "RESOURCE")
    resource_sha = _require_sha256(
        payload.get("resource_manifest_sha256"), "RESOURCE_DIGEST_INVALID"
    )
    if _sha256_file(resource_path) != resource_sha:
        _fail("RESOURCE_DIGEST_MISMATCH")
    resource_payload = _read_json(resource_path, "RESOURCE_JSON_INVALID")
    resource = _validate_resource_candidate(resource_payload)
    if payload.get("model_bindings_sha256") != resource[
        "model_bindings_sha256"
    ]:
        _fail("AUTHORIZATION_MODEL_BINDINGS_MISMATCH")

    if (
        not isinstance(payload.get("gpu_uuid"), str)
        or _GPU_UUID_RE.fullmatch(str(payload.get("gpu_uuid"))) is None
        or type(payload.get("physical_gpu_index")) is not int
        or int(payload["physical_gpu_index"]) < 0
        or type(payload.get("physical_gpu_count")) is not int
        or payload.get("physical_gpu_count") != 1
    ):
        _fail("AUTHORIZATION_GPU_IDENTITY_INVALID")
    if tuple(payload.get("model_order", ())) != tuple(SCIENTIFIC_MODELS):
        _fail("AUTHORIZATION_MODEL_ORDER_INVALID")
    for field in ("sequential_model_execution", "sequential_unit_execution"):
        _require_true(payload, field, f"AUTHORIZATION_SEQUENTIALITY_INVALID:{field}")
    for field in (
        "fallback_allowed",
        "auto_retry_allowed",
        "device_switch_allowed",
        "grid_reduction_allowed",
        "downstream_advance_allowed",
        "extension_authorized",
        "confirmation_authorized",
        "pilot_inputs_materialized",
        "pilot_started",
    ):
        _require_false(payload, field, f"AUTHORIZATION_SCOPE_INVALID:{field}")
    provenance = payload.get("forbidden_provenance")
    if not isinstance(provenance, Sequence) or isinstance(
        provenance, (str, bytes, bytearray)
    ) or list(provenance):
        _fail("PROVENANCE_FORBIDDEN")
    _validate_budget(payload)
    run_root = _require_fresh_run_root(payload.get("run_root"), home)
    return {
        "request_path": str(request_path),
        "request_sha256": _sha256_file(request_path),
        "protocol_path": str(protocol_path),
        "protocol_sha256": protocol_sha,
        "resource_path": str(resource_path),
        "resource_sha256": resource_sha,
        "model_bindings_sha256": resource["model_bindings_sha256"],
        "run_root": str(run_root),
    }


def _partition_without_digest(payload: Mapping[str, object]) -> dict[str, object]:
    return {key: value for key, value in payload.items() if key != "partition_sha256"}


def _build_partition_payload(
    *,
    schedule_identity_sha256: str,
    protocol_sha256: str,
    source_commit: str,
    source_tree: str,
    model_bindings_sha256: str,
) -> dict[str, object]:
    base = build_pilot_partition(
        schedule_identity_sha256,
        protocol_sha256=protocol_sha256,
    )
    base_payload = base.to_dict()
    primary = set(base.primary_scene_ids)
    extension = set(base.extension_scene_ids)
    core = set(base.core_scene_ids)
    payload: dict[str, object] = {
        "schema_version": PARTITION_SCHEMA_VERSION,
        "status": PARTITION_STATUS,
        "validation_class": DEVELOPMENT_EVIDENCE_ONLY,
        "scientific_result": NO_SCIENTIFIC_RESULT,
        "selector_version": SELECTOR_VERSION,
        "schedule_identity_sha256": schedule_identity_sha256,
        "protocol_sha256": protocol_sha256,
        "production_source_commit": source_commit,
        "production_source_tree": source_tree,
        "model_bindings_sha256": model_bindings_sha256,
        "model_order": list(SCIENTIFIC_MODELS),
        "state_ids": list(SCIENTIFIC_STATES),
        "primary_scene_ids": list(base.primary_scene_ids),
        "extension_scene_ids": list(base.extension_scene_ids),
        "core_scene_ids": list(base.core_scene_ids),
        "primary_unit_ids": list(base.primary_unit_ids),
        "extension_unit_ids": list(base.extension_unit_ids),
        "confirmation_15_unit_ids": list(base.confirmation_15_unit_ids),
        "confirmation_17_unit_ids": list(base.confirmation_17_unit_ids),
        "selector_payload_sha256": base.selector_payload_sha256,
        "selector_partition_sha256": base.partition_sha256,
        "primary_scope": base_payload["primary_scope"],
        "extension_scope": base_payload["extension_scope"],
        "confirmation_15_scope": base_payload["confirmation_15_scope"],
        "confirmation_17_scope": base_payload["confirmation_17_scope"],
        "disjointness_proof": {
            "primary_vs_extension_disjoint": not bool(primary & extension),
            "primary_vs_core_disjoint": not bool(primary & core),
            "extension_vs_core_disjoint": not bool(extension & core),
            "primary_vs_non_primary_disjoint": not bool(
                primary & (extension | core)
            ),
            "all_twenty_scenes_covered": (
                primary | extension | core == set(TEST_SCENE_IDS)
            ),
        },
        "attempt05_predictions_read": False,
        "gate2_predictions_read": False,
        "pilot_inputs_materialized": False,
        "pilot_started": False,
    }
    payload["partition_sha256"] = _domain_sha256(
        "georeliab:pilot-freeze-partition:v1",
        payload,
    )
    return payload


def validate_pilot_partition_manifest(
    payload: Mapping[str, object],
    *,
    expected_schedule_identity_sha256: str,
    expected_protocol_sha256: str,
    expected_model_bindings_sha256: str,
) -> dict[str, object]:
    """Validate the complete frozen Pilot partition identity."""

    if not isinstance(payload, Mapping) or set(payload) != _PARTITION_KEYS:
        _fail("PARTITION_SCHEMA_INVALID")
    if (
        payload.get("schema_version") != PARTITION_SCHEMA_VERSION
        or payload.get("status") != PARTITION_STATUS
        or payload.get("validation_class") != DEVELOPMENT_EVIDENCE_ONLY
        or payload.get("scientific_result") != NO_SCIENTIFIC_RESULT
    ):
        _fail("PARTITION_HEADER_INVALID")
    if payload.get("selector_version") != SELECTOR_VERSION:
        _fail("PARTITION_SELECTOR_VERSION_INVALID")
    schedule_sha = _require_sha256(
        expected_schedule_identity_sha256,
        "PARTITION_EXPECTED_SCHEDULE_INVALID",
    )
    protocol_sha = _require_sha256(
        expected_protocol_sha256,
        "PARTITION_EXPECTED_PROTOCOL_INVALID",
    )
    model_sha = _require_sha256(
        expected_model_bindings_sha256,
        "PARTITION_EXPECTED_MODEL_BINDINGS_INVALID",
    )
    if payload.get("schedule_identity_sha256") != schedule_sha:
        _fail("PARTITION_SCHEDULE_IDENTITY_MISMATCH")
    if payload.get("protocol_sha256") != protocol_sha:
        _fail("PARTITION_PROTOCOL_MISMATCH")
    if payload.get("model_bindings_sha256") != model_sha:
        _fail("PARTITION_MODEL_BINDINGS_MISMATCH")
    _require_git_oid(
        payload.get("production_source_commit"),
        "PARTITION_SOURCE_COMMIT_INVALID",
    )
    _require_git_oid(
        payload.get("production_source_tree"),
        "PARTITION_SOURCE_TREE_INVALID",
    )
    expected = build_pilot_partition(schedule_sha, protocol_sha256=protocol_sha)
    expected_payload = expected.to_dict()
    comparisons = {
        "primary_scene_ids": list(expected.primary_scene_ids),
        "extension_scene_ids": list(expected.extension_scene_ids),
        "core_scene_ids": list(expected.core_scene_ids),
        "primary_unit_ids": list(expected.primary_unit_ids),
        "extension_unit_ids": list(expected.extension_unit_ids),
        "confirmation_15_unit_ids": list(expected.confirmation_15_unit_ids),
        "confirmation_17_unit_ids": list(expected.confirmation_17_unit_ids),
        "selector_payload_sha256": expected.selector_payload_sha256,
        "selector_partition_sha256": expected.partition_sha256,
        "primary_scope": expected_payload["primary_scope"],
        "extension_scope": expected_payload["extension_scope"],
        "confirmation_15_scope": expected_payload["confirmation_15_scope"],
        "confirmation_17_scope": expected_payload["confirmation_17_scope"],
        "model_order": list(SCIENTIFIC_MODELS),
        "state_ids": list(SCIENTIFIC_STATES),
    }
    for field, expected_value in comparisons.items():
        if payload.get(field) != expected_value:
            _fail(f"PARTITION_IDENTITY_MISMATCH:{field}")
    expected_disjointness = {
        "primary_vs_extension_disjoint": True,
        "primary_vs_core_disjoint": True,
        "extension_vs_core_disjoint": True,
        "primary_vs_non_primary_disjoint": True,
        "all_twenty_scenes_covered": True,
    }
    if payload.get("disjointness_proof") != expected_disjointness:
        _fail("PARTITION_DISJOINTNESS_INVALID")
    for field in (
        "attempt05_predictions_read",
        "gate2_predictions_read",
        "pilot_inputs_materialized",
        "pilot_started",
    ):
        _require_false(payload, field, f"PARTITION_STATE_INVALID:{field}")
    partition_sha = _require_sha256(
        payload.get("partition_sha256"), "PARTITION_DIGEST_INVALID"
    )
    expected_sha = _domain_sha256(
        "georeliab:pilot-freeze-partition:v1",
        _partition_without_digest(payload),
    )
    if partition_sha != expected_sha:
        _fail("PARTITION_DIGEST_MISMATCH")
    return dict(payload)


def _authorization_without_digest(
    payload: Mapping[str, object],
) -> dict[str, object]:
    return {
        key: value
        for key, value in payload.items()
        if key != "authorization_sha256"
    }


def _build_authorization_payload(
    *,
    request: Mapping[str, object],
    request_path: Path,
    admission_path: Path,
    admission_sha256: str,
    resource_path: Path,
    resource_sha256: str,
    protocol_path: Path,
    protocol_sha256: str,
    partition_path: Path,
    partition_bytes: bytes,
    partition: Mapping[str, object],
    run_root: str,
) -> dict[str, object]:
    payload: dict[str, object] = {
        "schema_version": AUTHORIZATION_SCHEMA_VERSION,
        "status": AUTHORIZATION_STATUS,
        "validation_class": DEVELOPMENT_EVIDENCE_ONLY,
        "scientific_result": NO_SCIENTIFIC_RESULT,
        "authorization_scope": REQUESTED_SCOPE,
        "user_approved": True,
        "authorization_note": request["authorization_note"],
        "approval_request_path": str(request_path),
        "approval_request_sha256": _sha256_file(request_path),
        "admission_report_path": str(admission_path),
        "admission_report_sha256": admission_sha256,
        "resource_manifest_path": str(resource_path),
        "resource_manifest_sha256": resource_sha256,
        "protocol_path": str(protocol_path),
        "protocol_sha256": protocol_sha256,
        "schedule_identity_sha256": partition["schedule_identity_sha256"],
        "production_source_commit": partition["production_source_commit"],
        "production_source_tree": partition["production_source_tree"],
        "model_bindings_sha256": partition["model_bindings_sha256"],
        "partition_manifest_path": str(partition_path),
        "partition_manifest_sha256": _sha256_bytes(partition_bytes),
        "partition_sha256": partition["partition_sha256"],
        "unit_inventory_sha256": _domain_sha256(
            "georeliab:pilot-primary-unit-inventory:v1",
            partition["primary_unit_ids"],
        ),
        "gpu_uuid": request["gpu_uuid"],
        "physical_gpu_index": request["physical_gpu_index"],
        "physical_gpu_count": 1,
        "model_order": list(SCIENTIFIC_MODELS),
        "sequential_model_execution": True,
        "sequential_unit_execution": True,
        "fallback_allowed": False,
        "auto_retry_allowed": False,
        "device_switch_allowed": False,
        "grid_reduction_allowed": False,
        "downstream_advance_allowed": False,
        "extension_authorized": False,
        "confirmation_authorized": False,
        "max_gpu_seconds": request["max_gpu_seconds"],
        "max_wall_seconds": request["max_wall_seconds"],
        "max_storage_bytes": request["max_storage_bytes"],
        "run_root": run_root,
        "pilot_execution_authorized": True,
        "pilot_partition_frozen": True,
        "pilot_inputs_materialized": False,
        "pilot_started": False,
        "automatic_progression_allowed": False,
        "next_action": NEXT_ACTION,
    }
    payload["authorization_sha256"] = _domain_sha256(
        "georeliab:pilot-authorization:v1", payload
    )
    return payload


def _validate_authorization_payload(
    payload: Mapping[str, object],
    *,
    partition: Mapping[str, object],
    partition_path: Path,
) -> dict[str, object]:
    if not isinstance(payload, Mapping) or set(payload) != _AUTHORIZATION_KEYS:
        _fail("AUTHORIZATION_SCHEMA_INVALID")
    if (
        payload.get("schema_version") != AUTHORIZATION_SCHEMA_VERSION
        or payload.get("status") != AUTHORIZATION_STATUS
        or payload.get("validation_class") != DEVELOPMENT_EVIDENCE_ONLY
        or payload.get("scientific_result") != NO_SCIENTIFIC_RESULT
        or payload.get("authorization_scope") != REQUESTED_SCOPE
    ):
        _fail("AUTHORIZATION_HEADER_INVALID")
    if payload.get("user_approved") is not True:
        _fail("AUTHORIZATION_USER_APPROVAL_MISSING")
    if not isinstance(payload.get("authorization_note"), str) or not str(
        payload.get("authorization_note")
    ).strip():
        _fail("AUTHORIZATION_NOTE_MISSING")
    bindings = {
        "schedule_identity_sha256": partition["schedule_identity_sha256"],
        "protocol_sha256": partition["protocol_sha256"],
        "production_source_commit": partition["production_source_commit"],
        "production_source_tree": partition["production_source_tree"],
        "model_bindings_sha256": partition["model_bindings_sha256"],
        "partition_sha256": partition["partition_sha256"],
    }
    for field, expected in bindings.items():
        if payload.get(field) != expected:
            _fail(f"AUTHORIZATION_BINDING_MISMATCH:{field}")
    if payload.get("partition_manifest_path") != str(partition_path):
        _fail("AUTHORIZATION_PARTITION_PATH_MISMATCH")
    if payload.get("partition_manifest_sha256") != _sha256_file(partition_path):
        _fail("AUTHORIZATION_PARTITION_DIGEST_MISMATCH")
    expected_units_sha = _domain_sha256(
        "georeliab:pilot-primary-unit-inventory:v1",
        partition["primary_unit_ids"],
    )
    if payload.get("unit_inventory_sha256") != expected_units_sha:
        _fail("AUTHORIZATION_UNIT_INVENTORY_MISMATCH")
    if (
        not isinstance(payload.get("gpu_uuid"), str)
        or _GPU_UUID_RE.fullmatch(str(payload.get("gpu_uuid"))) is None
        or type(payload.get("physical_gpu_index")) is not int
        or int(payload["physical_gpu_index"]) < 0
        or payload.get("physical_gpu_count") != 1
        or type(payload.get("physical_gpu_count")) is not int
    ):
        _fail("AUTHORIZATION_GPU_IDENTITY_INVALID")
    if tuple(payload.get("model_order", ())) != tuple(SCIENTIFIC_MODELS):
        _fail("AUTHORIZATION_MODEL_ORDER_INVALID")
    for field in ("sequential_model_execution", "sequential_unit_execution"):
        _require_true(payload, field, f"AUTHORIZATION_SEQUENTIALITY_INVALID:{field}")
    for field in (
        "fallback_allowed",
        "auto_retry_allowed",
        "device_switch_allowed",
        "grid_reduction_allowed",
        "downstream_advance_allowed",
        "extension_authorized",
        "confirmation_authorized",
        "pilot_inputs_materialized",
        "pilot_started",
        "automatic_progression_allowed",
    ):
        _require_false(payload, field, f"AUTHORIZATION_STATE_INVALID:{field}")
    _require_true(
        payload,
        "pilot_execution_authorized",
        "AUTHORIZATION_EXECUTION_FLAG_INVALID",
    )
    _require_true(
        payload,
        "pilot_partition_frozen",
        "AUTHORIZATION_PARTITION_FLAG_INVALID",
    )
    if payload.get("next_action") != NEXT_ACTION:
        _fail("AUTHORIZATION_NEXT_ACTION_INVALID")
    _validate_budget(payload)
    _require_sha256(
        payload.get("approval_request_sha256"),
        "AUTHORIZATION_REQUEST_DIGEST_INVALID",
    )
    for label in ("admission_report", "resource_manifest", "protocol"):
        path_value = payload.get(f"{label}_path")
        digest_value = payload.get(f"{label}_sha256")
        if not isinstance(path_value, str):
            _fail(f"AUTHORIZATION_EXTERNAL_PATH_INVALID:{label}")
        path = Path(path_value)
        if not path.is_file() or path.is_symlink():
            _fail(f"AUTHORIZATION_EXTERNAL_FILE_INVALID:{label}")
        if _sha256_file(path.resolve(strict=True)) != digest_value:
            _fail(f"AUTHORIZATION_EXTERNAL_DIGEST_MISMATCH:{label}")
    request_path_value = payload.get("approval_request_path")
    if not isinstance(request_path_value, str):
        _fail("AUTHORIZATION_REQUEST_PATH_INVALID")
    request_path = Path(request_path_value)
    if not request_path.is_file() or request_path.is_symlink():
        _fail("AUTHORIZATION_REQUEST_FILE_INVALID")
    if _sha256_file(request_path.resolve(strict=True)) != payload.get(
        "approval_request_sha256"
    ):
        _fail("AUTHORIZATION_REQUEST_DIGEST_MISMATCH")
    run_root = Path(str(payload.get("run_root", "")))
    if not run_root.is_absolute() or run_root.exists():
        _fail("AUTHORIZATION_RUN_ROOT_NOT_FRESH")
    authorization_sha = _require_sha256(
        payload.get("authorization_sha256"), "AUTHORIZATION_DIGEST_INVALID"
    )
    expected_sha = _domain_sha256(
        "georeliab:pilot-authorization:v1",
        _authorization_without_digest(payload),
    )
    if authorization_sha != expected_sha:
        _fail("AUTHORIZATION_DIGEST_MISMATCH")
    return dict(payload)


def _write_json_no_clobber(path: Path, value: object) -> bytes:
    encoded = _canonical_bytes(value)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("xb") as handle:
            handle.write(encoded)
    except FileExistsError as exc:
        raise PilotFreezeAuthorizationError(f"OUTPUT_FILE_EXISTS:{path}") from exc
    return encoded


def _write_manifest(root: Path) -> None:
    lines = []
    for relative in sorted(_BUNDLE_FILES):
        path = root.joinpath(*PurePosixPath(relative).parts)
        lines.append(f"{_sha256_file(path)}  {relative}")
    manifest = root / "MANIFEST.sha256"
    try:
        with manifest.open("x", encoding="ascii", newline="\n") as handle:
            handle.write("\n".join(lines) + "\n")
    except FileExistsError as exc:
        raise PilotFreezeAuthorizationError(
            f"OUTPUT_FILE_EXISTS:{manifest}"
        ) from exc


def prepare_pilot_partition_authorization(
    *,
    admission_report_path: Path,
    approval_request_path: Path,
    output_root: Path,
    home_root: Path,
) -> dict[str, object]:
    """Prepare a sealed, non-executing Pilot partition/authorization bundle."""

    home = _require_real_home(home_root)
    output = _require_fresh_output_root(output_root, home)
    admission_path = _require_home_file(
        Path(admission_report_path), home, "ADMISSION"
    )
    request_path = _require_home_file(
        Path(approval_request_path), home, "AUTHORIZATION_REQUEST"
    )
    request = _read_json(request_path, "AUTHORIZATION_REQUEST_JSON_INVALID")
    request_admission_path = request.get("admission_report_path")
    if request_admission_path != str(admission_path):
        _fail("ADMISSION_REQUEST_PATH_MISMATCH")
    admission_sha = _require_sha256(
        request.get("admission_report_sha256"),
        "ADMISSION_REQUEST_DIGEST_INVALID",
    )
    if _sha256_file(admission_path) != admission_sha:
        _fail("ADMISSION_DIGEST_MISMATCH")
    admission_payload = _read_json(admission_path, "ADMISSION_JSON_INVALID")
    admission = _validate_admission(admission_payload, home=home)
    approval = _validate_approval_request(
        request,
        home=home,
        request_path=request_path,
        admission_path=admission_path,
        admission_sha256=admission_sha,
        admission=admission,
    )

    schedule_sha = str(admission["schedule_identity_sha256"])
    protocol_sha = str(approval["protocol_sha256"])
    model_sha = str(approval["model_bindings_sha256"])
    partition = _build_partition_payload(
        schedule_identity_sha256=schedule_sha,
        protocol_sha256=protocol_sha,
        source_commit=str(admission["source_commit"]),
        source_tree=str(admission["source_tree"]),
        model_bindings_sha256=model_sha,
    )
    partition = validate_pilot_partition_manifest(
        partition,
        expected_schedule_identity_sha256=schedule_sha,
        expected_protocol_sha256=protocol_sha,
        expected_model_bindings_sha256=model_sha,
    )
    partition_path = output / "manifests" / "pilot-partition-manifest.json"
    partition_bytes = _canonical_bytes(partition)
    authorization_path = output / "manifests" / "pilot-authorization.json"
    authorization = _build_authorization_payload(
        request=request,
        request_path=request_path,
        admission_path=admission_path,
        admission_sha256=admission_sha,
        resource_path=Path(str(approval["resource_path"])),
        resource_sha256=str(approval["resource_sha256"]),
        protocol_path=Path(str(approval["protocol_path"])),
        protocol_sha256=protocol_sha,
        partition_path=partition_path,
        partition_bytes=partition_bytes,
        partition=partition,
        run_root=str(approval["run_root"]),
    )
    audit_payload = {
        "schema_version": SCHEMA_VERSION,
        "status": READY_STATUS,
        "validation_class": DEVELOPMENT_EVIDENCE_ONLY,
        "scientific_result": NO_SCIENTIFIC_RESULT,
        "admission_report_path": str(admission_path),
        "admission_report_sha256": admission_sha,
        "approval_request_path": str(request_path),
        "approval_request_sha256": str(approval["request_sha256"]),
        "partition_manifest_path": str(partition_path),
        "partition_manifest_sha256": _sha256_bytes(partition_bytes),
        "partition_sha256": partition["partition_sha256"],
        "authorization_path": str(authorization_path),
        "authorization_sha256": authorization["authorization_sha256"],
        "schedule_identity_sha256": schedule_sha,
        "protocol_sha256": protocol_sha,
        "model_bindings_sha256": model_sha,
        "primary_scene_count": 3,
        "primary_unit_count": 60,
        "pilot_execution_authorized": True,
        "pilot_partition_frozen": True,
        "pilot_inputs_materialized": False,
        "pilot_started": False,
        "confirmation_started": False,
        "automatic_progression_allowed": False,
        "blockers": [],
        "next_action": NEXT_ACTION,
    }

    output.mkdir(parents=True, exist_ok=False)
    _write_json_no_clobber(partition_path, partition)
    _write_json_no_clobber(authorization_path, authorization)
    _write_json_no_clobber(output / "pilot-freeze-audit.json", audit_payload)
    _write_manifest(output)
    return verify_pilot_freeze_bundle(output)


def _manifest_entries(path: Path) -> dict[str, str]:
    try:
        lines = path.read_text(encoding="ascii").splitlines()
    except (OSError, UnicodeError) as exc:
        raise PilotFreezeAuthorizationError(f"MANIFEST_UNREADABLE:{exc}") from exc
    entries: dict[str, str] = {}
    for line in lines:
        match = _MANIFEST_RE.fullmatch(line)
        if match is None:
            _fail("MANIFEST_ROW_INVALID")
        digest, relative_text = match.groups()
        relative = PurePosixPath(relative_text)
        if (
            relative.is_absolute()
            or ".." in relative.parts
            or relative_text in entries
        ):
            _fail("MANIFEST_PATH_INVALID")
        entries[relative_text] = digest
    if set(entries) != _BUNDLE_FILES:
        _fail("MANIFEST_FILE_SET_INVALID")
    return entries


def verify_pilot_freeze_bundle(root: Path) -> dict[str, object]:
    """Verify exact-file coverage and hashes for a prepared freeze bundle."""

    root = Path(root)
    if not root.is_absolute() or not root.is_dir() or root.is_symlink():
        _fail("BUNDLE_ROOT_INVALID")
    resolved = root.resolve(strict=True)
    paths = tuple(resolved.rglob("*"))
    if any(path.is_symlink() for path in paths):
        _fail("MANIFEST_SYMLINK_FORBIDDEN")
    actual_files = {
        path.relative_to(resolved).as_posix()
        for path in paths
        if path.is_file() and path.name != "MANIFEST.sha256"
    }
    if actual_files != _BUNDLE_FILES:
        _fail("MANIFEST_DIRECTORY_FILE_SET_INVALID")
    manifest_path = resolved / "MANIFEST.sha256"
    if not manifest_path.is_file() or manifest_path.is_symlink():
        _fail("MANIFEST_MISSING")
    entries = _manifest_entries(manifest_path)
    for relative, expected_sha in entries.items():
        path = resolved.joinpath(*PurePosixPath(relative).parts)
        if _sha256_file(path) != expected_sha:
            _fail(f"MANIFEST_DIGEST_MISMATCH:{relative}")

    partition_path = resolved / "manifests" / "pilot-partition-manifest.json"
    authorization_path = resolved / "manifests" / "pilot-authorization.json"
    audit_path = resolved / "pilot-freeze-audit.json"
    partition_payload = _read_json(partition_path, "PARTITION_JSON_INVALID")
    authorization_payload = _read_json(
        authorization_path, "AUTHORIZATION_JSON_INVALID"
    )
    schedule_sha = _require_sha256(
        authorization_payload.get("schedule_identity_sha256"),
        "AUTHORIZATION_SCHEDULE_IDENTITY_INVALID",
    )
    protocol_sha = _require_sha256(
        authorization_payload.get("protocol_sha256"),
        "AUTHORIZATION_PROTOCOL_DIGEST_INVALID",
    )
    model_sha = _require_sha256(
        authorization_payload.get("model_bindings_sha256"),
        "AUTHORIZATION_MODEL_BINDINGS_DIGEST_INVALID",
    )
    partition = validate_pilot_partition_manifest(
        partition_payload,
        expected_schedule_identity_sha256=schedule_sha,
        expected_protocol_sha256=protocol_sha,
        expected_model_bindings_sha256=model_sha,
    )
    authorization = _validate_authorization_payload(
        authorization_payload,
        partition=partition,
        partition_path=partition_path,
    )
    audit_payload = _read_json(audit_path, "FREEZE_AUDIT_JSON_INVALID")
    required_audit = {
        "schema_version": SCHEMA_VERSION,
        "status": READY_STATUS,
        "validation_class": DEVELOPMENT_EVIDENCE_ONLY,
        "scientific_result": NO_SCIENTIFIC_RESULT,
        "partition_manifest_path": str(partition_path),
        "partition_manifest_sha256": _sha256_file(partition_path),
        "partition_sha256": partition["partition_sha256"],
        "authorization_path": str(authorization_path),
        "authorization_sha256": authorization["authorization_sha256"],
        "schedule_identity_sha256": schedule_sha,
        "protocol_sha256": protocol_sha,
        "model_bindings_sha256": model_sha,
        "primary_scene_count": 3,
        "primary_unit_count": 60,
        "pilot_execution_authorized": True,
        "pilot_partition_frozen": True,
        "pilot_inputs_materialized": False,
        "pilot_started": False,
        "confirmation_started": False,
        "automatic_progression_allowed": False,
        "blockers": [],
        "next_action": NEXT_ACTION,
    }
    for field, expected in required_audit.items():
        if audit_payload.get(field) != expected:
            _fail(f"FREEZE_AUDIT_BINDING_MISMATCH:{field}")
    for prefix in ("admission_report", "approval_request"):
        path_value = audit_payload.get(f"{prefix}_path")
        digest_value = audit_payload.get(f"{prefix}_sha256")
        if not isinstance(path_value, str) or not Path(path_value).is_file():
            _fail(f"FREEZE_AUDIT_EXTERNAL_PATH_INVALID:{prefix}")
        if _sha256_file(Path(path_value)) != digest_value:
            _fail(f"FREEZE_AUDIT_EXTERNAL_DIGEST_MISMATCH:{prefix}")
    return dict(audit_payload)
