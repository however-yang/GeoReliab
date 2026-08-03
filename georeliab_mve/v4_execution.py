"""CPU-only execution governance for the GeoReliab v4 scientific MVE.

This module does not launch adapters, choose devices, or create production
runtime receipts.  It admits only already-materialized v4 Task 2/3 objects and
returns fail-closed controller decisions for the next external action.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
import hashlib
import json
import math
import os
import shutil
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse
from uuid import uuid4

from .v4_counterfactuals import (
    SCIENTIFIC_MODELS,
    ScientificExecutionUnit,
    ScientificSchedule,
    validate_scientific_schedule,
    validate_v4_split_assignment,
)
from .v4_gates import WarningGateDecision, evaluate_warning_gate
from .v4_metrics import validate_native_warning_calibration_inventory
from .v4_records import (
    TASK_AUDIT_RECORD_SCHEMA_VERSION,
    WARNING_EVIDENCE_SCHEMA_VERSION,
    TaskAuditRecord,
    WarningEvidence,
    parse_task_audit_record,
)
from .v4_science_lock import (
    V4_ARTIFACT_RECORD_SCHEMA_VERSION,
    V4_PROTOCOL_ID,
    V4_PROTOCOL_SHA256,
    V4_RECORD_ORIGIN_SCHEMA_VERSION,
    V4_SCIENTIFIC_BUNDLE_SCHEMA_VERSION,
    V4ScienceLockError,
    validate_v4_scientific_bundle_structure,
    v4_protocol_provenance,
)
from .v4_statistics import build_warning_evidence


ENGINEERING_SANITY = "ENGINEERING_SANITY"
SCIENTIFIC_MVE = "SCIENTIFIC_MVE"
NUMERICAL_REPEAT = "NUMERICAL_REPEAT"
FINALIZE = "FINALIZE"
GPU_SELECTION_REQUIRED = "GPU_SELECTION_REQUIRED"
V4_GPU_RECEIPT_SCHEMA_VERSION = "georeliab-v4-explicit-gpu-selection-1.0"
V4_EXECUTION_DECISION_SCHEMA_VERSION = "georeliab-v4-execution-decision-1.0"
V4_HARDWARE_PREFLIGHT_SCHEMA_VERSION = "georeliab-v4-hardware-preflight-1.0"

GPU_TARGET_SECONDS = 35 * 3600
GPU_CATASTROPHE_SECONDS = 50 * 3600
BYTE_TARGET = 150_000_000_000
BYTE_CATASTROPHE = 1_000_000_000_000

GPU_BEARING_STAGES = frozenset(
    {ENGINEERING_SANITY, SCIENTIFIC_MVE, NUMERICAL_REPEAT}
)
ALLOWED_TERMINAL_STATUSES = frozenset(
    {
        "MVE_GO_TO_EXTERNAL_VALIDATION",
        "MVE_SCIENTIFIC_NO_GO",
        "MVE_BLOCKED_ENDPOINT",
        "MVE_BLOCKED_EVIDENCE_TAMPER",
        "MVE_BLOCKED_V1_EVIDENCE",
        "MVE_BLOCKED_EVIDENCE_CONTRACT",
        "MVE_BLOCKED_SCHEDULE_BINDING_REQUIRED",
        "MVE_BLOCKED_CALIBRATION_BINDING_REQUIRED",
        "MVE_BLOCKED_SOURCE_RECORD_BINDING_REQUIRED",
    }
)


class V4ExecutionError(RuntimeError):
    """Raised when v4 execution governance cannot prove an invariant."""


@dataclass(frozen=True, slots=True)
class V4ExecutionReceipt:
    explicit_user_selection: bool
    project_commit: str
    project_tree: str
    protocol_id: str
    protocol_sha256: str
    scope: str
    stage: str
    schedule_sha256: str | None
    hardware_preflight_path: str
    hardware_preflight_sha256: str
    requested_physical_index: int
    resolved_physical_index: int
    device_uuid: str
    device_model: str
    driver_version: str
    total_memory_bytes: int
    max_concurrent_gpus: int
    sequential_model_execution: bool
    sequential_unit_execution: bool
    fallback_allowed: bool
    device_switch_allowed: bool
    retry_allowed: bool
    nonce: str
    schema_version: str = V4_GPU_RECEIPT_SCHEMA_VERSION

    def to_dict(self) -> dict[str, object]:
        return asdict(self)

    @classmethod
    def from_mapping(cls, value: object) -> "V4ExecutionReceipt":
        if isinstance(value, V4ExecutionReceipt):
            return value
        if not isinstance(value, Mapping):
            raise V4ExecutionError("V4_GPU_RECEIPT_SCHEMA_REQUIRED")
        required = set(cls.__dataclass_fields__)
        actual = {key for key in value if isinstance(key, str)}
        if actual != required:
            raise V4ExecutionError("V4_GPU_RECEIPT_SCHEMA_REQUIRED")
        if value.get("schema_version") != V4_GPU_RECEIPT_SCHEMA_VERSION:
            raise V4ExecutionError("V4_GPU_RECEIPT_SCHEMA_REQUIRED")
        try:
            return cls(**{key: value[key] for key in required})
        except TypeError as exc:
            raise V4ExecutionError("V4_GPU_RECEIPT_SCHEMA_REQUIRED") from exc


@dataclass(frozen=True, slots=True)
class V4ControllerDecision:
    status: str
    reason_code: str
    unit: ScientificExecutionUnit | None = None
    record_path: Path | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": V4_EXECUTION_DECISION_SCHEMA_VERSION,
            "status": self.status,
            "reason_code": self.reason_code,
            "unit": None if self.unit is None else self.unit.to_dict(),
            "record_path": None if self.record_path is None else str(self.record_path),
        }


@dataclass(frozen=True, slots=True)
class V4ResourceLedger:
    gpu_inference_seconds: float
    wall_runtime_seconds: float
    new_logical_bytes: int
    new_allocated_bytes: int
    peak_device_memory_bytes: int
    stage: str
    model_id: str
    unit_sha256: str
    baseline_inventory_sha256: str = "0" * 64


@dataclass(frozen=True, slots=True)
class V4ResourceDecision:
    status: str
    reason_code: str


@dataclass(frozen=True, slots=True)
class V4RepeatRequest:
    original_unit_sha256: str
    repeat_unit_sha256: str
    boundary_distance: float
    exact_recipe_sha256: str
    enters_ci: bool
    schema_version: str = "georeliab-v4-numerical-repeat-request-1.0"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_json(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise V4ExecutionError("V4_JSON_UNREADABLE") from exc
    if not isinstance(value, dict):
        raise V4ExecutionError("V4_JSON_NOT_OBJECT")
    return value




def _is_strict_int(value: object) -> bool:
    return type(value) is int


def _is_non_negative_int(value: object) -> bool:
    return _is_strict_int(value) and value >= 0


def _is_positive_int(value: object) -> bool:
    return _is_strict_int(value) and value > 0


def _is_sha(value: object, length: int = 64) -> bool:
    return (
        isinstance(value, str)
        and len(value) == length
        and value == value.lower()
        and all(character in "0123456789abcdef" for character in value)
    )


def _expect(condition: bool, reason_code: str) -> None:
    if not condition:
        raise V4ExecutionError(reason_code)


def v4_stage_entry_status(scope: str) -> V4ControllerDecision:
    if scope == FINALIZE:
        return V4ControllerDecision(
            status="V4_CPU_ONLY_FINALIZE_READY",
            reason_code="V4_FINALIZE_NEVER_CONSUMES_GPU_RECEIPT",
        )
    if scope in GPU_BEARING_STAGES:
        return V4ControllerDecision(
            status=GPU_SELECTION_REQUIRED,
            reason_code="V4_GPU_BEARING_STAGE_REQUIRES_EXPLICIT_RECEIPT",
        )
    return V4ControllerDecision(
        status="MVE_BLOCKED_UNKNOWN_STAGE",
        reason_code="V4_STAGE_SCOPE_UNKNOWN",
    )


def validate_v4_gpu_receipt(
    receipt: V4ExecutionReceipt | Mapping[str, object],
    *,
    project_commit: str,
    project_tree: str,
    scope: str,
    stage: str,
    schedule_sha256: str | None,
    hardware_preflight_path: Path,
    hardware_preflight_sha256: str,
    requested_physical_index: int,
    visible_gpu_count: int,
    active_gpu_count: int,
    lock_active: bool,
    stage_stopped: bool = False,
) -> V4ExecutionReceipt:
    """Validate an exact v4-only one-stage GPU authorization."""

    parsed = V4ExecutionReceipt.from_mapping(receipt)
    _expect(parsed.schema_version == V4_GPU_RECEIPT_SCHEMA_VERSION, "V4_GPU_RECEIPT_SCHEMA_REQUIRED")
    _expect(type(requested_physical_index) is int and requested_physical_index >= 0, "V4_GPU_RECEIPT_DEVICE_MISMATCH")
    _expect(type(visible_gpu_count) is int, "V4_GPU_RECEIPT_SINGLE_VISIBLE_GPU_REQUIRED")
    _expect(type(active_gpu_count) is int, "V4_GPU_RECEIPT_NO_ACTIVE_COMPUTE_PROCESS")
    _expect(type(lock_active) is bool, "V4_GPU_RECEIPT_LOCK_CONFLICT")
    _expect(type(stage_stopped) is bool, "V4_GPU_RECEIPT_EXPIRED_AT_STAGE_STOP")
    _expect(not stage_stopped, "V4_GPU_RECEIPT_EXPIRED_AT_STAGE_STOP")
    _expect(parsed.explicit_user_selection is True, "GPU_SELECTION_REQUIRED")
    _expect(
        parsed.project_commit == project_commit and parsed.project_tree == project_tree,
        "V4_GPU_RECEIPT_GIT_MISMATCH",
    )
    _expect(
        parsed.protocol_id == V4_PROTOCOL_ID
        and parsed.protocol_sha256 == V4_PROTOCOL_SHA256,
        "V4_GPU_RECEIPT_PROTOCOL_MISMATCH",
    )
    _expect(parsed.scope == scope and parsed.stage == stage, "V4_GPU_RECEIPT_SCOPE_MISMATCH")
    _expect(parsed.scope in GPU_BEARING_STAGES, "V4_GPU_RECEIPT_SCOPE_MISMATCH")
    _expect(parsed.schedule_sha256 == schedule_sha256, "V4_GPU_RECEIPT_SCHEDULE_MISMATCH")
    _expect(hardware_preflight_path.is_file(), "V4_GPU_RECEIPT_HARDWARE_PREFLIGHT_MISSING")
    actual_preflight_sha = _sha256_file(hardware_preflight_path)
    _expect(
        actual_preflight_sha == hardware_preflight_sha256,
        "V4_GPU_RECEIPT_HARDWARE_PREFLIGHT_MISMATCH",
    )
    _expect(
        Path(parsed.hardware_preflight_path).resolve()
        == hardware_preflight_path.resolve()
        and parsed.hardware_preflight_sha256 == actual_preflight_sha,
        "V4_GPU_RECEIPT_HARDWARE_PREFLIGHT_MISMATCH",
    )
    _expect(
        _is_non_negative_int(parsed.requested_physical_index)
        and _is_non_negative_int(parsed.resolved_physical_index)
        and parsed.requested_physical_index == requested_physical_index
        and parsed.resolved_physical_index == requested_physical_index,
        "V4_GPU_RECEIPT_DEVICE_MISMATCH",
    )
    _expect(visible_gpu_count == 1, "V4_GPU_RECEIPT_SINGLE_VISIBLE_GPU_REQUIRED")
    _expect(active_gpu_count == 0, "V4_GPU_RECEIPT_NO_ACTIVE_COMPUTE_PROCESS")
    _expect(not lock_active, "V4_GPU_RECEIPT_LOCK_CONFLICT")
    _expect(type(parsed.max_concurrent_gpus) is int and parsed.max_concurrent_gpus == 1, "V4_GPU_RECEIPT_SINGLE_GPU_REQUIRED")
    _expect(
        parsed.sequential_model_execution is True
        and parsed.sequential_unit_execution is True,
        "V4_GPU_RECEIPT_SEQUENTIAL_REQUIRED",
    )
    _expect(
        parsed.fallback_allowed is False
        and parsed.device_switch_allowed is False
        and parsed.retry_allowed is False,
        "V4_GPU_RECEIPT_NO_FALLBACK_REQUIRED",
    )
    _expect(_is_sha(parsed.project_commit, 40), "V4_GPU_RECEIPT_GIT_MISMATCH")
    _expect(_is_sha(parsed.project_tree, 40), "V4_GPU_RECEIPT_GIT_MISMATCH")
    _expect(_is_sha(parsed.hardware_preflight_sha256), "V4_GPU_RECEIPT_HARDWARE_PREFLIGHT_MISMATCH")
    _expect(
        _is_positive_int(parsed.total_memory_bytes)
        and isinstance(parsed.device_uuid, str)
        and bool(parsed.device_uuid.strip())
        and isinstance(parsed.device_model, str)
        and bool(parsed.device_model.strip())
        and isinstance(parsed.driver_version, str)
        and bool(parsed.driver_version.strip()),
        "V4_GPU_RECEIPT_DEVICE_MISMATCH",
    )
    preflight = _load_json(hardware_preflight_path)
    expected_preflight_keys = {
        "schema_version",
        "status",
        "project_commit",
        "project_tree",
        "scope",
        "stage",
        "requested_physical_index",
        "visible_gpu_count",
        "compute_process_count",
        "stable_sample_count",
        "environment",
        "devices",
        "samples",
        "model_environment_probes",
    }
    _expect(set(preflight) == expected_preflight_keys, "V4_GPU_PREFLIGHT_SCHEMA_REQUIRED")
    _expect(
        preflight.get("schema_version") == V4_HARDWARE_PREFLIGHT_SCHEMA_VERSION
        and preflight.get("status") == "PASS"
        and preflight.get("project_commit") == project_commit
        and preflight.get("project_tree") == project_tree
        and preflight.get("scope") == scope
        and preflight.get("stage") == stage
        and type(preflight.get("requested_physical_index")) is int
        and preflight.get("requested_physical_index") == requested_physical_index
        and type(preflight.get("visible_gpu_count")) is int
        and preflight.get("visible_gpu_count") == 1
        and type(preflight.get("compute_process_count")) is int
        and preflight.get("compute_process_count") == 0
        and type(preflight.get("stable_sample_count")) is int
        and preflight.get("stable_sample_count") == 2,
        "V4_GPU_PREFLIGHT_NOT_PASSING_EXACT_REQUEST",
    )
    environment = preflight.get("environment")
    _expect(
        isinstance(environment, Mapping)
        and set(environment) == {"CUDA_VISIBLE_DEVICES", "GEORELIAB_PHYSICAL_GPU_DEVICE"}
        and environment.get("CUDA_VISIBLE_DEVICES") == str(requested_physical_index)
        and environment.get("GEORELIAB_PHYSICAL_GPU_DEVICE") == f"cuda:{requested_physical_index}",
        "V4_GPU_PREFLIGHT_ENVIRONMENT_MISMATCH",
    )
    samples = preflight.get("samples")
    _expect(
        isinstance(samples, list) and len(samples) == 2,
        "V4_GPU_PREFLIGHT_STABLE_SAMPLES_REQUIRED",
    )
    for index, sample in enumerate(samples):
        _expect(
            isinstance(sample, Mapping)
            and set(sample) == {
                "sample_index",
                "visible_gpu_count",
                "compute_process_count",
                "device_uuid",
                "physical_index",
            }
            and type(sample.get("sample_index")) is int
            and sample.get("sample_index") == index
            and type(sample.get("visible_gpu_count")) is int
            and sample.get("visible_gpu_count") == 1
            and type(sample.get("compute_process_count")) is int
            and sample.get("compute_process_count") == 0
            and isinstance(sample.get("device_uuid"), str)
            and bool(str(sample.get("device_uuid")).strip())
            and sample.get("device_uuid") == parsed.device_uuid
            and type(sample.get("physical_index")) is int
            and sample.get("physical_index") == parsed.resolved_physical_index,
            "V4_GPU_PREFLIGHT_STABLE_SAMPLES_REQUIRED",
        )
    probes = preflight.get("model_environment_probes")
    _expect(
        isinstance(probes, list) and len(probes) == len(SCIENTIFIC_MODELS),
        "V4_GPU_PREFLIGHT_TORCH_PROBES_REQUIRED",
    )
    seen_probe_models: set[str] = set()
    for probe in probes:
        _expect(
            isinstance(probe, Mapping)
            and set(probe) == {
                "model_id",
                "torch_device_count",
                "torch_cuda_available",
                "torch_current_device",
                "mapped_device_uuid",
                "compute_process_count",
            },
            "V4_GPU_PREFLIGHT_TORCH_PROBES_REQUIRED",
        )
        model_id = probe.get("model_id")
        _expect(isinstance(model_id, str) and model_id in SCIENTIFIC_MODELS, "V4_GPU_PREFLIGHT_TORCH_PROBES_REQUIRED")
        _expect(model_id not in seen_probe_models, "V4_GPU_PREFLIGHT_TORCH_PROBES_REQUIRED")
        seen_probe_models.add(model_id)
        _expect(
            type(probe.get("torch_device_count")) is int
            and probe.get("torch_device_count") == 1
            and probe.get("torch_cuda_available") is True
            and type(probe.get("torch_current_device")) is int
            and probe.get("torch_current_device") == 0
            and isinstance(probe.get("mapped_device_uuid"), str)
            and bool(str(probe.get("mapped_device_uuid")).strip())
            and probe.get("mapped_device_uuid") == parsed.device_uuid
            and type(probe.get("compute_process_count")) is int
            and probe.get("compute_process_count") == 0,
            "V4_GPU_PREFLIGHT_TORCH_PROBES_REQUIRED",
        )
    _expect(seen_probe_models == set(SCIENTIFIC_MODELS), "V4_GPU_PREFLIGHT_TORCH_PROBES_REQUIRED")
    devices = preflight.get("devices")
    _expect(
        isinstance(devices, list) and len(devices) == 1,
        "V4_GPU_RECEIPT_SINGLE_VISIBLE_GPU_REQUIRED",
    )
    device = devices[0]
    expected_device_keys = {
        "physical_index",
        "uuid",
        "model",
        "driver_version",
        "total_memory_bytes",
        "compute_process_count",
    }
    _expect(
        isinstance(device, Mapping) and set(device) == expected_device_keys,
        "V4_GPU_RECEIPT_DEVICE_MISMATCH",
    )
    _expect(
        type(device.get("physical_index")) is int
        and device.get("physical_index") == parsed.resolved_physical_index
        and isinstance(device.get("uuid"), str)
        and bool(str(device.get("uuid")).strip())
        and device.get("uuid") == parsed.device_uuid
        and isinstance(device.get("model"), str)
        and bool(str(device.get("model")).strip())
        and device.get("model") == parsed.device_model
        and isinstance(device.get("driver_version"), str)
        and bool(str(device.get("driver_version")).strip())
        and device.get("driver_version") == parsed.driver_version
        and type(device.get("total_memory_bytes")) is int
        and device.get("total_memory_bytes") == parsed.total_memory_bytes
        and type(device.get("compute_process_count")) is int
        and device.get("compute_process_count") == 0,
        "V4_GPU_RECEIPT_DEVICE_MISMATCH",
    )
    return parsed


def authorize_next_scientific_dispatch(
    schedule: ScientificSchedule,
    root: Path,
    receipt: V4ExecutionReceipt | Mapping[str, object],
    *,
    project_commit: str,
    project_tree: str,
    hardware_preflight_path: Path,
    hardware_preflight_sha256: str,
    requested_physical_index: int,
    visible_gpu_count: int,
    active_gpu_count: int,
    lock_active: bool,
    stage_stopped: bool = False,
) -> V4ControllerDecision:
    """Return one dispatchable unit only after exact receipt validation."""

    current = first_incomplete_unit(schedule, root)
    if current.status != GPU_SELECTION_REQUIRED or current.unit is None:
        return current
    validate_v4_gpu_receipt(
        receipt,
        project_commit=project_commit,
        project_tree=project_tree,
        scope=SCIENTIFIC_MVE,
        stage=SCIENTIFIC_MVE,
        schedule_sha256=schedule.schedule_sha256,
        hardware_preflight_path=hardware_preflight_path,
        hardware_preflight_sha256=hardware_preflight_sha256,
        requested_physical_index=requested_physical_index,
        visible_gpu_count=visible_gpu_count,
        active_gpu_count=active_gpu_count,
        lock_active=lock_active,
        stage_stopped=stage_stopped,
    )
    return V4ControllerDecision(
        status="V4_DISPATCH_AUTHORIZED",
        reason_code="V4_EXACT_RECEIPT_AUTHORIZES_ONE_CANONICAL_UNIT",
        unit=current.unit,
        record_path=current.record_path,
    )


def canonical_record_path(root: Path, unit: ScientificExecutionUnit) -> Path:
    return (
        root
        / "stage"
        / SCIENTIFIC_MVE
        / "records"
        / unit.model_id
        / f"scan{unit.scene_id:03d}"
        / f"{unit.state_id}.json"
    )


def _canonical_units(schedule: ScientificSchedule) -> tuple[ScientificExecutionUnit, ...]:
    validated = validate_scientific_schedule(schedule)
    if len(validated.units) != 400:
        raise V4ExecutionError("V4_SCHEDULE_NOT_EXACT_400")
    return tuple(validated.units)


def _unit_key(unit: ScientificExecutionUnit) -> tuple[str, int, str]:
    return unit.model_id, unit.scene_id, unit.state_id


def _record_key(record: TaskAuditRecord) -> tuple[str, int, str]:
    return record.model_id, record.scene_id, record.state_id


def _read_record(path: Path) -> TaskAuditRecord:
    try:
        return parse_task_audit_record(path.read_bytes())
    except Exception as exc:
        raise V4ExecutionError("V4_EXISTING_ARTIFACT_INVALID") from exc


def admit_existing_task_record(
    path: Path,
    unit: ScientificExecutionUnit,
    *,
    root: Path,
) -> TaskAuditRecord:
    expected_path = canonical_record_path(root, unit)
    if path.resolve() != expected_path.resolve():
        raise V4ExecutionError("V4_RECORD_CANONICAL_PATH_MISMATCH")
    record = _read_record(path)
    if (
        _record_key(record) != _unit_key(unit)
        or record.execution_unit_sha256 != unit.execution_unit_sha256
        or record.state_identity_sha256 != unit.state_identity_sha256
        or record.pair_identity_sha256 != unit.pair_identity_sha256
    ):
        raise V4ExecutionError("V4_RECORD_SOURCE_LINKAGE_MISMATCH")
    if record.record_sha256 != _sha256_json(record.payload()):
        raise V4ExecutionError("V4_RECORD_HASH_MISMATCH")
    return record


def _allowed_record_paths(root: Path, units: Sequence[ScientificExecutionUnit]) -> set[Path]:
    return {canonical_record_path(root, unit).resolve() for unit in units}


_LEGACY_ATTEMPT05_BUNDLE_MEMBERS = frozenset(
    {
        "audit_record.json",
        "dense_audit.npz",
        "geometry_prediction.npz",
        "gt_points.npz",
        "native_confidence.npz",
        "prediction_artifact.json",
        "run_manifest.json",
        "task_audit_record.json",
        "valid_mask.npz",
    }
)


def _validated_legacy_bundle_members(
    root: Path,
    units: Sequence[ScientificExecutionUnit],
) -> set[Path]:
    record_root = root / "stage" / SCIENTIFIC_MVE / "records"
    by_key = {_unit_key(unit): unit for unit in units}
    allowed: set[Path] = set()
    if not record_root.exists():
        return allowed
    for task_path in record_root.glob("*/scan*/task_audit_record.json"):
        record = _read_record(task_path)
        unit = by_key.get(_record_key(record))
        if unit is None:
            raise V4ExecutionError("V4_RECORD_UNEXPECTED_EXTRA")
        canonical = canonical_record_path(root, unit)
        if canonical.exists() and (
            not canonical.is_file()
            or canonical.read_bytes() != task_path.read_bytes()
        ):
            raise V4ExecutionError("V4_LEGACY_BUNDLE_PROJECTION_MISMATCH")
        observed_bundle_members = {
            member.name
            for member in task_path.parent.iterdir()
            if member.is_file() and member.name in _LEGACY_ATTEMPT05_BUNDLE_MEMBERS
        }
        if observed_bundle_members != _LEGACY_ATTEMPT05_BUNDLE_MEMBERS:
            raise V4ExecutionError("V4_LEGACY_BUNDLE_MEMBER_SET_MISMATCH")
        for member in task_path.parent.iterdir():
            if member.is_dir() or member.name not in _LEGACY_ATTEMPT05_BUNDLE_MEMBERS:
                continue
            allowed.add(member.resolve())
    return allowed


def _reject_extra_or_partial_record_files(root: Path, units: Sequence[ScientificExecutionUnit]) -> None:
    record_root = root / "stage" / SCIENTIFIC_MVE / "records"
    if not record_root.exists():
        return
    allowed = _allowed_record_paths(root, units)
    legacy_allowed = _validated_legacy_bundle_members(root, units)
    for member in record_root.rglob("*"):
        if not member.is_file():
            continue
        if member.name.endswith(".partial") or ".partial" in member.name:
            raise V4ExecutionError("V4_RECORD_PARTIAL_ARTIFACT")
        if member.resolve() in legacy_allowed:
            continue
        if member.suffix == ".json" and member.resolve() not in allowed:
            raise V4ExecutionError("V4_RECORD_UNEXPECTED_EXTRA")
        if member.suffix != ".json":
            raise V4ExecutionError("V4_RECORD_UNEXPECTED_EXTRA")


def canonical_record_inventory(
    schedule: ScientificSchedule,
    record_paths: Sequence[Path],
    *,
    root: Path,
) -> tuple[TaskAuditRecord, ...]:
    units = _canonical_units(schedule)
    _reject_extra_or_partial_record_files(root, units)
    paths = tuple(record_paths)
    if len(paths) != 400:
        raise V4ExecutionError("V4_RECORD_COUNT_NOT_400")
    if len({path.resolve() for path in paths}) != len(paths):
        raise V4ExecutionError("V4_RECORD_DUPLICATE")
    records: list[TaskAuditRecord] = []
    seen: set[tuple[str, int, str]] = set()
    for unit, path in zip(units, paths, strict=True):
        if path.resolve() != canonical_record_path(root, unit).resolve():
            raise V4ExecutionError("V4_RECORD_CANONICAL_PATH_MISMATCH")
        record = _read_record(path)
        key = _record_key(record)
        if key in seen:
            raise V4ExecutionError("V4_RECORD_DUPLICATE")
        if key != _unit_key(unit):
            raise V4ExecutionError("V4_RECORD_ORDER_DRIFT")
        if (
            record.execution_unit_sha256 != unit.execution_unit_sha256
            or record.state_identity_sha256 != unit.state_identity_sha256
            or record.pair_identity_sha256 != unit.pair_identity_sha256
        ):
            raise V4ExecutionError("V4_RECORD_SOURCE_LINKAGE_MISMATCH")
        seen.add(key)
        records.append(record)
    return tuple(records)


def first_incomplete_unit(
    schedule: ScientificSchedule,
    root: Path,
    *,
    attempt_rows: Sequence[Mapping[str, object]] = (),
) -> V4ControllerDecision:
    units = _canonical_units(schedule)
    try:
        _reject_extra_or_partial_record_files(root, units)
    except V4ExecutionError as exc:
        return V4ControllerDecision(status="MVE_BLOCKED_RECORD_CONFLICT", reason_code=str(exc))
    for row in attempt_rows:
        state = row.get("state")
        if state == "failed":
            model_id = str(row.get("model_id", SCIENTIFIC_MODELS[0]))
            reason = (
                "V4_MODEL_A_FAILURE_BLOCKS_MODEL_B"
                if model_id == SCIENTIFIC_MODELS[0]
                else "V4_ATTEMPT_FAILURE_STOPS_STAGE"
            )
            status = (
                "MVE_BLOCKED_MODEL_A_FAILURE"
                if model_id == SCIENTIFIC_MODELS[0]
                else "MVE_BLOCKED_ATTEMPT_FAILURE"
            )
            return V4ControllerDecision(status=status, reason_code=reason)

    first_missing_index: int | None = None
    for index, unit in enumerate(units):
        path = canonical_record_path(root, unit)
        if not path.exists():
            first_missing_index = index
            break
        try:
            admit_existing_task_record(path, unit, root=root)
        except V4ExecutionError as exc:
            return V4ControllerDecision(
                status="MVE_BLOCKED_RECORD_CONFLICT",
                reason_code=str(exc),
                unit=unit,
                record_path=path,
            )
    if first_missing_index is None:
        return V4ControllerDecision(
            status="MVE_BLOCKED_FINALIZE_REQUIRED",
            reason_code="V4_EXACT_400_RECORDS_READY_FOR_CPU_FINALIZE",
        )
    for later in units[first_missing_index + 1 :]:
        if canonical_record_path(root, later).exists():
            return V4ControllerDecision(
                status="MVE_BLOCKED_RECORD_ORDER_DRIFT",
                reason_code="V4_FIRST_INCOMPLETE_CANONICAL_UNIT_ONLY",
                unit=units[first_missing_index],
                record_path=canonical_record_path(root, units[first_missing_index]),
            )
    unit = units[first_missing_index]
    return V4ControllerDecision(
        status=GPU_SELECTION_REQUIRED,
        reason_code="V4_GPU_SELECTION_REQUIRED_FOR_FIRST_INCOMPLETE_UNIT",
        unit=unit,
        record_path=canonical_record_path(root, unit),
    )


def _validate_ledger(value: V4ResourceLedger | None, label: str) -> V4ResourceLedger:
    if not isinstance(value, V4ResourceLedger):
        raise V4ExecutionError("V4_LEDGER_UNRECONCILED")
    if (
        isinstance(value.gpu_inference_seconds, bool)
        or isinstance(value.wall_runtime_seconds, bool)
        or isinstance(value.new_logical_bytes, bool)
        or isinstance(value.new_allocated_bytes, bool)
        or isinstance(value.peak_device_memory_bytes, bool)
        or not isinstance(value.gpu_inference_seconds, (int, float))
        or not isinstance(value.wall_runtime_seconds, (int, float))
        or not isinstance(value.new_logical_bytes, int)
        or not isinstance(value.new_allocated_bytes, int)
        or not isinstance(value.peak_device_memory_bytes, int)
        or not math.isfinite(float(value.gpu_inference_seconds))
        or not math.isfinite(float(value.wall_runtime_seconds))
        or value.gpu_inference_seconds < 0
        or value.wall_runtime_seconds < 0
        or value.new_logical_bytes < 0
        or value.new_allocated_bytes < 0
        or value.peak_device_memory_bytes < 0
        or value.stage not in GPU_BEARING_STAGES
        or value.model_id not in SCIENTIFIC_MODELS
        or not _is_sha(value.unit_sha256)
        or not _is_sha(value.baseline_inventory_sha256)
    ):
        raise V4ExecutionError(f"V4_LEDGER_UNRECONCILED:{label}")
    return value


def evaluate_resource_governance(
    ledger: V4ResourceLedger | None,
    *,
    predicted_next: V4ResourceLedger | None,
) -> V4ResourceDecision:
    current = _validate_ledger(ledger, "current")
    predicted = _validate_ledger(predicted_next, "predicted_next")
    if (
        current.baseline_inventory_sha256 != predicted.baseline_inventory_sha256
        or current.stage != predicted.stage
    ):
        raise V4ExecutionError("V4_LEDGER_UNRECONCILED:baseline_or_stage")
    if current.gpu_inference_seconds >= GPU_CATASTROPHE_SECONDS:
        return V4ResourceDecision("MVE_BLOCKED_RESOURCE_CATASTROPHE", "V4_CATASTROPHE_GPUH_FUSE")
    if (
        current.new_logical_bytes >= BYTE_CATASTROPHE
        or current.new_allocated_bytes >= BYTE_CATASTROPHE
    ):
        return V4ResourceDecision("MVE_BLOCKED_RESOURCE_CATASTROPHE", "V4_CATASTROPHE_STORAGE_FUSE")
    if current.gpu_inference_seconds >= GPU_TARGET_SECONDS:
        return V4ResourceDecision("MVE_BLOCKED_REAUTHORIZATION_REQUIRED", "V4_REAUTHORIZE_GPUH_TARGET_REACHED")
    if current.new_logical_bytes >= BYTE_TARGET:
        return V4ResourceDecision("MVE_BLOCKED_REAUTHORIZATION_REQUIRED", "V4_REAUTHORIZE_LOGICAL_BYTES_TARGET_REACHED")
    if current.new_allocated_bytes >= BYTE_TARGET:
        return V4ResourceDecision("MVE_BLOCKED_REAUTHORIZATION_REQUIRED", "V4_REAUTHORIZE_ALLOCATED_BYTES_TARGET_REACHED")
    if current.gpu_inference_seconds + predicted.gpu_inference_seconds > GPU_TARGET_SECONDS:
        return V4ResourceDecision("MVE_BLOCKED_REAUTHORIZATION_REQUIRED", "V4_REAUTHORIZE_GPUH_PREDICTED_NEXT_ITEM")
    if current.new_logical_bytes + predicted.new_logical_bytes > BYTE_TARGET:
        return V4ResourceDecision("MVE_BLOCKED_REAUTHORIZATION_REQUIRED", "V4_REAUTHORIZE_LOGICAL_BYTES_PREDICTED_NEXT_ITEM")
    if current.new_allocated_bytes + predicted.new_allocated_bytes > BYTE_TARGET:
        return V4ResourceDecision("MVE_BLOCKED_REAUTHORIZATION_REQUIRED", "V4_REAUTHORIZE_ALLOCATED_BYTES_PREDICTED_NEXT_ITEM")
    return V4ResourceDecision("PASS", "V4_RESOURCE_LEDGER_WITHIN_AUTHORIZED_TARGETS")


def numerical_repeat_status(
    *,
    boundary_distances: Sequence[float],
    existing_repeat_count: int,
    enters_ci: bool,
    repeat_requests: Sequence[V4RepeatRequest] = (),
    scientific_schedule: ScientificSchedule | None = None,
    expected_recipe_sha256: str | None = None,
) -> V4ControllerDecision:
    if (
        isinstance(existing_repeat_count, bool)
        or not isinstance(existing_repeat_count, int)
        or existing_repeat_count < 0
    ):
        return V4ControllerDecision(
            status="MVE_BLOCKED_REPEAT_PROTOCOL_VIOLATION",
            reason_code="V4_REPEAT_COUNT_INVALID",
        )
    distances = tuple(boundary_distances)
    if any(
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or float(value) < 0.0
        for value in distances
    ):
        return V4ControllerDecision(
            status="MVE_BLOCKED_REPEAT_PROTOCOL_VIOLATION",
            reason_code="V4_REPEAT_BOUNDARY_DISTANCE_INVALID",
        )
    if scientific_schedule is None:
        return V4ControllerDecision(
            status="MVE_BLOCKED_REPEAT_PROTOCOL_VIOLATION",
            reason_code="V4_REPEAT_SCHEDULE_REQUIRED",
        )
    if expected_recipe_sha256 is None or not _is_sha(expected_recipe_sha256):
        return V4ControllerDecision(
            status="MVE_BLOCKED_REPEAT_PROTOCOL_VIOLATION",
            reason_code="V4_REPEAT_RECIPE_SHA_REQUIRED",
        )
    requests = tuple(repeat_requests)
    if not requests:
        return V4ControllerDecision(
            status="MVE_BLOCKED_REPEAT_PROTOCOL_VIOLATION",
            reason_code="V4_REPEAT_REQUEST_REQUIRED",
        )
    if existing_repeat_count + len(requests) > 2:
        return V4ControllerDecision(
            status="MVE_BLOCKED_REPEAT_LIMIT_REACHED",
            reason_code="V4_NUMERICAL_REPEAT_MAX_TWO",
        )
    try:
        original_members = {
            unit.execution_unit_sha256
            for unit in _canonical_units(scientific_schedule)
        }
    except Exception:
        return V4ControllerDecision(
            status="MVE_BLOCKED_REPEAT_PROTOCOL_VIOLATION",
            reason_code="V4_REPEAT_SCHEDULE_INVALID",
        )
    originals: set[str] = set()
    repeats: set[str] = set()
    for request in requests:
        if (
            not isinstance(request, V4RepeatRequest)
            or request.schema_version != "georeliab-v4-numerical-repeat-request-1.0"
            or not _is_sha(request.original_unit_sha256)
            or not _is_sha(request.repeat_unit_sha256)
            or not _is_sha(request.exact_recipe_sha256)
            or request.enters_ci
            or not math.isfinite(float(request.boundary_distance))
            or request.boundary_distance not in distances
            or request.exact_recipe_sha256 != expected_recipe_sha256
            or request.original_unit_sha256 not in original_members
            or request.original_unit_sha256 in originals
            or request.repeat_unit_sha256 in repeats
            or request.original_unit_sha256 == request.repeat_unit_sha256
        ):
            return V4ControllerDecision(
                status="MVE_BLOCKED_REPEAT_PROTOCOL_VIOLATION",
                reason_code="V4_REPEAT_REQUEST_NOT_EXACT_RECIPE_LINKED",
            )
        originals.add(request.original_unit_sha256)
        repeats.add(request.repeat_unit_sha256)
    if enters_ci:
        return V4ControllerDecision(
            status="MVE_BLOCKED_REPEAT_PROTOCOL_VIOLATION",
            reason_code="V4_REPEAT_NEVER_ENTERS_CI",
        )
    if not distances or min(float(value) for value in distances) > 0.02:
        return V4ControllerDecision(
            status="MVE_BLOCKED_REPEAT_NOT_ELIGIBLE",
            reason_code="V4_REPEAT_BOUNDARY_DISTANCE_GT_0_02",
        )
    return V4ControllerDecision(
        status=GPU_SELECTION_REQUIRED,
        reason_code="V4_NUMERICAL_REPEAT_EXACT_RECIPE_GPU_SELECTION_REQUIRED",
    )


def _record_origin(source_root: Path) -> dict[str, object]:
    return {
        "schema_version": V4_RECORD_ORIGIN_SCHEMA_VERSION,
        "project_line": "v4",
        "protocol_provenance": v4_protocol_provenance(source_root),
    }


def _artifact_record(
    source_root: Path,
    *,
    record_kind: str,
    source_uri: str,
    source_sha256: str,
    source_schema_version: str,
    data: Mapping[str, object] | Sequence[object],
) -> dict[str, object]:
    return {
        "schema_version": V4_ARTIFACT_RECORD_SCHEMA_VERSION,
        "record_kind": record_kind,
        "origin": _record_origin(source_root),
        "source_uri": source_uri,
        "source_sha256": source_sha256,
        "source_schema_version": source_schema_version,
        "data": data,
    }


def _envelope(source_root: Path, artifact: Mapping[str, object]) -> dict[str, object]:
    return {
        "project_line": "v4",
        "protocol_provenance": v4_protocol_provenance(source_root),
        "artifact": dict(artifact),
    }


def _inventory_payload(records: Sequence[TaskAuditRecord], paths: Sequence[Path]) -> list[dict[str, object]]:
    return [
        {
            "index": index,
            "model_id": record.model_id,
            "scene_id": record.scene_id,
            "state_id": record.state_id,
            "execution_unit_sha256": record.execution_unit_sha256,
            "record_sha256": record.record_sha256,
            "source_uri": path.resolve().as_uri(),
            "source_sha256": _sha256_file(path),
        }
        for index, (record, path) in enumerate(zip(records, paths, strict=True))
    ]


def _object_data(value: Any) -> Mapping[str, object]:
    if isinstance(value, Mapping):
        return dict(value)
    if hasattr(value, "to_dict"):
        data = value.to_dict()
        if isinstance(data, Mapping):
            return dict(data)
    return {"schema_version": "georeliab-v4-bound-python-object-1.0", "repr": repr(value)}


def _digest_bound_summary(
    *,
    schema_version: str,
    source_sha256: str,
    item_count: int | None = None,
    extra: Mapping[str, object] | None = None,
) -> dict[str, object]:
    payload: dict[str, object] = {
        "schema_version": schema_version,
        "source_digest_sha256": source_sha256,
    }
    if item_count is not None:
        payload["item_count"] = item_count
    if extra is not None:
        payload.update(dict(extra))
    return payload


def _canonical_json_bytes_for_payload(payload: object) -> bytes:
    return json.dumps(payload, sort_keys=True, separators=(",", ":")).encode(
        "utf-8"
    ) + b"\n"


def _path_from_file_uri(uri: str) -> Path:
    parsed = urlparse(uri)
    if parsed.scheme != "file":
        raise V4ExecutionError("V4_ARTIFACT_SOURCE_URI_NOT_FILE")
    if parsed.netloc:
        return Path(f"//{parsed.netloc}{unquote(parsed.path)}")
    raw_path = unquote(parsed.path)
    if len(raw_path) >= 3 and raw_path[0] == "/" and raw_path[2] == ":":
        return Path(raw_path[1:])
    return Path(raw_path)


def _verify_artifact_source_files(artifacts: Sequence[Mapping[str, object]]) -> None:
    for envelope in artifacts:
        artifact = envelope.get("artifact")
        if not isinstance(artifact, Mapping):
            raise V4ExecutionError("V4_ARTIFACT_ENVELOPE_INVALID")
        source_uri = artifact.get("source_uri")
        source_sha256 = artifact.get("source_sha256")
        if not isinstance(source_uri, str) or not isinstance(source_sha256, str):
            raise V4ExecutionError("V4_ARTIFACT_SOURCE_BINDING_INVALID")
        if _sha256_file(_path_from_file_uri(source_uri)) != source_sha256:
            raise V4ExecutionError("V4_ARTIFACT_SOURCE_SHA_MISMATCH")

def _relative_publication_path(output_dir: Path, path: Path) -> Path:
    try:
        relative = path.resolve().relative_to(output_dir.resolve())
    except ValueError as exc:
        raise V4ExecutionError("V4_PUBLICATION_PATH_OUTSIDE_OUTPUT_DIR") from exc
    if not relative.parts or any(part in {"", ".", ".."} for part in relative.parts):
        raise V4ExecutionError("V4_PUBLICATION_PATH_OUTSIDE_OUTPUT_DIR")
    return relative


def _expected_publication_map(
    output_dir: Path,
    publications: Sequence[tuple[Path, bytes]],
) -> dict[Path, bytes]:
    expected: dict[Path, bytes] = {}
    for path, payload in publications:
        relative = _relative_publication_path(output_dir, path)
        previous = expected.get(relative)
        if previous is not None and previous != payload:
            raise V4ExecutionError("V4_IMMUTABLE_PUBLICATION_CONFLICT")
        expected[relative] = payload
    return expected


def _directory_matches_expected(output_dir: Path, expected: Mapping[Path, bytes]) -> bool:
    if not output_dir.is_dir():
        return False
    actual_files = {
        path.relative_to(output_dir)
        for path in output_dir.rglob("*")
        if path.is_file()
    }
    actual_dirs = [path for path in output_dir.rglob("*") if path.is_dir()]
    if actual_dirs or actual_files != set(expected):
        return False
    return all((output_dir / relative).read_bytes() == payload for relative, payload in expected.items())


def _verify_staged_publication(staging_dir: Path, expected: Mapping[Path, bytes]) -> None:
    if not _directory_matches_expected(staging_dir, expected):
        raise V4ExecutionError("V4_STAGED_PUBLICATION_VERIFY_FAILED")


def _write_staged_publications(staging_dir: Path, expected: Mapping[Path, bytes]) -> None:
    for relative, payload in expected.items():
        path = staging_dir / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("xb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())


def _verify_artifact_source_files_in_root(
    artifacts: Sequence[Mapping[str, object]],
    *,
    output_dir: Path,
    read_root: Path,
) -> None:
    for envelope in artifacts:
        artifact = envelope.get("artifact")
        if not isinstance(artifact, Mapping):
            raise V4ExecutionError("V4_ARTIFACT_ENVELOPE_INVALID")
        source_uri = artifact.get("source_uri")
        source_sha256 = artifact.get("source_sha256")
        if not isinstance(source_uri, str) or not isinstance(source_sha256, str):
            raise V4ExecutionError("V4_ARTIFACT_SOURCE_BINDING_INVALID")
        canonical_path = _path_from_file_uri(source_uri)
        try:
            relative = canonical_path.resolve().relative_to(output_dir.resolve())
        except ValueError:
            path = canonical_path
        else:
            path = read_root / relative
        if _sha256_file(path) != source_sha256:
            raise V4ExecutionError("V4_ARTIFACT_SOURCE_SHA_MISMATCH")


def _rename_staging_directory(staging_dir: Path, output_dir: Path) -> None:
    staging_dir.rename(output_dir)


def _after_publication_lock_acquired(
    output_dir: Path,
    expected: Mapping[Path, bytes],
) -> None:
    return None


def _publish_directory_atomically(
    *,
    output_dir: Path,
    publications: Sequence[tuple[Path, bytes]],
    artifacts: Sequence[Mapping[str, object]],
) -> None:
    expected = _expected_publication_map(output_dir, publications)
    if output_dir.exists():
        if _directory_matches_expected(output_dir, expected):
            _verify_artifact_source_files(artifacts)
            return
        raise V4ExecutionError("V4_IMMUTABLE_PUBLICATION_CONFLICT")

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging_dir = output_dir.with_name(f".{output_dir.name}.{uuid4().hex}.partial")
    lock_dir = output_dir.with_name(f".{output_dir.name}.publish.lock")
    lock_acquired = False
    try:
        staging_dir.mkdir(mode=0o700)
        _write_staged_publications(staging_dir, expected)
        _verify_staged_publication(staging_dir, expected)
        _verify_artifact_source_files_in_root(
            artifacts,
            output_dir=output_dir,
            read_root=staging_dir,
        )
        try:
            lock_dir.mkdir(mode=0o700)
        except FileExistsError as exc:
            raise V4ExecutionError("V4_PUBLICATION_LOCK_CONFLICT") from exc
        lock_acquired = True
        _after_publication_lock_acquired(output_dir, expected)
        if output_dir.exists():
            if _directory_matches_expected(output_dir, expected):
                shutil.rmtree(staging_dir, ignore_errors=True)
                return
            raise V4ExecutionError("V4_IMMUTABLE_PUBLICATION_CONFLICT")
        _rename_staging_directory(staging_dir, output_dir)
    except Exception:
        if staging_dir.exists():
            shutil.rmtree(staging_dir, ignore_errors=True)
        raise
    finally:
        if lock_acquired and lock_dir.exists():
            shutil.rmtree(lock_dir, ignore_errors=True)

def _path_sha_inventory(paths: Sequence[Path]) -> list[dict[str, object]]:
    return [
        {"path": path.resolve().as_uri(), "sha256": _sha256_file(path)}
        for path in sorted(paths, key=lambda item: item.resolve().as_posix())
    ]


def finalize_v4_scientific_bundle(
    *,
    source_root: Path,
    output_dir: Path,
    record_paths: Sequence[Path],
    scientific_schedule: ScientificSchedule,
    model_independent_states: Sequence[Any],
    split_assignment: Any,
    native_warning_calibrations: Sequence[Any],
) -> dict[str, object]:
    """Recompute WarningEvidence and atomically publish a digest-bound bundle."""

    schedule = validate_scientific_schedule(scientific_schedule)
    if len(tuple(model_independent_states)) != 200:
        raise V4ExecutionError("V4_STATE_INVENTORY_NOT_EXACT_200")
    try:
        validate_v4_split_assignment(split_assignment)
        calibrations = validate_native_warning_calibration_inventory(
            native_warning_calibrations,
            split_assignment=split_assignment,
        )
    except Exception as exc:
        raise V4ExecutionError("V4_CALIBRATION_OR_SPLIT_INVALID") from exc
    paths = tuple(record_paths)
    if len(paths) != 400:
        raise V4ExecutionError("V4_RECORD_COUNT_NOT_400")
    root = paths[0].parents[5]
    records = canonical_record_inventory(schedule, paths, root=root)
    before_inventory = _path_sha_inventory(paths)
    inventory = _inventory_payload(records, paths)
    inventory_sha256 = _sha256_json(inventory)
    evidence = build_warning_evidence(
        records,
        scientific_schedule=schedule,
        model_independent_states=model_independent_states,
        native_warning_calibrations=calibrations,
        split_assignment=split_assignment,
    )
    decision = evaluate_warning_gate(
        evidence,
        scientific_schedule=schedule,
        model_independent_states=model_independent_states,
        native_warning_calibrations=calibrations,
        split_assignment=split_assignment,
        task_records=records,
    )
    if not isinstance(decision, WarningGateDecision):
        raise V4ExecutionError("V4_WARNING_GATE_DECISION_INVALID")
    if decision.status not in ALLOWED_TERMINAL_STATUSES:
        raise V4ExecutionError("V4_TERMINAL_DECISION_UNSUPPORTED")
    after_inventory = _path_sha_inventory(record_paths)
    if before_inventory != after_inventory:
        raise V4ExecutionError("V4_FINALIZE_SOURCE_RECORD_MUTATION")

    schedule_payload = schedule.to_dict()
    inventory_payload = {"records": inventory, "inventory_sha256": inventory_sha256}
    evidence_payload = evidence.to_dict()
    decision_payload = {
        "schema_version": "georeliab-v4-warning-gate-decision-1.0",
        "status": decision.status,
        "reason_code": decision.reason_code,
        "strong_model_id": decision.strong_model_id,
        "strong_family": decision.strong_family,
    }
    governance_values = [
        ("split-assignment", split_assignment, "georeliab-v4-splits-1.0", None),
        (
            "model-independent-state-inventory",
            {"states": [_object_data(row) for row in model_independent_states]},
            "georeliab-v4-model-independent-state-inventory-1.0",
            200,
        ),
        (
            "native-warning-calibration-inventory",
            {"calibrations": [_object_data(row) for row in calibrations]},
            "georeliab-v4-native-warning-calibration-inventory-1.0",
            len(calibrations),
        ),
    ]
    governance_values.extend(
        (
            f"native-warning-calibration-{index:02d}",
            calibration,
            "georeliab-v4-native-warning-calibration-1.0",
            1,
        )
        for index, calibration in enumerate(calibrations)
    )
    governance_payloads = []
    for name, value, schema, count in governance_values:
        data = _object_data(value)
        data_bytes = _canonical_json_bytes_for_payload(data)
        governance_payloads.append((name, schema, count, data, data_bytes))

    schedule_path = output_dir / "scientific-schedule.json"
    inventory_path = output_dir / "canonical-400-record-inventory.json"
    evidence_path = output_dir / "warning-evidence.json"
    decision_path = output_dir / "warning-gate-decision.json"
    publication_path = output_dir / "v4-scientific-bundle.json"

    schedule_bytes = _canonical_json_bytes_for_payload(schedule_payload)
    inventory_bytes = _canonical_json_bytes_for_payload(inventory_payload)
    evidence_bytes = (
        evidence.canonical_json_bytes()
        if isinstance(evidence, WarningEvidence)
        else _canonical_json_bytes_for_payload(evidence_payload)
    )
    decision_bytes = _canonical_json_bytes_for_payload(decision_payload)
    schedule_sha256 = _sha256_bytes(schedule_bytes)
    inventory_file_sha256 = _sha256_bytes(inventory_bytes)
    evidence_file_sha256 = _sha256_bytes(evidence_bytes)
    decision_file_sha256 = _sha256_bytes(decision_bytes)

    artifacts = [
        _envelope(
            source_root,
            _artifact_record(
                source_root,
                record_kind="artifact",
                source_uri=path.resolve().as_uri(),
                source_sha256=_sha256_file(path),
                source_schema_version=TASK_AUDIT_RECORD_SCHEMA_VERSION,
                data=record.to_dict(),
            ),
        )
        for record, path in zip(records, paths, strict=True)
    ]
    artifacts.extend(
        [
            _envelope(
                source_root,
                _artifact_record(
                    source_root,
                    record_kind="artifact",
                    source_uri=schedule_path.resolve().as_uri(),
                    source_sha256=schedule_sha256,
                    source_schema_version=schedule.schema_version,
                    data=schedule_payload,
                ),
            ),
            _envelope(
                source_root,
                _artifact_record(
                    source_root,
                    record_kind="artifact",
                    source_uri=inventory_path.resolve().as_uri(),
                    source_sha256=inventory_file_sha256,
                    source_schema_version="georeliab-v4-canonical-400-record-inventory-1.0",
                    data=_digest_bound_summary(
                        schema_version="georeliab-v4-canonical-400-record-inventory-summary-1.0",
                        source_sha256=inventory_file_sha256,
                        item_count=len(inventory),
                        extra={"inventory_sha256": inventory_sha256},
                    ),
                ),
            ),
            _envelope(
                source_root,
                _artifact_record(
                    source_root,
                    record_kind="evidence",
                    source_uri=evidence_path.resolve().as_uri(),
                    source_sha256=evidence_file_sha256,
                    source_schema_version=WARNING_EVIDENCE_SCHEMA_VERSION,
                    data=evidence_payload,
                ),
            ),
            _envelope(
                source_root,
                _artifact_record(
                    source_root,
                    record_kind="evidence",
                    source_uri=decision_path.resolve().as_uri(),
                    source_sha256=decision_file_sha256,
                    source_schema_version="georeliab-v4-warning-gate-decision-1.0",
                    data=decision_payload,
                ),
            ),
        ]
    )
    source_publications: list[tuple[Path, bytes]] = [
        (schedule_path, schedule_bytes),
        (inventory_path, inventory_bytes),
        (evidence_path, evidence_bytes),
        (decision_path, decision_bytes),
    ]
    for name, schema, count, _data, data_bytes in governance_payloads:
        path = output_dir / f"{name}.json"
        digest = _sha256_bytes(data_bytes)
        source_publications.append((path, data_bytes))
        artifacts.append(
            _envelope(
                source_root,
                _artifact_record(
                    source_root,
                    record_kind="artifact",
                    source_uri=path.resolve().as_uri(),
                    source_sha256=digest,
                    source_schema_version=schema,
                    data=_digest_bound_summary(
                        schema_version=f"{schema}-summary",
                        source_sha256=digest,
                        item_count=count,
                    ),
                ),
            )
        )

    bundle = {
        "schema_version": V4_SCIENTIFIC_BUNDLE_SCHEMA_VERSION,
        "project_line": "v4",
        "scientific_validity": "SCIENTIFIC",
        "protocol_provenance": v4_protocol_provenance(source_root),
        "artifacts": artifacts,
    }
    try:
        validate_v4_scientific_bundle_structure(source_root, bundle)
    except V4ScienceLockError as exc:
        raise V4ExecutionError("V4_TASK1_BUNDLE_VALIDATION_FAILED") from exc

    bundle_bytes = _canonical_json_bytes_for_payload(bundle)
    _publish_directory_atomically(
        output_dir=output_dir,
        publications=(*source_publications, (publication_path, bundle_bytes)),
        artifacts=artifacts,
    )
    return {
        "schema_version": "georeliab-v4-finalize-report-1.0",
        "bundle": bundle,
        "decision": {
            "status": decision.status,
            "reason_code": decision.reason_code,
            "strong_model_id": decision.strong_model_id,
            "strong_family": decision.strong_family,
        },
        "source_inventory_before": before_inventory,
        "source_inventory_after": after_inventory,
        "source_inventory_before_sha256": _sha256_json(before_inventory),
        "source_inventory_after_sha256": _sha256_json(after_inventory),
        "bundle_path": str(publication_path),
        "bundle_sha256": _sha256_file(publication_path),
    }
