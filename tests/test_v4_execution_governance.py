from __future__ import annotations

from dataclasses import replace
import hashlib
import json
from pathlib import Path
import shutil
from urllib.parse import unquote, urlparse
from uuid import uuid4

import pytest

from georeliab_mve.v4_counterfactuals import (
    SCIENTIFIC_MODELS,
    SCIENTIFIC_STATES,
    TEST_SCENE_IDS,
    ScientificExecutionUnit,
    ScientificSchedule,
    canonical_json_sha256,
)
from georeliab_mve.v4_execution import (
    FINALIZE,
    GPU_SELECTION_REQUIRED,
    ENGINEERING_SANITY,
    NUMERICAL_REPEAT,
    SCIENTIFIC_MVE,
    V4ExecutionError,
    V4ExecutionReceipt,
    V4ResourceLedger,
    V4RepeatRequest,
    admit_existing_task_record,
    canonical_record_inventory,
    canonical_record_path,
    evaluate_resource_governance,
    finalize_v4_scientific_bundle,
    first_incomplete_unit,
    numerical_repeat_status,
    authorize_next_scientific_dispatch,
    validate_v4_gpu_receipt,
    v4_stage_entry_status,
    _path_from_file_uri as _v4_path_from_file_uri,
)
from georeliab_mve.v4_gates import MVE_SCIENTIFIC_NO_GO, WarningGateDecision
from georeliab_mve.v4_records import (
    TASK_AUDIT_RECORD_SCHEMA_VERSION,
    WARNING_EVIDENCE_SCHEMA_VERSION,
    TaskAuditRecord,
    parse_task_audit_record,
)
from georeliab_mve.v4_science_lock import (
    V4_ARTIFACT_RECORD_SCHEMA_VERSION,
    V4_PROTOCOL_ID,
    V4_PROTOCOL_SHA256,
    V4_PROTOCOL_VERSION,
    V4ScienceLockError,
)


COMMIT = "1" * 40
TREE = "2" * 40
SCHEDULE_SHA = "3" * 64
PREFLIGHT_SHA = "4" * 64
DEVICE_UUID = "GPU-" + "5" * 32


def _sha(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


@pytest.fixture()
def tmp_path() -> Path:
    path = Path(".pytest-local") / uuid4().hex
    path.mkdir(parents=True, exist_ok=False)
    try:
        yield path.resolve()
    finally:
        shutil.rmtree(path, ignore_errors=True)


def _unit(model_id: str, scene_id: int, state_id: str) -> ScientificExecutionUnit:
    provenance = {
        "schema_version": "georeliab-v4-protocol-provenance-1.0",
        "protocol_id": V4_PROTOCOL_ID,
        "protocol_version": V4_PROTOCOL_VERSION,
        "protocol_sha256": V4_PROTOCOL_SHA256,
    }
    payload = {
        "schema_version": "georeliab-v4-scientific-execution-unit-1.0",
        "protocol_provenance": provenance,
        "dataset": "DTU",
        "model_id": model_id,
        "scene_id": scene_id,
        "state_id": state_id,
        "state_identity_sha256": _sha(f"state:{scene_id}:{state_id}"),
        "pair_identity_sha256": (
            None if state_id == "L3" else _sha(f"pair:{scene_id}:{state_id}")
        ),
    }
    return ScientificExecutionUnit.from_dict(
        {**payload, "execution_unit_sha256": canonical_json_sha256(payload)}
    )


@pytest.fixture()
def schedule() -> ScientificSchedule:
    units = tuple(
        _unit(model_id, scene_id, state_id)
        for model_id in SCIENTIFIC_MODELS
        for scene_id in TEST_SCENE_IDS
        for state_id in SCIENTIFIC_STATES
    )
    provenance = {
        "schema_version": "georeliab-v4-protocol-provenance-1.0",
        "protocol_id": V4_PROTOCOL_ID,
        "protocol_version": V4_PROTOCOL_VERSION,
        "protocol_sha256": V4_PROTOCOL_SHA256,
    }
    payload = {
        "schema_version": "georeliab-v4-scientific-schedule-1.0",
        "protocol_provenance": provenance,
        "models": list(SCIENTIFIC_MODELS),
        "scene_ids": list(TEST_SCENE_IDS),
        "state_ids": list(SCIENTIFIC_STATES),
        "units": [unit.to_dict() for unit in units],
    }
    return ScientificSchedule(
        protocol_provenance_items=tuple(sorted(provenance.items())),
        models=SCIENTIFIC_MODELS,
        scene_ids=TEST_SCENE_IDS,
        state_ids=SCIENTIFIC_STATES,
        units=units,
        schedule_sha256=canonical_json_sha256(payload),
    )




def _write_preflight(tmp_path: Path) -> tuple[Path, str]:
    path = tmp_path / "preflight" / "hardware.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": "georeliab-v4-hardware-preflight-1.0",
        "status": "PASS",
        "project_commit": COMMIT,
        "project_tree": TREE,
        "scope": SCIENTIFIC_MVE,
        "stage": SCIENTIFIC_MVE,
        "requested_physical_index": 1,
        "visible_gpu_count": 1,
        "compute_process_count": 0,
        "stable_sample_count": 2,
        "environment": {
            "CUDA_VISIBLE_DEVICES": "1",
            "GEORELIAB_PHYSICAL_GPU_DEVICE": "cuda:1",
        },
        "devices": [
            {
                "physical_index": 1,
                "uuid": DEVICE_UUID,
                "model": "NVIDIA A100-SXM4-80GB",
                "driver_version": "555.55",
                "total_memory_bytes": 80_000_000_000,
                "compute_process_count": 0,
            }
        ],
        "samples": [
            {
                "sample_index": index,
                "visible_gpu_count": 1,
                "compute_process_count": 0,
                "device_uuid": DEVICE_UUID,
                "physical_index": 1,
            }
            for index in range(2)
        ],
        "model_environment_probes": [
            {
                "model_id": model_id,
                "torch_device_count": 1,
                "torch_cuda_available": True,
                "torch_current_device": 0,
                "mapped_device_uuid": DEVICE_UUID,
                "compute_process_count": 0,
            }
            for model_id in SCIENTIFIC_MODELS
        ],
    }
    path.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")
    return path, hashlib.sha256(path.read_bytes()).hexdigest()


def _rewrite_preflight(path: Path, mutate) -> str:
    payload = json.loads(path.read_text(encoding="utf-8"))
    mutate(payload)
    path.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _receipt(**updates: object) -> V4ExecutionReceipt:
    receipt = V4ExecutionReceipt(
        explicit_user_selection=True,
        project_commit=COMMIT,
        project_tree=TREE,
        protocol_id=V4_PROTOCOL_ID,
        protocol_sha256=V4_PROTOCOL_SHA256,
        scope=SCIENTIFIC_MVE,
        stage=SCIENTIFIC_MVE,
        schedule_sha256=SCHEDULE_SHA,
        hardware_preflight_path=str(updates.pop("hardware_preflight_path", "preflight/hardware.json")),
        hardware_preflight_sha256=str(updates.pop("hardware_preflight_sha256", PREFLIGHT_SHA)),
        requested_physical_index=1,
        resolved_physical_index=1,
        device_uuid=DEVICE_UUID,
        device_model="NVIDIA A100-SXM4-80GB",
        driver_version="555.55",
        total_memory_bytes=80_000_000_000,
        max_concurrent_gpus=1,
        sequential_model_execution=True,
        sequential_unit_execution=True,
        fallback_allowed=False,
        device_switch_allowed=False,
        retry_allowed=False,
        nonce="pytest-only",
    )
    return replace(receipt, **updates)


def test_file_uri_conversion_preserves_posix_absolute_and_windows_drive() -> None:
    assert _v4_path_from_file_uri("file:///srv/private/result.json").as_posix() == "/srv/private/result.json"
    assert _v4_path_from_file_uri("file:///D:/Workspace/result.json") == Path("D:/Workspace/result.json")


def test_gpu_bearing_stage_entries_stop_for_receipt_and_finalize_is_cpu_only() -> None:
    assert v4_stage_entry_status(ENGINEERING_SANITY).status == GPU_SELECTION_REQUIRED
    assert v4_stage_entry_status(SCIENTIFIC_MVE).status == GPU_SELECTION_REQUIRED
    assert v4_stage_entry_status(NUMERICAL_REPEAT).status == GPU_SELECTION_REQUIRED
    finalize = v4_stage_entry_status(FINALIZE)
    assert finalize.status == "V4_CPU_ONLY_FINALIZE_READY"
    assert finalize.reason_code == "V4_FINALIZE_NEVER_CONSUMES_GPU_RECEIPT"


def test_gpu_receipt_rejects_crossing_history_and_requires_single_sequential_gpu(
    tmp_path: Path,
) -> None:
    preflight_path, preflight_sha = _write_preflight(tmp_path)
    receipt = _receipt(
        hardware_preflight_path=preflight_path,
        hardware_preflight_sha256=preflight_sha,
    )
    assert validate_v4_gpu_receipt(
        receipt,
        project_commit=COMMIT,
        project_tree=TREE,
        scope=SCIENTIFIC_MVE,
        stage=SCIENTIFIC_MVE,
        schedule_sha256=SCHEDULE_SHA,
        hardware_preflight_path=preflight_path,
        hardware_preflight_sha256=preflight_sha,
        requested_physical_index=1,
        visible_gpu_count=1,
        active_gpu_count=0,
        lock_active=False,
    ) == receipt

    cases = [
        (_receipt(scope=FINALIZE, hardware_preflight_path=preflight_path, hardware_preflight_sha256=preflight_sha), "V4_GPU_RECEIPT_SCOPE_MISMATCH"),
        (_receipt(project_commit="9" * 40, hardware_preflight_path=preflight_path, hardware_preflight_sha256=preflight_sha), "V4_GPU_RECEIPT_GIT_MISMATCH"),
        (_receipt(protocol_sha256="a" * 64, hardware_preflight_path=preflight_path, hardware_preflight_sha256=preflight_sha), "V4_GPU_RECEIPT_PROTOCOL_MISMATCH"),
        (_receipt(schedule_sha256="b" * 64, hardware_preflight_path=preflight_path, hardware_preflight_sha256=preflight_sha), "V4_GPU_RECEIPT_SCHEDULE_MISMATCH"),
        (_receipt(requested_physical_index=0, hardware_preflight_path=preflight_path, hardware_preflight_sha256=preflight_sha), "V4_GPU_RECEIPT_DEVICE_MISMATCH"),
        (_receipt(max_concurrent_gpus=2, hardware_preflight_path=preflight_path, hardware_preflight_sha256=preflight_sha), "V4_GPU_RECEIPT_SINGLE_GPU_REQUIRED"),
        (_receipt(sequential_unit_execution=False, hardware_preflight_path=preflight_path, hardware_preflight_sha256=preflight_sha), "V4_GPU_RECEIPT_SEQUENTIAL_REQUIRED"),
        (_receipt(fallback_allowed=True, hardware_preflight_path=preflight_path, hardware_preflight_sha256=preflight_sha), "V4_GPU_RECEIPT_NO_FALLBACK_REQUIRED"),
    ]
    for bad_receipt, reason in cases:
        with pytest.raises(V4ExecutionError, match=reason):
            validate_v4_gpu_receipt(
                bad_receipt,
                project_commit=COMMIT,
                project_tree=TREE,
                scope=SCIENTIFIC_MVE,
                stage=SCIENTIFIC_MVE,
                schedule_sha256=SCHEDULE_SHA,
                hardware_preflight_path=preflight_path,
                hardware_preflight_sha256=preflight_sha,
                requested_physical_index=1,
                visible_gpu_count=1,
                active_gpu_count=0,
                lock_active=False,
            )

    for kwargs, reason in [
        ({"visible_gpu_count": 2}, "V4_GPU_RECEIPT_SINGLE_VISIBLE_GPU_REQUIRED"),
        ({"active_gpu_count": 1}, "V4_GPU_RECEIPT_NO_ACTIVE_COMPUTE_PROCESS"),
        ({"lock_active": True}, "V4_GPU_RECEIPT_LOCK_CONFLICT"),
    ]:
        with pytest.raises(V4ExecutionError, match=reason):
            validate_v4_gpu_receipt(
                receipt,
                project_commit=COMMIT,
                project_tree=TREE,
                scope=SCIENTIFIC_MVE,
                stage=SCIENTIFIC_MVE,
                schedule_sha256=SCHEDULE_SHA,
                hardware_preflight_path=preflight_path,
                hardware_preflight_sha256=preflight_sha,
                requested_physical_index=1,
                visible_gpu_count=kwargs.get("visible_gpu_count", 1),
                active_gpu_count=kwargs.get("active_gpu_count", 0),
                lock_active=kwargs.get("lock_active", False),
            )

    old_v1 = {"schema_version": "georeliab-explicit-gpu-selection-v1"}
    with pytest.raises(V4ExecutionError, match="V4_GPU_RECEIPT_SCHEMA_REQUIRED"):
        validate_v4_gpu_receipt(
            old_v1,
            project_commit=COMMIT,
            project_tree=TREE,
            scope=SCIENTIFIC_MVE,
            stage=SCIENTIFIC_MVE,
            schedule_sha256=SCHEDULE_SHA,
            hardware_preflight_path=preflight_path,
            hardware_preflight_sha256=preflight_sha,
            requested_physical_index=1,
            visible_gpu_count=1,
            active_gpu_count=0,
            lock_active=False,
        )

    wrong_schema_instance = replace(receipt, schema_version="georeliab-explicit-gpu-selection-v1")
    with pytest.raises(V4ExecutionError, match="V4_GPU_RECEIPT_SCHEMA_REQUIRED"):
        validate_v4_gpu_receipt(
            wrong_schema_instance,
            project_commit=COMMIT,
            project_tree=TREE,
            scope=SCIENTIFIC_MVE,
            stage=SCIENTIFIC_MVE,
            schedule_sha256=SCHEDULE_SHA,
            hardware_preflight_path=preflight_path,
            hardware_preflight_sha256=preflight_sha,
            requested_physical_index=1,
            visible_gpu_count=1,
            active_gpu_count=0,
            lock_active=False,
        )

    bad_preflight = tmp_path / "preflight" / "bad.json"
    bad_preflight.write_text(
        '{"schema_version":"georeliab-v4-hardware-preflight-1.0","status":"FAIL"}\n',
        encoding="utf-8",
    )
    bad_sha = hashlib.sha256(bad_preflight.read_bytes()).hexdigest()
    bad_receipt = _receipt(hardware_preflight_path=bad_preflight, hardware_preflight_sha256=bad_sha)
    with pytest.raises(V4ExecutionError, match="V4_GPU_PREFLIGHT_SCHEMA_REQUIRED|V4_GPU_PREFLIGHT_NOT_PASSING"):
        validate_v4_gpu_receipt(
            bad_receipt,
            project_commit=COMMIT,
            project_tree=TREE,
            scope=SCIENTIFIC_MVE,
            stage=SCIENTIFIC_MVE,
            schedule_sha256=SCHEDULE_SHA,
            hardware_preflight_path=bad_preflight,
            hardware_preflight_sha256=bad_sha,
            requested_physical_index=1,
            visible_gpu_count=1,
            active_gpu_count=0,
            lock_active=False,
        )




@pytest.mark.parametrize(
    ("receipt_update", "reason"),
    [
        ({"requested_physical_index": True}, "V4_GPU_RECEIPT_DEVICE_MISMATCH"),
        ({"resolved_physical_index": True}, "V4_GPU_RECEIPT_DEVICE_MISMATCH"),
        ({"total_memory_bytes": True}, "V4_GPU_RECEIPT_DEVICE_MISMATCH"),
        ({"max_concurrent_gpus": True}, "V4_GPU_RECEIPT_SINGLE_GPU_REQUIRED"),
        ({"device_uuid": ""}, "V4_GPU_RECEIPT_DEVICE_MISMATCH"),
        ({"device_model": ""}, "V4_GPU_RECEIPT_DEVICE_MISMATCH"),
        ({"driver_version": ""}, "V4_GPU_RECEIPT_DEVICE_MISMATCH"),
    ],
)
def test_gpu_receipt_rejects_bool_numeric_impostors_and_empty_metadata(
    tmp_path: Path,
    receipt_update: dict[str, object],
    reason: str,
) -> None:
    preflight_path, preflight_sha = _write_preflight(tmp_path)
    receipt = _receipt(
        hardware_preflight_path=preflight_path,
        hardware_preflight_sha256=preflight_sha,
        **receipt_update,
    )
    with pytest.raises(V4ExecutionError, match=reason):
        validate_v4_gpu_receipt(
            receipt,
            project_commit=COMMIT,
            project_tree=TREE,
            scope=SCIENTIFIC_MVE,
            stage=SCIENTIFIC_MVE,
            schedule_sha256=SCHEDULE_SHA,
            hardware_preflight_path=preflight_path,
            hardware_preflight_sha256=preflight_sha,
            requested_physical_index=1,
            visible_gpu_count=1,
            active_gpu_count=0,
            lock_active=False,
        )


@pytest.mark.parametrize(
    ("call_args", "reason"),
    [
        ({"requested_physical_index": True}, "V4_GPU_RECEIPT_DEVICE_MISMATCH"),
        ({"visible_gpu_count": True}, "V4_GPU_RECEIPT_SINGLE_VISIBLE_GPU_REQUIRED"),
        ({"active_gpu_count": False}, "V4_GPU_RECEIPT_NO_ACTIVE_COMPUTE_PROCESS"),
    ],
)
def test_gpu_receipt_rejects_bool_numeric_call_args(
    tmp_path: Path,
    call_args: dict[str, object],
    reason: str,
) -> None:
    preflight_path, preflight_sha = _write_preflight(tmp_path)
    receipt = _receipt(
        hardware_preflight_path=preflight_path,
        hardware_preflight_sha256=preflight_sha,
    )
    kwargs = {
        "requested_physical_index": 1,
        "visible_gpu_count": 1,
        "active_gpu_count": 0,
    }
    kwargs.update(call_args)
    with pytest.raises(V4ExecutionError, match=reason):
        validate_v4_gpu_receipt(
            receipt,
            project_commit=COMMIT,
            project_tree=TREE,
            scope=SCIENTIFIC_MVE,
            stage=SCIENTIFIC_MVE,
            schedule_sha256=SCHEDULE_SHA,
            hardware_preflight_path=preflight_path,
            hardware_preflight_sha256=preflight_sha,
            lock_active=False,
            **kwargs,
        )


@pytest.mark.parametrize(
    ("mutate", "reason"),
    [
        (lambda row: row.__setitem__("requested_physical_index", True), "V4_GPU_PREFLIGHT_NOT_PASSING_EXACT_REQUEST"),
        (lambda row: row.__setitem__("visible_gpu_count", True), "V4_GPU_PREFLIGHT_NOT_PASSING_EXACT_REQUEST"),
        (lambda row: row.__setitem__("compute_process_count", False), "V4_GPU_PREFLIGHT_NOT_PASSING_EXACT_REQUEST"),
        (lambda row: row.__setitem__("stable_sample_count", True), "V4_GPU_PREFLIGHT_NOT_PASSING_EXACT_REQUEST"),
        (lambda row: row["samples"][0].__setitem__("sample_index", False), "V4_GPU_PREFLIGHT_STABLE_SAMPLES_REQUIRED"),
        (lambda row: row["samples"][0].__setitem__("visible_gpu_count", True), "V4_GPU_PREFLIGHT_STABLE_SAMPLES_REQUIRED"),
        (lambda row: row["samples"][0].__setitem__("compute_process_count", False), "V4_GPU_PREFLIGHT_STABLE_SAMPLES_REQUIRED"),
        (lambda row: row["samples"][0].__setitem__("physical_index", True), "V4_GPU_PREFLIGHT_STABLE_SAMPLES_REQUIRED"),
        (lambda row: row["model_environment_probes"][0].__setitem__("torch_device_count", True), "V4_GPU_PREFLIGHT_TORCH_PROBES_REQUIRED"),
        (lambda row: row["model_environment_probes"][0].__setitem__("torch_current_device", False), "V4_GPU_PREFLIGHT_TORCH_PROBES_REQUIRED"),
        (lambda row: row["model_environment_probes"][0].__setitem__("compute_process_count", False), "V4_GPU_PREFLIGHT_TORCH_PROBES_REQUIRED"),
        (lambda row: row["devices"][0].__setitem__("physical_index", True), "V4_GPU_RECEIPT_DEVICE_MISMATCH"),
        (lambda row: row["devices"][0].__setitem__("total_memory_bytes", True), "V4_GPU_RECEIPT_DEVICE_MISMATCH"),
        (lambda row: row["devices"][0].__setitem__("compute_process_count", False), "V4_GPU_RECEIPT_DEVICE_MISMATCH"),
        (lambda row: row["devices"][0].__setitem__("uuid", ""), "V4_GPU_RECEIPT_DEVICE_MISMATCH"),
        (lambda row: row["devices"][0].__setitem__("model", ""), "V4_GPU_RECEIPT_DEVICE_MISMATCH"),
        (lambda row: row["devices"][0].__setitem__("driver_version", ""), "V4_GPU_RECEIPT_DEVICE_MISMATCH"),
    ],
)
def test_gpu_preflight_rejects_bool_numeric_impostors_and_empty_metadata(
    tmp_path: Path,
    mutate,
    reason: str,
) -> None:
    preflight_path, _preflight_sha = _write_preflight(tmp_path)
    preflight_sha = _rewrite_preflight(preflight_path, mutate)
    receipt = _receipt(
        hardware_preflight_path=preflight_path,
        hardware_preflight_sha256=preflight_sha,
    )
    with pytest.raises(V4ExecutionError, match=reason):
        validate_v4_gpu_receipt(
            receipt,
            project_commit=COMMIT,
            project_tree=TREE,
            scope=SCIENTIFIC_MVE,
            stage=SCIENTIFIC_MVE,
            schedule_sha256=SCHEDULE_SHA,
            hardware_preflight_path=preflight_path,
            hardware_preflight_sha256=preflight_sha,
            requested_physical_index=1,
            visible_gpu_count=1,
            active_gpu_count=0,
            lock_active=False,
        )


def test_authorized_dispatch_validates_receipt_and_expires_at_stage_stop(
    schedule: ScientificSchedule,
    tmp_path: Path,
) -> None:
    preflight_path, preflight_sha = _write_preflight(tmp_path)
    receipt = _receipt(
        schedule_sha256=schedule.schedule_sha256,
        hardware_preflight_path=preflight_path,
        hardware_preflight_sha256=preflight_sha,
    )
    decision = authorize_next_scientific_dispatch(
        schedule,
        tmp_path / "runtime",
        receipt,
        project_commit=COMMIT,
        project_tree=TREE,
        hardware_preflight_path=preflight_path,
        hardware_preflight_sha256=preflight_sha,
        requested_physical_index=1,
        visible_gpu_count=1,
        active_gpu_count=0,
        lock_active=False,
    )
    assert decision.status == "V4_DISPATCH_AUTHORIZED"
    assert decision.unit == schedule.units[0]

    with pytest.raises(V4ExecutionError, match="V4_GPU_RECEIPT_EXPIRED_AT_STAGE_STOP"):
        authorize_next_scientific_dispatch(
            schedule,
            tmp_path / "runtime",
            receipt,
            project_commit=COMMIT,
            project_tree=TREE,
            hardware_preflight_path=preflight_path,
            hardware_preflight_sha256=preflight_sha,
            requested_physical_index=1,
            visible_gpu_count=1,
            active_gpu_count=0,
            lock_active=False,
            stage_stopped=True,
        )


def test_exact_400_inventory_order_resume_and_invalid_record_retention(
    schedule: ScientificSchedule,
    tmp_path: Path,
) -> None:
    first = first_incomplete_unit(schedule, tmp_path)
    assert first.status == GPU_SELECTION_REQUIRED
    assert first.unit == schedule.units[0]

    paths = [canonical_record_path(tmp_path, unit) for unit in schedule.units]
    records = [
        _minimal_record(unit, valid=(index != 3), reason_code="VALID" if index != 3 else "INVALID_MODEL_OUTPUT")
        for index, unit in enumerate(schedule.units)
    ]
    _write_record(paths[0], records[0])
    _write_record(paths[1], records[1])
    assert first_incomplete_unit(schedule, tmp_path).unit == schedule.units[2]

    _write_record(paths[3], records[3])
    stopped = first_incomplete_unit(schedule, tmp_path)
    assert stopped.status == "MVE_BLOCKED_RECORD_ORDER_DRIFT"
    assert stopped.reason_code == "V4_FIRST_INCOMPLETE_CANONICAL_UNIT_ONLY"

    paths[3].unlink()
    _write_record(paths[2], records[2])
    _write_record(paths[3], records[3])
    assert first_incomplete_unit(schedule, tmp_path).unit == schedule.units[4]

    paths[2].write_text("partial", encoding="utf-8")
    stopped = first_incomplete_unit(schedule, tmp_path)
    assert stopped.status == "MVE_BLOCKED_RECORD_CONFLICT"
    assert stopped.reason_code == "V4_EXISTING_ARTIFACT_INVALID"

    with pytest.raises(V4ExecutionError, match="V4_RECORD_COUNT_NOT_400"):
        canonical_record_inventory(schedule, paths[:-1], root=tmp_path)
    with pytest.raises(V4ExecutionError, match="V4_RECORD_DUPLICATE"):
        canonical_record_inventory(schedule, (*paths[:-1], paths[0]), root=tmp_path)


def test_resume_rejects_extra_and_partial_record_files(
    schedule: ScientificSchedule,
    tmp_path: Path,
) -> None:
    extra = tmp_path / "stage" / SCIENTIFIC_MVE / "records" / "VGGT" / "extra.json"
    extra.parent.mkdir(parents=True, exist_ok=True)
    extra.write_text("{}\n", encoding="utf-8")
    decision = first_incomplete_unit(schedule, tmp_path)
    assert decision.status == "MVE_BLOCKED_RECORD_CONFLICT"
    assert decision.reason_code == "V4_RECORD_UNEXPECTED_EXTRA"

    extra.unlink()
    partial = canonical_record_path(tmp_path, schedule.units[0]).with_suffix(".json.partial")
    partial.parent.mkdir(parents=True, exist_ok=True)
    partial.write_text("partial\n", encoding="utf-8")
    decision = first_incomplete_unit(schedule, tmp_path)
    assert decision.reason_code == "V4_RECORD_PARTIAL_ARTIFACT"


def test_model_a_failure_blocks_model_b(schedule: ScientificSchedule) -> None:
    rows = [{"unit_sha256": schedule.units[0].execution_unit_sha256, "state": "failed"}]
    decision = first_incomplete_unit(schedule, Path("unused"), attempt_rows=rows)
    assert decision.status == "MVE_BLOCKED_MODEL_A_FAILURE"
    assert decision.reason_code == "V4_MODEL_A_FAILURE_BLOCKS_MODEL_B"


def test_admit_existing_artifact_requires_hash_identity_linkage_and_path(
    schedule: ScientificSchedule,
    tmp_path: Path,
) -> None:
    unit = schedule.units[0]
    record = _minimal_record(unit)
    path = canonical_record_path(tmp_path, unit)
    _write_record(path, record)
    assert admit_existing_task_record(path, unit, root=tmp_path) == record

    wrong_path = tmp_path / "wrong.json"
    _write_record(wrong_path, record)
    with pytest.raises(V4ExecutionError, match="V4_RECORD_CANONICAL_PATH_MISMATCH"):
        admit_existing_task_record(wrong_path, unit, root=tmp_path)

    other_record = _minimal_record(schedule.units[1])
    _write_record(path, other_record)
    with pytest.raises(V4ExecutionError, match="V4_RECORD_SOURCE_LINKAGE_MISMATCH"):
        admit_existing_task_record(path, unit, root=tmp_path)


def test_canonical_inventory_admits_only_a_byte_identical_legacy_attempt05_bundle(
    schedule: ScientificSchedule,
    tmp_path: Path,
) -> None:
    paths = [canonical_record_path(tmp_path, unit) for unit in schedule.units]
    records = [_minimal_record(unit) for unit in schedule.units]
    for path, record in zip(paths, records, strict=True):
        _write_record(path, record)
    legacy = paths[0].parent
    (legacy / "task_audit_record.json").write_bytes(paths[0].read_bytes())
    for name in (
        "audit_record.json",
        "prediction_artifact.json",
        "run_manifest.json",
    ):
        (legacy / name).write_text("{}\n", encoding="utf-8")
    for name in (
        "dense_audit.npz",
        "geometry_prediction.npz",
        "gt_points.npz",
        "native_confidence.npz",
        "valid_mask.npz",
    ):
        (legacy / name).write_bytes(b"legacy")

    assert canonical_record_inventory(schedule, paths, root=tmp_path) == tuple(records)

    (legacy / "task_audit_record.json").write_text("{}\n", encoding="utf-8")
    with pytest.raises(V4ExecutionError):
        canonical_record_inventory(schedule, paths, root=tmp_path)


def test_first_incomplete_allows_valid_legacy_bundle_before_projection(
    schedule: ScientificSchedule,
    tmp_path: Path,
) -> None:
    unit = schedule.units[0]
    record = _minimal_record(unit)
    legacy = (
        tmp_path
        / "stage"
        / SCIENTIFIC_MVE
        / "records"
        / unit.model_id
        / f"scan{unit.scene_id:03d}"
    )
    legacy.mkdir(parents=True, exist_ok=True)
    (legacy / "task_audit_record.json").write_bytes(record.canonical_json_bytes())
    for name in (
        "audit_record.json",
        "prediction_artifact.json",
        "run_manifest.json",
    ):
        (legacy / name).write_text("{}\n", encoding="utf-8")
    for name in (
        "dense_audit.npz",
        "geometry_prediction.npz",
        "gt_points.npz",
        "native_confidence.npz",
        "valid_mask.npz",
    ):
        (legacy / name).write_bytes(b"legacy")

    decision = first_incomplete_unit(schedule, tmp_path)
    assert decision.status == GPU_SELECTION_REQUIRED
    assert decision.unit == unit


def test_resource_limits_stop_independently_and_fail_closed() -> None:
    under = V4ResourceLedger(
        gpu_inference_seconds=10.0,
        wall_runtime_seconds=11.0,
        new_logical_bytes=100,
        new_allocated_bytes=100,
        peak_device_memory_bytes=100,
        stage=SCIENTIFIC_MVE,
        model_id="VGGT",
        unit_sha256="a" * 64,
    )
    assert evaluate_resource_governance(under, predicted_next=under).status == "PASS"

    assert evaluate_resource_governance(
        replace(under, gpu_inference_seconds=35 * 3600),
        predicted_next=under,
    ).reason_code == "V4_REAUTHORIZE_GPUH_TARGET_REACHED"
    assert evaluate_resource_governance(
        replace(under, new_logical_bytes=150_000_000_000),
        predicted_next=under,
    ).reason_code == "V4_REAUTHORIZE_LOGICAL_BYTES_TARGET_REACHED"
    assert evaluate_resource_governance(
        replace(under, new_allocated_bytes=150_000_000_000),
        predicted_next=under,
    ).reason_code == "V4_REAUTHORIZE_ALLOCATED_BYTES_TARGET_REACHED"
    assert evaluate_resource_governance(
        replace(under, gpu_inference_seconds=50 * 3600),
        predicted_next=under,
    ).reason_code == "V4_CATASTROPHE_GPUH_FUSE"
    assert evaluate_resource_governance(
        replace(under, new_allocated_bytes=1_000_000_000_000),
        predicted_next=under,
    ).reason_code == "V4_CATASTROPHE_STORAGE_FUSE"
    with pytest.raises(V4ExecutionError, match="V4_LEDGER_UNRECONCILED"):
        evaluate_resource_governance(None, predicted_next=under)
    with pytest.raises(V4ExecutionError, match="V4_LEDGER_UNRECONCILED"):
        evaluate_resource_governance(
            replace(under, new_logical_bytes=1.5),
            predicted_next=under,
        )
    with pytest.raises(V4ExecutionError, match="V4_LEDGER_UNRECONCILED"):
        evaluate_resource_governance(
            replace(under, peak_device_memory_bytes="100"),
            predicted_next=under,
        )


def test_resource_ledger_reconciles_related_baseline_stage_and_finite_values() -> None:
    base = V4ResourceLedger(
        gpu_inference_seconds=1.0,
        wall_runtime_seconds=1.0,
        new_logical_bytes=1,
        new_allocated_bytes=1,
        peak_device_memory_bytes=1,
        stage=SCIENTIFIC_MVE,
        model_id="VGGT",
        unit_sha256="a" * 64,
        baseline_inventory_sha256="b" * 64,
    )
    with pytest.raises(V4ExecutionError, match="baseline_or_stage"):
        evaluate_resource_governance(
            base,
            predicted_next=replace(base, baseline_inventory_sha256="c" * 64),
        )
    with pytest.raises(V4ExecutionError, match="baseline_or_stage"):
        evaluate_resource_governance(
            base,
            predicted_next=replace(base, stage=ENGINEERING_SANITY),
        )
    with pytest.raises(V4ExecutionError, match="V4_LEDGER_UNRECONCILED"):
        evaluate_resource_governance(
            replace(base, gpu_inference_seconds=float("nan")),
            predicted_next=base,
        )


@pytest.mark.parametrize(
    ("distances", "existing", "expected"),
    [
        ((0.02,), 0, "MVE_BLOCKED_REPEAT_PROTOCOL_VIOLATION"),
        ((0.0200000000001,), 0, "MVE_BLOCKED_REPEAT_NOT_ELIGIBLE"),
        ((0.01,), 2, "MVE_BLOCKED_REPEAT_LIMIT_REACHED"),
    ],
)
def test_repeat_boundary_exact_max_two_and_no_ci_inflation(
    schedule: ScientificSchedule,
    distances: tuple[float, ...],
    existing: int,
    expected: str,
) -> None:
    recipe_sha = "c" * 64
    linked = V4RepeatRequest(
        original_unit_sha256=schedule.units[0].execution_unit_sha256,
        repeat_unit_sha256="b" * 64,
        boundary_distance=distances[0],
        exact_recipe_sha256=recipe_sha,
        enters_ci=False,
    )
    request_args = {"repeat_requests": (linked,)} if expected != "MVE_BLOCKED_REPEAT_PROTOCOL_VIOLATION" else {}
    assert numerical_repeat_status(
        boundary_distances=distances,
        existing_repeat_count=existing,
        enters_ci=False,
        scientific_schedule=schedule,
        expected_recipe_sha256=recipe_sha,
        **request_args,
    ).status == expected
    exact_linked = V4RepeatRequest(
        original_unit_sha256=schedule.units[0].execution_unit_sha256,
        repeat_unit_sha256="d" * 64,
        boundary_distance=0.02,
        exact_recipe_sha256=recipe_sha,
        enters_ci=False,
    )
    assert numerical_repeat_status(
        boundary_distances=(0.01,),
        existing_repeat_count=0,
        enters_ci=True,
        repeat_requests=(replace(exact_linked, boundary_distance=0.01),),
        scientific_schedule=schedule,
        expected_recipe_sha256=recipe_sha,
    ).reason_code == "V4_REPEAT_NEVER_ENTERS_CI"
    assert numerical_repeat_status(
        boundary_distances=(0.02,),
        existing_repeat_count=0,
        enters_ci=False,
        repeat_requests=(exact_linked,),
        scientific_schedule=schedule,
        expected_recipe_sha256=recipe_sha,
    ).status == GPU_SELECTION_REQUIRED
    assert numerical_repeat_status(
        boundary_distances=(0.02,),
        existing_repeat_count=0,
        enters_ci=False,
        repeat_requests=(exact_linked,),
        expected_recipe_sha256=recipe_sha,
    ).reason_code == "V4_REPEAT_SCHEDULE_REQUIRED"
    assert numerical_repeat_status(
        boundary_distances=(0.02,),
        existing_repeat_count=0,
        enters_ci=False,
        repeat_requests=(exact_linked,),
        scientific_schedule=schedule,
    ).reason_code == "V4_REPEAT_RECIPE_SHA_REQUIRED"
    assert numerical_repeat_status(
        boundary_distances=(0.02,),
        existing_repeat_count=0,
        enters_ci=False,
        repeat_requests=(exact_linked,),
        scientific_schedule=schedule,
        expected_recipe_sha256="not-a-sha",
    ).reason_code == "V4_REPEAT_RECIPE_SHA_REQUIRED"
    second_linked = replace(exact_linked, repeat_unit_sha256="e" * 64)
    assert numerical_repeat_status(
        boundary_distances=(0.02,),
        existing_repeat_count=1,
        enters_ci=False,
        repeat_requests=(exact_linked, second_linked),
        scientific_schedule=schedule,
        expected_recipe_sha256=recipe_sha,
    ).reason_code == "V4_NUMERICAL_REPEAT_MAX_TWO"
    bad = replace(exact_linked, enters_ci=True)
    assert numerical_repeat_status(
        boundary_distances=(0.02,),
        existing_repeat_count=0,
        enters_ci=False,
        repeat_requests=(bad,),
        scientific_schedule=schedule,
        expected_recipe_sha256=recipe_sha,
    ).reason_code == "V4_REPEAT_REQUEST_NOT_EXACT_RECIPE_LINKED"
    duplicate = (exact_linked, replace(exact_linked, repeat_unit_sha256=exact_linked.repeat_unit_sha256))
    assert numerical_repeat_status(
        boundary_distances=(0.02,),
        existing_repeat_count=0,
        enters_ci=False,
        repeat_requests=duplicate,
        scientific_schedule=schedule,
        expected_recipe_sha256=recipe_sha,
    ).reason_code == "V4_REPEAT_REQUEST_NOT_EXACT_RECIPE_LINKED"


def _fake_finalize_inputs(
    schedule: ScientificSchedule,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    output_name: str,
) -> dict[str, object]:
    records = tuple(_minimal_record(unit) for unit in schedule.units)
    record_paths = []
    for unit, record in zip(schedule.units, records, strict=True):
        path = canonical_record_path(tmp_path / f"runtime-{output_name}", unit)
        _write_record(path, record)
        record_paths.append(path)
    monkeypatch.setattr(
        "georeliab_mve.v4_execution.build_warning_evidence",
        lambda task_records, **kwargs: _FakeEvidence(),
    )
    monkeypatch.setattr(
        "georeliab_mve.v4_execution.evaluate_warning_gate",
        lambda value, **kwargs: WarningGateDecision(
            status=MVE_SCIENTIFIC_NO_GO,
            reason_code="FROZEN_WARNING_GATE_NOT_MET",
        ),
    )
    monkeypatch.setattr(
        "georeliab_mve.v4_execution.validate_v4_split_assignment",
        lambda _split: None,
    )
    monkeypatch.setattr(
        "georeliab_mve.v4_execution.validate_native_warning_calibration_inventory",
        lambda calibrations, *, split_assignment: calibrations,
    )
    return {
        "source_root": Path(__file__).resolve().parents[1],
        "output_dir": tmp_path / output_name,
        "record_paths": record_paths,
        "scientific_schedule": schedule,
        "model_independent_states": tuple(object() for _ in range(200)),
        "split_assignment": object(),
        "native_warning_calibrations": (object(), object()),
    }


def _partial_dirs(parent: Path) -> list[Path]:
    return sorted(parent.glob(".*.partial"))


def test_finalize_rejects_bad_record_before_creating_output_dir(
    schedule: ScientificSchedule,
    tmp_path: Path,
) -> None:
    record_paths = [canonical_record_path(tmp_path / "runtime", unit) for unit in schedule.units]
    _write_record(record_paths[0], _minimal_record(schedule.units[0]))
    output = tmp_path / "published"
    with pytest.raises(V4ExecutionError):
        finalize_v4_scientific_bundle(
            source_root=Path(__file__).resolve().parents[1],
            output_dir=output,
            record_paths=record_paths,
            scientific_schedule=schedule,
            model_independent_states=(),
            split_assignment=object(),
            native_warning_calibrations=(),
        )
    assert not output.exists()


def test_finalize_recomputes_gate_wraps_digest_bound_sources_and_is_read_only(
    schedule: ScientificSchedule,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    records = tuple(_minimal_record(unit) for unit in schedule.units)
    record_paths = []
    for unit, record in zip(schedule.units, records, strict=True):
        path = canonical_record_path(tmp_path / "runtime", unit)
        _write_record(path, record)
        record_paths.append(path)
    before = {path: path.read_bytes() for path in record_paths}

    evidence = _FakeEvidence()
    calls = {}

    def fake_build_warning_evidence(task_records, **kwargs):
        calls["build_records"] = tuple(task_records)
        calls["build_kwargs"] = kwargs
        return evidence

    def fake_evaluate_warning_gate(value, **kwargs):
        calls["gate_value"] = value
        calls["gate_kwargs"] = kwargs
        return WarningGateDecision(
            status=MVE_SCIENTIFIC_NO_GO,
            reason_code="FROZEN_WARNING_GATE_NOT_MET",
        )

    monkeypatch.setattr(
        "georeliab_mve.v4_execution.build_warning_evidence",
        fake_build_warning_evidence,
    )
    monkeypatch.setattr(
        "georeliab_mve.v4_execution.evaluate_warning_gate",
        fake_evaluate_warning_gate,
    )
    monkeypatch.setattr(
        "georeliab_mve.v4_execution.validate_v4_split_assignment",
        lambda _split: None,
    )
    monkeypatch.setattr(
        "georeliab_mve.v4_execution.validate_native_warning_calibration_inventory",
        lambda calibrations, *, split_assignment: calibrations,
    )

    states = tuple(object() for _ in range(200))
    split_assignment = object()
    calibrations = (object(), object())
    output_dir = tmp_path / "published"
    bundle = finalize_v4_scientific_bundle(
        source_root=Path(__file__).resolve().parents[1],
        output_dir=output_dir,
        record_paths=record_paths,
        scientific_schedule=schedule,
        model_independent_states=states,
        split_assignment=split_assignment,
        native_warning_calibrations=calibrations,
    )

    assert calls["build_records"] == records
    assert calls["gate_value"] is evidence
    assert calls["gate_kwargs"]["task_records"] == records
    assert {path: path.read_bytes() for path in record_paths} == before
    assert bundle["bundle"]["scientific_validity"] == "SCIENTIFIC"
    assert bundle["decision"]["status"] == MVE_SCIENTIFIC_NO_GO
    assert bundle["source_inventory_before_sha256"] == bundle["source_inventory_after_sha256"]
    artifact_records = [row["artifact"] for row in bundle["bundle"]["artifacts"]]
    assert artifact_records[0]["source_schema_version"] == TASK_AUDIT_RECORD_SCHEMA_VERSION
    assert any(
        row["source_schema_version"] == WARNING_EVIDENCE_SCHEMA_VERSION
        for row in artifact_records
    )
    assert all(
        row["schema_version"] == V4_ARTIFACT_RECORD_SCHEMA_VERSION
        and row["source_uri"]
        and len(row["source_sha256"]) == 64
        for row in artifact_records
    )
    assert (output_dir / "v4-scientific-bundle.json").is_file()
    assert (output_dir / "warning-gate-decision.json").is_file()
    for row in artifact_records:
        source_path = _file_uri_path(row["source_uri"])
        assert hashlib.sha256(source_path.read_bytes()).hexdigest() == row["source_sha256"]
    second = finalize_v4_scientific_bundle(
        source_root=Path(__file__).resolve().parents[1],
        output_dir=output_dir,
        record_paths=record_paths,
        scientific_schedule=schedule,
        model_independent_states=states,
        split_assignment=split_assignment,
        native_warning_calibrations=calibrations,
    )
    assert second["bundle_sha256"] == bundle["bundle_sha256"]


def test_finalize_validates_bundle_before_publication(
    schedule: ScientificSchedule,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    records = tuple(_minimal_record(unit) for unit in schedule.units)
    record_paths = []
    for unit, record in zip(schedule.units, records, strict=True):
        path = canonical_record_path(tmp_path / "runtime", unit)
        _write_record(path, record)
        record_paths.append(path)

    monkeypatch.setattr(
        "georeliab_mve.v4_execution.build_warning_evidence",
        lambda task_records, **kwargs: _FakeEvidence(),
    )
    monkeypatch.setattr(
        "georeliab_mve.v4_execution.evaluate_warning_gate",
        lambda value, **kwargs: WarningGateDecision(
            status=MVE_SCIENTIFIC_NO_GO,
            reason_code="FROZEN_WARNING_GATE_NOT_MET",
        ),
    )
    monkeypatch.setattr(
        "georeliab_mve.v4_execution.validate_v4_split_assignment",
        lambda _split: None,
    )
    monkeypatch.setattr(
        "georeliab_mve.v4_execution.validate_native_warning_calibration_inventory",
        lambda calibrations, *, split_assignment: calibrations,
    )
    monkeypatch.setattr(
        "georeliab_mve.v4_execution.validate_v4_scientific_bundle_structure",
        lambda _root, _bundle: (_ for _ in ()).throw(V4ScienceLockError("boom")),
    )

    output = tmp_path / "published-fail"
    with pytest.raises(V4ExecutionError, match="V4_TASK1_BUNDLE_VALIDATION_FAILED"):
        finalize_v4_scientific_bundle(
            source_root=Path(__file__).resolve().parents[1],
            output_dir=output,
            record_paths=record_paths,
            scientific_schedule=schedule,
            model_independent_states=tuple(object() for _ in range(200)),
            split_assignment=object(),
            native_warning_calibrations=(object(), object()),
        )
    assert not output.exists()


def test_finalize_publication_conflict_fails_closed(
    schedule: ScientificSchedule,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    records = tuple(_minimal_record(unit) for unit in schedule.units)
    record_paths = []
    for unit, record in zip(schedule.units, records, strict=True):
        path = canonical_record_path(tmp_path / "runtime", unit)
        _write_record(path, record)
        record_paths.append(path)

    monkeypatch.setattr(
        "georeliab_mve.v4_execution.build_warning_evidence",
        lambda task_records, **kwargs: _FakeEvidence(),
    )
    monkeypatch.setattr(
        "georeliab_mve.v4_execution.evaluate_warning_gate",
        lambda value, **kwargs: WarningGateDecision(
            status=MVE_SCIENTIFIC_NO_GO,
            reason_code="FROZEN_WARNING_GATE_NOT_MET",
        ),
    )
    monkeypatch.setattr(
        "georeliab_mve.v4_execution.validate_v4_split_assignment",
        lambda _split: None,
    )
    monkeypatch.setattr(
        "georeliab_mve.v4_execution.validate_native_warning_calibration_inventory",
        lambda calibrations, *, split_assignment: calibrations,
    )

    output = tmp_path / "published-conflict"
    output.mkdir()
    (output / "warning-gate-decision.json").write_text("{}\n", encoding="utf-8")
    with pytest.raises(V4ExecutionError, match="V4_IMMUTABLE_PUBLICATION_CONFLICT"):
        finalize_v4_scientific_bundle(
            source_root=Path(__file__).resolve().parents[1],
            output_dir=output,
            record_paths=record_paths,
            scientific_schedule=schedule,
            model_independent_states=tuple(object() for _ in range(200)),
            split_assignment=object(),
            native_warning_calibrations=(object(), object()),
        )
    assert (output / "warning-gate-decision.json").read_text(encoding="utf-8") == "{}\n"
    assert not (output / "scientific-schedule.json").exists()
    assert not (output / "canonical-400-record-inventory.json").exists()
    assert not (output / "warning-evidence.json").exists()
    assert not (output / "v4-scientific-bundle.json").exists()


def test_finalize_mid_write_failure_leaves_canonical_absent_and_cleans_staging(
    schedule: ScientificSchedule,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    kwargs = _fake_finalize_inputs(schedule, tmp_path, monkeypatch, output_name="published-write-fail")

    def fail_after_first_write(staging_dir: Path, expected):
        relative, payload = next(iter(expected.items()))
        path = staging_dir / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
        raise V4ExecutionError("INJECTED_WRITE_FAILURE")

    monkeypatch.setattr(
        "georeliab_mve.v4_execution._write_staged_publications",
        fail_after_first_write,
    )
    with pytest.raises(V4ExecutionError, match="INJECTED_WRITE_FAILURE"):
        finalize_v4_scientific_bundle(**kwargs)
    assert not kwargs["output_dir"].exists()
    assert _partial_dirs(tmp_path) == []


def test_finalize_staged_verify_failure_leaves_canonical_absent_and_cleans_staging(
    schedule: ScientificSchedule,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    kwargs = _fake_finalize_inputs(schedule, tmp_path, monkeypatch, output_name="published-verify-fail")
    monkeypatch.setattr(
        "georeliab_mve.v4_execution._verify_staged_publication",
        lambda _staging_dir, _expected: (_ for _ in ()).throw(
            V4ExecutionError("INJECTED_VERIFY_FAILURE")
        ),
    )
    with pytest.raises(V4ExecutionError, match="INJECTED_VERIFY_FAILURE"):
        finalize_v4_scientific_bundle(**kwargs)
    assert not kwargs["output_dir"].exists()
    assert _partial_dirs(tmp_path) == []


def test_finalize_publication_lock_conflict_leaves_canonical_absent_and_cleans_staging(
    schedule: ScientificSchedule,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    kwargs = _fake_finalize_inputs(schedule, tmp_path, monkeypatch, output_name="published-lock-fail")
    lock_dir = tmp_path / ".published-lock-fail.publish.lock"
    lock_dir.mkdir()
    with pytest.raises(V4ExecutionError, match="V4_PUBLICATION_LOCK_CONFLICT"):
        finalize_v4_scientific_bundle(**kwargs)
    assert not kwargs["output_dir"].exists()
    assert _partial_dirs(tmp_path) == []
    assert lock_dir.is_dir()


def test_finalize_idempotent_race_after_lock_cleans_owned_staging(
    schedule: ScientificSchedule,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    kwargs = _fake_finalize_inputs(schedule, tmp_path, monkeypatch, output_name="published-race-exact")

    def materialize_exact_output_after_lock(output_dir: Path, expected) -> None:
        output_dir.mkdir()
        for relative, payload in expected.items():
            path = output_dir / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(payload)

    monkeypatch.setattr(
        "georeliab_mve.v4_execution._after_publication_lock_acquired",
        materialize_exact_output_after_lock,
    )
    finalize_v4_scientific_bundle(**kwargs)
    assert kwargs["output_dir"].is_dir()
    assert _partial_dirs(tmp_path) == []
    assert not (tmp_path / ".published-race-exact.publish.lock").exists()


def test_finalize_rename_failure_leaves_canonical_absent_and_cleans_staging(
    schedule: ScientificSchedule,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    kwargs = _fake_finalize_inputs(schedule, tmp_path, monkeypatch, output_name="published-rename-fail")
    monkeypatch.setattr(
        "georeliab_mve.v4_execution._rename_staging_directory",
        lambda _staging_dir, _output_dir: (_ for _ in ()).throw(
            OSError("INJECTED_RENAME_FAILURE")
        ),
    )
    with pytest.raises(OSError, match="INJECTED_RENAME_FAILURE"):
        finalize_v4_scientific_bundle(**kwargs)
    assert not kwargs["output_dir"].exists()
    assert _partial_dirs(tmp_path) == []


def _file_uri_path(uri: str) -> Path:
    parsed = urlparse(uri)
    assert parsed.scheme == "file"
    if parsed.netloc:
        return Path(f"//{parsed.netloc}{unquote(parsed.path)}")
    raw_path = unquote(parsed.path)
    if len(raw_path) >= 3 and raw_path[0] == "/" and raw_path[2] == ":":
        return Path(raw_path[1:])
    return Path(raw_path)


def _minimal_record(
    unit: ScientificExecutionUnit,
    *,
    valid: bool = False,
    reason_code: str = "INVALID_MODEL_OUTPUT",
) -> TaskAuditRecord:
    payload = {
        "schema_version": TASK_AUDIT_RECORD_SCHEMA_VERSION,
        "protocol_provenance": unit.protocol_provenance,
        "dataset": "DTU",
        "model_id": unit.model_id,
        "scene_id": unit.scene_id,
        "state_id": unit.state_id,
        "ordered_view_ids": [1, 7, 13, 19, 25, 31, 37, 43],
        "state_identity_sha256": unit.state_identity_sha256,
        "pair_identity_sha256": unit.pair_identity_sha256,
        "execution_unit_sha256": unit.execution_unit_sha256,
        "valid": valid,
        "reason_code": reason_code,
        "point_main_loss": 1.0,
        "fscore_1mm": 0.0,
        "fscore_2mm": 0.0,
        "fscore_5mm": 0.0,
        "median_predicted_error_mm": 1_000_000_000_000.0,
        "static_rank": -1.0,
        "static_rank_defined": False,
        "static_rank_reason_code": "INVALID_MODEL_OUTPUT",
        "pose_pairs": [
            {
                "view_a": view_a,
                "view_b": view_b,
                "rotation_error_deg": 180.0,
                "translation_direction_error_deg": 180.0,
                "pair_error_deg": 180.0,
            }
            for i, view_a in enumerate((1, 7, 13, 19, 25, 31, 37, 43))
            for view_b in (1, 7, 13, 19, 25, 31, 37, 43)[i + 1 :]
        ],
        "auc_5deg": 0.0,
        "auc_10deg": 0.0,
        "auc_20deg": 0.0,
        "pose_main_loss": 1.0,
        "median_pair_error_deg": 180.0,
        "pose_failure": True,
        "native_warning_score": 1.0,
        "calibration_identifier": "6" * 64,
        "alarm_threshold": 0.5,
        "alarm": True,
    }
    payload["record_sha256"] = canonical_json_sha256(payload)
    return parse_task_audit_record(payload)


def _write_record(path: Path, record: TaskAuditRecord) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(record.canonical_json_bytes())


class _FakeEvidence:
    evidence_sha256 = "7" * 64
    schema_version = WARNING_EVIDENCE_SCHEMA_VERSION

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "evidence_sha256": self.evidence_sha256,
        }

    def canonical_json_bytes(self) -> bytes:
        return b'{"evidence_sha256":"%s","schema_version":"%s"}\n' % (
            self.evidence_sha256.encode("ascii"),
            self.schema_version.encode("ascii"),
        )
