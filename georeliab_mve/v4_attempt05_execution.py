"""Attempt-05 execution controller for the GeoReliab v4 MVE.

This module is intentionally a controller layer.  It consumes the Attempt-04
authorization, binds all runtime paths to that signed grant, and exposes
fail-closed gates for dispatch, resume, resource budgets, ledgering, and final
CPU publication.  It does not call the legacy runner, delete partial outputs,
retry work, switch devices, or synthesize scientific warning sentinels.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
from typing import Any

import numpy as np

from .v4_attempt04_authorization import (
    AUTHORIZED_FINALIZER,
    BYTE_CATASTROPHE,
    BYTE_TARGET,
    GPU_CATASTROPHE_SECONDS,
    GPU_TARGET_SECONDS,
    MANIFEST_FILE_SHA256,
    SCHEDULE_FILE_SHA256,
    SCIENTIFIC_ANCHOR_COMMIT,
    SCIENTIFIC_ANCHOR_TREE,
    _EXPECTED_CLOSURE_DIGESTS as ATTEMPT04_CLOSURE_DIGESTS,
    validate_attempt04_execution_authorization,
)
from .v4_counterfactuals import (
    FOG_STATES,
    ModelIndependentState,
    SCIENTIFIC_MODELS,
    TEST_SCENE_IDS,
    ScientificSchedule,
    build_scientific_schedule,
    validate_scientific_schedule,
)
from .v4_execution import (
    V4ControllerDecision,
    V4ExecutionError,
    V4ResourceDecision,
    canonical_record_path,
    finalize_v4_scientific_bundle,
    first_incomplete_unit,
)
from .v4_science_lock import V4_PROTOCOL_ID, V4_PROTOCOL_SHA256


ATTEMPT_ID = "attempt-05"
ATTEMPT05_RUN_NAME = "v4-mve-attempt-05"
START_RECEIPT_SCHEMA = "georeliab-v4-attempt-05-start-receipt-1.0"
LEDGER_SCHEMA = "georeliab-v4-attempt-05-hash-chain-ledger-1.0"
DTU_PROJECTION_SCHEMA = "georeliab-v4-attempt-05-dtu-projection-c2w-1.0"
Q90_FREEZE_SCHEMA = "georeliab-v4-attempt-05-q90-freeze-1.0"
PREFLIGHT_SCHEMA = "georeliab-v4-attempt-05-execution-preflight-1.0"


@dataclass(frozen=True, slots=True)
class Attempt05AuthorizedContext:
    authorization_path: Path
    authorization_sha256: str
    authorization: Mapping[str, Any]
    runtime_root: Path
    run_root: Path
    artifact_root: Path
    gpu_ledger_path: Path
    final_evidence_path: Path
    tooling_commit: str
    tooling_tree: str
    selected_gpu: Mapping[str, Any]


@dataclass(frozen=True, slots=True)
class DTUProjectionDecomposition:
    projection: tuple[tuple[float, float, float, float], ...]
    intrinsic_k: tuple[tuple[float, float, float], ...]
    world_to_camera_rotation: tuple[tuple[float, float, float], ...]
    camera_center_world: tuple[float, float, float]
    camera_to_world: tuple[tuple[float, float, float, float], ...]
    max_reprojection_abs_error: float
    schema_version: str = DTU_PROJECTION_SCHEMA

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "projection": [list(row) for row in self.projection],
            "intrinsic_k": [list(row) for row in self.intrinsic_k],
            "world_to_camera_rotation": [
                list(row) for row in self.world_to_camera_rotation
            ],
            "camera_center_world": list(self.camera_center_world),
            "camera_to_world": [list(row) for row in self.camera_to_world],
            "max_reprojection_abs_error": self.max_reprojection_abs_error,
        }


def _canonical_json_bytes(value: object) -> bytes:
    return (
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("ascii")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sha256_json(value: object) -> str:
    return _sha256_bytes(_canonical_json_bytes(value))


def _is_sha(value: object, length: int = 64) -> bool:
    return (
        isinstance(value, str)
        and len(value) == length
        and value == value.lower()
        and all(character in "0123456789abcdef" for character in value)
    )


def _under_root(root: Path, path: Path) -> Path:
    resolved_root = root.resolve()
    resolved = path.resolve()
    try:
        resolved.relative_to(resolved_root)
    except ValueError as exc:
        raise V4ExecutionError("V4_ATTEMPT05_RUNTIME_PATH_ESCAPE") from exc
    return resolved


def _expected_authorized_scope() -> dict[str, object]:
    return {
        "models": list(SCIENTIFIC_MODELS),
        "model_count": 2,
        "dataset": "DTU",
        "paired_lighting_states": ["L1", "L2", "L3", "L4", "L5", "L6", "L7"],
        "rectified_member_count": 960,
        "synthetic_corruption": "Koschmieder fog",
        "synthetic_severity_axis": "beta-only",
        "scientific_unit_count": 400,
        "primary_endpoint": "Pose",
        "supporting_evidence": ["Fusion", "F-score"],
        "single_gpu": True,
        "models_sequential": True,
        "units_sequential": True,
        "fallback_allowed": False,
        "device_switch_allowed": False,
        "forbidden": [
            "UAVLight",
            "v4.1",
            "third model",
            "second corruption family",
            "seed expansion",
            "severity expansion",
            "scene expansion",
            "view expansion",
            "model matrix expansion",
        ],
    }


def _require_attempt05_runtime_paths(
    authorization: Mapping[str, Any],
) -> tuple[Path, Path, Path, Path, Path]:
    runtime_root_value = authorization.get("runtime_root")
    runtime_paths = authorization.get("runtime_paths")
    if not isinstance(runtime_root_value, str) or not isinstance(runtime_paths, Mapping):
        raise V4ExecutionError("V4_ATTEMPT05_RUNTIME_PATH_ESCAPE")
    runtime_root = Path(runtime_root_value)
    if not runtime_root.is_absolute():
        raise V4ExecutionError("V4_ATTEMPT05_RUNTIME_PATH_ESCAPE")
    required_keys = {
        "run_root",
        "artifact_root",
        "gpu_ledger_path",
        "final_evidence_path",
    }
    if set(runtime_paths) != required_keys:
        raise V4ExecutionError("V4_ATTEMPT05_RUNTIME_PATH_ESCAPE")
    for value in runtime_paths.values():
        if not isinstance(value, str):
            raise V4ExecutionError("V4_ATTEMPT05_RUNTIME_PATH_ESCAPE")
        _under_root(runtime_root, Path(value))

    runtime_root = runtime_root.resolve()
    run_root = runtime_root / "runs" / ATTEMPT05_RUN_NAME
    artifact_root = runtime_root / "artifacts" / ATTEMPT05_RUN_NAME
    gpu_ledger_path = runtime_root / "logs" / "ledgers" / f"{ATTEMPT05_RUN_NAME}.jsonl"
    final_evidence_path = artifact_root / "final-evidence"
    return runtime_root, run_root, artifact_root, gpu_ledger_path, final_evidence_path


def _validate_attempt05_authorization_payload(
    authorization: Mapping[str, Any],
) -> None:
    if authorization.get("status") != "V4_MVE_EXECUTION_AUTHORIZED":
        raise V4ExecutionError("V4_ATTEMPT05_AUTHORIZATION_REQUIRED")
    if (
        authorization.get("scientific_anchor_commit") != SCIENTIFIC_ANCHOR_COMMIT
        or authorization.get("scientific_anchor_tree") != SCIENTIFIC_ANCHOR_TREE
        or authorization.get("protocol_id") != V4_PROTOCOL_ID
        or authorization.get("protocol_sha256") != V4_PROTOCOL_SHA256
        or authorization.get("schedule_file_sha256") != SCHEDULE_FILE_SHA256
        or authorization.get("rectified_manifest_sha256") != MANIFEST_FILE_SHA256
        or authorization.get("closure_digests") != ATTEMPT04_CLOSURE_DIGESTS
    ):
        raise V4ExecutionError("V4_ATTEMPT05_AUTHORIZATION_SCOPE_MISMATCH")
    if authorization.get("authorized_scope") != _expected_authorized_scope():
        raise V4ExecutionError("V4_ATTEMPT05_AUTHORIZATION_SCOPE_MISMATCH")
    if authorization.get("budget") != {
        "authorization_gpu_seconds": GPU_TARGET_SECONDS,
        "authorization_storage_bytes": BYTE_TARGET,
        "catastrophe_gpu_seconds": GPU_CATASTROPHE_SECONDS,
        "catastrophe_storage_bytes": BYTE_CATASTROPHE,
    }:
        raise V4ExecutionError("V4_ATTEMPT05_BUDGET_SCOPE_MISMATCH")
    if authorization.get("finalizer") != AUTHORIZED_FINALIZER:
        raise V4ExecutionError("V4_ATTEMPT05_FINALIZER_SCOPE_MISMATCH")
    if (
        authorization.get("execution_lock_created") is not False
        or authorization.get("gpu_ledger_created") is not False
        or authorization.get("dispatcher_called") is not False
        or authorization.get("torch_probe_invocations") != 0
        or authorization.get("model_loads") != 0
        or authorization.get("model_forwards") != 0
        or authorization.get("gpu_inference_seconds") != 0
        or authorization.get("scientific_result") != "NO_SCIENTIFIC_RESULT"
    ):
        raise V4ExecutionError("V4_ATTEMPT05_AUTHORIZATION_SCOPE_MISMATCH")
    selected = authorization.get("selected_gpu")
    if (
        not isinstance(selected, Mapping)
        or not str(selected.get("uuid", "")).startswith("GPU-")
        or not selected.get("pci_bus_id")
        or type(selected.get("index")) is not int
        or selected.get("model") != "NVIDIA A100 80GB PCIe"
    ):
        raise V4ExecutionError("V4_ATTEMPT05_GPU_IDENTITY_UNPROVEN")
    _require_attempt05_runtime_paths(authorization)


def _authorized_context_from_payload(
    authorization: Mapping[str, Any],
    authorization_path: Path,
) -> Attempt05AuthorizedContext:
    _validate_attempt05_authorization_payload(authorization)
    runtime_root, run_root, artifact_root, gpu_ledger_path, final_evidence_path = (
        _require_attempt05_runtime_paths(authorization)
    )
    return Attempt05AuthorizedContext(
        authorization_path=authorization_path.resolve(),
        authorization_sha256=(
            _sha256_file(authorization_path)
            if authorization_path.is_file()
            else _sha256_json(dict(authorization))
        ),
        authorization=dict(authorization),
        runtime_root=runtime_root,
        run_root=run_root,
        artifact_root=artifact_root,
        gpu_ledger_path=gpu_ledger_path,
        final_evidence_path=final_evidence_path,
        tooling_commit=str(authorization["tooling_commit"]),
        tooling_tree=str(authorization["tooling_tree"]),
        selected_gpu=dict(authorization["selected_gpu"]),
    )


def load_attempt05_authorized_context(
    *,
    authorization_path: Path,
) -> Attempt05AuthorizedContext:
    authorization = validate_attempt04_execution_authorization(authorization_path)
    return _authorized_context_from_payload(authorization, authorization_path)




def _schedule_key_set(schedule: ScientificSchedule) -> list[list[object]]:
    validated = validate_scientific_schedule(schedule)
    return sorted(
        [unit.model_id, unit.scene_id, unit.state_id]
        for unit in validated.units
    )


def build_attempt05_scientific_schedule(
    model_independent_states: Sequence[ModelIndependentState],
) -> ScientificSchedule:
    """Build the 400-unit scientific schedule from the 200-state input closure."""

    try:
        return build_scientific_schedule(tuple(model_independent_states))
    except Exception as exc:
        raise V4ExecutionError("V4_ATTEMPT05_SCIENTIFIC_SCHEDULE_BUILD_FAILED") from exc



def _validate_calibration_schedule(
    calibration_schedule: Sequence[Mapping[str, object]],
) -> tuple[dict[str, object], ...]:
    rows = tuple(calibration_schedule)
    if len(rows) != 40 or any(not isinstance(row, Mapping) for row in rows):
        raise V4ExecutionError("V4_ATTEMPT05_CALIBRATION_SCHEDULE_REQUIRED")
    normalized: list[dict[str, object]] = []
    seen: set[tuple[str, int, str]] = set()
    by_model: dict[str, int] = {model: 0 for model in SCIENTIFIC_MODELS}
    for row in rows:
        model_id = row.get("model_id")
        scene_id = row.get("scene_id")
        state_id = row.get("state_id")
        unit_sha = row.get("calibration_unit_sha256")
        if (
            model_id not in SCIENTIFIC_MODELS
            or type(scene_id) is not int
            or scene_id in TEST_SCENE_IDS
            or state_id != "L3"
            or not _is_sha(unit_sha)
        ):
            raise V4ExecutionError("V4_ATTEMPT05_CALIBRATION_SCHEDULE_REQUIRED")
        key = (str(model_id), scene_id, "L3")
        if key in seen:
            raise V4ExecutionError("V4_ATTEMPT05_CALIBRATION_SCHEDULE_REQUIRED")
        seen.add(key)
        by_model[str(model_id)] += 1
        normalized.append(
            {
                "model_id": str(model_id),
                "scene_id": scene_id,
                "state_id": "L3",
                "calibration_unit_sha256": str(unit_sha),
            }
        )
    if any(count != 20 for count in by_model.values()):
        raise V4ExecutionError("V4_ATTEMPT05_CALIBRATION_SCHEDULE_REQUIRED")
    return tuple(sorted(normalized, key=lambda row: (str(row["model_id"]), int(row["scene_id"]))))

def _cpu_input_closure(
    *,
    schedule: ScientificSchedule,
    model_independent_states: Sequence[object],
    calibration_schedule: Sequence[Mapping[str, object]],
) -> dict[str, int]:
    validated = validate_scientific_schedule(schedule)
    units = tuple(validated.units)
    if len(units) != 400:
        raise V4ExecutionError("V4_SCHEDULE_NOT_EXACT_400")
    if len(tuple(model_independent_states)) != 200:
        raise V4ExecutionError("V4_STATE_INVENTORY_NOT_EXACT_200")
    calibration_rows = _validate_calibration_schedule(calibration_schedule)
    test_l3 = sum(unit.state_id == "L3" for unit in units)
    fog_units = sum(unit.state_id in FOG_STATES for unit in units)
    return {
        "calibration_l3_units": 40,
        "scientific_units": len(units),
        "rectified_non_l3_members": 960,
        "test_l3_units": test_l3,
        "calibration_l3_units_against_native_warning": 40,
        "fog_units_bound_to_l3": fog_units,
        "model_independent_states": len(tuple(model_independent_states)),
        "calibration_schedule_units": len(calibration_rows),
    }


def _start_receipt_path(context: Attempt05AuthorizedContext) -> Path:
    return context.run_root / "v4-attempt05-start-receipt.json"




def _preflight_path(context: Attempt05AuthorizedContext) -> Path:
    return context.run_root / "v4-attempt05-execution-preflight.json"


def _sample_device(sample: Mapping[str, object]) -> Mapping[str, object]:
    devices = sample.get("devices")
    if not isinstance(devices, Sequence) or isinstance(devices, (str, bytes, bytearray)) or len(devices) != 1:
        raise V4ExecutionError("V4_ATTEMPT05_PREFLIGHT_SINGLE_AUTHORIZED_GPU_REQUIRED")
    device = devices[0]
    if not isinstance(device, Mapping):
        raise V4ExecutionError("V4_ATTEMPT05_PREFLIGHT_SINGLE_AUTHORIZED_GPU_REQUIRED")
    return device


def _compute_process_count(device: Mapping[str, object]) -> int:
    if "compute_process_count" in device:
        value = device.get("compute_process_count")
        if type(value) is not int or value < 0:
            raise V4ExecutionError("V4_ATTEMPT05_PREFLIGHT_COMPUTE_PROCESS_PRESENT")
        return value
    processes = device.get("compute_processes")
    if isinstance(processes, Sequence) and not isinstance(processes, (str, bytes, bytearray)):
        return len(processes)
    raise V4ExecutionError("V4_ATTEMPT05_PREFLIGHT_COMPUTE_PROCESS_PRESENT")


def _default_revision_checker() -> Mapping[str, object]:
    root = Path(__file__).resolve().parents[1]
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        tree = subprocess.run(
            ["git", "rev-parse", "HEAD^{tree}"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError) as exc:
        raise V4ExecutionError("V4_ATTEMPT05_TOOLING_REVISION_UNAVAILABLE") from exc
    return {"commit": commit, "tree": tree, "clean": status == ""}


def _validate_actual_revision(
    revision: Mapping[str, object],
    *,
    attempt05_tooling_commit: str,
    attempt05_tooling_tree: str,
) -> dict[str, object]:
    commit = revision.get("commit")
    tree = revision.get("tree")
    clean = revision.get("clean")
    if commit != attempt05_tooling_commit or tree != attempt05_tooling_tree:
        raise V4ExecutionError("V4_ATTEMPT05_TOOLING_REVISION_MISMATCH")
    if clean is not True:
        raise V4ExecutionError("V4_ATTEMPT05_TOOLING_TREE_DIRTY")
    return {"commit": str(commit), "tree": str(tree), "clean": True}


def _validate_preflight_payload(
    payload: Mapping[str, Any],
    *,
    context: Attempt05AuthorizedContext,
) -> None:
    digest = payload.get("preflight_sha256")
    unsigned = {key: value for key, value in payload.items() if key != "preflight_sha256"}
    if (
        payload.get("schema_version") != PREFLIGHT_SCHEMA
        or payload.get("attempt_id") != ATTEMPT_ID
        or payload.get("attempt04_authorization_sha256") != context.authorization_sha256
        or payload.get("sample_count") != 3
        or payload.get("sample_interval_seconds") != 5.0
        or payload.get("output_roots_absent") is not True
        or payload.get("ledger_absent") is not True
        or payload.get("dispatcher_called") != 0
        or payload.get("model_loads") != 0
        or payload.get("model_forwards") != 0
        or payload.get("gpu_inference_seconds") != 0
        or not _is_sha(payload.get("attempt05_tooling_commit"), 40)
        or not _is_sha(payload.get("attempt05_tooling_tree"), 40)
        or payload.get("actual_tooling_revision") != {
            "commit": payload.get("attempt05_tooling_commit"),
            "tree": payload.get("attempt05_tooling_tree"),
            "clean": True,
        }
        or not _is_sha(digest)
        or _sha256_json(unsigned) != digest
    ):
        raise V4ExecutionError("V4_ATTEMPT05_PREFLIGHT_TAMPER")
    selected = payload.get("authorized_gpu")
    if selected != context.selected_gpu:
        raise V4ExecutionError("V4_ATTEMPT05_PREFLIGHT_GPU_IDENTITY_MISMATCH")


def create_attempt05_execution_preflight(
    *,
    authorization_path: Path,
    attempt05_tooling_commit: str,
    attempt05_tooling_tree: str,
    nvidia_smi_sampler: Callable[[], Mapping[str, object]],
    sleeper: Callable[[float], object],
    cuda_mapping_probe: Callable[[], Mapping[str, object]] | None = None,
    revision_checker: Callable[[], Mapping[str, object]] | None = None,
) -> dict[str, object]:
    if not _is_sha(attempt05_tooling_commit, 40) or not _is_sha(attempt05_tooling_tree, 40):
        raise V4ExecutionError("V4_ATTEMPT05_TOOLING_REVISION_REQUIRED")
    context = load_attempt05_authorized_context(authorization_path=authorization_path)
    path = _preflight_path(context)
    if path.exists() or path.with_name(path.name + ".partial").exists():
        raise V4ExecutionError("V4_ATTEMPT05_PREFLIGHT_IMMUTABLE")
    if context.run_root.exists() or context.artifact_root.exists() or context.gpu_ledger_path.exists():
        raise V4ExecutionError("V4_ATTEMPT05_PREFLIGHT_OUTPUT_COLLISION")
    actual_revision = _validate_actual_revision(
        (revision_checker or _default_revision_checker)(),
        attempt05_tooling_commit=attempt05_tooling_commit,
        attempt05_tooling_tree=attempt05_tooling_tree,
    )
    samples: list[dict[str, object]] = []
    selected = context.selected_gpu
    stable_total: int | None = None
    stable_free: int | None = None
    for index in range(3):
        raw = nvidia_smi_sampler()
        if not isinstance(raw, Mapping):
            raise V4ExecutionError("V4_ATTEMPT05_PREFLIGHT_SAMPLE_INVALID")
        device = _sample_device(raw)
        total = device.get("total_memory_bytes")
        free = device.get("free_memory_bytes")
        util = device.get("utilization_gpu_percent")
        if (
            device.get("uuid") != selected.get("uuid")
            or device.get("pci_bus_id") != selected.get("pci_bus_id")
            or device.get("index") != selected.get("index")
        ):
            raise V4ExecutionError("V4_ATTEMPT05_PREFLIGHT_GPU_IDENTITY_MISMATCH")
        if _compute_process_count(device) != 0:
            raise V4ExecutionError("V4_ATTEMPT05_PREFLIGHT_COMPUTE_PROCESS_PRESENT")
        if util != 0:
            raise V4ExecutionError("V4_ATTEMPT05_PREFLIGHT_GPU_NOT_IDLE")
        if type(total) is not int or type(free) is not int or free < 16 * 1024 * 1024 * 1024:
            raise V4ExecutionError("V4_ATTEMPT05_PREFLIGHT_MEMORY_NOT_STABLE")
        if stable_total is None:
            stable_total = total
            stable_free = free
        elif total != stable_total or free != stable_free:
            raise V4ExecutionError("V4_ATTEMPT05_PREFLIGHT_MEMORY_NOT_STABLE")
        samples.append(
            {
                "sample_index": index,
                "uuid": str(device["uuid"]),
                "pci_bus_id": str(device["pci_bus_id"]),
                "index": int(device["index"]),
                "utilization_gpu_percent": 0,
                "total_memory_bytes": total,
                "free_memory_bytes": free,
                "compute_process_count": 0,
            }
        )
        if index != 2:
            sleeper(5.0)
    cuda_probe_payload: dict[str, object] | None = None
    if cuda_mapping_probe is not None:
        probe = cuda_mapping_probe()
        if not isinstance(probe, Mapping):
            raise V4ExecutionError("V4_ATTEMPT05_CUDA_MAPPING_PROBE_INVALID")
        if (
            probe.get("visible_device_count") != 1
            or probe.get("mapped_uuid") != selected.get("uuid")
            or probe.get("mapped_pci_bus_id") != selected.get("pci_bus_id")
            or probe.get("model_loads") != 0
            or probe.get("model_forwards") != 0
            or probe.get("checkpoint_loads") != 0
        ):
            raise V4ExecutionError("V4_ATTEMPT05_CUDA_MAPPING_PROBE_INVALID")
        cuda_probe_payload = dict(probe)
    payload: dict[str, object] = {
        "schema_version": PREFLIGHT_SCHEMA,
        "attempt_id": ATTEMPT_ID,
        "attempt04_authorization_sha256": context.authorization_sha256,
        "attempt05_tooling_commit": attempt05_tooling_commit,
        "attempt05_tooling_tree": attempt05_tooling_tree,
        "actual_tooling_revision": actual_revision,
        "authorized_gpu": dict(selected),
        "sample_count": 3,
        "sample_interval_seconds": 5.0,
        "samples": samples,
        "cuda_mapping_probe": cuda_probe_payload,
        "output_roots_absent": True,
        "ledger_absent": True,
        "dispatcher_called": 0,
        "model_loads": 0,
        "model_forwards": 0,
        "gpu_inference_seconds": 0,
    }
    payload["preflight_sha256"] = _sha256_json(payload)
    _write_immutable_json(path, payload)
    return {**payload, "preflight_path": str(path), "preflight_file_sha256": _sha256_file(path)}


def validate_attempt05_execution_preflight(
    path: str | Path,
    *,
    authorization_path: Path,
) -> dict[str, object]:
    context = load_attempt05_authorized_context(authorization_path=authorization_path)
    preflight_path = Path(path)
    if preflight_path.resolve() != _preflight_path(context).resolve():
        raise V4ExecutionError("V4_ATTEMPT05_PREFLIGHT_PATH_MISMATCH")
    payload = _load_json(preflight_path)
    _validate_preflight_payload(payload, context=context)
    return {**payload, "preflight_path": str(preflight_path), "preflight_file_sha256": _sha256_file(preflight_path)}

def _write_immutable_json(path: Path, payload: Mapping[str, object]) -> None:
    if path.exists() or path.with_name(path.name + ".partial").exists():
        raise V4ExecutionError("V4_ATTEMPT05_START_RECEIPT_IMMUTABLE")
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".partial")
    try:
        with partial.open("xb") as handle:
            handle.write(_canonical_json_bytes(payload))
            handle.flush()
            os.fsync(handle.fileno())
        partial.replace(path)
    finally:
        try:
            partial.unlink()
        except FileNotFoundError:
            pass


def create_attempt05_start_receipt(
    *,
    authorization_path: Path,
    schedule: ScientificSchedule,
    model_independent_states: Sequence[object],
    split_assignment: object,
    calibration_schedule: Sequence[object],
    resume: bool = False,
    attempt05_tooling_commit: str | None = None,
    attempt05_tooling_tree: str | None = None,
    input_storage: Mapping[str, object] | None = None,
) -> dict[str, object]:
    if type(resume) is not bool:
        raise V4ExecutionError("V4_ATTEMPT05_RESUME_FLAG_INVALID")
    context = load_attempt05_authorized_context(authorization_path=authorization_path)
    if (attempt05_tooling_commit is None) != (attempt05_tooling_tree is None):
        raise V4ExecutionError("V4_ATTEMPT05_TOOLING_REVISION_REQUIRED")
    if attempt05_tooling_commit is None or attempt05_tooling_tree is None:
        raise V4ExecutionError("V4_ATTEMPT05_TOOLING_REVISION_REQUIRED")
    tooling_commit = attempt05_tooling_commit
    tooling_tree = attempt05_tooling_tree
    if not _is_sha(tooling_commit, 40) or not _is_sha(tooling_tree, 40):
        raise V4ExecutionError("V4_ATTEMPT05_TOOLING_REVISION_REQUIRED")
    preflight = validate_attempt05_execution_preflight(_preflight_path(context), authorization_path=authorization_path)
    if (
        preflight.get("attempt05_tooling_commit") != tooling_commit
        or preflight.get("attempt05_tooling_tree") != tooling_tree
    ):
        raise V4ExecutionError("V4_ATTEMPT05_PREFLIGHT_TOOLING_MISMATCH")
    receipt_path = _start_receipt_path(context)
    if resume:
        return _load_start_receipt(context)
    closure = _cpu_input_closure(
        schedule=schedule,
        model_independent_states=model_independent_states,
        calibration_schedule=calibration_schedule,
    )
    if not isinstance(input_storage, Mapping):
        raise V4ExecutionError("V4_ATTEMPT05_INPUT_STORAGE_BINDING_INVALID")
    storage = dict(input_storage)
    logical_bytes = storage.get("logical_bytes")
    allocated_bytes = storage.get("allocated_bytes")
    input_closure_sha256 = storage.get("input_closure_sha256")
    if (
        storage.get("scope")
        != "NEW_CALIBRATION_L3_FOG_PNG_AND_INPUT_CLOSURE_"
        "PAYLOADS_EXCLUDING_MANIFEST"
        or
        type(logical_bytes) is not int
        or type(allocated_bytes) is not int
        or logical_bytes < 0
        or allocated_bytes < 0
        or not _is_sha(input_closure_sha256, 64)
    ):
        raise V4ExecutionError("V4_ATTEMPT05_INPUT_STORAGE_BINDING_INVALID")
    payload = {
        "schema_version": START_RECEIPT_SCHEMA,
        "attempt_id": ATTEMPT_ID,
        "attempt04_consumed": True,
        "attempt04_authorization_path": str(context.authorization_path),
        "attempt04_authorization_sha256": context.authorization_sha256,
        "attempt04_tooling_commit": context.tooling_commit,
        "attempt04_tooling_tree": context.tooling_tree,
        "attempt05_tooling_commit": tooling_commit,
        "attempt05_tooling_tree": tooling_tree,
        "preflight_path": preflight["preflight_path"],
        "preflight_file_sha256": preflight["preflight_file_sha256"],
        "preflight_payload_sha256": preflight["preflight_sha256"],
        "scientific_anchor_commit": SCIENTIFIC_ANCHOR_COMMIT,
        "scientific_anchor_tree": SCIENTIFIC_ANCHOR_TREE,
        "protocol_id": V4_PROTOCOL_ID,
        "protocol_sha256": V4_PROTOCOL_SHA256,
        "rectified_resource_schedule_sha256": SCHEDULE_FILE_SHA256,
        "scientific_schedule_sha256": schedule.schedule_sha256,
        "schedule_key_set_sha256": _sha256_json(_schedule_key_set(schedule)),
        "calibration_schedule_sha256": _sha256_json(_validate_calibration_schedule(calibration_schedule)),
        "cpu_input_closure": closure,
        "budgeted_input_storage": {
            "scope": storage["scope"],
            "logical_bytes": logical_bytes,
            "allocated_bytes": allocated_bytes,
            "input_closure_sha256": input_closure_sha256,
        },
        "runtime_root": str(context.runtime_root),
        "runtime_paths": {
            "run_root": str(context.run_root),
            "artifact_root": str(context.artifact_root),
            "gpu_ledger_path": str(context.gpu_ledger_path),
            "final_evidence_path": str(context.final_evidence_path),
        },
        "selected_gpu": dict(context.selected_gpu),
        "single_gpu_serial": {
            "max_concurrent_gpus": 1,
            "models_sequential": True,
            "units_sequential": True,
            "fallback_allowed": False,
            "device_switch_allowed": False,
            "retry_allowed": False,
        },
        "runner_execute_item_called": False,
        "adapter_exception_conversion_allowed": False,
        "partial_deletion_allowed": False,
        "split_assignment_bound": split_assignment is not None,
        "calibration_outputs_available": False,
        "q90_freeze_artifact_path": None,
        "q90_freeze_artifact_sha256": None,
    }
    payload["start_receipt_sha256"] = _sha256_json(payload)
    _write_immutable_json(receipt_path, payload)
    append_attempt05_ledger_event(
        ledger_path=context.gpu_ledger_path,
        event_type="START_RECEIPT",
        payload={
            "start_receipt_path": str(receipt_path),
            "start_receipt_sha256": _sha256_file(receipt_path),
        },
    )
    return payload


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise V4ExecutionError("V4_ATTEMPT05_JSON_UNREADABLE") from exc
    if not isinstance(value, dict):
        raise V4ExecutionError("V4_ATTEMPT05_JSON_OBJECT_REQUIRED")
    return value


def _load_start_receipt(context: Attempt05AuthorizedContext) -> dict[str, Any]:
    path = _start_receipt_path(context)
    if not path.is_file():
        raise V4ExecutionError("V4_ATTEMPT05_START_RECEIPT_REQUIRED")
    payload = _load_json(path)
    digest = payload.get("start_receipt_sha256")
    unsigned = {key: value for key, value in payload.items() if key != "start_receipt_sha256"}
    if (
        payload.get("schema_version") != START_RECEIPT_SCHEMA
        or payload.get("attempt_id") != ATTEMPT_ID
        or payload.get("attempt04_authorization_sha256") != context.authorization_sha256
        or not _is_sha(digest)
        or _sha256_json(unsigned) != digest
    ):
        raise V4ExecutionError("V4_ATTEMPT05_START_RECEIPT_TAMPER")
    return payload



def _q90_freeze_path(context: Attempt05AuthorizedContext) -> Path:
    return context.artifact_root / "v4-attempt05-q90-freeze.json"


def _validate_q90_rows(
    q90_calibrations: Sequence[Mapping[str, object]],
) -> tuple[dict[str, object], ...]:
    rows = tuple(q90_calibrations)
    if len(rows) != len(SCIENTIFIC_MODELS) or any(not isinstance(row, Mapping) for row in rows):
        raise V4ExecutionError("V4_ATTEMPT05_Q90_FREEZE_SCHEMA_REQUIRED")
    normalized: list[dict[str, object]] = []
    seen: set[str] = set()
    for row in rows:
        model_id = row.get("model_id")
        threshold = row.get("alarm_threshold")
        if (
            model_id not in SCIENTIFIC_MODELS
            or model_id in seen
            or row.get("state_id") != "L3"
            or row.get("scene_count") != 20
            or row.get("quantile_probability") != 0.90
            or row.get("quantile_method") != "linear"
            or isinstance(threshold, bool)
            or not isinstance(threshold, (int, float))
            or not math.isfinite(float(threshold))
            or not _is_sha(row.get("calibration_identifier"))
        ):
            raise V4ExecutionError("V4_ATTEMPT05_Q90_FREEZE_SCHEMA_REQUIRED")
        seen.add(str(model_id))
        normalized.append(
            {
                "model_id": str(model_id),
                "state_id": "L3",
                "scene_count": 20,
                "quantile_probability": 0.90,
                "quantile_method": "linear",
                "alarm_threshold": float(threshold),
                "calibration_identifier": str(row["calibration_identifier"]),
            }
        )
    if tuple(row["model_id"] for row in sorted(normalized, key=lambda item: SCIENTIFIC_MODELS.index(str(item["model_id"])))) != SCIENTIFIC_MODELS:
        raise V4ExecutionError("V4_ATTEMPT05_Q90_FREEZE_SCHEMA_REQUIRED")
    return tuple(sorted(normalized, key=lambda item: SCIENTIFIC_MODELS.index(str(item["model_id"]))))


def create_attempt05_q90_freeze_artifact(
    *,
    authorization_path: Path,
    calibration_schedule_sha256: str,
    q90_calibrations: Sequence[Mapping[str, object]],
    attempt05_tooling_commit: str,
    attempt05_tooling_tree: str,
) -> dict[str, object]:
    if not _is_sha(attempt05_tooling_commit, 40) or not _is_sha(attempt05_tooling_tree, 40):
        raise V4ExecutionError("V4_ATTEMPT05_TOOLING_REVISION_REQUIRED")
    if not _is_sha(calibration_schedule_sha256):
        raise V4ExecutionError("V4_ATTEMPT05_CALIBRATION_SCHEDULE_REQUIRED")
    context = load_attempt05_authorized_context(authorization_path=authorization_path)
    start = _load_start_receipt(context)
    if start.get("calibration_schedule_sha256") != calibration_schedule_sha256:
        raise V4ExecutionError("V4_ATTEMPT05_Q90_FREEZE_SCHEDULE_MISMATCH")
    path = _q90_freeze_path(context)
    if path.with_name(path.name + ".partial").exists():
        raise V4ExecutionError("V4_ATTEMPT05_Q90_FREEZE_PARTIAL_BLOCKED")
    expected_rows = list(_validate_q90_rows(q90_calibrations))
    if path.exists():
        existing = validate_attempt05_q90_freeze_artifact(path, authorization_path=authorization_path)
        if (
            existing.get("calibration_schedule_sha256") != calibration_schedule_sha256
            or existing.get("attempt05_tooling_commit") != attempt05_tooling_commit
            or existing.get("attempt05_tooling_tree") != attempt05_tooling_tree
            or existing.get("q90_calibrations") != expected_rows
        ):
            raise V4ExecutionError("V4_ATTEMPT05_Q90_FREEZE_MISMATCH")
        return existing
    payload: dict[str, object] = {
        "schema_version": Q90_FREEZE_SCHEMA,
        "attempt_id": ATTEMPT_ID,
        "attempt04_authorization_sha256": context.authorization_sha256,
        "start_receipt_sha256": start["start_receipt_sha256"],
        "calibration_schedule_sha256": calibration_schedule_sha256,
        "attempt05_tooling_commit": attempt05_tooling_commit,
        "attempt05_tooling_tree": attempt05_tooling_tree,
        "q90_calibrations": expected_rows,
    }
    payload["q90_freeze_artifact_sha256"] = _sha256_json(payload)
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".partial")
    try:
        with partial.open("xb") as handle:
            handle.write(_canonical_json_bytes(payload))
            handle.flush()
            os.fsync(handle.fileno())
        partial.replace(path)
    finally:
        try:
            partial.unlink()
        except FileNotFoundError:
            pass
    payload["q90_freeze_artifact_path"] = str(path)
    append_attempt05_ledger_event(
        ledger_path=context.gpu_ledger_path,
        event_type="Q90_FREEZE",
        payload={
            "q90_freeze_artifact_path": str(path),
            "q90_freeze_artifact_sha256": _sha256_file(path),
        },
    )
    return payload


def validate_attempt05_q90_freeze_artifact(
    path: str | Path,
    *,
    authorization_path: Path,
) -> dict[str, object]:
    context = load_attempt05_authorized_context(authorization_path=authorization_path)
    freeze_path = Path(path)
    if freeze_path.resolve() != _q90_freeze_path(context).resolve():
        raise V4ExecutionError("V4_ATTEMPT05_Q90_FREEZE_PATH_MISMATCH")
    payload = _load_json(freeze_path)
    digest = payload.get("q90_freeze_artifact_sha256")
    unsigned = {key: value for key, value in payload.items() if key != "q90_freeze_artifact_sha256"}
    start = _load_start_receipt(context)
    if (
        payload.get("schema_version") != Q90_FREEZE_SCHEMA
        or payload.get("attempt_id") != ATTEMPT_ID
        or payload.get("attempt04_authorization_sha256") != context.authorization_sha256
        or payload.get("start_receipt_sha256") != start.get("start_receipt_sha256")
        or payload.get("calibration_schedule_sha256") != start.get("calibration_schedule_sha256")
        or not _is_sha(payload.get("attempt05_tooling_commit"), 40)
        or not _is_sha(payload.get("attempt05_tooling_tree"), 40)
        or not _is_sha(digest)
        or _sha256_json(unsigned) != digest
    ):
        raise V4ExecutionError("V4_ATTEMPT05_Q90_FREEZE_TAMPER")
    payload["q90_calibrations"] = list(_validate_q90_rows(payload.get("q90_calibrations", ())))
    result = dict(payload)
    result["q90_freeze_artifact_path"] = str(freeze_path)
    return result

def _ledger_event_hash(row: Mapping[str, object]) -> str:
    unsigned = {key: value for key, value in row.items() if key != "event_sha256"}
    return _sha256_json(unsigned)


def _read_ledger_rows(ledger_path: Path) -> list[dict[str, Any]]:
    if not ledger_path.exists():
        return []
    rows = []
    previous = "0" * 64
    try:
        lines = ledger_path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise V4ExecutionError("V4_ATTEMPT05_LEDGER_UNREADABLE") from exc
    for index, line in enumerate(lines):
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise V4ExecutionError("V4_ATTEMPT05_LEDGER_CHAIN_TAMPER") from exc
        if not isinstance(row, dict):
            raise V4ExecutionError("V4_ATTEMPT05_LEDGER_CHAIN_TAMPER")
        if (
            row.get("schema_version") != LEDGER_SCHEMA
            or row.get("sequence_index") != index
            or row.get("previous_event_sha256") != previous
            or row.get("event_sha256") != _ledger_event_hash(row)
        ):
            raise V4ExecutionError("V4_ATTEMPT05_LEDGER_CHAIN_TAMPER")
        previous = str(row["event_sha256"])
        rows.append(row)
    return rows


@dataclass(frozen=True, slots=True)
class Attempt05LedgerResumeTotals:
    gpu_inference_seconds: float = 0.0
    wall_runtime_seconds: float = 0.0
    logical_bytes: int = 0
    allocated_bytes: int = 0
    calibration_units_completed: int = 0
    scientific_units_completed: int = 0
    completed_units: int = 0
    invalid_units: int = 0
    failed_units: int = 0
    peak_memory_mb: float = 0.0
    run_started: bool = False
    finalized: bool = False
    calibration_unit_keys: frozenset[tuple[str, int, str]] = frozenset()
    scientific_unit_keys: frozenset[tuple[str, int, str]] = frozenset()
    projection_unit_keys: frozenset[tuple[str, int, str]] = frozenset()


def _payload_float(payload: Mapping[str, object], *names: str) -> float | None:
    for name in names:
        value = payload.get(name)
        if isinstance(value, bool):
            raise V4ExecutionError("V4_ATTEMPT05_LEDGER_RESOURCE_INVALID")
        if isinstance(value, (int, float)):
            number = float(value)
            if not math.isfinite(number) or number < 0:
                raise V4ExecutionError("V4_ATTEMPT05_LEDGER_RESOURCE_INVALID")
            return number
    return None


def _payload_int(payload: Mapping[str, object], *names: str) -> int | None:
    value = _payload_float(payload, *names)
    if value is None:
        return None
    if not float(value).is_integer():
        raise V4ExecutionError("V4_ATTEMPT05_LEDGER_RESOURCE_INVALID")
    return int(value)


def rehydrate_attempt05_ledger_totals(ledger_path: Path) -> Attempt05LedgerResumeTotals:
    rows = _read_ledger_rows(ledger_path)
    gpu = 0.0
    wall = 0.0
    logical = 0
    allocated = 0
    calibration_completed = 0
    scientific_completed = 0
    invalid = 0
    failed = 0
    peak = 0.0
    run_started = False
    finalized = False
    seen_units: set[tuple[str, str, int, str]] = set()
    calibration_unit_keys: set[tuple[str, int, str]] = set()
    scientific_unit_keys: set[tuple[str, int, str]] = set()
    projection_unit_keys: set[tuple[str, int, str]] = set()
    for row in rows:
        payload = row.get("payload")
        if not isinstance(payload, Mapping):
            raise V4ExecutionError("V4_ATTEMPT05_LEDGER_CHAIN_TAMPER")
        event_type = row.get("event_type")
        if event_type == "MVE_RUN_STARTED":
            if run_started:
                raise V4ExecutionError("V4_ATTEMPT05_LEDGER_DUPLICATE_PHASE_EVENT")
            run_started = True
        if event_type == "MVE_FINALIZED":
            if finalized:
                raise V4ExecutionError("V4_ATTEMPT05_LEDGER_DUPLICATE_PHASE_EVENT")
            finalized = True
        next_gpu = _payload_float(payload, "gpu_inference_seconds_total")
        next_wall = _payload_float(payload, "wall_runtime_seconds_total")
        next_logical = _payload_int(payload, "logical_bytes_total", "storage_bytes_total")
        next_allocated = _payload_int(payload, "allocated_bytes_total")
        if next_gpu is not None:
            if next_gpu < gpu:
                raise V4ExecutionError("V4_ATTEMPT05_LEDGER_RESOURCE_NON_MONOTONIC")
            gpu = next_gpu
        if next_wall is not None:
            if next_wall < wall:
                raise V4ExecutionError("V4_ATTEMPT05_LEDGER_RESOURCE_NON_MONOTONIC")
            wall = next_wall
        if next_logical is not None:
            if next_logical < logical:
                raise V4ExecutionError("V4_ATTEMPT05_LEDGER_RESOURCE_NON_MONOTONIC")
            logical = next_logical
        if next_allocated is not None:
            if next_allocated < allocated:
                raise V4ExecutionError("V4_ATTEMPT05_LEDGER_RESOURCE_NON_MONOTONIC")
            allocated = next_allocated
        peak = max(peak, _payload_float(payload, "peak_memory_mb_total", "peak_memory_mb") or 0.0)
        if event_type in {"CALIBRATION_UNIT_COMPLETE", "SCIENTIFIC_UNIT_COMPLETE"}:
            model_id = payload.get("model_id")
            scene_id = payload.get("scene_id")
            state_id = payload.get("state_id")
            if not isinstance(model_id, str) or type(scene_id) is not int or not isinstance(state_id, str):
                raise V4ExecutionError("V4_ATTEMPT05_LEDGER_UNIT_IDENTITY_INVALID")
            key = (str(event_type), model_id, scene_id, state_id)
            if key in seen_units:
                raise V4ExecutionError("V4_ATTEMPT05_LEDGER_DUPLICATE_UNIT")
            seen_units.add(key)
            if event_type == "CALIBRATION_UNIT_COMPLETE":
                calibration_completed += 1
                calibration_unit_keys.add((model_id, scene_id, state_id))
            else:
                scientific_completed += 1
                scientific_unit_keys.add((model_id, scene_id, state_id))
        if event_type == "CANONICAL_RECORD_PROJECTION":
            model_id = payload.get("model_id")
            scene_id = payload.get("scene_id")
            state_id = payload.get("state_id")
            if (
                not isinstance(model_id, str)
                or type(scene_id) is not int
                or not isinstance(state_id, str)
            ):
                raise V4ExecutionError("V4_ATTEMPT05_LEDGER_UNIT_IDENTITY_INVALID")
            key = (str(event_type), model_id, scene_id, state_id)
            if key in seen_units:
                raise V4ExecutionError("V4_ATTEMPT05_LEDGER_DUPLICATE_UNIT")
            seen_units.add(key)
            projection_unit_keys.add((model_id, scene_id, state_id))
        status = str(payload.get("status", ""))
        if status == "INVALID_FAILURE_RECORDED" or status == "RESUMED_INVALID_FAILURE_RECORDED":
            invalid += 1
        if status.endswith("FAILED") or event_type == "UNIT_FAILED":
            failed += 1
    completed = calibration_completed + scientific_completed
    if completed > 440:
        raise V4ExecutionError("V4_ATTEMPT05_LEDGER_COMPLETED_COUNT_INVALID")
    return Attempt05LedgerResumeTotals(
        gpu_inference_seconds=gpu,
        wall_runtime_seconds=wall,
        logical_bytes=logical,
        allocated_bytes=allocated,
        calibration_units_completed=calibration_completed,
        scientific_units_completed=scientific_completed,
        completed_units=completed,
        invalid_units=invalid,
        failed_units=failed,
        peak_memory_mb=peak,
        run_started=run_started,
        finalized=finalized,
        calibration_unit_keys=frozenset(calibration_unit_keys),
        scientific_unit_keys=frozenset(scientific_unit_keys),
        projection_unit_keys=frozenset(projection_unit_keys),
    )


def append_attempt05_ledger_event(
    *,
    ledger_path: Path,
    event_type: str,
    payload: Mapping[str, object],
) -> dict[str, object]:
    if not isinstance(event_type, str) or not event_type:
        raise V4ExecutionError("V4_ATTEMPT05_LEDGER_EVENT_INVALID")
    rows = _read_ledger_rows(ledger_path)
    previous = rows[-1]["event_sha256"] if rows else "0" * 64
    payload_dict = dict(payload)
    payload_dict.setdefault("retry_count", 0)
    row: dict[str, object] = {
        "schema_version": LEDGER_SCHEMA,
        "sequence_index": len(rows),
        "previous_event_sha256": previous,
        "event_type": event_type,
        "payload": payload_dict,
    }
    row["event_sha256"] = _ledger_event_hash(row)
    ledger_path.parent.mkdir(parents=True, exist_ok=True)
    with ledger_path.open("ab") as handle:
        handle.write(_canonical_json_bytes(row))
        handle.flush()
        os.fsync(handle.fileno())
    return row


def _block_partial_outputs(root: Path) -> None:
    if not root.exists():
        return
    for path in root.rglob("*"):
        if any(part.endswith(".partial") for part in path.relative_to(root).parts):
            raise V4ExecutionError("V4_ATTEMPT05_RESUME_PARTIAL_BLOCKED")


def authorize_attempt05_next_dispatch(
    *,
    authorization_path: Path,
    schedule: ScientificSchedule,
    resume: bool = False,
) -> V4ControllerDecision:
    if type(resume) is not bool:
        raise V4ExecutionError("V4_ATTEMPT05_RESUME_FLAG_INVALID")
    context = load_attempt05_authorized_context(authorization_path=authorization_path)
    _load_start_receipt(context)
    _block_partial_outputs(context.run_root)
    decision = first_incomplete_unit(schedule, context.run_root)
    if decision.status == "GPU_SELECTION_REQUIRED":
        return V4ControllerDecision(
            status="PASS",
            reason_code="V4_ATTEMPT05_NEXT_UNIT_AUTHORIZED",
            unit=decision.unit,
            record_path=decision.record_path,
        )
    if decision.status == "MVE_BLOCKED_FINALIZE_REQUIRED":
        return V4ControllerDecision(
            status="PASS",
            reason_code="CPU_ONLY_FINALIZE_READY",
            unit=decision.unit,
            record_path=decision.record_path,
        )
    if decision.status == "V4_DISPATCH_AUTHORIZED":
        raise V4ExecutionError("V4_ATTEMPT05_INTERNAL_DISPATCH_SCOPE_ERROR")
    return decision


def evaluate_attempt05_resource_gate(
    *,
    gpu_inference_seconds: float,
    wall_runtime_seconds: float,
    new_logical_bytes: int,
    new_allocated_bytes: int,
) -> V4ResourceDecision:
    values = (gpu_inference_seconds, wall_runtime_seconds)
    if (
        any(isinstance(value, bool) or not isinstance(value, (int, float)) for value in values)
        or any(not math.isfinite(float(value)) or value < 0 for value in values)
        or isinstance(new_logical_bytes, bool)
        or isinstance(new_allocated_bytes, bool)
        or not isinstance(new_logical_bytes, int)
        or not isinstance(new_allocated_bytes, int)
        or new_logical_bytes < 0
        or new_allocated_bytes < 0
    ):
        raise V4ExecutionError("V4_ATTEMPT05_RESOURCE_LEDGER_INVALID")
    if gpu_inference_seconds >= GPU_CATASTROPHE_SECONDS:
        return V4ResourceDecision(
            "MVE_BLOCKED_RESOURCE_CATASTROPHE", "V4_CATASTROPHE_GPUH_FUSE"
        )
    if new_logical_bytes >= BYTE_CATASTROPHE or new_allocated_bytes >= BYTE_CATASTROPHE:
        return V4ResourceDecision(
            "MVE_BLOCKED_RESOURCE_CATASTROPHE", "V4_CATASTROPHE_STORAGE_FUSE"
        )
    if gpu_inference_seconds >= GPU_TARGET_SECONDS:
        return V4ResourceDecision(
            "MVE_BLOCKED_REAUTHORIZATION_REQUIRED",
            "V4_REAUTHORIZE_GPUH_TARGET_REACHED",
        )
    if new_logical_bytes >= BYTE_TARGET:
        return V4ResourceDecision(
            "MVE_BLOCKED_REAUTHORIZATION_REQUIRED",
            "V4_REAUTHORIZE_LOGICAL_BYTES_TARGET_REACHED",
        )
    if new_allocated_bytes >= BYTE_TARGET:
        return V4ResourceDecision(
            "MVE_BLOCKED_REAUTHORIZATION_REQUIRED",
            "V4_REAUTHORIZE_ALLOCATED_BYTES_TARGET_REACHED",
        )
    return V4ResourceDecision("PASS", "V4_ATTEMPT05_RESOURCE_WITHIN_AUTHORIZATION")


def finalize_attempt05_scientific_bundle(
    *,
    authorization_path: Path,
    record_paths: Sequence[Path],
    scientific_schedule: ScientificSchedule,
    model_independent_states: Sequence[object],
    split_assignment: object,
    native_warning_calibrations: Sequence[object],
) -> dict[str, object]:
    context = load_attempt05_authorized_context(authorization_path=authorization_path)
    paths = tuple(record_paths)
    if len(paths) != 400:
        raise V4ExecutionError("V4_RECORD_COUNT_NOT_400")
    expected = tuple(
        canonical_record_path(context.run_root, unit)
        for unit in validate_scientific_schedule(scientific_schedule).units
    )
    if len(expected) != 400:
        raise V4ExecutionError("V4_SCHEDULE_NOT_EXACT_400")
    if all(path.exists() for path in paths) and tuple(path.resolve() for path in paths) != tuple(
        path.resolve() for path in expected
    ):
        raise V4ExecutionError("V4_RECORD_CANONICAL_PATH_MISMATCH")
    return finalize_v4_scientific_bundle(
        source_root=Path(__file__).resolve().parents[1],
        output_dir=context.final_evidence_path.with_suffix(""),
        record_paths=paths,
        scientific_schedule=scientific_schedule,
        model_independent_states=model_independent_states,
        split_assignment=split_assignment,
        native_warning_calibrations=native_warning_calibrations,
    )


def _rq_decomposition(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    reversed_matrix = np.flipud(np.fliplr(matrix))
    q_rev, r_rev = np.linalg.qr(reversed_matrix.T)
    r = np.flipud(np.fliplr(r_rev.T))
    q = np.flipud(np.fliplr(q_rev.T))
    signs = np.sign(np.diag(r))
    signs[signs == 0.0] = 1.0
    transform = np.diag(signs)
    k = r @ transform
    rotation = transform @ q
    if np.linalg.det(rotation) < 0:
        k[:, 2] *= -1.0
        rotation[2, :] *= -1.0
    if k[2, 2] == 0:
        raise V4ExecutionError("V4_ATTEMPT05_DTU_PROJECTION_DEGENERATE")
    k = k / k[2, 2]
    return k, rotation


def decompose_dtu_projection_to_camera_to_world(
    projection_3x4: Sequence[Sequence[float]],
    *,
    max_reprojection_abs_error: float = 1e-7,
) -> DTUProjectionDecomposition:
    try:
        projection = np.asarray(projection_3x4, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise V4ExecutionError("V4_ATTEMPT05_DTU_PROJECTION_INVALID") from exc
    if projection.shape != (3, 4) or not np.all(np.isfinite(projection)):
        raise V4ExecutionError("V4_ATTEMPT05_DTU_PROJECTION_INVALID")
    left = projection[:, :3]
    if abs(float(np.linalg.det(left))) < 1e-12:
        raise V4ExecutionError("V4_ATTEMPT05_DTU_PROJECTION_DEGENERATE")
    k, rotation = _rq_decomposition(left)
    if not np.allclose(rotation @ rotation.T, np.eye(3), rtol=0.0, atol=1e-7):
        raise V4ExecutionError("V4_ATTEMPT05_DTU_ROTATION_INVALID")
    if not math.isclose(float(np.linalg.det(rotation)), 1.0, rel_tol=0.0, abs_tol=1e-7):
        raise V4ExecutionError("V4_ATTEMPT05_DTU_ROTATION_INVALID")
    center = -np.linalg.solve(left, projection[:, 3])
    translation = -rotation @ center
    reconstructed = k @ np.column_stack([rotation, translation])
    scale = float(np.sum(projection * reconstructed) / np.sum(reconstructed * reconstructed))
    reconstructed *= scale
    error = float(np.max(np.abs(projection - reconstructed)))
    if error > max_reprojection_abs_error:
        raise V4ExecutionError("V4_ATTEMPT05_DTU_REPROJECTION_MISMATCH")
    c2w = np.eye(4, dtype=np.float64)
    c2w[:3, :3] = rotation.T
    c2w[:3, 3] = center
    return DTUProjectionDecomposition(
        projection=tuple(tuple(float(x) for x in row) for row in projection),
        intrinsic_k=tuple(tuple(float(x) for x in row) for row in k),
        world_to_camera_rotation=tuple(
            tuple(float(x) for x in row) for row in rotation
        ),
        camera_center_world=tuple(float(x) for x in center),
        camera_to_world=tuple(tuple(float(x) for x in row) for row in c2w),
        max_reprojection_abs_error=error,
    )


def decompose_ordered_dtu_projections(
    projections_by_view_id: Mapping[int, Sequence[Sequence[float]]],
    *,
    ordered_view_ids: Sequence[int],
) -> tuple[DTUProjectionDecomposition, ...]:
    ordered = tuple(ordered_view_ids)
    if len(ordered) != 8 or len(set(ordered)) != 8:
        raise V4ExecutionError("V4_ATTEMPT05_DTU_VIEW_ORDER_INVALID")
    if set(projections_by_view_id) != set(ordered):
        raise V4ExecutionError("V4_ATTEMPT05_DTU_VIEW_ORDER_INVALID")
    return tuple(
        decompose_dtu_projection_to_camera_to_world(projections_by_view_id[view_id])
        for view_id in ordered
    )











