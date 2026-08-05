"""Attempt-05 production inference bridge for GeoReliab v4.

This module is deliberately below the governance/CLI layer. It executes one
scientific unit at an already-authorized atomic boundary, calls the supplied
adapter directly, and converts validated PredictionArtifact v1.1 payloads into
strict v4 TaskAuditRecords.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse

import numpy as np

from .adapters import RenderedView
from .artifact_storage import write_deterministic_npz
from .audit import audit_prediction_arrays, model_risk_from_confidence
from .contracts import (
    AuditRecord,
    ContractError,
    PredictionArtifact,
    RunManifest,
    RunMode,
    SampleKey,
    validate_artifact_bundle,
    write_json_artifact,
)
from .v4_counterfactuals import FOG_STATES, SCIENTIFIC_STATES, ScientificExecutionUnit
from .v4_attempt05_recovery import (
    FailureEnvelope,
    atomic_write_bytes,
    failure_envelope_for,
    rename_noreplace,
)
from .v4_execution import V4ExecutionError
from .v4_metrics import (
    NativeWarningCalibration,
    compute_point_task_metrics,
    compute_relative_pose_metrics,
    native_warning_score,
)
from .v4_records import (
    Task3ContractError,
    TaskAuditRecord,
    build_task_audit_record,
    read_task_audit_record,
    write_task_audit_record,
)


class Attempt05RuntimeError(V4ExecutionError):
    """Raised when Attempt-05 runtime execution must fail closed.

    The legacy string reason remains the exception message for compatibility,
    while failure_envelope retains stage, original exception and traceback.
    """

    def __init__(
        self,
        reason_code: str,
        *,
        failure_envelope: FailureEnvelope | None = None,
    ) -> None:
        super().__init__(reason_code)
        self.reason_code = reason_code
        self.failure_envelope = failure_envelope

    @classmethod
    def from_exception(
        cls,
        exc: BaseException,
        *,
        stage: str,
        unit: ScientificExecutionUnit | None = None,
        unit_key: tuple[str, int, str] | None = None,
        reason_code: str,
        worker_pid: int | None = None,
        heartbeat_age_seconds: float | None = None,
    ) -> "Attempt05RuntimeError":
        resolved_unit_key = unit_key
        if resolved_unit_key is None and unit is not None:
            resolved_unit_key = (unit.model_id, unit.scene_id, unit.state_id)
        envelope = failure_envelope_for(
            exc,
            attempt_id="attempt-05",
            stage=stage,
            unit_key=resolved_unit_key,
            reason_code=reason_code,
            worker_pid=worker_pid,
            heartbeat_age_seconds=heartbeat_age_seconds,
        )
        return cls(envelope.reason_code, failure_envelope=envelope)


CALIBRATION_WARNING_EVIDENCE_SCHEMA = "georeliab-v4-attempt05-calibration-warning-evidence-1.0"


@dataclass(frozen=True, slots=True)
class DTUProjectionDecomposition:
    view_id: int
    projection: tuple[tuple[float, float, float, float], ...]
    intrinsic_k: tuple[tuple[float, float, float], ...]
    world_to_camera_rotation: tuple[tuple[float, float, float], ...]
    camera_center_world: tuple[float, float, float]
    camera_to_world: tuple[tuple[float, float, float, float], ...]
    max_reprojection_abs_error: float
    det_r: float


@dataclass(frozen=True, slots=True)
class Attempt05UnitResult:
    status: str
    record_path: Path
    prediction: PredictionArtifact | None
    record: TaskAuditRecord


@dataclass(frozen=True, slots=True)
class Attempt05CalibrationResult:
    status: str
    evidence_path: Path
    prediction: PredictionArtifact | None
    warning_score: float
    evidence_sha256: str


@dataclass(frozen=True, slots=True)
class _RecordBuildResult:
    record: TaskAuditRecord
    dense_payload: Mapping[str, np.ndarray]
    gt_points_payload: Mapping[str, np.ndarray]
    audit_record: AuditRecord


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_json_bytes(payload: Mapping[str, Any]) -> bytes:
    return (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")


def _json_payload_sha256(payload: Mapping[str, Any]) -> str:
    return hashlib.sha256(_canonical_json_bytes(payload)).hexdigest()


def _write_json_payload(path: Path, payload: Mapping[str, Any]) -> None:
    atomic_write_bytes(path, _canonical_json_bytes(payload))


def _file_uri_path(uri: str, label: str) -> Path:
    parsed = urlparse(uri)
    if parsed.scheme != "file" or parsed.netloc not in ("", "localhost"):
        raise Attempt05RuntimeError(f"V4_ATTEMPT05_{label.upper()}_URI_NOT_FILE")
    path = Path(unquote(parsed.path))
    if os.name == "nt" and path.drive == "" and len(path.parts) > 1:
        raw = unquote(parsed.path)
        if raw.startswith("/") and len(raw) > 3 and raw[2] == ":":
            path = Path(raw[1:])
    if not path.is_file():
        raise Attempt05RuntimeError(f"V4_ATTEMPT05_{label.upper()}_MISSING")
    return path


def _load_npz(uri: str, digest: str | None, label: str) -> Mapping[str, np.ndarray]:
    path = _file_uri_path(uri, label)
    if digest and _sha256_file(path) != digest:
        raise Attempt05RuntimeError(f"V4_ATTEMPT05_{label.upper()}_DIGEST_MISMATCH")
    try:
        with np.load(path, allow_pickle=False) as payload:
            return {key: payload[key] for key in payload.files}
    except Exception as exc:
        raise Attempt05RuntimeError(f"V4_ATTEMPT05_{label.upper()}_NPZ_INVALID") from exc


def _validate_prediction(
    prediction: object,
    *,
    manifest: RunManifest,
    sample_key: SampleKey,
) -> PredictionArtifact:
    if not isinstance(prediction, PredictionArtifact):
        raise Attempt05RuntimeError("V4_ATTEMPT05_ADAPTER_RETURNED_NON_PREDICTION")
    try:
        validated = PredictionArtifact.from_dict(prediction.to_dict())
    except (AttributeError, ContractError) as exc:
        raise Attempt05RuntimeError("V4_ATTEMPT05_PREDICTION_ARTIFACT_INVALID") from exc
    if (
        validated.schema_version != "1.1"
        or validated.run_id != manifest.run_id
        or validated.sample_key != str(sample_key)
    ):
        raise Attempt05RuntimeError("V4_ATTEMPT05_PREDICTION_ARTIFACT_INVALID")
    return validated


def build_cpu_input_closure(
    *,
    calibration_l3_units: Sequence[object],
    model_independent_states: Sequence[object],
    scientific_units: Sequence[ScientificExecutionUnit],
    rectified_bindings: Sequence[object],
) -> dict[str, int]:
    calibration = tuple(calibration_l3_units)
    states = tuple(model_independent_states)
    units = tuple(scientific_units)
    bindings = tuple(rectified_bindings)
    if len(calibration) != 40:
        raise Attempt05RuntimeError("V4_ATTEMPT05_CALIBRATION_L3_NOT_40")
    if len(states) != 200:
        raise Attempt05RuntimeError("V4_ATTEMPT05_MODEL_INDEPENDENT_STATES_NOT_200")
    if len(units) != 400:
        raise Attempt05RuntimeError("V4_ATTEMPT05_SCHEDULE_NOT_400")
    if len(bindings) != 960:
        raise Attempt05RuntimeError("V4_ATTEMPT05_RECTIFIED_BINDINGS_NOT_960")
    if any(not isinstance(unit, ScientificExecutionUnit) for unit in units):
        raise Attempt05RuntimeError("V4_ATTEMPT05_SCHEDULE_UNIT_INVALID")
    test_l3 = sum(unit.state_id == "L3" for unit in units)
    fog = sum(unit.state_id in FOG_STATES for unit in units)
    non_l3 = sum(unit.state_id != "L3" for unit in units)
    if test_l3 != 40 or fog != 120 or non_l3 != 360:
        raise Attempt05RuntimeError("V4_ATTEMPT05_SCHEDULE_POPULATION_INVALID")
    if any(unit.state_id not in SCIENTIFIC_STATES for unit in units):
        raise Attempt05RuntimeError("V4_ATTEMPT05_SCHEDULE_STATE_INVALID")
    return {
        "calibration_l3_units": len(calibration),
        "model_independent_states": len(states),
        "scientific_units": len(units),
        "schedule_units": len(units),
        "rectified_non_l3_members": len(bindings),
        "test_l3_units": test_l3,
        "non_l3_scientific_units": non_l3,
        "fog_bindings_to_l3": fog,
    }


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
    if np.linalg.det(rotation) < 0.0:
        k[:, 2] *= -1.0
        rotation[2, :] *= -1.0
    if abs(float(k[2, 2])) <= 1e-12:
        raise Attempt05RuntimeError("V4_ATTEMPT05_DTU_PROJECTION_DEGENERATE")
    return k / k[2, 2], rotation


def decompose_dtu_projection_to_camera_to_world(
    projection_3x4: Sequence[Sequence[float]],
    *,
    view_id: int,
    max_reprojection_abs_error: float = 1e-7,
) -> DTUProjectionDecomposition:
    try:
        projection = np.asarray(projection_3x4, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise Attempt05RuntimeError("V4_ATTEMPT05_DTU_PROJECTION_INVALID") from exc
    if (
        isinstance(view_id, bool)
        or not isinstance(view_id, int)
        or projection.shape != (3, 4)
        or not np.all(np.isfinite(projection))
    ):
        raise Attempt05RuntimeError("V4_ATTEMPT05_DTU_PROJECTION_INVALID")
    left = projection[:, :3]
    if abs(float(np.linalg.det(left))) <= 1e-12:
        raise Attempt05RuntimeError("V4_ATTEMPT05_DTU_PROJECTION_DEGENERATE")
    k, rotation = _rq_decomposition(left)
    det_r = float(np.linalg.det(rotation))
    if not np.allclose(rotation @ rotation.T, np.eye(3), rtol=0.0, atol=1e-7):
        raise Attempt05RuntimeError("V4_ATTEMPT05_DTU_ROTATION_INVALID")
    if not math.isclose(det_r, 1.0, rel_tol=0.0, abs_tol=1e-7):
        raise Attempt05RuntimeError("V4_ATTEMPT05_DTU_ROTATION_INVALID")
    center = -np.linalg.solve(left, projection[:, 3])
    translation = -rotation @ center
    reconstructed = k @ np.column_stack([rotation, translation])
    scale = float(np.sum(projection * reconstructed) / np.sum(reconstructed * reconstructed))
    reconstructed *= scale
    error = float(np.max(np.abs(projection - reconstructed)))
    if error > max_reprojection_abs_error:
        raise Attempt05RuntimeError("V4_ATTEMPT05_DTU_REPROJECTION_MISMATCH")
    c2w = np.eye(4, dtype=np.float64)
    c2w[:3, :3] = rotation.T
    c2w[:3, 3] = center
    return DTUProjectionDecomposition(
        view_id=view_id,
        projection=tuple(tuple(float(item) for item in row) for row in projection),
        intrinsic_k=tuple(tuple(float(item) for item in row) for row in k),
        world_to_camera_rotation=tuple(tuple(float(item) for item in row) for row in rotation),
        camera_center_world=tuple(float(item) for item in center),
        camera_to_world=tuple(tuple(float(item) for item in row) for row in c2w),
        max_reprojection_abs_error=error,
        det_r=det_r,
    )


def decompose_ordered_dtu_projections(
    projections_by_view_id: Mapping[int, Sequence[Sequence[float]]],
    *,
    ordered_view_ids: Sequence[int],
) -> tuple[DTUProjectionDecomposition, ...]:
    ordered = tuple(ordered_view_ids)
    if len(ordered) != 8 or len(set(ordered)) != 8 or set(projections_by_view_id) != set(ordered):
        raise Attempt05RuntimeError("V4_ATTEMPT05_DTU_VIEW_ORDER_INVALID")
    return tuple(
        decompose_dtu_projection_to_camera_to_world(
            projections_by_view_id[view_id],
            view_id=view_id,
        )
        for view_id in ordered
    )


def _record_path(output_dir: Path) -> Path:
    return output_dir / "task_audit_record.json"


def _partial_dir(output_dir: Path) -> Path:
    return output_dir.with_name(output_dir.name + ".partial")


def _promote_staging_dir(
    staging_dir: Path,
    output_dir: Path,
    *,
    stage: str,
    unit: ScientificExecutionUnit | None = None,
    unit_key: tuple[str, int, str] | None = None,
) -> None:
    """Promote a completed bundle without clobbering an existing identity."""

    try:
        rename_noreplace(staging_dir, output_dir)
    except Attempt05RuntimeError:
        raise
    except Exception as exc:
        key = unit_key
        if key is None and unit is not None:
            key = (unit.model_id, unit.scene_id, unit.state_id)
        raise Attempt05RuntimeError.from_exception(
            exc,
            stage=stage,
            unit=unit,
            unit_key=key,
            reason_code="V4_ATTEMPT05_ATOMIC_PROMOTION_FAILED",
        ) from exc


def _block_or_resume(output_dir: Path, *, resume: bool) -> TaskAuditRecord | None:
    record_path = _record_path(output_dir)
    if _partial_dir(output_dir).exists() or record_path.with_name(record_path.name + ".partial").exists():
        raise Attempt05RuntimeError("V4_ATTEMPT05_PARTIAL_EXISTS")
    if output_dir.exists():
        if not resume:
            raise Attempt05RuntimeError("V4_ATTEMPT05_RECORD_ALREADY_EXISTS")
        if not record_path.is_file():
            raise Attempt05RuntimeError("V4_ATTEMPT05_RESUME_INVALID_RECORD")
        try:
            record = read_task_audit_record(record_path)
        except Task3ContractError as exc:
            raise Attempt05RuntimeError("V4_ATTEMPT05_RESUME_INVALID_RECORD") from exc
        return record
    return None


def _native_risk_by_view(
    risk: np.ndarray,
    view_ids: np.ndarray,
    ordered_view_ids: Sequence[int],
) -> dict[int, np.ndarray]:
    result: dict[int, np.ndarray] = {}
    for view_id in ordered_view_ids:
        values = risk[view_ids == view_id]
        if len(values) == 0:
            raise Attempt05RuntimeError("V4_MVE_BLOCKED_NATIVE_WARNING_UNAVAILABLE")
        result[int(view_id)] = values
    return result


def _native_warning_from_prediction(
    prediction: PredictionArtifact,
    *,
    model_id: str,
    ordered_view_ids: Sequence[int],
    unit: ScientificExecutionUnit | None = None,
    unit_key: tuple[str, int, str] | None = None,
) -> float:
    try:
        geometry = _load_npz(
            prediction.geometry_prediction_uri,
            prediction.payload_digests.get("geometry_prediction_uri"),
            "geometry",
        )
        confidence = _load_npz(
            prediction.native_confidence_uri,
            prediction.payload_digests.get("native_confidence_uri"),
            "confidence",
        )
        raw_confidence = np.asarray(confidence["raw_confidence"], dtype=np.float64)
        view_ids = np.asarray(geometry["view_id"], dtype=np.int64)
        risk = model_risk_from_confidence(model_id, raw_confidence)
        score = native_warning_score(
            _native_risk_by_view(risk, view_ids, ordered_view_ids),
            ordered_view_ids=ordered_view_ids,
        )
    except Exception as exc:
        if isinstance(exc, Attempt05RuntimeError):
            raise
        raise Attempt05RuntimeError.from_exception(
            exc,
            stage="native_warning",
            unit=unit,
            unit_key=unit_key,
            reason_code="V4_MVE_BLOCKED_NATIVE_WARNING_UNAVAILABLE",
        ) from exc
    if not math.isfinite(float(score)):
        raise Attempt05RuntimeError("V4_MVE_BLOCKED_NATIVE_WARNING_UNAVAILABLE")
    return float(score)


def _prediction_with_uris(
    prediction: PredictionArtifact,
    *,
    geometry_path: Path,
    confidence_path: Path,
    valid_mask_path: Path,
    payload_digests: Mapping[str, str],
) -> PredictionArtifact:
    return PredictionArtifact(
        run_id=prediction.run_id,
        sample_key=prediction.sample_key,
        geometry_prediction_uri=geometry_path.resolve().as_uri(),
        native_confidence_uri=confidence_path.resolve().as_uri(),
        valid_mask_uri=valid_mask_path.resolve().as_uri(),
        hook_location=prediction.hook_location,
        runtime_seconds=prediction.runtime_seconds,
        peak_memory_mb=prediction.peak_memory_mb,
        invalid_prediction=prediction.invalid_prediction,
        payload_digests=dict(payload_digests),
        schema_version=prediction.schema_version,
    )


def _audit_with_final_uris(audit: AuditRecord, *, bundle_dir: Path) -> AuditRecord:
    metadata = dict(audit.metadata)
    metadata["dense_audit_uri"] = (bundle_dir / "dense_audit.npz").resolve().as_uri()
    metadata["gt_points_uri"] = (bundle_dir / "gt_points.npz").resolve().as_uri()
    return AuditRecord(
        run_id=audit.run_id,
        sample_key=audit.sample_key,
        gt_error=audit.gt_error,
        failure_label=audit.failure_label,
        selection_score=audit.selection_score,
        coverage=audit.coverage,
        accepted=audit.accepted,
        downstream_outcome=audit.downstream_outcome,
        invalid_prediction=audit.invalid_prediction,
        metadata=metadata,
    )


def _copy_prediction_payloads(
    prediction: PredictionArtifact,
    staging_dir: Path,
    *,
    unit: ScientificExecutionUnit | None = None,
    unit_key: tuple[str, int, str] | None = None,
) -> PredictionArtifact:
    """Copy prediction payloads into the transaction tree with an envelope."""

    try:
        sources = {
            "geometry_prediction_uri": _file_uri_path(prediction.geometry_prediction_uri, "geometry"),
            "native_confidence_uri": _file_uri_path(prediction.native_confidence_uri, "confidence"),
            "valid_mask_uri": _file_uri_path(prediction.valid_mask_uri, "valid_mask"),
        }
        targets = {
            "geometry_prediction_uri": staging_dir / "geometry_prediction.npz",
            "native_confidence_uri": staging_dir / "native_confidence.npz",
            "valid_mask_uri": staging_dir / "valid_mask.npz",
        }
        labels = {
            "geometry_prediction_uri": "geometry",
            "native_confidence_uri": "confidence",
            "valid_mask_uri": "valid_mask",
        }
        digests: dict[str, str] = {}
        for key, source in sources.items():
            expected = prediction.payload_digests.get(key, "")
            if expected and _sha256_file(source) != expected:
                raise Attempt05RuntimeError("V4_ATTEMPT05_PREDICTION_PAYLOAD_DIGEST_MISMATCH")
            arrays = _load_npz(source.resolve().as_uri(), expected, labels[key])
            write_deterministic_npz(targets[key], arrays)
            digests[key] = _sha256_file(targets[key])
        return _prediction_with_uris(
            prediction,
            geometry_path=targets["geometry_prediction_uri"],
            confidence_path=targets["native_confidence_uri"],
            valid_mask_path=targets["valid_mask_uri"],
            payload_digests=digests,
        )
    except Attempt05RuntimeError:
        raise
    except Exception as exc:
        raise Attempt05RuntimeError.from_exception(
            exc,
            stage="prediction_payload_copy",
            unit=unit,
            unit_key=unit_key,
            reason_code="V4_ATTEMPT05_PREDICTION_PAYLOAD_COPY_FAILED",
        ) from exc


def _write_npz(path: Path, payload: Mapping[str, np.ndarray]) -> str:
    write_deterministic_npz(path, payload)
    return _sha256_file(path)


def _prediction_record_impl(
    *,
    unit: ScientificExecutionUnit,
    manifest: RunManifest,
    prediction: PredictionArtifact,
    calibration: NativeWarningCalibration,
    ordered_view_ids: tuple[int, ...],
    gt_points: Any,
    gt_camera_c2w: Any,
    observability_mask: Any,
    gt_dtu_camera_c2w: Any,
    observability_bb: Any | None,
    observability_res: float | None,
    staging_dir: Path,
) -> _RecordBuildResult:
    gt_payload = {"gt_points": np.asarray(gt_points, dtype=np.float64)}
    if prediction.invalid_prediction:
        warning = 1e12
        try:
            record = build_task_audit_record(
                execution_unit=unit,
                calibration=calibration,
                ordered_view_ids=ordered_view_ids,
                valid=False,
                reason_code="INVALID_MODEL_OUTPUT",
                native_warning_score_value=warning,
            )
        except Task3ContractError as exc:
            raise Attempt05RuntimeError("V4_ATTEMPT05_TASK_RECORD_INVALID") from exc
        dense_payload = {
            "voxel_points": np.empty((0, 3), dtype=np.float64),
            "raw_confidence": np.empty((0,), dtype=np.float64),
            "risk": np.empty((0,), dtype=np.float64),
            "gt_error": np.empty((0,), dtype=np.float64),
            "failure_label": np.empty((0,), dtype=bool),
            "provenance_count": np.empty((0,), dtype=np.int64),
        }
        dense_path = staging_dir / "dense_audit.npz"
        gt_path = staging_dir / "gt_points.npz"
        dense_sha = _write_npz(dense_path, dense_payload)
        gt_sha = _write_npz(gt_path, gt_payload)
        audit_record = AuditRecord(
            run_id=manifest.run_id,
            sample_key=prediction.sample_key,
            gt_error=1e12,
            failure_label=True,
            selection_score=1e12,
            coverage=1.0,
            accepted=False,
            downstream_outcome=0.0,
            invalid_prediction=True,
            metadata={
                "dense_audit_uri": dense_path.resolve().as_uri(),
                "dense_audit_sha256": dense_sha,
                "gt_points_uri": gt_path.resolve().as_uri(),
                "gt_points_sha256": gt_sha,
            },
        )
        return _RecordBuildResult(record, dense_payload, gt_payload, audit_record)

    geometry = _load_npz(
        prediction.geometry_prediction_uri,
        prediction.payload_digests.get("geometry_prediction_uri"),
        "geometry",
    )
    confidence = _load_npz(
        prediction.native_confidence_uri,
        prediction.payload_digests.get("native_confidence_uri"),
        "confidence",
    )
    valid = _load_npz(
        prediction.valid_mask_uri,
        prediction.payload_digests.get("valid_mask_uri"),
        "valid_mask",
    )
    try:
        raw_confidence = np.asarray(confidence["raw_confidence"], dtype=np.float64)
        risk = model_risk_from_confidence(unit.model_id, raw_confidence)
        warning = _native_warning_from_prediction(
            prediction,
            model_id=unit.model_id,
            ordered_view_ids=ordered_view_ids,
            unit=unit,
        )
    except Exception as exc:
        raise Attempt05RuntimeError("V4_MVE_BLOCKED_NATIVE_WARNING_UNAVAILABLE") from exc

    try:
        audit = audit_prediction_arrays(
            points_world=geometry["points_world"],
            pred_camera_centers=np.asarray(geometry["camera_c2w"], dtype=np.float64)[:, :3, 3],
            gt_camera_centers=np.asarray(gt_camera_c2w, dtype=np.float64)[:, :3, 3],
            raw_confidence=raw_confidence,
            risk=risk,
            valid_mask=valid["valid_mask"],
            gt_points=gt_points,
            observability_mask=observability_mask,
            observability_bb=observability_bb,
            observability_res=observability_res,
            voxel_size_mm=0.2,
        )
    except Exception as exc:
        if isinstance(exc, Attempt05RuntimeError):
            raise
        raise Attempt05RuntimeError.from_exception(
            exc,
            stage="gt_array_audit",
            unit=unit,
            reason_code="V4_ATTEMPT05_GT_ARRAY_AUDIT_FAILED",
        ) from exc

    try:
        point_metrics = compute_point_task_metrics(audit.voxel_points, gt_points, audit.risk)
    except Exception as exc:
        if isinstance(exc, Attempt05RuntimeError):
            raise
        raise Attempt05RuntimeError.from_exception(
            exc,
            stage="point_metrics",
            unit=unit,
            reason_code="V4_ATTEMPT05_POINT_METRIC_FAILED",
        ) from exc

    try:
        pose_metrics = compute_relative_pose_metrics(
            geometry["camera_c2w"],
            gt_dtu_camera_c2w,
            ordered_view_ids=ordered_view_ids,
        )
    except Exception as exc:
        if isinstance(exc, Attempt05RuntimeError):
            raise
        raise Attempt05RuntimeError.from_exception(
            exc,
            stage="pose_metrics",
            unit=unit,
            reason_code="V4_ATTEMPT05_POSE_METRIC_FAILED",
        ) from exc

    try:
        record = build_task_audit_record(
            execution_unit=unit,
            calibration=calibration,
            ordered_view_ids=ordered_view_ids,
            valid=True,
            reason_code="VALID",
            native_warning_score_value=warning,
            point_metrics=point_metrics,
            pose_metrics=pose_metrics,
        )
    except Exception as exc:
        if isinstance(exc, Attempt05RuntimeError):
            raise
        raise Attempt05RuntimeError.from_exception(
            exc,
            stage="task_record_build",
            unit=unit,
            reason_code="V4_ATTEMPT05_TASK_RECORD_BUILD_FAILED",
        ) from exc

    dense_payload = {
        "voxel_points": audit.voxel_points,
        "raw_confidence": audit.raw_confidence,
        "risk": audit.risk,
        "gt_error": audit.gt_error,
        "failure_label": audit.failure_label,
        "provenance_count": audit.provenance_count,
    }
    dense_path = staging_dir / "dense_audit.npz"
    gt_path = staging_dir / "gt_points.npz"
    dense_sha = _write_npz(dense_path, dense_payload)
    gt_sha = _write_npz(gt_path, gt_payload)
    audit_record = AuditRecord(
        run_id=manifest.run_id,
        sample_key=prediction.sample_key,
        gt_error=float(np.median(audit.gt_error)),
        failure_label=bool(np.any(audit.failure_label)),
        selection_score=float(np.median(audit.risk)),
        coverage=1.0,
        accepted=True,
        downstream_outcome=audit.summary["fscore_2mm"],
        invalid_prediction=False,
        metadata={
            "dense_audit_uri": dense_path.resolve().as_uri(),
            "dense_audit_sha256": dense_sha,
            "gt_points_uri": gt_path.resolve().as_uri(),
            "gt_points_sha256": gt_sha,
        },
    )
    return _RecordBuildResult(record, dense_payload, gt_payload, audit_record)


def _prediction_record(**kwargs: Any) -> _RecordBuildResult:
    """Build a record while preserving the underlying failure envelope."""

    unit = kwargs.get("unit")
    try:
        return _prediction_record_impl(**kwargs)
    except Attempt05RuntimeError:
        raise
    except Exception as exc:
        if not isinstance(unit, ScientificExecutionUnit):
            unit = None
        raise Attempt05RuntimeError.from_exception(
            exc,
            stage="prediction_record",
            unit=unit,
            reason_code="V4_ATTEMPT05_PREDICTION_RECORD_FAILED",
        ) from exc


def _validate_staged_bundle(
    *,
    staging_dir: Path,
    manifest: RunManifest,
    prediction: PredictionArtifact,
    build: _RecordBuildResult,
    unit: ScientificExecutionUnit | None = None,
) -> TaskAuditRecord:
    try:
        validate_artifact_bundle(manifest, prediction, build.audit_record)
        reread = read_task_audit_record(staging_dir / "task_audit_record.json")
        if reread.record_sha256 != build.record.record_sha256:
            raise Attempt05RuntimeError("V4_ATTEMPT05_TASK_RECORD_DIGEST_MISMATCH")
        for name in ("geometry_prediction", "native_confidence", "valid_mask", "dense_audit", "gt_points"):
            if not (staging_dir / f"{name}.npz").is_file():
                raise Attempt05RuntimeError("V4_ATTEMPT05_STAGED_PAYLOAD_MISSING")
        return reread
    except Attempt05RuntimeError:
        raise
    except Exception as exc:
        raise Attempt05RuntimeError.from_exception(
            exc,
            stage="staged_bundle_validation",
            unit=unit,
            reason_code="V4_ATTEMPT05_STAGED_BUNDLE_INVALID",
        ) from exc


def _calibration_evidence_path(output_dir: Path) -> Path:
    return output_dir / "native_warning_evidence.json"


def _load_calibration_evidence(output_dir: Path, *, manifest: RunManifest, sample_key: SampleKey) -> Mapping[str, Any]:
    evidence_path = _calibration_evidence_path(output_dir)
    if not evidence_path.is_file():
        raise Attempt05RuntimeError("V4_ATTEMPT05_RESUME_CALIBRATION_EVIDENCE_MISSING")
    if (output_dir / "task_audit_record.json").exists() or (output_dir / "audit_record.json").exists():
        raise Attempt05RuntimeError("V4_ATTEMPT05_CALIBRATION_RECORD_FORBIDDEN")
    try:
        payload = json.loads(evidence_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise Attempt05RuntimeError("V4_ATTEMPT05_CALIBRATION_EVIDENCE_INVALID") from exc
    if not isinstance(payload, Mapping):
        raise Attempt05RuntimeError("V4_ATTEMPT05_CALIBRATION_EVIDENCE_INVALID")
    embedded_sha = payload.get("evidence_sha256")
    unsigned = dict(payload)
    unsigned.pop("evidence_sha256", None)
    if (
        payload.get("schema_version") != CALIBRATION_WARNING_EVIDENCE_SCHEMA
        or payload.get("calibration_only") is not True
        or payload.get("scientific_record_created") is not False
        or payload.get("run_id") != manifest.run_id
        or payload.get("sample_key") != str(sample_key)
        or not isinstance(payload.get("warning_score"), (int, float))
        or not math.isfinite(float(payload["warning_score"]))
        or embedded_sha != _json_payload_sha256(unsigned)
    ):
        raise Attempt05RuntimeError("V4_ATTEMPT05_CALIBRATION_EVIDENCE_INVALID")
    return payload


def _calibration_block_or_resume(
    output_dir: Path,
    *,
    manifest: RunManifest,
    sample_key: SampleKey,
    resume: bool,
) -> Mapping[str, Any] | None:
    if _partial_dir(output_dir).exists() or _calibration_evidence_path(output_dir).with_name("native_warning_evidence.json.partial").exists():
        raise Attempt05RuntimeError("V4_ATTEMPT05_PARTIAL_EXISTS")
    if output_dir.exists():
        if not resume:
            raise Attempt05RuntimeError("V4_ATTEMPT05_CALIBRATION_ALREADY_EXISTS")
        return _load_calibration_evidence(output_dir, manifest=manifest, sample_key=sample_key)
    return None


def _build_calibration_evidence(
    *,
    manifest: RunManifest,
    sample_key: SampleKey,
    prediction: PredictionArtifact,
    model_id: str,
    scene_id: int,
    ordered_view_ids: Sequence[int],
    warning_score: float,
) -> Mapping[str, Any]:
    payload: dict[str, Any] = {
        "schema_version": CALIBRATION_WARNING_EVIDENCE_SCHEMA,
        "calibration_only": True,
        "scientific_record_created": False,
        "model_id": model_id,
        "scene_id": scene_id,
        "state_id": "L3",
        "ordered_view_ids": [int(view_id) for view_id in ordered_view_ids],
        "run_id": manifest.run_id,
        "sample_key": str(sample_key),
        "prediction_artifact_uri": "prediction_artifact.json",
        "prediction_payload_digests": dict(prediction.payload_digests),
        "warning_score": float(warning_score),
    }
    unsigned = dict(payload)
    payload["evidence_sha256"] = _json_payload_sha256(unsigned)
    return payload


def _validate_calibration_bundle(
    *,
    output_dir: Path,
    manifest: RunManifest,
    sample_key: SampleKey,
    model_id: str,
    ordered_view_ids: Sequence[int],
) -> Mapping[str, Any]:
    try:
        prediction_payload = json.loads((output_dir / "prediction_artifact.json").read_text(encoding="utf-8"))
        reread_prediction = PredictionArtifact.from_dict(prediction_payload)
    except (OSError, json.JSONDecodeError, ContractError) as exc:
        raise Attempt05RuntimeError("V4_ATTEMPT05_CALIBRATION_BUNDLE_INVALID") from exc
    _validate_prediction(reread_prediction, manifest=manifest, sample_key=sample_key)
    warning = _native_warning_from_prediction(
        reread_prediction,
        model_id=model_id,
        ordered_view_ids=ordered_view_ids,
    )
    evidence = _load_calibration_evidence(output_dir, manifest=manifest, sample_key=sample_key)
    if not math.isclose(float(evidence["warning_score"]), warning, rel_tol=0.0, abs_tol=0.0):
        raise Attempt05RuntimeError("V4_ATTEMPT05_CALIBRATION_EVIDENCE_INVALID")
    for name in ("geometry_prediction", "native_confidence", "valid_mask"):
        if not (output_dir / f"{name}.npz").is_file():
            raise Attempt05RuntimeError("V4_ATTEMPT05_CALIBRATION_PAYLOAD_MISSING")
    if (output_dir / "task_audit_record.json").exists() or (output_dir / "audit_record.json").exists():
        raise Attempt05RuntimeError("V4_ATTEMPT05_CALIBRATION_RECORD_FORBIDDEN")
    return evidence


def execute_attempt05_calibration_l3(
    *,
    manifest: RunManifest,
    sample_key: SampleKey,
    model_id: str,
    scene_id: int,
    rendered_views: Sequence[RenderedView],
    adapter: Any,
    output_dir: Path,
    resume: bool = False,
) -> Attempt05CalibrationResult:
    if not isinstance(manifest, RunManifest) or manifest.mode is not RunMode.REAL:
        raise Attempt05RuntimeError("V4_ATTEMPT05_MANIFEST_INVALID")
    if not isinstance(sample_key, SampleKey) or sample_key.condition != "L3":
        raise Attempt05RuntimeError("V4_ATTEMPT05_SAMPLE_KEY_INVALID")
    if manifest.model != model_id or manifest.dataset != "dtu" or manifest.split != sample_key.split:
        raise Attempt05RuntimeError("V4_ATTEMPT05_MANIFEST_UNIT_MISMATCH")
    if isinstance(scene_id, bool) or not isinstance(scene_id, int) or scene_id <= 0:
        raise Attempt05RuntimeError("V4_ATTEMPT05_CALIBRATION_SCENE_INVALID")
    ordered_view_ids = tuple(view.view_id for view in rendered_views)
    if len(ordered_view_ids) != 8 or len(set(ordered_view_ids)) != 8:
        raise Attempt05RuntimeError("V4_ATTEMPT05_VIEW_ORDER_INVALID")

    resumed = _calibration_block_or_resume(
        output_dir,
        manifest=manifest,
        sample_key=sample_key,
        resume=resume,
    )
    evidence_path = _calibration_evidence_path(output_dir)
    if resumed is not None:
        resumed = _validate_calibration_bundle(
            output_dir=output_dir,
            manifest=manifest,
            sample_key=sample_key,
            model_id=model_id,
            ordered_view_ids=ordered_view_ids,
        )
        return Attempt05CalibrationResult(
            status="CALIBRATION_RESUMED_VALID",
            evidence_path=evidence_path,
            prediction=None,
            warning_score=float(resumed["warning_score"]),
            evidence_sha256=str(resumed["evidence_sha256"]),
        )

    staging_dir = _partial_dir(output_dir)
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    try:
        staging_dir.mkdir(exist_ok=False)
    except FileExistsError as exc:
        raise Attempt05RuntimeError("V4_ATTEMPT05_PARTIAL_EXISTS") from exc

    try:
        prediction = adapter.predict_sample(manifest, sample_key, tuple(rendered_views))
    except Attempt05RuntimeError:
        raise
    except Exception as exc:
        raise Attempt05RuntimeError.from_exception(
            exc,
            stage="adapter_predict",
            unit_key=(model_id, scene_id, "L3"),
            reason_code="V4_ATTEMPT05_ADAPTER_EXCEPTION",
        ) from exc
    try:
        prediction = _validate_prediction(prediction, manifest=manifest, sample_key=sample_key)
    except Attempt05RuntimeError:
        raise
    except Exception as exc:
        raise Attempt05RuntimeError.from_exception(
            exc,
            stage="prediction_validate",
            unit_key=(model_id, scene_id, "L3"),
            reason_code="V4_ATTEMPT05_PREDICTION_ARTIFACT_INVALID",
        ) from exc
    if prediction.invalid_prediction:
        raise Attempt05RuntimeError('V4_ATTEMPT05_CALIBRATION_INVALID_PREDICTION')
    staged_prediction = _copy_prediction_payloads(
        prediction, staging_dir, unit_key=(model_id, scene_id, "L3")
    )
    warning = _native_warning_from_prediction(
        staged_prediction,
        model_id=model_id,
        ordered_view_ids=ordered_view_ids,
        unit_key=(model_id, scene_id, "L3"),
    )
    final_prediction = _prediction_with_uris(
        staged_prediction,
        geometry_path=output_dir / "geometry_prediction.npz",
        confidence_path=output_dir / "native_confidence.npz",
        valid_mask_path=output_dir / "valid_mask.npz",
        payload_digests=staged_prediction.payload_digests,
    )
    evidence = _build_calibration_evidence(
        manifest=manifest,
        sample_key=sample_key,
        prediction=final_prediction,
        model_id=model_id,
        scene_id=scene_id,
        ordered_view_ids=ordered_view_ids,
        warning_score=warning,
    )
    try:
        write_json_artifact(staging_dir / "run_manifest.json", manifest)
        write_json_artifact(staging_dir / "prediction_artifact.json", final_prediction)
        _write_json_payload(staging_dir / "native_warning_evidence.json", evidence)
    except Attempt05RuntimeError:
        raise
    except Exception as exc:
        raise Attempt05RuntimeError.from_exception(
            exc,
            stage="calibration_staged_write",
            unit_key=(model_id, scene_id, "L3"),
            reason_code="V4_ATTEMPT05_STAGED_WRITE_FAILED",
        ) from exc
    if (staging_dir / "task_audit_record.json").exists() or (staging_dir / "audit_record.json").exists():
        raise Attempt05RuntimeError("V4_ATTEMPT05_CALIBRATION_RECORD_FORBIDDEN")
    _promote_staging_dir(
        staging_dir,
        output_dir,
        stage="calibration_promotion",
        unit_key=(model_id, scene_id, "L3"),
    )
    try:
        final_evidence = _validate_calibration_bundle(
            output_dir=output_dir,
            manifest=manifest,
            sample_key=sample_key,
            model_id=model_id,
            ordered_view_ids=ordered_view_ids,
        )
    except Attempt05RuntimeError:
        raise
    except Exception as exc:
        raise Attempt05RuntimeError.from_exception(
            exc,
            stage="calibration_final_validation",
            unit_key=(model_id, scene_id, "L3"),
            reason_code="V4_ATTEMPT05_FINAL_BUNDLE_INVALID",
        ) from exc
    return Attempt05CalibrationResult(
        status="CALIBRATION_WARNING_RECORDED",
        evidence_path=evidence_path,
        prediction=final_prediction,
        warning_score=float(final_evidence["warning_score"]),
        evidence_sha256=str(final_evidence["evidence_sha256"]),
    )


def execute_attempt05_unit(
    *,
    unit: ScientificExecutionUnit,
    manifest: RunManifest,
    sample_key: SampleKey,
    rendered_views: Sequence[RenderedView],
    adapter: Any,
    calibration: NativeWarningCalibration,
    output_dir: Path,
    gt_points: Any,
    gt_camera_c2w: Any,
    observability_mask: Any,
    gt_dtu_camera_c2w: Any,
    observability_bb: Any | None = None,
    observability_res: float | None = None,
    resume: bool = False,
) -> Attempt05UnitResult:
    if not isinstance(unit, ScientificExecutionUnit):
        raise Attempt05RuntimeError("V4_ATTEMPT05_UNIT_INVALID")
    if not isinstance(manifest, RunManifest) or manifest.mode is not RunMode.REAL:
        raise Attempt05RuntimeError("V4_ATTEMPT05_MANIFEST_INVALID")
    if not isinstance(sample_key, SampleKey):
        raise Attempt05RuntimeError("V4_ATTEMPT05_SAMPLE_KEY_INVALID")
    if manifest.model != unit.model_id or manifest.dataset != "dtu" or manifest.split != "test":
        raise Attempt05RuntimeError("V4_ATTEMPT05_MANIFEST_UNIT_MISMATCH")
    if calibration.model_id != unit.model_id:
        raise Attempt05RuntimeError("V4_ATTEMPT05_CALIBRATION_MODEL_MISMATCH")
    ordered_view_ids = tuple(view.view_id for view in rendered_views)
    if len(ordered_view_ids) != 8 or len(set(ordered_view_ids)) != 8:
        raise Attempt05RuntimeError("V4_ATTEMPT05_VIEW_ORDER_INVALID")

    resumed = _block_or_resume(output_dir, resume=resume)
    record_path = _record_path(output_dir)
    if resumed is not None:
        return Attempt05UnitResult(
            status="RESUMED_VALID_COMPLETE" if resumed.valid else "RESUMED_INVALID_FAILURE_RECORDED",
            record_path=record_path,
            prediction=None,
            record=resumed,
        )

    staging_dir = _partial_dir(output_dir)
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    try:
        staging_dir.mkdir(exist_ok=False)
    except FileExistsError as exc:
        raise Attempt05RuntimeError("V4_ATTEMPT05_PARTIAL_EXISTS") from exc

    try:
        prediction = adapter.predict_sample(manifest, sample_key, tuple(rendered_views))
    except Attempt05RuntimeError:
        raise
    except Exception as exc:
        raise Attempt05RuntimeError.from_exception(
            exc,
            stage="adapter_predict",
            unit=unit,
            reason_code="V4_ATTEMPT05_ADAPTER_EXCEPTION",
        ) from exc
    try:
        prediction = _validate_prediction(prediction, manifest=manifest, sample_key=sample_key)
    except Attempt05RuntimeError:
        raise
    except Exception as exc:
        raise Attempt05RuntimeError.from_exception(
            exc,
            stage="prediction_validate",
            unit=unit,
            reason_code="V4_ATTEMPT05_PREDICTION_ARTIFACT_INVALID",
        ) from exc
    staged_prediction = _copy_prediction_payloads(prediction, staging_dir, unit=unit)
    build = _prediction_record(
        unit=unit,
        manifest=manifest,
        prediction=staged_prediction,
        calibration=calibration,
        ordered_view_ids=ordered_view_ids,
        gt_points=gt_points,
        gt_camera_c2w=gt_camera_c2w,
        observability_mask=observability_mask,
        gt_dtu_camera_c2w=gt_dtu_camera_c2w,
        observability_bb=observability_bb,
        observability_res=observability_res,
        staging_dir=staging_dir,
    )
    final_prediction = _prediction_with_uris(
        staged_prediction,
        geometry_path=output_dir / "geometry_prediction.npz",
        confidence_path=output_dir / "native_confidence.npz",
        valid_mask_path=output_dir / "valid_mask.npz",
        payload_digests=staged_prediction.payload_digests,
    )
    final_audit = _audit_with_final_uris(build.audit_record, bundle_dir=output_dir)
    try:
        write_json_artifact(staging_dir / "run_manifest.json", manifest)
        write_json_artifact(staging_dir / "prediction_artifact.json", final_prediction)
        write_json_artifact(staging_dir / "audit_record.json", final_audit)
        write_task_audit_record(staging_dir / "task_audit_record.json", build.record)
    except Attempt05RuntimeError:
        raise
    except Exception as exc:
        raise Attempt05RuntimeError.from_exception(
            exc,
            stage="unit_staged_write",
            unit=unit,
            reason_code="V4_ATTEMPT05_STAGED_WRITE_FAILED",
        ) from exc
    staged_record = _validate_staged_bundle(
        staging_dir=staging_dir,
        manifest=manifest,
        prediction=staged_prediction,
        build=build,
        unit=unit,
    )
    _promote_staging_dir(
        staging_dir,
        output_dir,
        stage="unit_promotion",
        unit=unit,
    )
    try:
        validate_artifact_bundle(manifest, final_prediction, final_audit)
        final_record = read_task_audit_record(record_path)
    except Attempt05RuntimeError:
        raise
    except Exception as exc:
        raise Attempt05RuntimeError.from_exception(
            exc,
            stage="unit_final_validation",
            unit=unit,
            reason_code="V4_ATTEMPT05_FINAL_BUNDLE_INVALID",
        ) from exc
    if final_record.record_sha256 != staged_record.record_sha256:
        raise Attempt05RuntimeError("V4_ATTEMPT05_TASK_RECORD_DIGEST_MISMATCH")
    return Attempt05UnitResult(
        status="VALID_COMPLETE" if final_record.valid else "INVALID_FAILURE_RECORDED",
        record_path=record_path,
        prediction=final_prediction,
        record=final_record,
    )
