"""Strict canonical records for GeoReliab v4 Task 3.

The schemas in this module admit only v4 scientific objects.  They do not
wrap or deserialize v1 ``AuditRecord`` or P2/P3 evidence payloads.
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
from uuid import uuid4

from .v4_counterfactuals import (
    FOG_BOUNDARY_LAG_SEQUENCE,
    LIGHTING_STATES,
    SCIENTIFIC_MODELS,
    SCIENTIFIC_STATES,
    TEST_SCENE_IDS,
    ScientificExecutionUnit,
    canonical_json_bytes,
    canonical_json_sha256,
)
from .v4_metrics import (
    POSE_FAILURE_THRESHOLD_DEG,
    POSE_PAIR_COUNT,
    NativeWarningCalibration,
    PointTaskMetrics,
    PosePairMetrics,
    PoseTaskMetrics,
    V4MetricError,
    empirical_pose_auc,
)
from .v4_science_lock import (
    V4_PROTOCOL_ID,
    V4_PROTOCOL_PROVENANCE_SCHEMA_VERSION,
    V4_PROTOCOL_SHA256,
    V4_PROTOCOL_VERSION,
)


TASK_AUDIT_RECORD_SCHEMA_VERSION = "georeliab-v4-task-audit-record-1.0"
INVALID_POINT_ERROR_MM = 1_000_000_000_000.0
INVALID_POSE_ERROR_DEG = 180.0
MVE_BLOCKED_NATIVE_WARNING_UNAVAILABLE = "MVE_BLOCKED_NATIVE_WARNING_UNAVAILABLE"

_SHA256_CHARACTERS = frozenset("0123456789abcdef")
_TASK_AUDIT_KEYS = frozenset(
    {
        "schema_version",
        "protocol_provenance",
        "dataset",
        "model_id",
        "scene_id",
        "state_id",
        "ordered_view_ids",
        "state_identity_sha256",
        "pair_identity_sha256",
        "execution_unit_sha256",
        "valid",
        "reason_code",
        "point_main_loss",
        "fscore_1mm",
        "fscore_2mm",
        "fscore_5mm",
        "median_predicted_error_mm",
        "static_rank",
        "static_rank_defined",
        "static_rank_reason_code",
        "pose_pairs",
        "auc_5deg",
        "auc_10deg",
        "auc_20deg",
        "pose_main_loss",
        "median_pair_error_deg",
        "pose_failure",
        "native_warning_score",
        "calibration_identifier",
        "alarm_threshold",
        "alarm",
        "record_sha256",
    }
)


class Task3ContractError(ValueError):
    """Raised when a v4 Task 3 record or evidence contract fails closed."""


def _expected_provenance() -> dict[str, str]:
    return {
        "schema_version": V4_PROTOCOL_PROVENANCE_SCHEMA_VERSION,
        "protocol_id": V4_PROTOCOL_ID,
        "protocol_version": V4_PROTOCOL_VERSION,
        "protocol_sha256": V4_PROTOCOL_SHA256,
    }


def _closed_mapping(
    value: object,
    expected: frozenset[str],
    *,
    label: str,
) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise Task3ContractError(f"{label} must be a JSON object")
    non_strings = {key for key in value if not isinstance(key, str)}
    actual = {key for key in value if isinstance(key, str)}
    missing = expected - actual
    unexpected = actual - expected
    if not missing and not unexpected and not non_strings:
        return value
    reasons = []
    if missing:
        reasons.append("missing keys " + ", ".join(sorted(missing)))
    if unexpected:
        reasons.append("unexpected keys " + ", ".join(sorted(unexpected)))
    if non_strings:
        reasons.append(
            "non-string keys " + ", ".join(sorted(repr(key) for key in non_strings))
        )
    raise Task3ContractError(f"{label} closed schema violation: {'; '.join(reasons)}")


def _json_list(value: object, *, label: str) -> list[Any]:
    if not isinstance(value, list):
        raise Task3ContractError(f"{label} must be a JSON list")
    return value


def _finite(value: object, *, label: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
    ):
        raise Task3ContractError(f"{label} must be finite")
    return float(value)


def _integer(value: object, *, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise Task3ContractError(f"{label} must be an integer")
    return value


def _boolean(value: object, *, label: str) -> bool:
    if not isinstance(value, bool):
        raise Task3ContractError(f"{label} must be boolean")
    return value


def _sha256(value: object, *, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or value != value.lower()
        or any(character not in _SHA256_CHARACTERS for character in value)
    ):
        raise Task3ContractError(f"{label} must be a lowercase SHA-256")
    return value


def _duplicate_rejecting_object(
    pairs: list[tuple[str, Any]],
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise Task3ContractError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _parse_json(value: bytes | str, *, label: str) -> object:
    try:
        text = value.decode("utf-8") if isinstance(value, bytes) else value
        return json.loads(text, object_pairs_hook=_duplicate_rejecting_object)
    except UnicodeDecodeError as exc:
        raise Task3ContractError(f"{label} is not UTF-8 JSON") from exc
    except json.JSONDecodeError as exc:
        raise Task3ContractError(f"{label} is not valid JSON: {exc}") from exc


def _canonical_file_bytes(value: object) -> bytes:
    return canonical_json_bytes(value) + b"\n"


def _ordered_views(value: object) -> tuple[int, ...]:
    rows = _json_list(value, label="ordered_view_ids")
    result = tuple(
        _integer(view_id, label=f"ordered_view_ids[{index}]")
        for index, view_id in enumerate(rows)
    )
    if len(result) != 8 or len(set(result)) != 8:
        raise Task3ContractError(
            "ordered_view_ids must contain exactly eight distinct views"
        )
    return result


def _protocol_items(value: object) -> tuple[tuple[str, str], ...]:
    expected = _expected_provenance()
    if not isinstance(value, Mapping) or dict(value) != expected:
        raise Task3ContractError(
            "record provenance is not the exact immutable v4 protocol"
        )
    return tuple(sorted(expected.items()))


def _task_record_payload(
    *,
    protocol_provenance: Mapping[str, str],
    dataset: str,
    model_id: str,
    scene_id: int,
    state_id: str,
    ordered_view_ids: tuple[int, ...],
    state_identity_sha256: str,
    pair_identity_sha256: str | None,
    execution_unit_sha256: str,
    valid: bool,
    reason_code: str,
    point_main_loss: float,
    fscore_1mm: float,
    fscore_2mm: float,
    fscore_5mm: float,
    median_predicted_error_mm: float,
    static_rank: float,
    static_rank_defined: bool,
    static_rank_reason_code: str,
    pose_pairs: tuple[PosePairMetrics, ...],
    auc_5deg: float,
    auc_10deg: float,
    auc_20deg: float,
    pose_main_loss: float,
    median_pair_error_deg: float,
    pose_failure: bool,
    native_warning_score: float,
    calibration_identifier: str,
    alarm_threshold: float,
    alarm: bool,
) -> dict[str, object]:
    return {
        "schema_version": TASK_AUDIT_RECORD_SCHEMA_VERSION,
        "protocol_provenance": dict(protocol_provenance),
        "dataset": dataset,
        "model_id": model_id,
        "scene_id": scene_id,
        "state_id": state_id,
        "ordered_view_ids": list(ordered_view_ids),
        "state_identity_sha256": state_identity_sha256,
        "pair_identity_sha256": pair_identity_sha256,
        "execution_unit_sha256": execution_unit_sha256,
        "valid": valid,
        "reason_code": reason_code,
        "point_main_loss": point_main_loss,
        "fscore_1mm": fscore_1mm,
        "fscore_2mm": fscore_2mm,
        "fscore_5mm": fscore_5mm,
        "median_predicted_error_mm": median_predicted_error_mm,
        "static_rank": static_rank,
        "static_rank_defined": static_rank_defined,
        "static_rank_reason_code": static_rank_reason_code,
        "pose_pairs": [pair.to_dict() for pair in pose_pairs],
        "auc_5deg": auc_5deg,
        "auc_10deg": auc_10deg,
        "auc_20deg": auc_20deg,
        "pose_main_loss": pose_main_loss,
        "median_pair_error_deg": median_pair_error_deg,
        "pose_failure": pose_failure,
        "native_warning_score": native_warning_score,
        "calibration_identifier": calibration_identifier,
        "alarm_threshold": alarm_threshold,
        "alarm": alarm,
    }


@dataclass(frozen=True, slots=True)
class TaskAuditRecord:
    """One strict v4 observation for a model, test scene, and state."""

    protocol_provenance_items: tuple[tuple[str, str], ...]
    dataset: str
    model_id: str
    scene_id: int
    state_id: str
    ordered_view_ids: tuple[int, ...]
    state_identity_sha256: str
    pair_identity_sha256: str | None
    execution_unit_sha256: str
    valid: bool
    reason_code: str
    point_main_loss: float
    fscore_1mm: float
    fscore_2mm: float
    fscore_5mm: float
    median_predicted_error_mm: float
    static_rank: float
    static_rank_defined: bool
    static_rank_reason_code: str
    pose_pairs: tuple[PosePairMetrics, ...]
    auc_5deg: float
    auc_10deg: float
    auc_20deg: float
    pose_main_loss: float
    median_pair_error_deg: float
    pose_failure: bool
    native_warning_score: float
    calibration_identifier: str
    alarm_threshold: float
    alarm: bool
    record_sha256: str
    schema_version: str = TASK_AUDIT_RECORD_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != TASK_AUDIT_RECORD_SCHEMA_VERSION:
            raise Task3ContractError("TaskAuditRecord schema mismatch")
        if dict(self.protocol_provenance_items) != _expected_provenance():
            raise Task3ContractError(
                "TaskAuditRecord is not bound to the exact v4 protocol"
            )
        if self.dataset != "DTU":
            raise Task3ContractError("TaskAuditRecord dataset must be DTU")
        if self.model_id not in SCIENTIFIC_MODELS:
            raise Task3ContractError("TaskAuditRecord model must be VGGT or MASt3R")
        if self.scene_id not in TEST_SCENE_IDS:
            raise Task3ContractError(
                "TaskAuditRecord scene must be a frozen test scene"
            )
        if self.state_id not in SCIENTIFIC_STATES:
            raise Task3ContractError(
                "TaskAuditRecord state is outside the frozen schedule"
            )
        if len(self.ordered_view_ids) != 8 or len(set(self.ordered_view_ids)) != 8:
            raise Task3ContractError(
                "TaskAuditRecord requires eight distinct ordered views"
            )
        if any(
            isinstance(view_id, bool) or not isinstance(view_id, int)
            for view_id in self.ordered_view_ids
        ):
            raise Task3ContractError("TaskAuditRecord view ids must be integers")
        _sha256(self.state_identity_sha256, label="state identity")
        _sha256(self.execution_unit_sha256, label="execution-unit identity")
        if self.state_id == "L3":
            if self.pair_identity_sha256 is not None:
                raise Task3ContractError(
                    "L3 TaskAuditRecord pair identity must be null"
                )
        else:
            _sha256(self.pair_identity_sha256, label="pair identity")
        _boolean(self.valid, label="valid")
        if not isinstance(self.reason_code, str) or not self.reason_code:
            raise Task3ContractError("reason_code must be a non-empty string")
        if self.valid and self.reason_code != "VALID":
            raise Task3ContractError("valid records require reason_code=VALID")
        if not self.valid and self.reason_code == "VALID":
            raise Task3ContractError("invalid records require an invalid reason")

        point_values = (
            self.point_main_loss,
            self.fscore_1mm,
            self.fscore_2mm,
            self.fscore_5mm,
            self.median_predicted_error_mm,
            self.static_rank,
        )
        if any(not math.isfinite(value) for value in point_values):
            raise Task3ContractError("TaskAuditRecord point metrics must be finite")
        if not all(
            0.0 <= value <= 1.0
            for value in (
                self.point_main_loss,
                self.fscore_1mm,
                self.fscore_2mm,
                self.fscore_5mm,
            )
        ):
            raise Task3ContractError("point losses/F-scores must be in [0, 1]")
        if not math.isclose(
            self.point_main_loss,
            1.0 - self.fscore_2mm,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise Task3ContractError("point main loss must equal 1-F-score@2mm")
        if self.median_predicted_error_mm < 0.0:
            raise Task3ContractError("median point error must be non-negative")
        if not -1.0 <= self.static_rank <= 1.0:
            raise Task3ContractError("StaticRank must be in [-1, 1]")
        _boolean(self.static_rank_defined, label="StaticRank defined")
        allowed_static_reasons = {
            "DEFINED",
            "DEGENERATE_CONSTANT_RISK_OR_ERROR",
            "INVALID_MODEL_OUTPUT",
        }
        if self.static_rank_reason_code not in allowed_static_reasons:
            raise Task3ContractError("StaticRank reason code is unsupported")
        if self.static_rank_defined is not (self.static_rank_reason_code == "DEFINED"):
            raise Task3ContractError("StaticRank diagnostic fields are inconsistent")

        if len(self.pose_pairs) != POSE_PAIR_COUNT or any(
            not isinstance(pair, PosePairMetrics) for pair in self.pose_pairs
        ):
            raise Task3ContractError(
                "TaskAuditRecord requires exactly 28 pose-pair metrics"
            )
        expected_pair_ids = tuple(
            (self.ordered_view_ids[first], self.ordered_view_ids[second])
            for first in range(8)
            for second in range(first + 1, 8)
        )
        actual_pair_ids = tuple((pair.view_a, pair.view_b) for pair in self.pose_pairs)
        if actual_pair_ids != expected_pair_ids:
            raise Task3ContractError(
                "pose pairs must follow the exact 28 unordered view pairs"
            )
        aucs = (self.auc_5deg, self.auc_10deg, self.auc_20deg)
        if any(not math.isfinite(value) or not 0.0 <= value <= 1.0 for value in aucs):
            raise Task3ContractError("pose AUCs must be finite in [0, 1]")
        if not self.auc_5deg <= self.auc_10deg <= self.auc_20deg:
            raise Task3ContractError("pose AUCs must be monotone")
        pair_errors = tuple(pair.pair_error_deg for pair in self.pose_pairs)
        expected_aucs = tuple(
            empirical_pose_auc(pair_errors, threshold)
            for threshold in (5.0, 10.0, 20.0)
        )
        if any(
            not math.isclose(
                actual,
                expected,
                rel_tol=0.0,
                abs_tol=1e-12,
            )
            for actual, expected in zip(aucs, expected_aucs, strict=True)
        ):
            raise Task3ContractError("pose AUCs do not match the 28 pair errors")
        if not math.isclose(
            self.pose_main_loss,
            1.0 - self.auc_10deg,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise Task3ContractError("pose main loss must equal 1-AUC@10deg")
        expected_median = sorted(pair_errors)
        expected_median_value = (expected_median[13] + expected_median[14]) / 2.0
        if not math.isclose(
            self.median_pair_error_deg,
            expected_median_value,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise Task3ContractError(
                "median pair error does not match the 28 pair errors"
            )
        if self.pose_failure is not (
            self.median_pair_error_deg > POSE_FAILURE_THRESHOLD_DEG
        ):
            raise Task3ContractError(
                "pose failure must use strict median(pair_error)>10deg"
            )
        score = _finite(
            self.native_warning_score,
            label="native warning score",
        )
        _sha256(self.calibration_identifier, label="calibration identifier")
        threshold = _finite(self.alarm_threshold, label="alarm threshold")
        if self.alarm is not (score >= threshold):
            raise Task3ContractError(
                "alarm label must use the frozen inclusive threshold"
            )

        if not self.valid:
            invalid_contract = (
                self.point_main_loss == 1.0
                and self.fscore_1mm == 0.0
                and self.fscore_2mm == 0.0
                and self.fscore_5mm == 0.0
                and self.median_predicted_error_mm == INVALID_POINT_ERROR_MM
                and self.static_rank == -1.0
                and not self.static_rank_defined
                and self.static_rank_reason_code == "INVALID_MODEL_OUTPUT"
                and self.auc_5deg == 0.0
                and self.auc_10deg == 0.0
                and self.auc_20deg == 0.0
                and self.pose_main_loss == 1.0
                and self.median_pair_error_deg == INVALID_POSE_ERROR_DEG
                and self.pose_failure
                and all(
                    pair.rotation_error_deg == INVALID_POSE_ERROR_DEG
                    and pair.translation_direction_error_deg == INVALID_POSE_ERROR_DEG
                    and pair.pair_error_deg == INVALID_POSE_ERROR_DEG
                    for pair in self.pose_pairs
                )
            )
            if not invalid_contract:
                raise Task3ContractError(
                    "invalid output does not use the frozen failure sentinels"
                )

        _sha256(self.record_sha256, label="TaskAuditRecord digest")
        if self.record_sha256 != canonical_json_sha256(self.payload()):
            raise Task3ContractError("TaskAuditRecord digest tamper or mismatch")

    @property
    def protocol_provenance(self) -> dict[str, str]:
        return dict(self.protocol_provenance_items)

    def payload(self) -> dict[str, object]:
        return _task_record_payload(
            protocol_provenance=self.protocol_provenance,
            dataset=self.dataset,
            model_id=self.model_id,
            scene_id=self.scene_id,
            state_id=self.state_id,
            ordered_view_ids=self.ordered_view_ids,
            state_identity_sha256=self.state_identity_sha256,
            pair_identity_sha256=self.pair_identity_sha256,
            execution_unit_sha256=self.execution_unit_sha256,
            valid=self.valid,
            reason_code=self.reason_code,
            point_main_loss=self.point_main_loss,
            fscore_1mm=self.fscore_1mm,
            fscore_2mm=self.fscore_2mm,
            fscore_5mm=self.fscore_5mm,
            median_predicted_error_mm=self.median_predicted_error_mm,
            static_rank=self.static_rank,
            static_rank_defined=self.static_rank_defined,
            static_rank_reason_code=self.static_rank_reason_code,
            pose_pairs=self.pose_pairs,
            auc_5deg=self.auc_5deg,
            auc_10deg=self.auc_10deg,
            auc_20deg=self.auc_20deg,
            pose_main_loss=self.pose_main_loss,
            median_pair_error_deg=self.median_pair_error_deg,
            pose_failure=self.pose_failure,
            native_warning_score=self.native_warning_score,
            calibration_identifier=self.calibration_identifier,
            alarm_threshold=self.alarm_threshold,
            alarm=self.alarm,
        )

    def to_dict(self) -> dict[str, object]:
        return {**self.payload(), "record_sha256": self.record_sha256}

    def canonical_json_bytes(self) -> bytes:
        return _canonical_file_bytes(self.to_dict())


def _invalid_pose_pairs(
    ordered_view_ids: tuple[int, ...],
) -> tuple[PosePairMetrics, ...]:
    return tuple(
        PosePairMetrics(
            view_a=ordered_view_ids[first],
            view_b=ordered_view_ids[second],
            rotation_error_deg=INVALID_POSE_ERROR_DEG,
            translation_direction_error_deg=INVALID_POSE_ERROR_DEG,
            pair_error_deg=INVALID_POSE_ERROR_DEG,
        )
        for first in range(8)
        for second in range(first + 1, 8)
    )


def build_task_audit_record(
    *,
    execution_unit: ScientificExecutionUnit,
    calibration: NativeWarningCalibration,
    ordered_view_ids: Sequence[int],
    valid: bool,
    reason_code: str,
    native_warning_score_value: float | None,
    point_metrics: PointTaskMetrics | None = None,
    pose_metrics: PoseTaskMetrics | None = None,
) -> TaskAuditRecord:
    """Build one strict record, retaining invalid outputs as full failures."""

    if not isinstance(execution_unit, ScientificExecutionUnit):
        raise Task3ContractError(
            "TaskAuditRecord requires a Task 2 ScientificExecutionUnit"
        )
    try:
        unit = ScientificExecutionUnit.from_dict(execution_unit.to_dict())
    except Exception as exc:
        raise Task3ContractError(
            f"invalid Task 2 execution-unit identity: {exc}"
        ) from exc
    if not isinstance(calibration, NativeWarningCalibration):
        raise Task3ContractError(
            "TaskAuditRecord requires a frozen native-warning calibration"
        )
    if calibration.model_id != unit.model_id:
        raise Task3ContractError(
            "calibration model does not match execution-unit model"
        )
    ordered = tuple(ordered_view_ids)
    if len(ordered) != 8 or len(set(ordered)) != 8:
        raise Task3ContractError(
            "TaskAuditRecord requires eight distinct ordered views"
        )
    if (
        native_warning_score_value is None
        or isinstance(native_warning_score_value, bool)
        or not isinstance(native_warning_score_value, (int, float))
        or not math.isfinite(float(native_warning_score_value))
    ):
        raise Task3ContractError(
            f"{MVE_BLOCKED_NATIVE_WARNING_UNAVAILABLE}: "
            "every scientific output requires a finite native warning score"
        )
    warning_score = float(native_warning_score_value)
    if not isinstance(valid, bool):
        raise Task3ContractError("valid must be an explicit boolean")

    if valid:
        if not isinstance(point_metrics, PointTaskMetrics) or not isinstance(
            pose_metrics, PoseTaskMetrics
        ):
            raise Task3ContractError(
                "valid outputs require point and pose metric records"
            )
        point_main_loss = point_metrics.point_main_loss
        fscore_1mm = point_metrics.fscore_1mm
        fscore_2mm = point_metrics.fscore_2mm
        fscore_5mm = point_metrics.fscore_5mm
        median_point_error = point_metrics.median_predicted_error_mm
        static_rank = point_metrics.static_rank
        static_rank_defined = point_metrics.static_rank_defined
        static_rank_reason = point_metrics.static_rank_reason_code
        pairs = pose_metrics.pairs
        auc_5 = pose_metrics.auc_5deg
        auc_10 = pose_metrics.auc_10deg
        auc_20 = pose_metrics.auc_20deg
        pose_main_loss = pose_metrics.pose_main_loss
        median_pair_error = pose_metrics.median_pair_error_deg
        pose_failure = pose_metrics.pose_failure
    else:
        point_main_loss = 1.0
        fscore_1mm = fscore_2mm = fscore_5mm = 0.0
        median_point_error = INVALID_POINT_ERROR_MM
        static_rank = -1.0
        static_rank_defined = False
        static_rank_reason = "INVALID_MODEL_OUTPUT"
        pairs = _invalid_pose_pairs(ordered)
        auc_5 = auc_10 = auc_20 = 0.0
        pose_main_loss = 1.0
        median_pair_error = INVALID_POSE_ERROR_DEG
        pose_failure = True

    payload = _task_record_payload(
        protocol_provenance=unit.protocol_provenance,
        dataset=unit.dataset,
        model_id=unit.model_id,
        scene_id=unit.scene_id,
        state_id=unit.state_id,
        ordered_view_ids=ordered,
        state_identity_sha256=unit.state_identity_sha256,
        pair_identity_sha256=unit.pair_identity_sha256,
        execution_unit_sha256=unit.execution_unit_sha256,
        valid=valid,
        reason_code=reason_code,
        point_main_loss=point_main_loss,
        fscore_1mm=fscore_1mm,
        fscore_2mm=fscore_2mm,
        fscore_5mm=fscore_5mm,
        median_predicted_error_mm=median_point_error,
        static_rank=static_rank,
        static_rank_defined=static_rank_defined,
        static_rank_reason_code=static_rank_reason,
        pose_pairs=pairs,
        auc_5deg=auc_5,
        auc_10deg=auc_10,
        auc_20deg=auc_20,
        pose_main_loss=pose_main_loss,
        median_pair_error_deg=median_pair_error,
        pose_failure=pose_failure,
        native_warning_score=warning_score,
        calibration_identifier=calibration.calibration_identifier,
        alarm_threshold=calibration.alarm_threshold,
        alarm=calibration.alarm_for(warning_score),
    )
    return TaskAuditRecord(
        protocol_provenance_items=tuple(sorted(unit.protocol_provenance.items())),
        dataset=unit.dataset,
        model_id=unit.model_id,
        scene_id=unit.scene_id,
        state_id=unit.state_id,
        ordered_view_ids=ordered,
        state_identity_sha256=unit.state_identity_sha256,
        pair_identity_sha256=unit.pair_identity_sha256,
        execution_unit_sha256=unit.execution_unit_sha256,
        valid=valid,
        reason_code=reason_code,
        point_main_loss=point_main_loss,
        fscore_1mm=fscore_1mm,
        fscore_2mm=fscore_2mm,
        fscore_5mm=fscore_5mm,
        median_predicted_error_mm=median_point_error,
        static_rank=static_rank,
        static_rank_defined=static_rank_defined,
        static_rank_reason_code=static_rank_reason,
        pose_pairs=pairs,
        auc_5deg=auc_5,
        auc_10deg=auc_10,
        auc_20deg=auc_20,
        pose_main_loss=pose_main_loss,
        median_pair_error_deg=median_pair_error,
        pose_failure=pose_failure,
        native_warning_score=warning_score,
        calibration_identifier=calibration.calibration_identifier,
        alarm_threshold=calibration.alarm_threshold,
        alarm=calibration.alarm_for(warning_score),
        record_sha256=canonical_json_sha256(payload),
    )


def parse_task_audit_record(
    value: bytes | str | Mapping[str, Any],
) -> TaskAuditRecord:
    """Strictly parse and revalidate one canonical TaskAuditRecord."""

    parsed: object = (
        _parse_json(value, label="TaskAuditRecord")
        if isinstance(value, (bytes, str))
        else value
    )
    row = _closed_mapping(
        parsed,
        _TASK_AUDIT_KEYS,
        label="TaskAuditRecord",
    )
    if row["schema_version"] != TASK_AUDIT_RECORD_SCHEMA_VERSION:
        raise Task3ContractError("TaskAuditRecord schema is not the v4 Task 3 schema")
    record_sha256 = _sha256(
        row["record_sha256"],
        label="TaskAuditRecord digest",
    )
    unsigned = {key: item for key, item in row.items() if key != "record_sha256"}
    if record_sha256 != canonical_json_sha256(unsigned):
        raise Task3ContractError("TaskAuditRecord digest tamper or mismatch")
    dataset = row["dataset"]
    model_id = row["model_id"]
    state_id = row["state_id"]
    reason_code = row["reason_code"]
    static_reason = row["static_rank_reason_code"]
    calibration_identifier = row["calibration_identifier"]
    if not all(
        isinstance(value, str)
        for value in (
            dataset,
            model_id,
            state_id,
            reason_code,
            static_reason,
            calibration_identifier,
        )
    ):
        raise Task3ContractError("TaskAuditRecord string fields are invalid")
    pair_identity = row["pair_identity_sha256"]
    if pair_identity is not None and not isinstance(pair_identity, str):
        raise Task3ContractError("pair_identity_sha256 must be a SHA-256 or null")
    pose_rows = _json_list(row["pose_pairs"], label="pose_pairs")
    try:
        pose_pairs = tuple(PosePairMetrics.from_dict(item) for item in pose_rows)
    except V4MetricError as exc:
        raise Task3ContractError(f"invalid pose pair: {exc}") from exc
    return TaskAuditRecord(
        protocol_provenance_items=_protocol_items(row["protocol_provenance"]),
        dataset=dataset,
        model_id=model_id,
        scene_id=_integer(row["scene_id"], label="scene_id"),
        state_id=state_id,
        ordered_view_ids=_ordered_views(row["ordered_view_ids"]),
        state_identity_sha256=_sha256(
            row["state_identity_sha256"],
            label="state identity",
        ),
        pair_identity_sha256=pair_identity,
        execution_unit_sha256=_sha256(
            row["execution_unit_sha256"],
            label="execution-unit identity",
        ),
        valid=_boolean(row["valid"], label="valid"),
        reason_code=reason_code,
        point_main_loss=_finite(
            row["point_main_loss"],
            label="point main loss",
        ),
        fscore_1mm=_finite(row["fscore_1mm"], label="F-score@1mm"),
        fscore_2mm=_finite(row["fscore_2mm"], label="F-score@2mm"),
        fscore_5mm=_finite(row["fscore_5mm"], label="F-score@5mm"),
        median_predicted_error_mm=_finite(
            row["median_predicted_error_mm"],
            label="median predicted point error",
        ),
        static_rank=_finite(row["static_rank"], label="StaticRank"),
        static_rank_defined=_boolean(
            row["static_rank_defined"],
            label="StaticRank defined",
        ),
        static_rank_reason_code=static_reason,
        pose_pairs=pose_pairs,
        auc_5deg=_finite(row["auc_5deg"], label="AUC@5deg"),
        auc_10deg=_finite(row["auc_10deg"], label="AUC@10deg"),
        auc_20deg=_finite(row["auc_20deg"], label="AUC@20deg"),
        pose_main_loss=_finite(
            row["pose_main_loss"],
            label="pose main loss",
        ),
        median_pair_error_deg=_finite(
            row["median_pair_error_deg"],
            label="median pair error",
        ),
        pose_failure=_boolean(
            row["pose_failure"],
            label="pose failure",
        ),
        native_warning_score=_finite(
            row["native_warning_score"],
            label="native warning score",
        ),
        calibration_identifier=_sha256(
            calibration_identifier,
            label="calibration identifier",
        ),
        alarm_threshold=_finite(
            row["alarm_threshold"],
            label="alarm threshold",
        ),
        alarm=_boolean(row["alarm"], label="alarm"),
        record_sha256=record_sha256,
    )


def _fsync_parent_directory(path: Path) -> None:
    flags = getattr(os, "O_DIRECTORY", 0)
    try:
        descriptor = os.open(path, os.O_RDONLY | flags)
    except OSError:
        return
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def write_task_audit_record(path: Path, record: TaskAuditRecord) -> str:
    """Atomically publish canonical JSON without overwriting a conflict."""

    if not isinstance(record, TaskAuditRecord):
        raise Task3ContractError("write requires a TaskAuditRecord")
    expected = record.canonical_json_bytes()
    expected_sha256 = hashlib.sha256(expected).hexdigest()
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise Task3ContractError(
            f"cannot create TaskAuditRecord directory: {path.parent}"
        ) from exc
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    temporary_created = False
    try:
        with temporary.open("xb") as handle:
            temporary_created = True
            handle.write(expected)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError:
            existing = path.read_bytes()
            if existing != expected:
                raise Task3ContractError(
                    f"existing TaskAuditRecord conflicts with payload: {path}"
                )
            return hashlib.sha256(existing).hexdigest()
        except OSError as exc:
            raise Task3ContractError(
                f"cannot atomically publish TaskAuditRecord: {path}"
            ) from exc
        _fsync_parent_directory(path.parent)
        return expected_sha256
    except Task3ContractError:
        raise
    except OSError as exc:
        raise Task3ContractError(
            f"cannot prepare TaskAuditRecord publication: {path}"
        ) from exc
    finally:
        if temporary_created:
            try:
                temporary.unlink(missing_ok=True)
            except OSError as exc:
                raise Task3ContractError(
                    f"cannot clean TaskAuditRecord temporary file: {temporary}"
                ) from exc


def read_task_audit_record(path: Path) -> TaskAuditRecord:
    try:
        payload = path.read_bytes()
    except OSError as exc:
        raise Task3ContractError(f"cannot read TaskAuditRecord: {path}") from exc
    return parse_task_audit_record(payload)


WARNING_EVIDENCE_SCHEMA_VERSION = "georeliab-v4-warning-evidence-1.0"
WARNING_BOOTSTRAP_RESAMPLES = 10_000
WARNING_BOOTSTRAP_SEED = int(
    hashlib.sha256(b"GeoReliab-v4-ranking-warning:task-3-bootstrap").hexdigest()[:8],
    16,
)
WARNING_BOOTSTRAP_CONFIDENCE = 0.95
PRIMARY_BOOTSTRAP_DRAW_GROUP = "PRIMARY_SCENE_BLOCK"
PRIMARY_LIGHTING_ORIGIN = "REAL_DTU_LIGHTING"
SYNTHETIC_FOG_ORIGIN = "SYNTHETIC_FOG_ONLY"
WARNING_FAMILIES = ("ranking-warning", "task-transfer")
WARNING_HYPOTHESES = tuple(
    f"{model_id}:{family}"
    for model_id in SCIENTIFIC_MODELS
    for family in WARNING_FAMILIES
)

_METRIC_ESTIMATE_KEYS = frozenset(
    {
        "point_estimate",
        "ci_lower",
        "ci_upper",
        "n_scenes",
        "defined",
        "reason_code",
        "bootstrap_draw_group",
    }
)
_BOOTSTRAP_METADATA_KEYS = frozenset(
    {
        "n_resamples",
        "seed",
        "confidence",
        "unit",
        "method",
        "repeated_runs_included",
    }
)
_HOLM_EVIDENCE_KEYS = frozenset(
    {
        "hypothesis_id",
        "model_id",
        "family",
        "gap_metric",
        "null_margin",
        "reverse_effect_draw_count",
        "bootstrap_draw_count",
        "raw_p",
        "adjusted_p",
        "alpha",
        "non_reversal_excluded",
        "raw_reason_code",
        "adjusted_reason_code",
    }
)
_MODEL_WARNING_EVIDENCE_KEYS = frozenset(
    {
        "model_id",
        "calibration_identifier",
        "static_rank",
        "crr_point",
        "crr_pose",
        "rwg_pose",
        "sfr_pose",
        "boundary_lag",
        "naurc_point",
        "naurc_pose",
        "ttg_pose",
        "pose_failure_scene_count",
        "fog_no_failure_scene_count",
        "ranking_warning_strong",
        "task_transfer_strong",
        "ranking_warning_reason_code",
        "task_transfer_reason_code",
    }
)
_WARNING_EVIDENCE_KEYS = frozenset(
    {
        "schema_version",
        "protocol_provenance",
        "dataset",
        "primary_evidence_origin",
        "primary_state_ids",
        "scientific_schedule_sha256",
        "input_record_inventory_sha256",
        "input_record_count",
        "bootstrap_metadata",
        "models",
        "holm",
        "evidence_sha256",
    }
)


@dataclass(frozen=True, slots=True)
class MetricEstimate:
    """Point estimate and deterministic scene-level percentile interval."""

    point_estimate: float
    ci_lower: float
    ci_upper: float
    n_scenes: int
    defined: bool
    reason_code: str
    bootstrap_draw_group: str

    def __post_init__(self) -> None:
        point = _finite(self.point_estimate, label="metric point estimate")
        lower = _finite(self.ci_lower, label="metric CI lower")
        upper = _finite(self.ci_upper, label="metric CI upper")
        if lower > upper:
            raise Task3ContractError("metric CI lower must not exceed upper")
        if point != self.point_estimate:
            raise Task3ContractError("metric point estimate must be canonical")
        if (
            isinstance(self.n_scenes, bool)
            or not isinstance(self.n_scenes, int)
            or self.n_scenes < 0
        ):
            raise Task3ContractError(
                "metric scene count must be a non-negative integer"
            )
        if not isinstance(self.defined, bool):
            raise Task3ContractError("metric defined flag must be boolean")
        if self.defined and self.n_scenes == 0:
            raise Task3ContractError("defined metric must contain at least one scene")
        if not isinstance(self.reason_code, str) or not self.reason_code:
            raise Task3ContractError("metric reason code must be non-empty")
        if self.defined and self.reason_code != "DEFINED":
            raise Task3ContractError("defined metric must use reason code DEFINED")
        if self.bootstrap_draw_group != PRIMARY_BOOTSTRAP_DRAW_GROUP:
            raise Task3ContractError(
                "metric must use the frozen primary scene-block draw group"
            )

    def to_dict(self) -> dict[str, object]:
        return {
            "point_estimate": self.point_estimate,
            "ci_lower": self.ci_lower,
            "ci_upper": self.ci_upper,
            "n_scenes": self.n_scenes,
            "defined": self.defined,
            "reason_code": self.reason_code,
            "bootstrap_draw_group": self.bootstrap_draw_group,
        }

    @classmethod
    def from_dict(cls, value: object) -> MetricEstimate:
        row = _closed_mapping(
            value,
            _METRIC_ESTIMATE_KEYS,
            label="MetricEstimate",
        )
        reason = row["reason_code"]
        draw_group = row["bootstrap_draw_group"]
        if not isinstance(reason, str) or not isinstance(draw_group, str):
            raise Task3ContractError(
                "MetricEstimate reason and draw group must be strings"
            )
        return cls(
            point_estimate=_finite(
                row["point_estimate"],
                label="metric point estimate",
            ),
            ci_lower=_finite(row["ci_lower"], label="metric CI lower"),
            ci_upper=_finite(row["ci_upper"], label="metric CI upper"),
            n_scenes=_integer(row["n_scenes"], label="metric scene count"),
            defined=_boolean(row["defined"], label="metric defined"),
            reason_code=reason,
            bootstrap_draw_group=draw_group,
        )


@dataclass(frozen=True, slots=True)
class BootstrapMetadata:
    """Frozen primary-evidence resampling contract."""

    n_resamples: int
    seed: int
    confidence: float
    unit: str
    method: str
    repeated_runs_included: bool

    def __post_init__(self) -> None:
        if self.n_resamples != WARNING_BOOTSTRAP_RESAMPLES:
            raise Task3ContractError(
                "primary evidence requires exactly 10000 resamples"
            )
        if self.seed != WARNING_BOOTSTRAP_SEED:
            raise Task3ContractError("bootstrap seed must equal the frozen Task 3 seed")
        if self.confidence != WARNING_BOOTSTRAP_CONFIDENCE:
            raise Task3ContractError("bootstrap confidence must equal 0.95")
        if self.unit != "scene":
            raise Task3ContractError("bootstrap unit must be scene")
        if self.method != "percentile":
            raise Task3ContractError("bootstrap method must be percentile")
        if self.repeated_runs_included is not False:
            raise Task3ContractError(
                "repeated numerical reruns cannot be bootstrap units"
            )

    @classmethod
    def frozen(cls) -> BootstrapMetadata:
        return cls(
            n_resamples=WARNING_BOOTSTRAP_RESAMPLES,
            seed=WARNING_BOOTSTRAP_SEED,
            confidence=WARNING_BOOTSTRAP_CONFIDENCE,
            unit="scene",
            method="percentile",
            repeated_runs_included=False,
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "n_resamples": self.n_resamples,
            "seed": self.seed,
            "confidence": self.confidence,
            "unit": self.unit,
            "method": self.method,
            "repeated_runs_included": self.repeated_runs_included,
        }

    @classmethod
    def from_dict(cls, value: object) -> BootstrapMetadata:
        row = _closed_mapping(
            value,
            _BOOTSTRAP_METADATA_KEYS,
            label="BootstrapMetadata",
        )
        unit = row["unit"]
        method = row["method"]
        if not isinstance(unit, str) or not isinstance(method, str):
            raise Task3ContractError("bootstrap unit and method must be strings")
        return cls(
            n_resamples=_integer(
                row["n_resamples"],
                label="bootstrap resamples",
            ),
            seed=_integer(row["seed"], label="bootstrap seed"),
            confidence=_finite(
                row["confidence"],
                label="bootstrap confidence",
            ),
            unit=unit,
            method=method,
            repeated_runs_included=_boolean(
                row["repeated_runs_included"],
                label="bootstrap repeated runs",
            ),
        )


@dataclass(frozen=True, slots=True)
class HolmEvidence:
    """Raw and four-hypothesis Holm-adjusted non-reversal evidence."""

    hypothesis_id: str
    model_id: str
    family: str
    gap_metric: str
    null_margin: float
    reverse_effect_draw_count: int
    bootstrap_draw_count: int
    raw_p: float
    adjusted_p: float
    alpha: float
    non_reversal_excluded: bool
    raw_reason_code: str
    adjusted_reason_code: str

    def __post_init__(self) -> None:
        if self.model_id not in SCIENTIFIC_MODELS:
            raise Task3ContractError("Holm evidence model is outside v4 MVE")
        if self.family not in WARNING_FAMILIES:
            raise Task3ContractError("Holm evidence family is invalid")
        if self.hypothesis_id != f"{self.model_id}:{self.family}":
            raise Task3ContractError(
                "Holm hypothesis identity does not match model and family"
            )
        expected_gap = "RWG_POSE" if self.family == "ranking-warning" else "TTG_POSE"
        if self.gap_metric != expected_gap:
            raise Task3ContractError("Holm gap metric does not match its gate family")
        if self.null_margin != -0.10:
            raise Task3ContractError("Holm non-reversal null margin must equal -0.10")
        if self.bootstrap_draw_count != WARNING_BOOTSTRAP_RESAMPLES:
            raise Task3ContractError(
                "Holm evidence must use exactly 10000 bootstrap draws"
            )
        if (
            isinstance(self.reverse_effect_draw_count, bool)
            or not isinstance(self.reverse_effect_draw_count, int)
            or not 0 <= self.reverse_effect_draw_count <= self.bootstrap_draw_count
        ):
            raise Task3ContractError("Holm reverse-effect draw count is invalid")
        for label, value in (
            ("raw p", self.raw_p),
            ("adjusted p", self.adjusted_p),
            ("alpha", self.alpha),
        ):
            parsed = _finite(value, label=f"Holm {label}")
            if not 0.0 <= parsed <= 1.0:
                raise Task3ContractError(f"Holm {label} must be in [0, 1]")
        if self.alpha != 0.05:
            raise Task3ContractError("Holm alpha must equal 0.05")
        expected_raw_p = (self.reverse_effect_draw_count + 1) / (
            self.bootstrap_draw_count + 1
        )
        if not math.isclose(
            self.raw_p,
            expected_raw_p,
            rel_tol=0.0,
            abs_tol=1e-15,
        ):
            raise Task3ContractError(
                "Holm raw p does not match its bootstrap tail count"
            )
        if self.adjusted_p < self.raw_p:
            raise Task3ContractError("Holm adjusted p cannot be below raw p")
        expected_excluded = self.adjusted_p <= self.alpha
        if self.non_reversal_excluded is not expected_excluded:
            raise Task3ContractError("Holm decision does not match adjusted p-value")
        expected_raw_reason = (
            "RAW_NON_REVERSAL_EXCLUDED"
            if self.raw_p <= self.alpha
            else "RAW_NON_REVERSAL_NOT_EXCLUDED"
        )
        expected_adjusted_reason = (
            "HOLM_NON_REVERSAL_EXCLUDED"
            if expected_excluded
            else "HOLM_NON_REVERSAL_NOT_EXCLUDED"
        )
        if self.raw_reason_code != expected_raw_reason:
            raise Task3ContractError("Holm raw reason code does not match raw p-value")
        if self.adjusted_reason_code != expected_adjusted_reason:
            raise Task3ContractError(
                "Holm adjusted reason code does not match adjusted p-value"
            )

    def to_dict(self) -> dict[str, object]:
        return {
            "hypothesis_id": self.hypothesis_id,
            "model_id": self.model_id,
            "family": self.family,
            "gap_metric": self.gap_metric,
            "null_margin": self.null_margin,
            "reverse_effect_draw_count": (self.reverse_effect_draw_count),
            "bootstrap_draw_count": self.bootstrap_draw_count,
            "raw_p": self.raw_p,
            "adjusted_p": self.adjusted_p,
            "alpha": self.alpha,
            "non_reversal_excluded": self.non_reversal_excluded,
            "raw_reason_code": self.raw_reason_code,
            "adjusted_reason_code": self.adjusted_reason_code,
        }

    @classmethod
    def from_dict(cls, value: object) -> HolmEvidence:
        row = _closed_mapping(
            value,
            _HOLM_EVIDENCE_KEYS,
            label="HolmEvidence",
        )
        string_fields = (
            "hypothesis_id",
            "model_id",
            "family",
            "gap_metric",
            "raw_reason_code",
            "adjusted_reason_code",
        )
        if any(not isinstance(row[field], str) for field in string_fields):
            raise Task3ContractError("Holm text fields must be strings")
        return cls(
            hypothesis_id=row["hypothesis_id"],
            model_id=row["model_id"],
            family=row["family"],
            gap_metric=row["gap_metric"],
            null_margin=_finite(
                row["null_margin"],
                label="Holm null margin",
            ),
            reverse_effect_draw_count=_integer(
                row["reverse_effect_draw_count"],
                label="Holm reverse-effect draw count",
            ),
            bootstrap_draw_count=_integer(
                row["bootstrap_draw_count"],
                label="Holm bootstrap draw count",
            ),
            raw_p=_finite(row["raw_p"], label="Holm raw p"),
            adjusted_p=_finite(
                row["adjusted_p"],
                label="Holm adjusted p",
            ),
            alpha=_finite(row["alpha"], label="Holm alpha"),
            non_reversal_excluded=_boolean(
                row["non_reversal_excluded"],
                label="Holm non-reversal decision",
            ),
            raw_reason_code=row["raw_reason_code"],
            adjusted_reason_code=row["adjusted_reason_code"],
        )


def _ranking_warning_strong(
    *,
    static_rank: MetricEstimate,
    crr_pose: MetricEstimate,
    rwg_pose: MetricEstimate,
    sfr_pose: MetricEstimate,
) -> bool:
    return (
        sfr_pose.ci_lower >= 0.30
        and static_rank.ci_lower >= 0.35
        and crr_pose.ci_upper <= 0.15
        and rwg_pose.ci_lower >= 0.20
    )


def _task_transfer_strong(
    *,
    naurc_point: MetricEstimate,
    ttg_pose: MetricEstimate,
    sfr_pose: MetricEstimate,
) -> bool:
    return (
        sfr_pose.ci_lower >= 0.30
        and naurc_point.ci_upper <= 0.50
        and ttg_pose.ci_lower >= 0.20
    )


@dataclass(frozen=True, slots=True)
class ModelWarningEvidence:
    """All Task 3 estimates and frozen branch decisions for one model."""

    model_id: str
    calibration_identifier: str
    static_rank: MetricEstimate
    crr_point: MetricEstimate
    crr_pose: MetricEstimate
    rwg_pose: MetricEstimate
    sfr_pose: MetricEstimate
    boundary_lag: MetricEstimate
    naurc_point: MetricEstimate
    naurc_pose: MetricEstimate
    ttg_pose: MetricEstimate
    pose_failure_scene_count: int
    fog_no_failure_scene_count: int
    ranking_warning_strong: bool
    task_transfer_strong: bool
    ranking_warning_reason_code: str
    task_transfer_reason_code: str

    def __post_init__(self) -> None:
        if self.model_id not in SCIENTIFIC_MODELS:
            raise Task3ContractError(
                "model warning evidence is outside the two-model MVE"
            )
        _sha256(
            self.calibration_identifier,
            label="calibration identifier",
        )
        estimates = (
            self.static_rank,
            self.crr_point,
            self.crr_pose,
            self.rwg_pose,
            self.sfr_pose,
            self.boundary_lag,
            self.naurc_point,
            self.naurc_pose,
            self.ttg_pose,
        )
        if any(not isinstance(item, MetricEstimate) for item in estimates):
            raise Task3ContractError(
                "model warning metrics must be MetricEstimate objects"
            )
        if not math.isclose(
            self.rwg_pose.point_estimate,
            self.static_rank.point_estimate - self.crr_pose.point_estimate,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise Task3ContractError("RWG-pose must equal StaticRank minus CRR-pose")
        if not math.isclose(
            self.ttg_pose.point_estimate,
            self.naurc_pose.point_estimate - self.naurc_point.point_estimate,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise Task3ContractError("TTG-pose must equal nAURC-pose minus nAURC-point")
        for label, value in (
            ("pose failure scene count", self.pose_failure_scene_count),
            ("fog no-failure scene count", self.fog_no_failure_scene_count),
        ):
            if (
                isinstance(value, bool)
                or not isinstance(value, int)
                or not 0 <= value <= len(TEST_SCENE_IDS)
            ):
                raise Task3ContractError(
                    f"{label} must be an integer in the test-scene range"
                )
        if (
            self.pose_failure_scene_count > 0
            and self.sfr_pose.n_scenes != self.pose_failure_scene_count
        ):
            raise Task3ContractError(
                "SFR scene count must equal distinct pose-failure scenes"
            )
        if self.boundary_lag.n_scenes + self.fog_no_failure_scene_count != len(
            TEST_SCENE_IDS
        ):
            raise Task3ContractError(
                "Boundary Lag and no-failure counts must cover 20 scenes"
            )
        expected_ranking = _ranking_warning_strong(
            static_rank=self.static_rank,
            crr_pose=self.crr_pose,
            rwg_pose=self.rwg_pose,
            sfr_pose=self.sfr_pose,
        )
        expected_transfer = _task_transfer_strong(
            naurc_point=self.naurc_point,
            ttg_pose=self.ttg_pose,
            sfr_pose=self.sfr_pose,
        )
        if self.ranking_warning_strong is not expected_ranking:
            raise Task3ContractError(
                "ranking-warning decision does not match frozen thresholds"
            )
        if self.task_transfer_strong is not expected_transfer:
            raise Task3ContractError(
                "task-transfer decision does not match frozen thresholds"
            )
        expected_ranking_reason = (
            "STRONG_RANKING_WARNING"
            if expected_ranking
            else "RANKING_WARNING_THRESHOLDS_NOT_MET"
        )
        expected_transfer_reason = (
            "STRONG_TASK_TRANSFER"
            if expected_transfer
            else "TASK_TRANSFER_THRESHOLDS_NOT_MET"
        )
        if self.ranking_warning_reason_code != expected_ranking_reason:
            raise Task3ContractError(
                "ranking-warning reason does not match branch decision"
            )
        if self.task_transfer_reason_code != expected_transfer_reason:
            raise Task3ContractError(
                "task-transfer reason does not match branch decision"
            )

    @classmethod
    def create(
        cls,
        *,
        model_id: str,
        calibration_identifier: str,
        static_rank: MetricEstimate,
        crr_point: MetricEstimate,
        crr_pose: MetricEstimate,
        rwg_pose: MetricEstimate,
        sfr_pose: MetricEstimate,
        boundary_lag: MetricEstimate,
        naurc_point: MetricEstimate,
        naurc_pose: MetricEstimate,
        ttg_pose: MetricEstimate,
        pose_failure_scene_count: int,
        fog_no_failure_scene_count: int,
    ) -> ModelWarningEvidence:
        ranking = _ranking_warning_strong(
            static_rank=static_rank,
            crr_pose=crr_pose,
            rwg_pose=rwg_pose,
            sfr_pose=sfr_pose,
        )
        transfer = _task_transfer_strong(
            naurc_point=naurc_point,
            ttg_pose=ttg_pose,
            sfr_pose=sfr_pose,
        )
        return cls(
            model_id=model_id,
            calibration_identifier=calibration_identifier,
            static_rank=static_rank,
            crr_point=crr_point,
            crr_pose=crr_pose,
            rwg_pose=rwg_pose,
            sfr_pose=sfr_pose,
            boundary_lag=boundary_lag,
            naurc_point=naurc_point,
            naurc_pose=naurc_pose,
            ttg_pose=ttg_pose,
            pose_failure_scene_count=pose_failure_scene_count,
            fog_no_failure_scene_count=fog_no_failure_scene_count,
            ranking_warning_strong=ranking,
            task_transfer_strong=transfer,
            ranking_warning_reason_code=(
                "STRONG_RANKING_WARNING"
                if ranking
                else "RANKING_WARNING_THRESHOLDS_NOT_MET"
            ),
            task_transfer_reason_code=(
                "STRONG_TASK_TRANSFER"
                if transfer
                else "TASK_TRANSFER_THRESHOLDS_NOT_MET"
            ),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "model_id": self.model_id,
            "calibration_identifier": self.calibration_identifier,
            "static_rank": self.static_rank.to_dict(),
            "crr_point": self.crr_point.to_dict(),
            "crr_pose": self.crr_pose.to_dict(),
            "rwg_pose": self.rwg_pose.to_dict(),
            "sfr_pose": self.sfr_pose.to_dict(),
            "boundary_lag": self.boundary_lag.to_dict(),
            "naurc_point": self.naurc_point.to_dict(),
            "naurc_pose": self.naurc_pose.to_dict(),
            "ttg_pose": self.ttg_pose.to_dict(),
            "pose_failure_scene_count": self.pose_failure_scene_count,
            "fog_no_failure_scene_count": self.fog_no_failure_scene_count,
            "ranking_warning_strong": self.ranking_warning_strong,
            "task_transfer_strong": self.task_transfer_strong,
            "ranking_warning_reason_code": (self.ranking_warning_reason_code),
            "task_transfer_reason_code": self.task_transfer_reason_code,
        }

    @classmethod
    def from_dict(cls, value: object) -> ModelWarningEvidence:
        row = _closed_mapping(
            value,
            _MODEL_WARNING_EVIDENCE_KEYS,
            label="ModelWarningEvidence",
        )
        string_fields = (
            "model_id",
            "calibration_identifier",
            "ranking_warning_reason_code",
            "task_transfer_reason_code",
        )
        if any(not isinstance(row[field], str) for field in string_fields):
            raise Task3ContractError(
                "model warning evidence text fields must be strings"
            )
        return cls(
            model_id=row["model_id"],
            calibration_identifier=row["calibration_identifier"],
            static_rank=MetricEstimate.from_dict(row["static_rank"]),
            crr_point=MetricEstimate.from_dict(row["crr_point"]),
            crr_pose=MetricEstimate.from_dict(row["crr_pose"]),
            rwg_pose=MetricEstimate.from_dict(row["rwg_pose"]),
            sfr_pose=MetricEstimate.from_dict(row["sfr_pose"]),
            boundary_lag=MetricEstimate.from_dict(row["boundary_lag"]),
            naurc_point=MetricEstimate.from_dict(row["naurc_point"]),
            naurc_pose=MetricEstimate.from_dict(row["naurc_pose"]),
            ttg_pose=MetricEstimate.from_dict(row["ttg_pose"]),
            pose_failure_scene_count=_integer(
                row["pose_failure_scene_count"],
                label="pose failure scene count",
            ),
            fog_no_failure_scene_count=_integer(
                row["fog_no_failure_scene_count"],
                label="fog no-failure scene count",
            ),
            ranking_warning_strong=_boolean(
                row["ranking_warning_strong"],
                label="ranking-warning strong",
            ),
            task_transfer_strong=_boolean(
                row["task_transfer_strong"],
                label="task-transfer strong",
            ),
            ranking_warning_reason_code=row["ranking_warning_reason_code"],
            task_transfer_reason_code=row["task_transfer_reason_code"],
        )


def _warning_evidence_payload(
    *,
    protocol_provenance: Mapping[str, str],
    dataset: str,
    primary_evidence_origin: str,
    primary_state_ids: tuple[str, ...],
    scientific_schedule_sha256: str,
    input_record_inventory_sha256: str,
    input_record_count: int,
    bootstrap_metadata: BootstrapMetadata,
    models: tuple[ModelWarningEvidence, ...],
    holm: tuple[HolmEvidence, ...],
) -> dict[str, object]:
    return {
        "schema_version": WARNING_EVIDENCE_SCHEMA_VERSION,
        "protocol_provenance": dict(protocol_provenance),
        "dataset": dataset,
        "primary_evidence_origin": primary_evidence_origin,
        "primary_state_ids": list(primary_state_ids),
        "scientific_schedule_sha256": scientific_schedule_sha256,
        "input_record_inventory_sha256": (input_record_inventory_sha256),
        "input_record_count": input_record_count,
        "bootstrap_metadata": bootstrap_metadata.to_dict(),
        "models": [model.to_dict() for model in models],
        "holm": [item.to_dict() for item in holm],
    }


@dataclass(frozen=True, slots=True)
class WarningEvidence:
    """Strict v4-only evidence bundle consumed by the frozen Task 3 gate."""

    protocol_provenance_items: tuple[tuple[str, str], ...]
    dataset: str
    primary_evidence_origin: str
    primary_state_ids: tuple[str, ...]
    scientific_schedule_sha256: str
    input_record_inventory_sha256: str
    input_record_count: int
    bootstrap_metadata: BootstrapMetadata
    models: tuple[ModelWarningEvidence, ...]
    holm: tuple[HolmEvidence, ...]
    evidence_sha256: str
    schema_version: str = WARNING_EVIDENCE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != WARNING_EVIDENCE_SCHEMA_VERSION:
            raise Task3ContractError("WarningEvidence schema mismatch")
        if dict(self.protocol_provenance_items) != _expected_provenance():
            raise Task3ContractError(
                "WarningEvidence is not bound to the exact v4 protocol"
            )
        if self.dataset != "DTU":
            raise Task3ContractError("WarningEvidence dataset must be DTU")
        if self.primary_evidence_origin not in {
            PRIMARY_LIGHTING_ORIGIN,
            SYNTHETIC_FOG_ORIGIN,
        }:
            raise Task3ContractError("WarningEvidence primary origin is unsupported")
        if (
            not self.primary_state_ids
            or any(
                not isinstance(state_id, str) or not state_id
                for state_id in self.primary_state_ids
            )
            or len(set(self.primary_state_ids)) != len(self.primary_state_ids)
        ):
            raise Task3ContractError(
                "WarningEvidence primary states must be distinct strings"
            )
        expected_primary_states = (
            LIGHTING_STATES
            if self.primary_evidence_origin == PRIMARY_LIGHTING_ORIGIN
            else FOG_BOUNDARY_LAG_SEQUENCE
        )
        if self.primary_state_ids != expected_primary_states:
            raise Task3ContractError(
                "WarningEvidence primary states do not match its origin"
            )
        _sha256(
            self.scientific_schedule_sha256,
            label="scientific schedule fingerprint",
        )
        _sha256(
            self.input_record_inventory_sha256,
            label="input record inventory",
        )
        if (
            isinstance(self.input_record_count, bool)
            or not isinstance(self.input_record_count, int)
            or self.input_record_count != 400
        ):
            raise Task3ContractError(
                "WarningEvidence input record count must equal 400"
            )
        if not isinstance(self.bootstrap_metadata, BootstrapMetadata):
            raise Task3ContractError("WarningEvidence bootstrap metadata is invalid")
        if tuple(model.model_id for model in self.models) != SCIENTIFIC_MODELS:
            raise Task3ContractError(
                "WarningEvidence requires exactly VGGT then MASt3R"
            )
        if any(not isinstance(item, HolmEvidence) for item in self.holm):
            raise Task3ContractError(
                "WarningEvidence Holm rows must be HolmEvidence objects"
            )
        if tuple(item.hypothesis_id for item in self.holm) != WARNING_HYPOTHESES:
            raise Task3ContractError(
                "WarningEvidence requires the exact four Holm hypotheses"
            )
        ordered_holm = sorted(
            self.holm,
            key=lambda item: (
                item.raw_p,
                WARNING_HYPOTHESES.index(item.hypothesis_id),
            ),
        )
        adjusted_by_id: dict[str, float] = {}
        running_max = 0.0
        for rank, item in enumerate(ordered_holm):
            candidate = min(
                1.0,
                (len(ordered_holm) - rank) * item.raw_p,
            )
            running_max = max(running_max, candidate)
            adjusted_by_id[item.hypothesis_id] = running_max
        model_by_id = {model.model_id: model for model in self.models}
        for item in self.holm:
            if not math.isclose(
                item.adjusted_p,
                adjusted_by_id[item.hypothesis_id],
                rel_tol=0.0,
                abs_tol=1e-15,
            ):
                raise Task3ContractError(
                    "Holm adjusted p does not match the exact four-family "
                    "step-down calculation"
                )
            model = model_by_id[item.model_id]
            gap = model.rwg_pose if item.family == "ranking-warning" else model.ttg_pose
            if item.non_reversal_excluded and gap.ci_lower <= item.null_margin:
                raise Task3ContractError(
                    "Holm non-reversal exclusion conflicts with the "
                    "corresponding gap confidence interval"
                )
        _sha256(self.evidence_sha256, label="WarningEvidence digest")
        if self.evidence_sha256 != canonical_json_sha256(self.payload()):
            raise Task3ContractError("WarningEvidence digest tamper or mismatch")

    @property
    def protocol_provenance(self) -> dict[str, str]:
        return dict(self.protocol_provenance_items)

    def payload(self) -> dict[str, object]:
        return _warning_evidence_payload(
            protocol_provenance=self.protocol_provenance,
            dataset=self.dataset,
            primary_evidence_origin=self.primary_evidence_origin,
            primary_state_ids=self.primary_state_ids,
            scientific_schedule_sha256=self.scientific_schedule_sha256,
            input_record_inventory_sha256=(self.input_record_inventory_sha256),
            input_record_count=self.input_record_count,
            bootstrap_metadata=self.bootstrap_metadata,
            models=self.models,
            holm=self.holm,
        )

    def to_dict(self) -> dict[str, object]:
        return {
            **self.payload(),
            "evidence_sha256": self.evidence_sha256,
        }

    def canonical_json_bytes(self) -> bytes:
        return _canonical_file_bytes(self.to_dict())


def build_warning_evidence_record(
    *,
    primary_evidence_origin: str,
    primary_state_ids: Sequence[str],
    scientific_schedule_sha256: str,
    input_record_inventory_sha256: str,
    input_record_count: int,
    bootstrap_metadata: BootstrapMetadata,
    models: Sequence[ModelWarningEvidence],
    holm: Sequence[HolmEvidence],
) -> WarningEvidence:
    """Construct a digest-bound v4 WarningEvidence record."""

    state_tuple = tuple(primary_state_ids)
    model_tuple = tuple(models)
    holm_tuple = tuple(holm)
    provenance = _expected_provenance()
    payload = _warning_evidence_payload(
        protocol_provenance=provenance,
        dataset="DTU",
        primary_evidence_origin=primary_evidence_origin,
        primary_state_ids=state_tuple,
        scientific_schedule_sha256=scientific_schedule_sha256,
        input_record_inventory_sha256=input_record_inventory_sha256,
        input_record_count=input_record_count,
        bootstrap_metadata=bootstrap_metadata,
        models=model_tuple,
        holm=holm_tuple,
    )
    return WarningEvidence(
        protocol_provenance_items=tuple(sorted(provenance.items())),
        dataset="DTU",
        primary_evidence_origin=primary_evidence_origin,
        primary_state_ids=state_tuple,
        scientific_schedule_sha256=scientific_schedule_sha256,
        input_record_inventory_sha256=input_record_inventory_sha256,
        input_record_count=input_record_count,
        bootstrap_metadata=bootstrap_metadata,
        models=model_tuple,
        holm=holm_tuple,
        evidence_sha256=canonical_json_sha256(payload),
    )


def parse_warning_evidence(
    value: bytes | str | Mapping[str, Any],
) -> WarningEvidence:
    """Parse strict canonical v4 evidence and reject v1/raw runtime objects."""

    parsed: object
    if isinstance(value, (bytes, str)):
        parsed = _parse_json(value, label="WarningEvidence")
    else:
        parsed = value
    row = _closed_mapping(
        parsed,
        _WARNING_EVIDENCE_KEYS,
        label="WarningEvidence",
    )
    if row["schema_version"] != WARNING_EVIDENCE_SCHEMA_VERSION:
        raise Task3ContractError("WarningEvidence schema does not identify v4 evidence")
    evidence_sha256 = _sha256(
        row["evidence_sha256"],
        label="WarningEvidence digest",
    )
    payload = {key: item for key, item in row.items() if key != "evidence_sha256"}
    if canonical_json_sha256(payload) != evidence_sha256:
        raise Task3ContractError("WarningEvidence digest tamper or mismatch")
    provenance_items = _protocol_items(row["protocol_provenance"])
    if row["dataset"] != "DTU":
        raise Task3ContractError("WarningEvidence dataset must be DTU")
    origin = row["primary_evidence_origin"]
    if not isinstance(origin, str):
        raise Task3ContractError("WarningEvidence primary origin must be a string")
    primary_state_rows = _json_list(
        row["primary_state_ids"],
        label="WarningEvidence primary states",
    )
    if any(not isinstance(state_id, str) for state_id in primary_state_rows):
        raise Task3ContractError("WarningEvidence primary states must be strings")
    model_rows = _json_list(
        row["models"],
        label="WarningEvidence models",
    )
    holm_rows = _json_list(
        row["holm"],
        label="WarningEvidence Holm evidence",
    )
    return WarningEvidence(
        protocol_provenance_items=provenance_items,
        dataset="DTU",
        primary_evidence_origin=origin,
        primary_state_ids=tuple(primary_state_rows),
        scientific_schedule_sha256=_sha256(
            row["scientific_schedule_sha256"],
            label="scientific schedule fingerprint",
        ),
        input_record_inventory_sha256=_sha256(
            row["input_record_inventory_sha256"],
            label="input record inventory",
        ),
        input_record_count=_integer(
            row["input_record_count"],
            label="input record count",
        ),
        bootstrap_metadata=BootstrapMetadata.from_dict(row["bootstrap_metadata"]),
        models=tuple(ModelWarningEvidence.from_dict(item) for item in model_rows),
        holm=tuple(HolmEvidence.from_dict(item) for item in holm_rows),
        evidence_sha256=evidence_sha256,
    )


def write_warning_evidence(path: Path, evidence: WarningEvidence) -> str:
    """Atomically publish canonical evidence without replacing conflicts."""

    if not isinstance(evidence, WarningEvidence):
        raise Task3ContractError("write requires WarningEvidence")
    expected = evidence.canonical_json_bytes()
    expected_sha256 = hashlib.sha256(expected).hexdigest()
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise Task3ContractError(
            f"cannot create WarningEvidence directory: {path.parent}"
        ) from exc
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    temporary_created = False
    try:
        with temporary.open("xb") as handle:
            temporary_created = True
            handle.write(expected)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError:
            existing = path.read_bytes()
            if existing != expected:
                raise Task3ContractError(
                    f"existing WarningEvidence conflicts with payload: {path}"
                )
            return hashlib.sha256(existing).hexdigest()
        except OSError as exc:
            raise Task3ContractError(
                f"cannot atomically publish WarningEvidence: {path}"
            ) from exc
        return expected_sha256
    except Task3ContractError:
        raise
    except OSError as exc:
        raise Task3ContractError(
            f"cannot prepare WarningEvidence publication: {path}"
        ) from exc
    finally:
        if temporary_created:
            try:
                temporary.unlink(missing_ok=True)
            except OSError as exc:
                raise Task3ContractError(
                    f"cannot clean WarningEvidence temporary file: {temporary}"
                ) from exc


def read_warning_evidence(path: Path) -> WarningEvidence:
    try:
        payload = path.read_bytes()
    except OSError as exc:
        raise Task3ContractError(f"cannot read WarningEvidence: {path}") from exc
    return parse_warning_evidence(payload)
