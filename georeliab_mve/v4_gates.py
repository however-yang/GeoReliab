"""Frozen scientific decision logic for GeoReliab v4 Task 3.

This is deliberately not a Task 4 execution controller.  It consumes one
strict WarningEvidence record and returns only the frozen scientific outcome
or a specific evidence blocker.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from .v4_counterfactuals import (
    LIGHTING_STATES,
    SCIENTIFIC_MODELS,
    CounterfactualContractError,
    ModelIndependentState,
    ScientificSchedule,
    V4SplitAssignment,
    build_scientific_schedule,
    validate_scientific_schedule,
)
from .v4_metrics import (
    NativeWarningCalibration,
    V4MetricError,
    validate_native_warning_calibration_inventory,
)
from .v4_records import (
    PRIMARY_LIGHTING_ORIGIN,
    ModelWarningEvidence,
    Task3ContractError,
    TaskAuditRecord,
    WarningEvidence,
    parse_warning_evidence,
)
from .v4_statistics import build_warning_evidence


MVE_GO_TO_EXTERNAL_VALIDATION = "MVE_GO_TO_EXTERNAL_VALIDATION"
MVE_SCIENTIFIC_NO_GO = "MVE_SCIENTIFIC_NO_GO"
MVE_BLOCKED_ENDPOINT = "MVE_BLOCKED_ENDPOINT"
MVE_BLOCKED_EVIDENCE_TAMPER = "MVE_BLOCKED_EVIDENCE_TAMPER"
MVE_BLOCKED_V1_EVIDENCE = "MVE_BLOCKED_V1_EVIDENCE"
MVE_BLOCKED_EVIDENCE_CONTRACT = "MVE_BLOCKED_EVIDENCE_CONTRACT"
MVE_BLOCKED_SCHEDULE_BINDING_REQUIRED = "MVE_BLOCKED_SCHEDULE_BINDING_REQUIRED"
MVE_BLOCKED_CALIBRATION_BINDING_REQUIRED = "MVE_BLOCKED_CALIBRATION_BINDING_REQUIRED"
MVE_BLOCKED_SOURCE_RECORD_BINDING_REQUIRED = (
    "MVE_BLOCKED_SOURCE_RECORD_BINDING_REQUIRED"
)


@dataclass(frozen=True, slots=True)
class WarningGateDecision:
    """One closed scientific gate result."""

    status: str
    reason_code: str
    strong_model_id: str | None = None
    strong_family: str | None = None

    def __post_init__(self) -> None:
        allowed_statuses = {
            MVE_GO_TO_EXTERNAL_VALIDATION,
            MVE_SCIENTIFIC_NO_GO,
            MVE_BLOCKED_ENDPOINT,
            MVE_BLOCKED_EVIDENCE_TAMPER,
            MVE_BLOCKED_V1_EVIDENCE,
            MVE_BLOCKED_EVIDENCE_CONTRACT,
            MVE_BLOCKED_SCHEDULE_BINDING_REQUIRED,
            MVE_BLOCKED_CALIBRATION_BINDING_REQUIRED,
            MVE_BLOCKED_SOURCE_RECORD_BINDING_REQUIRED,
        }
        if self.status not in allowed_statuses:
            raise Task3ContractError("warning gate status is unsupported")
        if not isinstance(self.reason_code, str) or not self.reason_code:
            raise Task3ContractError("warning gate reason code must be non-empty")
        if self.status == MVE_GO_TO_EXTERNAL_VALIDATION:
            if (
                self.strong_model_id not in SCIENTIFIC_MODELS
                or self.strong_family not in {"ranking-warning", "task-transfer"}
            ):
                raise Task3ContractError(
                    "GO decision requires a valid strong model and family"
                )
        elif self.strong_model_id is not None or self.strong_family is not None:
            raise Task3ContractError("non-GO decisions cannot claim a strong route")


def _looks_like_v1(value: object) -> bool:
    if isinstance(value, Mapping):
        keys = {key for key in value if isinstance(key, str)}
        legacy_keys = {
            "scientific_validity",
            "bundle_index",
            "p2_evidence",
            "p3_evidence",
            "audit_record",
        }
        if keys & legacy_keys:
            return True
        schema = value.get("schema_version")
        return isinstance(schema, str) and (
            "v1" in schema.lower() or schema.startswith("georeliab-audit")
        )
    if isinstance(value, bytes):
        lowered = value.lower()
        return (
            b"scientific_validity" in lowered
            or b"bundle_index" in lowered
            or b"georeliab-v1" in lowered
        )
    if isinstance(value, str):
        lowered = value.lower()
        return (
            "scientific_validity" in lowered
            or "bundle_index" in lowered
            or "georeliab-v1" in lowered
        )
    return False


def _coerce_evidence(
    value: WarningEvidence | bytes | str | Mapping[str, Any],
) -> WarningEvidence | WarningGateDecision:
    if isinstance(value, WarningEvidence):
        return value
    if _looks_like_v1(value):
        return WarningGateDecision(
            status=MVE_BLOCKED_V1_EVIDENCE,
            reason_code="RAW_V1_OR_PRIOR_P2_P3_EVIDENCE_REJECTED",
        )
    try:
        return parse_warning_evidence(value)
    except Task3ContractError as exc:
        message = str(exc).lower()
        if "digest" in message or "tamper" in message:
            return WarningGateDecision(
                status=MVE_BLOCKED_EVIDENCE_TAMPER,
                reason_code="WARNING_EVIDENCE_DIGEST_MISMATCH",
            )
        return WarningGateDecision(
            status=MVE_BLOCKED_EVIDENCE_CONTRACT,
            reason_code="WARNING_EVIDENCE_CONTRACT_REJECTED",
        )


def _other_model(model_id: str) -> str:
    if model_id == SCIENTIFIC_MODELS[0]:
        return SCIENTIFIC_MODELS[1]
    return SCIENTIFIC_MODELS[0]


def _model_by_id(
    evidence: WarningEvidence,
    model_id: str,
) -> ModelWarningEvidence:
    return next(model for model in evidence.models if model.model_id == model_id)


def _strong_for_family(
    model: ModelWarningEvidence,
    family: str,
) -> bool:
    if family == "ranking-warning":
        return model.ranking_warning_strong
    return model.task_transfer_strong


def _gap_for_family(
    model: ModelWarningEvidence,
    family: str,
) -> float:
    if family == "ranking-warning":
        return model.rwg_pose.point_estimate
    return model.ttg_pose.point_estimate


def _state_inventory_matches_schedule(
    states: Sequence[ModelIndependentState],
    schedule: ScientificSchedule,
) -> bool:
    rows = tuple(states)
    if any(not isinstance(state, ModelIndependentState) for state in rows):
        return False
    try:
        rebuilt_schedule = build_scientific_schedule(rows)
    except CounterfactualContractError:
        return False
    return rebuilt_schedule.schedule_sha256 == schedule.schedule_sha256


def _evaluate_rebuilt_warning_evidence(
    evidence: WarningEvidence,
) -> WarningGateDecision:
    """Apply thresholds only after the public gate rebuilt exact evidence."""

    if not isinstance(evidence, WarningEvidence):
        raise Task3ContractError(
            "threshold evaluation requires strict rebuilt WarningEvidence"
        )
    if evidence.input_record_count != 400:
        return WarningGateDecision(
            status=MVE_BLOCKED_EVIDENCE_CONTRACT,
            reason_code="PRIMARY_RECORD_INVENTORY_NOT_400",
        )
    if any(model.pose_failure_scene_count < 8 for model in evidence.models):
        return WarningGateDecision(
            status=MVE_BLOCKED_ENDPOINT,
            reason_code="FEWER_THAN_8_DISTINCT_POSE_FAILURE_SCENES",
        )
    if (
        evidence.primary_evidence_origin != PRIMARY_LIGHTING_ORIGIN
        or evidence.primary_state_ids != LIGHTING_STATES
    ):
        return WarningGateDecision(
            status=MVE_SCIENTIFIC_NO_GO,
            reason_code="REAL_DTU_LIGHTING_PRIMARY_EVIDENCE_REQUIRED",
        )

    holm_by_id = {item.hypothesis_id: item for item in evidence.holm}
    for strong_model_id in SCIENTIFIC_MODELS:
        strong_model = _model_by_id(evidence, strong_model_id)
        for family in ("ranking-warning", "task-transfer"):
            if not _strong_for_family(strong_model, family):
                continue
            directional_model_id = _other_model(strong_model_id)
            directional_model = _model_by_id(
                evidence,
                directional_model_id,
            )
            if _gap_for_family(directional_model, family) < 0.0:
                continue
            directional_holm = holm_by_id[f"{directional_model_id}:{family}"]
            if not directional_holm.non_reversal_excluded:
                continue
            return WarningGateDecision(
                status=MVE_GO_TO_EXTERNAL_VALIDATION,
                reason_code="ONE_STRONG_ONE_DIRECTIONAL_REAL_DTU",
                strong_model_id=strong_model_id,
                strong_family=family,
            )
    return WarningGateDecision(
        status=MVE_SCIENTIFIC_NO_GO,
        reason_code="FROZEN_WARNING_GATE_NOT_MET",
    )


def evaluate_warning_gate(
    value: WarningEvidence | bytes | str | Mapping[str, Any],
    *,
    scientific_schedule: ScientificSchedule | None = None,
    model_independent_states: Sequence[ModelIndependentState] | None = None,
    native_warning_calibrations: Sequence[NativeWarningCalibration] | None = None,
    split_assignment: V4SplitAssignment | None = None,
    task_records: Sequence[TaskAuditRecord] | None = None,
) -> WarningGateDecision:
    """Apply the real-DTU, one-strong-one-directional frozen gate."""

    coerced = _coerce_evidence(value)
    if isinstance(coerced, WarningGateDecision):
        return coerced
    evidence = coerced

    if scientific_schedule is None:
        return WarningGateDecision(
            status=MVE_BLOCKED_SCHEDULE_BINDING_REQUIRED,
            reason_code="TASK2_SCIENTIFIC_SCHEDULE_REQUIRED",
        )
    try:
        schedule = validate_scientific_schedule(scientific_schedule)
    except CounterfactualContractError:
        return WarningGateDecision(
            status=MVE_BLOCKED_SCHEDULE_BINDING_REQUIRED,
            reason_code="TASK2_SCIENTIFIC_SCHEDULE_INVALID",
        )
    if evidence.scientific_schedule_sha256 != schedule.schedule_sha256:
        return WarningGateDecision(
            status=MVE_BLOCKED_SCHEDULE_BINDING_REQUIRED,
            reason_code="TASK2_SCIENTIFIC_SCHEDULE_MISMATCH",
        )
    if model_independent_states is None:
        return WarningGateDecision(
            status=MVE_BLOCKED_SCHEDULE_BINDING_REQUIRED,
            reason_code="TASK2_MODEL_INDEPENDENT_STATE_INVENTORY_REQUIRED",
        )
    if not _state_inventory_matches_schedule(
        model_independent_states,
        schedule,
    ):
        return WarningGateDecision(
            status=MVE_BLOCKED_SCHEDULE_BINDING_REQUIRED,
            reason_code="TASK2_MODEL_INDEPENDENT_STATE_INVENTORY_MISMATCH",
        )
    if native_warning_calibrations is None or split_assignment is None:
        return WarningGateDecision(
            status=MVE_BLOCKED_CALIBRATION_BINDING_REQUIRED,
            reason_code="TASK3_NATIVE_WARNING_CALIBRATION_REQUIRED",
        )
    try:
        calibrations = validate_native_warning_calibration_inventory(
            native_warning_calibrations,
            split_assignment=split_assignment,
        )
    except V4MetricError:
        return WarningGateDecision(
            status=MVE_BLOCKED_CALIBRATION_BINDING_REQUIRED,
            reason_code="TASK3_NATIVE_WARNING_CALIBRATION_INVALID",
        )
    calibration_by_model = {
        calibration.model_id: calibration for calibration in calibrations
    }
    if any(
        model.calibration_identifier
        != calibration_by_model[model.model_id].calibration_identifier
        for model in evidence.models
    ):
        return WarningGateDecision(
            status=MVE_BLOCKED_CALIBRATION_BINDING_REQUIRED,
            reason_code="TASK3_NATIVE_WARNING_CALIBRATION_MISMATCH",
        )
    if task_records is None:
        return WarningGateDecision(
            status=MVE_BLOCKED_SOURCE_RECORD_BINDING_REQUIRED,
            reason_code="TASK3_SOURCE_TASK_RECORDS_REQUIRED",
        )
    try:
        rebuilt = build_warning_evidence(
            task_records,
            scientific_schedule=schedule,
            model_independent_states=model_independent_states,
            native_warning_calibrations=calibrations,
            split_assignment=split_assignment,
        )
    except Task3ContractError:
        return WarningGateDecision(
            status=MVE_BLOCKED_SOURCE_RECORD_BINDING_REQUIRED,
            reason_code="TASK3_SOURCE_TASK_RECORDS_INVALID",
        )
    if rebuilt != evidence or rebuilt.evidence_sha256 != evidence.evidence_sha256:
        return WarningGateDecision(
            status=MVE_BLOCKED_SOURCE_RECORD_BINDING_REQUIRED,
            reason_code="TASK3_WARNING_EVIDENCE_REBUILD_MISMATCH",
        )
    return _evaluate_rebuilt_warning_evidence(rebuilt)
