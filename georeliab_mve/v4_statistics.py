"""Deterministic scene-level statistics for GeoReliab v4 Task 3.

Only strict v4 TaskAuditRecord objects are admitted.  Every primary confidence
interval uses the same frozen 10,000-draw scene-block bootstrap; state rows and
repeated numerical runs are never promoted to independent sampling units.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from functools import lru_cache
import math
from statistics import median
from typing import TypeVar

import numpy as np

from .v4_counterfactuals import (
    FOG_BOUNDARY_LAG_SEQUENCE,
    LIGHTING_STATES,
    SCIENTIFIC_MODELS,
    SCIENTIFIC_STATES,
    TEST_SCENE_IDS,
    CounterfactualContractError,
    ModelIndependentState,
    ScientificSchedule,
    V4SplitAssignment,
    build_scientific_schedule,
    canonical_json_sha256,
    validate_scientific_schedule,
)
from .v4_metrics import (
    NativeWarningCalibration,
    V4MetricError,
    boundary_lag_for_scene,
    counterfactual_rank_response,
    linear_quantile,
    normalized_aurc,
    relative_warning_gap,
    validate_native_warning_calibration_inventory,
)
from .v4_records import (
    PRIMARY_BOOTSTRAP_DRAW_GROUP,
    PRIMARY_LIGHTING_ORIGIN,
    WARNING_BOOTSTRAP_RESAMPLES,
    WARNING_BOOTSTRAP_SEED,
    WARNING_HYPOTHESES,
    BootstrapMetadata,
    HolmEvidence,
    MetricEstimate,
    ModelWarningEvidence,
    Task3ContractError,
    TaskAuditRecord,
    WarningEvidence,
    build_warning_evidence_record,
)


V4_BOOTSTRAP_RESAMPLES = WARNING_BOOTSTRAP_RESAMPLES
V4_BOOTSTRAP_SEED = WARNING_BOOTSTRAP_SEED
FAMILY_HYPOTHESES = WARNING_HYPOTHESES
LIGHTING_COUNTERFACTUAL_STATES = tuple(
    state_id for state_id in LIGHTING_STATES if state_id != "L3"
)
EXPECTED_TASK_RECORD_COUNT = (
    len(SCIENTIFIC_MODELS) * len(TEST_SCENE_IDS) * len(SCIENTIFIC_STATES)
)

_SceneKey = TypeVar("_SceneKey")


@dataclass(frozen=True, slots=True)
class SceneBootstrapResult:
    """A scalar estimate and all frozen scene-block bootstrap draws."""

    estimate: MetricEstimate
    metadata: BootstrapMetadata
    draws: tuple[float, ...]

    def __post_init__(self) -> None:
        if len(self.draws) != V4_BOOTSTRAP_RESAMPLES:
            raise Task3ContractError("scene bootstrap must retain exactly 10000 draws")
        if any(not math.isfinite(value) for value in self.draws):
            raise Task3ContractError("scene bootstrap draws must be finite")


@lru_cache(maxsize=32)
def _scene_draw_indices(scene_count: int) -> np.ndarray:
    if (
        isinstance(scene_count, bool)
        or not isinstance(scene_count, int)
        or scene_count <= 0
    ):
        raise Task3ContractError("scene bootstrap requires at least one scene")
    generator = np.random.default_rng(V4_BOOTSTRAP_SEED)
    indices = generator.integers(
        0,
        scene_count,
        size=(V4_BOOTSTRAP_RESAMPLES, scene_count),
        dtype=np.int64,
    )
    indices.setflags(write=False)
    return indices


def _metric_estimate(
    point_estimate: float,
    draws: Sequence[float],
    *,
    n_scenes: int,
    defined: bool = True,
    reason_code: str = "DEFINED",
) -> MetricEstimate:
    draw_values = tuple(float(value) for value in draws)
    if len(draw_values) != V4_BOOTSTRAP_RESAMPLES:
        raise Task3ContractError(
            "metric confidence interval requires exactly 10000 draws"
        )
    if not math.isfinite(float(point_estimate)) or any(
        not math.isfinite(value) for value in draw_values
    ):
        raise Task3ContractError("metric estimate and draws must be finite")
    return MetricEstimate(
        point_estimate=float(point_estimate),
        ci_lower=linear_quantile(draw_values, 0.025),
        ci_upper=linear_quantile(draw_values, 0.975),
        n_scenes=n_scenes,
        defined=defined,
        reason_code=reason_code,
        bootstrap_draw_group=PRIMARY_BOOTSTRAP_DRAW_GROUP,
    )


def _undefined_estimate(
    *,
    reason_code: str,
    n_scenes: int = 0,
) -> MetricEstimate:
    return MetricEstimate(
        point_estimate=0.0,
        ci_lower=0.0,
        ci_upper=0.0,
        n_scenes=n_scenes,
        defined=False,
        reason_code=reason_code,
        bootstrap_draw_group=PRIMARY_BOOTSTRAP_DRAW_GROUP,
    )


def scene_block_bootstrap(
    blocks: Mapping[_SceneKey, Sequence[float]],
    statistic: Callable[[Sequence[float]], float],
) -> SceneBootstrapResult:
    """Bootstrap whole scene blocks with the exact frozen draw count/seed."""

    if not isinstance(blocks, Mapping) or not blocks:
        raise Task3ContractError("scene bootstrap blocks must be a non-empty mapping")
    try:
        ordered_keys = tuple(sorted(blocks, key=lambda item: repr(item)))
    except TypeError as exc:
        raise Task3ContractError(
            "scene bootstrap keys are not deterministically sortable"
        ) from exc
    normalized: list[tuple[float, ...]] = []
    for key in ordered_keys:
        values = tuple(float(value) for value in blocks[key])
        if not values or any(not math.isfinite(value) for value in values):
            raise Task3ContractError(
                f"scene bootstrap block {key!r} must be finite and non-empty"
            )
        normalized.append(values)

    def evaluate(selected: Sequence[int]) -> float:
        flattened = tuple(value for index in selected for value in normalized[index])
        try:
            result = float(statistic(flattened))
        except (TypeError, ValueError, ZeroDivisionError) as exc:
            raise Task3ContractError("scene bootstrap statistic failed") from exc
        if not math.isfinite(result):
            raise Task3ContractError("scene bootstrap statistic must be finite")
        return result

    point = evaluate(tuple(range(len(normalized))))
    draw_indices = _scene_draw_indices(len(normalized))
    draws = tuple(evaluate(row) for row in draw_indices)
    estimate = _metric_estimate(
        point,
        draws,
        n_scenes=len(normalized),
    )
    return SceneBootstrapResult(
        estimate=estimate,
        metadata=BootstrapMetadata.frozen(),
        draws=draws,
    )


def holm_adjust_four(
    reverse_effect_draw_counts: Mapping[str, int],
) -> dict[str, HolmEvidence]:
    """Apply frozen Holm step-down adjustment to the exact 2x2 family."""

    if (
        set(reverse_effect_draw_counts) != set(FAMILY_HYPOTHESES)
        or len(reverse_effect_draw_counts) != 4
    ):
        raise Task3ContractError(
            "Holm adjustment requires the exact four model-family hypotheses"
        )
    normalized: dict[str, float] = {}
    counts: dict[str, int] = {}
    for hypothesis_id in FAMILY_HYPOTHESES:
        value = reverse_effect_draw_counts[hypothesis_id]
        if (
            isinstance(value, bool)
            or not isinstance(value, int)
            or not 0 <= value <= V4_BOOTSTRAP_RESAMPLES
        ):
            raise Task3ContractError(
                "Holm reverse-effect draw counts must be in 0..10000"
            )
        counts[hypothesis_id] = value
        normalized[hypothesis_id] = (value + 1) / (V4_BOOTSTRAP_RESAMPLES + 1)
    ordered = sorted(
        normalized.items(),
        key=lambda item: (item[1], FAMILY_HYPOTHESES.index(item[0])),
    )
    adjusted_by_id: dict[str, float] = {}
    running_max = 0.0
    family_count = len(ordered)
    for rank, (hypothesis_id, raw_p) in enumerate(ordered):
        candidate = min(1.0, (family_count - rank) * raw_p)
        running_max = max(running_max, candidate)
        adjusted_by_id[hypothesis_id] = running_max
    result: dict[str, HolmEvidence] = {}
    for hypothesis_id in FAMILY_HYPOTHESES:
        model_id, family = hypothesis_id.split(":", maxsplit=1)
        raw_p = normalized[hypothesis_id]
        adjusted_p = adjusted_by_id[hypothesis_id]
        result[hypothesis_id] = HolmEvidence(
            hypothesis_id=hypothesis_id,
            model_id=model_id,
            family=family,
            gap_metric=("RWG_POSE" if family == "ranking-warning" else "TTG_POSE"),
            null_margin=-0.10,
            reverse_effect_draw_count=counts[hypothesis_id],
            bootstrap_draw_count=V4_BOOTSTRAP_RESAMPLES,
            raw_p=raw_p,
            adjusted_p=adjusted_p,
            alpha=0.05,
            non_reversal_excluded=adjusted_p <= 0.05,
            raw_reason_code=(
                "RAW_NON_REVERSAL_EXCLUDED"
                if raw_p <= 0.05
                else "RAW_NON_REVERSAL_NOT_EXCLUDED"
            ),
            adjusted_reason_code=(
                "HOLM_NON_REVERSAL_EXCLUDED"
                if adjusted_p <= 0.05
                else "HOLM_NON_REVERSAL_NOT_EXCLUDED"
            ),
        )
    return result


def _record_key(record: TaskAuditRecord) -> tuple[str, int, str]:
    return (record.model_id, record.scene_id, record.state_id)


def _record_sort_key(record: TaskAuditRecord) -> tuple[int, int, int]:
    return (
        SCIENTIFIC_MODELS.index(record.model_id),
        TEST_SCENE_IDS.index(record.scene_id),
        SCIENTIFIC_STATES.index(record.state_id),
    )


def _validated_model_independent_states(
    states: Sequence[ModelIndependentState] | None,
    scientific_schedule: ScientificSchedule,
) -> dict[tuple[int, str], ModelIndependentState]:
    if states is None:
        raise Task3ContractError(
            "Task 3 requires the exact Task 2 model-independent state inventory"
        )
    rows = tuple(states)
    if any(not isinstance(state, ModelIndependentState) for state in rows):
        raise Task3ContractError(
            "Task 3 state inventory admits only strict ModelIndependentState objects"
        )
    try:
        rebuilt_schedule = build_scientific_schedule(rows)
    except CounterfactualContractError as exc:
        raise Task3ContractError(
            "Task 2 model-independent state inventory is invalid"
        ) from exc
    if rebuilt_schedule.schedule_sha256 != scientific_schedule.schedule_sha256:
        raise Task3ContractError(
            "Task 2 model-independent state inventory does not match the "
            "ScientificSchedule"
        )
    return {(state.scene_id, state.state_id): state for state in rows}


def _validated_records(
    records: Sequence[TaskAuditRecord],
    scientific_schedule: ScientificSchedule,
    model_independent_states: Mapping[tuple[int, str], ModelIndependentState],
    calibrations: Mapping[str, NativeWarningCalibration],
) -> tuple[TaskAuditRecord, ...]:
    try:
        schedule = validate_scientific_schedule(scientific_schedule)
    except CounterfactualContractError as exc:
        raise Task3ContractError(
            "Task 3 requires a validated Task 2 ScientificSchedule"
        ) from exc
    rows = tuple(records)
    if any(not isinstance(record, TaskAuditRecord) for record in rows):
        raise Task3ContractError(
            "WarningEvidence admits only strict v4 TaskAuditRecord objects"
        )
    seen: set[tuple[str, int, str]] = set()
    for record in rows:
        key = _record_key(record)
        if key in seen:
            raise Task3ContractError(
                "duplicate or repeat task record cannot increase sample size"
            )
        seen.add(key)
    if len(rows) != EXPECTED_TASK_RECORD_COUNT:
        raise Task3ContractError(
            "WarningEvidence requires exactly 400 unique task records"
        )
    expected = {
        (model_id, scene_id, state_id)
        for model_id in SCIENTIFIC_MODELS
        for scene_id in TEST_SCENE_IDS
        for state_id in SCIENTIFIC_STATES
    }
    if seen != expected:
        raise Task3ContractError(
            "task-record inventory does not equal the frozen 400-unit schedule"
        )
    ordered = tuple(sorted(rows, key=_record_sort_key))
    schedule_by_key = {
        (unit.model_id, unit.scene_id, unit.state_id): unit for unit in schedule.units
    }
    for record in ordered:
        unit = schedule_by_key[_record_key(record)]
        if (
            record.execution_unit_sha256 != unit.execution_unit_sha256
            or record.state_identity_sha256 != unit.state_identity_sha256
            or record.pair_identity_sha256 != unit.pair_identity_sha256
        ):
            raise Task3ContractError(
                "task record identity does not match the exact Task 2 "
                "ScientificSchedule unit"
            )
        state = model_independent_states[(record.scene_id, record.state_id)]
        if record.ordered_view_ids != state.ordered_view_ids:
            raise Task3ContractError(
                "task record ordered view identity does not match the exact "
                "Task 2 model-independent state"
            )
        calibration = calibrations[record.model_id]
        if record.calibration_identifier != calibration.calibration_identifier:
            raise Task3ContractError(
                "task record calibration identifier does not match the "
                "authoritative model calibration"
            )
        if record.alarm_threshold != calibration.alarm_threshold:
            raise Task3ContractError(
                "task record calibration threshold does not match the "
                "authoritative model calibration"
            )
        if record.alarm is not calibration.alarm_for(record.native_warning_score):
            raise Task3ContractError(
                "task record alarm does not match the authoritative model calibration"
            )

    for scene_id in TEST_SCENE_IDS:
        for state_id in SCIENTIFIC_STATES:
            paired = tuple(
                record
                for record in ordered
                if record.scene_id == scene_id and record.state_id == state_id
            )
            state_identities = {record.state_identity_sha256 for record in paired}
            pair_identities = {record.pair_identity_sha256 for record in paired}
            ordered_views = {record.ordered_view_ids for record in paired}
            if (
                len(state_identities) != 1
                or len(pair_identities) != 1
                or len(ordered_views) != 1
            ):
                raise Task3ContractError(
                    "cross-model records do not share Task 2 scene identity"
                )
    return ordered


def _estimate_with_indices(
    point: float,
    draws: np.ndarray,
    *,
    n_scenes: int,
    defined: bool = True,
    reason_code: str = "DEFINED",
) -> MetricEstimate:
    return _metric_estimate(
        point,
        draws.tolist(),
        n_scenes=n_scenes,
        defined=defined,
        reason_code=reason_code,
    )


def _rank_draws(
    warning_by_scene: np.ndarray,
    loss_by_scene: np.ndarray,
    draw_indices: np.ndarray,
) -> np.ndarray:
    draws = np.empty(V4_BOOTSTRAP_RESAMPLES, dtype=np.float64)
    for draw_index, selected in enumerate(draw_indices):
        warning = warning_by_scene[selected].reshape(-1)
        losses = loss_by_scene[selected].reshape(-1)
        draws[draw_index] = counterfactual_rank_response(
            warning,
            losses,
        ).value
    return draws


def _naurc_draws(
    warning_by_scene: np.ndarray,
    loss_by_scene: np.ndarray,
    draw_indices: np.ndarray,
) -> np.ndarray:
    draws = np.empty(V4_BOOTSTRAP_RESAMPLES, dtype=np.float64)
    for draw_index, selected in enumerate(draw_indices):
        warning = warning_by_scene[selected].reshape(-1)
        losses = loss_by_scene[selected].reshape(-1)
        try:
            draws[draw_index] = normalized_aurc(warning, losses).naurc
        except V4MetricError as exc:
            raise Task3ContractError(
                "selective-risk denominator failed closed in a scene draw"
            ) from exc
    return draws


@dataclass(frozen=True, slots=True)
class _ModelComputation:
    evidence: ModelWarningEvidence
    rwg_draws: tuple[float, ...]
    ttg_draws: tuple[float, ...]


def _compute_model(
    model_id: str,
    lookup: Mapping[tuple[str, int, str], TaskAuditRecord],
) -> _ModelComputation:
    scene_count = len(TEST_SCENE_IDS)
    draw_indices = _scene_draw_indices(scene_count)

    lighting = tuple(
        tuple(lookup[(model_id, scene_id, state_id)] for state_id in LIGHTING_STATES)
        for scene_id in TEST_SCENE_IDS
    )
    static_by_scene = np.asarray(
        [
            median(record.static_rank for record in scene_rows)
            for scene_rows in lighting
        ],
        dtype=np.float64,
    )
    static_point = float(median(static_by_scene.tolist()))
    static_draws = np.median(static_by_scene[draw_indices], axis=1)
    all_static_defined = all(
        record.static_rank_defined for scene_rows in lighting for record in scene_rows
    )
    static_rank = _estimate_with_indices(
        static_point,
        static_draws,
        n_scenes=scene_count,
        defined=all_static_defined,
        reason_code=(
            "DEFINED" if all_static_defined else "DEGENERATE_STATIC_RANK_RETAINED"
        ),
    )

    delta_warning = np.empty(
        (scene_count, len(LIGHTING_COUNTERFACTUAL_STATES)),
        dtype=np.float64,
    )
    delta_point = np.empty_like(delta_warning)
    delta_pose = np.empty_like(delta_warning)
    for scene_index, scene_id in enumerate(TEST_SCENE_IDS):
        baseline = lookup[(model_id, scene_id, "L3")]
        for state_index, state_id in enumerate(LIGHTING_COUNTERFACTUAL_STATES):
            record = lookup[(model_id, scene_id, state_id)]
            delta_warning[scene_index, state_index] = (
                record.native_warning_score - baseline.native_warning_score
            )
            delta_point[scene_index, state_index] = (
                record.point_main_loss - baseline.point_main_loss
            )
            delta_pose[scene_index, state_index] = (
                record.pose_main_loss - baseline.pose_main_loss
            )
    crr_point_result = counterfactual_rank_response(
        delta_warning.reshape(-1),
        delta_point.reshape(-1),
    )
    crr_pose_result = counterfactual_rank_response(
        delta_warning.reshape(-1),
        delta_pose.reshape(-1),
    )
    crr_point_draws = _rank_draws(
        delta_warning,
        delta_point,
        draw_indices,
    )
    crr_pose_draws = _rank_draws(
        delta_warning,
        delta_pose,
        draw_indices,
    )
    crr_point = _estimate_with_indices(
        crr_point_result.value,
        crr_point_draws,
        n_scenes=scene_count,
        defined=crr_point_result.defined,
        reason_code=crr_point_result.reason_code,
    )
    crr_pose = _estimate_with_indices(
        crr_pose_result.value,
        crr_pose_draws,
        n_scenes=scene_count,
        defined=crr_pose_result.defined,
        reason_code=crr_pose_result.reason_code,
    )
    rwg_point = relative_warning_gap(
        static_point,
        crr_pose_result.value,
    )
    rwg_draws = static_draws - crr_pose_draws
    rwg_pose = _estimate_with_indices(
        rwg_point,
        rwg_draws,
        n_scenes=scene_count,
        defined=crr_pose_result.defined and all_static_defined,
        reason_code=(
            "DEFINED"
            if crr_pose_result.defined and all_static_defined
            else "DEGENERATE_COMPONENT_RETAINED"
        ),
    )

    failure_scene_rows: list[tuple[TaskAuditRecord, ...]] = []
    for scene_rows in lighting:
        failures = tuple(record for record in scene_rows if record.pose_failure)
        if failures:
            failure_scene_rows.append(failures)
    failure_scene_count = len(failure_scene_rows)
    if failure_scene_count:
        silent_by_scene = np.asarray(
            [sum(not record.alarm for record in rows) for rows in failure_scene_rows],
            dtype=np.float64,
        )
        failure_by_scene = np.asarray(
            [len(rows) for rows in failure_scene_rows],
            dtype=np.float64,
        )
        sfr_point = float(np.sum(silent_by_scene) / np.sum(failure_by_scene))
        failure_draw_indices = _scene_draw_indices(failure_scene_count)
        sfr_draws = np.sum(silent_by_scene[failure_draw_indices], axis=1) / np.sum(
            failure_by_scene[failure_draw_indices], axis=1
        )
        sfr_pose = _estimate_with_indices(
            sfr_point,
            sfr_draws,
            n_scenes=failure_scene_count,
        )
    else:
        sfr_pose = _undefined_estimate(reason_code="NO_POSE_FAILURE_SCENES")

    lags: list[float] = []
    fog_no_failure_count = 0
    for scene_id in TEST_SCENE_IDS:
        fog_rows = tuple(
            lookup[(model_id, scene_id, state_id)]
            for state_id in FOG_BOUNDARY_LAG_SEQUENCE
        )
        result = boundary_lag_for_scene(
            FOG_BOUNDARY_LAG_SEQUENCE,
            alarms=tuple(record.alarm for record in fog_rows),
            pose_failures=tuple(record.pose_failure for record in fog_rows),
        )
        if result.included:
            if result.lag is None:
                raise Task3ContractError("included Boundary Lag cannot be null")
            lags.append(float(result.lag))
        else:
            fog_no_failure_count += 1
    if lags:
        lag_array = np.asarray(lags, dtype=np.float64)
        lag_draw_indices = _scene_draw_indices(len(lags))
        lag_draws = np.median(lag_array[lag_draw_indices], axis=1)
        boundary_lag = _estimate_with_indices(
            float(median(lags)),
            lag_draws,
            n_scenes=len(lags),
        )
    else:
        boundary_lag = _undefined_estimate(reason_code="NO_FOG_FAILURE_SCENES")

    warning_by_scene = np.asarray(
        [
            [record.native_warning_score for record in scene_rows]
            for scene_rows in lighting
        ],
        dtype=np.float64,
    )
    point_loss_by_scene = np.asarray(
        [[record.point_main_loss for record in scene_rows] for scene_rows in lighting],
        dtype=np.float64,
    )
    pose_loss_by_scene = np.asarray(
        [[record.pose_main_loss for record in scene_rows] for scene_rows in lighting],
        dtype=np.float64,
    )
    try:
        point_selective = normalized_aurc(
            warning_by_scene.reshape(-1),
            point_loss_by_scene.reshape(-1),
        )
        pose_selective = normalized_aurc(
            warning_by_scene.reshape(-1),
            pose_loss_by_scene.reshape(-1),
        )
    except V4MetricError as exc:
        raise Task3ContractError("selective-risk denominator failed closed") from exc
    point_naurc_draws = _naurc_draws(
        warning_by_scene,
        point_loss_by_scene,
        draw_indices,
    )
    pose_naurc_draws = _naurc_draws(
        warning_by_scene,
        pose_loss_by_scene,
        draw_indices,
    )
    ttg_point = pose_selective.naurc - point_selective.naurc
    ttg_draws = pose_naurc_draws - point_naurc_draws
    naurc_point = _estimate_with_indices(
        point_selective.naurc,
        point_naurc_draws,
        n_scenes=scene_count,
    )
    naurc_pose = _estimate_with_indices(
        pose_selective.naurc,
        pose_naurc_draws,
        n_scenes=scene_count,
    )
    ttg_pose = _estimate_with_indices(
        ttg_point,
        ttg_draws,
        n_scenes=scene_count,
    )

    calibration_identifier = lighting[0][0].calibration_identifier
    evidence = ModelWarningEvidence.create(
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
        pose_failure_scene_count=failure_scene_count,
        fog_no_failure_scene_count=fog_no_failure_count,
    )
    return _ModelComputation(
        evidence=evidence,
        rwg_draws=tuple(float(value) for value in rwg_draws),
        ttg_draws=tuple(float(value) for value in ttg_draws),
    )


def _bootstrap_reverse_effect_count(draws: Sequence[float]) -> int:
    if len(draws) != V4_BOOTSTRAP_RESAMPLES:
        raise Task3ContractError(
            "non-reversal test requires exactly 10000 bootstrap draws"
        )
    return sum(value <= -0.10 for value in draws)


@lru_cache(maxsize=4)
def _build_warning_evidence_cached(
    ordered: tuple[TaskAuditRecord, ...],
    scientific_schedule_sha256: str,
) -> WarningEvidence:
    lookup = {_record_key(record): record for record in ordered}
    computations = tuple(
        _compute_model(model_id, lookup) for model_id in SCIENTIFIC_MODELS
    )
    reverse_effect_counts: dict[str, int] = {}
    for model_id, computation in zip(
        SCIENTIFIC_MODELS,
        computations,
        strict=True,
    ):
        reverse_effect_counts[f"{model_id}:ranking-warning"] = (
            _bootstrap_reverse_effect_count(computation.rwg_draws)
        )
        reverse_effect_counts[f"{model_id}:task-transfer"] = (
            _bootstrap_reverse_effect_count(computation.ttg_draws)
        )
    holm = holm_adjust_four(reverse_effect_counts)
    inventory_sha256 = canonical_json_sha256(
        {
            "schema_version": ("georeliab-v4-warning-input-record-inventory-1.0"),
            "record_sha256": [record.record_sha256 for record in ordered],
        }
    )
    return build_warning_evidence_record(
        primary_evidence_origin=PRIMARY_LIGHTING_ORIGIN,
        primary_state_ids=LIGHTING_STATES,
        scientific_schedule_sha256=scientific_schedule_sha256,
        input_record_inventory_sha256=inventory_sha256,
        input_record_count=len(ordered),
        bootstrap_metadata=BootstrapMetadata.frozen(),
        models=tuple(computation.evidence for computation in computations),
        holm=tuple(holm[hypothesis_id] for hypothesis_id in FAMILY_HYPOTHESES),
    )


def build_warning_evidence(
    records: Sequence[TaskAuditRecord],
    *,
    scientific_schedule: ScientificSchedule,
    model_independent_states: Sequence[ModelIndependentState] | None = None,
    native_warning_calibrations: Sequence[NativeWarningCalibration] | None = None,
    split_assignment: V4SplitAssignment | None = None,
) -> WarningEvidence:
    """Build the exact v4 primary evidence from the frozen 400-unit schedule."""

    try:
        schedule = validate_scientific_schedule(scientific_schedule)
    except CounterfactualContractError as exc:
        raise Task3ContractError(
            "Task 3 requires a validated Task 2 ScientificSchedule"
        ) from exc
    state_by_key = _validated_model_independent_states(
        model_independent_states,
        schedule,
    )
    try:
        validated_calibrations = validate_native_warning_calibration_inventory(
            native_warning_calibrations,
            split_assignment=split_assignment,
        )
    except V4MetricError as exc:
        raise Task3ContractError(
            f"native-warning calibration binding rejected: {exc}"
        ) from exc
    calibration_by_model = {
        calibration.model_id: calibration for calibration in validated_calibrations
    }
    ordered = _validated_records(
        records,
        schedule,
        state_by_key,
        calibration_by_model,
    )
    return _build_warning_evidence_cached(
        ordered,
        schedule.schedule_sha256,
    )
