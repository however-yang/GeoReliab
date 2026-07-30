"""CPU-only numeric metrics for GeoReliab v4 Task 3.

This module deliberately has no dependency on the v1 audit or gate objects.
It implements the frozen point, relative-pose, and native-warning definitions
from ``.superpowers/v4-ranking-warning-plan.md``.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import math
from statistics import fmean, median
from typing import Any

import numpy as np

from .metrics import MetricError, spearman_correlation
from .v4_counterfactuals import (
    CounterfactualContractError,
    DTU_OFFICIAL_SCENE_SET,
    SCIENTIFIC_MODELS,
    TEST_SCENE_IDS,
    V4SplitAssignment,
    canonical_json_sha256,
    validate_v4_split_assignment,
)
from .v4_science_lock import (
    V4_PROTOCOL_ID,
    V4_PROTOCOL_PROVENANCE_SCHEMA_VERSION,
    V4_PROTOCOL_SHA256,
    V4_PROTOCOL_VERSION,
)


CALIBRATION_SCHEMA_VERSION = "georeliab-v4-native-warning-calibration-1.0"
CALIBRATION_QUANTILE = 0.90
CALIBRATION_CARDINALITY = 20
POINT_THRESHOLDS_MM = (1.0, 2.0, 5.0)
POSE_THRESHOLDS_DEG = (5.0, 10.0, 20.0)
POSE_FAILURE_THRESHOLD_DEG = 10.0
POSE_VIEW_COUNT = 8
POSE_PAIR_COUNT = 28


class V4MetricError(ValueError):
    """Raised when a frozen v4 metric is undefined or violates its contract."""


def _protocol_provenance() -> dict[str, str]:
    return {
        "schema_version": V4_PROTOCOL_PROVENANCE_SCHEMA_VERSION,
        "protocol_id": V4_PROTOCOL_ID,
        "protocol_version": V4_PROTOCOL_VERSION,
        "protocol_sha256": V4_PROTOCOL_SHA256,
    }


def _finite_float(value: object, *, label: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float, np.integer, np.floating))
        or not math.isfinite(float(value))
    ):
        raise V4MetricError(f"{label} must be finite")
    return float(value)


def _sha256(value: object, *, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or value != value.lower()
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise V4MetricError(f"{label} must be a lowercase SHA-256")
    return value


def _points(value: Any, *, label: str, nonempty: bool = True) -> np.ndarray:
    try:
        result = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise V4MetricError(f"{label} must be a numeric point array") from exc
    if result.ndim != 2 or result.shape[1] != 3:
        raise V4MetricError(f"{label} must have shape (N, 3)")
    if nonempty and len(result) == 0:
        raise V4MetricError(f"{label} must not be empty")
    if not np.all(np.isfinite(result)):
        raise V4MetricError(f"{label} must be finite")
    return result


def _vector(
    value: Any,
    *,
    label: str,
    expected_length: int | None = None,
) -> np.ndarray:
    try:
        result = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise V4MetricError(f"{label} must be a numeric vector") from exc
    if result.ndim != 1:
        raise V4MetricError(f"{label} must be one-dimensional")
    if expected_length is not None and len(result) != expected_length:
        raise V4MetricError(
            f"{label} must contain exactly {expected_length} observations"
        )
    if len(result) == 0 or not np.all(np.isfinite(result)):
        raise V4MetricError(f"{label} must contain finite observations")
    return result


def _nearest_errors(points: np.ndarray, targets: np.ndarray) -> np.ndarray:
    """Return Euclidean nearest-neighbour distances without filtering points."""

    if len(points) == 0 or len(targets) == 0:
        raise V4MetricError("nearest-neighbour inputs must not be empty")
    try:
        from scipy.spatial import cKDTree
    except ImportError:
        cKDTree = None
    if cKDTree is not None:
        distances, _ = cKDTree(targets).query(points, k=1)
        result = np.asarray(distances, dtype=np.float64)
        if not np.all(np.isfinite(result)):
            raise V4MetricError("nearest-neighbour distances must be finite")
        return result

    # Dependency-free fallback.  Chunking prevents a full N x M x 3 array.
    chunks: list[np.ndarray] = []
    for start in range(0, len(points), 4096):
        block = points[start : start + 4096]
        squared = np.sum((block[:, None, :] - targets[None, :, :]) ** 2, axis=2)
        chunks.append(np.sqrt(np.min(squared, axis=1)))
    return np.concatenate(chunks)


def _fscore(
    prediction_to_gt: np.ndarray,
    gt_to_prediction: np.ndarray,
    threshold_mm: float,
) -> float:
    precision = float(np.mean(prediction_to_gt <= threshold_mm))
    recall = float(np.mean(gt_to_prediction <= threshold_mm))
    if precision + recall == 0.0:
        return 0.0
    return 2.0 * precision * recall / (precision + recall)


@dataclass(frozen=True, slots=True)
class PointTaskMetrics:
    """Frozen point-task metrics for one fixed model/scene/state."""

    point_main_loss: float
    fscore_1mm: float
    fscore_2mm: float
    fscore_5mm: float
    median_predicted_error_mm: float
    static_rank: float
    static_rank_defined: bool
    static_rank_reason_code: str
    prediction_count: int
    ground_truth_count: int

    def __post_init__(self) -> None:
        values = (
            self.point_main_loss,
            self.fscore_1mm,
            self.fscore_2mm,
            self.fscore_5mm,
            self.median_predicted_error_mm,
            self.static_rank,
        )
        if any(not math.isfinite(float(value)) for value in values):
            raise V4MetricError("point metrics must be finite")
        if not all(
            0.0 <= value <= 1.0
            for value in (
                self.point_main_loss,
                self.fscore_1mm,
                self.fscore_2mm,
                self.fscore_5mm,
            )
        ):
            raise V4MetricError("point losses and F-scores must be in [0, 1]")
        if not math.isclose(
            self.point_main_loss,
            1.0 - self.fscore_2mm,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise V4MetricError("point main loss must equal 1-F-score@2mm")
        if self.median_predicted_error_mm < 0.0:
            raise V4MetricError("median point error must be non-negative")
        if not -1.0 <= self.static_rank <= 1.0:
            raise V4MetricError("StaticRank must be in [-1, 1]")
        if not isinstance(self.static_rank_defined, bool):
            raise V4MetricError("StaticRank defined flag must be boolean")
        expected_reason = (
            "DEFINED"
            if self.static_rank_defined
            else "DEGENERATE_CONSTANT_RISK_OR_ERROR"
        )
        if self.static_rank_reason_code != expected_reason:
            raise V4MetricError("StaticRank diagnostic is inconsistent")
        if self.prediction_count < 1 or self.ground_truth_count < 1:
            raise V4MetricError("valid point metrics require non-empty point sets")


def compute_point_task_metrics(
    predicted_points: Any,
    ground_truth_points: Any,
    native_point_risk: Any,
) -> PointTaskMetrics:
    """Compute bidirectional F-score and within-state point StaticRank.

    Every prediction participates; this API intentionally exposes no
    confidence threshold or accepted-mask argument.
    """

    predicted = _points(predicted_points, label="predicted_points")
    ground_truth = _points(ground_truth_points, label="ground_truth_points")
    risk = _vector(
        native_point_risk,
        label="native_point_risk",
        expected_length=len(predicted),
    )
    prediction_to_gt = _nearest_errors(predicted, ground_truth)
    gt_to_prediction = _nearest_errors(ground_truth, predicted)
    fscores = {
        threshold: _fscore(prediction_to_gt, gt_to_prediction, threshold)
        for threshold in POINT_THRESHOLDS_MM
    }
    static_rank_defined = True
    static_rank_reason = "DEFINED"
    try:
        static_rank = spearman_correlation(
            prediction_to_gt.tolist(),
            risk.tolist(),
        )
    except MetricError as exc:
        if "constant input" not in str(exc):
            raise V4MetricError(f"StaticRank is undefined: {exc}") from exc
        static_rank = 0.0
        static_rank_defined = False
        static_rank_reason = "DEGENERATE_CONSTANT_RISK_OR_ERROR"
    fscore_2mm = fscores[2.0]
    return PointTaskMetrics(
        point_main_loss=1.0 - fscore_2mm,
        fscore_1mm=fscores[1.0],
        fscore_2mm=fscore_2mm,
        fscore_5mm=fscores[5.0],
        median_predicted_error_mm=float(np.median(prediction_to_gt)),
        static_rank=static_rank,
        static_rank_defined=static_rank_defined,
        static_rank_reason_code=static_rank_reason,
        prediction_count=len(predicted),
        ground_truth_count=len(ground_truth),
    )


def linear_quantile(values: Sequence[float], probability: float) -> float:
    """Return the deterministic type-7/NumPy-linear empirical quantile."""

    if not 0.0 <= probability <= 1.0 or not math.isfinite(probability):
        raise V4MetricError("quantile probability must be in [0, 1]")
    observations = [
        _finite_float(value, label="quantile observation") for value in values
    ]
    if not observations:
        raise V4MetricError("quantile requires at least one observation")
    observations.sort()
    position = (len(observations) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return observations[lower]
    weight = position - lower
    return observations[lower] * (1.0 - weight) + observations[upper] * weight


def native_warning_score(
    native_risk_by_view: Mapping[int, Sequence[float]],
    *,
    ordered_view_ids: Sequence[int],
) -> float:
    """Compute median-over-eight-views of per-view linear Q90 point risk."""

    ordered = tuple(ordered_view_ids)
    if (
        len(ordered) != POSE_VIEW_COUNT
        or len(set(ordered)) != POSE_VIEW_COUNT
        or not isinstance(native_risk_by_view, Mapping)
        or set(native_risk_by_view) != set(ordered)
    ):
        raise V4MetricError(
            "native warning requires the exact ordered eight-view identity"
        )
    quantiles = []
    for view_id in ordered:
        risks = native_risk_by_view[view_id]
        if isinstance(risks, (str, bytes, bytearray)):
            raise V4MetricError(f"view {view_id} risk must be numeric")
        quantiles.append(linear_quantile(tuple(risks), CALIBRATION_QUANTILE))
    result = float(median(quantiles))
    if not math.isfinite(result):
        raise V4MetricError("native warning score must be finite")
    return result


@dataclass(frozen=True, slots=True)
class CalibrationWarningSample:
    """One calibration-L3 scene warning score before fitting."""

    model_id: str
    scene_id: int
    state_id: str
    warning_score: float
    split_fingerprint_sha256: str
    inventory_sha256: str

    def __post_init__(self) -> None:
        if self.model_id not in SCIENTIFIC_MODELS:
            raise V4MetricError("calibration model must be VGGT or MASt3R")
        if (
            isinstance(self.scene_id, bool)
            or not isinstance(self.scene_id, int)
            or self.scene_id not in DTU_OFFICIAL_SCENE_SET
        ):
            raise V4MetricError("calibration scene must be an official DTU scene")
        if not isinstance(self.state_id, str):
            raise V4MetricError("calibration state_id must be a string")
        _finite_float(self.warning_score, label="calibration warning score")
        _sha256(
            self.split_fingerprint_sha256,
            label="calibration split fingerprint",
        )
        _sha256(self.inventory_sha256, label="calibration inventory fingerprint")


def _calibration_payload(
    *,
    model_id: str,
    scene_ids: tuple[int, ...],
    sorted_warning_scores: tuple[float, ...],
    alarm_threshold: float,
    split_fingerprint_sha256: str,
    inventory_sha256: str,
) -> dict[str, object]:
    return {
        "schema_version": CALIBRATION_SCHEMA_VERSION,
        "protocol_provenance": _protocol_provenance(),
        "model_id": model_id,
        "state_id": "L3",
        "scene_ids": list(scene_ids),
        "sorted_warning_scores": list(sorted_warning_scores),
        "quantile_probability": CALIBRATION_QUANTILE,
        "quantile_method": "linear",
        "alarm_threshold": alarm_threshold,
        "split_schema_version": "georeliab-v4-splits-1.0",
        "split_fingerprint_sha256": split_fingerprint_sha256,
        "inventory_sha256": inventory_sha256,
    }


@dataclass(frozen=True, slots=True)
class NativeWarningCalibration:
    """Frozen per-model empirical warning scale and alarm threshold."""

    model_id: str
    scene_ids: tuple[int, ...]
    sorted_warning_scores: tuple[float, ...]
    alarm_threshold: float
    split_fingerprint_sha256: str
    inventory_sha256: str
    calibration_identifier: str
    schema_version: str = CALIBRATION_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != CALIBRATION_SCHEMA_VERSION:
            raise V4MetricError("native-warning calibration schema mismatch")
        if self.model_id not in SCIENTIFIC_MODELS:
            raise V4MetricError("calibration model must be VGGT or MASt3R")
        if len(self.scene_ids) != CALIBRATION_CARDINALITY:
            raise V4MetricError("calibration requires exactly 20 scenes")
        if len(set(self.scene_ids)) != len(self.scene_ids):
            raise V4MetricError("calibration scenes must be unique")
        if any(scene not in DTU_OFFICIAL_SCENE_SET for scene in self.scene_ids):
            raise V4MetricError("calibration scenes must be official DTU scenes")
        overlap = sorted(set(self.scene_ids) & set(TEST_SCENE_IDS))
        if overlap:
            raise V4MetricError(f"calibration/test overlap is forbidden: {overlap}")
        if len(self.sorted_warning_scores) != CALIBRATION_CARDINALITY:
            raise V4MetricError("calibration scale requires exactly 20 scores")
        scores = tuple(
            _finite_float(value, label="calibration warning score")
            for value in self.sorted_warning_scores
        )
        if tuple(sorted(scores)) != scores:
            raise V4MetricError("calibration empirical scale must be sorted")
        threshold = _finite_float(
            self.alarm_threshold,
            label="calibration alarm threshold",
        )
        expected_threshold = linear_quantile(scores, CALIBRATION_QUANTILE)
        if threshold != expected_threshold:
            raise V4MetricError(
                "calibration alarm threshold is not the frozen linear Q90"
            )
        split_fingerprint = _sha256(
            self.split_fingerprint_sha256,
            label="calibration split fingerprint",
        )
        inventory_fingerprint = _sha256(
            self.inventory_sha256,
            label="calibration inventory fingerprint",
        )
        payload = _calibration_payload(
            model_id=self.model_id,
            scene_ids=self.scene_ids,
            sorted_warning_scores=scores,
            alarm_threshold=threshold,
            split_fingerprint_sha256=split_fingerprint,
            inventory_sha256=inventory_fingerprint,
        )
        expected_identifier = canonical_json_sha256(payload)
        if self.calibration_identifier != expected_identifier:
            raise V4MetricError("calibration identifier tamper or mismatch")

    @property
    def protocol_provenance(self) -> dict[str, str]:
        return _protocol_provenance()

    def to_dict(self) -> dict[str, object]:
        payload = _calibration_payload(
            model_id=self.model_id,
            scene_ids=self.scene_ids,
            sorted_warning_scores=self.sorted_warning_scores,
            alarm_threshold=self.alarm_threshold,
            split_fingerprint_sha256=self.split_fingerprint_sha256,
            inventory_sha256=self.inventory_sha256,
        )
        return {
            **payload,
            "calibration_identifier": self.calibration_identifier,
        }

    def alarm_for(self, warning_score: float) -> bool:
        score = _finite_float(warning_score, label="native warning score")
        return score >= self.alarm_threshold


def validate_native_warning_calibration_inventory(
    calibrations: Sequence[NativeWarningCalibration] | None,
    *,
    split_assignment: V4SplitAssignment | None,
) -> tuple[NativeWarningCalibration, ...]:
    """Revalidate the exact two-model calibration inventory against one split."""

    if calibrations is None or split_assignment is None:
        raise V4MetricError(
            "native-warning calibration inventory and split assignment are required"
        )
    if not isinstance(split_assignment, V4SplitAssignment):
        raise V4MetricError("native-warning calibration requires a V4SplitAssignment")
    try:
        validate_v4_split_assignment(split_assignment)
    except CounterfactualContractError as exc:
        raise V4MetricError(f"invalid native-warning calibration split: {exc}") from exc

    rows = tuple(calibrations)
    if len(rows) != len(SCIENTIFIC_MODELS) or any(
        not isinstance(row, NativeWarningCalibration) for row in rows
    ):
        raise V4MetricError(
            "native-warning calibration inventory requires exactly two strict records"
        )

    validated_by_model: dict[str, NativeWarningCalibration] = {}
    for row in rows:
        validated = NativeWarningCalibration(
            model_id=row.model_id,
            scene_ids=tuple(row.scene_ids),
            sorted_warning_scores=tuple(row.sorted_warning_scores),
            alarm_threshold=row.alarm_threshold,
            split_fingerprint_sha256=row.split_fingerprint_sha256,
            inventory_sha256=row.inventory_sha256,
            calibration_identifier=row.calibration_identifier,
            schema_version=row.schema_version,
        )
        if validated.model_id in validated_by_model:
            raise V4MetricError(
                "native-warning calibration inventory contains a duplicate model"
            )
        if validated.scene_ids != split_assignment.calibration:
            raise V4MetricError(
                "native-warning calibration scene set does not match the exact split"
            )
        if (
            validated.split_fingerprint_sha256 != split_assignment.fingerprint_sha256
            or validated.inventory_sha256 != split_assignment.inventory_sha256
        ):
            raise V4MetricError(
                "native-warning calibration does not bind to the exact split inventory"
            )
        validated_by_model[validated.model_id] = validated

    if set(validated_by_model) != set(SCIENTIFIC_MODELS):
        raise V4MetricError(
            "native-warning calibration inventory must contain VGGT and MASt3R"
        )
    return tuple(validated_by_model[model_id] for model_id in SCIENTIFIC_MODELS)


def fit_native_warning_calibration(
    samples: Sequence[CalibrationWarningSample],
    *,
    split_assignment: V4SplitAssignment,
) -> NativeWarningCalibration:
    """Fit one model only on the exact validated calibration split."""

    rows = tuple(samples)
    if len(rows) != CALIBRATION_CARDINALITY:
        raise V4MetricError("calibration requires exactly 20 scene samples")
    if any(not isinstance(row, CalibrationWarningSample) for row in rows):
        raise V4MetricError(
            "calibration inputs must be CalibrationWarningSample values"
        )
    models = {row.model_id for row in rows}
    if len(models) != 1:
        raise V4MetricError("calibration cannot contain mixed models")
    if any(row.state_id != "L3" for row in rows):
        raise V4MetricError("native-warning calibration accepts L3 only")
    scene_ids = tuple(row.scene_id for row in rows)
    if len(set(scene_ids)) != len(scene_ids):
        raise V4MetricError("calibration scenes must be unique")
    overlap = sorted(set(scene_ids) & set(TEST_SCENE_IDS))
    if overlap:
        raise V4MetricError(
            f"calibration/test overlap is forbidden; test scenes={overlap}"
        )
    if not isinstance(split_assignment, V4SplitAssignment):
        raise V4MetricError(
            "calibration requires an explicit validated V4SplitAssignment"
        )
    try:
        validate_v4_split_assignment(split_assignment)
    except CounterfactualContractError as exc:
        raise V4MetricError(f"invalid calibration split: {exc}") from exc
    if set(scene_ids) != set(split_assignment.calibration):
        raise V4MetricError(
            "calibration samples must equal the exact calibration split"
        )
    if any(
        row.split_fingerprint_sha256 != split_assignment.fingerprint_sha256
        or row.inventory_sha256 != split_assignment.inventory_sha256
        for row in rows
    ):
        raise V4MetricError(
            "calibration samples do not bind to the exact calibration split "
            "and inventory fingerprints"
        )
    by_scene = {row.scene_id: row for row in rows}
    scene_ids = split_assignment.calibration
    sorted_scores = tuple(
        sorted(float(by_scene[scene_id].warning_score) for scene_id in scene_ids)
    )
    threshold = linear_quantile(sorted_scores, CALIBRATION_QUANTILE)
    model_id = next(iter(models))
    payload = _calibration_payload(
        model_id=model_id,
        scene_ids=scene_ids,
        sorted_warning_scores=sorted_scores,
        alarm_threshold=threshold,
        split_fingerprint_sha256=split_assignment.fingerprint_sha256,
        inventory_sha256=split_assignment.inventory_sha256,
    )
    return NativeWarningCalibration(
        model_id=model_id,
        scene_ids=scene_ids,
        sorted_warning_scores=sorted_scores,
        alarm_threshold=threshold,
        split_fingerprint_sha256=split_assignment.fingerprint_sha256,
        inventory_sha256=split_assignment.inventory_sha256,
        calibration_identifier=canonical_json_sha256(payload),
    )


def _angle_from_unit_vectors(left: np.ndarray, right: np.ndarray) -> float:
    left_norm = float(np.linalg.norm(left))
    right_norm = float(np.linalg.norm(right))
    if (
        not math.isfinite(left_norm)
        or not math.isfinite(right_norm)
        or left_norm <= 1e-12
        or right_norm <= 1e-12
    ):
        raise V4MetricError("degenerate relative translation direction")
    cosine = float(np.dot(left / left_norm, right / right_norm))
    if cosine >= 1.0 - 1e-12:
        return 0.0
    if cosine <= -1.0 + 1e-12:
        return 180.0
    return math.degrees(math.acos(max(-1.0, min(1.0, cosine))))


def _rotation_geodesic_degrees(
    predicted_relative: np.ndarray,
    ground_truth_relative: np.ndarray,
) -> float:
    delta = predicted_relative @ ground_truth_relative.T
    cosine = (float(np.trace(delta)) - 1.0) / 2.0
    if cosine >= 1.0 - 1e-12:
        return 0.0
    if cosine <= -1.0 + 1e-12:
        return 180.0
    return math.degrees(math.acos(max(-1.0, min(1.0, cosine))))


def _pose_array(value: Any, *, label: str) -> np.ndarray:
    try:
        result = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise V4MetricError(f"{label} must be numeric camera-to-world poses") from exc
    if result.shape not in {
        (POSE_VIEW_COUNT, 4, 4),
        (POSE_VIEW_COUNT, 3, 4),
    }:
        raise V4MetricError(f"{label} must have shape (8, 4, 4) or (8, 3, 4)")
    if not np.all(np.isfinite(result)):
        raise V4MetricError(f"{label} must be finite")
    if result.shape[1:] == (4, 4) and not np.allclose(
        result[:, 3, :],
        np.asarray([0.0, 0.0, 0.0, 1.0]),
        rtol=0.0,
        atol=1e-9,
    ):
        raise V4MetricError(f"{label} homogeneous rows are invalid")
    rotations = result[:, :3, :3]
    identity = np.eye(3)
    for index, rotation in enumerate(rotations):
        determinant = float(np.linalg.det(rotation))
        if determinant <= 0.0:
            raise V4MetricError(
                f"{label}[{index}] is a reflection, not a proper rotation"
            )
        if not np.allclose(
            rotation.T @ rotation,
            identity,
            rtol=0.0,
            atol=1e-6,
        ) or not math.isclose(determinant, 1.0, rel_tol=0.0, abs_tol=1e-6):
            raise V4MetricError(f"{label}[{index}] is not a proper rotation")
    return result


@dataclass(frozen=True, slots=True)
class PosePairMetrics:
    view_a: int
    view_b: int
    rotation_error_deg: float
    translation_direction_error_deg: float
    pair_error_deg: float

    def __post_init__(self) -> None:
        if (
            isinstance(self.view_a, bool)
            or isinstance(self.view_b, bool)
            or not isinstance(self.view_a, int)
            or not isinstance(self.view_b, int)
            or self.view_a == self.view_b
        ):
            raise V4MetricError("pose pair view ids must be distinct integers")
        rotation = _finite_float(
            self.rotation_error_deg,
            label="rotation error",
        )
        translation = _finite_float(
            self.translation_direction_error_deg,
            label="translation-direction error",
        )
        pair = _finite_float(self.pair_error_deg, label="pair error")
        if not all(0.0 <= value <= 180.0 for value in (rotation, translation, pair)):
            raise V4MetricError("pose errors must be in [0, 180] degrees")
        if pair != max(rotation, translation):
            raise V4MetricError(
                "pair error must equal max(rotation, translation-direction)"
            )

    def to_dict(self) -> dict[str, int | float]:
        return {
            "view_a": self.view_a,
            "view_b": self.view_b,
            "rotation_error_deg": self.rotation_error_deg,
            "translation_direction_error_deg": (self.translation_direction_error_deg),
            "pair_error_deg": self.pair_error_deg,
        }

    @classmethod
    def from_dict(cls, value: object) -> "PosePairMetrics":
        expected = {
            "view_a",
            "view_b",
            "rotation_error_deg",
            "translation_direction_error_deg",
            "pair_error_deg",
        }
        if not isinstance(value, Mapping) or set(value) != expected:
            raise V4MetricError("pose pair closed schema violation")
        view_a = value["view_a"]
        view_b = value["view_b"]
        if (
            isinstance(view_a, bool)
            or isinstance(view_b, bool)
            or not isinstance(view_a, int)
            or not isinstance(view_b, int)
        ):
            raise V4MetricError("pose pair view ids must be integers")
        return cls(
            view_a=view_a,
            view_b=view_b,
            rotation_error_deg=_finite_float(
                value["rotation_error_deg"],
                label="rotation error",
            ),
            translation_direction_error_deg=_finite_float(
                value["translation_direction_error_deg"],
                label="translation-direction error",
            ),
            pair_error_deg=_finite_float(
                value["pair_error_deg"],
                label="pair error",
            ),
        )


def empirical_pose_auc(
    pair_errors_deg: Sequence[float],
    threshold_deg: float,
) -> float:
    """Integrate the empirical recall curve and normalize by the threshold."""

    threshold = _finite_float(threshold_deg, label="pose AUC threshold")
    if threshold <= 0.0:
        raise V4MetricError("pose AUC threshold must be positive")
    errors = [
        _finite_float(value, label="pose pair error") for value in pair_errors_deg
    ]
    if not errors or any(value < 0.0 for value in errors):
        raise V4MetricError("pose AUC requires non-negative pair errors")
    result = fmean(max(0.0, 1.0 - error / threshold) for error in errors)
    return max(0.0, min(1.0, result))


@dataclass(frozen=True, slots=True)
class PoseTaskMetrics:
    pairs: tuple[PosePairMetrics, ...]
    auc_5deg: float
    auc_10deg: float
    auc_20deg: float
    pose_main_loss: float
    median_pair_error_deg: float
    pose_failure: bool

    def __post_init__(self) -> None:
        if len(self.pairs) != POSE_PAIR_COUNT or any(
            not isinstance(pair, PosePairMetrics) for pair in self.pairs
        ):
            raise V4MetricError("pose metrics require exactly 28 pair records")
        actual_pairs = [(pair.view_a, pair.view_b) for pair in self.pairs]
        if len(set(actual_pairs)) != POSE_PAIR_COUNT:
            raise V4MetricError("pose metrics contain duplicate view pairs")
        aucs = (self.auc_5deg, self.auc_10deg, self.auc_20deg)
        if any(not math.isfinite(value) or not 0.0 <= value <= 1.0 for value in aucs):
            raise V4MetricError("pose AUC values must be finite in [0, 1]")
        if not self.auc_5deg <= self.auc_10deg <= self.auc_20deg:
            raise V4MetricError("pose AUCs must be monotone by threshold")
        pair_errors = tuple(pair.pair_error_deg for pair in self.pairs)
        expected_aucs = tuple(
            empirical_pose_auc(pair_errors, threshold)
            for threshold in POSE_THRESHOLDS_DEG
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
            raise V4MetricError("pose AUC values do not match the 28 pair errors")
        if not math.isclose(
            self.pose_main_loss,
            1.0 - self.auc_10deg,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise V4MetricError("pose main loss must equal 1-AUC@10deg")
        expected_median = float(median(pair_errors))
        if not math.isclose(
            self.median_pair_error_deg,
            expected_median,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise V4MetricError("median pair error does not match the 28 pairs")
        if self.pose_failure is not (
            self.median_pair_error_deg > POSE_FAILURE_THRESHOLD_DEG
        ):
            raise V4MetricError("pose failure must use strict median>10deg")


def compute_relative_pose_metrics(
    predicted_camera_to_world: Any,
    ground_truth_camera_to_world: Any,
    *,
    ordered_view_ids: Sequence[int],
) -> PoseTaskMetrics:
    """Compute the frozen 28-pair camera-to-world relative-pose metric."""

    ordered = tuple(ordered_view_ids)
    if len(ordered) != POSE_VIEW_COUNT or len(set(ordered)) != POSE_VIEW_COUNT:
        raise V4MetricError("pose metric requires eight distinct ordered views")
    if any(
        isinstance(view_id, bool) or not isinstance(view_id, int) for view_id in ordered
    ):
        raise V4MetricError("ordered view ids must be integers")
    predicted = _pose_array(
        predicted_camera_to_world,
        label="predicted_camera_to_world",
    )
    ground_truth = _pose_array(
        ground_truth_camera_to_world,
        label="ground_truth_camera_to_world",
    )
    predicted_rotations = predicted[:, :3, :3]
    predicted_centers = predicted[:, :3, 3]
    ground_truth_rotations = ground_truth[:, :3, :3]
    ground_truth_centers = ground_truth[:, :3, 3]

    pairs: list[PosePairMetrics] = []
    for first in range(POSE_VIEW_COUNT):
        for second in range(first + 1, POSE_VIEW_COUNT):
            predicted_relative_rotation = (
                predicted_rotations[first].T @ predicted_rotations[second]
            )
            ground_truth_relative_rotation = (
                ground_truth_rotations[first].T @ ground_truth_rotations[second]
            )
            rotation_error = _rotation_geodesic_degrees(
                predicted_relative_rotation,
                ground_truth_relative_rotation,
            )
            predicted_translation_first_camera = predicted_rotations[first].T @ (
                predicted_centers[second] - predicted_centers[first]
            )
            ground_truth_translation_first_camera = ground_truth_rotations[first].T @ (
                ground_truth_centers[second] - ground_truth_centers[first]
            )
            translation_error = _angle_from_unit_vectors(
                predicted_translation_first_camera,
                ground_truth_translation_first_camera,
            )
            pairs.append(
                PosePairMetrics(
                    view_a=ordered[first],
                    view_b=ordered[second],
                    rotation_error_deg=rotation_error,
                    translation_direction_error_deg=translation_error,
                    pair_error_deg=max(rotation_error, translation_error),
                )
            )
    if len(pairs) != POSE_PAIR_COUNT:
        raise V4MetricError("pose metric did not produce all 28 unordered pairs")
    pair_errors = tuple(pair.pair_error_deg for pair in pairs)
    auc_5 = empirical_pose_auc(pair_errors, 5.0)
    auc_10 = empirical_pose_auc(pair_errors, 10.0)
    auc_20 = empirical_pose_auc(pair_errors, 20.0)
    median_error = float(median(pair_errors))
    return PoseTaskMetrics(
        pairs=tuple(pairs),
        auc_5deg=auc_5,
        auc_10deg=auc_10,
        auc_20deg=auc_20,
        pose_main_loss=1.0 - auc_10,
        median_pair_error_deg=median_error,
        pose_failure=median_error > POSE_FAILURE_THRESHOLD_DEG,
    )


@dataclass(frozen=True, slots=True)
class RankCorrelationResult:
    """Spearman result with an explicit scientific degeneracy diagnostic."""

    value: float
    defined: bool
    reason_code: str
    observation_count: int

    def __post_init__(self) -> None:
        _finite_float(self.value, label="rank correlation")
        if not isinstance(self.defined, bool):
            raise V4MetricError("rank correlation defined flag must be boolean")
        if not isinstance(self.reason_code, str) or not self.reason_code:
            raise V4MetricError("rank correlation reason code must be non-empty")
        if (
            isinstance(self.observation_count, bool)
            or not isinstance(self.observation_count, int)
            or self.observation_count < 2
        ):
            raise V4MetricError(
                "rank correlation must contain at least two observations"
            )
        if self.defined and self.reason_code != "DEFINED":
            raise V4MetricError("defined rank correlation must use reason code DEFINED")


def counterfactual_rank_response(
    delta_warning: Sequence[float],
    delta_task_loss: Sequence[float],
) -> RankCorrelationResult:
    """Spearman response over paired changes relative to the L3 baseline.

    A constant warning or loss delta is scientifically meaningful: it is
    retained as rho=0 with a separate degeneracy diagnostic instead of being
    treated as blocked evidence.
    """

    warning = _vector(delta_warning, label="delta warning")
    task_loss = _vector(delta_task_loss, label="delta task loss")
    if len(warning) != len(task_loss) or len(warning) < 2:
        raise V4MetricError("counterfactual rank response requires equal lengths >= 2")
    try:
        value = spearman_correlation(warning, task_loss)
    except MetricError as exc:
        if "constant input" not in str(exc):
            raise V4MetricError("counterfactual rank response is invalid") from exc
        return RankCorrelationResult(
            value=0.0,
            defined=False,
            reason_code="DEGENERATE_CONSTANT_DELTA_WARNING_OR_LOSS",
            observation_count=len(warning),
        )
    return RankCorrelationResult(
        value=float(value),
        defined=True,
        reason_code="DEFINED",
        observation_count=len(warning),
    )


def relative_warning_gap(static_rank: float, crr_pose: float) -> float:
    """Return the frozen ranking-warning gap StaticRank - CRR-pose."""

    return _finite_float(
        _finite_float(static_rank, label="StaticRank")
        - _finite_float(crr_pose, label="CRR-pose"),
        label="relative warning gap",
    )


def silent_failure_rate(
    pose_failures: Sequence[bool],
    alarms: Sequence[bool],
) -> float:
    """Return P(no alarm | pose failure), retaining every failure."""

    failures = tuple(pose_failures)
    alarm_values = tuple(alarms)
    if len(failures) != len(alarm_values) or not failures:
        raise V4MetricError("silent failure rate requires equal non-empty sequences")
    if any(not isinstance(value, bool) for value in (*failures, *alarm_values)):
        raise V4MetricError("silent failure rate requires boolean failures and alarms")
    failure_count = sum(failures)
    if failure_count == 0:
        raise V4MetricError("silent failure rate requires at least one pose failure")
    silent_count = sum(
        failure and not alarm
        for failure, alarm in zip(failures, alarm_values, strict=True)
    )
    return silent_count / failure_count


@dataclass(frozen=True, slots=True)
class BoundaryLagResult:
    """One scene's deterministic fog boundary-lag diagnostic."""

    first_failure_level: int | None
    alarm_level: int | None
    lag: int | None
    classification: str
    included: bool

    def __post_init__(self) -> None:
        if not isinstance(self.included, bool):
            raise V4MetricError("Boundary Lag included flag must be boolean")
        allowed = {
            "EARLY_WARNING",
            "SAME_LEVEL",
            "LATE_WARNING",
            "NEVER_WARNING",
            "NO_FAILURE",
        }
        if self.classification not in allowed:
            raise V4MetricError("Boundary Lag classification is invalid")
        if not self.included:
            if (
                self.classification != "NO_FAILURE"
                or self.first_failure_level is not None
                or self.lag is not None
            ):
                raise V4MetricError("no-failure Boundary Lag must be excluded")
            return
        if self.first_failure_level not in range(4):
            raise V4MetricError("Boundary Lag failure level must be in 0..3")
        if self.alarm_level not in range(5):
            raise V4MetricError("Boundary Lag alarm level must be in 0..4")
        if self.lag != self.alarm_level - self.first_failure_level:
            raise V4MetricError(
                "Boundary Lag must equal alarm level minus failure level"
            )


def boundary_lag_for_scene(
    state_ids: Sequence[str],
    *,
    alarms: Sequence[bool],
    pose_failures: Sequence[bool],
) -> BoundaryLagResult:
    """Evaluate only the frozen L3 -> fog-s1 -> fog-s2 -> fog-s3 sequence."""

    frozen = ("L3", "fog-s1", "fog-s2", "fog-s3")
    states = tuple(state_ids)
    alarm_values = tuple(alarms)
    failures = tuple(pose_failures)
    if states != frozen:
        raise V4MetricError(
            "Boundary Lag requires the frozen fog sequence "
            "L3 -> fog-s1 -> fog-s2 -> fog-s3"
        )
    if len(alarm_values) != 4 or len(failures) != 4:
        raise V4MetricError("Boundary Lag requires four alarm and failure labels")
    if any(not isinstance(value, bool) for value in (*alarm_values, *failures)):
        raise V4MetricError("Boundary Lag labels must be boolean")
    first_failure = next(
        (index for index, failed in enumerate(failures) if failed),
        None,
    )
    first_alarm = next(
        (index for index, alarmed in enumerate(alarm_values) if alarmed),
        4,
    )
    if first_failure is None:
        return BoundaryLagResult(
            first_failure_level=None,
            alarm_level=first_alarm,
            lag=None,
            classification="NO_FAILURE",
            included=False,
        )
    lag = first_alarm - first_failure
    if first_alarm == 4:
        classification = "NEVER_WARNING"
    elif lag < 0:
        classification = "EARLY_WARNING"
    elif lag == 0:
        classification = "SAME_LEVEL"
    else:
        classification = "LATE_WARNING"
    return BoundaryLagResult(
        first_failure_level=first_failure,
        alarm_level=first_alarm,
        lag=lag,
        classification=classification,
        included=True,
    )


@dataclass(frozen=True, slots=True)
class SelectiveRiskResult:
    """Frozen selective-risk summary."""

    signal_aurc: float
    oracle_aurc: float
    random_aurc: float
    naurc: float

    def __post_init__(self) -> None:
        for label, value in (
            ("signal AURC", self.signal_aurc),
            ("oracle AURC", self.oracle_aurc),
            ("random AURC", self.random_aurc),
            ("nAURC", self.naurc),
        ):
            _finite_float(value, label=label)


def _tie_averaged_aurc(
    scores: np.ndarray,
    losses: np.ndarray,
) -> float:
    """Expected prefix AURC under all permutations within score ties."""

    order = np.argsort(scores, kind="stable")
    ordered_scores = scores[order]
    ordered_losses = losses[order]
    prefix_loss = 0.0
    prefix_size = 0
    prefix_risks: list[float] = []
    group_start = 0
    while group_start < len(ordered_scores):
        group_end = group_start + 1
        while (
            group_end < len(ordered_scores)
            and ordered_scores[group_end] == ordered_scores[group_start]
        ):
            group_end += 1
        group_losses = ordered_losses[group_start:group_end]
        group_total = float(np.sum(group_losses))
        group_size = group_end - group_start
        for admitted_in_group in range(1, group_size + 1):
            expected_group_loss = (admitted_in_group / group_size) * group_total
            prefix_risks.append(
                (prefix_loss + expected_group_loss) / (prefix_size + admitted_in_group)
            )
        prefix_loss += group_total
        prefix_size += group_size
        group_start = group_end
    return float(fmean(prefix_risks))


def normalized_aurc(
    warning_scores: Sequence[float],
    task_losses: Sequence[float],
) -> SelectiveRiskResult:
    """Compute tie-averaged signal/oracle/random AURC and normalized AURC."""

    warnings = _vector(warning_scores, label="warning scores")
    losses = _vector(task_losses, label="task losses")
    if len(warnings) != len(losses):
        raise V4MetricError("normalized AURC requires equal warning and loss lengths")
    signal = _tie_averaged_aurc(warnings, losses)
    oracle = _tie_averaged_aurc(losses, losses)
    random = float(fmean(losses))
    denominator = random - oracle
    if not math.isfinite(denominator) or denominator <= 0.0:
        raise V4MetricError("normalized AURC denominator must be finite and positive")
    normalized = (signal - oracle) / denominator
    return SelectiveRiskResult(
        signal_aurc=signal,
        oracle_aurc=oracle,
        random_aurc=random,
        naurc=normalized,
    )
