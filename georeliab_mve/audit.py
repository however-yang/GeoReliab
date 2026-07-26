"""Geometry audit, scene-level statistics, and GeoReliab evidence assembly."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
import random
import hashlib
from statistics import fmean
from typing import Any, Mapping, Sequence

import numpy as np

from .contracts import AuditRecord, PredictionArtifact, RunManifest, RunMode, ScientificValidity, read_json_artifact, validate_artifact_bundle, write_json_artifact
from .gates import (
    DownstreamHarmEvidence,
    GEORELIAB_FAILURE_AUROC_THRESHOLD,
    GEORELIAB_RHO_DECLINE_THRESHOLD,
    GeoReliabConditionEvidence,
    GeoReliabGateInput,
    ZeroUpdateEvidence,
)
from .metrics import MetricError, binary_auroc, spearman_correlation
from .statistics import BootstrapResult, holm_adjust, paired_scene_bootstrap


class AuditError(ValueError):
    """Raised when an audit input violates the frozen scientific protocol."""


@dataclass(frozen=True, slots=True)
class Sim3:
    scale: float
    rotation: np.ndarray
    translation: np.ndarray

    def apply(self, points: np.ndarray) -> np.ndarray:
        values = _points_array(points, 'points')
        return self.scale * (values @ self.rotation.T) + self.translation


@dataclass(frozen=True, slots=True)
class DenseAuditResult:
    voxel_points: np.ndarray
    raw_confidence: np.ndarray
    risk: np.ndarray
    gt_error: np.ndarray
    failure_label: np.ndarray
    labels_1mm: np.ndarray
    labels_5mm: np.ndarray
    provenance_count: np.ndarray
    aligned_camera_centers: np.ndarray
    summary: dict[str, Any]


@dataclass(frozen=True, slots=True)
class SceneAUROCBranch:
    eligible: bool
    effect: float | None
    ci_lower: float | None
    ci_upper: float | None
    n_eligible_scenes: int
    n_resamples: int
    reason_codes: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class HarmResult:
    effect_vs_random: float
    ci_lower: float
    ci_upper: float
    n_scenes: int
    n_random_masks: int
    n_resamples: int

    def to_evidence(self, model: str, condition: str) -> DownstreamHarmEvidence:
        return DownstreamHarmEvidence(
            model=model,
            condition=condition,
            effect_vs_random=self.effect_vs_random,
            ci_upper=self.ci_upper,
        )


@dataclass(frozen=True, slots=True)
class ZeroUpdateResult:
    native_auroc: float
    zero_update_auroc: float
    auroc_gain: float
    ci_lower: float
    ci_upper: float
    n_scenes: int
    n_resamples: int

    def to_evidence(self, model: str, condition: str) -> ZeroUpdateEvidence:
        return ZeroUpdateEvidence(
            model=model,
            condition=condition,
            auroc_gain=self.auroc_gain,
            ci_lower=self.ci_lower,
        )


@dataclass(frozen=True, slots=True)
class DiagnosticCalibration:
    intercept: float
    slope: float
    constant_probability: float | None = None

    def apply(self, risks: Sequence[float]) -> list[float]:
        values = [float(value) for value in risks]
        if self.constant_probability is not None:
            return [self.constant_probability for _ in values]
        return [1.0 / (1.0 + math.exp(-(self.intercept + self.slope * value))) for value in values]


@dataclass(frozen=True, slots=True)
class GeoReliabEvidencePayload:
    scientific_validity: ScientificValidity
    run_mode: RunMode
    split: str
    evidence_schema_version: str
    required_models_ready: tuple[str, ...]
    required_datasets_ready: bool
    tartanair_native_fog_sanity: bool
    conditions: tuple[GeoReliabConditionEvidence, ...]
    downstream_harm: tuple[DownstreamHarmEvidence, ...]
    zero_update: tuple[ZeroUpdateEvidence, ...]
    p5_skip_reason: str | None
    schedule_counts: Mapping[str, int]
    invalid_counts: Mapping[str, int]
    statistics: Mapping[str, Any]

    def to_gate_input(self) -> GeoReliabGateInput:
        return GeoReliabGateInput(
            scientific_validity=self.scientific_validity,
            required_models_ready=self.required_models_ready,
            required_datasets_ready=self.required_datasets_ready,
            tartanair_native_fog_sanity=self.tartanair_native_fog_sanity,
            conditions=self.conditions,
            downstream_harm=self.downstream_harm,
            zero_update=self.zero_update,
            run_mode=self.run_mode,
            evidence_schema_version=self.evidence_schema_version,
            split=self.split,
            schedule_counts=dict(self.schedule_counts),
            downstream_schedule_counts=dict(self.statistics.get('downstream_schedule_counts', {})),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            'scientific_validity': self.scientific_validity.value,
            'run_mode': self.run_mode.value,
            'split': self.split,
            'evidence_schema_version': self.evidence_schema_version,
            'required_models_ready': list(self.required_models_ready),
            'required_datasets_ready': self.required_datasets_ready,
            'tartanair_native_fog_sanity': self.tartanair_native_fog_sanity,
            'conditions': [asdict(item) for item in self.conditions],
            'downstream_harm': [asdict(item) for item in self.downstream_harm],
            'zero_update': [asdict(item) for item in self.zero_update],
            'p5_skip_reason': self.p5_skip_reason,
            'schedule_counts': dict(self.schedule_counts),
            'invalid_counts': dict(self.invalid_counts),
            'statistics': dict(self.statistics),
        }


def _points_array(value: Any, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.ndim != 2 or array.shape[1] != 3 or not np.all(np.isfinite(array)):
        raise AuditError(f'{name} must have shape (N, 3) and be finite')
    return array


def _vector(value: Any, name: str, *, dtype=np.float64) -> np.ndarray:
    array = np.asarray(value, dtype=dtype)
    if array.ndim != 1:
        raise AuditError(f'{name} must be one-dimensional')
    return array


def umeyama_sim3(source: np.ndarray, target: np.ndarray) -> Sim3:
    """Estimate a proper Sim(3); reflection and degenerate layouts fail closed."""

    src = _points_array(source, 'source')
    dst = _points_array(target, 'target')
    if src.shape != dst.shape or src.shape[0] < 3:
        raise AuditError('source and target camera centers must match with >=3 points')
    src_mean = src.mean(axis=0)
    dst_mean = dst.mean(axis=0)
    src_centered = src - src_mean
    dst_centered = dst - dst_mean
    if np.linalg.matrix_rank(src_centered) < 3 or np.linalg.matrix_rank(dst_centered) < 3:
        raise AuditError('degenerate camera layout cannot define a 3D Sim(3)')
    covariance = dst_centered.T @ src_centered / src.shape[0]
    u_mat, singular_values, vt_mat = np.linalg.svd(covariance)
    rotation = u_mat @ vt_mat
    if np.linalg.det(rotation) <= 0:
        raise AuditError('reflection is prohibited in Sim(3) alignment')
    variance = float(np.mean(np.sum(src_centered * src_centered, axis=1)))
    if variance <= 0 or singular_values[-1] <= 0:
        raise AuditError('degenerate camera layout cannot define a 3D Sim(3)')
    scale = float(np.sum(singular_values) / variance)
    if not math.isfinite(scale) or scale <= 0:
        raise AuditError('invalid Sim(3) scale')
    translation = dst_mean - scale * (rotation @ src_mean)
    return Sim3(scale=scale, rotation=rotation, translation=translation)


def _nearest_errors(points: np.ndarray, gt_points: np.ndarray) -> np.ndarray:
    gt = _points_array(gt_points, 'gt_points')
    if len(points) == 0:
        return np.empty((0,), dtype=np.float64)
    if len(gt) == 0:
        raise AuditError('gt_points must not be empty for valid predictions')
    try:
        from scipy.spatial import cKDTree
    except ImportError:
        cKDTree = None
    if cKDTree is not None:
        distances, _ = cKDTree(gt).query(points, k=1)
        return np.asarray(distances, dtype=np.float64)
    chunks: list[np.ndarray] = []
    for start in range(0, len(points), 4096):
        block = points[start : start + 4096]
        squared = np.sum((block[:, None, :] - gt[None, :, :]) ** 2, axis=2)
        chunks.append(np.sqrt(np.min(squared, axis=1)))
    return np.concatenate(chunks)


def _voxel_downsample(
    points: np.ndarray,
    raw_confidence: np.ndarray,
    risk: np.ndarray,
    *,
    voxel_size_mm: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    voxel_size = voxel_size_mm
    if voxel_size <= 0:
        raise AuditError('voxel_size_mm must be positive')
    if len(points) == 0:
        return (
            np.empty((0, 3), dtype=np.float64),
            np.empty((0,), dtype=np.float64),
            np.empty((0,), dtype=np.float64),
            np.empty((0,), dtype=np.int64),
        )
    keys = np.floor(points / voxel_size).astype(np.int64)
    groups: dict[tuple[int, int, int], list[int]] = {}
    for index, key in enumerate(keys):
        groups.setdefault(tuple(int(item) for item in key), []).append(index)
    voxel_points: list[np.ndarray] = []
    voxel_conf: list[float] = []
    voxel_risk: list[float] = []
    counts: list[int] = []
    for key in sorted(groups):
        indices = groups[key]
        voxel_points.append(points[indices].mean(axis=0))
        voxel_conf.append(float(np.median(raw_confidence[indices])))
        voxel_risk.append(float(np.median(risk[indices])))
        counts.append(len(indices))
    return (
        np.vstack(voxel_points),
        np.asarray(voxel_conf, dtype=np.float64),
        np.asarray(voxel_risk, dtype=np.float64),
        np.asarray(counts, dtype=np.int64),
    )


def fscore_at_threshold(
    errors: Sequence[float],
    accepted_mask: Sequence[bool],
    *,
    threshold_mm: float = 2.0,
    gt_to_selected_errors: Sequence[float] | None = None,
) -> float:
    error_values = _vector(errors, 'errors')
    accepted = _vector(accepted_mask, 'accepted_mask', dtype=bool)
    if error_values.shape != accepted.shape:
        raise AuditError('errors and accepted_mask shapes must match')
    if len(error_values) == 0:
        return 0.0
    true_positive = int(np.count_nonzero((error_values <= threshold_mm) & accepted))
    selected = int(np.count_nonzero(accepted))
    if gt_to_selected_errors is None:
        raise AuditError('true reconstruction F-score requires GT-to-selected-prediction errors')
    completeness_errors = _vector(gt_to_selected_errors, 'gt_to_selected_errors')
    if selected == 0 or len(completeness_errors) == 0:
        return 0.0
    precision = true_positive / selected
    recall = int(np.count_nonzero(completeness_errors <= threshold_mm)) / len(completeness_errors)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def gt_to_selected_errors(
    pred_points: Any, gt_points: Any, accepted_mask: Sequence[bool]
) -> np.ndarray:
    points = _points_array(pred_points, 'pred_points')
    gt = _points_array(gt_points, 'gt_points')
    accepted = _vector(accepted_mask, 'accepted_mask', dtype=bool)
    if len(points) != len(accepted):
        raise AuditError('pred_points and accepted_mask shapes must match')
    if not np.any(accepted):
        return np.full((len(gt),), math.inf, dtype=np.float64)
    return _nearest_errors(gt, points[accepted])


def audit_prediction_arrays(
    *,
    points_world: Any,
    pred_camera_centers: Any,
    gt_camera_centers: Any,
    raw_confidence: Any,
    risk: Any,
    valid_mask: Any,
    gt_points: Any,
    observability_mask: Any,
    observability_bb: Any | None = None,
    observability_res: float | None = None,
    voxel_size_mm: float = 0.2,
    invalid_prediction: bool = False,
) -> DenseAuditResult:
    if invalid_prediction:
        return DenseAuditResult(
            voxel_points=np.empty((0, 3), dtype=np.float64),
            raw_confidence=np.empty((0,), dtype=np.float64),
            risk=np.empty((0,), dtype=np.float64),
            gt_error=np.asarray([math.inf]),
            failure_label=np.asarray([True]),
            labels_1mm=np.asarray([True]),
            labels_5mm=np.asarray([True]),
            provenance_count=np.empty((0,), dtype=np.int64),
            aligned_camera_centers=np.empty((0, 3), dtype=np.float64),
            summary={
                'invalid_prediction': True,
                'invalid_failure_count': 1,
                'model_valid_count': 0,
                'observable_count': 0,
                'voxel_count': 0,
                'fscore_2mm': 0.0,
            },
        )
    points = _points_array(points_world, 'points_world')
    pred_centers = _points_array(pred_camera_centers, 'pred_camera_centers')
    gt_centers = _points_array(gt_camera_centers, 'gt_camera_centers')
    confidence = _vector(raw_confidence, 'raw_confidence')
    risk_values = _vector(risk, 'risk')
    valid = _vector(valid_mask, 'valid_mask', dtype=bool)
    raw_observable = np.asarray(observability_mask)
    if not (len(points) == len(confidence) == len(risk_values) == len(valid)):
        raise AuditError('point, confidence, risk, and valid shapes must match')
    if not np.all(np.isfinite(confidence[valid])) or not np.all(np.isfinite(risk_values[valid])):
        raise AuditError('valid model outputs must have finite confidence and risk')
    sim3 = umeyama_sim3(pred_centers, gt_centers)
    aligned = sim3.apply(points)
    if raw_observable.ndim == 3:
        if observability_bb is None or observability_res is None:
            raise AuditError('3-D DTU observability requires BB and Res metadata')
        observable = dtu_observability_mask_for_points(
            aligned,
            obs_mask=raw_observable,
            bb=observability_bb,
            res=observability_res,
        )
    else:
        observable = _vector(raw_observable, 'observability_mask', dtype=bool)
        if len(observable) != len(points):
            raise AuditError('point and observability shapes must match')
    keep = valid & observable
    voxel_points, voxel_conf, voxel_risk, provenance = _voxel_downsample(
        aligned[keep],
        confidence[keep],
        risk_values[keep],
        voxel_size_mm=voxel_size_mm,
    )
    gt_error = _nearest_errors(voxel_points, gt_points)
    gt_to_pred_error = _nearest_errors(_points_array(gt_points, 'gt_points'), voxel_points) if len(voxel_points) else np.full((len(_points_array(gt_points, 'gt_points')),), math.inf)
    failure_2mm = gt_error > 2.0
    fscore = fscore_at_threshold(
        gt_error,
        np.ones(len(gt_error), dtype=bool),
        gt_to_selected_errors=gt_to_pred_error,
    )
    return DenseAuditResult(
        voxel_points=voxel_points,
        raw_confidence=voxel_conf,
        risk=voxel_risk,
        gt_error=gt_error,
        failure_label=failure_2mm,
        labels_1mm=gt_error > 1.0,
        labels_5mm=gt_error > 5.0,
        provenance_count=provenance,
        aligned_camera_centers=sim3.apply(pred_centers),
        summary={
            'invalid_prediction': False,
            'invalid_failure_count': 0,
            'model_valid_count': int(np.count_nonzero(valid)),
            'observable_count': int(np.count_nonzero(keep)),
            'voxel_count': int(len(voxel_points)),
            'fscore_2mm': fscore,
        },
    )


def _scene_rho(record: Mapping[str, Sequence[float]]) -> float:
    try:
        return spearman_correlation(record['risk'], record['error'])
    except (KeyError, MetricError) as exc:
        raise AuditError(f'invalid scene rho record: {exc}') from exc


def _bootstrap_difference(
    baseline: Mapping[str, float],
    treatment: Mapping[str, float],
    *,
    n_resamples: int,
    seed: int,
) -> BootstrapResult:
    return paired_scene_bootstrap(
        baseline,
        treatment,
        n_resamples=n_resamples,
        seed=seed,
    )


def scene_auroc_branch(
    scene_labels_scores: Mapping[str, tuple[Sequence[int | bool], Sequence[float]]],
    *,
    n_resamples: int = 10_000,
    seed: int = 0,
) -> SceneAUROCBranch:
    aurocs: dict[str, float] = {}
    for scene, (labels, scores) in scene_labels_scores.items():
        try:
            aurocs[scene] = binary_auroc(labels, scores)
        except MetricError:
            continue
    if len(aurocs) < 16:
        return SceneAUROCBranch(
            eligible=False,
            effect=None,
            ci_lower=None,
            ci_upper=None,
            n_eligible_scenes=len(aurocs),
            n_resamples=n_resamples,
            reason_codes=('AUROC_ELIGIBLE_SCENES_LT_16',),
        )
    baseline = {scene: 0.65 for scene in aurocs}
    result = paired_scene_bootstrap(
        baseline,
        aurocs,
        n_resamples=n_resamples,
        seed=seed,
    )
    return SceneAUROCBranch(
        eligible=True,
        effect=result.effect + 0.65,
        ci_lower=result.ci_lower + 0.65,
        ci_upper=result.ci_upper + 0.65,
        n_eligible_scenes=len(aurocs),
        n_resamples=n_resamples,
        reason_codes=(),
    )


def build_condition_evidence(
    *,
    model: str,
    corruption: str,
    scene_condition_records: Mapping[str, Mapping[str, Mapping[str, Sequence[float]]]],
    corruption_qa: bool,
    cross_view_consistent: bool,
    gt_geometry_invariant: bool,
    expected_scene_count: int = 20,
    n_resamples: int = 10_000,
    seed: int = 0,
) -> GeoReliabConditionEvidence:
    required = ('clean', '1', '2', '3')
    if any(key not in scene_condition_records for key in required):
        raise AuditError('condition records require clean and severities 1/2/3')
    scene_sets = {condition: set(scene_condition_records[condition]) for condition in required}
    first_scenes = scene_sets['clean']
    if any(scenes != first_scenes for scenes in scene_sets.values()):
        raise AuditError('condition records require identical frozen scene keys')
    if len(first_scenes) != expected_scene_count:
        raise AuditError(f'condition records require exactly {expected_scene_count} scenes')
    rhos = {
        condition: {
            scene: _scene_rho(record)
            for scene, record in scene_condition_records[condition].items()
        }
        for condition in required
    }
    clean_mean = fmean(rhos['clean'].values())
    severity_means = tuple(fmean(rhos[str(index)].values()) for index in (1, 2, 3))
    if clean_mean >= 0.2:
        relative_ci_lower, _, relative_raw_p = _relative_decline_bootstrap_ci(
            rhos['clean'], rhos['3'], n_resamples=n_resamples, seed=seed + 11
        )
    else:
        relative_ci_lower = None
        relative_raw_p = None
    auroc = scene_auroc_branch(
        {
            scene: (record['failure'], record['risk'])
            for scene, record in scene_condition_records['3'].items()
            if 'failure' in record
        },
        n_resamples=n_resamples,
        seed=seed + 23,
    )
    return GeoReliabConditionEvidence(
        model=model,
        corruption=corruption,
        clean_rho=clean_mean,
        severity_rhos=severity_means,
        failure_auroc=auroc.effect if auroc.effect is not None else 1.0,
        corruption_severity_monotonic=corruption_qa,
        cross_view_consistent=cross_view_consistent,
        gt_geometry_invariant=gt_geometry_invariant,
        relative_decline_ci_lower=relative_ci_lower,
        failure_auroc_ci_upper=auroc.ci_upper,
        scene_ids=tuple(sorted(first_scenes)),
        scene_count=len(first_scenes),
        n_resamples=n_resamples,
        relative_decline_raw_p=relative_raw_p,
        failure_auroc_raw_p=None,
    )


def holm_primary_comparisons(p_values: Mapping[str, float]):
    return holm_adjust(p_values)


def _relative_decline_bootstrap_ci(
    clean: Mapping[str, float],
    severity3: Mapping[str, float],
    *,
    n_resamples: int,
    seed: int,
) -> tuple[float, float, float]:
    if set(clean) != set(severity3):
        raise AuditError('relative decline requires paired scene keys')
    scenes = sorted(clean)
    rng = random.Random(seed)
    estimates: list[float] = []
    for _ in range(n_resamples):
        sampled = [scenes[rng.randrange(len(scenes))] for _ in scenes]
        clean_mean = fmean(clean[scene] for scene in sampled)
        s3_mean = fmean(severity3[scene] for scene in sampled)
        estimates.append((clean_mean - s3_mean) / clean_mean if clean_mean >= 0.2 else float('-inf'))
    estimates.sort()
    raw_p = sum(value <= GEORELIAB_RHO_DECLINE_THRESHOLD for value in estimates) / len(estimates)
    return (
        estimates[int(0.025 * (n_resamples - 1))],
        estimates[int(0.975 * (n_resamples - 1))],
        raw_p,
    )


def fit_diagnostic_calibration(
    *, split: str, risks: Sequence[float], failures: Sequence[int | bool]
) -> DiagnosticCalibration:
    if split != 'calibration':
        raise AuditError('ECE mapping must be fitted on calibration-clean only')
    risk_values = _vector(risks, 'risks')
    labels = np.asarray(failures)
    if labels.shape != risk_values.shape:
        raise AuditError('risks and failures shapes must match')
    if len(risk_values) < 2:
        raise AuditError('diagnostic calibration requires at least two samples')
    y = labels.astype(float)
    if np.any((y != 0.0) & (y != 1.0)):
        raise AuditError('failures must be binary for diagnostic calibration')
    if np.all(y == y[0]):
        return DiagnosticCalibration(intercept=0.0, slope=0.0, constant_probability=float(y[0]))
    centered = risk_values - float(np.mean(risk_values))
    scale = float(np.std(centered)) or 1.0
    x = centered / scale
    intercept = math.log(float(np.mean(y)) / (1.0 - float(np.mean(y))))
    slope = 0.0
    for _ in range(400):
        logits = np.clip(intercept + slope * x, -40.0, 40.0)
        pred = 1.0 / (1.0 + np.exp(-logits))
        grad_i = float(np.sum(pred - y))
        grad_s = float(np.sum((pred - y) * x))
        intercept -= 0.05 * grad_i / len(y)
        slope = max(0.0, slope - 0.05 * grad_s / len(y))
    return DiagnosticCalibration(intercept=intercept - slope * float(np.mean(risk_values)) / scale, slope=slope / scale)


def coverage_auc(
    errors: Sequence[float],
    risk: Sequence[float],
    coverages: Sequence[float],
    *,
    gt_to_selected_errors_by_coverage: Mapping[float, Sequence[float]] | None = None,
    pred_points: Any | None = None,
    gt_points: Any | None = None,
) -> float:
    error_values = _vector(errors, 'errors')
    risk_values = _vector(risk, 'risk')
    if error_values.shape != risk_values.shape:
        raise AuditError('errors and risk shapes must match')
    if len(error_values) == 0:
        return 0.0
    order = np.argsort(risk_values, kind='mergesort')
    coverage_values = [float(item) for item in coverages]
    utilities: list[float] = []
    for coverage in coverage_values:
        if not 0 < float(coverage) <= 1:
            raise AuditError('coverages must be in (0, 1]')
        retain = max(1, int(math.ceil(len(order) * float(coverage))))
        accepted = np.zeros(len(order), dtype=bool)
        accepted[order[:retain]] = True
        if pred_points is not None and gt_points is not None:
            completeness_errors = gt_to_selected_errors(pred_points, gt_points, accepted)
        elif gt_to_selected_errors_by_coverage is not None:
            completeness_errors = gt_to_selected_errors_by_coverage[float(coverage)]
        else:
            raise AuditError('coverage-AUC requires GT completeness evidence for each coverage')
        utilities.append(
            fscore_at_threshold(
                error_values,
                accepted,
                gt_to_selected_errors=completeness_errors,
            )
        )
    return _coverage_axis_auc(coverage_values, utilities)


def _coverage_axis_auc(coverages: Sequence[float], utilities: Sequence[float]) -> float:
    pairs = sorted(zip(coverages, utilities, strict=True))
    if len(pairs) == 1:
        return float(pairs[0][1])
    xs = np.asarray([item[0] for item in pairs], dtype=np.float64)
    ys = np.asarray([item[1] for item in pairs], dtype=np.float64)
    width = float(xs[-1] - xs[0])
    if width <= 0:
        raise AuditError('coverage-AUC requires at least two distinct coverages')
    area = float(np.sum((xs[1:] - xs[:-1]) * (ys[1:] + ys[:-1]) / 2.0))
    return area / width


def _random_coverage_auc(
    errors: np.ndarray,
    coverages: Sequence[float],
    pred_points: np.ndarray,
    gt_points: np.ndarray,
    *,
    n_random_masks: int,
    seed: int,
) -> float:
    generator = random.Random(seed)
    per_mask_auc: list[float] = []
    for _ in range(n_random_masks):
        utilities: list[float] = []
        for coverage in coverages:
            retain = max(1, int(math.ceil(len(errors) * float(coverage))))
            selected = set(generator.sample(range(len(errors)), retain))
            accepted = np.fromiter((index in selected for index in range(len(errors))), dtype=bool)
            utilities.append(
                fscore_at_threshold(
                    errors,
                    accepted,
                    gt_to_selected_errors=gt_to_selected_errors(pred_points, gt_points, accepted),
                )
            )
        per_mask_auc.append(_coverage_axis_auc([float(item) for item in coverages], utilities))
    return fmean(per_mask_auc)


def native_vs_random_harm(
    scene_records: Mapping[str, tuple[Sequence[float], Sequence[float], Any, Any]],
    *,
    coverages: Sequence[float] = (0.9, 0.7, 0.5, 0.3),
    n_random_masks: int = 100,
    seed: int = 0,
    n_resamples: int = 10_000,
) -> HarmResult:
    native: dict[str, float] = {}
    random_auc: dict[str, float] = {}
    for scene, (errors, risk, pred_points, gt_points) in scene_records.items():
        error_values = _vector(errors, f'{scene}.errors')
        risk_values = _vector(risk, f'{scene}.risk')
        pred_array = _points_array(pred_points, f'{scene}.pred_points')
        gt_array = _points_array(gt_points, f'{scene}.gt_points')
        if error_values.shape != risk_values.shape:
            raise AuditError(f'{scene} errors/risk shapes must match')
        if len(pred_array) != len(error_values):
            raise AuditError(f'{scene} pred_points must match errors')
        native[scene] = coverage_auc(
            error_values,
            risk_values,
            coverages,
            pred_points=pred_array,
            gt_points=gt_array,
        )
        random_auc[scene] = _random_coverage_auc(
            error_values,
            coverages,
            pred_array,
            gt_array,
            n_random_masks=n_random_masks,
            seed=seed + sum(ord(ch) for ch in scene),
        )
    result = paired_scene_bootstrap(
        random_auc,
        native,
        n_resamples=n_resamples,
        seed=seed,
    )
    return HarmResult(
        effect_vs_random=result.effect,
        ci_lower=result.ci_lower,
        ci_upper=result.ci_upper,
        n_scenes=result.n_scenes,
        n_random_masks=n_random_masks,
        n_resamples=n_resamples,
    )


def align_subset_to_full_prediction(
    subset_points: Any,
    subset_camera_centers: Any,
    full_camera_centers: Any,
    *,
    subset_view_ids: Sequence[int] | None = None,
) -> np.ndarray:
    subset_cameras = _points_array(subset_camera_centers, 'subset_camera_centers')
    full_cameras = _points_array(full_camera_centers, 'full_camera_centers')
    target_cameras = full_cameras
    if subset_view_ids is not None:
        ids = np.asarray(subset_view_ids, dtype=np.int64)
        if ids.ndim != 1 or len(ids) != len(subset_cameras):
            raise AuditError('subset_view_ids must match subset camera count')
        if np.any(ids < 0) or np.any(ids >= len(full_cameras)) or len(set(ids.tolist())) != len(ids):
            raise AuditError('subset_view_ids must be unique valid full-camera indexes')
        target_cameras = full_cameras[ids]
    sim3 = umeyama_sim3(subset_cameras, target_cameras)
    return sim3.apply(subset_points)


def compute_zero_update_disagreement_risk(
    *,
    full_points: Any,
    full_camera_centers: Any,
    subset_predictions: Sequence[Mapping[str, Any]],
) -> np.ndarray:
    full = _points_array(full_points, 'full_points')
    cameras = _points_array(full_camera_centers, 'full_camera_centers')
    if len(subset_predictions) != 4:
        raise AuditError('zero-update requires exactly four 6-of-8 subset predictions')
    expected_omissions = ({0, 4}, {1, 5}, {2, 6}, {3, 7})
    distances: list[np.ndarray] = []
    for index, (subset, omitted) in enumerate(zip(subset_predictions, expected_omissions, strict=True)):
        try:
            view_ids = tuple(int(item) for item in subset['view_ids'])
            if set(range(8)) - set(view_ids) != omitted:
                raise AuditError(f'subset {index} does not match frozen omitted views {sorted(omitted)}')
            aligned = align_subset_to_full_prediction(
                subset['points'],
                subset['camera_centers'],
                cameras,
                subset_view_ids=view_ids,
            )
        except KeyError as exc:
            raise AuditError(f'subset {index} missing {exc.args[0]}') from exc
        distances.append(_nearest_errors(full, aligned))
    median_distance = np.median(np.vstack(distances), axis=0)
    centered = full - np.median(full, axis=0)
    scale = float(np.median(np.linalg.norm(centered, axis=1)))
    if not math.isfinite(scale) or scale <= 0:
        scale = float(np.sqrt(np.mean(np.sum(centered * centered, axis=1))))
    if not math.isfinite(scale) or scale <= 0:
        raise AuditError('robust predicted scene scale is degenerate')
    return median_distance / scale


def evaluate_zero_update_gain(
    scene_records: Mapping[str, Mapping[str, Sequence[float]]],
    *,
    n_resamples: int = 10_000,
    seed: int = 0,
) -> ZeroUpdateResult:
    native: dict[str, float] = {}
    zero: dict[str, float] = {}
    for scene, record in scene_records.items():
        try:
            labels = record['failure']
            native[scene] = binary_auroc(labels, record['native_risk'])
            zero[scene] = binary_auroc(labels, record['zero_update_risk'])
        except (KeyError, MetricError) as exc:
            raise AuditError(f'invalid zero-update scene record {scene}: {exc}') from exc
    result = paired_scene_bootstrap(native, zero, n_resamples=n_resamples, seed=seed)
    return ZeroUpdateResult(
        native_auroc=fmean(native.values()),
        zero_update_auroc=fmean(zero.values()),
        auroc_gain=result.effect,
        ci_lower=result.ci_lower,
        ci_upper=result.ci_upper,
        n_scenes=result.n_scenes,
        n_resamples=n_resamples,
    )


def _condition_from_dict(item: Mapping[str, Any]) -> GeoReliabConditionEvidence:
    if any(key in item and isinstance(item[key], list) for key in ('severity_rhos', 'scene_ids', 'branch_reason_codes')):
        item = dict(item)
        for key in ('severity_rhos', 'scene_ids', 'branch_reason_codes'):
            if key in item and isinstance(item[key], list):
                item[key] = tuple(item[key])
    return GeoReliabConditionEvidence(**dict(item))


def _with_holm_metadata(
    conditions: Sequence[GeoReliabConditionEvidence],
) -> tuple[GeoReliabConditionEvidence, ...]:
    p_values: dict[str, float] = {}
    for item in conditions:
        prefix = f'{item.model}:{item.corruption}'
        if item.relative_decline_raw_p is not None:
            p_values[f'{prefix}:relative_decline'] = item.relative_decline_raw_p
        if item.failure_auroc_raw_p is not None:
            p_values[f'{prefix}:failure_auroc'] = item.failure_auroc_raw_p
    if not p_values:
        return tuple(conditions)
    adjusted = holm_primary_comparisons(p_values)
    patched = []
    for item in conditions:
        values = item.to_dict()
        rel_key = f'{item.model}:{item.corruption}:relative_decline'
        auroc_key = f'{item.model}:{item.corruption}:failure_auroc'
        reasons = list(item.branch_reason_codes)
        if rel_key in adjusted:
            values['relative_decline_adjusted_p'] = adjusted[rel_key].adjusted_p
            values['relative_decline_holm_rejected'] = adjusted[rel_key].rejected
            if adjusted[rel_key].rejected:
                reasons.append('RELATIVE_DECLINE_HOLM_SUPPORTED')
        if auroc_key in adjusted:
            values['failure_auroc_adjusted_p'] = adjusted[auroc_key].adjusted_p
            values['failure_auroc_holm_rejected'] = adjusted[auroc_key].rejected
            if adjusted[auroc_key].rejected:
                reasons.append('FAILURE_AUROC_HOLM_SUPPORTED')
        values['branch_reason_codes'] = tuple(sorted(set(reasons)))
        values['severity_rhos'] = tuple(values['severity_rhos'])
        values['scene_ids'] = tuple(values['scene_ids'])
        patched.append(GeoReliabConditionEvidence(**values))
    return tuple(patched)


def build_georeliab_evidence(
    *,
    condition_evidence: Sequence[GeoReliabConditionEvidence],
    downstream_harm: Sequence[DownstreamHarmEvidence],
    zero_update: Sequence[ZeroUpdateEvidence],
    required_models_ready: Sequence[str],
    required_datasets_ready: bool,
    tartanair_native_fog_sanity: bool,
    run_mode: str | RunMode,
    split: str,
    evidence_schema_version: str = '1.1',
    schedule_counts: Mapping[str, int] | None = None,
    invalid_counts: Mapping[str, int] | None = None,
    statistics: Mapping[str, Any] | None = None,
) -> GeoReliabEvidencePayload:
    mode = RunMode(run_mode)
    if mode is not RunMode.REAL or split != 'test' or evidence_schema_version != '1.1':
        raise AuditError('GeoReliab gate evidence must be schema-v1.1 REAL test evidence')
    condition_evidence = _with_holm_metadata(tuple(condition_evidence))
    provisional = GeoReliabGateInput(
        scientific_validity=ScientificValidity.SCIENTIFIC,
        required_models_ready=tuple(required_models_ready),
        required_datasets_ready=required_datasets_ready,
        tartanair_native_fog_sanity=tartanair_native_fog_sanity,
        conditions=tuple(condition_evidence),
        downstream_harm=(),
        zero_update=(),
        run_mode=mode,
        evidence_schema_version=evidence_schema_version,
        split=split,
        schedule_counts=dict(schedule_counts or {}),
        downstream_schedule_counts={},
    )
    from .gates import evaluate_georeliab_gate

    p4 = evaluate_georeliab_gate(provisional)
    p5_skip_reason = None
    final_harm = tuple(downstream_harm)
    final_zero = tuple(zero_update)
    if 'CONFIDENCE_FAILURE_GATE_NOT_MET' in p4.reason_codes:
        p5_skip_reason = 'P4_CONFIDENCE_PHENOMENON_GATE_FAILED'
        final_harm = ()
        final_zero = ()
    stats = dict(statistics or {})
    if 'primary_p_values' in stats and 'holm_primary' not in stats:
        stats['holm_primary'] = {
            name: result.to_dict()
            for name, result in holm_primary_comparisons(stats['primary_p_values']).items()
        }
    return GeoReliabEvidencePayload(
        scientific_validity=ScientificValidity.SCIENTIFIC,
        run_mode=mode,
        split=split,
        evidence_schema_version=evidence_schema_version,
        required_models_ready=tuple(required_models_ready),
        required_datasets_ready=required_datasets_ready,
        tartanair_native_fog_sanity=tartanair_native_fog_sanity,
        conditions=tuple(condition_evidence),
        downstream_harm=final_harm,
        zero_update=final_zero,
        p5_skip_reason=p5_skip_reason,
        schedule_counts=dict(schedule_counts or {}),
        invalid_counts=dict(invalid_counts or {}),
        statistics=stats,
    )


def load_georeliab_evidence_input(path: Path) -> GeoReliabEvidencePayload:
    payload = json.loads(path.read_text(encoding='utf-8'))
    if not isinstance(payload, dict):
        raise AuditError('audit input must be a JSON object')
    mode = RunMode(payload.get('run_mode'))
    split = payload.get('split')
    schema = payload.get('evidence_schema_version', '1.1')
    conditions = tuple(_condition_from_dict(item) for item in payload.get('conditions', ()))
    harm = tuple(DownstreamHarmEvidence(**item) for item in payload.get('downstream_harm', ()))
    zero = tuple(ZeroUpdateEvidence(**item) for item in payload.get('zero_update', ()))
    return build_georeliab_evidence(
        condition_evidence=conditions,
        downstream_harm=harm,
        zero_update=zero,
        required_models_ready=payload.get('required_models_ready', ('VGGT', 'MASt3R')),
        required_datasets_ready=bool(payload.get('required_datasets_ready', True)),
        tartanair_native_fog_sanity=bool(payload.get('tartanair_native_fog_sanity', True)),
        run_mode=mode,
        split=split,
        evidence_schema_version=schema,
        schedule_counts=payload.get('schedule_counts', {}),
        invalid_counts=payload.get('invalid_counts', {}),
        statistics=payload.get('statistics', {}),
    )


def load_stage_evidence_manifest(path: Path) -> GeoReliabEvidencePayload:
    payload = json.loads(path.read_text(encoding='utf-8'))
    if not isinstance(payload, dict) or payload.get('schema_version') != 'stage-evidence-v1':
        raise AuditError('stage evidence manifest requires schema_version stage-evidence-v1')
    bundle_index = payload.get('bundle_index')
    if not isinstance(bundle_index, list) or not bundle_index:
        raise AuditError('stage evidence manifest requires a non-empty validated bundle_index')
    for item in bundle_index:
        if not isinstance(item, dict) or not item.get('scene_summary_sha256') or not item.get('scene_summary_path'):
            raise AuditError('bundle_index entries must bind scene_summary path and sha256')
        scene_path = Path(item['scene_summary_path'])
        if _file_sha256(scene_path) != item['scene_summary_sha256']:
            raise AuditError('stage evidence scene summary digest mismatch')
    payload = dict(payload)
    payload.setdefault('run_mode', 'real')
    payload.setdefault('split', 'test')
    payload.setdefault('evidence_schema_version', '1.1')
    return load_georeliab_evidence_payload(payload)


def load_georeliab_evidence_payload(payload: Mapping[str, Any]) -> GeoReliabEvidencePayload:
    mode = RunMode(payload.get('run_mode'))
    split = payload.get('split')
    schema = payload.get('evidence_schema_version', '1.1')
    conditions = tuple(_condition_from_dict(item) for item in payload.get('conditions', ()))
    harm = tuple(DownstreamHarmEvidence(**item) for item in payload.get('downstream_harm', ()))
    zero = tuple(ZeroUpdateEvidence(**item) for item in payload.get('zero_update', ()))
    return build_georeliab_evidence(
        condition_evidence=conditions,
        downstream_harm=harm,
        zero_update=zero,
        required_models_ready=payload.get('required_models_ready', ('VGGT', 'MASt3R')),
        required_datasets_ready=bool(payload.get('required_datasets_ready', True)),
        tartanair_native_fog_sanity=bool(payload.get('tartanair_native_fog_sanity', True)),
        run_mode=mode,
        split=split,
        evidence_schema_version=schema,
        schedule_counts=payload.get('schedule_counts', {}),
        invalid_counts=payload.get('invalid_counts', {}),
        statistics={
            **dict(payload.get('statistics', {})),
            'downstream_schedule_counts': dict(payload.get('downstream_schedule_counts', {})),
        },
    )


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def _load_npz_uri(uri: str) -> dict[str, np.ndarray]:
    from urllib.parse import unquote, urlparse

    parsed = urlparse(uri)
    if parsed.scheme != 'file':
        raise AuditError('bundle audit requires local file URI payloads')
    path_text = unquote(parsed.path)
    if len(path_text) >= 3 and path_text[0] == '/' and path_text[2] == ':':
        path_text = path_text[1:]
    with np.load(Path(path_text), allow_pickle=False) as payload:
        return {name: payload[name] for name in payload.files}


def write_dense_audit_bundle(
    *,
    manifest_path: Path,
    prediction_path: Path,
    gt_points_path: Path,
    gt_cameras_path: Path,
    obs_mask_path: Path,
    obs_bb: Sequence[float],
    obs_res: float,
    output_dir: Path,
) -> dict[str, Any]:
    manifest = read_json_artifact(manifest_path, RunManifest)
    prediction = read_json_artifact(prediction_path, PredictionArtifact)
    if not isinstance(manifest, RunManifest) or not isinstance(prediction, PredictionArtifact):
        raise AuditError('invalid manifest or prediction artifact type')
    if manifest.mode is not RunMode.REAL or manifest.split != 'test':
        raise AuditError('bundle audit requires schema-v1.1 REAL test artifacts')
    geometry = _load_npz_uri(prediction.geometry_prediction_uri)
    confidence = _load_npz_uri(prediction.native_confidence_uri)
    mask = _load_npz_uri(prediction.valid_mask_uri)
    camera_centers = np.asarray(geometry['camera_c2w'], dtype=np.float64)[:, :3, 3]
    gt_points = np.load(gt_points_path, allow_pickle=False)
    gt_cameras = np.load(gt_cameras_path, allow_pickle=False)
    obs_mask = np.load(obs_mask_path, allow_pickle=False)
    bb_values = np.asarray(list(obs_bb), dtype=np.float64)
    if bb_values.shape != (6,):
        raise AuditError('obs-bb must provide six comma-separated values')
    if prediction.invalid_prediction:
        result = audit_prediction_arrays(
            points_world=np.empty((0, 3)),
            pred_camera_centers=np.empty((0, 3)),
            gt_camera_centers=np.empty((0, 3)),
            raw_confidence=np.empty((0,)),
            risk=np.empty((0,)),
            valid_mask=np.empty((0,), dtype=bool),
            gt_points=np.empty((0, 3)),
            observability_mask=np.empty((0,), dtype=bool),
            invalid_prediction=True,
        )
    else:
        result = audit_prediction_arrays(
            points_world=geometry['points_world'],
            pred_camera_centers=camera_centers,
            gt_camera_centers=gt_cameras,
            raw_confidence=confidence['raw_confidence'],
            risk=model_risk_from_confidence(manifest.model, confidence['raw_confidence']),
            valid_mask=mask['valid_mask'],
            gt_points=gt_points,
            observability_mask=obs_mask,
            observability_bb=bb_values.reshape(2, 3),
            observability_res=obs_res,
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    dense_path = output_dir / 'dense_audit.npz'
    dense_gt_error = result.gt_error if not prediction.invalid_prediction else np.empty((0,), dtype=np.float64)
    dense_failure = result.failure_label if not prediction.invalid_prediction else np.empty((0,), dtype=bool)
    np.savez(
        dense_path,
        voxel_points=result.voxel_points,
        raw_confidence=result.raw_confidence,
        risk=result.risk,
        gt_error=dense_gt_error,
        failure_label=dense_failure,
        provenance_count=result.provenance_count,
    )
    audit = AuditRecord(
        run_id=manifest.run_id,
        sample_key=prediction.sample_key,
        gt_error=1e12 if prediction.invalid_prediction else float(np.median(result.gt_error)),
        failure_label=bool(np.any(result.failure_label)) if len(result.failure_label) else True,
        selection_score=float(np.median(result.risk)) if len(result.risk) else 1e12,
        coverage=1.0,
        accepted=not prediction.invalid_prediction,
        downstream_outcome=result.summary['fscore_2mm'],
        invalid_prediction=prediction.invalid_prediction,
        metadata={
            'dense_audit_uri': dense_path.resolve().as_uri(),
            'dense_audit_sha256': _file_sha256(dense_path),
        },
    )
    audit_path = output_dir / 'audit_record.json'
    write_json_artifact(audit_path, audit)
    validate_artifact_bundle(manifest, prediction, audit)
    summary = {
        'run_id': manifest.run_id,
        'sample_key': prediction.sample_key,
        'dense_audit_uri': dense_path.resolve().as_uri(),
        'audit_record': str(audit_path),
        'summary': result.summary,
        'invalid_counts': {'invalid_prediction': int(prediction.invalid_prediction)},
        'schedule_counts': {'bundle': 1},
    }
    (output_dir / 'scene_summary.json').write_text(
        json.dumps(summary, indent=2, sort_keys=True) + '\n', encoding='utf-8'
    )
    return summary
def dtu_observability_mask_for_points(
    points: Any,
    *,
    obs_mask: Any,
    bb: Any,
    res: float,
) -> np.ndarray:
    values = _points_array(points, 'points')
    volume = np.asarray(obs_mask, dtype=bool)
    bounds = np.asarray(bb, dtype=np.float64)
    if volume.ndim != 3:
        raise AuditError('DTU ObsMask must be a 3-D volume')
    if bounds.shape != (2, 3):
        raise AuditError('DTU BB must have shape (2, 3)')
    if not math.isfinite(float(res)) or float(res) <= 0:
        raise AuditError('DTU Res must be finite and positive')
    xyz_indices = np.rint((values - bounds[0]) / float(res)).astype(np.int64)
    indices = xyz_indices[:, [1, 0, 2]]
    inside = np.all((indices >= 0) & (indices < np.asarray(volume.shape)), axis=1)
    result = np.zeros(len(values), dtype=bool)
    valid_indices = indices[inside]
    result[inside] = volume[
        valid_indices[:, 0], valid_indices[:, 1], valid_indices[:, 2]
    ]
    return result


def model_risk_from_confidence(model: str, raw_confidence: Any) -> np.ndarray:
    confidence = _vector(raw_confidence, 'raw_confidence')
    if model == 'VGGT':
        return -np.log(np.maximum(confidence - 1.0, 1e-12))
    if model == 'MASt3R':
        return -np.log(np.maximum(confidence, 1e-12))
    raise AuditError(f'unknown model for native confidence risk conversion: {model}')


def load_official_dtu_evidence(
    *,
    sample_key: str,
    frozen_materialization: Path,
    split_manifest: Path,
) -> dict[str, Any]:
    from .contracts import SampleKey

    key = SampleKey.parse(sample_key)
    if key.dataset != 'dtu':
        raise AuditError('official DTU evidence requires dtu sample_key')
    frozen = json.loads(frozen_materialization.read_text(encoding='utf-8'))
    splits = json.loads(split_manifest.read_text(encoding='utf-8'))
    if key.scene not in splits.get(key.split, []):
        raise AuditError('sample scene/split is not present in split manifest')
    try:
        scene_record = frozen['scenes'][key.scene]
        ply_path = Path(scene_record['ply_path'])
        expected = scene_record['ply_sha256']
    except KeyError as exc:
        raise AuditError(f'frozen materialization missing {exc.args[0]}') from exc
    actual = _file_sha256(ply_path)
    if actual != expected:
        raise AuditError('official DTU source digest mismatch')
    return {
        'scene': key.scene,
        'split': key.split,
        'ply_path': str(ply_path),
        'ply_sha256': actual,
    }
