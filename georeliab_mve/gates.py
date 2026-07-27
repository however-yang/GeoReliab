"""Pre-registered scientific gates and final track-selection matrix."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
import math
from typing import Any, Iterable

from .contracts import RunMode, ScientificValidity


GEOMETRY_DELTA_THRESHOLD = 0.05
GEOMETRY_RECOVERY_THRESHOLD = 0.30
GEOMETRY_EQUIVALENCE_MARGIN = 0.02
GEORELIAB_RHO_DECLINE_THRESHOLD = 0.50
GEORELIAB_FAILURE_AUROC_THRESHOLD = 0.65
GEORELIAB_ZERO_UPDATE_GAIN_THRESHOLD = 0.10
GEOMETRY_REQUIRED_BENCHMARKS = frozenset(("VSI-Bench", "CVT-Bench"))
GEOMETRY_MODEL_CANDIDATES = frozenset(
    ("Spatial-MLLM", "SpatialStack", "GUIDE")
)
GEORELIAB_MVE_REQUIRED_MODELS = frozenset(("VGGT", "MASt3R"))
GEOMETRY_REQUIRED_SAMPLE_CLASS = "geometry-required"
GEOMETRY_CONTROL_SAMPLE_CLASS = "semantic-control"
GEOMETRY_SAMPLE_CLASSES = frozenset(
    (GEOMETRY_REQUIRED_SAMPLE_CLASS, GEOMETRY_CONTROL_SAMPLE_CLASS)
)
GEORELIAB_REQUIRED_CORRUPTIONS = frozenset(
    ("fog", "low-light-noise", "defocus")
)


class GateStatus(str, Enum):
    PASS = "PASS"
    FAIL = "FAIL"
    BLOCKED = "BLOCKED"

    @property
    def is_terminal(self) -> bool:
        return self in (GateStatus.PASS, GateStatus.FAIL)


class SelectedTrack(str, Enum):
    GEOMETRY = "GEOMETRY_CAUSAL_AUDIT"
    GEORELIAB = "GEORELIAB"
    STOP = "STOP_AND_RETURN_RESOURCES"
    BLOCKED = "BLOCKED_PENDING_EVIDENCE"
    BLOCKED_PENDING_GEOMETRY = 'BLOCKED_PENDING_GEOMETRY'
    NON_SCIENTIFIC = "BLOCKED_NON_SCIENTIFIC_FIXTURE"
    NON_SCIENTIFIC_SMOKE = 'BLOCKED_NON_SCIENTIFIC_SMOKE'


@dataclass(frozen=True, slots=True)
class GateDecision:
    lane: str
    status: GateStatus
    reason_codes: tuple[str, ...]
    details: dict[str, Any]
    scientific_validity: ScientificValidity

    def to_dict(self) -> dict[str, Any]:
        return {
            "lane": self.lane,
            "status": self.status.value,
            "reason_codes": list(self.reason_codes),
            "details": self.details,
            "scientific_validity": self.scientific_validity.value,
        }


@dataclass(frozen=True, slots=True)
class SelectionDecision:
    selected_track: SelectedTrack
    reason: str
    geometry_status: GateStatus
    georeliab_status: GateStatus
    scientific_validity: ScientificValidity

    def to_dict(self) -> dict[str, str]:
        return {
            "selected_track": self.selected_track.value,
            "reason": self.reason,
            "geometry_status": self.geometry_status.value,
            "georeliab_status": self.georeliab_status.value,
            "scientific_validity": self.scientific_validity.value,
        }


@dataclass(frozen=True, slots=True)
class GeometryEvidence:
    model: str
    benchmark: str
    sample_class: str
    stratum: str
    delta_geom: float
    ci_lower: float
    recovery: float
    equivalent_by_tost: bool
    post_fusion_changed: bool
    semantic_control_unchanged: bool

    def __post_init__(self) -> None:
        if not isinstance(self.model, str) or not self.model:
            raise ValueError('model must be a non-empty string')
        if self.benchmark not in GEOMETRY_REQUIRED_BENCHMARKS:
            raise ValueError(
                "benchmark must be one of "
                f"{sorted(GEOMETRY_REQUIRED_BENCHMARKS)}"
            )
        if self.sample_class not in GEOMETRY_SAMPLE_CLASSES:
            raise ValueError(
                f"sample_class must be one of {sorted(GEOMETRY_SAMPLE_CLASSES)}"
            )
        if not isinstance(self.stratum, str) or not self.stratum:
            raise ValueError('stratum must be a non-empty string')
        for name in ("delta_geom", "ci_lower", "recovery"):
            value = getattr(self, name)
            if isinstance(value, bool) or not math.isfinite(float(value)):
                raise ValueError(f"{name} must be finite")
        for name in (
            'equivalent_by_tost',
            'post_fusion_changed',
            'semantic_control_unchanged',
        ):
            if type(getattr(self, name)) is not bool:
                raise ValueError(f'{name} must be boolean')


@dataclass(frozen=True, slots=True)
class GeometryGateInput:
    scientific_validity: ScientificValidity
    reproducible_checkpoints: tuple[str, ...]
    hookable_models: tuple[str, ...]
    required_datasets_ready: bool
    zeroing_effective: bool
    matched_intervention_effective: bool
    evidence: tuple[GeometryEvidence, ...]
    fixed_inputs_verified: bool
    run_mode: RunMode = RunMode.REAL
    evidence_schema_version: str = '1.1'


def _models_with_geometry_coverage(
    evidence: Iterable[GeometryEvidence],
) -> tuple[set[str], set[str], set[str]]:
    by_model_strata: dict[str, set[str]] = {}
    by_model_benchmarks: dict[str, set[str]] = {}
    all_strata: set[str] = set()
    all_benchmarks: set[str] = set()
    for item in evidence:
        by_model_strata.setdefault(item.model, set()).add(item.stratum)
        by_model_benchmarks.setdefault(item.model, set()).add(item.benchmark)
        all_strata.add(item.stratum)
        all_benchmarks.add(item.benchmark)
    models = {
        model
        for model, strata in by_model_strata.items()
        if len(strata) >= 2
        and GEOMETRY_REQUIRED_BENCHMARKS <= by_model_benchmarks[model]
    }
    return models, all_strata, all_benchmarks


def _models_with_control_coverage(
    evidence: Iterable[GeometryEvidence],
) -> set[str]:
    by_model: dict[str, set[str]] = {}
    for item in evidence:
        if (
            item.sample_class == GEOMETRY_CONTROL_SAMPLE_CLASS
            and item.post_fusion_changed
            and item.semantic_control_unchanged
        ):
            by_model.setdefault(item.model, set()).add(item.benchmark)
    return {
        model
        for model, benchmarks in by_model.items()
        if GEOMETRY_REQUIRED_BENCHMARKS <= benchmarks
    }


def _scientific_evidence_reason(
    scientific_validity: ScientificValidity,
    run_mode: RunMode,
    evidence_schema_version: str,
) -> str | None:
    if run_mode is RunMode.SMOKE or (
        scientific_validity is ScientificValidity.NON_SCIENTIFIC_SMOKE
    ):
        return 'NON_SCIENTIFIC_SMOKE'
    if run_mode is RunMode.FIXTURE or (
        scientific_validity is ScientificValidity.NON_SCIENTIFIC_FIXTURE
    ):
        return 'NON_SCIENTIFIC_FIXTURE'
    if scientific_validity is not ScientificValidity.SCIENTIFIC:
        return 'NON_SCIENTIFIC_EVIDENCE'
    if run_mode is not RunMode.REAL:
        return 'SCIENTIFIC_REAL_TEST_EVIDENCE_REQUIRED'
    if evidence_schema_version != '1.1':
        return 'SCHEMA_V1_1_TEST_EVIDENCE_REQUIRED'
    return None


def evaluate_geometry_gate(value: GeometryGateInput) -> GateDecision:
    """Apply the approved Geometry MVE engineering and scientific gates."""

    evidence_reason = _scientific_evidence_reason(
        value.scientific_validity, value.run_mode, value.evidence_schema_version
    )
    if evidence_reason is not None:
        return GateDecision(
            lane="geometry",
            status=GateStatus.BLOCKED,
            reason_codes=(evidence_reason,),
            details={},
            scientific_validity=value.scientific_validity,
        )
    reported_ready_models = (
        set(value.reproducible_checkpoints) & set(value.hookable_models)
    )
    ready_models = reported_ready_models & GEOMETRY_MODEL_CANDIDATES
    if len(ready_models) < 2 or not value.required_datasets_ready:
        return GateDecision(
            lane="geometry",
            status=GateStatus.BLOCKED,
            reason_codes=("ENGINEERING_GATE_NOT_MET",),
            details={
                "ready_models": sorted(ready_models),
                "unexpected_models": sorted(
                    reported_ready_models - GEOMETRY_MODEL_CANDIDATES
                ),
                "required_datasets_ready": value.required_datasets_ready,
            },
            scientific_validity=value.scientific_validity,
        )
    if not value.fixed_inputs_verified:
        return GateDecision(
            lane="geometry",
            status=GateStatus.FAIL,
            reason_codes=("FIXED_INPUT_CONTROL_NOT_MET",),
            details={},
            scientific_validity=value.scientific_validity,
        )
    if not value.matched_intervention_effective:
        reason = (
            "ZEROING_ONLY_OOD_ARTIFACT"
            if value.zeroing_effective
            else "MATCHED_INTERVENTION_NOT_EFFECTIVE"
        )
        return GateDecision(
            lane="geometry",
            status=GateStatus.FAIL,
            reason_codes=(reason,),
            details={},
            scientific_validity=value.scientific_validity,
        )

    ready_evidence = tuple(
        item for item in value.evidence if item.model in ready_models
    )
    control_models = _models_with_control_coverage(ready_evidence)
    faithful = [
        item
        for item in ready_evidence
        if item.sample_class == GEOMETRY_REQUIRED_SAMPLE_CLASS
        and item.post_fusion_changed
        and item.delta_geom >= GEOMETRY_DELTA_THRESHOLD
        and item.ci_lower > 0
        and item.recovery >= GEOMETRY_RECOVERY_THRESHOLD
    ]
    faithful_models, faithful_strata, faithful_benchmarks = (
        _models_with_geometry_coverage(faithful)
    )
    faithful_models &= control_models
    if len(faithful_models) >= 2:
        return GateDecision(
            lane="geometry",
            status=GateStatus.PASS,
            reason_codes=("FAITHFUL_GEOMETRY_USE",),
            details={
                "qualifying_models": sorted(faithful_models),
                "qualifying_strata": sorted(faithful_strata),
                "qualifying_benchmarks": sorted(faithful_benchmarks),
                "control_models": sorted(control_models),
            },
            scientific_validity=value.scientific_validity,
        )

    nondependence = [
        item
        for item in ready_evidence
        if item.sample_class == GEOMETRY_REQUIRED_SAMPLE_CLASS
        and item.equivalent_by_tost
        and item.post_fusion_changed
    ]
    (
        nondependence_models,
        nondependence_strata,
        nondependence_benchmarks,
    ) = _models_with_geometry_coverage(nondependence)
    nondependence_models &= control_models
    if len(nondependence_models) >= 2:
        return GateDecision(
            lane="geometry",
            status=GateStatus.PASS,
            reason_codes=("CAUSAL_NONDEPENDENCE",),
            details={
                "qualifying_models": sorted(nondependence_models),
                "qualifying_strata": sorted(nondependence_strata),
                "qualifying_benchmarks": sorted(nondependence_benchmarks),
                "control_models": sorted(control_models),
            },
            scientific_validity=value.scientific_validity,
        )

    return GateDecision(
        lane="geometry",
        status=GateStatus.FAIL,
        reason_codes=("SCIENTIFIC_GATE_NOT_MET",),
        details={
            "faithful_models": sorted(faithful_models),
            "faithful_strata": sorted(faithful_strata),
            "faithful_benchmarks": sorted(faithful_benchmarks),
            "nondependence_models": sorted(nondependence_models),
            "nondependence_strata": sorted(nondependence_strata),
            "nondependence_benchmarks": sorted(nondependence_benchmarks),
            "control_models": sorted(control_models),
        },
        scientific_validity=value.scientific_validity,
    )


@dataclass(frozen=True, slots=True)
class GeoReliabConditionEvidence:
    model: str
    corruption: str
    clean_rho: float
    severity_rhos: tuple[float, float, float]
    failure_auroc: float
    corruption_severity_monotonic: bool
    cross_view_consistent: bool
    gt_geometry_invariant: bool
    relative_decline_ci_lower: float | None = None
    failure_auroc_ci_upper: float | None = None
    extreme_ood_only: bool = False
    scene_ids: tuple[str, ...] = ()
    scene_count: int = 0
    invalid_count: int = 0
    n_resamples: int = 0
    relative_decline_raw_p: float | None = None
    relative_decline_adjusted_p: float | None = None
    relative_decline_holm_rejected: bool = False
    failure_auroc_raw_p: float | None = None
    failure_auroc_adjusted_p: float | None = None
    failure_auroc_holm_rejected: bool = False
    branch_reason_codes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.model, str) or not self.model:
            raise ValueError("model must be a non-empty string")
        if self.corruption not in GEORELIAB_REQUIRED_CORRUPTIONS:
            raise ValueError(
                "corruption must be one of "
                f"{sorted(GEORELIAB_REQUIRED_CORRUPTIONS)}"
            )
        try:
            severity_rhos = tuple(self.severity_rhos)
        except TypeError as exc:
            raise ValueError("severity_rhos must contain severities 1, 2, and 3") from exc
        if len(severity_rhos) != 3:
            raise ValueError("severity_rhos must contain severities 1, 2, and 3")
        if any(
            isinstance(item, bool)
            or not isinstance(item, (int, float))
            or not math.isfinite(float(item))
            or not -1 <= float(item) <= 1
            for item in severity_rhos
        ):
            raise ValueError("severity_rhos must contain three finite correlations")
        object.__setattr__(
            self, "severity_rhos", tuple(float(item) for item in severity_rhos)
        )
        values = (
            self.clean_rho,
            self.failure_auroc,
            self.relative_decline_ci_lower,
            self.failure_auroc_ci_upper,
            self.relative_decline_raw_p,
            self.relative_decline_adjusted_p,
            self.failure_auroc_raw_p,
            self.failure_auroc_adjusted_p,
        )
        if any(
            value is not None
            and (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
            )
            for value in values
        ):
            raise ValueError("GeoReliab evidence values must be null or finite")
        if not -1 <= float(self.clean_rho) <= 1:
            raise ValueError("clean_rho must be in [-1, 1]")
        if not 0 <= float(self.failure_auroc) <= 1:
            raise ValueError("failure_auroc must be in [0, 1]")
        for name in (
            "corruption_severity_monotonic",
            "cross_view_consistent",
            "gt_geometry_invariant",
            "extreme_ood_only",
        ):
            if type(getattr(self, name)) is not bool:
                raise ValueError(f"{name} must be boolean")
        if self.scene_count < 0 or self.invalid_count < 0 or self.n_resamples < 0:
            raise ValueError('scene_count, invalid_count, and n_resamples must be non-negative')
        for name in ('relative_decline_holm_rejected', 'failure_auroc_holm_rejected'):
            if type(getattr(self, name)) is not bool:
                raise ValueError(f'{name} must be boolean')
        if any(not isinstance(item, str) or not item for item in self.scene_ids):
            raise ValueError('scene_ids must contain non-empty strings')
        if any(not isinstance(item, str) or not item for item in self.branch_reason_codes):
            raise ValueError('branch_reason_codes must contain non-empty strings')

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result['severity_rhos'] = list(self.severity_rhos)
        result['scene_ids'] = list(self.scene_ids)
        result['branch_reason_codes'] = list(self.branch_reason_codes)
        return result

    @property
    def severity3_rho(self) -> float:
        return self.severity_rhos[-1]

    def rho_ramp_monotonic(self) -> bool:
        directional = (self.clean_rho, *self.severity_rhos)
        return all(left >= right for left, right in zip(directional, directional[1:]))

    def verification_passed(self) -> bool:
        return (
            self.corruption_severity_monotonic
            and self.rho_ramp_monotonic()
            and self.cross_view_consistent
            and self.gt_geometry_invariant
        )

    def confidence_gate_hit(self) -> bool:
        clean_rho = self.clean_rho
        relative_decline = (
            (clean_rho - self.severity3_rho) / clean_rho
            if clean_rho >= 0.2
            else float("-inf")
        )
        relative_supported = (
            relative_decline > GEORELIAB_RHO_DECLINE_THRESHOLD
            and self.relative_decline_ci_lower is not None
            and self.relative_decline_ci_lower > GEORELIAB_RHO_DECLINE_THRESHOLD
            and self.relative_decline_adjusted_p is not None
            and self.relative_decline_adjusted_p <= 0.05
            and self.relative_decline_holm_rejected
        )
        auroc_supported = (
            self.failure_auroc < GEORELIAB_FAILURE_AUROC_THRESHOLD
            and self.failure_auroc_ci_upper is not None
            and self.failure_auroc_ci_upper < GEORELIAB_FAILURE_AUROC_THRESHOLD
            and self.failure_auroc_adjusted_p is not None
            and self.failure_auroc_adjusted_p <= 0.05
            and self.failure_auroc_holm_rejected
        )
        return relative_supported or auroc_supported


@dataclass(frozen=True, slots=True)
class DownstreamHarmEvidence:
    model: str
    condition: str
    effect_vs_random: float
    ci_upper: float

    def __post_init__(self) -> None:
        if not isinstance(self.model, str) or not self.model:
            raise ValueError('model must be a non-empty string')
        if not isinstance(self.condition, str) or not self.condition:
            raise ValueError('condition must be a non-empty string')
        if any(
            isinstance(value, bool) or not math.isfinite(float(value))
            for value in (self.effect_vs_random, self.ci_upper)
        ):
            raise ValueError('harm evidence values must be finite numbers')

    def qualifies(self) -> bool:
        return self.effect_vs_random < 0 and self.ci_upper < 0


@dataclass(frozen=True, slots=True)
class ZeroUpdateEvidence:
    model: str
    condition: str
    auroc_gain: float
    ci_lower: float

    def __post_init__(self) -> None:
        if not isinstance(self.model, str) or not self.model:
            raise ValueError('model must be a non-empty string')
        if not isinstance(self.condition, str) or not self.condition:
            raise ValueError('condition must be a non-empty string')
        if any(
            isinstance(value, bool) or not math.isfinite(float(value))
            for value in (self.auroc_gain, self.ci_lower)
        ):
            raise ValueError('zero-update evidence values must be finite numbers')

    def qualifies(self) -> bool:
        return (
            self.auroc_gain >= GEORELIAB_ZERO_UPDATE_GAIN_THRESHOLD
            and self.ci_lower > 0
        )


@dataclass(frozen=True, slots=True)
class GeoReliabGateInput:
    scientific_validity: ScientificValidity
    required_models_ready: tuple[str, ...]
    required_datasets_ready: bool
    tartanair_native_fog_sanity: bool
    conditions: tuple[GeoReliabConditionEvidence, ...]
    downstream_harm: tuple[DownstreamHarmEvidence, ...]
    zero_update: tuple[ZeroUpdateEvidence, ...]
    run_mode: RunMode = RunMode.REAL
    evidence_schema_version: str = '1.1'
    split: str = 'test'
    schedule_counts: dict[str, int] = None
    downstream_schedule_counts: dict[str, int] = None


def evaluate_georeliab_gate(value: GeoReliabGateInput) -> GateDecision:
    """Apply the approved GeoReliab MVE gate without optimistic fallback."""

    evidence_reason = _scientific_evidence_reason(
        value.scientific_validity, value.run_mode, value.evidence_schema_version
    )
    if evidence_reason is not None:
        return GateDecision(
            lane="georeliab",
            status=GateStatus.BLOCKED,
            reason_codes=(evidence_reason,),
            details={},
            scientific_validity=value.scientific_validity,
        )
    reported_ready_models = set(value.required_models_ready)
    ready_models = reported_ready_models & GEORELIAB_MVE_REQUIRED_MODELS
    if (
        not GEORELIAB_MVE_REQUIRED_MODELS <= reported_ready_models
        or not value.required_datasets_ready
    ):
        return GateDecision(
            lane="georeliab",
            status=GateStatus.BLOCKED,
            reason_codes=("ENGINEERING_GATE_NOT_MET",),
            details={
                "missing_models": sorted(
                    GEORELIAB_MVE_REQUIRED_MODELS - reported_ready_models
                ),
                "unexpected_models": sorted(
                    reported_ready_models - GEORELIAB_MVE_REQUIRED_MODELS
                ),
                "ready_models": sorted(ready_models),
                "required_datasets_ready": value.required_datasets_ready,
            },
            scientific_validity=value.scientific_validity,
        )

    if value.split != 'test':
        return GateDecision(
            lane='georeliab',
            status=GateStatus.FAIL,
            reason_codes=('P3_SCHEDULE_COUNTS_INVALID',),
            details={'split': value.split, 'schedule_counts': value.schedule_counts or {}},
            scientific_validity=value.scientific_validity,
        )
    if not _p3_schedule_counts_valid(value.schedule_counts):
        return GateDecision(
            lane='georeliab',
            status=GateStatus.BLOCKED,
            reason_codes=('P3_SCHEDULE_COUNTS_INVALID',),
            details={'split': value.split, 'schedule_counts': value.schedule_counts or {}},
            scientific_validity=value.scientific_validity,
        )

    condition_reason = _validate_georeliab_condition_grid(value.conditions)
    if condition_reason is not None:
        return GateDecision(
            lane='georeliab',
            status=GateStatus.FAIL,
            reason_codes=(condition_reason,),
            details={'condition_count': len(value.conditions)},
            scientific_validity=value.scientific_validity,
        )
    if value.schedule_counts.get('invalid', 0) > 0 or any(item.invalid_count > 0 for item in value.conditions):
        return GateDecision(
            lane='georeliab',
            status=GateStatus.FAIL,
            reason_codes=('INVALID_OUTPUT_IN_VERIFIED_CONDITION',),
            details={
                'condition_invalid': sum(item.invalid_count for item in value.conditions),
                'schedule_invalid': value.schedule_counts.get('invalid'),
            },
            scientific_validity=value.scientific_validity,
        )

    verified_conditions = [
        condition
        for condition in value.conditions
        if condition.model in ready_models and condition.verification_passed()
    ]
    qualifying_conditions = [
        condition
        for condition in verified_conditions
        if not condition.extreme_ood_only and condition.confidence_gate_hit()
    ]
    by_model: dict[str, set[str]] = {}
    for condition in qualifying_conditions:
        by_model.setdefault(condition.model, set()).add(condition.corruption)
    confidence_models = {
        model
        for model, corruptions in by_model.items()
        if GEORELIAB_REQUIRED_CORRUPTIONS <= corruptions
    }
    confidence_pass = len(confidence_models) >= 2

    harm_by_condition = {
        (item.model, item.condition): item
        for item in value.downstream_harm
        if item.model in ready_models and item.qualifies()
    }
    harm = list(harm_by_condition.values())
    harm_pass = len(harm) >= 2
    zero_update_by_condition = {
        (item.model, item.condition): item
        for item in value.zero_update
        if item.model in ready_models and item.qualifies()
    }
    zero_update = list(zero_update_by_condition.values())
    zero_update_pass = bool(zero_update)

    reason_codes: list[str] = []
    if not value.tartanair_native_fog_sanity:
        reason_codes.append("TARTANAIR_SANITY_GATE_NOT_MET")
    if not confidence_pass:
        reason_codes.append("CONFIDENCE_FAILURE_GATE_NOT_MET")
        return GateDecision(
            lane="georeliab",
            status=GateStatus.FAIL,
            reason_codes=tuple(reason_codes),
            details={
                "confidence_models": sorted(confidence_models),
                "verified_condition_count": len(verified_conditions),
                "tartanair_native_fog_sanity": value.tartanair_native_fog_sanity,
                "p5_skip_reason": "P4_CONFIDENCE_PHENOMENON_GATE_FAILED",
            },
            scientific_validity=value.scientific_validity,
        )
    if reason_codes:
        return GateDecision(
            lane="georeliab",
            status=GateStatus.FAIL,
            reason_codes=tuple(reason_codes),
            details={
                "confidence_models": sorted(confidence_models),
                "verified_condition_count": len(verified_conditions),
                "tartanair_native_fog_sanity": value.tartanair_native_fog_sanity,
            },
            scientific_validity=value.scientific_validity,
        )
    if not _p5_downstream_schedule_counts_valid(value.downstream_schedule_counts):
        return GateDecision(
            lane='georeliab',
            status=GateStatus.BLOCKED,
            reason_codes=('P5_DOWNSTREAM_SCHEDULE_COUNTS_INVALID',),
            details={'downstream_schedule_counts': value.downstream_schedule_counts or {}},
            scientific_validity=value.scientific_validity,
        )
    execution_reason = _validate_p5_execution_grid(value.downstream_harm, value.zero_update)
    if execution_reason is not None:
        return GateDecision(
            lane='georeliab',
            status=GateStatus.FAIL,
            reason_codes=(execution_reason,),
            details={'downstream_harm_count': len(value.downstream_harm), 'zero_update_count': len(value.zero_update)},
            scientific_validity=value.scientific_validity,
        )
    if not harm_pass:
        reason_codes.append("DOWNSTREAM_HARM_GATE_NOT_MET")
    if not zero_update_pass:
        reason_codes.append("ZERO_UPDATE_GATE_NOT_MET")
    status = GateStatus.PASS if not reason_codes else GateStatus.FAIL
    if status is GateStatus.PASS:
        reason_codes.append("ALL_GEORELIAB_GATES_MET")
    return GateDecision(
        lane="georeliab",
        status=status,
        reason_codes=tuple(reason_codes),
        details={
            "confidence_models": sorted(confidence_models),
            "verified_condition_count": len(verified_conditions),
            "tartanair_native_fog_sanity": value.tartanair_native_fog_sanity,
            "qualifying_harm_conditions": [
                f"{item.model}:{item.condition}" for item in harm
            ],
            "qualifying_zero_update_conditions": [
                f"{item.model}:{item.condition}" for item in zero_update
            ],
        },
        scientific_validity=value.scientific_validity,
    )


def _p3_schedule_counts_valid(counts: dict[str, int] | None) -> bool:
    if not isinstance(counts, dict):
        return False
    if counts.get('scheduled') != 400 or counts.get('completed') != 400 or counts.get('missing') != 0:
        return False
    invalid = counts.get('invalid')
    return isinstance(invalid, int) and 0 <= invalid <= 400


def _p5_downstream_schedule_counts_valid(counts: dict[str, int] | None) -> bool:
    if not isinstance(counts, dict):
        return False
    return counts.get('scheduled') == 6 and counts.get('completed') == 6 and counts.get('missing') == 0


def _validate_georeliab_condition_grid(
    conditions: tuple[GeoReliabConditionEvidence, ...]
) -> str | None:
    expected_pairs = {
        (model, corruption)
        for model in GEORELIAB_MVE_REQUIRED_MODELS
        for corruption in GEORELIAB_REQUIRED_CORRUPTIONS
    }
    observed_pairs = [(item.model, item.corruption) for item in conditions]
    if len(observed_pairs) != len(expected_pairs) or set(observed_pairs) != expected_pairs:
        return 'CONDITION_GRID_INVALID'
    if len(set(observed_pairs)) != len(observed_pairs):
        return 'CONDITION_GRID_INVALID'
    scene_sets = {tuple(item.scene_ids) for item in conditions}
    if len(scene_sets) != 1:
        return 'CONDITION_SCENE_KEYS_INVALID'
    for item in conditions:
        if item.scene_count != 20 or len(item.scene_ids) != 20 or item.n_resamples != 10_000:
            return 'CONDITION_PROVENANCE_INVALID'
    return None


def _validate_p5_execution_grid(
    downstream_harm: tuple[DownstreamHarmEvidence, ...],
    zero_update: tuple[ZeroUpdateEvidence, ...]
) -> str | None:
    expected = {
        (model, f'{corruption}-s2')
        for model in GEORELIAB_MVE_REQUIRED_MODELS
        for corruption in GEORELIAB_REQUIRED_CORRUPTIONS
    }
    harm_pairs = [(item.model, item.condition) for item in downstream_harm]
    zero_pairs = [(item.model, item.condition) for item in zero_update]
    if len(harm_pairs) != 6 or set(harm_pairs) != expected or len(set(harm_pairs)) != 6:
        return 'DOWNSTREAM_EXECUTION_GRID_INVALID'
    if len(zero_pairs) != 6 or set(zero_pairs) != expected or len(set(zero_pairs)) != 6:
        return 'ZERO_UPDATE_EXECUTION_GRID_INVALID'
    return None

def select_track(
    geometry: GateDecision, georeliab: GateDecision
) -> SelectionDecision:
    """Apply the four pre-registered final selection rules."""

    if (
        geometry.scientific_validity is not ScientificValidity.SCIENTIFIC
        or georeliab.scientific_validity is not ScientificValidity.SCIENTIFIC
    ):
        validity = (
            ScientificValidity.NON_SCIENTIFIC_SMOKE
            if (
                geometry.scientific_validity is ScientificValidity.NON_SCIENTIFIC_SMOKE
                or georeliab.scientific_validity is ScientificValidity.NON_SCIENTIFIC_SMOKE
            )
            else ScientificValidity.NON_SCIENTIFIC_FIXTURE
        )
        return SelectionDecision(
            selected_track=(
                SelectedTrack.NON_SCIENTIFIC_SMOKE
                if validity is ScientificValidity.NON_SCIENTIFIC_SMOKE
                else SelectedTrack.NON_SCIENTIFIC
            ),
            reason=(
                'smoke evidence cannot select a scientific track'
                if validity is ScientificValidity.NON_SCIENTIFIC_SMOKE
                else 'fixture evidence cannot select a scientific track'
            ),
            geometry_status=geometry.status,
            georeliab_status=georeliab.status,
            scientific_validity=validity,
        )
    if not geometry.status.is_terminal or not georeliab.status.is_terminal:
        pending_georeliab = (
            geometry.status is GateStatus.BLOCKED
            and georeliab.status is GateStatus.PASS
        )
        return SelectionDecision(
            selected_track=(
                SelectedTrack.BLOCKED_PENDING_GEOMETRY
                if pending_georeliab
                else SelectedTrack.BLOCKED
            ),
            reason=(
                'GEORELIAB_PASS_PENDING_GEOMETRY'
                if pending_georeliab
                else 'required scientific evidence is non-terminal'
            ),
            geometry_status=geometry.status,
            georeliab_status=georeliab.status,
            scientific_validity=ScientificValidity.SCIENTIFIC,
        )
    if geometry.status is GateStatus.PASS:
        return SelectionDecision(
            selected_track=SelectedTrack.GEOMETRY,
            reason=(
                "Geometry passed and has priority, including when both lanes pass"
            ),
            geometry_status=geometry.status,
            georeliab_status=georeliab.status,
            scientific_validity=ScientificValidity.SCIENTIFIC,
        )
    if georeliab.status is GateStatus.PASS:
        return SelectionDecision(
            selected_track=SelectedTrack.GEORELIAB,
            reason="Geometry did not pass and GeoReliab passed",
            geometry_status=geometry.status,
            georeliab_status=georeliab.status,
            scientific_validity=ScientificValidity.SCIENTIFIC,
        )
    return SelectionDecision(
        selected_track=SelectedTrack.STOP,
        reason=(
            "both scientific gates failed; do not auto-switch to Deformable World"
        ),
        geometry_status=geometry.status,
        georeliab_status=georeliab.status,
        scientific_validity=ScientificValidity.SCIENTIFIC,
    )
