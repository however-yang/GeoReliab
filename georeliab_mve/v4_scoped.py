"""Scoped, development-only evidence contracts for GeoReliab v4.

The original v4 evidence and gate objects intentionally remain frozen at the
400-unit/20-scene inventory.  This module provides a *separate* wrapper for
the pre-registered 3-scene development pilot and its optional 5-scene
extension.  It never mutates, relaxes, or replaces the formal gate.

All selectors and manifests are deterministic and outcome independent.  The
module is CPU-only; it accepts already materialised rows and does not import
model adapters or launch inference.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
import math
from statistics import median
from typing import Any

from .metrics import MetricError, binary_auroc
from .v4_qualification import ScheduleIdentityManifest
from .v4_counterfactuals import (
    FOG_BOUNDARY_LAG_SEQUENCE,
    SCIENTIFIC_MODELS,
    SCIENTIFIC_STATES,
    TEST_SCENE_IDS,
)
from .v4_metrics import boundary_lag_for_scene


SCOPED_SCHEMA_VERSION = "georeliab-v4-scoped-evidence-1.0"
PARTITION_SCHEMA_VERSION = "georeliab-v4-pilot-partition-1.0"
POWER_SCHEMA_VERSION = "georeliab-v4-synthetic-power-design-1.0"
SCOPE_ADDENDUM_SCHEMA_VERSION = "georeliab-v4-confirmation-scope-addendum-1.0"
SELECTOR_VERSION = "georeliab-v4-pilot-selector-sha256-v1"
CONFIRMATION_SCOPE_VERSION = "georeliab-v4-confirmation-scope-v1"

V4_PILOT_GO_TO_FULL_MVE = "V4_PILOT_GO_TO_FULL_MVE"
V4_PILOT_SCIENTIFIC_NO_GO = "V4_PILOT_SCIENTIFIC_NO_GO"
V4_PILOT_INCONCLUSIVE = "V4_PILOT_INCONCLUSIVE"
V4_PILOT_BLOCKED_EXECUTION = "V4_PILOT_BLOCKED_EXECUTION"
FULL_CONFIRMATION_FORBIDDEN = "FULL_CONFIRMATION_FORBIDDEN"
V4_RECOVERY_CPU_QUALIFIED = "V4_RECOVERY_CPU_QUALIFIED"

PROTOCOL_DECISION_GO = "V4_PROTOCOL_GO"
PROTOCOL_DECISION_NO_GO = "V4_PROTOCOL_NO_GO"
GO_FAMILY_RANKING_WARNING = "RANKING_WARNING"
GO_FAMILY_TASK_TRANSFER = "TASK_TRANSFER"
GO_FAMILY_BOTH = "BOTH"
GO_FAMILY_NONE = "NONE"

PAPER_RANKING_NOT_WARNING_SUPPORTED = "RANKING_IS_NOT_WARNING_SUPPORTED"
PAPER_RANKING_SUPPORTED_LAG_NOT_SUPPORTED = (
    "RANKING_SUPPORTED_WARNING_LAG_NOT_SUPPORTED"
)
PAPER_TASK_TRANSFER_ONLY_SUPPORTED = "TASK_TRANSFER_ONLY_SUPPORTED"
PAPER_NO_PRIMARY_CLAIM = "NO_PRIMARY_PAPER_CLAIM_SUPPORTED"

FROZEN_STRATA: tuple[tuple[str, tuple[int, ...]], ...] = (
    ("S0", (1, 9, 10, 11)),
    ("S1", (12, 13, 23, 24)),
    ("S2", (29, 32, 33, 34)),
    ("S3", (48, 49, 62, 75)),
    ("S4", (77, 110, 114, 118)),
)


def _canonical_bytes(value: object) -> bytes:
    """Canonical JSON used by every new manifest in this module."""

    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _sha_domain(domain: str, value: object) -> str:
    return hashlib.sha256(domain.encode("ascii") + b"\0" + _canonical_bytes(value)).hexdigest()


def _sha_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _valid_sha(value: object, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or value != value.lower()
        or any(char not in "0123456789abcdef" for char in value)
    ):
        raise ValueError(f"{label} must be a lowercase SHA-256")
    return value


def _scene_ids(value: Iterable[int], *, label: str = "scene_ids") -> tuple[int, ...]:
    result = tuple(int(scene) for scene in value)
    if not result or len(set(result)) != len(result):
        raise ValueError(f"{label} must be non-empty and unique")
    if any(scene not in TEST_SCENE_IDS for scene in result):
        raise ValueError(f"{label} contains a scene outside the frozen schedule")
    return result


def _ordered_units(scene_ids: Sequence[int]) -> tuple[str, ...]:
    return tuple(
        f"{model}:{scene}:{state}"
        for model in SCIENTIFIC_MODELS
        for scene in scene_ids
        for state in SCIENTIFIC_STATES
    )


@dataclass(frozen=True, slots=True)
class ConfirmationScopeAddendum:
    """A machine-readable, non-protocol scope restriction."""

    schedule_identity_sha256: str
    protocol_sha256: str
    role: str
    scene_ids: tuple[int, ...]
    expected_record_count: int
    scope_sha256: str
    addendum_sha256: str
    schema_version: str = SCOPE_ADDENDUM_SCHEMA_VERSION

    def __post_init__(self) -> None:
        _valid_sha(self.schedule_identity_sha256, "schedule_identity_sha256")
        _valid_sha(self.protocol_sha256, "protocol_sha256")
        _valid_sha(self.scope_sha256, "scope_sha256")
        _valid_sha(self.addendum_sha256, "addendum_sha256")
        if self.role not in {"PRIMARY", "EXTENSION", "CONFIRMATION_15", "CONFIRMATION_17", "CONFIRMATION_20", "FULL20"}:
            raise ValueError("unsupported confirmation scope role")
        scenes = _scene_ids(self.scene_ids)
        if scenes != self.scene_ids:
            raise ValueError("scene_ids must retain deterministic order")
        if self.expected_record_count != len(scenes) * len(SCIENTIFIC_MODELS) * len(SCIENTIFIC_STATES):
            raise ValueError("expected_record_count does not match scope")
        expected_scope = _sha_domain(
            "georeliab:confirmation-scope:v1",
            {"role": self.role, "scene_ids": list(self.scene_ids), "schedule_identity_sha256": self.schedule_identity_sha256},
        )
        expected_addendum = _sha_domain(
            "georeliab:scope-addendum:v1",
            {
                "schema_version": self.schema_version,
                "schedule_identity_sha256": self.schedule_identity_sha256,
                "protocol_sha256": self.protocol_sha256,
                "role": self.role,
                "scene_ids": list(self.scene_ids),
                "expected_record_count": self.expected_record_count,
                "scope_sha256": self.scope_sha256,
            },
        )
        if self.scope_sha256 != expected_scope or self.addendum_sha256 != expected_addendum:
            raise ValueError("confirmation scope digest mismatch")

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "schedule_identity_sha256": self.schedule_identity_sha256,
            "protocol_sha256": self.protocol_sha256,
            "role": self.role,
            "scene_ids": list(self.scene_ids),
            "expected_record_count": self.expected_record_count,
            "scope_sha256": self.scope_sha256,
            "addendum_sha256": self.addendum_sha256,
        }


@dataclass(frozen=True, slots=True)
class PilotPartitionManifest:
    schedule_identity_sha256: str
    primary_scene_ids: tuple[int, ...]
    extension_scene_ids: tuple[int, ...]
    core_scene_ids: tuple[int, ...]
    selector_payload_sha256: str
    partition_sha256: str
    primary_scope: ConfirmationScopeAddendum
    extension_scope: ConfirmationScopeAddendum
    confirmation_15_scope: ConfirmationScopeAddendum
    confirmation_17_scope: ConfirmationScopeAddendum
    schema_version: str = PARTITION_SCHEMA_VERSION

    def __post_init__(self) -> None:
        _valid_sha(self.schedule_identity_sha256, "schedule_identity_sha256")
        _valid_sha(self.selector_payload_sha256, "selector_payload_sha256")
        _valid_sha(self.partition_sha256, "partition_sha256")
        primary = _scene_ids(self.primary_scene_ids)
        extension = _scene_ids(self.extension_scene_ids)
        core = _scene_ids(self.core_scene_ids)
        if len(primary) != 3 or len(extension) != 2 or len(core) != 15:
            raise ValueError("partition must contain 3 primary, 2 extension, and 15 core scenes")
        if set(primary) & set(extension) or set(primary) & set(core) or set(extension) & set(core):
            raise ValueError("partition scopes must be disjoint")
        if set(primary) | set(extension) | set(core) != set(TEST_SCENE_IDS):
            raise ValueError("partition must cover all 20 frozen scenes")
        if tuple(self.primary_scope.scene_ids) != primary or tuple(self.extension_scope.scene_ids) != extension:
            raise ValueError("scope addendum scene IDs do not match partition")
        if tuple(self.confirmation_15_scope.scene_ids) != core:
            raise ValueError("15-scene confirmation scope must be core")
        if tuple(self.confirmation_17_scope.scene_ids) != tuple(extension) + tuple(core):
            raise ValueError("17-scene confirmation scope must be extension plus core")

        scopes = (self.primary_scope, self.extension_scope, self.confirmation_15_scope, self.confirmation_17_scope)
        if any(scope.schedule_identity_sha256 != self.schedule_identity_sha256 for scope in scopes):
            raise ValueError("scope addendum schedule identity mismatch")
        selector_payload = {
            "selector_version": SELECTOR_VERSION,
            "schedule_identity_sha256": self.schedule_identity_sha256,
            "strata": [[name, list(scenes)] for name, scenes in FROZEN_STRATA],
            "primary_scene_ids": list(primary),
            "extension_scene_ids": list(extension),
            "core_scene_ids": list(core),
        }
        if self.selector_payload_sha256 != _sha_domain("georeliab:pilot-selector-manifest:v1", selector_payload):
            raise ValueError("pilot selector digest mismatch")
        partition_payload = {
            "schema_version": self.schema_version,
            "schedule_identity_sha256": self.schedule_identity_sha256,
            "primary_scene_ids": list(primary),
            "extension_scene_ids": list(extension),
            "core_scene_ids": list(core),
            "selector_payload_sha256": self.selector_payload_sha256,
            "addenda": {name: scope.addendum_sha256 for name, scope in (("primary_scope", self.primary_scope), ("extension_scope", self.extension_scope), ("confirmation_15_scope", self.confirmation_15_scope), ("confirmation_17_scope", self.confirmation_17_scope))},
        }
        if self.partition_sha256 != _sha_domain("georeliab:pilot-partition:v1", partition_payload):
            raise ValueError("pilot partition digest mismatch")

    @property
    def primary_unit_ids(self) -> tuple[str, ...]:
        return _ordered_units(self.primary_scene_ids)

    @property
    def extension_unit_ids(self) -> tuple[str, ...]:
        return _ordered_units(self.extension_scene_ids)

    @property
    def confirmation_15_unit_ids(self) -> tuple[str, ...]:
        return _ordered_units(self.core_scene_ids)

    @property
    def confirmation_17_unit_ids(self) -> tuple[str, ...]:
        return _ordered_units(self.extension_scene_ids + self.core_scene_ids)

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "schedule_identity_sha256": self.schedule_identity_sha256,
            "primary_scene_ids": list(self.primary_scene_ids),
            "extension_scene_ids": list(self.extension_scene_ids),
            "core_scene_ids": list(self.core_scene_ids),
            "selector_payload_sha256": self.selector_payload_sha256,
            "partition_sha256": self.partition_sha256,
            "primary_scope": self.primary_scope.to_dict(),
            "extension_scope": self.extension_scope.to_dict(),
            "confirmation_15_scope": self.confirmation_15_scope.to_dict(),
            "confirmation_17_scope": self.confirmation_17_scope.to_dict(),
        }


def _select_scene(schedule_identity_sha256: str, role: str, stratum: str, scenes: Sequence[int]) -> int:
    candidates = [
        (
            _sha_domain(
                "georeliab:pilot-selector:v1",
                {
                    "selector_version": SELECTOR_VERSION,
                    "schedule_identity_sha256": schedule_identity_sha256,
                    "role": role,
                    "stratum": stratum,
                    "scene_id": scene,
                },
            ),
            scene,
        )
        for scene in scenes
    ]
    return min(candidates)[1]


def build_pilot_partition(
    schedule_identity_sha256: str,
    *,
    protocol_sha256: str = "0" * 64,
) -> PilotPartitionManifest:
    """Select one outcome-independent scene per frozen stratum/role."""

    _valid_sha(schedule_identity_sha256, "schedule_identity_sha256")
    _valid_sha(protocol_sha256, "protocol_sha256")
    strata = dict(FROZEN_STRATA)
    primary = tuple(
        _select_scene(schedule_identity_sha256, "PRIMARY", stratum, strata[stratum])
        for stratum in ("S0", "S2", "S4")
    )
    extension = tuple(
        _select_scene(schedule_identity_sha256, "EXTENSION", stratum, strata[stratum])
        for stratum in ("S1", "S3")
    )
    used = set(primary) | set(extension)
    core = tuple(scene for scene in TEST_SCENE_IDS if scene not in used)
    selector_payload = {
        "selector_version": SELECTOR_VERSION,
        "schedule_identity_sha256": schedule_identity_sha256,
        "strata": [[name, list(scenes)] for name, scenes in FROZEN_STRATA],
        "primary_scene_ids": primary,
        "extension_scene_ids": extension,
        "core_scene_ids": core,
    }
    selector_hash = _sha_domain("georeliab:pilot-selector-manifest:v1", selector_payload)

    def scope(role: str, scenes: tuple[int, ...]) -> ConfirmationScopeAddendum:
        count = len(scenes) * len(SCIENTIFIC_MODELS) * len(SCIENTIFIC_STATES)
        scope_hash = _sha_domain(
            "georeliab:confirmation-scope:v1",
            {"role": role, "scene_ids": list(scenes), "schedule_identity_sha256": schedule_identity_sha256},
        )
        addendum_fields = {
            "schema_version": SCOPE_ADDENDUM_SCHEMA_VERSION,
            "schedule_identity_sha256": schedule_identity_sha256,
            "protocol_sha256": protocol_sha256,
            "role": role,
            "scene_ids": tuple(scenes),
            "expected_record_count": count,
            "scope_sha256": scope_hash,
        }
        return ConfirmationScopeAddendum(
            **addendum_fields,

            addendum_sha256=_sha_domain("georeliab:scope-addendum:v1", addendum_fields),
        )

    addenda = {
        "primary_scope": scope("PRIMARY", primary),
        "extension_scope": scope("EXTENSION", extension),
        "confirmation_15_scope": scope("CONFIRMATION_15", core),
        "confirmation_17_scope": scope("CONFIRMATION_17", extension + core),
    }
    fields = {
        "schema_version": PARTITION_SCHEMA_VERSION,
        "schedule_identity_sha256": schedule_identity_sha256,
        "primary_scene_ids": primary,
        "extension_scene_ids": extension,
        "core_scene_ids": core,
        "selector_payload_sha256": selector_hash,
    }
    return PilotPartitionManifest(
        **fields,
        partition_sha256=_sha_domain("georeliab:pilot-partition:v1", {**fields, "addenda": {name: value.addendum_sha256 for name, value in addenda.items()}}),
        **addenda,
    )


@dataclass(frozen=True, slots=True)
class PilotModelMetrics:
    model_id: str
    auroc: float
    boundary_lag_median: float
    late_warning_proportion: float
    positive_ranking_scene_count: int
    positive_warning_scene_count: int
    scene_count: int
    defined: bool = True
    reason_code: str = "DEFINED"

    def to_dict(self) -> dict[str, object]:
        """Return the metric record without exposing scientific aggregates."""

        return {
            "model_id": self.model_id,
            "auroc": self.auroc,
            "boundary_lag_median": self.boundary_lag_median,
            "late_warning_proportion": self.late_warning_proportion,
            "positive_ranking_scene_count": self.positive_ranking_scene_count,
            "positive_warning_scene_count": self.positive_warning_scene_count,
            "scene_count": self.scene_count,
            "defined": self.defined,
            "reason_code": self.reason_code,
        }


@dataclass(frozen=True, slots=True)
class PilotDecision:
    status: str
    reason_code: str
    macro_auroc: float | None
    pooled_boundary_lag_median: float | None
    late_warning_proportion: float | None
    models: tuple[PilotModelMetrics, ...]
    loso_auroc: tuple[float, ...] = ()
    loso_boundary_lag: tuple[float, ...] = ()
    invalid_count: int = 0
    duplicate_count: int = 0
    identity_mismatch_count: int = 0
    paper_claim_qualification: str = PAPER_NO_PRIMARY_CLAIM
    scene_dominance_ranking: tuple[float, ...] = ()
    scene_dominance_lag: tuple[float, ...] = ()
    scene_dominance_pass: bool = False

    def __post_init__(self) -> None:
        allowed_statuses = {
            V4_PILOT_GO_TO_FULL_MVE,
            V4_PILOT_SCIENTIFIC_NO_GO,
            V4_PILOT_INCONCLUSIVE,
            V4_PILOT_BLOCKED_EXECUTION,
        }
        if self.status not in allowed_statuses:
            raise ValueError("unsupported pilot decision status")
        for field_name in (
            "invalid_count",
            "duplicate_count",
            "identity_mismatch_count",
        ):
            value = getattr(self, field_name)
            if type(value) is not int or value < 0:
                raise ValueError(f"{field_name} must be a non-negative integer")
        for field_name in (
            "models",
            "loso_auroc",
            "loso_boundary_lag",
            "scene_dominance_ranking",
            "scene_dominance_lag",
        ):
            if not isinstance(getattr(self, field_name), tuple):
                raise ValueError(f"{field_name} must be an immutable tuple")

    @property
    def execution_blocked(self) -> bool:
        return self.status == V4_PILOT_BLOCKED_EXECUTION or any(
            (
                self.invalid_count,
                self.duplicate_count,
                self.identity_mismatch_count,
            )
        )

    def to_dict(self) -> dict[str, object]:
        """Serialize the complete machine-readable pilot decision."""

        return {
            "status": self.status,
            "reason_code": self.reason_code,
            "macro_auroc": self.macro_auroc,
            "pooled_boundary_lag_median": self.pooled_boundary_lag_median,
            "late_warning_proportion": self.late_warning_proportion,
            "models": [model.to_dict() for model in self.models],
            "loso_auroc": list(self.loso_auroc),
            "loso_boundary_lag": list(self.loso_boundary_lag),
            "scene_dominance_ranking": list(self.scene_dominance_ranking),
            "scene_dominance_lag": list(self.scene_dominance_lag),
            "scene_dominance_pass": self.scene_dominance_pass,
            "invalid_count": self.invalid_count,
            "duplicate_count": self.duplicate_count,
            "identity_mismatch_count": self.identity_mismatch_count,
            "execution_blocked": self.execution_blocked,
            "paper_claim_qualification": self.paper_claim_qualification,
            "scientific_result": "NO_SCIENTIFIC_RESULT",
        }


def _row_value(row: object, key: str) -> Any:
    if isinstance(row, Mapping):
        return row[key]
    return getattr(row, key)


def _normalise_rows(records: Sequence[object], scene_ids: tuple[int, ...]) -> dict[tuple[str, int, str], object]:
    expected = {
        (model, scene, state)
        for model in SCIENTIFIC_MODELS
        for scene in scene_ids
        for state in SCIENTIFIC_STATES
    }
    lookup: dict[tuple[str, int, str], object] = {}
    for row in records:
        try:
            key = (str(_row_value(row, "model_id")), int(_row_value(row, "scene_id")), str(_row_value(row, "state_id")))
        except (AttributeError, KeyError, TypeError, ValueError) as exc:
            raise ValueError("pilot record is missing model/scene/state identity") from exc
        if key not in expected:
            raise ValueError("pilot record lies outside the selected scope")
        if key in lookup:
            raise ValueError("duplicate pilot record identity")
        lookup[key] = row
    if set(lookup) != expected:
        raise ValueError("pilot scope is missing one or more records")
    return lookup


def _model_metrics(lookup: Mapping[tuple[str, int, str], object], model: str, scenes: tuple[int, ...]) -> PilotModelMetrics:
    labels: list[bool] = []
    scores: list[float] = []
    scene_auroc: list[float | None] = []
    lags: list[float] = []
    positive_scene = 0
    positive_warning = 0
    for scene in scenes:
        rows = [lookup[(model, scene, state)] for state in SCIENTIFIC_STATES]
        scene_labels = [bool(_row_value(row, "pose_failure")) for row in rows]
        scene_scores = [float(_row_value(row, "native_warning_score")) for row in rows]
        labels.extend(scene_labels)
        scores.extend(scene_scores)
        try:
            local_auc = binary_auroc(scene_labels, scene_scores)
        except MetricError:
            local_auc = None
        scene_auroc.append(local_auc)
        if local_auc is not None and local_auc > 0.5:
            positive_scene += 1
        fog_rows = [lookup[(model, scene, state)] for state in FOG_BOUNDARY_LAG_SEQUENCE]
        lag_result = boundary_lag_for_scene(
            FOG_BOUNDARY_LAG_SEQUENCE,
            alarms=[bool(_row_value(row, "alarm")) for row in fog_rows],
            pose_failures=[bool(_row_value(row, "pose_failure")) for row in fog_rows],
        )
        if lag_result.included and lag_result.lag is not None:
            lags.append(float(lag_result.lag))
            if lag_result.lag > 0:
                positive_warning += 1
    try:
        auc = binary_auroc(labels, scores)
    except MetricError:
        return PilotModelMetrics(model, 0.0, 0.0, 0.0, positive_scene, positive_warning, len(scenes), False, "UNDEFINED_AUROC")
    if not lags:
        return PilotModelMetrics(model, auc, 0.0, 0.0, positive_scene, positive_warning, len(scenes), False, "NO_FAILURE_GROUPS")
    return PilotModelMetrics(
        model,
        auc,
        float(median(lags)),
        sum(lag > 0 for lag in lags) / len(lags),
        positive_scene,
        positive_warning,
        len(scenes),
    )


def _scene_dominance_ratios(full_value: float, neutral_value: float, loso_values: Sequence[float]) -> tuple[tuple[float, ...], bool]:
    """Return leave-one-scene-out dominance ratios with fail-closed zero denom."""

    denominator = abs(float(full_value) - float(neutral_value))
    if denominator <= 1e-12 or not loso_values:
        return (), False
    ratios = tuple(abs(float(full_value) - float(value)) / denominator for value in loso_values)
    return ratios, all(math.isfinite(value) and value <= 0.50 for value in ratios)


def evaluate_pilot(
    records: Sequence[object],
    *,
    scene_ids: Sequence[int],
    partition: PilotPartitionManifest,
    invalid_count: int = 0,
    duplicate_count: int = 0,
    identity_mismatch_count: int = 0,
    extension: bool = False,
) -> PilotDecision:
    """Evaluate development-only ranking and Boundary Lag pilot criteria."""

    if not isinstance(partition, PilotPartitionManifest):
        raise TypeError("pilot evaluator requires a frozen PilotPartitionManifest")
    for field_name, value in (
        ("invalid_count", invalid_count),
        ("duplicate_count", duplicate_count),
        ("identity_mismatch_count", identity_mismatch_count),
    ):
        if type(value) is not int or value < 0:
            raise ValueError(f"{field_name} must be a non-negative integer")
    scenes = _scene_ids(scene_ids)
    expected_scenes = (
        partition.primary_scene_ids + partition.extension_scene_ids
        if extension
        else partition.primary_scene_ids
    )
    if scenes != expected_scenes:
        return PilotDecision(
            status=V4_PILOT_BLOCKED_EXECUTION,
            reason_code="PILOT_PARTITION_SCOPE_MISMATCH",
            macro_auroc=None,
            pooled_boundary_lag_median=None,
            late_warning_proportion=None,
            models=(),
            invalid_count=invalid_count,
            duplicate_count=duplicate_count,
            identity_mismatch_count=identity_mismatch_count + 1,
        )
    try:
        lookup = _normalise_rows(records, scenes)
    except ValueError as exc:
        return PilotDecision(
            status=V4_PILOT_BLOCKED_EXECUTION,
            reason_code=str(exc),
            macro_auroc=None,
            pooled_boundary_lag_median=None,
            late_warning_proportion=None,
            models=(),
            invalid_count=invalid_count,
            duplicate_count=duplicate_count,
            identity_mismatch_count=identity_mismatch_count,
        )
    models = tuple(_model_metrics(lookup, model, scenes) for model in SCIENTIFIC_MODELS)
    if invalid_count or duplicate_count or identity_mismatch_count:
        return PilotDecision(
            status=V4_PILOT_BLOCKED_EXECUTION,
            reason_code="EXECUTION_INTEGRITY_FAILURE",
            macro_auroc=None,
            pooled_boundary_lag_median=None,
            late_warning_proportion=None,
            models=models,
            invalid_count=invalid_count,
            duplicate_count=duplicate_count,
            identity_mismatch_count=identity_mismatch_count,
        )
    if any(not model.defined for model in models):
        return PilotDecision(
            status=V4_PILOT_INCONCLUSIVE,
            reason_code="UNDEFINED_MODEL_METRIC",
            macro_auroc=None,
            pooled_boundary_lag_median=None,
            late_warning_proportion=None,
            models=models,
        )
    macro_auc = sum(model.auroc for model in models) / len(models)
    all_lags = [
        float(lag)
        for model in models
        for lag in _scene_lags(lookup, model.model_id, scenes)
    ]
    if not all_lags:
        return PilotDecision(
            status=V4_PILOT_INCONCLUSIVE,
            reason_code="NO_FAILURE_GROUPS",
            macro_auroc=macro_auc,
            pooled_boundary_lag_median=None,
            late_warning_proportion=None,
            models=models,
        )
    pooled_lag = float(median(all_lags))
    late = sum(lag > 0 for lag in all_lags) / len(all_lags)
    min_positive = math.ceil(2 * len(scenes) / 3)
    ranking_ok = macro_auc >= 0.60 and all(model.auroc > 0.50 for model in models)
    warning_ok = pooled_lag > 0 and late >= 0.60 and all(model.boundary_lag_median > 0 for model in models)
    direction_ok = all(
        model.positive_ranking_scene_count >= min_positive
        and model.positive_warning_scene_count >= min_positive
        for model in models
    )
    conflict = any(
        left * right < 0
        for left, right in (
            (models[0].auroc - 0.5, models[1].auroc - 0.5),
            (models[0].boundary_lag_median, models[1].boundary_lag_median),
            (models[0].late_warning_proportion - 0.5, models[1].late_warning_proportion - 0.5),
        )
    )
    loso_auc: list[float] = []
    loso_lag: list[float] = []
    for omitted in scenes:
        reduced = tuple(scene for scene in scenes if scene != omitted)
        if len(reduced) < 2:
            continue
        scoped = _normalise_rows([row for key, row in lookup.items() if key[1] in reduced], reduced)
        aucs = [_model_metrics(scoped, model, reduced).auroc for model in SCIENTIFIC_MODELS]
        lag_values = [lag for model in SCIENTIFIC_MODELS for lag in _scene_lags(scoped, model, reduced)]
        loso_auc.append(sum(aucs) / len(aucs))
        loso_lag.append(float(median(lag_values)) if lag_values else 0.0)
    loso_ok = bool(loso_auc) and all(value > 0.50 for value in loso_auc) and all(value > 0 for value in loso_lag)
    ranking_dominance, ranking_dominance_ok = _scene_dominance_ratios(macro_auc, 0.50, loso_auc)
    lag_dominance, lag_dominance_ok = _scene_dominance_ratios(pooled_lag, 0.0, loso_lag)
    scene_dominance_ok = ranking_dominance_ok and lag_dominance_ok
    if conflict or macro_auc < 0.55 or pooled_lag <= 0:
        status, reason = V4_PILOT_SCIENTIFIC_NO_GO, "PREREGISTERED_NO_GO"
    elif ranking_ok and warning_ok and direction_ok and loso_ok and scene_dominance_ok:
        status, reason = V4_PILOT_GO_TO_FULL_MVE, "ALL_PRIMARY_PILOT_CRITERIA_MET"
    else:
        status = V4_PILOT_INCONCLUSIVE
        reason = "SCENE_DOMINANCE_FAILURE" if not scene_dominance_ok else "PRIMARY_CRITERIA_NOT_DECISIVE"
    paper = PAPER_NO_PRIMARY_CLAIM
    if ranking_ok and warning_ok:
        paper = PAPER_RANKING_NOT_WARNING_SUPPORTED
    elif ranking_ok:
        paper = PAPER_RANKING_SUPPORTED_LAG_NOT_SUPPORTED
    return PilotDecision(
        status=status,
        reason_code=reason,
        macro_auroc=macro_auc,
        pooled_boundary_lag_median=pooled_lag,
        late_warning_proportion=late,
        models=models,
        loso_auroc=tuple(loso_auc),
        loso_boundary_lag=tuple(loso_lag),
        paper_claim_qualification=paper,
        scene_dominance_ranking=ranking_dominance,
        scene_dominance_lag=lag_dominance,
        scene_dominance_pass=scene_dominance_ok,
    )

@dataclass(frozen=True, slots=True)
class PilotGateDecision:
    """Gate-3 decision without launching an extension or full confirmation."""

    status: str
    reason_code: str
    primary_status: str
    extension_status: str | None
    extension_allowed: bool
    full_confirmation_forbidden: bool
    formal_confirmation_scene_count: int | None = None
    extension_authorized: bool = False

    def __post_init__(self) -> None:
        allowed_statuses = {
            V4_PILOT_GO_TO_FULL_MVE,
            V4_PILOT_SCIENTIFIC_NO_GO,
            V4_PILOT_INCONCLUSIVE,
            V4_PILOT_BLOCKED_EXECUTION,
            FULL_CONFIRMATION_FORBIDDEN,
        }
        if self.status not in allowed_statuses:
            raise ValueError("unsupported pilot gate status")
        if self.primary_status not in allowed_statuses:
            raise ValueError("unsupported primary pilot status")
        if self.extension_status is not None and self.extension_status not in allowed_statuses:
            raise ValueError("unsupported extension pilot status")
        if self.formal_confirmation_scene_count not in {None, 15, 17}:
            raise ValueError("formal confirmation scope must be 15 or 17 scenes")
        if self.full_confirmation_forbidden and self.formal_confirmation_scene_count is not None:
            raise ValueError("forbidden full confirmation cannot carry a formal scope")

    def to_dict(self) -> dict[str, object]:
        return {
            "status": self.status,
            "reason_code": self.reason_code,
            "primary_status": self.primary_status,
            "extension_status": self.extension_status,
            "extension_allowed": self.extension_allowed,
            "full_confirmation_forbidden": self.full_confirmation_forbidden,
            "formal_confirmation_scene_count": self.formal_confirmation_scene_count,
            "extension_authorized": self.extension_authorized,
            "scientific_result": "NO_SCIENTIFIC_RESULT",
        }


def evaluate_pilot_gate(
    primary: PilotDecision,
    *,
    extension: PilotDecision | None = None,
    power_15_pass: bool,
    power_17_pass: bool,
    extension_authorized: bool = False,
) -> PilotGateDecision:
    """Apply the pre-registered Pilot/Extension precedence rules.

    This is a pure decision function.  It never runs the extension and never
    promotes development evidence to a formal scientific result.
    """

    if primary.status == V4_PILOT_BLOCKED_EXECUTION or primary.execution_blocked:
        return PilotGateDecision(
            V4_PILOT_BLOCKED_EXECUTION,
            "PRIMARY_EXECUTION_INTEGRITY_FAILURE",
            primary.status,
            extension.status if extension else None,
            False,
            True,
            None,
            extension_authorized,
        )
    if primary.status == V4_PILOT_SCIENTIFIC_NO_GO:
        return PilotGateDecision(
            V4_PILOT_SCIENTIFIC_NO_GO,
            "PRIMARY_SCIENTIFIC_NO_GO",
            primary.status,
            extension.status if extension else None,
            False,
            True,
            None,
            extension_authorized,
        )
    if primary.status == V4_PILOT_GO_TO_FULL_MVE:
        if not power_17_pass:
            return PilotGateDecision(
                FULL_CONFIRMATION_FORBIDDEN,
                "POWER_17_SCENE_DESIGN_GATE_FAILED",
                primary.status,
                None,
                False,
                True,
                None,
                extension_authorized,
            )
        # Primary GO leaves the development extension unrun; formal scope is
        # the pre-registered 17-scene confirmation, subject to new auth.
        return PilotGateDecision(
            V4_PILOT_GO_TO_FULL_MVE,
            "PRIMARY_GO_FORMAL_17_SCENE_SCOPE",
            primary.status,
            None,
            False,
            False,
            17,
            extension_authorized,
        )
    # Primary INCONCLUSIVE: only a passing 15-scene power design may expose
    # the extension, and a fresh explicit authorization is required to run it.
    if not power_15_pass:
        return PilotGateDecision(
            FULL_CONFIRMATION_FORBIDDEN,
            "POWER_15_SCENE_DESIGN_GATE_FAILED",
            primary.status,
            None,
            False,
            True,
            None,
            extension_authorized,
        )
    if extension is None:
        return PilotGateDecision(
            V4_PILOT_INCONCLUSIVE,
            "EXTENSION_REQUIRES_NEW_AUTHORIZATION",
            primary.status,
            None,
            True,
            False,
            None,
            extension_authorized,
        )
    if (
        extension.status == V4_PILOT_BLOCKED_EXECUTION
        or extension.execution_blocked
    ):
        return PilotGateDecision(
            V4_PILOT_BLOCKED_EXECUTION,
            "EXTENSION_EXECUTION_INTEGRITY_FAILURE",
            primary.status,
            extension.status,
            False,
            True,
            None,
            extension_authorized,
        )
    if not extension_authorized:
        return PilotGateDecision(
            V4_PILOT_INCONCLUSIVE,
            "EXTENSION_REQUIRES_NEW_AUTHORIZATION",
            primary.status,
            extension.status,
            True,
            False,
            None,
            False,
        )
    if extension.status != V4_PILOT_GO_TO_FULL_MVE:
        return PilotGateDecision(
            FULL_CONFIRMATION_FORBIDDEN,
            "EXTENSION_NOT_GO_FULL_CONFIRMATION_FORBIDDEN",
            primary.status,
            extension.status,
            False,
            True,
            None,
            extension_authorized,
        )
    if not power_17_pass:
        return PilotGateDecision(
            FULL_CONFIRMATION_FORBIDDEN,
            "POWER_17_SCENE_DESIGN_GATE_FAILED",
            primary.status,
            extension.status,
            False,
            True,
            None,
            extension_authorized,
        )
    return PilotGateDecision(
        V4_PILOT_GO_TO_FULL_MVE,
        "EXTENSION_GO_FORMAL_17_SCENE_SCOPE",
        primary.status,
        extension.status,
        False,
        False,
        17,
        extension_authorized,
    )

def _scene_lags(lookup: Mapping[tuple[str, int, str], object], model: str, scenes: tuple[int, ...]) -> tuple[float, ...]:
    values: list[float] = []
    for scene in scenes:
        rows = [lookup[(model, scene, state)] for state in FOG_BOUNDARY_LAG_SEQUENCE]
        result = boundary_lag_for_scene(
            FOG_BOUNDARY_LAG_SEQUENCE,
            alarms=[bool(_row_value(row, "alarm")) for row in rows],
            pose_failures=[bool(_row_value(row, "pose_failure")) for row in rows],
        )
        if result.included and result.lag is not None:
            values.append(float(result.lag))
    return tuple(values)


@dataclass(frozen=True, slots=True)
class SyntheticPowerDesignManifest:
    """Pre-registered, outcome-independent synthetic power design.

    This object describes the design only; it never executes the simulation.  The
    eight sensitivity cells vary one factor at a time around the frozen main
    scenario, while the task-transfer family remains a non-directional null.
    """

    static_rank: float = 0.45
    crr: float = 0.05
    rwg: float = 0.40
    sfr: float = 0.40
    task_transfer_non_directional_null: bool = True
    scope_scene_counts: tuple[int, ...] = (15, 17)
    outer_replicates: int = 10_000
    inner_bootstrap_replicates: int = 10_000
    target_margin: float = 0.10
    scene_sd: float = 0.10
    cross_metric_corr: float = 0.30
    cross_model_corr: float = 0.60
    failure_prevalence: float = 0.65
    scene_sd_sensitivity: tuple[float, ...] = (0.05, 0.15)
    cross_metric_corr_sensitivity: tuple[float, ...] = (0.0, 0.60)
    cross_model_corr_sensitivity: tuple[float, ...] = (0.30, 0.90)
    failure_prevalence_sensitivity: tuple[float, ...] = (0.50, 0.80)
    # Kept under the legacy name for read-only consumers; it is sensitivity
    # values only, not a post-hoc selection grid containing the main value.
    failure_prevalence_grid: tuple[float, ...] = (0.50, 0.80)
    holm_method: str = "four-holm"
    streaming: bool = True
    allow_downsampling: bool = False
    batch_size: int = 128
    # Machine-readable generator, factor, and link metadata. These fields are
    # included in the identity payload so seeds cannot be reused with a drifted
    # synthetic data-generating process.
    generator_family: str = "GAUSSIAN_COPULA_SCENE_MODEL_EFFECTS"
    generator_version: str = "georeliab-synthetic-generator-v1"
    factor_specs: tuple[tuple[str, str], ...] = (
        ("scene_sd", "result_scale_scene_random_effect"),
        ("cross_metric_corr", "cross_metric_covariance"),
        ("cross_model_corr", "cross_model_covariance"),
        ("failure_prevalence", "correlated_failure_scene_bernoulli"),
    )
    link_specs: tuple[tuple[str, str], ...] = (
        ("static_rank", "fisher_z_to_result_scale[-1,1]"),
        ("crr", "fisher_z_to_result_scale[-1,1]"),
        ("sfr", "logit_to_probability"),
        ("rwg", "static_rank_minus_crr"),
        ("alarm", "mechanical_from_sfr"),
    )
    covariance_structure: str = "KRONECKER_CROSS_METRIC_X_CROSS_MODEL"
    failure_scene_process: str = "CORRELATED_BERNOULLI_ONSET_FOG_S1_S2_S3"
    alarm_link: str = "SFR_MECHANICAL_ALARM"
    seed_domain: str = "georeliab:power-seed:v1"
    seed_version: str = "counter-based-v1"
    schema_version: str = POWER_SCHEMA_VERSION
    identity_sha256: str = ""

    def __post_init__(self) -> None:
        if tuple(self.scope_scene_counts) != (15, 17):
            raise ValueError("power design must cover exactly the 15- and 17-scene scopes")
        for name in ("outer_replicates", "inner_bootstrap_replicates"):
            value = getattr(self, name)
            if type(value) is not int or value != 10_000:
                raise ValueError(f"{name} is preregistered at exactly 10000")
        if type(self.batch_size) is not int or self.batch_size <= 0:
            raise ValueError("batch_size must be a positive integer")
        if self.generator_family != "GAUSSIAN_COPULA_SCENE_MODEL_EFFECTS" or self.generator_version != "georeliab-synthetic-generator-v1":
            raise ValueError("synthetic generator metadata is not preregistered")
        expected_factor_specs = (
            ("scene_sd", "result_scale_scene_random_effect"),
            ("cross_metric_corr", "cross_metric_covariance"),
            ("cross_model_corr", "cross_model_covariance"),
            ("failure_prevalence", "correlated_failure_scene_bernoulli"),
        )
        expected_link_specs = (
            ("static_rank", "fisher_z_to_result_scale[-1,1]"),
            ("crr", "fisher_z_to_result_scale[-1,1]"),
            ("sfr", "logit_to_probability"),
            ("rwg", "static_rank_minus_crr"),
            ("alarm", "mechanical_from_sfr"),
        )
        if tuple(self.factor_specs) != expected_factor_specs or tuple(self.link_specs) != expected_link_specs:
            raise ValueError("synthetic factor/link metadata is not preregistered")
        if self.covariance_structure != "KRONECKER_CROSS_METRIC_X_CROSS_MODEL":
            raise ValueError("synthetic covariance structure is not preregistered")
        if self.failure_scene_process != "CORRELATED_BERNOULLI_ONSET_FOG_S1_S2_S3":
            raise ValueError("synthetic failure-scene process is not preregistered")
        if self.alarm_link != "SFR_MECHANICAL_ALARM":
            raise ValueError("synthetic alarm link is not preregistered")
        if self.seed_domain != "georeliab:power-seed:v1" or self.seed_version != "counter-based-v1":
            raise ValueError("synthetic seed domain/version is not preregistered")
        if self.holm_method != "four-holm":
            raise ValueError("power design requires exact four-Holm")
        if self.streaming is not True or self.allow_downsampling is not False:
            raise ValueError("power design requires streaming without downsampling")
        if self.task_transfer_non_directional_null is not True:
            raise ValueError("task-transfer family is a fixed non-directional null")
        for name in ("static_rank", "crr", "rwg", "sfr"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or not -1.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [-1,1]")
        if not math.isclose(self.rwg, self.static_rank - self.crr, rel_tol=0.0, abs_tol=1e-12):
            raise ValueError("RWG must be derived as StaticRank minus CRR")
        for name in ("target_margin", "scene_sd", "cross_metric_corr", "cross_model_corr", "failure_prevalence"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError(f"{name} must be in [0,1]")
        frozen_grids = {
            "scene_sd_sensitivity": (0.05, 0.15),
            "cross_metric_corr_sensitivity": (0.0, 0.60),
            "cross_model_corr_sensitivity": (0.30, 0.90),
            "failure_prevalence_sensitivity": (0.50, 0.80),
            "failure_prevalence_grid": (0.50, 0.80),
        }
        for name, expected in frozen_grids.items():
            if tuple(getattr(self, name)) != expected:
                raise ValueError(f"{name} is not the preregistered sensitivity grid")
        if self.identity_sha256:
            _valid_sha(self.identity_sha256, "identity_sha256")

    def payload(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "static_rank": self.static_rank,
            "crr": self.crr,
            "rwg": self.rwg,
            "sfr": self.sfr,
            "task_transfer_non_directional_null": self.task_transfer_non_directional_null,
            "scope_scene_counts": list(self.scope_scene_counts),
            "outer_replicates": self.outer_replicates,
            "inner_bootstrap_replicates": self.inner_bootstrap_replicates,
            "target_margin": self.target_margin,
            "scene_sd": self.scene_sd,
            "cross_metric_corr": self.cross_metric_corr,
            "cross_model_corr": self.cross_model_corr,
            "failure_prevalence": self.failure_prevalence,
            "scene_sd_sensitivity": list(self.scene_sd_sensitivity),
            "cross_metric_corr_sensitivity": list(self.cross_metric_corr_sensitivity),
            "cross_model_corr_sensitivity": list(self.cross_model_corr_sensitivity),
            "failure_prevalence_sensitivity": list(self.failure_prevalence_sensitivity),
            "failure_prevalence_grid": list(self.failure_prevalence_grid),
            "holm_method": self.holm_method,
            "streaming": self.streaming,
            "allow_downsampling": self.allow_downsampling,
            "batch_size": self.batch_size,
            "generator_family": self.generator_family,
            "generator_version": self.generator_version,
            "factor_specs": [list(item) for item in self.factor_specs],
            "link_specs": [list(item) for item in self.link_specs],
            "covariance_structure": self.covariance_structure,
            "failure_scene_process": self.failure_scene_process,
            "alarm_link": self.alarm_link,
            "seed_domain": self.seed_domain,
            "seed_version": self.seed_version,
        }

    def sensitivity_scenarios(self) -> tuple[dict[str, float | str], ...]:
        """Return the eight one-factor-at-a-time cells in stable order."""
        cells: list[dict[str, float | str]] = []
        for name, values in (
            ("scene_sd", self.scene_sd_sensitivity),
            ("cross_metric_corr", self.cross_metric_corr_sensitivity),
            ("cross_model_corr", self.cross_model_corr_sensitivity),
            ("failure_prevalence", self.failure_prevalence_sensitivity),
        ):
            for value in values:
                cells.append({"factor": name, "value": float(value)})
        return tuple(cells)

    def seed(self, *, scope_scene_count: int, scenario: str = "main") -> int:
        if scope_scene_count not in self.scope_scene_counts:
            raise ValueError("unknown power-design scope")
        if not isinstance(scenario, str) or not scenario:
            raise ValueError("scenario must be a non-empty string")
        digest = _sha_domain(
            self.seed_domain,
            {
                "seed_version": self.seed_version,
                "design": self.payload(),
                "scope": scope_scene_count,
                "scenario": scenario,
            },
        )
        return int(digest[:16], 16)


def build_synthetic_power_design(**overrides: object) -> SyntheticPowerDesignManifest:
    """Construct the frozen synthetic design; overrides are explicit and validated."""

    fields = {
        "static_rank": 0.45,
        "crr": 0.05,
        "rwg": 0.40,
        "sfr": 0.40,
        "task_transfer_non_directional_null": True,
        "scope_scene_counts": (15, 17),
        "outer_replicates": 10_000,
        "inner_bootstrap_replicates": 10_000,
        "target_margin": 0.10,
        "scene_sd": 0.10,
        "cross_metric_corr": 0.30,
        "cross_model_corr": 0.60,
        "failure_prevalence": 0.65,
        "scene_sd_sensitivity": (0.05, 0.15),
        "cross_metric_corr_sensitivity": (0.0, 0.60),
        "cross_model_corr_sensitivity": (0.30, 0.90),
        "failure_prevalence_sensitivity": (0.50, 0.80),
        "failure_prevalence_grid": (0.50, 0.80),
        "holm_method": "four-holm",
        "streaming": True,
        "allow_downsampling": False,
        "batch_size": 128,
        "generator_family": "GAUSSIAN_COPULA_SCENE_MODEL_EFFECTS",
        "generator_version": "georeliab-synthetic-generator-v1",
        "factor_specs": (
            ("scene_sd", "result_scale_scene_random_effect"),
            ("cross_metric_corr", "cross_metric_covariance"),
            ("cross_model_corr", "cross_model_covariance"),
            ("failure_prevalence", "correlated_failure_scene_bernoulli"),
        ),
        "link_specs": (
            ("static_rank", "fisher_z_to_result_scale[-1,1]"),
            ("crr", "fisher_z_to_result_scale[-1,1]"),
            ("sfr", "logit_to_probability"),
            ("rwg", "static_rank_minus_crr"),
            ("alarm", "mechanical_from_sfr"),
        ),
        "covariance_structure": "KRONECKER_CROSS_METRIC_X_CROSS_MODEL",
        "failure_scene_process": "CORRELATED_BERNOULLI_ONSET_FOG_S1_S2_S3",
        "alarm_link": "SFR_MECHANICAL_ALARM",
        "seed_domain": "georeliab:power-seed:v1",
        "seed_version": "counter-based-v1",
    }
    unknown = set(overrides) - set(fields)
    if unknown:
        raise TypeError(f"unknown synthetic power fields: {sorted(unknown)}")
    fields.update(overrides)
    manifest = SyntheticPowerDesignManifest(**fields)
    return SyntheticPowerDesignManifest(
        **fields,
        identity_sha256=_sha_domain("georeliab:synthetic-power-design:v1", manifest.payload()),
    )


@dataclass(frozen=True, slots=True)
class PaperClaimQualification:
    protocol_decision: str
    go_family: str
    qualification: str
    boundary_lag_supported: bool = False

    def __post_init__(self) -> None:
        if self.protocol_decision not in {PROTOCOL_DECISION_GO, PROTOCOL_DECISION_NO_GO}:
            raise ValueError("unsupported protocol decision")
        if self.go_family not in {GO_FAMILY_RANKING_WARNING, GO_FAMILY_TASK_TRANSFER, GO_FAMILY_BOTH, GO_FAMILY_NONE}:
            raise ValueError("unsupported gate family")
        if self.qualification not in {PAPER_RANKING_NOT_WARNING_SUPPORTED, PAPER_RANKING_SUPPORTED_LAG_NOT_SUPPORTED, PAPER_TASK_TRANSFER_ONLY_SUPPORTED, PAPER_NO_PRIMARY_CLAIM}:
            raise ValueError("unsupported paper claim qualification")


def qualify_boundary_lag_claim(
    *,
    model_lag_medians: Mapping[str, float] | Sequence[float],
    pooled_late_warning_proportion: float,
    pooled_median_lag_ci_lower: float,
    loso_lag_medians: Sequence[float],
) -> bool:
    """Return whether the preregistered warning-lag paper claim is qualified.

    This helper is deliberately stricter than the protocol gate: both model
    medians must be positive, the pooled late-warning proportion must reach
    60%, the pooled bootstrap CI must exclude zero on the lower side, and no
    leave-one-scene-out lag may reverse direction. Invalid or empty inputs
    fail closed to ``False``.
    """

    if isinstance(model_lag_medians, Mapping):
        model_values = tuple(model_lag_medians.values())
    else:
        model_values = tuple(model_lag_medians)
    try:
        model_values = tuple(float(value) for value in model_values)
        loso_values = tuple(float(value) for value in loso_lag_medians)
        late = float(pooled_late_warning_proportion)
        ci_lower = float(pooled_median_lag_ci_lower)
    except (TypeError, ValueError):
        return False
    if len(model_values) < 2 or not loso_values:
        return False
    if not all(math.isfinite(value) and value > 0.0 for value in model_values):
        return False
    if not math.isfinite(late) or late < 0.60:
        return False
    if not math.isfinite(ci_lower) or ci_lower <= 0.0:
        return False
    return all(math.isfinite(value) and value > 0.0 for value in loso_values)


def classify_paper_claim(
    *,
    protocol_decision: str,
    go_family: str,
    ranking_warning_pass: bool,
    boundary_lag_pass: bool,
) -> PaperClaimQualification:
    if protocol_decision != PROTOCOL_DECISION_GO:
        return PaperClaimQualification(protocol_decision, go_family, PAPER_NO_PRIMARY_CLAIM, False)
    if ranking_warning_pass and boundary_lag_pass:
        qualification = PAPER_RANKING_NOT_WARNING_SUPPORTED
    elif ranking_warning_pass:
        qualification = PAPER_RANKING_SUPPORTED_LAG_NOT_SUPPORTED
    elif go_family == GO_FAMILY_TASK_TRANSFER:
        qualification = PAPER_TASK_TRANSFER_ONLY_SUPPORTED
    else:
        qualification = PAPER_NO_PRIMARY_CLAIM
    return PaperClaimQualification(protocol_decision, go_family, qualification, bool(ranking_warning_pass and boundary_lag_pass))


__all__ = [
    "ScheduleIdentityManifest",
    "ConfirmationScopeAddendum",
    "PilotPartitionManifest",
    "build_pilot_partition",
    "PilotModelMetrics",
    "PilotDecision",
    "evaluate_pilot",
    "PilotGateDecision",
    "evaluate_pilot_gate",
    "SyntheticPowerDesignManifest",
    "build_synthetic_power_design",
    "PaperClaimQualification",
    "classify_paper_claim",
    "qualify_boundary_lag_claim",
    "V4_PILOT_GO_TO_FULL_MVE",
    "V4_PILOT_SCIENTIFIC_NO_GO",
    "V4_PILOT_INCONCLUSIVE",
    "V4_PILOT_BLOCKED_EXECUTION",
    "FULL_CONFIRMATION_FORBIDDEN",
    "PROTOCOL_DECISION_GO",
    "PROTOCOL_DECISION_NO_GO",
    "GO_FAMILY_RANKING_WARNING",
    "GO_FAMILY_TASK_TRANSFER",
    "GO_FAMILY_BOTH",
    "GO_FAMILY_NONE",
    "PAPER_RANKING_NOT_WARNING_SUPPORTED",
    "PAPER_RANKING_SUPPORTED_LAG_NOT_SUPPORTED",
    "PAPER_TASK_TRANSFER_ONLY_SUPPORTED",
    "PAPER_NO_PRIMARY_CLAIM",
]


def build_schedule_identity_manifest(
    raw_file_bytes: bytes,
    parsed_schedule: object,
    ordered_unit_ids: Sequence[str],
    *,
    schema_version: str = "georeliab-v4-scientific-schedule-1.0",
    canonicalizer_version: str = "georeliab-canonical-json-v1",
) -> ScheduleIdentityManifest:
    """Build a domain-separated manifest without comparing raw and semantic SHA."""
    if not isinstance(raw_file_bytes, bytes):
        raise TypeError("raw_file_bytes must be bytes")
    ordered = tuple(ordered_unit_ids)
    return ScheduleIdentityManifest.from_schedule_bytes(
        raw_file_bytes,
        parsed_schedule,
        ordered,
        schema_version=schema_version,
        canonicalizer_version=canonicalizer_version,
    )


def build_confirmation_scope_addendum(
    *,
    schedule_identity_sha256: str,
    protocol_sha256: str,
    role: str,
    scene_ids: Sequence[int],
) -> ConfirmationScopeAddendum:
    """Create one immutable scope addendum bound to protocol and schedule identity."""
    scenes = tuple(scene_ids)
    count = len(scenes) * len(SCIENTIFIC_MODELS) * len(SCIENTIFIC_STATES)
    scope_hash = _sha_domain(
        "georeliab:confirmation-scope:v1",
        {"role": role, "scene_ids": list(scenes), "schedule_identity_sha256": schedule_identity_sha256},
    )
    fields = {
        "schema_version": SCOPE_ADDENDUM_SCHEMA_VERSION,
        "schedule_identity_sha256": schedule_identity_sha256,
        "protocol_sha256": protocol_sha256,
        "role": role,
        "scene_ids": scenes,
        "expected_record_count": count,
        "scope_sha256": scope_hash,
    }
    return ConfirmationScopeAddendum(
        **fields,
        addendum_sha256=_sha_domain("georeliab:scope-addendum:v1", fields),
    )


@dataclass(frozen=True, slots=True)
class ScopedWarningEvidence:
    """A scope-bound reference to evidence; it carries no scientific values itself."""

    scope: ConfirmationScopeAddendum
    input_record_inventory_sha256: str
    input_record_count: int
    evidence: object
    source_attempt_id: str | None = None
    schema_version: str = SCOPED_SCHEMA_VERSION

    def __post_init__(self) -> None:
        _valid_sha(self.input_record_inventory_sha256, "input_record_inventory_sha256")
        if self.input_record_count != self.scope.expected_record_count:
            raise ValueError("scoped evidence count does not match scope addendum")
        if not isinstance(self.evidence, Mapping) and not hasattr(self.evidence, "to_dict"):
            raise ValueError("scoped evidence must be a mapping or strict evidence object")
        if self.source_attempt_id is not None:
            # Attempt-05 is permanently non-resumable and must never be used
            # as scoped/development evidence. Normalize separators and case
            # so aliases such as ``attempt_05`` or replay suffixes cannot
            # bypass the source quarantine.
            normalized_source = self.source_attempt_id.casefold().replace("-", "").replace("_", "")
            if "attempt05" in normalized_source:
                raise ValueError("Attempt-05 predictions are forbidden in scoped evidence")
            if any(
                marker in normalized_source
                for marker in ("localgate2", "gate2smoke", "recoverysmoke")
            ):
                raise ValueError(
                    "Gate 2 smoke predictions are forbidden in scoped evidence"
                )

    @property
    def schedule_identity_sha256(self) -> str:
        return self.scope.schedule_identity_sha256

    @property
    def scope_sha256(self) -> str:
        return self.scope.scope_sha256

    def to_dict(self) -> dict[str, object]:
        payload = self.evidence.to_dict() if hasattr(self.evidence, "to_dict") else dict(self.evidence)
        return {
            "schema_version": self.schema_version,
            "scope": self.scope.to_dict(),
            "input_record_inventory_sha256": self.input_record_inventory_sha256,
            "input_record_count": self.input_record_count,
            "source_attempt_id": self.source_attempt_id,
            "evidence": payload,
        }


def build_scoped_warning_evidence(
    evidence: object,
    *,
    scope: ConfirmationScopeAddendum,
    input_record_inventory_sha256: str,
    input_record_count: int,
    source_attempt_id: str | None = None,
) -> ScopedWarningEvidence:
    """Bind existing WarningEvidence (including full-20 parity evidence) to a scope."""
    return ScopedWarningEvidence(scope, input_record_inventory_sha256, input_record_count, evidence, source_attempt_id)


def _coverage_item_count(value: object, scene_ids: tuple[int, ...]) -> int | None:
    """Extract and validate one model's declared record coverage."""

    expected_scene_ids = set(scene_ids)
    expected_per_scene = len(SCIENTIFIC_STATES)
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value if value >= 0 else None
    if isinstance(value, Mapping):
        declared_scenes = value.get("scene_ids")
        if declared_scenes is not None:
            try:
                if set(int(scene) for scene in declared_scenes) != expected_scene_ids:
                    return None
            except (TypeError, ValueError):
                return None
        for key in ("record_count", "expected_record_count", "count", "completed_count", "n_records"):
            declared = value.get(key)
            if isinstance(declared, int) and not isinstance(declared, bool):
                return declared if declared >= 0 else None
        declared_states = value.get("state_ids")
        if declared_scenes is not None and declared_states is not None:
            try:
                if set(str(state) for state in declared_states) == set(SCIENTIFIC_STATES):
                    return len(scene_ids) * expected_per_scene
            except (TypeError, ValueError):
                return None
        nested_records = value.get("records")
        if nested_records is not None:
            return _coverage_item_count(nested_records, scene_ids)
        # A nested scene -> count mapping is accepted only when every selected
        # scene is present and each scene covers all ten frozen states.
        scene_counts: dict[int, int] = {}
        for key, item in value.items():
            try:
                scene = int(key)
            except (TypeError, ValueError):
                continue
            if scene in expected_scene_ids and isinstance(item, int) and not isinstance(item, bool):
                scene_counts[scene] = item
        if scene_counts:
            if set(scene_counts) != expected_scene_ids or any(
                count != expected_per_scene for count in scene_counts.values()
            ):
                return None
            return sum(scene_counts.values())
        return None
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        identities: set[tuple[object, ...]] = set()
        for item in value:
            if isinstance(item, Mapping):
                try:
                    if "model_id" in item:
                        identity = (
                            str(item["model_id"]),
                            int(item["scene_id"]),
                            str(item["state_id"]),
                        )
                    else:
                        identity = (
                            int(item["scene_id"]),
                            str(item["state_id"]),
                        )
                except (KeyError, TypeError, ValueError):
                    return None
                if identity in identities:
                    return None
                identities.add(identity)
            else:
                identities.add((repr(item),))
        return len(identities)
    return None


def _coverage_by_model(
    coverage: object,
    scene_ids: tuple[int, ...],
) -> tuple[tuple[tuple[str, int], ...], bool]:
    """Normalize direct or nested coverage declarations fail-closed."""

    if isinstance(coverage, Mapping):
        candidate = coverage.get("by_model", coverage.get("models", coverage))
        if not isinstance(candidate, Mapping):
            return (), False
        if set(str(key) for key in candidate) != set(SCIENTIFIC_MODELS):
            return (), False
        counts = tuple(
            (model, _coverage_item_count(candidate[model], scene_ids) or -1)
            for model in SCIENTIFIC_MODELS
        )
    elif isinstance(coverage, Sequence) and not isinstance(coverage, (str, bytes, bytearray)):
        grouped: dict[str, list[Mapping[str, object]]] = {model: [] for model in SCIENTIFIC_MODELS}
        seen: set[tuple[str, int, str]] = set()
        for item in coverage:
            if not isinstance(item, Mapping):
                return (), False
            try:
                identity = (str(item["model_id"]), int(item["scene_id"]), str(item["state_id"]))
            except (KeyError, TypeError, ValueError):
                return (), False
            if identity in seen or identity[0] not in grouped or identity[1] not in scene_ids or identity[2] not in SCIENTIFIC_STATES:
                return (), False
            seen.add(identity)
            grouped[identity[0]].append(item)
        counts = tuple((model, len(grouped[model])) for model in SCIENTIFIC_MODELS)
    else:
        return (), False
    expected = len(scene_ids) * len(SCIENTIFIC_STATES)
    valid = all(count == expected for _, count in counts)
    return counts, valid


def _decision_status(value: object | None) -> object:
    status = getattr(value, "status", None)
    if status is None and isinstance(value, Mapping):
        status = value.get("status")
    return status


def _parity_agrees(result: object, formal_decision: object | None) -> bool:
    """Compare a parity callback result to the original evaluator decision."""

    if isinstance(result, bool):
        return result
    if isinstance(result, Mapping) and isinstance(result.get("matches"), bool):
        return bool(result["matches"])
    if formal_decision is None:
        return False
    if result is formal_decision:
        return True
    result_payload = result.to_dict() if hasattr(result, "to_dict") else result
    formal_payload = formal_decision.to_dict() if hasattr(formal_decision, "to_dict") else formal_decision
    if isinstance(result_payload, Mapping) and isinstance(formal_payload, Mapping):
        for key in ("status", "reason_code", "strong_model_id", "strong_family"):
            if key in result_payload or key in formal_payload:
                if result_payload.get(key) != formal_payload.get(key):
                    return False
        return True
    result_status = _decision_status(result)
    formal_status = _decision_status(formal_decision)
    result_reason = getattr(result, "reason_code", None)
    formal_reason = getattr(formal_decision, "reason_code", None)
    return result_status == formal_status and (
        result_reason is None or formal_reason is None or result_reason == formal_reason
    )


@dataclass(frozen=True, slots=True)
class ScopedWarningGateDecision:
    protocol_decision: str
    reason_code: str
    formal_decision: object | None = None
    scope_sha256: str | None = None
    scene_ids: tuple[int, ...] = ()
    expected_count: int | None = None
    input_record_count: int | None = None
    coverage: tuple[tuple[str, int], ...] = ()
    coverage_valid: bool = False
    parity_required: bool = False
    parity_checked: bool = False

    def to_dict(self) -> dict[str, object]:
        if hasattr(self.formal_decision, "to_dict"):
            payload = self.formal_decision.to_dict()
        elif isinstance(self.formal_decision, Mapping):
            payload = dict(self.formal_decision)
        elif self.formal_decision is None:
            payload = None
        else:
            payload = {
                key: getattr(self.formal_decision, key)
                for key in ("status", "reason_code", "strong_model_id", "strong_family")
                if hasattr(self.formal_decision, key)
            }
        if isinstance(payload, Mapping):
            payload = dict(payload)
            payload["scientific_result"] = "NO_SCIENTIFIC_RESULT"
        return {
            "protocol_decision": self.protocol_decision,
            "status": self.protocol_decision,
            "reason_code": self.reason_code,
            "scope_sha256": self.scope_sha256,
            "scene_ids": list(self.scene_ids),
            "expected_count": self.expected_count,
            "expected_record_count": self.expected_count,
            "input_record_count": self.input_record_count,
            "coverage": {model: count for model, count in self.coverage},
            "coverage_valid": self.coverage_valid,
            "parity_required": self.parity_required,
            "parity_checked": self.parity_checked,
            "formal_decision": payload,
            "scientific_result": "NO_SCIENTIFIC_RESULT",
        }


def evaluate_scoped_warning_gate(
    scoped_evidence: ScopedWarningEvidence,
    *,
    formal_decision: object | None = None,
    scene_ids: Sequence[int] | None = None,
    expected_count: int | None = None,
    coverage: object | None = None,
    parity_evaluator: object | None = None,
) -> ScopedWarningGateDecision:
    """Validate scope closure before preserving the original gate decision.

    This wrapper never recomputes or relaxes the frozen warning gate.  It only
    proves that the requested scene scope and per-model record coverage close
    exactly; a 20-scene scope additionally requires an explicit parity callback
    whose result agrees with the original evaluator.
    """

    if not isinstance(scoped_evidence, ScopedWarningEvidence):
        raise TypeError("scoped_evidence must be ScopedWarningEvidence")
    scenes = tuple(scoped_evidence.scope.scene_ids if scene_ids is None else scene_ids)
    try:
        normalized_scenes = _scene_ids(scenes)
    except (TypeError, ValueError):
        normalized_scenes = ()
    expected = len(normalized_scenes) * len(SCIENTIFIC_MODELS) * len(SCIENTIFIC_STATES)
    requested_expected = expected if expected_count is None else expected_count
    counts, coverage_valid = _coverage_by_model(coverage, normalized_scenes) if coverage is not None else ((), False)
    parity_required = len(normalized_scenes) == len(TEST_SCENE_IDS)
    parity_checked = False
    if normalized_scenes != tuple(scoped_evidence.scope.scene_ids):
        reason = "SCOPED_SCENE_SCOPE_MISMATCH"
    elif requested_expected != expected or scoped_evidence.input_record_count != expected:
        reason = "SCOPED_EXPECTED_COUNT_MISMATCH"
    elif not coverage_valid:
        reason = "SCOPED_COVERAGE_MISMATCH"
    elif parity_required and parity_evaluator is None:
        reason = "SCOPED_20_SCENE_PARITY_REQUIRED"
    else:
        reason = "ORIGINAL_GATE_DECISION_PRESERVED"
        if parity_required:
            try:
                parity_checked = _parity_agrees(parity_evaluator(scoped_evidence), formal_decision)
            except Exception:
                parity_checked = False
            if not parity_checked:
                reason = "SCOPED_20_SCENE_PARITY_MISMATCH"
    status = _decision_status(formal_decision)
    protocol = PROTOCOL_DECISION_GO if reason == "ORIGINAL_GATE_DECISION_PRESERVED" and status in {"MVE_GO_TO_EXTERNAL_VALIDATION", PROTOCOL_DECISION_GO} else PROTOCOL_DECISION_NO_GO
    return ScopedWarningGateDecision(
        protocol,
        reason,
        formal_decision,
        scoped_evidence.scope_sha256,
        normalized_scenes,
        requested_expected,
        scoped_evidence.input_record_count,
        counts,
        coverage_valid,
        parity_required,
        parity_checked,
    )


def build_pilot_partition_manifest(*args: object, **kwargs: object) -> PilotPartitionManifest:
    """Compatibility alias with an explicit manifest-oriented name."""
    return build_pilot_partition(*args, **kwargs)
__all__.extend([
    "build_schedule_identity_manifest",
    "build_confirmation_scope_addendum",
    "ScopedWarningEvidence",
    "build_scoped_warning_evidence",
    "ScopedWarningGateDecision",
    "evaluate_scoped_warning_gate",
    "build_pilot_partition_manifest",
])
