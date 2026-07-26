"""Versioned artifact contracts frozen by the approved dual-MVE plan.

The contracts are deliberately strict. Invalid predictions are preserved as
failures instead of being silently filtered, and fixture artifacts can never be
mistaken for scientific evidence.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
import json
import math
from pathlib import Path
import re
from typing import Any, ClassVar, Mapping, Self


SCHEMA_VERSION = "1.0"
_SLUG_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


class ContractError(ValueError):
    """Raised when an artifact violates a frozen contract."""


class RunMode(str, Enum):
    REAL = "real"
    FIXTURE = "fixture"


class ScientificValidity(str, Enum):
    SCIENTIFIC = "SCIENTIFIC"
    NON_SCIENTIFIC_FIXTURE = "NON_SCIENTIFIC_FIXTURE"


def _require_slug(value: str, field_name: str) -> str:
    if not isinstance(value, str) or not _SLUG_RE.fullmatch(value):
        raise ContractError(
            f"{field_name} must match {_SLUG_RE.pattern!r}; got {value!r}"
        )
    return value


def _require_finite(value: float, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ContractError(f"{field_name} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ContractError(f"{field_name} must be finite")
    return result


@dataclass(frozen=True, slots=True)
class SampleKey:
    """Canonical sample identity.

    Wire format:
    ``dataset/split/scene/view_set/condition/severity/seed``.
    """

    dataset: str
    split: str
    scene: str
    view_set: str
    condition: str
    severity: str
    seed: str

    PART_COUNT: ClassVar[int] = 7

    def __post_init__(self) -> None:
        for name in (
            "dataset",
            "split",
            "scene",
            "view_set",
            "condition",
            "severity",
            "seed",
        ):
            _require_slug(getattr(self, name), name)

    def __str__(self) -> str:
        return "/".join(
            (
                self.dataset,
                self.split,
                self.scene,
                self.view_set,
                self.condition,
                self.severity,
                self.seed,
            )
        )

    @classmethod
    def parse(cls, value: str) -> Self:
        if not isinstance(value, str):
            raise ContractError("sample_key must be a string")
        parts = value.split("/")
        if len(parts) != cls.PART_COUNT:
            raise ContractError(
                "sample_key must contain exactly seven slash-separated fields: "
                "dataset/split/scene/view_set/condition/severity/seed"
            )
        return cls(*parts)


@dataclass(frozen=True, slots=True)
class RunManifest:
    """Provenance required for every model invocation."""

    run_id: str
    mode: RunMode
    scientific_validity: ScientificValidity
    model: str
    checkpoint_hash: str
    dataset: str
    split: str
    seed: int
    intervention_version: str
    corruption_version: str
    environment: Mapping[str, str]
    rgb_digest: str
    prompt_digest: str
    decoder_digest: str
    schema_version: str = SCHEMA_VERSION

    def __post_init__(self) -> None:
        if not isinstance(self.mode, RunMode):
            raise ContractError('mode must be a RunMode')
        if not isinstance(self.scientific_validity, ScientificValidity):
            raise ContractError('scientific_validity must be a ScientificValidity')
        _require_slug(self.run_id, "run_id")
        _require_slug(self.model, "model")
        _require_slug(self.dataset, "dataset")
        _require_slug(self.split, "split")
        _require_slug(self.intervention_version, "intervention_version")
        _require_slug(self.corruption_version, "corruption_version")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int) or self.seed < 0:
            raise ContractError("seed must be a non-negative integer")
        if self.mode is RunMode.REAL:
            if self.scientific_validity is not ScientificValidity.SCIENTIFIC:
                raise ContractError("real runs must be marked SCIENTIFIC")
            if not _SHA256_RE.fullmatch(self.checkpoint_hash):
                raise ContractError(
                    "real-run checkpoint_hash must be a lowercase SHA-256 digest"
                )
        elif self.scientific_validity is not ScientificValidity.NON_SCIENTIFIC_FIXTURE:
            raise ContractError(
                "fixture runs must be marked NON_SCIENTIFIC_FIXTURE"
            )
        for name in ("rgb_digest", "prompt_digest", "decoder_digest"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value:
                raise ContractError(f"{name} must be a non-empty string")
        if not isinstance(self.environment, Mapping) or not self.environment:
            raise ContractError("environment must be a non-empty mapping")
        if any(
            not isinstance(key, str)
            or not key
            or not isinstance(value, str)
            or not value
            for key, value in self.environment.items()
        ):
            raise ContractError("environment keys and values must be non-empty strings")
        if self.schema_version != SCHEMA_VERSION:
            raise ContractError(
                f"unsupported schema_version {self.schema_version!r}; "
                f"expected {SCHEMA_VERSION!r}"
            )

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["mode"] = self.mode.value
        result["scientific_validity"] = self.scientific_validity.value
        result["environment"] = dict(self.environment)
        return result

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> Self:
        data = dict(value)
        try:
            data["mode"] = RunMode(data["mode"])
            data["scientific_validity"] = ScientificValidity(
                data["scientific_validity"]
            )
            return cls(**data)
        except KeyError as exc:
            raise ContractError(f"missing RunManifest field: {exc.args[0]}") from exc
        except TypeError as exc:
            raise ContractError(f"invalid RunManifest fields: {exc}") from exc


@dataclass(frozen=True, slots=True)
class PredictionArtifact:
    """Locations and runtime metadata for one prediction."""

    run_id: str
    sample_key: str
    geometry_prediction_uri: str
    native_confidence_uri: str
    valid_mask_uri: str
    hook_location: str | None
    runtime_seconds: float
    peak_memory_mb: float
    invalid_prediction: bool = False
    schema_version: str = SCHEMA_VERSION

    def __post_init__(self) -> None:
        _require_slug(self.run_id, 'run_id')
        SampleKey.parse(self.sample_key)
        if type(self.invalid_prediction) is not bool:
            raise ContractError('invalid_prediction must be boolean')
        for name in (
            "geometry_prediction_uri",
            "native_confidence_uri",
            "valid_mask_uri",
        ):
            value = getattr(self, name)
            if not isinstance(value, str) or not value:
                raise ContractError(f"{name} must be a non-empty URI")
        if self.hook_location is not None and (
            not isinstance(self.hook_location, str) or not self.hook_location
        ):
            raise ContractError("hook_location must be null or a non-empty string")
        if _require_finite(self.runtime_seconds, "runtime_seconds") < 0:
            raise ContractError("runtime_seconds must be non-negative")
        if _require_finite(self.peak_memory_mb, "peak_memory_mb") < 0:
            raise ContractError("peak_memory_mb must be non-negative")
        if self.schema_version != SCHEMA_VERSION:
            raise ContractError("unsupported PredictionArtifact schema_version")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> Self:
        try:
            return cls(**dict(value))
        except TypeError as exc:
            raise ContractError(f"invalid PredictionArtifact fields: {exc}") from exc


@dataclass(frozen=True, slots=True)
class AuditRecord:
    """Evaluation record; invalid model output is retained as a failed sample."""

    run_id: str
    sample_key: str
    gt_error: float | None
    failure_label: bool
    selection_score: float
    coverage: float
    accepted: bool
    downstream_outcome: float | None
    invalid_prediction: bool = False
    metadata: Mapping[str, str] = field(default_factory=dict)
    schema_version: str = SCHEMA_VERSION

    def __post_init__(self) -> None:
        _require_slug(self.run_id, 'run_id')
        SampleKey.parse(self.sample_key)
        for name in ('failure_label', 'accepted', 'invalid_prediction'):
            if type(getattr(self, name)) is not bool:
                raise ContractError(f'{name} must be boolean')
        if self.gt_error is not None:
            _require_finite(self.gt_error, "gt_error")
        _require_finite(self.selection_score, "selection_score")
        coverage = _require_finite(self.coverage, "coverage")
        if not 0 <= coverage <= 1:
            raise ContractError("coverage must be in [0, 1]")
        if self.downstream_outcome is not None:
            _require_finite(self.downstream_outcome, "downstream_outcome")
        if self.invalid_prediction and not self.failure_label:
            raise ContractError("invalid predictions must be counted as failures")
        if self.invalid_prediction and self.accepted:
            raise ContractError("invalid predictions cannot be accepted")
        if any(
            not isinstance(key, str)
            or not key
            or not isinstance(value, str)
            or not value
            for key, value in self.metadata.items()
        ):
            raise ContractError("metadata keys and values must be non-empty strings")
        if self.schema_version != SCHEMA_VERSION:
            raise ContractError("unsupported AuditRecord schema_version")

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["metadata"] = dict(self.metadata)
        return result

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> Self:
        try:
            return cls(**dict(value))
        except TypeError as exc:
            raise ContractError(f"invalid AuditRecord fields: {exc}") from exc


def write_json_artifact(
    path: Path, artifact: RunManifest | PredictionArtifact | AuditRecord
) -> None:
    """Write one artifact using a stable, human-readable JSON representation."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(artifact.to_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def read_json_artifact(
    path: Path,
    artifact_type: type[RunManifest] | type[PredictionArtifact] | type[AuditRecord],
) -> RunManifest | PredictionArtifact | AuditRecord:
    """Read and validate a JSON artifact."""

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ContractError(f"cannot read artifact {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ContractError("artifact JSON root must be an object")
    return artifact_type.from_dict(payload)


def validate_artifact_linkage(
    manifest: RunManifest,
    prediction: PredictionArtifact,
    audit: AuditRecord,
) -> None:
    """Ensure records cannot cross-link runs, samples, or invalidity state."""

    if prediction.run_id != manifest.run_id or audit.run_id != manifest.run_id:
        raise ContractError('artifact run_id does not match RunManifest')
    if prediction.sample_key != audit.sample_key:
        raise ContractError('PredictionArtifact and AuditRecord sample_key mismatch')
    sample_key = SampleKey.parse(prediction.sample_key)
    if sample_key.dataset != manifest.dataset or sample_key.split != manifest.split:
        raise ContractError(
            "sample_key dataset/split does not match RunManifest: "
            f"{sample_key.dataset}/{sample_key.split} != "
            f"{manifest.dataset}/{manifest.split}"
        )
    if sample_key.seed != str(manifest.seed):
        raise ContractError(
            "sample_key seed does not match RunManifest: "
            f"{sample_key.seed} != {manifest.seed}"
        )
    if prediction.invalid_prediction != audit.invalid_prediction:
        raise ContractError(
            "PredictionArtifact and AuditRecord invalid_prediction mismatch"
        )

