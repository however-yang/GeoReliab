"""Versioned artifact contracts frozen by the approved dual-MVE plan.

The contracts are deliberately strict. Invalid predictions are preserved as
failures instead of being silently filtered, and fixture artifacts can never be
mistaken for scientific evidence.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Any, ClassVar, Mapping, Self
from urllib.parse import unquote, urlparse

import numpy as np


SCHEMA_VERSION = '1.1'
LEGACY_SCHEMA_VERSION = '1.0'
_SLUG_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_GIT_HASH_RE = re.compile(r'^[0-9a-f]{40,64}$')


class ContractError(ValueError):
    """Raised when an artifact violates a frozen contract."""


class RunMode(str, Enum):
    REAL = "real"
    SMOKE = 'smoke'
    FIXTURE = "fixture"


class ScientificValidity(str, Enum):
    SCIENTIFIC = "SCIENTIFIC"
    NON_SCIENTIFIC_SMOKE = 'NON_SCIENTIFIC_SMOKE'
    NON_SCIENTIFIC_FIXTURE = "NON_SCIENTIFIC_FIXTURE"


@dataclass(frozen=True, slots=True)
class ScientificProvenance:
    '''Immutable source and input provenance required for real-model runs.'''

    project_commit: str
    project_tree: str
    model_source_commit: str
    environment_lock_sha256: str
    corruption_manifest_sha256: str
    split_view_manifest_sha256: str
    dust3r_source_commit: str | None = None
    croco_source_commit: str | None = None

    def __post_init__(self) -> None:
        for name in ('project_commit', 'project_tree', 'model_source_commit'):
            if not _GIT_HASH_RE.fullmatch(getattr(self, name)):
                raise ContractError(f'{name} must be a lowercase git object hash')
        for name in (
            'environment_lock_sha256',
            'corruption_manifest_sha256',
            'split_view_manifest_sha256',
        ):
            if not _SHA256_RE.fullmatch(getattr(self, name)):
                raise ContractError(f'{name} must be a lowercase SHA-256 digest')
        for name in ('dust3r_source_commit', 'croco_source_commit'):
            value = getattr(self, name)
            if value is not None and not _GIT_HASH_RE.fullmatch(value):
                raise ContractError(f'{name} must be null or a lowercase git object hash')

    def to_dict(self) -> dict[str, str | None]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> Self:
        try:
            return cls(**dict(value))
        except (TypeError, KeyError) as exc:
            raise ContractError(f'invalid scientific provenance: {exc}') from exc


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
    provenance: ScientificProvenance | None = None
    schema_version: str = SCHEMA_VERSION
    _legacy_v1_0: bool = field(default=False, repr=False, compare=False)

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
                raise ContractError('real runs must be marked SCIENTIFIC')
        elif self.mode is RunMode.SMOKE:
            if self.scientific_validity is not ScientificValidity.NON_SCIENTIFIC_SMOKE:
                raise ContractError('smoke runs must be marked NON_SCIENTIFIC_SMOKE')
        elif self.scientific_validity is not ScientificValidity.NON_SCIENTIFIC_FIXTURE:
            raise ContractError(
                'fixture runs must be marked NON_SCIENTIFIC_FIXTURE'
            )
        if self.mode in (RunMode.REAL, RunMode.SMOKE):
            if not _SHA256_RE.fullmatch(self.checkpoint_hash):
                raise ContractError(
                    'real/smoke checkpoint_hash must be a lowercase SHA-256 digest'
                )
            if not self._legacy_v1_0 and self.provenance is None:
                raise ContractError('real/smoke runs require full scientific provenance')
            if self.provenance is not None and not isinstance(
                self.provenance, ScientificProvenance
            ):
                raise ContractError('provenance must be ScientificProvenance')
            if self.model == 'MASt3R' and self.provenance is not None and (
                self.provenance.dust3r_source_commit is None
                or self.provenance.croco_source_commit is None
            ):
                raise ContractError('MASt3R provenance requires DUSt3R and CroCo commits')
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
        if self._legacy_v1_0 and self.mode in (RunMode.REAL, RunMode.SMOKE):
            raise ContractError('legacy real/smoke artifacts cannot be rewritten without provenance')
        result = asdict(self)
        result.pop('_legacy_v1_0')
        result['mode'] = self.mode.value
        result['scientific_validity'] = self.scientific_validity.value
        result['environment'] = dict(self.environment)
        return result

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> Self:
        data = dict(value)
        try:
            source_schema = data.get('schema_version', LEGACY_SCHEMA_VERSION)
            if source_schema not in (LEGACY_SCHEMA_VERSION, SCHEMA_VERSION):
                raise ContractError(f'unsupported schema_version {source_schema!r}')
            if source_schema == LEGACY_SCHEMA_VERSION and data.get('mode') == RunMode.SMOKE.value:
                raise ContractError('v1.0 artifacts cannot use smoke mode')
            data['schema_version'] = SCHEMA_VERSION
            data['_legacy_v1_0'] = source_schema == LEGACY_SCHEMA_VERSION
            data['mode'] = RunMode(data['mode'])
            data['scientific_validity'] = ScientificValidity(
                data['scientific_validity']
            )
            if data.get('provenance') is not None:
                data['provenance'] = ScientificProvenance.from_dict(data['provenance'])
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
    payload_digests: Mapping[str, str] = field(default_factory=dict)
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
        if not isinstance(self.payload_digests, Mapping):
            raise ContractError('payload_digests must be a mapping')
        allowed_digests = {
            'geometry_prediction_uri',
            'native_confidence_uri',
            'valid_mask_uri',
        }
        if set(self.payload_digests) - allowed_digests:
            raise ContractError('payload_digests contains an unknown payload key')
        if any(
            not isinstance(key, str)
            or not isinstance(value, str)
            or (value and not _SHA256_RE.fullmatch(value))
            for key, value in self.payload_digests.items()
        ):
            raise ContractError('payload digests must be empty or lowercase SHA-256 digests')
        if self.schema_version != SCHEMA_VERSION:
            raise ContractError("unsupported PredictionArtifact schema_version")

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result['payload_digests'] = dict(self.payload_digests)
        return result

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> Self:
        try:
            data = dict(value)
            source_schema = data.get('schema_version', LEGACY_SCHEMA_VERSION)
            if source_schema not in (LEGACY_SCHEMA_VERSION, SCHEMA_VERSION):
                raise ContractError(f'unsupported schema_version {source_schema!r}')
            data['schema_version'] = SCHEMA_VERSION
            return cls(**data)
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
            data = dict(value)
            source_schema = data.get('schema_version', LEGACY_SCHEMA_VERSION)
            if source_schema not in (LEGACY_SCHEMA_VERSION, SCHEMA_VERSION):
                raise ContractError(f'unsupported schema_version {source_schema!r}')
            data['schema_version'] = SCHEMA_VERSION
            return cls(**data)
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


def _file_uri_path(uri: str, field_name: str) -> str:
    parsed = urlparse(uri)
    if parsed.scheme != 'file' or parsed.netloc not in ('', 'localhost'):
        raise ContractError(f'{field_name} must be a local file URI for validation')
    payload_path = unquote(parsed.path)
    if re.match(r'^/[A-Za-z]:/', payload_path):
        payload_path = payload_path[1:]
    return payload_path


def _local_payload_path(uri: str, field_name: str) -> Path:
    return Path(_file_uri_path(uri, field_name))


def _load_npz(uri: str, field_name: str) -> dict[str, np.ndarray]:
    path = _local_payload_path(uri, field_name)
    try:
        with np.load(path, allow_pickle=False) as payload:
            return {name: payload[name] for name in payload.files}
    except (OSError, ValueError) as exc:
        raise ContractError(f'cannot read {field_name} NPZ payload: {exc}') from exc


def _require_payload_keys(
    payload: Mapping[str, np.ndarray], keys: tuple[str, ...], field_name: str
) -> None:
    missing = sorted(set(keys) - set(payload))
    if missing:
        raise ContractError(f'{field_name} missing required payload keys: {missing}')


def _require_finite_array(value: np.ndarray, field_name: str) -> None:
    if not np.issubdtype(value.dtype, np.number) or not np.all(np.isfinite(value)):
        raise ContractError(f'{field_name} must contain only finite numeric values')


def _validate_payload_digest(uri: str, expected: str, field_name: str) -> None:
    if not expected:
        return
    path = _local_payload_path(uri, field_name)
    try:
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError as exc:
        raise ContractError(f'cannot hash {field_name}: {exc}') from exc
    if actual != expected:
        raise ContractError(f'{field_name} payload digest mismatch')


def validate_artifact_bundle(
    manifest: RunManifest,
    prediction: PredictionArtifact,
    audit: AuditRecord,
) -> None:
    '''Fail closed on cross-links, payload schema violations, and digest drift.'''

    validate_artifact_linkage(manifest, prediction, audit)
    geometry = _load_npz(prediction.geometry_prediction_uri, 'geometry_prediction_uri')
    confidence = _load_npz(prediction.native_confidence_uri, 'native_confidence_uri')
    mask_payload = _load_npz(prediction.valid_mask_uri, 'valid_mask_uri')
    _require_payload_keys(
        geometry,
        ('points_world', 'camera_c2w', 'intrinsics', 'pixel_xy', 'view_id'),
        'geometry_prediction_uri',
    )
    _require_payload_keys(confidence, ('raw_confidence',), 'native_confidence_uri')
    _require_payload_keys(mask_payload, ('valid_mask',), 'valid_mask_uri')
    points = geometry['points_world']
    pixels = geometry['pixel_xy']
    view_id = geometry['view_id']
    camera_c2w = geometry['camera_c2w']
    intrinsics = geometry['intrinsics']
    if points.ndim != 2 or points.shape[1] != 3:
        raise ContractError('points_world must have shape (N, 3)')
    if pixels.shape != (len(points), 2) or view_id.shape != (len(points),):
        raise ContractError('geometry point, pixel_xy, and view_id shapes are incompatible')
    if camera_c2w.ndim != 3 or camera_c2w.shape[1:] != (4, 4):
        raise ContractError('camera_c2w must have shape (V, 4, 4)')
    if intrinsics.shape != (camera_c2w.shape[0], 3, 3):
        raise ContractError('intrinsics must have shape (V, 3, 3) matching camera_c2w')
    for value, name in (
        (points, 'points_world'),
        (pixels, 'pixel_xy'),
        (camera_c2w, 'camera_c2w'),
        (intrinsics, 'intrinsics'),
    ):
        _require_finite_array(value, name)
    if not np.issubdtype(view_id.dtype, np.integer) or np.any(view_id < 0) or np.any(view_id >= len(camera_c2w)):
        raise ContractError('view_id must contain valid integer camera indexes')
    raw_confidence = confidence['raw_confidence']
    valid_mask = mask_payload['valid_mask']
    if raw_confidence.shape != (len(points),) or valid_mask.shape != (len(points),):
        raise ContractError('native confidence and valid mask shapes must match points_world')
    if valid_mask.dtype != np.bool_:
        raise ContractError('valid_mask must be boolean')
    if not np.all(np.isfinite(raw_confidence[valid_mask])):
        raise ContractError('valid mask marks non-finite raw confidence as valid')
    if 'dense_audit_uri' not in audit.metadata:
        raise ContractError('AuditRecord metadata requires dense_audit_uri')
    dense = _load_npz(audit.metadata['dense_audit_uri'], 'dense_audit_uri')
    _require_payload_keys(
        dense,
        (
            'voxel_points',
            'raw_confidence',
            'risk',
            'gt_error',
            'failure_label',
            'provenance_count',
        ),
        'dense_audit_uri',
    )
    expected_count = int(np.count_nonzero(valid_mask))
    voxel_points = dense['voxel_points']
    if voxel_points.shape != (expected_count, 3):
        raise ContractError('invalid-mask/confidence filtering drift in dense voxel points')
    for key in ('raw_confidence', 'risk', 'gt_error', 'provenance_count', 'failure_label'):
        if dense[key].shape != (expected_count,):
            raise ContractError(f'invalid-mask/confidence filtering drift in dense {key}')
    for key in ('voxel_points', 'raw_confidence', 'risk', 'gt_error'):
        _require_finite_array(dense[key], f'dense_audit_uri:{key}')
    if not np.array_equal(dense['raw_confidence'], raw_confidence[valid_mask]):
        raise ContractError('invalid-mask/confidence filtering drift in dense raw confidence')
    if dense['failure_label'].dtype != np.bool_:
        raise ContractError('dense failure_label must be boolean')
    if not np.issubdtype(dense['provenance_count'].dtype, np.integer) or np.any(
        dense['provenance_count'] < 1
    ):
        raise ContractError('dense provenance_count must contain positive integers')
    if not np.array_equal(dense['failure_label'], dense['gt_error'] > 0.002):
        raise ContractError('dense failure_label must be the 2 mm GT-error label')
    for field_name in (
        'geometry_prediction_uri',
        'native_confidence_uri',
        'valid_mask_uri',
    ):
        expected_digest = prediction.payload_digests.get(field_name, '')
        if (
            manifest.scientific_validity is ScientificValidity.SCIENTIFIC
            and not expected_digest
        ):
            raise ContractError(f'{field_name} requires a payload digest for scientific evidence')
        _validate_payload_digest(
            getattr(prediction, field_name),
            expected_digest,
            field_name,
        )
    dense_digest = audit.metadata.get('dense_audit_sha256', '')
    if manifest.scientific_validity is ScientificValidity.SCIENTIFIC and not dense_digest:
        raise ContractError('dense_audit_uri requires a payload digest for scientific evidence')
    if dense_digest and not _SHA256_RE.fullmatch(dense_digest):
        raise ContractError('dense_audit_sha256 must be a lowercase SHA-256 digest')
    _validate_payload_digest(
        audit.metadata['dense_audit_uri'], dense_digest, 'dense_audit_uri'
    )

