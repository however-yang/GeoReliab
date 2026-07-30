"""Fail-closed v4 counterfactual identity, split, and schedule contracts.

This module implements Task 2 of ``.superpowers/v4-ranking-warning-plan.md``.
It is deliberately CPU-only and independent of model adapters, checkpoints,
devices, and execution paths.  The old v1 lighting-3 materializer remains
unchanged.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
import hashlib
import json
import os
from pathlib import Path
import re
from typing import Any
from uuid import uuid4

from .v4_science_lock import (
    NO_SCIENTIFIC_RESULT,
    V4_PROTOCOL_ID,
    V4_PROTOCOL_PROVENANCE_SCHEMA_VERSION,
    V4_PROTOCOL_SHA256,
    V4_PROTOCOL_VERSION,
    v4_protocol_provenance,
)


COUNTERFACTUAL_PAIR_SCHEMA_VERSION = "georeliab-v4-counterfactual-pair-1.0"
MODEL_INDEPENDENT_STATE_SCHEMA_VERSION = "georeliab-v4-model-independent-state-1.0"
SCIENTIFIC_SCHEDULE_SCHEMA_VERSION = "georeliab-v4-scientific-schedule-1.0"
SCIENTIFIC_EXECUTION_UNIT_SCHEMA_VERSION = "georeliab-v4-scientific-execution-unit-1.0"
V4_SPLIT_SCHEMA_VERSION = "georeliab-v4-splits-1.0"

TEST_SCENE_IDS = (
    1,
    9,
    10,
    11,
    12,
    13,
    23,
    24,
    29,
    32,
    33,
    34,
    48,
    49,
    62,
    75,
    77,
    110,
    114,
    118,
)
DTU_OFFICIAL_SCENE_IDS = (*range(1, 78), *range(82, 129))
DTU_OFFICIAL_SCENE_SET = frozenset(DTU_OFFICIAL_SCENE_IDS)
EXCLUDED_SUPPORT_SCENE_IDS = frozenset({4, 15})
LIGHTING_STATES = ("L1", "L2", "L3", "L4", "L5", "L6", "L7")
FOG_STATES = ("fog-s1", "fog-s2", "fog-s3")
FOG_BOUNDARY_LAG_SEQUENCE = ("L3", *FOG_STATES)
SCIENTIFIC_STATES = (*LIGHTING_STATES, *FOG_STATES)
SCIENTIFIC_MODELS = ("VGGT", "MASt3R")
MVE_GO_TO_EXTERNAL_VALIDATION = "MVE_GO_TO_EXTERNAL_VALIDATION"

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_PAIR_KEYS = frozenset(
    {
        "schema_version",
        "protocol_provenance",
        "pair_identity_sha256",
        "payload_sha256",
        "payload",
    }
)
_PAIR_PAYLOAD_KEYS = frozenset(
    {
        "dataset",
        "scene_id",
        "axis",
        "axis_semantics",
        "reference_state",
        "counterfactual_state",
    }
)
_STATE_KEYS = frozenset(
    {
        "schema_version",
        "protocol_provenance",
        "dataset",
        "scene_id",
        "state_id",
        "ordered_view_ids",
        "gt_point_cloud_sha256",
        "observability_mask_sha256",
        "camera_sha256_by_view",
        "input_sha256_by_view",
        "source_state_id",
        "source_input_sha256_by_view",
        "state_identity_sha256",
    }
)
_VIEW_DIGEST_KEYS = frozenset({"view_id", "sha256"})
_SCHEDULE_KEYS = frozenset(
    {
        "schema_version",
        "protocol_provenance",
        "models",
        "scene_ids",
        "state_ids",
        "units",
        "schedule_sha256",
    }
)
_UNIT_KEYS = frozenset(
    {
        "schema_version",
        "protocol_provenance",
        "dataset",
        "model_id",
        "scene_id",
        "state_id",
        "state_identity_sha256",
        "pair_identity_sha256",
        "execution_unit_sha256",
    }
)


class CounterfactualContractError(ValueError):
    """Raised when a v4 identity, split, or schedule fails closed."""


class CounterfactualAxis(str, Enum):
    DTU_LIGHTING = "DTU_LIGHTING"
    SYNTHETIC_FOG = "SYNTHETIC_FOG"


class AxisSemantics(str, Enum):
    UNORDERED_DISCRETE = "UNORDERED_DISCRETE"
    ORDERED = "ORDERED"
    PHYSICAL_SANITY_NON_SCIENTIFIC = "PHYSICAL_SANITY_NON_SCIENTIFIC"


class DatasetRole(str, Enum):
    SCIENTIFIC = "SCIENTIFIC"
    PHYSICAL_SANITY_NON_SCIENTIFIC = "PHYSICAL_SANITY_NON_SCIENTIFIC"


def _default_source_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _require_closed_schema(
    value: object,
    expected: frozenset[str],
    *,
    label: str,
) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise CounterfactualContractError(f"{label} must be a JSON object")
    non_string = {key for key in value if not isinstance(key, str)}
    actual = {key for key in value if isinstance(key, str)}
    missing = expected - actual
    unexpected = actual - expected
    if not missing and not unexpected and not non_string:
        return value
    reasons = []
    if missing:
        reasons.append("missing keys " + ", ".join(sorted(missing)))
    if unexpected:
        reasons.append("unexpected keys " + ", ".join(sorted(unexpected)))
    if non_string:
        reasons.append(
            "non-string keys " + ", ".join(sorted(repr(key) for key in non_string))
        )
    raise CounterfactualContractError(
        f"{label} closed schema violation: {'; '.join(reasons)}"
    )


def _require_json_list(value: object, *, label: str) -> list[Any]:
    if not isinstance(value, list):
        raise CounterfactualContractError(f"{label} must be a JSON list")
    return value


def _require_sha256(value: object, *, label: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise CounterfactualContractError(f"{label} must be a lowercase SHA-256 digest")
    return value


def _require_scene_id(value: object, *, label: str = "scene_id") -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise CounterfactualContractError(f"{label} must be a positive integer")
    return value


def _duplicate_rejecting_object(
    pairs: list[tuple[str, Any]],
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise CounterfactualContractError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _parse_json(value: bytes | str, *, label: str) -> object:
    try:
        text = value.decode("utf-8") if isinstance(value, bytes) else value
        return json.loads(text, object_pairs_hook=_duplicate_rejecting_object)
    except UnicodeDecodeError as exc:
        raise CounterfactualContractError(f"{label} is not UTF-8 JSON") from exc
    except json.JSONDecodeError as exc:
        raise CounterfactualContractError(f"{label} is not valid JSON: {exc}") from exc


def canonical_json_bytes(value: object) -> bytes:
    """Return the one canonical JSON encoding used by all Task 2 digests."""

    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
    except (TypeError, ValueError) as exc:
        raise CounterfactualContractError(
            f"value is outside the canonical JSON domain: {exc}"
        ) from exc


def canonical_json_sha256(value: object) -> str:
    """Hash canonical JSON without filesystem or insertion-order dependence."""

    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _canonical_file_bytes(value: object) -> bytes:
    return canonical_json_bytes(value) + b"\n"


def _expected_protocol_provenance() -> dict[str, str]:
    return {
        "schema_version": V4_PROTOCOL_PROVENANCE_SCHEMA_VERSION,
        "protocol_id": V4_PROTOCOL_ID,
        "protocol_version": V4_PROTOCOL_VERSION,
        "protocol_sha256": V4_PROTOCOL_SHA256,
    }


def _locked_protocol_provenance(source_root: Path | None) -> dict[str, str]:
    root = _default_source_root() if source_root is None else source_root
    try:
        provenance = v4_protocol_provenance(root)
    except Exception as exc:
        raise CounterfactualContractError(
            f"cannot bind Task 2 identity to the Task 1 v4 lock: {exc}"
        ) from exc
    _validate_protocol_provenance(provenance)
    return provenance


def _validate_protocol_provenance(value: object) -> dict[str, str]:
    expected = _expected_protocol_provenance()
    if not isinstance(value, Mapping) or dict(value) != expected:
        raise CounterfactualContractError(
            f"protocol provenance is not fixed to {V4_PROTOCOL_ID}@{V4_PROTOCOL_SHA256}"
        )
    return expected


def _protocol_items(value: object) -> tuple[tuple[str, str], ...]:
    return tuple(sorted(_validate_protocol_provenance(value).items()))


@dataclass(frozen=True)
class AssetEvidence:
    """A verified source member; location is intentionally non-semantic."""

    member: str
    sha256: str
    source_uri: str

    def __post_init__(self) -> None:
        if not isinstance(self.member, str) or not self.member:
            raise CounterfactualContractError(
                "asset evidence member must be a non-empty string"
            )
        _require_sha256(self.sha256, label="asset evidence sha256")
        if not isinstance(self.source_uri, str) or not self.source_uri.strip():
            raise CounterfactualContractError(
                "asset evidence source_uri must be non-empty"
            )


@dataclass(frozen=True)
class ViewDigest:
    view_id: int
    sha256: str

    def __post_init__(self) -> None:
        if (
            isinstance(self.view_id, bool)
            or not isinstance(self.view_id, int)
            or not 1 <= self.view_id <= 49
        ):
            raise CounterfactualContractError(
                "view_id must be an integer in the official DTU range 1..49"
            )
        _require_sha256(self.sha256, label=f"view {self.view_id} sha256")

    def to_dict(self) -> dict[str, object]:
        return {"view_id": self.view_id, "sha256": self.sha256}

    @classmethod
    def from_dict(cls, value: object, *, label: str) -> "ViewDigest":
        row = _require_closed_schema(value, _VIEW_DIGEST_KEYS, label=label)
        return cls(
            view_id=_require_scene_id(row["view_id"], label=f"{label}.view_id"),
            sha256=_require_sha256(row["sha256"], label=f"{label}.sha256"),
        )


def _validate_ordered_view_ids(value: object) -> tuple[int, ...]:
    rows = _require_json_list(value, label="ordered_view_ids")
    if len(rows) != 8:
        raise CounterfactualContractError(
            "ordered_view_ids must contain exactly eight views"
        )
    result = []
    for index, view_id in enumerate(rows):
        if (
            isinstance(view_id, bool)
            or not isinstance(view_id, int)
            or not 1 <= view_id <= 49
        ):
            raise CounterfactualContractError(
                f"ordered_view_ids[{index}] must be an integer in 1..49"
            )
        result.append(view_id)
    if len(set(result)) != len(result):
        raise CounterfactualContractError("ordered_view_ids contains duplicate views")
    return tuple(result)


def _ordered_views_from_sequence(value: Sequence[int]) -> tuple[int, ...]:
    if isinstance(value, (str, bytes, bytearray)):
        raise CounterfactualContractError(
            "ordered_view_ids must be a sequence of eight integers"
        )
    return _validate_ordered_view_ids(list(value))


def _parse_view_digests(
    value: object,
    *,
    ordered_view_ids: tuple[int, ...],
    label: str,
) -> tuple[ViewDigest, ...]:
    rows = _require_json_list(value, label=label)
    result = tuple(
        ViewDigest.from_dict(row, label=f"{label}[{index}]")
        for index, row in enumerate(rows)
    )
    if tuple(row.view_id for row in result) != ordered_view_ids:
        if len({row.view_id for row in result}) != len(result):
            raise CounterfactualContractError(f"{label} contains duplicate views")
        raise CounterfactualContractError(
            f"{label} must match the exact ordered eight-view identity"
        )
    return result


def _validate_view_digests(
    value: tuple[ViewDigest, ...],
    *,
    ordered_view_ids: tuple[int, ...],
    label: str,
) -> None:
    if not isinstance(value, tuple) or not all(
        isinstance(row, ViewDigest) for row in value
    ):
        raise CounterfactualContractError(
            f"{label} must be an immutable tuple of ViewDigest values"
        )
    if tuple(row.view_id for row in value) != ordered_view_ids:
        if len({row.view_id for row in value}) != len(value):
            raise CounterfactualContractError(f"{label} contains duplicate views")
        raise CounterfactualContractError(
            f"{label} must match the exact ordered eight-view identity"
        )


def _state_payload(
    *,
    protocol_provenance: Mapping[str, str],
    dataset: str,
    scene_id: int,
    state_id: str,
    ordered_view_ids: tuple[int, ...],
    gt_point_cloud_sha256: str,
    observability_mask_sha256: str,
    camera_sha256_by_view: tuple[ViewDigest, ...],
    input_sha256_by_view: tuple[ViewDigest, ...],
    source_state_id: str,
    source_input_sha256_by_view: tuple[ViewDigest, ...],
) -> dict[str, object]:
    return {
        "schema_version": MODEL_INDEPENDENT_STATE_SCHEMA_VERSION,
        "protocol_provenance": dict(protocol_provenance),
        "dataset": dataset,
        "scene_id": scene_id,
        "state_id": state_id,
        "ordered_view_ids": list(ordered_view_ids),
        "gt_point_cloud_sha256": gt_point_cloud_sha256,
        "observability_mask_sha256": observability_mask_sha256,
        "camera_sha256_by_view": [row.to_dict() for row in camera_sha256_by_view],
        "input_sha256_by_view": [row.to_dict() for row in input_sha256_by_view],
        "source_state_id": source_state_id,
        "source_input_sha256_by_view": [
            row.to_dict() for row in source_input_sha256_by_view
        ],
    }


@dataclass(frozen=True)
class ModelIndependentState:
    """One DTU state bound only to scene/data semantics and v4 provenance."""

    protocol_provenance_items: tuple[tuple[str, str], ...]
    dataset: str
    scene_id: int
    state_id: str
    ordered_view_ids: tuple[int, ...]
    gt_point_cloud_sha256: str
    observability_mask_sha256: str
    camera_sha256_by_view: tuple[ViewDigest, ...]
    input_sha256_by_view: tuple[ViewDigest, ...]
    source_state_id: str
    source_input_sha256_by_view: tuple[ViewDigest, ...]
    state_identity_sha256: str
    schema_version: str = MODEL_INDEPENDENT_STATE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != MODEL_INDEPENDENT_STATE_SCHEMA_VERSION:
            raise CounterfactualContractError("model-independent state schema mismatch")
        provenance = dict(self.protocol_provenance_items)
        _validate_protocol_provenance(provenance)
        if self.dataset != "DTU":
            raise CounterfactualContractError(
                "model-independent scientific states must use DTU"
            )
        _require_scene_id(self.scene_id)
        if self.scene_id not in DTU_OFFICIAL_SCENE_SET:
            raise CounterfactualContractError(
                "model-independent state scene is not an official DTU scan"
            )
        if self.state_id not in SCIENTIFIC_STATES:
            raise CounterfactualContractError(
                f"state_id is outside the frozen scientific states: {self.state_id!r}"
            )
        if len(self.ordered_view_ids) != 8 or len(set(self.ordered_view_ids)) != 8:
            raise CounterfactualContractError(
                "ordered_view_ids must contain eight distinct views"
            )
        for view_id in self.ordered_view_ids:
            if (
                isinstance(view_id, bool)
                or not isinstance(view_id, int)
                or not 1 <= view_id <= 49
            ):
                raise CounterfactualContractError(
                    "ordered_view_ids must use official DTU views 1..49"
                )
        _require_sha256(self.gt_point_cloud_sha256, label="GT point-cloud sha256")
        _require_sha256(
            self.observability_mask_sha256,
            label="observability-mask sha256",
        )
        _validate_view_digests(
            self.camera_sha256_by_view,
            ordered_view_ids=self.ordered_view_ids,
            label="camera_sha256_by_view",
        )
        _validate_view_digests(
            self.input_sha256_by_view,
            ordered_view_ids=self.ordered_view_ids,
            label="input_sha256_by_view",
        )
        _validate_view_digests(
            self.source_input_sha256_by_view,
            ordered_view_ids=self.ordered_view_ids,
            label="source_input_sha256_by_view",
        )
        if self.state_id in LIGHTING_STATES:
            if (
                self.source_state_id != self.state_id
                or self.source_input_sha256_by_view != self.input_sha256_by_view
            ):
                raise CounterfactualContractError(
                    "DTU lighting states must bind to their own official inputs"
                )
        elif self.source_state_id != "L3":
            raise CounterfactualContractError(
                "synthetic fog states must bind to the L3 source state"
            )

        expected = canonical_json_sha256(self.identity_payload())
        _require_sha256(self.state_identity_sha256, label="state identity SHA-256")
        if self.state_identity_sha256 != expected:
            raise CounterfactualContractError("state identity tamper or mismatch")

    @property
    def protocol_provenance(self) -> dict[str, str]:
        return dict(self.protocol_provenance_items)

    @property
    def camera_identity_sha256(self) -> str:
        return canonical_json_sha256(
            [row.to_dict() for row in self.camera_sha256_by_view]
        )

    @property
    def scene_identity_sha256(self) -> str:
        return canonical_json_sha256(
            {
                "protocol_provenance": self.protocol_provenance,
                "dataset": self.dataset,
                "scene_id": self.scene_id,
                "ordered_view_ids": list(self.ordered_view_ids),
                "gt_point_cloud_sha256": self.gt_point_cloud_sha256,
                "observability_mask_sha256": (self.observability_mask_sha256),
                "camera_sha256_by_view": [
                    row.to_dict() for row in self.camera_sha256_by_view
                ],
            }
        )

    def identity_payload(self) -> dict[str, object]:
        return _state_payload(
            protocol_provenance=self.protocol_provenance,
            dataset=self.dataset,
            scene_id=self.scene_id,
            state_id=self.state_id,
            ordered_view_ids=self.ordered_view_ids,
            gt_point_cloud_sha256=self.gt_point_cloud_sha256,
            observability_mask_sha256=self.observability_mask_sha256,
            camera_sha256_by_view=self.camera_sha256_by_view,
            input_sha256_by_view=self.input_sha256_by_view,
            source_state_id=self.source_state_id,
            source_input_sha256_by_view=self.source_input_sha256_by_view,
        )

    def to_dict(self) -> dict[str, object]:
        return {
            **self.identity_payload(),
            "state_identity_sha256": self.state_identity_sha256,
        }

    @classmethod
    def from_dict(cls, value: object) -> "ModelIndependentState":
        row = _require_closed_schema(
            value, _STATE_KEYS, label="model-independent state"
        )
        if row["schema_version"] != MODEL_INDEPENDENT_STATE_SCHEMA_VERSION:
            raise CounterfactualContractError("model-independent state schema mismatch")
        provenance_items = _protocol_items(row["protocol_provenance"])
        dataset = row["dataset"]
        if not isinstance(dataset, str):
            raise CounterfactualContractError("state dataset must be a string")
        scene_id = _require_scene_id(row["scene_id"])
        state_id = row["state_id"]
        if not isinstance(state_id, str):
            raise CounterfactualContractError("state_id must be a string")
        ordered = _validate_ordered_view_ids(row["ordered_view_ids"])
        camera = _parse_view_digests(
            row["camera_sha256_by_view"],
            ordered_view_ids=ordered,
            label="camera_sha256_by_view",
        )
        inputs = _parse_view_digests(
            row["input_sha256_by_view"],
            ordered_view_ids=ordered,
            label="input_sha256_by_view",
        )
        source_inputs = _parse_view_digests(
            row["source_input_sha256_by_view"],
            ordered_view_ids=ordered,
            label="source_input_sha256_by_view",
        )
        source_state_id = row["source_state_id"]
        if not isinstance(source_state_id, str):
            raise CounterfactualContractError("source_state_id must be a string")
        return cls(
            protocol_provenance_items=provenance_items,
            dataset=dataset,
            scene_id=scene_id,
            state_id=state_id,
            ordered_view_ids=ordered,
            gt_point_cloud_sha256=_require_sha256(
                row["gt_point_cloud_sha256"],
                label="GT point-cloud sha256",
            ),
            observability_mask_sha256=_require_sha256(
                row["observability_mask_sha256"],
                label="observability-mask sha256",
            ),
            camera_sha256_by_view=camera,
            input_sha256_by_view=inputs,
            source_state_id=source_state_id,
            source_input_sha256_by_view=source_inputs,
            state_identity_sha256=_require_sha256(
                row["state_identity_sha256"],
                label="state identity SHA-256",
            ),
        )


def _new_state(
    *,
    protocol_provenance: Mapping[str, str],
    scene_id: int,
    state_id: str,
    ordered_view_ids: tuple[int, ...],
    gt_point_cloud_sha256: str,
    observability_mask_sha256: str,
    camera_sha256_by_view: tuple[ViewDigest, ...],
    input_sha256_by_view: tuple[ViewDigest, ...],
    source_state_id: str,
    source_input_sha256_by_view: tuple[ViewDigest, ...],
) -> ModelIndependentState:
    payload = _state_payload(
        protocol_provenance=protocol_provenance,
        dataset="DTU",
        scene_id=scene_id,
        state_id=state_id,
        ordered_view_ids=ordered_view_ids,
        gt_point_cloud_sha256=gt_point_cloud_sha256,
        observability_mask_sha256=observability_mask_sha256,
        camera_sha256_by_view=camera_sha256_by_view,
        input_sha256_by_view=input_sha256_by_view,
        source_state_id=source_state_id,
        source_input_sha256_by_view=source_input_sha256_by_view,
    )
    return ModelIndependentState(
        protocol_provenance_items=tuple(sorted(protocol_provenance.items())),
        dataset="DTU",
        scene_id=scene_id,
        state_id=state_id,
        ordered_view_ids=ordered_view_ids,
        gt_point_cloud_sha256=gt_point_cloud_sha256,
        observability_mask_sha256=observability_mask_sha256,
        camera_sha256_by_view=camera_sha256_by_view,
        input_sha256_by_view=input_sha256_by_view,
        source_state_id=source_state_id,
        source_input_sha256_by_view=source_input_sha256_by_view,
        state_identity_sha256=canonical_json_sha256(payload),
    )


def _normalize_member(value: str) -> str:
    return value.replace("\\", "/")


def _require_evidence_views(
    value: Mapping[int, AssetEvidence],
    *,
    ordered_view_ids: tuple[int, ...],
    label: str,
) -> None:
    if not isinstance(value, Mapping):
        raise CounterfactualContractError(f"{label} must be a mapping")
    if set(value) != set(ordered_view_ids):
        raise CounterfactualContractError(
            f"{label} must contain the exact ordered eight-view identity"
        )
    for view_id in ordered_view_ids:
        if not isinstance(value[view_id], AssetEvidence):
            raise CounterfactualContractError(
                f"{label}[{view_id}] must be AssetEvidence"
            )


def materialize_dtu_state_identity(
    *,
    source_root: Path | None,
    scene_id: int,
    state_id: str,
    ordered_view_ids: Sequence[int],
    rgb_inputs: Mapping[int, AssetEvidence],
    cameras: Mapping[int, AssetEvidence],
    gt_point_cloud: AssetEvidence,
    observability_mask: AssetEvidence,
    clean_source_inputs: Mapping[int, AssetEvidence] | None = None,
) -> ModelIndependentState:
    """Validate v4 L1-L7/fog evidence and discard non-semantic locations."""

    scene = _require_scene_id(scene_id)
    if state_id not in SCIENTIFIC_STATES:
        raise CounterfactualContractError(
            f"state_id is outside the frozen scientific states: {state_id!r}"
        )
    ordered = _ordered_views_from_sequence(ordered_view_ids)
    _require_evidence_views(rgb_inputs, ordered_view_ids=ordered, label="rgb_inputs")
    _require_evidence_views(cameras, ordered_view_ids=ordered, label="cameras")
    if not isinstance(gt_point_cloud, AssetEvidence) or not isinstance(
        observability_mask, AssetEvidence
    ):
        raise CounterfactualContractError(
            "GT point cloud and observability mask require AssetEvidence"
        )

    expected_gt = f"Points/stl/stl{scene:03d}_total.ply"
    if _normalize_member(gt_point_cloud.member) != expected_gt:
        raise CounterfactualContractError(
            f"GT point-cloud member must be exact official {expected_gt}"
        )
    expected_mask = f"MVS Data/ObsMask/ObsMask{scene}_10.mat"
    if _normalize_member(observability_mask.member) != expected_mask:
        raise CounterfactualContractError(
            f"observability-mask member must be exact official {expected_mask}"
        )

    camera_rows = []
    input_rows = []
    for view_id in ordered:
        camera = cameras[view_id]
        expected_camera = f"MVS Data/Calibration/cal18/pos_{view_id:03d}.txt"
        if _normalize_member(camera.member) != expected_camera:
            raise CounterfactualContractError(
                f"camera view {view_id} must be exact official {expected_camera}"
            )
        camera_rows.append(ViewDigest(view_id, camera.sha256))

        rgb = rgb_inputs[view_id]
        if state_id in LIGHTING_STATES:
            expected_rgb = (
                f"Rectified/scan{scene}/rect_{view_id:03d}_{state_id[1:]}_r5000.png"
            )
            if _normalize_member(rgb.member) != expected_rgb:
                raise CounterfactualContractError(
                    f"RGB view {view_id} must use exact official "
                    f"{state_id} naming {expected_rgb}"
                )
        elif not _normalize_member(rgb.member).startswith(
            f"SyntheticFog/scan{scene}/{state_id}/"
        ):
            raise CounterfactualContractError(
                f"synthetic fog view {view_id} is not bound to scan{scene}/{state_id}"
            )
        input_rows.append(ViewDigest(view_id, rgb.sha256))

    input_tuple = tuple(input_rows)
    if state_id in LIGHTING_STATES:
        if clean_source_inputs is not None:
            raise CounterfactualContractError(
                "DTU lighting states cannot carry a second clean source"
            )
        source_state_id = state_id
        source_rows = input_tuple
    else:
        if clean_source_inputs is None:
            raise CounterfactualContractError(
                "synthetic fog states require exact L3 clean source evidence"
            )
        _require_evidence_views(
            clean_source_inputs,
            ordered_view_ids=ordered,
            label="clean_source_inputs",
        )
        source_rows_list = []
        for view_id in ordered:
            source = clean_source_inputs[view_id]
            expected_source = f"Rectified/scan{scene}/rect_{view_id:03d}_3_r5000.png"
            if _normalize_member(source.member) != expected_source:
                raise CounterfactualContractError(
                    f"fog source view {view_id} must use exact official L3 "
                    f"naming {expected_source}"
                )
            source_rows_list.append(ViewDigest(view_id, source.sha256))
        source_state_id = "L3"
        source_rows = tuple(source_rows_list)

    provenance = _locked_protocol_provenance(source_root)
    return _new_state(
        protocol_provenance=provenance,
        scene_id=scene,
        state_id=state_id,
        ordered_view_ids=ordered,
        gt_point_cloud_sha256=gt_point_cloud.sha256,
        observability_mask_sha256=observability_mask.sha256,
        camera_sha256_by_view=tuple(camera_rows),
        input_sha256_by_view=input_tuple,
        source_state_id=source_state_id,
        source_input_sha256_by_view=source_rows,
    )


def _require_shared_scene_identity(
    reference: ModelIndependentState,
    candidate: ModelIndependentState,
    *,
    label: str,
) -> None:
    if candidate.scene_id != reference.scene_id:
        raise CounterfactualContractError(f"{label} changes scene identity")
    if candidate.ordered_view_ids != reference.ordered_view_ids:
        raise CounterfactualContractError(f"{label} changes ordered view identity")
    if candidate.camera_sha256_by_view != reference.camera_sha256_by_view:
        raise CounterfactualContractError(f"{label} changes camera identity")
    if candidate.gt_point_cloud_sha256 != reference.gt_point_cloud_sha256:
        raise CounterfactualContractError(f"{label} changes GT point-cloud identity")
    if candidate.observability_mask_sha256 != reference.observability_mask_sha256:
        raise CounterfactualContractError(
            f"{label} changes observability-mask identity"
        )
    if candidate.protocol_provenance != reference.protocol_provenance:
        raise CounterfactualContractError(f"{label} changes protocol provenance")


def validate_dtu_lighting_states(
    states: Iterable[ModelIndependentState],
) -> dict[str, object]:
    """Validate L1-L7 as an unordered set with one shared scene identity."""

    rows = tuple(states)
    if len(rows) != 7 or any(
        not isinstance(row, ModelIndependentState) for row in rows
    ):
        raise CounterfactualContractError(
            "DTU lighting validation requires exactly seven state identities"
        )
    by_state = {row.state_id: row for row in rows}
    if len(by_state) != len(rows) or set(by_state) != set(LIGHTING_STATES):
        raise CounterfactualContractError(
            "DTU lighting identities must contain exactly L1-L7 once each"
        )
    reference = by_state["L3"]
    for state_id, state in by_state.items():
        _require_shared_scene_identity(reference, state, label=f"DTU {state_id}")
    return {
        "axis": CounterfactualAxis.DTU_LIGHTING.value,
        "axis_semantics": AxisSemantics.UNORDERED_DISCRETE.value,
        "states": LIGHTING_STATES,
        "scene_id": reference.scene_id,
        "ordered_view_ids": reference.ordered_view_ids,
        "scene_identity_sha256": reference.scene_identity_sha256,
    }


def validate_fog_states(
    states: Sequence[ModelIndependentState],
) -> dict[str, object]:
    """Validate only L3 -> fog-s1 -> fog-s2 -> fog-s3."""

    rows = tuple(states)
    if (
        tuple(
            row.state_id if isinstance(row, ModelIndependentState) else None
            for row in rows
        )
        != FOG_BOUNDARY_LAG_SEQUENCE
    ):
        raise CounterfactualContractError(
            "fog identities must use the frozen ordered sequence "
            "L3 -> fog-s1 -> fog-s2 -> fog-s3"
        )
    reference = rows[0]
    for state in rows[1:]:
        _require_shared_scene_identity(reference, state, label=f"fog {state.state_id}")
        if (
            state.source_state_id != "L3"
            or state.source_input_sha256_by_view != reference.input_sha256_by_view
        ):
            raise CounterfactualContractError(
                f"fog {state.state_id} does not bind to the exact L3 source"
            )
    return {
        "axis": CounterfactualAxis.SYNTHETIC_FOG.value,
        "axis_semantics": AxisSemantics.ORDERED.value,
        "states": FOG_BOUNDARY_LAG_SEQUENCE,
        "scene_id": reference.scene_id,
        "ordered_view_ids": reference.ordered_view_ids,
        "scene_identity_sha256": reference.scene_identity_sha256,
    }


def validate_boundary_lag_sequence(
    axis: CounterfactualAxis,
    state_ids: Sequence[str],
) -> None:
    """Admit Boundary Lag only on the frozen ordered fog sequence."""

    if axis is not CounterfactualAxis.SYNTHETIC_FOG:
        raise CounterfactualContractError(
            "Boundary Lag is forbidden for DTU lighting and every non-fog axis"
        )
    if tuple(state_ids) != FOG_BOUNDARY_LAG_SEQUENCE:
        raise CounterfactualContractError(
            "Boundary Lag requires the frozen fog sequence "
            "L3 -> fog-s1 -> fog-s2 -> fog-s3"
        )


def _pair_payload(
    *,
    reference_state: ModelIndependentState,
    counterfactual_state: ModelIndependentState,
    axis: CounterfactualAxis,
    axis_semantics: AxisSemantics,
) -> dict[str, object]:
    return {
        "dataset": "DTU",
        "scene_id": reference_state.scene_id,
        "axis": axis.value,
        "axis_semantics": axis_semantics.value,
        "reference_state": reference_state.to_dict(),
        "counterfactual_state": counterfactual_state.to_dict(),
    }


@dataclass(frozen=True)
class CounterfactualPairManifest:
    """Strict semantic pair identity with no model or location fields."""

    protocol_provenance_items: tuple[tuple[str, str], ...]
    dataset: str
    scene_id: int
    axis: CounterfactualAxis
    axis_semantics: AxisSemantics
    reference_state: ModelIndependentState
    counterfactual_state: ModelIndependentState
    pair_identity_sha256: str
    payload_sha256: str
    schema_version: str = COUNTERFACTUAL_PAIR_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != COUNTERFACTUAL_PAIR_SCHEMA_VERSION:
            raise CounterfactualContractError("counterfactual pair schema mismatch")
        provenance = dict(self.protocol_provenance_items)
        _validate_protocol_provenance(provenance)
        if self.dataset != "DTU":
            raise CounterfactualContractError("counterfactual pairs must use DTU")
        _require_scene_id(self.scene_id)
        if not isinstance(self.reference_state, ModelIndependentState) or not (
            isinstance(self.counterfactual_state, ModelIndependentState)
        ):
            raise CounterfactualContractError(
                "pair states must be model-independent state identities"
            )
        if self.reference_state.protocol_provenance != provenance or (
            self.counterfactual_state.protocol_provenance != provenance
        ):
            raise CounterfactualContractError(
                "pair states do not use the exact pair protocol provenance"
            )
        _require_shared_scene_identity(
            self.reference_state,
            self.counterfactual_state,
            label="counterfactual pair",
        )
        if self.scene_id != self.reference_state.scene_id:
            raise CounterfactualContractError(
                "pair scene_id does not match state identity"
            )
        if self.reference_state.state_id != "L3":
            raise CounterfactualContractError(
                "counterfactual pair reference state must be L3"
            )

        if self.axis is CounterfactualAxis.DTU_LIGHTING:
            if self.axis_semantics is not AxisSemantics.UNORDERED_DISCRETE:
                raise CounterfactualContractError(
                    "DTU lighting axis semantics must be UNORDERED_DISCRETE"
                )
            if (
                self.counterfactual_state.state_id not in LIGHTING_STATES
                or self.counterfactual_state.state_id == "L3"
            ):
                raise CounterfactualContractError(
                    "DTU lighting pair counterfactual must be L1-L2 or L4-L7"
                )
        elif self.axis is CounterfactualAxis.SYNTHETIC_FOG:
            if self.axis_semantics is not AxisSemantics.ORDERED:
                raise CounterfactualContractError(
                    "synthetic fog axis semantics must be ORDERED"
                )
            if self.counterfactual_state.state_id not in FOG_STATES:
                raise CounterfactualContractError(
                    "synthetic fog pair counterfactual must be fog-s1..fog-s3"
                )
            if (
                self.counterfactual_state.source_state_id != "L3"
                or self.counterfactual_state.source_input_sha256_by_view
                != self.reference_state.input_sha256_by_view
            ):
                raise CounterfactualContractError(
                    "synthetic fog pair is not bound to the exact L3 source"
                )
        else:
            raise CounterfactualContractError(
                f"unsupported counterfactual axis: {self.axis!r}"
            )

        expected = canonical_json_sha256(self.identity_payload())
        _require_sha256(self.pair_identity_sha256, label="pair identity SHA-256")
        _require_sha256(self.payload_sha256, label="payload SHA-256")
        if self.pair_identity_sha256 != expected or self.payload_sha256 != expected:
            raise CounterfactualContractError(
                "counterfactual pair payload tamper or digest mismatch"
            )

    @property
    def protocol_provenance(self) -> dict[str, str]:
        return dict(self.protocol_provenance_items)

    def identity_payload(self) -> dict[str, object]:
        return _pair_payload(
            reference_state=self.reference_state,
            counterfactual_state=self.counterfactual_state,
            axis=self.axis,
            axis_semantics=self.axis_semantics,
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "protocol_provenance": self.protocol_provenance,
            "pair_identity_sha256": self.pair_identity_sha256,
            "payload_sha256": self.payload_sha256,
            "payload": self.identity_payload(),
        }

    def canonical_json_bytes(self) -> bytes:
        return _canonical_file_bytes(self.to_dict())


def build_counterfactual_pair_manifest(
    *,
    reference_state: ModelIndependentState,
    counterfactual_state: ModelIndependentState,
    axis: CounterfactualAxis,
    axis_semantics: AxisSemantics,
) -> CounterfactualPairManifest:
    if not isinstance(axis, CounterfactualAxis) or not isinstance(
        axis_semantics, AxisSemantics
    ):
        raise CounterfactualContractError(
            "pair axis and axis semantics must use the frozen enums"
        )
    if not isinstance(reference_state, ModelIndependentState):
        raise CounterfactualContractError("reference_state must be model-independent")
    payload = _pair_payload(
        reference_state=reference_state,
        counterfactual_state=counterfactual_state,
        axis=axis,
        axis_semantics=axis_semantics,
    )
    digest = canonical_json_sha256(payload)
    return CounterfactualPairManifest(
        protocol_provenance_items=tuple(
            sorted(reference_state.protocol_provenance.items())
        ),
        dataset="DTU",
        scene_id=reference_state.scene_id,
        axis=axis,
        axis_semantics=axis_semantics,
        reference_state=reference_state,
        counterfactual_state=counterfactual_state,
        pair_identity_sha256=digest,
        payload_sha256=digest,
    )


def parse_counterfactual_pair_manifest(
    value: bytes | str | Mapping[str, Any],
) -> CounterfactualPairManifest:
    parsed: object = (
        _parse_json(value, label="counterfactual pair manifest")
        if isinstance(value, (bytes, str))
        else value
    )
    row = _require_closed_schema(
        parsed, _PAIR_KEYS, label="counterfactual pair manifest"
    )
    if row["schema_version"] != COUNTERFACTUAL_PAIR_SCHEMA_VERSION:
        raise CounterfactualContractError(
            "counterfactual pair manifest schema mismatch"
        )
    provenance_items = _protocol_items(row["protocol_provenance"])
    payload = _require_closed_schema(
        row["payload"], _PAIR_PAYLOAD_KEYS, label="pair identity payload"
    )
    if payload["dataset"] != "DTU":
        raise CounterfactualContractError("pair dataset must be exactly DTU")
    scene_id = _require_scene_id(payload["scene_id"])
    try:
        axis = CounterfactualAxis(payload["axis"])
        axis_semantics = AxisSemantics(payload["axis_semantics"])
    except (TypeError, ValueError) as exc:
        raise CounterfactualContractError(
            "pair axis or axis semantics is unsupported"
        ) from exc
    reference = ModelIndependentState.from_dict(payload["reference_state"])
    counterfactual = ModelIndependentState.from_dict(payload["counterfactual_state"])
    return CounterfactualPairManifest(
        protocol_provenance_items=provenance_items,
        dataset="DTU",
        scene_id=scene_id,
        axis=axis,
        axis_semantics=axis_semantics,
        reference_state=reference,
        counterfactual_state=counterfactual,
        pair_identity_sha256=_require_sha256(
            row["pair_identity_sha256"], label="pair identity SHA-256"
        ),
        payload_sha256=_require_sha256(row["payload_sha256"], label="payload SHA-256"),
    )


def validate_counterfactual_pair_manifest(
    value: (CounterfactualPairManifest | bytes | str | Mapping[str, Any]),
) -> CounterfactualPairManifest:
    """Reparse and validate a pair even when given a constructed instance."""

    if isinstance(value, CounterfactualPairManifest):
        return parse_counterfactual_pair_manifest(value.to_dict())
    return parse_counterfactual_pair_manifest(value)


def write_counterfactual_pair_manifest(
    path: Path,
    manifest: CounterfactualPairManifest,
) -> str:
    """Publish canonical JSON atomically without overwriting another writer."""

    if not isinstance(manifest, CounterfactualPairManifest):
        raise CounterfactualContractError("write requires CounterfactualPairManifest")
    expected = manifest.canonical_json_bytes()
    expected_sha256 = hashlib.sha256(expected).hexdigest()
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise CounterfactualContractError(
            f"cannot create pair manifest directory: {path.parent}"
        ) from exc
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    temporary_created = False
    published_sha256: str
    try:
        with temporary.open("xb") as handle:
            temporary_created = True
            handle.write(expected)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            # A same-directory hard link is an atomic create-if-absent publish:
            # unlike os.replace(), it can never overwrite a concurrent target.
            os.link(temporary, path)
        except FileExistsError:
            try:
                existing = path.read_bytes()
            except OSError as exc:
                raise CounterfactualContractError(
                    f"cannot read concurrently published pair manifest: {path}"
                ) from exc
            if existing != expected:
                raise CounterfactualContractError(
                    f"existing pair manifest conflicts with canonical payload: {path}"
                )
            published_sha256 = hashlib.sha256(existing).hexdigest()
        except OSError as exc:
            raise CounterfactualContractError(
                f"cannot atomically publish pair manifest: {path}"
            ) from exc
        else:
            published_sha256 = expected_sha256
    except CounterfactualContractError:
        raise
    except OSError as exc:
        raise CounterfactualContractError(
            f"cannot prepare pair manifest for atomic publication: {path}"
        ) from exc
    finally:
        if temporary_created:
            try:
                temporary.unlink(missing_ok=True)
            except OSError as exc:
                raise CounterfactualContractError(
                    f"cannot clean temporary pair manifest: {temporary}"
                ) from exc
    return published_sha256


def read_counterfactual_pair_manifest(
    path: Path,
) -> CounterfactualPairManifest:
    try:
        payload = path.read_bytes()
    except OSError as exc:
        raise CounterfactualContractError(f"cannot read pair manifest: {path}") from exc
    return parse_counterfactual_pair_manifest(payload)


@dataclass(frozen=True)
class VerifiedSceneInventory:
    """Explicit completeness evidence used before any model output exists."""

    scene_id: int
    verified_complete: bool
    inventory_sha256: str

    def __post_init__(self) -> None:
        _require_scene_id(self.scene_id)
        if self.scene_id not in DTU_OFFICIAL_SCENE_SET:
            raise CounterfactualContractError(
                "verified inventory scene is not an official DTU scan"
            )
        if not isinstance(self.verified_complete, bool):
            raise CounterfactualContractError(
                "verified_complete must be an explicit boolean"
            )
        _require_sha256(self.inventory_sha256, label="inventory evidence SHA-256")

    def to_dict(self) -> dict[str, object]:
        return {
            "scene_id": self.scene_id,
            "verified_complete": self.verified_complete,
            "inventory_sha256": self.inventory_sha256,
        }


def _support_sort_key(scene_id: int) -> tuple[str, int]:
    digest = hashlib.sha256(f"GeoReliab-v4:scan{scene_id}".encode("utf-8")).hexdigest()
    return digest, scene_id


@dataclass(frozen=True)
class V4SplitAssignment:
    protocol_provenance_items: tuple[tuple[str, str], ...]
    calibration: tuple[int, ...]
    dev: tuple[int, ...]
    reference: tuple[int, ...]
    test: tuple[int, ...]
    excluded_incomplete: tuple[int, ...]
    unassigned_verified: tuple[int, ...]
    inventory_sha256: str
    fingerprint_sha256: str
    schema_version: str = V4_SPLIT_SCHEMA_VERSION

    @property
    def protocol_provenance(self) -> dict[str, str]:
        return dict(self.protocol_provenance_items)

    @property
    def assigned_scene_ids(self) -> tuple[int, ...]:
        return (
            *self.calibration,
            *self.dev,
            *self.reference,
            *self.test,
        )

    def payload(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "protocol_provenance": self.protocol_provenance,
            "assignment_hash": 'SHA256("GeoReliab-v4:scan<ID>")',
            "splits": {
                "calibration": list(self.calibration),
                "dev": list(self.dev),
                "reference": list(self.reference),
                "test": list(self.test),
            },
            "excluded_scene_ids": sorted(EXCLUDED_SUPPORT_SCENE_IDS),
            "excluded_incomplete": list(self.excluded_incomplete),
            "unassigned_verified": list(self.unassigned_verified),
            "inventory_sha256": self.inventory_sha256,
        }

    def to_dict(self) -> dict[str, object]:
        return {
            **self.payload(),
            "fingerprint_sha256": self.fingerprint_sha256,
        }


def validate_v4_split_assignment(
    assignment: V4SplitAssignment,
) -> None:
    if not isinstance(assignment, V4SplitAssignment):
        raise CounterfactualContractError("split validation requires V4SplitAssignment")
    _validate_protocol_provenance(assignment.protocol_provenance)
    if assignment.schema_version != V4_SPLIT_SCHEMA_VERSION:
        raise CounterfactualContractError("v4 split schema mismatch")
    expected_counts = {
        "calibration": 20,
        "dev": 5,
        "reference": 5,
        "test": 20,
    }
    splits = {
        "calibration": assignment.calibration,
        "dev": assignment.dev,
        "reference": assignment.reference,
        "test": assignment.test,
    }
    for name, expected_count in expected_counts.items():
        values = splits[name]
        if len(values) != expected_count or len(set(values)) != len(values):
            raise CounterfactualContractError(
                f"{name} split must contain exactly {expected_count} unique scenes"
            )
    if assignment.test != TEST_SCENE_IDS:
        raise CounterfactualContractError("test split must equal the frozen 20 scenes")
    names = tuple(splits)
    for index, left_name in enumerate(names):
        left = set(splits[left_name])
        for right_name in names[index + 1 :]:
            overlap = left & set(splits[right_name])
            if overlap:
                raise CounterfactualContractError(
                    "v4 splits are not scene-disjoint: "
                    f"{left_name}/{right_name}={sorted(overlap)}"
                )
    eligible = (
        *assignment.calibration,
        *assignment.dev,
        *assignment.reference,
        *assignment.unassigned_verified,
    )
    if len(set(eligible)) != len(eligible):
        raise CounterfactualContractError(
            "verified support inventory contains duplicate scene assignments"
        )
    if any(
        scene_id not in DTU_OFFICIAL_SCENE_SET
        or scene_id in TEST_SCENE_IDS
        or scene_id in EXCLUDED_SUPPORT_SCENE_IDS
        for scene_id in eligible
    ):
        raise CounterfactualContractError(
            "support assignment contains an ineligible DTU scene"
        )
    ordered_eligible = tuple(sorted(eligible, key=_support_sort_key))
    if (
        assignment.calibration != ordered_eligible[:20]
        or assignment.dev != ordered_eligible[20:25]
        or assignment.reference != ordered_eligible[25:30]
        or assignment.unassigned_verified != ordered_eligible[30:]
    ):
        raise CounterfactualContractError(
            "support splits do not follow the frozen SHA256 ordering"
        )
    if EXCLUDED_SUPPORT_SCENE_IDS & set(assignment.assigned_scene_ids):
        raise CounterfactualContractError("excluded scans 4/15 entered a v4 split")
    if set(assignment.excluded_incomplete) & set(assignment.assigned_scene_ids):
        raise CounterfactualContractError(
            "an explicitly incomplete scene entered a v4 split"
        )
    _require_sha256(assignment.inventory_sha256, label="split inventory SHA-256")
    expected_fingerprint = canonical_json_sha256(assignment.payload())
    if assignment.fingerprint_sha256 != expected_fingerprint:
        raise CounterfactualContractError("v4 split fingerprint tamper or mismatch")


def construct_v4_splits(
    inventory: Iterable[VerifiedSceneInventory],
    *,
    source_root: Path | None = None,
) -> V4SplitAssignment:
    """Construct frozen test plus 20/5/5 support splits from explicit evidence."""

    by_scene: dict[int, VerifiedSceneInventory] = {}
    for row in inventory:
        if not isinstance(row, VerifiedSceneInventory):
            raise CounterfactualContractError(
                "split inventory rows must be VerifiedSceneInventory"
            )
        if row.scene_id in by_scene:
            raise CounterfactualContractError(
                f"duplicate verified inventory for scan{row.scene_id}"
            )
        by_scene[row.scene_id] = row

    missing_test = sorted(set(TEST_SCENE_IDS) - set(by_scene))
    if missing_test:
        raise CounterfactualContractError(
            f"test inventory is missing scenes: {missing_test}"
        )
    incomplete_test = sorted(
        scene_id
        for scene_id in TEST_SCENE_IDS
        if not by_scene[scene_id].verified_complete
    )
    if incomplete_test:
        raise CounterfactualContractError(
            f"test scene inventory is incomplete: {incomplete_test}"
        )

    eligible = sorted(
        (
            scene_id
            for scene_id, row in by_scene.items()
            if row.verified_complete
            and scene_id not in TEST_SCENE_IDS
            and scene_id not in EXCLUDED_SUPPORT_SCENE_IDS
        ),
        key=_support_sort_key,
    )
    if len(eligible) < 30:
        raise CounterfactualContractError(
            "v4 support splits require at least 30 explicitly "
            "verified-complete eligible scenes"
        )
    calibration = tuple(eligible[:20])
    dev = tuple(eligible[20:25])
    reference = tuple(eligible[25:30])
    unassigned = tuple(eligible[30:])
    incomplete = tuple(
        sorted(
            scene_id
            for scene_id, row in by_scene.items()
            if not row.verified_complete
            and scene_id not in TEST_SCENE_IDS
            and scene_id not in EXCLUDED_SUPPORT_SCENE_IDS
        )
    )
    inventory_payload = [by_scene[scene_id].to_dict() for scene_id in sorted(by_scene)]
    inventory_sha256 = canonical_json_sha256(inventory_payload)
    provenance = _locked_protocol_provenance(source_root)
    partial = V4SplitAssignment(
        protocol_provenance_items=tuple(sorted(provenance.items())),
        calibration=calibration,
        dev=dev,
        reference=reference,
        test=TEST_SCENE_IDS,
        excluded_incomplete=incomplete,
        unassigned_verified=unassigned,
        inventory_sha256=inventory_sha256,
        fingerprint_sha256="0" * 64,
    )
    result = V4SplitAssignment(
        protocol_provenance_items=partial.protocol_provenance_items,
        calibration=calibration,
        dev=dev,
        reference=reference,
        test=TEST_SCENE_IDS,
        excluded_incomplete=incomplete,
        unassigned_verified=unassigned,
        inventory_sha256=inventory_sha256,
        fingerprint_sha256=canonical_json_sha256(partial.payload()),
    )
    validate_v4_split_assignment(result)
    return result


def validate_dataset_admission(
    dataset: str,
    role: DatasetRole,
    *,
    scientific: bool,
    decision_status: str = NO_SCIENTIFIC_RESULT,
    protocol_line: str = "v4",
) -> DatasetRole:
    """Keep TartanAir non-scientific and UAVLight outside the v4 runtime."""

    if not isinstance(role, DatasetRole) or not isinstance(scientific, bool):
        raise CounterfactualContractError(
            "dataset role and explicit scientific boolean are required"
        )
    if dataset == "DTU":
        if not scientific or role is not DatasetRole.SCIENTIFIC:
            raise CounterfactualContractError(
                "DTU scientific admission requires the SCIENTIFIC role"
            )
        return role
    if dataset == "TartanAir":
        if scientific or role is not DatasetRole.PHYSICAL_SANITY_NON_SCIENTIFIC:
            raise CounterfactualContractError(
                "TartanAir may be PHYSICAL_SANITY_NON_SCIENTIFIC only "
                "and cannot enter scientific evidence"
            )
        return role
    if dataset == "UAVLight":
        raise CounterfactualContractError(
            "UAVLight requires a separate future protocol implementation; "
            "the v4 runtime cannot admit it scientifically"
        )
    raise CounterfactualContractError(
        f"dataset is outside the frozen v4/v4.1 admission boundary: {dataset!r}"
    )


def _unit_identity_payload(
    *,
    protocol_provenance: Mapping[str, str],
    model_id: str,
    scene_id: int,
    state_id: str,
    state_identity_sha256: str,
    pair_identity_sha256: str | None,
) -> dict[str, object]:
    return {
        "schema_version": SCIENTIFIC_EXECUTION_UNIT_SCHEMA_VERSION,
        "protocol_provenance": dict(protocol_provenance),
        "dataset": "DTU",
        "model_id": model_id,
        "scene_id": scene_id,
        "state_id": state_id,
        "state_identity_sha256": state_identity_sha256,
        "pair_identity_sha256": pair_identity_sha256,
    }


@dataclass(frozen=True)
class ScientificExecutionUnit:
    protocol_provenance_items: tuple[tuple[str, str], ...]
    dataset: str
    model_id: str
    scene_id: int
    state_id: str
    state_identity_sha256: str
    pair_identity_sha256: str | None
    execution_unit_sha256: str
    schema_version: str = SCIENTIFIC_EXECUTION_UNIT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != SCIENTIFIC_EXECUTION_UNIT_SCHEMA_VERSION:
            raise CounterfactualContractError(
                "scientific execution-unit schema mismatch"
            )
        provenance = dict(self.protocol_provenance_items)
        _validate_protocol_provenance(provenance)
        if self.dataset != "DTU":
            raise CounterfactualContractError(
                "v4 scientific schedule unit dataset must be DTU"
            )
        if self.model_id not in SCIENTIFIC_MODELS:
            raise CounterfactualContractError(
                "v4 scientific schedule model must be exactly VGGT or MASt3R"
            )
        if self.scene_id not in TEST_SCENE_IDS:
            raise CounterfactualContractError(
                "v4 scientific schedule unit scene is not a frozen test scene"
            )
        if self.state_id not in SCIENTIFIC_STATES:
            raise CounterfactualContractError(
                "v4 scientific schedule unit state is not frozen"
            )
        _require_sha256(self.state_identity_sha256, label="state identity SHA-256")
        if self.state_id == "L3":
            if self.pair_identity_sha256 is not None:
                raise CounterfactualContractError(
                    "L3 must be one reused clean/reference unit without a "
                    "second pair execution"
                )
        else:
            _require_sha256(self.pair_identity_sha256, label="pair identity SHA-256")
        _require_sha256(
            self.execution_unit_sha256,
            label="execution-unit identity SHA-256",
        )
        expected = canonical_json_sha256(self.identity_payload())
        if self.execution_unit_sha256 != expected:
            raise CounterfactualContractError(
                "execution-unit identity tamper or mismatch"
            )

    @property
    def protocol_provenance(self) -> dict[str, str]:
        return dict(self.protocol_provenance_items)

    def identity_payload(self) -> dict[str, object]:
        return _unit_identity_payload(
            protocol_provenance=self.protocol_provenance,
            model_id=self.model_id,
            scene_id=self.scene_id,
            state_id=self.state_id,
            state_identity_sha256=self.state_identity_sha256,
            pair_identity_sha256=self.pair_identity_sha256,
        )

    def to_dict(self) -> dict[str, object]:
        return {
            **self.identity_payload(),
            "execution_unit_sha256": self.execution_unit_sha256,
        }

    @classmethod
    def from_dict(cls, value: object) -> "ScientificExecutionUnit":
        row = _require_closed_schema(
            value, _UNIT_KEYS, label="scientific execution unit"
        )
        if row["schema_version"] != SCIENTIFIC_EXECUTION_UNIT_SCHEMA_VERSION:
            raise CounterfactualContractError(
                "scientific execution-unit schema mismatch"
            )
        model_id = row["model_id"]
        state_id = row["state_id"]
        dataset = row["dataset"]
        if not isinstance(model_id, str) or not isinstance(state_id, str):
            raise CounterfactualContractError(
                "execution-unit model_id/state_id must be strings"
            )
        if not isinstance(dataset, str):
            raise CounterfactualContractError("execution-unit dataset must be a string")
        pair_digest = row["pair_identity_sha256"]
        if pair_digest is not None and not isinstance(pair_digest, str):
            raise CounterfactualContractError(
                "pair_identity_sha256 must be a SHA-256 or null"
            )
        return cls(
            protocol_provenance_items=_protocol_items(row["protocol_provenance"]),
            dataset=dataset,
            model_id=model_id,
            scene_id=_require_scene_id(row["scene_id"]),
            state_id=state_id,
            state_identity_sha256=_require_sha256(
                row["state_identity_sha256"],
                label="state identity SHA-256",
            ),
            pair_identity_sha256=pair_digest,
            execution_unit_sha256=_require_sha256(
                row["execution_unit_sha256"],
                label="execution-unit identity SHA-256",
            ),
        )


def _new_execution_unit(
    *,
    protocol_provenance: Mapping[str, str],
    model_id: str,
    state: ModelIndependentState,
    pair_identity_sha256: str | None,
) -> ScientificExecutionUnit:
    payload = _unit_identity_payload(
        protocol_provenance=protocol_provenance,
        model_id=model_id,
        scene_id=state.scene_id,
        state_id=state.state_id,
        state_identity_sha256=state.state_identity_sha256,
        pair_identity_sha256=pair_identity_sha256,
    )
    return ScientificExecutionUnit(
        protocol_provenance_items=tuple(sorted(protocol_provenance.items())),
        dataset="DTU",
        model_id=model_id,
        scene_id=state.scene_id,
        state_id=state.state_id,
        state_identity_sha256=state.state_identity_sha256,
        pair_identity_sha256=pair_identity_sha256,
        execution_unit_sha256=canonical_json_sha256(payload),
    )


def _schedule_payload(
    *,
    protocol_provenance: Mapping[str, str],
    units: tuple[ScientificExecutionUnit, ...],
) -> dict[str, object]:
    return {
        "schema_version": SCIENTIFIC_SCHEDULE_SCHEMA_VERSION,
        "protocol_provenance": dict(protocol_provenance),
        "models": list(SCIENTIFIC_MODELS),
        "scene_ids": list(TEST_SCENE_IDS),
        "state_ids": list(SCIENTIFIC_STATES),
        "units": [unit.to_dict() for unit in units],
    }


@dataclass(frozen=True)
class ScientificSchedule:
    protocol_provenance_items: tuple[tuple[str, str], ...]
    models: tuple[str, ...]
    scene_ids: tuple[int, ...]
    state_ids: tuple[str, ...]
    units: tuple[ScientificExecutionUnit, ...]
    schedule_sha256: str
    schema_version: str = SCIENTIFIC_SCHEDULE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != SCIENTIFIC_SCHEDULE_SCHEMA_VERSION:
            raise CounterfactualContractError("scientific schedule schema mismatch")
        provenance = dict(self.protocol_provenance_items)
        _validate_protocol_provenance(provenance)
        if self.models != SCIENTIFIC_MODELS:
            raise CounterfactualContractError(
                "scientific schedule models must be exactly VGGT and MASt3R"
            )
        if self.scene_ids != TEST_SCENE_IDS:
            raise CounterfactualContractError(
                "scientific schedule scene_ids must be the exact 20 test scenes"
            )
        if self.state_ids != SCIENTIFIC_STATES:
            raise CounterfactualContractError(
                "scientific schedule state_ids must be the exact 10 states"
            )
        if len(self.units) != 400:
            raise CounterfactualContractError(
                "scientific schedule must contain exactly 400 units"
            )
        if any(not isinstance(unit, ScientificExecutionUnit) for unit in self.units):
            raise CounterfactualContractError(
                "scientific schedule units must use the strict v4 unit schema"
            )
        unit_ids = [unit.execution_unit_sha256 for unit in self.units]
        if len(set(unit_ids)) != len(unit_ids):
            raise CounterfactualContractError(
                "scientific schedule contains duplicate execution units"
            )
        actual_keys = [
            (unit.model_id, unit.scene_id, unit.state_id) for unit in self.units
        ]
        if len(set(actual_keys)) != len(actual_keys):
            raise CounterfactualContractError(
                "scientific schedule contains duplicate model/scene/state units"
            )
        expected_keys = tuple(
            (model, scene_id, state_id)
            for model in SCIENTIFIC_MODELS
            for scene_id in TEST_SCENE_IDS
            for state_id in SCIENTIFIC_STATES
        )
        actual_key_set = set(actual_keys)
        expected_key_set = set(expected_keys)
        if actual_key_set != expected_key_set:
            missing = len(expected_key_set - actual_key_set)
            extra = len(actual_key_set - expected_key_set)
            raise CounterfactualContractError(
                "scientific schedule has missing or extra units: "
                f"missing={missing}, extra={extra}"
            )
        if tuple(actual_keys) != expected_keys:
            raise CounterfactualContractError(
                "scientific schedule units must use the exact canonical unit sequence"
            )
        for unit in self.units:
            if unit.protocol_provenance != provenance:
                raise CounterfactualContractError(
                    "scientific schedule unit protocol provenance mismatch"
                )
        model_independent: dict[tuple[int, str], tuple[str, str | None]] = {}
        for unit in self.units:
            key = (unit.scene_id, unit.state_id)
            identities = (
                unit.state_identity_sha256,
                unit.pair_identity_sha256,
            )
            previous = model_independent.setdefault(key, identities)
            if previous != identities:
                raise CounterfactualContractError(
                    "models do not consume identical model-independent "
                    f"state/pair identity for scan{unit.scene_id}/"
                    f"{unit.state_id}"
                )
        if len(model_independent) != 200:
            raise CounterfactualContractError(
                "scientific schedule must expose exactly 200 model-independent states"
            )
        for model in SCIENTIFIC_MODELS:
            if sum(unit.model_id == model for unit in self.units) != 200:
                raise CounterfactualContractError(
                    f"scientific schedule model {model} must have 200 units"
                )
        _require_sha256(self.schedule_sha256, label="schedule fingerprint SHA-256")
        expected_fingerprint = canonical_json_sha256(self.identity_payload())
        if self.schedule_sha256 != expected_fingerprint:
            raise CounterfactualContractError(
                "scientific schedule fingerprint tamper or mismatch"
            )

    @property
    def protocol_provenance(self) -> dict[str, str]:
        return dict(self.protocol_provenance_items)

    def identity_payload(self) -> dict[str, object]:
        return _schedule_payload(
            protocol_provenance=self.protocol_provenance,
            units=self.units,
        )

    def to_dict(self) -> dict[str, object]:
        return {
            **self.identity_payload(),
            "schedule_sha256": self.schedule_sha256,
        }

    def canonical_json_bytes(self) -> bytes:
        return _canonical_file_bytes(self.to_dict())


def build_scientific_schedule(
    states: Iterable[ModelIndependentState],
) -> ScientificSchedule:
    """Build the exact deterministic 400-unit, two-model v4 schedule."""

    rows = tuple(states)
    if len(rows) != 200 or any(
        not isinstance(row, ModelIndependentState) for row in rows
    ):
        raise CounterfactualContractError(
            "schedule construction requires exactly 200 "
            "model-independent state identities"
        )
    by_key: dict[tuple[int, str], ModelIndependentState] = {}
    for state in rows:
        key = (state.scene_id, state.state_id)
        if key in by_key:
            raise CounterfactualContractError(
                f"duplicate model-independent state scan{state.scene_id}/"
                f"{state.state_id}"
            )
        by_key[key] = state
    expected = {
        (scene_id, state_id)
        for scene_id in TEST_SCENE_IDS
        for state_id in SCIENTIFIC_STATES
    }
    if set(by_key) != expected:
        missing = sorted(expected - set(by_key))
        extra = sorted(set(by_key) - expected)
        raise CounterfactualContractError(
            f"schedule state inventory is not exact: missing={missing}, extra={extra}"
        )

    pair_identity_by_key: dict[tuple[int, str], str | None] = {}
    for scene_id in TEST_SCENE_IDS:
        scene_states = {
            state_id: by_key[(scene_id, state_id)] for state_id in SCIENTIFIC_STATES
        }
        validate_dtu_lighting_states(
            tuple(scene_states[state_id] for state_id in LIGHTING_STATES)
        )
        validate_fog_states(
            tuple(scene_states[state_id] for state_id in FOG_BOUNDARY_LAG_SEQUENCE)
        )
        reference = scene_states["L3"]
        for state_id in SCIENTIFIC_STATES:
            if state_id == "L3":
                pair_identity_by_key[(scene_id, state_id)] = None
                continue
            if state_id in LIGHTING_STATES:
                axis = CounterfactualAxis.DTU_LIGHTING
                semantics = AxisSemantics.UNORDERED_DISCRETE
            else:
                axis = CounterfactualAxis.SYNTHETIC_FOG
                semantics = AxisSemantics.ORDERED
            pair = build_counterfactual_pair_manifest(
                reference_state=reference,
                counterfactual_state=scene_states[state_id],
                axis=axis,
                axis_semantics=semantics,
            )
            pair_identity_by_key[(scene_id, state_id)] = pair.pair_identity_sha256

    provenance = rows[0].protocol_provenance
    units = tuple(
        _new_execution_unit(
            protocol_provenance=provenance,
            model_id=model,
            state=by_key[(scene_id, state_id)],
            pair_identity_sha256=pair_identity_by_key[(scene_id, state_id)],
        )
        for model in SCIENTIFIC_MODELS
        for scene_id in TEST_SCENE_IDS
        for state_id in SCIENTIFIC_STATES
    )
    payload = _schedule_payload(protocol_provenance=provenance, units=units)
    return ScientificSchedule(
        protocol_provenance_items=tuple(sorted(provenance.items())),
        models=SCIENTIFIC_MODELS,
        scene_ids=TEST_SCENE_IDS,
        state_ids=SCIENTIFIC_STATES,
        units=units,
        schedule_sha256=canonical_json_sha256(payload),
    )


def parse_scientific_schedule(
    value: bytes | str | Mapping[str, Any],
) -> ScientificSchedule:
    parsed: object = (
        _parse_json(value, label="scientific schedule")
        if isinstance(value, (bytes, str))
        else value
    )
    row = _require_closed_schema(parsed, _SCHEDULE_KEYS, label="scientific schedule")
    if row["schema_version"] != SCIENTIFIC_SCHEDULE_SCHEMA_VERSION:
        raise CounterfactualContractError("scientific schedule schema mismatch")
    models_raw = _require_json_list(row["models"], label="schedule models")
    scenes_raw = _require_json_list(row["scene_ids"], label="schedule scene_ids")
    states_raw = _require_json_list(row["state_ids"], label="schedule state_ids")
    if not all(isinstance(model, str) for model in models_raw):
        raise CounterfactualContractError("schedule models must be strings")
    if not all(
        isinstance(scene_id, int) and not isinstance(scene_id, bool)
        for scene_id in scenes_raw
    ):
        raise CounterfactualContractError("schedule scene_ids must be integers")
    if not all(isinstance(state_id, str) for state_id in states_raw):
        raise CounterfactualContractError("schedule state_ids must be strings")
    units_raw = _require_json_list(row["units"], label="schedule units")
    units = tuple(ScientificExecutionUnit.from_dict(unit) for unit in units_raw)
    return ScientificSchedule(
        protocol_provenance_items=_protocol_items(row["protocol_provenance"]),
        models=tuple(models_raw),
        scene_ids=tuple(scenes_raw),
        state_ids=tuple(states_raw),
        units=units,
        schedule_sha256=_require_sha256(
            row["schedule_sha256"],
            label="schedule fingerprint SHA-256",
        ),
    )


def validate_scientific_schedule(
    value: ScientificSchedule | bytes | str | Mapping[str, Any],
) -> ScientificSchedule:
    """Reparse and validate the exact closed 400-unit schedule contract."""

    if isinstance(value, ScientificSchedule):
        return parse_scientific_schedule(value.to_dict())
    return parse_scientific_schedule(value)
