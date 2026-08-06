"""CPU-only v4 Rectified member closure and missing-member materialization."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import re
import zlib
from typing import Any

from . import toml_compat as tomllib
from .preparation import PreparationError
from .tartanair_range import RemoteZipEntry, RemoteZipIndex, extract_range_members_evidence
from .tartanair_range import index_remote_zip
from .v4_counterfactuals import (
    DTU_LIGHTING_MAPPING_PROVENANCE,
    DTU_LIGHTING_SEMANTIC_TO_PHYSICAL_TOKEN,
    ModelIndependentState,
    SCIENTIFIC_MODELS,
    SCIENTIFIC_STATES,
    TEST_SCENE_IDS,
    canonical_json_sha256,
    parse_scientific_schedule,
)


RECTIFIED_CLOSURE_SCHEMA_VERSION = "georeliab-v4-rectified-member-closure-1.0"
RECTIFIED_MEMBER_SCHEMA_VERSION = "georeliab-v4-rectified-member-1.0"
RECTIFIED_EXPECTED_SET_SCHEMA_VERSION = (
    "georeliab-v4-rectified-expected-set-1.0"
)
RECTIFIED_MATERIALIZE_SCHEMA_VERSION = (
    "georeliab-v4-rectified-materialize-missing-1.0"
)
RECTIFIED_BOOTSTRAP_SCHEMA_VERSION = (
    "georeliab-v4-rectified-production-bootstrap-1.0"
)
RECTIFIED_MEMBER_ILLUMINATIONS = ("L1", "L2", "L4", "L5", "L6", "L7")
RECTIFIED_ACCEPTANCE_SCENE_COUNT = 20
RECTIFIED_ACCEPTANCE_VIEWS_PER_SCENE = 8
RECTIFIED_ACCEPTANCE_BASE_UNITS = 160
RECTIFIED_ACCEPTANCE_MEMBER_COUNT = 960
RECTIFIED_ACCEPTANCE_SCHEDULE_UNITS = 400
RECTIFIED_ACCEPTANCE_NON_L3_LIGHTING_UNITS = 240
RECTIFIED_ACCEPTANCE_L3_REFERENCE_UNITS = 40
RECTIFIED_ACCEPTANCE_FOG_UNITS = 120
RECTIFIED_RANGE_MAX_WORKERS = 16
REFERENCE_ILLUMINATION_ROLE = {
    "L3": "REFERENCE_EXCLUDED_FROM_RECTIFIED_MEMBER_CLOSURE"
}

MANIFEST_NAME = "v4-rectified-member-manifest.jsonl"
SCHEMA_NAME = "v4-rectified-member-manifest.schema.json"
EXPECTED_SET_NAME = "v4-rectified-member-expected-set.json"
AUDIT_NAME = "v4-rectified-member-closure-audit.json"
RECEIPT_NAME = "v4-rectified-member-closure-receipt.json"
MANIFEST_SHA_NAME = "v4-rectified-member-manifest.sha256"
ORDERED_MEMBER_LIST_SHA_NAME = "v4-rectified-member-list.sha256"
GROUP_INDEX_SHA_NAME = "v4-rectified-member-group-index.sha256"
SCHEDULE_BINDING_SHA_NAME = "v4-rectified-schedule-member-binding.sha256"
MATERIALIZE_RECEIPT_NAME = "v4-rectified-member-materialize-receipt.json"
BOOTSTRAP_SCHEDULE_NAME = "v4-rectified-resource-schedule-400.json"

_RECTIFIED_RE = re.compile(
    r"^Rectified/scan(?P<scene>[1-9][0-9]*)/"
    r"rect_(?P<view>[0-9]{3})_(?P<illumination>[0-9]+)_r5000\.png$"
)
_SHA_RE = re.compile(r"^[0-9a-f]{64}$")
_MEMBER_KEYS = frozenset(
    {
        "schema_version",
        "canonical_member_id",
        "scene_id",
        "camera_id",
        "view_id",
        "illumination_id",
        "source_archive_illumination_token",
        "illumination_mapping_provenance",
        "paired_counterfactual_group_id",
        "counterfactual_group_id",
        "schedule_unit_id",
        "schedule_unit_ids",
        "normalized_relative_path",
        "resolved_physical_path",
        "file_size",
        "sha256",
        "width",
        "height",
        "channels",
        "dtype",
        "source_dataset_root_identity",
        "symlink_status",
        "symlink_target",
        "excluded_reference_role_by_illumination",
    }
)


class V4RectifiedClosureError(RuntimeError):
    """Raised when v4 Rectified membership cannot be proven."""


def _semantic_physical_map() -> dict[str, str]:
    mapping = dict(DTU_LIGHTING_SEMANTIC_TO_PHYSICAL_TOKEN)
    expected_keys = {"L1", "L2", "L3", "L4", "L5", "L6", "L7"}
    if set(mapping) != expected_keys:
        raise V4RectifiedClosureError("V4_RECTIFIED_MAPPING_TAMPER")
    if set(mapping.values()) != {"0", "1", "2", "3", "4", "5", "6"}:
        raise V4RectifiedClosureError("V4_RECTIFIED_MAPPING_TAMPER")
    if mapping.get("L3") != "3" or mapping.get("L7") != "0":
        raise V4RectifiedClosureError("V4_RECTIFIED_MAPPING_TAMPER")
    provenance = DTU_LIGHTING_MAPPING_PROVENANCE
    if not isinstance(provenance, Mapping) or provenance.get("bijection") != (
        "L1:1,L2:2,L3:3,L4:4,L5:5,L6:6,L7:0"
    ):
        raise V4RectifiedClosureError("V4_RECTIFIED_MAPPING_TAMPER")
    return mapping


def _json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii") + b"\n"


def _json_sha(value: object) -> str:
    return hashlib.sha256(_json_bytes(value)).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise V4RectifiedClosureError("V4_RECTIFIED_JSON_INVALID") from exc
    if not isinstance(payload, dict):
        raise V4RectifiedClosureError("V4_RECTIFIED_JSON_OBJECT_REQUIRED")
    return payload


def _is_sha(value: object) -> bool:
    return isinstance(value, str) and _SHA_RE.fullmatch(value) is not None


def _views_per_scene(views_by_scene: Mapping[int, Sequence[int]]) -> int:
    widths = {len(tuple(views)) for views in views_by_scene.values()}
    if len(widths) != 1:
        raise V4RectifiedClosureError("V4_RECTIFIED_SCENE_VIEW_MISMATCH")
    return next(iter(widths))


def _cardinality_equation(
    scene_count: int,
    views_per_scene: int,
    member_count: int,
    product: int,
) -> str:
    return f"{scene_count} * {views_per_scene} * {member_count} = {product}"


def _schedule_partition_counts(unit_sets: Mapping[str, Sequence[str]]) -> dict[str, int]:
    return {
        "non_l3_lighting": len(unit_sets["non_l3_lighting"]),
        "l3_reference": len(unit_sets["l3_reference"]),
        "fog": len(unit_sets["fog"]),
        "full_schedule": len(unit_sets["full_schedule"]),
    }


def _atomic_bytes(
    path: Path,
    content: bytes,
    *,
    validator: Callable[[Path], None] | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".partial")
    try:
        with partial.open("wb") as handle:
            handle.write(content)
            handle.flush()
            try:
                os.fsync(handle.fileno())
            except OSError:
                pass
        if validator is not None:
            validator(partial)
        partial.replace(path)
    except Exception:
        try:
            partial.unlink()
        except FileNotFoundError:
            pass
        raise


def _atomic_json(
    path: Path,
    payload: object,
    *,
    validator: Callable[[Path], None] | None = None,
) -> None:
    _atomic_bytes(path, _json_bytes(payload), validator=validator)


def _load_schedule(path: Path):
    try:
        return parse_scientific_schedule(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise V4RectifiedClosureError("V4_RECTIFIED_SCHEDULE_INVALID") from exc


def _load_states(path: Path) -> tuple[ModelIndependentState, ...]:
    payload = _load_json(path)
    raw = payload.get("states", payload.get("model_independent_states"))
    if not isinstance(raw, list):
        raise V4RectifiedClosureError("V4_RECTIFIED_STATE_INVENTORY_INVALID")
    try:
        return tuple(ModelIndependentState.from_dict(row) for row in raw)
    except Exception as exc:
        raise V4RectifiedClosureError("V4_RECTIFIED_STATE_INVENTORY_INVALID") from exc



def _load_resource_expected_set(path: Path) -> dict[str, object]:
    payload = _load_json(path)
    if payload.get("schema_version") != RECTIFIED_EXPECTED_SET_SCHEMA_VERSION:
        raise V4RectifiedClosureError("V4_RECTIFIED_EXPECTED_SET_INVALID")
    _validate_resource_expected_set_authority(payload)
    _validate_expected_set_schedule_accounting(payload)
    return payload


def _load_protocol_mve(protocol_path: Path) -> Mapping[str, object]:
    try:
        payload = tomllib.loads(protocol_path.read_text(encoding="utf-8"))
    except (tomllib.TOMLDecodeError, UnicodeDecodeError, OSError) as exc:
        raise V4RectifiedClosureError("V4_RECTIFIED_PROTOCOL_INVALID") from exc
    mve = payload.get("mve")
    if not isinstance(mve, Mapping):
        raise V4RectifiedClosureError("V4_RECTIFIED_PROTOCOL_INVALID")
    if tuple(mve.get("models", ())) != SCIENTIFIC_MODELS:
        raise V4RectifiedClosureError("V4_RECTIFIED_PROTOCOL_INVALID")
    if tuple(mve.get("states", ())) != SCIENTIFIC_STATES:
        raise V4RectifiedClosureError("V4_RECTIFIED_PROTOCOL_INVALID")
    if list(mve.get("test_scene_ids", ())) != list(TEST_SCENE_IDS):
        raise V4RectifiedClosureError("V4_RECTIFIED_PROTOCOL_INVALID")
    if (
        mve.get("views_per_scene") != RECTIFIED_ACCEPTANCE_VIEWS_PER_SCENE
        or mve.get("scientific_unit_count") != RECTIFIED_ACCEPTANCE_SCHEDULE_UNITS
    ):
        raise V4RectifiedClosureError("V4_RECTIFIED_PROTOCOL_INVALID")
    return mve


def _split_groups(split_view_manifest_path: Path) -> tuple[dict[int, tuple[int, ...]], tuple[tuple[int, int], ...]]:
    payload = _load_json(split_view_manifest_path)
    splits = payload.get("splits")
    scenes = payload.get("test_scene_ids")
    if isinstance(splits, Mapping):
        scenes = splits.get("test", scenes)
    if scenes != list(TEST_SCENE_IDS):
        raise V4RectifiedClosureError("V4_RECTIFIED_SPLIT_TEST_SCENES_MISMATCH")
    views = payload.get("views")
    if not isinstance(views, Mapping):
        raise V4RectifiedClosureError("V4_RECTIFIED_SPLIT_VIEW_MANIFEST_INVALID")
    views_by_scene: dict[int, tuple[int, ...]] = {}
    groups: list[tuple[int, int]] = []
    for scene_id in TEST_SCENE_IDS:
        raw = views.get(str(scene_id), views.get(f"scan{scene_id}"))
        if not isinstance(raw, list):
            raise V4RectifiedClosureError("V4_RECTIFIED_SPLIT_VIEW_MANIFEST_INVALID")
        try:
            ordered = tuple(int(item) for item in raw)
        except (TypeError, ValueError) as exc:
            raise V4RectifiedClosureError("V4_RECTIFIED_SPLIT_VIEW_MANIFEST_INVALID") from exc
        if (
            len(ordered) != RECTIFIED_ACCEPTANCE_VIEWS_PER_SCENE
            or len(set(ordered)) != RECTIFIED_ACCEPTANCE_VIEWS_PER_SCENE
        ):
            raise V4RectifiedClosureError("V4_RECTIFIED_SCENE_VIEW_MISMATCH")
        views_by_scene[scene_id] = ordered
        groups.extend((scene_id, view_id) for view_id in ordered)
    if len(groups) != RECTIFIED_ACCEPTANCE_BASE_UNITS:
        raise V4RectifiedClosureError("V4_RECTIFIED_EXPECTED_CARDINALITY_MISMATCH")
    return views_by_scene, tuple(groups)


def _semantic_unit_role(state_id: str) -> str:
    if state_id == "L3":
        return "l3_reference"
    if state_id in RECTIFIED_MEMBER_ILLUMINATIONS:
        return "non_l3_lighting"
    if state_id.startswith("fog-"):
        return "fog"
    raise V4RectifiedClosureError("V4_RECTIFIED_PROTOCOL_INVALID")


def _semantic_resource_schedule(
    *,
    protocol_path: Path,
    split_view_manifest_path: Path,
) -> tuple[dict[str, object], dict[str, object], tuple[tuple[int, int], ...]]:
    _load_protocol_mve(protocol_path)
    views_by_scene, groups = _split_groups(split_view_manifest_path)
    units: list[dict[str, object]] = []
    unit_sets: dict[str, list[str]] = {
        "non_l3_lighting": [],
        "l3_reference": [],
        "fog": [],
        "full_schedule": [],
    }
    unit_ids_by_scene_state: dict[str, list[str]] = {}
    for model in SCIENTIFIC_MODELS:
        for scene_id in TEST_SCENE_IDS:
            ordered_views = list(views_by_scene[scene_id])
            for state_id in SCIENTIFIC_STATES:
                role = _semantic_unit_role(state_id)
                payload = {
                    "kind": "v4-rectified-resource-schedule-unit",
                    "protocol_sha256": _sha256_file(protocol_path),
                    "split_view_manifest_sha256": _sha256_file(split_view_manifest_path),
                    "model": model,
                    "scene": scene_id,
                    "state": state_id,
                    "ordered_views": ordered_views,
                    "role": role,
                }
                unit_sha = canonical_json_sha256(payload)
                unit = {**payload, "schedule_unit_sha256": unit_sha}
                units.append(unit)
                unit_sets[role].append(unit_sha)
                unit_sets["full_schedule"].append(unit_sha)
                unit_ids_by_scene_state.setdefault(f"{scene_id}:{state_id}", []).append(unit_sha)
    for ids in unit_ids_by_scene_state.values():
        ids.sort()
    for ids in unit_sets.values():
        ids.sort()
    if (
        len(unit_sets["non_l3_lighting"]) != RECTIFIED_ACCEPTANCE_NON_L3_LIGHTING_UNITS
        or len(unit_sets["l3_reference"]) != RECTIFIED_ACCEPTANCE_L3_REFERENCE_UNITS
        or len(unit_sets["fog"]) != RECTIFIED_ACCEPTANCE_FOG_UNITS
        or len(unit_sets["full_schedule"]) != RECTIFIED_ACCEPTANCE_SCHEDULE_UNITS
    ):
        raise V4RectifiedClosureError("V4_RECTIFIED_SCHEDULE_BINDING_UNPROVEN")
    if set(unit_sets["full_schedule"]) != (
        set(unit_sets["non_l3_lighting"])
        | set(unit_sets["l3_reference"])
        | set(unit_sets["fog"])
    ):
        raise V4RectifiedClosureError("V4_RECTIFIED_SCHEDULE_BINDING_UNPROVEN")
    schedule = {
        "schema_version": RECTIFIED_BOOTSTRAP_SCHEMA_VERSION,
        "resource_kind": "rectified_resource_schedule_not_scientific_evidence",
        "protocol_path": str(protocol_path.resolve()),
        "protocol_sha256": _sha256_file(protocol_path),
        "split_view_manifest_path": str(split_view_manifest_path.resolve()),
        "split_view_manifest_sha256": _sha256_file(split_view_manifest_path),
        "models": list(SCIENTIFIC_MODELS),
        "scene_ids": list(TEST_SCENE_IDS),
        "state_ids": list(SCIENTIFIC_STATES),
        "unit_count": len(units),
        "units": units,
        "schedule_unit_sets": unit_sets,
        "schedule_unit_ids_by_scene_state": unit_ids_by_scene_state,
    }
    expected_set = _expected_set_from_groups(
        schedule_path=protocol_path,
        schedule_sha256=_sha256_file(protocol_path),
        scientific_schedule_sha256=_json_sha(schedule),
        state_inventory_path=split_view_manifest_path,
        state_inventory_sha256=_sha256_file(split_view_manifest_path),
        groups=groups,
        views_by_scene=views_by_scene,
        unit_ids_by_state=unit_ids_by_scene_state,
        unit_sets=unit_sets,
        derivation_source="protocol_and_split_view_manifest_resource_schedule",
    )
    expected_set["resource_schedule_sha256"] = _json_sha(schedule)
    expected_set["resource_schedule_kind"] = schedule["resource_kind"]
    return schedule, expected_set, groups


def _sha_list(value: object) -> list[str]:
    if not isinstance(value, list) or any(not _is_sha(item) for item in value):
        raise V4RectifiedClosureError("V4_RECTIFIED_SCHEDULE_BINDING_UNPROVEN")
    if len(value) != len(set(value)):
        raise V4RectifiedClosureError("V4_RECTIFIED_SCHEDULE_BINDING_UNPROVEN")
    return list(value)


def _expected_schedule_keys() -> set[str]:
    return {
        f"{scene_id}:{state_id}"
        for scene_id in TEST_SCENE_IDS
        for state_id in SCIENTIFIC_STATES
    }


def _validate_expected_set_schedule_accounting(expected_set: Mapping[str, object]) -> None:
    unit_sets = expected_set.get("schedule_unit_sets")
    if not isinstance(unit_sets, Mapping):
        raise V4RectifiedClosureError("V4_RECTIFIED_SCHEDULE_BINDING_UNPROVEN")
    required_unit_sets = {"non_l3_lighting", "l3_reference", "fog", "full_schedule"}
    if set(unit_sets) != required_unit_sets:
        raise V4RectifiedClosureError("V4_RECTIFIED_SCHEDULE_BINDING_UNPROVEN")
    non_l3 = set(_sha_list(unit_sets["non_l3_lighting"]))
    l3 = set(_sha_list(unit_sets["l3_reference"]))
    fog = set(_sha_list(unit_sets["fog"]))
    full = set(_sha_list(unit_sets["full_schedule"]))
    if (
        len(non_l3) != RECTIFIED_ACCEPTANCE_NON_L3_LIGHTING_UNITS
        or len(l3) != RECTIFIED_ACCEPTANCE_L3_REFERENCE_UNITS
        or len(fog) != RECTIFIED_ACCEPTANCE_FOG_UNITS
        or len(full) != RECTIFIED_ACCEPTANCE_SCHEDULE_UNITS
    ):
        raise V4RectifiedClosureError("V4_RECTIFIED_EXPECTED_CARDINALITY_MISMATCH")
    if non_l3 & l3 or non_l3 & fog or l3 & fog or (non_l3 | l3 | fog) != full:
        raise V4RectifiedClosureError("V4_RECTIFIED_SCHEDULE_BINDING_UNPROVEN")

    by_scene_state = expected_set.get("schedule_unit_ids_by_scene_state")
    if not isinstance(by_scene_state, Mapping):
        raise V4RectifiedClosureError("V4_RECTIFIED_SCHEDULE_BINDING_UNPROVEN")
    if set(by_scene_state) != _expected_schedule_keys():
        raise V4RectifiedClosureError("V4_RECTIFIED_SCHEDULE_BINDING_UNPROVEN")
    ids_from_keys: set[str] = set()
    for scene_id in TEST_SCENE_IDS:
        for state_id in SCIENTIFIC_STATES:
            ids = _sha_list(by_scene_state[f"{scene_id}:{state_id}"])
            if len(ids) != len(SCIENTIFIC_MODELS):
                raise V4RectifiedClosureError("V4_RECTIFIED_SCHEDULE_BINDING_UNPROVEN")
            ids_set = set(ids)
            if state_id == "L3":
                expected_partition = l3
            elif state_id in RECTIFIED_MEMBER_ILLUMINATIONS:
                expected_partition = non_l3
            elif state_id.startswith("fog-"):
                expected_partition = fog
            else:
                raise V4RectifiedClosureError("V4_RECTIFIED_SCHEDULE_BINDING_UNPROVEN")
            if not ids_set <= expected_partition:
                raise V4RectifiedClosureError("V4_RECTIFIED_SCHEDULE_BINDING_UNPROVEN")
            ids_from_keys.update(ids_set)
    if ids_from_keys != full:
        raise V4RectifiedClosureError("V4_RECTIFIED_SCHEDULE_BINDING_UNPROVEN")


def _validate_resource_schedule_payload(schedule: Mapping[str, object]) -> None:
    required_schedule_keys = {
        "schema_version",
        "resource_kind",
        "protocol_path",
        "protocol_sha256",
        "split_view_manifest_path",
        "split_view_manifest_sha256",
        "models",
        "scene_ids",
        "state_ids",
        "unit_count",
        "units",
        "schedule_unit_sets",
        "schedule_unit_ids_by_scene_state",
    }
    if set(schedule) != required_schedule_keys:
        raise V4RectifiedClosureError("V4_RECTIFIED_RESOURCE_SCHEDULE_TAMPER")
    if schedule.get("schema_version") != RECTIFIED_BOOTSTRAP_SCHEMA_VERSION:
        raise V4RectifiedClosureError("V4_RECTIFIED_RESOURCE_SCHEDULE_TAMPER")
    if schedule.get("resource_kind") != "rectified_resource_schedule_not_scientific_evidence":
        raise V4RectifiedClosureError("V4_RECTIFIED_RESOURCE_SCHEDULE_TAMPER")
    if tuple(schedule.get("models", ())) != SCIENTIFIC_MODELS:
        raise V4RectifiedClosureError("V4_RECTIFIED_RESOURCE_SCHEDULE_TAMPER")
    if tuple(schedule.get("scene_ids", ())) != TEST_SCENE_IDS:
        raise V4RectifiedClosureError("V4_RECTIFIED_RESOURCE_SCHEDULE_TAMPER")
    if tuple(schedule.get("state_ids", ())) != SCIENTIFIC_STATES:
        raise V4RectifiedClosureError("V4_RECTIFIED_RESOURCE_SCHEDULE_TAMPER")
    if not _is_sha(schedule.get("protocol_sha256")) or not _is_sha(schedule.get("split_view_manifest_sha256")):
        raise V4RectifiedClosureError("V4_RECTIFIED_RESOURCE_SCHEDULE_TAMPER")
    units = schedule.get("units")
    if not isinstance(units, list):
        raise V4RectifiedClosureError("V4_RECTIFIED_RESOURCE_SCHEDULE_TAMPER")
    if (
        schedule.get("unit_count") != len(units)
        or len(units) != RECTIFIED_ACCEPTANCE_SCHEDULE_UNITS
    ):
        raise V4RectifiedClosureError("V4_RECTIFIED_EXPECTED_CARDINALITY_MISMATCH")

    required_unit_keys = {
        "kind",
        "protocol_sha256",
        "split_view_manifest_sha256",
        "model",
        "scene",
        "state",
        "ordered_views",
        "role",
        "schedule_unit_sha256",
    }
    rebuilt_unit_sets: dict[str, list[str]] = {
        "non_l3_lighting": [],
        "l3_reference": [],
        "fog": [],
        "full_schedule": [],
    }
    rebuilt_by_scene_state: dict[str, list[str]] = {}
    seen_model_scene_state: set[tuple[str, int, str]] = set()
    views_by_scene: dict[int, set[tuple[int, ...]]] = {scene_id: set() for scene_id in TEST_SCENE_IDS}
    for unit in units:
        if not isinstance(unit, Mapping) or set(unit) != required_unit_keys:
            raise V4RectifiedClosureError("V4_RECTIFIED_RESOURCE_SCHEDULE_TAMPER")
        if unit.get("kind") != "v4-rectified-resource-schedule-unit":
            raise V4RectifiedClosureError("V4_RECTIFIED_RESOURCE_SCHEDULE_TAMPER")
        model = unit.get("model")
        scene = unit.get("scene")
        state = unit.get("state")
        if model not in SCIENTIFIC_MODELS or scene not in TEST_SCENE_IDS or state not in SCIENTIFIC_STATES:
            raise V4RectifiedClosureError("V4_RECTIFIED_RESOURCE_SCHEDULE_TAMPER")
        if unit.get("protocol_sha256") != schedule.get("protocol_sha256"):
            raise V4RectifiedClosureError("V4_RECTIFIED_RESOURCE_SCHEDULE_TAMPER")
        if unit.get("split_view_manifest_sha256") != schedule.get("split_view_manifest_sha256"):
            raise V4RectifiedClosureError("V4_RECTIFIED_RESOURCE_SCHEDULE_TAMPER")
        role = _semantic_unit_role(str(state))
        if unit.get("role") != role:
            raise V4RectifiedClosureError("V4_RECTIFIED_RESOURCE_SCHEDULE_TAMPER")
        raw_views = unit.get("ordered_views")
        if not isinstance(raw_views, list):
            raise V4RectifiedClosureError("V4_RECTIFIED_RESOURCE_SCHEDULE_TAMPER")
        try:
            ordered_views = tuple(int(view) for view in raw_views)
        except (TypeError, ValueError) as exc:
            raise V4RectifiedClosureError("V4_RECTIFIED_RESOURCE_SCHEDULE_TAMPER") from exc
        if (
            len(ordered_views) != RECTIFIED_ACCEPTANCE_VIEWS_PER_SCENE
            or len(set(ordered_views)) != RECTIFIED_ACCEPTANCE_VIEWS_PER_SCENE
        ):
            raise V4RectifiedClosureError("V4_RECTIFIED_EXPECTED_CARDINALITY_MISMATCH")
        views_by_scene[int(scene)].add(ordered_views)
        identity = (str(model), int(scene), str(state))
        if identity in seen_model_scene_state:
            raise V4RectifiedClosureError("V4_RECTIFIED_RESOURCE_SCHEDULE_TAMPER")
        seen_model_scene_state.add(identity)
        unit_without_sha = {key: unit[key] for key in required_unit_keys if key != "schedule_unit_sha256"}
        unit_sha = canonical_json_sha256(unit_without_sha)
        if unit.get("schedule_unit_sha256") != unit_sha:
            raise V4RectifiedClosureError("V4_RECTIFIED_RESOURCE_SCHEDULE_TAMPER")
        rebuilt_unit_sets[role].append(unit_sha)
        rebuilt_unit_sets["full_schedule"].append(unit_sha)
        rebuilt_by_scene_state.setdefault(f"{scene}:{state}", []).append(unit_sha)
    expected_identities = {
        (model, scene_id, state_id)
        for model in SCIENTIFIC_MODELS
        for scene_id in TEST_SCENE_IDS
        for state_id in SCIENTIFIC_STATES
    }
    if seen_model_scene_state != expected_identities:
        raise V4RectifiedClosureError("V4_RECTIFIED_RESOURCE_SCHEDULE_TAMPER")
    if any(len(scene_views) != 1 for scene_views in views_by_scene.values()):
        raise V4RectifiedClosureError("V4_RECTIFIED_SCENE_VIEW_MISMATCH")
    for ids in rebuilt_unit_sets.values():
        ids.sort()
    for ids in rebuilt_by_scene_state.values():
        ids.sort()
    if schedule.get("schedule_unit_sets") != rebuilt_unit_sets:
        raise V4RectifiedClosureError("V4_RECTIFIED_SCHEDULE_BINDING_UNPROVEN")
    if schedule.get("schedule_unit_ids_by_scene_state") != rebuilt_by_scene_state:
        raise V4RectifiedClosureError("V4_RECTIFIED_SCHEDULE_BINDING_UNPROVEN")
    _validate_expected_set_schedule_accounting(schedule)


def _resource_schedule_views(schedule: Mapping[str, object]) -> dict[int, tuple[int, ...]]:
    units = schedule.get("units")
    if not isinstance(units, list):
        raise V4RectifiedClosureError("V4_RECTIFIED_RESOURCE_SCHEDULE_TAMPER")
    views_by_scene: dict[int, set[tuple[int, ...]]] = {scene_id: set() for scene_id in TEST_SCENE_IDS}
    for unit in units:
        if not isinstance(unit, Mapping):
            raise V4RectifiedClosureError("V4_RECTIFIED_RESOURCE_SCHEDULE_TAMPER")
        scene = unit.get("scene")
        raw_views = unit.get("ordered_views")
        if scene not in TEST_SCENE_IDS or not isinstance(raw_views, list):
            raise V4RectifiedClosureError("V4_RECTIFIED_RESOURCE_SCHEDULE_TAMPER")
        views_by_scene[int(scene)].add(tuple(int(view) for view in raw_views))
    if any(len(values) != 1 for values in views_by_scene.values()):
        raise V4RectifiedClosureError("V4_RECTIFIED_SCENE_VIEW_MISMATCH")
    return {scene_id: next(iter(values)) for scene_id, values in views_by_scene.items()}


def _validate_expected_set_cardinality_binding(
    expected_set: Mapping[str, object],
    schedule: Mapping[str, object],
) -> None:
    views_by_scene = _resource_schedule_views(schedule)
    expected_groups = [
        {"scene_id": scene_id, "view_id": view_id}
        for scene_id in TEST_SCENE_IDS
        for view_id in views_by_scene[scene_id]
    ]
    expected_base_units = len(expected_groups)
    expected_member_count = expected_base_units * len(RECTIFIED_MEMBER_ILLUMINATIONS)
    if expected_set.get("scene_ids") != list(TEST_SCENE_IDS):
        raise V4RectifiedClosureError("V4_RECTIFIED_EXPECTED_SET_INVALID")
    if expected_set.get("view_ids_by_scene") != {
        str(scene_id): list(views_by_scene[scene_id]) for scene_id in TEST_SCENE_IDS
    }:
        raise V4RectifiedClosureError("V4_RECTIFIED_EXPECTED_SET_INVALID")
    if expected_set.get("groups") != expected_groups:
        raise V4RectifiedClosureError("V4_RECTIFIED_EXPECTED_CARDINALITY_MISMATCH")
    if (
        expected_base_units != RECTIFIED_ACCEPTANCE_BASE_UNITS
        or expected_member_count != RECTIFIED_ACCEPTANCE_MEMBER_COUNT
    ):
        raise V4RectifiedClosureError("V4_RECTIFIED_EXPECTED_CARDINALITY_MISMATCH")
    if expected_set.get("scene_count") != len(TEST_SCENE_IDS):
        raise V4RectifiedClosureError("V4_RECTIFIED_EXPECTED_CARDINALITY_MISMATCH")
    if expected_set.get("views_per_scene") != _views_per_scene(views_by_scene):
        raise V4RectifiedClosureError("V4_RECTIFIED_EXPECTED_CARDINALITY_MISMATCH")
    if expected_set.get("model_independent_expected_base_units") != expected_base_units:
        raise V4RectifiedClosureError("V4_RECTIFIED_EXPECTED_CARDINALITY_MISMATCH")
    if expected_set.get("expected_member_count") != expected_member_count:
        raise V4RectifiedClosureError("V4_RECTIFIED_EXPECTED_CARDINALITY_MISMATCH")
    proof = expected_set.get("expected_cardinality_proof")
    if not isinstance(proof, Mapping):
        raise V4RectifiedClosureError("V4_RECTIFIED_EXPECTED_CARDINALITY_MISMATCH")
    expected_equation = _cardinality_equation(
        len(TEST_SCENE_IDS),
        _views_per_scene(views_by_scene),
        len(RECTIFIED_MEMBER_ILLUMINATIONS),
        expected_member_count,
    )
    if proof.get("expected_base_units") != expected_base_units:
        raise V4RectifiedClosureError("V4_RECTIFIED_EXPECTED_CARDINALITY_MISMATCH")
    if proof.get("member_illuminations") != len(RECTIFIED_MEMBER_ILLUMINATIONS):
        raise V4RectifiedClosureError("V4_RECTIFIED_EXPECTED_CARDINALITY_MISMATCH")
    if proof.get("equation") != expected_equation or proof.get("product") != expected_member_count:
        raise V4RectifiedClosureError("V4_RECTIFIED_EXPECTED_CARDINALITY_MISMATCH")


def _validate_resource_expected_set_authority(expected_set: Mapping[str, object]) -> None:
    schedule_path_value = expected_set.get("resource_schedule_path")
    schedule_file_sha = expected_set.get("resource_schedule_file_sha256")
    if not isinstance(schedule_path_value, str) or not _is_sha(schedule_file_sha):
        raise V4RectifiedClosureError("V4_RECTIFIED_RESOURCE_SCHEDULE_REQUIRED")
    schedule_path = Path(schedule_path_value)
    if not schedule_path.is_file() or _sha256_file(schedule_path) != schedule_file_sha:
        raise V4RectifiedClosureError("V4_RECTIFIED_RESOURCE_SCHEDULE_TAMPER")
    schedule = _load_json(schedule_path)
    _validate_resource_schedule_payload(schedule)
    if expected_set.get("resource_schedule_sha256") != _json_sha(schedule):
        raise V4RectifiedClosureError("V4_RECTIFIED_RESOURCE_SCHEDULE_TAMPER")
    if expected_set.get("resource_schedule_kind") != schedule.get("resource_kind"):
        raise V4RectifiedClosureError("V4_RECTIFIED_RESOURCE_SCHEDULE_TAMPER")
    for key in ("schedule_unit_sets", "schedule_unit_ids_by_scene_state"):
        if expected_set.get(key) != schedule.get(key):
            raise V4RectifiedClosureError("V4_RECTIFIED_SCHEDULE_BINDING_UNPROVEN")
    _validate_expected_set_cardinality_binding(expected_set, schedule)


def _expected_set_from_groups(
    *,
    schedule_path: Path,
    schedule_sha256: str,
    scientific_schedule_sha256: str,
    state_inventory_path: Path,
    state_inventory_sha256: str,
    groups: Sequence[tuple[int, int]],
    views_by_scene: Mapping[int, Sequence[int]],
    unit_ids_by_state: Mapping[str, Sequence[str]],
    unit_sets: Mapping[str, Sequence[str]],
    derivation_source: str,
) -> dict[str, object]:
    expected_base_units = len(groups)
    expected_member_count = expected_base_units * len(RECTIFIED_MEMBER_ILLUMINATIONS)
    if (
        expected_base_units != RECTIFIED_ACCEPTANCE_BASE_UNITS
        or expected_member_count != RECTIFIED_ACCEPTANCE_MEMBER_COUNT
    ):
        raise V4RectifiedClosureError("V4_RECTIFIED_EXPECTED_CARDINALITY_MISMATCH")
    expected = {
        "schema_version": RECTIFIED_EXPECTED_SET_SCHEMA_VERSION,
        "derivation_source": derivation_source,
        "schedule_path": str(schedule_path.resolve()),
        "schedule_sha256": schedule_sha256,
        "scientific_schedule_sha256": scientific_schedule_sha256,
        "state_inventory_path": str(state_inventory_path.resolve()),
        "state_inventory_sha256": state_inventory_sha256,
        "model_independent_expected_base_units": expected_base_units,
        "expected_member_count": expected_member_count,
        "expected_base_units_times_member_illuminations": expected_member_count,
        "scene_count": len(TEST_SCENE_IDS),
        "views_per_scene": _views_per_scene(views_by_scene),
        "members_per_group": len(RECTIFIED_MEMBER_ILLUMINATIONS),
        "scene_ids": list(TEST_SCENE_IDS),
        "view_ids_by_scene": {
            str(scene_id): list(views_by_scene[scene_id]) for scene_id in TEST_SCENE_IDS
        },
        "member_illumination_ids": list(RECTIFIED_MEMBER_ILLUMINATIONS),
        "excluded_reference_role": dict(REFERENCE_ILLUMINATION_ROLE),
        "semantic_to_physical_suffix": _semantic_physical_map(),
        "illumination_mapping_provenance": dict(DTU_LIGHTING_MAPPING_PROVENANCE),
        "schedule_unit_ids_by_scene_state": {
            str(key): list(value) for key, value in sorted(unit_ids_by_state.items())
        },
        "schedule_binding_accounting": {
            "non_l3_lighting_units": len(unit_sets["non_l3_lighting"]),
            "l3_reference_units": len(unit_sets["l3_reference"]),
            "fog_units": len(unit_sets["fog"]),
            "full_schedule_units": len(unit_sets["full_schedule"]),
            "manifest_rows_cover_lighting_views": expected_member_count,
            "full_schedule_unit_union_proven": True,
        },
        "schedule_unit_sets": {key: list(value) for key, value in unit_sets.items()},
        "expected_cardinality_proof": {
            "scene_count_from_v4_protocol": len(TEST_SCENE_IDS),
            "views_per_scene_from_split_view_manifest": _views_per_scene(views_by_scene),
            "expected_base_units": expected_base_units,
            "member_illuminations": len(RECTIFIED_MEMBER_ILLUMINATIONS),
            "equation": _cardinality_equation(
                len(views_by_scene),
                _views_per_scene(views_by_scene),
                len(RECTIFIED_MEMBER_ILLUMINATIONS),
                expected_member_count,
            ),
            "product": expected_member_count,
        },
        "groups": [
            {"scene_id": scene_id, "view_id": view_id} for scene_id, view_id in groups
        ],
    }
    _validate_expected_set_schedule_accounting(expected)
    return expected

def _derive_base_groups(
    *,
    schedule_path: Path,
    state_inventory_path: Path,
) -> tuple[dict[str, object], tuple[tuple[int, int], ...]]:
    schedule = _load_schedule(schedule_path)
    states = _load_states(state_inventory_path)
    if schedule.models != SCIENTIFIC_MODELS or schedule.scene_ids != TEST_SCENE_IDS:
        raise V4RectifiedClosureError("V4_RECTIFIED_SCHEDULE_INVALID")
    state_by_key = {(state.scene_id, state.state_id): state for state in states}
    schedule_state_keys = {
        (unit.scene_id, unit.state_id)
        for unit in schedule.units
        if unit.model_id == SCIENTIFIC_MODELS[0]
    }
    if set(state_by_key) != schedule_state_keys:
        raise V4RectifiedClosureError("V4_RECTIFIED_SCHEDULE_MANIFEST_MISMATCH")
    unit_ids_by_state: dict[tuple[int, str], list[str]] = {}
    for unit in schedule.units:
        unit_ids_by_state.setdefault((unit.scene_id, unit.state_id), []).append(
            unit.execution_unit_sha256
        )
    for ids in unit_ids_by_state.values():
        ids.sort()
    non_l3_lighting_unit_ids = sorted(
        unit.execution_unit_sha256
        for unit in schedule.units
        if unit.state_id in RECTIFIED_MEMBER_ILLUMINATIONS
    )
    l3_reference_unit_ids = sorted(
        unit.execution_unit_sha256 for unit in schedule.units if unit.state_id == "L3"
    )
    fog_unit_ids = sorted(
        unit.execution_unit_sha256
        for unit in schedule.units
        if str(unit.state_id).startswith("fog-")
    )
    full_schedule_unit_ids = sorted(unit.execution_unit_sha256 for unit in schedule.units)
    categorized = set(non_l3_lighting_unit_ids) | set(l3_reference_unit_ids) | set(fog_unit_ids)
    if categorized != set(full_schedule_unit_ids):
        raise V4RectifiedClosureError("V4_RECTIFIED_SCHEDULE_BINDING_UNPROVEN")
    if (
        len(non_l3_lighting_unit_ids) != RECTIFIED_ACCEPTANCE_NON_L3_LIGHTING_UNITS
        or len(l3_reference_unit_ids) != RECTIFIED_ACCEPTANCE_L3_REFERENCE_UNITS
        or len(fog_unit_ids) != RECTIFIED_ACCEPTANCE_FOG_UNITS
        or len(full_schedule_unit_ids) != RECTIFIED_ACCEPTANCE_SCHEDULE_UNITS
    ):
        raise V4RectifiedClosureError("V4_RECTIFIED_SCHEDULE_BINDING_UNPROVEN")
    groups: list[tuple[int, int]] = []
    views_by_scene: dict[int, tuple[int, ...]] = {}
    for scene_id in schedule.scene_ids:
        scene_states = [
            state_by_key.get((scene_id, state_id))
            for state_id in schedule.state_ids
        ]
        if any(state is None for state in scene_states):
            raise V4RectifiedClosureError("V4_RECTIFIED_SCHEDULE_MANIFEST_MISMATCH")
        view_sets = {state.ordered_view_ids for state in scene_states if state}
        if len(view_sets) != 1:
            raise V4RectifiedClosureError("V4_RECTIFIED_SCENE_VIEW_MISMATCH")
        views = next(iter(view_sets))
        if len(views) != RECTIFIED_ACCEPTANCE_VIEWS_PER_SCENE:
            raise V4RectifiedClosureError("V4_RECTIFIED_SCENE_VIEW_MISMATCH")
        views_by_scene[scene_id] = views
        groups.extend((scene_id, view_id) for view_id in views)
    expected_base_units = len(groups)
    expected_member_count = expected_base_units * len(RECTIFIED_MEMBER_ILLUMINATIONS)
    if expected_member_count != RECTIFIED_ACCEPTANCE_MEMBER_COUNT:
        raise V4RectifiedClosureError("V4_RECTIFIED_EXPECTED_CARDINALITY_MISMATCH")
    expected = {
        "schema_version": RECTIFIED_EXPECTED_SET_SCHEMA_VERSION,
        "schedule_path": str(schedule_path.resolve()),
        "schedule_sha256": _sha256_file(schedule_path),
        "scientific_schedule_sha256": schedule.schedule_sha256,
        "state_inventory_path": str(state_inventory_path.resolve()),
        "state_inventory_sha256": _sha256_file(state_inventory_path),
        "model_independent_expected_base_units": expected_base_units,
        "expected_member_count": expected_member_count,
        "expected_base_units_times_member_illuminations": expected_member_count,
        "scene_count": len(schedule.scene_ids),
        "views_per_scene": _views_per_scene(views_by_scene),
        "members_per_group": len(RECTIFIED_MEMBER_ILLUMINATIONS),
        "scene_ids": list(schedule.scene_ids),
        "view_ids_by_scene": {
            str(scene_id): list(views_by_scene[scene_id])
            for scene_id in schedule.scene_ids
        },
        "member_illumination_ids": list(RECTIFIED_MEMBER_ILLUMINATIONS),
        "excluded_reference_role": dict(REFERENCE_ILLUMINATION_ROLE),
        "semantic_to_physical_suffix": _semantic_physical_map(),
        "illumination_mapping_provenance": dict(DTU_LIGHTING_MAPPING_PROVENANCE),
        "schedule_unit_ids_by_scene_state": {
            f"{scene_id}:{state_id}": unit_ids_by_state[(scene_id, state_id)]
            for scene_id, state_id in sorted(unit_ids_by_state)
        },
        "schedule_binding_accounting": {
            "non_l3_lighting_units": len(non_l3_lighting_unit_ids),
            "l3_reference_units": len(l3_reference_unit_ids),
            "fog_units": len(fog_unit_ids),
            "full_schedule_units": len(full_schedule_unit_ids),
            "manifest_rows_cover_lighting_views": expected_member_count,
            "full_schedule_unit_union_proven": True,
        },
        "schedule_unit_sets": {
            "non_l3_lighting": non_l3_lighting_unit_ids,
            "l3_reference": l3_reference_unit_ids,
            "fog": fog_unit_ids,
            "full_schedule": full_schedule_unit_ids,
        },
        "expected_cardinality_proof": {
            "scene_count_from_schedule": len(schedule.scene_ids),
            "views_per_scene_from_state_inventory": _views_per_scene(views_by_scene),
            "expected_base_units": expected_base_units,
            "member_illuminations": len(RECTIFIED_MEMBER_ILLUMINATIONS),
            "equation": _cardinality_equation(
                len(views_by_scene),
                _views_per_scene(views_by_scene),
                len(RECTIFIED_MEMBER_ILLUMINATIONS),
                expected_member_count,
            ),
            "product": expected_member_count,
        },
        "groups": [
            {"scene_id": scene_id, "view_id": view_id} for scene_id, view_id in groups
        ],
    }
    return expected, tuple(groups)


def _expected_relative_path(scene_id: int, view_id: int, illumination_id: str) -> str:
    token = _semantic_physical_map()[illumination_id]
    return (
        f"Rectified/scan{scene_id}/"
        f"rect_{view_id:03d}_{token}_r5000.png"
    )


def _expected_member_rows(expected: Mapping[str, object]) -> tuple[dict[str, object], ...]:
    groups = expected.get("groups")
    if not isinstance(groups, list):
        raise V4RectifiedClosureError("V4_RECTIFIED_EXPECTED_SET_INVALID")
    rows: list[dict[str, object]] = []
    schedule_sha = str(expected.get("scientific_schedule_sha256"))
    for group in groups:
        if not isinstance(group, Mapping):
            raise V4RectifiedClosureError("V4_RECTIFIED_EXPECTED_SET_INVALID")
        scene_id = int(group["scene_id"])
        view_id = int(group["view_id"])
        group_id = f"DTU:scan{scene_id}:view{view_id:03d}"
        for illumination_id in RECTIFIED_MEMBER_ILLUMINATIONS:
            relative = _expected_relative_path(scene_id, view_id, illumination_id)
            schedule_ids_raw = expected.get("schedule_unit_ids_by_scene_state")
            if not isinstance(schedule_ids_raw, Mapping):
                raise V4RectifiedClosureError("V4_RECTIFIED_EXPECTED_SET_INVALID")
            schedule_unit_ids = schedule_ids_raw.get(f"{scene_id}:{illumination_id}")
            if (
                not isinstance(schedule_unit_ids, list)
                or len(schedule_unit_ids) != 2
                or any(not _is_sha(value) for value in schedule_unit_ids)
            ):
                raise V4RectifiedClosureError("V4_RECTIFIED_SCHEDULE_BINDING_UNPROVEN")
            identity_payload = {
                "schedule_sha256": schedule_sha,
                "scene_id": scene_id,
                "view_id": view_id,
                "illumination_id": illumination_id,
                "source_archive_illumination_token": _semantic_physical_map()[illumination_id],
                "normalized_relative_path": relative,
            }
            rows.append(
                {
                    "scene_id": scene_id,
                    "view_id": view_id,
                    "camera_id": view_id,
                    "illumination_id": illumination_id,
                    "source_archive_illumination_token": _semantic_physical_map()[illumination_id],
                    "counterfactual_group_id": group_id,
                    "paired_counterfactual_group_id": group_id,
                    "normalized_relative_path": relative,
                    "canonical_member_id": canonical_json_sha256(identity_payload),
                    "schedule_unit_id": canonical_json_sha256(
                        {
                            "kind": "rectified-member-binding",
                            "schedule_unit_ids": schedule_unit_ids,
                            **identity_payload,
                        }
                    ),
                    "schedule_unit_ids": schedule_unit_ids,
                }
            )
    return tuple(rows)


def _png_shape(path: Path) -> tuple[int, int, int, str]:
    with path.open("rb") as handle:
        header = handle.read(33)
    if len(header) < 33 or not header.startswith(b"\x89PNG\r\n\x1a\n"):
        raise V4RectifiedClosureError("V4_RECTIFIED_IMAGE_HEADER_INVALID")
    if header[12:16] != b"IHDR":
        raise V4RectifiedClosureError("V4_RECTIFIED_IMAGE_HEADER_INVALID")
    width = int.from_bytes(header[16:20], "big")
    height = int.from_bytes(header[20:24], "big")
    bit_depth = header[24]
    color_type = header[25]
    channels_by_type = {0: 1, 2: 3, 3: 1, 4: 2, 6: 4}
    channels = channels_by_type.get(color_type)
    if channels is None or bit_depth not in {8, 16}:
        raise V4RectifiedClosureError("V4_RECTIFIED_IMAGE_HEADER_INVALID")
    return width, height, channels, f"uint{bit_depth}"


def _source_root_identity(root: Path) -> str:
    resolved = root.resolve()
    return canonical_json_sha256(
        {
            "kind": "v4-rectified-source-root",
            "resolved_root": str(resolved),
        }
    )


def _symlink_info(path: Path) -> tuple[str, str | None]:
    if not path.is_symlink():
        return "not_symlink", None
    try:
        target = os.readlink(path)
    except OSError as exc:
        raise V4RectifiedClosureError("V4_RECTIFIED_SYMLINK_DRIFT") from exc
    return "symlink", target


def _member_path_under_root(root: Path, relative: str) -> Path:
    relative_path = Path(relative)
    if relative_path.is_absolute() or ".." in relative_path.parts:
        raise V4RectifiedClosureError("V4_RECTIFIED_PATH_ESCAPE")
    root_resolved = root.resolve()
    path = root / relative_path
    current = path
    while current != root and current != current.parent:
        if current.exists() and current.is_symlink():
            raise V4RectifiedClosureError("V4_RECTIFIED_SYMLINK_FORBIDDEN")
        current = current.parent
    try:
        path.resolve().relative_to(root_resolved)
    except ValueError as exc:
        raise V4RectifiedClosureError("V4_RECTIFIED_PATH_ESCAPE") from exc
    return path


def _materialized_member(root: Path, expected: Mapping[str, object]) -> dict[str, object]:
    relative = str(expected["normalized_relative_path"])
    path = _member_path_under_root(root, relative)
    if not path.exists():
        raise V4RectifiedClosureError("V4_RECTIFIED_MEMBER_MISSING")
    if not path.is_file():
        raise V4RectifiedClosureError("V4_RECTIFIED_MEMBER_NOT_FILE")
    width, height, channels, dtype = _png_shape(path)
    status, target = _symlink_info(path)
    return {
        "schema_version": RECTIFIED_MEMBER_SCHEMA_VERSION,
        **expected,
        "resolved_physical_path": str(path.resolve()),
        "file_size": path.stat().st_size,
        "sha256": _sha256_file(path),
        "width": width,
        "height": height,
        "channels": channels,
        "dtype": dtype,
        "source_dataset_root_identity": _source_root_identity(root),
        "symlink_status": status,
        "symlink_target": target,
        "excluded_reference_role_by_illumination": dict(REFERENCE_ILLUMINATION_ROLE),
        "illumination_mapping_provenance": dict(DTU_LIGHTING_MAPPING_PROVENANCE),
    }


def _scan_unexpected_illuminations(
    root: Path,
    groups: Sequence[tuple[int, int]],
) -> None:
    mapping = _semantic_physical_map()
    allowed = {mapping[illumination] for illumination in RECTIFIED_MEMBER_ILLUMINATIONS}
    for scene_id, view_id in groups:
        directory = root / "Rectified" / f"scan{scene_id}"
        if not directory.exists():
            continue
        for path in directory.glob(f"rect_{view_id:03d}_*_r5000.png"):
            relative = path.relative_to(root).as_posix()
            match = _RECTIFIED_RE.fullmatch(relative)
            if match is None:
                raise V4RectifiedClosureError("V4_RECTIFIED_UNEXPECTED_MEMBER")
            illumination_number = match.group("illumination")
            if illumination_number == mapping["L3"]:
                continue
            if illumination_number not in allowed:
                raise V4RectifiedClosureError(
                    "V4_RECTIFIED_UNEXPECTED_ILLUMINATION"
                )


def _read_manifest(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise V4RectifiedClosureError(
                f"V4_RECTIFIED_MANIFEST_JSON_INVALID:{line_number}"
            ) from exc
        if not isinstance(row, dict):
            raise V4RectifiedClosureError("V4_RECTIFIED_MANIFEST_ROW_INVALID")
        rows.append(row)
    return rows


def _manifest_bytes(rows: Sequence[Mapping[str, object]]) -> bytes:
    return b"".join(_json_bytes(dict(row)) for row in rows)


def _validate_row_schema(row: Mapping[str, object]) -> None:
    if set(row) != _MEMBER_KEYS:
        if "excluded_reference_role_by_illumination" not in row:
            raise V4RectifiedClosureError("V4_RECTIFIED_L3_ROLE_UNDECLARED")
        raise V4RectifiedClosureError("V4_RECTIFIED_MEMBER_SCHEMA_INVALID")
    if row.get("schema_version") != RECTIFIED_MEMBER_SCHEMA_VERSION:
        raise V4RectifiedClosureError("V4_RECTIFIED_MEMBER_SCHEMA_INVALID")
    if row.get("illumination_id") == "L3":
        raise V4RectifiedClosureError("V4_RECTIFIED_REFERENCE_INCLUDED")
    if row.get("illumination_id") not in RECTIFIED_MEMBER_ILLUMINATIONS:
        raise V4RectifiedClosureError("V4_RECTIFIED_UNEXPECTED_ILLUMINATION")
    if row.get("excluded_reference_role_by_illumination") != REFERENCE_ILLUMINATION_ROLE:
        raise V4RectifiedClosureError("V4_RECTIFIED_L3_ROLE_UNDECLARED")
    if row.get("illumination_mapping_provenance") != DTU_LIGHTING_MAPPING_PROVENANCE:
        raise V4RectifiedClosureError("V4_RECTIFIED_MAPPING_TAMPER")
    illumination = str(row.get("illumination_id"))
    if row.get("source_archive_illumination_token") != _semantic_physical_map()[illumination]:
        raise V4RectifiedClosureError("V4_RECTIFIED_MAPPING_TAMPER")
    schedule_unit_ids = row.get("schedule_unit_ids")
    if (
        not isinstance(schedule_unit_ids, list)
        or len(schedule_unit_ids) != 2
        or any(not _is_sha(value) for value in schedule_unit_ids)
    ):
        raise V4RectifiedClosureError("V4_RECTIFIED_SCHEDULE_BINDING_UNPROVEN")
    for key in ("canonical_member_id", "schedule_unit_id", "sha256"):
        if not _is_sha(row.get(key)):
            raise V4RectifiedClosureError("V4_RECTIFIED_MEMBER_HASH_INVALID")


def _validate_manifest_rows(
    *,
    root: Path,
    expected_rows: Sequence[Mapping[str, object]],
    actual_rows: Sequence[Mapping[str, object]],
    expected_set: Mapping[str, object],
) -> dict[str, object]:
    if len(actual_rows) != len(expected_rows):
        raise V4RectifiedClosureError("V4_RECTIFIED_MANIFEST_CARDINALITY_MISMATCH")
    expected_by_member = {str(row["canonical_member_id"]): row for row in expected_rows}
    if len(expected_by_member) != len(expected_rows):
        raise V4RectifiedClosureError("V4_RECTIFIED_EXPECTED_SET_INVALID")
    raw_member_ids = [str(row.get("canonical_member_id")) for row in actual_rows]
    if len(set(raw_member_ids)) != len(raw_member_ids):
        raise V4RectifiedClosureError("V4_RECTIFIED_DUPLICATE_MEMBER")
    raw_paths_lower = [str(row.get("normalized_relative_path")).lower() for row in actual_rows]
    if len(set(raw_paths_lower)) != len(raw_paths_lower):
        raise V4RectifiedClosureError("V4_RECTIFIED_PATH_NORMALIZATION_DUPLICATE")
    raw_groups: dict[str, list[str]] = {}
    for row in actual_rows:
        group_id = str(row.get("counterfactual_group_id"))
        raw_groups.setdefault(group_id, []).append(str(row.get("illumination_id")))
    for illuminations in raw_groups.values():
        if all(value in RECTIFIED_MEMBER_ILLUMINATIONS for value in illuminations) and sorted(illuminations) != sorted(RECTIFIED_MEMBER_ILLUMINATIONS):
            raise V4RectifiedClosureError("V4_RECTIFIED_GROUP_INCOMPLETE")

    seen_members: set[str] = set()
    seen_paths_lower: set[str] = set()
    groups: dict[str, list[str]] = {}
    dimensions_by_group: dict[str, tuple[object, ...]] = {}
    for row in actual_rows:
        _validate_row_schema(row)
        member_id = str(row["canonical_member_id"])
        seen_members.add(member_id)
        relative = str(row["normalized_relative_path"])
        lowered = relative.lower()
        seen_paths_lower.add(lowered)
        expected = expected_by_member.get(member_id)
        if expected is None:
            raise V4RectifiedClosureError("V4_RECTIFIED_ORPHAN_MEMBER")
        for key, value in expected.items():
            if row.get(key) != value:
                raise V4RectifiedClosureError("V4_RECTIFIED_MEMBER_IDENTITY_MISMATCH")
        path = _member_path_under_root(root, relative)
        status, target = _symlink_info(path)
        if row.get("symlink_status") != status or row.get("symlink_target") != target:
            raise V4RectifiedClosureError("V4_RECTIFIED_SYMLINK_DRIFT")
        if str(path.resolve()) != row.get("resolved_physical_path"):
            raise V4RectifiedClosureError("V4_RECTIFIED_SYMLINK_DRIFT")
        if not path.exists() or _sha256_file(path) != row.get("sha256"):
            raise V4RectifiedClosureError("V4_RECTIFIED_MEMBER_HASH_MISMATCH")
        if path.stat().st_size != row.get("file_size"):
            raise V4RectifiedClosureError("V4_RECTIFIED_MEMBER_HASH_MISMATCH")
        width, height, channels, dtype = _png_shape(path)
        if row.get("width") != width or row.get("height") != height:
            raise V4RectifiedClosureError("V4_RECTIFIED_DIMENSION_MISMATCH")
        if row.get("channels") != channels:
            raise V4RectifiedClosureError("V4_RECTIFIED_DIMENSION_MISMATCH")
        if row.get("dtype") != dtype:
            raise V4RectifiedClosureError("V4_RECTIFIED_DTYPE_MISMATCH")
        group_id = str(row["counterfactual_group_id"])
        groups.setdefault(group_id, []).append(str(row["illumination_id"]))
        dimensions = (row["width"], row["height"], row["channels"], row["dtype"])
        previous_dimensions = dimensions_by_group.setdefault(group_id, dimensions)
        if previous_dimensions != dimensions:
            raise V4RectifiedClosureError("V4_RECTIFIED_DIMENSION_MISMATCH")
    if seen_members != set(expected_by_member):
        raise V4RectifiedClosureError("V4_RECTIFIED_DANGLING_SCHEDULE_MEMBER")
    for illuminations in groups.values():
        if sorted(illuminations) != sorted(RECTIFIED_MEMBER_ILLUMINATIONS):
            raise V4RectifiedClosureError("V4_RECTIFIED_GROUP_INCOMPLETE")
    expected_group_count = int(expected_set.get("model_independent_expected_base_units", -1))
    if len(groups) != expected_group_count:
        raise V4RectifiedClosureError("V4_RECTIFIED_GROUP_CARDINALITY_MISMATCH")
    unit_sets = expected_set.get("schedule_unit_sets")
    if not isinstance(unit_sets, Mapping):
        raise V4RectifiedClosureError("V4_RECTIFIED_SCHEDULE_BINDING_UNPROVEN")
    full = set(unit_sets.get("full_schedule", []))
    categorized = (
        set(unit_sets.get("non_l3_lighting", []))
        | set(unit_sets.get("l3_reference", []))
        | set(unit_sets.get("fog", []))
    )
    if len(full) != RECTIFIED_ACCEPTANCE_SCHEDULE_UNITS or categorized != full:
        raise V4RectifiedClosureError("V4_RECTIFIED_SCHEDULE_BINDING_UNPROVEN")
    ordered_member_ids = [str(row["canonical_member_id"]) for row in actual_rows]
    group_index = {
        group_id: sorted(illuminations) for group_id, illuminations in sorted(groups.items())
    }
    return {
        "schema_version": RECTIFIED_CLOSURE_SCHEMA_VERSION,
        "status": "PASS",
        "manifest_rows": len(actual_rows),
        "group_count": len(groups),
        "ordered_member_list_sha256": _json_sha(ordered_member_ids),
        "group_index_sha256": _json_sha(group_index),
        "schedule_to_member_binding_sha256": _json_sha(
            [
                {
                    "schedule_unit_id": row["schedule_unit_id"],
                    "schedule_unit_ids": row["schedule_unit_ids"],
                    "canonical_member_id": row["canonical_member_id"],
                }
                for row in actual_rows
            ]
        ),
    }


def _schema_payload() -> dict[str, object]:
    return {
        "schema_version": RECTIFIED_CLOSURE_SCHEMA_VERSION,
        "member_schema_version": RECTIFIED_MEMBER_SCHEMA_VERSION,
        "required_keys": sorted(_MEMBER_KEYS),
        "forbidden_identity_inputs": ["timestamp", "mtime"],
        "member_illumination_ids": list(RECTIFIED_MEMBER_ILLUMINATIONS),
        "excluded_reference_role": dict(REFERENCE_ILLUMINATION_ROLE),
    }


def _write_text_digest(path: Path, digest: str) -> None:
    _atomic_bytes(path, (digest + "\n").encode("ascii"))


def _publish_closure(
    *,
    root: Path,
    schedule_path: Path | None,
    state_inventory_path: Path | None,
    output_dir: Path,
    manifest_rows: Sequence[Mapping[str, object]],
    expected_set: Mapping[str, object],
) -> dict[str, object]:
    expected_rows = _expected_member_rows(expected_set)
    audit = _validate_manifest_rows(
        root=root,
        expected_rows=expected_rows,
        actual_rows=manifest_rows,
        expected_set=expected_set,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / MANIFEST_NAME
    schema_path = output_dir / SCHEMA_NAME
    expected_path = output_dir / EXPECTED_SET_NAME
    audit_path = output_dir / AUDIT_NAME
    receipt_path = output_dir / RECEIPT_NAME

    def validate_manifest_file(path: Path) -> None:
        _validate_manifest_rows(
            root=root,
            expected_rows=expected_rows,
            actual_rows=_read_manifest(path),
            expected_set=expected_set,
        )

    _atomic_json(schema_path, _schema_payload())
    _atomic_json(expected_path, expected_set)
    _atomic_bytes(
        manifest_path,
        _manifest_bytes(manifest_rows),
        validator=validate_manifest_file,
    )
    audit.update(
        {
            "manifest_path": str(manifest_path),
            "manifest_sha256": _sha256_file(manifest_path),
            "expected_set_path": str(expected_path),
            "expected_set_sha256": _sha256_file(expected_path),
            "schedule_path": str(schedule_path.resolve()) if schedule_path is not None else expected_set.get("schedule_path"),
            "schedule_sha256": _sha256_file(schedule_path) if schedule_path is not None else expected_set.get("schedule_sha256"),
            "state_inventory_path": str(state_inventory_path.resolve()) if state_inventory_path is not None else expected_set.get("state_inventory_path"),
            "state_inventory_sha256": _sha256_file(state_inventory_path) if state_inventory_path is not None else expected_set.get("state_inventory_sha256"),
        }
    )
    _atomic_json(audit_path, audit)
    _write_text_digest(output_dir / MANIFEST_SHA_NAME, audit["manifest_sha256"])
    _write_text_digest(
        output_dir / ORDERED_MEMBER_LIST_SHA_NAME,
        str(audit["ordered_member_list_sha256"]),
    )
    _write_text_digest(output_dir / GROUP_INDEX_SHA_NAME, str(audit["group_index_sha256"]))
    _write_text_digest(
        output_dir / SCHEDULE_BINDING_SHA_NAME,
        str(audit["schedule_to_member_binding_sha256"]),
    )
    receipt = {
        "schema_version": RECTIFIED_CLOSURE_SCHEMA_VERSION,
        "status": "PASS",
        "reason_code": "V4_RECTIFIED_MEMBER_CLOSURE_PASS",
        "manifest_path": str(manifest_path),
        "manifest_sha256": audit["manifest_sha256"],
        "schema_path": str(schema_path),
        "expected_set_path": str(expected_path),
        "closure_audit_path": str(audit_path),
        "expected_base_units": expected_set["model_independent_expected_base_units"],
        "expected_member_count": expected_set["expected_member_count"],
        "excluded_reference_role": dict(REFERENCE_ILLUMINATION_ROLE),
        "ordered_member_list_sha256": audit["ordered_member_list_sha256"],
        "group_index_sha256": audit["group_index_sha256"],
        "schedule_to_member_binding_sha256": audit["schedule_to_member_binding_sha256"],
        "manifest_sha256_path": str(output_dir / MANIFEST_SHA_NAME),
        "ordered_member_list_sha256_path": str(output_dir / ORDERED_MEMBER_LIST_SHA_NAME),
        "group_index_sha256_path": str(output_dir / GROUP_INDEX_SHA_NAME),
        "schedule_to_member_binding_sha256_path": str(output_dir / SCHEDULE_BINDING_SHA_NAME),
    }
    _atomic_json(receipt_path, receipt)
    return {**receipt, "receipt_path": str(receipt_path)}



def _groups_from_expected_set(expected_set: Mapping[str, object]) -> tuple[tuple[int, int], ...]:
    groups = expected_set.get("groups")
    if not isinstance(groups, list):
        raise V4RectifiedClosureError("V4_RECTIFIED_EXPECTED_SET_INVALID")
    result: list[tuple[int, int]] = []
    for group in groups:
        if not isinstance(group, Mapping):
            raise V4RectifiedClosureError("V4_RECTIFIED_EXPECTED_SET_INVALID")
        result.append((int(group["scene_id"]), int(group["view_id"])))
    if (
        len(result) != RECTIFIED_ACCEPTANCE_BASE_UNITS
        or len(set(result)) != RECTIFIED_ACCEPTANCE_BASE_UNITS
    ):
        raise V4RectifiedClosureError("V4_RECTIFIED_EXPECTED_CARDINALITY_MISMATCH")
    return tuple(result)


def _resolve_expected_set(
    *,
    schedule_path: Path | None,
    state_inventory_path: Path | None,
    expected_set_path: Path | None,
) -> tuple[dict[str, object], tuple[tuple[int, int], ...]]:
    _ = (schedule_path, state_inventory_path)
    if expected_set_path is None:
        raise V4RectifiedClosureError("V4_RECTIFIED_EXPECTED_SET_REQUIRED")
    expected_set = _load_resource_expected_set(expected_set_path)
    return expected_set, _groups_from_expected_set(expected_set)



def _validate_bootstrap_schedule_file(path: Path) -> None:
    staged = _load_json(path)
    if (
        staged.get("resource_kind") != "rectified_resource_schedule_not_scientific_evidence"
        or staged.get("unit_count") != RECTIFIED_ACCEPTANCE_SCHEDULE_UNITS
    ):
        raise V4RectifiedClosureError("V4_RECTIFIED_SCHEDULE_BINDING_UNPROVEN")

def prepare_rectified_resource_schedule(
    *,
    protocol_path: Path,
    split_view_manifest_path: Path,
    output_dir: Path,
) -> dict[str, object]:
    """Create the CPU-only resource schedule and Rectified expected set."""

    schedule, expected_set, _groups = _semantic_resource_schedule(
        protocol_path=protocol_path,
        split_view_manifest_path=split_view_manifest_path,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    schedule_path = output_dir / BOOTSTRAP_SCHEDULE_NAME
    expected_path = output_dir / EXPECTED_SET_NAME
    _atomic_json(
        schedule_path,
        schedule,
        validator=_validate_bootstrap_schedule_file,
    )
    expected_set["resource_schedule_path"] = str(schedule_path.resolve())
    expected_set["resource_schedule_file_sha256"] = _sha256_file(schedule_path)
    _atomic_json(
        expected_path,
        expected_set,
        validator=lambda staged: _load_resource_expected_set(staged),
    )
    accounting = expected_set["schedule_binding_accounting"]
    return {
        "schema_version": RECTIFIED_BOOTSTRAP_SCHEMA_VERSION,
        "status": "PASS",
        "reason_code": "V4_RECTIFIED_RESOURCE_SCHEDULE_PASS",
        "resource_schedule_path": str(schedule_path),
        "resource_schedule_sha256": _sha256_file(schedule_path),
        "expected_set_path": str(expected_path),
        "expected_set_sha256": _sha256_file(expected_path),
        "unit_count": accounting["full_schedule_units"],
        "expected_base_units": expected_set["model_independent_expected_base_units"],
        "expected_member_count": expected_set["expected_member_count"],
        "unit_partition": {
            "non_l3_lighting": accounting["non_l3_lighting_units"],
            "l3_reference": accounting["l3_reference_units"],
            "fog": accounting["fog_units"],
            "full_schedule": accounting["full_schedule_units"],
            "union_disjointness_proven": accounting["full_schedule_unit_union_proven"],
        },
    }

def create_rectified_member_closure(
    *,
    root: Path,
    output_dir: Path,
    schedule_path: Path | None = None,
    state_inventory_path: Path | None = None,
    expected_set_path: Path | None = None,
) -> dict[str, object]:
    """Generate and validate the exact 960-member v4 Rectified manifest."""

    resolved_root = root.resolve()
    expected_set, groups = _resolve_expected_set(
        schedule_path=schedule_path,
        state_inventory_path=state_inventory_path,
        expected_set_path=expected_set_path,
    )
    _scan_unexpected_illuminations(resolved_root, groups)
    expected_rows = _expected_member_rows(expected_set)
    manifest_rows = [
        _materialized_member(resolved_root, expected) for expected in expected_rows
    ]
    return _publish_closure(
        root=resolved_root,
        schedule_path=schedule_path,
        state_inventory_path=state_inventory_path,
        output_dir=output_dir,
        manifest_rows=manifest_rows,
        expected_set=expected_set,
    )


def validate_rectified_member_closure(
    *,
    root: Path,
    manifest_path: Path,
    output_dir: Path,
    schedule_path: Path | None = None,
    state_inventory_path: Path | None = None,
    expected_set_path: Path | None = None,
) -> dict[str, object]:
    """Validate an existing v4 Rectified member manifest and publish audit artifacts."""

    resolved_root = root.resolve()
    expected_set, groups = _resolve_expected_set(
        schedule_path=schedule_path,
        state_inventory_path=state_inventory_path,
        expected_set_path=expected_set_path,
    )
    _scan_unexpected_illuminations(resolved_root, groups)
    manifest_rows = _read_manifest(manifest_path)
    return _publish_closure(
        root=resolved_root,
        schedule_path=schedule_path,
        state_inventory_path=state_inventory_path,
        output_dir=output_dir,
        manifest_rows=manifest_rows,
        expected_set=expected_set,
    )


def audit_rectified_member_closure_read_only(
    *,
    root: Path,
    manifest_path: Path,
    expected_set_path: Path,
) -> dict[str, object]:
    resolved_root = root.resolve()
    expected_set, groups = _resolve_expected_set(
        schedule_path=None,
        state_inventory_path=None,
        expected_set_path=expected_set_path,
    )
    _scan_unexpected_illuminations(resolved_root, groups)
    audit = _validate_manifest_rows(
        root=resolved_root,
        expected_rows=_expected_member_rows(expected_set),
        actual_rows=_read_manifest(manifest_path),
        expected_set=expected_set,
    )
    return {
        **audit,
        'manifest_path': str(manifest_path.resolve()),
        'manifest_sha256': _sha256_file(manifest_path),
        'expected_set_path': str(expected_set_path.resolve()),
        'expected_set_sha256': _sha256_file(expected_set_path),
    }


def _index_entries(index: object) -> Mapping[str, object]:
    if isinstance(index, RemoteZipIndex):
        return index.entries
    if isinstance(index, Mapping):
        entries = index.get("entries", index)
        if isinstance(entries, Mapping):
            return entries
    raise V4RectifiedClosureError("V4_RECTIFIED_OFFICIAL_INDEX_INVALID")


def _entry_sha(entry: object) -> str | None:
    if isinstance(entry, Mapping):
        value = entry.get("sha256") or entry.get("raw_sha256")
        if isinstance(value, str):
            return value
    return None


def _entry_size_crc(entry: object) -> tuple[int | None, int | None]:
    if isinstance(entry, RemoteZipEntry):
        return entry.uncompressed_size, entry.crc32
    if not isinstance(entry, Mapping):
        return None, None
    raw_size = entry.get("uncompressed_size", entry.get("bytes"))
    raw_crc = entry.get("crc32")
    try:
        size = int(raw_size) if raw_size is not None else None
    except (TypeError, ValueError):
        size = None
    try:
        if isinstance(raw_crc, str):
            crc = int(raw_crc, 16)
        elif raw_crc is not None:
            crc = int(raw_crc)
        else:
            crc = None
    except (TypeError, ValueError):
        crc = None
    return size, crc


def _file_sha_crc(path: Path) -> tuple[str, int]:
    digest = hashlib.sha256()
    crc = 0
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
            crc = zlib.crc32(block, crc)
    return digest.hexdigest(), crc & 0xFFFFFFFF


def _validate_existing_against_official_entry(path: Path, entry: object) -> None:
    expected_size, expected_crc = _entry_size_crc(entry)
    if expected_size is not None and path.stat().st_size != expected_size:
        raise V4RectifiedClosureError("V4_RECTIFIED_OFFICIAL_MEMBER_HASH_MISMATCH")
    actual_sha, actual_crc = _file_sha_crc(path)
    if expected_crc is not None and actual_crc != expected_crc:
        raise V4RectifiedClosureError("V4_RECTIFIED_OFFICIAL_MEMBER_HASH_MISMATCH")
    expected_sha = _entry_sha(entry)
    if expected_sha is not None and actual_sha != expected_sha:
        raise V4RectifiedClosureError("V4_RECTIFIED_OFFICIAL_MEMBER_HASH_MISMATCH")
    if expected_sha is None and expected_size is None and expected_crc is None:
        raise V4RectifiedClosureError("V4_RECTIFIED_OFFICIAL_INDEX_INVALID")


def _materialize_range_members_concurrently(
    *,
    archive: Path | str,
    index: object,
    members: Sequence[str],
    root: Path,
) -> tuple[list[dict[str, object]], int]:
    """Extract independent members concurrently while retaining canonical order."""

    if not members:
        return [], 0
    worker_count = min(RECTIFIED_RANGE_MAX_WORKERS, len(members))

    def extract_one(member: str) -> dict[str, object]:
        evidence = extract_range_members_evidence(
            str(archive),
            index,  # type: ignore[arg-type]
            [member],
            root,
        )
        try:
            return dict(evidence[member])
        except KeyError as exc:
            raise V4RectifiedClosureError(
                "V4_RECTIFIED_MATERIALIZE_EVIDENCE_MISSING"
            ) from exc

    with ThreadPoolExecutor(
        max_workers=worker_count,
        thread_name_prefix="v4-rectified-range",
    ) as executor:
        futures = {member: executor.submit(extract_one, member) for member in members}
        try:
            rows = [futures[member].result() for member in members]
        except Exception:
            for future in futures.values():
                future.cancel()
            raise

    return [
        {
            "member": member,
            "sha256": row.get("raw_sha256"),
            "disposition": row.get("disposition"),
        }
        for member, row in zip(members, rows, strict=True)
    ], worker_count


def materialize_missing_rectified_members(
    *,
    root: Path,
    schedule_path: Path | None = None,
    state_inventory_path: Path | None = None,
    expected_set_path: Path | None = None,
    official_rectified_archive: Path | str,
    output_dir: Path,
    indexer: Callable[[Path | str], object] | None = None,
    extractor: Callable[[Path | str, str, Path, str], None] | None = None,
) -> dict[str, object]:
    """Fetch only missing expected official Rectified members into ``root``."""

    resolved_root = root.resolve()
    expected_set, _groups = _resolve_expected_set(
        schedule_path=schedule_path,
        state_inventory_path=state_inventory_path,
        expected_set_path=expected_set_path,
    )
    expected_rows = _expected_member_rows(expected_set)
    expected_members = [str(row["normalized_relative_path"]) for row in expected_rows]
    archive = official_rectified_archive
    try:
        index = (indexer or (lambda value: index_remote_zip(str(value))))(archive)
    except PreparationError as exc:
        raise V4RectifiedClosureError("V4_RECTIFIED_OFFICIAL_INDEX_INVALID") from exc
    entries = _index_entries(index)
    missing: list[str] = []
    for member in expected_members:
        if member not in entries:
            raise V4RectifiedClosureError("V4_RECTIFIED_OFFICIAL_MEMBER_MISSING")
        destination = _member_path_under_root(resolved_root, member)
        if destination.exists():
            if not destination.is_file():
                raise V4RectifiedClosureError("V4_RECTIFIED_MEMBER_NOT_FILE")
            _validate_existing_against_official_entry(destination, entries[member])
        else:
            missing.append(member)
    materialized: list[dict[str, object]] = []
    range_worker_count = 0
    if extractor is None:
        try:
            materialized, range_worker_count = _materialize_range_members_concurrently(
                archive=archive,
                index=index,
                members=missing,
                root=resolved_root,
            )
        except PreparationError as exc:
            raise V4RectifiedClosureError("V4_RECTIFIED_OFFICIAL_MEMBER_HASH_MISMATCH") from exc
    else:
        range_worker_count = 1 if missing else 0
        for member in missing:
            destination = _member_path_under_root(resolved_root, member)
            partial = destination.with_name(destination.name + ".partial")
            expected_sha = _entry_sha(entries[member])
            if expected_sha is None:
                expected_sha = ""
            try:
                extractor(archive, member, partial, expected_sha)
                if not partial.is_file():
                    raise V4RectifiedClosureError(
                        "V4_RECTIFIED_MATERIALIZE_PARTIAL_MISSING"
                    )
                actual_sha = _sha256_file(partial)
                if expected_sha and actual_sha != expected_sha:
                    raise V4RectifiedClosureError(
                        "V4_RECTIFIED_OFFICIAL_MEMBER_HASH_MISMATCH"
                    )
                destination.parent.mkdir(parents=True, exist_ok=True)
                partial.replace(destination)
                materialized.append(
                    {
                        "member": member,
                        "sha256": actual_sha,
                        "disposition": "written",
                    }
                )
            except Exception:
                try:
                    partial.unlink()
                except FileNotFoundError:
                    pass
                raise
    output_dir.mkdir(parents=True, exist_ok=True)
    receipt = {
        "schema_version": RECTIFIED_MATERIALIZE_SCHEMA_VERSION,
        "status": "PASS",
        "reason_code": "V4_RECTIFIED_MATERIALIZE_MISSING_PASS",
        "root": str(resolved_root),
        "official_rectified_archive": str(official_rectified_archive),
        "expected_set_sha256": _json_sha(expected_set),
        "missing_count": len(missing),
        "materialized_count": len(materialized),
        "range_worker_count": range_worker_count,
        "materialized_members": materialized,
    }
    receipt_path = output_dir / MATERIALIZE_RECEIPT_NAME
    _atomic_json(receipt_path, receipt)
    return {**receipt, "receipt_path": str(receipt_path)}
