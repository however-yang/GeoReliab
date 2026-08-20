"""Test-only Pilot resource, input-closure, and CPU preflight harness.

This module audits already-staged local evidence.  It deliberately lives in
``tests`` and never downloads data, loads a model, probes a GPU, dispatches a
unit, creates a run root, or advances Pilot execution.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import hashlib
from itertools import combinations
import json
from pathlib import Path, PurePosixPath
import re

from georeliab_mve.v4_counterfactuals import (
    AssetEvidence,
    CounterfactualContractError,
    FOG_STATES,
    ModelIndependentState,
    SCIENTIFIC_MODELS,
    SCIENTIFIC_STATES,
    materialize_dtu_state_identity,
)
from georeliab_mve.v4_science_lock import V4_PROTOCOL_SHA256
from tests import pilot_partition_authorization_audit as freeze
from tests import pilot_round2_harness


RESOURCE_READY = "V4_PILOT_RESOURCE_CANDIDATE_READY"
INPUT_PREFLIGHT_READY = "V4_PILOT_INPUT_RESOURCE_PREFLIGHT_READY"
DEVELOPMENT_EVIDENCE_ONLY = "DEVELOPMENT_EVIDENCE_ONLY"
NO_SCIENTIFIC_RESULT = "NO_SCIENTIFIC_RESULT"

RESOURCE_REQUEST_SCHEMA = "georeliab-v4-pilot-resource-audit-request-1.0"
RESOURCE_CANDIDATE_SCHEMA = "georeliab-v4-pilot-resource-candidate-1.0"
STAGED_INPUT_SCHEMA = "georeliab-v4-pilot-staged-input-inventory-1.0"
INPUT_CLOSURE_SCHEMA = "georeliab-v4-pilot-input-closure-1.0"
INPUT_PREFLIGHT_SCHEMA = "georeliab-v4-pilot-input-resource-preflight-1.0"
SCHEDULE_VIEWS_SCHEMA = "georeliab-v4-pilot-schedule-views-1.0"

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_GIT_OID_RE = re.compile(r"^[0-9a-f]{40}$")
_MANIFEST_RE = re.compile(r"^([0-9a-f]{64})  ([^\r\n]+)$")

_RESOURCE_REQUEST_KEYS = frozenset(
    {
        "schema_version",
        "status",
        "validation_class",
        "scientific_result",
        "resource_root",
        "model_order",
        "models",
        "source_evidence_class",
        "historical_resource_audit_reused",
        "download_allowed",
        "gpu_probe_allowed",
        "pilot_started",
        "automatic_progression_allowed",
    }
)
_MODEL_KEYS = frozenset(
    {
        "model_id",
        "source_root",
        "source_commit",
        "source_tree_sha256",
        "checkpoint_path",
        "checkpoint_sha256",
        "adapter_id",
        "adapter_path",
        "adapter_sha256",
        "config_path",
        "config_sha256",
        "environment_path",
        "environment_sha256",
    }
)
_STAGED_KEYS = frozenset(
    {
        "schema_version",
        "status",
        "validation_class",
        "scientific_result",
        "freeze_root",
        "resource_manifest_path",
        "resource_manifest_sha256",
        "schedule_manifest_path",
        "schedule_manifest_sha256",
        "schedule_identity_sha256",
        "protocol_sha256",
        "input_root",
        "fog_calibration_path",
        "fog_calibration_sha256",
        "scenes",
        "attempt05_predictions_read",
        "gate2_predictions_read",
        "prediction_outputs_reused",
        "receipt_or_ledger_reused",
        "historical_provenance",
        "pilot_started",
        "automatic_progression_allowed",
    }
)
_SCENE_KEYS = frozenset(
    {
        "scene_id",
        "ordered_view_ids",
        "cameras",
        "gt_point_cloud",
        "observability_mask",
        "states",
    }
)
_STATE_KEYS = frozenset({"state_id", "rgb_inputs"})
_ASSET_KEYS = frozenset({"member", "path", "sha256"})
_FOG_ASSET_KEYS = frozenset({"member", "path", "sha256", "fog_generation"})
_FOG_GENERATION_KEYS = frozenset(
    {
        "source_state_id",
        "source_sha256",
        "calibration_sha256",
        "recipe_sha256",
    }
)
_BUNDLE_FILES = frozenset(
    {
        "manifests/pilot-model-independent-states.json",
        "manifests/pilot-unit-records.json",
        "manifests/pilot-input-closure.json",
        "pilot-input-preflight.json",
    }
)


class PilotInputResourceError(ValueError):
    """Raised when a Pilot input/resource contract fails closed."""


def _fail(reason: str) -> None:
    raise PilotInputResourceError(reason)


def _canonical_bytes(value: object) -> bytes:
    try:
        text = json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
        )
    except (TypeError, ValueError) as exc:
        raise PilotInputResourceError(
            f"CANONICAL_JSON_INVALID:{type(exc).__name__}:{exc}"
        ) from exc
    return (text + "\n").encode("utf-8")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_value(value: object) -> str:
    return _sha256_bytes(_canonical_bytes(value))


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
    except OSError as exc:
        raise PilotInputResourceError(
            f"ASSET_DIGEST_UNREADABLE:{path}:{exc}"
        ) from exc
    return digest.hexdigest()


def _read_json(path: Path, reason: str) -> object:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise PilotInputResourceError(
            f"{reason}:{type(exc).__name__}:{exc}"
        ) from exc


def _read_object(path: Path, reason: str) -> Mapping[str, object]:
    value = _read_json(path, reason)
    if not isinstance(value, Mapping):
        _fail(f"{reason}:JSON_OBJECT_REQUIRED")
    return value


def _inside(path: Path, root: Path) -> bool:
    return path == root or root in path.parents


def _real_home(home_root: Path) -> Path:
    raw = Path(home_root)
    if not raw.is_absolute() or not raw.is_dir() or raw.is_symlink():
        _fail("HOME_ROOT_INVALID")
    return raw.resolve(strict=True)


def _home_file(path: Path, home: Path, reason: str) -> Path:
    raw = Path(path)
    if not raw.is_absolute() or not raw.is_file() or raw.is_symlink():
        _fail(f"{reason}_FILE_INVALID")
    resolved = raw.resolve(strict=True)
    if not _inside(resolved, home):
        _fail(f"{reason}_ROOT_INVALID")
    return resolved


def _home_directory(path: Path, home: Path, reason: str) -> Path:
    raw = Path(path)
    if not raw.is_absolute() or not raw.is_dir() or raw.is_symlink():
        _fail(f"{reason}_DIRECTORY_INVALID")
    resolved = raw.resolve(strict=True)
    if not _inside(resolved, home):
        _fail(f"{reason}_ROOT_INVALID")
    if any(member.is_symlink() for member in raw.rglob("*")):
        _fail(f"{reason}_SYMLINK_FORBIDDEN")
    return resolved


def _fresh_output(path: Path, home: Path) -> Path:
    raw = Path(path)
    if not raw.is_absolute():
        _fail("OUTPUT_ROOT_INVALID")
    resolved = raw.resolve(strict=False)
    expected = (home / "georeliab-v4-pilot" / "readiness").resolve(
        strict=False
    )
    if expected not in resolved.parents:
        _fail("OUTPUT_ROOT_SCOPE_INVALID")
    if raw.exists() or resolved.exists():
        _fail("OUTPUT_ROOT_EXISTS")
    return resolved


def _require_sha(value: object, reason: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        _fail(reason)
    return value


def _require_false(payload: Mapping[str, object], field: str, reason: str) -> None:
    if payload.get(field) is not False:
        _fail(reason)


def _tree_rows(root: Path) -> list[dict[str, object]]:
    members = tuple(root.rglob("*"))
    if any(path.is_symlink() for path in members):
        _fail("RESOURCE_SOURCE_SYMLINK_FORBIDDEN")
    rows = []
    for path in sorted(members):
        if path.is_file():
            rows.append(
                {
                    "relative_path": path.relative_to(root).as_posix(),
                    "sha256": _sha256_file(path),
                    "size_bytes": path.stat().st_size,
                }
            )
    if not rows:
        _fail("RESOURCE_SOURCE_TREE_EMPTY")
    return rows


def _validate_resource_model(
    row: object,
    *,
    expected_model: str,
    resource_root: Path,
) -> dict[str, object]:
    if not isinstance(row, Mapping) or set(row) != _MODEL_KEYS:
        _fail("RESOURCE_MODEL_SCHEMA_INVALID")
    if row.get("model_id") != expected_model:
        _fail("RESOURCE_MODEL_ORDER_INVALID")
    commit = row.get("source_commit")
    if not isinstance(commit, str) or _GIT_OID_RE.fullmatch(commit) is None:
        _fail("RESOURCE_SOURCE_COMMIT_INVALID")

    source_text = row.get("source_root")
    if not isinstance(source_text, str):
        _fail("RESOURCE_SOURCE_ROOT_INVALID")
    source = _home_directory(Path(source_text), resource_root, "RESOURCE_SOURCE")
    source_rows = _tree_rows(source)
    source_sha = _sha256_value(source_rows)
    if row.get("source_tree_sha256") != source_sha:
        _fail("RESOURCE_SOURCE_TREE_DIGEST_MISMATCH")

    files: dict[str, Path] = {}
    for label in ("checkpoint", "adapter", "config", "environment"):
        path_text = row.get(f"{label}_path")
        if not isinstance(path_text, str):
            _fail(f"RESOURCE_{label.upper()}_PATH_INVALID")
        path = _home_file(Path(path_text), resource_root, f"RESOURCE_{label.upper()}")
        expected = _require_sha(
            row.get(f"{label}_sha256"),
            f"RESOURCE_{label.upper()}_DIGEST_INVALID",
        )
        if _sha256_file(path) != expected:
            _fail(f"RESOURCE_{label.upper()}_DIGEST_MISMATCH")
        files[label] = path
    adapter_id = row.get("adapter_id")
    if not isinstance(adapter_id, str) or not adapter_id.strip():
        _fail("RESOURCE_ADAPTER_ID_INVALID")

    return {
        "model_id": expected_model,
        "source_root": str(source),
        "source_commit": commit,
        "source_tree_sha256": source_sha,
        "checkpoint_path": str(files["checkpoint"]),
        "checkpoint_sha256": row["checkpoint_sha256"],
        "adapter_id": adapter_id,
        "adapter_path": str(files["adapter"]),
        "adapter_sha256": row["adapter_sha256"],
        "config_path": str(files["config"]),
        "config_sha256": row["config_sha256"],
        "environment_path": str(files["environment"]),
        "environment_sha256": row["environment_sha256"],
    }


def prepare_pilot_resource_candidate(
    *, request_path: Path, output_path: Path, home_root: Path
) -> dict[str, object]:
    """Seal two independently audited, already-local model resources."""

    home = _real_home(home_root)
    request_file = _home_file(request_path, home, "RESOURCE_REQUEST")
    output = _fresh_output(output_path, home)
    request = _read_object(request_file, "RESOURCE_REQUEST_JSON_INVALID")
    if set(request) != _RESOURCE_REQUEST_KEYS:
        _fail("RESOURCE_REQUEST_SCHEMA_INVALID")
    if (
        request.get("schema_version") != RESOURCE_REQUEST_SCHEMA
        or request.get("status") != "PILOT_RESOURCE_AUDIT_REQUESTED"
        or request.get("validation_class") != DEVELOPMENT_EVIDENCE_ONLY
        or request.get("scientific_result") != NO_SCIENTIFIC_RESULT
    ):
        _fail("RESOURCE_REQUEST_HEADER_INVALID")
    if request.get("source_evidence_class") != "FRESH_PILOT_RESOURCE_CANDIDATE":
        _fail("HISTORICAL_RESOURCE_PROMOTION_FORBIDDEN")
    for field in (
        "historical_resource_audit_reused",
        "download_allowed",
        "gpu_probe_allowed",
        "pilot_started",
        "automatic_progression_allowed",
    ):
        _require_false(request, field, f"HISTORICAL_RESOURCE_STATE_INVALID:{field}")
    if tuple(request.get("model_order", ())) != tuple(SCIENTIFIC_MODELS):
        _fail("RESOURCE_MODEL_ORDER_INVALID")

    root_text = request.get("resource_root")
    if not isinstance(root_text, str):
        _fail("RESOURCE_ROOT_INVALID")
    resource_root = _home_directory(Path(root_text), home, "RESOURCE")
    resource_scope = (home / "georeliab-v4-pilot" / "resources").resolve(
        strict=False
    )
    if resource_scope not in resource_root.parents:
        _fail("RESOURCE_ROOT_SCOPE_INVALID")
    rows = request.get("models")
    if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes, bytearray)):
        _fail("RESOURCE_MODEL_BINDINGS_INVALID")
    if len(rows) != 2:
        _fail("RESOURCE_MODEL_BINDINGS_INVALID")
    models = [
        _validate_resource_model(
            row,
            expected_model=model_id,
            resource_root=resource_root,
        )
        for model_id, row in zip(SCIENTIFIC_MODELS, rows, strict=True)
    ]
    payload: dict[str, object] = {
        "schema_version": RESOURCE_CANDIDATE_SCHEMA,
        "status": RESOURCE_READY,
        "validation_class": DEVELOPMENT_EVIDENCE_ONLY,
        "scientific_result": NO_SCIENTIFIC_RESULT,
        "resource_request_path": str(request_file),
        "resource_request_sha256": _sha256_file(request_file),
        "resource_root": str(resource_root),
        "model_order": list(SCIENTIFIC_MODELS),
        "models": models,
        "model_bindings_sha256": _sha256_value(models),
        "source_evidence_class": "FRESH_PILOT_RESOURCE_CANDIDATE",
        "historical_resource_audit_reused": False,
        "download_performed": False,
        "gpu_probe_performed": False,
        "pilot_inputs_materialized": False,
        "pilot_started": False,
        "automatic_progression_allowed": False,
        "next_action": "FREEZE_PILOT_PARTITION_AFTER_EXPLICIT_AUTHORIZATION",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    try:
        with output.open("xb") as handle:
            handle.write(_canonical_bytes(payload))
    except FileExistsError as exc:
        raise PilotInputResourceError(f"OUTPUT_FILE_EXISTS:{output}") from exc
    return dict(payload)


def _asset(
    value: object,
    *,
    input_root: Path,
    reason: str,
    fog: bool = False,
) -> tuple[AssetEvidence, Mapping[str, object]]:
    expected_keys = _FOG_ASSET_KEYS if fog else _ASSET_KEYS
    if not isinstance(value, Mapping) or set(value) != expected_keys:
        _fail(f"{reason}_ASSET_SCHEMA_INVALID")
    member = value.get("member")
    if not isinstance(member, str) or not member:
        _fail(f"{reason}_ASSET_MEMBER_INVALID")
    relative = PurePosixPath(member)
    if relative.is_absolute() or ".." in relative.parts:
        _fail(f"{reason}_ASSET_MEMBER_INVALID")
    path_text = value.get("path")
    if not isinstance(path_text, str):
        _fail(f"{reason}_ASSET_PATH_INVALID")
    path = _home_file(Path(path_text), input_root, f"{reason}_ASSET")
    expected_path = input_root.joinpath(*relative.parts)
    if path != expected_path:
        _fail(f"{reason}_ASSET_PATH_MISMATCH")
    digest = _require_sha(value.get("sha256"), f"{reason}_ASSET_DIGEST_INVALID")
    if _sha256_file(path) != digest:
        _fail(f"{reason}_ASSET_DIGEST_MISMATCH")
    return AssetEvidence(member=member, sha256=digest, source_uri=str(path)), value


def _validate_schedule(
    path: Path,
    *,
    expected_sha: object,
    schedule_identity: str,
    scene_ids: tuple[int, ...],
) -> dict[int, tuple[int, ...]]:
    if _sha256_file(path) != expected_sha:
        _fail("BINDING_SCHEDULE_MANIFEST_DIGEST_MISMATCH")
    value = _read_object(path, "INPUT_SCHEDULE_JSON_INVALID")
    if set(value) != {
        "schema_version",
        "schedule_identity_sha256",
        "scene_ids",
        "ordered_views_by_scene",
    }:
        _fail("INPUT_SCHEDULE_SCHEMA_INVALID")
    if (
        value.get("schema_version") != SCHEDULE_VIEWS_SCHEMA
        or value.get("schedule_identity_sha256") != schedule_identity
        or tuple(value.get("scene_ids", ())) != scene_ids
    ):
        _fail("BINDING_SCHEDULE_IDENTITY_MISMATCH")
    rows = value.get("ordered_views_by_scene")
    if not isinstance(rows, Mapping) or set(rows) != {str(scene) for scene in scene_ids}:
        _fail("INPUT_VIEW_INVENTORY_INVALID")
    result: dict[int, tuple[int, ...]] = {}
    for scene in scene_ids:
        views = rows[str(scene)]
        if not isinstance(views, Sequence) or isinstance(views, (str, bytes)):
            _fail("INPUT_VIEW_INVENTORY_INVALID")
        ordered = tuple(views)
        if (
            len(ordered) != 8
            or len(set(ordered)) != 8
            or any(type(view) is not int or not 1 <= view <= 49 for view in ordered)
        ):
            _fail("INPUT_VIEW_INVENTORY_INVALID")
        result[scene] = ordered
    return result


def _validate_scene(
    value: object,
    *,
    scene_id: int,
    ordered_views: tuple[int, ...],
    input_root: Path,
    calibration_sha: str,
    source_root: Path,
) -> list[dict[str, object]]:
    if not isinstance(value, Mapping) or set(value) != _SCENE_KEYS:
        _fail("INPUT_SCENE_SCHEMA_INVALID")
    if value.get("scene_id") != scene_id:
        _fail("INPUT_SCENE_IDENTITY_MISMATCH")
    if tuple(value.get("ordered_view_ids", ())) != ordered_views:
        _fail("INPUT_VIEW_ORDER_MISMATCH")

    camera_rows = value.get("cameras")
    if not isinstance(camera_rows, Sequence) or len(camera_rows) != 8:
        _fail("INPUT_CAMERA_INVENTORY_INVALID")
    cameras: dict[int, AssetEvidence] = {}
    for view, row in zip(ordered_views, camera_rows, strict=True):
        evidence, _ = _asset(row, input_root=input_root, reason="INPUT_CAMERA")
        cameras[view] = evidence
    gt, _ = _asset(
        value.get("gt_point_cloud"), input_root=input_root, reason="INPUT_GT"
    )
    mask, _ = _asset(
        value.get("observability_mask"),
        input_root=input_root,
        reason="INPUT_MASK",
    )

    state_rows = value.get("states")
    if not isinstance(state_rows, Sequence) or isinstance(
        state_rows, (str, bytes, bytearray)
    ):
        _fail("INPUT_STATE_INVENTORY_INVALID")
    if len(state_rows) != len(SCIENTIFIC_STATES):
        _fail("INPUT_STATE_INVENTORY_INVALID")
    state_map: dict[str, Mapping[str, object]] = {}
    for expected_state, row in zip(SCIENTIFIC_STATES, state_rows, strict=True):
        if not isinstance(row, Mapping) or set(row) != _STATE_KEYS:
            _fail("INPUT_STATE_SCHEMA_INVALID")
        if row.get("state_id") != expected_state:
            _fail("INPUT_STATE_ORDER_MISMATCH")
        state_map[expected_state] = row

    evidence_by_state: dict[str, dict[int, AssetEvidence]] = {}
    metadata_by_state: dict[str, dict[int, Mapping[str, object]]] = {}
    for state_id in SCIENTIFIC_STATES:
        rgb_rows = state_map[state_id].get("rgb_inputs")
        if not isinstance(rgb_rows, Sequence) or len(rgb_rows) != 8:
            _fail("INPUT_VIEW_INVENTORY_INVALID")
        evidence_by_state[state_id] = {}
        metadata_by_state[state_id] = {}
        for view, row in zip(ordered_views, rgb_rows, strict=True):
            evidence, metadata = _asset(
                row,
                input_root=input_root,
                reason="INPUT_RGB",
                fog=state_id in FOG_STATES,
            )
            evidence_by_state[state_id][view] = evidence
            metadata_by_state[state_id][view] = metadata

    l3 = evidence_by_state["L3"]
    states: list[dict[str, object]] = []
    for state_id in SCIENTIFIC_STATES:
        if state_id in FOG_STATES:
            for view in ordered_views:
                generation = metadata_by_state[state_id][view].get("fog_generation")
                if not isinstance(generation, Mapping) or set(generation) != (
                    _FOG_GENERATION_KEYS
                ):
                    _fail("FOG_GENERATION_SCHEMA_INVALID")
                if (
                    generation.get("source_state_id") != "L3"
                    or generation.get("source_sha256") != l3[view].sha256
                    or generation.get("calibration_sha256") != calibration_sha
                ):
                    _fail("FOG_L3_SOURCE_BINDING_MISMATCH")
                _require_sha(
                    generation.get("recipe_sha256"),
                    "FOG_RECIPE_DIGEST_INVALID",
                )
        try:
            state = materialize_dtu_state_identity(
                source_root=source_root,
                scene_id=scene_id,
                state_id=state_id,
                ordered_view_ids=ordered_views,
                rgb_inputs=evidence_by_state[state_id],
                cameras=cameras,
                gt_point_cloud=gt,
                observability_mask=mask,
                clean_source_inputs=l3 if state_id in FOG_STATES else None,
            )
        except CounterfactualContractError as exc:
            raise PilotInputResourceError(f"INPUT_STATE_IDENTITY_INVALID:{exc}") from exc
        states.append(state.to_dict())
    return states


def _storage_accounting(root: Path) -> dict[str, int]:
    paths = tuple(root.rglob("*"))
    if any(path.is_symlink() for path in paths):
        _fail("INPUT_SYMLINK_FORBIDDEN")
    files = tuple(path for path in paths if path.is_file())
    return {
        "logical_bytes": sum(path.stat().st_size for path in files),
        "allocated_bytes": sum(path.stat().st_blocks * 512 for path in files),
        "regular_file_count": len(files),
    }


def _write_json_no_clobber(path: Path, value: object) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("xb") as handle:
            handle.write(_canonical_bytes(value))
    except FileExistsError as exc:
        raise PilotInputResourceError(f"OUTPUT_FILE_EXISTS:{path}") from exc


def _write_manifest(root: Path) -> None:
    lines = [
        f"{_sha256_file(root.joinpath(*PurePosixPath(relative).parts))}  {relative}"
        for relative in sorted(_BUNDLE_FILES)
    ]
    try:
        with (root / "MANIFEST.sha256").open(
            "x", encoding="ascii", newline="\n"
        ) as handle:
            handle.write("\n".join(lines) + "\n")
    except FileExistsError as exc:
        raise PilotInputResourceError("MANIFEST_EXISTS") from exc


def prepare_pilot_input_resource_preflight(
    *,
    freeze_root: Path,
    staged_inventory_path: Path,
    output_root: Path,
    home_root: Path,
    source_root: Path,
) -> dict[str, object]:
    """Validate fresh inputs and seal a non-executing CPU preflight bundle."""

    home = _real_home(home_root)
    output = _fresh_output(output_root, home)
    frozen = _home_directory(freeze_root, home, "BINDING_FREEZE")
    try:
        freeze.verify_pilot_freeze_bundle(frozen)
    except freeze.PilotFreezeAuthorizationError as exc:
        raise PilotInputResourceError(f"BINDING_FREEZE_INVALID:{exc}") from exc
    partition_path = frozen / "manifests/pilot-partition-manifest.json"
    authorization_path = frozen / "manifests/pilot-authorization.json"
    partition = _read_object(partition_path, "BINDING_PARTITION_JSON_INVALID")
    authorization = _read_object(
        authorization_path, "BINDING_AUTHORIZATION_JSON_INVALID"
    )

    staged_path = _home_file(staged_inventory_path, home, "INPUT_INVENTORY")
    staged = _read_object(staged_path, "INPUT_INVENTORY_JSON_INVALID")
    if set(staged) != _STAGED_KEYS:
        _fail("INPUT_INVENTORY_SCHEMA_INVALID")
    if (
        staged.get("schema_version") != STAGED_INPUT_SCHEMA
        or staged.get("status") != "PILOT_FRESH_INPUTS_STAGED"
        or staged.get("validation_class") != DEVELOPMENT_EVIDENCE_ONLY
        or staged.get("scientific_result") != NO_SCIENTIFIC_RESULT
    ):
        _fail("INPUT_INVENTORY_HEADER_INVALID")
    for field in (
        "attempt05_predictions_read",
        "gate2_predictions_read",
        "prediction_outputs_reused",
        "receipt_or_ledger_reused",
        "pilot_started",
        "automatic_progression_allowed",
    ):
        _require_false(staged, field, f"PROVENANCE_FORBIDDEN:{field}")
    if staged.get("historical_provenance") != []:
        _fail("PROVENANCE_HISTORICAL_REUSE_FORBIDDEN")
    if staged.get("freeze_root") != str(frozen):
        _fail("BINDING_FREEZE_ROOT_MISMATCH")

    schedule_identity = _require_sha(
        partition.get("schedule_identity_sha256"),
        "BINDING_SCHEDULE_IDENTITY_INVALID",
    )
    if staged.get("schedule_identity_sha256") != schedule_identity:
        _fail("BINDING_SCHEDULE_IDENTITY_MISMATCH")
    if (
        staged.get("protocol_sha256") != V4_PROTOCOL_SHA256
        or staged.get("protocol_sha256") != partition.get("protocol_sha256")
        or staged.get("protocol_sha256") != authorization.get("protocol_sha256")
    ):
        _fail("BINDING_PROTOCOL_DIGEST_MISMATCH")

    resource_text = staged.get("resource_manifest_path")
    if not isinstance(resource_text, str):
        _fail("BINDING_RESOURCE_PATH_INVALID")
    resource_path = _home_file(Path(resource_text), home, "BINDING_RESOURCE")
    resource_sha = _require_sha(
        staged.get("resource_manifest_sha256"),
        "BINDING_RESOURCE_DIGEST_INVALID",
    )
    if (
        _sha256_file(resource_path) != resource_sha
        or authorization.get("resource_manifest_path") != str(resource_path)
        or authorization.get("resource_manifest_sha256") != resource_sha
    ):
        _fail("BINDING_RESOURCE_DIGEST_MISMATCH")
    resource = _read_object(resource_path, "BINDING_RESOURCE_JSON_INVALID")
    if resource.get("model_bindings_sha256") != partition.get(
        "model_bindings_sha256"
    ):
        _fail("BINDING_MODEL_RESOURCE_DIGEST_MISMATCH")

    schedule_text = staged.get("schedule_manifest_path")
    if not isinstance(schedule_text, str):
        _fail("BINDING_SCHEDULE_PATH_INVALID")
    schedule_path = _home_file(Path(schedule_text), home, "BINDING_SCHEDULE")
    scenes = tuple(partition.get("primary_scene_ids", ()))
    if len(scenes) != 3 or any(type(scene) is not int for scene in scenes):
        _fail("INPUT_PRIMARY_SCENE_INVENTORY_INVALID")
    views_by_scene = _validate_schedule(
        schedule_path,
        expected_sha=staged.get("schedule_manifest_sha256"),
        schedule_identity=schedule_identity,
        scene_ids=scenes,
    )

    input_text = staged.get("input_root")
    if not isinstance(input_text, str):
        _fail("INPUT_ROOT_INVALID")
    input_root = _home_directory(Path(input_text), home, "INPUT")
    input_scope = (home / "georeliab-v4-pilot" / "inputs").resolve(
        strict=False
    )
    if input_scope not in input_root.parents:
        _fail("INPUT_ROOT_NOT_FRESH_PILOT_SCOPE")
    calibration_text = staged.get("fog_calibration_path")
    if not isinstance(calibration_text, str):
        _fail("FOG_CALIBRATION_PATH_INVALID")
    calibration_path = _home_file(
        Path(calibration_text), input_root, "FOG_CALIBRATION"
    )
    calibration_sha = _require_sha(
        staged.get("fog_calibration_sha256"),
        "FOG_CALIBRATION_DIGEST_INVALID",
    )
    if _sha256_file(calibration_path) != calibration_sha:
        _fail("FOG_CALIBRATION_DIGEST_MISMATCH")

    scene_rows = staged.get("scenes")
    if not isinstance(scene_rows, Sequence) or isinstance(
        scene_rows, (str, bytes, bytearray)
    ):
        _fail("INPUT_SCENE_INVENTORY_INVALID")
    if len(scene_rows) != 3:
        _fail("INPUT_SCENE_INVENTORY_INVALID")
    state_records: list[dict[str, object]] = []
    for scene_id, row in zip(scenes, scene_rows, strict=True):
        state_records.extend(
            _validate_scene(
                row,
                scene_id=scene_id,
                ordered_views=views_by_scene[scene_id],
                input_root=input_root,
                calibration_sha=calibration_sha,
                source_root=Path(source_root),
            )
        )
    if len(state_records) != 30:
        _fail("INPUT_STATE_INVENTORY_INVALID")

    by_identity = {
        (row["scene_id"], row["state_id"]): row["state_identity_sha256"]
        for row in state_records
    }
    unit_ids = tuple(partition.get("primary_unit_ids", ()))
    units: list[dict[str, object]] = []
    for unit_id in unit_ids:
        try:
            model_id, scene_text, state_id = str(unit_id).split(":", 2)
            scene_id = int(scene_text)
            state_sha = by_identity[(scene_id, state_id)]
        except (ValueError, KeyError) as exc:
            raise PilotInputResourceError(
                "INPUT_UNIT_IDENTITY_INVALID"
            ) from exc
        ordered = views_by_scene[scene_id]
        units.append(
            {
                "unit_id": unit_id,
                "model_id": model_id,
                "scene_id": scene_id,
                "state_id": state_id,
                "state_identity_sha256": state_sha,
                "ordered_view_ids": list(ordered),
                "pose_pairs": [list(pair) for pair in combinations(ordered, 2)],
                "prediction_provenance": [],
                "receipt_provenance": [],
                "ledger_provenance": [],
            }
        )
    try:
        pilot_round2_harness.validate_pilot_records(
            units,
            expected_unit_ids=unit_ids,
            expected_views_by_scene=views_by_scene,
        )
    except pilot_round2_harness.PilotRound2ContractError as exc:
        raise PilotInputResourceError(f"INPUT_UNIT_INVENTORY_INVALID:{exc}") from exc

    storage = _storage_accounting(input_root)
    max_storage = authorization.get("max_storage_bytes")
    if type(max_storage) is not int or storage["allocated_bytes"] > max_storage:
        _fail("INPUT_STORAGE_BUDGET_EXCEEDED")
    run_root = Path(str(authorization.get("run_root", "")))
    if not run_root.is_absolute() or run_root.exists():
        _fail("BINDING_RUN_ROOT_NOT_FRESH")

    states_path = output / "manifests/pilot-model-independent-states.json"
    units_path = output / "manifests/pilot-unit-records.json"
    closure_path = output / "manifests/pilot-input-closure.json"
    preflight_path = output / "pilot-input-preflight.json"
    states_bytes = _canonical_bytes(state_records)
    units_bytes = _canonical_bytes(units)
    closure: dict[str, object] = {
        "schema_version": INPUT_CLOSURE_SCHEMA,
        "status": "V4_PILOT_FRESH_INPUT_CLOSURE_SEALED",
        "validation_class": DEVELOPMENT_EVIDENCE_ONLY,
        "scientific_result": NO_SCIENTIFIC_RESULT,
        "freeze_root": str(frozen),
        "partition_manifest_path": str(partition_path),
        "partition_manifest_sha256": _sha256_file(partition_path),
        "authorization_path": str(authorization_path),
        "authorization_sha256": _sha256_file(authorization_path),
        "resource_manifest_path": str(resource_path),
        "resource_manifest_sha256": resource_sha,
        "schedule_manifest_path": str(schedule_path),
        "schedule_manifest_sha256": _sha256_file(schedule_path),
        "schedule_identity_sha256": schedule_identity,
        "protocol_sha256": V4_PROTOCOL_SHA256,
        "input_root": str(input_root),
        "scene_ids": list(scenes),
        "state_ids": list(SCIENTIFIC_STATES),
        "model_order": list(SCIENTIFIC_MODELS),
        "state_inventory_sha256": _sha256_bytes(states_bytes),
        "unit_inventory_sha256": _sha256_bytes(units_bytes),
        "primary_scene_count": 3,
        "state_identity_count": 30,
        "execution_unit_count": 60,
        "storage_accounting": storage,
        "max_storage_bytes": max_storage,
        "attempt05_predictions_read": False,
        "gate2_predictions_read": False,
        "prediction_outputs_reused": False,
        "receipt_or_ledger_reused": False,
        "pilot_inputs_materialized": True,
        "pilot_started": False,
        "gpu_preflight_started": False,
        "automatic_progression_allowed": False,
    }
    closure_bytes = _canonical_bytes(closure)
    preflight: dict[str, object] = {
        "schema_version": INPUT_PREFLIGHT_SCHEMA,
        "status": INPUT_PREFLIGHT_READY,
        "validation_class": DEVELOPMENT_EVIDENCE_ONLY,
        "scientific_result": NO_SCIENTIFIC_RESULT,
        "input_closure_path": str(closure_path),
        "input_closure_sha256": _sha256_bytes(closure_bytes),
        "primary_scene_count": 3,
        "state_identity_count": 30,
        "execution_unit_count": 60,
        "pilot_execution_authorized": True,
        "pilot_partition_frozen": True,
        "pilot_inputs_materialized": True,
        "gpu_preflight_started": False,
        "pilot_started": False,
        "confirmation_started": False,
        "automatic_progression_allowed": False,
        "run_root": str(run_root),
        "next_action": "RUN_SEPARATE_GPU_PREFLIGHT_AFTER_USER_REVIEW",
    }

    output.mkdir(parents=True, exist_ok=False)
    _write_json_no_clobber(states_path, state_records)
    _write_json_no_clobber(units_path, units)
    _write_json_no_clobber(closure_path, closure)
    _write_json_no_clobber(preflight_path, preflight)
    _write_manifest(output)
    return verify_pilot_input_resource_bundle(output)


def _manifest_entries(path: Path) -> dict[str, str]:
    try:
        lines = path.read_text(encoding="ascii").splitlines()
    except (OSError, UnicodeError) as exc:
        raise PilotInputResourceError(f"MANIFEST_UNREADABLE:{exc}") from exc
    entries: dict[str, str] = {}
    for line in lines:
        match = _MANIFEST_RE.fullmatch(line)
        if match is None:
            _fail("MANIFEST_ROW_INVALID")
        digest, relative_text = match.groups()
        relative = PurePosixPath(relative_text)
        if relative.is_absolute() or ".." in relative.parts or relative_text in entries:
            _fail("MANIFEST_PATH_INVALID")
        entries[relative_text] = digest
    if set(entries) != _BUNDLE_FILES:
        _fail("MANIFEST_FILE_SET_INVALID")
    return entries


def verify_pilot_input_resource_bundle(root: Path) -> dict[str, object]:
    """Verify exact file coverage, digests, and sealed closure semantics."""

    raw = Path(root)
    if not raw.is_absolute() or not raw.is_dir() or raw.is_symlink():
        _fail("MANIFEST_BUNDLE_ROOT_INVALID")
    resolved = raw.resolve(strict=True)
    members = tuple(resolved.rglob("*"))
    if any(path.is_symlink() for path in members):
        _fail("MANIFEST_SYMLINK_FORBIDDEN")
    actual = {
        path.relative_to(resolved).as_posix()
        for path in members
        if path.is_file() and path.name != "MANIFEST.sha256"
    }
    if actual != _BUNDLE_FILES:
        _fail("MANIFEST_UNLISTED_OR_MISSING_FILE")
    manifest = resolved / "MANIFEST.sha256"
    if not manifest.is_file() or manifest.is_symlink():
        _fail("MANIFEST_MISSING")
    for relative, expected in _manifest_entries(manifest).items():
        path = resolved.joinpath(*PurePosixPath(relative).parts)
        if _sha256_file(path) != expected:
            _fail(f"MANIFEST_DIGEST_MISMATCH:{relative}")

    states_path = resolved / "manifests/pilot-model-independent-states.json"
    units_path = resolved / "manifests/pilot-unit-records.json"
    closure_path = resolved / "manifests/pilot-input-closure.json"
    preflight_path = resolved / "pilot-input-preflight.json"
    states = _read_json(states_path, "MANIFEST_STATES_JSON_INVALID")
    units = _read_json(units_path, "MANIFEST_UNITS_JSON_INVALID")
    if not isinstance(states, list) or len(states) != 30:
        _fail("MANIFEST_STATE_INVENTORY_INVALID")
    if not isinstance(units, list) or len(units) != 60:
        _fail("MANIFEST_UNIT_INVENTORY_INVALID")
    try:
        parsed_states = [ModelIndependentState.from_dict(row) for row in states]
    except CounterfactualContractError as exc:
        raise PilotInputResourceError(f"MANIFEST_STATE_IDENTITY_INVALID:{exc}") from exc
    if len({row.state_identity_sha256 for row in parsed_states}) != 30:
        _fail("MANIFEST_STATE_IDENTITY_DUPLICATE")

    closure = _read_object(closure_path, "MANIFEST_CLOSURE_JSON_INVALID")
    if (
        closure.get("schema_version") != INPUT_CLOSURE_SCHEMA
        or closure.get("status") != "V4_PILOT_FRESH_INPUT_CLOSURE_SEALED"
        or closure.get("validation_class") != DEVELOPMENT_EVIDENCE_ONLY
        or closure.get("scientific_result") != NO_SCIENTIFIC_RESULT
        or closure.get("state_inventory_sha256") != _sha256_file(states_path)
        or closure.get("unit_inventory_sha256") != _sha256_file(units_path)
        or closure.get("primary_scene_count") != 3
        or closure.get("state_identity_count") != 30
        or closure.get("execution_unit_count") != 60
    ):
        _fail("MANIFEST_CLOSURE_BINDING_MISMATCH")
    preflight = _read_object(preflight_path, "MANIFEST_PREFLIGHT_JSON_INVALID")
    required = {
        "schema_version": INPUT_PREFLIGHT_SCHEMA,
        "status": INPUT_PREFLIGHT_READY,
        "validation_class": DEVELOPMENT_EVIDENCE_ONLY,
        "scientific_result": NO_SCIENTIFIC_RESULT,
        "input_closure_path": str(closure_path),
        "input_closure_sha256": _sha256_file(closure_path),
        "primary_scene_count": 3,
        "state_identity_count": 30,
        "execution_unit_count": 60,
        "pilot_execution_authorized": True,
        "pilot_partition_frozen": True,
        "pilot_inputs_materialized": True,
        "gpu_preflight_started": False,
        "pilot_started": False,
        "confirmation_started": False,
        "automatic_progression_allowed": False,
        "next_action": "RUN_SEPARATE_GPU_PREFLIGHT_AFTER_USER_REVIEW",
    }
    for field, expected in required.items():
        if preflight.get(field) != expected:
            _fail(f"MANIFEST_PREFLIGHT_BINDING_MISMATCH:{field}")
    run_root = Path(str(preflight.get("run_root", "")))
    if not run_root.is_absolute() or run_root.exists():
        _fail("MANIFEST_RUN_ROOT_NOT_FRESH")
    return dict(preflight)
