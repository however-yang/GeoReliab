"""CPU-only input closure for GeoReliab v4 MVE Attempt-05.

This module prepares the immutable input manifests consumed by the Attempt-05
execution controller.  It deliberately performs no GPU probing, model loading,
forward pass, metric computation, or scientific finalization.  Missing or
ambiguous source files fail closed before execution can start.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np

from .prepared_inputs import (
    DTU_IMAGE_SIZE,
    decode_srgb_png,
    derive_dtu_depth,
    parse_dtu_binary_ply,
    parse_dtu_projection,
)
from .preparation import CorruptionCalibration, select_fps_views, deterministic_png
from .preparation_round2 import fog_render
from .v4_attempt04_authorization import MANIFEST_FILE_SHA256, SCHEDULE_FILE_SHA256
from .v4_attempt05_execution import (
    Attempt05AuthorizedContext,
    build_attempt05_scientific_schedule,
)
from .v4_counterfactuals import (
    AssetEvidence,
    DTU_LIGHTING_SEMANTIC_TO_PHYSICAL_TOKEN,
    FOG_STATES,
    LIGHTING_STATES,
    SCIENTIFIC_MODELS,
    TEST_SCENE_IDS,
    V4SplitAssignment,
    VerifiedSceneInventory,
    construct_v4_splits,
    materialize_dtu_state_identity,
    validate_scientific_schedule,
    validate_v4_split_assignment,
)
from .v4_execution import V4ExecutionError


ATTEMPT05_INPUT_CLOSURE_SCHEMA = "georeliab-v4-attempt-05-input-closure-1.0"
ATTEMPT05_CALIBRATION_SCHEDULE_SCHEMA = (
    "georeliab-v4-attempt-05-calibration-schedule-1.0"
)
ATTEMPT05_FOG_BINDING_SCHEMA = "georeliab-v4-attempt-05-fog-binding-1.0"


class Attempt05InputClosureError(V4ExecutionError):
    """Raised when Attempt-05 input closure must fail before GPU use."""


@dataclass(frozen=True, slots=True)
class _FogMaterializationResult:
    records: tuple[dict[str, object], ...]
    created_paths: tuple[Path, ...]


@dataclass(frozen=True, slots=True)
class Attempt05InputClosure:
    status: str
    split_assignment: V4SplitAssignment
    model_independent_states: tuple[Any, ...]
    scientific_schedule: Any
    calibration_schedule: tuple[dict[str, object], ...]
    manifest_path: Path
    manifest_sha256: str
    state_inventory_path: Path
    state_inventory_sha256: str
    scientific_schedule_path: Path
    scientific_schedule_sha256: str
    split_assignment_path: Path
    split_assignment_sha256: str
    calibration_schedule_path: Path
    calibration_schedule_sha256: str
    runtime_binding_path: Path
    runtime_binding_sha256: str


def _canonical_json_bytes(value: object) -> bytes:
    return (
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("ascii")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_json(value: object) -> str:
    return _sha256_bytes(_canonical_json_bytes(value))


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _allocated_bytes(path: Path) -> int:
    stat = path.stat()
    blocks = getattr(stat, "st_blocks", None)
    return int(blocks) * 512 if blocks is not None else int(stat.st_size)


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise Attempt05InputClosureError("V4_MVE_BLOCKED_INPUT_JSON_UNREADABLE") from exc
    if not isinstance(payload, dict):
        raise Attempt05InputClosureError("V4_MVE_BLOCKED_INPUT_JSON_SCHEMA")
    return payload


def _write_atomic_json(path: Path, payload: Mapping[str, Any]) -> str:
    data = _canonical_json_bytes(dict(payload))
    digest = _sha256_bytes(data)
    partial = path.with_name(path.name + ".partial")
    if path.exists() or partial.exists():
        raise Attempt05InputClosureError("V4_MVE_BLOCKED_INPUT_IMMUTABLE_COLLISION")
    path.parent.mkdir(parents=True, exist_ok=True)
    with partial.open("xb") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())
    try:
        parsed = json.loads(partial.read_text(encoding="utf-8"))
        if not isinstance(parsed, dict):
            raise Attempt05InputClosureError("V4_MVE_BLOCKED_INPUT_JSON_SCHEMA")
        partial.replace(path)
    except Exception:
        try:
            partial.unlink(missing_ok=True)
        finally:
            raise
    return digest


def _asset_evidence(path: Path, *, member: str) -> AssetEvidence:
    try:
        resolved = path.resolve(strict=True)
        if not resolved.is_file():
            raise OSError("not a regular file")
    except OSError as exc:
        raise Attempt05InputClosureError("V4_MVE_BLOCKED_INPUT_MEMBER_MISSING") from exc
    return AssetEvidence(member=member, sha256=_sha256_file(resolved), source_uri=str(resolved))


def _sample_root(root: Path) -> Path:
    return root / "SampleSet" / "SampleSet" / "MVS Data"


def _points_root(root: Path) -> Path:
    return root / "Points" / "Points"


def _rectified_root(root: Path) -> Path:
    return root / "Rectified"


def _gt_path(root: Path, scene_id: int) -> Path:
    return _points_root(root) / "stl" / f"stl{scene_id:03d}_total.ply"


def _mask_path(root: Path, scene_id: int) -> Path:
    return _sample_root(root) / "ObsMask" / f"ObsMask{scene_id}_10.mat"


def _camera_path(root: Path, view_id: int) -> Path:
    return _sample_root(root) / "Calibration" / "cal18" / f"pos_{view_id:03d}.txt"


def _rgb_path(root: Path, scene_id: int, view_id: int, state_id: str) -> Path:
    if state_id not in LIGHTING_STATES:
        raise Attempt05InputClosureError("V4_MVE_BLOCKED_INPUT_STATE_SCOPE")
    token = DTU_LIGHTING_SEMANTIC_TO_PHYSICAL_TOKEN[state_id]
    return _rectified_root(root) / f"scan{scene_id}" / f"rect_{view_id:03d}_{token}_r5000.png"


def _resource_receipt_path(context: Attempt05AuthorizedContext) -> Path:
    raw = context.authorization.get("resource_receipt_path")
    if not isinstance(raw, str) or not raw:
        raise Attempt05InputClosureError("V4_MVE_BLOCKED_INPUT_AUTHORIZATION_TAMPER")
    path = Path(raw)
    if not path.is_file():
        raise Attempt05InputClosureError("V4_MVE_BLOCKED_INPUT_RESOURCE_RECEIPT_MISSING")
    expected = context.authorization.get("resource_receipt_sha256")
    if isinstance(expected, str) and expected and _sha256_file(path) != expected:
        raise Attempt05InputClosureError("V4_MVE_BLOCKED_INPUT_RESOURCE_RECEIPT_TAMPER")
    return path


def _closure_schedule_path(context: Attempt05AuthorizedContext) -> Path:
    receipt = _read_json(_resource_receipt_path(context))
    closure_files = receipt.get("closure_files")
    if not isinstance(closure_files, Mapping):
        raise Attempt05InputClosureError("V4_MVE_BLOCKED_INPUT_RESOURCE_RECEIPT_TAMPER")
    schedule = closure_files.get("schedule")
    if not isinstance(schedule, Mapping):
        raise Attempt05InputClosureError("V4_MVE_BLOCKED_INPUT_RESOURCE_RECEIPT_TAMPER")
    path_raw = schedule.get("path")
    if not isinstance(path_raw, str):
        raise Attempt05InputClosureError("V4_MVE_BLOCKED_INPUT_RESOURCE_RECEIPT_TAMPER")
    path = Path(path_raw)
    if not path.is_file() or _sha256_file(path) != SCHEDULE_FILE_SHA256:
        raise Attempt05InputClosureError("V4_MVE_BLOCKED_INPUT_SCHEDULE_HASH_MISMATCH")
    if schedule.get("sha256") != SCHEDULE_FILE_SHA256:
        raise Attempt05InputClosureError("V4_MVE_BLOCKED_INPUT_SCHEDULE_HASH_MISMATCH")
    return path


def _extract_scene_and_views(unit: Mapping[str, Any]) -> tuple[int, tuple[int, ...]]:
    scene_value = unit.get("scene_id", unit.get("scene"))
    if scene_value is None:
        payload = unit.get("payload")
        if isinstance(payload, Mapping):
            scene_value = payload.get("scene_id", payload.get("scene"))
    ordered = unit.get("ordered_views", unit.get("ordered_view_ids"))
    if ordered is None:
        payload = unit.get("payload")
        if isinstance(payload, Mapping):
            ordered = payload.get("ordered_views", payload.get("ordered_view_ids"))
    if ordered is None:
        state_identity = unit.get("state_identity")
        if isinstance(state_identity, Mapping):
            ordered = state_identity.get("ordered_views", state_identity.get("ordered_view_ids"))
            scene_value = scene_value if scene_value is not None else state_identity.get("scene_id")
    if scene_value is None or not isinstance(ordered, Sequence) or isinstance(ordered, (str, bytes)):
        raise Attempt05InputClosureError("V4_MVE_BLOCKED_INPUT_SCHEDULE_SCHEMA")
    views = tuple(int(view) for view in ordered)
    if len(views) != 8 or len(set(views)) != 8:
        raise Attempt05InputClosureError("V4_MVE_BLOCKED_INPUT_SCHEDULE_VIEW_ORDER")
    return int(scene_value), views


def _view_order_from_resource_schedule(schedule_path: Path) -> dict[int, tuple[int, ...]]:
    if _sha256_file(schedule_path) != SCHEDULE_FILE_SHA256:
        raise Attempt05InputClosureError("V4_MVE_BLOCKED_INPUT_SCHEDULE_HASH_MISMATCH")
    payload = _read_json(schedule_path)
    units = payload.get("units", payload.get("execution_units"))
    if not isinstance(units, list) or len(units) != 400:
        raise Attempt05InputClosureError("V4_MVE_BLOCKED_INPUT_SCHEDULE_CARDINALITY")
    by_scene: dict[int, tuple[int, ...]] = {}
    for unit in units:
        if not isinstance(unit, Mapping):
            raise Attempt05InputClosureError("V4_MVE_BLOCKED_INPUT_SCHEDULE_SCHEMA")
        scene, views = _extract_scene_and_views(unit)
        existing = by_scene.setdefault(scene, views)
        if existing != views:
            raise Attempt05InputClosureError("V4_MVE_BLOCKED_INPUT_SCHEDULE_VIEW_ORDER")
    if set(by_scene) != set(TEST_SCENE_IDS):
        raise Attempt05InputClosureError("V4_MVE_BLOCKED_INPUT_SCHEDULE_SCENE_SET")
    return by_scene


def _view_order_from_closure_manifest(manifest_path: Path) -> dict[int, tuple[int, ...]]:
    if _sha256_file(manifest_path) != MANIFEST_FILE_SHA256:
        raise Attempt05InputClosureError("V4_MVE_BLOCKED_INPUT_CLOSURE_HASH_MISMATCH")
    by_scene: dict[int, list[int]] = {}
    seen_scene_view: set[tuple[int, int]] = set()
    lines = manifest_path.read_text(encoding="utf-8").splitlines()
    if len(lines) != 960:
        raise Attempt05InputClosureError("V4_MVE_BLOCKED_INPUT_CLOSURE_CARDINALITY")
    for line in lines:
        row = json.loads(line)
        if not isinstance(row, dict):
            raise Attempt05InputClosureError("V4_MVE_BLOCKED_INPUT_CLOSURE_SCHEMA")
        scene = int(row.get("scene_id"))
        view = int(row.get("view_id"))
        illumination = str(row.get("illumination_id"))
        if illumination == "L3" or illumination not in {"L1", "L2", "L4", "L5", "L6", "L7"}:
            raise Attempt05InputClosureError("V4_MVE_BLOCKED_INPUT_CLOSURE_ILLUMINATION")
        key = (scene, view)
        if key not in seen_scene_view:
            by_scene.setdefault(scene, []).append(view)
            seen_scene_view.add(key)
    result: dict[int, tuple[int, ...]] = {}
    for scene, views in by_scene.items():
        if len(views) != 8 or len(set(views)) != 8:
            raise Attempt05InputClosureError("V4_MVE_BLOCKED_INPUT_VIEW_CLOSURE")
        result[scene] = tuple(views)
    return result




def _png_dimensions(path: Path) -> tuple[int, int]:
    try:
        with path.open("rb") as handle:
            header = handle.read(24)
    except OSError as exc:
        raise Attempt05InputClosureError("V4_MVE_BLOCKED_INPUT_PNG_UNREADABLE") from exc
    if len(header) < 24 or header[:8] != b"\x89PNG\r\n\x1a\n" or header[12:16] != b"IHDR":
        raise Attempt05InputClosureError("V4_MVE_BLOCKED_INPUT_PNG_SCHEMA")
    width = int.from_bytes(header[16:20], "big")
    height = int.from_bytes(header[20:24], "big")
    if width <= 0 or height <= 0:
        raise Attempt05InputClosureError("V4_MVE_BLOCKED_INPUT_PNG_DIMENSIONS")
    return width, height


def _view_evidence(path: Path, *, member: str, source_sha256: str | None = None) -> dict[str, object]:
    asset = _asset_evidence(path, member=member)
    width, height = _png_dimensions(path)
    return {
        "view_id": int(Path(member).name.split("_")[1]),
        "member": member,
        "path": asset.source_uri,
        "sha256": asset.sha256,
        "source_sha256": source_sha256,
        "width": width,
        "height": height,
        "channels": 3,
        "dtype": "uint8",
    }


def _runtime_binding_for_scene_state(
    *,
    root: Path,
    fog_root: Path,
    scene_id: int,
    state_id: str,
    ordered_views: Sequence[int],
) -> dict[str, object]:
    gt = _asset_evidence(
        _gt_path(root, scene_id),
        member=f"Points/Points/stl/stl{scene_id:03d}_total.ply",
    )
    mask = _asset_evidence(
        _mask_path(root, scene_id),
        member=f"SampleSet/SampleSet/MVS Data/ObsMask/ObsMask{scene_id}_10.mat",
    )
    cameras = []
    for view_id in ordered_views:
        cam_member = f"SampleSet/SampleSet/MVS Data/Calibration/cal18/pos_{view_id:03d}.txt"
        cam = _asset_evidence(_camera_path(root, view_id), member=cam_member)
        cameras.append({"view_id": view_id, "member": cam_member, "path": cam.source_uri, "sha256": cam.sha256})
    l3_by_view: dict[int, AssetEvidence] = {}
    for view_id in ordered_views:
        member = _official_rgb_member(scene_id, view_id, "L3")
        l3_by_view[view_id] = _asset_evidence(_rgb_path(root, scene_id, view_id, "L3"), member=member)
    views = []
    for view_id in ordered_views:
        if state_id in LIGHTING_STATES:
            member = _official_rgb_member(scene_id, view_id, state_id)
            path = _rgb_path(root, scene_id, view_id, state_id)
            source_sha = None
        elif state_id in FOG_STATES:
            member = _fog_member(scene_id, view_id, state_id)
            path = fog_root / f"scan{scene_id}" / state_id / f"rect_{view_id:03d}_3_r5000.png"
            source_sha = l3_by_view[view_id].sha256
        else:
            raise Attempt05InputClosureError("V4_MVE_BLOCKED_INPUT_STATE_SCOPE")
        row = _view_evidence(path, member=member, source_sha256=source_sha)
        if int(row["view_id"]) != view_id:
            raise Attempt05InputClosureError("V4_MVE_BLOCKED_INPUT_VIEW_MEMBER_MISMATCH")
        views.append(row)
    state = _state_for_scene(
        root=root,
        source_root=None,
        scene_id=scene_id,
        state_id=state_id,
        ordered_views=tuple(ordered_views),
        fog_root=fog_root,
    )
    return {
        "state_identity_sha256": state.state_identity_sha256,
        "scene_id": scene_id,
        "state_id": state_id,
        "ordered_view_ids": list(ordered_views),
        "views": views,
        "cameras": cameras,
        "gt_point_cloud": {"path": gt.source_uri, "sha256": gt.sha256},
        "observability_mask": {"path": mask.source_uri, "sha256": mask.sha256},
    }




def _camera_center_from_projection(path: Path) -> np.ndarray:
    matrix = parse_dtu_projection(path)
    _, _, vh = np.linalg.svd(matrix)
    homogeneous = vh[-1]
    if abs(float(homogeneous[-1])) < 1e-12:
        raise Attempt05InputClosureError("V4_MVE_BLOCKED_INPUT_CAMERA_CENTER_INVALID")
    center = homogeneous[:3] / homogeneous[-1]
    if center.shape != (3,) or not np.isfinite(center).all():
        raise Attempt05InputClosureError("V4_MVE_BLOCKED_INPUT_CAMERA_CENTER_INVALID")
    return np.asarray(center, dtype=np.float64)


def _fps_views_for_scene(root: Path, scene_id: int) -> tuple[int, ...]:
    centers: dict[int, np.ndarray] = {}
    for view_id in range(1, 50):
        camera_path = _camera_path(root, view_id)
        centers[view_id] = _camera_center_from_projection(camera_path)
    try:
        selected = select_fps_views(centers)
    except Exception as exc:
        raise Attempt05InputClosureError("V4_MVE_BLOCKED_INPUT_FPS_VIEW_DERIVATION") from exc
    if len(selected) != 8:
        raise Attempt05InputClosureError("V4_MVE_BLOCKED_INPUT_FPS_VIEW_DERIVATION")
    return selected


def _scene_has_required_l3_assets(
    root: Path,
    scene_id: int,
    ordered_views: Sequence[int],
) -> bool:
    if len(tuple(ordered_views)) != 8:
        return False
    if not _gt_path(root, scene_id).is_file():
        return False
    if not _mask_path(root, scene_id).is_file():
        return False
    for view_id in ordered_views:
        if not _camera_path(root, int(view_id)).is_file():
            return False
        if not _rgb_path(root, scene_id, int(view_id), "L3").is_file():
            return False
    return True


def _verified_complete_scenes(
    root: Path,
    closure_scenes: Iterable[int],
    authoritative_view_order: Mapping[int, Sequence[int]],
) -> tuple[int, ...]:
    candidates = set(int(scene) for scene in closure_scenes) | set(TEST_SCENE_IDS)
    candidates.update(range(1, 78))
    candidates.update(range(82, 129))
    candidates.discard(4)
    candidates.discard(15)
    complete: list[int] = []
    for scene in sorted(candidates):
        ordered = authoritative_view_order.get(scene)
        if ordered is None:
            try:
                ordered = _fps_views_for_scene(root, scene)
            except Attempt05InputClosureError:
                continue
        if _scene_has_required_l3_assets(root, scene, tuple(int(v) for v in ordered)):
            complete.append(scene)
    if len(complete) < 50:
        raise Attempt05InputClosureError("V4_MVE_BLOCKED_INPUT_COMPLETE_SCENE_INVENTORY")
    return tuple(complete)


def _load_corruption_calibration(path: Path) -> tuple[CorruptionCalibration, str]:
    if not path.is_file():
        raise Attempt05InputClosureError("V4_MVE_BLOCKED_INPUT_CORRUPTION_CALIBRATION_MISSING")
    payload = _read_json(path)
    try:
        calibration = CorruptionCalibration(
            d_ref=float(payload["d_ref"]),
            airlight=tuple(float(value) for value in payload["airlight"]),  # type: ignore[arg-type]
            fog_betas=tuple(float(value) for value in payload["fog_betas"]),  # type: ignore[arg-type]
            defocus_scales=tuple(float(value) for value in payload["defocus_scales"]),  # type: ignore[arg-type]
            implementation_version=str(payload.get("implementation_version", "")),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise Attempt05InputClosureError("V4_MVE_BLOCKED_INPUT_CORRUPTION_CALIBRATION_SCHEMA") from exc
    if (
        len(calibration.airlight) != 3
        or len(calibration.fog_betas) != 3
        or len(calibration.defocus_scales) != 3
        or calibration.d_ref <= 0
        or not all(np.isfinite(v) for v in (*calibration.airlight, *calibration.fog_betas, *calibration.defocus_scales, calibration.d_ref))
    ):
        raise Attempt05InputClosureError("V4_MVE_BLOCKED_INPUT_CORRUPTION_CALIBRATION_SCHEMA")
    return calibration, _sha256_file(path)


def _write_atomic_bytes(path: Path, data: bytes) -> str:
    partial = path.with_name(path.name + ".partial")
    if partial.exists():
        raise Attempt05InputClosureError("V4_MVE_BLOCKED_INPUT_PARTIAL_PRESENT")
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        try:
            existing = path.read_bytes()
        except OSError as exc:
            raise Attempt05InputClosureError("V4_MVE_BLOCKED_INPUT_MEMBER_MISSING") from exc
        if existing != data:
            raise Attempt05InputClosureError("V4_MVE_BLOCKED_INPUT_FOG_DIGEST_MISMATCH")
        return "reused"
    with partial.open("xb") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())
    try:
        partial.replace(path)
    except Exception:
        partial.unlink(missing_ok=True)
        raise
    return "created"


def _cleanup_created_fog(paths: Sequence[Path], fog_root: Path) -> None:
    try:
        resolved_root = fog_root.resolve(strict=False)
    except OSError as exc:
        raise Attempt05InputClosureError("V4_MVE_BLOCKED_INPUT_FOG_CLEANUP_SCOPE") from exc
    for path in reversed(tuple(paths)):
        resolved_path = path.resolve(strict=False)
        try:
            resolved_path.relative_to(resolved_root)
        except ValueError as exc:
            raise Attempt05InputClosureError("V4_MVE_BLOCKED_INPUT_FOG_CLEANUP_SCOPE") from exc
        try:
            if path.exists():
                path.unlink()
        except OSError as exc:
            raise Attempt05InputClosureError("V4_MVE_BLOCKED_INPUT_FOG_CLEANUP_FAILED") from exc


def _fog_recipe_sha256(
    *,
    calibration: CorruptionCalibration,
    corruption_calibration_sha256: str,
    severity: int,
) -> str:
    beta = calibration.fog_betas[severity - 1]
    return _sha256_json(
        {
            "schema_version": ATTEMPT05_FOG_BINDING_SCHEMA,
            "renderer": "Koschmieder",
            "state_id": f"fog-s{severity}",
            "severity": severity,
            "beta": beta,
            "d_ref": calibration.d_ref,
            "airlight": list(calibration.airlight),
            "implementation_version": calibration.implementation_version,
            "corruption_calibration_sha256": corruption_calibration_sha256,
        }
    )


def _materialize_fog_members(
    root: Path,
    fog_root: Path,
    view_order: Mapping[int, Sequence[int]],
    calibration: CorruptionCalibration,
    *,
    corruption_calibration_sha256: str,
) -> _FogMaterializationResult:
    if len(corruption_calibration_sha256) != 64:
        raise Attempt05InputClosureError("V4_MVE_BLOCKED_INPUT_CORRUPTION_CALIBRATION_SCHEMA")
    ply_cache: dict[int, np.ndarray] = {}
    gt_sha_cache: dict[int, str] = {}
    created_paths: list[Path] = []
    records: list[dict[str, object]] = []
    try:
        for scene_id in TEST_SCENE_IDS:
            ordered = view_order.get(scene_id)
            if ordered is None:
                raise Attempt05InputClosureError("V4_MVE_BLOCKED_INPUT_TEST_VIEW_CLOSURE")
            if len(tuple(ordered)) != 8:
                raise Attempt05InputClosureError("V4_MVE_BLOCKED_INPUT_TEST_VIEW_CLOSURE")
            gt_path = _gt_path(root, scene_id)
            gt_sha_cache[scene_id] = _sha256_file(gt_path)
            for state_id in FOG_STATES:
                severity = int(str(state_id)[-1])
                recipe_sha256 = _fog_recipe_sha256(
                    calibration=calibration,
                    corruption_calibration_sha256=corruption_calibration_sha256,
                    severity=severity,
                )
                for view_id in ordered:
                    view = int(view_id)
                    fog_path = fog_root / f"scan{scene_id}" / state_id / f"rect_{view:03d}_3_r5000.png"
                    if fog_path.with_name(fog_path.name + ".partial").exists():
                        raise Attempt05InputClosureError("V4_MVE_BLOCKED_INPUT_PARTIAL_PRESENT")
                    l3_path = _rgb_path(root, scene_id, view, "L3")
                    l3_sha = _sha256_file(l3_path)
                    camera_path = _camera_path(root, view)
                    camera_sha = _sha256_file(camera_path)
                    image, _ = decode_srgb_png(l3_path, expected_size=DTU_IMAGE_SIZE)
                    points = ply_cache.get(scene_id)
                    if points is None:
                        points = parse_dtu_binary_ply(gt_path)
                        ply_cache[scene_id] = points
                    depth, _ = derive_dtu_depth(points, camera_path, image_size=DTU_IMAGE_SIZE)
                    rendered, _ = fog_render(
                        image,
                        depth,
                        calibration,
                        severity=severity,
                        gt_digest=gt_sha_cache[scene_id],
                        raw_source_sha256=l3_sha,
                    )
                    png_bytes = deterministic_png(rendered)
                    materialization_status = _write_atomic_bytes(fog_path, png_bytes)
                    if materialization_status == "created":
                        created_paths.append(fog_path)
                    record_payload = {
                        "schema_version": ATTEMPT05_FOG_BINDING_SCHEMA,
                        "scene_id": scene_id,
                        "state_id": state_id,
                        "severity": severity,
                        "view_id": view,
                        "fog_member": _fog_member(scene_id, view, state_id),
                        "fog_path": str(fog_path.resolve(strict=True)),
                        "fog_png_sha256": _sha256_bytes(png_bytes),
                        "source_l3_member": _official_rgb_member(scene_id, view, "L3"),
                        "source_l3_path": str(l3_path.resolve(strict=True)),
                        "source_l3_sha256": l3_sha,
                        "gt_sha256": gt_sha_cache[scene_id],
                        "camera_sha256": camera_sha,
                        "corruption_calibration_sha256": corruption_calibration_sha256,
                        "fog_recipe_sha256": recipe_sha256,
                    }
                    records.append(
                        {
                            **record_payload,
                            "fog_binding_sha256": _sha256_json(record_payload),
                            "materialization_status": materialization_status,
                        }
                    )
    except Exception:
        _cleanup_created_fog(created_paths, fog_root)
        raise
    return _FogMaterializationResult(records=tuple(records), created_paths=tuple(created_paths))


def _make_inventory(
    complete_scene_ids: Iterable[int],
) -> tuple[VerifiedSceneInventory, ...]:
    return tuple(
        VerifiedSceneInventory(
            scene_id=scene_id,
            verified_complete=True,
            inventory_sha256=_sha256_json(
                {
                    "attempt": "attempt-05",
                    "dataset": "DTU",
                    "scene_id": scene_id,
                    "verified_complete": True,
                }
            ),
        )
        for scene_id in sorted(set(complete_scene_ids))
    )


def build_attempt05_v4_split_assignment(
    *,
    complete_scene_ids: Iterable[int],
    source_root: Path | None,
) -> V4SplitAssignment:
    assignment = construct_v4_splits(
        _make_inventory(complete_scene_ids),
        source_root=source_root,
    )
    validate_v4_split_assignment(assignment)
    return assignment


def _official_rgb_member(scene_id: int, view_id: int, state_id: str) -> str:
    token = DTU_LIGHTING_SEMANTIC_TO_PHYSICAL_TOKEN[state_id]
    return f"Rectified/scan{scene_id}/rect_{view_id:03d}_{token}_r5000.png"


def _fog_member(scene_id: int, view_id: int, state_id: str) -> str:
    return f"SyntheticFog/scan{scene_id}/{state_id}/rect_{view_id:03d}_3_r5000.png"


def _state_for_scene(
    *,
    root: Path,
    source_root: Path | None,
    scene_id: int,
    state_id: str,
    ordered_views: tuple[int, ...],
    fog_root: Path,
) -> Any:
    cameras = {
        view_id: _asset_evidence(
            _camera_path(root, view_id),
            member=f"SampleSet/SampleSet/MVS Data/Calibration/cal18/pos_{view_id:03d}.txt",
        )
        for view_id in ordered_views
    }
    gt = _asset_evidence(
        _gt_path(root, scene_id),
        member=f"Points/Points/stl/stl{scene_id:03d}_total.ply",
    )
    mask = _asset_evidence(
        _mask_path(root, scene_id),
        member=f"SampleSet/SampleSet/MVS Data/ObsMask/ObsMask{scene_id}_10.mat",
    )
    l3_source = {
        view_id: _asset_evidence(
            _rgb_path(root, scene_id, view_id, "L3"),
            member=_official_rgb_member(scene_id, view_id, "L3"),
        )
        for view_id in ordered_views
    }
    if state_id in LIGHTING_STATES:
        rgb_inputs = {
            view_id: _asset_evidence(
                _rgb_path(root, scene_id, view_id, state_id),
                member=_official_rgb_member(scene_id, view_id, state_id),
            )
            for view_id in ordered_views
        }
        clean_source = None
    elif state_id in FOG_STATES:
        rgb_inputs = {
            view_id: _asset_evidence(
                fog_root / f"scan{scene_id}" / state_id / f"rect_{view_id:03d}_3_r5000.png",
                member=_fog_member(scene_id, view_id, state_id),
            )
            for view_id in ordered_views
        }
        clean_source = l3_source
    else:
        raise Attempt05InputClosureError("V4_MVE_BLOCKED_INPUT_STATE_SCOPE")
    return materialize_dtu_state_identity(
        source_root=source_root,
        scene_id=scene_id,
        state_id=state_id,
        ordered_view_ids=ordered_views,
        rgb_inputs=rgb_inputs,
        cameras=cameras,
        gt_point_cloud=gt,
        observability_mask=mask,
        clean_source_inputs=clean_source,
    )


def build_attempt05_calibration_schedule(
    *,
    split_assignment: V4SplitAssignment,
) -> tuple[dict[str, object], ...]:
    rows: list[dict[str, object]] = []
    for model_id in SCIENTIFIC_MODELS:
        for scene_id in split_assignment.calibration:
            payload = {
                "schema_version": ATTEMPT05_CALIBRATION_SCHEDULE_SCHEMA,
                "attempt_id": "attempt-05",
                "model_id": model_id,
                "scene_id": scene_id,
                "state_id": "L3",
                "scientific": False,
                "excluded_from_ci": True,
            }
            rows.append({**payload, "calibration_unit_sha256": _sha256_json(payload)})
    return tuple(rows)


def prepare_attempt05_inputs(
    *,
    context: Attempt05AuthorizedContext,
    dtu_root: Path,
    rectified_closure_manifest: Path,
    fog_root: Path,
    output_dir: Path,
    source_root: Path | None,
) -> Attempt05InputClosure:
    """Create the complete CPU input closure for Attempt-05.

    Test-scene view order is derived from the frozen closure resource schedule.
    Fog PNGs are materialized deterministically from frozen calibration metadata
    before any model execution can start.
    """

    if output_dir.exists() or output_dir.with_name(output_dir.name + ".partial").exists():
        raise Attempt05InputClosureError("V4_MVE_BLOCKED_INPUT_IMMUTABLE_COLLISION")
    schedule_view_order = _view_order_from_resource_schedule(_closure_schedule_path(context))
    closure_view_order = _view_order_from_closure_manifest(rectified_closure_manifest)
    for scene_id, ordered_views in schedule_view_order.items():
        closure_views = closure_view_order.get(scene_id)
        if closure_views is None or set(closure_views) != set(ordered_views):
            raise Attempt05InputClosureError("V4_MVE_BLOCKED_INPUT_VIEW_CLOSURE")
    view_order = schedule_view_order
    complete_scenes = _verified_complete_scenes(dtu_root, view_order, view_order)
    split_assignment = build_attempt05_v4_split_assignment(
        complete_scene_ids=complete_scenes,
        source_root=source_root,
    )
    corruption_calibration_path = context.runtime_root / "manifests" / "corruption_calibration.json"
    corruption_calibration, corruption_calibration_sha = _load_corruption_calibration(
        corruption_calibration_path
    )
    fog_result: _FogMaterializationResult | None = None
    output_partial = output_dir.with_name(output_dir.name + ".partial")
    try:
        fog_result = _materialize_fog_members(
            dtu_root,
            fog_root,
            view_order,
            corruption_calibration,
            corruption_calibration_sha256=corruption_calibration_sha,
        )
        states: list[Any] = []
        runtime_bindings: list[dict[str, object]] = []
        for scene_id in TEST_SCENE_IDS:
            ordered_views = view_order.get(scene_id)
            if ordered_views is None:
                raise Attempt05InputClosureError("V4_MVE_BLOCKED_INPUT_TEST_VIEW_CLOSURE")
            for state_id in (*LIGHTING_STATES, *FOG_STATES):
                states.append(
                    _state_for_scene(
                        root=dtu_root,
                        source_root=source_root,
                        scene_id=scene_id,
                        state_id=state_id,
                        ordered_views=ordered_views,
                        fog_root=fog_root,
                    )
                )
                runtime_bindings.append(
                    _runtime_binding_for_scene_state(
                        root=dtu_root,
                        fog_root=fog_root,
                        scene_id=scene_id,
                        state_id=state_id,
                        ordered_views=ordered_views,
                    )
                )
        scientific_schedule = build_attempt05_scientific_schedule(states)
        validate_scientific_schedule(scientific_schedule)

        calibration_schedule = build_attempt05_calibration_schedule(
            split_assignment=split_assignment
        )
        calibration_l3_bindings = []
        for scene_id in split_assignment.calibration:
            ordered_views = _fps_views_for_scene(dtu_root, scene_id)
            state = _state_for_scene(
                root=dtu_root,
                source_root=source_root,
                scene_id=scene_id,
                state_id="L3",
                ordered_views=ordered_views,
                fog_root=fog_root,
            )
            calibration_l3_bindings.append(state.to_dict())
            runtime_bindings.append(
                _runtime_binding_for_scene_state(
                    root=dtu_root,
                    fog_root=fog_root,
                    scene_id=scene_id,
                    state_id="L3",
                    ordered_views=ordered_views,
                )
            )

        output_partial.mkdir(parents=True, exist_ok=False)
        split_path = output_partial / "v4-split-assignment.json"
        states_path = output_partial / "v4-model-independent-states.json"
        schedule_path = output_partial / "v4-scientific-schedule-400.json"
        calibration_path = output_partial / "v4-calibration-l3-schedule-40.json"
        runtime_binding_path = output_partial / "v4-runtime-state-bindings.json"
        fog_binding_path = output_partial / "v4-fog-binding.json"
        manifest_path = output_partial / "v4-attempt05-input-closure.json"
        split_sha = _write_atomic_json(split_path, split_assignment.to_dict())
        states_payload = {
            "schema_version": ATTEMPT05_INPUT_CLOSURE_SCHEMA,
            "attempt_id": "attempt-05",
            "model_independent_state_count": 200,
            "states": [state.to_dict() for state in states],
        }
        states_sha = _write_atomic_json(states_path, states_payload)
        schedule_sha = _write_atomic_json(schedule_path, scientific_schedule.to_dict())
        calibration_payload = {
            "schema_version": ATTEMPT05_CALIBRATION_SCHEDULE_SCHEMA,
            "attempt_id": "attempt-05",
            "calibration_schedule": list(calibration_schedule),
            "calibration_l3_state_bindings": calibration_l3_bindings,
        }
        calibration_sha = _write_atomic_json(calibration_path, calibration_payload)
        fog_binding_payload = {
            "schema_version": ATTEMPT05_FOG_BINDING_SCHEMA,
            "attempt_id": "attempt-05",
            "corruption_calibration_sha256": corruption_calibration_sha,
            "fog_png_members": len(fog_result.records),
            "bindings": list(fog_result.records),
        }
        fog_binding_sha = _write_atomic_json(fog_binding_path, fog_binding_payload)
        runtime_binding_payload = {
            "schema_version": "georeliab-v4-attempt-05-runtime-state-bindings-1.0",
            "attempt_id": "attempt-05",
            "binding_count": len(runtime_bindings),
            "fog_binding_sha256": fog_binding_sha,
            "bindings": runtime_bindings,
        }
        runtime_binding_sha = _write_atomic_json(runtime_binding_path, runtime_binding_payload)
        budget_paths = {
            Path(str(row["fog_path"])).resolve(strict=True)
            for row in fog_result.records
        }
        budget_paths.update(
            {
                split_path,
                states_path,
                schedule_path,
                calibration_path,
                fog_binding_path,
                runtime_binding_path,
            }
        )
        input_logical_bytes = sum(path.stat().st_size for path in budget_paths)
        input_allocated_bytes = sum(_allocated_bytes(path) for path in budget_paths)
        manifest_payload = {
            "schema_version": ATTEMPT05_INPUT_CLOSURE_SCHEMA,
            "status": "PASS",
            "attempt_id": "attempt-05",
            "run_name": "v4-mve-attempt-05",
            "scientific_result": "NO_SCIENTIFIC_RESULT",
            "attempt04_authorization_sha256": context.authorization_sha256,
            "rectified_closure_manifest_sha256": _sha256_file(rectified_closure_manifest),
            "corruption_calibration_path": str(corruption_calibration_path),
            "corruption_calibration_sha256": corruption_calibration_sha,
            "split_assignment_sha256": split_sha,
            "state_inventory_sha256": states_sha,
            "scientific_schedule_sha256": schedule_sha,
            "calibration_schedule_sha256": calibration_sha,
            "fog_binding_sha256": fog_binding_sha,
            "runtime_state_bindings_sha256": runtime_binding_sha,
            "scientific_units": len(scientific_schedule.units),
            "scientific_state_count": len(states),
            "calibration_l3_units": len(calibration_schedule),
            "test_l3_reference_members": 160,
            "calibration_l3_reference_members": 160,
            "rectified_non_l3_members": 960,
            "fog_png_members": 480,
            "max_model_execution_units": 440,
            "budgeted_input_storage": {
                "scope": "FOG_PNG_AND_INPUT_CLOSURE_PAYLOADS_EXCLUDING_MANIFEST",
                "logical_bytes": input_logical_bytes,
                "allocated_bytes": input_allocated_bytes,
                "path_count": len(budget_paths),
            },
            "output_files": {
                "split_assignment": split_path.name,
                "model_independent_states": states_path.name,
                "scientific_schedule": schedule_path.name,
                "calibration_schedule": calibration_path.name,
                "fog_binding": fog_binding_path.name,
                "runtime_state_bindings": runtime_binding_path.name,
            },
        }
        manifest_sha = _write_atomic_json(manifest_path, manifest_payload)
        output_partial.replace(output_dir)
    except Exception:
        # Keep a failed partial for forensic inspection; callers must not treat
        # it as valid input closure. Remove only fog PNGs created by this failed
        # invocation so final-looking uncommitted fog cannot be admitted later.
        if fog_result is not None:
            _cleanup_created_fog(fog_result.created_paths, fog_root)
        raise

    return Attempt05InputClosure(
        status="PASS",
        split_assignment=split_assignment,
        model_independent_states=tuple(states),
        scientific_schedule=scientific_schedule,
        calibration_schedule=calibration_schedule,
        manifest_path=output_dir / "v4-attempt05-input-closure.json",
        manifest_sha256=manifest_sha,
        state_inventory_path=output_dir / "v4-model-independent-states.json",
        state_inventory_sha256=states_sha,
        scientific_schedule_path=output_dir / "v4-scientific-schedule-400.json",
        scientific_schedule_sha256=schedule_sha,
        split_assignment_path=output_dir / "v4-split-assignment.json",
        split_assignment_sha256=split_sha,
        calibration_schedule_path=output_dir / "v4-calibration-l3-schedule-40.json",
        calibration_schedule_sha256=calibration_sha,
        runtime_binding_path=output_dir / "v4-runtime-state-bindings.json",
        runtime_binding_sha256=runtime_binding_sha,
    )

def validate_attempt05_input_closure(path: Path) -> dict[str, Any]:
    payload = _read_json(path)
    if (
        payload.get("schema_version") != ATTEMPT05_INPUT_CLOSURE_SCHEMA
        or payload.get("status") != "PASS"
        or payload.get("attempt_id") != "attempt-05"
        or payload.get("scientific_result") != "NO_SCIENTIFIC_RESULT"
        or payload.get("scientific_units") != 400
        or payload.get("scientific_state_count") != 200
        or payload.get("calibration_l3_units") != 40
        or payload.get("rectified_non_l3_members") != 960
        or payload.get("fog_png_members") != 480
        or payload.get("max_model_execution_units") != 440
    ):
        raise Attempt05InputClosureError("V4_MVE_BLOCKED_INPUT_MANIFEST_TAMPER")
    storage = payload.get("budgeted_input_storage")
    if (
        not isinstance(storage, Mapping)
        or storage.get("scope")
        != "FOG_PNG_AND_INPUT_CLOSURE_PAYLOADS_EXCLUDING_MANIFEST"
        or type(storage.get("logical_bytes")) is not int
        or type(storage.get("allocated_bytes")) is not int
        or int(storage["logical_bytes"]) <= 0
        or int(storage["allocated_bytes"]) <= 0
    ):
        raise Attempt05InputClosureError("V4_MVE_BLOCKED_INPUT_STORAGE_BINDING_INVALID")
    return payload
