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
import re
from typing import Any
import zipfile
import zlib

import numpy as np

from .prepared_inputs import (
    DTU_IMAGE_SIZE,
    decode_srgb_png,
    derive_dtu_depth,
    parse_dtu_binary_ply,
)
from .preparation import CorruptionCalibration, deterministic_png
from .preparation import PreparationError
from .preparation_round2 import fog_render
from .tartanair_range import extract_range_members_evidence, index_remote_zip
from .v4_attempt04_authorization import MANIFEST_FILE_SHA256, SCHEDULE_FILE_SHA256
from .v4_attempt05_execution import (
    Attempt05AuthorizedContext,
    build_attempt05_scientific_schedule,
)
from .v4_counterfactuals import (
    AssetEvidence,
    DTU_LIGHTING_SEMANTIC_TO_PHYSICAL_TOKEN,
    DTU_OFFICIAL_SCENE_SET,
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
ATTEMPT05_CALIBRATION_L3_MATERIALIZATION_SCHEMA = (
    "georeliab-v4-attempt-05-calibration-l3-materialization-1.0"
)
DTU_INVENTORY_SCHEMA = "dtu-official-inventory-v1"
DTU_INVENTORY_FILE_SHA256 = (
    "e0000c803279620f302d68b55f2321e6e31be2a2f02f2e690d7ecf3bac24d197"
)
_MASK_MEMBER_RE = re.compile(
    r"^SampleSet/MVS Data/ObsMask/ObsMask([0-9]+)_10[.]mat$"
)
_POINT_MEMBER_RE = re.compile(r"^Points/stl/stl([0-9]{3})_total[.]ply$")


class Attempt05InputClosureError(V4ExecutionError):
    """Raised when Attempt-05 input closure must fail before GPU use."""


@dataclass(frozen=True, slots=True)
class _FogMaterializationResult:
    records: tuple[dict[str, object], ...]
    created_paths: tuple[Path, ...]


@dataclass(frozen=True, slots=True)
class _VerifiedDtuInventory:
    path: Path
    file_sha256: str
    rows: tuple[VerifiedSceneInventory, ...]


@dataclass(frozen=True, slots=True)
class _RectifiedArchiveBinding:
    url: str
    content_length: int
    etag: str


@dataclass(frozen=True, slots=True)
class _CalibrationL3MaterializationResult:
    records: tuple[dict[str, object], ...]
    central_directory_sha256: str
    observed_etag: str
    normalized_etag: str
    member_inventory_sha256: str
    written_member_count: int


@dataclass(frozen=True, slots=True)
class _SupportAssetMaterializationResult:
    records: tuple[dict[str, object], ...]
    written_paths: tuple[Path, ...]
    member_inventory_sha256: str


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


def _official_gt_member(scene_id: int) -> str:
    return f"Points/stl/stl{scene_id:03d}_total.ply"


def _official_mask_member(scene_id: int) -> str:
    return f"SampleSet/MVS Data/ObsMask/ObsMask{scene_id}_10.mat"


def _official_camera_member(view_id: int) -> str:
    return f"SampleSet/MVS Data/Calibration/cal18/pos_{view_id:03d}.txt"


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


def _rectified_archive_binding(
    context: Attempt05AuthorizedContext,
) -> _RectifiedArchiveBinding:
    receipt = _read_json(_resource_receipt_path(context))
    resource_bindings = receipt.get("resource_bindings")
    archives = (
        resource_bindings.get("dtu_archives")
        if isinstance(resource_bindings, Mapping)
        else None
    )
    rectified = archives.get("Rectified") if isinstance(archives, Mapping) else None
    central = (
        rectified.get("central_directory_identity")
        if isinstance(rectified, Mapping)
        else None
    )
    if not isinstance(rectified, Mapping) or not isinstance(central, Mapping):
        raise Attempt05InputClosureError(
            "V4_MVE_BLOCKED_INPUT_RECTIFIED_ARCHIVE_BINDING"
        )
    url = rectified.get("url")
    content_length = rectified.get("bytes")
    etag = rectified.get("etag")
    if (
        not isinstance(url, str)
        or not url.startswith("https://")
        or type(content_length) is not int
        or content_length <= 0
        or not isinstance(etag, str)
        or not etag
        or central.get("archive_bytes") != content_length
        or central.get("archive_etag") != etag
    ):
        raise Attempt05InputClosureError(
            "V4_MVE_BLOCKED_INPUT_RECTIFIED_ARCHIVE_BINDING"
        )
    return _RectifiedArchiveBinding(
        url=url,
        content_length=content_length,
        etag=etag,
    )


def _complete_scene_ids_from_archive_members(
    *,
    sample_members: Iterable[str],
    point_members: Iterable[str],
) -> frozenset[int]:
    """Derive GT-complete DTU scenes from exact frozen archive membership."""

    def scene_ids(members: Iterable[str], pattern: re.Pattern[str]) -> set[int]:
        names = tuple(str(member) for member in members)
        if len({name.casefold() for name in names}) != len(names):
            raise Attempt05InputClosureError(
                "V4_MVE_BLOCKED_INPUT_ARCHIVE_MEMBER_IDENTITY"
            )
        result: set[int] = set()
        for name in names:
            match = pattern.fullmatch(name)
            if match is None:
                continue
            scene_id = int(match.group(1))
            if scene_id not in DTU_OFFICIAL_SCENE_SET or scene_id in result:
                raise Attempt05InputClosureError(
                    "V4_MVE_BLOCKED_INPUT_ARCHIVE_MEMBER_IDENTITY"
                )
            result.add(scene_id)
        return result

    masks = scene_ids(sample_members, _MASK_MEMBER_RE)
    points = scene_ids(point_members, _POINT_MEMBER_RE)
    return frozenset(masks & points)


def _authorized_dtu_archive_paths(
    context: Attempt05AuthorizedContext,
) -> dict[str, Path]:
    receipt = _read_json(_resource_receipt_path(context))
    bindings = receipt.get("resource_bindings")
    archives = bindings.get("dtu_archives") if isinstance(bindings, Mapping) else None
    if not isinstance(archives, Mapping):
        raise Attempt05InputClosureError(
            "V4_MVE_BLOCKED_INPUT_RESOURCE_RECEIPT_TAMPER"
        )

    resolved: dict[str, Path] = {}
    for archive_name in ("SampleSet", "Points"):
        row = archives.get(archive_name)
        if not isinstance(row, Mapping):
            raise Attempt05InputClosureError(
                "V4_MVE_BLOCKED_INPUT_RESOURCE_RECEIPT_TAMPER"
            )
        raw_path = row.get("path")
        expected_bytes = row.get("bytes")
        expected_sha = row.get("sha256")
        if (
            not isinstance(raw_path, str)
            or type(expected_bytes) is not int
            or expected_bytes <= 0
            or not isinstance(expected_sha, str)
            or len(expected_sha) != 64
        ):
            raise Attempt05InputClosureError(
                "V4_MVE_BLOCKED_INPUT_RESOURCE_RECEIPT_TAMPER"
            )
        try:
            path = Path(raw_path).resolve(strict=True)
        except OSError as exc:
            raise Attempt05InputClosureError(
                "V4_MVE_BLOCKED_INPUT_RESOURCE_ARCHIVE_MISSING"
            ) from exc
        if (
            not path.is_file()
            or path.stat().st_size != expected_bytes
            or _sha256_file(path) != expected_sha
        ):
            raise Attempt05InputClosureError(
                "V4_MVE_BLOCKED_INPUT_RESOURCE_ARCHIVE_TAMPER"
            )
        resolved[archive_name] = path

    return resolved


def _authorized_complete_scene_ids(
    archive_paths: Mapping[str, Path],
) -> frozenset[int]:
    try:
        with zipfile.ZipFile(archive_paths["SampleSet"]) as sample_zip:
            sample_members = tuple(info.filename for info in sample_zip.infolist())
        with zipfile.ZipFile(archive_paths["Points"]) as point_zip:
            point_members = tuple(info.filename for info in point_zip.infolist())
    except (KeyError, OSError, zipfile.BadZipFile) as exc:
        raise Attempt05InputClosureError(
            "V4_MVE_BLOCKED_INPUT_RESOURCE_ARCHIVE_UNREADABLE"
        ) from exc
    complete = _complete_scene_ids_from_archive_members(
        sample_members=sample_members,
        point_members=point_members,
    )
    if not set(TEST_SCENE_IDS) <= complete or len(complete) < 40:
        raise Attempt05InputClosureError(
            "V4_MVE_BLOCKED_INPUT_COMPLETE_SCENE_CLOSURE"
        )
    return complete


def _crc32_file(path: Path) -> int:
    value = 0
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value = zlib.crc32(block, value)
    return value & 0xFFFFFFFF


def _materialize_assigned_support_assets(
    *,
    dtu_root: Path,
    scene_ids: Iterable[int],
    archive_paths: Mapping[str, Path],
) -> _SupportAssetMaterializationResult:
    assignments = (
        (
            "SampleSet",
            lambda scene_id: f"SampleSet/MVS Data/ObsMask/ObsMask{scene_id}_10.mat",
            lambda scene_id: _mask_path(dtu_root, scene_id),
        ),
        (
            "Points",
            lambda scene_id: f"Points/stl/stl{scene_id:03d}_total.ply",
            lambda scene_id: _gt_path(dtu_root, scene_id),
        ),
    )
    records: list[dict[str, object]] = []
    written: list[Path] = []
    ordered_scenes = tuple(sorted({int(scene_id) for scene_id in scene_ids}))
    try:
        for archive_name, member_for_scene, path_for_scene in assignments:
            with zipfile.ZipFile(archive_paths[archive_name]) as archive:
                for scene_id in ordered_scenes:
                    member = member_for_scene(scene_id)
                    try:
                        info = archive.getinfo(member)
                    except KeyError as exc:
                        raise Attempt05InputClosureError(
                            "V4_MVE_BLOCKED_INPUT_SUPPORT_ARCHIVE_MEMBER_MISSING"
                        ) from exc
                    destination = path_for_scene(scene_id)
                    partial = destination.with_name(destination.name + ".partial")
                    if partial.exists():
                        raise Attempt05InputClosureError(
                            "V4_MVE_BLOCKED_INPUT_SUPPORT_ASSET_PARTIAL"
                        )
                    disposition = "reused"
                    if destination.exists():
                        if (
                            not destination.is_file()
                            or destination.stat().st_size != info.file_size
                            or _crc32_file(destination) != info.CRC
                        ):
                            raise Attempt05InputClosureError(
                                "V4_MVE_BLOCKED_INPUT_SUPPORT_ASSET_CONFLICT"
                            )
                    else:
                        disposition = "written"
                        destination.parent.mkdir(parents=True, exist_ok=True)
                        try:
                            with archive.open(info) as source, partial.open("xb") as target:
                                for block in iter(lambda: source.read(1024 * 1024), b""):
                                    target.write(block)
                                target.flush()
                                os.fsync(target.fileno())
                            if (
                                partial.stat().st_size != info.file_size
                                or _crc32_file(partial) != info.CRC
                            ):
                                raise Attempt05InputClosureError(
                                    "V4_MVE_BLOCKED_INPUT_SUPPORT_ASSET_CRC"
                                )
                            partial.replace(destination)
                            written.append(destination)
                        except Exception:
                            partial.unlink(missing_ok=True)
                            raise
                    records.append(
                        {
                            "archive": archive_name,
                            "scene_id": scene_id,
                            "member": member,
                            "path": str(destination.resolve(strict=True)),
                            "bytes": info.file_size,
                            "crc32": f"{info.CRC:08x}",
                            "sha256": _sha256_file(destination),
                            "disposition": disposition,
                        }
                    )
    except (OSError, zipfile.BadZipFile) as exc:
        raise Attempt05InputClosureError(
            "V4_MVE_BLOCKED_INPUT_SUPPORT_ARCHIVE_UNREADABLE"
        ) from exc
    return _SupportAssetMaterializationResult(
        records=tuple(records),
        written_paths=tuple(written),
        member_inventory_sha256=_sha256_json(records),
    )


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
        member=_official_gt_member(scene_id),
    )
    mask = _asset_evidence(
        _mask_path(root, scene_id),
        member=_official_mask_member(scene_id),
    )
    cameras = []
    for view_id in ordered_views:
        cam_member = _official_camera_member(view_id)
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




def _global_frozen_view_order(
    authoritative_view_order: Mapping[int, Sequence[int]],
) -> tuple[int, ...]:
    """Return the single DTU camera order already frozen by the schedule.

    DTU calibration cameras are shared by every scan. Attempt-05 therefore
    must not recompute FPS from 49 camera files after materialization retained
    only the eight selected cameras. Every test scene must bind the same
    ordered tuple, which is then reused for scene-disjoint calibration.
    """

    orders = {
        tuple(int(view_id) for view_id in ordered_views)
        for ordered_views in authoritative_view_order.values()
    }
    if len(orders) != 1:
        raise Attempt05InputClosureError("V4_MVE_BLOCKED_INPUT_VIEW_ORDER_DRIFT")
    order = next(iter(orders))
    if len(order) != 8 or len(set(order)) != 8:
        raise Attempt05InputClosureError("V4_MVE_BLOCKED_INPUT_VIEW_ORDER_DRIFT")
    return order


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
    complete_scene_ids: Iterable[int] | None = None,
    inventory: Iterable[VerifiedSceneInventory] | None = None,
    source_root: Path | None,
) -> V4SplitAssignment:
    if (complete_scene_ids is None) == (inventory is None):
        raise Attempt05InputClosureError(
            "V4_MVE_BLOCKED_INPUT_DTU_INVENTORY_REQUIRED"
        )
    rows = (
        tuple(inventory)
        if inventory is not None
        else _make_inventory(complete_scene_ids or ())
    )
    assignment = construct_v4_splits(rows, source_root=source_root)
    validate_v4_split_assignment(assignment)
    return assignment


def _official_rgb_member(scene_id: int, view_id: int, state_id: str) -> str:
    token = DTU_LIGHTING_SEMANTIC_TO_PHYSICAL_TOKEN[state_id]
    return f"Rectified/scan{scene_id}/rect_{view_id:03d}_{token}_r5000.png"


def _expected_calibration_l3_members(
    *,
    split_assignment: V4SplitAssignment,
    ordered_views: Sequence[int],
) -> tuple[str, ...]:
    validate_v4_split_assignment(split_assignment)
    views = tuple(int(view_id) for view_id in ordered_views)
    if len(views) != 8 or len(set(views)) != 8:
        raise Attempt05InputClosureError(
            "V4_MVE_BLOCKED_INPUT_CALIBRATION_VIEW_ORDER"
        )
    members = tuple(
        _official_rgb_member(scene_id, view_id, "L3")
        for scene_id in split_assignment.calibration
        for view_id in views
    )
    if len(members) != 160 or len(set(members)) != 160:
        raise Attempt05InputClosureError(
            "V4_MVE_BLOCKED_INPUT_CALIBRATION_L3_CARDINALITY"
        )
    return members


def _normalize_strong_etag(value: str | None) -> str:
    if not isinstance(value, str):
        raise Attempt05InputClosureError(
            "V4_MVE_BLOCKED_INPUT_RECTIFIED_ARCHIVE_IDENTITY"
        )
    raw = value.strip()
    if raw.startswith("W/"):
        raise Attempt05InputClosureError(
            "V4_MVE_BLOCKED_INPUT_RECTIFIED_ARCHIVE_IDENTITY"
        )
    if len(raw) >= 2 and raw[0] == '"' and raw[-1] == '"':
        raw = raw[1:-1]
    if not raw or '"' in raw:
        raise Attempt05InputClosureError(
            "V4_MVE_BLOCKED_INPUT_RECTIFIED_ARCHIVE_IDENTITY"
        )
    return raw


def _materialize_calibration_l3_members(
    *,
    dtu_root: Path,
    split_assignment: V4SplitAssignment,
    ordered_views: Sequence[int],
    archive: _RectifiedArchiveBinding,
) -> _CalibrationL3MaterializationResult:
    members = _expected_calibration_l3_members(
        split_assignment=split_assignment,
        ordered_views=ordered_views,
    )
    for member in members:
        output = dtu_root / member
        if output.with_name(output.name + ".partial").exists():
            raise Attempt05InputClosureError(
                "V4_MVE_BLOCKED_INPUT_CALIBRATION_L3_PARTIAL"
            )
    try:
        index = index_remote_zip(archive.url)
    except PreparationError as exc:
        raise Attempt05InputClosureError(
            "V4_MVE_BLOCKED_INPUT_RECTIFIED_ARCHIVE_INDEX"
        ) from exc
    normalized_etag = _normalize_strong_etag(index.etag)
    if (
        index.content_length != archive.content_length
        or normalized_etag != _normalize_strong_etag(archive.etag)
        or not isinstance(index.central_directory_sha256, str)
        or len(index.central_directory_sha256) != 64
    ):
        raise Attempt05InputClosureError(
            "V4_MVE_BLOCKED_INPUT_RECTIFIED_ARCHIVE_IDENTITY"
        )
    if any(member not in index.entries for member in members):
        raise Attempt05InputClosureError(
            "V4_MVE_BLOCKED_INPUT_RECTIFIED_ARCHIVE_MEMBER_MISSING"
        )
    try:
        evidence = extract_range_members_evidence(
            archive.url,
            index,
            members,
            dtu_root,
        )
    except PreparationError as exc:
        raise Attempt05InputClosureError(
            "V4_MVE_BLOCKED_INPUT_RECTIFIED_ARCHIVE_EXTRACTION"
        ) from exc
    if set(evidence) != set(members):
        raise Attempt05InputClosureError(
            "V4_MVE_BLOCKED_INPUT_RECTIFIED_ARCHIVE_RESULT_SET"
        )
    records: list[dict[str, object]] = []
    for ordinal, member in enumerate(members):
        raw = evidence.get(member)
        entry = index.entries[member]
        expected_path = (dtu_root / member).resolve(strict=False)
        if not isinstance(raw, Mapping):
            raise Attempt05InputClosureError(
                "V4_MVE_BLOCKED_INPUT_RECTIFIED_ARCHIVE_RESULT_SCHEMA"
            )
        path_raw = raw.get("path")
        raw_sha = raw.get("raw_sha256")
        disposition = raw.get("disposition")
        try:
            resolved = Path(str(path_raw)).resolve(strict=True)
        except OSError as exc:
            raise Attempt05InputClosureError(
                "V4_MVE_BLOCKED_INPUT_RECTIFIED_ARCHIVE_RESULT_PATH"
            ) from exc
        if (
            resolved != expected_path
            or not resolved.is_file()
            or raw.get("member") != member
            or raw.get("compressed_size") != entry.compressed_size
            or raw.get("uncompressed_size") != entry.uncompressed_size
            or raw.get("crc32") != f"{entry.crc32:08x}"
            or not isinstance(raw_sha, str)
            or len(raw_sha) != 64
            or _sha256_file(resolved) != raw_sha
            or disposition not in {"reused", "written"}
        ):
            raise Attempt05InputClosureError(
                "V4_MVE_BLOCKED_INPUT_RECTIFIED_ARCHIVE_RESULT_SCHEMA"
            )
        records.append(
            {
                "ordinal": ordinal,
                "scene_id": split_assignment.calibration[ordinal // 8],
                "view_id": int(Path(member).name.split("_")[1]),
                "member": member,
                "path": str(resolved),
                "raw_sha256": raw_sha,
                "compressed_size": entry.compressed_size,
                "uncompressed_size": entry.uncompressed_size,
                "crc32": f"{entry.crc32:08x}",
                "disposition": disposition,
            }
        )
    return _CalibrationL3MaterializationResult(
        records=tuple(records),
        central_directory_sha256=index.central_directory_sha256,
        observed_etag=str(index.etag),
        normalized_etag=normalized_etag,
        member_inventory_sha256=_sha256_json(records),
        written_member_count=sum(
            record["disposition"] == "written" for record in records
        ),
    )


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
            member=_official_camera_member(view_id),
        )
        for view_id in ordered_views
    }
    gt = _asset_evidence(
        _gt_path(root, scene_id),
        member=_official_gt_member(scene_id),
    )
    mask = _asset_evidence(
        _mask_path(root, scene_id),
        member=_official_mask_member(scene_id),
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
    global_view_order = _global_frozen_view_order(view_order)
    archive_paths = _authorized_dtu_archive_paths(context)
    complete_scene_ids = _authorized_complete_scene_ids(archive_paths)
    inventory = _load_verified_dtu_inventory(
        context.runtime_root / "manifests" / "dtu_inventory.json",
        complete_scene_ids=complete_scene_ids,
    )
    split_assignment = build_attempt05_v4_split_assignment(
        inventory=inventory.rows,
        source_root=source_root,
    )
    rectified_archive = _rectified_archive_binding(context)
    calibration_materialization = _materialize_calibration_l3_members(
        dtu_root=dtu_root,
        split_assignment=split_assignment,
        ordered_views=global_view_order,
        archive=rectified_archive,
    )
    support_materialization = _materialize_assigned_support_assets(
        dtu_root=dtu_root,
        scene_ids=(*TEST_SCENE_IDS, *split_assignment.calibration),
        archive_paths=archive_paths,
    )
    for scene_id in (*TEST_SCENE_IDS, *split_assignment.calibration):
        ordered_views = view_order.get(scene_id, global_view_order)
        if not _scene_has_required_l3_assets(dtu_root, scene_id, ordered_views):
            raise Attempt05InputClosureError(
                "V4_MVE_BLOCKED_INPUT_ASSIGNED_SCENE_ASSET_MISSING"
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
            ordered_views = global_view_order
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
        calibration_materialization_path = (
            output_partial / "v4-calibration-l3-materialization.json"
        )
        support_materialization_path = (
            output_partial / "v4-support-asset-materialization.json"
        )
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
        calibration_materialization_payload = {
            "schema_version": ATTEMPT05_CALIBRATION_L3_MATERIALIZATION_SCHEMA,
            "attempt_id": "attempt-05",
            "source_inventory_path": str(inventory.path),
            "source_inventory_sha256": inventory.file_sha256,
            "archive": {
                "url": rectified_archive.url,
                "content_length": rectified_archive.content_length,
                "etag": rectified_archive.etag,
                "observed_etag": calibration_materialization.observed_etag,
                "normalized_etag": calibration_materialization.normalized_etag,
                "central_directory_sha256": (
                    calibration_materialization.central_directory_sha256
                ),
            },
            "member_count": len(calibration_materialization.records),
            "written_member_count": (
                calibration_materialization.written_member_count
            ),
            "member_inventory_sha256": (
                calibration_materialization.member_inventory_sha256
            ),
            "members": list(calibration_materialization.records),
        }
        calibration_materialization_sha = _write_atomic_json(
            calibration_materialization_path,
            calibration_materialization_payload,
        )
        support_materialization_payload = {
            "schema_version": (
                "georeliab-v4-attempt-05-support-asset-materialization-1.0"
            ),
            "attempt_id": "attempt-05",
            "member_count": len(support_materialization.records),
            "written_member_count": len(support_materialization.written_paths),
            "member_inventory_sha256": (
                support_materialization.member_inventory_sha256
            ),
            "members": list(support_materialization.records),
        }
        support_materialization_sha = _write_atomic_json(
            support_materialization_path,
            support_materialization_payload,
        )
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
                calibration_materialization_path,
                support_materialization_path,
                fog_binding_path,
                runtime_binding_path,
            }
        )
        budget_paths.update(
            Path(str(record["path"])).resolve(strict=True)
            for record in calibration_materialization.records
            if record["disposition"] == "written"
        )
        budget_paths.update(support_materialization.written_paths)
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
            "dtu_inventory_path": str(inventory.path),
            "dtu_inventory_sha256": inventory.file_sha256,
            "corruption_calibration_path": str(corruption_calibration_path),
            "corruption_calibration_sha256": corruption_calibration_sha,
            "split_assignment_sha256": split_sha,
            "state_inventory_sha256": states_sha,
            "scientific_schedule_sha256": schedule_sha,
            "calibration_schedule_sha256": calibration_sha,
            "calibration_l3_materialization_sha256": (
                calibration_materialization_sha
            ),
            "calibration_l3_member_inventory_sha256": (
                calibration_materialization.member_inventory_sha256
            ),
            "support_asset_materialization_sha256": support_materialization_sha,
            "support_asset_member_inventory_sha256": (
                support_materialization.member_inventory_sha256
            ),
            "support_asset_members": len(support_materialization.records),
            "support_asset_written_members": len(
                support_materialization.written_paths
            ),
            "fog_binding_sha256": fog_binding_sha,
            "runtime_state_bindings_sha256": runtime_binding_sha,
            "scientific_units": len(scientific_schedule.units),
            "scientific_state_count": len(states),
            "calibration_l3_units": len(calibration_schedule),
            "test_l3_reference_members": 160,
            "calibration_l3_reference_members": 160,
            "calibration_l3_written_members": (
                calibration_materialization.written_member_count
            ),
            "rectified_non_l3_members": 960,
            "fog_png_members": 480,
            "max_model_execution_units": 440,
            "budgeted_input_storage": {
                "scope": (
                    "NEW_CALIBRATION_L3_FOG_PNG_AND_INPUT_CLOSURE_"
                    "PAYLOADS_EXCLUDING_MANIFEST"
                ),
                "logical_bytes": input_logical_bytes,
                "allocated_bytes": input_allocated_bytes,
                "path_count": len(budget_paths),
            },
            "output_files": {
                "split_assignment": split_path.name,
                "model_independent_states": states_path.name,
                "scientific_schedule": schedule_path.name,
                "calibration_schedule": calibration_path.name,
                "calibration_l3_materialization": (
                    calibration_materialization_path.name
                ),
                "support_asset_materialization": support_materialization_path.name,
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
        or payload.get("calibration_l3_reference_members") != 160
        or payload.get("rectified_non_l3_members") != 960
        or payload.get("fog_png_members") != 480
        or payload.get("max_model_execution_units") != 440
    ):
        raise Attempt05InputClosureError("V4_MVE_BLOCKED_INPUT_MANIFEST_TAMPER")
    storage = payload.get("budgeted_input_storage")
    if (
        not isinstance(storage, Mapping)
        or storage.get("scope")
        != (
            "NEW_CALIBRATION_L3_FOG_PNG_AND_INPUT_CLOSURE_"
            "PAYLOADS_EXCLUDING_MANIFEST"
        )
        or type(storage.get("logical_bytes")) is not int
        or type(storage.get("allocated_bytes")) is not int
        or int(storage["logical_bytes"]) <= 0
        or int(storage["allocated_bytes"]) <= 0
    ):
        raise Attempt05InputClosureError("V4_MVE_BLOCKED_INPUT_STORAGE_BINDING_INVALID")
    return payload


def _load_verified_dtu_inventory(
    path: Path,
    *,
    complete_scene_ids: Iterable[int] = DTU_OFFICIAL_SCENE_SET,
) -> _VerifiedDtuInventory:
    try:
        resolved = path.resolve(strict=True)
    except OSError as exc:
        raise Attempt05InputClosureError(
            "V4_MVE_BLOCKED_INPUT_DTU_INVENTORY_MISSING"
        ) from exc
    file_sha256 = _sha256_file(resolved)
    if file_sha256 != DTU_INVENTORY_FILE_SHA256:
        raise Attempt05InputClosureError(
            "V4_MVE_BLOCKED_INPUT_DTU_INVENTORY_HASH_MISMATCH"
        )
    payload = _read_json(resolved)
    if set(payload) != {"schema_version", "scenes"} or payload.get(
        "schema_version"
    ) != DTU_INVENTORY_SCHEMA:
        raise Attempt05InputClosureError(
            "V4_MVE_BLOCKED_INPUT_DTU_INVENTORY_SCHEMA"
        )
    scenes = payload.get("scenes")
    if not isinstance(scenes, list) or len(scenes) != len(DTU_OFFICIAL_SCENE_SET):
        raise Attempt05InputClosureError(
            "V4_MVE_BLOCKED_INPUT_DTU_INVENTORY_CARDINALITY"
        )
    complete = frozenset(int(scene_id) for scene_id in complete_scene_ids)
    if not set(TEST_SCENE_IDS) <= complete or not complete <= DTU_OFFICIAL_SCENE_SET:
        raise Attempt05InputClosureError(
            "V4_MVE_BLOCKED_INPUT_COMPLETE_SCENE_CLOSURE"
        )
    rows: list[VerifiedSceneInventory] = []
    seen: set[int] = set()
    expected_camera_ids = {str(view_id) for view_id in range(1, 50)}
    for raw in scenes:
        if not isinstance(raw, Mapping) or set(raw) != {
            "scene_id",
            "camera_centers",
            "rgb_files",
            "points_path",
            "mask_path",
        }:
            raise Attempt05InputClosureError(
                "V4_MVE_BLOCKED_INPUT_DTU_INVENTORY_SCHEMA"
            )
        scene_id = raw.get("scene_id")
        if type(scene_id) is not int or scene_id not in DTU_OFFICIAL_SCENE_SET:
            raise Attempt05InputClosureError(
                "V4_MVE_BLOCKED_INPUT_DTU_INVENTORY_SCENE"
            )
        if scene_id in seen:
            raise Attempt05InputClosureError(
                "V4_MVE_BLOCKED_INPUT_DTU_INVENTORY_DUPLICATE"
            )
        seen.add(scene_id)
        centers = raw.get("camera_centers")
        if not isinstance(centers, Mapping) or set(centers) != expected_camera_ids:
            raise Attempt05InputClosureError(
                "V4_MVE_BLOCKED_INPUT_DTU_INVENTORY_CAMERA"
            )
        for center in centers.values():
            if (
                not isinstance(center, list)
                or len(center) != 3
                or any(
                    isinstance(value, bool)
                    or not isinstance(value, (int, float))
                    or not np.isfinite(float(value))
                    for value in center
                )
            ):
                raise Attempt05InputClosureError(
                    "V4_MVE_BLOCKED_INPUT_DTU_INVENTORY_CAMERA"
                )
        rgb_files = raw.get("rgb_files")
        expected_rgb = [
            f"rect_{view_id:03d}_3_r5000.png" for view_id in range(1, 50)
        ]
        if rgb_files != expected_rgb:
            raise Attempt05InputClosureError(
                "V4_MVE_BLOCKED_INPUT_DTU_INVENTORY_RGB"
            )
        if raw.get("points_path") != f"Points/stl/stl{scene_id:03d}_total.ply":
            raise Attempt05InputClosureError(
                "V4_MVE_BLOCKED_INPUT_DTU_INVENTORY_POINTS"
            )
        if (
            raw.get("mask_path")
            != f"SampleSet/MVS Data/ObsMask/ObsMask{scene_id}_10.mat"
        ):
            raise Attempt05InputClosureError(
                "V4_MVE_BLOCKED_INPUT_DTU_INVENTORY_MASK"
            )
        rows.append(
            VerifiedSceneInventory(
                scene_id=scene_id,
                verified_complete=scene_id in complete,
                inventory_sha256=_sha256_json(dict(raw)),
            )
        )
    if seen != DTU_OFFICIAL_SCENE_SET:
        raise Attempt05InputClosureError(
            "V4_MVE_BLOCKED_INPUT_DTU_INVENTORY_SCENE_SET"
        )
    return _VerifiedDtuInventory(
        path=resolved,
        file_sha256=file_sha256,
        rows=tuple(sorted(rows, key=lambda row: row.scene_id)),
    )
