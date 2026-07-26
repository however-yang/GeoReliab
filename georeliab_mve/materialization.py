"""Fail-closed acquisition and provenance for frozen DTU/TartanAir inputs."""

from __future__ import annotations

from collections.abc import Mapping
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
from typing import Any
import zipfile
import zlib

import numpy as np

from .preparation import DtuScene, PreparationError, atomic_json
from .tartanair import P000_FRAME_COUNT, select_uniform_aligned_p000_members
from .tartanair_range import (
    RemoteZipIndex,
    extract_range_members_evidence,
    index_remote_zip,
)


DTU_SCAN_IDS = tuple((*range(1, 78), *range(82, 129)))
DTU_SCAN_SET = frozenset(DTU_SCAN_IDS)
DTU_LIGHTING3_COUNT = len(DTU_SCAN_IDS) * 49
TARTANAIR_P000_COUNT = P000_FRAME_COUNT
_RECTIFIED_RE = re.compile(
    r"^Rectified/scan(?P<scene>[1-9][0-9]*)/"
    r"rect_(?P<view>00[1-9]|0[1-4][0-9])_3_r5000\.png$"
)
_TARTAN_IMAGE_RE = re.compile(
    r"^GreatMarsh/Data_easy/P000/image_lcam_front/"
    r"(?P<frame>[0-9]{6})_lcam_front\.png$"
)
_TARTAN_DEPTH_RE = re.compile(
    r"^GreatMarsh/Data_easy/P000/depth_lcam_front/"
    r"(?P<frame>[0-9]{6})_lcam_front_depth\.png$"
)
FROZEN_TYPING_EXTENSIONS_SITE = Path("/home/smli/miniforge3/pkgs/typing_extensions-4.15.0-pyhcf101f3_0/site-packages")
FROZEN_TYPING_EXTENSIONS_VERSION = "4.15.0"
FROZEN_TYPING_EXTENSIONS_SHA256 = "433d11d170d3a24d2eb065ebc1bfe848cea7e3d7ce68567ab52bea2d4c2f7ed8"
FROZEN_TYPING_EXTENSIONS_DIST_INFO = "typing_extensions-4.15.0.dist-info"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def normal_etag(value: object) -> str:
    if not isinstance(value, str) or not value.strip():
        raise PreparationError("remote archive evidence is missing an ETag")
    return value.strip().removeprefix("W/").strip().strip('"').lower()


def require_sha256(value: object, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(char not in "0123456789abcdef" for char in value)
    ):
        raise PreparationError(f"{label} must be a lowercase SHA-256")
    return value


def require_git_commit(value: object, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 40
        or any(char not in "0123456789abcdef" for char in value.lower())
    ):
        raise PreparationError(f"{label} must be a 40-character Git commit")
    return value.lower()


def verify_typing_extensions_dependency(
    *,
    site: Path,
    expected_sha256: str,
    expected_version: str,
    enforce_home_prefix: bool = True,
) -> dict[str, Any]:
    """Verify the frozen typing_extensions module used by isolated probes."""

    if enforce_home_prefix:
        frozen_site = FROZEN_TYPING_EXTENSIONS_SITE.as_posix()
        requested_site = site.as_posix().rstrip("/")
        if requested_site != frozen_site:
            raise PreparationError(
                f"typing_extensions site must equal the frozen package cache: {FROZEN_TYPING_EXTENSIONS_SITE}"
            )
        try:
            normalized = site.resolve().as_posix().rstrip("/")
        except OSError as exc:
            raise PreparationError(f"cannot resolve typing_extensions site: {site}") from exc
        if normalized != frozen_site:
            raise PreparationError(
                f"typing_extensions site must resolve to the frozen package cache: {FROZEN_TYPING_EXTENSIONS_SITE}"
            )
    expected_sha = require_sha256(expected_sha256, "typing_extensions.py digest")
    if expected_version != FROZEN_TYPING_EXTENSIONS_VERSION:
        raise PreparationError(
            f"typing_extensions version mismatch: {expected_version} != {FROZEN_TYPING_EXTENSIONS_VERSION}"
        )
    path = site / "typing_extensions.py"
    if not path.is_file():
        raise PreparationError(f"frozen typing_extensions.py is missing: {path}")
    actual_sha = sha256_file(path)
    if actual_sha != expected_sha:
        raise PreparationError(
            f"typing_extensions.py SHA-256 mismatch: {actual_sha} != {expected_sha}"
        )
    return {
        "site": str(site),
        "path": str(path),
        "version": expected_version,
        "sha256": actual_sha,
    }


def validate_rectified_index(index: RemoteZipIndex) -> dict[int, tuple[str, ...]]:
    """Require the exact official scan/view inventory for lighting 3."""

    by_scene: dict[int, dict[int, str]] = {}
    lighting3_names = [
        name for name in index.entries if name.endswith("_3_r5000.png")
    ]
    if len(lighting3_names) != DTU_LIGHTING3_COUNT:
        raise PreparationError(
            "Rectified index must contain exactly 6076 official lighting-3 members"
        )
    for name in lighting3_names:
        match = _RECTIFIED_RE.fullmatch(name)
        if match is None:
            raise PreparationError(
                f"Rectified index contains misnamed lighting-3 member: {name}"
            )
        scene, view = int(match.group("scene")), int(match.group("view"))
        if scene not in DTU_SCAN_SET:
            raise PreparationError(
                f"Rectified index contains unexpected official scan id: {scene}"
            )
        if view in by_scene.setdefault(scene, {}):
            raise PreparationError(
                f"Rectified index contains duplicate scan{scene} view {view}"
            )
        by_scene[scene][view] = name
    if set(by_scene) != DTU_SCAN_SET:
        raise PreparationError(
            "Rectified index must contain exactly scans 1..77,82..128; "
            f"missing={sorted(DTU_SCAN_SET - set(by_scene))}, "
            f"unexpected={sorted(set(by_scene) - DTU_SCAN_SET)}"
        )
    expected_views = set(range(1, 50))
    for scene, members in by_scene.items():
        if set(members) != expected_views:
            raise PreparationError(
                f"Rectified scan{scene} must contain views 1..49; "
                f"missing={sorted(expected_views - set(members))}, "
                f"unexpected={sorted(set(members) - expected_views)}"
            )
    return {
        scene: tuple(by_scene[scene][view] for view in range(1, 50))
        for scene in DTU_SCAN_IDS
    }


def validate_tartanair_indexes(
    image: RemoteZipIndex, depth: RemoteZipIndex
) -> tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...]]:
    """Require exactly 3,537 aligned P000 image/depth members."""

    def collect(
        entries: Mapping[str, object], pattern: re.Pattern[str], modality: str
    ) -> dict[str, str]:
        candidates = [
            name
            for name in entries
            if "GreatMarsh/Data_easy/P000/" in name
            and f"/{modality}_lcam_front/" in name
            and not name.endswith("/")
        ]
        if len(candidates) != TARTANAIR_P000_COUNT:
            raise PreparationError(
                f"TartanAir {modality} archive must contain exactly 3537 P000 members"
            )
        result: dict[str, str] = {}
        for name in candidates:
            match = pattern.fullmatch(name)
            if match is None:
                raise PreparationError(
                    f"TartanAir {modality} archive contains misnamed P000 member: {name}"
                )
            frame = match.group("frame")
            if frame in result:
                raise PreparationError(
                    f"TartanAir {modality} archive contains duplicate frame {frame}"
                )
            result[frame] = name
        expected = {f"{index:06d}" for index in range(TARTANAIR_P000_COUNT)}
        if set(result) != expected:
            raise PreparationError(
                f"TartanAir {modality} P000 frame ids are not exactly 000000..003536"
            )
        return result

    images = collect(image.entries, _TARTAN_IMAGE_RE, "image")
    depths = collect(depth.entries, _TARTAN_DEPTH_RE, "depth")
    if set(images) != set(depths):
        raise PreparationError("TartanAir P000 RGB/depth frame ids are misaligned")
    selection = select_uniform_aligned_p000_members(
        tuple(images.values()), tuple(depths.values())
    )
    frame_ids = tuple(
        _TARTAN_IMAGE_RE.fullmatch(name).group("frame")  # type: ignore[union-attr]
        for name in selection.image_members
    )
    return frame_ids, selection.image_members, selection.depth_members


def validate_remote_indexes(
    resources: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, RemoteZipIndex]]:
    """Index all remote archives and validate identities and inventories."""

    specs = (
        ("Rectified.zip", "dtu_rectified_url", "dtu_rectified_bytes", "dtu_rectified_etag"),
        ("tartanair-image", "tartanair_image_url", "tartanair_image_bytes", "tartanair_image_etag"),
        ("tartanair-depth", "tartanair_depth_url", "tartanair_depth_bytes", "tartanair_depth_etag"),
    )
    indexes: dict[str, RemoteZipIndex] = {}
    evidence_rows: list[dict[str, Any]] = []
    for name, url_key, bytes_key, etag_key in specs:
        url = str(resources[url_key])
        index = index_remote_zip(url)
        if (
            index.content_length != int(resources[bytes_key])
            or normal_etag(index.etag) != normal_etag(resources[etag_key])
        ):
            raise PreparationError(f"{name} remote identity mismatch")
        require_sha256(index.central_directory_sha256, f"{name} central-directory digest")
        indexes[name] = index
        evidence_rows.append(
            {
                "name": name,
                "url": url,
                "bytes": index.content_length,
                "etag": normal_etag(index.etag),
                "central_directory_sha256": index.central_directory_sha256,
                "member_count": len(index.entries),
            }
        )

    rectified = validate_rectified_index(indexes["Rectified.zip"])
    frame_ids, image_members, depth_members = validate_tartanair_indexes(
        indexes["tartanair-image"], indexes["tartanair-depth"]
    )
    for row in evidence_rows:
        if row["name"] == "Rectified.zip":
            row["required_member_count"] = sum(map(len, rectified.values()))
        elif row["name"] == "tartanair-image":
            row["p000_member_count"] = TARTANAIR_P000_COUNT
            row["selected_members"] = list(image_members)
        else:
            row["p000_member_count"] = TARTANAIR_P000_COUNT
            row["selected_members"] = list(depth_members)
    return (
        {
            "schema_version": "remote-zip-evidence-v1",
            "remote_indexes": evidence_rows,
            "tartanair_selected_frame_ids": list(frame_ids),
        },
        indexes,
    )


def _unique_suffix_member(
    archive: zipfile.ZipFile, suffix: str, label: str
) -> str:
    normalized = suffix.replace("\\", "/")
    matches = [
        info.filename
        for info in archive.infolist()
        if not info.is_dir()
        and info.filename.replace("\\", "/").endswith(normalized)
    ]
    if len(matches) != 1:
        raise PreparationError(
            f"{label} must resolve to exactly one verified archive member: {suffix}"
        )
    return matches[0]


def _projection_center(raw: bytes, label: str) -> np.ndarray:
    try:
        lines = [line.decode("ascii") for line in raw.splitlines()]
        matrix = np.loadtxt(lines, dtype=np.float64)
    except (UnicodeDecodeError, ValueError) as exc:
        raise PreparationError(f"{label} is not an ASCII 3x4 camera matrix") from exc
    if matrix.size != 12:
        raise PreparationError(f"{label} is not a 3x4 camera matrix")
    matrix = matrix.reshape(3, 4)
    try:
        center = -np.linalg.solve(matrix[:, :3], matrix[:, 3])
    except np.linalg.LinAlgError as exc:
        raise PreparationError(f"{label} has a singular camera matrix") from exc
    if center.shape != (3,) or not np.isfinite(center).all():
        raise PreparationError(f"{label} has an invalid camera center")
    return center


def build_dtu_archive_inventory(
    sample_set_path: Path,
    points_path: Path,
    rectified: RemoteZipIndex,
) -> tuple[tuple[DtuScene, ...], dict[str, Any]]:
    """Build the canonical inventory without extracting whole archives."""

    rectified_members = validate_rectified_index(rectified)
    with zipfile.ZipFile(sample_set_path) as sample_archive, zipfile.ZipFile(
        points_path
    ) as points_archive:
        camera_members = {
            view: _unique_suffix_member(
                sample_archive,
                f"MVS Data/Calibration/cal18/pos_{view:03d}.txt",
                f"DTU camera pos_{view:03d}",
            )
            for view in range(1, 50)
        }
        centers = {
            view: _projection_center(
                sample_archive.read(member), f"DTU camera pos_{view:03d}"
            )
            for view, member in camera_members.items()
        }
        scenes: list[DtuScene] = []
        sources: list[dict[str, Any]] = []
        for scene in DTU_SCAN_IDS:
            points_member = _unique_suffix_member(
                points_archive,
                f"Points/stl/stl{scene:03d}_total.ply",
                f"DTU scan{scene} points",
            )
            mask_member = _unique_suffix_member(
                sample_archive,
                f"MVS Data/ObsMask/ObsMask{scene}_10.mat",
                f"DTU scan{scene} observability mask",
            )
            scenes.append(
                DtuScene(
                    scene_id=scene,
                    rgb_files=tuple(
                        Path(name).name for name in rectified_members[scene]
                    ),
                    camera_centers=dict(centers),
                    points_path=points_member,
                    mask_path=mask_member,
                )
            )
            sources.append(
                {
                    "scene_id": scene,
                    "rectified_members": list(rectified_members[scene]),
                    "points_member": points_member,
                    "mask_member": mask_member,
                }
            )
    return (
        tuple(scenes),
        {
            "schema_version": "dtu-archive-inventory-provenance-v1",
            "camera_members": {
                str(view): member for view, member in camera_members.items()
            },
            "scenes": sources,
        },
    )


def _atomic_materialize_bytes(
    destination: Path, data: bytes, *, label: str
) -> tuple[str, str]:
    digest = hashlib.sha256(data).hexdigest()
    if destination.exists():
        if sha256_file(destination) != digest:
            raise PreparationError(
                f"existing materialized {label} fails frozen digest verification"
            )
        return digest, "reused"
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_suffix(destination.suffix + ".partial")
    partial.write_bytes(data)
    if hashlib.sha256(partial.read_bytes()).hexdigest() != digest:
        raise PreparationError(f"partial materialized {label} digest mismatch")
    partial.replace(destination)
    return digest, "written"


def _local_member_evidence(
    *,
    archive: zipfile.ZipFile,
    archive_path: Path,
    archive_sha256: str,
    member: str,
    destination_root: Path,
    archive_name: str,
) -> dict[str, Any]:
    require_sha256(archive_sha256, f"{archive_name} archive digest")
    try:
        info = archive.getinfo(member)
        data = archive.read(info)
    except (KeyError, OSError, zipfile.BadZipFile) as exc:
        raise PreparationError(
            f"verified {archive_name} archive is missing member {member}"
        ) from exc
    if len(data) != info.file_size or zlib.crc32(data) & 0xFFFFFFFF != info.CRC:
        raise PreparationError(f"{archive_name} member integrity mismatch: {member}")
    output = destination_root / archive_name.removesuffix(".zip") / member
    digest, disposition = _atomic_materialize_bytes(
        output, data, label=f"{archive_name}:{member}"
    )
    return {
        "archive_name": archive_name,
        "archive_path": str(archive_path),
        "archive_sha256": archive_sha256,
        "member": member,
        "compressed_size": info.compress_size,
        "uncompressed_size": info.file_size,
        "crc32": f"{info.CRC:08x}",
        "raw_sha256": digest,
        "path": str(output),
        "disposition": disposition,
    }


def _remote_archive_evidence(
    name: str,
    url: str,
    index: RemoteZipIndex,
    members: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    return {
        "archive_name": name,
        "url": url,
        "bytes": index.content_length,
        "etag": normal_etag(index.etag),
        "central_directory_sha256": index.central_directory_sha256,
        "members": dict(members),
    }


def materialize_frozen_selection(
    *,
    root: Path,
    resources: Mapping[str, Any],
    split_manifest_path: Path,
    dtu_inventory_provenance_path: Path,
    typing_extensions_site: Path = FROZEN_TYPING_EXTENSIONS_SITE,
    enforce_typing_extensions_home: bool = True,
) -> dict[str, Any]:
    """Materialize only the 45x8 DTU views and 100 aligned TartanAir pairs."""

    try:
        split_payload = json.loads(split_manifest_path.read_text(encoding="utf-8"))
        provenance = json.loads(
            dtu_inventory_provenance_path.read_text(encoding="utf-8")
        )
    except (OSError, json.JSONDecodeError) as exc:
        raise PreparationError(
            "materialization requires canonical split and inventory provenance"
        ) from exc
    if split_payload.get("schema_version") != "dtu-preparation-v1":
        raise PreparationError("split/view manifest has an unsupported schema")
    expected_scenes = set().union(
        *(set(values) for values in split_payload["splits"].values())
    )
    if expected_scenes != set(map(int, split_payload["views"])):
        raise PreparationError(
            "split/view manifest must contain views for every frozen scene"
        )
    if len(expected_scenes) != 45:
        raise PreparationError("split/view manifest must contain exactly 45 scenes")

    typing_dependency = verify_typing_extensions_dependency(
        site=typing_extensions_site,
        expected_sha256=str(resources.get("typing_extensions_sha256", FROZEN_TYPING_EXTENSIONS_SHA256)),
        expected_version=str(resources.get("typing_extensions_version", FROZEN_TYPING_EXTENSIONS_VERSION)),
        enforce_home_prefix=enforce_typing_extensions_home,
    )

    _, indexes = validate_remote_indexes(resources)
    rectified = indexes["Rectified.zip"]
    tartan_image = indexes["tartanair-image"]
    tartan_depth = indexes["tartanair-depth"]
    _, image_members, depth_members = validate_tartanair_indexes(
        tartan_image, tartan_depth
    )
    scene_sources = {
        int(row["scene_id"]): row for row in provenance.get("scenes", [])
    }
    camera_members = {
        int(view): member
        for view, member in provenance.get("camera_members", {}).items()
    }
    if set(scene_sources) != DTU_SCAN_SET or set(camera_members) != set(
        range(1, 50)
    ):
        raise PreparationError("DTU inventory provenance is incomplete")
    for view, member in camera_members.items():
        normalized = str(member).replace("\\", "/")
        expected = f"MVS Data/Calibration/cal18/pos_{view:03d}.txt"
        if not normalized.endswith(expected):
            raise PreparationError(
                f"DTU camera provenance view {view} must reference exact pos_{view:03d}.txt"
            )

    dtu_members: list[str] = []
    for scene in sorted(expected_scenes):
        by_view: dict[int, str] = {}
        for name in scene_sources[scene]["rectified_members"]:
            match = _RECTIFIED_RE.fullmatch(str(name))
            if match is None or int(match.group("scene")) != scene:
                raise PreparationError(
                    f"DTU inventory provenance contains a misbound scan{scene} RGB member"
                )
            by_view[int(match.group("view"))] = str(name)
        for view in split_payload["views"][str(scene)]:
            if int(view) not in by_view:
                raise PreparationError(
                    f"split/view manifest references missing scan{scene} view {view}"
                )
            dtu_members.append(by_view[int(view)])
    dtu_remote = extract_range_members_evidence(
        str(resources["dtu_rectified_url"]),
        rectified,
        dtu_members,
        root / "materialized",
    )
    tartan_image_remote = extract_range_members_evidence(
        str(resources["tartanair_image_url"]),
        tartan_image,
        image_members,
        root / "materialized",
    )
    tartan_depth_remote = extract_range_members_evidence(
        str(resources["tartanair_depth_url"]),
        tartan_depth,
        depth_members,
        root / "materialized",
    )

    from .archive_round1 import verify_archive

    sample_sha = str(resources["dtu_sampleset_sha256"])
    points_sha = str(resources["dtu_points_sha256"])
    sample_path, points_path = root / "SampleSet.zip", root / "Points.zip"
    verify_archive(
        sample_path,
        expected_bytes=int(resources["dtu_sampleset_bytes"]),
        expected_sha256=sample_sha,
    )
    verify_archive(
        points_path,
        expected_bytes=int(resources["dtu_points_bytes"]),
        expected_sha256=points_sha,
    )
    dtu_rows: list[dict[str, Any]] = []
    with zipfile.ZipFile(sample_path) as sample_archive, zipfile.ZipFile(
        points_path
    ) as points_archive:
        for scene in sorted(expected_scenes):
            source = scene_sources[scene]
            points_evidence = _local_member_evidence(
                archive=points_archive,
                archive_path=points_path,
                archive_sha256=points_sha,
                member=source["points_member"],
                destination_root=root / "materialized",
                archive_name="Points.zip",
            )
            mask_evidence = _local_member_evidence(
                archive=sample_archive,
                archive_path=sample_path,
                archive_sha256=sample_sha,
                member=source["mask_member"],
                destination_root=root / "materialized",
                archive_name="SampleSet.zip",
            )
            camera_evidence: dict[str, Any] = {}
            rgb_evidence: dict[str, Any] = {}
            by_view: dict[int, str] = {}
            for name in source["rectified_members"]:
                match = _RECTIFIED_RE.fullmatch(str(name))
                if match is None or int(match.group("scene")) != scene:
                    raise PreparationError(
                        f"DTU inventory provenance contains a misbound scan{scene} RGB member"
                    )
                by_view[int(match.group("view"))] = str(name)
            for view_value in split_payload["views"][str(scene)]:
                view = int(view_value)
                camera_evidence[str(view)] = _local_member_evidence(
                    archive=sample_archive,
                    archive_path=sample_path,
                    archive_sha256=sample_sha,
                    member=camera_members[view],
                    destination_root=root / "materialized",
                    archive_name="SampleSet.zip",
                )
                rgb_evidence[str(view)] = dtu_remote[by_view[view]]
            split = next(
                name
                for name, scenes in split_payload["splits"].items()
                if scene in scenes
            )
            dtu_rows.append(
                {
                    "scene_id": scene,
                    "split": split,
                    "points": points_evidence,
                    "mask": mask_evidence,
                    "cameras": camera_evidence,
                    "rgb": rgb_evidence,
                }
            )

    tartan_pairs = []
    for image_member, depth_member in zip(image_members, depth_members):
        image_match = _TARTAN_IMAGE_RE.fullmatch(image_member)
        if image_match is None:
            raise PreparationError("selected TartanAir RGB member is misnamed")
        tartan_pairs.append(
            {
                "frame_id": image_match.group("frame"),
                "rgb": tartan_image_remote[image_member],
                "depth": tartan_depth_remote[depth_member],
            }
        )
    payload = {
        "schema_version": "frozen-materialization-v1",
        "split_view_manifest_path": str(split_manifest_path),
        "split_view_manifest_sha256": sha256_file(split_manifest_path),
        "dtu_inventory_provenance_path": str(dtu_inventory_provenance_path),
        "dtu_inventory_provenance_sha256": sha256_file(
            dtu_inventory_provenance_path
        ),
        "dependencies": {"typing_extensions": typing_dependency},
        "archives": {
            "Rectified.zip": _remote_archive_evidence(
                "Rectified.zip",
                str(resources["dtu_rectified_url"]),
                rectified,
                dtu_remote,
            ),
            "tartanair-image": _remote_archive_evidence(
                "tartanair-image",
                str(resources["tartanair_image_url"]),
                tartan_image,
                tartan_image_remote,
            ),
            "tartanair-depth": _remote_archive_evidence(
                "tartanair-depth",
                str(resources["tartanair_depth_url"]),
                tartan_depth,
                tartan_depth_remote,
            ),
            "SampleSet.zip": {
                "path": str(sample_path),
                "bytes": int(resources["dtu_sampleset_bytes"]),
                "sha256": sample_sha,
            },
            "Points.zip": {
                "path": str(points_path),
                "bytes": int(resources["dtu_points_bytes"]),
                "sha256": points_sha,
            },
        },
        "dtu": dtu_rows,
        "tartanair": {
            "source_commit": require_git_commit(
                resources.get("tartanair_hf_commit"), "TartanAir source commit"
            ),
            "environment": "GreatMarsh",
            "difficulty": "Data_easy",
            "trajectory": "P000",
            "camera": "lcam_front",
            "selection": "uniform-100-of-3537-v1",
            "pairs": tartan_pairs,
        },
    }
    output = root / "manifests" / "frozen_materialization.json"
    atomic_json(output, payload)
    return {
        "materialization_path": str(output),
        "materialization_sha256": sha256_file(output),
        "dtu_scene_count": len(dtu_rows),
        "dtu_rgb_count": len(dtu_members),
        "tartanair_pair_count": len(tartan_pairs),
    }


def verify_materialized_member(
    evidence: Mapping[str, Any], *, materialized_root: Path | None = None,
) -> None:
    try:
        path = Path(str(evidence["path"]))
        expected_sha = require_sha256(
            evidence["raw_sha256"], "materialized raw digest"
        )
        expected_size = int(evidence["uncompressed_size"])
        expected_crc = int(str(evidence["crc32"]), 16)
    except (KeyError, TypeError, ValueError) as exc:
        raise PreparationError("materialized member evidence is incomplete") from exc
    if materialized_root is not None:
        try:
            path.resolve().relative_to(materialized_root.resolve())
        except (OSError, ValueError) as exc:
            raise PreparationError(
                f"materialized member escapes the frozen data root: {path}"
            ) from exc
    if (
        not path.is_file()
        or path.stat().st_size != expected_size
        or sha256_file(path) != expected_sha
    ):
        raise PreparationError(
            f"materialized member digest/size mismatch: {path}"
        )
    crc = 0
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            crc = zlib.crc32(block, crc)
    if crc & 0xFFFFFFFF != expected_crc:
        raise PreparationError(f"materialized member CRC mismatch: {path}")


def verify_materialization_manifest(
    path: Path, *, split_manifest_path: Path, enforce_typing_extensions_home: bool = True
) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PreparationError("frozen materialization manifest is unreadable") from exc
    if payload.get("schema_version") != "frozen-materialization-v1":
        raise PreparationError("frozen materialization manifest schema mismatch")
    split_sha = sha256_file(split_manifest_path)
    if payload.get("split_view_manifest_sha256") != split_sha:
        raise PreparationError(
            "materialization is not bound to the canonical split/view manifest"
        )
    try:
        inventory_path = Path(str(payload["dtu_inventory_provenance_path"]))
        inventory_sha = require_sha256(
            payload["dtu_inventory_provenance_sha256"],
            "DTU inventory provenance digest",
        )
    except (KeyError, TypeError) as exc:
        raise PreparationError(
            "materialization is missing DTU inventory provenance"
        ) from exc
    if not inventory_path.is_file() or sha256_file(inventory_path) != inventory_sha:
        raise PreparationError("DTU inventory provenance digest mismatch")
    try:
        split_payload = json.loads(split_manifest_path.read_text(encoding="utf-8"))
        split_membership = {
            int(scene): split
            for split, scenes in split_payload["splits"].items()
            for scene in scenes
        }
        expected_views = {
            int(scene): tuple(map(int, views))
            for scene, views in split_payload["views"].items()
        }
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise PreparationError("canonical split/view manifest is unreadable") from exc
    if split_payload.get("schema_version") != "dtu-preparation-v1":
        raise PreparationError("canonical split/view manifest schema mismatch")
    materialized_root = path.parent.parent / "materialized"
    try:
        dependency = payload["dependencies"]["typing_extensions"]
        verify_typing_extensions_dependency(
            site=Path(str(dependency["site"])),
            expected_sha256=str(dependency["sha256"]),
            expected_version=str(dependency["version"]),
            enforce_home_prefix=enforce_typing_extensions_home,
        )
        tartan = payload["tartanair"]
        source_commit = require_git_commit(
            tartan["source_commit"], "TartanAir source commit"
        )
        image_url = str(payload["archives"]["tartanair-image"]["url"])
        depth_url = str(payload["archives"]["tartanair-depth"]["url"])
    except (KeyError, TypeError) as exc:
        raise PreparationError("TartanAir source or dependency provenance is incomplete") from exc
    if any(f"/resolve/{source_commit}/" not in url for url in (image_url, depth_url)):
        raise PreparationError(
            "TartanAir archive URLs are not bound to the frozen source commit"
        )
    dtu_rows = payload.get("dtu")
    pairs = payload.get("tartanair", {}).get("pairs")
    if not isinstance(dtu_rows, list) or not isinstance(pairs, list):
        raise PreparationError("materialization evidence is incomplete")
    if len(dtu_rows) != 45 or len(pairs) != 100:
        raise PreparationError(
            "materialization must contain 45 DTU scenes and 100 TartanAir pairs"
        )
    seen_frames: list[str] = []
    seen_scenes: set[int] = set()
    for row in dtu_rows:
        try:
            scene = int(row["scene_id"])
            split = str(row["split"])
            rgb = row["rgb"]
            cameras = row["cameras"]
        except (KeyError, TypeError, ValueError) as exc:
            raise PreparationError("DTU materialization row is incomplete") from exc
        if scene in seen_scenes or split_membership.get(scene) != split:
            raise PreparationError("DTU materialization contains duplicate or cross-split scene")
        if set(map(int, rgb)) != set(expected_views.get(scene, ())) or set(map(int, cameras)) != set(expected_views.get(scene, ())):
            raise PreparationError(
                f"materialized scan{scene} views do not match the frozen FPS manifest"
            )
        seen_scenes.add(scene)
        verify_materialized_member(row["points"], materialized_root=materialized_root)
        verify_materialized_member(row["mask"], materialized_root=materialized_root)
        if len(rgb) != 8 or len(cameras) != 8:
            raise PreparationError(
                f"materialized scan{scene} must contain eight views"
            )
        for member in rgb.values():
            verify_materialized_member(member, materialized_root=materialized_root)
        for view_text, member in cameras.items():
            view = int(view_text)
            normalized = str(member.get("member", "")).replace("\\", "/")
            expected = f"MVS Data/Calibration/cal18/pos_{view:03d}.txt"
            if not normalized.endswith(expected):
                raise PreparationError(
                    f"materialized DTU camera view {view} is not exact pos_{view:03d}.txt"
                )
            verify_materialized_member(member, materialized_root=materialized_root)
    if seen_scenes != set(split_membership):
        raise PreparationError("DTU materialization does not contain the exact frozen 45 scenes")
    for pair in pairs:
        frame = str(pair.get("frame_id"))
        seen_frames.append(frame)
        rgb_match = _TARTAN_IMAGE_RE.fullmatch(
            str(pair.get("rgb", {}).get("member", ""))
        )
        depth_match = _TARTAN_DEPTH_RE.fullmatch(
            str(pair.get("depth", {}).get("member", ""))
        )
        if (
            rgb_match is None
            or depth_match is None
            or rgb_match.group("frame") != frame
            or depth_match.group("frame") != frame
        ):
            raise PreparationError(
                "TartanAir materialization contains a cross-frame RGB/depth swap"
            )
        verify_materialized_member(pair["rgb"], materialized_root=materialized_root)
        verify_materialized_member(pair["depth"], materialized_root=materialized_root)
    if len(set(seen_frames)) != 100:
        raise PreparationError(
            "TartanAir materialization must contain 100 unique frame ids"
        )
    return payload


def _git_head(path: Path) -> str:
    try:
        result = subprocess.run(
            ["git", "-C", str(path), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            timeout=30,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise PreparationError(f"cannot verify upstream git identity: {path}") from exc
    return result.stdout.strip().lower()


def _env_python(env_path: Path) -> Path:
    for candidate in (
        env_path / "bin" / "python",
        env_path / "Scripts" / "python.exe",
    ):
        if candidate.is_file():
            return candidate
    raise PreparationError(f"frozen environment has no Python executable: {env_path}")


def _python_torch_versions(
    env_path: Path, *, cache_root: Path, typing_site: Path = FROZEN_TYPING_EXTENSIONS_SITE,
    typing_sha256: str = FROZEN_TYPING_EXTENSIONS_SHA256,
    typing_version: str = FROZEN_TYPING_EXTENSIONS_VERSION,
) -> tuple[str, str]:
    python = _env_python(env_path)
    cache_root.mkdir(parents=True, exist_ok=True)
    environment = dict(os.environ)
    environment.update(
        {
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONPYCACHEPREFIX": str(cache_root / "pycache"),
            "XDG_CACHE_HOME": str(cache_root),
            "TORCH_HOME": str(cache_root / "torch"),
        }
    )
    try:
        result = subprocess.run(
            [
                str(python),
                "-I",
                "-B",
                "-c",
                (
                    "import hashlib, platform, sys\n"
                    "from pathlib import Path\n"
                    "site=Path(sys.argv[1])\n"
                    "expected_sha=sys.argv[2]\n"
                    "expected_version=sys.argv[3]\n"
                    "sys.path.insert(0, str(site))\n"
                    "import typing_extensions\n"
                    "from importlib import metadata\n"
                    "actual=Path(typing_extensions.__file__).resolve()\n"
                    "assert actual == (site / 'typing_extensions.py').resolve()\n"
                    "assert hashlib.sha256(actual.read_bytes()).hexdigest() == expected_sha\n"
                    f"expected_dist_info = (site / '{FROZEN_TYPING_EXTENSIONS_DIST_INFO}').resolve()\n"
                    "assert metadata.version('typing_extensions') == expected_version\n"
                    "dist = metadata.distribution('typing_extensions')\n"
                    "dist_root = Path(dist.locate_file('')).resolve()\n"
                    f"dist_info = Path(dist.locate_file('{FROZEN_TYPING_EXTENSIONS_DIST_INFO}')).resolve()\n"
                    "assert dist_root == site.resolve(), f'typing_extensions distribution root escaped frozen site: {dist_root}'\n"
                    "assert dist_info == expected_dist_info and dist_info.is_dir(), f'typing_extensions dist-info escaped frozen site: {dist_info}'\n"
                    "assert dist.version == expected_version\n"
                    "import torch\n"
                    "print(platform.python_version())\n"
                    "print(torch.__version__)"
                ),
                str(typing_site),
                typing_sha256,
                typing_version,
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=60,
            env=environment,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise PreparationError(
            f"cannot verify Python/Torch versions in {env_path}"
        ) from exc
    lines = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    if len(lines) != 2:
        raise PreparationError(
            f"frozen environment emitted invalid version evidence: {env_path}"
        )
    return lines[0], lines[1]


def verify_frozen_overlay_identities(
    *,
    runtime: Mapping[str, Any],
    resources: Mapping[str, Any],
    cache_root: Path,
    enforce_typing_extensions_home: bool = True,
) -> dict[str, Any]:
    """Read and verify all source/checkpoint/environment identities."""

    source_specs = (
        ("vggt", "vggt_source", "vggt_source_commit"),
        ("mast3r", "mast3r_source", "mast3r_source_commit"),
        ("dust3r", "dust3r_source", "dust3r_source_commit"),
        ("croco", "croco_source", "croco_source_commit"),
    )
    sources: dict[str, Any] = {}
    for name, path_key, commit_key in source_specs:
        path = Path(str(runtime[path_key]))
        expected = require_git_commit(resources[commit_key], commit_key)
        actual = _git_head(path)
        if actual != expected:
            raise PreparationError(
                f"{name} source commit mismatch: {actual} != {expected}"
            )
        sources[name] = {"path": str(path), "commit": actual}

    files: dict[str, Any] = {}
    for path_key, digest_key in (
        ("vggt_checkpoint", "vggt_checkpoint_sha256"),
        ("mast3r_checkpoint", "mast3r_checkpoint_sha256"),
        ("mast3r_config", "mast3r_config_sha256"),
    ):
        path = Path(str(resources[path_key]))
        expected = require_sha256(resources[digest_key], digest_key)
        if not path.is_file():
            raise PreparationError(f"frozen resource file is missing: {path}")
        actual = sha256_file(path)
        if actual != expected:
            raise PreparationError(
                f"{path_key} SHA-256 mismatch: {actual} != {expected}"
            )
        files[path_key] = {"path": str(path), "sha256": actual}

    dependencies = {
        "typing_extensions": verify_typing_extensions_dependency(
            site=Path(str(runtime.get("typing_extensions_site", FROZEN_TYPING_EXTENSIONS_SITE))),
            expected_sha256=str(resources.get("typing_extensions_sha256", FROZEN_TYPING_EXTENSIONS_SHA256)),
            expected_version=str(resources.get("typing_extensions_version", FROZEN_TYPING_EXTENSIONS_VERSION)),
            enforce_home_prefix=enforce_typing_extensions_home,
        )
    }
    environments: dict[str, Any] = {}
    for name in ("vggt", "mast3r"):
        env_path = Path(str(runtime[f"{name}_env"]))
        python, torch = _python_torch_versions(
            env_path,
            cache_root=cache_root / name,
            typing_site=Path(str(runtime.get("typing_extensions_site", FROZEN_TYPING_EXTENSIONS_SITE))),
            typing_sha256=str(resources.get("typing_extensions_sha256", FROZEN_TYPING_EXTENSIONS_SHA256)),
            typing_version=str(resources.get("typing_extensions_version", FROZEN_TYPING_EXTENSIONS_VERSION)),
        )
        expected_python = str(runtime[f"{name}_python"])
        expected_torch = str(runtime[f"{name}_torch"])
        if python != expected_python or torch != expected_torch:
            raise PreparationError(
                f"{name} environment mismatch: Python {python}/Torch {torch} != "
                f"Python {expected_python}/Torch {expected_torch}"
            )
        environments[name] = {
            "path": str(env_path),
            "python": python,
            "torch": torch,
        }
    return {
        "schema_version": "frozen-runtime-identity-v1",
        "sources": sources,
        "files": files,
        "environments": environments,
        "dependencies": dependencies,
        "verification_python": sys.version.split()[0],
    }
