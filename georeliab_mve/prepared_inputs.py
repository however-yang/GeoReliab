"""Deterministic production writers for frozen DTU and TartanAir inputs."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from io import BytesIO
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import sys
from typing import Any

import numpy as np

from .materialization import sha256_file, verify_materialization_manifest
from .preparation import PreparationError


WRITER_VERSION = "georeliab-prepared-writer-v1"
SRGB_DECODE_VERSION = "srgb8-exact-inverse-linear-f32-v1"
DTU_DEPTH_VERSION = "dtu-projective-zbuffer-edt-fill-v1"
TARTAN_DEPTH_VERSION = "tartanair-opencv-bgra-view-le-f32-v1"
DTU_IMAGE_SIZE = (1600, 1200)
TARTAN_IMAGE_SIZE = (640, 640)
_VERTEX_PROPERTIES = (
    ("float", "x"), ("float", "y"), ("float", "z"),
    ("float", "nx"), ("float", "ny"), ("float", "nz"),
    ("uchar", "red"), ("uchar", "green"), ("uchar", "blue"),
)
_PLY_DTYPE = np.dtype([
    ("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
    ("nx", "<f4"), ("ny", "<f4"), ("nz", "<f4"),
    ("red", "u1"), ("green", "u1"), ("blue", "u1"),
])
_STAGE_SPLIT = {"calibration": "calibration", "smoke": "dev", "test": "test"}
_STAGE_FILENAMES = {
    "calibration": "calibration_inputs.json",
    "smoke": "render_inputs_smoke.json",
    "test": "render_inputs_test.json",
}


def _sha_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _require_sha(value: object, label: str) -> str:
    if (not isinstance(value, str) or len(value) != 64
            or any(char not in "0123456789abcdef" for char in value)):
        raise PreparationError(f"{label} must be a lowercase SHA-256")
    return value


def _load_png(path: Path, *, mode: str, expected_size: tuple[int, int]) -> np.ndarray:
    try:
        from PIL import Image
    except ImportError as exc:  # pragma: no cover
        raise PreparationError("Pillow is required to decode official PNG inputs") from exc
    try:
        with Image.open(path) as image:
            if image.format != "PNG" or image.mode != mode or image.size != expected_size:
                raise PreparationError(
                    f"official PNG must be {expected_size[0]}x{expected_size[1]} {mode}: "
                    f"{path} (format={image.format}, mode={image.mode}, size={image.size})"
                )
            image.load()
            pixels = np.asarray(image, dtype=np.uint8)
    except (OSError, ValueError) as exc:
        raise PreparationError(f"official PNG is unreadable: {path}") from exc
    expected_shape = (expected_size[1], expected_size[0], len(mode))
    if pixels.shape != expected_shape or pixels.dtype != np.uint8:
        raise PreparationError(f"official PNG decoded to an unexpected layout: {path}")
    return pixels


def decode_srgb_png(path: Path, *, expected_size: tuple[int, int]) -> tuple[np.ndarray, dict[str, Any]]:
    """Decode an 8-bit RGB PNG with the exact IEC sRGB inverse transfer."""
    pixels = _load_png(path, mode="RGB", expected_size=expected_size)
    encoded = pixels.astype(np.float64) / 255.0
    linear = np.where(encoded <= 0.04045, encoded / 12.92,
                      ((encoded + 0.055) / 1.055) ** 2.4).astype("<f4")
    return linear, {
        "algorithm": SRGB_DECODE_VERSION,
        "input_encoding": "8-bit IEC sRGB PNG",
        "output_dtype": "little-endian float32",
        "output_shape": list(linear.shape),
    }


def decode_tartanair_depth_png(
    path: Path, *, expected_size: tuple[int, int] = TARTAN_IMAGE_SIZE,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Reproduce cv2 IMREAD_UNCHANGED BGRA followed by ``view('<f4')``."""
    rgba = _load_png(path, mode="RGBA", expected_size=expected_size)
    bgra = np.ascontiguousarray(rgba[..., [2, 1, 0, 3]], dtype=np.uint8)
    depth = bgra.view("<f4").reshape(expected_size[1], expected_size[0]).copy()
    valid = np.isfinite(depth) & (depth > 0)
    return depth, {
        "algorithm": TARTAN_DEPTH_VERSION,
        "channel_semantics": "Pillow RGBA -> OpenCV BGRA -> view(<f4)",
        "output_dtype": "little-endian float32",
        "output_shape": list(depth.shape),
        "validity_policy": "valid iff finite and > 0; finite positive sky values are preserved",
        "sanity_patch_policy": "a 32x32 patch is used only when every depth pixel is finite and > 0",
        "finite_positive_count": int(valid.sum()),
        "invalid_count": int(depth.size - valid.sum()),
    }


def parse_dtu_projection(path: Path) -> np.ndarray:
    try:
        tokens = path.read_text(encoding="ascii").split()
        if len(tokens) != 12:
            raise ValueError("token count")
        matrix = np.asarray([float(token) for token in tokens], dtype=np.float64).reshape(3, 4)
    except (OSError, UnicodeError, ValueError) as exc:
        raise PreparationError(f"DTU camera is not an ASCII 3x4 projection: {path}") from exc
    if not np.isfinite(matrix).all() or np.linalg.matrix_rank(matrix[:, :3]) != 3:
        raise PreparationError(f"DTU camera projection is nonfinite or singular: {path}")
    return matrix


def _parse_ply_header(path: Path) -> tuple[int, int]:
    try:
        with path.open("rb") as handle:
            lines: list[str] = []
            for _ in range(256):
                raw = handle.readline()
                if not raw:
                    raise PreparationError(f"DTU PLY has no end_header: {path}")
                line = raw.decode("ascii").rstrip("\r\n")
                lines.append(line)
                if line == "end_header":
                    break
            else:
                raise PreparationError(f"DTU PLY header exceeds 256 lines: {path}")
            offset = handle.tell()
    except (OSError, UnicodeDecodeError) as exc:
        raise PreparationError(f"DTU PLY header is unreadable: {path}") from exc
    if len(lines) < 4 or lines[0] != "ply" or lines[1] != "format binary_little_endian 1.0":
        raise PreparationError(f"DTU PLY must be binary_little_endian 1.0: {path}")
    vertex_count: int | None = None
    current_element: str | None = None
    vertex_properties: list[tuple[str, str]] = []
    for line in lines[2:-1]:
        if not line or line.startswith("comment ") or line.startswith("obj_info "):
            continue
        tokens = line.split()
        if tokens[:1] == ["element"] and len(tokens) == 3:
            try:
                count = int(tokens[2])
            except ValueError as exc:
                raise PreparationError(f"DTU PLY has an invalid element count: {path}") from exc
            if count < 0:
                raise PreparationError(f"DTU PLY has a negative element count: {path}")
            current_element = tokens[1]
            if current_element == "vertex":
                if vertex_count is not None:
                    raise PreparationError(f"DTU PLY has duplicate vertex elements: {path}")
                vertex_count = count
            elif count != 0:
                raise PreparationError(f"DTU PLY has unsupported nonempty element {current_element}: {path}")
        elif tokens[:1] == ["property"] and len(tokens) == 3 and current_element == "vertex":
            vertex_properties.append((tokens[1], tokens[2]))
        else:
            raise PreparationError(f"DTU PLY has unsupported header syntax: {line}")
    if vertex_count is None or vertex_count <= 0:
        raise PreparationError(f"DTU PLY requires a positive vertex count: {path}")
    if tuple(vertex_properties) != _VERTEX_PROPERTIES:
        raise PreparationError("DTU PLY vertex layout must be x,y,z,nx,ny,nz float plus RGB uchar")
    expected_bytes = offset + vertex_count * _PLY_DTYPE.itemsize
    if path.stat().st_size != expected_bytes:
        raise PreparationError("DTU PLY vertex count/payload length mismatch")
    return vertex_count, offset


def parse_dtu_binary_ply(path: Path) -> np.ndarray:
    """Return official structured-light XYZ as one bounded float32 copy."""
    count, offset = _parse_ply_header(path)
    try:
        rows = np.memmap(path, dtype=_PLY_DTYPE, mode="r", offset=offset, shape=(count,))
        xyz = np.empty((count, 3), dtype="<f4")
        xyz[:, 0], xyz[:, 1], xyz[:, 2] = rows["x"], rows["y"], rows["z"]
        normals_finite = (np.isfinite(rows["nx"]).all()
                          and np.isfinite(rows["ny"]).all()
                          and np.isfinite(rows["nz"]).all())
    except (OSError, ValueError) as exc:
        raise PreparationError(f"DTU PLY binary payload is unreadable: {path}") from exc
    if not np.isfinite(xyz).all() or not normals_finite:
        raise PreparationError(f"DTU PLY contains nonfinite vertex data: {path}")
    return xyz


def derive_dtu_depth(
    points_xyz: np.ndarray,
    camera_path: Path,
    *,
    image_size: tuple[int, int] = DTU_IMAGE_SIZE,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Project XYZ, z-buffer by q2, then nearest-fill holes/background.

    The DTU ObsMask is a 3-D evaluation volume and is deliberately not applied
    as a 2-D raster mask. Its raw digest remains bound in record provenance.
    """
    xyz = np.asarray(points_xyz)
    if xyz.ndim != 2 or xyz.shape[1] != 3 or xyz.size == 0 or not np.isfinite(xyz).all():
        raise PreparationError("DTU projection requires nonempty finite Nx3 points")
    width, height = image_size
    if width <= 0 or height <= 0:
        raise PreparationError("DTU projection image size must be positive")
    matrix = parse_dtu_projection(camera_path)
    zbuffer = np.full(height * width, np.inf, dtype=np.float64)
    valid_projected = 0
    positive_projective = 0
    for start in range(0, len(xyz), 250_000):
        chunk = np.asarray(xyz[start:start + 250_000], dtype=np.float64)
        homogeneous = np.empty((len(chunk), 4), dtype=np.float64)
        homogeneous[:, :3] = chunk
        homogeneous[:, 3] = 1.0
        projected = homogeneous @ matrix.T
        z = projected[:, 2]
        positive = np.isfinite(projected).all(axis=1) & (z > 0)
        positive_projective += int(positive.sum())
        if not positive.any():
            continue
        u = np.floor(projected[:, 0] / z + 0.5)
        v = np.floor(projected[:, 1] / z + 0.5)
        inside = positive & (u >= 0) & (u < width) & (v >= 0) & (v < height)
        valid_projected += int(inside.sum())
        if inside.any():
            pixel = v[inside].astype(np.int64) * width + u[inside].astype(np.int64)
            np.minimum.at(zbuffer, pixel, z[inside])
    raster = zbuffer.reshape(height, width)
    valid_pixels = np.isfinite(raster)
    unique_pixels = int(valid_pixels.sum())
    if unique_pixels == 0:
        raise PreparationError("DTU projection produced no in-frame positive depth")
    try:
        import scipy
        from scipy.ndimage import distance_transform_edt
    except ImportError as exc:  # pragma: no cover
        raise PreparationError("SciPy is required for frozen DTU depth hole filling") from exc
    if not valid_pixels.all():
        nearest = distance_transform_edt(
            ~valid_pixels, return_distances=False, return_indices=True,
        )
        raster = raster[tuple(nearest)]
    depth = np.asarray(raster, dtype="<f4")
    if (depth.shape != (height, width) or not np.isfinite(depth).all()
            or not (depth > 0).all()):
        raise PreparationError("DTU filled depth must be dense, finite, positive, and aligned")
    return depth, {
        "algorithm": DTU_DEPTH_VERSION,
        "projection_rule": "u,v=floor(q0/q2+0.5),floor(q1/q2+0.5); q2>0; z-buffer min q2",
        "hole_rule": "scipy-edt-nearest-valid-projective-z-v1",
        "scipy_version": scipy.__version__,
        "input_vertex_count": int(len(xyz)),
        "positive_projective_count": positive_projective,
        "valid_projected_count": valid_projected,
        "zbuffer_pixel_count": unique_pixels,
        "raw_projection_coverage": unique_pixels / float(width * height),
        "filled_pixel_count": int(width * height - unique_pixels),
        "output_shape": [height, width],
        "output_dtype": "little-endian float32",
        "units": "DTU millimetres",
        "observability_mask_applied": False,
        "observability_mask_role": "raw digest bound for Task3 audit masking; not a 2-D depth raster mask",
    }


@dataclass
class DtuPlyCache:
    """One-scene cache: enough for eight adjacent views, bounded in memory."""
    key: tuple[str, str] | None = None
    points: np.ndarray | None = None

    def load(self, evidence: Mapping[str, Any]) -> np.ndarray:
        try:
            path = Path(str(evidence["path"]))
            digest = _require_sha(evidence["raw_sha256"], "DTU PLY digest")
        except (KeyError, TypeError) as exc:
            raise PreparationError("DTU PLY evidence is incomplete") from exc
        current = (str(path), digest)
        if self.key == current and self.points is not None:
            return self.points
        if not path.is_file() or sha256_file(path) != digest:
            raise PreparationError("DTU PLY raw digest mismatch")
        self.key, self.points = current, parse_dtu_binary_ply(path)
        return self.points


def _require_camera_member(evidence: Mapping[str, Any], view_id: int) -> None:
    expected = f"MVS Data/Calibration/cal18/pos_{view_id:03d}.txt"
    member = str(evidence.get("member", "")).replace("\\", "/")
    if not member.endswith(expected):
        raise PreparationError(
            f"DTU view {view_id} camera provenance must reference exact pos_{view_id:03d}.txt"
        )


def decode_dtu_assets(
    assets: Mapping[str, Any], *, view_id: int,
    ply_cache: DtuPlyCache | None = None,
    image_size: tuple[int, int] = DTU_IMAGE_SIZE,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any], dict[str, Any]]:
    try:
        rgb_evidence, camera_evidence = assets["rgb"], assets["camera"]
        points_evidence, mask_evidence = assets["points"], assets["mask"]
        rgb_path = Path(str(rgb_evidence["path"]))
        camera_path = Path(str(camera_evidence["path"]))
    except (KeyError, TypeError) as exc:
        raise PreparationError("DTU prepared source assets are incomplete") from exc
    _require_camera_member(camera_evidence, view_id)
    for name in ("rgb", "camera", "points", "mask"):
        evidence = assets[name]
        path = Path(str(evidence.get("path", "")))
        digest = _require_sha(evidence.get("raw_sha256"), f"DTU {name} raw digest")
        if not path.is_file() or sha256_file(path) != digest:
            raise PreparationError(f"DTU {name} raw bytes fail digest verification")
    image, rgb_metadata = decode_srgb_png(rgb_path, expected_size=image_size)
    cache = ply_cache if ply_cache is not None else DtuPlyCache()
    points = cache.load(points_evidence)
    depth, depth_metadata = derive_dtu_depth(points, camera_path, image_size=image_size)
    if image.shape[:2] != depth.shape:
        raise PreparationError("DTU RGB/depth are not pixel-aligned")
    rgb_metadata.update({
        "input_sha256": rgb_evidence["raw_sha256"],
        "raw_width": image_size[0], "raw_height": image_size[1],
    })
    depth_metadata["input_sha256"] = {
        name: assets[name]["raw_sha256"] for name in ("camera", "points", "mask")
    }
    return image, depth, rgb_metadata, depth_metadata


def decode_tartanair_assets(
    assets: Mapping[str, Any], *, image_size: tuple[int, int] = TARTAN_IMAGE_SIZE,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any], dict[str, Any]]:
    try:
        rgb_evidence, depth_evidence = assets["rgb"], assets["depth"]
    except (KeyError, TypeError) as exc:
        raise PreparationError("TartanAir prepared source assets are incomplete") from exc
    for name, evidence in (("rgb", rgb_evidence), ("depth", depth_evidence)):
        path = Path(str(evidence.get("path", "")))
        digest = _require_sha(evidence.get("raw_sha256"), f"TartanAir {name} raw digest")
        if not path.is_file() or sha256_file(path) != digest:
            raise PreparationError(f"TartanAir {name} raw bytes fail digest verification")
    image, rgb_metadata = decode_srgb_png(Path(str(rgb_evidence["path"])), expected_size=image_size)
    depth, depth_metadata = decode_tartanair_depth_png(
        Path(str(depth_evidence["path"])), expected_size=image_size,
    )
    rgb_metadata["input_sha256"] = rgb_evidence["raw_sha256"]
    depth_metadata["input_sha256"] = depth_evidence["raw_sha256"]
    if image.shape[:2] != depth.shape:
        raise PreparationError("TartanAir RGB/depth are not pixel-aligned")
    return image, depth, rgb_metadata, depth_metadata


def npy_bytes(array: np.ndarray) -> bytes:
    stream = BytesIO()
    np.save(stream, np.asarray(array), allow_pickle=False)
    return stream.getvalue()


def implementation_evidence() -> dict[str, Any]:
    try:
        pillow = importlib.metadata.version("Pillow")
        scipy = importlib.metadata.version("scipy")
    except importlib.metadata.PackageNotFoundError as exc:
        raise PreparationError("Pillow and SciPy must exist in the preparation environment") from exc
    return {
        "writer_version": WRITER_VERSION,
        "source_module": "georeliab_mve.prepared_inputs",
        "source_sha256": sha256_file(Path(__file__)),
        "algorithms": {
            "rgb": SRGB_DECODE_VERSION,
            "dtu_depth": DTU_DEPTH_VERSION,
            "tartanair_depth": TARTAN_DEPTH_VERSION,
        },
        "dependencies": {
            "python": sys.version.split()[0],
            "numpy": np.__version__,
            "pillow": pillow,
            "scipy": scipy,
        },
    }


def _canonical_json_bytes(payload: Mapping[str, Any]) -> bytes:
    return (json.dumps(payload, sort_keys=True, indent=2) + "\n").encode("utf-8")


def _strict_atomic_bytes(path: Path, expected: bytes, *, label: str) -> str:
    """Reuse exact outputs, reject tampering, and recover expected partials."""
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(path.suffix + ".partial")
    if partial.exists():
        try:
            partial.unlink()
        except OSError as exc:
            raise PreparationError(f"cannot remove interrupted {label} partial: {partial}") from exc
    if path.exists():
        try:
            existing = path.read_bytes()
        except OSError as exc:
            raise PreparationError(f"existing {label} is unreadable: {path}") from exc
        if existing != expected:
            raise PreparationError(f"existing {label} is tampered or recipe-incompatible: {path}")
        return "reused"
    try:
        partial.write_bytes(expected)
        if partial.read_bytes() != expected:
            raise PreparationError(f"atomic {label} byte verification failed: {partial}")
        partial.replace(path)
    except OSError as exc:
        raise PreparationError(f"cannot atomically write {label}: {path}") from exc
    return "written"


def _strict_atomic_npy(path: Path, array: np.ndarray) -> tuple[str, str]:
    raw = npy_bytes(array)
    return _sha_bytes(raw), _strict_atomic_bytes(path, raw, label="prepared NPY")


def _strict_atomic_json(path: Path, payload: Mapping[str, Any]) -> tuple[str, str]:
    raw = _canonical_json_bytes(payload)
    return _sha_bytes(raw), _strict_atomic_bytes(path, raw, label="prepared JSON")


def _asset_row(evidence: Mapping[str, Any]) -> dict[str, str]:
    return {
        "member": str(evidence["member"]),
        "raw_sha256": _require_sha(evidence["raw_sha256"], "source asset digest"),
    }


def _dtu_materialized_map(payload: Mapping[str, Any]) -> dict[tuple[int, int], dict[str, Any]]:
    result: dict[tuple[int, int], dict[str, Any]] = {}
    for scene_row in payload.get("dtu", []):
        scene = int(scene_row["scene_id"])
        for view_text, rgb in scene_row["rgb"].items():
            view = int(view_text)
            key = (scene, view)
            if key in result:
                raise PreparationError("materialization has duplicate DTU scene/view")
            result[key] = {
                "split": scene_row["split"],
                "rgb": rgb,
                "camera": scene_row["cameras"][view_text],
                "points": scene_row["points"],
                "mask": scene_row["mask"],
            }
    return result


def _prepared_payload(
    *, root: Path, stage: str, split_payload: Mapping[str, Any],
    split_path: Path, split_sha: str, materialization_path: Path,
    materialization_sha: str,
    materialized: Mapping[tuple[int, int], Mapping[str, Any]],
    producer: Mapping[str, Any], dispositions: dict[str, int],
) -> dict[str, Any]:
    split = _STAGE_SPLIT[stage]
    cache = DtuPlyCache()
    rows: list[dict[str, Any]] = []
    for scene_value in split_payload["splits"][split]:
        scene = int(scene_value)
        for view_value in split_payload["views"][str(scene)]:
            view = int(view_value)
            assets = materialized.get((scene, view))
            if assets is None or assets.get("split") != split:
                raise PreparationError(f"prepared writer is missing frozen scan{scene}/view{view}")
            image, depth, rgb_meta, depth_meta = decode_dtu_assets(
                assets, view_id=view, ply_cache=cache, image_size=DTU_IMAGE_SIZE,
            )
            array_root = root / "prepared" / "arrays" / "dtu" / split / f"scan{scene:03d}"
            rgb_path = array_root / f"view{view:03d}_linear_rgb.npy"
            depth_path = array_root / f"view{view:03d}_gt_depth.npy"
            rgb_sha, rgb_disposition = _strict_atomic_npy(rgb_path, image)
            depth_sha, depth_disposition = _strict_atomic_npy(depth_path, depth)
            dispositions[rgb_disposition] += 1
            dispositions[depth_disposition] += 1
            rgb_meta["output_sha256"] = rgb_sha
            depth_meta["output_sha256"] = depth_sha
            rows.append({
                "scene_id": scene,
                "view_id": view,
                "sample_key": f"dtu/{split}/scan{scene}/view{view}/clean/0/0",
                "raw_rgb_path": str(assets["rgb"]["path"]),
                "raw_source_sha256": assets["rgb"]["raw_sha256"],
                "linear_rgb_npy": str(rgb_path),
                "linear_rgb_npy_sha256": rgb_sha,
                "depth_npy": str(depth_path),
                "depth_npy_sha256": depth_sha,
                "gt_digest": depth_sha,
                "source_assets": {
                    name: _asset_row(assets[name])
                    for name in ("rgb", "camera", "points", "mask")
                },
                "rgb_decode": rgb_meta,
                "depth_derivation": depth_meta,
            })
    return {
        "schema_version": "prepared-input-v2",
        "stage": stage,
        "split": split,
        "producer": dict(producer),
        "split_view_manifest_path": str(split_path),
        "split_view_manifest_sha256": split_sha,
        "materialization_path": str(materialization_path),
        "materialization_sha256": materialization_sha,
        "record_count": len(rows),
        "records": rows,
    }


def _tartanair_payload(
    *, root: Path, materialization: Mapping[str, Any],
    materialization_path: Path, materialization_sha: str,
    producer: Mapping[str, Any], dispositions: dict[str, int],
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for pair in materialization.get("tartanair", {}).get("pairs", []):
        frame = str(pair["frame_id"])
        image, depth, rgb_meta, depth_meta = decode_tartanair_assets(
            pair, image_size=TARTAN_IMAGE_SIZE,
        )
        array_root = root / "prepared" / "arrays" / "tartanair" / "P000"
        rgb_path = array_root / f"{frame}_linear_rgb.npy"
        depth_path = array_root / f"{frame}_depth.npy"
        rgb_sha, rgb_disposition = _strict_atomic_npy(rgb_path, image)
        depth_sha, depth_disposition = _strict_atomic_npy(depth_path, depth)
        dispositions[rgb_disposition] += 1
        dispositions[depth_disposition] += 1
        rgb_meta["output_sha256"] = rgb_sha
        depth_meta["output_sha256"] = depth_sha
        rows.append({
            "frame_id": frame,
            "raw_rgb_path": str(pair["rgb"]["path"]),
            "raw_depth_path": str(pair["depth"]["path"]),
            "rgb_npy": str(rgb_path), "rgb_npy_sha256": rgb_sha,
            "depth_npy": str(depth_path), "depth_npy_sha256": depth_sha,
            "source_assets": {name: _asset_row(pair[name]) for name in ("rgb", "depth")},
            "rgb_decode": rgb_meta, "depth_decode": depth_meta,
        })
    return {
        "schema_version": "tartanair-prepared-v2",
        "producer": dict(producer),
        "materialization_path": str(materialization_path),
        "materialization_sha256": materialization_sha,
        "record_count": len(rows), "records": rows,
    }


def _reject_home_write(root: Path) -> None:
    if os.name == "nt":
        return
    try:
        root.resolve().relative_to(Path("/home").resolve())
    except ValueError:
        return
    raise PreparationError("prepared writer refuses every output below /home")


def write_prepared_inputs(root: Path) -> dict[str, Any]:
    """Create every frozen DTU/Tartan prepared input from materialized bytes."""
    root = root.resolve()
    _reject_home_write(root)
    split_path = root / "manifests" / "split_view_manifest.json"
    materialization_path = root / "manifests" / "frozen_materialization.json"
    try:
        split_payload = json.loads(split_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PreparationError("prepared writer requires the frozen split/view manifest") from exc
    if split_payload.get("schema_version") != "dtu-preparation-v1":
        raise PreparationError("prepared writer split/view manifest schema mismatch")
    materialization = verify_materialization_manifest(
        materialization_path, split_manifest_path=split_path,
    )
    split_sha = sha256_file(split_path)
    materialization_sha = sha256_file(materialization_path)
    producer = implementation_evidence()
    materialized = _dtu_materialized_map(materialization)
    dispositions = {"written": 0, "reused": 0}
    prepared_paths: dict[str, str] = {}
    prepared_hashes: dict[str, str] = {}
    schedule_counts: dict[str, int] = {}
    expected_counts = {"calibration": 80, "smoke": 80, "test": 160}
    for stage in ("calibration", "smoke", "test"):
        payload = _prepared_payload(
            root=root, stage=stage, split_payload=split_payload,
            split_path=split_path, split_sha=split_sha,
            materialization_path=materialization_path,
            materialization_sha=materialization_sha,
            materialized=materialized, producer=producer,
            dispositions=dispositions,
        )
        expected = expected_counts[stage]
        if payload["record_count"] != expected:
            raise PreparationError(
                f"prepared writer {stage} schedule count {payload['record_count']} != {expected}"
            )
        path = root / "prepared" / _STAGE_FILENAMES[stage]
        digest, disposition = _strict_atomic_json(path, payload)
        dispositions[disposition] += 1
        prepared_paths[stage], prepared_hashes[stage] = str(path), digest
        schedule_counts[stage] = expected

    tartan_payload = _tartanair_payload(
        root=root, materialization=materialization,
        materialization_path=materialization_path,
        materialization_sha=materialization_sha,
        producer=producer, dispositions=dispositions,
    )
    if tartan_payload["record_count"] != 100:
        raise PreparationError("prepared writer TartanAir schedule count must be 100")
    tartan_path = root / "prepared" / "tartanair_p000_pairs.json"
    tartan_sha, tartan_disposition = _strict_atomic_json(tartan_path, tartan_payload)
    dispositions[tartan_disposition] += 1
    prepared_paths["tartanair"], prepared_hashes["tartanair"] = str(tartan_path), tartan_sha
    schedule_counts["tartanair"] = 100

    leftovers = sorted(str(path) for path in (root / "prepared").rglob("*.partial"))
    if leftovers:
        raise PreparationError(f"prepared writer left partial outputs: {leftovers}")
    writer_payload = {
        "schema_version": "prepared-writer-manifest-v1",
        "producer": producer,
        "split_view_manifest_path": str(split_path),
        "split_view_manifest_sha256": split_sha,
        "materialization_path": str(materialization_path),
        "materialization_sha256": materialization_sha,
        "operation_sequence": [
            "verify", "index", "manifests", "prepared", "calibration",
            "rendering --stage smoke", "rendering --stage test", "sanity",
        ],
        "schedule_counts": schedule_counts,
        "prepared_manifest_paths": prepared_paths,
        "prepared_manifest_sha256": prepared_hashes,
        "partial_leftovers": [],
    }
    writer_path = root / "manifests" / "prepared_inputs_writer.json"
    writer_sha, writer_disposition = _strict_atomic_json(writer_path, writer_payload)
    dispositions[writer_disposition] += 1
    return {
        "writer_manifest_path": str(writer_path),
        "writer_manifest_sha256": writer_sha,
        "schedule_counts": schedule_counts,
        "prepared_manifest_paths": prepared_paths,
        "prepared_manifest_sha256": prepared_hashes,
        **dispositions,
    }
