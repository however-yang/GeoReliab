from __future__ import annotations

from pathlib import Path
import hashlib
import json

import numpy as np
import pytest

from georeliab_mve.prepared_inputs import (
    decode_dtu_assets,
    decode_srgb_png,
    decode_tartanair_depth_png,
    derive_dtu_depth,
    parse_dtu_binary_ply,
    write_prepared_inputs,
)
from georeliab_mve.preparation import PreparationError
from georeliab_mve.preparation_round2 import load_prepared_batch


def _save_png(path: Path, pixels: np.ndarray, mode: str) -> None:
    from PIL import Image

    Image.fromarray(pixels, mode=mode).save(path, format="PNG")


def _write_binary_ply(path: Path, xyz: np.ndarray) -> None:
    rows = np.zeros(
        len(xyz),
        dtype=np.dtype(
            [
                ("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
                ("nx", "<f4"), ("ny", "<f4"), ("nz", "<f4"),
                ("red", "u1"), ("green", "u1"), ("blue", "u1"),
            ]
        ),
    )
    rows["x"], rows["y"], rows["z"] = xyz.T
    header = (
        "ply\r\nformat binary_little_endian 1.0\r\n"
        f"element vertex {len(rows)}\r\n"
        "property float x\r\nproperty float y\r\nproperty float z\r\n"
        "property float nx\r\nproperty float ny\r\nproperty float nz\r\n"
        "property uchar red\r\nproperty uchar green\r\nproperty uchar blue\r\n"
        "end_header\r\n"
    ).encode("ascii")
    path.write_bytes(header + rows.tobytes())


def test_realistic_rgb_png_uses_exact_inverse_srgb(tmp_path):
    path = tmp_path / "rgb.png"
    pixels = np.asarray([[[0, 128, 255], [10, 20, 30]]], dtype=np.uint8)
    _save_png(path, pixels, "RGB")

    decoded, metadata = decode_srgb_png(path, expected_size=(2, 1))
    encoded = pixels.astype(np.float64) / 255.0
    expected = np.where(
        encoded <= 0.04045,
        encoded / 12.92,
        ((encoded + 0.055) / 1.055) ** 2.4,
    ).astype("<f4")
    np.testing.assert_array_equal(decoded, expected)
    assert decoded.dtype == np.dtype("<f4")
    assert metadata["algorithm"] == "srgb8-exact-inverse-linear-f32-v1"


def test_tartanair_rgba_png_matches_opencv_bgra_little_endian_view(tmp_path):
    path = tmp_path / "depth.png"
    expected = np.asarray([[1.0, 12.5], [0.25, 12345.5]], dtype="<f4")
    bgra = expected.view(np.uint8).reshape(2, 2, 4)
    rgba = np.ascontiguousarray(bgra[..., [2, 1, 0, 3]])
    _save_png(path, rgba, "RGBA")

    decoded, metadata = decode_tartanair_depth_png(path, expected_size=(2, 2))
    np.testing.assert_array_equal(decoded, expected)
    assert metadata["channel_semantics"] == "Pillow RGBA -> OpenCV BGRA -> view(<f4)"
    assert metadata["validity_policy"] == "valid iff finite and > 0; finite positive sky values are preserved"


def test_binary_ply_projection_zbuffer_occlusion_and_nearest_hole_fill(tmp_path):
    ply = tmp_path / "points.ply"
    camera = tmp_path / "pos_001.txt"
    _write_binary_ply(
        ply,
        np.asarray([[1.0, 1.0, 1.0], [2.0, 2.0, 2.0]], dtype=np.float32),
    )
    camera.write_text("1 0 0 0\n0 1 0 0\n0 0 1 0\n", encoding="ascii")

    points = parse_dtu_binary_ply(ply)
    depth, metadata = derive_dtu_depth(
        points,
        camera,
        image_size=(4, 3),
    )
    np.testing.assert_array_equal(depth, np.ones((3, 4), dtype="<f4"))
    assert metadata["valid_projected_count"] == 2
    assert metadata["zbuffer_pixel_count"] == 1
    assert metadata["raw_projection_coverage"] == pytest.approx(1 / 12)
    assert metadata["hole_rule"] == "scipy-edt-nearest-valid-projective-z-v1"
    assert metadata["observability_mask_applied"] is False
    assert metadata["units"] == "DTU millimetres"


def _synthetic_frozen_schedule(root: Path) -> tuple[dict, dict]:
    split = {
        "schema_version": "dtu-preparation-v1",
        "splits": {
            "dev": list(range(1, 11)),
            "calibration": list(range(11, 21)),
            "reference-token": list(range(21, 26)),
            "test": list(range(26, 46)),
        },
        "views": {str(scene): list(range(1, 9)) for scene in range(1, 46)},
    }
    raw = root / "materialized" / "placeholder.bin"
    raw.parent.mkdir(parents=True)
    raw.write_bytes(b"official")
    digest = hashlib.sha256(raw.read_bytes()).hexdigest()
    dtu = []
    for split_name, scenes in split["splits"].items():
        for scene in scenes:
            points = {"path": str(raw), "member": f"Points/stl/stl{scene:03d}_total.ply", "raw_sha256": digest}
            mask = {"path": str(raw), "member": f"MVS Data/ObsMask/ObsMask{scene}_10.mat", "raw_sha256": digest}
            dtu.append({
                "scene_id": scene, "split": split_name,
                "points": points, "mask": mask,
                "rgb": {
                    str(view): {"path": str(raw), "member": f"Rectified/scan{scene}/rect_{view:03d}_3_r5000.png", "raw_sha256": digest}
                    for view in range(1, 9)
                },
                "cameras": {
                    str(view): {"path": str(raw), "member": f"MVS Data/Calibration/cal18/pos_{view:03d}.txt", "raw_sha256": digest}
                    for view in range(1, 9)
                },
            })
    tartan = {"pairs": [
        {
            "frame_id": f"{frame:06d}",
            "rgb": {"path": str(raw), "member": f"GreatMarsh/Data_easy/P000/image_lcam_front/{frame:06d}_lcam_front.png", "raw_sha256": digest},
            "depth": {"path": str(raw), "member": f"GreatMarsh/Data_easy/P000/depth_lcam_front/{frame:06d}_lcam_front_depth.png", "raw_sha256": digest},
        }
        for frame in range(100)
    ]}
    return split, {"schema_version": "frozen-materialization-v1", "dtu": dtu, "tartanair": tartan}


def test_production_writer_emits_exact_stage_manifests_and_resumes(monkeypatch, tmp_path):
    import georeliab_mve.prepared_inputs as prepared

    root = tmp_path / "data"
    manifests = root / "manifests"
    manifests.mkdir(parents=True)
    split, materialization = _synthetic_frozen_schedule(root)
    split_path = manifests / "split_view_manifest.json"
    materialization_path = manifests / "frozen_materialization.json"
    split_path.write_text(json.dumps(split), encoding="utf-8")
    materialization_path.write_text(json.dumps(materialization), encoding="utf-8")
    monkeypatch.setattr(prepared, "verify_materialization_manifest", lambda *_args, **_kwargs: materialization)

    def fake_dtu(assets, *, view_id, ply_cache, image_size):
        image = np.full((2, 3, 3), view_id / 10, dtype="<f4")
        depth = np.full((2, 3), view_id, dtype="<f4")
        return image, depth, {"algorithm": prepared.SRGB_DECODE_VERSION, "input_sha256": assets["rgb"]["raw_sha256"]}, {
            "algorithm": prepared.DTU_DEPTH_VERSION,
            "input_sha256": {name: assets[name]["raw_sha256"] for name in ("camera", "points", "mask")},
            "observability_mask_applied": False,
        }

    def fake_tartan(assets, *, image_size):
        image = np.full((2, 2, 3), 0.25, dtype="<f4")
        depth = np.full((2, 2), 2.0, dtype="<f4")
        return image, depth, {"algorithm": prepared.SRGB_DECODE_VERSION, "input_sha256": assets["rgb"]["raw_sha256"]}, {
            "algorithm": prepared.TARTAN_DEPTH_VERSION, "input_sha256": assets["depth"]["raw_sha256"]
        }

    monkeypatch.setattr(prepared, "decode_dtu_assets", fake_dtu)
    monkeypatch.setattr(prepared, "decode_tartanair_assets", fake_tartan)
    first = write_prepared_inputs(root)
    assert first["schedule_counts"] == {"calibration": 80, "smoke": 80, "test": 160, "tartanair": 100}
    assert set(first["prepared_manifest_paths"]) == {"calibration", "smoke", "test", "tartanair"}
    assert not (root / "prepared" / "render_inputs.json").exists()
    assert json.loads((root / "prepared" / "render_inputs_smoke.json").read_text())["split"] == "dev"
    assert json.loads((root / "prepared" / "render_inputs_test.json").read_text())["split"] == "test"
    writer = json.loads((manifests / "prepared_inputs_writer.json").read_text())
    assert writer["partial_leftovers"] == []
    assert "pillow" in writer["producer"]["dependencies"]
    second = write_prepared_inputs(root)
    assert second["written"] == 0 and second["reused"] == first["written"]

    output = next((root / "prepared" / "arrays" / "dtu").rglob("*.npy"))
    partial = output.with_suffix(output.suffix + ".partial")
    partial.write_bytes(b"interrupted")
    resumed = write_prepared_inputs(root)
    assert resumed["written"] == 0 and not partial.exists()
    output.write_bytes(b"tampered")
    with pytest.raises(PreparationError, match="tampered or recipe-incompatible"):
        write_prepared_inputs(root)


def _raw_asset(path: Path, member: str) -> dict[str, str]:
    return {
        "path": str(path), "member": member,
        "raw_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def test_loader_recomputes_official_bytes_and_rejects_self_rehashed_arrays_and_raw_tamper(
    monkeypatch, tmp_path,
):
    import georeliab_mve.materialization as materialization_module
    import georeliab_mve.prepared_inputs as prepared

    root = tmp_path / "data"
    manifests, prepared_root, raw_root = root / "manifests", root / "prepared", root / "materialized"
    manifests.mkdir(parents=True)
    prepared_root.mkdir()
    raw_root.mkdir()
    rgb_path = raw_root / "rect_001_3_r5000.png"
    camera_path = raw_root / "pos_001.txt"
    points_path = raw_root / "stl001_total.ply"
    mask_path = raw_root / "ObsMask1_10.mat"
    _save_png(rgb_path, np.asarray([[[0, 128, 255], [255, 64, 0]]], dtype=np.uint8), "RGB")
    camera_path.write_text("1 0 0 0\n0 1 0 0\n0 0 1 0\n", encoding="ascii")
    _write_binary_ply(points_path, np.asarray([[1.0, 0.0, 1.0]], dtype=np.float32))
    mask_path.write_bytes(b"official-3d-obsmask")
    assets = {
        "rgb": _raw_asset(rgb_path, "Rectified/scan1/rect_001_3_r5000.png"),
        "camera": _raw_asset(camera_path, "MVS Data/Calibration/cal18/pos_001.txt"),
        "points": _raw_asset(points_path, "Points/stl/stl001_total.ply"),
        "mask": _raw_asset(mask_path, "MVS Data/ObsMask/ObsMask1_10.mat"),
    }
    split_payload = {
        "schema_version": "dtu-preparation-v1",
        "splits": {"dev": [1]}, "views": {"1": [1]},
    }
    split_path = manifests / "split_view_manifest.json"
    split_path.write_text(json.dumps(split_payload), encoding="utf-8")
    materialization_path = manifests / "frozen_materialization.json"
    materialization_path.write_text("{}", encoding="utf-8")
    frozen = {
        "schema_version": "frozen-materialization-v1",
        "dtu": [{
            "scene_id": 1, "split": "dev",
            "rgb": {"1": assets["rgb"]}, "cameras": {"1": assets["camera"]},
            "points": assets["points"], "mask": assets["mask"],
        }],
        "tartanair": {"pairs": []},
    }
    monkeypatch.setattr(materialization_module, "verify_materialization_manifest", lambda *_args, **_kwargs: frozen)
    monkeypatch.setattr(prepared, "DTU_IMAGE_SIZE", (2, 1))
    image, depth, rgb_meta, depth_meta = decode_dtu_assets(
        assets, view_id=1, image_size=(2, 1),
    )
    rgb_npy = prepared_root / "arrays" / "dtu" / "dev" / "scan001" / "view001_linear_rgb.npy"
    depth_npy = prepared_root / "arrays" / "dtu" / "dev" / "scan001" / "view001_gt_depth.npy"
    rgb_npy.parent.mkdir(parents=True)
    np.save(rgb_npy, image, allow_pickle=False)
    np.save(depth_npy, depth, allow_pickle=False)
    rgb_sha = hashlib.sha256(rgb_npy.read_bytes()).hexdigest()
    depth_sha = hashlib.sha256(depth_npy.read_bytes()).hexdigest()
    rgb_meta["output_sha256"], depth_meta["output_sha256"] = rgb_sha, depth_sha
    row = {
        "scene_id": 1, "view_id": 1,
        "sample_key": "dtu/dev/scan1/view1/clean/0/0",
        "raw_rgb_path": str(rgb_path), "raw_source_sha256": assets["rgb"]["raw_sha256"],
        "linear_rgb_npy": str(rgb_npy), "linear_rgb_npy_sha256": rgb_sha,
        "depth_npy": str(depth_npy), "depth_npy_sha256": depth_sha,
        "gt_digest": depth_sha,
        "source_assets": {
            name: {"member": assets[name]["member"], "raw_sha256": assets[name]["raw_sha256"]}
            for name in ("rgb", "camera", "points", "mask")
        },
        "rgb_decode": rgb_meta, "depth_derivation": depth_meta,
    }
    payload = {
        "schema_version": "prepared-input-v2", "stage": "smoke", "split": "dev",
        "producer": prepared.implementation_evidence(),
        "split_view_manifest_path": str(split_path),
        "split_view_manifest_sha256": hashlib.sha256(split_path.read_bytes()).hexdigest(),
        "materialization_path": str(materialization_path),
        "materialization_sha256": hashlib.sha256(materialization_path.read_bytes()).hexdigest(),
        "record_count": 1, "records": [row],
    }
    prepared_path = prepared_root / "render_inputs_smoke.json"
    prepared_path.write_text(json.dumps(payload), encoding="utf-8")
    assert len(load_prepared_batch(prepared_path, expected_stage="smoke").records) == 1

    np.save(rgb_npy, np.zeros_like(image), allow_pickle=False)
    forged_sha = hashlib.sha256(rgb_npy.read_bytes()).hexdigest()
    payload["records"][0]["linear_rgb_npy_sha256"] = forged_sha
    payload["records"][0]["rgb_decode"]["output_sha256"] = forged_sha
    prepared_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(PreparationError, match="does not match deterministic decode"):
        load_prepared_batch(prepared_path, expected_stage="smoke")

    np.save(rgb_npy, image, allow_pickle=False)
    payload["records"][0]["linear_rgb_npy_sha256"] = rgb_sha
    payload["records"][0]["rgb_decode"]["output_sha256"] = rgb_sha
    prepared_path.write_text(json.dumps(payload), encoding="utf-8")
    camera2 = raw_root / "pos_002.txt"
    camera2.write_bytes(camera_path.read_bytes())
    swapped = _raw_asset(camera2, "MVS Data/Calibration/cal18/pos_002.txt")
    frozen["dtu"][0]["cameras"]["1"] = swapped
    payload["records"][0]["source_assets"]["camera"] = {
        "member": swapped["member"], "raw_sha256": swapped["raw_sha256"],
    }
    prepared_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(PreparationError, match="not exact pos_001"):
        load_prepared_batch(prepared_path, expected_stage="smoke")

    frozen["dtu"][0]["cameras"]["1"] = assets["camera"]
    payload["records"][0]["source_assets"]["camera"] = {
        "member": assets["camera"]["member"], "raw_sha256": assets["camera"]["raw_sha256"],
    }
    prepared_path.write_text(json.dumps(payload), encoding="utf-8")
    for name, raw_path in (("rgb", rgb_path), ("points", points_path),
                           ("camera", camera_path), ("mask", mask_path)):
        original = raw_path.read_bytes()
        raw_path.write_bytes(original + b"tamper")
        with pytest.raises(PreparationError, match="(digest|raw bytes)"):
            load_prepared_batch(prepared_path, expected_stage="smoke")
        raw_path.write_bytes(original)


def test_tartanair_loader_recomputes_100_official_pairs_and_rejects_depth_tamper(
    monkeypatch, tmp_path,
):
    import georeliab_mve.materialization as materialization_module
    import georeliab_mve.prepared_inputs as prepared
    from georeliab_mve.preparation_round2 import load_tartanair_prepared_pairs

    root = tmp_path / "data"
    manifests, prepared_root, raw_root = root / "manifests", root / "prepared", root / "materialized"
    manifests.mkdir(parents=True)
    prepared_root.mkdir()
    raw_root.mkdir()
    split_path = manifests / "split_view_manifest.json"
    split_path.write_text('{"schema_version":"dtu-preparation-v1"}', encoding="utf-8")
    materialization_path = manifests / "frozen_materialization.json"
    materialization_path.write_text(json.dumps({
        "schema_version": "frozen-materialization-v1",
        "split_view_manifest_path": str(split_path),
    }), encoding="utf-8")
    pairs, records = [], []
    monkeypatch.setattr(prepared, "TARTAN_IMAGE_SIZE", (2, 2))
    for frame_index in range(100):
        frame = f"{frame_index:06d}"
        rgb_raw = raw_root / f"{frame}_rgb.png"
        depth_raw = raw_root / f"{frame}_depth.png"
        rgb_pixels = np.full((2, 2, 3), frame_index % 256, dtype=np.uint8)
        expected_depth = np.full((2, 2), frame_index + 1.0, dtype="<f4")
        bgra = expected_depth.view(np.uint8).reshape(2, 2, 4)
        rgba = np.ascontiguousarray(bgra[..., [2, 1, 0, 3]])
        _save_png(rgb_raw, rgb_pixels, "RGB")
        _save_png(depth_raw, rgba, "RGBA")
        pair = {
            "frame_id": frame,
            "rgb": _raw_asset(rgb_raw, f"GreatMarsh/Data_easy/P000/image_lcam_front/{frame}_lcam_front.png"),
            "depth": _raw_asset(depth_raw, f"GreatMarsh/Data_easy/P000/depth_lcam_front/{frame}_lcam_front_depth.png"),
        }
        pairs.append(pair)
        image, depth, rgb_meta, depth_meta = prepared.decode_tartanair_assets(
            pair, image_size=(2, 2),
        )
        array_root = prepared_root / "arrays" / "tartanair" / "P000"
        array_root.mkdir(parents=True, exist_ok=True)
        rgb_npy, depth_npy = array_root / f"{frame}_linear_rgb.npy", array_root / f"{frame}_depth.npy"
        np.save(rgb_npy, image, allow_pickle=False)
        np.save(depth_npy, depth, allow_pickle=False)
        rgb_sha = hashlib.sha256(rgb_npy.read_bytes()).hexdigest()
        depth_sha = hashlib.sha256(depth_npy.read_bytes()).hexdigest()
        rgb_meta["output_sha256"], depth_meta["output_sha256"] = rgb_sha, depth_sha
        records.append({
            "frame_id": frame,
            "raw_rgb_path": str(rgb_raw), "raw_depth_path": str(depth_raw),
            "rgb_npy": str(rgb_npy), "rgb_npy_sha256": rgb_sha,
            "depth_npy": str(depth_npy), "depth_npy_sha256": depth_sha,
            "source_assets": {
                name: {"member": pair[name]["member"], "raw_sha256": pair[name]["raw_sha256"]}
                for name in ("rgb", "depth")
            },
            "rgb_decode": rgb_meta, "depth_decode": depth_meta,
        })
    frozen = {"schema_version": "frozen-materialization-v1", "tartanair": {"pairs": pairs}}
    monkeypatch.setattr(materialization_module, "verify_materialization_manifest", lambda *_args, **_kwargs: frozen)
    payload = {
        "schema_version": "tartanair-prepared-v2",
        "producer": prepared.implementation_evidence(),
        "materialization_path": str(materialization_path),
        "materialization_sha256": hashlib.sha256(materialization_path.read_bytes()).hexdigest(),
        "record_count": 100, "records": records,
    }
    prepared_path = prepared_root / "tartanair_p000_pairs.json"
    prepared_path.write_text(json.dumps(payload), encoding="utf-8")
    assert len(load_tartanair_prepared_pairs(prepared_path)) == 100

    first_depth_raw = Path(records[0]["raw_depth_path"])
    original = first_depth_raw.read_bytes()
    first_depth_raw.write_bytes(original + b"tamper")
    with pytest.raises(PreparationError, match="(raw bytes|digest|tampered)"):
        load_tartanair_prepared_pairs(prepared_path)
    first_depth_raw.write_bytes(original)

    first_depth_npy = Path(records[0]["depth_npy"])
    np.save(first_depth_npy, np.zeros((2, 2), dtype="<f4"), allow_pickle=False)
    forged = hashlib.sha256(first_depth_npy.read_bytes()).hexdigest()
    payload["records"][0]["depth_npy_sha256"] = forged
    payload["records"][0]["depth_decode"]["output_sha256"] = forged
    prepared_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(PreparationError, match="does not match official BGRA"):
        load_tartanair_prepared_pairs(prepared_path)
