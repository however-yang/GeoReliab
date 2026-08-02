from __future__ import annotations

import json
from pathlib import Path

import pytest

from georeliab_mve.v4_attempt05_inputs import (
    Attempt05InputClosureError,
    _camera_path,
    _global_frozen_view_order,
    _gt_path,
    _load_corruption_calibration,
    _materialize_fog_members,
    _mask_path,
    _rgb_path,
    _scene_has_required_l3_assets,
    _verified_complete_scenes,
    _view_order_from_closure_manifest,
    _view_order_from_resource_schedule,
    build_attempt05_calibration_schedule,
    build_attempt05_v4_split_assignment,
    validate_attempt05_input_closure,
)
from georeliab_mve.v4_counterfactuals import TEST_SCENE_IDS


def _sha() -> str:
    return "a" * 64


def test_v4_split_uses_twenty_calibration_and_scene_disjoint() -> None:
    scenes = [*range(1, 78), *range(82, 129)]
    assignment = build_attempt05_v4_split_assignment(
        complete_scene_ids=scenes,
        source_root=Path.cwd(),
    )

    assert assignment.test == TEST_SCENE_IDS
    assert len(assignment.calibration) == 20
    assert len(assignment.dev) == 5
    assert len(assignment.reference) == 5
    assert set(assignment.calibration).isdisjoint(assignment.test)


def test_calibration_schedule_is_forty_l3_non_scientific_units() -> None:
    assignment = build_attempt05_v4_split_assignment(
        complete_scene_ids=[*range(1, 78), *range(82, 129)],
        source_root=Path.cwd(),
    )
    rows = build_attempt05_calibration_schedule(split_assignment=assignment)

    assert len(rows) == 40
    assert [row["model_id"] for row in rows[:20]] == ["VGGT"] * 20
    assert [row["model_id"] for row in rows[20:]] == ["MASt3R"] * 20
    assert {row["state_id"] for row in rows} == {"L3"}
    assert {row["scientific"] for row in rows} == {False}
    assert {row["excluded_from_ci"] for row in rows} == {True}
    assert all(
        row["scene_id"] not in TEST_SCENE_IDS
        and isinstance(row["calibration_unit_sha256"], str)
        and len(row["calibration_unit_sha256"]) == 64
        for row in rows
    )


def test_closure_manifest_requires_exact_960_non_l3_rows(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.jsonl"
    rows = []
    for scene_id in TEST_SCENE_IDS:
        for view_id in range(1, 9):
            for state_id in ("L1", "L2", "L4", "L5", "L6", "L7"):
                rows.append(
                    {
                        "scene_id": scene_id,
                        "view_id": view_id,
                        "illumination_id": state_id,
                    }
                )
    manifest.write_text("\n".join(json.dumps(row) for row in rows) + "\n")

    with pytest.raises(Attempt05InputClosureError, match="HASH_MISMATCH"):
        _view_order_from_closure_manifest(manifest)


def test_closure_manifest_rejects_l3_even_at_960_rows(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest = tmp_path / "manifest.jsonl"
    rows = []
    first = True
    for scene_id in TEST_SCENE_IDS:
        for view_id in range(1, 9):
            for state_id in ("L1", "L2", "L4", "L5", "L6", "L7"):
                rows.append(
                    {
                        "scene_id": scene_id,
                        "view_id": view_id,
                        "illumination_id": "L3" if first else state_id,
                    }
                )
                first = False
    manifest.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
    import georeliab_mve.v4_attempt05_inputs as inputs

    monkeypatch.setattr(inputs, "MANIFEST_FILE_SHA256", inputs._sha256_file(manifest))

    with pytest.raises(Attempt05InputClosureError, match="ILLUMINATION"):
        _view_order_from_closure_manifest(manifest)


def test_validate_attempt05_input_closure_rejects_metric_claims(tmp_path: Path) -> None:
    payload = {
        "schema_version": "georeliab-v4-attempt-05-input-closure-1.0",
        "status": "PASS",
        "attempt_id": "attempt-05",
        "scientific_result": "SCIENTIFIC_RESULT_AVAILABLE",
        "scientific_units": 400,
        "scientific_state_count": 200,
        "calibration_l3_units": 40,
        "rectified_non_l3_members": 960,
        "fog_png_members": 480,
        "max_model_execution_units": 440,
    }
    path = tmp_path / "closure.json"
    path.write_text(json.dumps(payload))

    with pytest.raises(Attempt05InputClosureError, match="MANIFEST_TAMPER"):
        validate_attempt05_input_closure(path)



def test_attempt05_uses_nested_materialized_dtu_paths() -> None:
    root = Path("/srv/private/smli/GeoReliab/materialized")

    assert _gt_path(root, 1) == root / "Points" / "Points" / "stl" / "stl001_total.ply"
    assert _mask_path(root, 1) == root / "SampleSet" / "SampleSet" / "MVS Data" / "ObsMask" / "ObsMask1_10.mat"
    assert _camera_path(root, 7) == root / "SampleSet" / "SampleSet" / "MVS Data" / "Calibration" / "cal18" / "pos_007.txt"
    assert _rgb_path(root, 1, 7, "L3") == root / "Rectified" / "scan1" / "rect_007_3_r5000.png"


def test_scene_completeness_requires_only_authoritative_eight_l3_views(tmp_path: Path) -> None:
    root = tmp_path / "materialized"
    for path in (
        _gt_path(root, 1),
        _mask_path(root, 1),
        *(_camera_path(root, view) for view in range(1, 9)),
        *(_rgb_path(root, 1, view, "L3") for view in range(1, 9)),
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"x")

    assert _scene_has_required_l3_assets(root, 1, tuple(range(1, 9)))
    assert not _scene_has_required_l3_assets(root, 1, (*range(1, 8), 49))


def test_resource_schedule_is_authoritative_for_ordered_views(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    schedule = tmp_path / "schedule.json"
    ordered = (8, 1, 7, 2, 6, 3, 5, 4)
    units = [
        {"scene_id": scene, "ordered_view_ids": list(ordered), "unit_id": f"{scene}-{index}"}
        for scene in TEST_SCENE_IDS
        for index in range(20)
    ]
    schedule.write_text(json.dumps({"units": units}), encoding="utf-8")
    import georeliab_mve.v4_attempt05_inputs as inputs

    monkeypatch.setattr(inputs, "SCHEDULE_FILE_SHA256", inputs._sha256_file(schedule))

    assert _view_order_from_resource_schedule(schedule) == {scene: ordered for scene in TEST_SCENE_IDS}


def test_calibration_reuses_frozen_global_view_order_without_all_49_cameras(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import georeliab_mve.v4_attempt05_inputs as inputs

    root = tmp_path / "materialized"
    ordered = (1, 39, 45, 49, 6, 16, 42, 27)
    observed_orders: dict[int, tuple[int, ...]] = {}

    def record_order(root: Path, scene_id: int, ordered_views: tuple[int, ...]) -> bool:
        observed_orders[scene_id] = tuple(ordered_views)
        return True

    monkeypatch.setattr(inputs, "_scene_has_required_l3_assets", record_order)

    authoritative = {scene: ordered for scene in TEST_SCENE_IDS}
    global_order = _global_frozen_view_order(authoritative)
    complete = _verified_complete_scenes(
        root,
        authoritative,
        authoritative,
        default_view_order=global_order,
    )

    assert global_order == ordered
    assert 82 in complete
    assert observed_orders[82] == ordered
    assert not _camera_path(root, 2).exists()


def test_frozen_global_view_order_rejects_scene_order_drift() -> None:
    with pytest.raises(Attempt05InputClosureError, match="VIEW_ORDER_DRIFT"):
        _global_frozen_view_order({1: tuple(range(1, 9)), 9: tuple(range(2, 10))})


def test_corruption_calibration_is_loaded_and_bound_by_sha(tmp_path: Path) -> None:
    path = tmp_path / "corruption_calibration.json"
    payload = {
        "d_ref": 12.5,
        "airlight": [0.1, 0.2, 0.3],
        "fog_betas": [0.01, 0.02, 0.03],
        "defocus_scales": [1.0, 2.0, 3.0],
        "implementation_version": "test-version",
    }
    path.write_text(json.dumps(payload), encoding="utf-8")

    calibration, digest = _load_corruption_calibration(path)

    assert calibration.d_ref == 12.5
    assert calibration.fog_betas == (0.01, 0.02, 0.03)
    assert len(digest) == 64


def test_fog_materialization_is_deterministic_and_atomic(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import georeliab_mve.v4_attempt05_inputs as inputs

    root = tmp_path / "materialized"
    fog_root = tmp_path / "fog"
    (_gt_path(root, 1)).parent.mkdir(parents=True, exist_ok=True)
    _gt_path(root, 1).write_bytes(b"ply")
    for view in range(1, 9):
        _rgb_path(root, 1, view, "L3").parent.mkdir(parents=True, exist_ok=True)
        _rgb_path(root, 1, view, "L3").write_bytes(b"rgb")
        _camera_path(root, view).parent.mkdir(parents=True, exist_ok=True)
        _camera_path(root, view).write_bytes(b"cam")

    monkeypatch.setattr(inputs, "TEST_SCENE_IDS", (1,))
    monkeypatch.setattr(inputs, "FOG_STATES", ("fog-s1",))
    monkeypatch.setattr(inputs, "_sha256_file", lambda path: "b" * 64)
    monkeypatch.setattr(inputs, "decode_srgb_png", lambda path, *, expected_size: (__import__("numpy").zeros((1, 1, 3)), {}))
    monkeypatch.setattr(inputs, "parse_dtu_binary_ply", lambda path: __import__("numpy").zeros((1, 3)))
    monkeypatch.setattr(inputs, "derive_dtu_depth", lambda points, camera_path, *, image_size: (__import__("numpy").ones((1, 1)), {}))
    monkeypatch.setattr(inputs, "fog_render", lambda image, depth, calibration, *, severity, gt_digest, raw_source_sha256: (image + severity, {}))
    monkeypatch.setattr(inputs, "deterministic_png", lambda image: b"deterministic-png")
    calibration = inputs.CorruptionCalibration(
        d_ref=1.0,
        airlight=(0.1, 0.1, 0.1),
        fog_betas=(0.1, 0.2, 0.3),
        defocus_scales=(1.0, 2.0, 3.0),
    )

    _materialize_fog_members(
        root,
        fog_root,
        {1: tuple(range(1, 9))},
        calibration,
        corruption_calibration_sha256="c" * 64,
    )

    out = fog_root / "scan1" / "fog-s1" / "rect_002_3_r5000.png"
    assert out.read_bytes() == b"deterministic-png"
    assert not out.with_name(out.name + ".partial").exists()


def test_fog_materialization_rejects_stale_existing_png(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import georeliab_mve.v4_attempt05_inputs as inputs

    root = tmp_path / "materialized"
    fog_root = tmp_path / "fog"
    (_gt_path(root, 1)).parent.mkdir(parents=True, exist_ok=True)
    _gt_path(root, 1).write_bytes(b"ply")
    for view in range(1, 9):
        _rgb_path(root, 1, view, "L3").parent.mkdir(parents=True, exist_ok=True)
        _rgb_path(root, 1, view, "L3").write_bytes(b"rgb")
        _camera_path(root, view).parent.mkdir(parents=True, exist_ok=True)
        _camera_path(root, view).write_bytes(b"cam")
    stale = fog_root / "scan1" / "fog-s1" / "rect_001_3_r5000.png"
    stale.parent.mkdir(parents=True, exist_ok=True)
    stale.write_bytes(b"stale-v1")

    monkeypatch.setattr(inputs, "TEST_SCENE_IDS", (1,))
    monkeypatch.setattr(inputs, "FOG_STATES", ("fog-s1",))
    monkeypatch.setattr(inputs, "_sha256_file", lambda path: "b" * 64)
    monkeypatch.setattr(inputs, "decode_srgb_png", lambda path, *, expected_size: (__import__("numpy").zeros((1, 1, 3)), {}))
    monkeypatch.setattr(inputs, "parse_dtu_binary_ply", lambda path: __import__("numpy").zeros((1, 3)))
    monkeypatch.setattr(inputs, "derive_dtu_depth", lambda points, camera_path, *, image_size: (__import__("numpy").ones((1, 1)), {}))
    monkeypatch.setattr(inputs, "fog_render", lambda image, depth, calibration, *, severity, gt_digest, raw_source_sha256: (image + severity, {}))
    monkeypatch.setattr(inputs, "deterministic_png", lambda image: b"deterministic-png")
    calibration = inputs.CorruptionCalibration(
        d_ref=1.0,
        airlight=(0.1, 0.1, 0.1),
        fog_betas=(0.1, 0.2, 0.3),
        defocus_scales=(1.0, 2.0, 3.0),
    )

    with pytest.raises(Attempt05InputClosureError, match="V4_MVE_BLOCKED_INPUT_FOG_DIGEST_MISMATCH"):
        _materialize_fog_members(
            root,
            fog_root,
            {1: tuple(range(1, 9))},
            calibration,
            corruption_calibration_sha256="c" * 64,
        )
    assert stale.read_bytes() == b"stale-v1"
    assert not stale.with_name(stale.name + ".partial").exists()



def test_fog_materialization_reuses_exact_existing_png_with_same_binding(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import georeliab_mve.v4_attempt05_inputs as inputs

    root = tmp_path / "materialized"
    fog_root = tmp_path / "fog"
    (_gt_path(root, 1)).parent.mkdir(parents=True, exist_ok=True)
    _gt_path(root, 1).write_bytes(b"ply")
    for view in range(1, 9):
        _rgb_path(root, 1, view, "L3").parent.mkdir(parents=True, exist_ok=True)
        _rgb_path(root, 1, view, "L3").write_bytes(b"rgb")
        _camera_path(root, view).parent.mkdir(parents=True, exist_ok=True)
        _camera_path(root, view).write_bytes(b"cam")

    monkeypatch.setattr(inputs, "TEST_SCENE_IDS", (1,))
    monkeypatch.setattr(inputs, "FOG_STATES", ("fog-s1",))
    monkeypatch.setattr(inputs, "_sha256_file", lambda path: "b" * 64)
    monkeypatch.setattr(inputs, "decode_srgb_png", lambda path, *, expected_size: (__import__("numpy").zeros((1, 1, 3)), {}))
    monkeypatch.setattr(inputs, "parse_dtu_binary_ply", lambda path: __import__("numpy").zeros((1, 3)))
    monkeypatch.setattr(inputs, "derive_dtu_depth", lambda points, camera_path, *, image_size: (__import__("numpy").ones((1, 1)), {}))
    monkeypatch.setattr(inputs, "fog_render", lambda image, depth, calibration, *, severity, gt_digest, raw_source_sha256: (image + severity, {}))
    monkeypatch.setattr(inputs, "deterministic_png", lambda image: b"deterministic-png")
    calibration = inputs.CorruptionCalibration(
        d_ref=1.0,
        airlight=(0.1, 0.1, 0.1),
        fog_betas=(0.1, 0.2, 0.3),
        defocus_scales=(1.0, 2.0, 3.0),
    )

    first = _materialize_fog_members(
        root,
        fog_root,
        {1: tuple(range(1, 9))},
        calibration,
        corruption_calibration_sha256="c" * 64,
    )
    second = _materialize_fog_members(
        root,
        fog_root,
        {1: tuple(range(1, 9))},
        calibration,
        corruption_calibration_sha256="c" * 64,
    )

    assert {record["materialization_status"] for record in first.records} == {"created"}
    assert {record["materialization_status"] for record in second.records} == {"reused"}
    assert [record["fog_binding_sha256"] for record in first.records] == [
        record["fog_binding_sha256"] for record in second.records
    ]
    for record in second.records:
        assert record["corruption_calibration_sha256"] == "c" * 64
        assert record["fog_recipe_sha256"]


def test_fog_materialization_removes_created_png_after_failed_closure(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import georeliab_mve.v4_attempt05_inputs as inputs

    root = tmp_path / "materialized"
    fog_root = tmp_path / "fog"
    (_gt_path(root, 1)).parent.mkdir(parents=True, exist_ok=True)
    _gt_path(root, 1).write_bytes(b"ply")
    for view in range(1, 9):
        _rgb_path(root, 1, view, "L3").parent.mkdir(parents=True, exist_ok=True)
        _rgb_path(root, 1, view, "L3").write_bytes(b"rgb")
        _camera_path(root, view).parent.mkdir(parents=True, exist_ok=True)
        _camera_path(root, view).write_bytes(b"cam")

    monkeypatch.setattr(inputs, "TEST_SCENE_IDS", (1,))
    monkeypatch.setattr(inputs, "FOG_STATES", ("fog-s1",))
    monkeypatch.setattr(inputs, "_sha256_file", lambda path: "b" * 64)
    monkeypatch.setattr(inputs, "decode_srgb_png", lambda path, *, expected_size: (__import__("numpy").zeros((1, 1, 3)), {}))
    monkeypatch.setattr(inputs, "parse_dtu_binary_ply", lambda path: __import__("numpy").zeros((1, 3)))
    monkeypatch.setattr(inputs, "derive_dtu_depth", lambda points, camera_path, *, image_size: (__import__("numpy").ones((1, 1)), {}))
    monkeypatch.setattr(inputs, "deterministic_png", lambda image: b"deterministic-png")

    calls = {"count": 0}

    def render_or_fail(image, depth, calibration, *, severity, gt_digest, raw_source_sha256):
        calls["count"] += 1
        if calls["count"] == 2:
            raise RuntimeError("simulated closure failure")
        return image + severity, {}

    monkeypatch.setattr(inputs, "fog_render", render_or_fail)
    calibration = inputs.CorruptionCalibration(
        d_ref=1.0,
        airlight=(0.1, 0.1, 0.1),
        fog_betas=(0.1, 0.2, 0.3),
        defocus_scales=(1.0, 2.0, 3.0),
    )

    with pytest.raises(RuntimeError, match="simulated closure failure"):
        _materialize_fog_members(
            root,
            fog_root,
            {1: tuple(range(1, 9))},
            calibration,
            corruption_calibration_sha256="c" * 64,
        )

    first = fog_root / "scan1" / "fog-s1" / "rect_001_3_r5000.png"
    assert not first.exists()
    assert not first.with_name(first.name + ".partial").exists()
