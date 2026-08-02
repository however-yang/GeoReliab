from __future__ import annotations

import json
from pathlib import Path

import pytest

from georeliab_mve.v4_attempt05_inputs import (
    Attempt05InputClosureError,
    _RectifiedArchiveBinding,
    _camera_path,
    _expected_calibration_l3_members,
    _global_frozen_view_order,
    _gt_path,
    _load_verified_dtu_inventory,
    _load_corruption_calibration,
    _materialize_calibration_l3_members,
    _materialize_fog_members,
    _mask_path,
    _rgb_path,
    _scene_has_required_l3_assets,
    _view_order_from_closure_manifest,
    _view_order_from_resource_schedule,
    build_attempt05_calibration_schedule,
    build_attempt05_v4_split_assignment,
    validate_attempt05_input_closure,
)
from georeliab_mve.tartanair_range import RemoteZipEntry, RemoteZipIndex
from georeliab_mve.v4_counterfactuals import DTU_OFFICIAL_SCENE_IDS, TEST_SCENE_IDS


def _sha() -> str:
    return "a" * 64


def _inventory_payload() -> dict[str, object]:
    scenes = []
    for scene_id in DTU_OFFICIAL_SCENE_IDS:
        scenes.append(
            {
                "scene_id": scene_id,
                "camera_centers": {
                    str(view_id): [float(view_id), 0.0, 1.0]
                    for view_id in range(1, 50)
                },
                "rgb_files": [
                    f"rect_{view_id:03d}_3_r5000.png"
                    for view_id in range(1, 50)
                ],
                "points_path": f"Points/stl/stl{scene_id:03d}_total.ply",
                "mask_path": f"SampleSet/MVS Data/ObsMask/ObsMask{scene_id}_10.mat",
            }
        )
    return {"schema_version": "dtu-official-inventory-v1", "scenes": scenes}


def _write_inventory(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    import georeliab_mve.v4_attempt05_inputs as inputs

    path = tmp_path / "dtu_inventory.json"
    path.write_text(json.dumps(_inventory_payload()), encoding="utf-8")
    monkeypatch.setattr(inputs, "DTU_INVENTORY_FILE_SHA256", inputs._sha256_file(path))
    return path


def _sha256_file_for_test(path: Path) -> str:
    import hashlib

    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_frozen_inventory_drives_v4_split_without_local_rgb_presence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    inventory = _load_verified_dtu_inventory(_write_inventory(tmp_path, monkeypatch))
    assignment = build_attempt05_v4_split_assignment(
        inventory=inventory.rows,
        source_root=Path.cwd(),
    )

    assert len(inventory.rows) == 124
    assert assignment.calibration == (
        82, 52, 74, 73, 94, 122, 105, 90, 115, 107,
        5, 63, 76, 126, 92, 20, 127, 38, 17, 71,
    )
    assert inventory.file_sha256 == _sha256_file_for_test(inventory.path)


@pytest.mark.parametrize("mutation", ["duplicate", "rgb_count", "camera_nan"])
def test_frozen_inventory_rejects_schema_and_identity_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutation: str,
) -> None:
    import georeliab_mve.v4_attempt05_inputs as inputs

    payload = _inventory_payload()
    scenes = payload["scenes"]
    assert isinstance(scenes, list)
    if mutation == "duplicate":
        scenes[-1] = dict(scenes[0])
    elif mutation == "rgb_count":
        scenes[0]["rgb_files"] = scenes[0]["rgb_files"][:-1]
    else:
        scenes[0]["camera_centers"]["1"][0] = float("nan")
    path = tmp_path / "dtu_inventory.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    monkeypatch.setattr(inputs, "DTU_INVENTORY_FILE_SHA256", inputs._sha256_file(path))

    with pytest.raises(Attempt05InputClosureError, match="DTU_INVENTORY"):
        _load_verified_dtu_inventory(path)


def test_calibration_l3_expected_set_is_exactly_twenty_scenes_by_eight_views() -> None:
    assignment = build_attempt05_v4_split_assignment(
        complete_scene_ids=DTU_OFFICIAL_SCENE_IDS,
        source_root=Path.cwd(),
    )
    members = _expected_calibration_l3_members(
        split_assignment=assignment,
        ordered_views=(1, 39, 45, 49, 6, 16, 42, 27),
    )

    assert len(members) == 160
    assert len(set(members)) == 160
    assert all("_3_r5000.png" in member for member in members)
    assert {int(member.split("scan", 1)[1].split("/", 1)[0]) for member in members} == set(
        assignment.calibration
    )


def _remote_index(
    members: tuple[str, ...],
    *,
    length: int = 1234,
    etag: str = '"frozen-etag"',
) -> RemoteZipIndex:
    entries = {
        member: RemoteZipEntry(member, 0, 3, 3, 0, index)
        for index, member in enumerate(members)
    }
    return RemoteZipIndex(length, etag, entries, "b" * 64)


def test_calibration_l3_materialization_binds_archive_and_reuses_existing_members(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import georeliab_mve.v4_attempt05_inputs as inputs

    assignment = build_attempt05_v4_split_assignment(
        complete_scene_ids=DTU_OFFICIAL_SCENE_IDS,
        source_root=Path.cwd(),
    )
    views = (1, 39, 45, 49, 6, 16, 42, 27)
    members = _expected_calibration_l3_members(
        split_assignment=assignment,
        ordered_views=views,
    )
    monkeypatch.setattr(inputs, "index_remote_zip", lambda _url: _remote_index(members))

    def extract(_url, _index, requested, destination):
        result = {}
        for ordinal, member in enumerate(requested):
            path = destination / member
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"png")
            result[member] = {
                "member": member,
                "path": str(path),
                "compressed_size": 3,
                "uncompressed_size": 3,
                "crc32": "00000000",
                "raw_sha256": _sha256_file_for_test(path),
                "disposition": "reused" if ordinal == 0 else "written",
            }
        return result

    monkeypatch.setattr(inputs, "extract_range_members_evidence", extract)
    result = _materialize_calibration_l3_members(
        dtu_root=tmp_path,
        split_assignment=assignment,
        ordered_views=views,
        archive=_RectifiedArchiveBinding(
            url="https://example.invalid/Rectified.zip",
            content_length=1234,
            etag="frozen-etag",
        ),
    )

    assert len(result.records) == 160
    assert result.records[0]["disposition"] == "reused"
    assert result.written_member_count == 159
    assert result.central_directory_sha256 == "b" * 64
    assert result.observed_etag == '"frozen-etag"'
    assert result.normalized_etag == "frozen-etag"


@pytest.mark.parametrize("drift", ["bytes", "etag", "extra"])
def test_calibration_l3_materialization_rejects_archive_or_result_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    drift: str,
) -> None:
    import georeliab_mve.v4_attempt05_inputs as inputs

    assignment = build_attempt05_v4_split_assignment(
        complete_scene_ids=DTU_OFFICIAL_SCENE_IDS,
        source_root=Path.cwd(),
    )
    views = (1, 39, 45, 49, 6, 16, 42, 27)
    members = _expected_calibration_l3_members(
        split_assignment=assignment,
        ordered_views=views,
    )
    monkeypatch.setattr(
        inputs,
        "index_remote_zip",
        lambda _url: _remote_index(
            members,
            length=1235 if drift == "bytes" else 1234,
            etag='"changed-etag"' if drift == "etag" else '"frozen-etag"',
        ),
    )

    def extract(_url, _index, requested, destination):
        result = {
            member: {
                "member": member,
                "path": str(destination / member),
                "compressed_size": 3,
                "uncompressed_size": 3,
                "crc32": "00000000",
                "raw_sha256": "c" * 64,
                "disposition": "written",
            }
            for member in requested
        }
        if drift == "extra":
            result["Rectified/scan999/extra.png"] = dict(next(iter(result.values())))
        return result

    monkeypatch.setattr(inputs, "extract_range_members_evidence", extract)
    with pytest.raises(Attempt05InputClosureError, match="RECTIFIED_ARCHIVE"):
        _materialize_calibration_l3_members(
            dtu_root=tmp_path,
            split_assignment=assignment,
            ordered_views=views,
            archive=_RectifiedArchiveBinding(
                url="https://example.invalid/Rectified.zip",
                content_length=1234,
                etag="frozen-etag",
            ),
        )


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
