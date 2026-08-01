from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import threading
import zlib

import pytest

from georeliab_mve.preparation import PreparationError
from georeliab_mve.tartanair_range import RemoteZipEntry
from georeliab_mve.v4_counterfactuals import (
    AssetEvidence,
    DTU_LIGHTING_SEMANTIC_TO_PHYSICAL_TOKEN,
    LIGHTING_STATES,
    SCIENTIFIC_STATES,
    TEST_SCENE_IDS,
    build_scientific_schedule,
    canonical_json_sha256,
    materialize_dtu_state_identity,
)
from georeliab_mve.v4_rectified_closure import (
    V4RectifiedClosureError,
    create_rectified_member_closure,
    prepare_rectified_resource_schedule,
    materialize_missing_rectified_members,
    validate_rectified_member_closure,
)


SOURCE_ROOT = Path(__file__).resolve().parents[1]
ORDERED_VIEWS = (1, 7, 13, 19, 25, 31, 37, 43)
MEMBERS = ("L1", "L2", "L4", "L5", "L6", "L7")


def _sha(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _asset(member: str, label: str) -> AssetEvidence:
    return AssetEvidence(member=member, sha256=_sha(label), source_uri=f"file:///{member}")


def _state(scene_id: int, state_id: str):
    rgb_inputs = {}
    cameras = {}
    for view_id in ORDERED_VIEWS:
        if state_id in LIGHTING_STATES:
            member = f"Rectified/scan{scene_id}/rect_{view_id:03d}_{DTU_LIGHTING_SEMANTIC_TO_PHYSICAL_TOKEN[state_id]}_r5000.png"
        else:
            member = f"SyntheticFog/scan{scene_id}/{state_id}/view_{view_id:03d}.png"
        rgb_inputs[view_id] = _asset(member, f"rgb:{scene_id}:{state_id}:{view_id}")
        cameras[view_id] = _asset(
            f"MVS Data/Calibration/cal18/pos_{view_id:03d}.txt",
            f"camera:{scene_id}:{view_id}",
        )
    clean_source_inputs = None
    if state_id.startswith("fog-"):
        clean_source_inputs = {
            view_id: _asset(
                f"Rectified/scan{scene_id}/rect_{view_id:03d}_3_r5000.png",
                f"rgb:{scene_id}:L3:{view_id}",
            )
            for view_id in ORDERED_VIEWS
        }
    return materialize_dtu_state_identity(
        source_root=SOURCE_ROOT,
        scene_id=scene_id,
        state_id=state_id,
        ordered_view_ids=ORDERED_VIEWS,
        rgb_inputs=rgb_inputs,
        cameras=cameras,
        gt_point_cloud=_asset(f"Points/stl/stl{scene_id:03d}_total.ply", f"gt:{scene_id}"),
        observability_mask=_asset(
            f"MVS Data/ObsMask/ObsMask{scene_id}_10.mat",
            f"mask:{scene_id}",
        ),
        clean_source_inputs=clean_source_inputs,
    )


@pytest.fixture(scope="module")
def states_and_schedule() -> tuple[list[dict[str, object]], dict[str, object]]:
    states = [
        _state(scene_id, state_id)
        for scene_id in TEST_SCENE_IDS
        for state_id in SCIENTIFIC_STATES
    ]
    return [state.to_dict() for state in states], build_scientific_schedule(states).to_dict()


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")


def _resource_json_sha(payload: object) -> str:
    raw = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii") + b"\n"
    return hashlib.sha256(raw).hexdigest()


def _official_index(root: Path, *, overrides: dict[str, bytes] | None = None) -> dict[str, dict[str, object]]:
    overrides = overrides or {}
    entries: dict[str, dict[str, object]] = {}
    for scene_id in TEST_SCENE_IDS:
        for view_id in ORDERED_VIEWS:
            for illumination in MEMBERS:
                member = _member_path(Path("."), scene_id, view_id, illumination).as_posix()
                data = overrides.get(member)
                if data is None:
                    data = (root / member).read_bytes()
                entries[member] = {
                    "sha256": hashlib.sha256(data).hexdigest(),
                    "bytes": len(data),
                }
    return entries


def _png_bytes(width: int = 3, height: int = 2, *, color_type: int = 2, bit_depth: int = 8) -> bytes:
    return (
        b"\x89PNG\r\n\x1a\n"
        + (13).to_bytes(4, "big")
        + b"IHDR"
        + width.to_bytes(4, "big")
        + height.to_bytes(4, "big")
        + bytes([bit_depth, color_type, 0, 0, 0])
        + b"fixture-payload"
    )


def _member_path(root: Path, scene_id: int, view_id: int, illumination_id: str) -> Path:
    return root / "Rectified" / f"scan{scene_id}" / f"rect_{view_id:03d}_{DTU_LIGHTING_SEMANTIC_TO_PHYSICAL_TOKEN[illumination_id]}_r5000.png"


def _write_members(root: Path, *, color_type: int = 2, bit_depth: int = 8) -> None:
    for scene_id in TEST_SCENE_IDS:
        for view_id in ORDERED_VIEWS:
            for illumination_id in MEMBERS:
                path = _member_path(root, scene_id, view_id, illumination_id)
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(
                    _png_bytes(color_type=color_type, bit_depth=bit_depth)
                    + f":{scene_id}:{view_id}:{illumination_id}".encode("ascii")
                )


def _inputs(
    tmp_path: Path,
    states_and_schedule: tuple[list[dict[str, object]], dict[str, object]],
) -> dict[str, Path]:
    state_payloads, schedule_payload = states_and_schedule
    root = tmp_path / "runtime"
    _write_members(root)
    schedule = tmp_path / "schedule.json"
    state_inventory = tmp_path / "state_inventory.json"
    _write_json(schedule, schedule_payload)
    _write_json(
        state_inventory,
        {"schema_version": "georeliab-v4-state-inventory-1.0", "states": state_payloads},
    )
    bootstrap = prepare_rectified_resource_schedule(
        protocol_path=SOURCE_ROOT / "configs" / "georeliab_v4_protocol.toml",
        split_view_manifest_path=_split_manifest(tmp_path),
        output_dir=tmp_path / "bootstrap",
    )
    return {
        "root": root,
        "schedule": schedule,
        "state_inventory": state_inventory,
        "expected_set": Path(str(bootstrap["expected_set_path"])),
        "resource_schedule": Path(str(bootstrap["resource_schedule_path"])),
    }


def _create(
    tmp_path: Path,
    states_and_schedule: tuple[list[dict[str, object]], dict[str, object]],
) -> dict[str, object]:
    paths = _inputs(tmp_path, states_and_schedule)
    return create_rectified_member_closure(
        root=paths["root"],
        expected_set_path=paths["expected_set"],
        output_dir=tmp_path / "closure",
    )


def test_exact_960_success_and_artifact_hashes(
    tmp_path: Path,
    states_and_schedule: tuple[list[dict[str, object]], dict[str, object]],
) -> None:
    result = _create(tmp_path, states_and_schedule)

    assert result["status"] == "PASS"
    assert result["expected_base_units"] == 160
    assert result["expected_member_count"] == 960
    assert result["excluded_reference_role"]["L3"] == "REFERENCE_EXCLUDED_FROM_RECTIFIED_MEMBER_CLOSURE"
    manifest_path = Path(str(result["manifest_path"]))
    assert manifest_path.name == "v4-rectified-member-manifest.jsonl"
    assert len(manifest_path.read_text(encoding="utf-8").splitlines()) == 960
    assert Path(str(result["manifest_sha256_path"])).read_text(encoding="utf-8").strip() == result["manifest_sha256"]
    assert not list((tmp_path / "closure").glob("*.partial"))
    assert validate_rectified_member_closure(
        root=tmp_path / "runtime",
        expected_set_path=Path(str(result["expected_set_path"])),
        manifest_path=manifest_path,
        output_dir=tmp_path / "validated",
    )["status"] == "PASS"


@pytest.mark.parametrize(
    ("mutate", "reason"),
    [
        (
            lambda paths: _member_path(paths["root"], TEST_SCENE_IDS[0], ORDERED_VIEWS[0], "L1").unlink(),
            "V4_RECTIFIED_MEMBER_MISSING",
        ),
        (
            lambda paths: (paths["root"] / "Rectified" / f"scan{TEST_SCENE_IDS[0]}" / f"rect_{ORDERED_VIEWS[0]:03d}_8_r5000.png").write_bytes(_png_bytes()),
            "V4_RECTIFIED_UNEXPECTED_ILLUMINATION",
        ),
    ],
)
def test_filesystem_missing_and_l8_are_fail_closed(
    tmp_path: Path,
    states_and_schedule: tuple[list[dict[str, object]], dict[str, object]],
    mutate,
    reason: str,
) -> None:
    paths = _inputs(tmp_path, states_and_schedule)
    mutate(paths)
    with pytest.raises(V4RectifiedClosureError, match=reason):
        create_rectified_member_closure(
            root=paths["root"],
            expected_set_path=paths["expected_set"],
            output_dir=tmp_path / "closure",
        )
    assert not (tmp_path / "closure" / "v4-rectified-member-closure-receipt.json").exists()


@pytest.mark.parametrize(
    ("mutate", "reason"),
    [
        (lambda rows: rows.pop(0), "V4_RECTIFIED_MANIFEST_CARDINALITY_MISMATCH"),
        (lambda rows: rows.__setitem__(1, deepcopy(rows[0])), "V4_RECTIFIED_DUPLICATE_MEMBER"),
        (lambda rows: rows[0].__setitem__("scene_id", TEST_SCENE_IDS[1]), "V4_RECTIFIED_MEMBER_IDENTITY_MISMATCH"),
        (lambda rows: rows[0].__setitem__("illumination_id", "L3"), "V4_RECTIFIED_REFERENCE_INCLUDED"),
        (lambda rows: rows[0].pop("excluded_reference_role_by_illumination", None), "V4_RECTIFIED_L3_ROLE_UNDECLARED"),
        (lambda rows: rows[0].__setitem__("sha256", "0" * 64), "V4_RECTIFIED_MEMBER_HASH_MISMATCH"),
        (lambda rows: rows[0].__setitem__("normalized_relative_path", rows[1]["normalized_relative_path"].upper()), "V4_RECTIFIED_PATH_NORMALIZATION_DUPLICATE"),
        (
            lambda rows: [
                row.__setitem__("illumination_id", "L1")
                for row in rows
                if row["counterfactual_group_id"] == rows[0]["counterfactual_group_id"]
            ],
            "V4_RECTIFIED_GROUP_INCOMPLETE",
        ),
        (lambda rows: rows[0].__setitem__("width", 99), "V4_RECTIFIED_DIMENSION_MISMATCH"),
        (lambda rows: rows[0].__setitem__("dtype", "uint16"), "V4_RECTIFIED_DTYPE_MISMATCH"),
        (lambda rows: rows.append({**deepcopy(rows[0]), "schedule_unit_id": "orphan"}), "V4_RECTIFIED_MANIFEST_CARDINALITY_MISMATCH"),
    ],
)
def test_manifest_mutations_are_fail_closed(
    tmp_path: Path,
    states_and_schedule: tuple[list[dict[str, object]], dict[str, object]],
    mutate,
    reason: str,
) -> None:
    result = _create(tmp_path, states_and_schedule)
    manifest_path = Path(str(result["manifest_path"]))
    rows = [json.loads(line) for line in manifest_path.read_text(encoding="utf-8").splitlines()]
    mutate(rows)
    manifest_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )

    with pytest.raises(V4RectifiedClosureError, match=reason):
        validate_rectified_member_closure(
            root=tmp_path / "runtime",
            expected_set_path=Path(str(result["expected_set_path"])),
            manifest_path=manifest_path,
            output_dir=tmp_path / "validated",
        )


def test_schedule_and_state_inventory_mismatch_is_fail_closed(
    tmp_path: Path,
    states_and_schedule: tuple[list[dict[str, object]], dict[str, object]],
) -> None:
    state_payloads, schedule_payload = states_and_schedule
    payload = deepcopy(schedule_payload)
    payload["scene_ids"] = list(TEST_SCENE_IDS[:-1])
    payload["schedule_sha256"] = canonical_json_sha256(
        {key: value for key, value in payload.items() if key != "schedule_sha256"}
    )
    root = tmp_path / "runtime"
    _write_members(root)
    schedule = tmp_path / "schedule.json"
    state_inventory = tmp_path / "state_inventory.json"
    _write_json(schedule, payload)
    _write_json(state_inventory, {"states": state_payloads})

    with pytest.raises(V4RectifiedClosureError, match="V4_RECTIFIED_EXPECTED_SET_REQUIRED"):
        create_rectified_member_closure(
            root=root,
            expected_set_path=None,
            output_dir=tmp_path / "closure",
        )


def test_stale_schedule_hash_is_fail_closed(
    tmp_path: Path,
    states_and_schedule: tuple[list[dict[str, object]], dict[str, object]],
) -> None:
    state_payloads, schedule_payload = states_and_schedule
    payload = deepcopy(schedule_payload)
    payload["schedule_sha256"] = "0" * 64
    root = tmp_path / "runtime"
    _write_members(root)
    schedule = tmp_path / "schedule.json"
    state_inventory = tmp_path / "state_inventory.json"
    _write_json(schedule, payload)
    _write_json(state_inventory, {"states": state_payloads})

    with pytest.raises(V4RectifiedClosureError, match="V4_RECTIFIED_EXPECTED_SET_REQUIRED"):
        create_rectified_member_closure(
            root=root,
            expected_set_path=None,
            output_dir=tmp_path / "closure",
        )


def test_symlink_drift_is_fail_closed(
    tmp_path: Path,
    states_and_schedule: tuple[list[dict[str, object]], dict[str, object]],
) -> None:
    paths = _inputs(tmp_path, states_and_schedule)
    link = _member_path(paths["root"], TEST_SCENE_IDS[0], ORDERED_VIEWS[0], "L1")
    target = tmp_path / "target.png"
    target.write_bytes(link.read_bytes())
    link.unlink()
    link.symlink_to(target)

    with pytest.raises(V4RectifiedClosureError, match="V4_RECTIFIED_SYMLINK_FORBIDDEN"):
        create_rectified_member_closure(
            root=paths["root"],
            expected_set_path=paths["expected_set"],
            output_dir=tmp_path / "closure",
        )




def test_semantic_physical_mapping_is_explicit_and_forbids_suffix_7(
    tmp_path: Path,
    states_and_schedule: tuple[list[dict[str, object]], dict[str, object]],
) -> None:
    result = _create(tmp_path, states_and_schedule)
    manifest = Path(str(result["manifest_path"]))
    rows = [json.loads(line) for line in manifest.read_text(encoding="utf-8").splitlines()]
    by_illumination = {row["illumination_id"]: row for row in rows[:6]}

    assert by_illumination["L7"]["source_archive_illumination_token"] == "0"
    assert by_illumination["L7"]["normalized_relative_path"].endswith("_0_r5000.png")
    assert by_illumination["L1"]["normalized_relative_path"].endswith("_1_r5000.png")
    assert all("_7_r5000" not in row["normalized_relative_path"] for row in rows)
    expected_set = json.loads(
        Path(str(result["expected_set_path"])).read_text(encoding="utf-8")
    )
    assert expected_set["semantic_to_physical_suffix"]["L3"] == "3"
    assert expected_set["semantic_to_physical_suffix"]["L7"] == "0"


def test_mapping_tamper_is_fail_closed(
    tmp_path: Path,
    states_and_schedule: tuple[list[dict[str, object]], dict[str, object]],
) -> None:
    result = _create(tmp_path, states_and_schedule)
    manifest = Path(str(result["manifest_path"]))
    rows = [json.loads(line) for line in manifest.read_text(encoding="utf-8").splitlines()]
    rows[-1]["source_archive_illumination_token"] = "7"
    rows[-1]["normalized_relative_path"] = rows[-1]["normalized_relative_path"].replace(
        "_0_r5000.png", "_7_r5000.png"
    )
    manifest.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )

    with pytest.raises(V4RectifiedClosureError, match="V4_RECTIFIED_MAPPING_TAMPER"):
        validate_rectified_member_closure(
            root=tmp_path / "runtime",
            expected_set_path=Path(str(result["expected_set_path"])),
            manifest_path=manifest,
            output_dir=tmp_path / "validated",
        )

def test_materialize_missing_members_uses_expected_set_only_and_is_atomic(
    tmp_path: Path,
    states_and_schedule: tuple[list[dict[str, object]], dict[str, object]],
) -> None:
    paths = _inputs(tmp_path, states_and_schedule)
    missing = _member_path(paths["root"], TEST_SCENE_IDS[0], ORDERED_VIEWS[0], "L2")
    missing.unlink()
    requested: list[tuple[str, Path]] = []

    def fake_index(_archive: Path):
        entries = _official_index(
            paths["root"],
            overrides={
                _member_path(Path("."), TEST_SCENE_IDS[0], ORDERED_VIEWS[0], "L2").as_posix(): _png_bytes() + b"official-l2",
            },
        )
        entries[(Path("Rectified") / f"scan{TEST_SCENE_IDS[0]}" / f"rect_{ORDERED_VIEWS[0]:03d}_8_r5000.png").as_posix()] = {
            "sha256": _sha("official-l8"),
            "bytes": 1,
        }
        return entries

    def fake_extract(_archive: Path, member: str, destination: Path, expected_sha256: str) -> None:
        requested.append((member, destination))
        assert "L8" not in member
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(_png_bytes() + b"official-l2")
        assert hashlib.sha256(destination.read_bytes()).hexdigest() == expected_sha256

    result = materialize_missing_rectified_members(
        root=paths["root"],
        expected_set_path=paths["expected_set"],
        official_rectified_archive=tmp_path / "Rectified.zip",
        output_dir=tmp_path / "materialize",
        indexer=fake_index,
        extractor=fake_extract,
    )

    assert result["status"] == "PASS"
    assert result["materialized_count"] == 1
    assert requested == [
        (
            f"Rectified/scan{TEST_SCENE_IDS[0]}/rect_{ORDERED_VIEWS[0]:03d}_2_r5000.png",
            missing.with_name(missing.name + ".partial"),
        )
    ]
    assert missing.is_file()
    assert not missing.with_name(missing.name + ".partial").exists()
    assert Path(str(result["receipt_path"])).is_file()


def test_default_range_materialization_is_bounded_concurrent_and_ordered(
    tmp_path: Path,
    states_and_schedule: tuple[list[dict[str, object]], dict[str, object]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = _inputs(tmp_path, states_and_schedule)
    entries = _official_index(paths["root"])
    missing_members = [
        _member_path(paths["root"], TEST_SCENE_IDS[0], ORDERED_VIEWS[0], illumination)
        for illumination in ("L1", "L2", "L4")
    ]
    for member in missing_members:
        member.unlink()

    requested = [
        member.relative_to(paths["root"]).as_posix() for member in missing_members
    ]
    barrier = threading.Barrier(len(requested))
    observed: list[str] = []
    observed_lock = threading.Lock()

    def fake_range_extract(
        _archive: str,
        _index: object,
        members: list[str],
        destination: Path,
    ) -> dict[str, dict[str, object]]:
        assert len(members) == 1
        member = members[0]
        barrier.wait(timeout=2)
        output = destination / member
        output.parent.mkdir(parents=True, exist_ok=True)
        payload = _png_bytes()
        output.write_bytes(payload)
        with observed_lock:
            observed.append(member)
        return {
            member: {
                "member": member,
                "raw_sha256": hashlib.sha256(payload).hexdigest(),
                "disposition": "written",
            }
        }

    monkeypatch.setattr(
        "georeliab_mve.v4_rectified_closure.extract_range_members_evidence",
        fake_range_extract,
    )
    result = materialize_missing_rectified_members(
        root=paths["root"],
        expected_set_path=paths["expected_set"],
        official_rectified_archive="https://example.invalid/Rectified.zip",
        output_dir=tmp_path / "materialize-concurrent",
        indexer=lambda _archive: entries,
    )

    assert result["status"] == "PASS"
    assert result["materialized_count"] == len(requested)
    assert result["range_worker_count"] == len(requested)
    assert sorted(observed) == sorted(requested)
    assert [row["member"] for row in result["materialized_members"]] == requested
    assert all(member.is_file() for member in missing_members)


def test_public_cpu_only_cli_generates_closure(
    tmp_path: Path,
    states_and_schedule: tuple[list[dict[str, object]], dict[str, object]],
) -> None:
    paths = _inputs(tmp_path, states_and_schedule)
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "georeliab_mve",
            "v4-rectified-closure",
            "--root",
            str(paths["root"]),
            "--expected-set",
            str(paths["expected_set"]),
            "--output-dir",
            str(tmp_path / "closure"),
        ],
        cwd=SOURCE_ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    payload = json.loads(completed.stdout)
    assert payload["status"] == "PASS"
    assert "gpu" not in completed.stdout.lower()


def _split_manifest(tmp_path: Path) -> Path:
    split_path = tmp_path / "manifests" / "split_view_manifest.json"
    _write_json(
        split_path,
        {
            "schema_version": "dtu-preparation-v1",
            "splits": {"test": list(TEST_SCENE_IDS)},
            "views": {str(scene_id): list(ORDERED_VIEWS) for scene_id in TEST_SCENE_IDS},
        },
    )
    return split_path


def test_prepare_rectified_resource_schedule_bootstraps_expected_set_without_state_inventory(
    tmp_path: Path,
    states_and_schedule: tuple[list[dict[str, object]], dict[str, object]],
) -> None:
    from georeliab_mve.v4_rectified_closure import prepare_rectified_resource_schedule

    paths = _inputs(tmp_path, states_and_schedule)
    result = prepare_rectified_resource_schedule(
        protocol_path=SOURCE_ROOT / "configs" / "georeliab_v4_protocol.toml",
        split_view_manifest_path=_split_manifest(tmp_path),
        output_dir=tmp_path / "bootstrap",
    )

    assert result["status"] == "PASS"
    assert result["unit_count"] == 400
    assert result["expected_base_units"] == 160
    assert result["expected_member_count"] == 960
    assert result["unit_partition"] == {
        "non_l3_lighting": 240,
        "l3_reference": 40,
        "fog": 120,
        "full_schedule": 400,
        "union_disjointness_proven": True,
    }
    expected_set = json.loads(Path(str(result["expected_set_path"])).read_text(encoding="utf-8"))
    proof = expected_set["expected_cardinality_proof"]
    assert proof["equation"] == (
        f"{proof['scene_count_from_v4_protocol']} * "
        f"{proof['views_per_scene_from_split_view_manifest']} * "
        f"{proof['member_illuminations']} = {proof['product']}"
    )
    assert proof["product"] == expected_set["expected_member_count"]
    schedule = json.loads(Path(str(result["resource_schedule_path"])).read_text(encoding="utf-8"))
    assert schedule["resource_kind"] == "rectified_resource_schedule_not_scientific_evidence"
    assert len(schedule["units"]) == 400
    assert {unit["role"] for unit in schedule["units"]} == {"non_l3_lighting", "l3_reference", "fog"}
    assert {unit["state"] for unit in schedule["units"] if unit["role"] == "l3_reference"} == {"L3"}
    assert create_rectified_member_closure(
        root=paths["root"],
        expected_set_path=Path(str(result["expected_set_path"])),
        output_dir=tmp_path / "closure-from-expected",
    )["status"] == "PASS"


def test_public_rectified_apis_reject_direct_schedule_state_bypass(
    tmp_path: Path,
    states_and_schedule: tuple[list[dict[str, object]], dict[str, object]],
) -> None:
    paths = _inputs(tmp_path, states_and_schedule)
    valid = create_rectified_member_closure(
        root=paths["root"],
        expected_set_path=paths["expected_set"],
        output_dir=tmp_path / "valid-closure",
    )

    with pytest.raises(V4RectifiedClosureError, match="V4_RECTIFIED_EXPECTED_SET_REQUIRED"):
        create_rectified_member_closure(
            root=paths["root"],
            schedule_path=paths["schedule"],
            state_inventory_path=paths["state_inventory"],
            output_dir=tmp_path / "direct-create",
        )
    with pytest.raises(V4RectifiedClosureError, match="V4_RECTIFIED_EXPECTED_SET_REQUIRED"):
        validate_rectified_member_closure(
            root=paths["root"],
            schedule_path=paths["schedule"],
            state_inventory_path=paths["state_inventory"],
            manifest_path=Path(str(valid["manifest_path"])),
            output_dir=tmp_path / "direct-validate",
        )
    with pytest.raises(V4RectifiedClosureError, match="V4_RECTIFIED_EXPECTED_SET_REQUIRED"):
        materialize_missing_rectified_members(
            root=paths["root"],
            schedule_path=paths["schedule"],
            state_inventory_path=paths["state_inventory"],
            official_rectified_archive=tmp_path / "Rectified.zip",
            output_dir=tmp_path / "direct-materialize",
        )


def test_resource_schedule_unit_list_tamper_is_fail_closed_with_partitions_unchanged(
    tmp_path: Path,
    states_and_schedule: tuple[list[dict[str, object]], dict[str, object]],
) -> None:
    paths = _inputs(tmp_path, states_and_schedule)
    schedule_path = paths["resource_schedule"]
    expected_set_path = paths["expected_set"]
    schedule = json.loads(schedule_path.read_text(encoding="utf-8"))
    expected = json.loads(expected_set_path.read_text(encoding="utf-8"))
    untouched_partitions = deepcopy(schedule["schedule_unit_sets"])

    schedule["units"][0]["role"] = "fog"
    assert schedule["schedule_unit_sets"] == untouched_partitions
    _write_json(schedule_path, schedule)
    expected["resource_schedule_file_sha256"] = hashlib.sha256(
        schedule_path.read_bytes()
    ).hexdigest()
    expected["resource_schedule_sha256"] = _resource_json_sha(schedule)
    _write_json(expected_set_path, expected)

    with pytest.raises(V4RectifiedClosureError, match="V4_RECTIFIED_RESOURCE_SCHEDULE_TAMPER"):
        create_rectified_member_closure(
            root=paths["root"],
            expected_set_path=expected_set_path,
            output_dir=tmp_path / "closure",
        )

def test_expected_set_groups_are_bound_to_resource_schedule_views(
    tmp_path: Path,
    states_and_schedule: tuple[list[dict[str, object]], dict[str, object]],
) -> None:
    paths = _inputs(tmp_path, states_and_schedule)
    expected = json.loads(paths["expected_set"].read_text(encoding="utf-8"))
    expected["groups"][0]["view_id"] = 999
    _write_json(paths["expected_set"], expected)

    with pytest.raises(V4RectifiedClosureError, match="V4_RECTIFIED_EXPECTED_CARDINALITY_MISMATCH"):
        create_rectified_member_closure(
            root=paths["root"],
            expected_set_path=paths["expected_set"],
            output_dir=tmp_path / "closure",
        )


def test_symlink_member_and_parent_paths_are_rejected(
    tmp_path: Path,
    states_and_schedule: tuple[list[dict[str, object]], dict[str, object]],
) -> None:
    member_paths = _inputs(tmp_path / "member", states_and_schedule)
    member = _member_path(member_paths["root"], TEST_SCENE_IDS[0], ORDERED_VIEWS[0], "L1")
    target = tmp_path / "member-target.png"
    target.write_bytes(member.read_bytes())
    member.unlink()
    member.symlink_to(target)
    with pytest.raises(V4RectifiedClosureError, match="V4_RECTIFIED_SYMLINK_FORBIDDEN"):
        create_rectified_member_closure(
            root=member_paths["root"],
            expected_set_path=member_paths["expected_set"],
            output_dir=tmp_path / "member-closure",
        )

    parent_paths = _inputs(tmp_path / "parent", states_and_schedule)
    scan_dir = parent_paths["root"] / "Rectified" / f"scan{TEST_SCENE_IDS[0]}"
    real_scan_dir = parent_paths["root"] / "Rectified" / f"scan{TEST_SCENE_IDS[0]}-real"
    scan_dir.rename(real_scan_dir)
    scan_dir.symlink_to(real_scan_dir, target_is_directory=True)
    with pytest.raises(V4RectifiedClosureError, match="V4_RECTIFIED_SYMLINK_FORBIDDEN"):
        create_rectified_member_closure(
            root=parent_paths["root"],
            expected_set_path=parent_paths["expected_set"],
            output_dir=tmp_path / "parent-closure",
        )


def test_materialize_existing_member_mismatch_is_fail_closed_without_overwrite(
    tmp_path: Path,
    states_and_schedule: tuple[list[dict[str, object]], dict[str, object]],
) -> None:
    paths = _inputs(tmp_path, states_and_schedule)
    member = _member_path(paths["root"], TEST_SCENE_IDS[0], ORDERED_VIEWS[0], "L2")
    original = member.read_bytes()
    member.write_bytes(_png_bytes() + b"corrupt-existing")

    def fail_extract(*_args: object) -> None:
        raise AssertionError("extractor must not overwrite mismatched existing files")

    with pytest.raises(V4RectifiedClosureError, match="V4_RECTIFIED_OFFICIAL_MEMBER_HASH_MISMATCH"):
        materialize_missing_rectified_members(
            root=paths["root"],
            expected_set_path=paths["expected_set"],
            official_rectified_archive=tmp_path / "Rectified.zip",
            output_dir=tmp_path / "materialize",
            indexer=lambda _archive: _official_index(
                paths["root"],
                overrides={
                    _member_path(Path("."), TEST_SCENE_IDS[0], ORDERED_VIEWS[0], "L2").as_posix(): original,
                },
            ),
            extractor=fail_extract,
        )
    assert member.read_bytes().endswith(b"corrupt-existing")


def test_materialize_existing_member_crc_mismatch_without_sha_is_fail_closed(
    tmp_path: Path,
    states_and_schedule: tuple[list[dict[str, object]], dict[str, object]],
) -> None:
    paths = _inputs(tmp_path, states_and_schedule)
    entries: dict[str, RemoteZipEntry] = {}
    for scene_id in TEST_SCENE_IDS:
        for view_id in ORDERED_VIEWS:
            for illumination in MEMBERS:
                member = _member_path(Path("."), scene_id, view_id, illumination).as_posix()
                data = (paths["root"] / member).read_bytes()
                entries[member] = RemoteZipEntry(
                    name=member,
                    compression=0,
                    compressed_size=len(data),
                    uncompressed_size=len(data),
                    crc32=zlib.crc32(data) & 0xFFFFFFFF,
                    local_offset=0,
                )
    member = _member_path(paths["root"], TEST_SCENE_IDS[0], ORDERED_VIEWS[0], "L2")
    replacement = bytearray(member.read_bytes())
    replacement[-1] = (replacement[-1] + 1) % 256
    member.write_bytes(replacement)

    def fail_extract(*_args: object) -> None:
        raise AssertionError("extractor must not overwrite CRC-mismatched existing files")

    with pytest.raises(V4RectifiedClosureError, match="V4_RECTIFIED_OFFICIAL_MEMBER_HASH_MISMATCH"):
        materialize_missing_rectified_members(
            root=paths["root"],
            expected_set_path=paths["expected_set"],
            official_rectified_archive=tmp_path / "Rectified.zip",
            output_dir=tmp_path / "materialize-crc",
            indexer=lambda _archive: entries,
            extractor=fail_extract,
        )

def test_materialize_wraps_preparation_error_for_structured_cli(
    tmp_path: Path,
    states_and_schedule: tuple[list[dict[str, object]], dict[str, object]],
) -> None:
    paths = _inputs(tmp_path, states_and_schedule)

    def broken_index(_archive: Path) -> object:
        raise PreparationError("range index failed")

    with pytest.raises(V4RectifiedClosureError, match="V4_RECTIFIED_OFFICIAL_INDEX_INVALID"):
        materialize_missing_rectified_members(
            root=paths["root"],
            expected_set_path=paths["expected_set"],
            official_rectified_archive=tmp_path / "Rectified.zip",
            output_dir=tmp_path / "materialize",
            indexer=broken_index,
        )


def test_failed_closure_leaves_no_final_artifacts(
    tmp_path: Path,
    states_and_schedule: tuple[list[dict[str, object]], dict[str, object]],
) -> None:
    result = _create(tmp_path, states_and_schedule)
    manifest = Path(str(result["manifest_path"]))
    rows = [json.loads(line) for line in manifest.read_text(encoding="utf-8").splitlines()]
    rows.pop()
    manifest.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    output_dir = tmp_path / "failed-validate"

    with pytest.raises(V4RectifiedClosureError, match="V4_RECTIFIED_MANIFEST_CARDINALITY_MISMATCH"):
        validate_rectified_member_closure(
            root=tmp_path / "runtime",
            expected_set_path=Path(str(result["expected_set_path"])),
            manifest_path=manifest,
            output_dir=output_dir,
        )
    assert not output_dir.exists()

def test_prepare_rectified_resource_schedule_rejects_split_scene_and_view_drift(tmp_path: Path) -> None:
    from georeliab_mve.v4_rectified_closure import prepare_rectified_resource_schedule

    split_path = _split_manifest(tmp_path)
    payload = json.loads(split_path.read_text(encoding="utf-8"))
    payload["splits"]["test"] = list(TEST_SCENE_IDS[:-1])
    _write_json(split_path, payload)
    with pytest.raises(V4RectifiedClosureError, match="V4_RECTIFIED_SPLIT_TEST_SCENES_MISMATCH"):
        prepare_rectified_resource_schedule(
            protocol_path=SOURCE_ROOT / "configs" / "georeliab_v4_protocol.toml",
            split_view_manifest_path=split_path,
            output_dir=tmp_path / "bootstrap",
        )

    split_path = _split_manifest(tmp_path / "fresh")
    payload = json.loads(split_path.read_text(encoding="utf-8"))
    payload["views"][str(TEST_SCENE_IDS[0])] = list(ORDERED_VIEWS[:-1])
    _write_json(split_path, payload)
    with pytest.raises(V4RectifiedClosureError, match="V4_RECTIFIED_SCENE_VIEW_MISMATCH"):
        prepare_rectified_resource_schedule(
            protocol_path=SOURCE_ROOT / "configs" / "georeliab_v4_protocol.toml",
            split_view_manifest_path=split_path,
            output_dir=tmp_path / "bootstrap-view",
        )


def test_expected_set_schedule_partition_tamper_is_fail_closed(
    tmp_path: Path,
    states_and_schedule: tuple[list[dict[str, object]], dict[str, object]],
) -> None:
    from georeliab_mve.v4_rectified_closure import prepare_rectified_resource_schedule

    paths = _inputs(tmp_path, states_and_schedule)
    result = prepare_rectified_resource_schedule(
        protocol_path=SOURCE_ROOT / "configs" / "georeliab_v4_protocol.toml",
        split_view_manifest_path=_split_manifest(tmp_path),
        output_dir=tmp_path / "bootstrap",
    )
    expected_set_path = Path(str(result["expected_set_path"]))
    expected = json.loads(expected_set_path.read_text(encoding="utf-8"))
    expected["schedule_unit_sets"]["fog"].pop()
    _write_json(expected_set_path, expected)
    with pytest.raises(V4RectifiedClosureError, match="V4_RECTIFIED_SCHEDULE_BINDING_UNPROVEN"):
        create_rectified_member_closure(
            root=paths["root"],
            expected_set_path=expected_set_path,
            output_dir=tmp_path / "closure",
        )


def test_expected_set_scene_state_fake_schedule_ids_are_fail_closed(
    tmp_path: Path,
    states_and_schedule: tuple[list[dict[str, object]], dict[str, object]],
) -> None:
    from georeliab_mve.v4_rectified_closure import prepare_rectified_resource_schedule

    paths = _inputs(tmp_path, states_and_schedule)
    result = prepare_rectified_resource_schedule(
        protocol_path=SOURCE_ROOT / "configs" / "georeliab_v4_protocol.toml",
        split_view_manifest_path=_split_manifest(tmp_path),
        output_dir=tmp_path / "bootstrap",
    )
    valid_expected_set_path = Path(str(result["expected_set_path"]))
    closure = create_rectified_member_closure(
        root=paths["root"],
        expected_set_path=valid_expected_set_path,
        output_dir=tmp_path / "valid-closure",
    )

    expected = json.loads(valid_expected_set_path.read_text(encoding="utf-8"))
    expected["schedule_unit_ids_by_scene_state"]["1:L1"] = ["a" * 64, "b" * 64]
    tampered_expected_set_path = tmp_path / "tampered" / "expected-set.json"
    _write_json(tampered_expected_set_path, expected)

    with pytest.raises(V4RectifiedClosureError, match="V4_RECTIFIED_SCHEDULE_BINDING_UNPROVEN"):
        create_rectified_member_closure(
            root=paths["root"],
            expected_set_path=tampered_expected_set_path,
            output_dir=tmp_path / "tampered-create",
        )
    with pytest.raises(V4RectifiedClosureError, match="V4_RECTIFIED_SCHEDULE_BINDING_UNPROVEN"):
        validate_rectified_member_closure(
            root=paths["root"],
            manifest_path=Path(str(closure["manifest_path"])),
            expected_set_path=tampered_expected_set_path,
            output_dir=tmp_path / "tampered-validate",
        )

def test_rectified_cli_fail_closed_errors_are_structured_json(tmp_path: Path) -> None:
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "georeliab_mve",
            "v4-rectified-closure",
            "--root",
            str(tmp_path),
            "--output-dir",
            str(tmp_path / "closure"),
        ],
        cwd=SOURCE_ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    assert completed.returncode == 2
    payload = json.loads(completed.stdout)
    assert payload["status"] == "FAIL"
    assert payload["reason_code"] == "V4_RECTIFIED_EXPECTED_SET_REQUIRED"
    assert completed.stderr == ""
