from __future__ import annotations

from collections import Counter
from copy import deepcopy
import hashlib
import os
from pathlib import Path

import pytest

from georeliab_mve.v4_counterfactuals import (
    AssetEvidence,
    AxisSemantics,
    CounterfactualAxis,
    CounterfactualContractError,
    DatasetRole,
    FOG_BOUNDARY_LAG_SEQUENCE,
    LIGHTING_STATES,
    MVE_GO_TO_EXTERNAL_VALIDATION,
    ModelIndependentState,
    SCIENTIFIC_MODELS,
    SCIENTIFIC_STATES,
    TEST_SCENE_IDS,
    VerifiedSceneInventory,
    build_counterfactual_pair_manifest,
    build_scientific_schedule,
    canonical_json_sha256,
    construct_v4_splits,
    materialize_dtu_state_identity,
    parse_counterfactual_pair_manifest,
    parse_scientific_schedule,
    read_counterfactual_pair_manifest,
    validate_boundary_lag_sequence,
    validate_counterfactual_pair_manifest,
    validate_dataset_admission,
    validate_dtu_lighting_states,
    validate_fog_states,
    validate_scientific_schedule,
    validate_v4_split_assignment,
    write_counterfactual_pair_manifest,
)
from georeliab_mve.v4_science_lock import (
    NO_SCIENTIFIC_RESULT,
    V4_PROTOCOL_ID,
    V4_PROTOCOL_SHA256,
    validate_v4_science_lock,
)


SOURCE_ROOT = Path(__file__).resolve().parents[1]
ORDERED_VIEWS = (1, 7, 13, 19, 25, 31, 37, 43)
ALTERNATE_VIEWS = (2, 8, 14, 20, 26, 32, 38, 44)


def _sha(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _asset(
    member: str,
    semantic_label: str,
    *,
    uri_root: str,
) -> AssetEvidence:
    return AssetEvidence(
        member=member,
        sha256=_sha(semantic_label),
        source_uri=f"{uri_root}/{member}",
    )


def _state(
    scene_id: int,
    state_id: str,
    *,
    views: tuple[int, ...] = ORDERED_VIEWS,
    uri_root: str = "file:///materialized-a",
    input_salt: str = "",
    source_salt: str = "",
    camera_salt: str = "",
    gt_salt: str = "",
    mask_salt: str = "",
):
    rgb_inputs = {}
    cameras = {}
    for view_id in views:
        if state_id in LIGHTING_STATES:
            member = (
                f"Rectified/scan{scene_id}/rect_{view_id:03d}_{state_id[1:]}_r5000.png"
            )
        else:
            member = f"SyntheticFog/scan{scene_id}/{state_id}/view_{view_id:03d}.png"
        rgb_inputs[view_id] = _asset(
            member,
            f"rgb:{scene_id}:{state_id}:{view_id}:{input_salt}",
            uri_root=uri_root,
        )
        cameras[view_id] = _asset(
            f"MVS Data/Calibration/cal18/pos_{view_id:03d}.txt",
            f"camera:{scene_id}:{view_id}:{camera_salt}",
            uri_root=uri_root,
        )

    clean_source_inputs = None
    if state_id.startswith("fog-"):
        clean_source_inputs = {
            view_id: _asset(
                (f"Rectified/scan{scene_id}/rect_{view_id:03d}_3_r5000.png"),
                f"rgb:{scene_id}:L3:{view_id}:{source_salt}",
                uri_root=uri_root,
            )
            for view_id in views
        }

    return materialize_dtu_state_identity(
        source_root=SOURCE_ROOT,
        scene_id=scene_id,
        state_id=state_id,
        ordered_view_ids=views,
        rgb_inputs=rgb_inputs,
        cameras=cameras,
        gt_point_cloud=_asset(
            f"Points/stl/stl{scene_id:03d}_total.ply",
            f"gt:{scene_id}:{gt_salt}",
            uri_root=uri_root,
        ),
        observability_mask=_asset(
            f"MVS Data/ObsMask/ObsMask{scene_id}_10.mat",
            f"mask:{scene_id}:{mask_salt}",
            uri_root=uri_root,
        ),
        clean_source_inputs=clean_source_inputs,
    )


def _lighting_pair(
    *,
    scene_id: int = 1,
    counterfactual_state: str = "L1",
    views: tuple[int, ...] = ORDERED_VIEWS,
    uri_root: str = "file:///materialized-a",
    input_salt: str = "",
    camera_salt: str = "",
    gt_salt: str = "",
    mask_salt: str = "",
):
    reference = _state(
        scene_id,
        "L3",
        views=views,
        uri_root=uri_root,
        camera_salt=camera_salt,
        gt_salt=gt_salt,
        mask_salt=mask_salt,
    )
    counterfactual = _state(
        scene_id,
        counterfactual_state,
        views=views,
        uri_root=uri_root,
        input_salt=input_salt,
        camera_salt=camera_salt,
        gt_salt=gt_salt,
        mask_salt=mask_salt,
    )
    return build_counterfactual_pair_manifest(
        reference_state=reference,
        counterfactual_state=counterfactual,
        axis=CounterfactualAxis.DTU_LIGHTING,
        axis_semantics=AxisSemantics.UNORDERED_DISCRETE,
    )


def _all_states_for_scene(scene_id: int):
    lighting = [_state(scene_id, state_id) for state_id in LIGHTING_STATES]
    fog = [_state(scene_id, state_id) for state_id in ("fog-s1", "fog-s2", "fog-s3")]
    return (*lighting, *fog)


@pytest.fixture(scope="module")
def scientific_states():
    return tuple(
        state
        for scene_id in TEST_SCENE_IDS
        for state in _all_states_for_scene(scene_id)
    )


@pytest.fixture(scope="module")
def scientific_schedule(scientific_states):
    return build_scientific_schedule(scientific_states)


def _rehash_schedule(payload: dict[str, object]) -> dict[str, object]:
    payload["schedule_sha256"] = canonical_json_sha256(
        {key: value for key, value in payload.items() if key != "schedule_sha256"}
    )
    return payload


def test_pair_identity_is_model_checkpoint_path_and_json_order_invariant(
    tmp_path: Path,
) -> None:
    pair_a = _lighting_pair(uri_root="file:///first/location")
    pair_b = _lighting_pair(uri_root="s3://different-bucket/prefix")

    # Model/checkpoint/device are execution metadata and cannot enter this schema.
    execution_a = {
        "model": "VGGT",
        "checkpoint": "a" * 64,
        "device": "cuda:0",
    }
    execution_b = {
        "model": "MASt3R",
        "checkpoint": "b" * 64,
        "device": "cpu",
    }
    assert execution_a != execution_b
    assert pair_a.pair_identity_sha256 == pair_b.pair_identity_sha256
    assert pair_a.payload_sha256 == pair_b.payload_sha256

    serialized = pair_a.to_dict()
    reversed_top_level = dict(reversed(tuple(serialized.items())))
    parsed = parse_counterfactual_pair_manifest(reversed_top_level)
    assert parsed.pair_identity_sha256 == pair_a.pair_identity_sha256
    assert parsed.canonical_json_bytes() == pair_a.canonical_json_bytes()
    assert validate_counterfactual_pair_manifest(pair_a) == pair_a

    text = pair_a.canonical_json_bytes().decode("ascii")
    for forbidden in (
        "VGGT",
        "MASt3R",
        "checkpoint",
        "device",
        "source_uri",
        "file://",
        "s3://",
    ):
        assert forbidden not in text

    path = tmp_path / "pair.json"
    written_sha = write_counterfactual_pair_manifest(path, pair_a)
    assert written_sha == hashlib.sha256(path.read_bytes()).hexdigest()
    assert read_counterfactual_pair_manifest(path) == pair_a


def test_pair_manifest_atomic_publish_preserves_concurrent_conflict(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "pair.json"
    proposed = _lighting_pair(counterfactual_state="L1")
    concurrent = _lighting_pair(counterfactual_state="L2")
    concurrent_bytes = concurrent.canonical_json_bytes()
    real_fsync = os.fsync
    conflict_injected = False

    def inject_conflicting_target(file_descriptor: int) -> None:
        nonlocal conflict_injected
        real_fsync(file_descriptor)
        if not conflict_injected:
            with path.open("xb") as handle:
                handle.write(concurrent_bytes)
            conflict_injected = True

    monkeypatch.setattr(
        "georeliab_mve.v4_counterfactuals.os.fsync",
        inject_conflicting_target,
    )

    with pytest.raises(CounterfactualContractError, match="conflicts"):
        write_counterfactual_pair_manifest(path, proposed)

    assert conflict_injected
    assert path.read_bytes() == concurrent_bytes
    assert not tuple(tmp_path.glob(".pair.json.*.tmp"))

    identical_sha = write_counterfactual_pair_manifest(path, concurrent)
    assert identical_sha == hashlib.sha256(concurrent_bytes).hexdigest()
    assert path.read_bytes() == concurrent_bytes
    assert not tuple(tmp_path.glob(".pair.json.*.tmp"))


@pytest.mark.parametrize(
    "changed",
    (
        _lighting_pair(scene_id=9),
        _lighting_pair(views=ALTERNATE_VIEWS),
        _lighting_pair(input_salt="changed-input"),
        _lighting_pair(camera_salt="changed-camera"),
        _lighting_pair(gt_salt="changed-gt"),
        _lighting_pair(mask_salt="changed-mask"),
    ),
    ids=("scene", "views", "input", "camera", "gt", "mask"),
)
def test_pair_identity_changes_with_any_semantic_change(changed) -> None:
    baseline = _lighting_pair()
    assert changed.pair_identity_sha256 != baseline.pair_identity_sha256


def test_pair_manifest_closed_schema_hash_and_tamper_rejection() -> None:
    pair = _lighting_pair()
    payload = pair.to_dict()

    extra = deepcopy(payload)
    extra["model_name"] = "VGGT"
    with pytest.raises(CounterfactualContractError, match="unexpected"):
        parse_counterfactual_pair_manifest(extra)

    missing = deepcopy(payload)
    del missing["payload_sha256"]
    with pytest.raises(CounterfactualContractError, match="missing"):
        parse_counterfactual_pair_manifest(missing)

    bad_hash = deepcopy(payload)
    bad_hash["pair_identity_sha256"] = "NOT-A-HASH"
    with pytest.raises(CounterfactualContractError, match="SHA-256"):
        parse_counterfactual_pair_manifest(bad_hash)

    tampered = deepcopy(payload)
    tampered["payload"]["counterfactual_state"]["state_id"] = "L2"
    with pytest.raises(
        CounterfactualContractError,
        match="tamper|mismatch|official inputs",
    ):
        parse_counterfactual_pair_manifest(tampered)

    duplicate_views = deepcopy(payload)
    reference = duplicate_views["payload"]["reference_state"]
    reference["ordered_view_ids"][1] = reference["ordered_view_ids"][0]
    with pytest.raises(CounterfactualContractError, match="duplicate"):
        parse_counterfactual_pair_manifest(duplicate_views)

    duplicate_json_key = (
        pair.canonical_json_bytes()
        .decode("ascii")
        .replace(
            '"schema_version":',
            '"schema_version":"duplicate","schema_version":',
            1,
        )
    )
    with pytest.raises(CounterfactualContractError, match="duplicate JSON key"):
        parse_counterfactual_pair_manifest(duplicate_json_key)


def _inventory(*, incomplete: frozenset[int] = frozenset()):
    official_scene_ids = (*range(1, 78), *range(82, 129))
    return tuple(
        VerifiedSceneInventory(
            scene_id=scene_id,
            verified_complete=scene_id not in incomplete,
            inventory_sha256=_sha(f"inventory:{scene_id}:{scene_id not in incomplete}"),
        )
        for scene_id in official_scene_ids
    )


def test_v4_splits_are_deterministic_exact_and_scene_disjoint() -> None:
    inventory = _inventory(incomplete=frozenset({2}))
    assignment = construct_v4_splits(inventory)
    same_from_reverse_inventory = construct_v4_splits(reversed(inventory))

    assert assignment == same_from_reverse_inventory
    assert assignment.calibration == (
        82,
        52,
        74,
        73,
        94,
        122,
        105,
        90,
        115,
        107,
        5,
        63,
        76,
        126,
        92,
        20,
        127,
        38,
        17,
        71,
    )
    assert assignment.dev == (96, 124, 35, 28, 44)
    assert assignment.reference == (55, 98, 42, 104, 89)
    assert assignment.test == TEST_SCENE_IDS
    assert assignment.excluded_incomplete == (2,)
    assert 4 not in assignment.assigned_scene_ids
    assert 15 not in assignment.assigned_scene_ids

    splits = (
        set(assignment.calibration),
        set(assignment.dev),
        set(assignment.reference),
        set(assignment.test),
    )
    assert all(
        left.isdisjoint(right)
        for index, left in enumerate(splits)
        for right in splits[index + 1 :]
    )
    validate_v4_split_assignment(assignment)


def test_v4_splits_use_only_explicit_verified_inventory() -> None:
    with pytest.raises(CounterfactualContractError, match="test.*incomplete"):
        construct_v4_splits(_inventory(incomplete=frozenset({1})))

    too_few = tuple(
        row
        for row in _inventory()
        if row.scene_id in TEST_SCENE_IDS or row.scene_id in {4, 15, 2, 3}
    )
    with pytest.raises(CounterfactualContractError, match="30"):
        construct_v4_splits(too_few)

    duplicate = (*_inventory(), _inventory()[0])
    with pytest.raises(CounterfactualContractError, match="duplicate"):
        construct_v4_splits(duplicate)


def test_state_materialization_rejects_non_official_dtu_scene() -> None:
    with pytest.raises(CounterfactualContractError, match="official DTU"):
        _state(999, "L3")


def test_rehashed_standalone_state_rejects_non_official_dtu_scene() -> None:
    payload = deepcopy(_state(1, "L3").to_dict())
    payload["scene_id"] = 999
    payload["state_identity_sha256"] = canonical_json_sha256(
        {key: value for key, value in payload.items() if key != "state_identity_sha256"}
    )

    with pytest.raises(CounterfactualContractError, match="official DTU"):
        ModelIndependentState.from_dict(payload)


def test_lighting_materialization_requires_exact_names_and_shared_identity() -> None:
    states = tuple(_state(1, state_id) for state_id in LIGHTING_STATES)
    report = validate_dtu_lighting_states(states)

    assert report["states"] == LIGHTING_STATES
    assert report["ordered_view_ids"] == ORDERED_VIEWS
    assert len({state.scene_identity_sha256 for state in states}) == 1
    assert len({state.camera_identity_sha256 for state in states}) == 1
    assert len({state.gt_point_cloud_sha256 for state in states}) == 1
    assert len({state.observability_mask_sha256 for state in states}) == 1

    bad_camera = tuple(
        _state(1, state_id, camera_salt="swapped" if state_id == "L2" else "")
        for state_id in LIGHTING_STATES
    )
    with pytest.raises(CounterfactualContractError, match="camera"):
        validate_dtu_lighting_states(bad_camera)

    views = ORDERED_VIEWS
    rgb = {
        view_id: _asset(
            f"Rectified/scan1/rect_{view_id:03d}_1_r5000.png",
            f"rgb:1:L1:{view_id}:",
            uri_root="file:///data",
        )
        for view_id in views
    }
    rgb[views[0]] = _asset(
        "Rectified/scan1/rect_001_3_r5000.png",
        "rgb:1:L1:1:",
        uri_root="file:///data",
    )
    with pytest.raises(CounterfactualContractError, match="official.*L1"):
        materialize_dtu_state_identity(
            source_root=SOURCE_ROOT,
            scene_id=1,
            state_id="L1",
            ordered_view_ids=views,
            rgb_inputs=rgb,
            cameras={
                view_id: _asset(
                    f"MVS Data/Calibration/cal18/pos_{view_id:03d}.txt",
                    f"camera:1:{view_id}:",
                    uri_root="file:///data",
                )
                for view_id in views
            },
            gt_point_cloud=_asset(
                "Points/stl/stl001_total.ply",
                "gt:1:",
                uri_root="file:///data",
            ),
            observability_mask=_asset(
                "MVS Data/ObsMask/ObsMask1_10.mat",
                "mask:1:",
                uri_root="file:///data",
            ),
        )


def test_axis_semantics_refuse_lighting_severity_and_boundary_lag() -> None:
    reference = _state(1, "L3")
    counterfactual = _state(1, "L1")
    with pytest.raises(CounterfactualContractError, match="UNORDERED_DISCRETE"):
        build_counterfactual_pair_manifest(
            reference_state=reference,
            counterfactual_state=counterfactual,
            axis=CounterfactualAxis.DTU_LIGHTING,
            axis_semantics=AxisSemantics.ORDERED,
        )

    with pytest.raises(CounterfactualContractError, match="Boundary Lag"):
        validate_boundary_lag_sequence(
            CounterfactualAxis.DTU_LIGHTING,
            ("L3", "L1"),
        )

    pair_payload = _lighting_pair().to_dict()
    pair_payload["payload"]["severity_rank"] = 1
    with pytest.raises(CounterfactualContractError, match="unexpected"):
        parse_counterfactual_pair_manifest(pair_payload)


def test_fog_accepts_only_frozen_order_and_binds_each_state_to_l3() -> None:
    states = (
        _state(1, "L3"),
        _state(1, "fog-s1"),
        _state(1, "fog-s2"),
        _state(1, "fog-s3"),
    )
    report = validate_fog_states(states)
    assert report["states"] == FOG_BOUNDARY_LAG_SEQUENCE
    assert all(
        state.source_state_id == "L3"
        and state.source_input_sha256_by_view == states[0].input_sha256_by_view
        for state in states[1:]
    )
    validate_boundary_lag_sequence(
        CounterfactualAxis.SYNTHETIC_FOG,
        FOG_BOUNDARY_LAG_SEQUENCE,
    )

    with pytest.raises(CounterfactualContractError, match="frozen.*sequence"):
        validate_fog_states((states[0], states[2], states[1], states[3]))

    wrong_source = (
        _state(1, "L3"),
        _state(1, "fog-s1", source_salt="not-L3"),
        _state(1, "fog-s2"),
        _state(1, "fog-s3"),
    )
    with pytest.raises(CounterfactualContractError, match="L3 source"):
        validate_fog_states(wrong_source)


def test_schedule_is_exactly_400_and_reuses_l3_once(
    scientific_schedule,
) -> None:
    schedule = scientific_schedule
    assert schedule.models == SCIENTIFIC_MODELS
    assert schedule.scene_ids == TEST_SCENE_IDS
    assert schedule.state_ids == SCIENTIFIC_STATES
    assert len(schedule.units) == 400
    assert len({unit.execution_unit_sha256 for unit in schedule.units}) == 400

    counts = Counter(unit.model_id for unit in schedule.units)
    assert counts == {"VGGT": 200, "MASt3R": 200}
    l3_units = [unit for unit in schedule.units if unit.state_id == "L3"]
    assert len(l3_units) == 40
    assert all(unit.pair_identity_sha256 is None for unit in l3_units)
    assert all("fog-clean" not in unit.state_id for unit in schedule.units)

    by_model = {
        model: {
            (
                unit.scene_id,
                unit.state_id,
                unit.state_identity_sha256,
                unit.pair_identity_sha256,
            )
            for unit in schedule.units
            if unit.model_id == model
        }
        for model in SCIENTIFIC_MODELS
    }
    assert len(by_model["VGGT"]) == 200
    assert by_model["VGGT"] == by_model["MASt3R"]


def test_schedule_fingerprint_is_canonical_and_input_order_independent(
    scientific_states,
    scientific_schedule,
) -> None:
    reversed_schedule = build_scientific_schedule(reversed(scientific_states))
    assert reversed_schedule.schedule_sha256 == scientific_schedule.schedule_sha256
    assert (
        reversed_schedule.canonical_json_bytes()
        == scientific_schedule.canonical_json_bytes()
    )

    reversed_payload = dict(reversed(tuple(scientific_schedule.to_dict().items())))
    assert (
        parse_scientific_schedule(reversed_payload).schedule_sha256
        == scientific_schedule.schedule_sha256
    )
    assert validate_scientific_schedule(scientific_schedule) == scientific_schedule


def test_schedule_rejects_rehashed_noncanonical_unit_sequence(
    scientific_schedule,
) -> None:
    payload = deepcopy(scientific_schedule.to_dict())
    units = payload["units"]
    units[0], units[1] = units[1], units[0]
    _rehash_schedule(payload)

    with pytest.raises(CounterfactualContractError, match="canonical unit sequence"):
        parse_scientific_schedule(payload)


@pytest.mark.parametrize("mutation", ("missing", "duplicate", "extra"))
def test_schedule_rejects_missing_duplicate_and_extra_units(
    scientific_schedule,
    mutation: str,
) -> None:
    payload = deepcopy(scientific_schedule.to_dict())
    units = payload["units"]
    if mutation == "missing":
        units.pop()
    elif mutation == "duplicate":
        units[-1] = deepcopy(units[0])
    else:
        units.append(deepcopy(units[0]))
    _rehash_schedule(payload)

    with pytest.raises(
        CounterfactualContractError,
        match="400|duplicate|missing|extra",
    ):
        parse_scientific_schedule(payload)


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("models", ["VGGT", "MASt3R", "DUSt3R"]),
        ("scene_ids", [*TEST_SCENE_IDS, 999]),
        ("state_ids", [*SCIENTIFIC_STATES, "fog-clean"]),
    ),
)
def test_schedule_rejects_extra_models_scenes_and_states(
    scientific_schedule,
    field: str,
    value: list[object],
) -> None:
    payload = deepcopy(scientific_schedule.to_dict())
    payload[field] = value
    _rehash_schedule(payload)
    with pytest.raises(CounterfactualContractError, match="exact"):
        parse_scientific_schedule(payload)


def test_tartanair_is_non_scientific_and_uavlight_requires_future_protocol() -> None:
    assert (
        validate_dataset_admission(
            "TartanAir",
            DatasetRole.PHYSICAL_SANITY_NON_SCIENTIFIC,
            scientific=False,
            decision_status=NO_SCIENTIFIC_RESULT,
            protocol_line="v4",
        )
        is DatasetRole.PHYSICAL_SANITY_NON_SCIENTIFIC
    )
    with pytest.raises(CounterfactualContractError, match="scientific"):
        validate_dataset_admission(
            "TartanAir",
            DatasetRole.PHYSICAL_SANITY_NON_SCIENTIFIC,
            scientific=True,
            decision_status=NO_SCIENTIFIC_RESULT,
            protocol_line="v4",
        )

    with pytest.raises(CounterfactualContractError, match="separate future protocol"):
        validate_dataset_admission(
            "UAVLight",
            DatasetRole.SCIENTIFIC,
            scientific=True,
            decision_status=NO_SCIENTIFIC_RESULT,
            protocol_line="v4",
        )
    with pytest.raises(CounterfactualContractError, match="separate future protocol"):
        validate_dataset_admission(
            "UAVLight",
            DatasetRole.SCIENTIFIC,
            scientific=True,
            decision_status=MVE_GO_TO_EXTERNAL_VALIDATION,
            protocol_line="v4.1",
        )


@pytest.mark.parametrize(
    ("role", "scientific"),
    [
        (DatasetRole.SCIENTIFIC, False),
        (DatasetRole.PHYSICAL_SANITY_NON_SCIENTIFIC, True),
    ],
    ids=["scientific-boolean-required", "scientific-role-required"],
)
def test_uavlight_cannot_be_admitted_by_v4_even_with_post_go_arguments(
    role: DatasetRole,
    scientific: bool,
) -> None:
    with pytest.raises(
        CounterfactualContractError,
        match="separate future protocol",
    ):
        validate_dataset_admission(
            "UAVLight",
            role,
            scientific=scientific,
            decision_status=MVE_GO_TO_EXTERNAL_VALIDATION,
            protocol_line="v4.1",
        )


def test_v1_artifacts_and_non_dtu_datasets_cannot_enter_v4_schedule(
    scientific_schedule,
) -> None:
    old_v1_wrapper = deepcopy(scientific_schedule.to_dict())
    old_v1_wrapper["p2_schedule"] = {
        "schema_version": "georeliab-run-manifest-v1",
        "artifact_path": "old-v1.json",
    }
    _rehash_schedule(old_v1_wrapper)
    with pytest.raises(CounterfactualContractError, match="unexpected"):
        parse_scientific_schedule(old_v1_wrapper)

    tartan_unit = deepcopy(scientific_schedule.to_dict())
    tartan_unit["units"][0]["dataset"] = "TartanAir"
    _rehash_schedule(tartan_unit)
    with pytest.raises(CounterfactualContractError, match="DTU|tamper"):
        parse_scientific_schedule(tartan_unit)


def test_task1_and_v1_science_locks_remain_valid() -> None:
    report = validate_v4_science_lock(SOURCE_ROOT)
    assert report["protocol_id"] == V4_PROTOCOL_ID
    assert report["protocol_sha256"] == V4_PROTOCOL_SHA256
    assert report["status"] == "GEORELIAB_V4_PROTOCOL_READY"
