from __future__ import annotations

from copy import deepcopy
from functools import lru_cache
import hashlib
import math
from pathlib import Path
from statistics import fmean

import pytest

from georeliab_mve.v4_counterfactuals import (
    AssetEvidence,
    DTU_OFFICIAL_SCENE_IDS,
    SCIENTIFIC_MODELS,
    SCIENTIFIC_STATES,
    TEST_SCENE_IDS,
    ScientificExecutionUnit,
    ScientificSchedule,
    VerifiedSceneInventory,
    build_scientific_schedule,
    canonical_json_sha256,
    construct_v4_splits,
    materialize_dtu_state_identity,
)
from georeliab_mve.v4_gates import (
    MVE_BLOCKED_CALIBRATION_BINDING_REQUIRED,
    MVE_BLOCKED_ENDPOINT,
    MVE_BLOCKED_SOURCE_RECORD_BINDING_REQUIRED,
    MVE_BLOCKED_SCHEDULE_BINDING_REQUIRED,
    MVE_GO_TO_EXTERNAL_VALIDATION,
    MVE_SCIENTIFIC_NO_GO,
    _evaluate_rebuilt_warning_evidence,
    evaluate_warning_gate,
)
from georeliab_mve.v4_metrics import (
    CalibrationWarningSample,
    PointTaskMetrics,
    PosePairMetrics,
    PoseTaskMetrics,
    V4MetricError,
    boundary_lag_for_scene,
    counterfactual_rank_response,
    empirical_pose_auc,
    normalized_aurc,
    relative_warning_gap,
    silent_failure_rate,
)
from georeliab_mve.v4_records import (
    BootstrapMetadata,
    HolmEvidence,
    MetricEstimate,
    ModelWarningEvidence,
    Task3ContractError,
    build_task_audit_record,
    build_warning_evidence_record,
    parse_task_audit_record,
    parse_warning_evidence,
    read_warning_evidence,
    write_warning_evidence,
)
from georeliab_mve.v4_statistics import (
    FAMILY_HYPOTHESES,
    V4_BOOTSTRAP_RESAMPLES,
    V4_BOOTSTRAP_SEED,
    build_warning_evidence,
    holm_adjust_four,
    scene_block_bootstrap,
)


ORDERED_VIEWS = (1, 7, 13, 19, 25, 31, 37, 43)
LIGHTING_STATES = ("L1", "L2", "L3", "L4", "L5", "L6", "L7")
SOURCE_ROOT = Path(__file__).resolve().parents[1]


def _sha(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


@lru_cache(maxsize=1)
def _split_assignment():
    return construct_v4_splits(
        tuple(
            VerifiedSceneInventory(
                scene_id=scene_id,
                verified_complete=True,
                inventory_sha256=_sha(f"inventory:{scene_id}"),
            )
            for scene_id in DTU_OFFICIAL_SCENE_IDS
        )
    )


def _fit_calibration(model_id: str, split, *, warning_offset: float = 0.0):
    samples = tuple(
        CalibrationWarningSample(
            model_id=model_id,
            scene_id=scene_id,
            state_id="L3",
            warning_score=warning_offset + index / 20.0,
            split_fingerprint_sha256=split.fingerprint_sha256,
            inventory_sha256=split.inventory_sha256,
        )
        for index, scene_id in enumerate(split.calibration)
    )
    from georeliab_mve.v4_metrics import fit_native_warning_calibration

    return fit_native_warning_calibration(samples, split_assignment=split)


@lru_cache(maxsize=2)
def _calibration(model_id: str):
    return _fit_calibration(model_id, _split_assignment())


def _calibrations():
    return tuple(_calibration(model_id) for model_id in SCIENTIFIC_MODELS)


def _asset(member: str, semantic_label: str) -> AssetEvidence:
    return AssetEvidence(
        member=member,
        sha256=_sha(semantic_label),
        source_uri=f"file:///task3-tests/{member}",
    )


def _state(
    scene_id: int,
    state_id: str,
    *,
    identity_namespace: str = "",
):
    rgb_inputs = {}
    cameras = {}
    for view_id in ORDERED_VIEWS:
        if state_id in LIGHTING_STATES:
            member = (
                f"Rectified/scan{scene_id}/rect_{view_id:03d}_{state_id[1:]}_r5000.png"
            )
        else:
            member = f"SyntheticFog/scan{scene_id}/{state_id}/view_{view_id:03d}.png"
        rgb_inputs[view_id] = _asset(
            member,
            f"{identity_namespace}rgb:{scene_id}:{state_id}:{view_id}",
        )
        cameras[view_id] = _asset(
            f"MVS Data/Calibration/cal18/pos_{view_id:03d}.txt",
            f"{identity_namespace}camera:{scene_id}:{view_id}",
        )

    clean_source_inputs = None
    if state_id.startswith("fog-"):
        clean_source_inputs = {
            view_id: _asset(
                f"Rectified/scan{scene_id}/rect_{view_id:03d}_3_r5000.png",
                f"{identity_namespace}rgb:{scene_id}:L3:{view_id}",
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
        gt_point_cloud=_asset(
            f"Points/stl/stl{scene_id:03d}_total.ply",
            f"{identity_namespace}gt:{scene_id}",
        ),
        observability_mask=_asset(
            f"MVS Data/ObsMask/ObsMask{scene_id}_10.mat",
            f"{identity_namespace}mask:{scene_id}",
        ),
        clean_source_inputs=clean_source_inputs,
    )


@lru_cache(maxsize=2)
def _state_inventory(identity_namespace: str = ""):
    return tuple(
        _state(
            scene_id,
            state_id,
            identity_namespace=identity_namespace,
        )
        for scene_id in TEST_SCENE_IDS
        for state_id in SCIENTIFIC_STATES
    )


@lru_cache(maxsize=2)
def _scientific_schedule(identity_namespace: str = "") -> ScientificSchedule:
    return build_scientific_schedule(_state_inventory(identity_namespace))


def _unit(
    model_id: str,
    scene_id: int,
    state_id: str,
    *,
    identity_namespace: str = "",
) -> ScientificExecutionUnit:
    return next(
        unit
        for unit in _scientific_schedule(identity_namespace).units
        if (
            unit.model_id == model_id
            and unit.scene_id == scene_id
            and unit.state_id == state_id
        )
    )


def _build_evidence(
    records,
    *,
    scientific_schedule=None,
    model_independent_states=None,
    native_warning_calibrations=None,
    split_assignment=None,
):
    return build_warning_evidence(
        records,
        scientific_schedule=(
            _scientific_schedule()
            if scientific_schedule is None
            else scientific_schedule
        ),
        model_independent_states=(
            _state_inventory()
            if model_independent_states is None
            else model_independent_states
        ),
        native_warning_calibrations=(
            _calibrations()
            if native_warning_calibrations is None
            else native_warning_calibrations
        ),
        split_assignment=(
            _split_assignment() if split_assignment is None else split_assignment
        ),
    )


def _rehashed_task_record(record, **updates):
    payload = record.to_dict()
    payload.update(updates)
    payload["record_sha256"] = canonical_json_sha256(
        {key: value for key, value in payload.items() if key != "record_sha256"}
    )
    return parse_task_audit_record(payload)


def _record_with_view_order(record, ordered_view_ids):
    pose_pairs = deepcopy(record.to_dict()["pose_pairs"])
    expected_pairs = tuple(
        (ordered_view_ids[first], ordered_view_ids[second])
        for first in range(8)
        for second in range(first + 1, 8)
    )
    for pose_pair, (view_a, view_b) in zip(
        pose_pairs,
        expected_pairs,
        strict=True,
    ):
        pose_pair["view_a"] = view_a
        pose_pair["view_b"] = view_b
    return _rehashed_task_record(
        record,
        ordered_view_ids=list(ordered_view_ids),
        pose_pairs=pose_pairs,
    )


@lru_cache(maxsize=1)
def _alternate_split_assignment():
    excluded_scene = _split_assignment().calibration[0]
    return construct_v4_splits(
        tuple(
            VerifiedSceneInventory(
                scene_id=scene_id,
                verified_complete=(scene_id != excluded_scene),
                inventory_sha256=_sha(f"alternate-inventory:{scene_id}"),
            )
            for scene_id in DTU_OFFICIAL_SCENE_IDS
        )
    )


def _point_metrics(loss: float, static_rank: float = 0.7) -> PointTaskMetrics:
    fscore = 1.0 - loss
    return PointTaskMetrics(
        point_main_loss=loss,
        fscore_1mm=max(0.0, fscore - 0.1),
        fscore_2mm=fscore,
        fscore_5mm=min(1.0, fscore + 0.1),
        median_predicted_error_mm=loss * 10.0,
        static_rank=static_rank,
        static_rank_defined=True,
        static_rank_reason_code="DEFINED",
        prediction_count=100,
        ground_truth_count=100,
    )


def _pose_metrics(pair_error: float) -> PoseTaskMetrics:
    pairs = tuple(
        PosePairMetrics(
            view_a=ORDERED_VIEWS[first],
            view_b=ORDERED_VIEWS[second],
            rotation_error_deg=pair_error,
            translation_direction_error_deg=pair_error,
            pair_error_deg=pair_error,
        )
        for first in range(8)
        for second in range(first + 1, 8)
    )
    errors = tuple(pair.pair_error_deg for pair in pairs)
    auc_5 = empirical_pose_auc(errors, 5.0)
    auc_10 = empirical_pose_auc(errors, 10.0)
    auc_20 = empirical_pose_auc(errors, 20.0)
    return PoseTaskMetrics(
        pairs=pairs,
        auc_5deg=auc_5,
        auc_10deg=auc_10,
        auc_20deg=auc_20,
        pose_main_loss=1.0 - auc_10,
        median_pair_error_deg=pair_error,
        pose_failure=pair_error > 10.0,
    )


@pytest.fixture(scope="module")
def full_records():
    records = []
    for model_index, model_id in enumerate(SCIENTIFIC_MODELS):
        calibration = _calibration(model_id)
        for scene_index, scene_id in enumerate(TEST_SCENE_IDS):
            for state_index, state_id in enumerate(SCIENTIFIC_STATES):
                point_loss = 0.1 + 0.03 * (state_index % 7)
                pose_error = float((scene_index + 2 * state_index + model_index) % 9)
                if scene_index < 8 and state_id == "L7":
                    pose_error = 12.0
                warning = 0.05 + 0.025 * state_index + 0.0001 * scene_index
                records.append(
                    build_task_audit_record(
                        execution_unit=_unit(model_id, scene_id, state_id),
                        calibration=calibration,
                        ordered_view_ids=ORDERED_VIEWS,
                        valid=True,
                        reason_code="VALID",
                        native_warning_score_value=warning,
                        point_metrics=_point_metrics(point_loss),
                        pose_metrics=_pose_metrics(pose_error),
                    )
                )
    return tuple(records)


def test_crr_constant_delta_is_zero_scientific_result_not_blocked() -> None:
    normal = counterfactual_rank_response([1.0, 2.0, 3.0], [3.0, 2.0, 1.0])
    constant = counterfactual_rank_response([0.0, 0.0, 0.0], [1.0, 2.0, 3.0])

    assert normal.value == pytest.approx(-1.0)
    assert normal.defined is True
    assert normal.reason_code == "DEFINED"
    assert constant.value == 0.0
    assert constant.defined is False
    assert constant.reason_code == "DEGENERATE_CONSTANT_DELTA_WARNING_OR_LOSS"
    assert relative_warning_gap(0.7, constant.value) == pytest.approx(0.7)


def test_sfr_and_boundary_lag_cover_every_deterministic_case() -> None:
    assert silent_failure_rate([True, True, False], [False, True, False]) == 0.5

    early = boundary_lag_for_scene(
        ("L3", "fog-s1", "fog-s2", "fog-s3"),
        alarms=(True, True, True, True),
        pose_failures=(False, False, True, True),
    )
    same = boundary_lag_for_scene(
        ("L3", "fog-s1", "fog-s2", "fog-s3"),
        alarms=(False, True, True, True),
        pose_failures=(False, True, True, True),
    )
    late = boundary_lag_for_scene(
        ("L3", "fog-s1", "fog-s2", "fog-s3"),
        alarms=(False, False, False, True),
        pose_failures=(False, True, True, True),
    )
    never = boundary_lag_for_scene(
        ("L3", "fog-s1", "fog-s2", "fog-s3"),
        alarms=(False, False, False, False),
        pose_failures=(False, False, True, True),
    )
    no_failure = boundary_lag_for_scene(
        ("L3", "fog-s1", "fog-s2", "fog-s3"),
        alarms=(False, False, True, True),
        pose_failures=(False, False, False, False),
    )

    assert (early.lag, early.classification) == (-2, "EARLY_WARNING")
    assert (same.lag, same.classification) == (0, "SAME_LEVEL")
    assert (late.lag, late.classification) == (2, "LATE_WARNING")
    assert never.alarm_level == 4
    assert (never.lag, never.classification) == (2, "NEVER_WARNING")
    assert no_failure.included is False
    assert no_failure.lag is None
    assert no_failure.classification == "NO_FAILURE"

    with pytest.raises(V4MetricError, match="frozen fog sequence"):
        boundary_lag_for_scene(
            ("L3", "L1", "L2", "L4"),
            alarms=(False, False, False, False),
            pose_failures=(False, True, True, True),
        )


def test_naurc_is_tie_safe_and_fails_closed_on_zero_denominator() -> None:
    ordered = normalized_aurc([0.0, 1.0], [0.0, 1.0])
    all_tied = normalized_aurc([0.5, 0.5], [0.0, 1.0])
    tied_reversed_input = normalized_aurc([0.5, 0.5], [1.0, 0.0])

    assert ordered.signal_aurc == pytest.approx(0.25)
    assert ordered.oracle_aurc == pytest.approx(0.25)
    assert ordered.random_aurc == pytest.approx(0.5)
    assert ordered.naurc == pytest.approx(0.0)
    assert all_tied.signal_aurc == pytest.approx(all_tied.random_aurc)
    assert all_tied.naurc == pytest.approx(1.0)
    assert tied_reversed_input == all_tied

    with pytest.raises(V4MetricError, match="denominator"):
        normalized_aurc([0.0, 1.0], [0.5, 0.5])


def test_scene_block_bootstrap_is_exact_deterministic_and_not_row_level() -> None:
    blocks = {
        1: (1.0, 1.0, 1.0),
        2: (3.0,),
        3: (5.0, 5.0),
    }

    first = scene_block_bootstrap(blocks, fmean)
    second = scene_block_bootstrap(blocks, fmean)

    assert first == second
    assert first.metadata.n_resamples == V4_BOOTSTRAP_RESAMPLES == 10_000
    assert first.metadata.seed == V4_BOOTSTRAP_SEED
    assert first.metadata.unit == "scene"
    assert first.metadata.repeated_runs_included is False
    assert first.estimate.n_scenes == 3
    assert first.estimate.bootstrap_draw_group == "PRIMARY_SCENE_BLOCK"
    assert len(first.draws) == 10_000


def test_holm_adjustment_requires_exact_four_model_family_hypotheses() -> None:
    reverse_counts = {
        "VGGT:ranking-warning": 99,
        "VGGT:task-transfer": 199,
        "MASt3R:ranking-warning": 299,
        "MASt3R:task-transfer": 1999,
    }
    adjusted = holm_adjust_four(reverse_counts)

    assert tuple(adjusted) == FAMILY_HYPOTHESES
    denominator = V4_BOOTSTRAP_RESAMPLES + 1
    assert adjusted["VGGT:ranking-warning"].adjusted_p == pytest.approx(
        4 * 100 / denominator
    )
    assert adjusted["VGGT:task-transfer"].adjusted_p == pytest.approx(
        3 * 200 / denominator
    )
    assert adjusted["MASt3R:ranking-warning"].adjusted_p == pytest.approx(
        2 * 300 / denominator
    )
    assert adjusted["MASt3R:task-transfer"].adjusted_p == pytest.approx(
        2000 / denominator
    )
    assert adjusted["VGGT:ranking-warning"].reverse_effect_draw_count == 99
    assert adjusted["VGGT:ranking-warning"].bootstrap_draw_count == 10_000
    assert adjusted["VGGT:ranking-warning"].non_reversal_excluded is True
    assert adjusted["VGGT:task-transfer"].non_reversal_excluded is False

    with pytest.raises(Task3ContractError, match="exact four"):
        holm_adjust_four({"VGGT:ranking-warning": 99})


def test_full_evidence_uses_400_unique_records_and_exact_10000_bootstraps(
    full_records,
    tmp_path: Path,
) -> None:
    evidence = _build_evidence(full_records)
    rerun = _build_evidence(tuple(reversed(full_records)))

    assert evidence.input_record_count == 400
    assert len(evidence.input_record_inventory_sha256) == 64
    assert evidence.bootstrap_metadata.n_resamples == 10_000
    assert evidence.bootstrap_metadata.unit == "scene"
    assert evidence == rerun
    assert tuple(model.model_id for model in evidence.models) == SCIENTIFIC_MODELS
    assert all(model.pose_failure_scene_count == 8 for model in evidence.models)
    assert all(model.static_rank.point_estimate == 0.7 for model in evidence.models)
    assert all(model.sfr_pose.point_estimate == 1.0 for model in evidence.models)
    assert set(item.hypothesis_id for item in evidence.holm) == set(FAMILY_HYPOTHESES)

    parsed = parse_warning_evidence(evidence.canonical_json_bytes())
    assert parsed == evidence
    path = tmp_path / "warning-evidence.json"
    written_sha = write_warning_evidence(path, evidence)
    assert written_sha == hashlib.sha256(path.read_bytes()).hexdigest()
    assert read_warning_evidence(path) == evidence

    duplicate = (*full_records, full_records[0])
    with pytest.raises(Task3ContractError, match="duplicate|repeat"):
        _build_evidence(duplicate)


def test_evidence_rejects_shared_wrong_ordered_view_identity(full_records) -> None:
    reversed_views = list(reversed(ORDERED_VIEWS))
    wrong_records = tuple(
        _record_with_view_order(record, reversed_views)
        if record.scene_id == TEST_SCENE_IDS[0] and record.state_id == "L1"
        else record
        for record in full_records
    )

    with pytest.raises(Task3ContractError, match="ordered view identity"):
        _build_evidence(wrong_records)


def test_evidence_requires_exact_authoritative_calibration_binding(
    full_records,
) -> None:
    with pytest.raises(Task3ContractError, match="calibration.*required"):
        build_warning_evidence(
            full_records,
            scientific_schedule=_scientific_schedule(),
            model_independent_states=_state_inventory(),
        )

    fake_identifier = _sha("fabricated-calibration-identifier")
    fake_identifier_records = tuple(
        _rehashed_task_record(record, calibration_identifier=fake_identifier)
        if record.model_id == "VGGT"
        else record
        for record in full_records
    )
    with pytest.raises(Task3ContractError, match="calibration.*identifier"):
        _build_evidence(fake_identifier_records)

    drifted_threshold = _calibration("MASt3R").alarm_threshold + 0.05
    drifted_threshold_records = tuple(
        _rehashed_task_record(
            record,
            alarm_threshold=drifted_threshold,
            alarm=record.native_warning_score >= drifted_threshold,
        )
        if record.model_id == "MASt3R"
        else record
        for record in full_records
    )
    with pytest.raises(Task3ContractError, match="calibration.*threshold"):
        _build_evidence(drifted_threshold_records)

    with pytest.raises(Task3ContractError, match="calibration.*split"):
        _build_evidence(
            full_records,
            split_assignment=_alternate_split_assignment(),
        )


def test_warning_evidence_closed_schema_digest_v1_and_nonoverwrite(
    full_records,
    tmp_path: Path,
) -> None:
    evidence = _build_evidence(full_records)
    payload = evidence.to_dict()

    extra = deepcopy(payload)
    extra["p3_evidence"] = {"scientific_validity": "SCIENTIFIC"}
    with pytest.raises(Task3ContractError, match="closed schema|unexpected"):
        parse_warning_evidence(extra)

    tampered = deepcopy(payload)
    tampered["models"][0]["static_rank"]["point_estimate"] = -1.0
    with pytest.raises(Task3ContractError, match="tamper|digest"):
        parse_warning_evidence(tampered)

    with pytest.raises(Task3ContractError, match="schema|evidence"):
        parse_warning_evidence(
            {
                "scientific_validity": "SCIENTIFIC",
                "bundle_index": "old-v1-P2.json",
            }
        )

    path = tmp_path / "warning-evidence.json"
    path.write_bytes(b"concurrent-writer\n")
    with pytest.raises(Task3ContractError, match="conflict"):
        write_warning_evidence(path, evidence)
    assert path.read_bytes() == b"concurrent-writer\n"


def test_rehashed_partial_origin_or_schedule_identity_bypass_is_blocked(
    full_records,
) -> None:
    evidence = _build_evidence(full_records)

    partial = evidence.to_dict()
    partial["input_record_count"] = 399
    partial["evidence_sha256"] = canonical_json_sha256(
        {key: value for key, value in partial.items() if key != "evidence_sha256"}
    )
    with pytest.raises(Task3ContractError, match="count must equal 400"):
        parse_warning_evidence(partial)
    assert (
        _bound_public_gate(partial, full_records).status
        == "MVE_BLOCKED_EVIDENCE_CONTRACT"
    )

    origin_mismatch = evidence.to_dict()
    origin_mismatch["primary_evidence_origin"] = "SYNTHETIC_FOG_ONLY"
    origin_mismatch["evidence_sha256"] = canonical_json_sha256(
        {
            key: value
            for key, value in origin_mismatch.items()
            if key != "evidence_sha256"
        }
    )
    with pytest.raises(Task3ContractError, match="states.*origin"):
        parse_warning_evidence(origin_mismatch)

    with pytest.raises(Task3ContractError, match="ScientificSchedule unit"):
        _build_evidence(
            full_records,
            scientific_schedule=_scientific_schedule("alternate:"),
            model_independent_states=_state_inventory("alternate:"),
        )


def _bound_public_gate(value, records):
    return evaluate_warning_gate(
        value,
        scientific_schedule=_scientific_schedule(),
        model_independent_states=_state_inventory(),
        native_warning_calibrations=_calibrations(),
        split_assignment=_split_assignment(),
        task_records=records,
    )


def _estimate(
    point: float,
    *,
    lower: float | None = None,
    upper: float | None = None,
    n_scenes: int = 20,
    defined: bool = True,
    reason_code: str = "DEFINED",
) -> MetricEstimate:
    return MetricEstimate(
        point_estimate=point,
        ci_lower=point if lower is None else lower,
        ci_upper=point if upper is None else upper,
        n_scenes=n_scenes,
        defined=defined,
        reason_code=reason_code,
        bootstrap_draw_group="PRIMARY_SCENE_BLOCK",
    )


def _model_evidence(
    model_id: str,
    *,
    static_point: float = 0.5,
    static_lower: float = 0.35,
    crr_pose_point: float = 0.0,
    crr_pose_upper: float = 0.15,
    rwg_lower: float = 0.20,
    sfr_lower: float = 0.30,
    naurc_point_upper: float = 0.8,
    ttg_point: float = 0.0,
    ttg_lower: float = 0.0,
    failure_scenes: int = 8,
    calibration_identifier: str | None = None,
) -> ModelWarningEvidence:
    return ModelWarningEvidence.create(
        model_id=model_id,
        calibration_identifier=(
            _calibration(model_id).calibration_identifier
            if calibration_identifier is None
            else calibration_identifier
        ),
        static_rank=_estimate(
            static_point,
            lower=static_lower,
            upper=0.7,
        ),
        crr_point=_estimate(0.4, lower=0.2, upper=0.6),
        crr_pose=_estimate(
            crr_pose_point,
            lower=-0.1,
            upper=crr_pose_upper,
        ),
        rwg_pose=_estimate(
            static_point - crr_pose_point,
            lower=rwg_lower,
            upper=0.5,
        ),
        sfr_pose=_estimate(
            0.5,
            lower=sfr_lower,
            upper=0.8,
            n_scenes=failure_scenes,
        ),
        boundary_lag=_estimate(1.0, lower=0.0, upper=2.0),
        naurc_point=_estimate(0.6, lower=0.3, upper=naurc_point_upper),
        naurc_pose=_estimate(0.6 + ttg_point, lower=0.4, upper=1.0),
        ttg_pose=_estimate(ttg_point, lower=ttg_lower, upper=0.5),
        pose_failure_scene_count=failure_scenes,
        fog_no_failure_scene_count=0,
    )


def _holm(
    *,
    ranking_other_passes: bool = True,
    transfer_other_passes: bool = True,
) -> tuple[HolmEvidence, ...]:
    reverse_counts = {hypothesis_id: 0 for hypothesis_id in FAMILY_HYPOTHESES}
    for hypothesis_id in FAMILY_HYPOTHESES:
        other_ranking = (
            hypothesis_id == "MASt3R:ranking-warning" and not ranking_other_passes
        )
        other_transfer = (
            hypothesis_id == "MASt3R:task-transfer" and not transfer_other_passes
        )
        if other_ranking or other_transfer:
            reverse_counts[hypothesis_id] = 2_000
    adjusted = holm_adjust_four(reverse_counts)
    return tuple(adjusted[item] for item in FAMILY_HYPOTHESES)


def _manual_evidence(
    *,
    vggt: ModelWarningEvidence | None = None,
    mast3r: ModelWarningEvidence | None = None,
    primary_origin: str = "REAL_DTU_LIGHTING",
    holm: tuple[HolmEvidence, ...] | None = None,
):
    primary_states = (
        ("L3", "fog-s1", "fog-s2", "fog-s3")
        if primary_origin == "SYNTHETIC_FOG_ONLY"
        else LIGHTING_STATES
    )
    return build_warning_evidence_record(
        primary_evidence_origin=primary_origin,
        primary_state_ids=primary_states,
        scientific_schedule_sha256=(_scientific_schedule().schedule_sha256),
        input_record_inventory_sha256="a" * 64,
        input_record_count=400,
        bootstrap_metadata=BootstrapMetadata.frozen(),
        models=(
            _model_evidence("VGGT") if vggt is None else vggt,
            _model_evidence(
                "MASt3R",
                static_point=0.0,
                static_lower=0.0,
                rwg_lower=-0.05,
            )
            if mast3r is None
            else mast3r,
        ),
        holm=_holm() if holm is None else holm,
    )


def _threshold_decision(value):
    return _evaluate_rebuilt_warning_evidence(value)


def test_gate_requires_exact_task2_schedule_and_holm_gap_consistency() -> None:
    evidence = _manual_evidence()
    assert (
        evaluate_warning_gate(evidence).status == MVE_BLOCKED_SCHEDULE_BINDING_REQUIRED
    )
    assert (
        evaluate_warning_gate(
            evidence,
            scientific_schedule=_scientific_schedule("alternate:"),
        ).status
        == MVE_BLOCKED_SCHEDULE_BINDING_REQUIRED
    )
    assert (
        evaluate_warning_gate(
            evidence,
            scientific_schedule=_scientific_schedule(),
            model_independent_states=_state_inventory(),
        ).status
        == MVE_BLOCKED_CALIBRATION_BINDING_REQUIRED
    )
    assert (
        evaluate_warning_gate(
            evidence,
            scientific_schedule=_scientific_schedule(),
            native_warning_calibrations=_calibrations(),
            split_assignment=_split_assignment(),
        ).status
        == MVE_BLOCKED_SCHEDULE_BINDING_REQUIRED
    )
    assert (
        evaluate_warning_gate(
            evidence,
            scientific_schedule=_scientific_schedule(),
            model_independent_states=_state_inventory("alternate:"),
            native_warning_calibrations=_calibrations(),
            split_assignment=_split_assignment(),
        ).status
        == MVE_BLOCKED_SCHEDULE_BINDING_REQUIRED
    )

    fake_calibration_evidence = _manual_evidence(
        vggt=_model_evidence(
            "VGGT",
            calibration_identifier=_sha("fake-gate-calibration"),
        )
    )
    assert (
        evaluate_warning_gate(
            fake_calibration_evidence,
            scientific_schedule=_scientific_schedule(),
            model_independent_states=_state_inventory(),
            native_warning_calibrations=_calibrations(),
            split_assignment=_split_assignment(),
        ).status
        == MVE_BLOCKED_CALIBRATION_BINDING_REQUIRED
    )
    drifted_calibrations = (
        _fit_calibration(
            "VGGT",
            _split_assignment(),
            warning_offset=0.05,
        ),
        _calibration("MASt3R"),
    )
    assert (
        evaluate_warning_gate(
            evidence,
            scientific_schedule=_scientific_schedule(),
            model_independent_states=_state_inventory(),
            native_warning_calibrations=drifted_calibrations,
            split_assignment=_split_assignment(),
        ).status
        == MVE_BLOCKED_CALIBRATION_BINDING_REQUIRED
    )
    assert (
        evaluate_warning_gate(
            evidence,
            scientific_schedule=_scientific_schedule(),
            model_independent_states=_state_inventory(),
            native_warning_calibrations=_calibrations(),
            split_assignment=_alternate_split_assignment(),
        ).status
        == MVE_BLOCKED_CALIBRATION_BINDING_REQUIRED
    )

    with pytest.raises(Task3ContractError, match="gap confidence interval"):
        _manual_evidence(
            mast3r=_model_evidence(
                "MASt3R",
                static_point=0.0,
                static_lower=0.0,
                rwg_lower=-0.20,
            )
        )


def test_public_gate_rebuilds_exact_source_records_before_thresholds(
    full_records,
) -> None:
    evidence = _build_evidence(full_records)
    admitted = _bound_public_gate(evidence, full_records)
    assert not admitted.status.startswith("MVE_BLOCKED_")

    different_states = evaluate_warning_gate(
        evidence,
        scientific_schedule=_scientific_schedule(),
        model_independent_states=_state_inventory("alternate:"),
        native_warning_calibrations=_calibrations(),
        split_assignment=_split_assignment(),
        task_records=full_records,
    )
    assert different_states.status == MVE_BLOCKED_SCHEDULE_BINDING_REQUIRED

    different_calibrations = evaluate_warning_gate(
        evidence,
        scientific_schedule=_scientific_schedule(),
        model_independent_states=_state_inventory(),
        native_warning_calibrations=(
            _fit_calibration(
                "VGGT",
                _split_assignment(),
                warning_offset=0.05,
            ),
            _calibration("MASt3R"),
        ),
        split_assignment=_split_assignment(),
        task_records=full_records,
    )
    assert different_calibrations.status == MVE_BLOCKED_CALIBRATION_BINDING_REQUIRED

    missing_records = evaluate_warning_gate(
        evidence,
        scientific_schedule=_scientific_schedule(),
        model_independent_states=_state_inventory(),
        native_warning_calibrations=_calibrations(),
        split_assignment=_split_assignment(),
    )
    assert missing_records.status == MVE_BLOCKED_SOURCE_RECORD_BINDING_REQUIRED
    assert missing_records.reason_code == "TASK3_SOURCE_TASK_RECORDS_REQUIRED"

    invalid_records = (*full_records[:-1], full_records[0])
    invalid = _bound_public_gate(evidence, invalid_records)
    assert invalid.status == MVE_BLOCKED_SOURCE_RECORD_BINDING_REQUIRED
    assert invalid.reason_code == "TASK3_SOURCE_TASK_RECORDS_INVALID"

    rehashed_metric = evidence.to_dict()
    lag = rehashed_metric["models"][0]["boundary_lag"]
    lag["point_estimate"] += 0.5
    lag["ci_lower"] += 0.5
    lag["ci_upper"] += 0.5
    rehashed_metric["evidence_sha256"] = canonical_json_sha256(
        {
            key: value
            for key, value in rehashed_metric.items()
            if key != "evidence_sha256"
        }
    )
    mismatch = _bound_public_gate(rehashed_metric, full_records)
    assert mismatch.status == MVE_BLOCKED_SOURCE_RECORD_BINDING_REQUIRED
    assert mismatch.reason_code == "TASK3_WARNING_EVIDENCE_REBUILD_MISMATCH"


def test_frozen_gate_accepts_exact_ranking_boundaries_one_strong_directional() -> None:
    decision = _threshold_decision(_manual_evidence())

    assert decision.status == MVE_GO_TO_EXTERNAL_VALIDATION
    assert decision.strong_model_id == "VGGT"
    assert decision.strong_family == "ranking-warning"


@pytest.mark.parametrize(
    "changed",
    (
        {"static_lower": math.nextafter(0.35, -math.inf)},
        {"crr_pose_upper": math.nextafter(0.15, math.inf)},
        {"rwg_lower": math.nextafter(0.20, -math.inf)},
        {"sfr_lower": math.nextafter(0.30, -math.inf)},
    ),
    ids=("static", "crr-pose", "rwg", "sfr"),
)
def test_ranking_gate_fails_immediately_outside_each_boundary(changed) -> None:
    evidence = _manual_evidence(vggt=_model_evidence("VGGT", **changed))
    assert _threshold_decision(evidence).status == MVE_SCIENTIFIC_NO_GO


def test_transfer_gate_exact_boundaries_and_each_failure() -> None:
    strong_transfer = _model_evidence(
        "VGGT",
        static_lower=0.0,
        rwg_lower=0.0,
        naurc_point_upper=0.50,
        ttg_point=0.20,
        ttg_lower=0.20,
    )
    directional_other = _model_evidence(
        "MASt3R",
        static_point=0.0,
        static_lower=0.0,
        rwg_lower=-0.05,
        naurc_point_upper=0.8,
        ttg_point=0.0,
        ttg_lower=-0.05,
    )
    evidence = _manual_evidence(vggt=strong_transfer, mast3r=directional_other)
    decision = _threshold_decision(evidence)
    assert decision.status == MVE_GO_TO_EXTERNAL_VALIDATION
    assert decision.strong_family == "task-transfer"

    bad_naurc = _manual_evidence(
        vggt=_model_evidence(
            "VGGT",
            static_lower=0.0,
            rwg_lower=0.0,
            naurc_point_upper=math.nextafter(0.50, math.inf),
            ttg_point=0.20,
            ttg_lower=0.20,
        ),
        mast3r=directional_other,
    )
    bad_ttg = _manual_evidence(
        vggt=_model_evidence(
            "VGGT",
            static_lower=0.0,
            rwg_lower=0.0,
            naurc_point_upper=0.50,
            ttg_point=0.20,
            ttg_lower=math.nextafter(0.20, -math.inf),
        ),
        mast3r=directional_other,
    )
    assert _threshold_decision(bad_naurc).status == MVE_SCIENTIFIC_NO_GO
    assert _threshold_decision(bad_ttg).status == MVE_SCIENTIFIC_NO_GO


def test_gate_rejects_reverse_other_holm_failure_synthetic_and_low_endpoint() -> None:
    reverse = _manual_evidence(
        mast3r=_model_evidence(
            "MASt3R",
            static_point=math.nextafter(0.0, -math.inf),
            static_lower=-0.05,
            rwg_lower=-0.05,
        )
    )
    holm_failure = _manual_evidence(holm=_holm(ranking_other_passes=False))
    synthetic = _manual_evidence(primary_origin="SYNTHETIC_FOG_ONLY")
    low_endpoint = _manual_evidence(vggt=_model_evidence("VGGT", failure_scenes=7))

    assert _threshold_decision(reverse).status == MVE_SCIENTIFIC_NO_GO
    assert _threshold_decision(holm_failure).status == MVE_SCIENTIFIC_NO_GO
    assert _threshold_decision(synthetic).status == MVE_SCIENTIFIC_NO_GO
    assert _threshold_decision(low_endpoint).status == MVE_BLOCKED_ENDPOINT


def test_tampered_or_v1_gate_input_returns_specific_blocked_status(
    full_records,
) -> None:
    payload = _build_evidence(full_records).to_dict()
    payload["models"][0]["rwg_pose"]["point_estimate"] = -1.0
    tampered = evaluate_warning_gate(payload)
    legacy = evaluate_warning_gate(
        {
            "scientific_validity": "SCIENTIFIC",
            "bundle_index": "old-v1-P3.json",
        }
    )

    assert tampered.status == "MVE_BLOCKED_EVIDENCE_TAMPER"
    assert legacy.status == "MVE_BLOCKED_V1_EVIDENCE"
