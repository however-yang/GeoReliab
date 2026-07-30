from __future__ import annotations

from copy import deepcopy
from functools import lru_cache
import hashlib
import math
from pathlib import Path

import numpy as np
import pytest

from georeliab_mve.v4_counterfactuals import (
    DTU_OFFICIAL_SCENE_IDS,
    ScientificExecutionUnit,
    VerifiedSceneInventory,
    canonical_json_sha256,
    construct_v4_splits,
)
from georeliab_mve.v4_metrics import (
    CalibrationWarningSample,
    V4MetricError,
    compute_point_task_metrics,
    compute_relative_pose_metrics,
    fit_native_warning_calibration,
    native_warning_score,
)
from georeliab_mve.v4_records import (
    TASK_AUDIT_RECORD_SCHEMA_VERSION,
    Task3ContractError,
    build_task_audit_record,
    parse_task_audit_record,
    read_task_audit_record,
    write_task_audit_record,
)
from georeliab_mve.v4_science_lock import (
    V4_PROTOCOL_ID,
    V4_PROTOCOL_SHA256,
    V4_PROTOCOL_VERSION,
)


ORDERED_VIEWS = (1, 7, 13, 19, 25, 31, 37, 43)


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


def _calibration_sample(
    model_id: str,
    scene_id: int,
    state_id: str,
    warning_score: float,
    *,
    split_assignment=None,
):
    split = _split_assignment() if split_assignment is None else split_assignment
    return CalibrationWarningSample(
        model_id=model_id,
        scene_id=scene_id,
        state_id=state_id,
        warning_score=warning_score,
        split_fingerprint_sha256=split.fingerprint_sha256,
        inventory_sha256=split.inventory_sha256,
    )


def _unit(
    *,
    model_id: str = "VGGT",
    scene_id: int = 1,
    state_id: str = "L1",
) -> ScientificExecutionUnit:
    provenance = {
        "schema_version": "georeliab-v4-protocol-provenance-1.0",
        "protocol_id": V4_PROTOCOL_ID,
        "protocol_version": V4_PROTOCOL_VERSION,
        "protocol_sha256": V4_PROTOCOL_SHA256,
    }
    payload = {
        "schema_version": "georeliab-v4-scientific-execution-unit-1.0",
        "protocol_provenance": provenance,
        "dataset": "DTU",
        "model_id": model_id,
        "scene_id": scene_id,
        "state_id": state_id,
        "state_identity_sha256": _sha(f"state:{scene_id}:{state_id}"),
        "pair_identity_sha256": (
            None if state_id == "L3" else _sha(f"pair:{scene_id}:{state_id}")
        ),
    }
    return ScientificExecutionUnit.from_dict(
        {
            **payload,
            "execution_unit_sha256": canonical_json_sha256(payload),
        }
    )


def _calibration(model_id: str = "VGGT"):
    split = _split_assignment()
    samples = tuple(
        _calibration_sample(
            model_id=model_id,
            scene_id=scene_id,
            state_id="L3",
            warning_score=float(index),
        )
        for index, scene_id in enumerate(split.calibration)
    )
    return fit_native_warning_calibration(samples, split_assignment=split)


def _poses() -> np.ndarray:
    poses = np.repeat(np.eye(4, dtype=np.float64)[None, :, :], 8, axis=0)
    poses[:, :3, 3] = np.asarray(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 1.0, 0.5],
            [-1.0, 0.5, 1.0],
            [0.5, -1.0, 1.5],
            [1.5, 0.5, -0.5],
        ],
        dtype=np.float64,
    )
    return poses


def _rotation_z(degrees: float) -> np.ndarray:
    radians = math.radians(degrees)
    cosine = math.cos(radians)
    sine = math.sin(radians)
    return np.asarray(
        [[cosine, -sine, 0.0], [sine, cosine, 0.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )


def _sim3_transform(poses: np.ndarray) -> np.ndarray:
    rotation = _rotation_z(37.0)
    scale = 2.5
    translation = np.asarray([3.0, -4.0, 1.0])
    transformed = poses.copy()
    transformed[:, :3, :3] = np.einsum("ij,njk->nik", rotation, poses[:, :3, :3])
    transformed[:, :3, 3] = (
        scale * np.einsum("ij,nj->ni", rotation, poses[:, :3, 3]) + translation
    )
    return transformed


def test_point_metrics_use_bidirectional_fscore_without_confidence_filtering() -> None:
    predicted = np.asarray([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]])
    ground_truth = np.asarray([[0.0, 0.0, 0.0], [1.5, 0.0, 0.0]])
    risk = np.asarray([0.1, 0.9])

    result = compute_point_task_metrics(predicted, ground_truth, risk)

    # At 2 mm: precision=1/2 and recall=2/2, hence F=2/3.
    assert result.fscore_2mm == pytest.approx(2.0 / 3.0)
    assert result.point_main_loss == pytest.approx(1.0 / 3.0)
    assert result.fscore_1mm == pytest.approx(0.5)
    assert result.fscore_5mm == pytest.approx(2.0 / 3.0)
    assert result.median_predicted_error_mm == pytest.approx(4.25)
    assert result.static_rank == pytest.approx(1.0)
    assert result.prediction_count == 2
    assert result.ground_truth_count == 2


def test_constant_native_risk_is_zero_rank_with_degenerate_diagnostic() -> None:
    result = compute_point_task_metrics(
        np.asarray([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]]),
        np.asarray([[0.0, 0.0, 0.0], [1.5, 0.0, 0.0]]),
        np.asarray([0.5, 0.5]),
    )

    assert result.static_rank == 0.0
    assert result.static_rank_defined is False
    assert result.static_rank_reason_code == "DEGENERATE_CONSTANT_RISK_OR_ERROR"


def test_native_warning_is_median_of_per_view_linear_q90() -> None:
    risks = {
        view_id: [float(index), float(index + 10)]
        for index, view_id in enumerate(ORDERED_VIEWS)
    }
    # Each Q90 is index+9; median of 9..16 is 12.5.
    assert native_warning_score(risks, ordered_view_ids=ORDERED_VIEWS) == 12.5

    with pytest.raises(V4MetricError, match="exact ordered eight-view"):
        native_warning_score(
            {view_id: risks[view_id] for view_id in ORDERED_VIEWS[:-1]},
            ordered_view_ids=ORDERED_VIEWS,
        )


def test_calibration_is_exactly_twenty_model_specific_l3_scenes_and_frozen() -> None:
    calibration = _calibration()
    split = _split_assignment()

    assert calibration.model_id == "VGGT"
    assert calibration.scene_ids == split.calibration
    assert calibration.split_fingerprint_sha256 == split.fingerprint_sha256
    assert calibration.inventory_sha256 == split.inventory_sha256
    assert calibration.sorted_warning_scores == tuple(float(i) for i in range(20))
    assert calibration.alarm_threshold == pytest.approx(17.1)
    assert len(calibration.calibration_identifier) == 64
    assert calibration.alarm_for(17.1) is True
    assert calibration.alarm_for(math.nextafter(17.1, -math.inf)) is False

    wrong_state = [
        _calibration_sample("VGGT", scene, "L3", float(index))
        for index, scene in enumerate(split.calibration)
    ]
    wrong_state[0] = _calibration_sample("VGGT", 82, "L2", 0.0)
    with pytest.raises(V4MetricError, match="L3"):
        fit_native_warning_calibration(wrong_state, split_assignment=split)

    overlap = [
        _calibration_sample("VGGT", scene, "L3", float(index))
        for index, scene in enumerate((1, *split.calibration[1:]))
    ]
    with pytest.raises(V4MetricError, match="test overlap|test scene"):
        fit_native_warning_calibration(overlap, split_assignment=split)

    mixed = [
        _calibration_sample(
            "MASt3R" if index == 0 else "VGGT",
            scene,
            "L3",
            float(index),
        )
        for index, scene in enumerate(split.calibration)
    ]
    with pytest.raises(V4MetricError, match="mixed models"):
        fit_native_warning_calibration(mixed, split_assignment=split)

    with pytest.raises(V4MetricError, match="exactly 20"):
        fit_native_warning_calibration(
            tuple(
                _calibration_sample("VGGT", scene, "L3", float(index))
                for index, scene in enumerate(split.calibration[:-1])
            ),
            split_assignment=split,
        )


def test_calibration_rejects_a_different_validated_inventory_split() -> None:
    split = _split_assignment()
    wrong_inventory = construct_v4_splits(
        tuple(
            VerifiedSceneInventory(
                scene_id=scene_id,
                verified_complete=scene_id != 2,
                inventory_sha256=_sha(f"other-inventory:{scene_id}"),
            )
            for scene_id in DTU_OFFICIAL_SCENE_IDS
        )
    )
    samples = tuple(
        _calibration_sample("VGGT", scene, "L3", float(index))
        for index, scene in enumerate(split.calibration)
    )

    with pytest.raises(V4MetricError, match="exact calibration split"):
        fit_native_warning_calibration(
            samples,
            split_assignment=wrong_inventory,
        )


def test_pose_uses_all_twenty_eight_pairs_and_known_geodesic_angle() -> None:
    ground_truth = _poses()
    predicted = ground_truth.copy()
    predicted[1, :3, :3] = _rotation_z(30.0)

    result = compute_relative_pose_metrics(
        predicted,
        ground_truth,
        ordered_view_ids=ORDERED_VIEWS,
    )

    assert len(result.pairs) == 28
    assert result.pairs[0].view_a == ORDERED_VIEWS[0]
    assert result.pairs[0].view_b == ORDERED_VIEWS[1]
    assert result.pairs[0].rotation_error_deg == pytest.approx(30.0)
    assert result.pairs[0].translation_direction_error_deg == pytest.approx(0.0)
    assert result.pairs[0].pair_error_deg == pytest.approx(30.0)
    assert 0.0 <= result.auc_5deg <= result.auc_10deg <= result.auc_20deg <= 1.0
    assert result.pose_main_loss == pytest.approx(1.0 - result.auc_10deg)


def test_pose_metric_is_invariant_to_shared_reflection_free_sim3() -> None:
    ground_truth = _poses()
    predicted = ground_truth.copy()
    predicted[2, :3, :3] = _rotation_z(12.0)
    predicted[4, :3, 3] += np.asarray([0.2, -0.1, 0.3])

    baseline = compute_relative_pose_metrics(
        predicted,
        ground_truth,
        ordered_view_ids=ORDERED_VIEWS,
    )
    transformed = compute_relative_pose_metrics(
        _sim3_transform(predicted),
        _sim3_transform(ground_truth),
        ordered_view_ids=ORDERED_VIEWS,
    )

    assert [pair.pair_error_deg for pair in transformed.pairs] == pytest.approx(
        [pair.pair_error_deg for pair in baseline.pairs],
        abs=1e-8,
    )
    assert transformed.auc_10deg == pytest.approx(baseline.auc_10deg)
    assert transformed.median_pair_error_deg == pytest.approx(
        baseline.median_pair_error_deg
    )


def test_pose_degeneracy_nonfinite_and_reflections_fail_closed() -> None:
    valid = _poses()

    degenerate = valid.copy()
    degenerate[1, :3, 3] = degenerate[0, :3, 3]
    with pytest.raises(V4MetricError, match="degenerate"):
        compute_relative_pose_metrics(
            degenerate,
            valid,
            ordered_view_ids=ORDERED_VIEWS,
        )

    nonfinite = valid.copy()
    nonfinite[0, 0, 0] = math.nan
    with pytest.raises(V4MetricError, match="finite"):
        compute_relative_pose_metrics(
            nonfinite,
            valid,
            ordered_view_ids=ORDERED_VIEWS,
        )

    reflected = valid.copy()
    reflected[0, :3, :3] = np.diag([-1.0, 1.0, 1.0])
    with pytest.raises(V4MetricError, match="proper rotation|reflection"):
        compute_relative_pose_metrics(
            reflected,
            valid,
            ordered_view_ids=ORDERED_VIEWS,
        )


def test_task_record_is_closed_v4_bound_and_retains_invalid_outputs(
    tmp_path: Path,
) -> None:
    record = build_task_audit_record(
        execution_unit=_unit(),
        calibration=_calibration(),
        ordered_view_ids=ORDERED_VIEWS,
        valid=False,
        reason_code="INVALID_NONFINITE_MODEL_OUTPUT",
        native_warning_score_value=18.0,
    )

    assert record.schema_version == TASK_AUDIT_RECORD_SCHEMA_VERSION
    assert record.protocol_provenance["protocol_sha256"] == V4_PROTOCOL_SHA256
    assert record.valid is False
    assert record.point_main_loss == 1.0
    assert record.fscore_1mm == record.fscore_2mm == record.fscore_5mm == 0.0
    assert record.pose_main_loss == 1.0
    assert record.auc_5deg == record.auc_10deg == record.auc_20deg == 0.0
    assert record.pose_failure is True
    assert len(record.pose_pairs) == 28
    assert all(pair.pair_error_deg == 180.0 for pair in record.pose_pairs)
    assert math.isfinite(record.native_warning_score)
    assert record.alarm is True

    parsed = parse_task_audit_record(record.canonical_json_bytes())
    assert parsed == record
    assert set(record.to_dict()) == {
        "schema_version",
        "protocol_provenance",
        "dataset",
        "model_id",
        "scene_id",
        "state_id",
        "ordered_view_ids",
        "state_identity_sha256",
        "pair_identity_sha256",
        "execution_unit_sha256",
        "valid",
        "reason_code",
        "point_main_loss",
        "fscore_1mm",
        "fscore_2mm",
        "fscore_5mm",
        "median_predicted_error_mm",
        "static_rank",
        "static_rank_defined",
        "static_rank_reason_code",
        "pose_pairs",
        "auc_5deg",
        "auc_10deg",
        "auc_20deg",
        "pose_main_loss",
        "median_pair_error_deg",
        "pose_failure",
        "native_warning_score",
        "calibration_identifier",
        "alarm_threshold",
        "alarm",
        "record_sha256",
    }
    text = record.canonical_json_bytes().decode("ascii")
    for forbidden in ("selection_score", "accepted", "learned_head", "P2", "P3"):
        assert forbidden not in text

    path = tmp_path / "record.json"
    written_sha = write_task_audit_record(path, record)
    assert written_sha == hashlib.sha256(path.read_bytes()).hexdigest()
    assert read_task_audit_record(path) == record


def test_task_record_digest_tamper_v1_and_nonoverwrite_rejection(
    tmp_path: Path,
) -> None:
    record = build_task_audit_record(
        execution_unit=_unit(),
        calibration=_calibration(),
        ordered_view_ids=ORDERED_VIEWS,
        valid=False,
        reason_code="INVALID_MODEL_OUTPUT",
        native_warning_score_value=18.0,
    )
    payload = record.to_dict()

    extra = deepcopy(payload)
    extra["selection_score"] = 0.9
    with pytest.raises(Task3ContractError, match="closed schema|unexpected"):
        parse_task_audit_record(extra)

    tampered = deepcopy(payload)
    tampered["point_main_loss"] = 0.0
    with pytest.raises(Task3ContractError, match="tamper|digest"):
        parse_task_audit_record(tampered)

    old_v1 = {
        "run_id": "legacy",
        "sample_key": "scan1",
        "failure_label": True,
        "selection_score": 0.5,
        "coverage": 1.0,
        "accepted": True,
        "downstream_outcome": 1.0,
    }
    with pytest.raises(Task3ContractError, match="schema|record"):
        parse_task_audit_record(old_v1)

    path = tmp_path / "record.json"
    path.write_bytes(b"concurrent-writer\n")
    with pytest.raises(Task3ContractError, match="conflict"):
        write_task_audit_record(path, record)
    assert path.read_bytes() == b"concurrent-writer\n"


def test_invalid_output_without_finite_native_warning_is_blocked_evidence() -> None:
    with pytest.raises(
        Task3ContractError,
        match="MVE_BLOCKED_NATIVE_WARNING_UNAVAILABLE",
    ):
        build_task_audit_record(
            execution_unit=_unit(),
            calibration=_calibration(),
            ordered_view_ids=ORDERED_VIEWS,
            valid=False,
            reason_code="INVALID_MODEL_OUTPUT",
            native_warning_score_value=None,
        )

    with pytest.raises(
        Task3ContractError,
        match="MVE_BLOCKED_NATIVE_WARNING_UNAVAILABLE",
    ):
        build_task_audit_record(
            execution_unit=_unit(),
            calibration=_calibration(),
            ordered_view_ids=ORDERED_VIEWS,
            valid=False,
            reason_code="INVALID_MODEL_OUTPUT",
            native_warning_score_value=math.nan,
        )
