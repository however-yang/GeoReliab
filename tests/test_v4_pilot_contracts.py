"""Test-only contracts for the preregistered GeoReliab v4 Pilot.

This suite is deliberately CPU-only.  It does not materialize inputs, import
model adapters, select a GPU, or write scientific evidence.  A failing test is
a Pilot readiness blocker; it must not be weakened or silently xfailed to make
the suite green.
"""

from __future__ import annotations

from dataclasses import replace
import json

import pytest

from georeliab_mve.v4_counterfactuals import (
    FOG_BOUNDARY_LAG_SEQUENCE,
    FOG_STATES,
    LIGHTING_STATES,
    SCIENTIFIC_MODELS,
    SCIENTIFIC_STATES,
    TEST_SCENE_IDS,
)
from georeliab_mve.v4_metrics import POSE_PAIR_COUNT, POSE_VIEW_COUNT
from georeliab_mve.v4_science_lock import NO_SCIENTIFIC_RESULT
from georeliab_mve.v4_scoped import (
    FULL_CONFIRMATION_FORBIDDEN,
    V4_PILOT_BLOCKED_EXECUTION,
    V4_PILOT_GO_TO_FULL_MVE,
    V4_PILOT_INCONCLUSIVE,
    V4_PILOT_SCIENTIFIC_NO_GO,
    build_pilot_partition,
    build_scoped_warning_evidence,
    evaluate_pilot,
    evaluate_pilot_gate,
)


def _strong_rows(
    scenes: tuple[int, ...],
    *,
    classes: bool = True,
    reverse_models: frozenset[str] = frozenset(),
    alarm_state_by_model: dict[str, str] | None = None,
) -> list[dict[str, object]]:
    """Create complete metric rows with an unambiguous late-warning effect."""

    alarms = alarm_state_by_model or {
        model: "fog-s3" for model in SCIENTIFIC_MODELS
    }
    rows: list[dict[str, object]] = []
    for model in SCIENTIFIC_MODELS:
        for scene in scenes:
            for state_index, state in enumerate(SCIENTIFIC_STATES):
                failure = classes and state in {"fog-s2", "fog-s3"}
                ranking_label = not failure if model in reverse_models else failure
                rows.append(
                    {
                        "model_id": model,
                        "scene_id": scene,
                        "state_id": state,
                        "pose_failure": failure,
                        "native_warning_score": (
                            float(ranking_label) + state_index / 100.0
                        ),
                        "alarm": state == alarms[model],
                    }
                )
    return rows


def _borderline_ranking_rows(
    scenes: tuple[int, ...],
) -> list[dict[str, object]]:
    """Create AUROC 14/24 = 0.5833..., the preregistered grey zone."""

    assert len(scenes) == 3
    negative_below_counts = (5, 5, 4)
    rows: list[dict[str, object]] = []
    for model in SCIENTIFIC_MODELS:
        for scene_index, scene in enumerate(scenes):
            negative_index = 0
            for state in SCIENTIFIC_STATES:
                failure = state in {"fog-s2", "fog-s3"}
                if failure:
                    score = 0.0
                else:
                    score = (
                        -1.0
                        if negative_index < negative_below_counts[scene_index]
                        else 1.0
                    )
                    negative_index += 1
                rows.append(
                    {
                        "model_id": model,
                        "scene_id": scene,
                        "state_id": state,
                        "pose_failure": failure,
                        "native_warning_score": score,
                        "alarm": state == "fog-s3",
                    }
                )
    return rows


def test_pilot_inventory_is_exactly_sixty_and_partition_is_closed() -> None:
    partition = build_pilot_partition("a" * 64)

    assert len(partition.primary_unit_ids) == 60
    assert len(set(partition.primary_unit_ids)) == 60
    assert len(partition.extension_unit_ids) == 40
    assert len(partition.confirmation_15_unit_ids) == 300
    assert len(partition.confirmation_17_unit_ids) == 340

    primary = set(partition.primary_scene_ids)
    extension = set(partition.extension_scene_ids)
    core = set(partition.core_scene_ids)
    assert not primary & extension
    assert not primary & core
    assert not extension & core
    assert primary | extension | core == set(TEST_SCENE_IDS)


def test_schedule_identity_changes_every_partition_binding() -> None:
    first = build_pilot_partition("a" * 64)
    second = build_pilot_partition("b" * 64)

    assert first.schedule_identity_sha256 != second.schedule_identity_sha256
    assert first.selector_payload_sha256 != second.selector_payload_sha256
    assert first.partition_sha256 != second.partition_sha256
    assert first.primary_scope.scope_sha256 != second.primary_scope.scope_sha256
    assert (
        first.confirmation_17_scope.addendum_sha256
        != second.confirmation_17_scope.addendum_sha256
    )


def test_partition_and_scope_digest_tampering_is_rejected() -> None:
    partition = build_pilot_partition("a" * 64)

    with pytest.raises(ValueError, match="partition digest"):
        replace(partition, partition_sha256="f" * 64)
    with pytest.raises(ValueError, match="expected_record_count"):
        replace(
            partition.primary_scope,
            expected_record_count=partition.primary_scope.expected_record_count - 1,
        )


def test_axis_semantics_keep_lighting_unordered_and_lag_fog_only() -> None:
    assert LIGHTING_STATES == ("L1", "L2", "L3", "L4", "L5", "L6", "L7")
    assert FOG_STATES == ("fog-s1", "fog-s2", "fog-s3")
    assert SCIENTIFIC_STATES == (*LIGHTING_STATES, *FOG_STATES)
    assert len(SCIENTIFIC_STATES) == 10
    assert SCIENTIFIC_STATES.count("L3") == 1
    assert FOG_BOUNDARY_LAG_SEQUENCE == (
        "L3",
        "fog-s1",
        "fog-s2",
        "fog-s3",
    )
    assert POSE_VIEW_COUNT == 8
    assert POSE_PAIR_COUNT == 28


@pytest.mark.parametrize(
    ("mutation", "reason_fragment"),
    (
        ("missing", "missing"),
        ("duplicate", "duplicate"),
        ("outside", "outside"),
    ),
)
def test_pilot_coverage_fails_closed(
    mutation: str,
    reason_fragment: str,
) -> None:
    partition = build_pilot_partition("a" * 64)
    scenes = partition.primary_scene_ids
    rows = _strong_rows(scenes)
    if mutation == "missing":
        rows.pop()
    elif mutation == "duplicate":
        rows.append(dict(rows[0]))
    else:
        rows[0] = {**rows[0], "state_id": "fog-s4"}

    decision = evaluate_pilot(rows, scene_ids=scenes, partition=partition)

    assert decision.status == V4_PILOT_BLOCKED_EXECUTION
    assert reason_fragment in decision.reason_code
    assert decision.to_dict()["scientific_result"] == NO_SCIENTIFIC_RESULT


@pytest.mark.parametrize(
    "counter_name",
    ("invalid_count", "duplicate_count", "identity_mismatch_count"),
)
def test_integrity_counters_override_undefined_metrics(counter_name: str) -> None:
    partition = build_pilot_partition("a" * 64)
    scenes = partition.primary_scene_ids
    decision = evaluate_pilot(
        _strong_rows(scenes, classes=False),
        scene_ids=scenes,
        partition=partition,
        **{counter_name: 1},
    )

    assert decision.status == V4_PILOT_BLOCKED_EXECUTION
    assert decision.reason_code == "EXECUTION_INTEGRITY_FAILURE"
    assert decision.execution_blocked
    assert decision.to_dict()[counter_name] == 1


def test_undefined_metrics_are_inconclusive_without_denominator_deletion() -> None:
    partition = build_pilot_partition("a" * 64)
    scenes = partition.primary_scene_ids
    decision = evaluate_pilot(
        _strong_rows(scenes, classes=False),
        scene_ids=scenes,
        partition=partition,
    )

    assert decision.status == V4_PILOT_INCONCLUSIVE
    assert decision.reason_code == "UNDEFINED_MODEL_METRIC"
    assert len(decision.models) == len(SCIENTIFIC_MODELS)
    assert all(not model.defined for model in decision.models)


def test_macro_auroc_grey_zone_is_inconclusive_not_go_or_no_go() -> None:
    partition = build_pilot_partition("a" * 64)
    scenes = partition.primary_scene_ids
    decision = evaluate_pilot(
        _borderline_ranking_rows(scenes),
        scene_ids=scenes,
        partition=partition,
    )

    assert decision.macro_auroc == pytest.approx(14 / 24)
    assert 0.55 <= decision.macro_auroc < 0.60
    assert decision.pooled_boundary_lag_median > 0
    assert decision.status == V4_PILOT_INCONCLUSIVE


def test_model_direction_conflict_is_preregistered_scientific_no_go() -> None:
    partition = build_pilot_partition("a" * 64)
    scenes = partition.primary_scene_ids
    decision = evaluate_pilot(
        _strong_rows(scenes, reverse_models=frozenset({"MASt3R"})),
        scene_ids=scenes,
        partition=partition,
    )

    assert decision.status == V4_PILOT_SCIENTIFIC_NO_GO
    assert decision.reason_code == "PREREGISTERED_NO_GO"
    assert decision.to_dict()["scientific_result"] == NO_SCIENTIFIC_RESULT


def test_nonpositive_boundary_lag_is_preregistered_scientific_no_go() -> None:
    partition = build_pilot_partition("a" * 64)
    scenes = partition.primary_scene_ids
    decision = evaluate_pilot(
        _strong_rows(
            scenes,
            alarm_state_by_model={model: "fog-s2" for model in SCIENTIFIC_MODELS},
        ),
        scene_ids=scenes,
        partition=partition,
    )

    assert decision.pooled_boundary_lag_median == 0
    assert decision.status == V4_PILOT_SCIENTIFIC_NO_GO


def test_pilot_go_remains_development_only_and_has_no_formal_marker() -> None:
    partition = build_pilot_partition("a" * 64)
    scenes = partition.primary_scene_ids
    decision = evaluate_pilot(
        _strong_rows(scenes), scene_ids=scenes, partition=partition
    )
    payload = decision.to_dict()

    assert decision.status == V4_PILOT_GO_TO_FULL_MVE
    assert payload["scientific_result"] == NO_SCIENTIFIC_RESULT
    serialized = json.dumps(payload, sort_keys=True)
    assert "MVE_FINALIZED" not in serialized
    assert "MVE_GO_TO_EXTERNAL_VALIDATION" not in serialized


def test_gate_is_pure_and_never_auto_starts_extension_or_confirmation(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    partition = build_pilot_partition("a" * 64)
    primary = evaluate_pilot(
        _borderline_ranking_rows(partition.primary_scene_ids),
        scene_ids=partition.primary_scene_ids,
        partition=partition,
    )
    extension_scenes = partition.primary_scene_ids + partition.extension_scene_ids
    extension = evaluate_pilot(
        _strong_rows(extension_scenes),
        scene_ids=extension_scenes,
        partition=partition,
        extension=True,
    )
    monkeypatch.chdir(tmp_path)

    unauthorized = evaluate_pilot_gate(
        primary,
        extension=extension,
        power_15_pass=True,
        power_17_pass=True,
        extension_authorized=False,
    )

    assert unauthorized.status == V4_PILOT_INCONCLUSIVE
    assert unauthorized.extension_allowed
    assert unauthorized.formal_confirmation_scene_count is None
    assert list(tmp_path.iterdir()) == []

    authorized = evaluate_pilot_gate(
        primary,
        extension=extension,
        power_15_pass=True,
        power_17_pass=True,
        extension_authorized=True,
    )
    assert authorized.status == V4_PILOT_GO_TO_FULL_MVE
    assert authorized.formal_confirmation_scene_count == 17
    assert authorized.to_dict()["scientific_result"] == NO_SCIENTIFIC_RESULT
    assert list(tmp_path.iterdir()) == []


def test_failed_power_gate_forbids_formal_confirmation() -> None:
    partition = build_pilot_partition("a" * 64)
    scenes = partition.primary_scene_ids
    primary = evaluate_pilot(
        _strong_rows(scenes), scene_ids=scenes, partition=partition
    )

    decision = evaluate_pilot_gate(
        primary,
        power_15_pass=True,
        power_17_pass=False,
    )

    assert decision.status == FULL_CONFIRMATION_FORBIDDEN
    assert decision.full_confirmation_forbidden
    assert decision.formal_confirmation_scene_count is None


def test_contract_gap_coverage_block_must_survive_gate_finalization() -> None:
    partition = build_pilot_partition("a" * 64)
    scenes = partition.primary_scene_ids
    blocked = evaluate_pilot(
        _strong_rows(scenes)[:-1],
        scene_ids=scenes,
        partition=partition,
    )

    assert blocked.status == V4_PILOT_BLOCKED_EXECUTION
    assert blocked.execution_blocked
    gate = evaluate_pilot_gate(
        blocked,
        power_15_pass=True,
        power_17_pass=True,
    )
    assert gate.status == V4_PILOT_BLOCKED_EXECUTION
    assert gate.full_confirmation_forbidden


def test_contract_gap_extension_must_match_frozen_five_scene_inventory() -> None:
    partition = build_pilot_partition("a" * 64)
    authorized_scenes = partition.primary_scene_ids + partition.extension_scene_ids
    wrong_scenes = partition.primary_scene_ids + partition.core_scene_ids[:2]

    authorized = evaluate_pilot(
        _strong_rows(authorized_scenes),
        scene_ids=authorized_scenes,
        partition=partition,
        extension=True,
    )
    wrong = evaluate_pilot(
        _strong_rows(wrong_scenes),
        scene_ids=wrong_scenes,
        partition=partition,
        extension=True,
    )

    assert authorized.status == V4_PILOT_GO_TO_FULL_MVE
    assert wrong.status == V4_PILOT_BLOCKED_EXECUTION


def test_contract_gap_five_scene_scope_requires_extension_mode() -> None:
    partition = build_pilot_partition("a" * 64)
    scenes = partition.primary_scene_ids + partition.extension_scene_ids

    decision = evaluate_pilot(
        _strong_rows(scenes),
        scene_ids=scenes,
        partition=partition,
        extension=False,
    )

    assert decision.status == V4_PILOT_BLOCKED_EXECUTION


def test_contract_gap_local_gate2_predictions_are_quarantined_from_pilot() -> None:
    partition = build_pilot_partition("a" * 64, protocol_sha256="b" * 64)

    with pytest.raises(ValueError, match="Gate 2|smoke|forbidden"):
        build_scoped_warning_evidence(
            {"development": True},
            scope=partition.primary_scope,
            input_record_inventory_sha256="c" * 64,
            input_record_count=partition.primary_scope.expected_record_count,
            source_attempt_id="local-gate2-run-01",
        )
