from __future__ import annotations

import pytest

from georeliab_mve.v4_counterfactuals import SCIENTIFIC_MODELS, SCIENTIFIC_STATES
from georeliab_mve.v4_scoped import (
    GO_FAMILY_RANKING_WARNING,
    PAPER_NO_PRIMARY_CLAIM,
    PAPER_RANKING_NOT_WARNING_SUPPORTED,
    PROTOCOL_DECISION_GO,
    V4_PILOT_BLOCKED_EXECUTION,
    V4_PILOT_GO_TO_FULL_MVE,
    V4_PILOT_INCONCLUSIVE,
    FULL_CONFIRMATION_FORBIDDEN,
    build_pilot_partition,
    build_confirmation_scope_addendum,
    build_schedule_identity_manifest,
    build_scoped_warning_evidence,
    build_synthetic_power_design,
    classify_paper_claim,
    evaluate_pilot,
    evaluate_pilot_gate,
    evaluate_scoped_warning_gate,
    qualify_boundary_lag_claim,
)


def _pilot_rows(scenes, *, warning=True, classes=True):
    rows = []
    for model in SCIENTIFIC_MODELS:
        for scene in scenes:
            for state_index, state in enumerate(SCIENTIFIC_STATES):
                failure = classes and state in {"fog-s2", "fog-s3"}
                rows.append(
                    {
                        "model_id": model,
                        "scene_id": scene,
                        "state_id": state,
                        "pose_failure": failure,
                        "native_warning_score": (float(failure) if warning else 0.0)
                        + state_index / 100.0,
                        "alarm": state == "fog-s3" if warning else False,
                    }
                )
    return rows


def test_partition_is_deterministic_and_disjoint():
    first = build_pilot_partition("a" * 64)
    second = build_pilot_partition("a" * 64)
    assert first.to_dict() == second.to_dict()
    assert len(first.primary_scene_ids) == 3
    assert len(first.extension_scene_ids) == 2
    assert len(first.core_scene_ids) == 15
    assert not set(first.primary_scene_ids) & set(first.extension_scene_ids)
    assert set(first.confirmation_17_scope.scene_ids) == set(
        first.extension_scene_ids + first.core_scene_ids
    )


def test_schedule_identity_separates_raw_and_semantic_domains():
    manifest = build_schedule_identity_manifest(b"{}", {"units": ["u"]}, ["u"])
    assert manifest.raw_sha256 != manifest.semantic_sha256
    assert manifest.unit_count == 1
    assert len(manifest.schedule_identity_sha256) == 64


def test_power_design_is_preregistered_and_has_deterministic_seeds():
    design = build_synthetic_power_design()
    assert (design.static_rank, design.crr, design.rwg, design.sfr) == (
        0.45,
        0.05,
        0.40,
        0.40,
    )
    assert design.task_transfer_non_directional_null
    assert design.scope_scene_counts == (15, 17)
    assert design.outer_replicates == 10_000
    assert design.inner_bootstrap_replicates == 10_000
    assert len(design.sensitivity_scenarios()) == 8
    assert design.holm_method == "four-holm"
    assert design.streaming and not design.allow_downsampling
    assert design.generator_family == "GAUSSIAN_COPULA_SCENE_MODEL_EFFECTS"
    assert design.seed_domain == "georeliab:power-seed:v1" and design.seed_version == "counter-based-v1"
    assert len(design.factor_specs) == 4 and len(design.link_specs) == 5
    assert design.seed(scope_scene_count=15) == design.seed(scope_scene_count=15)
    assert design.seed(scope_scene_count=15) != design.seed(scope_scene_count=17)
    with pytest.raises(ValueError, match="RWG"):
        build_synthetic_power_design(rwg=0.39)


def test_pilot_go_undefined_and_execution_block_precedence():
    partition = build_pilot_partition("a" * 64)
    scenes = partition.primary_scene_ids
    decision = evaluate_pilot(
        _pilot_rows(scenes), scene_ids=scenes, partition=partition
    )
    assert decision.status == V4_PILOT_GO_TO_FULL_MVE
    undefined = evaluate_pilot(
        _pilot_rows(scenes, classes=False),
        scene_ids=scenes,
        partition=partition,
    )
    assert undefined.status == V4_PILOT_INCONCLUSIVE
    blocked = evaluate_pilot(
        _pilot_rows(scenes),
        scene_ids=scenes,
        partition=partition,
        invalid_count=1,
    )
    assert blocked.status == V4_PILOT_BLOCKED_EXECUTION


def test_pilot_gate_extension_and_power_precedence():
    partition = build_pilot_partition("a" * 64)
    scenes = partition.primary_scene_ids
    primary_go = evaluate_pilot(
        _pilot_rows(scenes), scene_ids=scenes, partition=partition
    )
    gate = evaluate_pilot_gate(primary_go, power_15_pass=True, power_17_pass=True)
    assert gate.status == V4_PILOT_GO_TO_FULL_MVE
    assert gate.formal_confirmation_scene_count == 17
    assert not gate.extension_allowed
    primary_inconclusive = evaluate_pilot(
        _pilot_rows(scenes, classes=False),
        scene_ids=scenes,
        partition=partition,
    )
    pending = evaluate_pilot_gate(
        primary_inconclusive,
        power_15_pass=True,
        power_17_pass=False,
    )
    assert pending.status == V4_PILOT_INCONCLUSIVE
    assert pending.extension_allowed and not pending.full_confirmation_forbidden
    blocked = evaluate_pilot_gate(
        primary_inconclusive,
        power_15_pass=False,
        power_17_pass=True,
    )
    assert blocked.full_confirmation_forbidden


def test_scope_rejects_attempt05_and_claim_is_separate_from_protocol():
    partition = build_pilot_partition("a" * 64, protocol_sha256="b" * 64)
    scope = partition.primary_scope
    evidence = {"development": True}
    for source in ("v4-mve-attempt-05", "attempt05-replay", "ATTEMPT_05"):
        with pytest.raises(ValueError, match="Attempt-05"):
            build_scoped_warning_evidence(
                evidence,
                scope=scope,
                input_record_inventory_sha256="c" * 64,
                input_record_count=60,
                source_attempt_id=source,
            )
    claim = classify_paper_claim(
        protocol_decision=PROTOCOL_DECISION_GO,
        go_family=GO_FAMILY_RANKING_WARNING,
        ranking_warning_pass=True,
        boundary_lag_pass=True,
    )
    assert claim.qualification == PAPER_RANKING_NOT_WARNING_SUPPORTED
    no_claim = classify_paper_claim(
        protocol_decision=PROTOCOL_DECISION_GO,
        go_family=GO_FAMILY_RANKING_WARNING,
        ranking_warning_pass=False,
        boundary_lag_pass=True,
    )
    assert no_claim.qualification == PAPER_NO_PRIMARY_CLAIM

def test_boundary_lag_claim_requires_ci_and_loso_support():
    assert qualify_boundary_lag_claim(
        model_lag_medians={"VGGT": 1.0, "MASt3R": 2.0},
        pooled_late_warning_proportion=0.60,
        pooled_median_lag_ci_lower=0.01,
        loso_lag_medians=(0.5, 1.0),
    )
    assert not qualify_boundary_lag_claim(
        model_lag_medians=(1.0, -0.1),
        pooled_late_warning_proportion=0.90,
        pooled_median_lag_ci_lower=0.01,
        loso_lag_medians=(0.5,),
    )


def test_scoped_evaluator_closes_count_and_nested_coverage():
    partition = build_pilot_partition("a" * 64)
    scope = partition.primary_scope
    evidence = build_scoped_warning_evidence(
        {"development": True},
        scope=scope,
        input_record_inventory_sha256="c" * 64,
        input_record_count=scope.expected_record_count,
    )
    formal = {"status": "MVE_GO_TO_EXTERNAL_VALIDATION", "reason_code": "OK"}
    decision = evaluate_scoped_warning_gate(
        evidence,
        formal_decision=formal,
        scene_ids=scope.scene_ids,
        expected_count=scope.expected_record_count,
        coverage={
            "VGGT": {"scene_ids": list(scope.scene_ids), "record_count": 30},
            "MASt3R": {"scene_ids": list(scope.scene_ids), "record_count": 30},
        },
    )
    assert decision.protocol_decision == PROTOCOL_DECISION_GO
    assert decision.coverage_valid and not decision.parity_required
    assert decision.to_dict()["scientific_result"] == "NO_SCIENTIFIC_RESULT"
    mismatch = evaluate_scoped_warning_gate(
        evidence,
        formal_decision=formal,
        scene_ids=scope.scene_ids,
        expected_count=scope.expected_record_count - 1,
        coverage={"VGGT": 30, "MASt3R": 30},
    )
    assert mismatch.protocol_decision != PROTOCOL_DECISION_GO
    assert mismatch.reason_code == "SCOPED_EXPECTED_COUNT_MISMATCH"


def test_scoped_20_scene_requires_agreeing_parity_callback():
    partition = build_pilot_partition("a" * 64)
    all_scenes = tuple(sorted(partition.primary_scene_ids + partition.extension_scene_ids + partition.core_scene_ids))
    scope = build_confirmation_scope_addendum(
        schedule_identity_sha256="a" * 64,
        protocol_sha256="b" * 64,
        role="FULL20",
        scene_ids=all_scenes,
    )
    evidence = build_scoped_warning_evidence(
        {"development": True},
        scope=scope,
        input_record_inventory_sha256="c" * 64,
        input_record_count=scope.expected_record_count,
    )
    formal = {"status": "MVE_GO_TO_EXTERNAL_VALIDATION", "reason_code": "OK"}
    coverage = {model: {"scene_ids": list(all_scenes), "record_count": 200} for model in SCIENTIFIC_MODELS}
    missing = evaluate_scoped_warning_gate(
        evidence,
        formal_decision=formal,
        expected_count=400,
        coverage=coverage,
    )
    assert missing.reason_code == "SCOPED_20_SCENE_PARITY_REQUIRED"
    mismatch = evaluate_scoped_warning_gate(
        evidence,
        formal_decision=formal,
        expected_count=400,
        coverage=coverage,
        parity_evaluator=lambda _: {"status": "MVE_SCIENTIFIC_NO_GO", "reason_code": "MISMATCH"},
    )
    assert mismatch.reason_code == "SCOPED_20_SCENE_PARITY_MISMATCH"
    assert not mismatch.parity_checked
    matched = evaluate_scoped_warning_gate(
        evidence,
        formal_decision=formal,
        expected_count=400,
        coverage=coverage,
        parity_evaluator=lambda _: formal,
    )
    assert matched.protocol_decision == PROTOCOL_DECISION_GO
    assert matched.parity_required and matched.parity_checked

def test_extension_non_go_forbids_full_confirmation():
    partition = build_pilot_partition("a" * 64)
    primary_scenes = partition.primary_scene_ids
    extension_scenes = partition.primary_scene_ids + partition.extension_scene_ids
    primary = evaluate_pilot(
        _pilot_rows(primary_scenes, classes=False),
        scene_ids=primary_scenes,
        partition=partition,
    )
    extension = evaluate_pilot(
        _pilot_rows(extension_scenes, classes=False),
        scene_ids=extension_scenes,
        partition=partition,
        extension=True,
    )
    gate = evaluate_pilot_gate(primary, extension=extension, power_15_pass=True, power_17_pass=True, extension_authorized=True)
    assert gate.status == FULL_CONFIRMATION_FORBIDDEN
    assert gate.full_confirmation_forbidden
