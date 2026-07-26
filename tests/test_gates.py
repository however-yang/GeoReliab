from __future__ import annotations

from dataclasses import replace

import pytest

from georeliab_mve.contracts import ScientificValidity
from georeliab_mve.gates import (
    DownstreamHarmEvidence,
    GateDecision,
    GateStatus,
    GeometryEvidence,
    GeometryGateInput,
    GeoReliabConditionEvidence,
    GeoReliabGateInput,
    SelectedTrack,
    ZeroUpdateEvidence,
    evaluate_geometry_gate,
    evaluate_georeliab_gate,
    select_track,
)


SCIENTIFIC = ScientificValidity.SCIENTIFIC


def geometry_evidence(
    *,
    faithful=True,
    equivalent=False,
    benchmarks=("VSI-Bench", "CVT-Bench"),
    include_controls=True,
    models=("Spatial-MLLM", "SpatialStack"),
):
    rows = []
    for model in models:
        for benchmark in benchmarks:
            for stratum in ("distance", "direction"):
                rows.append(
                    GeometryEvidence(
                        model=model,
                        benchmark=benchmark,
                        sample_class="geometry-required",
                        stratum=stratum,
                        delta_geom=0.06 if faithful else 0.0,
                        ci_lower=0.01 if faithful else -0.01,
                        recovery=0.35 if faithful else 0.0,
                        equivalent_by_tost=equivalent,
                        post_fusion_changed=True,
                        semantic_control_unchanged=False,
                    )
                )
            if include_controls:
                rows.append(
                    GeometryEvidence(
                        model=model,
                        benchmark=benchmark,
                        sample_class="semantic-control",
                        stratum="semantic-2d",
                        delta_geom=0.0,
                        ci_lower=-0.01,
                        recovery=0.0,
                        equivalent_by_tost=False,
                        post_fusion_changed=True,
                        semantic_control_unchanged=True,
                    )
                )
    return tuple(rows)


def geometry_input(**overrides):
    fields = {
        "scientific_validity": SCIENTIFIC,
        "reproducible_checkpoints": ("Spatial-MLLM", "SpatialStack"),
        "hookable_models": ("Spatial-MLLM", "SpatialStack"),
        "required_datasets_ready": True,
        "fixed_inputs_verified": True,
        "zeroing_effective": True,
        "matched_intervention_effective": True,
        "evidence": geometry_evidence(),
    }
    fields.update(overrides)
    return GeometryGateInput(**fields)


def georeliab_input(**overrides):
    conditions = tuple(
        GeoReliabConditionEvidence(
            model=model,
            corruption=corruption,
            clean_rho=0.8,
            severity_rhos=(0.6, 0.4, 0.2),
            failure_auroc=0.70,
            corruption_severity_monotonic=True,
            cross_view_consistent=True,
            gt_geometry_invariant=True,
            relative_decline_ci_lower=0.60,
        )
        for model in ("VGGT", "MASt3R")
        for corruption in ("fog", "low-light-noise", "defocus")
    )
    fields = {
        "scientific_validity": SCIENTIFIC,
        "required_models_ready": ("VGGT", "MASt3R"),
        "required_datasets_ready": True,
        "tartanair_native_fog_sanity": True,
        "conditions": conditions,
        "downstream_harm": (
            DownstreamHarmEvidence("VGGT", "fog", -0.02, -0.01),
            DownstreamHarmEvidence("MASt3R", "defocus", -0.03, -0.01),
        ),
        "zero_update": (
            ZeroUpdateEvidence("VGGT", "fog", 0.10, 0.01),
        ),
    }
    fields.update(overrides)
    return GeoReliabGateInput(**fields)


def test_geometry_faithful_use_passes():
    result = evaluate_geometry_gate(geometry_input())
    assert result.status is GateStatus.PASS
    assert result.reason_codes == ('FAITHFUL_GEOMETRY_USE',)


def test_geometry_causal_nondependence_passes():
    result = evaluate_geometry_gate(
        geometry_input(evidence=geometry_evidence(faithful=False, equivalent=True))
    )
    assert result.status is GateStatus.PASS
    assert result.reason_codes == ('CAUSAL_NONDEPENDENCE',)


def test_geometry_zeroing_only_is_rejected():
    result = evaluate_geometry_gate(
        geometry_input(matched_intervention_effective=False)
    )
    assert result.status is GateStatus.FAIL
    assert result.reason_codes == ('ZEROING_ONLY_OOD_ARTIFACT',)


def test_geometry_missing_checkpoint_is_blocked():
    result = evaluate_geometry_gate(
        geometry_input(reproducible_checkpoints=('Spatial-MLLM',))
    )
    assert result.status is GateStatus.BLOCKED


def test_geometry_requires_both_benchmarks():
    result = evaluate_geometry_gate(
        geometry_input(evidence=geometry_evidence(benchmarks=("VSI-Bench",)))
    )
    assert result.status is GateStatus.FAIL


def test_geometry_requires_semantic_controls():
    result = evaluate_geometry_gate(
        geometry_input(evidence=geometry_evidence(include_controls=False))
    )
    assert result.status is GateStatus.FAIL


def test_geometry_requires_fixed_inputs():
    result = evaluate_geometry_gate(geometry_input(fixed_inputs_verified=False))
    assert result.status is GateStatus.FAIL
    assert result.reason_codes == ("FIXED_INPUT_CONTROL_NOT_MET",)


def test_geometry_rejects_models_outside_frozen_candidates():
    result = evaluate_geometry_gate(
        geometry_input(
            reproducible_checkpoints=("foo", "bar"),
            hookable_models=("foo", "bar"),
            evidence=geometry_evidence(models=("foo", "bar")),
        )
    )
    assert result.status is GateStatus.BLOCKED
    assert result.details["unexpected_models"] == ["bar", "foo"]


def test_georeliab_all_three_gates_pass():
    result = evaluate_georeliab_gate(georeliab_input())
    assert result.status is GateStatus.PASS


def test_georeliab_point_estimate_without_ci_cannot_pass():
    no_ci = tuple(
        GeoReliabConditionEvidence(
            model=item.model,
            corruption=item.corruption,
            clean_rho=item.clean_rho,
            severity_rhos=item.severity_rhos,
            failure_auroc=item.failure_auroc,
            corruption_severity_monotonic=item.corruption_severity_monotonic,
            cross_view_consistent=item.cross_view_consistent,
            gt_geometry_invariant=item.gt_geometry_invariant,
        )
        for item in georeliab_input().conditions
    )
    result = evaluate_georeliab_gate(georeliab_input(conditions=no_ci))
    assert result.status is GateStatus.FAIL
    assert 'CONFIDENCE_FAILURE_GATE_NOT_MET' in result.reason_codes


def test_georeliab_ci_must_clear_threshold_not_touch_it():
    boundary = tuple(
        GeoReliabConditionEvidence(
            model=item.model,
            corruption=item.corruption,
            clean_rho=item.clean_rho,
            severity_rhos=item.severity_rhos,
            failure_auroc=item.failure_auroc,
            corruption_severity_monotonic=item.corruption_severity_monotonic,
            cross_view_consistent=item.cross_view_consistent,
            gt_geometry_invariant=item.gt_geometry_invariant,
            relative_decline_ci_lower=0.50,
        )
        for item in georeliab_input().conditions
    )
    result = evaluate_georeliab_gate(georeliab_input(conditions=boundary))
    assert result.status is GateStatus.FAIL


def test_georeliab_duplicate_harm_condition_does_not_satisfy_gate():
    duplicate = DownstreamHarmEvidence('VGGT', 'fog', -0.02, -0.01)
    result = evaluate_georeliab_gate(
        georeliab_input(downstream_harm=(duplicate, duplicate))
    )
    assert result.status is GateStatus.FAIL
    assert 'DOWNSTREAM_HARM_GATE_NOT_MET' in result.reason_codes


def test_georeliab_rejects_endpoint_only_severity_evidence():
    with pytest.raises(ValueError, match="severities 1, 2, and 3"):
        GeoReliabConditionEvidence(
            model="VGGT",
            corruption="fog",
            clean_rho=-0.8,
            severity_rhos=(-0.2,),
            failure_auroc=0.6,
            corruption_severity_monotonic=True,
            cross_view_consistent=True,
            gt_geometry_invariant=True,
        )


def test_georeliab_requires_corruption_verification():
    unverified = tuple(
        replace(item, cross_view_consistent=False)
        for item in georeliab_input().conditions
    )
    result = evaluate_georeliab_gate(
        georeliab_input(conditions=unverified)
    )
    assert result.status is GateStatus.FAIL
    assert "CONFIDENCE_FAILURE_GATE_NOT_MET" in result.reason_codes


def test_georeliab_requires_tartanair_native_fog_sanity():
    result = evaluate_georeliab_gate(
        georeliab_input(tartanair_native_fog_sanity=False)
    )
    assert result.status is GateStatus.FAIL
    assert "TARTANAIR_SANITY_GATE_NOT_MET" in result.reason_codes


def test_georeliab_rejects_non_monotonic_rho_ramp():
    non_monotonic = tuple(
        replace(item, severity_rhos=(-0.2, -0.7, -0.1))
        for item in georeliab_input().conditions
    )
    result = evaluate_georeliab_gate(
        georeliab_input(conditions=non_monotonic)
    )
    assert result.status is GateStatus.FAIL
    assert "CONFIDENCE_FAILURE_GATE_NOT_MET" in result.reason_codes


def test_georeliab_rejects_models_outside_frozen_candidates():
    result = evaluate_georeliab_gate(
        georeliab_input(required_models_ready=("foo", "bar"))
    )
    assert result.status is GateStatus.BLOCKED
    assert result.details["missing_models"] == ["MASt3R", "VGGT"]


def decision(lane, status, validity=SCIENTIFIC):
    return GateDecision(lane, status, (), {}, validity)


@pytest.mark.parametrize(
    ('geometry_status', 'geo_status', 'expected'),
    (
        (GateStatus.PASS, GateStatus.PASS, SelectedTrack.GEOMETRY),
        (GateStatus.PASS, GateStatus.FAIL, SelectedTrack.GEOMETRY),
        (GateStatus.FAIL, GateStatus.PASS, SelectedTrack.GEORELIAB),
        (
            GateStatus.BLOCKED,
            GateStatus.PASS,
            SelectedTrack.BLOCKED_PENDING_GEOMETRY,
        ),
        (GateStatus.FAIL, GateStatus.FAIL, SelectedTrack.STOP),
        (GateStatus.BLOCKED, GateStatus.FAIL, SelectedTrack.BLOCKED),
    ),
)
def test_final_selection_matrix(geometry_status, geo_status, expected):
    selected = select_track(
        decision('geometry', geometry_status),
        decision('georeliab', geo_status),
    )
    assert selected.selected_track is expected


def test_fixture_evidence_can_never_select_a_track():
    fixture = ScientificValidity.NON_SCIENTIFIC_FIXTURE
    selected = select_track(
        decision('geometry', GateStatus.BLOCKED, fixture),
        decision('georeliab', GateStatus.BLOCKED, fixture),
    )
    assert selected.selected_track is SelectedTrack.NON_SCIENTIFIC
    assert selected.to_dict()['selected_track'] == 'BLOCKED_NON_SCIENTIFIC_FIXTURE'

