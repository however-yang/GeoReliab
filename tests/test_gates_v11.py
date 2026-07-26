from __future__ import annotations

import pytest

from georeliab_mve.contracts import RunMode, ScientificValidity
from georeliab_mve.gates import (
    GateDecision,
    GateStatus,
    GeoReliabGateInput,
    SelectedTrack,
    evaluate_georeliab_gate,
    select_track,
)


def decision(lane, status, validity=ScientificValidity.SCIENTIFIC):
    return GateDecision(lane, status, (), {}, validity)


def test_smoke_evidence_cannot_enter_a_scientific_gate():
    result = evaluate_georeliab_gate(
        GeoReliabGateInput(
            scientific_validity=ScientificValidity.NON_SCIENTIFIC_SMOKE,
            required_models_ready=(),
            required_datasets_ready=False,
            tartanair_native_fog_sanity=False,
            conditions=(),
            downstream_harm=(),
            zero_update=(),
            run_mode=RunMode.SMOKE,
        )
    )
    assert result.status is GateStatus.BLOCKED
    assert result.reason_codes == ('NON_SCIENTIFIC_SMOKE',)


@pytest.mark.parametrize(
    ('geometry_status', 'geo_status', 'expected'),
    (
        (GateStatus.BLOCKED, GateStatus.PASS, SelectedTrack.BLOCKED_PENDING_GEOMETRY),
        (GateStatus.PASS, GateStatus.BLOCKED, SelectedTrack.BLOCKED),
        (GateStatus.FAIL, GateStatus.BLOCKED, SelectedTrack.BLOCKED),
    ),
)
def test_non_terminal_route_never_selects_a_final_track(
    geometry_status, geo_status, expected
):
    selected = select_track(
        decision('geometry', geometry_status),
        decision('georeliab', geo_status),
    )
    assert selected.selected_track is expected


def test_georeliab_pass_pending_geometry_has_explicit_reason_code():
    selected = select_track(
        decision('geometry', GateStatus.BLOCKED),
        decision('georeliab', GateStatus.PASS),
    )
    assert selected.reason == 'GEORELIAB_PASS_PENDING_GEOMETRY'


def test_smoke_selection_preserves_smoke_validity_and_reason():
    smoke = ScientificValidity.NON_SCIENTIFIC_SMOKE
    selected = select_track(
        decision('geometry', GateStatus.BLOCKED, smoke),
        decision('georeliab', GateStatus.BLOCKED, smoke),
    )
    assert selected.scientific_validity is smoke
    assert 'smoke' in selected.reason.lower()
