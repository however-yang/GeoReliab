from __future__ import annotations

from georeliab_mve.science_lock import (
    LOCKED_SCIENCE_FILES,
    validate_schedule_contract,
)


def test_science_lock_covers_gate_implementation() -> None:
    assert "georeliab_mve/gates.py" in LOCKED_SCIENCE_FILES


def test_frozen_schedule_contract_keeps_all_grid_sizes() -> None:
    report = validate_schedule_contract()
    assert report == {
        "status": "PASS",
        "preflight_items": 8,
        "p2_items": 200,
        "p3_items": 400,
        "p5_zero_update_items": 480,
    }
