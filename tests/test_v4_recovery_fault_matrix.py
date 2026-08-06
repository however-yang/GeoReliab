"""Focused tests for the CPU-only Gate 1 fault matrix."""

from pathlib import Path

import pytest

from georeliab_mve import v4_attempt05_recovery as recovery
from georeliab_mve import v4_recovery_fault_matrix as matrix


def test_fault_catalog_covers_required_surfaces_and_errno_kinds() -> None:
    cases = matrix.build_fault_matrix_cases()
    assert all(isinstance(case.fault_point, matrix.FaultPoint) for case in cases)
    points = {case.fault_point for case in cases}
    assert {
        "artifact:temp_write",
        "prepared_receipt:file_fsync",
        "promotion:rename_before",
        "promotion:rename_after",
        "projection:ledger_append",
        "completion:ledger_append",
        "pending_cleanup:cleanup",
        "checkpoint:file_fsync",
        "ledger:torn_tail",
        "terminal:ledger",
        "terminal:supervisor",
        "identity:symlink",
        "identity:path",
        "liveness:live",
        "liveness:dead",
    } <= points
    assert {case.fault_kind for case in cases} >= {"EACCES", "ENOSPC", "EIO", "ABRUPT_EXIT"}
    assert all(case.expected_classification in matrix.CLASSIFICATIONS for case in cases)


def test_state_normalization_is_four_class_and_active_is_nonterminal() -> None:
    assert matrix._normalise_state({"state": recovery.COMPLETE, "recovery_action": recovery.RECOVERY_ACTION_NOOP}) == (
        "COMPLETE",
        recovery.RECOVERY_ACTION_NOOP,
    )
    assert matrix._normalise_state(
        {"state": recovery.INCOMPLETE_SAFE_TO_RETRY, "recovery_action": recovery.RECOVERY_ACTION_RESUME_LEDGER_ONLY}
    ) == ("SAFE_RETRY", recovery.RECOVERY_ACTION_RESUME_LEDGER_ONLY)
    assert matrix._normalise_state(
        {"state": recovery.ORPHAN_REQUIRES_QUARANTINE, "recovery_action": recovery.RECOVERY_ACTION_QUARANTINE}
    ) == ("QUARANTINED", recovery.RECOVERY_ACTION_QUARANTINE)
    assert matrix._normalise_state(
        {"state": recovery.FATAL_IDENTITY_MISMATCH, "recovery_action": recovery.RECOVERY_ACTION_ABORT_FATAL}
    ) == ("FATAL_IDENTITY_MISMATCH", recovery.RECOVERY_ACTION_ABORT_FATAL)
    with pytest.raises(matrix.FaultMatrixError, match="ACTIVE_STATE_NOT_TERMINAL"):
        matrix._normalise_state({"state": recovery.ACTIVE_VALID_DEFER_NO_MUTATION, "recovery_action": recovery.RECOVERY_ACTION_NOOP})


def test_case_hashes_are_deterministic() -> None:
    first = [case.to_dict() for case in matrix.build_fault_matrix_cases()]
    second = [case.to_dict() for case in matrix.build_fault_matrix_cases()]
    assert first == second


def test_identity_and_liveness_cases_are_classified_without_scientific_payload(tmp_path: Path) -> None:
    selected = {
        "no-clobber-promotion",
        "identity-symlink",
        "identity-content-mutation",
        "identity-duplicate-key",
        "identity-path-mismatch",
        "liveness-live-worker",
        "liveness-dead-worker-stale-heartbeat",
    }
    cases = [case for case in matrix.build_fault_matrix_cases() if case.case_id in selected]
    assert {case.case_id for case in cases} == selected
    for case in cases:
        observed = matrix._run_case(case, tmp_path / case.case_id)
        assert observed[0] == case.expected_classification
        assert observed[1] == case.expected_action
        assert "NameError" not in observed[2]
        matrix._assert_no_scientific_payload({"case": case.to_dict(), "observation": observed[2]})


def test_observation_requires_expected_action_and_no_scientific_result() -> None:
    observation = matrix.FaultInjectionObservation(
        case_id="case",
        expected_classification="COMPLETE",
        expected_action=recovery.RECOVERY_ACTION_NOOP,
        observed_classification="COMPLETE",
        observed_action=recovery.RECOVERY_ACTION_NOOP,
        reason="ok",
        before_inventory_sha256="0" * 64,
        after_inventory_sha256="0" * 64,
        before_ledger_sha256="0" * 64,
        after_ledger_sha256="0" * 64,
        duplicate_count=0,
        overwrite_count=0,
    )
    assert observation.passed
    assert observation.to_dict()["expected_action"] == recovery.RECOVERY_ACTION_NOOP
    assert observation.to_dict()["scientific_result"] == matrix.NO_SCIENTIFIC_RESULT
