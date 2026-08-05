from __future__ import annotations

import json

import pytest

from georeliab_mve.v4_qualification import (
    ATTEMPT05_HISTORICAL_COUNT,
    NO_SCIENTIFIC_RESULT,
    CanonicalProvenance,
    HistoricalGpuBounds,
    ScheduleIdentityManifest,
    SOURCE_PATCH_ALLOWLIST,
    build_hourly_monitor_status,
    build_recovery_smoke_manifest,
    evaluate_historical_gpu_gate,
    evaluate_recovery_smoke,
    reject_attempt05_source,
    validate_recovery_smoke_receipts,
    write_json_no_clobber,
    V4_RECOVERY_RUNTIME_BLOCKED,
    V4_RECOVERY_RUNTIME_QUALIFIED,
)


def test_schedule_identity_separates_raw_and_semantic_domains() -> None:
    parsed = {"schema": "v4", "units": [{"unit_id": "a"}, {"unit_id": "b"}]}
    manifest = ScheduleIdentityManifest.from_schedule_bytes(
        b'{"units":[{"unit_id":"a"},{"unit_id":"b"}]}\n',
        parsed,
        ("a", "b"),
    )
    assert manifest.raw_sha256 != manifest.semantic_sha256
    assert manifest.schedule_identity_sha256 != manifest.semantic_sha256
    assert manifest.unit_count == 2


def test_identity_changes_with_order_or_semantic_content() -> None:
    first = ScheduleIdentityManifest.from_schedule_bytes(
        b"raw-a",
        {"units": ["a", "b"]},
        ("a", "b"),
    )
    reordered = ScheduleIdentityManifest.from_schedule_bytes(
        b"raw-a",
        {"units": ["b", "a"]},
        ("b", "a"),
    )
    changed = ScheduleIdentityManifest.from_schedule_bytes(
        b"raw-a",
        {"units": ["a", "c"]},
        ("a", "c"),
    )
    assert reordered.schedule_identity_sha256 != first.schedule_identity_sha256
    assert changed.semantic_sha256 != first.semantic_sha256


def test_canonical_provenance_requires_exact_allowlist() -> None:
    kwargs = dict(
        source_commit="0f4fd144",
        source_parent="6fd6c80e",
        source_tree="79918d43",
        patch_sha256="patch",
        stable_patch_id="422eba6b",
        canonical_base_commit="a9d3844",
        canonical_base_tree="24ae416",
        replay_commit="replay",
        replay_tree="tree",
        replay_patch_id="422eba6b",
        changed_paths=tuple(SOURCE_PATCH_ALLOWLIST),
        scientific_config_zero_drift=True,
    )
    assert CanonicalProvenance(**kwargs).to_dict()["scientific_config_zero_drift"]
    with pytest.raises(ValueError, match="allowlist"):
        CanonicalProvenance(**{**kwargs, "changed_paths": tuple(SOURCE_PATCH_ALLOWLIST[:-1])})


def test_budget_gate_fails_closed_on_unknown_history() -> None:
    result = evaluate_historical_gpu_gate(
        HistoricalGpuBounds(10.0, 49.0, 51.0, 51.0, 41 * 1024**3, False)
    )
    assert result["status"] == "V4_GPU_HISTORY_UNRESOLVED"


def test_monitor_does_not_count_attempt05_history() -> None:
    status = build_hourly_monitor_status(
        stage="GATE1_CPU",
        stage_completed=4,
        stage_total=20,
        attempt06_valid_completed=0,
        attempt06_elapsed_seconds=12.0,
        cumulative_materialization_elapsed_seconds=3600.0,
    )
    assert status["historical_attempt05"]["valid_completed"] == ATTEMPT05_HISTORICAL_COUNT
    assert status["current_progress"]["full400_materialization_progress"] == 0.0
    assert status["external_monitor_interval_seconds"] == 3600


def test_attempt05_source_is_rejected_for_new_attempt() -> None:
    with pytest.raises(ValueError, match="SOURCE_REJECTED"):
        reject_attempt05_source(attempt_id="attempt-06", source_attempt_id="attempt-05")


def test_write_json_no_clobber(tmp_path) -> None:
    path = tmp_path / "manifest.json"
    write_json_no_clobber(path, {"scientific_result": NO_SCIENTIFIC_RESULT})
    assert json.loads(path.read_text())["scientific_result"] == NO_SCIENTIFIC_RESULT
    with pytest.raises(FileExistsError):
        write_json_no_clobber(path, {"other": True})


def _smoke_rows(manifest):
    return [
        {
            "unit_key": unit_id,
            "inference_start_count": 2 if ordinal == 11 else 1,
            "completion_count": 1,
            "overwrite_count": 0,
            "gpu_uuid": manifest.gpu_uuid,
            "physical_gpu_index": manifest.physical_gpu_index,
            "canonical_present": True,
            "ledger_committed": True,
            "scientific_marker": NO_SCIENTIFIC_RESULT,
        }
        for ordinal, unit_id in enumerate(manifest.unit_keys, 1)
    ]


def _smoke_manifest():
    return build_recovery_smoke_manifest(
        schedule_identity_sha256="a" * 64,
        support_scene_ids=(1, 9, 10, 11, 12, 13, 23, 24, 29, 32),
    )


def test_recovery_smoke_contract_is_exactly_once_and_gpu0_only():
    manifest = _smoke_manifest()
    result = evaluate_recovery_smoke(manifest, observations=_smoke_rows(manifest))
    assert result["status"] == V4_RECOVERY_RUNTIME_QUALIFIED
    assert result["unit_count"] == 12
    assert result["scientific_result"] == NO_SCIENTIFIC_RESULT
    assert manifest.physical_gpu_index == 0


def test_recovery_smoke_rejects_gpu1_or_science_marker():
    manifest = _smoke_manifest()
    result = validate_recovery_smoke_receipts(
        manifest,
        _smoke_rows(manifest),
        gpu1_owners=("foreign-project",),
        scientific_markers=("MVE_FINALIZED",),
    )
    assert result["status"] == V4_RECOVERY_RUNTIME_BLOCKED
    assert "GPU1_PROJECT_PROCESS_PRESENT" in result["reason_codes"]
    assert "SCIENTIFIC_MARKER_PRESENT_GLOBAL" in result["reason_codes"]