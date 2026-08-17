"""Round-2 CPU-only contracts for a fresh 60-unit development Pilot."""

from __future__ import annotations

import hashlib
import importlib.util
from itertools import combinations
import json
from pathlib import Path
import sys

import pytest

from georeliab_mve.v4_counterfactuals import SCIENTIFIC_MODELS, SCIENTIFIC_STATES
from georeliab_mve.v4_scoped import (
    V4_PILOT_INCONCLUSIVE,
    V4_PILOT_SCIENTIFIC_NO_GO,
    build_pilot_partition,
    evaluate_pilot,
)


HARNESS_PATH = Path(__file__).with_name("pilot_round2_harness.py")
SPEC = importlib.util.spec_from_file_location("pilot_round2_harness", HARNESS_PATH)
assert SPEC is not None and SPEC.loader is not None
harness = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = harness
SPEC.loader.exec_module(harness)

ORDERED_VIEWS = (1, 7, 13, 19, 25, 31, 37, 43)
POSE_PAIRS = tuple(combinations(ORDERED_VIEWS, 2))
PROTOCOL_ID = "georeliab-v4"
PROTOCOL_SHA = "1" * 64
SCHEDULE_IDENTITY = "2" * 64
SOURCE_SHA = "3" * 64
GPU_UUID = "GPU-" + "4" * 32


def _partition():
    return build_pilot_partition("a" * 64)


def _pilot_records() -> list[dict[str, object]]:
    partition = _partition()
    records: list[dict[str, object]] = []
    for unit_id in partition.primary_unit_ids:
        model_id, scene_text, state_id = unit_id.split(":")
        records.append(
            {
                "unit_id": unit_id,
                "model_id": model_id,
                "scene_id": int(scene_text),
                "state_id": state_id,
                "ordered_view_ids": list(ORDERED_VIEWS),
                "pose_pairs": [list(pair) for pair in POSE_PAIRS],
                "prediction_provenance": {
                    "source_attempt_id": "development-pilot-01"
                },
                "receipt_provenance": {
                    "source_attempt_id": "development-pilot-01"
                },
                "ledger_provenance": {
                    "source_attempt_id": "development-pilot-01"
                },
            }
        )
    return records


@pytest.mark.parametrize(
    ("channel", "forbidden"),
    (
        ("prediction_provenance", "/archive/attempt-05/predictions"),
        ("receipt_provenance", "local_gate2_run_01"),
        ("ledger_provenance", "recovery-smoke-ledger"),
    ),
)
def test_nested_attempt05_and_gate2_provenance_is_rejected(
    channel: str,
    forbidden: str,
) -> None:
    payload = {"records": _pilot_records()}
    payload["records"][0][channel]["source_attempt_id"] = forbidden

    with pytest.raises(harness.PilotRound2ContractError, match="provenance"):
        harness.scan_forbidden_provenance(payload)


def test_exact_sixty_records_bind_eight_views_and_all_twenty_eight_pairs() -> None:
    partition = _partition()
    harness.validate_pilot_records(
        _pilot_records(),
        expected_unit_ids=partition.primary_unit_ids,
        expected_views_by_scene={
            scene_id: ORDERED_VIEWS for scene_id in partition.primary_scene_ids
        },
    )


@pytest.mark.parametrize(
    "mutation",
    ("view_order", "missing_view", "missing_pair", "duplicate_pair"),
)
def test_view_or_pose_pair_drift_fails_closed(mutation: str) -> None:
    partition = _partition()
    records = _pilot_records()
    if mutation == "view_order":
        records[0]["ordered_view_ids"][:2] = reversed(
            records[0]["ordered_view_ids"][:2]
        )
    elif mutation == "missing_view":
        records[0]["ordered_view_ids"].pop()
    elif mutation == "missing_pair":
        records[0]["pose_pairs"].pop()
    else:
        records[0]["pose_pairs"][-1] = records[0]["pose_pairs"][0]

    with pytest.raises(harness.PilotRound2ContractError, match="view|pair"):
        harness.validate_pilot_records(
            records,
            expected_unit_ids=partition.primary_unit_ids,
            expected_views_by_scene={
                scene_id: ORDERED_VIEWS for scene_id in partition.primary_scene_ids
            },
        )


def _execution_contract() -> dict[str, object]:
    return {
        "schema_version": "georeliab-v4-development-pilot-execution-contract-1.0",
        "validation_class": harness.DEVELOPMENT_EVIDENCE_ONLY,
        "scientific_result": harness.NO_SCIENTIFIC_RESULT,
        "unit_ids": list(_partition().primary_unit_ids),
        "gpu_uuid": GPU_UUID,
        "physical_gpu_count": 1,
        "model_order": list(SCIENTIFIC_MODELS),
        "sequential_model_execution": True,
        "sequential_unit_execution": True,
        "fallback_allowed": False,
        "auto_retry_allowed": False,
        "device_switch_allowed": False,
        "grid_reduction_allowed": False,
        "downstream_advance_allowed": False,
        "max_gpu_seconds": 21_600,
        "max_wall_seconds": 43_200,
        "max_storage_bytes": 25 * 1024**3,
        "output_root": "/home/hryang/georeliab-v4-pilot/runs/development-pilot-01",
    }


def test_valid_execution_contract_is_single_gpu_and_strictly_sequential() -> None:
    partition = _partition()
    harness.validate_pilot_execution_contract(
        _execution_contract(), expected_unit_ids=partition.primary_unit_ids
    )


@pytest.mark.parametrize(
    ("field", "bad_value"),
    (
        ("physical_gpu_count", 2),
        ("model_order", ["MASt3R", "VGGT"]),
        ("sequential_model_execution", False),
        ("sequential_unit_execution", False),
        ("fallback_allowed", True),
        ("auto_retry_allowed", True),
        ("device_switch_allowed", True),
        ("grid_reduction_allowed", True),
        ("downstream_advance_allowed", True),
    ),
)
def test_gpu_parallel_fallback_retry_switch_or_downstream_is_rejected(
    field: str,
    bad_value: object,
) -> None:
    payload = _execution_contract()
    payload[field] = bad_value

    with pytest.raises(harness.PilotRound2ContractError):
        harness.validate_pilot_execution_contract(
            payload, expected_unit_ids=_partition().primary_unit_ids
        )


def _common_evidence() -> dict[str, object]:
    return {
        "schema_version": "georeliab-v4-development-evidence-1.0",
        "protocol_id": PROTOCOL_ID,
        "protocol_sha256": PROTOCOL_SHA,
        "schedule_identity_sha256": SCHEDULE_IDENTITY,
        "created_at": "2026-08-18T00:00:00Z",
        "source_sha256": SOURCE_SHA,
        "validation_class": harness.DEVELOPMENT_EVIDENCE_ONLY,
        "scientific_result": harness.NO_SCIENTIFIC_RESULT,
    }


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_manifest(root: Path) -> None:
    files = sorted(
        path
        for path in root.rglob("*")
        if path.is_file() and path.name != "MANIFEST.sha256"
    )
    lines = [f"{_sha256(path)}  {path.relative_to(root).as_posix()}" for path in files]
    (root / "MANIFEST.sha256").write_text("\n".join(lines) + "\n", encoding="ascii")


def _evidence_root(tmp_path: Path) -> Path:
    root = tmp_path / "pilot-evidence"
    for relative in harness.REQUIRED_EVIDENCE_FILES:
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.suffix == ".json":
            path.write_text(
                json.dumps(_common_evidence(), sort_keys=True) + "\n",
                encoding="utf-8",
            )
        elif path.suffix == ".jsonl":
            path.write_text(
                json.dumps(_common_evidence(), sort_keys=True) + "\n",
                encoding="utf-8",
            )
        else:
            path.write_text("development evidence only\n", encoding="utf-8")
    _write_manifest(root)
    return root


def test_complete_evidence_bundle_and_manifest_verify(tmp_path: Path) -> None:
    harness.verify_pilot_evidence_bundle(_evidence_root(tmp_path))


@pytest.mark.parametrize("mutation", ("missing", "tamper", "unlisted", "symlink"))
def test_missing_tampered_unlisted_or_symlink_evidence_fails_closed(
    tmp_path: Path,
    mutation: str,
) -> None:
    root = _evidence_root(tmp_path)
    target = root / "logs/pilot-console.log"
    if mutation == "missing":
        target.unlink()
    elif mutation == "tamper":
        target.write_text("tampered\n", encoding="utf-8")
    elif mutation == "unlisted":
        (root / "unexpected.txt").write_text("unexpected\n", encoding="utf-8")
    else:
        source = root / "logs/pilot-gpu-preflight.log"
        target.unlink()
        target.symlink_to(source)
        _write_manifest(root)

    with pytest.raises(harness.PilotRound2ContractError):
        harness.verify_pilot_evidence_bundle(root)


def test_nonzero_command_exit_is_logged_without_false_pass(tmp_path: Path) -> None:
    status = harness.run_logged_command(
        ["/bin/bash", "-c", "printf 'before-failure\\n'; exit 7"],
        console_log=tmp_path / "console.log",
        status_path=tmp_path / "status.json",
    )

    assert status.raw_exit_code == 7
    assert status.tee_exit_code == 0
    assert not status.passed
    assert "before-failure" in status.console_log.read_text(encoding="utf-8")
    assert harness.validate_logged_status(status.status_path) == status


def test_zero_exit_is_the_only_logged_pass(tmp_path: Path) -> None:
    status = harness.run_logged_command(
        ["/bin/bash", "-c", "printf 'complete\\n'"],
        console_log=tmp_path / "console.log",
        status_path=tmp_path / "status.json",
    )

    assert status.raw_exit_code == 0
    assert status.tee_exit_code == 0
    assert status.passed


def test_tampered_logged_status_cannot_turn_failure_into_pass(tmp_path: Path) -> None:
    status = harness.run_logged_command(
        ["/bin/bash", "-c", "exit 9"],
        console_log=tmp_path / "console.log",
        status_path=tmp_path / "status.json",
    )
    payload = json.loads(status.status_path.read_text(encoding="utf-8"))
    payload["passed"] = True
    status.status_path.write_text(json.dumps(payload) + "\n", encoding="utf-8")

    with pytest.raises(harness.PilotRound2ContractError, match="exit|pass"):
        harness.validate_logged_status(status.status_path)


def _metric_rows(
    scenes: tuple[int, ...],
    *,
    score_mode: str = "strong",
    failure_mode: str = "fog",
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    dominant_scene = scenes[-1]
    for model in SCIENTIFIC_MODELS:
        for scene in scenes:
            for index, state in enumerate(SCIENTIFIC_STATES):
                if failure_mode == "lighting-only":
                    failure = state == "L7"
                else:
                    failure = state in {"fog-s2", "fog-s3"}
                if score_mode == "zero":
                    score = 0.5
                elif score_mode == "one-reversed" and scene == dominant_scene:
                    score = 2.0 * float(not failure) + index / 100.0
                else:
                    score = float(failure) + index / 100.0
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


def test_no_fog_failure_group_is_inconclusive_without_deleting_scenes() -> None:
    partition = _partition()
    decision = evaluate_pilot(
        _metric_rows(partition.primary_scene_ids, failure_mode="lighting-only"),
        scene_ids=partition.primary_scene_ids,
        partition=partition,
    )

    assert decision.status == V4_PILOT_INCONCLUSIVE
    assert len(decision.models) == len(SCIENTIFIC_MODELS)
    assert all(model.scene_count == 3 for model in decision.models)
    assert all(model.reason_code == "NO_FAILURE_GROUPS" for model in decision.models)


def test_zero_ranking_effect_is_scientific_no_go_not_model_conflict() -> None:
    partition = _partition()
    decision = evaluate_pilot(
        _metric_rows(partition.primary_scene_ids, score_mode="zero"),
        scene_ids=partition.primary_scene_ids,
        partition=partition,
    )

    assert decision.macro_auroc == pytest.approx(0.5)
    assert decision.status == V4_PILOT_SCIENTIFIC_NO_GO


def test_loso_and_scene_dominance_failure_stays_inconclusive() -> None:
    partition = _partition()
    decision = evaluate_pilot(
        _metric_rows(partition.primary_scene_ids, score_mode="one-reversed"),
        scene_ids=partition.primary_scene_ids,
        partition=partition,
    )

    assert decision.status == V4_PILOT_INCONCLUSIVE
    assert any(value <= 0.5 for value in decision.loso_auroc)
    assert not decision.scene_dominance_pass
    assert decision.reason_code == "SCENE_DOMINANCE_FAILURE"
