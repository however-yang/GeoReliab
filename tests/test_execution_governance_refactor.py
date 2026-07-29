from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import georeliab_mve.execution_governance as governance
import georeliab_mve.runner as runner
from georeliab_mve.execution_governance import (
    ExecutionGovernanceError,
    archive_superseded_results,
    create_p2a_selection_manifest,
    evaluate_p2a_completion,
    load_p2a_selection_manifest,
    record_gpu_selection,
    select_p2a_items,
    validate_gpu_selection,
    validate_superseded_archive,
)
from georeliab_mve.science_lock import BASE_PROJECT_COMMIT
from georeliab_mve.storage_audit import FileUsage
from georeliab_mve.storage_retention import build_retention_actions


COMMIT = "1" * 40
TREE = "2" * 40


def _schedule() -> tuple[SimpleNamespace, ...]:
    return tuple(
        SimpleNamespace(model=model, identity=f"{model.lower()}-{index:03d}")
        for model in ("VGGT", "MASt3R")
        for index in range(100)
    )


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")


def _old_manifest(path: Path, index: int) -> None:
    _write_json(
        path,
        {
            "run_id": f"old-{index}",
            "provenance": {"project_commit": BASE_PROJECT_COMMIT},
        },
    )
    path.with_name("payload.bin").write_bytes(f"payload-{index}".encode("ascii"))


def _populate_superseded_results(root: Path) -> None:
    for index in range(16):
        _old_manifest(
            root
            / "preflight-real"
            / f"repeat-{'a' if index < 8 else 'b'}"
            / "stage"
            / "preflight"
            / "bundles"
            / "vggt"
            / f"item-{index:03d}"
            / "run_manifest.json",
            index,
        )
    for index in range(75):
        _old_manifest(
            root
            / "stage"
            / "smoke"
            / "bundles"
            / ("vggt" if index % 2 == 0 else "mast3r")
            / f"item-{index:03d}"
            / "run_manifest.json",
            100 + index,
        )
    _write_json(root / "artifacts" / "p1_preflight.json", {"status": "OK"})


def test_p2a_selection_is_exact_deterministic_and_bound_to_full_schedule(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    items = _schedule()
    first = select_p2a_items(items)
    second = select_p2a_items(tuple(reversed(items)))
    assert [item.identity for item in first] == [item.identity for item in second]
    assert len(first) == 50
    assert sum(item.model == "VGGT" for item in first) == 25
    assert sum(item.model == "MASt3R" for item in first) == 25

    root = tmp_path / "runtime"
    artifacts = root / "artifacts"
    artifacts.mkdir(parents=True)
    storage_before = artifacts / "storage_before.json"
    storage_plan = artifacts / "storage_plan.json"
    _write_json(
        storage_before,
        {
            "projection": {
                "r1_status": "PASS",
                "full_worst_path_bytes": 800_000_000_000,
                "model_smoke_retained_byte_p95": {
                    "vggt": 100,
                    "mast3r": 200,
                },
            }
        },
    )
    _write_json(storage_plan, {"plan_payload_sha256": "3" * 64})
    monkeypatch.setattr(governance, "_git_identity", lambda _root: (COMMIT, TREE))
    monkeypatch.setattr(runner, "full_schedule", lambda _root, stage: items)

    payload = create_p2a_selection_manifest(
        root,
        source_root=tmp_path,
        storage_before_path=storage_before,
        storage_plan_path=storage_plan,
    )
    path = artifacts / "p2a_selection_manifest.json"
    loaded, rebound = load_p2a_selection_manifest(
        root,
        path,
        source_root=tmp_path,
    )
    assert payload == rebound
    assert len(loaded) == 50
    assert rebound["full_schedule_count"] == 200
    assert rebound["selected_per_model"] == {"vggt": 25, "mast3r": 25}

    tampered = dict(rebound)
    tampered["selected_items"] = list(rebound["selected_items"])
    tampered["selected_items"][0] = dict(tampered["selected_items"][0])
    tampered["selected_items"][0]["identity"] = "tampered"
    _write_json(path, tampered)
    with pytest.raises(ExecutionGovernanceError, match="frozen rule"):
        load_p2a_selection_manifest(root, path, source_root=tmp_path)


def test_p2a_selection_refuses_preexisting_smoke_execution(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "runtime"
    artifacts = root / "artifacts"
    artifacts.mkdir(parents=True)
    storage_before = artifacts / "storage_before.json"
    storage_plan = artifacts / "storage_plan.json"
    _write_json(
        storage_before,
        {
            "projection": {
                "r1_status": "PASS",
                "full_worst_path_bytes": 800_000_000_000,
                "model_smoke_retained_byte_p95": {
                    "vggt": 100,
                    "mast3r": 200,
                },
            }
        },
    )
    _write_json(storage_plan, {"plan_payload_sha256": "3" * 64})
    ledger = root / "stage" / "smoke" / "ledger.jsonl"
    ledger.parent.mkdir(parents=True)
    ledger.write_text(
        json.dumps(
            {
                "item_identity": "premature",
                "state": "completed",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(governance, "_git_identity", lambda _root: (COMMIT, TREE))
    monkeypatch.setattr(runner, "full_schedule", lambda _root, stage: _schedule())

    with pytest.raises(ExecutionGovernanceError, match="before any new-commit smoke"):
        create_p2a_selection_manifest(
            root,
            source_root=tmp_path,
            storage_before_path=storage_before,
            storage_plan_path=storage_plan,
        )


def test_selection_manifest_is_forbidden_for_scientific_stages(
    tmp_path: Path,
) -> None:
    context = runner.RunnerContext(
        root=tmp_path,
        output_root=tmp_path,
        config_path=None,
        device="cpu",
    )
    for stage in ("test", "zero-update"):
        with pytest.raises(runner.RunnerError, match="forbidden for P3/P5"):
            runner.run_stage(
                context,
                stage=stage,
                model="vggt",
                shard="0/1",
                dry_run=True,
                selection_manifest=tmp_path / "selection.json",
            )

    with pytest.raises(runner.RunnerError, match="requires model all"):
        runner.run_stage(
            context,
            stage="smoke",
            model="vggt",
            shard="0/1",
            dry_run=True,
            selection_manifest=tmp_path / "selection.json",
        )


def test_gpu_selection_requires_explicit_exact_commit_and_device(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(governance, "_git_identity", lambda _root: (COMMIT, TREE))
    with pytest.raises(ExecutionGovernanceError, match="GPU_SELECTION_REQUIRED"):
        record_gpu_selection(
            tmp_path,
            source_root=tmp_path,
            project_commit=COMMIT,
            device="cuda:1",
            explicit_user_selection=False,
        )

    payload = record_gpu_selection(
        tmp_path,
        source_root=tmp_path,
        project_commit=COMMIT,
        device="cuda:1",
        explicit_user_selection=True,
    )
    assert payload["max_concurrent_gpus"] == 1
    assert payload["multi_shard_execution"] == "SEQUENTIAL"
    assert (
        validate_gpu_selection(
            tmp_path,
            source_root=tmp_path,
            project_commit=COMMIT,
            device="cuda:1",
        )
        == payload
    )
    with pytest.raises(ExecutionGovernanceError, match="exact-commit/device"):
        validate_gpu_selection(
            tmp_path,
            source_root=tmp_path,
            project_commit=COMMIT,
            device="cuda:0",
        )


def test_superseded_archive_precedes_smoke_retention_and_is_idempotent(
    tmp_path: Path,
) -> None:
    root = tmp_path / "runtime"
    _populate_superseded_results(root)
    old_npz = (
        root
        / "stage"
        / "smoke"
        / "bundles"
        / "vggt"
        / "item-extra"
        / "geometry_prediction.npz"
    )
    old_npz.parent.mkdir(parents=True)
    np.savez(old_npz, points=np.arange(30, dtype=np.float64).reshape(10, 3))
    old_row = FileUsage(
        path=old_npz.relative_to(root).as_posix(),
        level="L1",
        logical_bytes=old_npz.stat().st_size,
        allocated_bytes=old_npz.stat().st_size,
        reason="scientific array cache",
    )
    assert build_retention_actions(root, [old_row]) == []

    receipt = archive_superseded_results(root)
    assert receipt["status"] == "PASS"
    assert receipt["p1_item_count"] == 16
    assert receipt["p2_item_count"] == 75
    assert not (root / "preflight-real").exists()
    assert not (root / "stage" / "smoke").exists()
    assert not (root / "artifacts" / "p1_preflight.json").exists()
    assert validate_superseded_archive(root) == receipt
    assert archive_superseded_results(root) == receipt

    new_npz = (
        root
        / "stage"
        / "smoke"
        / "bundles"
        / "vggt"
        / "new-commit-item"
        / "geometry_prediction.npz"
    )
    new_npz.parent.mkdir(parents=True)
    np.savez(new_npz, points=np.arange(30, dtype=np.float64).reshape(10, 3))
    new_row = FileUsage(
        path=new_npz.relative_to(root).as_posix(),
        level="L1",
        logical_bytes=new_npz.stat().st_size,
        allocated_bytes=new_npz.stat().st_size,
        reason="scientific array cache",
    )
    actions = build_retention_actions(root, [new_row])
    assert [row["action"] for row in actions] == ["lossless_reencode_npz"]

    archive_path = Path(receipt["archive_path"])
    archive_path.write_bytes(b"tampered")
    with pytest.raises(ExecutionGovernanceError, match="digest binding"):
        validate_superseded_archive(root)
    assert build_retention_actions(root, [new_row]) == []


def test_p2a_completion_admits_model_invalid_but_rejects_exceptions_and_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    selected = select_p2a_items(_schedule())
    rows = [
        {
            "item_identity": item.identity,
            "state": "completed",
            "reason_code": "MODEL_INVALID_OUTPUT" if index == 0 else "OK",
            "invalid_prediction": index == 0,
        }
        for index, item in enumerate(selected)
    ]
    for item in selected:
        bundle = (
            tmp_path
            / "stage"
            / "smoke"
            / "bundles"
            / item.model.lower()
            / item.identity
        )
        bundle.mkdir(parents=True)
        (bundle / "retained.bin").write_bytes(b"x" * 10)

    manifest = {
        "predicted_retained_bytes": 500,
        "full_worst_path_bytes": 800_000_000_000,
        "full_schedule_fingerprint": "4" * 64,
    }
    monkeypatch.setattr(
        governance,
        "load_p2a_selection_manifest",
        lambda *_args, **_kwargs: (selected, manifest),
    )
    monkeypatch.setattr(
        runner,
        "read_stage_ledger",
        lambda *_args, **_kwargs: {"rows": rows},
    )
    monkeypatch.setattr(
        runner,
        "load_completed_bundle",
        lambda *_args, **_kwargs: {"status": "OK"},
    )
    monkeypatch.setattr(
        governance,
        "validate_retention_receipt",
        lambda *_args, **_kwargs: {"status": "PASS"},
    )
    selection = tmp_path / "selection.json"
    selection.write_text("{}\n", encoding="utf-8")

    passed = evaluate_p2a_completion(
        tmp_path,
        selection_path=selection,
        source_root=tmp_path,
        write_terminal=False,
    )
    assert passed["status"] == "PASS"
    assert passed["completed"] == 50
    assert passed["missing"] == 0
    assert passed["invalid"] == 1
    assert passed["mast3r_retention_receipts_validated"] == 25
    assert passed["retained_bytes_relative_error"] == 0.0

    rows.append(
        {
            "item_identity": "premature-non-selected",
            "state": "completed",
            "reason_code": "OK",
        }
    )
    premature = evaluate_p2a_completion(
        tmp_path,
        selection_path=selection,
        source_root=tmp_path,
        write_terminal=False,
    )
    assert premature["reason_code"] == "P2A_NON_SELECTED_SMOKE_EXECUTION"
    rows.pop()

    rows[1]["reason_code"] = "ADAPTER_EXCEPTION"
    failed = evaluate_p2a_completion(
        tmp_path,
        selection_path=selection,
        source_root=tmp_path,
        write_terminal=False,
    )
    assert failed["status"] == "FAIL"
    assert failed["reason_code"] == "P2A_BUNDLE_OR_RETENTION_INVALID"

    rows[1]["reason_code"] = "OK"
    manifest["predicted_retained_bytes"] = 1_000
    drifted = evaluate_p2a_completion(
        tmp_path,
        selection_path=selection,
        source_root=tmp_path,
        write_terminal=False,
    )
    assert drifted["status"] == "FAIL"
    assert drifted["reason_code"] == "P2A_RETAINED_BYTES_PREDICTION_MISS"

    manifest["predicted_retained_bytes"] = 500
    frozen = evaluate_p2a_completion(
        tmp_path,
        selection_path=selection,
        source_root=tmp_path,
        write_terminal=True,
    )
    assert frozen["status"] == "PASS"
    assert frozen["ledger_boundary_row_count"] == 50
    assert (tmp_path / "artifacts" / "p2a_completion.json").is_file()

    rows.append(
        {
            "item_identity": "later-full-p2-item",
            "state": "completed",
            "reason_code": "OK",
        }
    )
    resumed = evaluate_p2a_completion(
        tmp_path,
        selection_path=selection,
        source_root=tmp_path,
        write_terminal=False,
    )
    assert resumed == frozen

    rows[0]["reason_code"] = "PREFIX_TAMPER"
    tampered = evaluate_p2a_completion(
        tmp_path,
        selection_path=selection,
        source_root=tmp_path,
        write_terminal=False,
    )
    assert tampered["reason_code"] == "P2A_LEDGER_BOUNDARY_INVALID"
