from __future__ import annotations

import hashlib
import json
from pathlib import Path
import zipfile

import numpy as np
import pytest

from georeliab_mve.artifact_storage import (
    npz_fingerprint,
    reencode_npz_lossless,
)
from georeliab_mve.science_lock import (
    BASE_PROJECT_COMMIT,
    LOCKED_SCIENCE_FILES,
    ScienceLockError,
    validate_science_lock,
)
from georeliab_mve.storage_audit import (
    PLAN_SCHEMA,
    StorageAuditError,
    apply_storage_plan,
    capture_storage_snapshot,
    classify_storage_path,
    resolve_under_root,
    storage_audit,
)


SOURCE_ROOT = Path(__file__).resolve().parents[1]


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_science_lock_matches_exact_f539_inputs():
    report = validate_science_lock(SOURCE_ROOT)
    assert report["status"] == "PASS"
    assert report["base_project_commit"] == BASE_PROJECT_COMMIT
    assert report["hash_algorithm"] == "sha256-canonical-lf-v1"
    assert {
        row["path"]: row["sha256"] for row in report["files"]
    } == dict(LOCKED_SCIENCE_FILES)


def test_science_lock_is_checkout_line_ending_invariant(tmp_path: Path):
    for relative in LOCKED_SCIENCE_FILES:
        target = tmp_path / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        raw = (SOURCE_ROOT / relative).read_bytes()
        canonical = raw.replace(b"\r\n", b"\n").replace(b"\r", b"\n")
        target.write_bytes(canonical)

    report = validate_science_lock(tmp_path)

    assert report["status"] == "PASS"
    assert report["hash_algorithm"] == "sha256-canonical-lf-v1"


def test_science_lock_fails_closed_on_tamper(tmp_path: Path):
    for relative in LOCKED_SCIENCE_FILES:
        target = tmp_path / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((SOURCE_ROOT / relative).read_bytes())
    (tmp_path / "configs" / "dual_mve_protocol.toml").write_text(
        "tampered", encoding="utf-8"
    )
    with pytest.raises(ScienceLockError, match="science lock mismatch"):
        validate_science_lock(tmp_path)


@pytest.mark.parametrize(
    ("relative", "expected"),
    (
        ("manifests/split_view_manifest.json", "L0"),
        ("stage/test/bundles/vggt/x/geometry_prediction.npz", "L1"),
        ("stage/smoke/bundles/mast3r/x/adapter/mast3r_cache/a.pth", "L2"),
        ("cache/Rectified.sparse-index.zip", "L2"),
        ("stage/test/x.partial", "L2"),
        ("stage/test/x.partial/nested/data.bin", "L2"),
    ),
)
def test_storage_level_classification(relative: str, expected: str):
    assert classify_storage_path(Path(relative))[0] == expected


def test_storage_audit_reports_logical_and_allocated_and_writes_plan(tmp_path: Path):
    (tmp_path / "manifests").mkdir()
    (tmp_path / "manifests" / "x.json").write_text("{}", encoding="utf-8")
    cache = tmp_path / "cache"
    cache.mkdir()
    sparse = cache / "Rectified.sparse-index.zip"
    sparse.write_bytes(b"x" * 17)
    snapshot, plan = storage_audit(tmp_path, source_root=SOURCE_ROOT)
    assert snapshot["logical_bytes"] >= 19
    assert snapshot["allocated_bytes"] >= 0
    assert snapshot["levels"]["L0"]["file_count"] >= 1
    assert snapshot["levels"]["L2"]["logical_bytes"] == 17
    assert plan["schema_version"] == PLAN_SCHEMA
    assert plan["mutation_enabled"] is True
    assert (tmp_path / "artifacts" / "storage_before.json").exists()
    assert (tmp_path / "artifacts" / "storage_plan.json").exists()
    assert snapshot["resource_accounting"]["wall_runtime_seconds"] == 0.0
    assert snapshot["resource_accounting"]["gpu_inference_seconds"] == 0.0


def test_path_boundary_rejects_root_and_escape(tmp_path: Path):
    with pytest.raises(StorageAuditError, match="runtime root itself"):
        resolve_under_root(tmp_path, tmp_path)
    with pytest.raises(StorageAuditError, match="escaped"):
        resolve_under_root(tmp_path, tmp_path.parent / "outside")


def test_apply_rejects_wrong_digest_and_audit_only_plan(tmp_path: Path):
    plan = {
        "schema_version": PLAN_SCHEMA,
        "runtime_root": str(tmp_path.resolve()),
        "mutation_enabled": False,
        "actions": [{"action": "review_ephemeral", "path": "../escape"}],
    }
    path = tmp_path / "plan.json"
    path.write_text(json.dumps(plan, sort_keys=True) + "\n", encoding="utf-8")
    with pytest.raises(StorageAuditError, match="digest mismatch"):
        apply_storage_plan(tmp_path, path, expected_plan_sha256="0" * 64)
    with pytest.raises(StorageAuditError, match="audit-only"):
        apply_storage_plan(tmp_path, path, expected_plan_sha256=_sha(path))


def test_digest_bound_apply_reencodes_npz_and_deletes_partial_losslessly(
    tmp_path: Path,
):
    geometry = (
        tmp_path
        / "stage"
        / "test"
        / "bundles"
        / "vggt"
        / "sample"
        / "geometry_prediction.npz"
    )
    geometry.parent.mkdir(parents=True)
    arrays = {
        "points_world": np.arange(24_000, dtype=np.float64).reshape(8_000, 3),
        "view_id": np.arange(8_000, dtype=np.int64) % 8,
    }
    np.savez(geometry, **arrays)
    before = npz_fingerprint(geometry)
    stale = tmp_path / "work.partial" / "stale.bin"
    stale.parent.mkdir()
    stale.write_bytes(b"ephemeral")

    _snapshot, plan = storage_audit(tmp_path, source_root=SOURCE_ROOT)
    plan_path = tmp_path / "artifacts" / "storage_plan.json"
    actions = plan["actions"]
    assert {action["action"] for action in actions} == {
        "lossless_reencode_npz",
        "delete_unreferenced_ephemeral",
    }

    receipt = apply_storage_plan(
        tmp_path,
        plan_path,
        expected_plan_sha256=_sha(plan_path),
    )

    assert receipt["status"] == "PASS"
    assert receipt["actions_applied"] == 2
    assert not stale.exists()
    assert npz_fingerprint(geometry) == before
    with zipfile.ZipFile(geometry) as archive:
        assert all(
            row.compress_type == zipfile.ZIP_DEFLATED
            for row in archive.infolist()
        )
    repeated = apply_storage_plan(
        tmp_path,
        plan_path,
        expected_plan_sha256=_sha(plan_path),
    )
    assert repeated["status"] == "PASS"
    assert npz_fingerprint(geometry) == before


def test_apply_resumes_a_digest_bound_quarantine(tmp_path: Path):
    stale = tmp_path / "orphan.partial" / "payload.bin"
    stale.parent.mkdir()
    stale.write_bytes(b"resume-me")
    _snapshot, plan = storage_audit(tmp_path, source_root=SOURCE_ROOT)
    plan_path = tmp_path / "artifacts" / "storage_plan.json"
    plan_sha = _sha(plan_path)
    action = plan["actions"][0]
    action_bytes = (
        json.dumps(
            action,
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        )
        + "\n"
    ).encode("utf-8")
    action_sha = hashlib.sha256(action_bytes).hexdigest()
    receipt_path = (
        tmp_path
        / "artifacts"
        / "storage-retention"
        / "actions"
        / f"000000-{action_sha[:16]}.json"
    )
    receipt_path.parent.mkdir(parents=True)
    ready = {
        "schema_version": "georeliab-storage-action-receipt-v1",
        "plan_file_sha256": plan_sha,
        "action_sha256": action_sha,
        "action": action,
        "status": "READY_TO_REMOVE",
        "result_sha256": None,
    }
    receipt_path.write_text(
        json.dumps(ready, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    quarantine = (
        tmp_path
        / "artifacts"
        / "storage-retention"
        / "quarantine"
        / action_sha
        / stale.name
    )
    quarantine.parent.mkdir(parents=True)
    stale.replace(quarantine)

    receipt = apply_storage_plan(
        tmp_path,
        plan_path,
        expected_plan_sha256=plan_sha,
    )

    assert receipt["status"] == "PASS"
    assert not quarantine.exists()
    assert not stale.exists()


def test_apply_resumes_after_atomic_npz_swap(tmp_path: Path):
    geometry = (
        tmp_path
        / "stage"
        / "test"
        / "bundles"
        / "vggt"
        / "sample"
        / "geometry_prediction.npz"
    )
    geometry.parent.mkdir(parents=True)
    np.savez(
        geometry,
        points_world=np.arange(12_000, dtype=np.float64).reshape(4_000, 3),
    )
    before = npz_fingerprint(geometry)
    _snapshot, plan = storage_audit(tmp_path, source_root=SOURCE_ROOT)
    plan_path = tmp_path / "artifacts" / "storage_plan.json"
    plan_sha = _sha(plan_path)
    action = plan["actions"][0]
    action_bytes = (
        json.dumps(
            action,
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        )
        + "\n"
    ).encode("utf-8")
    action_sha = hashlib.sha256(action_bytes).hexdigest()
    partial = geometry.with_name(
        geometry.name + ".storage-retention.partial"
    )
    backup = geometry.with_name(
        geometry.name + ".storage-retention.original"
    )
    reencode_npz_lossless(geometry, partial)
    result_sha = _sha(partial)
    receipt_path = (
        tmp_path
        / "artifacts"
        / "storage-retention"
        / "actions"
        / f"000000-{action_sha[:16]}.json"
    )
    receipt_path.parent.mkdir(parents=True)
    ready = {
        "schema_version": "georeliab-storage-action-receipt-v1",
        "plan_file_sha256": plan_sha,
        "action_sha256": action_sha,
        "action": action,
        "status": "READY_TO_REPLACE",
        "result_sha256": result_sha,
    }
    receipt_path.write_text(
        json.dumps(ready, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    geometry.replace(backup)
    partial.replace(geometry)

    receipt = apply_storage_plan(
        tmp_path,
        plan_path,
        expected_plan_sha256=plan_sha,
    )

    assert receipt["status"] == "PASS"
    assert not backup.exists()
    assert npz_fingerprint(geometry) == before


def test_repeated_dry_run_keeps_retention_actions_stable(tmp_path: Path):
    stale = tmp_path / "debug" / "stale.bin"
    stale.parent.mkdir()
    stale.write_bytes(b"stale")

    _first_snapshot, first = storage_audit(tmp_path, source_root=SOURCE_ROOT)
    _second_snapshot, second = storage_audit(tmp_path, source_root=SOURCE_ROOT)

    assert first["actions"] == second["actions"]


def test_referenced_debug_file_is_not_admitted_for_deletion(tmp_path: Path):
    debug = tmp_path / "debug" / "trace.bin"
    debug.parent.mkdir()
    debug.write_bytes(b"needed")
    manifest = tmp_path / "manifests" / "reference.json"
    manifest.parent.mkdir()
    manifest.write_text(
        json.dumps({"trace_uri": debug.resolve().as_uri()}),
        encoding="utf-8",
    )

    _snapshot, plan = storage_audit(tmp_path, source_root=SOURCE_ROOT)

    assert all(action["path"] != "debug/trace.bin" for action in plan["actions"])


def test_sparse_index_requires_and_revalidates_materialization_proof(tmp_path: Path):
    member = tmp_path / "materialized" / "Rectified" / "scan1" / "rgb.png"
    member.parent.mkdir(parents=True)
    member.write_bytes(b"rgb")
    sparse = tmp_path / "cache" / "Rectified.sparse-index.zip"
    sparse.parent.mkdir()
    sparse.write_bytes(b"sparse-index")
    materialization = tmp_path / "manifests" / "frozen_materialization.json"
    materialization.parent.mkdir()
    materialization.write_text(
        json.dumps(
            {
                "schema_version": "frozen-materialization-v1",
                "archives": {
                    "Rectified.zip": {
                        "bytes": 129_000_000_000,
                        "central_directory_sha256": "a" * 64,
                        "members": {
                            "rgb": {
                                "path": str(member),
                                "raw_sha256": _sha(member),
                                "uncompressed_size": member.stat().st_size,
                            }
                        },
                    }
                },
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    _snapshot, plan = storage_audit(tmp_path, source_root=SOURCE_ROOT)
    sparse_actions = [
        action
        for action in plan["actions"]
        if action["action"] == "delete_verified_sparse_index"
    ]
    assert len(sparse_actions) == 1
    apply_storage_plan(
        tmp_path,
        tmp_path / "artifacts" / "storage_plan.json",
        expected_plan_sha256=_sha(tmp_path / "artifacts" / "storage_plan.json"),
    )
    assert not sparse.exists()


def test_snapshot_records_legacy_accounting_as_unavailable_without_bundle(tmp_path: Path):
    ledger = tmp_path / "stage" / "smoke" / "ledger.jsonl"
    ledger.parent.mkdir(parents=True)
    ledger.write_text(
        json.dumps(
            {
                "state": "completed",
                "item_identity": "missing",
                "runtime_seconds": 12.0,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    snapshot = capture_storage_snapshot(tmp_path, source_root=SOURCE_ROOT)
    assert snapshot["resource_accounting"]["wall_runtime_seconds"] == 12.0
    assert snapshot["resource_accounting"]["status"] == "UNAVAILABLE"
    assert snapshot["status"] == "BLOCKED"
