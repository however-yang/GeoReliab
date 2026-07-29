from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

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
    assert {
        row["path"]: row["sha256"] for row in report["files"]
    } == dict(LOCKED_SCIENCE_FILES)


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
