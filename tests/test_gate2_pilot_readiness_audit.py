"""CPU-only contracts for Gate 2 to Pilot admission evidence."""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import pytest

from georeliab_mve.v4_attempt05_recovery import build_recovery_smoke_manifest


HARNESS_PATH = Path(__file__).with_name("gate2_pilot_readiness_audit.py")
SPEC = importlib.util.spec_from_file_location(
    "gate2_pilot_readiness_audit", HARNESS_PATH
)
assert SPEC is not None and SPEC.loader is not None
audit = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = audit
SPEC.loader.exec_module(audit)


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_manifest(root: Path) -> None:
    rows = []
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.name != "MANIFEST.sha256":
            rows.append(f"{_sha(path)}  {path.relative_to(root).as_posix()}")
    (root / "MANIFEST.sha256").write_text("\n".join(rows) + "\n", encoding="utf-8")


def _repo_commit(repo: Path) -> str:
    return subprocess.run(
        ("git", "-C", str(repo), "rev-parse", "HEAD"),
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _make_repo(tmp_path: Path) -> tuple[Path, str]:
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "example.py").write_text("VALUE = 1\n", encoding="utf-8")
    commands = (
        ("git", "init", "-q"),
        ("git", "config", "user.email", "audit@example.invalid"),
        ("git", "config", "user.name", "Audit Test"),
        ("git", "add", "example.py"),
        ("git", "commit", "-qm", "fixture"),
    )
    for command in commands:
        subprocess.run(command, cwd=repo, check=True)
    return repo, _repo_commit(repo)


def _write_junit(path: Path, *, tests: int = 886, failures: int = 0) -> None:
    path.write_text(
        (
            f'<testsuites><testsuite tests="{tests}" failures="{failures}" '
            'errors="0" skipped="3" time="1.25" /></testsuites>\n'
        ),
        encoding="utf-8",
    )


def _gate1_bundle(root: Path, commit: str) -> None:
    semantic_sha = "a" * 64
    qualification = {
        "status": audit.GATE1_STATUS,
        "runtime_status": audit.GATE1_RUNTIME_STATUS,
        "case_count": 147,
        "passed_case_count": 147,
        "unknown_count": 0,
        "broad_legacy_reason_count": 0,
        "all_failure_injections_classified": True,
        "gate2_started": False,
        "pilot_started": False,
        "attempt06_started": False,
        "tooling": {"passed": True, "returncode": 0},
        "semantic_result_sha256": semantic_sha,
        "scientific_result": audit.NO_SCIENTIFIC_RESULT,
    }
    source = {
        "current_commit": commit,
        "scientific_assets_zero_drift": True,
        "scientific_result": audit.NO_SCIENTIFIC_RESULT,
    }
    semantic = {
        "semantic_result_sha256": semantic_sha,
        "semantic_payload": {
            "all_failure_injections_classified": True,
            "unknown_count": 0,
        },
        "scientific_result": audit.NO_SCIENTIFIC_RESULT,
    }
    _write_json(root / "qualification.json", qualification)
    _write_json(root / "source-manifest.json", source)
    _write_json(root / "semantic-result.json", semantic)
    _write_manifest(root)


def _smoke_rows(smoke) -> list[dict[str, object]]:
    return [
        {
            "unit_key": unit_key,
            "inference_start_count": smoke.expected_inference_starts[unit_key],
            "completion_count": 1,
            "projection_count": 1,
            "overwrite_count": 0,
            "interruption_phase": smoke.interruption_plan.get(unit_key),
            "gpu_uuid": smoke.gpu_uuid,
            "physical_gpu_index": smoke.physical_gpu_index,
            "canonical_present": True,
            "ledger_committed": True,
            "scientific_marker": audit.NO_SCIENTIFIC_RESULT,
        }
        for unit_key in smoke.unit_keys
    ]


def _gate2_bundle(
    root: Path,
    commit: str,
    *,
    formal: bool = False,
) -> Path:
    manifests = root / "manifests"
    run = root / "runs" / "run-01"
    observations = run / "observations"
    observations.mkdir(parents=True)
    smoke = build_recovery_smoke_manifest(
        schedule_identity_sha256="b" * 64,
        support_scene_ids=tuple(range(1, 11)),
    )
    rows = _smoke_rows(smoke)
    for index, row in enumerate(rows):
        _write_json(observations / f"{index:02d}.json", row)

    view_root = root / "data"
    bindings = []
    ordered_views = list(range(1, 9))
    for scene_id in smoke.scene_ids:
        views = []
        for view_id in ordered_views:
            path = view_root / f"scene-{scene_id}" / f"view-{view_id}.bin"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(f"{scene_id}:{view_id}".encode("ascii"))
            views.append(
                {
                    "view_id": view_id,
                    "path": str(path),
                    "sha256": _sha(path),
                }
            )
        bindings.append(
            {
                "scene_id": scene_id,
                "ordered_view_ids": ordered_views,
                "views": views,
            }
        )
    closure = manifests / "local-gate2-input-closure.json"
    _write_json(
        closure,
        {
            "attempt05_predictions_read": False,
            "bindings": bindings,
            "scientific_result": audit.NO_SCIENTIFIC_RESULT,
        },
    )
    overlay = manifests / "local-gate2-overlay.toml"
    overlay.write_text("[local]\nvalidation = true\n", encoding="utf-8")
    _write_json(
        manifests / "local-gate2-resource-audit.json",
        {
            "formal_gate2_equivalent": formal,
            "scientific_result": audit.NO_SCIENTIFIC_RESULT,
        },
    )
    _write_json(
        run / "input-manifest.json",
        {
            "input_closure_path": str(closure),
            "input_closure_sha256": _sha(closure),
            "overlay_config": str(overlay),
            "overlay_sha256": _sha(overlay),
            "scientific_result": audit.NO_SCIENTIFIC_RESULT,
        },
    )
    _write_json(run / "smoke-manifest.json", smoke.to_dict())
    _write_json(
        run / "source-manifest.json",
        {
            "canonical_commit": commit,
            "production_source_zero_drift": True,
            "scientific_result": audit.NO_SCIENTIFIC_RESULT,
        },
    )
    qualification = {
        "status": audit.FORMAL_GATE2_STATUS if formal else audit.LOCAL_GATE2_STATUS,
        "validation_class": (
            audit.FORMAL_GATE2_CLASS if formal else audit.LOCAL_GATE2_CLASS
        ),
        "formal_gate2_equivalent": formal,
        "expected_unit_count": 12,
        "unit_count": 12,
        "recovery_runtime_status": audit.GATE2_RUNTIME_STATUS,
        "gate2_started": True,
        "pilot_started": False,
        "attempt06_started": False,
        "scientific_result": audit.NO_SCIENTIFIC_RESULT,
    }
    for field in audit._EMPTY_VIOLATION_FIELDS:
        qualification[field] = []
    _write_json(run / "qualification.json", qualification)
    _write_manifest(run)
    return run


def test_g0_requires_exact_clean_commit_and_passing_full_suite(tmp_path: Path) -> None:
    repo, commit = _make_repo(tmp_path)
    junit = tmp_path / "full.junit.xml"
    _write_junit(junit)

    result = audit.audit_g0(
        repo=repo,
        expected_commit=commit,
        junit_path=junit,
        expected_test_count=886,
        ruff_paths=("example.py",),
    )

    assert result["status"] == audit.G0_PASS
    assert result["cpu_suite"]["tests"] == 886


def test_g0_rejects_dirty_worktree(tmp_path: Path) -> None:
    repo, commit = _make_repo(tmp_path)
    junit = tmp_path / "full.junit.xml"
    _write_junit(junit)
    (repo / "untracked.txt").write_text("drift\n", encoding="utf-8")

    with pytest.raises(audit.ReadinessAuditError, match="WORKTREE_NOT_CLEAN"):
        audit.audit_g0(
            repo=repo,
            expected_commit=commit,
            junit_path=junit,
            expected_test_count=886,
            ruff_paths=("example.py",),
        )


@pytest.mark.parametrize("tests,failures", ((885, 0), (886, 1)))
def test_g0_rejects_wrong_test_count_or_failure(
    tmp_path: Path,
    tests: int,
    failures: int,
) -> None:
    repo, commit = _make_repo(tmp_path)
    junit = tmp_path / "full.junit.xml"
    _write_junit(junit, tests=tests, failures=failures)

    with pytest.raises(audit.ReadinessAuditError, match="CPU_TEST|CPU_SUITE"):
        audit.audit_g0(
            repo=repo,
            expected_commit=commit,
            junit_path=junit,
            expected_test_count=886,
            ruff_paths=("example.py",),
        )


def test_gate1_pass_requires_all_147_cases_and_clean_lineage(tmp_path: Path) -> None:
    repo = Path(__file__).resolve().parents[1]
    commit = _repo_commit(repo)
    root = tmp_path / "gate1"
    _gate1_bundle(root, commit)

    result = audit.audit_g1(repo=repo, root=root, target_commit=commit)

    assert result["status"] == audit.G1_PASS
    assert result["passed_case_count"] == 147
    assert result["lineage_bridge"]["critical_paths_zero_drift"] is True


@pytest.mark.parametrize(
    ("field", "bad_value", "reason"),
    (
        ("passed_case_count", 146, "CASES_NOT_ALL_PASS"),
        ("unknown_count", 1, "UNKNOWN_CLASSIFICATION"),
        ("pilot_started", True, "DOWNSTREAM_STARTED"),
        ("scientific_result", "MVE_FINALIZED", "SCIENTIFIC_RESULT"),
    ),
)
def test_gate1_status_mutations_fail_closed(
    tmp_path: Path,
    field: str,
    bad_value: object,
    reason: str,
) -> None:
    repo = Path(__file__).resolve().parents[1]
    commit = _repo_commit(repo)
    root = tmp_path / "gate1"
    _gate1_bundle(root, commit)
    qualification = json.loads((root / "qualification.json").read_text())
    qualification[field] = bad_value
    _write_json(root / "qualification.json", qualification)
    _write_manifest(root)

    with pytest.raises(audit.ReadinessAuditError, match=reason):
        audit.audit_g1(repo=repo, root=root, target_commit=commit)


def test_gate1_manifest_tamper_is_rejected(tmp_path: Path) -> None:
    repo = Path(__file__).resolve().parents[1]
    commit = _repo_commit(repo)
    root = tmp_path / "gate1"
    _gate1_bundle(root, commit)
    (root / "semantic-result.json").write_text("{}\n", encoding="utf-8")

    with pytest.raises(audit.ReadinessAuditError, match="MANIFEST_HASH_MISMATCH"):
        audit.audit_g1(repo=repo, root=root, target_commit=commit)


def test_manifest_allows_recorded_internal_fault_fixture_symlink(
    tmp_path: Path,
) -> None:
    root = tmp_path / "evidence"
    target = root / "fixture.bin"
    target.parent.mkdir(parents=True)
    target.write_bytes(b"fixture")
    link = root / "case" / "opaque-prediction.bin"
    link.parent.mkdir(parents=True)
    link.symlink_to(target)
    _write_manifest(root)

    result = audit.verify_sha256_manifest(root)

    assert result["internal_symlink_count"] == 1


def test_manifest_rejects_symlink_that_escapes_evidence_root(tmp_path: Path) -> None:
    root = tmp_path / "evidence"
    outside = tmp_path / "outside.bin"
    outside.write_bytes(b"outside")
    link = root / "case" / "opaque-prediction.bin"
    link.parent.mkdir(parents=True)
    link.symlink_to(outside)
    _write_manifest(root)

    with pytest.raises(audit.ReadinessAuditError, match="MANIFEST_FILE_UNSAFE"):
        audit.verify_sha256_manifest(root)


def test_local_gate2_integrity_passes_but_is_not_formal(tmp_path: Path) -> None:
    repo = Path(__file__).resolve().parents[1]
    commit = _repo_commit(repo)
    root = tmp_path / "gate2"
    run = _gate2_bundle(root, commit)

    result = audit.audit_g2(
        repo=repo,
        gate2_root=root,
        run_root=run,
        target_commit=commit,
    )

    assert result["status"] == audit.G2_LOCAL_ONLY
    assert result["formal_gate2_equivalent"] is False
    assert result["unit_count"] == 12
    assert result["controlled_interruption_count"] == 3


def test_exact_formal_gate2_contract_can_pass_without_authorizing_pilot(
    tmp_path: Path,
) -> None:
    repo = Path(__file__).resolve().parents[1]
    commit = _repo_commit(repo)
    root = tmp_path / "gate2"
    run = _gate2_bundle(root, commit, formal=True)

    g2 = audit.audit_g2(
        repo=repo,
        gate2_root=root,
        run_root=run,
        target_commit=commit,
    )
    report = audit.build_readiness_report(
        g0={"status": audit.G0_PASS},
        g1={"status": audit.G1_PASS},
        g2=g2,
    )

    assert g2["status"] == audit.G2_FORMAL_PASS
    assert report["status"] == audit.ADMISSION_READY
    assert report["pilot_execution_authorized"] is False
    assert report["pilot_started"] is False
    assert report["automatic_progression_allowed"] is False


@pytest.mark.parametrize(
    ("field", "bad_value", "reason"),
    (
        ("formal_gate2_equivalent", True, "LOCAL_EQUIVALENCE_MISMATCH"),
        ("pilot_started", True, "DOWNSTREAM_STARTED"),
        ("unit_count", 11, "UNIT_COUNT_INVALID"),
        ("scientific_result", "SCIENTIFIC", "SCIENTIFIC_RESULT"),
    ),
)
def test_gate2_mixed_or_unsafe_status_fails_closed(
    tmp_path: Path,
    field: str,
    bad_value: object,
    reason: str,
) -> None:
    repo = Path(__file__).resolve().parents[1]
    commit = _repo_commit(repo)
    root = tmp_path / "gate2"
    run = _gate2_bundle(root, commit)
    qualification = json.loads((run / "qualification.json").read_text())
    qualification[field] = bad_value
    _write_json(run / "qualification.json", qualification)
    _write_manifest(run)

    with pytest.raises(audit.ReadinessAuditError, match=reason):
        audit.audit_g2(
            repo=repo,
            gate2_root=root,
            run_root=run,
            target_commit=commit,
        )


def test_gate2_output_or_input_tamper_is_rejected(tmp_path: Path) -> None:
    repo = Path(__file__).resolve().parents[1]
    commit = _repo_commit(repo)
    root = tmp_path / "gate2"
    run = _gate2_bundle(root, commit)
    observation = next((run / "observations").glob("*.json"))
    observation.write_text("{}\n", encoding="utf-8")

    with pytest.raises(audit.ReadinessAuditError, match="MANIFEST_HASH_MISMATCH"):
        audit.audit_g2(
            repo=repo,
            gate2_root=root,
            run_root=run,
            target_commit=commit,
        )


def test_current_local_class_blocks_pilot_and_names_next_action() -> None:
    report = audit.build_readiness_report(
        g0={"status": audit.G0_PASS},
        g1={"status": audit.G1_PASS},
        g2={"status": audit.G2_LOCAL_ONLY},
    )

    assert report["status"] == audit.ADMISSION_BLOCKED
    assert report["blockers"] == [
        "G2_EVIDENCE_IS_LOCAL_DEVELOPMENT_ONLY",
        "FORMAL_GATE2_QUALIFICATION_MISSING",
    ]
    assert report["next_action"] == "PRODUCE_FRESH_FORMAL_GATE2_QUALIFICATION"
    assert report["pilot_execution_authorized"] is False
    assert report["pilot_partition_frozen"] is False
    assert report["confirmation_started"] is False


def test_audit_output_is_no_clobber(tmp_path: Path) -> None:
    output = tmp_path / "audit.json"
    audit._write_json_no_clobber(output, {"status": audit.ADMISSION_BLOCKED})

    with pytest.raises(audit.ReadinessAuditError, match="OUTPUT_EXISTS"):
        audit._write_json_no_clobber(output, {"status": audit.ADMISSION_READY})


def test_blocked_audit_is_shell_safe_unless_readiness_is_required() -> None:
    assert audit.audit_exit_code(audit.ADMISSION_BLOCKED, require_ready=False) == 0
    assert audit.audit_exit_code(audit.ADMISSION_BLOCKED, require_ready=True) == 3
    assert audit.audit_exit_code(audit.ADMISSION_READY, require_ready=True) == 0
