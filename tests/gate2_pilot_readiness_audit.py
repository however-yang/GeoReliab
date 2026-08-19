"""Test-only Gate 2 to Pilot admission auditor.

The auditor is deliberately outside :mod:`georeliab_mve`.  It reads existing
qualification evidence, verifies hashes and lineage, and emits a fail-closed
readiness report.  It never dispatches a GPU, starts a Pilot, promotes local
development evidence, or writes a scientific marker.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import subprocess
import sys
from typing import Mapping, Sequence
import xml.etree.ElementTree as ET

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from georeliab_mve.v4_attempt05_recovery import (  # noqa: E402
    RecoverySmokeManifest,
    evaluate_recovery_smoke,
)
from tests import local_gate2_prepare as local_gate2  # noqa: E402


SCHEMA_VERSION = "georeliab-v4-gate2-pilot-readiness-audit-1.0"
VALIDATION_CLASS = "GATE2_TO_PILOT_READINESS_AUDIT"
NO_SCIENTIFIC_RESULT = "NO_SCIENTIFIC_RESULT"
G0_PASS = "G0_SOURCE_TOOLCHAIN_PASS"
G1_PASS = "G1_CPU_FAULT_MATRIX_PASS"
G2_FORMAL_PASS = "G2_FORMAL_GPU_SMOKE_PASS"
G2_LOCAL_ONLY = "G2_LOCAL_DEVELOPMENT_ONLY"
ADMISSION_READY = "V4_PILOT_ADMISSION_READY_FOR_EXPLICIT_AUTHORIZATION"
ADMISSION_BLOCKED = "V4_PILOT_ADMISSION_BLOCKED"

GATE1_STATUS = "V4_RECOVERY_CPU_FAULT_MATRIX_PASS"
GATE1_RUNTIME_STATUS = "V4_RECOVERY_RUNTIME_READY"
GATE2_RUNTIME_STATUS = "V4_RECOVERY_RUNTIME_QUALIFIED"
FORMAL_GATE2_STATUS = "V4_GATE2_GPU_SMOKE_PASS"
LOCAL_GATE2_STATUS = "LOCAL_GATE2_DEVELOPMENT_PASS"
FORMAL_GATE2_CLASS = "FORMAL_GATE2"
LOCAL_GATE2_CLASS = "LOCAL_GATE2_DEVELOPMENT_VALIDATION"

GATE1_CRITICAL_PATHS = (
    "georeliab_mve/v4_attempt05_recovery.py",
    "georeliab_mve/v4_recovery_fault_matrix.py",
)
GATE2_CRITICAL_PATHS = (
    "georeliab_mve/adapters.py",
    "georeliab_mve/contracts.py",
    "georeliab_mve/toml_compat.py",
    "georeliab_mve/v4_attempt05_inputs.py",
    "georeliab_mve/v4_attempt05_recovery.py",
    "georeliab_mve/v4_counterfactuals.py",
)
G0_RUFF_PATHS = (
    "georeliab_mve/v4_attempt05_recovery.py",
    "georeliab_mve/v4_qualification.py",
    "georeliab_mve/v4_recovery_fault_matrix.py",
    "georeliab_mve/v4_scoped.py",
    "tests/pilot_round2_harness.py",
    "tests/test_v4_pilot_contracts.py",
    "tests/test_v4_pilot_contracts_round2.py",
    "tests/test_v4_qualification.py",
    "tests/test_v4_recovery_fault_matrix.py",
)

_GIT_OID_RE = re.compile(r"^(?:[0-9a-f]{40}|[0-9a-f]{64})$")
_MANIFEST_RE = re.compile(r"^([0-9a-f]{64})  ([^\r\n]+)$")
_EMPTY_VIOLATION_FIELDS = (
    "budget_violations",
    "closure_violations",
    "duplicate_unit_keys",
    "gpu_violations",
    "inference_count_mismatches",
    "invalid_unit_keys",
    "missing_unit_keys",
    "projection_violations",
    "scientific_marker_violations",
    "scientific_markers",
)


class ReadinessAuditError(ValueError):
    """Raised when qualification evidence is malformed or inconsistent."""


def _fail(reason: str) -> None:
    raise ReadinessAuditError(reason)


def _canonical_bytes(value: object) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
        + "\n"
    ).encode("utf-8")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_json(path: Path) -> Mapping[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ReadinessAuditError(f"INVALID_JSON:{path}:{exc}") from exc
    if not isinstance(value, Mapping):
        _fail(f"JSON_OBJECT_REQUIRED:{path}")
    return value


def _run_git(repo: Path, *args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    completed = subprocess.run(
        ("git", "-C", str(repo), *args),
        check=False,
        capture_output=True,
        text=True,
    )
    if check and completed.returncode != 0:
        _fail(f"GIT_COMMAND_FAILED:{' '.join(args)}:{completed.stderr.strip()}")
    return completed


def _require_bool(value: object, expected: bool, reason: str) -> None:
    if value is not expected:
        _fail(reason)


def _require_empty_sequence(value: object, reason: str) -> None:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        _fail(reason)
    if list(value):
        _fail(reason)


def _require_no_scientific_result(value: Mapping[str, object], reason: str) -> None:
    if value.get("scientific_result") != NO_SCIENTIFIC_RESULT:
        _fail(reason)


def verify_sha256_manifest(root: Path) -> dict[str, object]:
    """Verify every regular file named by an immutable SHA-256 manifest."""

    root = root.resolve()
    manifest_path = root / "MANIFEST.sha256"
    try:
        lines = manifest_path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError) as exc:
        raise ReadinessAuditError(f"MANIFEST_UNREADABLE:{manifest_path}:{exc}") from exc
    if not lines:
        _fail(f"MANIFEST_EMPTY:{manifest_path}")

    seen: set[str] = set()
    internal_symlink_count = 0
    for line in lines:
        match = _MANIFEST_RE.fullmatch(line)
        if match is None:
            _fail(f"MANIFEST_ROW_INVALID:{line}")
        expected, relative_text = match.groups()
        relative = PurePosixPath(relative_text)
        if relative.is_absolute() or ".." in relative.parts:
            _fail(f"MANIFEST_PATH_UNSAFE:{relative_text}")
        if relative_text in seen or relative.name == "MANIFEST.sha256":
            _fail(f"MANIFEST_PATH_INVALID:{relative_text}")
        seen.add(relative_text)
        path = root.joinpath(*relative.parts)
        try:
            resolved = path.resolve(strict=True)
        except OSError as exc:
            raise ReadinessAuditError(
                f"MANIFEST_FILE_MISSING:{relative_text}:{exc}"
            ) from exc
        if root not in resolved.parents or not path.is_file():
            _fail(f"MANIFEST_FILE_UNSAFE:{relative_text}")
        if path.is_symlink():
            internal_symlink_count += 1
        observed = _sha256_file(path)
        if observed != expected:
            _fail(f"MANIFEST_HASH_MISMATCH:{relative_text}")
    return {
        "entry_count": len(lines),
        "internal_symlink_count": internal_symlink_count,
        "manifest_path": str(manifest_path),
        "manifest_sha256": _sha256_file(manifest_path),
    }


def _junit_counts(path: Path) -> dict[str, int | float | str]:
    try:
        root = ET.parse(path).getroot()
    except (OSError, ET.ParseError) as exc:
        raise ReadinessAuditError(f"JUNIT_INVALID:{path}:{exc}") from exc
    suites = [root] if root.tag == "testsuite" else list(root.findall("testsuite"))
    if not suites:
        _fail(f"JUNIT_SUITE_MISSING:{path}")

    def total(name: str) -> int:
        try:
            return sum(int(suite.attrib.get(name, "0")) for suite in suites)
        except ValueError as exc:
            raise ReadinessAuditError(f"JUNIT_COUNT_INVALID:{name}") from exc

    elapsed = 0.0
    try:
        elapsed = sum(float(suite.attrib.get("time", "0")) for suite in suites)
    except ValueError as exc:
        raise ReadinessAuditError("JUNIT_TIME_INVALID") from exc
    return {
        "tests": total("tests"),
        "failures": total("failures"),
        "errors": total("errors"),
        "skipped": total("skipped"),
        "time_seconds": elapsed,
        "path": str(path.resolve()),
        "sha256": _sha256_file(path),
    }


def audit_g0(
    *,
    repo: Path,
    expected_commit: str,
    junit_path: Path,
    expected_test_count: int,
    ruff_paths: Sequence[str] = G0_RUFF_PATHS,
) -> dict[str, object]:
    """Qualify the exact clean source commit, Ruff, and full CPU JUnit."""

    if _GIT_OID_RE.fullmatch(expected_commit) is None:
        _fail("G0_EXPECTED_COMMIT_INVALID")
    repo = repo.resolve()
    head = _run_git(repo, "rev-parse", "HEAD").stdout.strip()
    if head != expected_commit:
        _fail("G0_SOURCE_COMMIT_MISMATCH")
    status = _run_git(repo, "status", "--porcelain", "--untracked-files=all").stdout
    if status:
        _fail("G0_WORKTREE_NOT_CLEAN")

    ruff = subprocess.run(
        ("ruff", "check", *ruff_paths),
        cwd=repo,
        check=False,
        capture_output=True,
        text=True,
    )
    if ruff.returncode != 0:
        _fail(f"G0_RUFF_FAILED:{ruff.stdout.strip()}:{ruff.stderr.strip()}")

    junit = _junit_counts(junit_path)
    if junit["tests"] != expected_test_count:
        _fail("G0_CPU_TEST_COUNT_MISMATCH")
    if junit["failures"] != 0 or junit["errors"] != 0:
        _fail("G0_CPU_SUITE_FAILED")
    return {
        "status": G0_PASS,
        "source_commit": head,
        "source_tree": _run_git(repo, "rev-parse", "HEAD^{tree}").stdout.strip(),
        "worktree_clean": True,
        "ruff": {
            "returncode": ruff.returncode,
            "scope": list(ruff_paths),
            "stdout": ruff.stdout.strip(),
            "stderr": ruff.stderr.strip(),
        },
        "cpu_suite": junit,
        "scientific_result": NO_SCIENTIFIC_RESULT,
    }


def _require_ancestor_and_unchanged(
    *,
    repo: Path,
    evidence_commit: str,
    target_commit: str,
    critical_paths: Sequence[str],
    label: str,
) -> dict[str, object]:
    if _GIT_OID_RE.fullmatch(evidence_commit) is None:
        _fail(f"{label}_SOURCE_COMMIT_INVALID")
    ancestor = _run_git(
        repo,
        "merge-base",
        "--is-ancestor",
        evidence_commit,
        target_commit,
        check=False,
    )
    if ancestor.returncode != 0:
        _fail(f"{label}_SOURCE_NOT_ANCESTOR")
    changed = _run_git(
        repo,
        "diff",
        "--name-only",
        evidence_commit,
        target_commit,
        "--",
        *critical_paths,
    ).stdout.splitlines()
    if changed:
        _fail(f"{label}_QUALIFIED_SOURCE_DRIFT:{','.join(changed)}")
    return {
        "evidence_commit": evidence_commit,
        "target_commit": target_commit,
        "ancestor": True,
        "critical_paths": list(critical_paths),
        "critical_paths_zero_drift": True,
    }


def audit_g1(*, repo: Path, root: Path, target_commit: str) -> dict[str, object]:
    """Verify Gate 1 terminal evidence and its scoped lineage bridge."""

    root = root.resolve()
    manifest = verify_sha256_manifest(root)
    qualification = _read_json(root / "qualification.json")
    source = _read_json(root / "source-manifest.json")
    semantic = _read_json(root / "semantic-result.json")
    _require_no_scientific_result(qualification, "G1_SCIENTIFIC_RESULT_FORBIDDEN")
    _require_no_scientific_result(source, "G1_SOURCE_SCIENTIFIC_RESULT_FORBIDDEN")
    _require_no_scientific_result(semantic, "G1_SEMANTIC_SCIENTIFIC_RESULT_FORBIDDEN")
    if qualification.get("status") != GATE1_STATUS:
        _fail("G1_TERMINAL_STATUS_NOT_PASS")
    if qualification.get("runtime_status") != GATE1_RUNTIME_STATUS:
        _fail("G1_RUNTIME_NOT_READY")
    if qualification.get("case_count") != 147:
        _fail("G1_CASE_COUNT_INVALID")
    if qualification.get("passed_case_count") != 147:
        _fail("G1_CASES_NOT_ALL_PASS")
    if qualification.get("unknown_count") != 0:
        _fail("G1_UNKNOWN_CLASSIFICATION_PRESENT")
    if qualification.get("broad_legacy_reason_count") != 0:
        _fail("G1_BROAD_LEGACY_REASON_PRESENT")
    _require_bool(
        qualification.get("all_failure_injections_classified"),
        True,
        "G1_UNCLASSIFIED_INJECTION",
    )
    for field in ("gate2_started", "pilot_started", "attempt06_started"):
        _require_bool(qualification.get(field), False, f"G1_DOWNSTREAM_STARTED:{field}")
    tooling = qualification.get("tooling")
    if not isinstance(tooling, Mapping) or tooling.get("passed") is not True:
        _fail("G1_TOOLING_NOT_PASS")
    if semantic.get("semantic_result_sha256") != qualification.get(
        "semantic_result_sha256"
    ):
        _fail("G1_SEMANTIC_DIGEST_MISMATCH")
    payload = semantic.get("semantic_payload")
    if not isinstance(payload, Mapping):
        _fail("G1_SEMANTIC_PAYLOAD_INVALID")
    if payload.get("unknown_count") != 0 or payload.get(
        "all_failure_injections_classified"
    ) is not True:
        _fail("G1_SEMANTIC_PAYLOAD_NOT_PASS")
    evidence_commit = source.get("current_commit")
    if not isinstance(evidence_commit, str):
        _fail("G1_SOURCE_COMMIT_MISSING")
    lineage = _require_ancestor_and_unchanged(
        repo=repo.resolve(),
        evidence_commit=evidence_commit,
        target_commit=target_commit,
        critical_paths=GATE1_CRITICAL_PATHS,
        label="G1",
    )
    return {
        "status": G1_PASS,
        "qualification_path": str((root / "qualification.json").resolve()),
        "case_count": 147,
        "passed_case_count": 147,
        "unknown_count": 0,
        "manifest": manifest,
        "lineage_bridge": lineage,
        "scientific_result": NO_SCIENTIFIC_RESULT,
    }


def _observation_rows(root: Path) -> list[Mapping[str, object]]:
    observations = root / "observations"
    paths = sorted(observations.glob("*.json"))
    if len(paths) != 12:
        _fail("G2_OBSERVATION_COUNT_INVALID")
    return [_read_json(path) for path in paths]


def _audit_local_input_binding(gate2_root: Path, run_root: Path) -> dict[str, object]:
    input_manifest = _read_json(run_root / "input-manifest.json")
    closure_text = input_manifest.get("input_closure_path")
    overlay_text = input_manifest.get("overlay_config")
    if not isinstance(closure_text, str) or not isinstance(overlay_text, str):
        _fail("G2_INPUT_BINDING_PATH_MISSING")
    closure_path = Path(closure_text).resolve()
    overlay_path = Path(overlay_text).resolve()
    if gate2_root.resolve() not in closure_path.parents:
        _fail("G2_INPUT_CLOSURE_OUTSIDE_ROOT")
    if gate2_root.resolve() not in overlay_path.parents:
        _fail("G2_OVERLAY_OUTSIDE_ROOT")
    if _sha256_file(closure_path) != input_manifest.get("input_closure_sha256"):
        _fail("G2_INPUT_CLOSURE_DIGEST_MISMATCH")
    if _sha256_file(overlay_path) != input_manifest.get("overlay_sha256"):
        _fail("G2_OVERLAY_DIGEST_MISMATCH")
    closure = _read_json(closure_path)
    if closure.get("attempt05_predictions_read") is not False:
        _fail("G2_ATTEMPT05_INPUT_REUSE_FORBIDDEN")
    bindings = closure.get("bindings")
    if not isinstance(bindings, Sequence) or isinstance(
        bindings, (str, bytes, bytearray)
    ):
        _fail("G2_INPUT_BINDINGS_INVALID")
    if len(bindings) != 6:
        _fail("G2_INPUT_SCENE_COUNT_INVALID")
    view_count = 0
    for binding in bindings:
        if not isinstance(binding, Mapping):
            _fail("G2_INPUT_BINDING_INVALID")
        views = binding.get("views")
        ordered = binding.get("ordered_view_ids")
        if not isinstance(views, Sequence) or isinstance(views, (str, bytes, bytearray)):
            _fail("G2_INPUT_VIEWS_INVALID")
        if not isinstance(ordered, Sequence) or isinstance(
            ordered, (str, bytes, bytearray)
        ):
            _fail("G2_INPUT_VIEW_ORDER_INVALID")
        if len(views) != 8 or len(ordered) != 8:
            _fail("G2_INPUT_VIEW_COUNT_INVALID")
        observed_order: list[object] = []
        for view in views:
            if not isinstance(view, Mapping):
                _fail("G2_INPUT_VIEW_INVALID")
            path_text = view.get("path")
            digest = view.get("sha256")
            if not isinstance(path_text, str) or not isinstance(digest, str):
                _fail("G2_INPUT_VIEW_IDENTITY_MISSING")
            path = Path(path_text).resolve()
            if gate2_root.resolve() not in path.parents:
                _fail("G2_INPUT_VIEW_OUTSIDE_ROOT")
            if _sha256_file(path) != digest:
                _fail("G2_INPUT_VIEW_DIGEST_MISMATCH")
            observed_order.append(view.get("view_id"))
            view_count += 1
        if observed_order != list(ordered):
            _fail("G2_INPUT_VIEW_ORDER_MISMATCH")
    return {
        "input_manifest_path": str((run_root / "input-manifest.json").resolve()),
        "input_closure_path": str(closure_path),
        "input_closure_sha256": _sha256_file(closure_path),
        "overlay_path": str(overlay_path),
        "overlay_sha256": _sha256_file(overlay_path),
        "scene_count": len(bindings),
        "view_count": view_count,
        "attempt05_predictions_read": False,
    }


def _resolved_path(value: object, *, reason: str) -> Path:
    if not isinstance(value, str) or not value:
        _fail(reason)
    return Path(value).expanduser().resolve()


def _require_exact_digest(
    path: Path, expected: object, *, reason: str
) -> str:
    if not path.is_file() or not isinstance(expected, str):
        _fail(reason)
    observed = _sha256_file(path)
    if observed != expected:
        _fail(reason)
    return observed


def _audit_formal_home_input_binding(
    gate2_root: Path, run_root: Path
) -> tuple[dict[str, object], Mapping[str, object]]:
    """Verify a formal closure whose immutable source bits stay in local root."""

    formal_root = gate2_root.resolve()
    input_manifest_path = (run_root / "input-manifest.json").resolve()
    input_manifest = _read_json(input_manifest_path)
    closure_path = _resolved_path(
        input_manifest.get("input_closure_path"),
        reason="G2_FORMAL_INPUT_CLOSURE_PATH_MISSING",
    )
    expected_closure_path = (
        formal_root / "manifests" / "formal-gate2-input-closure.json"
    ).resolve()
    if closure_path != expected_closure_path:
        _fail("G2_FORMAL_INPUT_CLOSURE_PATH_INVALID")
    closure_sha256 = _require_exact_digest(
        closure_path,
        input_manifest.get("input_closure_sha256"),
        reason="G2_INPUT_CLOSURE_DIGEST_MISMATCH",
    )
    if (
        _resolved_path(
            input_manifest.get("runtime_binding_path"),
            reason="G2_FORMAL_RUNTIME_BINDING_PATH_MISSING",
        )
        != closure_path
        or input_manifest.get("runtime_binding_sha256") != closure_sha256
    ):
        _fail("G2_FORMAL_RUNTIME_BINDING_IDENTITY_MISMATCH")
    if input_manifest.get("attempt05_predictions_read") is not False:
        _fail("G2_ATTEMPT05_INPUT_REUSE_FORBIDDEN")
    if input_manifest.get("prediction_outputs_reused") is not False:
        _fail("G2_PREDICTION_OUTPUT_REUSE_FORBIDDEN")
    _require_no_scientific_result(
        input_manifest, "G2_FORMAL_INPUT_SCIENTIFIC_RESULT_FORBIDDEN"
    )

    closure = _read_json(closure_path)
    try:
        closure = local_gate2.validate_formal_home_closure_payload(
            closure, expected_formal_root=formal_root
        )
    except Exception as exc:
        raise ReadinessAuditError(
            f"G2_FORMAL_CLOSURE_INVALID:{type(exc).__name__}:{exc}"
        ) from exc
    source_root = _resolved_path(
        closure.get("source_root"), reason="G2_FORMAL_SOURCE_ROOT_MISSING"
    )
    if source_root == formal_root:
        _fail("G2_FORMAL_SOURCE_ROOT_IDENTITY_MISMATCH")
    source_path = _resolved_path(
        closure.get("source_closure_path"),
        reason="G2_FORMAL_SOURCE_CLOSURE_PATH_MISSING",
    )
    if source_path != (
        source_root / "manifests" / "local-gate2-input-closure.json"
    ).resolve():
        _fail("G2_FORMAL_SOURCE_ROOT_IDENTITY_MISMATCH")
    source_sha256 = _require_exact_digest(
        source_path,
        closure.get("source_closure_sha256"),
        reason="G2_FORMAL_SOURCE_CLOSURE_DIGEST_MISMATCH",
    )
    try:
        source = local_gate2.validate_local_closure(source_path)
    except Exception as exc:
        raise ReadinessAuditError(
            f"G2_FORMAL_SOURCE_CLOSURE_INVALID:{type(exc).__name__}:{exc}"
        ) from exc
    if _resolved_path(
        source.get("root"), reason="G2_FORMAL_SOURCE_ROOT_IDENTITY_MISMATCH"
    ) != source_root:
        _fail("G2_FORMAL_SOURCE_ROOT_IDENTITY_MISMATCH")

    overlay_path = _resolved_path(
        closure.get("overlay_path"), reason="G2_FORMAL_OVERLAY_PATH_MISSING"
    )
    if (
        overlay_path
        != (source_root / "manifests" / "local-gate2-overlay.toml").resolve()
        or _resolved_path(
            input_manifest.get("overlay_config"),
            reason="G2_FORMAL_OVERLAY_PATH_MISSING",
        )
        != overlay_path
    ):
        _fail("G2_FORMAL_OVERLAY_PATH_INVALID")
    overlay_sha256 = _require_exact_digest(
        overlay_path,
        closure.get("overlay_sha256"),
        reason="G2_FORMAL_OVERLAY_DIGEST_MISMATCH",
    )
    if input_manifest.get("overlay_sha256") != overlay_sha256:
        _fail("G2_FORMAL_OVERLAY_DIGEST_MISMATCH")

    resource_path = _resolved_path(
        closure.get("resource_audit_path"),
        reason="G2_FORMAL_RESOURCE_AUDIT_PATH_MISSING",
    )
    if resource_path != (
        source_root / "manifests" / "local-gate2-resource-audit.json"
    ).resolve():
        _fail("G2_FORMAL_RESOURCE_AUDIT_PATH_INVALID")
    resource_sha256 = _require_exact_digest(
        resource_path,
        closure.get("resource_audit_sha256"),
        reason="G2_FORMAL_RESOURCE_AUDIT_DIGEST_MISMATCH",
    )
    resource = _read_json(resource_path)
    if resource.get("formal_gate2_equivalent") is not False:
        _fail("G2_FORMAL_LOCAL_RESOURCE_PROMOTION_FORBIDDEN")
    try:
        local_gate2._validate_existing_resource_audit(
            source_root=source_root,
            resource_audit=resource,
            overlay_path=overlay_path,
        )
    except Exception as exc:
        raise ReadinessAuditError(
            f"G2_FORMAL_RESOURCE_AUDIT_INVALID:{type(exc).__name__}:{exc}"
        ) from exc

    authorization_path = _resolved_path(
        input_manifest.get("authorization_manifest"),
        reason="G2_FORMAL_AUTHORIZATION_PATH_MISSING",
    )
    if authorization_path != (
        formal_root / "manifests" / "formal-gate2-authorization.json"
    ).resolve():
        _fail("G2_FORMAL_AUTHORIZATION_PATH_INVALID")
    authorization_sha256 = _require_exact_digest(
        authorization_path,
        input_manifest.get("authorization_sha256"),
        reason="G2_FORMAL_AUTHORIZATION_DIGEST_MISMATCH",
    )
    authorization = _read_json(authorization_path)
    try:
        authorization = local_gate2.validate_formal_gate2_authorization(
            authorization,
            expected_closure_sha256=closure_sha256,
            expected_output_root=run_root,
        )
    except Exception as exc:
        raise ReadinessAuditError(
            f"G2_FORMAL_AUTHORIZATION_INVALID:{type(exc).__name__}:{exc}"
        ) from exc
    if (
        authorization.get("production_source_commit")
        != closure.get("production_source_commit")
        or authorization.get("test_only_source_commit")
        != closure.get("test_only_source_commit")
        or authorization.get("schedule_identity_sha256")
        != closure.get("schedule_identity_sha256")
    ):
        _fail("G2_FORMAL_AUTHORIZATION_PROVENANCE_MISMATCH")

    bindings = source.get("bindings")
    if not isinstance(bindings, Sequence) or isinstance(
        bindings, (str, bytes, bytearray)
    ):
        _fail("G2_FORMAL_SOURCE_BINDINGS_INVALID")
    view_count = sum(
        len(binding.get("views", ()))
        for binding in bindings
        if isinstance(binding, Mapping)
    )
    if len(bindings) != 6 or view_count != 48:
        _fail("G2_FORMAL_SOURCE_BINDINGS_INVALID")
    return (
        {
            "topology": "FORMAL_HOME_DUAL_ROOT",
            "input_manifest_path": str(input_manifest_path),
            "formal_root": str(formal_root),
            "formal_closure_path": str(closure_path),
            "formal_closure_sha256": closure_sha256,
            "formal_closure_formal_equivalent": True,
            "source_root": str(source_root),
            "source_closure_path": str(source_path),
            "source_closure_sha256": source_sha256,
            "resource_audit_path": str(resource_path),
            "resource_audit_sha256": resource_sha256,
            "local_resource_formal_equivalent": False,
            "overlay_path": str(overlay_path),
            "overlay_sha256": overlay_sha256,
            "authorization_path": str(authorization_path),
            "authorization_sha256": authorization_sha256,
            "scene_count": len(bindings),
            "view_count": view_count,
            "attempt05_predictions_read": False,
            "prediction_outputs_reused": False,
        },
        resource,
    )


def audit_g2(
    *,
    repo: Path,
    gate2_root: Path,
    run_root: Path,
    target_commit: str,
) -> dict[str, object]:
    """Verify Gate 2 execution integrity without promoting its evidence class."""

    gate2_root = gate2_root.resolve()
    run_root = run_root.resolve()
    if gate2_root not in run_root.parents:
        _fail("G2_RUN_OUTSIDE_EVIDENCE_ROOT")
    manifest = verify_sha256_manifest(run_root)
    qualification = _read_json(run_root / "qualification.json")
    smoke_payload = _read_json(run_root / "smoke-manifest.json")
    source = _read_json(run_root / "source-manifest.json")
    input_manifest = _read_json(run_root / "input-manifest.json")
    closure_text = input_manifest.get("input_closure_path")
    formal_home = (
        isinstance(closure_text, str)
        and Path(closure_text).name == "formal-gate2-input-closure.json"
    )
    if formal_home:
        input_binding, resource = _audit_formal_home_input_binding(
            gate2_root, run_root
        )
    else:
        resource = _read_json(
            gate2_root / "manifests" / "local-gate2-resource-audit.json"
        )
        input_binding = _audit_local_input_binding(gate2_root, run_root)
    for value, reason in (
        (qualification, "G2_SCIENTIFIC_RESULT_FORBIDDEN"),
        (smoke_payload, "G2_SMOKE_SCIENTIFIC_RESULT_FORBIDDEN"),
        (source, "G2_SOURCE_SCIENTIFIC_RESULT_FORBIDDEN"),
        (resource, "G2_RESOURCE_SCIENTIFIC_RESULT_FORBIDDEN"),
    ):
        _require_no_scientific_result(value, reason)
    for field in _EMPTY_VIOLATION_FIELDS:
        _require_empty_sequence(
            qualification.get(field), f"G2_VIOLATION_PRESENT:{field}"
        )
    if qualification.get("expected_unit_count") != 12:
        _fail("G2_EXPECTED_UNIT_COUNT_INVALID")
    if qualification.get("unit_count") != 12:
        _fail("G2_UNIT_COUNT_INVALID")
    if qualification.get("recovery_runtime_status") != GATE2_RUNTIME_STATUS:
        _fail("G2_RECOVERY_RUNTIME_NOT_QUALIFIED")
    _require_bool(qualification.get("gate2_started"), True, "G2_NOT_STARTED")
    for field in ("pilot_started", "attempt06_started"):
        _require_bool(qualification.get(field), False, f"G2_DOWNSTREAM_STARTED:{field}")

    try:
        smoke = RecoverySmokeManifest.from_mapping(smoke_payload)
    except Exception as exc:
        raise ReadinessAuditError(f"G2_SMOKE_MANIFEST_INVALID:{exc}") from exc
    rows = _observation_rows(run_root)
    evaluated = evaluate_recovery_smoke(smoke, observations=rows)
    if evaluated.get("status") != GATE2_RUNTIME_STATUS:
        _fail("G2_OBSERVATIONS_NOT_QUALIFIED")
    for row in rows:
        if row.get("projection_count") != 1:
            _fail("G2_PROJECTION_COUNT_INVALID")
        if row.get("completion_count") != 1:
            _fail("G2_COMPLETION_COUNT_INVALID")
        if row.get("overwrite_count") != 0:
            _fail("G2_OVERWRITE_PRESENT")

    evidence_commit = source.get("canonical_commit")
    if not isinstance(evidence_commit, str):
        _fail("G2_SOURCE_COMMIT_MISSING")
    lineage = _require_ancestor_and_unchanged(
        repo=repo.resolve(),
        evidence_commit=evidence_commit,
        target_commit=target_commit,
        critical_paths=GATE2_CRITICAL_PATHS,
        label="G2",
    )
    status = qualification.get("status")
    validation_class = qualification.get("validation_class")
    equivalent = qualification.get("formal_gate2_equivalent")
    resource_equivalent = resource.get("formal_gate2_equivalent")
    if status == FORMAL_GATE2_STATUS:
        if validation_class != FORMAL_GATE2_CLASS:
            _fail("G2_FORMAL_CLASS_MISMATCH")
        if equivalent is not True:
            _fail("G2_FORMAL_EQUIVALENCE_MISMATCH")
        if formal_home:
            if (
                input_binding.get("formal_closure_formal_equivalent") is not True
                or resource_equivalent is not False
            ):
                _fail("G2_FORMAL_EQUIVALENCE_MISMATCH")
        elif resource_equivalent is not True:
            _fail("G2_FORMAL_EQUIVALENCE_MISMATCH")
        gate_status = G2_FORMAL_PASS
        formal = True
    elif status == LOCAL_GATE2_STATUS:
        if validation_class != LOCAL_GATE2_CLASS:
            _fail("G2_LOCAL_CLASS_MISMATCH")
        if equivalent is not False or resource_equivalent is not False:
            _fail("G2_LOCAL_EQUIVALENCE_MISMATCH")
        gate_status = G2_LOCAL_ONLY
        formal = False
    else:
        _fail("G2_TERMINAL_STATUS_NOT_ACCEPTED")
    return {
        "status": gate_status,
        "terminal_status": status,
        "validation_class": validation_class,
        "formal_gate2_equivalent": formal,
        "unit_count": 12,
        "controlled_interruption_count": len(smoke.interruption_plan),
        "gpu_uuid": smoke.gpu_uuid,
        "physical_gpu_index": smoke.physical_gpu_index,
        "manifest": manifest,
        "lineage_bridge": lineage,
        "input_binding": input_binding,
        "scientific_result": NO_SCIENTIFIC_RESULT,
    }


def build_readiness_report(
    *,
    g0: Mapping[str, object],
    g1: Mapping[str, object],
    g2: Mapping[str, object],
) -> dict[str, object]:
    """Build a non-authorizing transition report with no auto-progression."""

    blockers: list[str] = []
    if g0.get("status") != G0_PASS:
        blockers.append("G0_NOT_QUALIFIED")
    if g1.get("status") != G1_PASS:
        blockers.append("G1_NOT_QUALIFIED")
    if g2.get("status") != G2_FORMAL_PASS:
        blockers.extend(
            (
                "G2_EVIDENCE_IS_LOCAL_DEVELOPMENT_ONLY",
                "FORMAL_GATE2_QUALIFICATION_MISSING",
            )
        )
    can_request_authorization = not blockers
    return {
        "schema_version": SCHEMA_VERSION,
        "status": ADMISSION_READY if can_request_authorization else ADMISSION_BLOCKED,
        "validation_class": VALIDATION_CLASS,
        "scientific_result": NO_SCIENTIFIC_RESULT,
        "gates": {"g0": dict(g0), "g1": dict(g1), "g2": dict(g2)},
        "blockers": blockers,
        "can_request_pilot_execution_authorization": can_request_authorization,
        "pilot_execution_authorized": False,
        "pilot_partition_frozen": False,
        "pilot_started": False,
        "confirmation_started": False,
        "automatic_progression_allowed": False,
        "next_action": (
            "OBTAIN_EXPLICIT_PILOT_GPU_BUDGET_AUTHORIZATION_THEN_FREEZE_PARTITION"
            if can_request_authorization
            else "PRODUCE_FRESH_FORMAL_GATE2_QUALIFICATION"
        ),
    }


def audit(
    *,
    repo: Path,
    expected_commit: str,
    junit_path: Path,
    expected_test_count: int,
    gate1_root: Path,
    gate2_root: Path,
    gate2_run_root: Path,
) -> dict[str, object]:
    g0 = audit_g0(
        repo=repo,
        expected_commit=expected_commit,
        junit_path=junit_path,
        expected_test_count=expected_test_count,
    )
    g1 = audit_g1(repo=repo, root=gate1_root, target_commit=expected_commit)
    g2 = audit_g2(
        repo=repo,
        gate2_root=gate2_root,
        run_root=gate2_run_root,
        target_commit=expected_commit,
    )
    return build_readiness_report(g0=g0, g1=g1, g2=g2)


def _write_json_no_clobber(path: Path, value: object) -> None:
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("xb") as handle:
            handle.write(_canonical_bytes(value))
    except FileExistsError as exc:
        raise ReadinessAuditError(f"AUDIT_OUTPUT_EXISTS:{path}") from exc


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--junit", type=Path, required=True)
    parser.add_argument("--expected-test-count", type=int, required=True)
    parser.add_argument("--gate1-root", type=Path, required=True)
    parser.add_argument("--gate2-root", type=Path, required=True)
    parser.add_argument("--gate2-run-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--require-ready",
        action="store_true",
        help="return 3 when the valid audit conclusion is admission blocked",
    )
    return parser


def audit_exit_code(status: object, *, require_ready: bool) -> int:
    if status not in {ADMISSION_READY, ADMISSION_BLOCKED}:
        return 2
    if require_ready and status == ADMISSION_BLOCKED:
        return 3
    return 0


def auditor_provenance() -> dict[str, object]:
    """Bind a generated report to a clean, tracked test-only auditor commit."""

    source = Path(__file__).resolve()
    tracked = _run_git(
        PROJECT_ROOT,
        "ls-files",
        "--error-unmatch",
        source.relative_to(PROJECT_ROOT).as_posix(),
        check=False,
    )
    if tracked.returncode != 0:
        _fail("AUDITOR_SOURCE_NOT_TRACKED")
    worktree_status = _run_git(
        PROJECT_ROOT,
        "status",
        "--porcelain",
        "--untracked-files=all",
    ).stdout
    if worktree_status:
        _fail("AUDITOR_WORKTREE_NOT_CLEAN")
    return {
        "source_path": source.relative_to(PROJECT_ROOT).as_posix(),
        "source_sha256": _sha256_file(source),
        "source_commit": _run_git(PROJECT_ROOT, "rev-parse", "HEAD").stdout.strip(),
        "source_tree": _run_git(
            PROJECT_ROOT, "rev-parse", "HEAD^{tree}"
        ).stdout.strip(),
        "source_tracked": True,
        "worktree_clean": True,
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        report = audit(
            repo=args.repo,
            expected_commit=args.expected_commit,
            junit_path=args.junit,
            expected_test_count=args.expected_test_count,
            gate1_root=args.gate1_root,
            gate2_root=args.gate2_root,
            gate2_run_root=args.gate2_run_root,
        )
        report["auditor"] = auditor_provenance()
        _write_json_no_clobber(args.output, report)
    except ReadinessAuditError as exc:
        print(f"V4_GATE2_PILOT_READINESS_AUDIT_INVALID reason={exc}")
        return 2
    print(
        f"{report['status']} output={args.output.resolve()} "
        f"blockers={','.join(report['blockers']) or 'none'}"
    )
    return audit_exit_code(report["status"], require_ready=args.require_ready)


if __name__ == "__main__":
    raise SystemExit(main())
