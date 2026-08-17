"""Test-only Pilot Round-2 audit harness.

This module is intentionally outside ``georeliab_mve``.  It may validate
frozen manifests and synthetic fixtures, but it must never dispatch a GPU or
publish scientific evidence.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from itertools import combinations
import json
from pathlib import Path
from pathlib import PurePosixPath
import re
import subprocess
from typing import Mapping, Sequence


DEVELOPMENT_EVIDENCE_ONLY = "DEVELOPMENT_EVIDENCE_ONLY"
NO_SCIENTIFIC_RESULT = "NO_SCIENTIFIC_RESULT"
PILOT_MODELS = ("VGGT", "MASt3R")

REQUIRED_EVIDENCE_FILES = (
    "manifests/pilot-input-closure.json",
    "manifests/pilot-schedule-identity.json",
    "manifests/pilot-partition-manifest.json",
    "manifests/pilot-scope-addendum.json",
    "manifests/pilot-resource-audit.json",
    "manifests/pilot-source-manifest.json",
    "manifests/pilot-model-adapter-manifest.json",
    "manifests/pilot-calibration-threshold-manifest.json",
    "manifests/pilot-authorization.json",
    "manifests/pilot-budget-contract.json",
    "logs/pilot-gpu-preflight.log",
    "logs/pilot-cpu-contract-tests.log",
    "logs/pilot-dry-run-audit.log",
    "logs/pilot-console.log",
    "logs/pilot-event-log.jsonl",
    "decisions/pilot-coverage-audit.json",
    "decisions/pilot-transaction-closure-audit.json",
    "decisions/pilot-receipt-audit.json",
    "decisions/pilot-resource-accounting-audit.json",
    "decisions/pilot-metric-evidence.json",
    "decisions/pilot-decision.json",
    "decisions/pilot-manifest-verification.log",
)


class PilotRound2ContractError(ValueError):
    """Raised when test-only Pilot evidence fails closed."""


@dataclass(frozen=True, slots=True)
class LoggedCommandStatus:
    raw_exit_code: int
    tee_exit_code: int
    passed: bool
    console_log: Path
    status_path: Path


_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_MANIFEST_LINE_RE = re.compile(r"^([0-9a-f]{64})  ([^\r\n]+)$")
_FORBIDDEN_PROVENANCE_MARKERS = (
    "attempt05",
    "localgate2",
    "gate2smoke",
    "recoverysmoke",
)
_COMMON_EVIDENCE_KEYS = frozenset(
    {
        "schema_version",
        "protocol_id",
        "protocol_sha256",
        "schedule_identity_sha256",
        "created_at",
        "source_sha256",
        "validation_class",
        "scientific_result",
    }
)
_EXECUTION_KEYS = frozenset(
    {
        "schema_version",
        "validation_class",
        "scientific_result",
        "unit_ids",
        "gpu_uuid",
        "physical_gpu_count",
        "model_order",
        "sequential_model_execution",
        "sequential_unit_execution",
        "fallback_allowed",
        "auto_retry_allowed",
        "device_switch_allowed",
        "grid_reduction_allowed",
        "downstream_advance_allowed",
        "max_gpu_seconds",
        "max_wall_seconds",
        "max_storage_bytes",
        "output_root",
    }
)


def _fail(reason: str) -> None:
    raise PilotRound2ContractError(reason)


def _normalized_marker(value: object) -> str:
    return "".join(character for character in str(value).casefold() if character.isalnum())


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def scan_forbidden_provenance(value: object) -> None:
    """Reject historical prediction, receipt, ledger, root, or run references."""

    def walk(item: object, trail: str) -> None:
        if isinstance(item, Mapping):
            for key, child in item.items():
                walk(child, f"{trail}.{key}")
            return
        if isinstance(item, Sequence) and not isinstance(
            item, (str, bytes, bytearray)
        ):
            for index, child in enumerate(item):
                walk(child, f"{trail}[{index}]")
            return
        if isinstance(item, (str, Path)):
            normalized = _normalized_marker(item)
            if any(marker in normalized for marker in _FORBIDDEN_PROVENANCE_MARKERS):
                _fail(f"forbidden Pilot provenance at {trail}")

    walk(value, "evidence")


def validate_pilot_records(
    records: Sequence[Mapping[str, object]],
    *,
    expected_unit_ids: Sequence[str],
    expected_views_by_scene: Mapping[int, Sequence[int]],
) -> None:
    expected = tuple(expected_unit_ids)
    if len(expected) != 60 or len(set(expected)) != 60:
        _fail("Pilot expected unit inventory must contain exactly 60 identities")
    if len(records) != len(expected):
        _fail("Pilot record inventory must contain exactly 60 records")

    actual_ids: list[str] = []
    for ordinal, record in enumerate(records):
        if not isinstance(record, Mapping):
            _fail(f"Pilot record {ordinal} must be a mapping")
        try:
            unit_id = str(record["unit_id"])
            model_id = str(record["model_id"])
            scene_id = int(record["scene_id"])
            state_id = str(record["state_id"])
            ordered_views = tuple(record["ordered_view_ids"])
            pose_pairs = tuple(tuple(pair) for pair in record["pose_pairs"])
        except (KeyError, TypeError, ValueError) as exc:
            raise PilotRound2ContractError(
                f"Pilot record {ordinal} is missing an identity, view, or pair field"
            ) from exc

        actual_ids.append(unit_id)
        if model_id not in PILOT_MODELS:
            _fail(f"Pilot record {ordinal} has an unsupported model identity")
        if unit_id != f"{model_id}:{scene_id}:{state_id}":
            _fail(f"Pilot record {ordinal} unit identity does not match its fields")
        if scene_id not in expected_views_by_scene:
            _fail(f"Pilot record {ordinal} scene lies outside the frozen view inventory")
        expected_views = tuple(expected_views_by_scene[scene_id])
        if (
            len(expected_views) != 8
            or len(set(expected_views)) != 8
            or ordered_views != expected_views
        ):
            _fail(f"Pilot record {ordinal} ordered view identity mismatch")
        expected_pairs = tuple(combinations(expected_views, 2))
        if len(pose_pairs) != 28 or pose_pairs != expected_pairs:
            _fail(f"Pilot record {ordinal} unordered pose pair inventory mismatch")
        scan_forbidden_provenance(
            {
                key: record[key]
                for key in (
                    "prediction_provenance",
                    "receipt_provenance",
                    "ledger_provenance",
                )
                if key in record
            }
        )

    if tuple(actual_ids) != expected:
        _fail("Pilot record unit order or identity inventory mismatch")


def validate_pilot_execution_contract(
    payload: Mapping[str, object],
    *,
    expected_unit_ids: Sequence[str],
) -> None:
    if not isinstance(payload, Mapping) or set(payload) != _EXECUTION_KEYS:
        _fail("Pilot execution contract schema mismatch")
    if payload["schema_version"] != (
        "georeliab-v4-development-pilot-execution-contract-1.0"
    ):
        _fail("Pilot execution contract version mismatch")
    if payload["validation_class"] != DEVELOPMENT_EVIDENCE_ONLY:
        _fail("Pilot execution contract must remain development evidence")
    if payload["scientific_result"] != NO_SCIENTIFIC_RESULT:
        _fail("Pilot execution contract cannot claim a scientific result")

    expected = tuple(expected_unit_ids)
    try:
        actual = tuple(payload["unit_ids"])
    except TypeError as exc:
        raise PilotRound2ContractError("Pilot unit inventory must be a sequence") from exc
    if len(expected) != 60 or len(set(expected)) != 60 or actual != expected:
        _fail("Pilot execution contract must bind the exact ordered 60-unit inventory")
    if (
        not isinstance(payload["gpu_uuid"], str)
        or not str(payload["gpu_uuid"]).startswith("GPU-")
        or payload["physical_gpu_count"] != 1
        or type(payload["physical_gpu_count"]) is not int
    ):
        _fail("Pilot execution contract requires one explicit physical GPU UUID")
    if tuple(payload["model_order"]) != PILOT_MODELS:
        _fail("Pilot model order must be VGGT then MASt3R")
    if payload["sequential_model_execution"] is not True:
        _fail("Pilot models must execute sequentially")
    if payload["sequential_unit_execution"] is not True:
        _fail("Pilot units must execute sequentially")
    for field_name in (
        "fallback_allowed",
        "auto_retry_allowed",
        "device_switch_allowed",
        "grid_reduction_allowed",
        "downstream_advance_allowed",
    ):
        if payload[field_name] is not False:
            _fail(f"Pilot execution forbids {field_name}")
    for field_name in (
        "max_gpu_seconds",
        "max_wall_seconds",
        "max_storage_bytes",
    ):
        value = payload[field_name]
        if type(value) is not int or value <= 0:
            _fail(f"Pilot {field_name} must be a positive integer")
    output_root = Path(str(payload["output_root"]))
    if not output_root.is_absolute() or not output_root.is_relative_to(
        Path("/home/hryang")
    ):
        _fail("Pilot output root must be an absolute path below /home/hryang")


def _validate_common_evidence(payload: object, relative: str) -> None:
    if not isinstance(payload, Mapping) or not _COMMON_EVIDENCE_KEYS.issubset(payload):
        _fail(f"Pilot evidence metadata is incomplete: {relative}")
    if not isinstance(payload["schema_version"], str) or not payload[
        "schema_version"
    ]:
        _fail(f"Pilot evidence schema version is invalid: {relative}")
    for field_name in (
        "protocol_sha256",
        "schedule_identity_sha256",
        "source_sha256",
    ):
        if not isinstance(payload[field_name], str) or not _SHA256_RE.fullmatch(
            payload[field_name]
        ):
            _fail(f"Pilot evidence {field_name} is invalid: {relative}")
    if not isinstance(payload["protocol_id"], str) or not payload["protocol_id"]:
        _fail(f"Pilot evidence protocol_id is invalid: {relative}")
    if not isinstance(payload["created_at"], str) or not payload[
        "created_at"
    ].endswith("Z"):
        _fail(f"Pilot evidence timestamp is invalid: {relative}")
    if payload["validation_class"] != DEVELOPMENT_EVIDENCE_ONLY:
        _fail(f"Pilot evidence validation class is not development-only: {relative}")
    if payload["scientific_result"] != NO_SCIENTIFIC_RESULT:
        _fail(f"Pilot evidence contains a forbidden scientific result: {relative}")


def _manifest_entries(path: Path) -> dict[str, str]:
    try:
        lines = path.read_text(encoding="ascii").splitlines()
    except (OSError, UnicodeError) as exc:
        raise PilotRound2ContractError("Pilot MANIFEST.sha256 is unreadable") from exc
    if not lines:
        _fail("Pilot MANIFEST.sha256 is empty")
    result: dict[str, str] = {}
    for line in lines:
        match = _MANIFEST_LINE_RE.fullmatch(line)
        if match is None:
            _fail("Pilot MANIFEST.sha256 line is malformed")
        digest, relative = match.groups()
        candidate = PurePosixPath(relative)
        if candidate.is_absolute() or ".." in candidate.parts or relative in result:
            _fail("Pilot MANIFEST.sha256 contains an unsafe or duplicate path")
        result[relative] = digest
    return result


def verify_pilot_evidence_bundle(root: Path) -> None:
    root = Path(root)
    if not root.is_dir() or root.is_symlink():
        _fail("Pilot evidence root must be a real directory")
    manifest_path = root / "MANIFEST.sha256"
    if not manifest_path.is_file() or manifest_path.is_symlink():
        _fail("Pilot MANIFEST.sha256 is missing or unsafe")

    all_paths = tuple(root.rglob("*"))
    if any(path.is_symlink() for path in all_paths):
        _fail("Pilot evidence bundle contains a symlink")
    actual_files = {
        path.relative_to(root).as_posix()
        for path in all_paths
        if path.is_file() and path != manifest_path
    }
    required = set(REQUIRED_EVIDENCE_FILES)
    if not required.issubset(actual_files):
        _fail("Pilot evidence bundle is missing a required file")

    manifest = _manifest_entries(manifest_path)
    if set(manifest) != actual_files:
        _fail("Pilot MANIFEST.sha256 does not cover the exact directory")
    for relative, expected_sha256 in manifest.items():
        if _sha256_file(root / relative) != expected_sha256:
            _fail(f"Pilot evidence digest mismatch: {relative}")

    for relative in REQUIRED_EVIDENCE_FILES:
        path = root / relative
        if path.suffix == ".json":
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, UnicodeError, json.JSONDecodeError) as exc:
                raise PilotRound2ContractError(
                    f"Pilot JSON evidence is invalid: {relative}"
                ) from exc
            _validate_common_evidence(payload, relative)
        elif path.suffix == ".jsonl":
            try:
                lines = path.read_text(encoding="utf-8").splitlines()
                payloads = [json.loads(line) for line in lines]
            except (OSError, UnicodeError, json.JSONDecodeError) as exc:
                raise PilotRound2ContractError(
                    f"Pilot JSONL evidence is invalid: {relative}"
                ) from exc
            if not payloads:
                _fail(f"Pilot JSONL evidence is empty: {relative}")
            for payload in payloads:
                _validate_common_evidence(payload, relative)
        elif path.stat().st_size <= 0:
            _fail(f"Pilot log evidence is empty: {relative}")


def run_logged_command(
    command: Sequence[str],
    *,
    console_log: Path,
    status_path: Path,
) -> LoggedCommandStatus:
    if (
        not command
        or any(not isinstance(part, str) or not part for part in command)
    ):
        _fail("logged command must be a non-empty argv sequence")
    console_log = Path(console_log)
    status_path = Path(status_path)
    if console_log == status_path or console_log.exists() or status_path.exists():
        _fail("logged command outputs must be fresh and distinct")
    console_log.parent.mkdir(parents=True, exist_ok=True)
    status_path.parent.mkdir(parents=True, exist_ok=True)

    with console_log.open("x", encoding="utf-8", newline="") as handle:
        process = subprocess.Popen(
            list(command),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        assert process.stdout is not None
        for line in process.stdout:
            handle.write(line)
        raw_exit_code = process.wait()
        handle.flush()
    tee_exit_code = 0
    passed = raw_exit_code == 0 and tee_exit_code == 0
    payload = {
        "schema_version": "georeliab-v4-test-only-logged-command-status-1.0",
        "raw_exit_code": raw_exit_code,
        "tee_exit_code": tee_exit_code,
        "passed": passed,
        "console_log": str(console_log),
        "console_sha256": _sha256_file(console_log),
    }
    status_path.write_text(
        json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    return LoggedCommandStatus(
        raw_exit_code,
        tee_exit_code,
        passed,
        console_log,
        status_path,
    )


def validate_logged_status(path: Path) -> LoggedCommandStatus:
    status_path = Path(path)
    try:
        payload = json.loads(status_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise PilotRound2ContractError("logged exit status is unreadable") from exc
    expected_keys = {
        "schema_version",
        "raw_exit_code",
        "tee_exit_code",
        "passed",
        "console_log",
        "console_sha256",
    }
    if not isinstance(payload, Mapping) or set(payload) != expected_keys:
        _fail("logged exit status schema mismatch")
    if payload["schema_version"] != (
        "georeliab-v4-test-only-logged-command-status-1.0"
    ):
        _fail("logged exit status version mismatch")
    raw_exit_code = payload["raw_exit_code"]
    tee_exit_code = payload["tee_exit_code"]
    passed = payload["passed"]
    if (
        type(raw_exit_code) is not int
        or type(tee_exit_code) is not int
        or type(passed) is not bool
    ):
        _fail("logged exit status has invalid value types")
    if passed != (raw_exit_code == 0 and tee_exit_code == 0):
        _fail("logged exit status cannot turn a nonzero exit into pass")
    console_log = Path(str(payload["console_log"]))
    if (
        not console_log.is_file()
        or not isinstance(payload["console_sha256"], str)
        or _sha256_file(console_log) != payload["console_sha256"]
    ):
        _fail("logged console digest mismatch")
    return LoggedCommandStatus(
        raw_exit_code,
        tee_exit_code,
        passed,
        console_log,
        status_path,
    )
