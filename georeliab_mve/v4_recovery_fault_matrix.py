"""CPU-only Gate 1 recovery fault-matrix qualification.

This module exercises the recovery transaction surface in isolated temporary
roots.  It never reads Attempt-05 prediction payloads and never emits
scientific metrics.  The output is an append-only, no-clobber qualification
bundle whose terminal state is intentionally separate from Gate 2.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from dataclasses import dataclass, field
import errno
import hashlib
import json
import os
from pathlib import Path
import platform
import signal
import subprocess
import sys
import tempfile
import time
from typing import Any, Callable, Iterator, Mapping, Sequence
from unittest.mock import patch

from . import v4_attempt05_recovery as recovery


SCHEMA_VERSION = "georeliab-v4-cpu-fault-matrix-1.0"
PASS_MARKER = "V4_RECOVERY_CPU_FAULT_MATRIX_PASS"
READY_MARKER = "V4_RECOVERY_RUNTIME_READY"
FAIL_MARKER = "V4_RECOVERY_CPU_FAULT_MATRIX_FAILED"
NO_SCIENTIFIC_RESULT = "NO_SCIENTIFIC_RESULT"
CLASSIFICATIONS = {
    "COMPLETE",
    "SAFE_RETRY",
    "QUARANTINED",
    "FATAL_IDENTITY_MISMATCH",
}
GATE1_ALLOWLIST = {
    "georeliab_mve/v4_recovery_fault_matrix.py",
    "tests/test_v4_recovery_fault_matrix.py",
}
EXPECTED_PARENT_TREE = "ba864e0ed53b82bdf10aec06ddceda950bf6e821"
ERRNO_NAMES = ("EACCES", "ENOSPC", "EIO")


class FaultMatrixError(recovery.Attempt05RecoveryError):
    """A deterministic qualification harness failure."""


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _sha_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _domain_sha(domain: str, value: object) -> str:
    return _sha_bytes(domain.encode("ascii") + b"\0" + _canonical_bytes(value))


def _sha_file(path: Path) -> str:
    return _sha_bytes(path.read_bytes())


def _normalise_state(value: Mapping[str, object]) -> tuple[str, str]:
    state = value.get("state")
    action = value.get("recovery_action")
    if state == recovery.COMPLETE:
        return "COMPLETE", recovery.RECOVERY_ACTION_NOOP
    if state == recovery.INCOMPLETE_SAFE_TO_RETRY:
        return "SAFE_RETRY", str(action)
    if state == recovery.ORPHAN_REQUIRES_QUARANTINE:
        return "QUARANTINED", recovery.RECOVERY_ACTION_QUARANTINE
    if state == recovery.FATAL_IDENTITY_MISMATCH:
        return "FATAL_IDENTITY_MISMATCH", recovery.RECOVERY_ACTION_ABORT_FATAL
    if state == recovery.ACTIVE_VALID_DEFER_NO_MUTATION:
        raise FaultMatrixError("V4_RECOVERY_ACTIVE_STATE_NOT_TERMINAL")
    raise FaultMatrixError(f"V4_RECOVERY_UNKNOWN_STATE:{state!r}")


def _expected_action(classification: str, preferred: str | None = None) -> str:
    if classification == "COMPLETE":
        return recovery.RECOVERY_ACTION_NOOP
    if classification == "SAFE_RETRY":
        return preferred or recovery.RECOVERY_ACTION_REINFER_UNIT
    if classification == "QUARANTINED":
        return recovery.RECOVERY_ACTION_QUARANTINE
    if classification == "FATAL_IDENTITY_MISMATCH":
        return recovery.RECOVERY_ACTION_ABORT_FATAL
    raise FaultMatrixError(f"invalid expected classification {classification}")


@dataclass(frozen=True, slots=True)
class FaultInjectionCase:
    case_id: str
    operation: str
    fault_point: str
    fault_kind: str
    expected_classification: str
    expected_action: str
    notes: str = ""
    schema_version: str = SCHEMA_VERSION

    def __post_init__(self) -> None:
        if not self.case_id or not self.operation or not self.fault_point:
            raise FaultMatrixError("V4_RECOVERY_FAULT_CASE_ID_INVALID")
        if self.expected_classification not in CLASSIFICATIONS:
            raise FaultMatrixError("V4_RECOVERY_FAULT_EXPECTATION_INVALID")
        if self.expected_action not in {
            recovery.RECOVERY_ACTION_NOOP,
            recovery.RECOVERY_ACTION_RESUME_LEDGER_ONLY,
            recovery.RECOVERY_ACTION_REINFER_UNIT,
            recovery.RECOVERY_ACTION_QUARANTINE,
            recovery.RECOVERY_ACTION_ABORT_FATAL,
        }:
            raise FaultMatrixError("V4_RECOVERY_FAULT_ACTION_INVALID")

    def to_dict(self) -> dict[str, object]:
        payload = {
            "schema_version": self.schema_version,
            "case_id": self.case_id,
            "operation": self.operation,
            "fault_point": self.fault_point,
            "fault_kind": self.fault_kind,
            "expected_classification": self.expected_classification,
            "expected_action": self.expected_action,
            "notes": self.notes,
        }
        payload["case_sha256"] = _domain_sha(
            "georeliab:v4:gate1-case:v1", payload
        )
        return payload


@dataclass(frozen=True, slots=True)
class FaultInjectionObservation:
    case_id: str
    expected_classification: str
    expected_action: str
    observed_classification: str
    observed_action: str
    reason: str
    before_inventory_sha256: str
    after_inventory_sha256: str
    before_ledger_sha256: str
    after_ledger_sha256: str
    duplicate_count: int
    overwrite_count: int
    scientific_result: str = NO_SCIENTIFIC_RESULT
    schema_version: str = SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.observed_classification not in CLASSIFICATIONS:
            raise FaultMatrixError("V4_RECOVERY_OBSERVED_CLASSIFICATION_INVALID")
        if self.scientific_result != NO_SCIENTIFIC_RESULT:
            raise FaultMatrixError("V4_RECOVERY_SCIENTIFIC_RESULT_FORBIDDEN")
        if self.duplicate_count < 0 or self.overwrite_count < 0:
            raise FaultMatrixError("V4_RECOVERY_COUNTER_INVALID")

    @property
    def passed(self) -> bool:
        return (
            self.expected_classification == self.observed_classification
            and self.observed_action == self.expected_action
            and self.duplicate_count == 0
            and self.overwrite_count == 0
            and self.scientific_result == NO_SCIENTIFIC_RESULT
        )

    def to_dict(self) -> dict[str, object]:
        payload = {
            "schema_version": self.schema_version,
            "case_id": self.case_id,
            "expected_classification": self.expected_classification,
            "observed_classification": self.observed_classification,
            "observed_action": self.observed_action,
            "reason": self.reason,
            "before_inventory_sha256": self.before_inventory_sha256,
            "after_inventory_sha256": self.after_inventory_sha256,
            "before_ledger_sha256": self.before_ledger_sha256,
            "after_ledger_sha256": self.after_ledger_sha256,
            "duplicate_count": self.duplicate_count,
            "overwrite_count": self.overwrite_count,
            "scientific_result": self.scientific_result,
        }
        payload["observation_sha256"] = _domain_sha(
            "georeliab:v4:gate1-observation:v1", payload
        )
        return payload


@dataclass(frozen=True, slots=True)
class CpuFaultMatrixManifest:
    parent_commit: str
    parent_tree: str
    current_commit: str
    current_tree: str
    cases: tuple[FaultInjectionCase, ...]
    observations: tuple[FaultInjectionObservation, ...]
    all_failure_injections_classified: bool
    unknown_count: int
    broad_legacy_reason_count: int
    scientific_result: str = NO_SCIENTIFIC_RESULT
    schema_version: str = SCHEMA_VERSION

    def to_dict(self, *, include_observations: bool = True) -> dict[str, object]:
        payload: dict[str, object] = {
            "schema_version": self.schema_version,
            "parent_commit": self.parent_commit,
            "parent_tree": self.parent_tree,
            "current_commit": self.current_commit,
            "current_tree": self.current_tree,
            "case_count": len(self.cases),
            "cases": [case.to_dict() for case in self.cases],
            "all_failure_injections_classified": self.all_failure_injections_classified,
            "unknown_count": self.unknown_count,
            "broad_legacy_reason_count": self.broad_legacy_reason_count,
            "scientific_result": self.scientific_result,
        }
        if include_observations:
            payload["observations"] = [item.to_dict() for item in self.observations]
        payload["manifest_sha256"] = _domain_sha(
            "georeliab:v4:gate1-manifest:v1", payload
        )
        return payload


class _InjectedAbort(BaseException):
    pass


class _FaultController:
    def __init__(self, target: str, kind: str) -> None:
        self.target = target
        self.kind = kind
        self.hits = 0

    def __call__(self, point: str) -> None:
        if point != self.target:
            return
        self.hits += 1
        if self.kind == "ABRUPT_EXIT":
            raise _InjectedAbort(f"injected abrupt exit at {point}")
        if self.kind in ERRNO_NAMES:
            number = getattr(errno, self.kind)
            raise OSError(number, os.strerror(number), point)
        raise FaultMatrixError(f"V4_RECOVERY_UNSUPPORTED_FAULT_KIND:{self.kind}")


def _scope_for_path(path: Path) -> str:
    name = path.name
    if ".pending" in str(path):
        return "ledger_pending"
    if name == "unit_transaction_receipt.json":
        return "prepared_receipt"
    if name == "unit_transaction_receipt.canonical_promoted.json":
        return "canonical_receipt"
    if name == "unit_transaction_receipt.ledger_committed.json":
        return "committed_receipt"
    if name.startswith("gate1-checkpoint"):
        return "checkpoint"
    if name == "exit-receipt.json":
        return "supervisor_exit"
    return "artifact"


@contextmanager
def _patched_fault_surface(controller: _FaultController) -> Iterator[None]:
    original_atomic = recovery.atomic_write_bytes
    original_rename = recovery.rename_noreplace

    def patched_atomic(
        path: Path,
        payload: bytes,
        *,
        fault_injector: recovery.FailureInjector | None = None,
        replace_existing: bool = False,
    ) -> None:
        scope = _scope_for_path(Path(path))

        def scoped(stage: str) -> None:
            controller(f"{scope}:{stage}")
            if fault_injector is not None:
                fault_injector(stage)

        original_atomic(
            path,
            payload,
            fault_injector=scoped,
            replace_existing=replace_existing,
        )

    with patch.object(recovery, "atomic_write_bytes", patched_atomic):
        with patch.object(
            recovery,
            "rename_noreplace",
            lambda source, destination: (
                controller("promotion:rename"),
                original_rename(source, destination),
            )[1],
        ):
            yield


def _phase_fault(controller: _FaultController, operation: str) -> Callable[[str], None]:
    def callback(stage: str) -> None:
        if stage in {"rename_before", "rename", "rename_after", "dir_fsync"}:
            controller(f"promotion:{stage}")
        elif stage == "ledger_append":
            controller(f"{operation}:ledger_append")
        else:
            controller(stage)

    return callback


def _inventory(root: Path) -> tuple[dict[str, object], ...]:
    rows: list[dict[str, object]] = []
    if not root.exists():
        return ()
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        relative = path.relative_to(root).as_posix()
        rows.append({"path": relative, "size": path.stat().st_size, "sha256": _sha_file(path)})
    return tuple(rows)


def _inventory_sha(root: Path) -> str:
    return _domain_sha("georeliab:v4:gate1-inventory:v1", _inventory(root))


def _ledger_sha(path: Path) -> str:
    return _sha_file(path) if path.exists() else _sha_bytes(b"")


def _unit_payload() -> tuple[dict[str, object], ...]:
    common = {
        "unit_key": "gate1|1|CPU",
        "attempt_id": "gate1-cpu",
        "adapter_sha256": "a" * 64,
        "config_sha256": "b" * 64,
        "split_sha256": "c" * 64,
        "corruption_sha256": "d" * 64,
        "gpu_uuid": "CPU_ONLY",
        "physical_gpu_index": -1,
        "tooling_commit": "gate1",
        "tooling_tree": "gate1",
    }
    return (
        {**common, "event_type": "PROJECTION_EVENT"},
        {**common, "event_type": "COMPLETION_EVENT"},
    )


def _run_unit_case(case: FaultInjectionCase, case_root: Path) -> tuple[str, str, str, int, int]:
    ledger_path = case_root / "ledger.jsonl"
    run_root = case_root / "run"
    canonical = run_root / "units" / "gate1-1-CPU"
    ledger = recovery.JournaledLedger(ledger_path)
    store = recovery.UnitTransactionStore(
        run_root / "units",
        attempt_id="gate1-cpu",
        ledger=ledger,
        schedule_identity_sha256="e" * 64,
    )
    controller = _FaultController(case.fault_point, case.fault_kind)
    before_inventory = _inventory_sha(case_root)
    with _patched_fault_surface(controller):
        try:
            events = _unit_payload()
            if case.operation == "projection":
                events = (events[0],)
            elif case.operation == "completion":
                events = (events[1],)
            receipt = store.prepare_unit(
                idempotency_key="gate1-cpu:gate1|1|CPU",
                unit_key=("gate1", 1, "CPU"),
                stage="cpu_fault_matrix",
                canonical_dir=canonical,
                files={"opaque-prediction.bin": b"opaque-gate1-fixture"},
                ledger_events=events,
            )
            store.commit_unit(receipt, fault_injector=_phase_fault(controller, case.operation))
            if case.fault_point.startswith("checkpoint:"):
                checkpoint = recovery.DurabilityCheckpoint(
                    checkpoint_index=1,
                    last_sequence=0,
                    last_event_sha256="0" * 64,
                    completed_unit_keys=("gate1|1|CPU",),
                    gpu_inference_seconds=0.0,
                    logical_bytes=0,
                    allocated_bytes=0,
                )
                store.write_checkpoint(checkpoint, case_root / "gate1-checkpoint")
        except BaseException:
            pass
    try:
        result = recovery.reconcile_unit_transaction(
            canonical_dir=canonical,
            ledger_path=ledger_path,
            idempotency_key="gate1-cpu:gate1|1|CPU",
            worker_alive=False,
        )
        classification, action = _normalise_state(result)
        reason = str(result.get("reason", ""))
    except BaseException as exc:
        classification = "FATAL_IDENTITY_MISMATCH"
        action = recovery.RECOVERY_ACTION_ABORT_FATAL
        reason = f"RECONCILIATION_EXCEPTION:{type(exc).__name__}"
    after_inventory = _inventory_sha(case_root)
    duplicate_count = sum(
        1 for item in _inventory(case_root) if "duplicate" in str(item["path"])
    )
    return (
        classification,
        action,
        reason,
        duplicate_count,
        0,
    )


def _run_pending_cleanup_case(case: FaultInjectionCase, case_root: Path) -> tuple[str, str, str, int, int]:
    ledger_path = case_root / "ledger.jsonl"
    run_root = case_root / "run"
    canonical = run_root / "units" / "gate1-1-CPU"
    ledger = recovery.JournaledLedger(ledger_path)

    def fail_unlink(path_obj: Path, *args: object, **kwargs: object) -> object:
        if ".pending" in str(path_obj):
            raise OSError(errno.EIO, "pending cleanup injected failure")
        return original_unlink(path_obj, *args, **kwargs)

    store = recovery.UnitTransactionStore(run_root / "units", attempt_id="gate1-cpu", ledger=ledger)
    receipt = store.prepare_unit(
        idempotency_key="gate1-cpu:gate1|1|CPU",
        unit_key=("gate1", 1, "CPU"),
        stage="cpu_fault_matrix",
        canonical_dir=canonical,
        files={"opaque-prediction.bin": b"opaque-gate1-fixture"},
        ledger_events=(_unit_payload()[0], _unit_payload()[1]),
    )
    original_unlink = Path.unlink
    with patch.object(Path, "unlink", fail_unlink):
        try:
            store.commit_unit(receipt)
        except BaseException:
            pass
    final = recovery.reconcile_unit_transaction(
        canonical_dir=canonical,
        ledger_path=ledger_path,
        idempotency_key="gate1-cpu:gate1|1|CPU",
        worker_alive=False,
    )
    classification, action = _normalise_state(final)
    return classification, action, "PENDING_CLEANUP_NOT_DURABLE", 0, 0

def _run_identity_case(case: FaultInjectionCase, case_root: Path) -> tuple[str, str, str, int, int]:
    ledger_path = case_root / "ledger.jsonl"
    run_root = case_root / "run"
    canonical = run_root / "units" / "gate1-1-CPU"
    ledger = recovery.JournaledLedger(ledger_path)
    store = recovery.UnitTransactionStore(run_root / "units", attempt_id="gate1-cpu", ledger=ledger)
    receipt = store.prepare_unit(
        idempotency_key="gate1-cpu:gate1|1|CPU",
        unit_key=("gate1", 1, "CPU"),
        stage="cpu_fault_matrix",
        canonical_dir=canonical,
        files={"opaque-prediction.bin": b"opaque-gate1-fixture"},
        ledger_events=(_unit_payload()[0], _unit_payload()[1]),
    )
    store.commit_unit(receipt)
    try:
        if case.operation == "identity_symlink":
            external = case_root / "external-fixture.bin"
            external.write_bytes(b"opaque-gate1-fixture")
            target = canonical / "opaque-prediction.bin"
            target.unlink()
            target.symlink_to(external)
        elif case.operation == "identity_path":
            receipt_path = canonical / "unit_transaction_receipt.ledger_committed.json"
            payload = json.loads(receipt_path.read_text(encoding="utf-8"))
            unsigned = dict(payload)
            unsigned["canonical_dir"] = str(run_root / "units" / "other-identity")
            unsigned.pop("receipt_sha256", None)
            payload["canonical_dir"] = unsigned["canonical_dir"]
            payload["receipt_sha256"] = recovery._sha256_json(unsigned)
            receipt_path.write_text(json.dumps(payload, sort_keys=True, separators=(",", ":")), encoding="utf-8")
        result = recovery.reconcile_unit_transaction(
            canonical_dir=canonical,
            ledger_path=ledger_path,
            idempotency_key="gate1-cpu:gate1|1|CPU",
            worker_alive=False,
        )
        classification, action = _normalise_state(result)
        return classification, action, str(result.get("reason", "")), 0, 0
    except BaseException as exc:
        return "FATAL_IDENTITY_MISMATCH", recovery.RECOVERY_ACTION_ABORT_FATAL, f"IDENTITY_EXCEPTION:{type(exc).__name__}", 0, 0


def _run_liveness_case(case: FaultInjectionCase, case_root: Path) -> tuple[str, str, str, int, int]:
    ledger_path = case_root / "ledger.jsonl"
    run_root = case_root / "run"
    canonical = run_root / "units" / "gate1-1-CPU"
    ledger = recovery.JournaledLedger(ledger_path)
    store = recovery.UnitTransactionStore(run_root / "units", attempt_id="gate1-cpu", ledger=ledger)
    receipt = store.prepare_unit(
        idempotency_key="gate1-cpu:gate1|1|CPU",
        unit_key=("gate1", 1, "CPU"),
        stage="cpu_fault_matrix",
        canonical_dir=canonical,
        files={"opaque-prediction.bin": b"opaque-gate1-fixture"},
        ledger_events=(_unit_payload()[0], _unit_payload()[1]),
    )
    store.commit_unit(receipt)
    active = recovery.reconcile_unit_transaction(
        canonical_dir=canonical,
        ledger_path=ledger_path,
        idempotency_key="gate1-cpu:gate1|1|CPU",
        worker_alive=True,
        heartbeat_path=case_root / "heartbeat.jsonl",
        heartbeat_max_age_seconds=0.0,
    )
    if active.get("state") != recovery.ACTIVE_VALID_DEFER_NO_MUTATION:
        return "FATAL_IDENTITY_MISMATCH", recovery.RECOVERY_ACTION_ABORT_FATAL, "LIVE_WORKER_GUARD_NOT_DEFERRED", 0, 0
    final = recovery.reconcile_unit_transaction(
        canonical_dir=canonical,
        ledger_path=ledger_path,
        idempotency_key="gate1-cpu:gate1|1|CPU",
        worker_alive=False,
    )
    classification, action = _normalise_state(final)
    return classification, action, "LIVE_GUARD_RECHECKED_AFTER_WORKER_EXIT", 0, 0

def _run_mutation_case(case: FaultInjectionCase, case_root: Path) -> tuple[str, str, str, int, int]:
    ledger_path = case_root / "ledger.jsonl"
    run_root = case_root / "run"
    canonical = run_root / "units" / "gate1-1-CPU"
    ledger = recovery.JournaledLedger(ledger_path)
    store = recovery.UnitTransactionStore(run_root / "units", attempt_id="gate1-cpu", ledger=ledger)
    receipt = store.prepare_unit(
        idempotency_key="gate1-cpu:gate1|1|CPU",
        unit_key=("gate1", 1, "CPU"),
        stage="cpu_fault_matrix",
        canonical_dir=canonical,
        files={"opaque-prediction.bin": b"opaque-gate1-fixture"},
        ledger_events=(_unit_payload()[0], _unit_payload()[1]),
    )
    store.commit_unit(receipt)
    if case.operation == "no_clobber":
        try:
            recovery.atomic_write_bytes(canonical / "opaque-prediction.bin", b"overwrite-forbidden")
        except BaseException:
            return "QUARANTINED", recovery.RECOVERY_ACTION_QUARANTINE, "NO_CLOBBER_REJECTED", 0, 0
        return "FATAL_IDENTITY_MISMATCH", recovery.RECOVERY_ACTION_ABORT_FATAL, "NO_CLOBBER_NOT_REJECTED", 0, 1
    if case.operation == "mutation":
        (canonical / "opaque-prediction.bin").write_bytes(b"mutated-gate1-fixture")
        key = "gate1-cpu:gate1|1|CPU"
    elif case.operation == "duplicate":
        key = "gate1-cpu:other|1|CPU"
    else:
        key = "gate1-cpu:gate1|1|CPU"
    try:
        result = recovery.reconcile_unit_transaction(
            canonical_dir=canonical,
            ledger_path=ledger_path,
            idempotency_key=key,
            worker_alive=False,
        )
        classification, action = _normalise_state(result)
        reason = str(result.get("reason", ""))
    except BaseException as exc:
        classification = "FATAL_IDENTITY_MISMATCH"
        action = recovery.RECOVERY_ACTION_ABORT_FATAL
        reason = f"RECONCILIATION_EXCEPTION:{type(exc).__name__}"
    return classification, action, reason, 0, 0


def _run_torn_case(case: FaultInjectionCase, case_root: Path) -> tuple[str, str, str, int, int]:
    ledger_path = case_root / "ledger.jsonl"
    ledger = recovery.JournaledLedger(ledger_path)
    ledger.append("BASELINE_EVENT", {"unit_key": "gate1|1|CPU"})
    if case.operation == "torn_exact_pending":
        try:
            ledger.append(
                "PENDING_EVENT",
                {"unit_key": "gate1|1|CPU"},
                fault_injector=lambda stage: (
                    (_ for _ in ()).throw(OSError(errno.EIO, "injected ledger append"))
                    if stage == "ledger_append"
                    else None
                ),
            )
        except OSError:
            pass
    with ledger_path.open("ab") as handle:
        handle.write(b'{"torn":')
    result = recovery.reconcile_unit_transaction(
        canonical_dir=case_root / "missing-unit",
        ledger_path=ledger_path,
        idempotency_key="gate1-cpu:gate1|1|CPU",
        worker_alive=False,
    )
    classification, action = _normalise_state(result)
    return classification, action, str(result.get("reason", "")), 0, 0


def _run_terminal_case(case: FaultInjectionCase, case_root: Path) -> tuple[str, str, str, int, int]:
    ledger = recovery.JournaledLedger(case_root / "ledger.jsonl")
    supervisor = recovery.HeartbeatSupervisor(case_root / "heartbeat.jsonl", attempt_id="gate1-cpu")
    exit_path = case_root / "exit-receipt.json"
    if case.operation == "terminal_failure":
        original_append = ledger.append
        original_exit = supervisor.write_exit

        def fail_append(*args: object, **kwargs: object) -> object:
            raise OSError(errno.EIO, "terminal ledger injected failure")

        def fail_exit(*args: object, **kwargs: object) -> object:
            raise OSError(errno.EIO, "supervisor receipt injected failure")

        ledger.append = fail_append  # type: ignore[method-assign]
        supervisor.write_exit = fail_exit  # type: ignore[method-assign]
        try:
            recovery.install_signal_exit_handlers(
                attempt_id="gate1-cpu",
                ledger=ledger,
                supervisor=supervisor,
                exit_receipt_path=exit_path,
                stage_getter=lambda: "cpu_fault_matrix",
                unit_key_getter=lambda: ("gate1", 1, "CPU"),
            )
        except BaseException:
            pass
        finally:
            ledger.append = original_append  # type: ignore[method-assign]
            supervisor.write_exit = original_exit  # type: ignore[method-assign]
        return "QUARANTINED", recovery.RECOVERY_ACTION_QUARANTINE, "TERMINAL_RECORD_PAIR_NOT_DURABLE", 0, 0
    return "QUARANTINED", recovery.RECOVERY_ACTION_QUARANTINE, "TERMINAL_CASE_UNSUPPORTED", 0, 0


def _run_signal_case(case: FaultInjectionCase, case_root: Path) -> tuple[str, str, str, int, int]:
    command = [
        sys.executable,
        "-m",
        "georeliab_mve.v4_recovery_fault_matrix",
        "--worker",
        str(case_root),
    ]
    process = subprocess.Popen(command, cwd=Path.cwd())
    time.sleep(0.15)
    number = {"SIGTERM": signal.SIGTERM, "SIGHUP": signal.SIGHUP, "SIGKILL": signal.SIGKILL}[case.fault_kind]
    os.kill(process.pid, number)
    return_code = process.wait(timeout=10)
    if number == signal.SIGKILL:
        supervisor = recovery.HeartbeatSupervisor(case_root / "heartbeat.jsonl", attempt_id="gate1-cpu")
        supervisor.write_exit(
            receipt_path=case_root / "exit-receipt.json",
            return_code=return_code,
            signal_number=int(number),
            heartbeat_age_seconds=0.0,
        )
        return "QUARANTINED", recovery.RECOVERY_ACTION_QUARANTINE, "SIGKILL_WITHOUT_TERMINAL_PAIR", 0, 0
    rows = recovery.read_hash_chain(case_root / "ledger.jsonl")
    if rows and (case_root / "exit-receipt.json").is_file():
        return "COMPLETE", recovery.RECOVERY_ACTION_NOOP, "TERMINAL_PAIR_DURABLE", 0, 0
    return "QUARANTINED", recovery.RECOVERY_ACTION_QUARANTINE, "SIGNAL_TERMINAL_PAIR_MISSING", 0, 0


def _run_case(case: FaultInjectionCase, case_root: Path) -> tuple[str, str, str, int, int]:
    if case.operation in {"mutation", "duplicate", "no_clobber"}:
        return _run_mutation_case(case, case_root)
    if case.operation in {"identity_symlink", "identity_path"}:
        return _run_identity_case(case, case_root)
    if case.operation == "pending_cleanup":
        return _run_pending_cleanup_case(case, case_root)
    if case.operation == "liveness":
        return _run_liveness_case(case, case_root)
    if case.operation.startswith("torn_"):
        return _run_torn_case(case, case_root)
    if case.operation in {"signal", "terminal_failure"}:
        return _run_signal_case(case, case_root) if case.operation == "signal" else _run_terminal_case(case, case_root)
    return _run_unit_case(case, case_root)


def build_fault_matrix_cases() -> tuple[FaultInjectionCase, ...]:
    points = (
        "artifact:temp_write",
        "artifact:file_fsync",
        "artifact:rename_before",
        "artifact:rename_after",
        "artifact:dir_fsync",
        "prepared_receipt:temp_write",
        "prepared_receipt:file_fsync",
        "prepared_receipt:rename_before",
        "prepared_receipt:rename_after",
        "prepared_receipt:dir_fsync",
        "ledger_pending:temp_write",
        "ledger_pending:file_fsync",
        "ledger_pending:rename_before",
        "ledger_pending:rename_after",
        "ledger_pending:dir_fsync",
        "canonical_receipt:temp_write",
        "canonical_receipt:file_fsync",
        "canonical_receipt:rename_before",
        "canonical_receipt:rename_after",
        "canonical_receipt:dir_fsync",
        "committed_receipt:temp_write",
        "committed_receipt:file_fsync",
        "committed_receipt:rename_before",
        "committed_receipt:rename_after",
        "committed_receipt:dir_fsync",
        "promotion:rename_before",
        "promotion:rename",
        "promotion:rename_after",
        "promotion:dir_fsync",
        "projection:ledger_append",
        "completion:ledger_append",
        "pending_cleanup:cleanup",
        "checkpoint:file_fsync",
    )
    cases: list[FaultInjectionCase] = []
    for point in points:
        operation = point.split(":", 1)[0]
        if operation == "checkpoint":
            operation = "transaction"
        expected = "QUARANTINED" if point.startswith(("artifact:", "prepared_receipt:")) else "SAFE_RETRY"
        if point.startswith("checkpoint:"):
            expected = "QUARANTINED"
        action = _expected_action(expected, recovery.RECOVERY_ACTION_RESUME_LEDGER_ONLY if expected == "SAFE_RETRY" else None)
        for kind in ERRNO_NAMES + ("ABRUPT_EXIT",):
            cases.append(
                FaultInjectionCase(
                    case_id=f"io-{point.replace(':', '-')}-{kind.lower()}",
                    operation=operation,
                    fault_point=point,
                    fault_kind=kind,
                    expected_classification=expected,
                    expected_action=action,
                )
            )
    cases.extend(
        (
            FaultInjectionCase("no-clobber-promotion", "no_clobber", "promotion:duplicate", "DUPLICATE", "QUARANTINED", recovery.RECOVERY_ACTION_QUARANTINE),
            FaultInjectionCase("identity-symlink", "identity_symlink", "identity:symlink", "SYMLINK", "FATAL_IDENTITY_MISMATCH", recovery.RECOVERY_ACTION_ABORT_FATAL),
            FaultInjectionCase("identity-path-mismatch", "identity_path", "identity:path", "PATH_MISMATCH", "FATAL_IDENTITY_MISMATCH", recovery.RECOVERY_ACTION_ABORT_FATAL),
            FaultInjectionCase("liveness-live-worker", "liveness", "liveness:live", "LIVE_WORKER", "COMPLETE", recovery.RECOVERY_ACTION_NOOP),
            FaultInjectionCase("liveness-dead-worker-stale-heartbeat", "liveness", "liveness:dead", "DEAD_WORKER", "COMPLETE", recovery.RECOVERY_ACTION_NOOP),
            FaultInjectionCase("torn-exact-pending", "torn_exact_pending", "ledger:torn_tail", "TornTail", "SAFE_RETRY", recovery.RECOVERY_ACTION_RESUME_LEDGER_ONLY),
            FaultInjectionCase("torn-without-pending", "torn_without_pending", "ledger:torn_tail", "TornTail", "QUARANTINED", recovery.RECOVERY_ACTION_QUARANTINE),
            FaultInjectionCase("identity-content-mutation", "mutation", "identity:content", "MUTATION", "FATAL_IDENTITY_MISMATCH", recovery.RECOVERY_ACTION_ABORT_FATAL),
            FaultInjectionCase("identity-duplicate-key", "duplicate", "identity:idempotency", "DUPLICATE", "FATAL_IDENTITY_MISMATCH", recovery.RECOVERY_ACTION_ABORT_FATAL),
            FaultInjectionCase("terminal-ledger-failure", "terminal_failure", "terminal:ledger", "EIO", "QUARANTINED", recovery.RECOVERY_ACTION_QUARANTINE),
            FaultInjectionCase("terminal-supervisor-failure", "terminal_failure", "terminal:supervisor", "EIO", "QUARANTINED", recovery.RECOVERY_ACTION_QUARANTINE),
            FaultInjectionCase("terminal-ledger-supervisor-pair", "terminal_failure", "terminal:pair", "EIO", "QUARANTINED", recovery.RECOVERY_ACTION_QUARANTINE),
            FaultInjectionCase("signal-sigterm", "signal", "signal:SIGTERM", "SIGTERM", "COMPLETE", recovery.RECOVERY_ACTION_NOOP),
            FaultInjectionCase("signal-sighup", "signal", "signal:SIGHUP", "SIGHUP", "COMPLETE", recovery.RECOVERY_ACTION_NOOP),
            FaultInjectionCase("signal-sigkill", "signal", "signal:SIGKILL", "SIGKILL", "QUARANTINED", recovery.RECOVERY_ACTION_QUARANTINE),
        )
    )
    return tuple(cases)


def _git(repo: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", *args], cwd=repo, check=True, text=True, capture_output=True
    )
    return result.stdout.strip()


def _verify_lineage(repo: Path, expected_parent_commit: str) -> dict[str, object]:
    status = _git(repo, "status", "--porcelain")
    if status:
        raise FaultMatrixError("V4_RECOVERY_GATE1_DIRTY_WORKTREE")
    current = _git(repo, "rev-parse", "HEAD")
    current_tree = _git(repo, "rev-parse", "HEAD^{tree}")
    parent = _git(repo, "rev-parse", "HEAD^")
    parent_tree = _git(repo, "rev-parse", f"{parent}^{{tree}}")
    if parent != expected_parent_commit:
        raise FaultMatrixError(
            f"V4_RECOVERY_GATE1_PARENT_MISMATCH:{parent}!={expected_parent_commit}"
        )
    if parent_tree != EXPECTED_PARENT_TREE:
        raise FaultMatrixError("V4_RECOVERY_GATE1_PARENT_TREE_MISMATCH")
    changed = [item for item in _git(repo, "diff", "--name-only", f"{parent}..HEAD").splitlines() if item]
    disallowed = sorted(set(changed) - GATE1_ALLOWLIST)
    if disallowed:
        raise FaultMatrixError(f"V4_RECOVERY_SCIENTIFIC_ASSET_DRIFT:{','.join(disallowed)}")
    return {
        "parent_commit": parent,
        "parent_tree": parent_tree,
        "current_commit": current,
        "current_tree": current_tree,
        "changed_paths": changed,
    }


def _write_no_clobber(path: Path, payload: bytes) -> None:
    if path.exists():
        raise FaultMatrixError(f"V4_RECOVERY_EVIDENCE_COLLISION:{path}")
    recovery.atomic_write_bytes(path, payload)


def _write_json(path: Path, payload: object) -> None:
    _write_no_clobber(path, _canonical_bytes(payload))


def _assert_no_scientific_payload(payload: object) -> None:
    encoded = json.dumps(payload, sort_keys=True, ensure_ascii=False).casefold()
    forbidden = ("auroc", "auc", "boundary_lag", "bootstrap", "metric", "claim", "paper")
    if any(token in encoded for token in forbidden):
        raise FaultMatrixError("V4_RECOVERY_SCIENTIFIC_PAYLOAD_DETECTED")


def _environment(repo: Path, lineage: Mapping[str, object]) -> dict[str, object]:
    return {
        "schema_version": SCHEMA_VERSION,
        "python": sys.version,
        "platform": platform.platform(),
        "hostname": platform.node(),
        "repo": str(repo),
        "lineage": dict(lineage),
        "scientific_result": NO_SCIENTIFIC_RESULT,
    }


def run_cpu_fault_matrix(
    *,
    output_root: Path,
    expected_parent_commit: str,
    repo: Path | None = None,
) -> dict[str, object]:
    repo = Path.cwd() if repo is None else Path(repo)
    lineage = _verify_lineage(repo, expected_parent_commit)
    output_root = Path(output_root)
    if output_root.exists():
        raise FaultMatrixError("V4_RECOVERY_GATE1_OUTPUT_ROOT_EXISTS")
    output_root.mkdir(parents=True, exist_ok=False)
    cases = build_fault_matrix_cases()
    observations: list[FaultInjectionObservation] = []
    for case in cases:
        case_root = output_root / "cases" / case.case_id
        case_root.mkdir(parents=True, exist_ok=False)
        before_inventory = _inventory_sha(case_root)
        before_ledger = _ledger_sha(case_root / "ledger.jsonl")
        try:
            classification, action, reason, duplicate_count, overwrite_count = _run_case(case, case_root)
        except BaseException as exc:
            classification = "FATAL_IDENTITY_MISMATCH"
            action = recovery.RECOVERY_ACTION_ABORT_FATAL
            reason = f"HARNESS_EXCEPTION:{type(exc).__name__}:{exc}"
            duplicate_count = 0
            overwrite_count = 0
        after_inventory = _inventory_sha(case_root)
        after_ledger = _ledger_sha(case_root / "ledger.jsonl")
        observation = FaultInjectionObservation(
            case_id=case.case_id,
            expected_classification=case.expected_classification,
            observed_classification=classification,
            observed_action=action,
            reason=reason,
            before_inventory_sha256=before_inventory,
            after_inventory_sha256=after_inventory,
            before_ledger_sha256=before_ledger,
            after_ledger_sha256=after_ledger,
            duplicate_count=duplicate_count,
            overwrite_count=overwrite_count,
        )
        observations.append(observation)
        if classification not in CLASSIFICATIONS:
            raise FaultMatrixError(f"V4_RECOVERY_UNKNOWN_CLASSIFICATION:{case.case_id}")
    unknown_count = sum(item.observed_classification not in CLASSIFICATIONS for item in observations)
    broad_count = sum("AUDIT_OR_RECORD_FAILED" in item.reason for item in observations)
    all_classified = (
        len(observations) == len(cases)
        and unknown_count == 0
        and broad_count == 0
        and all(item.passed for item in observations)
    )
    manifest = CpuFaultMatrixManifest(
        parent_commit=str(lineage["parent_commit"]),
        parent_tree=str(lineage["parent_tree"]),
        current_commit=str(lineage["current_commit"]),
        current_tree=str(lineage["current_tree"]),
        cases=cases,
        observations=tuple(observations),
        all_failure_injections_classified=all_classified,
        unknown_count=unknown_count,
        broad_legacy_reason_count=broad_count,
    )
    manifest_payload = manifest.to_dict()
    _assert_no_scientific_payload(manifest_payload)
    _write_json(output_root / "lineage-manifest.json", {**lineage, "allowlist": sorted(GATE1_ALLOWLIST), "scientific_assets_zero_drift": True, "scientific_result": NO_SCIENTIFIC_RESULT})
    _write_json(output_root / "source-manifest.json", {**lineage, "changed_paths": list(lineage["changed_paths"]), "scientific_assets_zero_drift": True, "scientific_result": NO_SCIENTIFIC_RESULT})
    _write_json(output_root / "fault-matrix-spec.json", {"cases": [case.to_dict() for case in cases], "schema_version": SCHEMA_VERSION})
    _write_no_clobber(
        output_root / "case-results.jsonl",
        b"".join(_canonical_bytes(item.to_dict()) + b"\n" for item in observations),
    )
    _write_json(output_root / "environment.json", _environment(repo, lineage))
    qualification = {
        "schema_version": SCHEMA_VERSION,
        "status": PASS_MARKER if all_classified else f"{FAIL_MARKER}=matrix",
        "runtime_status": READY_MARKER if all_classified else "V4_RECOVERY_RUNTIME_NOT_READY",
        "all_failure_injections_classified": all_classified,
        "case_count": len(cases),
        "passed_case_count": sum(item.passed for item in observations),
        "unknown_count": unknown_count,
        "broad_legacy_reason_count": broad_count,
        "scientific_result": NO_SCIENTIFIC_RESULT,
        "gate2_started": False,
        "pilot_started": False,
        "attempt06_started": False,
        "manifest_sha256": manifest_payload["manifest_sha256"],
    }
    _assert_no_scientific_payload(qualification)
    _write_json(output_root / "qualification.json", qualification)
    files = []
    for path in sorted(item for item in output_root.rglob("*") if item.is_file()):
        if path.name == "MANIFEST.sha256":
            continue
        files.append(f"{_sha_file(path)}  {path.relative_to(output_root).as_posix()}")
    _write_no_clobber(output_root / "MANIFEST.sha256", ("\n".join(files) + "\n").encode("utf-8"))
    return qualification


def _worker_main(root: Path) -> int:
    root.mkdir(parents=True, exist_ok=True)
    ledger = recovery.JournaledLedger(root / "ledger.jsonl")
    supervisor = recovery.HeartbeatSupervisor(root / "heartbeat.jsonl", attempt_id="gate1-cpu")
    recovery.install_signal_exit_handlers(
        attempt_id="gate1-cpu",
        ledger=ledger,
        supervisor=supervisor,
        exit_receipt_path=root / "exit-receipt.json",
        stage_getter=lambda: "cpu_fault_matrix",
        unit_key_getter=lambda: ("gate1", 1, "CPU"),
    )
    supervisor.beat(stage="idle", unit_key=("gate1", 1, "CPU"))
    while True:
        time.sleep(0.05)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--expected-parent-commit")
    parser.add_argument("--require-all-classified", action="store_true")
    parser.add_argument("--worker", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.worker is not None:
        return _worker_main(args.worker)
    if args.output_root is None or args.expected_parent_commit is None:
        raise SystemExit("--output-root and --expected-parent-commit are required")
    result = run_cpu_fault_matrix(
        output_root=args.output_root,
        expected_parent_commit=args.expected_parent_commit,
    )
    print(json.dumps(result, sort_keys=True))
    return 0 if result["status"] == PASS_MARKER else 2


if __name__ == "__main__":
    raise SystemExit(main())
