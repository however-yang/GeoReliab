"""Durable recovery primitives for the post-Attempt-05 executor.

This module deliberately has two small, separate surfaces:

* identity-only forensic inspection, which hashes files but never parses or
  emits scientific metric values; and
* a journaled unit transaction used by a future attempt.  The transaction
  leaves enough state behind for a cold-start reconciler to finish a committed
  unit without running the model a second time.

The frozen Attempt-05 worktree is not modified by these helpers.  Callers
should place manifests and receipts in a new forensic or attempt root.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import signal
import traceback
from typing import Any
from uuid import uuid4

from .v4_execution import V4ExecutionError


RECOVERY_SCHEMA = "georeliab-v4-recovery-1.0"
FORENSIC_SCHEMA = "georeliab-v4-attempt05-forensic-1.0"
ELIGIBILITY_SCHEMA = "georeliab-v4-attempt05-resume-eligibility-1.0"
UNION_SCHEMA = "georeliab-v4-immutable-union-1.0"
LEDGER_SCHEMA = "georeliab-v4-durable-ledger-1.0"
DURABILITY_CHECKPOINT_INTERVAL = 5
EXTERNAL_MONITOR_INTERVAL_SECONDS = 60 * 60
INTERNAL_HEARTBEAT_INTERVAL_SECONDS = 60
AUTHORIZED_TOTAL_UNIT_COUNT = 400
ATTEMPT05_EXPECTED_VALID_COUNT = 199
TARGET_GPU_HOURS = 35.0
HARD_GPU_HOURS = 50.0
TARGET_STORAGE_BYTES = 150 * 1024**3
HARD_STORAGE_BYTES = 1 * 1024**4
NO_SCIENTIFIC_RESULT = "NO_SCIENTIFIC_RESULT"

# Binding evidence is identity metadata only; values are never emitted by the
# forensic report. Aliases keep the gate compatible with older/newer receipt
# writers while still requiring one stable value per binding category.
BINDING_EVIDENCE_GROUPS: Mapping[str, tuple[str, ...]] = {
    "adapter": ("adapter_sha256", "adapter_hash", "adapter_id"),
    "config": ("config_sha256", "config_hash", "config_id"),
    "split": ("split_sha256", "split_hash", "split_id"),
    "corruption": ("corruption_sha256", "corruption_hash", "corruption_id"),
    "gpu": ("gpu_uuid", "gpu_id"),
    "gpu_index": ("physical_gpu_index", "gpu_index"),
    "tooling_commit": ("tooling_commit", "commit_sha", "commit"),
    "tooling_tree": ("tooling_tree", "tree_sha", "tree"),
}

FailureInjector = Callable[[str], None]


class Attempt05RecoveryError(V4ExecutionError):
    """Fail-closed error raised by the durable recovery layer."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _canonical_json_bytes(value: object) -> bytes:
    return (
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _fsync_dir(path: Path) -> None:
    """Persist a directory entry where the platform exposes directory fsync."""

    flags = getattr(os, "O_DIRECTORY", 0)
    try:
        descriptor = os.open(path, os.O_RDONLY | flags)
    except OSError:
        # Windows and some test filesystems do not permit opening directories.
        # The file fsync still provides the strongest available guarantee.
        return
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def atomic_write_bytes(
    path: Path,
    payload: bytes,
    *,
    fault_injector: FailureInjector | None = None,
) -> None:
    """Write bytes using same-filesystem temp -> fsync -> atomic replace.

    A failed operation intentionally leaves its ``.partial`` file for forensic
    classification.  Successful writes never leave a partial sibling.
    """

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.partial")

    def inject(stage: str) -> None:
        if fault_injector is not None:
            fault_injector(stage)

    try:
        inject("temp_write")
        with temporary.open("xb") as handle:
            handle.write(payload)
            handle.flush()
            inject("file_fsync")
            os.fsync(handle.fileno())
        inject("rename")
        os.replace(temporary, path)
        inject("dir_fsync")
        _fsync_dir(path.parent)
    except Exception:
        # Do not hide the original error and do not remove forensic evidence.
        raise


def atomic_write_json(
    path: Path,
    payload: Mapping[str, object],
    *,
    fault_injector: FailureInjector | None = None,
) -> None:
    atomic_write_bytes(
        path,
        _canonical_json_bytes(payload),
        fault_injector=fault_injector,
    )


def _sha256_json(value: object) -> str:
    return sha256_bytes(_canonical_json_bytes(value))


def _require_text(value: object, field_name: str) -> str:
    if not isinstance(value, str) or not value:
        raise Attempt05RecoveryError(f"V4_RECOVERY_INVALID_{field_name.upper()}")
    return value


def _normalize_unit_key(value: object) -> str:
    if isinstance(value, str) and value:
        return value
    if isinstance(value, Mapping):
        model = value.get("model_id")
        scene = value.get("scene_id")
        state = value.get("state_id")
        if isinstance(model, str) and type(scene) is int and isinstance(state, str):
            return f"{model}|{scene}|{state}"
    if isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray, str)):
        values = tuple(value)
        if len(values) == 3 and isinstance(values[0], str) and type(values[1]) is int and isinstance(values[2], str):
            return f"{values[0]}|{values[1]}|{values[2]}"
    raise Attempt05RecoveryError("V4_RECOVERY_UNIT_IDENTITY_INVALID")


@dataclass(frozen=True, slots=True)
class FailureEnvelope:
    """Machine-readable failure context that replaces broad exception collapse."""

    attempt_id: str
    unit_key: str | None
    stage: str
    reason_code: str
    exception_type: str
    message: str
    traceback_text: str
    traceback_sha256: str
    occurred_at: str
    exit_code: int | None = None
    signal_number: int | None = None
    scientific_result: str = NO_SCIENTIFIC_RESULT
    cause_reason_code: str | None = None
    schema_version: str = RECOVERY_SCHEMA

    def __post_init__(self) -> None:
        if self.scientific_result != NO_SCIENTIFIC_RESULT:
            raise Attempt05RecoveryError("V4_RECOVERY_SCIENTIFIC_RESULT_FORBIDDEN")

    @classmethod
    def from_exception(
        cls,
        exc: BaseException,
        *,
        attempt_id: str,
        stage: str,
        unit_key: object | None = None,
        reason_code: str | None = None,
        cause_reason_code: str | None = None,
        exit_code: int | None = None,
        signal_number: int | None = None,
    ) -> "FailureEnvelope":
        trace = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
        default = f"V4_{attempt_id.upper().replace('-', '_')}_{stage.upper()}_FAILED"
        selected = reason_code or default
        return cls(
            attempt_id=attempt_id,
            unit_key=None if unit_key is None else _normalize_unit_key(unit_key),
            stage=stage,
            reason_code=selected,
            exception_type=type(exc).__name__,
            message=str(exc),
            traceback_text=trace,
            traceback_sha256=sha256_bytes(trace.encode("utf-8")),
            occurred_at=_utc_now(),
            exit_code=exit_code,
            signal_number=signal_number,
            cause_reason_code=cause_reason_code,
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "attempt_id": self.attempt_id,
            "unit_key": self.unit_key,
            "stage": self.stage,
            "reason_code": self.reason_code,
            "exception_type": self.exception_type,
            "message": self.message,
            "traceback": self.traceback_text,
            "traceback_sha256": self.traceback_sha256,
            "occurred_at": self.occurred_at,
            "exit_code": self.exit_code,
            "signal": self.signal_number,
            "scientific_result": self.scientific_result,
            "cause_reason_code": self.cause_reason_code,
        }

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> "FailureEnvelope":
        trace = _require_text(value.get("traceback"), "traceback")
        trace_sha = _require_text(value.get("traceback_sha256"), "traceback_sha256")
        if sha256_bytes(trace.encode("utf-8")) != trace_sha:
            raise Attempt05RecoveryError("V4_RECOVERY_FAILURE_TRACE_TAMPER")
        signal_value = value.get("signal")
        exit_value = value.get("exit_code")
        if signal_value is not None and type(signal_value) is not int:
            raise Attempt05RecoveryError("V4_RECOVERY_FAILURE_SIGNAL_INVALID")
        if exit_value is not None and type(exit_value) is not int:
            raise Attempt05RecoveryError("V4_RECOVERY_FAILURE_EXIT_INVALID")
        scientific_result = value.get("scientific_result", NO_SCIENTIFIC_RESULT)
        if scientific_result != NO_SCIENTIFIC_RESULT:
            raise Attempt05RecoveryError("V4_RECOVERY_SCIENTIFIC_RESULT_FORBIDDEN")
        return cls(
            schema_version=_require_text(value.get("schema_version"), "schema_version"),
            attempt_id=_require_text(value.get("attempt_id"), "attempt_id"),
            unit_key=None if value.get("unit_key") is None else _normalize_unit_key(value.get("unit_key")),
            stage=_require_text(value.get("stage"), "stage"),
            reason_code=_require_text(value.get("reason_code"), "reason_code"),
            exception_type=_require_text(value.get("exception_type"), "exception_type"),
            message=_require_text(value.get("message"), "message"),
            traceback_text=trace,
            traceback_sha256=trace_sha,
            occurred_at=_require_text(value.get("occurred_at"), "occurred_at"),
            exit_code=exit_value,
            signal_number=signal_value,
            scientific_result=scientific_result,
            cause_reason_code=value.get("cause_reason_code"),
        )


def failure_envelope_for(
    exc: BaseException,
    *,
    attempt_id: str = "attempt-06",
    stage: str,
    unit_key: object | None = None,
    reason_code: str | None = None,
) -> FailureEnvelope:
    """Create a split, attributable reason code for an executor failure."""

    cause = str(exc) if str(exc).startswith("V4_") else None
    return FailureEnvelope.from_exception(
        exc,
        attempt_id=attempt_id,
        stage=stage,
        unit_key=unit_key,
        reason_code=reason_code,
        cause_reason_code=cause,
    )


@dataclass(frozen=True, slots=True)
class UnitTransactionReceipt:
    attempt_id: str
    idempotency_key: str
    unit_key: str
    stage: str
    state: str
    canonical_dir: str
    files: Mapping[str, str]
    ledger_events: tuple[Mapping[str, object], ...] = ()
    ledger_sequences: tuple[int, ...] = ()
    created_at: str = field(default_factory=_utc_now)
    updated_at: str = field(default_factory=_utc_now)
    schema_version: str = RECOVERY_SCHEMA

    VALID_STATES = frozenset({"PREPARED", "CANONICAL_PROMOTED", "LEDGER_COMMITTED"})

    def payload(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "attempt_id": self.attempt_id,
            "idempotency_key": self.idempotency_key,
            "unit_key": self.unit_key,
            "stage": self.stage,
            "state": self.state,
            "canonical_dir": self.canonical_dir,
            "files": dict(self.files),
            "ledger_events": [dict(event) for event in self.ledger_events],
            "ledger_sequences": list(self.ledger_sequences),
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }

    def to_dict(self) -> dict[str, object]:
        payload = self.payload()
        payload["receipt_sha256"] = _sha256_json(payload)
        return payload

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> "UnitTransactionReceipt":
        unsigned = dict(value)
        receipt_sha = _require_text(unsigned.pop("receipt_sha256", None), "receipt_sha256")
        if receipt_sha != _sha256_json(unsigned):
            raise Attempt05RecoveryError("V4_RECOVERY_RECEIPT_TAMPER")
        files = value.get("files")
        events = value.get("ledger_events", ())
        sequences = value.get("ledger_sequences", ())
        if (
            not isinstance(files, Mapping)
            or not isinstance(events, Sequence)
            or isinstance(events, (str, bytes))
            or any(not isinstance(item, Mapping) for item in events)
            or not isinstance(sequences, Sequence)
            or isinstance(sequences, (str, bytes))
            or any(type(item) is not int for item in sequences)
        ):
            raise Attempt05RecoveryError("V4_RECOVERY_RECEIPT_INVALID")
        state = _require_text(value.get("state"), "state")
        if state not in cls.VALID_STATES:
            raise Attempt05RecoveryError("V4_RECOVERY_RECEIPT_STATE_INVALID")
        return cls(
            schema_version=_require_text(value.get("schema_version"), "schema_version"),
            attempt_id=_require_text(value.get("attempt_id"), "attempt_id"),
            idempotency_key=_require_text(value.get("idempotency_key"), "idempotency_key"),
            unit_key=_normalize_unit_key(value.get("unit_key")),
            stage=_require_text(value.get("stage"), "stage"),
            state=state,
            canonical_dir=_require_text(value.get("canonical_dir"), "canonical_dir"),
            files={_require_text(key, "file_name"): _require_text(item, "file_sha256") for key, item in files.items()},
            ledger_events=tuple(dict(item) for item in events),
            ledger_sequences=tuple(sequences),
            created_at=_require_text(value.get("created_at"), "created_at"),
            updated_at=_require_text(value.get("updated_at"), "updated_at"),
        )


@dataclass(frozen=True, slots=True)
class DurabilityCheckpoint:
    attempt_id: str
    checkpoint_index: int
    last_sequence: int
    last_event_sha256: str
    completed_unit_keys: tuple[str, ...]
    gpu_inference_seconds: float
    logical_bytes: int
    allocated_bytes: int
    created_at: str = field(default_factory=_utc_now)
    schema_version: str = RECOVERY_SCHEMA

    def to_dict(self) -> dict[str, object]:
        payload = {
            "schema_version": self.schema_version,
            "attempt_id": self.attempt_id,
            "checkpoint_index": self.checkpoint_index,
            "last_sequence": self.last_sequence,
            "last_event_sha256": self.last_event_sha256,
            "completed_unit_keys": list(self.completed_unit_keys),
            "gpu_inference_seconds": self.gpu_inference_seconds,
            "logical_bytes": self.logical_bytes,
            "allocated_bytes": self.allocated_bytes,
            "created_at": self.created_at,
        }
        payload["checkpoint_sha256"] = _sha256_json(payload)
        return payload


def _event_hash(row: Mapping[str, object]) -> str:
    return _sha256_json({key: value for key, value in row.items() if key != "event_sha256"})


def read_hash_chain(ledger_path: Path, *, allow_empty: bool = True) -> list[dict[str, Any]]:
    if not ledger_path.exists():
        if allow_empty:
            return []
        raise Attempt05RecoveryError("V4_RECOVERY_LEDGER_MISSING")
    raw = ledger_path.read_bytes()
    if raw and not raw.endswith(b"\n"):
        raise Attempt05RecoveryError("V4_RECOVERY_LEDGER_TORN_TAIL")
    rows: list[dict[str, Any]] = []
    previous = "0" * 64
    for index, line in enumerate(raw.splitlines()):
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise Attempt05RecoveryError("V4_RECOVERY_LEDGER_CHAIN_INVALID") from exc
        if not isinstance(row, dict):
            raise Attempt05RecoveryError("V4_RECOVERY_LEDGER_CHAIN_INVALID")
        if row.get("sequence_index") != index or row.get("previous_event_sha256") != previous or row.get("event_sha256") != _event_hash(row):
            raise Attempt05RecoveryError("V4_RECOVERY_LEDGER_CHAIN_INVALID")
        previous = str(row["event_sha256"])
        rows.append(row)
    return rows


class JournaledLedger:
    """Append ledger rows through a pending-event journal."""

    def __init__(self, ledger_path: Path, *, schema_version: str = LEDGER_SCHEMA, pending_root: Path | None = None) -> None:
        self.ledger_path = Path(ledger_path)
        self.schema_version = schema_version
        self.pending_root = pending_root or self.ledger_path.with_name(self.ledger_path.name + ".pending")

    def append(
        self,
        event_type: str,
        payload: Mapping[str, object],
        *,
        fault_injector: FailureInjector | None = None,
    ) -> dict[str, object]:
        if not event_type:
            raise Attempt05RecoveryError("V4_RECOVERY_LEDGER_EVENT_INVALID")
        # Reconcile durable pending rows before allocating a new sequence. This
        # makes retry deterministic after a ledger append failure.
        self.reconcile_pending()
        rows = read_hash_chain(self.ledger_path)
        transaction_key = payload.get("_transaction_idempotency_key")
        if isinstance(transaction_key, str) and transaction_key:
            for existing in rows:
                existing_payload = existing.get("payload")
                if (
                    existing.get("event_type") == event_type
                    and isinstance(existing_payload, Mapping)
                    and existing_payload.get("_transaction_idempotency_key") == transaction_key
                ):
                    return existing
        schema = str(rows[0].get("schema_version")) if rows else self.schema_version
        previous = str(rows[-1]["event_sha256"]) if rows else "0" * 64
        row: dict[str, object] = {
            "schema_version": schema,
            "sequence_index": len(rows),
            "previous_event_sha256": previous,
            "event_type": event_type,
            "payload": dict(payload),
        }
        row["event_sha256"] = _event_hash(row)
        self.pending_root.mkdir(parents=True, exist_ok=True)
        pending = self.pending_root / f"{int(row['sequence_index']):08d}-{row['event_sha256']}.partial"
        atomic_write_json(pending, row, fault_injector=fault_injector)
        self.ledger_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            if fault_injector is not None:
                fault_injector("ledger_append")
            with self.ledger_path.open("ab") as handle:
                handle.write(_canonical_json_bytes(row))
                handle.flush()
                os.fsync(handle.fileno())
            _fsync_dir(self.ledger_path.parent)
        except Exception:
            raise
        pending.unlink(missing_ok=True)
        return row

    def reconcile_pending(self) -> list[dict[str, object]]:
        rows = read_hash_chain(self.ledger_path)
        by_hash = {str(row["event_sha256"]): row for row in rows}
        repaired: list[dict[str, object]] = []
        if not self.pending_root.exists():
            return repaired
        for pending in sorted(self.pending_root.glob("*.partial")):
            try:
                payload = json.loads(pending.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                raise Attempt05RecoveryError("V4_RECOVERY_PENDING_EVENT_INVALID") from exc
            if not isinstance(payload, dict) or payload.get("event_sha256") != _event_hash(payload):
                raise Attempt05RecoveryError("V4_RECOVERY_PENDING_EVENT_INVALID")
            event_hash = str(payload["event_sha256"])
            if event_hash in by_hash:
                pending.unlink(missing_ok=True)
                continue
            if payload.get("sequence_index") != len(rows) or payload.get("previous_event_sha256") != (str(rows[-1]["event_sha256"]) if rows else "0" * 64):
                raise Attempt05RecoveryError("V4_RECOVERY_PENDING_EVENT_ORDER_INVALID")
            self.ledger_path.parent.mkdir(parents=True, exist_ok=True)
            with self.ledger_path.open("ab") as handle:
                handle.write(_canonical_json_bytes(payload))
                handle.flush()
                os.fsync(handle.fileno())
            _fsync_dir(self.ledger_path.parent)
            rows.append(payload)
            by_hash[event_hash] = payload
            pending.unlink(missing_ok=True)
            repaired.append(payload)
        return repaired


def repair_torn_ledger_tail(ledger_path: Path) -> int:
    """Drop only an uncommitted, non-newline tail and return removed bytes."""

    path = Path(ledger_path)
    raw = path.read_bytes()
    if not raw or raw.endswith(b"\n"):
        return 0
    boundary = raw.rfind(b"\n") + 1
    if boundary <= 0:
        raise Attempt05RecoveryError("V4_RECOVERY_LEDGER_NO_COMMITTED_PREFIX")
    prefix = raw[:boundary]
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.partial")
    with temporary.open("xb") as handle:
        handle.write(prefix)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    _fsync_dir(path.parent)
    return len(raw) - len(prefix)


class UnitTransactionStore:
    """Prepare/promote/commit a unit without overwriting existing identities."""

    def __init__(self, root: Path, *, attempt_id: str, ledger: JournaledLedger | None = None) -> None:
        self.root = Path(root)
        self.attempt_id = attempt_id
        self.ledger = ledger

    def prepare_unit(
        self,
        *,
        idempotency_key: str,
        unit_key: object,
        stage: str,
        canonical_dir: Path,
        files: Mapping[str, bytes],
        ledger_events: Sequence[Mapping[str, object]] = (),
        fault_injector: FailureInjector | None = None,
    ) -> UnitTransactionReceipt:
        canonical_dir = Path(canonical_dir)
        if canonical_dir.exists():
            receipt_path = canonical_dir / "unit_transaction_receipt.json"
            if not receipt_path.is_file():
                raise Attempt05RecoveryError("V4_RECOVERY_CANONICAL_RECEIPT_MISSING")
            existing = UnitTransactionReceipt.from_mapping(json.loads(receipt_path.read_text(encoding="utf-8")))
            if existing.idempotency_key != idempotency_key or existing.unit_key != _normalize_unit_key(unit_key):
                raise Attempt05RecoveryError("V4_RECOVERY_DUPLICATE_IDENTITY")
            try:
                if Path(existing.canonical_dir).resolve() != canonical_dir.resolve():
                    raise Attempt05RecoveryError("V4_RECOVERY_CANONICAL_PATH_MISMATCH")
                for name, expected_sha256 in existing.files.items():
                    if Path(name).name != name:
                        raise Attempt05RecoveryError("V4_RECOVERY_FILE_NAME_INVALID")
                    candidate = canonical_dir / name
                    if not candidate.is_file() or sha256_file(candidate) != expected_sha256:
                        raise Attempt05RecoveryError("V4_RECOVERY_CANONICAL_HASH_MISMATCH")
            except OSError as exc:
                raise Attempt05RecoveryError("V4_RECOVERY_CANONICAL_HASH_UNREADABLE") from exc
            return existing
        staging = canonical_dir.with_name(canonical_dir.name + ".partial")
        if staging.exists():
            raise Attempt05RecoveryError("V4_RECOVERY_ORPHAN_PARTIAL")
        staging.mkdir(parents=True, exist_ok=False)
        file_hashes: dict[str, str] = {}
        try:
            for name, data in files.items():
                if Path(name).name != name:
                    raise Attempt05RecoveryError("V4_RECOVERY_FILE_NAME_INVALID")
                target = staging / name
                atomic_write_bytes(target, bytes(data), fault_injector=fault_injector)
                file_hashes[name] = sha256_file(target)
            receipt = UnitTransactionReceipt(
                attempt_id=self.attempt_id,
                idempotency_key=idempotency_key,
                unit_key=_normalize_unit_key(unit_key),
                stage=stage,
                state="PREPARED",
                canonical_dir=str(canonical_dir),
                files=file_hashes,
                ledger_events=tuple(dict(event) for event in ledger_events),
            )
            atomic_write_json(staging / "unit_transaction_receipt.json", receipt.to_dict(), fault_injector=fault_injector)
            _fsync_dir(staging)
            return receipt
        except Exception:
            # The staging tree is intentionally retained for forensic recovery.
            raise

    def promote_unit(self, receipt: UnitTransactionReceipt, *, fault_injector: FailureInjector | None = None) -> UnitTransactionReceipt:
        staging = Path(receipt.canonical_dir).with_name(Path(receipt.canonical_dir).name + ".partial")
        canonical = Path(receipt.canonical_dir)
        if canonical.exists():
            existing = UnitTransactionReceipt.from_mapping(json.loads((canonical / "unit_transaction_receipt.json").read_text(encoding="utf-8")))
            if existing.idempotency_key != receipt.idempotency_key:
                raise Attempt05RecoveryError("V4_RECOVERY_DUPLICATE_IDENTITY")
            return existing
        if not staging.exists():
            raise Attempt05RecoveryError("V4_RECOVERY_PREPARED_TREE_MISSING")
        if fault_injector is not None:
            fault_injector("rename")
        os.replace(staging, canonical)
        if fault_injector is not None:
            fault_injector("dir_fsync")
        _fsync_dir(canonical.parent)
        promoted = replace(receipt, state="CANONICAL_PROMOTED", updated_at=_utc_now())
        atomic_write_json(canonical / "unit_transaction_receipt.json", promoted.to_dict(), fault_injector=fault_injector)
        return promoted

    def commit_unit(self, receipt: UnitTransactionReceipt, *, fault_injector: FailureInjector | None = None) -> UnitTransactionReceipt:
        current = self.promote_unit(receipt, fault_injector=fault_injector)
        sequences: list[int] = []
        if self.ledger is not None:
            for event in current.ledger_events:
                event_type = _require_text(event.get("event_type"), "event_type")
                payload = event.get("payload", {})
                if not isinstance(payload, Mapping):
                    raise Attempt05RecoveryError("V4_RECOVERY_LEDGER_EVENT_INVALID")
                durable_payload = dict(payload)
                durable_payload.setdefault(
                    "_transaction_idempotency_key",
                    f"{current.idempotency_key}:{len(sequences)}",
                )
                row = self.ledger.append(
                    event_type,
                    durable_payload,
                    fault_injector=fault_injector,
                )
                sequences.append(int(row["sequence_index"]))
        committed = replace(
            current,
            state="LEDGER_COMMITTED",
            ledger_sequences=tuple(sequences),
            updated_at=_utc_now(),
        )
        atomic_write_json(Path(committed.canonical_dir) / "unit_transaction_receipt.json", committed.to_dict(), fault_injector=fault_injector)
        return committed

    def write_checkpoint(self, checkpoint: DurabilityCheckpoint, path: Path) -> None:
        atomic_write_json(path, checkpoint.to_dict())


def checkpoint_due(completed_unit_count: int) -> bool:
    return completed_unit_count > 0 and completed_unit_count % DURABILITY_CHECKPOINT_INTERVAL == 0


def build_monitor_status(
    *,
    attempt05_valid_completed: int,
    attempt06_valid_completed: int,
    attempt06_elapsed_seconds: float,
    cumulative_materialization_elapsed_seconds: float,
    gpu0_owners: Sequence[str] = (),
    gpu1_owners: Sequence[str] = (),
    cumulative_gpu_inference_seconds: float = 0.0,
    cumulative_storage_bytes: int = 0,
    authorized_total: int = AUTHORIZED_TOTAL_UNIT_COUNT,
) -> dict[str, object]:
    """Build the hourly external status without exposing scientific metrics."""

    if authorized_total <= 0:
        raise Attempt05RecoveryError("V4_RECOVERY_AUTHORIZED_TOTAL_INVALID")
    if attempt05_valid_completed < 0 or attempt06_valid_completed < 0:
        raise Attempt05RecoveryError("V4_RECOVERY_COMPLETION_NEGATIVE")
    completed = attempt05_valid_completed + attempt06_valid_completed
    if completed > authorized_total:
        raise Attempt05RecoveryError("V4_RECOVERY_COMPLETION_EXCEEDS_TOTAL")
    if attempt06_elapsed_seconds < 0 or cumulative_materialization_elapsed_seconds < 0:
        raise Attempt05RecoveryError("V4_RECOVERY_ELAPSED_NEGATIVE")
    if cumulative_gpu_inference_seconds < 0 or cumulative_storage_bytes < 0:
        raise Attempt05RecoveryError("V4_RECOVERY_USAGE_NEGATIVE")
    cumulative_gpu_hours = cumulative_gpu_inference_seconds / 3600.0
    cumulative_storage_gib = cumulative_storage_bytes / float(1024**3)
    return {
        "schema_version": RECOVERY_SCHEMA,
        "external_monitor_interval_seconds": EXTERNAL_MONITOR_INTERVAL_SECONDS,
        "internal_heartbeat_interval_seconds": INTERNAL_HEARTBEAT_INTERVAL_SECONDS,
        "authorized_total_units": authorized_total,
        "attempt05_valid_completed": attempt05_valid_completed,
        "attempt06_valid_completed": attempt06_valid_completed,
        "overall_completed_units": completed,
        "overall_progress": completed / authorized_total,
        "overall_progress_percent": 100.0 * completed / authorized_total,
        "attempt06_elapsed_seconds": attempt06_elapsed_seconds,
        "cumulative_materialization_elapsed_seconds": cumulative_materialization_elapsed_seconds,
        "gpu0_owners": sorted(str(owner) for owner in gpu0_owners),
        "gpu1_owners": sorted(str(owner) for owner in gpu1_owners),
        "cumulative_gpu_inference_seconds": cumulative_gpu_inference_seconds,
        "cumulative_gpu_inference_hours": cumulative_gpu_hours,
        "cumulative_storage_bytes": cumulative_storage_bytes,
        "cumulative_storage_gib": cumulative_storage_gib,
        "target_gpu_hours": TARGET_GPU_HOURS,
        "hard_gpu_hours": HARD_GPU_HOURS,
        "target_storage_bytes": TARGET_STORAGE_BYTES,
        "hard_storage_bytes": HARD_STORAGE_BYTES,
        "budget_status": {
            "gpu_target_exceeded": cumulative_gpu_hours > TARGET_GPU_HOURS,
            "gpu_hard_ceiling_exceeded": cumulative_gpu_hours > HARD_GPU_HOURS,
            "storage_target_exceeded": cumulative_storage_bytes > TARGET_STORAGE_BYTES,
            "storage_hard_ceiling_exceeded": cumulative_storage_bytes > HARD_STORAGE_BYTES,
        },
        "scientific_result": NO_SCIENTIFIC_RESULT,
    }


def reconcile_unit_transaction(
    *,
    canonical_dir: Path,
    ledger_path: Path,
    idempotency_key: str,
    pending_root: Path | None = None,
) -> dict[str, object]:
    """Classify a unit tree without rerunning inference or rewriting history.

    The result is deliberately identity/state-only. A caller may repair an
    INCOMPLETE transaction by replaying its pending receipt, but an ORPHAN or
    TORN_LEDGER_TAIL result is fail-closed until separately audited.
    """

    canonical = Path(canonical_dir)
    staging = canonical.with_name(canonical.name + ".partial")
    pending = Path(pending_root) if pending_root is not None else Path(ledger_path).with_name(Path(ledger_path).name + ".pending")
    result: dict[str, object] = {
        "state": "INCOMPLETE",
        "idempotency_key": idempotency_key,
        "canonical_dir": str(canonical),
        "pending_event_count": len(list(pending.glob("*.partial"))) if pending.exists() else 0,
    }
    try:
        rows = read_hash_chain(Path(ledger_path))
    except Attempt05RecoveryError as exc:
        if str(exc) == "V4_RECOVERY_LEDGER_TORN_TAIL":
            result.update({"state": "TORN_LEDGER_TAIL", "reason": str(exc)})
            return result
        result.update({"state": "ORPHAN", "reason": str(exc)})
        return result

    if not canonical.exists():
        if staging.exists():
            result.update({"reason": "CANONICAL_NOT_PROMOTED", "receipt_state": "PREPARED"})
        else:
            result.update({"reason": "NO_CANONICAL_OR_STAGING"})
        return result

    receipt_path = canonical / "unit_transaction_receipt.json"
    if not receipt_path.is_file():
        result.update({"state": "ORPHAN", "reason": "CANONICAL_RECEIPT_MISSING"})
        return result
    try:
        receipt = UnitTransactionReceipt.from_mapping(
            json.loads(receipt_path.read_text(encoding="utf-8"))
        )
    except (OSError, json.JSONDecodeError, Attempt05RecoveryError) as exc:
        result.update({"state": "ORPHAN", "reason": "CANONICAL_RECEIPT_INVALID", "detail": str(exc)})
        return result
    if receipt.idempotency_key != idempotency_key:
        result.update({"state": "ORPHAN", "reason": "IDEMPOTENCY_KEY_MISMATCH"})
        return result
    for name, expected_sha256 in receipt.files.items():
        if Path(name).name != name:
            result.update({"state": "ORPHAN", "reason": "FILE_NAME_INVALID"})
            return result
        candidate = canonical / name
        try:
            actual = sha256_file(candidate) if candidate.is_file() else None
        except OSError:
            actual = None
        if actual != expected_sha256:
            result.update({"state": "ORPHAN", "reason": "CANONICAL_HASH_MISMATCH", "file": name})
            return result

    result.update({"receipt_state": receipt.state, "unit_key": receipt.unit_key})
    if receipt.state != "LEDGER_COMMITTED":
        result["reason"] = "CANONICAL_PRESENT_LEDGER_PENDING"
        return result

    by_sequence = {int(row["sequence_index"]): row for row in rows}
    for sequence in receipt.ledger_sequences:
        row = by_sequence.get(sequence)
        if row is None:
            result.update({"reason": "LEDGER_SEQUENCE_MISSING"})
            return result
        payload = row.get("payload")
        transaction = payload.get("_transaction_idempotency_key") if isinstance(payload, Mapping) else None
        if not isinstance(transaction, str) or not transaction.startswith(idempotency_key + ":"):
            result.update({"reason": "LEDGER_IDEMPOTENCY_MISMATCH"})
            return result
    result["state"] = "COMPLETE"
    result["reason"] = "CANONICAL_AND_LEDGER_COMMITTED"
    return result


@dataclass(frozen=True, slots=True)
class ResumeEligibilityManifest:
    attempt_id: str
    schedule_sha256: str
    expected_schedule_count: int
    verified_unit_keys: tuple[str, ...]
    missing_unit_keys: tuple[str, ...]
    excluded_unit_keys: tuple[str, ...]
    checks: Mapping[str, object]
    verdict: str
    corpus_classification: str
    scientific_result: str = NO_SCIENTIFIC_RESULT
    generated_at: str = field(default_factory=_utc_now)
    schema_version: str = ELIGIBILITY_SCHEMA

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "attempt_id": self.attempt_id,
            "schedule_sha256": self.schedule_sha256,
            "expected_schedule_count": self.expected_schedule_count,
            "verified_unit_keys": list(self.verified_unit_keys),
            "missing_unit_keys": list(self.missing_unit_keys),
            "excluded_unit_keys": list(self.excluded_unit_keys),
            "checks": dict(self.checks),
            "verdict": self.verdict,
            "corpus_classification": self.corpus_classification,
            "scientific_result": self.scientific_result,
            "generated_at": self.generated_at,
        }


def _event_unit_key(row: Mapping[str, object]) -> str | None:
    payload = row.get("payload")
    return None if not isinstance(payload, Mapping) else _normalize_unit_key(payload) if {"model_id", "scene_id", "state_id"}.issubset(payload) else None


def _event_path_and_sha(row: Mapping[str, object]) -> tuple[str | None, str | None]:
    payload = row.get("payload")
    if not isinstance(payload, Mapping):
        return None, None
    path = next((payload.get(name) for name in ("projection_path", "record_path", "task_record_path", "canonical_path") if isinstance(payload.get(name), str)), None)
    digest = next((payload.get(name) for name in ("task_record_sha256", "record_sha256") if isinstance(payload.get(name), str)), None)
    return path, digest


def _safe_path(root: Path, value: str) -> Path:
    candidate = Path(value)
    resolved = candidate if candidate.is_absolute() else root / candidate
    resolved = resolved.resolve()
    try:
        resolved.relative_to(root.resolve())
    except ValueError as exc:
        raise Attempt05RecoveryError("V4_RECOVERY_PATH_ESCAPE") from exc
    return resolved


def _scan_forbidden_files(root: Path) -> list[str]:
    if not root.exists():
        return []
    found: list[str] = []
    for path in root.rglob("*"):
        relative = path.relative_to(root)
        if any(part.endswith(".partial") or part.endswith(".tmp") for part in relative.parts):
            found.append(relative.as_posix())
    return sorted(found)


def identity_only_audit(
    *,
    attempt_id: str,
    schedule_keys: Sequence[object],
    ledger_path: Path,
    run_root: Path,
    schedule_sha256: str,
    expected_schedule_sha256: str | None = None,
    input_closure_schedule_sha256: str | None = None,
    final_evidence_root: Path | None = None,
    expected_verified_count: int | None = None,
    require_binding_evidence: bool = False,
) -> ResumeEligibilityManifest:
    """Audit identities and hashes without parsing scientific record payloads."""

    normalized_schedule = tuple(_normalize_unit_key(key) for key in schedule_keys)
    schedule_set = set(normalized_schedule)
    checks: dict[str, object] = {
        "metric_values_inspected": False,
        "ledger_hash_chain": False,
        "unit_hash_reconciliation": False,
        "schedule_hash_match": False,
        "partials_or_temps": [],
        "final_evidence_present": False,
        "duplicate_units": [],
        "unknown_units": [],
        "invalid_unit_events": [],
        "verified_count": 0,
        "missing_count": 0,
        "reconciliation_failures": {
            "projection_or_completion_cardinality": 0,
            "path_or_sha_missing": 0,
            "canonical_missing": 0,
            "canonical_hash_mismatch": 0,
        },
        "binding_evidence_required": require_binding_evidence,
        "binding_evidence_present": not require_binding_evidence,
        "binding_evidence_missing_units": [],
        "binding_evidence_mismatch_units": [],
    }
    try:
        rows = read_hash_chain(ledger_path)
        checks["ledger_hash_chain"] = True
    except Attempt05RecoveryError as exc:
        checks["ledger_error"] = str(exc)
        rows = []
    projections: dict[str, list[tuple[str | None, str | None]]] = {}
    completions: dict[str, list[tuple[str | None, str | None]]] = {}
    binding_payloads: dict[str, list[Mapping[str, object]]] = {}
    unknown: set[str] = set()
    duplicates: set[str] = set()
    for row in rows:
        event_type = row.get("event_type")
        if event_type not in {"CANONICAL_RECORD_PROJECTION", "SCIENTIFIC_UNIT_COMPLETE"}:
            continue
        key = _event_unit_key(row)
        if key is None:
            checks.setdefault("invalid_unit_events", []).append(row.get("sequence_index"))
            continue
        if key not in schedule_set:
            unknown.add(key)
        if event_type == "SCIENTIFIC_UNIT_COMPLETE":
            payload = row.get("payload")
            status = payload.get("status") if isinstance(payload, Mapping) else None
            if status is not None and status not in {"VALID_COMPLETE", "RESUMED_VALID_COMPLETE"}:
                checks["invalid_unit_events"].append(row.get("sequence_index"))
                continue
        bucket = projections if event_type == "CANONICAL_RECORD_PROJECTION" else completions
        bucket.setdefault(key, []).append(_event_path_and_sha(row))
        if isinstance(row.get("payload"), Mapping):
            binding_payloads.setdefault(key, []).append(dict(row["payload"]))
        if len(bucket[key]) > 1:
            duplicates.add(key)
    checks["duplicate_units"] = sorted(duplicates)
    checks["unknown_units"] = sorted(unknown)

    verified: list[str] = []
    for key in normalized_schedule:
        projection = projections.get(key, [])
        completion = completions.get(key, [])
        if len(projection) != 1 or len(completion) != 1:
            checks["reconciliation_failures"]["projection_or_completion_cardinality"] += 1  # type: ignore[index]
            continue
        projection_path, projection_sha = projection[0]
        completion_path, completion_sha = completion[0]
        path_value = projection_path or completion_path
        digest_value = projection_sha or completion_sha
        if path_value is None or digest_value is None:
            checks["reconciliation_failures"]["path_or_sha_missing"] += 1  # type: ignore[index]
            continue
        if projection_sha is not None and completion_sha is not None and projection_sha != completion_sha:
            checks["reconciliation_failures"]["path_or_sha_missing"] += 1  # type: ignore[index]
            continue
        if projection_path is not None and completion_path is not None and projection_path != completion_path:
            checks["reconciliation_failures"]["path_or_sha_missing"] += 1  # type: ignore[index]
            continue
        binding_failed = False
        if require_binding_evidence:
            payloads = binding_payloads.get(key, [])
            missing_groups: list[str] = []
            mismatch_groups: list[str] = []
            for group, aliases in BINDING_EVIDENCE_GROUPS.items():
                observed: list[str] = []
                for payload in payloads:
                    candidate = next((payload[name] for name in aliases if name in payload), None)
                    if candidate is None:
                        missing_groups.append(group)
                        continue
                    observed.append(_sha256_json(candidate))
                if len(observed) != len(payloads) or not observed:
                    if group not in missing_groups:
                        missing_groups.append(group)
                elif len(set(observed)) != 1:
                    mismatch_groups.append(group)
            if missing_groups:
                checks["binding_evidence_missing_units"].append({"unit_key": key, "fields": sorted(set(missing_groups))})  # type: ignore[index]
            if mismatch_groups:
                checks["binding_evidence_mismatch_units"].append({"unit_key": key, "fields": sorted(set(mismatch_groups))})  # type: ignore[index]
            binding_failed = bool(missing_groups or mismatch_groups)
        try:
            path = _safe_path(run_root, path_value)
            if not path.is_file():
                checks["reconciliation_failures"]["canonical_missing"] += 1  # type: ignore[index]
                continue
            if sha256_file(path) != digest_value:
                checks["reconciliation_failures"]["canonical_hash_mismatch"] += 1  # type: ignore[index]
                continue
        except (OSError, Attempt05RecoveryError):
            checks["reconciliation_failures"]["canonical_missing"] += 1  # type: ignore[index]
            continue
        if binding_failed:
            continue
        verified.append(key)
    missing = [key for key in normalized_schedule if key not in set(verified)]
    checks["verified_count"] = len(verified)
    checks["missing_count"] = len(missing)
    if expected_verified_count is not None:
        checks["expected_verified_count"] = expected_verified_count
        checks["verified_count_matches_expected"] = len(verified) == expected_verified_count
    forbidden = _scan_forbidden_files(run_root)
    checks["partials_or_temps"] = forbidden
    if final_evidence_root is not None and final_evidence_root.exists() and any(final_evidence_root.rglob("*")):
        checks["final_evidence_present"] = True
    hash_values = [schedule_sha256]
    if expected_schedule_sha256 is not None:
        hash_values.append(expected_schedule_sha256)
    if input_closure_schedule_sha256 is not None:
        hash_values.append(input_closure_schedule_sha256)
    checks["schedule_hash_match"] = len(set(hash_values)) == 1
    checks["binding_evidence_present"] = not (
        checks["binding_evidence_missing_units"] or checks["binding_evidence_mismatch_units"]
    )
    # A partial corpus is eligible only when every present completion is
    # reconciled exactly once. Missing identities are expected for a frozen
    # interrupted attempt and are mechanically generated for the next attempt.
    checks["unit_hash_reconciliation"] = (
        bool(verified)
        and len(set(verified)) == len(verified)
        and len(verified) + len(missing) == len(normalized_schedule)
        and (expected_verified_count is None or len(verified) == expected_verified_count)
    )
    eligible = bool(
        checks["ledger_hash_chain"]
        and checks["unit_hash_reconciliation"]
        and checks["schedule_hash_match"]
        and not forbidden
        and not duplicates
        and not unknown
        and not checks["invalid_unit_events"]
        and not checks["final_evidence_present"]
        and bool(checks["binding_evidence_present"])
    )
    return ResumeEligibilityManifest(
        attempt_id=attempt_id,
        schedule_sha256=schedule_sha256,
        expected_schedule_count=len(normalized_schedule),
        verified_unit_keys=tuple(verified),
        missing_unit_keys=tuple(missing),
        excluded_unit_keys=tuple(sorted(unknown | duplicates)),
        checks=checks,
        verdict=(
            "V4_ATTEMPT05_PARTIAL_CORPUS_RESUME_ELIGIBLE"
            if eligible
            else "V4_ATTEMPT05_PARTIAL_CORPUS_NOT_RESUMABLE"
        ),
        corpus_classification=(
            "IMMUTABLE_PARTIAL_PREDICTION_CORPUS"
            if eligible
            else "CONDITIONALLY_RECOVERABLE_PARTIAL_RUN_STATE"
        ),
    )


def write_forensic_bundle(
    forensic_root: Path,
    *,
    source_paths: Mapping[str, Path],
    eligibility: ResumeEligibilityManifest,
    failure: FailureEnvelope | None = None,
    postmortem: Mapping[str, object] | None = None,
) -> dict[str, Path]:
    """Write independent reports; never writes any source path."""

    root = Path(forensic_root)
    root.mkdir(parents=True, exist_ok=True)
    source_manifest: dict[str, object] = {"schema_version": FORENSIC_SCHEMA, "generated_at": _utc_now(), "sources": {}}
    for label, source in sorted(source_paths.items()):
        path = Path(source)
        item: dict[str, object] = {"path": str(path), "exists": path.exists()}
        if path.is_file():
            stat = path.stat()
            item.update({"size": stat.st_size, "mtime_ns": stat.st_mtime_ns, "sha256": sha256_file(path)})
        elif path.is_dir():
            files: list[dict[str, object]] = []
            for child in sorted(path.rglob("*")):
                if child.is_file():
                    stat = child.stat()
                    files.append({"path": str(child.relative_to(path)), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns, "sha256": sha256_file(child)})
            item["files"] = files
        source_manifest["sources"][label] = item  # type: ignore[index]
    outputs: dict[str, Path] = {}
    for name, payload in (
        ("source-manifest.json", source_manifest),
        ("resume-eligibility.json", eligibility.to_dict()),
    ):
        target = root / name
        atomic_write_json(target, payload)
        outputs[name] = target
    if failure is not None:
        target = root / "failure-envelope.json"
        atomic_write_json(target, failure.to_dict())
        outputs[target.name] = target
    report_payload = dict(postmortem or {})
    report_payload.setdefault(
        "root_cause",
        "V4_ATTEMPT05_ROOT_CAUSE_UNOBSERVABLE_DUE_TO_EXCEPTION_COLLAPSE",
    )
    report = {
        "schema_version": FORENSIC_SCHEMA,
        "terminal_classification": "ATTEMPT05_TERMINAL_INFRASTRUCTURE_FAILURE",
        "reason_code": "V4_ATTEMPT05_AUDIT_OR_RECORD_FAILED",
        "scientific_result": NO_SCIENTIFIC_RESULT,
        "eligibility_verdict": eligibility.verdict,
        "postmortem": report_payload,
    }
    target = root / "postmortem.json"
    atomic_write_json(target, report)
    outputs[target.name] = target
    markdown = root / "postmortem.md"
    atomic_write_bytes(
        markdown,
        (
            "# GeoReliab v4 Attempt-05 Postmortem\n\n"
            "- terminal: `ATTEMPT05_TERMINAL_INFRASTRUCTURE_FAILURE`\n"
            "- reason: `V4_ATTEMPT05_AUDIT_OR_RECORD_FAILED`\n"
            "- scientific result: `NO_SCIENTIFIC_RESULT`\n"
            f"- resume verdict: `{eligibility.verdict}`\n"
        ).encode("utf-8"),
    )
    outputs[markdown.name] = markdown
    manifest_lines = []
    for name, path in sorted(outputs.items()):
        manifest_lines.append(f"{sha256_file(path)}  {name}")
    manifest_path = root / "MANIFEST.sha256"
    atomic_write_bytes(manifest_path, ("\n".join(manifest_lines) + "\n").encode("ascii"))
    outputs[manifest_path.name] = manifest_path
    return outputs


def build_missing_unit_schedule(
    schedule_keys: Sequence[object],
    eligibility: ResumeEligibilityManifest,
) -> tuple[str, ...]:
    expected = tuple(_normalize_unit_key(key) for key in schedule_keys)
    if tuple(expected) != tuple(dict.fromkeys(expected)):
        raise Attempt05RecoveryError("V4_RECOVERY_SCHEDULE_DUPLICATE_IDENTITY")
    if set(eligibility.verified_unit_keys) - set(expected):
        raise Attempt05RecoveryError("V4_RECOVERY_VERIFIED_UNIT_OUTSIDE_SCHEDULE")
    missing = tuple(key for key in expected if key not in set(eligibility.verified_unit_keys))
    if missing != tuple(eligibility.missing_unit_keys):
        raise Attempt05RecoveryError("V4_RECOVERY_MISSING_SET_MISMATCH")
    return missing


@dataclass(frozen=True, slots=True)
class ImmutableUnionManifest:
    schedule_sha256: str
    expected_count: int
    sources: Mapping[str, tuple[str, ...]]
    unit_keys: tuple[str, ...]
    scientific_result: str = NO_SCIENTIFIC_RESULT
    schema_version: str = UNION_SCHEMA

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "schedule_sha256": self.schedule_sha256,
            "expected_count": self.expected_count,
            "sources": {key: list(value) for key, value in self.sources.items()},
            "unit_keys": list(self.unit_keys),
            "scientific_result": self.scientific_result,
        }


def build_immutable_union_manifest(
    *,
    schedule_sha256: str,
    schedule_keys: Sequence[object],
    source_units: Mapping[str, Sequence[object]],
) -> ImmutableUnionManifest:
    expected = tuple(_normalize_unit_key(key) for key in schedule_keys)
    if len(expected) != len(set(expected)):
        raise Attempt05RecoveryError("V4_RECOVERY_SCHEDULE_DUPLICATE_IDENTITY")
    sources = {str(name): tuple(_normalize_unit_key(key) for key in values) for name, values in source_units.items()}
    flattened = [key for values in sources.values() for key in values]
    if len(flattened) != len(set(flattened)) or set(flattened) != set(expected):
        raise Attempt05RecoveryError("V4_RECOVERY_UNION_NOT_EXACT")
    return ImmutableUnionManifest(
        schedule_sha256=schedule_sha256,
        expected_count=len(expected),
        sources=sources,
        unit_keys=expected,
    )


def attempt06_gate(
    *,
    eligibility: ResumeEligibilityManifest,
    root_cause_resolved: bool,
    fault_injection_passed: bool,
    gpu_smoke_passed: bool,
    explicit_authorization: bool = False,
    missing_schedule_keys: Sequence[object] | None = None,
    expected_total_count: int | None = None,
    expected_missing_count: int | None = None,
) -> dict[str, object]:
    """Return a gate decision; this function never launches an execution."""

    missing_schedule_closed = False
    if missing_schedule_keys is not None:
        try:
            normalized_missing = tuple(_normalize_unit_key(key) for key in missing_schedule_keys)
            missing_schedule_closed = normalized_missing == tuple(eligibility.missing_unit_keys)
        except Attempt05RecoveryError:
            missing_schedule_closed = False
    count_closed = True
    if expected_total_count is not None:
        count_closed = (
            len(eligibility.verified_unit_keys) + len(eligibility.missing_unit_keys)
            == expected_total_count
        )
    if expected_missing_count is not None:
        count_closed = count_closed and len(eligibility.missing_unit_keys) == expected_missing_count
    checks = {
        "eligibility": eligibility.verdict == "V4_ATTEMPT05_PARTIAL_CORPUS_RESUME_ELIGIBLE",
        "root_cause_resolved": root_cause_resolved,
        "fault_injection_passed": fault_injection_passed,
        "gpu_smoke_passed": gpu_smoke_passed,
        "missing_schedule_closed": missing_schedule_closed,
        "missing_schedule_count_closed": count_closed,
        "explicit_authorization": explicit_authorization,
    }
    ready = all(checks.values())
    status = "V4_ATTEMPT06_AUTHORIZATION_READY" if ready else "V4_ATTEMPT06_BLOCKED_RECOVERY_GATE"
    if ready and not explicit_authorization:
        status = "V4_ATTEMPT06_READY_PENDING_EXPLICIT_AUTHORIZATION"
    return {
        "status": status,
        "checks": checks,
        "scientific_result": NO_SCIENTIFIC_RESULT,
        "launch_performed": False,
    }


def write_supervisor_exit_receipt(
    path: Path,
    *,
    attempt_id: str,
    return_code: int | None,
    signal_number: int | None,
    heartbeat_age_seconds: float | None,
    failure: FailureEnvelope | None = None,
) -> None:
    payload = {
        "schema_version": RECOVERY_SCHEMA,
        "attempt_id": attempt_id,
        "return_code": return_code,
        "signal": signal_number,
        "heartbeat_age_seconds": heartbeat_age_seconds,
        "failure": None if failure is None else failure.to_dict(),
        "scientific_result": NO_SCIENTIFIC_RESULT,
        "written_at": _utc_now(),
    }
    atomic_write_json(path, payload)


class HeartbeatSupervisor:
    """Small independent heartbeat/exit surface for a future executor."""

    def __init__(self, heartbeat_path: Path, *, attempt_id: str) -> None:
        self.heartbeat_path = Path(heartbeat_path)
        self.attempt_id = attempt_id

    def beat(
        self,
        *,
        stage: str,
        unit_key: object | None = None,
        payload: Mapping[str, object] | None = None,
    ) -> None:
        body: dict[str, object] = {
            "schema_version": RECOVERY_SCHEMA,
            "attempt_id": self.attempt_id,
            "stage": stage,
            "unit_key": None if unit_key is None else _normalize_unit_key(unit_key),
            "payload": dict(payload or {}),
            "written_at": _utc_now(),
            "scientific_result": NO_SCIENTIFIC_RESULT,
        }
        atomic_write_json(self.heartbeat_path, body)

    def write_exit(
        self,
        *,
        receipt_path: Path,
        return_code: int | None,
        signal_number: int | None,
        heartbeat_age_seconds: float | None,
        failure: FailureEnvelope | None = None,
    ) -> None:
        write_supervisor_exit_receipt(
            receipt_path,
            attempt_id=self.attempt_id,
            return_code=return_code,
            signal_number=signal_number,
            heartbeat_age_seconds=heartbeat_age_seconds,
            failure=failure,
        )

    def is_stale(self, *, now_epoch: float | None = None, max_age_seconds: float) -> bool:
        try:
            age = (now_epoch if now_epoch is not None else __import__("time").time()) - self.heartbeat_path.stat().st_mtime
        except OSError:
            return True
        return age > max_age_seconds


def install_signal_exit_handlers(
    *,
    attempt_id: str,
    ledger: JournaledLedger,
    supervisor: HeartbeatSupervisor,
    exit_receipt_path: Path,
    stage_getter: Callable[[], str],
    unit_key_getter: Callable[[], object | None] | None = None,
) -> dict[int, Any]:
    """Install fail-closed SIGTERM/SIGINT/SIGHUP handling for the main process.

    The handler writes a terminal ledger event and a separate supervisor exit
    receipt. If either write fails, it raises a recovery error instead of
    pretending the attempt completed cleanly. SIGKILL/OOM remains observable
    only through the independent supervisor's exit receipt.
    """

    previous: dict[int, Any] = {}

    def handle(signum: int, _frame: Any) -> None:
        signal_name = signal.Signals(signum).name
        stage = stage_getter()
        unit_key = unit_key_getter() if unit_key_getter is not None else None
        interrupted = InterruptedError(f"received {signal_name}")
        envelope = failure_envelope_for(
            interrupted,
            attempt_id=attempt_id,
            stage=stage,
            unit_key=unit_key,
            reason_code=f"V4_{attempt_id.upper().replace('-', '_')}_{signal_name}_INTERRUPTED",
        )
        terminal_payload = {
            "attempt_id": attempt_id,
            "status": "TERMINAL_INFRASTRUCTURE_FAILURE",
            "reason_code": envelope.reason_code,
            "stage": stage,
            "unit_key": envelope.unit_key,
            "signal": signum,
            "failure_envelope": envelope.to_dict(),
            "scientific_result": NO_SCIENTIFIC_RESULT,
        }
        failures: list[BaseException] = []
        try:
            ledger.append("ATTEMPT_TERMINAL", terminal_payload)
        except BaseException as exc:
            failures.append(exc)
        try:
            supervisor.write_exit(
                receipt_path=exit_receipt_path,
                return_code=None,
                signal_number=signum,
                heartbeat_age_seconds=None,
                failure=envelope,
            )
        except BaseException as exc:
            failures.append(exc)
        if failures:
            raise Attempt05RecoveryError("V4_RECOVERY_TERMINAL_RECORD_FAILED") from failures[0]
        raise SystemExit(128 + signum)

    for name in ("SIGTERM", "SIGINT", "SIGHUP"):
        candidate = getattr(signal, name, None)
        if candidate is not None:
            previous[int(candidate)] = signal.signal(candidate, handle)
    return previous


def restore_signal_exit_handlers(previous: Mapping[int, Any]) -> None:
    for number, handler in previous.items():
        signal.signal(signal.Signals(number), handler)


def classify_exit(return_code: int | None, signal_number: int | None) -> str:
    if signal_number == signal.SIGKILL:
        return "V4_ATTEMPT06_PROCESS_SIGKILL_OBSERVED"
    if signal_number is not None:
        return "V4_ATTEMPT06_PROCESS_SIGNAL_OBSERVED"
    if return_code not in (None, 0):
        return "V4_ATTEMPT06_PROCESS_EXIT_NONZERO"
    return "V4_ATTEMPT06_PROCESS_EXIT_CLEAN"
