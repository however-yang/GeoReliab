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
import errno
import ctypes
from pathlib import Path
import signal
import stat
import traceback
from typing import Any
from uuid import uuid4

from .v4_execution import V4ExecutionError


RECOVERY_SCHEMA = "georeliab-v4-recovery-1.0"
FORENSIC_SCHEMA = "georeliab-v4-attempt05-forensic-1.0"
ELIGIBILITY_SCHEMA = "georeliab-v4-attempt05-resume-eligibility-1.0"
UNION_SCHEMA = "georeliab-v4-immutable-union-1.0"
SAME_ATTEMPT_UNION_SCHEMA = "georeliab-v4-same-attempt-session-union-1.0"
RECOVERY_SMOKE_SCHEMA = "georeliab-v4-recovery-smoke-1.0"
RECOVERY_SMOKE_ATTEMPT_ID = "v4-recovery-smoke-01"
AUTHORIZED_GPU_UUID = "GPU-6ae218e6-3d51-b748-e308-1f0509e87886"
AUTHORIZED_PHYSICAL_GPU_INDEX = 0
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

# Recovery classifications are deliberately closed: callers must never turn an
# ambiguous on-disk state into a retry by guessing. Keep these strings stable
# because they are consumed by supervisors and qualification reports.
ACTIVE_VALID_DEFER_NO_MUTATION = "ACTIVE_VALID_DEFER_NO_MUTATION"
COMPLETE = "COMPLETE"
INCOMPLETE_SAFE_TO_RETRY = "INCOMPLETE_SAFE_TO_RETRY"
ORPHAN_REQUIRES_QUARANTINE = "ORPHAN_REQUIRES_QUARANTINE"
FATAL_IDENTITY_MISMATCH = "FATAL_IDENTITY_MISMATCH"

RECOVERY_ACTION_NOOP = "NOOP"
RECOVERY_ACTION_RESUME_LEDGER_ONLY = "RESUME_LEDGER_ONLY"
RECOVERY_ACTION_REINFER_UNIT = "REINFER_UNIT"
RECOVERY_ACTION_QUARANTINE = "QUARANTINE"
RECOVERY_ACTION_ABORT_FATAL = "ABORT_FATAL"

SCHEDULE_IDENTITY_SCHEMA = "georeliab-schedule-identity-1.0"
SCHEDULE_CANONICALIZER_VERSION = "georeliab-schedule-canonicalizer-1"
SCHEDULE_RAW_HASH_DOMAIN = "sha256:file-bytes:v1"
SCHEDULE_SEMANTIC_HASH_DOMAIN = "georeliab:schedule-semantic:v1"
ORDERED_UNIT_IDS_HASH_DOMAIN = "georeliab:ordered-unit-ids:v1"
SCHEDULE_IDENTITY_HASH_DOMAIN = "georeliab:schedule-identity:v1"
FILE_HASH_DOMAIN = "sha256:file-bytes:v1"

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


def _file_identity(metadata: os.stat_result) -> tuple[int, int, int, int, int]:
    """Return stable metadata used to guard canonical hash reads."""

    mtime_ns = getattr(metadata, "st_mtime_ns", None)
    ctime_ns = getattr(metadata, "st_ctime_ns", None)
    if not isinstance(mtime_ns, (int, float)):
        mtime_ns = float(metadata.st_mtime) * 1_000_000_000
    if not isinstance(ctime_ns, (int, float)):
        ctime_ns = float(metadata.st_ctime) * 1_000_000_000
    return (
        int(getattr(metadata, "st_dev", 0)),
        int(getattr(metadata, "st_ino", 0)),
        int(getattr(metadata, "st_size", 0)),
        int(mtime_ns),
        int(ctime_ns),
    )


def sha256_file(path: Path) -> str:
    """Hash one immutable regular file without following a symlink.

    The descriptor remains open for the complete read. Descriptor and path
    metadata are compared before and after hashing so a replace/write race is
    fail-closed instead of silently producing a transient canonical hash.
    """

    path = Path(path)
    before_path = os.lstat(path)
    if not stat.S_ISREG(before_path.st_mode):
        code = errno.ELOOP if stat.S_ISLNK(before_path.st_mode) else errno.EINVAL
        raise OSError(code, str(path))
    flags = os.O_RDONLY | int(getattr(os, "O_BINARY", 0)) | int(getattr(os, "O_NOFOLLOW", 0))
    descriptor = os.open(path, flags)
    try:
        before_fd = os.fstat(descriptor)
        if not stat.S_ISREG(before_fd.st_mode):
            raise OSError(errno.EINVAL, str(path))
        before_fd_identity = _file_identity(before_fd)
        before_path_identity = _file_identity(before_path)
        digest = hashlib.sha256()
        while True:
            block = os.read(descriptor, 1024 * 1024)
            if not block:
                break
            digest.update(block)
        after_fd = os.fstat(descriptor)
        after_path = os.lstat(path)
        if (
            _file_identity(after_fd) != before_fd_identity
            or _file_identity(after_path) != before_path_identity
        ):
            raise OSError(errno.EAGAIN, str(path))
        return digest.hexdigest()
    finally:
        os.close(descriptor)


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
    replace_existing: bool = False,
) -> None:
    """Write bytes through a no-clobber same-filesystem transaction.

    Production paths are append-only/immutable: an existing destination is
    never replaced. ``replace_existing=True`` is retained only as a
    compatibility guard for older callers; it is idempotent for byte-identical
    content and fails closed for any attempted mutation. A failed operation
    intentionally leaves its ``.partial`` file for forensic classification.
    """

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if replace_existing and path.exists():
        try:
            if path.read_bytes() == payload:
                return
        except OSError as exc:
            raise Attempt05RecoveryError("V4_RECOVERY_EXISTING_DESTINATION_UNREADABLE") from exc
        raise Attempt05RecoveryError("V4_RECOVERY_REPLACEMENT_FORBIDDEN")
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
        inject("rename_before")
        inject("rename")
        rename_noreplace(temporary, path)
        inject("rename_after")
        inject("dir_fsync")
        _fsync_dir(path.parent)
    except Exception:
        # Do not hide the original error and do not remove forensic evidence.
        raise


def rename_noreplace(source: Path, destination: Path) -> None:
    """Promote a directory without ever replacing an existing destination.

    Linux uses renameat2(RENAME_NOREPLACE) when available.  Windows
    ``os.rename`` fails when the destination exists.  On POSIX filesystems
    without renameat2 this helper fails closed rather than falling back to
    replace-by-default ``os.rename`` and risking a clobber race.
    """

    source = Path(source)
    destination = Path(destination)
    if destination.exists():
        raise Attempt05RecoveryError("V4_RECOVERY_DUPLICATE_IDENTITY")
    if os.name != "nt":
        try:
            libc = ctypes.CDLL(None, use_errno=True)
            renameat2 = getattr(libc, "renameat2", None)
        except (AttributeError, OSError):
            renameat2 = None
        if renameat2 is not None:
            renameat2.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_uint]  # type: ignore[attr-defined]
            renameat2.restype = ctypes.c_int  # type: ignore[attr-defined]
            result = renameat2(-100, os.fsencode(source), -100, os.fsencode(destination), 1)  # type: ignore[misc,operator]
            if result == 0:
                return
            error = ctypes.get_errno()
            if error == errno.EEXIST:
                raise Attempt05RecoveryError("V4_RECOVERY_DUPLICATE_IDENTITY")
            if error != errno.ENOSYS:
                raise OSError(error, os.strerror(error), str(source), str(destination))
    if os.name != "nt":
        # A plain POSIX os.rename is replace-by-default and is therefore not
        # an acceptable fallback for immutable unit promotion.  Fail closed
        # when renameat2 is unavailable rather than risking a clobber race.
        raise Attempt05RecoveryError("V4_RECOVERY_RENAME_NOREPLACE_UNAVAILABLE")
    try:
        os.rename(source, destination)
    except FileExistsError as exc:
        raise Attempt05RecoveryError("V4_RECOVERY_DUPLICATE_IDENTITY") from exc


def atomic_write_json(
    path: Path,
    payload: Mapping[str, object],
    *,
    fault_injector: FailureInjector | None = None,
    replace_existing: bool = False,
) -> None:
    atomic_write_bytes(
        path,
        _canonical_json_bytes(payload),
        fault_injector=fault_injector,
        replace_existing=replace_existing,
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


def _canonical_json_bytes_without_newline(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _domain_hash(domain: str, value: object) -> str:
    return sha256_bytes(domain.encode("utf-8") + b"\0" + _canonical_json_bytes_without_newline(value))


@dataclass(frozen=True, slots=True)
class ScheduleIdentityManifest:
    """Domain-separated binding for raw bytes and parsed schedule semantics."""

    raw_sha256: str
    semantic_sha256: str
    schema_version: str
    canonicalizer_version: str
    unit_count: int
    ordered_unit_ids_sha256: str
    schedule_identity_sha256: str

    def unsigned_payload(self) -> dict[str, object]:
        return {
            "raw_sha256": self.raw_sha256,
            "semantic_sha256": self.semantic_sha256,
            "schema_version": self.schema_version,
            "canonicalizer_version": self.canonicalizer_version,
            "unit_count": self.unit_count,
            "ordered_unit_ids_sha256": self.ordered_unit_ids_sha256,
        }

    def to_dict(self) -> dict[str, object]:
        payload = self.unsigned_payload()
        payload["schedule_identity_sha256"] = self.schedule_identity_sha256
        return payload

    @classmethod
    def build(
        cls,
        *,
        raw_sha256: str,
        semantic_payload: object,
        ordered_unit_ids: Sequence[object],
        schema_version: str = SCHEDULE_IDENTITY_SCHEMA,
        canonicalizer_version: str = SCHEDULE_CANONICALIZER_VERSION,
    ) -> "ScheduleIdentityManifest":
        normalized = [_normalize_unit_key(item) for item in ordered_unit_ids]
        semantic_sha = _domain_hash(SCHEDULE_SEMANTIC_HASH_DOMAIN, semantic_payload)
        ordered_sha = _domain_hash(ORDERED_UNIT_IDS_HASH_DOMAIN, normalized)
        unsigned = {
            "raw_sha256": _require_text(raw_sha256, "raw_sha256"),
            "semantic_sha256": semantic_sha,
            "schema_version": _require_text(schema_version, "schema_version"),
            "canonicalizer_version": _require_text(canonicalizer_version, "canonicalizer_version"),
            "unit_count": len(normalized),
            "ordered_unit_ids_sha256": ordered_sha,
        }
        identity = _domain_hash(SCHEDULE_IDENTITY_HASH_DOMAIN, unsigned)
        return cls(schedule_identity_sha256=identity, **unsigned)

    @classmethod
    def from_schedule_bytes(
        cls,
        raw_bytes: bytes,
        parsed_schedule: object,
        ordered_unit_ids: Sequence[object],
        *,
        schema_version: str = SCHEDULE_IDENTITY_SCHEMA,
        canonicalizer_version: str = SCHEDULE_CANONICALIZER_VERSION,
    ) -> "ScheduleIdentityManifest":
        """Build the canonical identity from raw bytes and parsed semantics."""

        if not isinstance(raw_bytes, (bytes, bytearray)):
            raise Attempt05RecoveryError("V4_RECOVERY_SCHEDULE_RAW_BYTES_INVALID")
        return cls.build(
            raw_sha256=sha256_bytes(bytes(raw_bytes)),
            semantic_payload=parsed_schedule,
            ordered_unit_ids=ordered_unit_ids,
            schema_version=schema_version,
            canonicalizer_version=canonicalizer_version,
        )

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> "ScheduleIdentityManifest":
        required = (
            "raw_sha256", "semantic_sha256", "schema_version",
            "canonicalizer_version", "unit_count",
            "ordered_unit_ids_sha256", "schedule_identity_sha256",
        )
        if any(name not in value for name in required):
            raise Attempt05RecoveryError("V4_RECOVERY_SCHEDULE_IDENTITY_MANIFEST_INVALID")
        unit_count = value.get("unit_count")
        if type(unit_count) is not int or unit_count < 0:
            raise Attempt05RecoveryError("V4_RECOVERY_SCHEDULE_IDENTITY_COUNT_INVALID")
        payload = {
            "raw_sha256": _require_text(value.get("raw_sha256"), "raw_sha256"),
            "semantic_sha256": _require_text(value.get("semantic_sha256"), "semantic_sha256"),
            "schema_version": _require_text(value.get("schema_version"), "schema_version"),
            "canonicalizer_version": _require_text(value.get("canonicalizer_version"), "canonicalizer_version"),
            "unit_count": unit_count,
            "ordered_unit_ids_sha256": _require_text(value.get("ordered_unit_ids_sha256"), "ordered_unit_ids_sha256"),
        }
        identity = _require_text(value.get("schedule_identity_sha256"), "schedule_identity_sha256")
        if identity != _domain_hash(SCHEDULE_IDENTITY_HASH_DOMAIN, payload):
            raise Attempt05RecoveryError("V4_RECOVERY_SCHEDULE_IDENTITY_TAMPER")
        return cls(schedule_identity_sha256=identity, **payload)


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
    worker_pid: int | None = None
    heartbeat_age_seconds: float | None = None
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
        worker_pid: int | None = None,
        heartbeat_age_seconds: float | None = None,
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
            worker_pid=worker_pid,
            heartbeat_age_seconds=heartbeat_age_seconds,
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
            "worker_pid": self.worker_pid,
            "heartbeat_age_seconds": self.heartbeat_age_seconds,
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
        worker_pid = value.get("worker_pid")
        if worker_pid is not None and (type(worker_pid) is not int or worker_pid <= 0):
            raise Attempt05RecoveryError("V4_RECOVERY_FAILURE_WORKER_PID_INVALID")
        heartbeat_age = value.get("heartbeat_age_seconds")
        if heartbeat_age is not None and (type(heartbeat_age) not in (int, float) or heartbeat_age < 0):
            raise Attempt05RecoveryError("V4_RECOVERY_FAILURE_HEARTBEAT_AGE_INVALID")
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
            worker_pid=worker_pid,
            heartbeat_age_seconds=None if heartbeat_age is None else float(heartbeat_age),
        )


def failure_envelope_for(
    exc: BaseException,
    *,
    attempt_id: str = "attempt-06",
    stage: str,
    unit_key: object | None = None,
    reason_code: str | None = None,
    worker_pid: int | None = None,
    heartbeat_age_seconds: float | None = None,
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
        worker_pid=worker_pid,
        heartbeat_age_seconds=heartbeat_age_seconds,
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
    # Explicit domains prevent a raw schedule digest from being compared with
    # a semantic/identity digest merely because both fields are named SHA-256.
    schedule_identity_sha256: str | None = None
    schedule_hash_domain: str = SCHEDULE_IDENTITY_HASH_DOMAIN
    file_hash_domain: str = FILE_HASH_DOMAIN
    hash_domains: Mapping[str, str] = field(default_factory=dict)
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
            "schedule_identity_sha256": self.schedule_identity_sha256,
            "schedule_hash_domain": self.schedule_hash_domain,
            "file_hash_domain": self.file_hash_domain,
            "hash_domains": dict(self.hash_domains),
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
        schedule_identity = value.get("schedule_identity_sha256")
        if schedule_identity is not None:
            schedule_identity = _require_text(schedule_identity, "schedule_identity_sha256")
        schedule_domain = _require_text(value.get("schedule_hash_domain", SCHEDULE_IDENTITY_HASH_DOMAIN), "schedule_hash_domain")
        file_domain = _require_text(value.get("file_hash_domain", FILE_HASH_DOMAIN), "file_hash_domain")
        domains = value.get("hash_domains", {})
        if not isinstance(domains, Mapping) or any(not isinstance(k, str) or not isinstance(v, str) or not v for k, v in domains.items()):
            raise Attempt05RecoveryError("V4_RECOVERY_RECEIPT_HASH_DOMAINS_INVALID")
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
            schedule_identity_sha256=schedule_identity,
            schedule_hash_domain=schedule_domain,
            file_hash_domain=file_domain,
            hash_domains=dict(domains),
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


def segment_torn_ledger_tail(
    ledger_path: Path,
    *,
    segment_path: Path | None = None,
) -> dict[str, object]:
    """Archive an opaque torn tail without modifying the source ledger.

    The committed prefix remains byte-for-byte unchanged.  The segment is a
    separate append-only evidence file and is created no-clobber; callers may
    use its digest and prefix metadata to construct a future chain bridge.
    No event is guessed or re-serialized from the torn bytes.
    """

    path = Path(ledger_path)
    raw = path.read_bytes() if path.exists() else b""
    rows, tail = _read_committed_prefix(path)
    prefix_bytes = raw[: len(raw) - len(tail)] if tail else raw
    result: dict[str, object] = {
        "ledger_path": str(path),
        "committed_prefix_length": len(rows),
        "committed_prefix_sha256": sha256_bytes(prefix_bytes),
        "torn_tail_bytes": len(tail),
        "torn_tail_sha256": None if not tail else sha256_bytes(tail),
        "source_sha256": sha256_bytes(raw),
        "segment_path": None,
    }
    result["chain_bridge"] = build_ledger_chain_bridge(
        source_ledger_path=path,
        committed_prefix_length=len(rows),
        committed_prefix_sha256=str(result["committed_prefix_sha256"]),
        previous_event_sha256=str(rows[-1]["event_sha256"]) if rows else "0" * 64,
        torn_tail_sha256=None if not tail else str(result["torn_tail_sha256"]),
        source_ledger_sha256=str(result["source_sha256"]),
    )
    if not tail:
        return result
    digest = sha256_bytes(tail)
    target = Path(segment_path) if segment_path is not None else path.with_name(
        f"{path.name}.torn.{digest[:16]}.segment"
    )
    if target.exists():
        if target.read_bytes() != tail:
            raise Attempt05RecoveryError("V4_RECOVERY_TORN_SEGMENT_COLLISION")
    else:
        atomic_write_bytes(target, tail)
    result["segment_path"] = str(target)
    return result


def archive_torn_ledger_tail(
    ledger_path: Path,
    *,
    segment_path: Path | None = None,
) -> dict[str, object]:
    """Compatibility alias for the non-destructive torn-tail segmenter."""

    return segment_torn_ledger_tail(ledger_path, segment_path=segment_path)


def repair_torn_ledger_tail(ledger_path: Path) -> int:
    """Deprecated compatibility API that archives, never truncates, a tail."""

    result = segment_torn_ledger_tail(ledger_path)
    return int(result["torn_tail_bytes"])


_RECEIPT_STATE_RANK: Mapping[str, int] = {
    "PREPARED": 0,
    "CANONICAL_PROMOTED": 1,
    "LEDGER_COMMITTED": 2,
}


def _receipt_state_path(canonical_dir: Path, state: str) -> Path:
    canonical_dir = Path(canonical_dir)
    if state == "PREPARED":
        return canonical_dir / "unit_transaction_receipt.json"
    if state not in _RECEIPT_STATE_RANK:
        raise Attempt05RecoveryError("V4_RECOVERY_RECEIPT_STATE_INVALID")
    return canonical_dir / f"unit_transaction_receipt.{state.lower()}.json"


def _load_latest_receipt(canonical_dir: Path) -> UnitTransactionReceipt:
    canonical_dir = Path(canonical_dir)
    candidates = []
    base = _receipt_state_path(canonical_dir, "PREPARED")
    if base.is_file():
        candidates.append(base)
    candidates.extend(sorted(canonical_dir.glob("unit_transaction_receipt.*.json")))
    if not candidates:
        raise Attempt05RecoveryError("V4_RECOVERY_CANONICAL_RECEIPT_MISSING")
    receipts: list[UnitTransactionReceipt] = []
    for path in candidates:
        try:
            receipts.append(UnitTransactionReceipt.from_mapping(json.loads(path.read_text(encoding="utf-8"))))
        except (OSError, json.JSONDecodeError, Attempt05RecoveryError) as exc:
            raise Attempt05RecoveryError("V4_RECOVERY_CANONICAL_RECEIPT_INVALID") from exc
    return max(receipts, key=lambda item: (_RECEIPT_STATE_RANK[item.state], item.updated_at))


def _write_receipt_version(
    canonical_dir: Path,
    receipt: UnitTransactionReceipt,
    *,
    fault_injector: FailureInjector | None = None,
) -> None:
    target = _receipt_state_path(canonical_dir, receipt.state)
    payload = _canonical_json_bytes(receipt.to_dict())
    if target.exists():
        try:
            existing = UnitTransactionReceipt.from_mapping(
                json.loads(target.read_text(encoding="utf-8"))
            )
        except (OSError, json.JSONDecodeError, Attempt05RecoveryError) as exc:
            raise Attempt05RecoveryError("V4_RECOVERY_RECEIPT_VERSION_UNREADABLE") from exc
        # A retry with the same immutable receipt identity is a no-op. Any
        # differing state/hash is a collision and must fail closed.
        if existing.to_dict() == receipt.to_dict():
            return
        if (
            existing.attempt_id == receipt.attempt_id
            and existing.idempotency_key == receipt.idempotency_key
            and existing.unit_key == receipt.unit_key
            and existing.state == receipt.state
            and existing.files == receipt.files
            and existing.ledger_sequences == receipt.ledger_sequences
        ):
            return
        raise Attempt05RecoveryError("V4_RECOVERY_RECEIPT_VERSION_COLLISION")
    atomic_write_bytes(target, payload, fault_injector=fault_injector)


class UnitTransactionStore:
    """Prepare/promote/commit a unit without overwriting existing identities."""

    def __init__(
        self,
        root: Path,
        *,
        attempt_id: str,
        ledger: JournaledLedger | None = None,
        schedule_identity_sha256: str | None = None,
        schedule_hash_domain: str = SCHEDULE_IDENTITY_HASH_DOMAIN,
        hash_domains: Mapping[str, str] | None = None,
    ) -> None:
        self.root = Path(root)
        self.attempt_id = attempt_id
        self.ledger = ledger
        self.schedule_identity_sha256 = schedule_identity_sha256
        self.schedule_hash_domain = schedule_hash_domain
        self.hash_domains = dict(hash_domains or {})

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
        schedule_identity_sha256: str | None = None,
        schedule_hash_domain: str | None = None,
        hash_domains: Mapping[str, str] | None = None,
    ) -> UnitTransactionReceipt:
        canonical_dir = Path(canonical_dir)
        if canonical_dir.exists():
            existing = _load_latest_receipt(canonical_dir)
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
                schedule_identity_sha256=(
                    self.schedule_identity_sha256
                    if schedule_identity_sha256 is None
                    else schedule_identity_sha256
                ),
                schedule_hash_domain=schedule_hash_domain or self.schedule_hash_domain,
                hash_domains=dict(self.hash_domains if hash_domains is None else hash_domains),
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
            existing = _load_latest_receipt(canonical)
            if existing.idempotency_key != receipt.idempotency_key:
                raise Attempt05RecoveryError("V4_RECOVERY_DUPLICATE_IDENTITY")
            if existing.unit_key != receipt.unit_key:
                raise Attempt05RecoveryError("V4_RECOVERY_UNIT_IDENTITY_MISMATCH")
            if Path(existing.canonical_dir).resolve() != canonical.resolve():
                raise Attempt05RecoveryError("V4_RECOVERY_CANONICAL_PATH_MISMATCH")
            for name, expected_sha256 in existing.files.items():
                if Path(name).name != name:
                    raise Attempt05RecoveryError("V4_RECOVERY_FILE_NAME_INVALID")
                candidate = canonical / name
                if not candidate.is_file() or sha256_file(candidate) != expected_sha256:
                    raise Attempt05RecoveryError("V4_RECOVERY_CANONICAL_HASH_MISMATCH")
            return existing
        if not staging.exists():
            raise Attempt05RecoveryError("V4_RECOVERY_PREPARED_TREE_MISSING")
        if fault_injector is not None:
            # Expose both edges of the promotion window for qualification
            # fault injection while retaining the legacy ``rename`` hook.
            fault_injector("rename_before")
            fault_injector("rename")
        rename_noreplace(staging, canonical)
        if fault_injector is not None:
            fault_injector("rename_after")
            fault_injector("dir_fsync")
        _fsync_dir(canonical.parent)
        promoted = replace(receipt, state="CANONICAL_PROMOTED", updated_at=_utc_now())
        _write_receipt_version(canonical, promoted, fault_injector=fault_injector)
        return promoted

    def commit_unit(self, receipt: UnitTransactionReceipt, *, fault_injector: FailureInjector | None = None) -> UnitTransactionReceipt:
        current = self.promote_unit(receipt, fault_injector=fault_injector)
        if current.state == "LEDGER_COMMITTED":
            return current
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
        _write_receipt_version(Path(committed.canonical_dir), committed, fault_injector=fault_injector)
        return committed

    def write_checkpoint(self, checkpoint: DurabilityCheckpoint, path: Path) -> None:
        """Persist a checkpoint as an immutable, versioned record.

        A checkpoint path is a logical prefix, not a mutable singleton.
        Repeating the same checkpoint is idempotent; a later checkpoint gets a
        distinct no-clobber filename.
        """

        payload = checkpoint.to_dict()
        digest = str(payload["checkpoint_sha256"])
        path = Path(path)
        versioned = path.with_name(f"{path.name}.{checkpoint.checkpoint_index:08d}.{digest[:16]}.json")
        atomic_write_json(versioned, payload)


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


def _process_is_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        # The process exists but belongs to another user. Never mutate while
        # liveness is uncertain.
        return True
    except OSError:
        return False
    return True


def _read_committed_prefix(path: Path) -> tuple[list[dict[str, Any]], bytes]:
    """Read only complete newline-delimited events and return an opaque tail."""

    if not path.exists():
        return [], b""
    raw = path.read_bytes()
    lines = raw.splitlines(keepends=True)
    complete: list[bytes] = []
    tail = b""
    for line in lines:
        if line.endswith(b"\n"):
            complete.append(line)
        else:
            tail = line
    rows: list[dict[str, Any]] = []
    previous = "0" * 64
    for index, line in enumerate(complete):
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise Attempt05RecoveryError("V4_RECOVERY_LEDGER_CHAIN_INVALID") from exc
        if (
            not isinstance(row, dict)
            or row.get("sequence_index") != index
            or row.get("previous_event_sha256") != previous
            or row.get("event_sha256") != _event_hash(row)
        ):
            raise Attempt05RecoveryError("V4_RECOVERY_LEDGER_CHAIN_INVALID")
        previous = str(row["event_sha256"])
        rows.append(row)
    return rows, tail


def _pending_tail_matches(
    pending: Path,
    rows: Sequence[Mapping[str, object]],
) -> bool:
    expected_sequence = len(rows)
    expected_previous = str(rows[-1]["event_sha256"]) if rows else "0" * 64
    candidates = list(pending.glob("*.partial")) if pending.exists() else []
    if len(candidates) != 1:
        return False
    try:
        payload = json.loads(candidates[0].read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return (
        isinstance(payload, Mapping)
        and payload.get("event_sha256") == _event_hash(payload)
        and payload.get("sequence_index") == expected_sequence
        and payload.get("previous_event_sha256") == expected_previous
    )


@dataclass(frozen=True, slots=True)
class RecoveryDecision:
    """Machine-readable cold-start decision for one unit.

    ``state`` and ``recovery_action`` are closed vocabulary values.  The
    original result mapping remains available for compatibility, while this
    value gives supervisors a typed, serializable decision surface.
    """

    state: str
    recovery_action: str
    idempotency_key: str
    canonical_dir: str
    pending_event_count: int
    reason: str
    worker_pid: int | None = None
    worker_alive: bool | None = None
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.state not in {
            ACTIVE_VALID_DEFER_NO_MUTATION,
            COMPLETE,
            INCOMPLETE_SAFE_TO_RETRY,
            ORPHAN_REQUIRES_QUARANTINE,
            FATAL_IDENTITY_MISMATCH,
        }:
            raise Attempt05RecoveryError("V4_RECOVERY_DECISION_STATE_INVALID")
        if self.recovery_action not in {
            RECOVERY_ACTION_NOOP,
            RECOVERY_ACTION_RESUME_LEDGER_ONLY,
            RECOVERY_ACTION_REINFER_UNIT,
            RECOVERY_ACTION_QUARANTINE,
            RECOVERY_ACTION_ABORT_FATAL,
        }:
            raise Attempt05RecoveryError("V4_RECOVERY_DECISION_ACTION_INVALID")
        _require_text(self.idempotency_key, "idempotency_key")
        _require_text(self.canonical_dir, "canonical_dir")
        _require_text(self.reason, "reason")
        if type(self.pending_event_count) is not int or self.pending_event_count < 0:
            raise Attempt05RecoveryError("V4_RECOVERY_DECISION_PENDING_COUNT_INVALID")
        if self.worker_pid is not None and (type(self.worker_pid) is not int or self.worker_pid <= 0):
            raise Attempt05RecoveryError("V4_RECOVERY_DECISION_WORKER_PID_INVALID")
        if self.worker_alive is not None and type(self.worker_alive) is not bool:
            raise Attempt05RecoveryError("V4_RECOVERY_DECISION_WORKER_LIVENESS_INVALID")

    def to_dict(self) -> dict[str, object]:
        return {
            "state": self.state,
            "recovery_action": self.recovery_action,
            "idempotency_key": self.idempotency_key,
            "canonical_dir": self.canonical_dir,
            "pending_event_count": self.pending_event_count,
            "reason": self.reason,
            "worker_pid": self.worker_pid,
            "worker_alive": self.worker_alive,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> "RecoveryDecision":
        if not isinstance(value, Mapping):
            raise Attempt05RecoveryError("V4_RECOVERY_DECISION_INVALID")
        metadata = value.get("metadata", {})
        if not isinstance(metadata, Mapping):
            raise Attempt05RecoveryError("V4_RECOVERY_DECISION_METADATA_INVALID")
        return cls(
            state=_require_text(value.get("state"), "state"),
            recovery_action=_require_text(value.get("recovery_action"), "recovery_action"),
            idempotency_key=_require_text(value.get("idempotency_key"), "idempotency_key"),
            canonical_dir=_require_text(value.get("canonical_dir"), "canonical_dir"),
            pending_event_count=value.get("pending_event_count"),  # type: ignore[arg-type]
            reason=_require_text(value.get("reason"), "reason"),
            worker_pid=value.get("worker_pid"),  # type: ignore[arg-type]
            worker_alive=value.get("worker_alive"),  # type: ignore[arg-type]
            metadata=dict(metadata),
        )

    @classmethod
    def from_reconciliation(cls, value: Mapping[str, object]) -> "RecoveryDecision":
        """Parse a complete ``reconcile_unit_transaction`` result."""

        nested = value.get("recovery_decision")
        if isinstance(nested, Mapping):
            return cls.from_mapping(nested)
        return cls.from_mapping(value)


def _reconciliation_result(
    *,
    state: str,
    action: str,
    idempotency_key: str,
    canonical: Path,
    pending_count: int,
    reason: str,
    **extra: object,
) -> dict[str, object]:
    result: dict[str, object] = {
        "state": state,
        "recovery_action": action,
        "idempotency_key": idempotency_key,
        "canonical_dir": str(canonical),
        "pending_event_count": pending_count,
        "reason": reason,
    }
    result.update(extra)
    decision_metadata = dict(extra)
    decision_metadata.pop("worker_pid", None)
    decision_metadata.pop("worker_alive", None)
    decision = RecoveryDecision(
        state=state,
        recovery_action=action,
        idempotency_key=idempotency_key,
        canonical_dir=str(canonical),
        pending_event_count=pending_count,
        reason=reason,
        worker_pid=extra.get("worker_pid"),  # type: ignore[arg-type]
        worker_alive=extra.get("worker_alive"),  # type: ignore[arg-type]
        metadata=decision_metadata,
    )
    result["recovery_decision"] = decision.to_dict()
    return result


def reconcile_unit_transaction(
    *,
    canonical_dir: Path,
    ledger_path: Path,
    idempotency_key: str,
    pending_root: Path | None = None,
    worker_pid: int | None = None,
    worker_alive: bool | None = None,
    heartbeat_path: Path | None = None,
    heartbeat_max_age_seconds: float | None = None,
    now_epoch: float | None = None,
) -> dict[str, object]:
    """Classify a unit without inference, killing workers, or rewriting history.

    A stale heartbeat is not enough to mutate state: when a matching worker is
    alive (or liveness cannot be disproved) this returns
    ``ACTIVE_VALID_DEFER_NO_MUTATION``. Torn tails are safe to continue only
    when the complete prefix and exactly one matching pending event are both
    present; otherwise they are quarantined.
    """

    canonical = Path(canonical_dir)
    staging = canonical.with_name(canonical.name + ".partial")
    pending = Path(pending_root) if pending_root is not None else Path(ledger_path).with_name(Path(ledger_path).name + ".pending")
    pending_count = len(list(pending.glob("*.partial"))) if pending.exists() else 0

    live = worker_alive
    if live is None and worker_pid is not None:
        live = _process_is_alive(worker_pid)
    if live:
        return _reconciliation_result(
            state=ACTIVE_VALID_DEFER_NO_MUTATION,
            action=RECOVERY_ACTION_NOOP,
            idempotency_key=idempotency_key,
            canonical=canonical,
            pending_count=pending_count,
            reason="MATCHING_WORKER_ACTIVE_OR_HEARTBEAT_LIVENESS_UNCERTAIN",
            worker_pid=worker_pid,
            worker_alive=True,
            heartbeat_path=None if heartbeat_path is None else str(heartbeat_path),
        )

    try:
        rows, tail = _read_committed_prefix(Path(ledger_path))
        if tail:
            if _pending_tail_matches(pending, rows):
                return _reconciliation_result(
                    state=INCOMPLETE_SAFE_TO_RETRY,
                    action=RECOVERY_ACTION_RESUME_LEDGER_ONLY,
                    idempotency_key=idempotency_key,
                    canonical=canonical,
                    pending_count=pending_count,
                    reason="TORN_TAIL_WITH_EXACT_NEXT_PENDING_EVENT",
                    committed_prefix_length=len(rows),
                    torn_tail_sha256=sha256_bytes(tail),
                )
            return _reconciliation_result(
                state=ORPHAN_REQUIRES_QUARANTINE,
                action=RECOVERY_ACTION_QUARANTINE,
                idempotency_key=idempotency_key,
                canonical=canonical,
                pending_count=pending_count,
                reason="TORN_TAIL_WITHOUT_EXACT_NEXT_PENDING_EVENT",
                committed_prefix_length=len(rows),
                torn_tail_sha256=sha256_bytes(tail),
            )
    except Attempt05RecoveryError as exc:
        return _reconciliation_result(
            state=FATAL_IDENTITY_MISMATCH,
            action=RECOVERY_ACTION_ABORT_FATAL,
            idempotency_key=idempotency_key,
            canonical=canonical,
            pending_count=pending_count,
            reason=str(exc),
        )

    if not canonical.exists():
        if not staging.exists():
            return _reconciliation_result(
                state=INCOMPLETE_SAFE_TO_RETRY,
                action=RECOVERY_ACTION_REINFER_UNIT,
                idempotency_key=idempotency_key,
                canonical=canonical,
                pending_count=pending_count,
                reason="NO_CANONICAL_OR_STAGING",
            )
        staging_receipt = staging / "unit_transaction_receipt.json"
        if not staging_receipt.is_file():
            return _reconciliation_result(
                state=ORPHAN_REQUIRES_QUARANTINE,
                action=RECOVERY_ACTION_QUARANTINE,
                idempotency_key=idempotency_key,
                canonical=canonical,
                pending_count=pending_count,
                reason="PREPARED_TREE_RECEIPT_MISSING",
            )
        try:
            receipt = UnitTransactionReceipt.from_mapping(json.loads(staging_receipt.read_text(encoding="utf-8")))
        except (OSError, json.JSONDecodeError, Attempt05RecoveryError) as exc:
            return _reconciliation_result(
                state=ORPHAN_REQUIRES_QUARANTINE,
                action=RECOVERY_ACTION_QUARANTINE,
                idempotency_key=idempotency_key,
                canonical=canonical,
                pending_count=pending_count,
                reason="PREPARED_RECEIPT_INVALID",
                detail=str(exc),
            )
        if receipt.idempotency_key != idempotency_key:
            return _reconciliation_result(
                state=FATAL_IDENTITY_MISMATCH,
                action=RECOVERY_ACTION_ABORT_FATAL,
                idempotency_key=idempotency_key,
                canonical=canonical,
                pending_count=pending_count,
                reason="IDEMPOTENCY_KEY_MISMATCH",
            )
        return _reconciliation_result(
            state=INCOMPLETE_SAFE_TO_RETRY,
            action=RECOVERY_ACTION_RESUME_LEDGER_ONLY,
            idempotency_key=idempotency_key,
            canonical=canonical,
            pending_count=pending_count,
            reason="PREPARED_TREE_CAN_BE_PROMOTED_WITHOUT_INFERENCE",
            receipt_state=receipt.state,
        )

    try:
        receipt = _load_latest_receipt(canonical)
    except Attempt05RecoveryError as exc:
        reason = "CANONICAL_RECEIPT_MISSING" if "MISSING" in str(exc) else "CANONICAL_RECEIPT_INVALID"
        return _reconciliation_result(
            state=ORPHAN_REQUIRES_QUARANTINE,
            action=RECOVERY_ACTION_QUARANTINE,
            idempotency_key=idempotency_key,
            canonical=canonical,
            pending_count=pending_count,
            reason=reason,
            detail=str(exc),
        )
    if receipt.idempotency_key != idempotency_key:
        return _reconciliation_result(
            state=FATAL_IDENTITY_MISMATCH,
            action=RECOVERY_ACTION_ABORT_FATAL,
            idempotency_key=idempotency_key,
            canonical=canonical,
            pending_count=pending_count,
            reason="IDEMPOTENCY_KEY_MISMATCH",
        )
    try:
        receipt_path_value = Path(receipt.canonical_dir).resolve()
        canonical_path_value = canonical.resolve()
    except OSError:
        receipt_path_value = Path(receipt.canonical_dir)
        canonical_path_value = canonical
    if receipt_path_value != canonical_path_value:
        return _reconciliation_result(
            state=FATAL_IDENTITY_MISMATCH,
            action=RECOVERY_ACTION_ABORT_FATAL,
            idempotency_key=idempotency_key,
            canonical=canonical,
            pending_count=pending_count,
            reason="CANONICAL_PATH_MISMATCH",
        )
    for name, expected_sha256 in receipt.files.items():
        if Path(name).name != name:
            return _reconciliation_result(
                state=FATAL_IDENTITY_MISMATCH,
                action=RECOVERY_ACTION_ABORT_FATAL,
                idempotency_key=idempotency_key,
                canonical=canonical,
                pending_count=pending_count,
                reason="FILE_NAME_INVALID",
                file=name,
            )
        candidate = canonical / name
        try:
            actual = sha256_file(candidate) if candidate.is_file() else None
        except OSError:
            actual = None
        if actual != expected_sha256:
            return _reconciliation_result(
                state=FATAL_IDENTITY_MISMATCH,
                action=RECOVERY_ACTION_ABORT_FATAL,
                idempotency_key=idempotency_key,
                canonical=canonical,
                pending_count=pending_count,
                reason="CANONICAL_HASH_MISMATCH",
                file=name,
            )

    if receipt.state != "LEDGER_COMMITTED":
        return _reconciliation_result(
            state=INCOMPLETE_SAFE_TO_RETRY,
            action=RECOVERY_ACTION_RESUME_LEDGER_ONLY,
            idempotency_key=idempotency_key,
            canonical=canonical,
            pending_count=pending_count,
            reason="CANONICAL_PRESENT_LEDGER_PENDING",
            receipt_state=receipt.state,
            unit_key=receipt.unit_key,
        )

    by_sequence = {int(row["sequence_index"]): row for row in rows}
    for sequence in receipt.ledger_sequences:
        row = by_sequence.get(sequence)
        if row is None:
            return _reconciliation_result(
                state=INCOMPLETE_SAFE_TO_RETRY,
                action=RECOVERY_ACTION_RESUME_LEDGER_ONLY,
                idempotency_key=idempotency_key,
                canonical=canonical,
                pending_count=pending_count,
                reason="LEDGER_SEQUENCE_MISSING",
                receipt_state=receipt.state,
                unit_key=receipt.unit_key,
            )
        payload = row.get("payload")
        transaction = payload.get("_transaction_idempotency_key") if isinstance(payload, Mapping) else None
        if not isinstance(transaction, str) or not transaction.startswith(idempotency_key + ":"):
            return _reconciliation_result(
                state=FATAL_IDENTITY_MISMATCH,
                action=RECOVERY_ACTION_ABORT_FATAL,
                idempotency_key=idempotency_key,
                canonical=canonical,
                pending_count=pending_count,
                reason="LEDGER_IDEMPOTENCY_MISMATCH",
                sequence=sequence,
            )
    return _reconciliation_result(
        state=COMPLETE,
        action=RECOVERY_ACTION_NOOP,
        idempotency_key=idempotency_key,
        canonical=canonical,
        pending_count=pending_count,
        reason="CANONICAL_AND_LEDGER_COMMITTED",
        receipt_state=receipt.state,
        unit_key=receipt.unit_key,
    )


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
    schedule_identity_sha256: str | None = None
    schedule_identity_manifest: Mapping[str, object] | None = None
    scientific_result: str = NO_SCIENTIFIC_RESULT
    generated_at: str = field(default_factory=_utc_now)
    schema_version: str = ELIGIBILITY_SCHEMA

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "attempt_id": self.attempt_id,
            "schedule_sha256": self.schedule_sha256,
            "expected_schedule_count": self.expected_schedule_count,
            "schedule_identity_sha256": self.schedule_identity_sha256,
            "schedule_identity_manifest": (
                None if self.schedule_identity_manifest is None else dict(self.schedule_identity_manifest)
            ),
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
    schedule_identity_manifest: ScheduleIdentityManifest | Mapping[str, object] | None = None,
    expected_schedule_identity_sha256: str | None = None,
    expected_schedule_semantic_sha256: str | None = None,
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
        "schedule_raw_hash_match": False,
        "schedule_semantic_hash_match": False,
        "schedule_identity_hash_match": False,
        "legacy_hash_domain_unresolved": False,
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
        # Both journal projections must carry the same canonical path and
        # digest. Accepting one side as a fallback would allow an incomplete
        # receipt to masquerade as a verified unit.
        if (
            projection_path is None
            or projection_sha is None
            or completion_path is None
            or completion_sha is None
        ):
            checks["reconciliation_failures"]["path_or_sha_missing"] += 1  # type: ignore[index]
            continue
        if projection_sha != completion_sha or projection_path != completion_path:
            checks["reconciliation_failures"]["path_or_sha_missing"] += 1  # type: ignore[index]
            continue
        path_value = projection_path
        digest_value = projection_sha
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
    manifest_obj: ScheduleIdentityManifest | None = None
    if schedule_identity_manifest is not None:
        try:
            manifest_obj = (
                schedule_identity_manifest
                if isinstance(schedule_identity_manifest, ScheduleIdentityManifest)
                else ScheduleIdentityManifest.from_mapping(schedule_identity_manifest)
            )
            checks["schedule_raw_hash_match"] = manifest_obj.raw_sha256 == schedule_sha256
            checks["schedule_semantic_hash_match"] = (
                expected_schedule_semantic_sha256 is None
                or manifest_obj.semantic_sha256 == expected_schedule_semantic_sha256
            )
            checks["schedule_identity_hash_match"] = (
                expected_schedule_identity_sha256 is None
                or manifest_obj.schedule_identity_sha256 == expected_schedule_identity_sha256
            )
            checks["schedule_manifest_count_match"] = manifest_obj.unit_count == len(normalized_schedule)
            checks["schedule_manifest_order_match"] = (
                manifest_obj.ordered_unit_ids_sha256 == _domain_hash(ORDERED_UNIT_IDS_HASH_DOMAIN, normalized_schedule)
            )
            checks["schedule_hash_match"] = bool(
                checks["schedule_raw_hash_match"]
                and checks["schedule_semantic_hash_match"]
                and checks["schedule_identity_hash_match"]
                and checks["schedule_manifest_count_match"]
                and checks["schedule_manifest_order_match"]
            )
        except Attempt05RecoveryError as exc:
            checks["schedule_identity_error"] = str(exc)
    else:
        # Legacy callers provide a single undifferentiated schedule hash.
        # Equality is accepted for compatibility; disagreement is explicitly
        # unresolved instead of being described as semantic drift.
        legacy_values = [schedule_sha256]
        if expected_schedule_sha256 is not None:
            legacy_values.append(expected_schedule_sha256)
        if input_closure_schedule_sha256 is not None:
            legacy_values.append(input_closure_schedule_sha256)
        checks["schedule_hash_match"] = len(set(legacy_values)) == 1
        checks["schedule_raw_hash_match"] = checks["schedule_hash_match"]
        checks["schedule_semantic_hash_match"] = checks["schedule_hash_match"]
        if not checks["schedule_hash_match"]:
            checks["legacy_hash_domain_unresolved"] = True
    checks["schedule_identity_manifest_present"] = manifest_obj is not None
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
        schedule_identity_sha256=None if manifest_obj is None else manifest_obj.schedule_identity_sha256,
        schedule_identity_manifest=None if manifest_obj is None else manifest_obj.to_dict(),
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


@dataclass(frozen=True, slots=True)
class SameAttemptSessionUnionManifest:
    """Read-only union of recovery sessions from one Attempt-06 only.

    Session identifiers are provenance labels and never participate in the
    unit idempotency key.  Cross-Attempt and historical Attempt-05 sources are
    rejected before any union is materialized.
    """

    attempt_id: str
    schedule_identity_sha256: str
    expected_count: int
    sessions: Mapping[str, tuple[str, ...]]
    unit_keys: tuple[str, ...]
    scientific_result: str = NO_SCIENTIFIC_RESULT
    schema_version: str = SAME_ATTEMPT_UNION_SCHEMA

    def __post_init__(self) -> None:
        if not self.attempt_id or self.attempt_id.lower() in {"attempt-05", "attempt05"} or "attempt05" in self.attempt_id.lower():
            raise Attempt05RecoveryError("V4_RECOVERY_ATTEMPT05_SOURCE_FORBIDDEN")
        if not isinstance(self.schedule_identity_sha256, str) or len(self.schedule_identity_sha256) != 64:
            raise Attempt05RecoveryError("V4_RECOVERY_SCHEDULE_IDENTITY_INVALID")
        if type(self.expected_count) is not int or self.expected_count <= 0:
            raise Attempt05RecoveryError("V4_RECOVERY_UNION_COUNT_INVALID")
        if len(self.unit_keys) != self.expected_count or len(set(self.unit_keys)) != len(self.unit_keys):
            raise Attempt05RecoveryError("V4_RECOVERY_UNION_NOT_EXACT")
        if self.scientific_result != NO_SCIENTIFIC_RESULT:
            raise Attempt05RecoveryError("V4_RECOVERY_SCIENTIFIC_RESULT_FORBIDDEN")

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "attempt_id": self.attempt_id,
            "schedule_identity_sha256": self.schedule_identity_sha256,
            "expected_count": self.expected_count,
            "sessions": {name: list(values) for name, values in self.sessions.items()},
            "unit_keys": list(self.unit_keys),
            "scientific_result": self.scientific_result,
        }

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> "SameAttemptSessionUnionManifest":
        if not isinstance(value, Mapping):
            raise Attempt05RecoveryError("V4_RECOVERY_UNION_INVALID")
        raw_sessions = value.get("sessions", {})
        raw_units = value.get("unit_keys", ())
        if not isinstance(raw_sessions, Mapping) or not isinstance(raw_units, Sequence) or isinstance(raw_units, (str, bytes, bytearray)):
            raise Attempt05RecoveryError("V4_RECOVERY_UNION_INVALID")
        sessions: dict[str, tuple[str, ...]] = {}
        for name, values in raw_sessions.items():
            if not isinstance(name, str) or not name or not isinstance(values, Sequence) or isinstance(values, (str, bytes, bytearray)):
                raise Attempt05RecoveryError("V4_RECOVERY_UNION_SESSION_INVALID")
            sessions[name] = tuple(_normalize_unit_key(item) for item in values)
        return cls(
            attempt_id=_require_text(value.get("attempt_id"), "attempt_id"),
            schedule_identity_sha256=_require_text(value.get("schedule_identity_sha256"), "schedule_identity_sha256"),
            expected_count=value.get("expected_count"),  # type: ignore[arg-type]
            sessions=sessions,
            unit_keys=tuple(_normalize_unit_key(item) for item in raw_units),
            scientific_result=value.get("scientific_result", NO_SCIENTIFIC_RESULT),  # type: ignore[arg-type]
            schema_version=_require_text(value.get("schema_version", SAME_ATTEMPT_UNION_SCHEMA), "schema_version"),
        )


def build_same_attempt_session_union(
    *,
    attempt_id: str,
    schedule_identity_sha256: str,
    session_units: Mapping[str, Sequence[object]],
    expected_schedule_keys: Sequence[object] | None = None,
    expected_count: int | None = None,
    source_attempt_ids: Mapping[str, str] | None = None,
) -> SameAttemptSessionUnionManifest:
    """Build an exact union across sessions of one attempt, never across attempts."""

    if not isinstance(session_units, Mapping) or not session_units:
        raise Attempt05RecoveryError("V4_RECOVERY_UNION_SESSIONS_EMPTY")
    if not isinstance(attempt_id, str) or not attempt_id or "attempt05" in attempt_id.lower():
        raise Attempt05RecoveryError("V4_RECOVERY_ATTEMPT05_SOURCE_FORBIDDEN")
    if source_attempt_ids is not None:
        for session, source in source_attempt_ids.items():
            if source != attempt_id:
                raise Attempt05RecoveryError("V4_RECOVERY_CROSS_ATTEMPT_UNION_FORBIDDEN")
            if session not in session_units:
                raise Attempt05RecoveryError("V4_RECOVERY_UNION_SESSION_INVALID")
    sessions: dict[str, tuple[str, ...]] = {}
    flattened: list[str] = []
    for session, values in session_units.items():
        if not isinstance(session, str) or not session or not isinstance(values, Sequence) or isinstance(values, (str, bytes, bytearray)):
            raise Attempt05RecoveryError("V4_RECOVERY_UNION_SESSION_INVALID")
        normalized = tuple(_normalize_unit_key(item) for item in values)
        if len(normalized) != len(set(normalized)):
            raise Attempt05RecoveryError("V4_RECOVERY_UNION_DUPLICATE_IDENTITY")
        sessions[session] = normalized
        flattened.extend(normalized)
    if len(flattened) != len(set(flattened)):
        raise Attempt05RecoveryError("V4_RECOVERY_UNION_DUPLICATE_IDENTITY")
    expected = None if expected_schedule_keys is None else tuple(_normalize_unit_key(item) for item in expected_schedule_keys)
    if expected is not None:
        if len(expected) != len(set(expected)) or set(flattened) != set(expected):
            raise Attempt05RecoveryError("V4_RECOVERY_UNION_NOT_EXACT")
        units = expected
    else:
        units = tuple(sorted(flattened))
    count = len(units) if expected_count is None else expected_count
    if type(count) is not int or count != len(units):
        raise Attempt05RecoveryError("V4_RECOVERY_UNION_COUNT_INVALID")
    return SameAttemptSessionUnionManifest(
        attempt_id=attempt_id,
        schedule_identity_sha256=schedule_identity_sha256,
        expected_count=count,
        sessions=sessions,
        unit_keys=units,
    )


@dataclass(frozen=True, slots=True)
class RecoverySmokeManifest:
    """Deterministic non-scientific 12-unit recovery qualification plan."""

    attempt_id: str
    schedule_identity_sha256: str
    selector_version: str
    scene_ids: tuple[int, ...]
    unit_keys: tuple[str, ...]
    gpu_uuid: str
    physical_gpu_index: int
    interruption_plan: Mapping[str, str]
    expected_inference_starts: Mapping[str, int]
    scientific_result: str = NO_SCIENTIFIC_RESULT
    schema_version: str = RECOVERY_SMOKE_SCHEMA

    def __post_init__(self) -> None:
        if self.attempt_id != RECOVERY_SMOKE_ATTEMPT_ID:
            raise Attempt05RecoveryError("V4_RECOVERY_SMOKE_ATTEMPT_ID_INVALID")
        if not isinstance(self.schedule_identity_sha256, str) or len(self.schedule_identity_sha256) != 64:
            raise Attempt05RecoveryError("V4_RECOVERY_SCHEDULE_IDENTITY_INVALID")
        if len(self.scene_ids) != 6 or len(set(self.scene_ids)) != 6:
            raise Attempt05RecoveryError("V4_RECOVERY_SMOKE_SCENE_COUNT_INVALID")
        if len(self.unit_keys) != 12 or len(set(self.unit_keys)) != 12:
            raise Attempt05RecoveryError("V4_RECOVERY_SMOKE_UNIT_COUNT_INVALID")
        if self.gpu_uuid != AUTHORIZED_GPU_UUID:
            raise Attempt05RecoveryError("V4_RECOVERY_SMOKE_GPU_UUID_INVALID")
        if self.physical_gpu_index != AUTHORIZED_PHYSICAL_GPU_INDEX:
            raise Attempt05RecoveryError("V4_RECOVERY_SMOKE_GPU_INDEX_INVALID")
        if set(self.interruption_plan) != {self.unit_keys[3], self.unit_keys[7], self.unit_keys[10]}:
            raise Attempt05RecoveryError("V4_RECOVERY_SMOKE_INTERRUPTION_PLAN_INVALID")
        if any(value not in {
            "canonical_promotion_before_projection",
            "projection_before_completion",
            "inference_before_prepared_receipt",
        } for value in self.interruption_plan.values()):
            raise Attempt05RecoveryError("V4_RECOVERY_SMOKE_INTERRUPTION_PHASE_INVALID")
        if set(self.expected_inference_starts) != set(self.unit_keys):
            raise Attempt05RecoveryError("V4_RECOVERY_SMOKE_EXPECTATION_INVALID")
        if any(type(value) is not int or value not in {1, 2} for value in self.expected_inference_starts.values()):
            raise Attempt05RecoveryError("V4_RECOVERY_SMOKE_EXPECTATION_INVALID")
        if self.expected_inference_starts[self.unit_keys[10]] != 2:
            raise Attempt05RecoveryError("V4_RECOVERY_SMOKE_UNIT11_EXPECTATION_INVALID")
        if any(self.expected_inference_starts[key] != 1 for key in self.unit_keys if key != self.unit_keys[10]):
            raise Attempt05RecoveryError("V4_RECOVERY_SMOKE_EXPECTATION_INVALID")
        if self.scientific_result != NO_SCIENTIFIC_RESULT:
            raise Attempt05RecoveryError("V4_RECOVERY_SCIENTIFIC_RESULT_FORBIDDEN")

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "attempt_id": self.attempt_id,
            "schedule_identity_sha256": self.schedule_identity_sha256,
            "selector_version": self.selector_version,
            "scene_ids": list(self.scene_ids),
            "unit_keys": list(self.unit_keys),
            "gpu_uuid": self.gpu_uuid,
            "physical_gpu_index": self.physical_gpu_index,
            "interruption_plan": dict(self.interruption_plan),
            "expected_inference_starts": dict(self.expected_inference_starts),
            "scientific_result": self.scientific_result,
        }

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> "RecoverySmokeManifest":
        if not isinstance(value, Mapping):
            raise Attempt05RecoveryError("V4_RECOVERY_SMOKE_MANIFEST_INVALID")
        scenes = value.get("scene_ids")
        units = value.get("unit_keys")
        plan = value.get("interruption_plan")
        expected = value.get("expected_inference_starts")
        if not all(isinstance(item, Sequence) and not isinstance(item, (str, bytes, bytearray)) for item in (scenes, units)):
            raise Attempt05RecoveryError("V4_RECOVERY_SMOKE_MANIFEST_INVALID")
        if not isinstance(plan, Mapping) or not isinstance(expected, Mapping):
            raise Attempt05RecoveryError("V4_RECOVERY_SMOKE_MANIFEST_INVALID")
        return cls(
            attempt_id=_require_text(value.get("attempt_id"), "attempt_id"),
            schedule_identity_sha256=_require_text(value.get("schedule_identity_sha256"), "schedule_identity_sha256"),
            selector_version=_require_text(value.get("selector_version"), "selector_version"),
            scene_ids=tuple(int(item) for item in scenes),
            unit_keys=tuple(_normalize_unit_key(item) for item in units),
            gpu_uuid=_require_text(value.get("gpu_uuid"), "gpu_uuid"),
            physical_gpu_index=value.get("physical_gpu_index"),  # type: ignore[arg-type]
            interruption_plan={str(key): str(item) for key, item in plan.items()},
            expected_inference_starts={str(key): int(item) for key, item in expected.items()},
            scientific_result=value.get("scientific_result", NO_SCIENTIFIC_RESULT),  # type: ignore[arg-type]
            schema_version=_require_text(value.get("schema_version", RECOVERY_SMOKE_SCHEMA), "schema_version"),
        )


def select_recovery_smoke_scene_ids(
    support_scene_ids: Sequence[int],
    schedule_identity_sha256: str,
    *,
    selector_version: str = "georeliab:v4-recovery-smoke-selector:v1",
    count: int = 6,
) -> tuple[int, ...]:
    """Select six scenes deterministically without reading outcomes."""

    if type(count) is not int or count != 6:
        raise Attempt05RecoveryError("V4_RECOVERY_SMOKE_SCENE_COUNT_INVALID")
    if not isinstance(schedule_identity_sha256, str) or len(schedule_identity_sha256) != 64:
        raise Attempt05RecoveryError("V4_RECOVERY_SCHEDULE_IDENTITY_INVALID")
    unique = tuple(dict.fromkeys(int(item) for item in support_scene_ids))
    if len(unique) < count:
        raise Attempt05RecoveryError("V4_RECOVERY_SMOKE_SUPPORT_SCENES_INSUFFICIENT")
    ordered = sorted(
        unique,
        key=lambda scene: hashlib.sha256(
            f"{selector_version}\0{schedule_identity_sha256}\0{scene}".encode("utf-8")
        ).hexdigest(),
    )
    return tuple(sorted(ordered[:count]))


def build_recovery_smoke_manifest(
    *,
    schedule_identity_sha256: str,
    support_scene_ids: Sequence[int],
    model_ids: Sequence[str] = ("VGGT", "MASt3R"),
    gpu_uuid: str = AUTHORIZED_GPU_UUID,
    physical_gpu_index: int = AUTHORIZED_PHYSICAL_GPU_INDEX,
    selector_version: str = "georeliab:v4-recovery-smoke-selector:v1",
) -> RecoverySmokeManifest:
    scenes = select_recovery_smoke_scene_ids(
        support_scene_ids, schedule_identity_sha256, selector_version=selector_version
    )
    models = tuple(str(model) for model in model_ids)
    if len(models) != 2 or len(set(models)) != 2:
        raise Attempt05RecoveryError("V4_RECOVERY_SMOKE_MODEL_COUNT_INVALID")
    units = tuple(f"{model}|{scene}|L3" for scene in scenes for model in models)
    plan = {
        units[3]: "canonical_promotion_before_projection",
        units[7]: "projection_before_completion",
        units[10]: "inference_before_prepared_receipt",
    }
    expected = {key: 1 for key in units}
    expected[units[10]] = 2
    return RecoverySmokeManifest(
        attempt_id=RECOVERY_SMOKE_ATTEMPT_ID,
        schedule_identity_sha256=schedule_identity_sha256,
        selector_version=selector_version,
        scene_ids=scenes,
        unit_keys=units,
        gpu_uuid=gpu_uuid,
        physical_gpu_index=physical_gpu_index,
        interruption_plan=plan,
        expected_inference_starts=expected,
    )


def evaluate_recovery_smoke(
    manifest: RecoverySmokeManifest | Mapping[str, object],
    observations: Sequence[Mapping[str, object]] | None = None,
    *,
    records: Sequence[Mapping[str, object]] | None = None,
) -> dict[str, object]:
    """Evaluate exact-once smoke receipts without scientific metrics."""

    smoke = manifest if isinstance(manifest, RecoverySmokeManifest) else RecoverySmokeManifest.from_mapping(manifest)
    rows = tuple(observations if observations is not None else (records or ()))
    by_key: dict[str, Mapping[str, object]] = {}
    invalid: list[str] = []
    duplicate: list[str] = []
    for row in rows:
        if not isinstance(row, Mapping):
            invalid.append("non_mapping")
            continue
        key = row.get("unit_key")
        if not isinstance(key, str) or key not in smoke.unit_keys:
            invalid.append(str(key))
            continue
        if key in by_key:
            duplicate.append(key)
        by_key[key] = row
    missing = [key for key in smoke.unit_keys if key not in by_key]
    inference_counts: dict[str, int] = {}
    mismatches: list[str] = []
    gpu_violations: list[str] = []
    closure_violations: list[str] = []
    scientific_markers: list[str] = []
    for key, row in by_key.items():
        events = row.get("events")
        if isinstance(events, Sequence) and not isinstance(events, (str, bytes, bytearray)):
            count = sum(1 for event in events if str(event) == "inference_start")
        else:
            count = row.get("inference_start_count")
        # Qualification must fail closed when the receipt does not prove the
        # expected inference/recovery trace.  Defaults would turn an omitted
        # field into an accidental pass.
        if type(count) is not int:
            mismatches.append(key)
        else:
            inference_counts[key] = count
            if count != smoke.expected_inference_starts[key]:
                mismatches.append(key)
        if row.get("completion_count") != 1 or row.get("overwrite_count") != 0:
            mismatches.append(key)
        if row.get("gpu_uuid") != smoke.gpu_uuid or row.get("physical_gpu_index") != smoke.physical_gpu_index:
            gpu_violations.append(key)
        # A smoke completion is not qualified until both sides of the
        # canonical/ledger closure are explicitly attested.  These booleans
        # are receipt facts, not scientific metrics.
        if row.get("canonical_present") is not True or row.get("ledger_committed") is not True:
            closure_violations.append(key)
        marker = row.get("scientific_marker")
        if marker not in (None, "", NO_SCIENTIFIC_RESULT):
            scientific_markers.append(str(marker))
    passed = not (
        missing
        or invalid
        or duplicate
        or mismatches
        or gpu_violations
        or closure_violations
        or scientific_markers
    ) and len(by_key) == 12
    return {
        "schema_version": RECOVERY_SMOKE_SCHEMA,
        "attempt_id": smoke.attempt_id,
        "status": "V4_RECOVERY_RUNTIME_QUALIFIED" if passed else "V4_RECOVERY_RUNTIME_NOT_QUALIFIED",
        "unit_count": len(by_key),
        "expected_unit_count": 12,
        "missing_unit_keys": missing,
        "invalid_unit_keys": invalid,
        "duplicate_unit_keys": duplicate,
        "inference_start_counts": inference_counts,
        "inference_count_mismatches": mismatches,
        "gpu_violations": gpu_violations,
        "closure_violations": closure_violations,
        "scientific_markers": scientific_markers,
        "scientific_result": NO_SCIENTIFIC_RESULT,
    }


qualify_recovery_smoke = evaluate_recovery_smoke


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
    eligibility: ResumeEligibilityManifest | None = None,
    root_cause_resolved: bool = False,
    fault_injection_passed: bool = False,
    gpu_smoke_passed: bool = False,
    explicit_authorization: bool = False,
    missing_schedule_keys: Sequence[object] | None = None,
    expected_total_count: int | None = None,
    expected_missing_count: int | None = None,
    # Fresh-source Gate 4 controls.  These are intentionally explicit so a
    # historical Attempt-05 eligibility manifest can never authorize a new
    # run by accident.
    fresh_source_confirmed: bool | None = None,
    fresh_source: bool | None = None,
    attempt05_source_rejected: bool | None = None,
    recovery_runtime_qualified: bool | None = None,
    power_gate_passed: bool | None = None,
    pilot_gate_passed: bool | None = None,
    pilot_status: str | None = None,
    budget_gate_passed: bool | None = None,
    fresh_schedule_closed: bool | None = None,
) -> dict[str, object]:
    """Return a gate decision; this function never launches an execution."""

    # Gate 4 is always a fresh-source materialization. Historical Attempt-05
    # eligibility and missing-unit closure arguments remain accepted for
    # read-only compatibility, but can never make a launch eligible.
    if fresh_source_confirmed is None:
        fresh_source_confirmed = fresh_source
    if pilot_gate_passed is None and pilot_status is not None:
        pilot_gate_passed = pilot_status == "V4_PILOT_GO_TO_FULL_MVE"
    checks = {
        "fresh_source_confirmed": fresh_source_confirmed is True,
        "attempt05_source_rejected": attempt05_source_rejected is True,
        "recovery_runtime_qualified": recovery_runtime_qualified is True,
        "power_gate_passed": power_gate_passed is True,
        "pilot_gate_passed": pilot_gate_passed is True,
        "budget_gate_passed": budget_gate_passed is True,
        "fresh_schedule_closed": fresh_schedule_closed is True,
        "explicit_authorization": explicit_authorization,
        # Explicitly report historical arguments as non-authorizing evidence.
        "historical_partial_ignored": eligibility is None or eligibility.verdict != "V4_ATTEMPT05_PARTIAL_CORPUS_RESUME_ELIGIBLE",
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
        # Heartbeats are an append-only JSONL journal.  This keeps the fixed
        # path observable by stale-heartbeat monitors without replacing a
        # prior record or relying on an unsafe overwrite primitive.
        self.heartbeat_path.parent.mkdir(parents=True, exist_ok=True)
        line = _canonical_json_bytes(body)
        with self.heartbeat_path.open("ab") as handle:
            handle.write(line)
            handle.flush()
            os.fsync(handle.fileno())
        _fsync_dir(self.heartbeat_path.parent)

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

def build_ledger_chain_bridge(
    *,
    source_ledger_path: Path,
    committed_prefix_length: int,
    committed_prefix_sha256: str,
    previous_event_sha256: str,
    torn_tail_sha256: str | None,
    source_ledger_sha256: str,
) -> dict[str, object]:
    """Describe how a future segment continues an immutable ledger prefix.

    The bridge is metadata only: it never truncates or rewrites the source ledger.
    """
    if type(committed_prefix_length) is not int or committed_prefix_length < 0:
        raise Attempt05RecoveryError("V4_RECOVERY_CHAIN_BRIDGE_PREFIX_INVALID")
    for value, label in (
        (committed_prefix_sha256, "committed_prefix_sha256"),
        (previous_event_sha256, "previous_event_sha256"),
        (source_ledger_sha256, "source_ledger_sha256"),
    ):
        if not isinstance(value, str) or len(value) != 64 or any(char not in "0123456789abcdef" for char in value.lower()):
            raise Attempt05RecoveryError(f"V4_RECOVERY_CHAIN_BRIDGE_{label.upper()}_INVALID")
    if torn_tail_sha256 is not None and (
        not isinstance(torn_tail_sha256, str)
        or len(torn_tail_sha256) != 64
        or any(char not in "0123456789abcdef" for char in torn_tail_sha256.lower())
    ):
        raise Attempt05RecoveryError("V4_RECOVERY_CHAIN_BRIDGE_TORN_TAIL_INVALID")
    payload: dict[str, object] = {
        "schema_version": RECOVERY_SCHEMA,
        "source_ledger_path": str(Path(source_ledger_path)),
        "committed_prefix_length": committed_prefix_length,
        "committed_prefix_sha256": committed_prefix_sha256,
        "previous_event_sha256": previous_event_sha256,
        "next_sequence_index": committed_prefix_length,
        "torn_tail_sha256": torn_tail_sha256,
        "source_ledger_sha256": source_ledger_sha256,
    }
    payload["bridge_sha256"] = _sha256_json(payload)
    return payload
