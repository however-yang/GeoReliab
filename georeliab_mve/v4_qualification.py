"""Qualification, provenance, and fail-closed monitoring primitives.

This module is intentionally independent from the frozen Attempt-05 runtime.
It records provenance and resource decisions without reading scientific metric
payloads.  Attempt-05 is historical engineering evidence only; no helper in
this module may turn its 199 completed units into current scientific progress.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
import json
import math
from pathlib import Path
from typing import Mapping, Sequence


QUALIFICATION_SCHEMA = "georeliab-v4-qualification-1.0"
NO_SCIENTIFIC_RESULT = "NO_SCIENTIFIC_RESULT"
ATTEMPT05_HISTORICAL_COUNT = 199
AUTHORIZED_TOTAL_UNITS = 400
TARGET_GPU_HOURS = 35.0
HARD_GPU_HOURS = 50.0
TARGET_STORAGE_BYTES = 150 * 1024**3
HARD_STORAGE_BYTES = 1 * 1024**4

SOURCE_PATCH_ALLOWLIST = (
    "georeliab_mve/v4_attempt05_forensics.py",
    "georeliab_mve/v4_attempt05_recovery.py",
    "georeliab_mve/v4_attempt05_runtime.py",
    "georeliab_mve/v4_records.py",
    "tests/test_v4_attempt05_recovery.py",
    "tests/test_v4_attempt05_runtime.py",
)


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


@dataclass(frozen=True, slots=True)
class CanonicalProvenance:
    source_commit: str
    source_parent: str
    source_tree: str
    patch_sha256: str
    stable_patch_id: str
    canonical_base_commit: str
    canonical_base_tree: str
    replay_commit: str
    replay_tree: str
    replay_patch_id: str
    changed_paths: tuple[str, ...]
    scientific_config_zero_drift: bool
    schema_version: str = QUALIFICATION_SCHEMA

    def __post_init__(self) -> None:
        if tuple(sorted(self.changed_paths)) != tuple(self.changed_paths):
            raise ValueError("changed_paths must be sorted")
        if set(self.changed_paths) != set(SOURCE_PATCH_ALLOWLIST):
            raise ValueError("canonical replay path allowlist mismatch")
        for name in (
            "source_commit",
            "source_parent",
            "source_tree",
            "patch_sha256",
            "stable_patch_id",
            "canonical_base_commit",
            "canonical_base_tree",
            "replay_commit",
            "replay_tree",
            "replay_patch_id",
        ):
            value = getattr(self, name)
            if not isinstance(value, str) or not value:
                raise ValueError(f"{name} must be non-empty")
        if not self.scientific_config_zero_drift:
            raise ValueError("canonical promotion requires scientific zero drift")

    def to_dict(self) -> dict[str, object]:
        return asdict(self) | {"changed_paths": list(self.changed_paths)}


def write_json_no_clobber(path: Path, payload: Mapping[str, object]) -> None:
    """Write a new evidence manifest with the canonical no-clobber backend."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(path)
    # Import lazily so qualification remains a light CPU-only surface while
    # sharing the exact renameat2/RENAME_NOREPLACE implementation.
    from .v4_attempt05_recovery import Attempt05RecoveryError, atomic_write_bytes

    try:
        atomic_write_bytes(path, _canonical_json(payload))
    except Attempt05RecoveryError as exc:
        if path.exists():
            raise FileExistsError(path) from exc
        raise


@dataclass(frozen=True, slots=True)
class HistoricalGpuBounds:
    active_hours_lower: float
    active_hours_upper: float
    card_hours: float
    wall_hours: float
    cumulative_storage_bytes: int
    telemetry_complete: bool

    def __post_init__(self) -> None:
        numeric = (
            self.active_hours_lower,
            self.active_hours_upper,
            self.card_hours,
            self.wall_hours,
        )
        if any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or value < 0
            for value in numeric
        ):
            raise ValueError("GPU bounds must be finite and non-negative")
        if self.active_hours_lower > self.active_hours_upper:
            raise ValueError("active lower bound exceeds upper bound")
        if type(self.cumulative_storage_bytes) is not int or self.cumulative_storage_bytes < 0:
            raise ValueError("storage must be a non-negative integer")


def evaluate_historical_gpu_gate(
    bounds: HistoricalGpuBounds,
    *,
    additional_gpu_hours: float = 0.0,
    additional_storage_bytes: int = 0,
) -> dict[str, object]:
    """Fail closed when historical usage cannot be proved below hard limits."""

    if additional_gpu_hours < 0 or additional_storage_bytes < 0:
        raise ValueError("additional usage must be non-negative")
    lower = bounds.active_hours_lower + additional_gpu_hours
    upper = bounds.active_hours_upper + additional_gpu_hours
    storage = bounds.cumulative_storage_bytes + additional_storage_bytes
    if lower >= HARD_GPU_HOURS or storage >= HARD_STORAGE_BYTES:
        status = "V4_GPU_HARD_CEILING_BLOCKED"
        reason = "HARD_CEILING_REACHED"
    elif not bounds.telemetry_complete or upper >= HARD_GPU_HOURS:
        status = "V4_GPU_HISTORY_UNRESOLVED"
        reason = "CANNOT_PROVE_HARD_CEILING_HEADROOM"
    elif lower >= TARGET_GPU_HOURS or storage >= TARGET_STORAGE_BYTES:
        status = "V4_GPU_TARGET_REAUTHORIZATION_REQUIRED"
        reason = "TARGET_REACHED"
    else:
        status = "V4_GPU_WITHIN_CUMULATIVE_BUDGET"
        reason = "HISTORICAL_HEADROOM_PROVEN"
    return {
        "schema_version": QUALIFICATION_SCHEMA,
        "status": status,
        "reason_code": reason,
        "active_hours_lower": lower,
        "active_hours_upper": upper,
        "card_hours": bounds.card_hours,
        "wall_hours": bounds.wall_hours,
        "storage_bytes": storage,
        "target_gpu_hours": TARGET_GPU_HOURS,
        "hard_gpu_hours": HARD_GPU_HOURS,
        "target_storage_bytes": TARGET_STORAGE_BYTES,
        "hard_storage_bytes": HARD_STORAGE_BYTES,
        "scientific_result": NO_SCIENTIFIC_RESULT,
    }


def reject_attempt05_source(*, attempt_id: str, source_attempt_id: str) -> None:
    def token(value: str) -> str:
        return "".join(str(value).lower().split()).replace("-", "").replace("_", "")

    # Treat all historical spellings (attempt-05, attempt_05, attempt05,
    # v4-mve-attempt-05, and suffixed prediction/session aliases) as the same
    # forbidden cross-attempt source.
    if "attempt05" in token(attempt_id) or "attempt05" in token(source_attempt_id):
        raise ValueError("V4_ATTEMPT05_SOURCE_REJECTED_FOR_NEW_ATTEMPT")


def build_hourly_monitor_status(
    *,
    stage: str,
    stage_completed: int,
    stage_total: int,
    attempt06_valid_completed: int,
    attempt06_elapsed_seconds: float,
    cumulative_materialization_elapsed_seconds: float,
    gpu0_owners: Sequence[str] = (),
    gpu1_owners: Sequence[str] = (),
    cumulative_gpu_active_hours: float = 0.0,
    cumulative_card_hours: float = 0.0,
    cumulative_storage_bytes: int = 0,
    invalid_count: int = 0,
    duplicate_count: int = 0,
    identity_mismatch_count: int = 0,
    formal_valid_completed: int | None = None,
    formal_total: int | None = None,
    cumulative_wall_time_seconds: float | None = None,
    cumulative_gpu_utilization: float | None = None,
) -> dict[str, object]:
    """Build the one-hour status; historical Attempt-05 never counts progress."""

    if not isinstance(stage, str) or not stage:
        raise ValueError("stage must be non-empty")
    if type(stage_completed) is not int or type(stage_total) is not int or not (0 <= stage_completed <= stage_total):
        raise ValueError("invalid stage completion")
    if type(attempt06_valid_completed) is not int or not 0 <= attempt06_valid_completed <= AUTHORIZED_TOTAL_UNITS:
        raise ValueError("invalid Attempt-06 completion")
    for value in (attempt06_elapsed_seconds, cumulative_materialization_elapsed_seconds, cumulative_gpu_active_hours, cumulative_card_hours):
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)) or value < 0:
            raise ValueError("elapsed/resource values must be finite and non-negative")
    if type(cumulative_storage_bytes) is not int or cumulative_storage_bytes < 0:
        raise ValueError("invalid storage")
    if formal_valid_completed is not None or formal_total is not None:
        if type(formal_valid_completed) is not int or type(formal_total) is not int:
            raise ValueError("formal progress must use integer values")
        if formal_total <= 0 or not 0 <= formal_valid_completed <= formal_total:
            raise ValueError("invalid formal confirmation progress")
    for value in (cumulative_wall_time_seconds, cumulative_gpu_utilization):
        if value is None:
            continue
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)) or value < 0:
            raise ValueError("invalid cumulative monitor value")
    if cumulative_gpu_utilization is not None and cumulative_gpu_utilization > 1:
        raise ValueError("cumulative GPU utilization cannot exceed 1")
    return {
        "schema_version": QUALIFICATION_SCHEMA,
        "external_monitor_interval_seconds": 3600,
        "internal_heartbeat_interval_seconds": 60,
        "scientific_result": NO_SCIENTIFIC_RESULT,
        "historical_attempt05": {
            "valid_completed": ATTEMPT05_HISTORICAL_COUNT,
            "authorized_total": AUTHORIZED_TOTAL_UNITS,
            "classification": "NON_RESUMABLE_ENGINEERING_HISTORY",
        },
        "current_progress": {
            "attempt06_valid_completed": attempt06_valid_completed,
            "full400_materialization_progress": attempt06_valid_completed / AUTHORIZED_TOTAL_UNITS,
            "full400_materialization_progress_percent": 100.0 * attempt06_valid_completed / AUTHORIZED_TOTAL_UNITS,
        },
        "stage": {
            "name": stage,
            "completed": stage_completed,
            "total": stage_total,
            "progress": stage_completed / stage_total if stage_total else 0.0,
        },
        "formal_confirmation_progress": ({"valid_completed": formal_valid_completed, "total": formal_total, "progress": formal_valid_completed / formal_total, "progress_percent": 100.0 * formal_valid_completed / formal_total} if formal_valid_completed is not None and formal_total is not None else None),
        "attempt06_elapsed_seconds": attempt06_elapsed_seconds,
        "cumulative_materialization_elapsed_seconds": cumulative_materialization_elapsed_seconds,
        "cumulative_wall_time_seconds": cumulative_wall_time_seconds if cumulative_wall_time_seconds is not None else cumulative_materialization_elapsed_seconds,
        "cumulative_gpu_utilization": cumulative_gpu_utilization if cumulative_gpu_utilization is not None else (cumulative_gpu_active_hours / cumulative_card_hours if cumulative_card_hours > 0 else 0.0),
        "gpu0_owners": sorted(str(value) for value in gpu0_owners),
        "gpu1_owners": sorted(str(value) for value in gpu1_owners),
        "cumulative_gpu_active_hours": cumulative_gpu_active_hours,
        "cumulative_card_hours": cumulative_card_hours,
        "cumulative_storage_bytes": cumulative_storage_bytes,
        "invalid_count": invalid_count,
        "duplicate_count": duplicate_count,
        "identity_mismatch_count": identity_mismatch_count,
        "budget_status": {
            "target_exceeded": cumulative_gpu_active_hours >= TARGET_GPU_HOURS or cumulative_storage_bytes >= TARGET_STORAGE_BYTES,
            "hard_ceiling_exceeded": cumulative_gpu_active_hours >= HARD_GPU_HOURS or cumulative_storage_bytes >= HARD_STORAGE_BYTES,
        },
    }


# Recovery smoke is implemented once in the durable recovery module.  These
# compatibility exports keep qualification callers on the canonical contract
# without creating a second manifest/receipt schema here.
from .v4_attempt05_recovery import (  # noqa: E402  (late import avoids import-time work)
    AUTHORIZED_GPU_UUID,
    AUTHORIZED_PHYSICAL_GPU_INDEX,
    ScheduleIdentityManifest,
    RecoverySmokeManifest,
    build_recovery_smoke_manifest,
    evaluate_recovery_smoke,
)

AUTHORIZED_GPU_PHYSICAL_INDEX = AUTHORIZED_PHYSICAL_GPU_INDEX
V4_RECOVERY_RUNTIME_QUALIFIED = "V4_RECOVERY_RUNTIME_QUALIFIED"
V4_RECOVERY_RUNTIME_BLOCKED = "V4_RECOVERY_RUNTIME_NOT_QUALIFIED"


def validate_recovery_smoke_receipts(
    manifest: RecoverySmokeManifest,
    receipts: Sequence[Mapping[str, object]],
    *,
    gpu1_owners: Sequence[str] = (),
    scientific_markers: Sequence[str] = (),
) -> dict[str, object]:
    """Compatibility wrapper around the canonical smoke evaluator.

    The canonical evaluator remains in :mod:`v4_attempt05_recovery`; this
    wrapper only adds global GPU/scientific-marker guards used by older callers.
    """

    result = evaluate_recovery_smoke(manifest, observations=receipts)
    reasons = list(result.get("reason_codes", ()))
    if gpu1_owners:
        reasons.append("GPU1_PROJECT_PROCESS_PRESENT")
    if scientific_markers:
        reasons.append("SCIENTIFIC_MARKER_PRESENT_GLOBAL")
    if reasons:
        result["reason_codes"] = sorted(set(reasons))
        result["status"] = V4_RECOVERY_RUNTIME_BLOCKED
    result.setdefault("gpu1_owners", sorted(str(value) for value in gpu1_owners))
    result["scientific_result"] = NO_SCIENTIFIC_RESULT
    return result


__all__ = [
    "ScheduleIdentityManifest",
    "CanonicalProvenance",
    "HistoricalGpuBounds",
    "build_hourly_monitor_status",
    "evaluate_historical_gpu_gate",
    "reject_attempt05_source",
    "write_json_no_clobber",
    "RecoverySmokeManifest",
    "build_recovery_smoke_manifest",
    "evaluate_recovery_smoke",
    "validate_recovery_smoke_receipts",
    "V4_RECOVERY_RUNTIME_QUALIFIED",
    "V4_RECOVERY_RUNTIME_BLOCKED",
    "AUTHORIZED_GPU_UUID",
    "AUTHORIZED_GPU_PHYSICAL_INDEX",
]