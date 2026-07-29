"""Read-only storage accounting and digest-bound retention plans."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any, Iterable, Mapping

from .science_lock import BASE_PROJECT_COMMIT, validate_science_lock


PLAN_SCHEMA = "georeliab-storage-plan-v1"
SNAPSHOT_SCHEMA = "georeliab-storage-before-v1"
RECEIPT_SCHEMA = "georeliab-storage-apply-receipt-v1"
R1_LIMIT_BYTES = 900_000_000_000
HARD_LIMIT_BYTES = 1_000_000_000_000
REMAINING_RETAINED_TARGET_BYTES = 500_000_000_000
REPORTING_TARGET_BYTES = 300_000_000_000


class StorageAuditError(RuntimeError):
    """Raised when storage accounting or a retention plan cannot be trusted."""


@dataclass(frozen=True, slots=True)
class FileUsage:
    path: str
    level: str
    logical_bytes: int
    allocated_bytes: int
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "level": self.level,
            "logical_bytes": self.logical_bytes,
            "allocated_bytes": self.allocated_bytes,
            "reason": self.reason,
        }


def _json_bytes(value: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(dict(value), sort_keys=True, separators=(",", ":"), default=str)
        + "\n"
    ).encode("utf-8")


def _sha_json(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(_json_bytes(value)).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".partial")
    with partial.open("wb") as handle:
        handle.write(_json_bytes(payload))
        handle.flush()
        try:
            os.fsync(handle.fileno())
        except OSError:
            pass
    partial.replace(path)


def resolve_under_root(root: Path, target: Path, *, must_exist: bool = False) -> Path:
    """Resolve a target and reject roots, symlink escapes, and outside paths."""

    resolved_root = root.resolve()
    if resolved_root == Path(resolved_root.anchor):
        raise StorageAuditError("runtime root must not be a filesystem root")
    resolved = target.resolve(strict=must_exist)
    try:
        relative = resolved.relative_to(resolved_root)
    except ValueError as exc:
        raise StorageAuditError(f"storage target escaped runtime root: {resolved}") from exc
    if not relative.parts:
        raise StorageAuditError("storage action may not target the runtime root itself")
    return resolved


def _allocated_bytes(path: Path, logical_bytes: int) -> int:
    blocks = getattr(path.stat(), "st_blocks", None)
    if isinstance(blocks, int) and blocks >= 0:
        return blocks * 512
    return logical_bytes


def classify_storage_path(relative: Path) -> tuple[str, str]:
    """Classify one file according to the frozen L0/L1/L2 policy."""

    posix = relative.as_posix()
    name = relative.name
    parts = set(relative.parts)
    if (
        ".partial" in name
        or "mast3r_cache" in parts
        or posix.endswith("Rectified.sparse-index.zip")
        or "/debug/" in f"/{posix}/"
        or "/downloads/" in f"/{posix}/"
    ):
        return "L2", "ephemeral-work-or-download-intermediate"
    if name.endswith(".npz") and (
        any(
            token in name
            for token in (
                "geometry",
                "confidence",
                "valid_mask",
                "dense_audit",
                "subset_prediction",
            )
        )
        or "stage" in parts
    ):
        return "L1", "lossless-scientific-array-cache"
    if (
        relative.parts
        and relative.parts[0]
        in {
            "artifacts",
            "evidence",
            "manifests",
            "logs",
            "engineering-failures",
        }
    ) or name.endswith((".json", ".jsonl", ".toml", ".sha256")):
        return "L0", "immutable-governance-or-evidence"
    if "shared_gt" in parts:
        return "L0", "content-addressed-shared-ground-truth"
    return "L0", "retained-unclassified-input"


def scan_storage(root: Path) -> tuple[list[FileUsage], list[dict[str, str]]]:
    resolved_root = root.resolve()
    if not resolved_root.is_dir():
        raise StorageAuditError(f"runtime root is not a directory: {resolved_root}")
    rows: list[FileUsage] = []
    errors: list[dict[str, str]] = []
    for path in sorted(resolved_root.rglob("*")):
        try:
            if not path.is_file() or path.is_symlink():
                continue
            resolved = resolve_under_root(resolved_root, path, must_exist=True)
            relative = resolved.relative_to(resolved_root)
            logical = resolved.stat().st_size
            allocated = _allocated_bytes(resolved, logical)
            level, reason = classify_storage_path(relative)
            rows.append(FileUsage(relative.as_posix(), level, logical, allocated, reason))
        except OSError as exc:
            errors.append({"path": str(path), "reason": str(exc)})
    return rows, errors


def _read_json_lines(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise StorageAuditError(f"invalid ledger JSON {path}:{index}: {exc}") from exc
        if not isinstance(row, dict):
            raise StorageAuditError(f"ledger row is not an object: {path}:{index}")
        rows.append(row)
    return rows


def _finite_seconds(value: Any, field: str, path: Path) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise StorageAuditError(f"invalid {field} in {path}: {value!r}")
    seconds = float(value)
    if not math.isfinite(seconds) or seconds < 0:
        raise StorageAuditError(f"invalid {field} in {path}: {value!r}")
    return seconds


def _bundle_prediction_seconds(bundle_dir: Path) -> float:
    from .contracts import (
        AuditRecord,
        PredictionArtifact,
        RunManifest,
        read_json_artifact,
        validate_artifact_bundle,
    )

    manifest = read_json_artifact(bundle_dir / "run_manifest.json", RunManifest)
    prediction = read_json_artifact(
        bundle_dir / "prediction_artifact.json", PredictionArtifact
    )
    audit = read_json_artifact(bundle_dir / "audit_record.json", AuditRecord)
    validate_artifact_bundle(manifest, prediction, audit)
    return float(prediction.runtime_seconds)


def _find_bundle(root: Path, stage: str, identity: str, row: Mapping[str, Any]) -> Path:
    supplied = row.get("bundle_dir")
    if isinstance(supplied, str) and supplied:
        candidate = Path(supplied)
        if not candidate.is_absolute():
            candidate = root / candidate
        try:
            return resolve_under_root(root, candidate, must_exist=True)
        except (OSError, StorageAuditError):
            pass
    stage_root = root / "stage" / stage / "bundles"
    matches = sorted(stage_root.glob(f"*/{identity}")) if stage_root.exists() else []
    if len(matches) == 1:
        return resolve_under_root(root, matches[0], must_exist=True)
    raise StorageAuditError(
        f"cannot bind legacy ledger row to exactly one validated bundle: {stage}/{identity}"
    )


def account_runtime(root: Path) -> dict[str, Any]:
    """Separate wall time from validated model-inference time."""

    ledger_paths = sorted((root / "stage").glob("*/ledger.jsonl"))
    ledger_paths.extend(sorted(root.glob("preflight-real/*/stage/*/ledger.jsonl")))
    wall_seconds = 0.0
    explicit_gpu_seconds = 0.0
    legacy_latest: dict[tuple[str, str], tuple[Path, dict[str, Any]]] = {}
    issues: list[dict[str, str]] = []
    for ledger in ledger_paths:
        try:
            rows = _read_json_lines(ledger)
        except (OSError, StorageAuditError) as exc:
            issues.append({"path": str(ledger), "reason": str(exc)})
            continue
        for index, row in enumerate(rows):
            try:
                wall_value = row.get("wall_runtime_seconds", row.get("runtime_seconds", 0.0))
                wall_seconds += _finite_seconds(wall_value, "wall_runtime_seconds", ledger)
                if "gpu_inference_seconds" in row:
                    explicit_gpu_seconds += _finite_seconds(
                        row["gpu_inference_seconds"], "gpu_inference_seconds", ledger
                    )
                elif row.get("state") in {"completed", "skipped"}:
                    identity = str(
                        row.get("item_identity", row.get("identity", f"row-{index}"))
                    )
                    legacy_latest[(str(ledger), identity)] = (ledger, row)
                elif _finite_seconds(row.get("runtime_seconds", 0.0), "runtime_seconds", ledger) > 0:
                    issues.append(
                        {
                            "path": str(ledger),
                            "reason": "legacy non-complete row cannot be backfilled",
                        }
                    )
            except StorageAuditError as exc:
                issues.append({"path": str(ledger), "reason": str(exc)})
    legacy_gpu_seconds = 0.0
    failed_backfills = 0
    for (ledger_text, identity), (ledger, row) in legacy_latest.items():
        stage = ledger.parent.name
        try:
            bundle = _find_bundle(root, stage, identity, row)
            legacy_gpu_seconds += _bundle_prediction_seconds(bundle)
        except Exception as exc:
            failed_backfills += 1
            issues.append(
                {
                    "path": ledger_text,
                    "reason": f"legacy bundle validation failed for {identity}: {exc}",
                }
            )
    return {
        "wall_runtime_seconds": wall_seconds,
        "gpu_inference_seconds": explicit_gpu_seconds + legacy_gpu_seconds,
        "wall_runtime_hours": wall_seconds / 3600.0,
        "gpu_inference_hours": (explicit_gpu_seconds + legacy_gpu_seconds) / 3600.0,
        "legacy_backfilled_items": len(legacy_latest) - failed_backfills,
        "status": "OK" if not issues else "UNAVAILABLE",
        "issues": issues,
    }


def _percentile95(values: Iterable[int]) -> int:
    ordered = sorted(int(value) for value in values if int(value) >= 0)
    if not ordered:
        return 0
    index = max(0, math.ceil(0.95 * len(ordered)) - 1)
    return ordered[index]


def _bundle_sizes(root: Path, stage: str, model: str) -> list[int]:
    bundle_root = root / "stage" / stage / "bundles" / model
    sizes: list[int] = []
    if not bundle_root.exists():
        return sizes
    for bundle in sorted(bundle_root.iterdir()):
        if not bundle.is_dir() or bundle.name.endswith(".partial"):
            continue
        total = sum(
            path.stat().st_size
            for path in bundle.rglob("*")
            if path.is_file() and not path.is_symlink()
        )
        sizes.append(total)
    return sizes


def _projection(root: Path, current_logical: int, levels: Mapping[str, int]) -> dict[str, Any]:
    model_rates: dict[str, int] = {}
    for model in ("vggt", "mast3r"):
        smoke = _bundle_sizes(root, "smoke", model)
        preflight: list[int] = []
        for repeat in ("repeat-a", "repeat-b"):
            repeat_root = (
                root
                / "preflight-real"
                / repeat
                / "stage"
                / "preflight"
                / "bundles"
                / model
            )
            if repeat_root.exists():
                for bundle in repeat_root.iterdir():
                    if bundle.is_dir():
                        preflight.append(
                            sum(p.stat().st_size for p in bundle.rglob("*") if p.is_file())
                        )
        model_rates[model] = _percentile95(smoke or preflight)
    completed = {
        model: len(_bundle_sizes(root, "smoke", model))
        for model in ("vggt", "mast3r")
    }
    p2_remaining = sum(
        max(0, 100 - completed[model]) * model_rates[model] for model in model_rates
    )
    p3 = 200 * model_rates["vggt"] + 200 * model_rates["mast3r"]
    p5 = 240 * model_rates["vggt"] + 240 * model_rates["mast3r"]
    transient_peak = max(model_rates.values(), default=0)
    subtotal = current_logical + p2_remaining + p3 + p5 + transient_peak
    reserve = math.ceil(subtotal * 0.10)
    projected = subtotal + reserve
    return {
        "model_retained_byte_p95": model_rates,
        "p2_completed_by_model": completed,
        "p2_remaining_retained_bytes": p2_remaining,
        "p3_retained_bytes": p3,
        "conditional_p5_retained_bytes": p5,
        "single_task_temporary_peak_bytes": transient_peak,
        "reserve_bytes": reserve,
        "full_worst_path_bytes": projected,
        "r1_status": "PASS" if projected < R1_LIMIT_BYTES else "FAIL",
        "r1_limit_bytes": R1_LIMIT_BYTES,
        "hard_limit_bytes": HARD_LIMIT_BYTES,
        "remaining_retained_target_bytes": REMAINING_RETAINED_TARGET_BYTES,
        "reporting_target_bytes": REPORTING_TARGET_BYTES,
        "l1_current_bytes": int(levels.get("L1", 0)),
    }


def capture_storage_snapshot(root: Path, *, source_root: Path) -> dict[str, Any]:
    resolved_root = root.resolve()
    rows, scan_errors = scan_storage(resolved_root)
    levels: dict[str, dict[str, int]] = {
        level: {"logical_bytes": 0, "allocated_bytes": 0, "file_count": 0}
        for level in ("L0", "L1", "L2")
    }
    for row in rows:
        levels[row.level]["logical_bytes"] += row.logical_bytes
        levels[row.level]["allocated_bytes"] += row.allocated_bytes
        levels[row.level]["file_count"] += 1
    logical = sum(row.logical_bytes for row in rows)
    allocated = sum(row.allocated_bytes for row in rows)
    runtime = account_runtime(resolved_root)
    science_lock = validate_science_lock(source_root)
    projection = _projection(
        resolved_root,
        logical,
        {level: value["logical_bytes"] for level, value in levels.items()},
    )
    return {
        "schema_version": SNAPSHOT_SCHEMA,
        "base_project_commit": BASE_PROJECT_COMMIT,
        "runtime_root": str(resolved_root),
        "logical_bytes": logical,
        "allocated_bytes": allocated,
        "levels": levels,
        "reclaimable_l2_bytes": levels["L2"]["logical_bytes"],
        "scan_errors": scan_errors,
        "resource_accounting": runtime,
        "science_lock": science_lock,
        "projection": projection,
        "status": (
            "PASS"
            if not scan_errors
            and runtime["status"] == "OK"
            and projection["r1_status"] == "PASS"
            else "BLOCKED"
        ),
    }


def build_storage_plan(
    root: Path, snapshot: Mapping[str, Any], files: Iterable[FileUsage]
) -> dict[str, Any]:
    actions: list[dict[str, Any]] = []
    for row in files:
        if row.level != "L2":
            continue
        actions.append(
            {
                "action": "review_ephemeral",
                "path": row.path,
                "logical_bytes": row.logical_bytes,
                "allocated_bytes": row.allocated_bytes,
                "requires_verified_retention": True,
            }
        )
    payload: dict[str, Any] = {
        "schema_version": PLAN_SCHEMA,
        "base_project_commit": BASE_PROJECT_COMMIT,
        "runtime_root": str(root.resolve()),
        "storage_before_sha256": _sha_json(snapshot),
        "actions": actions,
        "projected_reclaimable_bytes": sum(
            int(action["logical_bytes"]) for action in actions
        ),
        "mutation_enabled": False,
        "reason": "audit-only-plan-before-retention-implementation",
    }
    payload["plan_payload_sha256"] = _sha_json(payload)
    return payload


def storage_audit(
    root: Path,
    *,
    source_root: Path,
    output_dir: Path | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    resolved_root = root.resolve()
    resolved_output = (
        resolve_under_root(resolved_root, output_dir)
        if output_dir is not None
        else resolved_root / "artifacts"
    )
    files, _errors = scan_storage(resolved_root)
    snapshot = capture_storage_snapshot(resolved_root, source_root=source_root)
    plan = build_storage_plan(resolved_root, snapshot, files)
    _atomic_json(resolved_output / "storage_before.json", snapshot)
    plan_path = resolved_output / "storage_plan.json"
    _atomic_json(plan_path, plan)
    returned_plan = dict(plan)
    returned_plan["plan_file_sha256"] = _sha256_file(plan_path)
    return snapshot, returned_plan


def apply_storage_plan(
    root: Path,
    plan_path: Path,
    *,
    expected_plan_sha256: str,
    receipt_path: Path | None = None,
) -> dict[str, Any]:
    """Validate a plan binding; mutation is enabled by the retention commit."""

    if len(expected_plan_sha256) != 64:
        raise StorageAuditError("expected plan SHA-256 must be 64 lowercase hex chars")
    actual_file_sha = _sha256_file(plan_path)
    if actual_file_sha != expected_plan_sha256:
        raise StorageAuditError(
            f"storage plan file digest mismatch: {actual_file_sha} != {expected_plan_sha256}"
        )
    payload = json.loads(plan_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("schema_version") != PLAN_SCHEMA:
        raise StorageAuditError("unsupported storage plan")
    if Path(str(payload.get("runtime_root", ""))).resolve() != root.resolve():
        raise StorageAuditError("storage plan is bound to another runtime root")
    if payload.get("mutation_enabled") is not True:
        raise StorageAuditError("storage plan is audit-only and cannot be applied")
    for action in payload.get("actions", []):
        if not isinstance(action, dict) or not isinstance(action.get("path"), str):
            raise StorageAuditError("storage plan contains an invalid action")
        resolve_under_root(root, root / action["path"])
    receipt = {
        "schema_version": RECEIPT_SCHEMA,
        "runtime_root": str(root.resolve()),
        "plan_file_sha256": actual_file_sha,
        "status": "PASS",
        "actions_applied": 0,
    }
    destination = receipt_path or root / "artifacts" / "storage_apply_receipt.json"
    resolve_under_root(root, destination)
    _atomic_json(destination, receipt)
    return receipt
