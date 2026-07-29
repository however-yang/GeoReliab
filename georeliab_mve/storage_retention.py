"""Digest-bound retention execution and worst-path storage projection.

The functions in this module are deliberately storage-only.  They never
change decoded array semantics, scientific schedules, or evidence admission.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
from typing import Any, Iterable, Mapping, Sequence
import zipfile
import zlib

from .artifact_storage import (
    assert_npz_equivalent,
    npz_fingerprint,
    reencode_npz_lossless,
    sha256_file,
)


R1_LIMIT_BYTES = 900_000_000_000
HARD_LIMIT_BYTES = 1_000_000_000_000
REMAINING_RETAINED_TARGET_BYTES = 500_000_000_000
REPORTING_TARGET_BYTES = 300_000_000_000
ZERO_UPDATE_RETAINED_FALLBACK_BYTES = 64_000_000
ACTION_RECEIPT_SCHEMA = "georeliab-storage-action-receipt-v1"
_NPZ_SAMPLE_BYTES = 8 * 1024 * 1024


class StorageRetentionError(RuntimeError):
    """Raised when a retention action cannot be proven safe."""


def _json_bytes(value: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(dict(value), sort_keys=True, separators=(",", ":"), default=str)
        + "\n"
    ).encode("utf-8")


def _sha_json(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(_json_bytes(value)).hexdigest()


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


def _resolve_member(root: Path, relative: str, *, must_exist: bool = False) -> Path:
    member = Path(relative)
    if member.is_absolute() or ".." in member.parts:
        raise StorageRetentionError(f"unsafe retention path: {relative}")
    resolved_root = root.resolve()
    resolved = (resolved_root / member).resolve(strict=must_exist)
    try:
        nested = resolved.relative_to(resolved_root)
    except ValueError as exc:
        raise StorageRetentionError(
            f"retention target escaped runtime root: {relative}"
        ) from exc
    if not nested.parts:
        raise StorageRetentionError("retention may not target the runtime root")
    return resolved


def _is_fixed_deflated_npz(path: Path) -> bool:
    try:
        with zipfile.ZipFile(path) as archive:
            rows = archive.infolist()
            return bool(rows) and all(
                row.compress_type == zipfile.ZIP_DEFLATED
                and row.date_time == (1980, 1, 1, 0, 0, 0)
                and not Path(row.filename).is_absolute()
                and ".." not in Path(row.filename).parts
                for row in rows
            )
    except (OSError, zipfile.BadZipFile):
        return False


def _sample_deflate_ratio(archive: zipfile.ZipFile, info: zipfile.ZipInfo) -> float:
    compressor = zlib.compressobj(level=9, wbits=-15)
    source_bytes = 0
    compressed_bytes = 0
    with archive.open(info) as member:
        while source_bytes < _NPZ_SAMPLE_BYTES:
            block = member.read(min(1024 * 1024, _NPZ_SAMPLE_BYTES - source_bytes))
            if not block:
                break
            source_bytes += len(block)
            compressed_bytes += len(compressor.compress(block))
    compressed_bytes += len(compressor.flush())
    if source_bytes == 0:
        return 1.0
    # A 10% guard and a 50% floor keep the dry-run conservative relative to
    # the measured 28%-50% geometry ratios.
    return min(1.0, max(0.50, 1.10 * compressed_bytes / source_bytes))


def estimate_npz_retained_bytes(path: Path) -> int:
    """Estimate deterministic lossless NPZ size without loading dense arrays."""

    try:
        with zipfile.ZipFile(path) as archive:
            rows = archive.infolist()
            if not rows:
                return path.stat().st_size
            if _is_fixed_deflated_npz(path):
                return path.stat().st_size
            total = 22
            for row in rows:
                name_bytes = len(row.filename.encode("utf-8"))
                ratio = _sample_deflate_ratio(archive, row)
                compressed = math.ceil(row.file_size * ratio)
                total += compressed + 76 + 2 * name_bytes
            return min(path.stat().st_size, total)
    except (OSError, zipfile.BadZipFile):
        return path.stat().st_size


def _walk_strings(value: Any) -> Iterable[str]:
    if isinstance(value, str):
        yield value
    elif isinstance(value, Mapping):
        for item in value.values():
            yield from _walk_strings(item)
    elif isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
        for item in value:
            yield from _walk_strings(item)


def _reference_tokens(root: Path) -> set[str]:
    tokens: set[str] = set()
    for path in sorted(root.rglob("*.json")):
        if path.is_symlink() or not path.is_file():
            continue
        try:
            relative = path.relative_to(root)
        except ValueError:
            continue
        if relative.as_posix() in {
            "artifacts/storage_before.json",
            "artifacts/storage_plan.json",
            "artifacts/storage_apply_receipt.json",
        } or relative.parts[:2] == (
            "artifacts",
            "storage-retention",
        ):
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        tokens.update(_walk_strings(payload))
    return tokens


def _is_referenced(root: Path, relative: str, tokens: set[str]) -> bool:
    path = _resolve_member(root, relative)
    candidates = {
        relative,
        relative.replace("/", os.sep),
        str(path),
        path.as_uri(),
    }
    return bool(candidates & tokens)


def _materialization_member_rows(payload: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    archive = payload.get("archives", {}).get("Rectified.zip", {})
    members = archive.get("members", {}) if isinstance(archive, Mapping) else {}
    if not isinstance(members, Mapping):
        return []
    return [row for row in members.values() if isinstance(row, Mapping)]


def verify_sparse_index_preconditions(
    root: Path,
    path: Path,
    *,
    expected_source_sha256: str | None = None,
    expected_manifest_sha256: str | None = None,
) -> dict[str, Any]:
    """Prove frozen materialization before admitting sparse-index removal."""

    manifest = root / "manifests" / "frozen_materialization.json"
    if not manifest.is_file():
        raise StorageRetentionError(
            "sparse index is retained until frozen materialization exists"
        )
    manifest_sha = sha256_file(manifest)
    if expected_manifest_sha256 and manifest_sha != expected_manifest_sha256:
        raise StorageRetentionError("frozen materialization digest changed")
    try:
        payload = json.loads(manifest.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise StorageRetentionError("frozen materialization is unreadable") from exc
    if payload.get("schema_version") != "frozen-materialization-v1":
        raise StorageRetentionError("frozen materialization schema mismatch")
    archive = payload.get("archives", {}).get("Rectified.zip", {})
    if (
        not isinstance(archive, Mapping)
        or not isinstance(archive.get("central_directory_sha256"), str)
        or len(str(archive.get("central_directory_sha256"))) != 64
        or not isinstance(archive.get("bytes"), int)
        or int(archive.get("bytes", 0)) <= 0
    ):
        raise StorageRetentionError("Rectified source archive summary is incomplete")
    members = _materialization_member_rows(payload)
    if not members:
        raise StorageRetentionError("frozen Rectified member inventory is empty")
    for row in members:
        member_path = Path(str(row.get("path", "")))
        try:
            member_path.resolve().relative_to(root.resolve())
        except (OSError, ValueError) as exc:
            raise StorageRetentionError(
                f"materialized member escaped runtime root: {member_path}"
            ) from exc
        expected_sha = str(row.get("raw_sha256", ""))
        expected_size = int(row.get("uncompressed_size", -1))
        if (
            len(expected_sha) != 64
            or not member_path.is_file()
            or member_path.stat().st_size != expected_size
            or sha256_file(member_path) != expected_sha
        ):
            raise StorageRetentionError(
                f"materialized Rectified member failed verification: {member_path}"
            )
    source_sha = sha256_file(path)
    if expected_source_sha256 and source_sha != expected_source_sha256:
        raise StorageRetentionError("sparse index source digest changed")
    return {
        "source_sha256": source_sha,
        "materialization_sha256": manifest_sha,
        "member_count": len(members),
        "central_directory_sha256": archive["central_directory_sha256"],
    }


def _superseded_archive_is_complete(root: Path) -> bool:
    from .execution_governance import (
        ExecutionGovernanceError,
        validate_superseded_archive,
    )

    try:
        return validate_superseded_archive(root).get("status") == "PASS"
    except (ExecutionGovernanceError, OSError):
        return False


def build_retention_actions(root: Path, files: Iterable[Any]) -> list[dict[str, Any]]:
    """Build deterministic, conservative actions from storage classifications."""

    resolved_root = root.resolve()
    tokens = _reference_tokens(resolved_root)
    superseded_archive_complete = _superseded_archive_is_complete(resolved_root)
    actions: list[dict[str, Any]] = []
    for row in sorted(files, key=lambda value: value.path):
        path = _resolve_member(resolved_root, row.path, must_exist=True)
        if (
            not superseded_archive_complete
            and (
                row.path.startswith("preflight-real/")
                or row.path.startswith("stage/smoke/")
            )
        ):
            # f539 member bytes must be archived before any lossless re-encode.
            continue
        if row.level == "L1" and path.suffix == ".npz":
            if not _is_fixed_deflated_npz(path):
                actions.append(
                    {
                        "action": "lossless_reencode_npz",
                        "path": row.path,
                        "source_sha256": sha256_file(path),
                        "source_bytes": path.stat().st_size,
                        "estimated_retained_bytes": estimate_npz_retained_bytes(path),
                    }
                )
            continue
        if row.level != "L2":
            continue
        parts = set(Path(row.path).parts)
        if "mast3r_cache" in parts:
            # Existing f539 caches are handled only after the superseded
            # archive is committed; future bundles finalize them pre-commit.
            continue
        if row.path.endswith("Rectified.sparse-index.zip"):
            try:
                proof = verify_sparse_index_preconditions(resolved_root, path)
            except StorageRetentionError:
                continue
            actions.append(
                {
                    "action": "delete_verified_sparse_index",
                    "path": row.path,
                    "source_sha256": proof["source_sha256"],
                    "source_bytes": path.stat().st_size,
                    "materialization_sha256": proof["materialization_sha256"],
                    "materialized_member_count": proof["member_count"],
                    "central_directory_sha256": proof[
                        "central_directory_sha256"
                    ],
                }
            )
            continue
        if _is_referenced(resolved_root, row.path, tokens):
            continue
        actions.append(
            {
                "action": "delete_unreferenced_ephemeral",
                "path": row.path,
                "source_sha256": sha256_file(path),
                "source_bytes": path.stat().st_size,
            }
        )
    return sorted(actions, key=lambda row: (str(row["action"]), str(row["path"])))


def _current_project_commit(source_root: Path) -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=source_root,
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
        value = result.stdout.strip().lower()
    except (OSError, subprocess.SubprocessError):
        return ""
    return value if len(value) == 40 else ""


def _bundle_project_commit(bundle: Path) -> str:
    manifest = bundle / "run_manifest.json"
    try:
        payload = json.loads(manifest.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return ""
    provenance = payload.get("provenance")
    if not isinstance(provenance, Mapping):
        return ""
    return str(provenance.get("project_commit", "")).lower()


def _bundle_storage_sample(bundle: Path, *, stage: str) -> dict[str, int]:
    retained = 0
    cache_bytes = 0
    live_bytes = 0
    for path in bundle.rglob("*"):
        if path.is_symlink() or not path.is_file():
            continue
        size = path.stat().st_size
        live_bytes += size
        relative = path.relative_to(bundle)
        if "mast3r_cache" in relative.parts:
            cache_bytes += size
            continue
        if path.name == "gt_points.npz":
            # New bundles point at one content-addressed shared GT payload.
            continue
        retained += (
            estimate_npz_retained_bytes(path)
            if path.suffix == ".npz"
            else size
        )
    test_retained = retained + (cache_bytes if stage == "test" else 0)
    return {
        "smoke_retained_bytes": retained,
        "test_retained_bytes": test_retained,
        "temporary_peak_bytes": max(live_bytes, live_bytes + cache_bytes),
    }


def _p95(values: Iterable[int]) -> int:
    rows = sorted(int(value) for value in values if int(value) >= 0)
    if not rows:
        return 0
    return rows[max(0, math.ceil(0.95 * len(rows)) - 1)]


def _bundle_samples(root: Path, model: str) -> list[dict[str, int]]:
    rows: list[dict[str, int]] = []
    smoke = root / "stage" / "smoke" / "bundles" / model
    if smoke.exists():
        rows.extend(
            _bundle_storage_sample(bundle, stage="smoke")
            for bundle in sorted(smoke.iterdir())
            if bundle.is_dir() and not bundle.name.endswith(".partial")
        )
    for repeat in ("repeat-a", "repeat-b"):
        preflight = (
            root
            / "preflight-real"
            / repeat
            / "stage"
            / "preflight"
            / "bundles"
            / model
        )
        if preflight.exists():
            rows.extend(
                _bundle_storage_sample(bundle, stage="preflight")
                for bundle in sorted(preflight.iterdir())
                if bundle.is_dir() and not bundle.name.endswith(".partial")
            )
    return rows


def _completed_for_commit(root: Path, model: str, commit: str) -> int:
    if not commit:
        return 0
    bundle_root = root / "stage" / "smoke" / "bundles" / model
    if not bundle_root.exists():
        return 0
    return sum(
        1
        for bundle in bundle_root.iterdir()
        if bundle.is_dir()
        and not bundle.name.endswith(".partial")
        and _bundle_project_commit(bundle) == commit
    )


def _planned_current_bytes(
    current_logical: int,
    actions: Iterable[Mapping[str, Any]],
) -> int:
    projected = int(current_logical)
    for action in actions:
        source = int(action.get("source_bytes", 0))
        if action.get("action") == "lossless_reencode_npz":
            projected -= max(
                0,
                source - int(action.get("estimated_retained_bytes", source)),
            )
        elif str(action.get("action", "")).startswith("delete_"):
            projected -= source
    return max(0, projected)


def project_full_path(
    root: Path,
    *,
    source_root: Path,
    current_logical: int,
    levels: Mapping[str, int],
    actions: Iterable[Mapping[str, Any]],
) -> dict[str, Any]:
    """Project the full P2-P5 retained path with one sequential temp bundle."""

    commit = _current_project_commit(source_root)
    smoke_rates: dict[str, int] = {}
    test_rates: dict[str, int] = {}
    temporary_rates: dict[str, int] = {}
    completed: dict[str, int] = {}
    for model in ("vggt", "mast3r"):
        samples = _bundle_samples(root, model)
        smoke_rates[model] = _p95(
            row["smoke_retained_bytes"] for row in samples
        )
        test_rates[model] = _p95(
            row["test_retained_bytes"] for row in samples
        )
        temporary_rates[model] = _p95(
            row["temporary_peak_bytes"] for row in samples
        )
        completed[model] = _completed_for_commit(root, model, commit)
    p2_remaining = sum(
        max(0, 100 - completed[model]) * smoke_rates[model]
        for model in smoke_rates
    )
    p3 = 200 * test_rates["vggt"] + 200 * test_rates["mast3r"]
    zero_root = root / "stage" / "zero-update" / "bundles"
    zero_sizes: dict[str, list[int]] = {"vggt": [], "mast3r": []}
    if zero_root.exists():
        for model in zero_sizes:
            model_root = zero_root / model
            if model_root.exists():
                zero_sizes[model] = [
                    sum(
                        path.stat().st_size
                        for path in bundle.rglob("*")
                        if path.is_file() and not path.is_symlink()
                    )
                    for bundle in model_root.iterdir()
                    if bundle.is_dir() and not bundle.name.endswith(".partial")
                ]
    p5_rates = {
        model: _p95(sizes) or ZERO_UPDATE_RETAINED_FALLBACK_BYTES
        for model, sizes in zero_sizes.items()
    }
    p5 = 240 * p5_rates["vggt"] + 240 * p5_rates["mast3r"]
    transient_peak = max(temporary_rates.values(), default=0)
    planned_current = _planned_current_bytes(current_logical, actions)
    subtotal = planned_current + p2_remaining + p3 + p5 + transient_peak
    reserve = math.ceil(subtotal * 0.10)
    projected = subtotal + reserve
    remaining_retained = p2_remaining + p3 + p5
    return {
        "active_project_commit": commit or None,
        "current_after_plan_bytes": planned_current,
        "model_smoke_retained_byte_p95": smoke_rates,
        "model_test_retained_byte_p95": test_rates,
        "model_zero_update_retained_byte_p95": p5_rates,
        "model_temporary_peak_byte_p95": temporary_rates,
        "p2_completed_by_model": completed,
        "p2_remaining_retained_bytes": p2_remaining,
        "p3_retained_bytes": p3,
        "conditional_p5_retained_bytes": p5,
        "remaining_retained_bytes": remaining_retained,
        "remaining_retained_target_status": (
            "PASS"
            if remaining_retained < REMAINING_RETAINED_TARGET_BYTES
            else "MISS"
        ),
        "single_task_temporary_peak_bytes": transient_peak,
        "reserve_bytes": reserve,
        "full_worst_path_bytes": projected,
        "r1_status": "PASS" if projected < R1_LIMIT_BYTES else "FAIL",
        "r1_limit_bytes": R1_LIMIT_BYTES,
        "hard_limit_bytes": HARD_LIMIT_BYTES,
        "remaining_retained_target_bytes": REMAINING_RETAINED_TARGET_BYTES,
        "reporting_target_bytes": REPORTING_TARGET_BYTES,
        "reporting_target_status": (
            "PASS" if planned_current < REPORTING_TARGET_BYTES else "MISS"
        ),
        "l1_current_bytes": int(levels.get("L1", 0)),
        "projection_source": (
            "post-plan-current-plus-model-retained-p95-and-single-task-peak"
        ),
    }


def _action_digest(action: Mapping[str, Any]) -> str:
    return _sha_json(action)


def _action_receipt_path(
    root: Path,
    plan_file_sha256: str,
    index: int,
    action: Mapping[str, Any],
) -> Path:
    return (
        root
        / "artifacts"
        / "storage-retention"
        / "actions"
        / f"{index:06d}-{_action_digest(action)[:16]}.json"
    )


def _load_receipt(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise StorageRetentionError(f"invalid action receipt: {path}") from exc
    if not isinstance(payload, dict):
        raise StorageRetentionError(f"action receipt is not an object: {path}")
    return payload


def _receipt_payload(
    *,
    plan_file_sha256: str,
    action: Mapping[str, Any],
    status: str,
    result_sha256: str | None = None,
) -> dict[str, Any]:
    return {
        "schema_version": ACTION_RECEIPT_SCHEMA,
        "plan_file_sha256": plan_file_sha256,
        "action_sha256": _action_digest(action),
        "action": dict(action),
        "status": status,
        "result_sha256": result_sha256,
    }


def _validate_pass_receipt(
    path: Path,
    *,
    plan_file_sha256: str,
    action: Mapping[str, Any],
) -> dict[str, Any] | None:
    payload = _load_receipt(path)
    if payload is None:
        return None
    if (
        payload.get("schema_version") != ACTION_RECEIPT_SCHEMA
        or payload.get("plan_file_sha256") != plan_file_sha256
        or payload.get("action_sha256") != _action_digest(action)
    ):
        raise StorageRetentionError("action receipt binding mismatch")
    return payload if payload.get("status") == "PASS" else None


def _apply_reencode(
    root: Path,
    action: Mapping[str, Any],
    *,
    plan_file_sha256: str,
    receipt_path: Path,
) -> dict[str, Any]:
    source = _resolve_member(root, str(action["path"]))
    partial = source.with_name(source.name + ".storage-retention.partial")
    backup = source.with_name(source.name + ".storage-retention.original")
    passed = _validate_pass_receipt(
        receipt_path,
        plan_file_sha256=plan_file_sha256,
        action=action,
    )
    if passed is not None:
        if not source.is_file() or sha256_file(source) != passed["result_sha256"]:
            raise StorageRetentionError("retained NPZ drifted after PASS receipt")
        if backup.exists():
            backup.unlink()
        return passed
    pending = _load_receipt(receipt_path)
    if pending is not None and (
        pending.get("schema_version") != ACTION_RECEIPT_SCHEMA
        or pending.get("plan_file_sha256") != plan_file_sha256
        or pending.get("action_sha256") != _action_digest(action)
    ):
        raise StorageRetentionError("pending NPZ receipt binding mismatch")
    if backup.exists():
        if source.exists():
            if (
                pending is None
                or pending.get("status") != "READY_TO_REPLACE"
                or sha256_file(backup) != action["source_sha256"]
                or sha256_file(source) != pending.get("result_sha256")
            ):
                raise StorageRetentionError(
                    "ambiguous NPZ recovery state contains source and original backup"
                )
            assert_npz_equivalent(backup, source)
            payload = _receipt_payload(
                plan_file_sha256=plan_file_sha256,
                action=action,
                status="PASS",
                result_sha256=str(pending["result_sha256"]),
            )
            _atomic_json(receipt_path, payload)
            backup.unlink()
            if partial.exists():
                partial.unlink()
            return payload
        backup.replace(source)
    if partial.exists():
        partial.unlink()
    if not source.is_file() or sha256_file(source) != action["source_sha256"]:
        raise StorageRetentionError("NPZ source digest changed before retention")
    reencode_npz_lossless(source, partial)
    assert_npz_equivalent(source, partial)
    result_sha = sha256_file(partial)
    _atomic_json(
        receipt_path,
        _receipt_payload(
            plan_file_sha256=plan_file_sha256,
            action=action,
            status="READY_TO_REPLACE",
            result_sha256=result_sha,
        ),
    )
    try:
        source.replace(backup)
        partial.replace(source)
        assert_npz_equivalent(backup, source)
        if sha256_file(source) != result_sha:
            raise StorageRetentionError("retained NPZ digest changed during replace")
        payload = _receipt_payload(
            plan_file_sha256=plan_file_sha256,
            action=action,
            status="PASS",
            result_sha256=result_sha,
        )
        _atomic_json(receipt_path, payload)
        backup.unlink()
        return payload
    except Exception:
        if backup.exists():
            if source.exists():
                source.unlink()
            backup.replace(source)
        raise
    finally:
        if partial.exists():
            partial.unlink()


def _apply_delete(
    root: Path,
    action: Mapping[str, Any],
    *,
    plan_file_sha256: str,
    receipt_path: Path,
) -> dict[str, Any]:
    source = _resolve_member(root, str(action["path"]))
    quarantine = (
        root
        / "artifacts"
        / "storage-retention"
        / "quarantine"
        / _action_digest(action)
        / source.name
    )
    _resolve_member(root, quarantine.relative_to(root).as_posix())
    passed = _validate_pass_receipt(
        receipt_path,
        plan_file_sha256=plan_file_sha256,
        action=action,
    )
    if passed is not None:
        if source.exists():
            raise StorageRetentionError("deleted source reappeared after PASS receipt")
        if quarantine.exists():
            if sha256_file(quarantine) != action["source_sha256"]:
                raise StorageRetentionError("quarantined source digest drift")
            quarantine.unlink()
        return passed
    if action["action"] == "delete_verified_sparse_index" and source.exists():
        verify_sparse_index_preconditions(
            root,
            source,
            expected_source_sha256=str(action["source_sha256"]),
            expected_manifest_sha256=str(action["materialization_sha256"]),
        )
    if source.exists() and quarantine.exists():
        raise StorageRetentionError("ambiguous deletion recovery state")
    if source.exists():
        if not source.is_file() or sha256_file(source) != action["source_sha256"]:
            raise StorageRetentionError("ephemeral source digest changed")
        quarantine.parent.mkdir(parents=True, exist_ok=True)
        _atomic_json(
            receipt_path,
            _receipt_payload(
                plan_file_sha256=plan_file_sha256,
                action=action,
                status="READY_TO_REMOVE",
            ),
        )
        source.replace(quarantine)
    if not quarantine.is_file() or sha256_file(quarantine) != action["source_sha256"]:
        raise StorageRetentionError("quarantined source is missing or invalid")
    payload = _receipt_payload(
        plan_file_sha256=plan_file_sha256,
        action=action,
        status="PASS",
    )
    _atomic_json(receipt_path, payload)
    quarantine.unlink()
    return payload


def apply_retention_actions(
    root: Path,
    actions: Iterable[Mapping[str, Any]],
    *,
    plan_file_sha256: str,
) -> list[dict[str, Any]]:
    """Apply ordered actions idempotently with per-action recovery receipts."""

    receipts: list[dict[str, Any]] = []
    for index, raw_action in enumerate(actions):
        action = dict(raw_action)
        kind = action.get("action")
        if kind not in {
            "lossless_reencode_npz",
            "delete_unreferenced_ephemeral",
            "delete_verified_sparse_index",
        }:
            raise StorageRetentionError(f"unsupported retention action: {kind}")
        receipt_path = _action_receipt_path(
            root,
            plan_file_sha256,
            index,
            action,
        )
        if kind == "lossless_reencode_npz":
            payload = _apply_reencode(
                root,
                action,
                plan_file_sha256=plan_file_sha256,
                receipt_path=receipt_path,
            )
        else:
            payload = _apply_delete(
                root,
                action,
                plan_file_sha256=plan_file_sha256,
                receipt_path=receipt_path,
            )
        receipts.append(
            {
                "path": str(receipt_path),
                "sha256": sha256_file(receipt_path),
                "status": payload["status"],
                "action_sha256": payload["action_sha256"],
            }
        )
    return receipts
