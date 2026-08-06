"""Storage-refactor rollout, superseded-result, P2-A, and GPU governance.

The functions in this module are intentionally outside the scientific artifact
contract.  They may archive superseded engineering runs, select a non-scientific
smoke canary, and bind an operator-selected GPU to one exact project commit.
They must never change the frozen schedule, thresholds, split, or corruption
parameters.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
from typing import Any, Iterable, Mapping, Sequence

from .artifact_storage import (
    ArtifactStorageError,
    sha256_file,
    validate_retention_receipt,
    validate_tar_gz_inventory,
    write_deterministic_tar_gz_inventory,
)
from .science_lock import BASE_PROJECT_COMMIT, ScienceLockError, validate_science_lock


ARCHIVE_SCHEMA = "georeliab-superseded-engineering-archive-v1"
ARCHIVE_INVENTORY_SCHEMA = "georeliab-superseded-inventory-v1"
GPU_SELECTION_SCHEMA = "georeliab-explicit-gpu-selection-v1"
P2A_SELECTION_SCHEMA = "georeliab-p2a-selection-v1"
P2A_COMPLETION_SCHEMA = "georeliab-p2a-completion-v1"
ROLLOUT_VALIDATION_SCHEMA = "georeliab-storage-refactor-validation-v1"
P2A_SELECTOR_VERSION = "GeoReliab-P2A-v1"
P2A_SELECTED_PER_MODEL = 25
P2A_SELECTED_TOTAL = 50
P2_FULL_TOTAL = 200
R1_LIMIT_BYTES = 900_000_000_000
HARD_STORAGE_LIMIT_BYTES = 1_000_000_000_000


class ExecutionGovernanceError(RuntimeError):
    """Raised when an execution-governance binding cannot be proven."""


def _json_bytes(payload: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(dict(payload), sort_keys=True, separators=(",", ":"), default=str)
        + "\n"
    ).encode("utf-8")


def _sha_json(payload: Any) -> str:
    raw = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ExecutionGovernanceError(f"cannot read governance JSON {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ExecutionGovernanceError(f"governance JSON is not an object: {path}")
    return payload


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".partial")
    try:
        with partial.open("wb") as handle:
            handle.write(_json_bytes(payload))
            handle.flush()
            try:
                os.fsync(handle.fileno())
            except OSError:
                pass
        partial.replace(path)
    finally:
        try:
            partial.unlink()
        except FileNotFoundError:
            pass


def _write_immutable_json(path: Path, payload: Mapping[str, Any]) -> None:
    if path.exists():
        existing = _load_json(path)
        if existing != dict(payload):
            raise ExecutionGovernanceError(f"immutable governance artifact mismatch: {path}")
        return
    _atomic_json(path, payload)


def _resolve_under_root(
    root: Path,
    target: Path,
    *,
    must_exist: bool = False,
) -> Path:
    resolved_root = root.resolve()
    if resolved_root == Path(resolved_root.anchor):
        raise ExecutionGovernanceError("runtime root must not be a filesystem root")
    resolved = target.resolve(strict=must_exist)
    try:
        relative = resolved.relative_to(resolved_root)
    except ValueError as exc:
        raise ExecutionGovernanceError(
            f"governance target escaped runtime root: {resolved}"
        ) from exc
    if not relative.parts:
        raise ExecutionGovernanceError("governance may not target the runtime root")
    return resolved


def _git_identity(source_root: Path) -> tuple[str, str]:
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=source_root,
            text=True,
            timeout=10,
        ).strip()
        tree = subprocess.check_output(
            ["git", "rev-parse", "HEAD^{tree}"],
            cwd=source_root,
            text=True,
            timeout=10,
        ).strip()
    except (OSError, subprocess.SubprocessError) as exc:
        raise ExecutionGovernanceError("cannot bind governance artifact to Git") from exc
    if len(commit) != 40 or len(tree) != 40:
        raise ExecutionGovernanceError("Git identity is not a full commit/tree SHA")
    return commit.lower(), tree.lower()


def _manifest_project_commit(path: Path) -> str:
    payload = _load_json(path)
    provenance = payload.get("provenance")
    if not isinstance(provenance, Mapping):
        return ""
    return str(provenance.get("project_commit", "")).lower()


def _superseded_manifest_paths(root: Path, commit: str) -> list[Path]:
    roots = (
        root / "preflight-real",
        root / "stage" / "smoke",
    )
    paths: list[Path] = []
    for source in roots:
        if not source.exists():
            continue
        for manifest in sorted(source.rglob("run_manifest.json")):
            if manifest.is_symlink():
                raise ExecutionGovernanceError(
                    f"superseded manifest may not be a symlink: {manifest}"
                )
            if _manifest_project_commit(manifest) == commit:
                paths.append(manifest)
    return paths


def _archive_paths(root: Path) -> tuple[Path, Path, Path]:
    directory = (
        root
        / "engineering-failures"
        / BASE_PROJECT_COMMIT
        / "storage-refactor"
    )
    return (
        directory / "superseded-p1-p2.tar.gz",
        directory / "superseded-p1-p2.inventory.json",
        directory / "superseded-p1-p2.receipt.json",
    )


def _archive_source_roots(root: Path) -> tuple[Path, ...]:
    candidates = (
        root / "preflight-real",
        root / "stage" / "smoke",
        root / "artifacts" / "p1_preflight.json",
    )
    return tuple(path for path in candidates if path.exists())


def _inventory_for_paths(root: Path, sources: Sequence[Path]) -> list[dict[str, Any]]:
    resolved_root = root.resolve()
    files: list[Path] = []
    for source in sources:
        resolved = _resolve_under_root(resolved_root, source, must_exist=True)
        if resolved.is_symlink():
            raise ExecutionGovernanceError(
                f"superseded archive source may not be a symlink: {resolved}"
            )
        if resolved.is_file():
            files.append(resolved)
            continue
        if not resolved.is_dir():
            raise ExecutionGovernanceError(
                f"superseded archive source is not a file or directory: {resolved}"
            )
        for member in sorted(resolved.rglob("*")):
            if member.is_symlink():
                raise ExecutionGovernanceError(
                    f"superseded archive member may not be a symlink: {member}"
                )
            if member.is_file():
                files.append(member.resolve())
    unique = sorted(set(files))
    return [
        {
            "relative_path": path.relative_to(resolved_root).as_posix(),
            "sha256": sha256_file(path),
            "bytes": path.stat().st_size,
        }
        for path in unique
    ]


def _validate_remaining_sources(
    root: Path,
    sources: Sequence[Path],
    inventory: Sequence[Mapping[str, Any]],
) -> None:
    expected = {str(row["relative_path"]): dict(row) for row in inventory}
    for row in _inventory_for_paths(root, tuple(path for path in sources if path.exists())):
        bound = expected.get(str(row["relative_path"]))
        if bound != row:
            raise ExecutionGovernanceError(
                f"superseded source changed after archive creation: {row['relative_path']}"
            )


def _remove_archived_source(root: Path, source: Path) -> None:
    resolved = _resolve_under_root(root, source, must_exist=True)
    if resolved.is_symlink():
        raise ExecutionGovernanceError(
            f"refusing to remove symlinked superseded source: {resolved}"
        )
    if resolved.is_dir():
        shutil.rmtree(resolved)
    else:
        resolved.unlink()


def _validate_superseded_counts(
    root: Path,
    *,
    expected_commit: str,
    expected_p1_items: int,
    expected_p2_items: int,
) -> tuple[int, int]:
    p1_paths = sorted((root / "preflight-real").rglob("run_manifest.json"))
    p2_paths = sorted((root / "stage" / "smoke").glob("bundles/*/*/run_manifest.json"))
    if len(p1_paths) != expected_p1_items or len(p2_paths) != expected_p2_items:
        raise ExecutionGovernanceError(
            "superseded grid count mismatch: "
            f"P1={len(p1_paths)}/{expected_p1_items}, "
            f"P2={len(p2_paths)}/{expected_p2_items}"
        )
    for path in (*p1_paths, *p2_paths):
        actual = _manifest_project_commit(path)
        if actual != expected_commit:
            raise ExecutionGovernanceError(
                f"superseded bundle commit mismatch: {path}: {actual} != {expected_commit}"
            )
    return len(p1_paths), len(p2_paths)


def validate_superseded_archive(
    root: Path,
    *,
    expected_commit: str = BASE_PROJECT_COMMIT,
) -> dict[str, Any]:
    resolved_root = root.resolve()
    archive_path, inventory_path, receipt_path = _archive_paths(resolved_root)
    receipt = _load_json(receipt_path)
    if (
        receipt.get("schema_version") != ARCHIVE_SCHEMA
        or receipt.get("status") != "PASS"
        or receipt.get("superseded_project_commit") != expected_commit
    ):
        raise ExecutionGovernanceError("superseded archive receipt is not terminal PASS")
    if (
        not archive_path.is_file()
        or sha256_file(archive_path) != receipt.get("archive_sha256")
        or not inventory_path.is_file()
        or sha256_file(inventory_path) != receipt.get("inventory_sha256")
    ):
        raise ExecutionGovernanceError("superseded archive digest binding failed")
    inventory_payload = _load_json(inventory_path)
    inventory = inventory_payload.get("members")
    if (
        inventory_payload.get("schema_version") != ARCHIVE_INVENTORY_SCHEMA
        or not isinstance(inventory, list)
        or any(not isinstance(row, dict) for row in inventory)
    ):
        raise ExecutionGovernanceError("superseded archive inventory is invalid")
    try:
        validate_tar_gz_inventory(archive_path, inventory)
    except ArtifactStorageError as exc:
        raise ExecutionGovernanceError(str(exc)) from exc
    if _superseded_manifest_paths(resolved_root, expected_commit):
        raise ExecutionGovernanceError(
            "superseded commit still appears in canonical P1/P2 paths"
        )
    return receipt


def superseded_archive_status(
    root: Path,
    *,
    expected_commit: str = BASE_PROJECT_COMMIT,
) -> dict[str, Any]:
    _archive_path, _inventory_path, receipt_path = _archive_paths(root.resolve())
    if receipt_path.exists():
        try:
            receipt = validate_superseded_archive(
                root,
                expected_commit=expected_commit,
            )
        except ExecutionGovernanceError as exc:
            return {
                "status": "BLOCKED",
                "reason_code": "SUPERSEDED_ARCHIVE_INVALID",
                "detail": str(exc),
            }
        return {
            "status": "PASS",
            "reason_code": "SUPERSEDED_ARCHIVE_COMPLETE",
            "receipt_path": str(receipt_path),
            "receipt_sha256": sha256_file(receipt_path),
            "archive_sha256": receipt["archive_sha256"],
        }
    if _superseded_manifest_paths(root.resolve(), expected_commit):
        return {
            "status": "BLOCKED",
            "reason_code": "SUPERSEDED_ARCHIVE_REQUIRED",
        }
    return {
        "status": "PASS",
        "reason_code": "SUPERSEDED_ARCHIVE_NOT_REQUIRED",
    }


def archive_superseded_results(
    root: Path,
    *,
    expected_commit: str = BASE_PROJECT_COMMIT,
    expected_p1_items: int = 16,
    expected_p2_items: int = 75,
) -> dict[str, Any]:
    """Archive exact old member bytes before removing old canonical paths."""

    resolved_root = root.resolve()
    archive_path, inventory_path, receipt_path = _archive_paths(resolved_root)
    if receipt_path.exists():
        existing = _load_json(receipt_path)
        if existing.get("status") == "PASS":
            return validate_superseded_archive(
                resolved_root,
                expected_commit=expected_commit,
            )
    else:
        existing = {}

    if existing:
        if (
            existing.get("schema_version") != ARCHIVE_SCHEMA
            or existing.get("status") != "READY_TO_REMOVE"
            or existing.get("superseded_project_commit") != expected_commit
        ):
            raise ExecutionGovernanceError("superseded archive recovery receipt is invalid")
        inventory_payload = _load_json(inventory_path)
        inventory = inventory_payload.get("members")
        if not isinstance(inventory, list):
            raise ExecutionGovernanceError("superseded archive recovery inventory is invalid")
        if (
            sha256_file(archive_path) != existing.get("archive_sha256")
            or sha256_file(inventory_path) != existing.get("inventory_sha256")
        ):
            raise ExecutionGovernanceError("superseded archive recovery digest mismatch")
        try:
            validate_tar_gz_inventory(archive_path, inventory)
        except ArtifactStorageError as exc:
            raise ExecutionGovernanceError(str(exc)) from exc
        sources = tuple(resolved_root / value for value in existing["source_paths"])
    else:
        p1_count, p2_count = _validate_superseded_counts(
            resolved_root,
            expected_commit=expected_commit,
            expected_p1_items=expected_p1_items,
            expected_p2_items=expected_p2_items,
        )
        sources = _archive_source_roots(resolved_root)
        if not sources:
            raise ExecutionGovernanceError("no superseded P1/P2 sources found")
        inventory = _inventory_for_paths(resolved_root, sources)
        try:
            write_deterministic_tar_gz_inventory(
                resolved_root,
                archive_path,
                inventory,
            )
            validate_tar_gz_inventory(archive_path, inventory)
        except ArtifactStorageError as exc:
            raise ExecutionGovernanceError(str(exc)) from exc
        inventory_payload = {
            "schema_version": ARCHIVE_INVENTORY_SCHEMA,
            "superseded_project_commit": expected_commit,
            "p1_item_count": p1_count,
            "p2_item_count": p2_count,
            "members": inventory,
        }
        _write_immutable_json(inventory_path, inventory_payload)
        existing = {
            "schema_version": ARCHIVE_SCHEMA,
            "status": "READY_TO_REMOVE",
            "superseded_project_commit": expected_commit,
            "p1_item_count": p1_count,
            "p2_item_count": p2_count,
            "archive_path": str(archive_path),
            "archive_sha256": sha256_file(archive_path),
            "inventory_path": str(inventory_path),
            "inventory_sha256": sha256_file(inventory_path),
            "member_count": len(inventory),
            "source_paths": [
                path.relative_to(resolved_root).as_posix() for path in sources
            ],
        }
        _atomic_json(receipt_path, existing)

    _validate_remaining_sources(resolved_root, sources, inventory)
    for source in sources:
        if source.exists():
            _remove_archived_source(resolved_root, source)
    passed = dict(existing)
    passed["status"] = "PASS"
    _atomic_json(receipt_path, passed)
    return validate_superseded_archive(
        resolved_root,
        expected_commit=expected_commit,
    )


def _selection_hash(identity: str) -> str:
    return hashlib.sha256(
        f"{P2A_SELECTOR_VERSION}:{identity}".encode("utf-8")
    ).hexdigest()


def select_p2a_items(items: Sequence[Any]) -> tuple[Any, ...]:
    """Select exactly 25 items per model from the full frozen P2 schedule."""

    if len(items) != P2_FULL_TOTAL:
        raise ExecutionGovernanceError(
            f"P2-A requires the complete 200-item P2 schedule, got {len(items)}"
        )
    selected: list[Any] = []
    for model in ("VGGT", "MASt3R"):
        candidates = [item for item in items if str(item.model) == model]
        if len(candidates) != 100:
            raise ExecutionGovernanceError(
                f"P2-A requires 100 {model} items, got {len(candidates)}"
            )
        ranked = sorted(
            candidates,
            key=lambda item: (_selection_hash(str(item.identity)), str(item.identity)),
        )
        selected.extend(ranked[:P2A_SELECTED_PER_MODEL])
    return tuple(sorted(selected, key=lambda item: str(item.identity)))


def _canonical_schedule_fingerprint(items: Sequence[Any]) -> str:
    return _sha_json([str(item.identity) for item in items])


def create_p2a_selection_manifest(
    root: Path,
    *,
    source_root: Path,
    storage_before_path: Path | None = None,
    storage_plan_path: Path | None = None,
    output_path: Path | None = None,
) -> dict[str, Any]:
    from .runner import full_schedule

    resolved_root = root.resolve()
    storage_before_path = storage_before_path or resolved_root / "artifacts" / "storage_before.json"
    storage_plan_path = storage_plan_path or resolved_root / "artifacts" / "storage_plan.json"
    output_path = output_path or resolved_root / "artifacts" / "p2a_selection_manifest.json"
    _resolve_under_root(resolved_root, storage_before_path, must_exist=True)
    _resolve_under_root(resolved_root, storage_plan_path, must_exist=True)
    _resolve_under_root(resolved_root, output_path)
    if not output_path.exists():
        smoke_root = resolved_root / "stage" / "smoke"
        ledger = smoke_root / "ledger.jsonl"
        evidence: list[str] = []
        if ledger.is_file() and any(
            line.strip()
            for line in ledger.read_text(encoding="utf-8").splitlines()
        ):
            evidence.append(str(ledger))
        for directory in (
            smoke_root / "bundles",
            smoke_root / "claims",
        ):
            if directory.exists() and any(
                member.is_file() for member in directory.rglob("*")
            ):
                evidence.append(str(directory))
        if evidence:
            raise ExecutionGovernanceError(
                "P2-A selection must be frozen before any new-commit smoke "
                f"execution: {evidence}"
            )
    snapshot = _load_json(storage_before_path)
    plan = _load_json(storage_plan_path)
    projection = snapshot.get("projection")
    if not isinstance(projection, Mapping) or projection.get("r1_status") != "PASS":
        raise ExecutionGovernanceError("P2-A requires a PASS storage projection")
    full_path = int(projection.get("full_worst_path_bytes", 0))
    if full_path <= 0 or full_path >= R1_LIMIT_BYTES:
        raise ExecutionGovernanceError("P2-A full-path projection is not below 900 GB")
    rates = projection.get("model_smoke_retained_byte_p95")
    if not isinstance(rates, Mapping):
        raise ExecutionGovernanceError("P2-A storage projection lacks model retained rates")
    predicted_by_model = {
        model: P2A_SELECTED_PER_MODEL * int(rates.get(model, 0))
        for model in ("vggt", "mast3r")
    }
    if any(value <= 0 for value in predicted_by_model.values()):
        raise ExecutionGovernanceError("P2-A model retained-byte prediction is unavailable")
    full = full_schedule(resolved_root, "smoke")
    selected = select_p2a_items(full)
    commit, tree = _git_identity(source_root.resolve())
    payload = {
        "schema_version": P2A_SELECTION_SCHEMA,
        "stage": "smoke",
        "scientific_validity": "NON_SCIENTIFIC_SMOKE",
        "selector_version": P2A_SELECTOR_VERSION,
        "project_commit": commit,
        "project_tree": tree,
        "full_schedule_count": len(full),
        "full_schedule_fingerprint": _canonical_schedule_fingerprint(full),
        "selected_count": len(selected),
        "selected_per_model": {
            "vggt": sum(item.model == "VGGT" for item in selected),
            "mast3r": sum(item.model == "MASt3R" for item in selected),
        },
        "selected_items": [
            {
                "identity": str(item.identity),
                "model": str(item.model),
                "selection_hash": _selection_hash(str(item.identity)),
            }
            for item in selected
        ],
        "storage_before_path": str(storage_before_path),
        "storage_before_sha256": sha256_file(storage_before_path),
        "storage_plan_path": str(storage_plan_path),
        "storage_plan_sha256": sha256_file(storage_plan_path),
        "storage_plan_payload_sha256": plan.get("plan_payload_sha256"),
        "predicted_retained_bytes_by_model": predicted_by_model,
        "predicted_retained_bytes": sum(predicted_by_model.values()),
        "full_worst_path_bytes": full_path,
        "r1_limit_bytes": R1_LIMIT_BYTES,
    }
    _write_immutable_json(output_path, payload)
    return payload


def load_p2a_selection_manifest(
    root: Path,
    path: Path,
    *,
    source_root: Path | None = None,
) -> tuple[tuple[Any, ...], dict[str, Any]]:
    from .runner import full_schedule

    resolved_root = root.resolve()
    resolved_path = _resolve_under_root(resolved_root, path, must_exist=True)
    payload = _load_json(resolved_path)
    if (
        payload.get("schema_version") != P2A_SELECTION_SCHEMA
        or payload.get("stage") != "smoke"
        or payload.get("scientific_validity") != "NON_SCIENTIFIC_SMOKE"
        or payload.get("selector_version") != P2A_SELECTOR_VERSION
    ):
        raise ExecutionGovernanceError("selection manifest is not a P2-A smoke manifest")
    source_root = (source_root or Path(__file__).resolve().parents[1]).resolve()
    commit, tree = _git_identity(source_root)
    if (
        payload.get("project_commit") != commit
        or payload.get("project_tree") != tree
    ):
        raise ExecutionGovernanceError("P2-A selection is bound to another project commit")
    full = full_schedule(resolved_root, "smoke")
    expected = select_p2a_items(full)
    if (
        payload.get("full_schedule_count") != P2_FULL_TOTAL
        or payload.get("full_schedule_fingerprint")
        != _canonical_schedule_fingerprint(full)
        or payload.get("selected_count") != P2A_SELECTED_TOTAL
    ):
        raise ExecutionGovernanceError("P2-A selection is not bound to the full P2 schedule")
    expected_rows = [
        {
            "identity": str(item.identity),
            "model": str(item.model),
            "selection_hash": _selection_hash(str(item.identity)),
        }
        for item in expected
    ]
    if payload.get("selected_items") != expected_rows:
        raise ExecutionGovernanceError("P2-A selected identities do not match the frozen rule")
    selected_per_model = payload.get("selected_per_model")
    if selected_per_model != {"vggt": 25, "mast3r": 25}:
        raise ExecutionGovernanceError("P2-A must contain 25 items per model")
    if int(payload.get("full_worst_path_bytes", 0)) >= R1_LIMIT_BYTES:
        raise ExecutionGovernanceError("P2-A is bound to a failing storage projection")
    return expected, payload


def _bundle_size(path: Path) -> int:
    return sum(
        member.stat().st_size
        for member in path.rglob("*")
        if member.is_file() and not member.is_symlink()
    )


def evaluate_p2a_completion(
    root: Path,
    *,
    selection_path: Path | None = None,
    source_root: Path | None = None,
    output_path: Path | None = None,
    write_terminal: bool = True,
) -> dict[str, Any]:
    from .runner import load_completed_bundle, read_stage_ledger

    resolved_root = root.resolve()
    selection_path = selection_path or resolved_root / "artifacts" / "p2a_selection_manifest.json"
    output_path = output_path or resolved_root / "artifacts" / "p2a_completion.json"
    selected, manifest = load_p2a_selection_manifest(
        resolved_root,
        selection_path,
        source_root=source_root,
    )
    rows = read_stage_ledger(resolved_root, "smoke")["rows"]
    selection_sha256 = sha256_file(selection_path)
    governed_rows = rows
    if output_path.exists():
        recorded = _load_json(output_path)
        if recorded.get("status") != "PASS":
            return recorded
        try:
            boundary_count = int(recorded["ledger_boundary_row_count"])
            boundary_sha256 = str(recorded["ledger_boundary_sha256"])
        except (KeyError, TypeError, ValueError):
            return {
                "schema_version": P2A_COMPLETION_SCHEMA,
                "status": "BLOCKED",
                "reason_code": "P2A_LEDGER_BOUNDARY_INVALID",
            }
        if (
            recorded.get("schema_version") != P2A_COMPLETION_SCHEMA
            or recorded.get("selection_manifest_sha256") != selection_sha256
            or boundary_count < 0
            or boundary_count > len(rows)
            or _sha_json(rows[:boundary_count]) != boundary_sha256
        ):
            return {
                "schema_version": P2A_COMPLETION_SCHEMA,
                "status": "BLOCKED",
                "reason_code": "P2A_LEDGER_BOUNDARY_INVALID",
            }
        governed_rows = rows[:boundary_count]
    selected_ids = {str(item.identity) for item in selected}
    unexpected = sorted(
        {
            str(row.get("item_identity", row.get("identity", "")))
            for row in governed_rows
            if str(row.get("item_identity", row.get("identity", "")))
            not in selected_ids
            and row.get("state") in {"completed", "skipped", "failed"}
        }
        - {""}
    )
    if unexpected:
        payload = {
            "schema_version": P2A_COMPLETION_SCHEMA,
            "status": "FAIL",
            "reason_code": "P2A_NON_SELECTED_SMOKE_EXECUTION",
            "unexpected_count": len(unexpected),
            "unexpected_identities": unexpected,
            "selection_manifest_sha256": selection_sha256,
        }
        if write_terminal:
            _write_immutable_json(output_path, payload)
        return payload
    latest: dict[str, dict[str, Any]] = {}
    for index, row in enumerate(governed_rows):
        latest[str(row.get("item_identity", row.get("identity", index)))] = row
    missing = [
        item.identity
        for item in selected
        if item.identity not in latest
    ]
    failed = [
        item.identity
        for item in selected
        if item.identity in latest
        and latest[item.identity].get("state") not in {"completed", "skipped"}
    ]
    if failed:
        payload = {
            "schema_version": P2A_COMPLETION_SCHEMA,
            "status": "FAIL",
            "reason_code": "P2A_EXECUTION_FAILURE",
            "failed_count": len(failed),
            "failed_identities": failed,
            "selection_manifest_sha256": sha256_file(selection_path),
        }
        if write_terminal:
            _write_immutable_json(output_path, payload)
        return payload
    if missing:
        return {
            "schema_version": P2A_COMPLETION_SCHEMA,
            "status": "BLOCKED_PENDING_EVIDENCE",
            "reason_code": "P2A_INCOMPLETE",
            "completed": P2A_SELECTED_TOTAL - len(missing),
            "missing": len(missing),
            "selection_manifest_sha256": sha256_file(selection_path),
        }
    invalid = 0
    actual_by_model = {"vggt": 0, "mast3r": 0}
    receipt_count = 0
    try:
        for item in selected:
            row = latest[item.identity]
            reason_code = str(row.get("reason_code", ""))
            if (
                reason_code == "ADAPTER_EXCEPTION"
                or reason_code.startswith("RUNNER_EXCEPTION")
            ):
                raise ExecutionGovernanceError(
                    f"P2-A contains an implementation exception: {item.identity}"
                )
            bundle = (
                resolved_root
                / "stage"
                / "smoke"
                / "bundles"
                / item.model.lower()
                / item.identity
            )
            load_completed_bundle(bundle)
            if item.model == "MASt3R":
                validate_retention_receipt(bundle)
                receipt_count += 1
            actual_by_model[item.model.lower()] += _bundle_size(bundle)
            if row.get("invalid_prediction") is True:
                invalid += 1
    except (OSError, ArtifactStorageError, ExecutionGovernanceError) as exc:
        payload = {
            "schema_version": P2A_COMPLETION_SCHEMA,
            "status": "FAIL",
            "reason_code": "P2A_BUNDLE_OR_RETENTION_INVALID",
            "detail": str(exc),
            "selection_manifest_sha256": sha256_file(selection_path),
        }
        if write_terminal:
            _write_immutable_json(output_path, payload)
        return payload
    predicted = int(manifest.get("predicted_retained_bytes", 0))
    actual = sum(actual_by_model.values())
    error = abs(actual - predicted) / predicted if predicted > 0 else float("inf")
    full_path = int(manifest.get("full_worst_path_bytes", 0))
    status = "PASS"
    reason = "P2A_COMPLETE"
    if error > 0.10:
        status, reason = "FAIL", "P2A_RETAINED_BYTES_PREDICTION_MISS"
    elif full_path >= R1_LIMIT_BYTES:
        status, reason = "FAIL", "P2A_R1_STORAGE_PROJECTION_FAILED"
    payload = {
        "schema_version": P2A_COMPLETION_SCHEMA,
        "status": status,
        "reason_code": reason,
        "scheduled": P2A_SELECTED_TOTAL,
        "completed": P2A_SELECTED_TOTAL,
        "missing": 0,
        "invalid": invalid,
        "mast3r_retention_receipts_validated": receipt_count,
        "selection_manifest_path": str(selection_path),
        "selection_manifest_sha256": selection_sha256,
        "ledger_boundary_row_count": len(governed_rows),
        "ledger_boundary_sha256": _sha_json(governed_rows),
        "full_schedule_count": P2_FULL_TOTAL,
        "full_schedule_fingerprint": manifest["full_schedule_fingerprint"],
        "actual_retained_bytes_by_model": actual_by_model,
        "actual_retained_bytes": actual,
        "predicted_retained_bytes": predicted,
        "retained_bytes_relative_error": error,
        "full_worst_path_bytes": full_path,
        "r1_limit_bytes": R1_LIMIT_BYTES,
    }
    if write_terminal:
        _write_immutable_json(output_path, payload)
    return payload


def p2a_completion_status(
    root: Path,
    *,
    source_root: Path | None = None,
) -> dict[str, Any]:
    output = root.resolve() / "artifacts" / "p2a_completion.json"
    if not output.exists():
        return {
            "status": "BLOCKED_PENDING_EVIDENCE",
            "reason_code": "P2A_COMPLETION_MISSING",
        }
    expected = evaluate_p2a_completion(
        root,
        source_root=source_root,
        write_terminal=False,
    )
    recorded = _load_json(output)
    if expected != recorded:
        return {
            "status": "BLOCKED",
            "reason_code": "P2A_COMPLETION_REVALIDATION_FAILED",
        }
    return recorded


def _gpu_selection_path(root: Path, project_commit: str) -> Path:
    return root / "artifacts" / "gpu-selections" / f"{project_commit}.json"


def record_gpu_selection(
    root: Path,
    *,
    source_root: Path,
    project_commit: str,
    device: str,
    explicit_user_selection: bool,
) -> dict[str, Any]:
    if not explicit_user_selection:
        raise ExecutionGovernanceError("GPU_SELECTION_REQUIRED")
    if device not in {"cuda:0", "cuda:1"}:
        raise ExecutionGovernanceError("selected GPU must be cuda:0 or cuda:1")
    commit, tree = _git_identity(source_root.resolve())
    if project_commit != commit:
        raise ExecutionGovernanceError(
            f"GPU selection commit mismatch: {project_commit} != {commit}"
        )
    payload = {
        "schema_version": GPU_SELECTION_SCHEMA,
        "status": "PASS",
        "authorization": "EXPLICIT_USER_SELECTION",
        "project_commit": commit,
        "project_tree": tree,
        "device": device,
        "max_concurrent_gpus": 1,
        "multi_shard_execution": "SEQUENTIAL",
        "supersedes_prior_commit_gpu_selections": True,
    }
    path = _gpu_selection_path(root.resolve(), commit)
    _resolve_under_root(root.resolve(), path)
    _write_immutable_json(path, payload)
    return payload


def validate_gpu_selection(
    root: Path,
    *,
    source_root: Path,
    project_commit: str,
    device: str,
) -> dict[str, Any]:
    commit, tree = _git_identity(source_root.resolve())
    if project_commit != commit:
        raise ExecutionGovernanceError("GPU selection request is not for the deployed commit")
    path = _gpu_selection_path(root.resolve(), commit)
    if not path.is_file():
        raise ExecutionGovernanceError("GPU_SELECTION_REQUIRED")
    payload = _load_json(path)
    expected = {
        "schema_version": GPU_SELECTION_SCHEMA,
        "status": "PASS",
        "authorization": "EXPLICIT_USER_SELECTION",
        "project_commit": commit,
        "project_tree": tree,
        "device": device,
        "max_concurrent_gpus": 1,
        "multi_shard_execution": "SEQUENTIAL",
        "supersedes_prior_commit_gpu_selections": True,
    }
    if payload != expected:
        raise ExecutionGovernanceError(
            "GPU_SELECTION_REQUIRED: exact-commit/device receipt mismatch"
        )
    return payload


def gpu_selection_status(
    root: Path,
    *,
    source_root: Path,
    project_commit: str | None = None,
    device: str | None = None,
) -> dict[str, Any]:
    commit, _tree = _git_identity(source_root.resolve())
    if project_commit is not None and project_commit != commit:
        return {
            "status": "GPU_SELECTION_REQUIRED",
            "reason_code": "GPU_SELECTION_COMMIT_MISMATCH",
        }
    path = _gpu_selection_path(root.resolve(), commit)
    if not path.exists():
        return {
            "status": "GPU_SELECTION_REQUIRED",
            "reason_code": "EXPLICIT_CURRENT_COMMIT_GPU_SELECTION_MISSING",
            "project_commit": commit,
        }
    payload = _load_json(path)
    selected = str(payload.get("device", ""))
    if device is not None and selected != device:
        return {
            "status": "GPU_SELECTION_REQUIRED",
            "reason_code": "GPU_SELECTION_DEVICE_MISMATCH",
            "project_commit": commit,
        }
    try:
        validate_gpu_selection(
            root,
            source_root=source_root,
            project_commit=commit,
            device=selected,
        )
    except ExecutionGovernanceError as exc:
        return {
            "status": "GPU_SELECTION_REQUIRED",
            "reason_code": "GPU_SELECTION_RECEIPT_INVALID",
            "detail": str(exc),
            "project_commit": commit,
        }
    return {
        "status": "PASS",
        "reason_code": "EXPLICIT_GPU_SELECTION_VALID",
        "project_commit": commit,
        "device": selected,
        "receipt_path": str(path),
        "receipt_sha256": sha256_file(path),
    }


def validate_storage_refactor_rollout(
    root: Path,
    *,
    source_root: Path,
    output_path: Path | None = None,
) -> dict[str, Any]:
    """Validate R1-R4 without selecting a GPU or starting an experiment."""

    from .storage_audit import scan_storage

    resolved_root = root.resolve()
    output_path = output_path or resolved_root / "artifacts" / "storage_refactor_validation.json"
    storage_before_path = resolved_root / "artifacts" / "storage_before.json"
    storage_plan_path = resolved_root / "artifacts" / "storage_plan.json"
    apply_receipt_path = resolved_root / "artifacts" / "storage_apply_receipt.json"
    snapshot = _load_json(storage_before_path)
    plan = _load_json(storage_plan_path)
    apply_receipt = _load_json(apply_receipt_path)
    rows, errors = scan_storage(resolved_root)
    actual_logical = sum(row.logical_bytes for row in rows)
    projection = snapshot.get("projection")
    r1_pass = bool(
        not errors
        and isinstance(projection, Mapping)
        and projection.get("r1_status") == "PASS"
        and 0 < int(projection.get("full_worst_path_bytes", 0)) < R1_LIMIT_BYTES
        and actual_logical < HARD_STORAGE_LIMIT_BYTES
    )
    try:
        science_lock = validate_science_lock(source_root.resolve())
        r2_pass = science_lock.get("status") == "PASS"
    except ScienceLockError as exc:
        science_lock = {"status": "FAIL", "detail": str(exc)}
        r2_pass = False
    plan_sha = sha256_file(storage_plan_path)
    action_receipts = apply_receipt.get("action_receipts")
    r3_pass = bool(
        apply_receipt.get("status") == "PASS"
        and apply_receipt.get("plan_file_sha256") == plan_sha
        and plan.get("runtime_root") == str(resolved_root)
        and isinstance(action_receipts, list)
        and int(apply_receipt.get("actions_applied", -1)) == len(action_receipts)
    )
    if r3_pass:
        for row in action_receipts:
            try:
                receipt_path = _resolve_under_root(
                    resolved_root,
                    Path(str(row["path"])),
                    must_exist=True,
                )
                receipt = _load_json(receipt_path)
                if (
                    row.get("status") != "PASS"
                    or receipt.get("status") != "PASS"
                    or sha256_file(receipt_path) != row.get("sha256")
                ):
                    r3_pass = False
                    break
            except (KeyError, OSError, ExecutionGovernanceError):
                r3_pass = False
                break
    commit, tree = _git_identity(source_root.resolve())
    launch_text = (source_root / "scripts" / "a100" / "launch_stage.sh").read_text(
        encoding="utf-8"
    )
    gpu_status = gpu_selection_status(
        resolved_root,
        source_root=source_root,
        project_commit=commit,
    )
    lock_path = resolved_root / "logs" / "control" / "georeliab-gpu-execution.lock"
    r4_pass = bool(
        "require_gpu_selection" in launch_text
        and "p2a)" in launch_text
        and "--shard 0/1" in launch_text
        and "--shard 1/2" not in launch_text
        and "CUDA_VISIBLE_DEVICES='$gpu_index'" in launch_text
        and "GEORELIAB_PHYSICAL_GPU_DEVICE='$device'" in launch_text
        and "assert_no_active_georeliab_execution" in launch_text
        and "selected physical GPU is unavailable" in launch_text
        and not lock_path.exists()
        and gpu_status.get("status") in {"GPU_SELECTION_REQUIRED", "PASS"}
    )
    payload = {
        "schema_version": ROLLOUT_VALIDATION_SCHEMA,
        "status": "PASS" if all((r1_pass, r2_pass, r3_pass, r4_pass)) else "BLOCKED",
        "project_commit": commit,
        "project_tree": tree,
        "runtime_root": str(resolved_root),
        "r1_storage": {
            "status": "PASS" if r1_pass else "FAIL",
            "actual_logical_bytes": actual_logical,
            "full_worst_path_bytes": (
                int(projection.get("full_worst_path_bytes", 0))
                if isinstance(projection, Mapping)
                else None
            ),
            "r1_limit_bytes": R1_LIMIT_BYTES,
            "hard_limit_bytes": HARD_STORAGE_LIMIT_BYTES,
            "scan_errors": errors,
        },
        "r2_protocol": {
            "status": "PASS" if r2_pass else "FAIL",
            "science_lock": science_lock,
        },
        "r3_artifact_equivalence": {
            "status": "PASS" if r3_pass else "FAIL",
            "plan_file_sha256": plan_sha,
            "actions_applied": apply_receipt.get("actions_applied"),
        },
        "r4_controller": {
            "status": "PASS" if r4_pass else "FAIL",
            "gpu_selection": gpu_status,
            "single_gpu_lock_active": lock_path.exists(),
            "selection_manifest_forbidden_outside_smoke": True,
        },
    }
    _write_immutable_json(output_path, payload)
    return payload
