"""Deterministic lossless artifact encoding and stage-specific retention.

This module implements the storage-only changes authorized by
``docs/STORAGE_REFACTOR_PROTOCOL.md``.  It never changes array dtype, shape, or
values, and it keeps every destructive operation inside an uncommitted bundle
directory until validation has succeeded.
"""

from __future__ import annotations

from dataclasses import dataclass
import gzip
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import tarfile
import time
from typing import Any, Iterable, Mapping
import zipfile

import numpy as np


NPZ_COMPRESSION_LEVEL = 9
RETENTION_RECEIPT_SCHEMA = "georeliab-retention-receipt-v1"
SHARED_GT_SCHEMA = "georeliab-shared-gt-v1"


class ArtifactStorageError(RuntimeError):
    """Raised when a storage transform cannot prove evidence equivalence."""


@dataclass(frozen=True, slots=True)
class SharedArtifact:
    path: Path
    sha256: str
    array_fingerprint: Mapping[str, Any]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_bytes(value: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(dict(value), sort_keys=True, separators=(",", ":"), default=str)
        + "\n"
    ).encode("utf-8")


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


def _array_fingerprint(value: Any) -> dict[str, Any]:
    array = np.asarray(value)
    if array.dtype.hasobject:
        raise ArtifactStorageError("object arrays are forbidden in evidence NPZ files")
    contiguous = np.ascontiguousarray(array)
    return {
        "dtype": array.dtype.str,
        "shape": list(array.shape),
        "c_order_sha256": hashlib.sha256(
            contiguous.tobytes(order="C")
        ).hexdigest(),
    }


def _validate_npz_member_name(name: Any) -> str:
    if not isinstance(name, str):
        raise ArtifactStorageError("NPZ array names must be strings")
    if (
        not name
        or name in {".", ".."}
        or "/" in name
        or "\\" in name
        or "\0" in name
    ):
        raise ArtifactStorageError(f"unsafe NPZ array name: {name!r}")
    return name


def arrays_fingerprint(arrays: Mapping[str, Any]) -> dict[str, Any]:
    """Describe decoded array semantics independent of the ZIP encoding."""

    names = sorted(_validate_npz_member_name(name) for name in arrays)
    return {
        "keys": names,
        "arrays": {
            name: _array_fingerprint(arrays[name])
            for name in names
        },
    }


def load_npz_arrays(path: Path) -> dict[str, np.ndarray]:
    try:
        with np.load(path, allow_pickle=False) as payload:
            return {name: payload[name] for name in payload.files}
    except (OSError, ValueError) as exc:
        raise ArtifactStorageError(f"cannot read NPZ artifact {path}: {exc}") from exc


def npz_fingerprint(path: Path) -> dict[str, Any]:
    return arrays_fingerprint(load_npz_arrays(path))


def assert_npz_equivalent(first: Path, second: Path) -> dict[str, Any]:
    first_fingerprint = npz_fingerprint(first)
    second_fingerprint = npz_fingerprint(second)
    if first_fingerprint != second_fingerprint:
        raise ArtifactStorageError(
            f"decoded NPZ semantics differ: {first} != {second}"
        )
    return first_fingerprint


def write_deterministic_npz(
    path: Path,
    arrays: Mapping[str, Any],
    *,
    member_order: Iterable[str] | None = None,
) -> None:
    """Write a deterministic ZIP-DEFLATED NPZ after exact validation.

    The default lexicographic member order makes writes independent of mapping
    insertion order.  Existing artifact families that exposed ``NpzFile.files``
    order may provide their frozen order explicitly.
    """

    expected = arrays_fingerprint(arrays)
    names = list(expected["keys"])
    if member_order is not None:
        names = [_validate_npz_member_name(name) for name in member_order]
        if len(names) != len(set(names)) or set(names) != set(expected["keys"]):
            raise ArtifactStorageError(
                "explicit NPZ member order must contain every array exactly once"
            )
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".partial")
    try:
        with partial.open("wb") as raw:
            with zipfile.ZipFile(
                raw,
                mode="w",
                compression=zipfile.ZIP_DEFLATED,
                compresslevel=NPZ_COMPRESSION_LEVEL,
                allowZip64=True,
            ) as archive:
                for name in names:
                    buffer = io.BytesIO()
                    np.save(buffer, np.asarray(arrays[name]), allow_pickle=False)
                    info = zipfile.ZipInfo(
                        f"{name}.npy",
                        date_time=(1980, 1, 1, 0, 0, 0),
                    )
                    info.create_system = 3
                    info.external_attr = 0o600 << 16
                    info.compress_type = zipfile.ZIP_DEFLATED
                    archive.writestr(
                        info,
                        buffer.getvalue(),
                        compress_type=zipfile.ZIP_DEFLATED,
                        compresslevel=NPZ_COMPRESSION_LEVEL,
                    )
            raw.flush()
            try:
                os.fsync(raw.fileno())
            except OSError:
                pass
        if npz_fingerprint(partial) != expected:
            raise ArtifactStorageError(
                f"deterministic NPZ verification failed before replace: {path}"
            )
        partial.replace(path)
    except Exception:
        try:
            partial.unlink()
        except FileNotFoundError:
            pass
        raise


def reencode_npz_lossless(source: Path, destination: Path) -> dict[str, Any]:
    """Losslessly re-encode an NPZ while keeping the source intact on failure."""

    arrays = load_npz_arrays(source)
    expected = arrays_fingerprint(arrays)
    write_deterministic_npz(destination, arrays)
    if npz_fingerprint(destination) != expected:
        raise ArtifactStorageError("lossless NPZ re-encoding changed decoded arrays")
    return {
        "source_sha256": sha256_file(source),
        "destination_sha256": sha256_file(destination),
        "array_fingerprint": expected,
    }


def write_shared_gt(runtime_root: Path, gt_points: Any) -> SharedArtifact:
    """Store one deterministic GT payload per unique content digest."""

    shared_root = runtime_root.resolve() / "shared_gt"
    shared_root.mkdir(parents=True, exist_ok=True)
    arrays = {"gt_points": np.asarray(gt_points)}
    expected = arrays_fingerprint(arrays)
    temporary = shared_root / (
        f".gt_points.{os.getpid()}.{time.time_ns()}.npz"
    )
    try:
        write_deterministic_npz(temporary, arrays)
        digest = sha256_file(temporary)
        destination = shared_root / f"{digest}.npz"
        if destination.exists():
            if sha256_file(destination) != digest:
                raise ArtifactStorageError(
                    f"content-addressed GT digest drift: {destination}"
                )
            if npz_fingerprint(destination) != expected:
                raise ArtifactStorageError(
                    f"content-addressed GT semantic drift: {destination}"
                )
        else:
            temporary.replace(destination)
        return SharedArtifact(destination, digest, expected)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def _safe_relative_files(root: Path) -> list[Path]:
    resolved_root = root.resolve()
    if not root.exists():
        return []
    if root.is_symlink() or not root.is_dir():
        raise ArtifactStorageError(f"retention root must be a real directory: {root}")
    files: list[Path] = []
    for candidate in sorted(root.rglob("*")):
        if candidate.is_symlink():
            raise ArtifactStorageError(
                f"retention refuses symbolic links: {candidate}"
            )
        if not candidate.is_file():
            continue
        resolved = candidate.resolve()
        try:
            resolved.relative_to(resolved_root)
        except ValueError as exc:
            raise ArtifactStorageError(
                f"retention member escaped its root: {candidate}"
            ) from exc
        files.append(candidate)
    return files


def file_inventory(root: Path) -> list[dict[str, Any]]:
    return [
        {
            "relative_path": path.relative_to(root).as_posix(),
            "sha256": sha256_file(path),
            "bytes": path.stat().st_size,
        }
        for path in _safe_relative_files(root)
    ]


def _validate_inventory(
    root: Path,
    inventory: Iterable[Mapping[str, Any]],
) -> None:
    expected = [dict(row) for row in inventory]
    actual = file_inventory(root)
    if actual != expected:
        raise ArtifactStorageError(f"retention inventory mismatch under {root}")


def write_deterministic_tar_gz(
    source_dir: Path,
    destination: Path,
) -> list[dict[str, Any]]:
    """Archive a cache with fixed ordering and metadata, then verify members."""

    inventory = file_inventory(source_dir)
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_name(destination.name + ".partial")
    try:
        with partial.open("wb") as raw:
            with gzip.GzipFile(
                filename="",
                mode="wb",
                compresslevel=9,
                fileobj=raw,
                mtime=0,
            ) as compressed:
                with tarfile.open(
                    fileobj=compressed,
                    mode="w",
                    format=tarfile.PAX_FORMAT,
                ) as archive:
                    for row in inventory:
                        member_path = source_dir / str(row["relative_path"])
                        info = tarfile.TarInfo(str(row["relative_path"]))
                        info.size = member_path.stat().st_size
                        info.mode = 0o600
                        info.mtime = 0
                        info.uid = 0
                        info.gid = 0
                        info.uname = ""
                        info.gname = ""
                        info.pax_headers = {}
                        with member_path.open("rb") as handle:
                            archive.addfile(info, handle)
            raw.flush()
            try:
                os.fsync(raw.fileno())
            except OSError:
                pass
        validate_tar_gz_inventory(partial, inventory)
        partial.replace(destination)
    except Exception:
        try:
            partial.unlink()
        except FileNotFoundError:
            pass
        raise
    return inventory


def validate_tar_gz_inventory(
    archive_path: Path,
    inventory: Iterable[Mapping[str, Any]],
) -> None:
    expected = [dict(row) for row in inventory]
    actual: list[dict[str, Any]] = []
    try:
        with tarfile.open(archive_path, mode="r:gz") as archive:
            for member in archive.getmembers():
                name = Path(member.name)
                if (
                    member.isdir()
                    or member.issym()
                    or member.islnk()
                    or name.is_absolute()
                    or ".." in name.parts
                ):
                    raise ArtifactStorageError(
                        f"unsafe cache archive member: {member.name}"
                    )
                extracted = archive.extractfile(member)
                if extracted is None:
                    raise ArtifactStorageError(
                        f"cache archive member is not a regular file: {member.name}"
                    )
                digest = hashlib.sha256()
                size = 0
                for block in iter(lambda: extracted.read(1024 * 1024), b""):
                    digest.update(block)
                    size += len(block)
                actual.append(
                    {
                        "relative_path": member.name,
                        "sha256": digest.hexdigest(),
                        "bytes": size,
                    }
                )
    except (OSError, tarfile.TarError) as exc:
        raise ArtifactStorageError(
            f"cannot validate cache archive {archive_path}: {exc}"
        ) from exc
    if actual != expected:
        raise ArtifactStorageError(
            f"cache archive inventory mismatch: {archive_path}"
        )


def _remove_retention_tree(bundle_root: Path, target: Path) -> None:
    resolved_root = bundle_root.resolve()
    resolved_target = target.resolve()
    try:
        relative = resolved_target.relative_to(resolved_root)
    except ValueError as exc:
        raise ArtifactStorageError(
            f"retention target escaped bundle root: {target}"
        ) from exc
    if not relative.parts:
        raise ArtifactStorageError("retention may not remove the bundle root")
    _safe_relative_files(target)
    shutil.rmtree(target)


def _resolve_bundle_member(bundle_dir: Path, relative: str) -> Path:
    member = Path(relative)
    if member.is_absolute() or ".." in member.parts:
        raise ArtifactStorageError(f"unsafe receipt path: {relative}")
    root = bundle_dir.resolve()
    resolved = (bundle_dir / member).resolve()
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise ArtifactStorageError(
            f"receipt path escaped bundle root: {relative}"
        ) from exc
    return resolved


def finalize_mast3r_cache(
    bundle_partial: Path,
    *,
    stage: str,
    allow_empty: bool = False,
) -> Path:
    """Apply the frozen P2/P3 MASt3R cache lifecycle inside a partial bundle."""

    if stage not in {"smoke", "test"}:
        raise ArtifactStorageError(f"unsupported MASt3R cache stage: {stage}")
    cache_dir = bundle_partial / "adapter" / "mast3r_cache"
    inventory = file_inventory(cache_dir)
    exception_empty = bool(allow_empty and not inventory)
    if not inventory and not exception_empty:
        raise ArtifactStorageError(
            "valid MASt3R bundle requires a non-empty verified cache inventory"
        )
    archive_path = bundle_partial / "mast3r_pairwise_cache.tar.gz"
    archive_sha = ""
    policy = (
        "adapter-exception-empty-cache"
        if exception_empty
        else "p2-inventory-only"
    )
    if stage == "test" and not exception_empty:
        policy = "p3-deterministic-tar-gz-until-p6"
        write_deterministic_tar_gz(cache_dir, archive_path)
        archive_sha = sha256_file(archive_path)
        validate_tar_gz_inventory(archive_path, inventory)
    ready_receipt = {
        "schema_version": RETENTION_RECEIPT_SCHEMA,
        "stage": stage,
        "policy": policy,
        "status": "READY_TO_REMOVE",
        "cache_relative_path": "adapter/mast3r_cache",
        "members": inventory,
        "archive_relative_path": (
            archive_path.relative_to(bundle_partial).as_posix()
            if stage == "test" and not exception_empty
            else None
        ),
        "archive_sha256": archive_sha or None,
    }
    receipt_path = bundle_partial / "cache_retention_receipt.json"
    _atomic_json(receipt_path, ready_receipt)
    if cache_dir.exists():
        _remove_retention_tree(bundle_partial, cache_dir)
    if cache_dir.exists():
        raise ArtifactStorageError("MASt3R cache remained after retention")
    final_receipt = dict(ready_receipt)
    final_receipt["status"] = "PASS"
    _atomic_json(receipt_path, final_receipt)
    validate_retention_receipt(bundle_partial)
    return receipt_path


def finalize_zero_update_adapter(
    bundle_partial: Path,
    *,
    subset_path: Path,
) -> Path:
    """Remove P5 adapter intermediates only after the subset is digest-bound."""

    if not subset_path.is_file():
        raise ArtifactStorageError("P5 subset prediction is missing")
    subset_relative = subset_path.resolve().relative_to(
        bundle_partial.resolve()
    ).as_posix()
    subset_digest = sha256_file(subset_path)
    adapter_dir = bundle_partial / "adapter"
    inventory = file_inventory(adapter_dir)
    receipt_path = bundle_partial / "retention_receipt.json"
    ready_receipt = {
        "schema_version": RETENTION_RECEIPT_SCHEMA,
        "stage": "zero-update",
        "policy": "p5-subset-only",
        "status": "READY_TO_REMOVE",
        "removed_relative_path": "adapter",
        "removed_members": inventory,
        "subset_relative_path": subset_relative,
        "subset_sha256": subset_digest,
    }
    _atomic_json(receipt_path, ready_receipt)
    if adapter_dir.exists():
        _remove_retention_tree(bundle_partial, adapter_dir)
    if adapter_dir.exists():
        raise ArtifactStorageError("P5 adapter intermediates remained after retention")
    final_receipt = dict(ready_receipt)
    final_receipt["status"] = "PASS"
    _atomic_json(receipt_path, final_receipt)
    validate_retention_receipt(bundle_partial)
    return receipt_path


def validate_retention_receipt(bundle_dir: Path) -> dict[str, Any] | None:
    cache_receipt = bundle_dir / "cache_retention_receipt.json"
    zero_receipt = bundle_dir / "retention_receipt.json"
    candidates = [path for path in (cache_receipt, zero_receipt) if path.exists()]
    if not candidates:
        return None
    if len(candidates) != 1:
        raise ArtifactStorageError("bundle contains conflicting retention receipts")
    receipt_path = candidates[0]
    try:
        payload = json.loads(receipt_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ArtifactStorageError(f"invalid retention receipt: {exc}") from exc
    if (
        not isinstance(payload, dict)
        or payload.get("schema_version") != RETENTION_RECEIPT_SCHEMA
        or payload.get("status") != "PASS"
    ):
        raise ArtifactStorageError("retention receipt is not a canonical PASS")
    stage = payload.get("stage")
    if stage in {"smoke", "test"}:
        if (bundle_dir / "adapter" / "mast3r_cache").exists():
            raise ArtifactStorageError("retained bundle still contains MASt3R cache")
        members = payload.get("members")
        policy = payload.get("policy")
        if not isinstance(members, list):
            raise ArtifactStorageError("cache retention receipt lacks inventory")
        if policy == "adapter-exception-empty-cache":
            if members:
                raise ArtifactStorageError(
                    "adapter-exception receipt must have an empty cache inventory"
                )
            if (
                payload.get("archive_relative_path") is not None
                or payload.get("archive_sha256") is not None
            ):
                raise ArtifactStorageError(
                    "adapter-exception receipt may not bind a cache archive"
                )
        elif policy == "p2-inventory-only" and stage == "smoke":
            if not members:
                raise ArtifactStorageError("P2 cache inventory must not be empty")
        elif policy == "p3-deterministic-tar-gz-until-p6" and stage == "test":
            if not members:
                raise ArtifactStorageError("P3 cache inventory must not be empty")
            relative = payload.get("archive_relative_path")
            digest = payload.get("archive_sha256")
            if not isinstance(relative, str) or not isinstance(digest, str):
                raise ArtifactStorageError("P3 cache receipt lacks archive binding")
            archive_path = _resolve_bundle_member(bundle_dir, relative)
            if not archive_path.is_file():
                raise ArtifactStorageError("P3 cache archive is missing")
            if sha256_file(archive_path) != digest:
                raise ArtifactStorageError("P3 cache archive digest mismatch")
            validate_tar_gz_inventory(archive_path, members)
        else:
            raise ArtifactStorageError(
                f"cache retention policy/stage mismatch: {policy}/{stage}"
            )
    elif stage == "zero-update":
        if (bundle_dir / "adapter").exists():
            raise ArtifactStorageError("P5 retained bundle still contains adapter output")
        relative = payload.get("subset_relative_path")
        digest = payload.get("subset_sha256")
        if not isinstance(relative, str) or not isinstance(digest, str):
            raise ArtifactStorageError("P5 retention receipt lacks subset binding")
        subset_path = _resolve_bundle_member(bundle_dir, relative)
        if not subset_path.is_file():
            raise ArtifactStorageError("P5 retained subset is missing")
        if sha256_file(subset_path) != digest:
            raise ArtifactStorageError("P5 retained subset digest mismatch")
        npz_fingerprint(subset_path)
    else:
        raise ArtifactStorageError(f"unsupported retention receipt stage: {stage}")
    return payload
