"""Fail-closed science locks for the storage-only refactor."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Mapping


BASE_PROJECT_COMMIT = "f5397b25806dcbf5b527b83c836b6c5f344122ae"

LOCKED_SCIENCE_FILES: Mapping[str, str] = {
    "configs/dual_mve_protocol.toml": (
        "71069d54ec11dd2fda4c3153a3b2a2ff0b60db5bb1855a39cbe7a7ee5d4f8198"
    ),
    "configs/a100_real_mve_overlay.toml": (
        "9dfa1b20aa4fcafb5808b5edafd5e30279b59da0887906472ec20dd804de4400"
    ),
    "georeliab_mve/splits.py": (
        "55e79e14b9f6dbe957500cfdf3c04723ebe950de973f66153061ddf89a89dc4e"
    ),
    "georeliab_mve/preparation_round2.py": (
        "891c7337d2f8129bddd44efe250e8a7e698f0c683aebb8070369abb38f465f2c"
    ),
    "georeliab_mve/gates.py": (
        "9063b624ae5543a3c825d7909749b25c831ba6324358d02db4dc58fefee15239"
    ),
}


class ScienceLockError(RuntimeError):
    """Raised when a frozen scientific input no longer matches f539."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_schedule_contract() -> dict[str, object]:
    """Lock schedule semantics while allowing storage-only runner edits."""

    from .preparation import TEST_SCENES
    from .runner import (
        CONDITIONS,
        GPU_HOUR_LIMIT,
        MODELS,
        PREFLIGHT_CONDITIONS,
        STORAGE_BYTE_LIMIT,
        ZERO_UPDATE_SUBSETS,
    )

    expected = {
        "models": ("VGGT", "MASt3R"),
        "conditions": (
            ("clean", 0),
            ("fog", 1),
            ("fog", 2),
            ("fog", 3),
            ("low-light-noise", 1),
            ("low-light-noise", 2),
            ("low-light-noise", 3),
            ("defocus", 1),
            ("defocus", 2),
            ("defocus", 3),
        ),
        "preflight_conditions": (
            ("clean", 0),
            ("fog", 1),
            ("low-light-noise", 1),
            ("defocus", 1),
        ),
        "zero_update_subsets": ((0, 4), (1, 5), (2, 6), (3, 7)),
        "test_scenes": (
            1, 9, 10, 11, 12, 13, 23, 24, 29, 32,
            33, 34, 48, 49, 62, 75, 77, 110, 114, 118,
        ),
        "gpu_hour_limit": 50.0,
        "storage_byte_limit": 1_000_000_000_000,
    }
    actual = {
        "models": tuple(MODELS),
        "conditions": tuple(CONDITIONS),
        "preflight_conditions": tuple(PREFLIGHT_CONDITIONS),
        "zero_update_subsets": tuple(ZERO_UPDATE_SUBSETS),
        "test_scenes": tuple(TEST_SCENES),
        "gpu_hour_limit": float(GPU_HOUR_LIMIT),
        "storage_byte_limit": int(STORAGE_BYTE_LIMIT),
    }
    if actual != expected:
        raise ScienceLockError("frozen schedule contract changed during storage refactor")
    return {
        "status": "PASS",
        "preflight_items": len(MODELS) * len(PREFLIGHT_CONDITIONS),
        "p2_items": len(MODELS) * 10 * len(CONDITIONS),
        "p3_items": len(MODELS) * len(TEST_SCENES) * len(CONDITIONS),
        "p5_zero_update_items": (
            len(MODELS) * len(TEST_SCENES) * 3 * len(ZERO_UPDATE_SUBSETS)
        ),
    }


def validate_science_lock(source_root: Path) -> dict[str, object]:
    """Validate every immutable scientific file without changing it."""

    root = source_root.resolve()
    rows: list[dict[str, str]] = []
    for relative, expected in LOCKED_SCIENCE_FILES.items():
        path = (root / relative).resolve()
        try:
            path.relative_to(root)
        except ValueError as exc:
            raise ScienceLockError(f"science lock escaped source root: {relative}") from exc
        if not path.is_file():
            raise ScienceLockError(f"missing science-locked file: {relative}")
        actual = sha256_file(path)
        rows.append({"path": relative, "sha256": actual, "expected_sha256": expected})
        if actual != expected:
            raise ScienceLockError(
                f"science lock mismatch for {relative}: {actual} != {expected}"
            )
    return {
        "schema_version": "georeliab-storage-science-lock-v1",
        "base_project_commit": BASE_PROJECT_COMMIT,
        "status": "PASS",
        "files": rows,
        "schedule_contract": validate_schedule_contract(),
    }
