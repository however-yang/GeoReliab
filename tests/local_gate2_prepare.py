"""Prepare a home-owned, non-scientific Gate 2 development closure.

This opt-in utility downloads only the 48 official DTU L3 PNG members used by
the deterministic six-scene recovery smoke.  It never downloads the complete
129 GB Rectified archive and it never writes outside the caller's home tree.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import struct
import sys
from types import SimpleNamespace
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from georeliab_mve.tartanair_range import (  # noqa: E402
    extract_range_members_evidence,
    index_remote_zip,
)
from georeliab_mve.adapters import verify_frozen_runtime  # noqa: E402
from georeliab_mve.runner import _frozen_runtime_from_config  # noqa: E402
from georeliab_mve.v4_attempt05_recovery import (  # noqa: E402
    AUTHORIZED_GPU_UUID,
    AUTHORIZED_PHYSICAL_GPU_INDEX,
    NO_SCIENTIFIC_RESULT,
    build_recovery_smoke_manifest,
    sha256_file,
)
from georeliab_mve.v4_counterfactuals import TEST_SCENE_IDS  # noqa: E402


SCHEMA_VERSION = "georeliab-v4-local-gate2-input-closure-1.0"
PLAN_SCHEMA_VERSION = "georeliab-v4-local-gate2-plan-1.0"
LOG_SCHEMA_VERSION = "georeliab-v4-local-gate2-preparation-log-1.0"
LOCAL_STATUS = "LOCAL_GATE2_DEVELOPMENT_INPUT_READY"
FORMAL_HOME_SCHEMA_VERSION = "georeliab-v4-formal-home-gate2-input-closure-1.0"
FORMAL_HOME_STATUS = "V4_FORMAL_HOME_GATE2_INPUT_CLOSURE_READY"
FORMAL_HOME_VALIDATION_CLASS = "FORMAL_GATE2_INPUT_CLOSURE"
FORMAL_GATE2_AUTH_SCHEMA_VERSION = "georeliab-v4-formal-gate2-authorization-1.0"
FORMAL_GATE2_AUTH_STATUS = "V4_FORMAL_GATE2_EXECUTION_AUTHORIZED"
RECTIFIED_URL = "https://roboimagedata2.compute.dtu.dk/data/MVS/Rectified.zip"
RECTIFIED_BYTES = 129_593_443_783
RECTIFIED_ETAG = "1e2c5f05c7-5652dcf644cb3"
CANONICAL_COMMIT = "a9c687c73eca6bc3a26659b3e0247c227e7c5238"
CANONICAL_TREE = "c8c47baff7deb082e502db83f8fc8c942f5d3e3d"
VGGT_CHECKPOINT_SHA256 = (
    "d15bf50a8615c8225ed48b51ea5cac673d82442ec0309036df555a053253afe0"
)
VGGT_SOURCE_COMMIT = "a288dd0f14786c93483e45524328726ab7b1b4ce"
MAST3R_CHECKPOINT_SHA256 = (
    "0a615eb05fa9db654050aa655945ee5696e7c6c1b7f93f1ee8c37249010f6feb"
)
MAST3R_CONFIG_SHA256 = (
    "718eb93dc4f9e4332b60cc0041af962d712cbd346d7770ce35c5b22cff68eae4"
)
MAST3R_SOURCE_COMMIT = "f5209afc300cec36239a7ac992263f36847bbba0"
DUST3R_SOURCE_COMMIT = "3cc8c88c413bb9e34c41db0e0eef99c2ee010b12"
CROCO_SOURCE_COMMIT = "d7de0705845239092414480bd829228723bf20de"
TYPING_EXTENSIONS_SHA256 = (
    "433d11d170d3a24d2eb065ebc1bfe848cea7e3d7ce68567ab52bea2d4c2f7ed8"
)

# This is the repository's frozen eight-view fixture order.  It is recorded in
# the local closure and is intentionally labelled development-only because the
# unavailable historical Attempt-05 schedule remains the formal authority.
LOCAL_ORDERED_VIEW_IDS = (1, 39, 45, 49, 6, 16, 42, 27)


class LocalGate2PreparationError(RuntimeError):
    """Fail-closed local preparation error."""


def _canonical_bytes(value: object) -> bytes:
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


def _sha_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


LOCAL_SCHEDULE_IDENTITY = _sha_bytes(
    _canonical_bytes(
        {
            "purpose": "local-gate2-development-recovery-smoke",
            "canonical_commit": CANONICAL_COMMIT,
            "canonical_tree": CANONICAL_TREE,
            "support_scene_ids": list(TEST_SCENE_IDS),
            "ordered_view_ids": list(LOCAL_ORDERED_VIEW_IDS),
            "state_id": "L3",
            "models": ["VGGT", "MASt3R"],
            "scientific_result": NO_SCIENTIFIC_RESULT,
        }
    )
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def require_home_owned_root(root: Path) -> Path:
    """Resolve ``root`` and require it to live below the current user's home."""

    resolved = root.expanduser().resolve()
    home = Path.home().resolve()
    if resolved == home or home not in resolved.parents:
        raise LocalGate2PreparationError(f"LOCAL_GATE2_ROOT_OUTSIDE_HOME:{resolved}")
    return resolved


def _atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + f".partial-{os.getpid()}")
    with partial.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(partial, path)


class PreparationLog:
    """Append matching JSONL and readable text records with fsync."""

    def __init__(self, root: Path) -> None:
        self.jsonl = root / "logs" / "preparation-events.jsonl"
        self.text = root / "logs" / "preparation.log"

    def write(self, event: str, **fields: object) -> None:
        row = {
            "schema_version": LOG_SCHEMA_VERSION,
            "recorded_at": _utc_now(),
            "event": event,
            "validation_class": "LOCAL_GATE2_DEVELOPMENT_VALIDATION",
            "scientific_result": NO_SCIENTIFIC_RESULT,
            **fields,
        }
        self.jsonl.parent.mkdir(parents=True, exist_ok=True)
        with self.jsonl.open("ab") as handle:
            handle.write(_canonical_bytes(row))
            handle.flush()
            os.fsync(handle.fileno())
        rendered = " ".join(
            f"{key}={json.dumps(value, sort_keys=True)}"
            for key, value in sorted(fields.items())
        )
        with self.text.open("ab") as handle:
            handle.write(f"[{row['recorded_at']}] {event} {rendered}\n".encode())
            handle.flush()
            os.fsync(handle.fileno())
        print(f"[{row['recorded_at']}] {event} {rendered}", flush=True)


def local_smoke_manifest() -> Mapping[str, object]:
    return build_recovery_smoke_manifest(
        schedule_identity_sha256=LOCAL_SCHEDULE_IDENTITY,
        support_scene_ids=TEST_SCENE_IDS,
    ).to_dict()


def selected_member_paths() -> tuple[str, ...]:
    smoke = local_smoke_manifest()
    return tuple(
        f"Rectified/scan{scene}/rect_{view_id:03d}_3_r5000.png"
        for scene in smoke["scene_ids"]  # type: ignore[index]
        for view_id in LOCAL_ORDERED_VIEW_IDS
    )


def _png_dimensions(path: Path) -> tuple[int, int]:
    with path.open("rb") as handle:
        header = handle.read(24)
    if (
        len(header) != 24
        or header[:8] != b"\x89PNG\r\n\x1a\n"
        or header[12:16] != b"IHDR"
    ):
        raise LocalGate2PreparationError(f"LOCAL_GATE2_NOT_PNG:{path}")
    width, height = struct.unpack(">II", header[16:24])
    if width <= 0 or height <= 0:
        raise LocalGate2PreparationError(f"LOCAL_GATE2_PNG_DIMENSIONS_INVALID:{path}")
    return width, height


def _archive_identity(index: object) -> dict[str, object]:
    return {
        "url": RECTIFIED_URL,
        "content_length": int(getattr(index, "content_length")),
        "etag": str(getattr(index, "etag")),
        "central_directory_sha256": str(getattr(index, "central_directory_sha256")),
    }


def _validate_archive_identity(identity: Mapping[str, object]) -> None:
    if identity.get("url") != RECTIFIED_URL:
        raise LocalGate2PreparationError("LOCAL_GATE2_ARCHIVE_URL_MISMATCH")
    if int(identity.get("content_length", -1)) != RECTIFIED_BYTES:
        raise LocalGate2PreparationError("LOCAL_GATE2_ARCHIVE_LENGTH_MISMATCH")
    etag = str(identity.get("etag", "")).strip('"')
    if etag != RECTIFIED_ETAG:
        raise LocalGate2PreparationError("LOCAL_GATE2_ARCHIVE_ETAG_MISMATCH")
    digest = str(identity.get("central_directory_sha256", ""))
    if len(digest) != 64:
        raise LocalGate2PreparationError("LOCAL_GATE2_ARCHIVE_INDEX_DIGEST_INVALID")


def build_plan(root: Path) -> dict[str, object]:
    resolved = require_home_owned_root(root)
    smoke = local_smoke_manifest()
    return {
        "schema_version": PLAN_SCHEMA_VERSION,
        "validation_class": "LOCAL_GATE2_DEVELOPMENT_VALIDATION",
        "formal_gate2_equivalent": False,
        "root": str(resolved),
        "data_root": str(resolved / "data" / "dtu-l3"),
        "resources_root": str(resolved / "resources"),
        "overlay_config": str(resolved / "manifests" / "local-gate2-overlay.toml"),
        "runs_root": str(resolved / "runs"),
        "logs_root": str(resolved / "logs"),
        "archive": {
            "url": RECTIFIED_URL,
            "expected_content_length": RECTIFIED_BYTES,
            "expected_etag": RECTIFIED_ETAG,
            "full_archive_download_forbidden": True,
        },
        "schedule_identity_sha256": LOCAL_SCHEDULE_IDENTITY,
        "scene_ids": smoke["scene_ids"],
        "ordered_view_ids": list(LOCAL_ORDERED_VIEW_IDS),
        "member_count": len(selected_member_paths()),
        "members": list(selected_member_paths()),
        "frozen_resources": {
            "vggt_source": {
                "url": "https://github.com/facebookresearch/vggt.git",
                "commit": VGGT_SOURCE_COMMIT,
            },
            "vggt_checkpoint": {
                "url": "https://huggingface.co/facebook/VGGT-1B/resolve/main/model.pt",
                "sha256": VGGT_CHECKPOINT_SHA256,
            },
            "mast3r_source": {
                "url": "https://github.com/naver/mast3r.git",
                "commit": MAST3R_SOURCE_COMMIT,
                "recursive_submodules_required": True,
                "dust3r_commit": DUST3R_SOURCE_COMMIT,
                "croco_commit": CROCO_SOURCE_COMMIT,
            },
            "mast3r_checkpoint": {
                "repository": "https://huggingface.co/naver/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric",
                "sha256": MAST3R_CHECKPOINT_SHA256,
                "config_sha256": MAST3R_CONFIG_SHA256,
            },
        },
        "scientific_result": NO_SCIENTIFIC_RESULT,
    }


def write_plan(root: Path) -> Path:
    resolved = require_home_owned_root(root)
    path = resolved / "manifests" / "local-gate2-plan.json"
    _atomic_write(path, _canonical_bytes(build_plan(resolved)))
    write_local_overlay(resolved)
    return path


def _toml_literal(value: Path) -> str:
    rendered = str(value.resolve())
    if "'" in rendered or "\n" in rendered or "\r" in rendered:
        raise LocalGate2PreparationError("LOCAL_GATE2_ROOT_NOT_TOML_SAFE")
    return f"'{rendered}'"


def local_overlay_text(root: Path) -> str:
    """Render the frozen model binding with every mutable path below ``root``."""

    resolved = require_home_owned_root(root)
    resources = resolved / "resources"
    mast3r_model = (
        resources
        / "models"
        / "mast3r"
        / "MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric"
    )
    values = {
        "root": resolved,
        "vggt_source": resources / "sources" / "vggt",
        "mast3r_source": resources / "sources" / "mast3r",
        "dust3r_source": resources / "sources" / "mast3r" / "dust3r",
        "croco_source": resources / "sources" / "mast3r" / "dust3r" / "croco",
        "vggt_env": resources / "envs" / "vggt_env",
        "mast3r_env": resources / "envs" / "mast3r_env",
        "typing": resources / "typing_extensions_site",
        "vggt_checkpoint": resources / "models" / "vggt" / "model.pt",
        "mast3r_checkpoint": mast3r_model / "model.safetensors",
        "mast3r_config": mast3r_model / "config.json",
    }
    lines = [
        "# Test-only home-owned Gate 2 development overlay.",
        "# It is not a formal Attempt-05 execution authorization.",
        "[runtime]",
        f"root = {_toml_literal(values['root'])}",
        f"vggt_source = {_toml_literal(values['vggt_source'])}",
        f"mast3r_source = {_toml_literal(values['mast3r_source'])}",
        f"dust3r_source = {_toml_literal(values['dust3r_source'])}",
        f"croco_source = {_toml_literal(values['croco_source'])}",
        f"vggt_env = {_toml_literal(values['vggt_env'])}",
        f"mast3r_env = {_toml_literal(values['mast3r_env'])}",
        f"typing_extensions_site = {_toml_literal(values['typing'])}",
        "vggt_python = '3.10.20'",
        "vggt_torch = '2.3.1+cu121'",
        "mast3r_python = '3.10.20'",
        "mast3r_torch = '2.5.1+cu121'",
        "",
        "[resources]",
        f"vggt_checkpoint = {_toml_literal(values['vggt_checkpoint'])}",
        f"vggt_checkpoint_sha256 = '{VGGT_CHECKPOINT_SHA256}'",
        f"vggt_source_commit = '{VGGT_SOURCE_COMMIT}'",
        f"mast3r_checkpoint = {_toml_literal(values['mast3r_checkpoint'])}",
        f"mast3r_checkpoint_sha256 = '{MAST3R_CHECKPOINT_SHA256}'",
        f"mast3r_config = {_toml_literal(values['mast3r_config'])}",
        f"mast3r_config_sha256 = '{MAST3R_CONFIG_SHA256}'",
        f"mast3r_source_commit = '{MAST3R_SOURCE_COMMIT}'",
        f"dust3r_source_commit = '{DUST3R_SOURCE_COMMIT}'",
        f"croco_source_commit = '{CROCO_SOURCE_COMMIT}'",
        "typing_extensions_version = '4.15.0'",
        f"typing_extensions_sha256 = '{TYPING_EXTENSIONS_SHA256}'",
        "",
        "[local_development]",
        "validation_class = 'LOCAL_GATE2_DEVELOPMENT_VALIDATION'",
        "formal_gate2_equivalent = false",
        "scientific_result = 'NO_SCIENTIFIC_RESULT'",
        "",
    ]
    return "\n".join(lines)


def write_local_overlay(root: Path) -> Path:
    resolved = require_home_owned_root(root)
    path = resolved / "manifests" / "local-gate2-overlay.toml"
    _atomic_write(path, local_overlay_text(resolved).encode("utf-8"))
    return path


def _closure_payload(
    root: Path,
    *,
    archive_identity: Mapping[str, object],
    extraction: Mapping[str, Mapping[str, Any]],
) -> dict[str, object]:
    data_root = root / "data" / "dtu-l3"
    smoke = local_smoke_manifest()
    bindings: list[dict[str, object]] = []
    member_rows: list[dict[str, object]] = []
    for scene in smoke["scene_ids"]:  # type: ignore[index]
        views: list[dict[str, object]] = []
        for view_id in LOCAL_ORDERED_VIEW_IDS:
            member = f"Rectified/scan{scene}/rect_{view_id:03d}_3_r5000.png"
            path = data_root / member
            row = extraction.get(member)
            if row is None or not path.is_file():
                raise LocalGate2PreparationError(
                    f"LOCAL_GATE2_SELECTED_MEMBER_MISSING:{member}"
                )
            digest = sha256_file(path)
            if row.get("raw_sha256") != digest:
                raise LocalGate2PreparationError(
                    f"LOCAL_GATE2_SELECTED_MEMBER_DIGEST_MISMATCH:{member}"
                )
            width, height = _png_dimensions(path)
            view = {
                "view_id": view_id,
                "path": str(path.resolve()),
                "sha256": digest,
                "source_sha256": digest,
                "width": width,
                "height": height,
            }
            views.append(view)
            member_rows.append(
                {
                    "scene_id": scene,
                    "state_id": "L3",
                    "view_id": view_id,
                    "member": member,
                    "path": str(path.resolve()),
                    "sha256": digest,
                    "bytes": path.stat().st_size,
                    "width": width,
                    "height": height,
                    "disposition": row.get("disposition"),
                }
            )
        bindings.append(
            {
                "scene_id": scene,
                "state_id": "L3",
                "ordered_view_ids": list(LOCAL_ORDERED_VIEW_IDS),
                "views": views,
            }
        )
    return {
        "schema_version": SCHEMA_VERSION,
        "status": LOCAL_STATUS,
        "validation_class": "LOCAL_GATE2_DEVELOPMENT_VALIDATION",
        "formal_gate2_equivalent": False,
        "limitation": (
            "The historical Attempt-05 schedule/input closure is unavailable; "
            "this closure is a deterministic adapter-and-recovery development smoke."
        ),
        "root": str(root),
        "data_root": str(data_root),
        "canonical_commit": CANONICAL_COMMIT,
        "canonical_tree": CANONICAL_TREE,
        "schedule_identity_sha256": LOCAL_SCHEDULE_IDENTITY,
        "smoke_manifest": smoke,
        "scene_ids": smoke["scene_ids"],
        "ordered_view_ids": list(LOCAL_ORDERED_VIEW_IDS),
        "state_id": "L3",
        "model_ids": ["VGGT", "MASt3R"],
        "archive": dict(archive_identity),
        "bindings": bindings,
        "members": member_rows,
        "member_count": len(member_rows),
        "attempt05_predictions_read": False,
        "scientific_result": NO_SCIENTIFIC_RESULT,
    }


def validate_local_closure(path: Path) -> dict[str, object]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise LocalGate2PreparationError("LOCAL_GATE2_CLOSURE_UNREADABLE") from exc
    if not isinstance(payload, dict):
        raise LocalGate2PreparationError("LOCAL_GATE2_CLOSURE_INVALID")
    if (
        payload.get("schema_version") != SCHEMA_VERSION
        or payload.get("status") != LOCAL_STATUS
    ):
        raise LocalGate2PreparationError("LOCAL_GATE2_CLOSURE_IDENTITY_INVALID")
    root = require_home_owned_root(Path(str(payload.get("root", ""))))
    if (
        path.resolve()
        != (root / "manifests" / "local-gate2-input-closure.json").resolve()
    ):
        raise LocalGate2PreparationError("LOCAL_GATE2_CLOSURE_PATH_INVALID")
    if payload.get("formal_gate2_equivalent") is not False:
        raise LocalGate2PreparationError("LOCAL_GATE2_FORMAL_CLAIM_FORBIDDEN")
    if payload.get("scientific_result") != NO_SCIENTIFIC_RESULT:
        raise LocalGate2PreparationError("LOCAL_GATE2_SCIENTIFIC_RESULT_FORBIDDEN")
    if payload.get("schedule_identity_sha256") != LOCAL_SCHEDULE_IDENTITY:
        raise LocalGate2PreparationError("LOCAL_GATE2_SCHEDULE_IDENTITY_MISMATCH")
    smoke = local_smoke_manifest()
    if payload.get("scene_ids") != smoke["scene_ids"]:
        raise LocalGate2PreparationError("LOCAL_GATE2_SCENE_SELECTION_MISMATCH")
    if payload.get("ordered_view_ids") != list(LOCAL_ORDERED_VIEW_IDS):
        raise LocalGate2PreparationError("LOCAL_GATE2_VIEW_ORDER_MISMATCH")
    archive = payload.get("archive")
    if not isinstance(archive, Mapping):
        raise LocalGate2PreparationError("LOCAL_GATE2_ARCHIVE_IDENTITY_MISSING")
    _validate_archive_identity(archive)
    members = payload.get("members")
    bindings = payload.get("bindings")
    if not isinstance(members, list) or len(members) != 48:
        raise LocalGate2PreparationError("LOCAL_GATE2_MEMBER_COUNT_INVALID")
    if (
        payload.get("member_count") != 48
        or not isinstance(bindings, list)
        or len(bindings) != 6
    ):
        raise LocalGate2PreparationError("LOCAL_GATE2_BINDING_COUNT_INVALID")
    expected_members = set(selected_member_paths())
    actual_members: set[str] = set()
    for row in members:
        if not isinstance(row, Mapping):
            raise LocalGate2PreparationError("LOCAL_GATE2_MEMBER_ROW_INVALID")
        member = str(row.get("member", ""))
        file_path = Path(str(row.get("path", ""))).resolve()
        expected_path = (root / "data" / "dtu-l3" / member).resolve()
        if member not in expected_members or file_path != expected_path:
            raise LocalGate2PreparationError("LOCAL_GATE2_MEMBER_PATH_INVALID")
        if member in actual_members or not file_path.is_file():
            raise LocalGate2PreparationError("LOCAL_GATE2_MEMBER_SET_INVALID")
        if sha256_file(file_path) != row.get("sha256"):
            raise LocalGate2PreparationError(
                f"LOCAL_GATE2_MEMBER_DIGEST_MISMATCH:{member}"
            )
        width, height = _png_dimensions(file_path)
        if [width, height] != [row.get("width"), row.get("height")]:
            raise LocalGate2PreparationError(
                f"LOCAL_GATE2_MEMBER_DIMENSION_MISMATCH:{member}"
            )
        actual_members.add(member)
    if actual_members != expected_members:
        raise LocalGate2PreparationError("LOCAL_GATE2_MEMBER_SET_INVALID")
    return payload


def _require_hex_identity(value: object, *, width: int, reason: str) -> str:
    rendered = str(value)
    if len(rendered) != width or any(
        character not in "0123456789abcdef" for character in rendered
    ):
        raise LocalGate2PreparationError(reason)
    return rendered


def _validate_formal_bindings(payload: Mapping[str, object]) -> None:
    smoke = payload.get("smoke_manifest")
    bindings = payload.get("bindings")
    if not isinstance(smoke, Mapping):
        raise LocalGate2PreparationError("FORMAL_GATE2_SMOKE_IDENTITY_INVALID")
    unit_keys = smoke.get("unit_keys")
    if not isinstance(unit_keys, list) or len(unit_keys) != 12:
        raise LocalGate2PreparationError("FORMAL_GATE2_UNIT_IDENTITY_INVALID")
    if len(set(str(value) for value in unit_keys)) != 12:
        raise LocalGate2PreparationError("FORMAL_GATE2_UNIT_IDENTITY_DUPLICATE")
    if dict(smoke) != dict(local_smoke_manifest()):
        raise LocalGate2PreparationError("FORMAL_GATE2_SMOKE_IDENTITY_INVALID")
    if not isinstance(bindings, list) or len(bindings) != 6:
        raise LocalGate2PreparationError("FORMAL_GATE2_BINDING_IDENTITY_INVALID")
    scene_ids = payload.get("scene_ids")
    if not isinstance(scene_ids, list) or len(scene_ids) != 6:
        raise LocalGate2PreparationError("FORMAL_GATE2_SCENE_IDENTITY_INVALID")
    ordered_view_ids = payload.get("ordered_view_ids")
    if ordered_view_ids != list(LOCAL_ORDERED_VIEW_IDS):
        raise LocalGate2PreparationError("FORMAL_GATE2_VIEW_IDENTITY_INVALID")
    seen_scenes: set[int] = set()
    view_count = 0
    for binding in bindings:
        if not isinstance(binding, Mapping):
            raise LocalGate2PreparationError("FORMAL_GATE2_BINDING_IDENTITY_INVALID")
        scene_id = int(binding.get("scene_id", -1))
        views = binding.get("views")
        if (
            scene_id not in scene_ids
            or scene_id in seen_scenes
            or binding.get("state_id") != "L3"
            or binding.get("ordered_view_ids") != list(LOCAL_ORDERED_VIEW_IDS)
            or not isinstance(views, list)
            or len(views) != 8
        ):
            raise LocalGate2PreparationError("FORMAL_GATE2_BINDING_IDENTITY_INVALID")
        actual_view_ids: list[int] = []
        for view in views:
            if not isinstance(view, Mapping):
                raise LocalGate2PreparationError("FORMAL_GATE2_VIEW_IDENTITY_INVALID")
            view_id = int(view.get("view_id", -1))
            actual_view_ids.append(view_id)
            _require_hex_identity(
                view.get("sha256"),
                width=64,
                reason="FORMAL_GATE2_VIEW_DIGEST_INVALID",
            )
            if view.get("source_sha256") != view.get("sha256"):
                raise LocalGate2PreparationError(
                    "FORMAL_GATE2_VIEW_PROVENANCE_INVALID"
                )
            if not str(view.get("path", "")):
                raise LocalGate2PreparationError("FORMAL_GATE2_VIEW_PATH_INVALID")
        if actual_view_ids != list(LOCAL_ORDERED_VIEW_IDS):
            raise LocalGate2PreparationError("FORMAL_GATE2_VIEW_IDENTITY_INVALID")
        seen_scenes.add(scene_id)
        view_count += len(views)
    if seen_scenes != set(int(value) for value in scene_ids) or view_count != 48:
        raise LocalGate2PreparationError("FORMAL_GATE2_BINDING_IDENTITY_INVALID")
    if payload.get("member_count") != 48:
        raise LocalGate2PreparationError("FORMAL_GATE2_MEMBER_COUNT_INVALID")


def build_formal_home_closure_payload(
    *,
    source_closure: Mapping[str, object],
    source_closure_path: Path,
    source_closure_sha256: str,
    resource_audit: Mapping[str, object],
    resource_audit_path: Path,
    resource_audit_sha256: str,
    overlay_path: Path,
    overlay_sha256: str,
    formal_root: Path,
    production_source_commit: str,
    production_source_tree: str,
    test_only_source_commit: str,
) -> dict[str, object]:
    """Rebind validated local bits into a distinct, non-executing formal closure."""

    if (
        source_closure.get("schema_version") != SCHEMA_VERSION
        or source_closure.get("status") != LOCAL_STATUS
        or source_closure.get("validation_class")
        != "LOCAL_GATE2_DEVELOPMENT_VALIDATION"
        or source_closure.get("formal_gate2_equivalent") is not False
    ):
        raise LocalGate2PreparationError("FORMAL_GATE2_SOURCE_IDENTITY_INVALID")
    if (
        source_closure.get("attempt05_predictions_read") is not False
        or source_closure.get("scientific_result") != NO_SCIENTIFIC_RESULT
        or source_closure.get("schedule_identity_sha256")
        != LOCAL_SCHEDULE_IDENTITY
    ):
        raise LocalGate2PreparationError("FORMAL_GATE2_SOURCE_PROVENANCE_INVALID")
    _validate_formal_bindings(source_closure)
    if (
        resource_audit.get("schema_version")
        != "georeliab-v4-local-gate2-resource-audit-1.0"
        or resource_audit.get("status")
        != "LOCAL_GATE2_DEVELOPMENT_RESOURCES_READY"
        or resource_audit.get("validation_class")
        != "LOCAL_GATE2_DEVELOPMENT_VALIDATION"
        or resource_audit.get("formal_gate2_equivalent") is not False
        or resource_audit.get("scientific_result") != NO_SCIENTIFIC_RESULT
    ):
        raise LocalGate2PreparationError("FORMAL_GATE2_RESOURCE_IDENTITY_INVALID")
    models = resource_audit.get("models")
    if not isinstance(models, list) or not all(
        isinstance(row, Mapping) for row in models
    ):
        raise LocalGate2PreparationError("FORMAL_GATE2_RESOURCE_MODELS_INVALID")
    if [row.get("model") for row in models] != ["VGGT", "MASt3R"]:
        raise LocalGate2PreparationError("FORMAL_GATE2_RESOURCE_MODELS_INVALID")
    for row in models:
        if not isinstance(row, Mapping):
            raise LocalGate2PreparationError("FORMAL_GATE2_RESOURCE_MODELS_INVALID")
        _require_hex_identity(
            row.get("checkpoint_sha256"),
            width=64,
            reason="FORMAL_GATE2_RESOURCE_DIGEST_INVALID",
        )
    identities = (
        (source_closure_sha256, 64, "FORMAL_GATE2_SOURCE_DIGEST_INVALID"),
        (resource_audit_sha256, 64, "FORMAL_GATE2_RESOURCE_DIGEST_INVALID"),
        (overlay_sha256, 64, "FORMAL_GATE2_OVERLAY_DIGEST_INVALID"),
        (production_source_commit, 40, "FORMAL_GATE2_PRODUCTION_IDENTITY_INVALID"),
        (production_source_tree, 40, "FORMAL_GATE2_PRODUCTION_IDENTITY_INVALID"),
        (test_only_source_commit, 40, "FORMAL_GATE2_TEST_ONLY_IDENTITY_INVALID"),
    )
    for value, width, reason in identities:
        _require_hex_identity(value, width=width, reason=reason)
    if resource_audit.get("overlay_sha256") != overlay_sha256:
        raise LocalGate2PreparationError("FORMAL_GATE2_OVERLAY_DIGEST_MISMATCH")
    resolved_formal_root = formal_root.expanduser().resolve()
    source_root = Path(str(source_closure.get("root", ""))).expanduser().resolve()
    if resolved_formal_root == source_root:
        raise LocalGate2PreparationError("FORMAL_GATE2_ROOT_PROVENANCE_INVALID")
    return {
        "schema_version": FORMAL_HOME_SCHEMA_VERSION,
        "status": FORMAL_HOME_STATUS,
        "validation_class": FORMAL_HOME_VALIDATION_CLASS,
        "formal_gate2_equivalent": True,
        "source_validation_class": "LOCAL_GATE2_DEVELOPMENT_VALIDATION",
        "input_bits_revalidated": True,
        "resource_bits_revalidated": True,
        "prediction_outputs_reused": False,
        "attempt05_predictions_read": False,
        "gate2_started": False,
        "pilot_started": False,
        "attempt06_started": False,
        "execution_authorized": False,
        "root": str(resolved_formal_root),
        "source_root": str(source_root),
        "source_closure_path": str(source_closure_path.expanduser().resolve()),
        "source_closure_sha256": source_closure_sha256,
        "resource_audit_path": str(resource_audit_path.expanduser().resolve()),
        "resource_audit_sha256": resource_audit_sha256,
        "overlay_path": str(overlay_path.expanduser().resolve()),
        "overlay_sha256": overlay_sha256,
        "production_source_commit": production_source_commit,
        "production_source_tree": production_source_tree,
        "test_only_source_commit": test_only_source_commit,
        "schedule_identity_sha256": source_closure["schedule_identity_sha256"],
        "smoke_manifest": deepcopy(source_closure["smoke_manifest"]),
        "scene_ids": deepcopy(source_closure["scene_ids"]),
        "ordered_view_ids": deepcopy(source_closure["ordered_view_ids"]),
        "state_id": "L3",
        "model_ids": ["VGGT", "MASt3R"],
        "bindings": deepcopy(source_closure["bindings"]),
        "member_count": 48,
        "resources": deepcopy(models),
        "scientific_result": NO_SCIENTIFIC_RESULT,
    }


def validate_formal_home_closure_payload(
    payload: Mapping[str, object], *, expected_formal_root: Path
) -> dict[str, object]:
    """Validate the separate formal identity; a relabelled local closure fails."""

    if (
        payload.get("schema_version") != FORMAL_HOME_SCHEMA_VERSION
        or payload.get("status") != FORMAL_HOME_STATUS
        or payload.get("validation_class") != FORMAL_HOME_VALIDATION_CLASS
    ):
        raise LocalGate2PreparationError("FORMAL_GATE2_CLOSURE_IDENTITY_INVALID")
    if (
        payload.get("formal_gate2_equivalent") is not True
        or payload.get("source_validation_class")
        != "LOCAL_GATE2_DEVELOPMENT_VALIDATION"
        or payload.get("input_bits_revalidated") is not True
        or payload.get("resource_bits_revalidated") is not True
    ):
        raise LocalGate2PreparationError("FORMAL_GATE2_REVALIDATION_PROVENANCE_INVALID")
    if (
        payload.get("prediction_outputs_reused") is not False
        or payload.get("attempt05_predictions_read") is not False
        or payload.get("gate2_started") is not False
        or payload.get("pilot_started") is not False
        or payload.get("attempt06_started") is not False
        or payload.get("execution_authorized") is not False
    ):
        raise LocalGate2PreparationError("FORMAL_GATE2_EXECUTION_PROVENANCE_INVALID")
    if payload.get("scientific_result") != NO_SCIENTIFIC_RESULT:
        raise LocalGate2PreparationError("FORMAL_GATE2_SCIENTIFIC_RESULT_FORBIDDEN")
    expected_root = expected_formal_root.expanduser().resolve()
    if Path(str(payload.get("root", ""))).expanduser().resolve() != expected_root:
        raise LocalGate2PreparationError("FORMAL_GATE2_ROOT_IDENTITY_INVALID")
    if payload.get("schedule_identity_sha256") != LOCAL_SCHEDULE_IDENTITY:
        raise LocalGate2PreparationError("FORMAL_GATE2_SCHEDULE_IDENTITY_INVALID")
    for field, width in (
        ("source_closure_sha256", 64),
        ("resource_audit_sha256", 64),
        ("overlay_sha256", 64),
        ("production_source_commit", 40),
        ("production_source_tree", 40),
        ("test_only_source_commit", 40),
    ):
        _require_hex_identity(
            payload.get(field),
            width=width,
            reason=f"FORMAL_GATE2_{field.upper()}_IDENTITY_INVALID",
        )
    if payload.get("model_ids") != ["VGGT", "MASt3R"]:
        raise LocalGate2PreparationError("FORMAL_GATE2_MODEL_IDENTITY_INVALID")
    _validate_formal_bindings(payload)
    return dict(payload)


def write_formal_home_closure(
    formal_root: Path, payload: Mapping[str, object]
) -> Path:
    """Write one immutable formal closure below a distinct home-owned root."""

    resolved = require_home_owned_root(formal_root)
    validated = validate_formal_home_closure_payload(
        payload, expected_formal_root=resolved
    )
    path = resolved / "manifests" / "formal-gate2-input-closure.json"
    if path.exists():
        raise LocalGate2PreparationError(
            f"FORMAL_GATE2_CLOSURE_NO_CLOBBER_COLLISION:{path}"
        )
    _atomic_write(path, _canonical_bytes(validated))
    return path


def validate_formal_gate2_authorization(
    authorization: Mapping[str, object],
    *,
    expected_closure_sha256: str,
    expected_output_root: Path,
) -> dict[str, object]:
    """Fail closed unless one machine-readable formal Gate 2 run is authorized."""

    _require_hex_identity(
        expected_closure_sha256,
        width=64,
        reason="FORMAL_GATE2_EXPECTED_CLOSURE_DIGEST_INVALID",
    )
    if (
        authorization.get("schema_version") != FORMAL_GATE2_AUTH_SCHEMA_VERSION
        or authorization.get("status") != FORMAL_GATE2_AUTH_STATUS
        or authorization.get("validation_class") != "FORMAL_GATE2"
    ):
        raise LocalGate2PreparationError("FORMAL_GATE2_AUTH_IDENTITY_INVALID")
    if authorization.get("user_approved") is not True or not str(
        authorization.get("authorization_note", "")
    ).strip():
        raise LocalGate2PreparationError("FORMAL_GATE2_USER_AUTHORIZATION_REQUIRED")
    if authorization.get("formal_closure_sha256") != expected_closure_sha256:
        raise LocalGate2PreparationError("FORMAL_GATE2_CLOSURE_DIGEST_MISMATCH")
    if (
        authorization.get("gpu_uuid") != AUTHORIZED_GPU_UUID
        or authorization.get("physical_gpu_index")
        != AUTHORIZED_PHYSICAL_GPU_INDEX
    ):
        raise LocalGate2PreparationError("FORMAL_GATE2_GPU_IDENTITY_MISMATCH")
    if authorization.get("pilot_started") is not False or authorization.get(
        "attempt06_started"
    ) is not False:
        raise LocalGate2PreparationError("FORMAL_GATE2_DOWNSTREAM_PILOT_FORBIDDEN")
    if authorization.get("scientific_result") != NO_SCIENTIFIC_RESULT:
        raise LocalGate2PreparationError("FORMAL_GATE2_SCIENTIFIC_RESULT_FORBIDDEN")
    if authorization.get("gate2_started") is not False:
        raise LocalGate2PreparationError("FORMAL_GATE2_AUTH_ALREADY_STARTED")
    expected_output = expected_output_root.expanduser().resolve()
    if Path(str(authorization.get("output_root", ""))).expanduser().resolve() != (
        expected_output
    ):
        raise LocalGate2PreparationError("FORMAL_GATE2_AUTH_OUTPUT_ROOT_MISMATCH")
    for field, ceiling in (
        ("max_gpu_seconds", 21_600),
        ("max_wall_seconds", 43_200),
        ("max_storage_bytes", 25 * 1024**3),
    ):
        value = authorization.get(field)
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            raise LocalGate2PreparationError("FORMAL_GATE2_AUTH_BUDGET_INVALID")
        if value <= 0 or value > ceiling:
            raise LocalGate2PreparationError("FORMAL_GATE2_AUTH_BUDGET_INVALID")
    for field in ("production_source_commit", "test_only_source_commit"):
        _require_hex_identity(
            authorization.get(field),
            width=40,
            reason="FORMAL_GATE2_AUTH_SOURCE_IDENTITY_INVALID",
        )
    if authorization.get("schedule_identity_sha256") != LOCAL_SCHEDULE_IDENTITY:
        raise LocalGate2PreparationError(
            "FORMAL_GATE2_AUTH_SCHEDULE_IDENTITY_INVALID"
        )
    return dict(authorization)


def materialize(root: Path, *, archive_url: str = RECTIFIED_URL) -> Path:
    resolved = require_home_owned_root(root)
    log = PreparationLog(resolved)
    write_plan(resolved)
    log.write(
        "LOCAL_PREPARATION_START",
        root=str(resolved),
        member_count=len(selected_member_paths()),
    )
    try:
        if archive_url != RECTIFIED_URL:
            raise LocalGate2PreparationError("LOCAL_GATE2_ARCHIVE_URL_MISMATCH")
        index = index_remote_zip(archive_url)
        identity = _archive_identity(index)
        _validate_archive_identity(identity)
        log.write("REMOTE_INDEX_VERIFIED", **identity)
        extraction = extract_range_members_evidence(
            archive_url,
            index,
            selected_member_paths(),
            resolved / "data" / "dtu-l3",
        )
        closure = _closure_payload(
            resolved,
            archive_identity=identity,
            extraction=extraction,
        )
        closure_path = resolved / "manifests" / "local-gate2-input-closure.json"
        _atomic_write(closure_path, _canonical_bytes(closure))
        validate_local_closure(closure_path)
        log.write(
            "LOCAL_PREPARATION_COMPLETE",
            closure_path=str(closure_path),
            closure_sha256=sha256_file(closure_path),
            member_count=closure["member_count"],
            status=LOCAL_STATUS,
        )
        return closure_path
    except BaseException as exc:
        log.write(
            "LOCAL_PREPARATION_FAILED",
            exception_type=type(exc).__name__,
            reason_code=str(exc),
        )
        raise


def audit_resources(root: Path) -> Path:
    """Fail closed unless both real-model resource bindings are frozen."""

    resolved = require_home_owned_root(root)
    overlay = resolved / "manifests" / "local-gate2-overlay.toml"
    if not overlay.is_file():
        raise LocalGate2PreparationError("LOCAL_GATE2_OVERLAY_MISSING")
    context = SimpleNamespace(
        root=resolved,
        output_root=resolved / "runs" / "resource-audit-placeholder",
        config_path=overlay,
        device="cuda:0",
    )
    rows = []
    log = PreparationLog(resolved)
    try:
        for model in ("VGGT", "MASt3R"):
            runtime = _frozen_runtime_from_config(model, context)
            evidence = verify_frozen_runtime(model, runtime)
            environment = evidence.environment
            if not isinstance(environment, Mapping):
                raise LocalGate2PreparationError(
                    f"LOCAL_GATE2_RESOURCE_ENVIRONMENT_INVALID:{model}"
                )
            rows.append(
                {
                    "model": model,
                    "source_commit": evidence.source_commit,
                    "checkpoint_sha256": evidence.checkpoint_sha256,
                    "config_sha256": evidence.config_sha256,
                    "dust3r_source_commit": evidence.dust3r_source_commit,
                    "croco_source_commit": evidence.croco_source_commit,
                    "environment": dict(environment),
                    "python_version": str(environment["python"]),
                    "torch_version": str(environment["torch"]),
                    "typing_extensions_version": str(
                        environment["typing_extensions_version"]
                    ),
                }
            )
    except Exception as exc:
        reason = f"LOCAL_GATE2_RESOURCE_AUDIT_FAILED:{model}:{type(exc).__name__}:{exc}"
        log.write(
            "LOCAL_RESOURCE_AUDIT_FAILED",
            model=model,
            exception_type=type(exc).__name__,
            reason_code=reason,
        )
        raise LocalGate2PreparationError(reason) from exc
    payload = {
        "schema_version": "georeliab-v4-local-gate2-resource-audit-1.0",
        "status": "LOCAL_GATE2_DEVELOPMENT_RESOURCES_READY",
        "validation_class": "LOCAL_GATE2_DEVELOPMENT_VALIDATION",
        "formal_gate2_equivalent": False,
        "overlay_path": str(overlay),
        "overlay_sha256": sha256_file(overlay),
        "models": rows,
        "scientific_result": NO_SCIENTIFIC_RESULT,
    }
    output = resolved / "manifests" / "local-gate2-resource-audit.json"
    _atomic_write(output, _canonical_bytes(payload))
    log.write(
        "LOCAL_RESOURCE_AUDIT_COMPLETE",
        status=payload["status"],
        evidence_path=str(output),
        evidence_sha256=sha256_file(output),
    )
    return output


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    for name in ("plan", "materialize", "audit", "audit-resources"):
        command = subparsers.add_parser(name)
        command.add_argument("--root", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    root = require_home_owned_root(args.root)
    if args.command == "plan":
        path = write_plan(root)
        print(path)
        return 0
    if args.command == "materialize":
        print(materialize(root))
        return 0
    if args.command == "audit-resources":
        print(audit_resources(root))
        return 0
    closure = root / "manifests" / "local-gate2-input-closure.json"
    validate_local_closure(closure)
    print(f"{LOCAL_STATUS} {closure}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
