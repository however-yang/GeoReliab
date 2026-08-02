from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import time
from typing import Any
from uuid import uuid4

from . import toml_compat as tomllib
from .v4_counterfactuals import SCIENTIFIC_MODELS
from .v4_execution import (
    BYTE_CATASTROPHE,
    BYTE_TARGET,
    GPU_CATASTROPHE_SECONDS,
    GPU_TARGET_SECONDS,
    V4ExecutionError,
)
from .v4_rectified_closure import (
    EXPECTED_SET_NAME,
    MANIFEST_NAME,
    MATERIALIZE_RECEIPT_NAME,
    RECEIPT_NAME,
    V4RectifiedClosureError,
    audit_rectified_member_closure_read_only,
)
from .v4_science_lock import (
    GEORELIAB_V4_PROTOCOL_READY,
    GPU_SELECTION_REQUIRED,
    NO_SCIENTIFIC_RESULT,
    V4_PROTOCOL_ID,
    V4_PROTOCOL_SHA256,
    V4ScienceLockError,
    validate_v4_science_lock,
)


ATTEMPT_ID = 'attempt-03'
SCHEMA_PREFIX = 'georeliab-v4-attempt-03'
RESOURCE_SCHEMA = f'{SCHEMA_PREFIX}-resource-revalidation-1.0'
INVENTORY_SCHEMA = f'{SCHEMA_PREFIX}-gpu-inventory-1.0'
PREFLIGHT_SCHEMA = f'{SCHEMA_PREFIX}-gpu-preflight-1.0'
RECEIPT_SCHEMA = f'{SCHEMA_PREFIX}-gpu-receipt-1.0'
AUTHORIZATION_SCHEMA = f'{SCHEMA_PREFIX}-execution-authorization-1.0'

SCIENTIFIC_ANCHOR_COMMIT = (
    '7381e60050143a78fca6a3ebde5706ae27d2c145'
)
SCIENTIFIC_ANCHOR_TREE = (
    'f4e2b1104496c817693aaa5989d0276d2ebe03e9'
)
AUTHORIZED_GPU_MODEL = 'NVIDIA A100 80GB PCIe'
SAMPLE_COUNT = 3
SAMPLE_INTERVAL_SECONDS = 5.0
NVIDIA_SMI_CALLS_PER_SAMPLE = 3
MIN_FREE_MEMORY_BYTES = 16 * 1024 * 1024 * 1024
AUTHORIZED_FINALIZER = (
    'georeliab_mve.v4_execution:finalize_v4_scientific_bundle'
)

SCHEDULE_FILE_SHA256 = (
    '47ed0464409d0189cb301930ecaf8db5b40b540ef0c5459dfea01fd92444a6c3'
)
EXPECTED_SET_FILE_SHA256 = (
    'b64139b0c89b6a2dd5b94982d372daee3746c3852a8f5a4a597a8d4ff456d450'
)
MATERIALIZE_RECEIPT_FILE_SHA256 = (
    '9af0fc7e832466ffd2de5c8a4fecf2f67513804797342edca3cf01ad67e889a9'
)
MANIFEST_FILE_SHA256 = (
    '1634e75bd09ca2b446ef32d768eddcd7fc547473f19c304b53b5711d7ac53dbb'
)
CLOSURE_RECEIPT_FILE_SHA256 = (
    '7a4d77966102812313a89060af36decb0fe5add6291ca018c406e5abe3df30b5'
)
ORDERED_MEMBER_LIST_SHA256 = (
    '521ef283be964c195f77acc55a1e8458c16302a3d74e1bc2cf82c6582d7e0377'
)
GROUP_INDEX_SHA256 = (
    'bfaa2423b554518b6648e2077cbab77fe568025b1a05ad91c7674cd271145017'
)
SCHEDULE_BINDING_SHA256 = (
    '42785287dbc4be2854bb0bfb3df3881f944942a26fdc422e37905b243a09e930'
)

EXPECTED_MEMBER_ILLUMINATIONS = ['L1', 'L2', 'L4', 'L5', 'L6', 'L7']
EXPECTED_L3_ROLE = 'REFERENCE_EXCLUDED_FROM_RECTIFIED_MEMBER_CLOSURE'
EXPECTED_MEMBER_COUNT = 960
EXPECTED_GROUP_COUNT = 160

_EXPECTED_CLOSURE_DIGESTS = {
    'schedule': SCHEDULE_FILE_SHA256,
    'expected_set': EXPECTED_SET_FILE_SHA256,
    'materialize_receipt': MATERIALIZE_RECEIPT_FILE_SHA256,
    'manifest': MANIFEST_FILE_SHA256,
    'closure_receipt': CLOSURE_RECEIPT_FILE_SHA256,
    'ordered_member_list': ORDERED_MEMBER_LIST_SHA256,
    'group_index': GROUP_INDEX_SHA256,
    'schedule_binding': SCHEDULE_BINDING_SHA256,
}

_EXPECTED_FILE_DIGESTS = {
    'schedule': SCHEDULE_FILE_SHA256,
    'expected_set': EXPECTED_SET_FILE_SHA256,
    'materialize_receipt': MATERIALIZE_RECEIPT_FILE_SHA256,
    'manifest': MANIFEST_FILE_SHA256,
    'closure_receipt': CLOSURE_RECEIPT_FILE_SHA256,
}
_NUMERICAL_ANCHOR_PATHS = (
    'georeliab_mve/adapters.py',
    'georeliab_mve/runner.py',
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def _json_bytes(value: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(
            dict(value),
            sort_keys=True,
            separators=(',', ':'),
            ensure_ascii=True,
            allow_nan=False,
        )
        + '\n'
    ).encode('ascii')


def _sha_json(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(_json_bytes(value)).hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding='utf-8'))
    except (OSError, json.JSONDecodeError) as exc:
        raise V4ExecutionError('V4_ATTEMPT03_JSON_UNREADABLE') from exc
    if not isinstance(value, dict):
        raise V4ExecutionError('V4_ATTEMPT03_JSON_OBJECT_REQUIRED')
    return value


def _is_sha(value: object, length: int = 64) -> bool:
    return (
        isinstance(value, str)
        and len(value) == length
        and value == value.lower()
        and all(char in '0123456789abcdef' for char in value)
    )


def _signed(payload: Mapping[str, Any], field: str) -> dict[str, Any]:
    result = dict(payload)
    result[field] = _sha_json(payload)
    return result


def _validate_signature(
    payload: Mapping[str, Any], field: str, reason: str
) -> None:
    expected = payload.get(field)
    unsigned = {key: value for key, value in payload.items() if key != field}
    if not _is_sha(expected) or _sha_json(unsigned) != expected:
        raise V4ExecutionError(reason)


def _atomic_json(
    path: Path,
    payload: Mapping[str, Any],
    validator: Callable[[Mapping[str, Any]], None],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + '.partial')
    if path.exists() or partial.exists():
        raise V4ExecutionError('V4_ATTEMPT03_ARTIFACT_COLLISION')
    try:
        with partial.open('wb') as handle:
            handle.write(_json_bytes(payload))
            handle.flush()
            try:
                os.fsync(handle.fileno())
            except OSError:
                pass
        staged = _load_json(partial)
        validator(staged)
        partial.replace(path)
    finally:
        try:
            partial.unlink()
        except FileNotFoundError:
            pass


def _run_text(command: Sequence[str]) -> str:
    completed = subprocess.run(
        tuple(command),
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    return completed.stdout


def _git_value(worktree: Path, *args: str) -> str:
    try:
        return _run_text(('git', '-C', str(worktree), *args)).strip()
    except (OSError, subprocess.SubprocessError) as exc:
        raise V4ExecutionError(
            'V4_ATTEMPT03_AUTHORIZATION_REVISION_UNRESOLVED'
        ) from exc


def _tooling_revision(worktree: Path) -> tuple[str, str]:
    commit = _git_value(worktree, 'rev-parse', 'HEAD')
    tree = _git_value(worktree, 'rev-parse', 'HEAD^{tree}')
    if not _is_sha(commit, 40) or not _is_sha(tree, 40):
        raise V4ExecutionError(
            'V4_ATTEMPT03_AUTHORIZATION_REVISION_UNRESOLVED'
        )
    if _git_value(worktree, 'status', '--porcelain'):
        raise V4ExecutionError('V4_ATTEMPT03_WORKTREE_NOT_CLEAN')
    return commit, tree


def _attempt_path(path: Path) -> Path:
    resolved = path.resolve()
    parts = tuple(part.lower() for part in resolved.parts)
    if parts.count(ATTEMPT_ID) != 1:
        raise V4ExecutionError('V4_ATTEMPT03_PATH_SCOPE_MISMATCH')
    if 'attempt-01' in parts or 'attempt-02' in parts:
        raise V4ExecutionError('V4_ATTEMPT03_HISTORY_REUSE_FORBIDDEN')
    return resolved


def _attempt_root(path: Path) -> Path:
    resolved = _attempt_path(path)
    index = tuple(part.lower() for part in resolved.parts).index(ATTEMPT_ID)
    return Path(*resolved.parts[: index + 1])


def _required_attempt_root(runtime_root: Path) -> Path:
    return (
        runtime_root.resolve()
        / 'authorization-attempts'
        / ATTEMPT_ID
    )


def _require_attempt_root(runtime_root: Path, path: Path) -> Path:
    resolved = _attempt_path(path)
    if _attempt_root(resolved) != _required_attempt_root(runtime_root):
        raise V4ExecutionError('V4_ATTEMPT03_PATH_SCOPE_MISMATCH')
    return resolved


def _require_fresh_artifact(path: Path) -> None:
    if path.exists() or path.with_name(path.name + '.partial').exists():
        raise V4ExecutionError('V4_ATTEMPT03_ARTIFACT_COLLISION')


def _assert_file_digest(path: Path, expected: str, reason: str) -> None:
    if not path.is_file() or _sha256_file(path) != expected:
        raise V4ExecutionError(reason)


def _file_binding(path: Path) -> dict[str, object]:
    if not path.is_file():
        raise V4ExecutionError('V4_ATTEMPT03_RESOURCE_FILE_MISSING')
    return {
        'path': str(path.resolve()),
        'sha256': _sha256_file(path),
        'bytes': path.stat().st_size,
    }


def _anchor_blob(worktree: Path, relative: str) -> bytes:
    try:
        completed = subprocess.run(
            (
                'git',
                '-C',
                str(worktree),
                'show',
                f'{SCIENTIFIC_ANCHOR_COMMIT}:{relative}',
            ),
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise V4ExecutionError(
            'V4_ATTEMPT03_SCIENTIFIC_ANCHOR_UNAVAILABLE'
        ) from exc
    return completed.stdout


def _validate_numerical_anchor(worktree: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for relative in _NUMERICAL_ANCHOR_PATHS:
        current = (worktree / relative).read_bytes()
        anchor = _anchor_blob(worktree, relative)
        if current != anchor:
            raise V4ExecutionError(
                'V4_ATTEMPT03_NUMERICAL_PATH_DRIFT'
            )
        rows.append(
            {
                'path': relative,
                'sha256': hashlib.sha256(current).hexdigest(),
            }
        )
    return rows


def _read_overlay(path: Path) -> dict[str, Any]:
    try:
        payload = tomllib.loads(path.read_text(encoding='utf-8'))
    except (OSError, tomllib.TOMLDecodeError) as exc:
        raise V4ExecutionError('V4_ATTEMPT03_OVERLAY_UNREADABLE') from exc
    if not isinstance(payload, dict):
        raise V4ExecutionError('V4_ATTEMPT03_OVERLAY_UNREADABLE')
    return payload


def _resource_bindings(
    *,
    runtime_root: Path,
    overlay_path: Path,
    materialize_receipt_path: Path,
    manifest_path: Path,
    closure_receipt_path: Path,
) -> dict[str, object]:
    overlay = _read_overlay(overlay_path)
    resources = overlay.get('resources')
    if not isinstance(resources, Mapping):
        raise V4ExecutionError('V4_ATTEMPT03_OVERLAY_RESOURCES_REQUIRED')
    sampleset = runtime_root / 'cache' / 'SampleSet.zip'
    points = runtime_root / 'cache' / 'Points.zip'
    environment = (
        runtime_root / 'artifacts' / 'environment_locks'
        / 'a100_environment_lock.json'
    )
    for path, key in (
        (sampleset, 'dtu_sampleset_sha256'),
        (points, 'dtu_points_sha256'),
    ):
        expected = resources.get(key)
        if not _is_sha(expected):
            raise V4ExecutionError('V4_ATTEMPT03_RESOURCE_SHA_REQUIRED')
        _assert_file_digest(
            path, str(expected), 'V4_ATTEMPT03_RESOURCE_HASH_MISMATCH'
        )
    models = {
        'VGGT': {
            'checkpoint': _file_binding(
                Path(str(resources.get('vggt_checkpoint')))
            ),
            'checkpoint_sha256': resources.get(
                'vggt_checkpoint_sha256'
            ),
            'upstream_commits': {
                'vggt': resources.get('vggt_source_commit'),
            },
        },
        'MASt3R': {
            'checkpoint': _file_binding(
                Path(str(resources.get('mast3r_checkpoint')))
            ),
            'checkpoint_sha256': resources.get(
                'mast3r_checkpoint_sha256'
            ),
            'config': _file_binding(
                Path(str(resources.get('mast3r_config')))
            ),
            'config_sha256': resources.get('mast3r_config_sha256'),
            'upstream_commits': {
                'mast3r': resources.get('mast3r_source_commit'),
                'dust3r': resources.get('dust3r_source_commit'),
                'croco': resources.get('croco_source_commit'),
            },
        },
    }
    for model in models.values():
        checkpoint = model['checkpoint']
        if checkpoint['sha256'] != model['checkpoint_sha256']:
            raise V4ExecutionError(
                'V4_ATTEMPT03_MODEL_RESOURCE_HASH_MISMATCH'
            )
        config = model.get('config')
        if isinstance(config, Mapping):
            if config['sha256'] != model.get('config_sha256'):
                raise V4ExecutionError(
                    'V4_ATTEMPT03_MODEL_RESOURCE_HASH_MISMATCH'
                )
        commits = model.get('upstream_commits')
        if not isinstance(commits, Mapping) or any(
            not _is_sha(value, 40) for value in commits.values()
        ):
            raise V4ExecutionError(
                'V4_ATTEMPT03_UPSTREAM_COMMIT_REQUIRED'
            )
    rectified_url = resources.get('dtu_rectified_url')
    rectified_bytes = resources.get('dtu_rectified_bytes')
    rectified_etag = resources.get('dtu_rectified_etag')
    if (
        not isinstance(rectified_url, str)
        or not rectified_url
        or type(rectified_bytes) is not int
        or rectified_bytes <= 0
        or not isinstance(rectified_etag, str)
        or not rectified_etag
    ):
        raise V4ExecutionError(
            'V4_ATTEMPT03_RECTIFIED_SOURCE_IDENTITY_REQUIRED'
        )
    rectified = {
        'url': rectified_url,
        'bytes': rectified_bytes,
        'etag': rectified_etag,
        'central_directory_identity': {
            'identity_kind': (
                'official_archive_length_etag_and_frozen_materialization'
            ),
            'archive_bytes': rectified_bytes,
            'archive_etag': rectified_etag,
            'materialize_receipt': _file_binding(
                materialize_receipt_path
            ),
        },
        'referenced_member_inventory': {
            'member_count': EXPECTED_MEMBER_COUNT,
            'manifest': _file_binding(manifest_path),
            'ordered_member_list_sha256': ORDERED_MEMBER_LIST_SHA256,
            'group_index_sha256': GROUP_INDEX_SHA256,
            'schedule_binding_sha256': SCHEDULE_BINDING_SHA256,
            'closure_receipt': _file_binding(closure_receipt_path),
        },
    }
    return {
        'overlay': _file_binding(overlay_path),
        'models': models,
        'dtu_archives': {
            'SampleSet': _file_binding(sampleset),
            'Points': _file_binding(points),
            'Rectified': rectified,
        },
        'environment_lock': _file_binding(environment),
    }


def _validate_resource_payload(payload: Mapping[str, Any]) -> None:
    if payload.get('schema_version') != RESOURCE_SCHEMA:
        raise V4ExecutionError('V4_ATTEMPT03_RESOURCE_SCHEMA_REQUIRED')
    if payload.get('attempt_id') != ATTEMPT_ID:
        raise V4ExecutionError('V4_ATTEMPT03_ID_MISMATCH')
    if (
        payload.get('status') != 'PASS'
        or payload.get('reason_code')
        != 'V4_RESOURCE_CLOSURE_REVALIDATED'
    ):
        raise V4ExecutionError('V4_RESOURCE_CLOSURE_REVALIDATION_FAILED')
    if (
        not _is_sha(payload.get('tooling_commit'), 40)
        or not _is_sha(payload.get('tooling_tree'), 40)
    ):
        raise V4ExecutionError('V4_ATTEMPT03_RESOURCE_RECEIPT_TAMPER')
    if payload.get('scientific_anchor_commit') != SCIENTIFIC_ANCHOR_COMMIT:
        raise V4ExecutionError('V4_ATTEMPT03_SCIENTIFIC_ANCHOR_MISMATCH')
    if payload.get('scientific_anchor_tree') != SCIENTIFIC_ANCHOR_TREE:
        raise V4ExecutionError('V4_ATTEMPT03_SCIENTIFIC_ANCHOR_MISMATCH')
    if (
        payload.get('protocol_id') != V4_PROTOCOL_ID
        or payload.get('protocol_sha256') != V4_PROTOCOL_SHA256
        or payload.get('schedule_file_sha256')
        != SCHEDULE_FILE_SHA256
        or payload.get('ordered_member_list_sha256')
        != ORDERED_MEMBER_LIST_SHA256
        or payload.get('group_index_sha256') != GROUP_INDEX_SHA256
        or payload.get('schedule_binding_sha256')
        != SCHEDULE_BINDING_SHA256
    ):
        raise V4ExecutionError('V4_ATTEMPT03_RESOURCE_RECEIPT_TAMPER')
    if (
        payload.get('member_count') != EXPECTED_MEMBER_COUNT
        or payload.get('group_count') != EXPECTED_GROUP_COUNT
        or payload.get('member_illuminations')
        != EXPECTED_MEMBER_ILLUMINATIONS
        or payload.get('l3_role') != EXPECTED_L3_ROLE
    ):
        raise V4ExecutionError('V4_RESOURCE_CLOSURE_REVALIDATION_FAILED')
    if any(
        payload.get(key) != 0
        for key in (
            'missing_count',
            'duplicate_count',
            'orphan_count',
            'symlink_count',
            'partial_count',
        )
    ):
        raise V4ExecutionError('V4_RESOURCE_CLOSURE_REVALIDATION_FAILED')
    if (
        payload.get('nvidia_smi_invocations') != 0
        or payload.get('torch_probe_invocations') != 0
        or payload.get('model_loads') != 0
        or payload.get('model_forwards') != 0
        or payload.get('scientific_result') != NO_SCIENTIFIC_RESULT
    ):
        raise V4ExecutionError('V4_ATTEMPT03_GPU_PROBE_SCOPE_VIOLATION')
    science_lock = payload.get('science_lock')
    if (
        not isinstance(science_lock, Mapping)
        or science_lock.get('status') != GEORELIAB_V4_PROTOCOL_READY
        or science_lock.get('execution_status') != GPU_SELECTION_REQUIRED
        or science_lock.get('scientific_result_status')
        != NO_SCIENTIFIC_RESULT
        or science_lock.get('protocol_id') != V4_PROTOCOL_ID
        or science_lock.get('protocol_sha256') != V4_PROTOCOL_SHA256
    ):
        raise V4ExecutionError('V4_ATTEMPT03_SCIENCE_LOCK_TAMPER')
    numerical = payload.get('numerical_anchor_paths')
    if (
        not isinstance(numerical, list)
        or any(not isinstance(item, Mapping) for item in numerical)
        or [item.get('path') for item in numerical]
        != list(_NUMERICAL_ANCHOR_PATHS)
        or any(
            not _is_sha(item.get('sha256'))
            for item in numerical
        )
    ):
        raise V4ExecutionError('V4_ATTEMPT03_NUMERICAL_PATH_DRIFT')
    closure_files = payload.get('closure_files')
    if (
        not isinstance(closure_files, Mapping)
        or set(closure_files) != set(_EXPECTED_FILE_DIGESTS)
    ):
        raise V4ExecutionError('V4_ATTEMPT03_RESOURCE_RECEIPT_TAMPER')
    for label, expected in _EXPECTED_FILE_DIGESTS.items():
        item = closure_files.get(label)
        if (
            not isinstance(item, Mapping)
            or item.get('sha256') != expected
            or not isinstance(item.get('path'), str)
            or not item.get('path')
        ):
            raise V4ExecutionError(
                'V4_ATTEMPT03_RESOURCE_RECEIPT_TAMPER'
            )
    _validate_signature(
        payload,
        'resource_revalidation_sha256',
        'V4_ATTEMPT03_RESOURCE_RECEIPT_TAMPER',
    )


def _validate_resource_failure_payload(payload: Mapping[str, Any]) -> None:
    if payload.get('schema_version') != RESOURCE_SCHEMA:
        raise V4ExecutionError('V4_ATTEMPT03_RESOURCE_SCHEMA_REQUIRED')
    if payload.get('attempt_id') != ATTEMPT_ID:
        raise V4ExecutionError('V4_ATTEMPT03_ID_MISMATCH')
    if (
        payload.get('status') != 'FAIL'
        or payload.get('reason_code')
        != 'V4_RESOURCE_CLOSURE_REVALIDATION_FAILED'
    ):
        raise V4ExecutionError('V4_ATTEMPT03_RESOURCE_BLOCKER_TAMPER')
    if (
        not _is_sha(payload.get('tooling_commit'), 40)
        or not _is_sha(payload.get('tooling_tree'), 40)
        or payload.get('scientific_anchor_commit')
        != SCIENTIFIC_ANCHOR_COMMIT
        or payload.get('scientific_anchor_tree') != SCIENTIFIC_ANCHOR_TREE
        or payload.get('protocol_id') != V4_PROTOCOL_ID
        or payload.get('protocol_sha256') != V4_PROTOCOL_SHA256
    ):
        raise V4ExecutionError('V4_ATTEMPT03_RESOURCE_BLOCKER_TAMPER')
    underlying = payload.get('underlying_reason_code')
    if not isinstance(underlying, str) or not underlying:
        raise V4ExecutionError('V4_ATTEMPT03_RESOURCE_BLOCKER_TAMPER')
    runtime_root = Path(str(payload.get('runtime_root')))
    attempt_root = Path(str(payload.get('attempt_root')))
    if (
        not runtime_root.is_absolute()
        or attempt_root != _required_attempt_root(runtime_root)
    ):
        raise V4ExecutionError('V4_ATTEMPT03_PATH_SCOPE_MISMATCH')
    if (
        payload.get('nvidia_smi_invocations') != 0
        or payload.get('torch_probe_invocations') != 0
        or payload.get('model_loads') != 0
        or payload.get('model_forwards') != 0
        or payload.get('gpu_inference_seconds') != 0
        or payload.get('pass_receipt_generated') is not False
        or payload.get('execution_authorization_generated') is not False
        or payload.get('scientific_result') != NO_SCIENTIFIC_RESULT
    ):
        raise V4ExecutionError('V4_ATTEMPT03_GPU_PROBE_SCOPE_VIOLATION')
    _validate_signature(
        payload,
        'resource_revalidation_sha256',
        'V4_ATTEMPT03_RESOURCE_BLOCKER_TAMPER',
    )


def _verify_file_binding(value: object) -> None:
    if not isinstance(value, Mapping):
        raise V4ExecutionError('V4_ATTEMPT03_RESOURCE_RECEIPT_TAMPER')
    path = Path(str(value.get('path')))
    digest = value.get('sha256')
    if (
        not _is_sha(digest)
        or not path.is_file()
        or _sha256_file(path) != digest
        or path.stat().st_size != value.get('bytes')
    ):
        raise V4ExecutionError('V4_ATTEMPT03_RESOURCE_RECEIPT_TAMPER')


def _verify_resource_bindings(value: object) -> None:
    if not isinstance(value, Mapping):
        raise V4ExecutionError('V4_ATTEMPT03_RESOURCE_RECEIPT_TAMPER')
    _verify_file_binding(value.get('overlay'))
    _verify_file_binding(value.get('environment_lock'))
    models = value.get('models')
    if not isinstance(models, Mapping) or set(models) != set(
        SCIENTIFIC_MODELS
    ):
        raise V4ExecutionError('V4_ATTEMPT03_RESOURCE_RECEIPT_TAMPER')
    for model in models.values():
        if not isinstance(model, Mapping):
            raise V4ExecutionError(
                'V4_ATTEMPT03_RESOURCE_RECEIPT_TAMPER'
            )
        _verify_file_binding(model.get('checkpoint'))
        checkpoint = model.get('checkpoint')
        if (
            not isinstance(checkpoint, Mapping)
            or model.get('checkpoint_sha256')
            != checkpoint.get('sha256')
        ):
            raise V4ExecutionError(
                'V4_ATTEMPT03_RESOURCE_RECEIPT_TAMPER'
            )
        if model.get('config') is not None:
            _verify_file_binding(model.get('config'))
            config = model.get('config')
            if (
                not isinstance(config, Mapping)
                or model.get('config_sha256') != config.get('sha256')
            ):
                raise V4ExecutionError(
                    'V4_ATTEMPT03_RESOURCE_RECEIPT_TAMPER'
                )
        commits = model.get('upstream_commits')
        if not isinstance(commits, Mapping) or any(
            not _is_sha(commit, 40) for commit in commits.values()
        ):
            raise V4ExecutionError(
                'V4_ATTEMPT03_RESOURCE_RECEIPT_TAMPER'
            )
    archives = value.get('dtu_archives')
    if not isinstance(archives, Mapping):
        raise V4ExecutionError('V4_ATTEMPT03_RESOURCE_RECEIPT_TAMPER')
    _verify_file_binding(archives.get('SampleSet'))
    _verify_file_binding(archives.get('Points'))
    rectified = archives.get('Rectified')
    if (
        not isinstance(rectified, Mapping)
        or not rectified.get('url')
        or not rectified.get('etag')
        or type(rectified.get('bytes')) is not int
        or rectified.get('bytes', 0) <= 0
    ):
        raise V4ExecutionError('V4_ATTEMPT03_RESOURCE_RECEIPT_TAMPER')
    central = rectified.get('central_directory_identity')
    members = rectified.get('referenced_member_inventory')
    if (
        not isinstance(central, Mapping)
        or central.get('identity_kind')
        != 'official_archive_length_etag_and_frozen_materialization'
        or central.get('archive_bytes') != rectified.get('bytes')
        or central.get('archive_etag') != rectified.get('etag')
        or not isinstance(members, Mapping)
        or members.get('member_count') != EXPECTED_MEMBER_COUNT
        or members.get('ordered_member_list_sha256')
        != ORDERED_MEMBER_LIST_SHA256
        or members.get('group_index_sha256') != GROUP_INDEX_SHA256
        or members.get('schedule_binding_sha256')
        != SCHEDULE_BINDING_SHA256
    ):
        raise V4ExecutionError('V4_ATTEMPT03_RESOURCE_RECEIPT_TAMPER')
    _verify_file_binding(central.get('materialize_receipt'))
    _verify_file_binding(members.get('manifest'))
    _verify_file_binding(members.get('closure_receipt'))
    if (
        central['materialize_receipt'].get('sha256')
        != MATERIALIZE_RECEIPT_FILE_SHA256
        or members['manifest'].get('sha256') != MANIFEST_FILE_SHA256
        or members['closure_receipt'].get('sha256')
        != CLOSURE_RECEIPT_FILE_SHA256
    ):
        raise V4ExecutionError('V4_ATTEMPT03_RESOURCE_RECEIPT_TAMPER')


def _revalidate_attempt03_resources_pass(
    *,
    worktree: Path,
    runtime_root: Path,
    rectified_root: Path,
    closure_root: Path,
    overlay_path: Path,
    output_path: Path,
) -> dict[str, object]:
    runtime_root = runtime_root.resolve()
    output_path = _require_attempt_root(runtime_root, output_path)
    _require_fresh_artifact(output_path)
    closure_root = _under_root(runtime_root, closure_root)
    rectified_root = _under_root(runtime_root, rectified_root)
    bootstrap = closure_root / 'bootstrap'
    materialize = closure_root / 'materialize'
    validation = closure_root / 'validation'
    schedule_path = bootstrap / 'v4-rectified-resource-schedule-400.json'
    expected_path = bootstrap / EXPECTED_SET_NAME
    materialize_path = materialize / MATERIALIZE_RECEIPT_NAME
    manifest_path = validation / MANIFEST_NAME
    closure_receipt_path = validation / RECEIPT_NAME
    paths = {
        'schedule': schedule_path,
        'expected_set': expected_path,
        'materialize_receipt': materialize_path,
        'manifest': manifest_path,
        'closure_receipt': closure_receipt_path,
    }
    for label, path in paths.items():
        _assert_file_digest(
            path,
            _EXPECTED_FILE_DIGESTS[label],
            'V4_RESOURCE_CLOSURE_REVALIDATION_FAILED',
        )
    if (
        next(closure_root.rglob('*.partial'), None) is not None
        or next(rectified_root.rglob('*.partial'), None) is not None
    ):
        raise V4ExecutionError('V4_RESOURCE_CLOSURE_REVALIDATION_FAILED')
    science_lock = validate_v4_science_lock(worktree)
    numerical_paths = _validate_numerical_anchor(worktree)
    schedule = _load_json(schedule_path)
    if schedule.get('unit_count') != 400:
        raise V4ExecutionError('V4_RESOURCE_CLOSURE_REVALIDATION_FAILED')
    audit = audit_rectified_member_closure_read_only(
        root=rectified_root,
        manifest_path=manifest_path,
        expected_set_path=expected_path,
    )
    receipt = _load_json(closure_receipt_path)
    if (
        audit.get('status') != 'PASS'
        or audit.get('manifest_rows') != EXPECTED_MEMBER_COUNT
        or audit.get('group_count') != EXPECTED_GROUP_COUNT
        or receipt.get('status') != 'PASS'
        or receipt.get('expected_base_units') != EXPECTED_GROUP_COUNT
        or receipt.get('expected_member_count') != EXPECTED_MEMBER_COUNT
        or receipt.get('excluded_reference_role') != {
            'L3': EXPECTED_L3_ROLE
        }
        or audit.get('ordered_member_list_sha256')
        != ORDERED_MEMBER_LIST_SHA256
        or audit.get('group_index_sha256') != GROUP_INDEX_SHA256
        or audit.get('schedule_to_member_binding_sha256')
        != SCHEDULE_BINDING_SHA256
    ):
        raise V4ExecutionError('V4_RESOURCE_CLOSURE_REVALIDATION_FAILED')
    tooling_commit, tooling_tree = _tooling_revision(worktree)
    resources = _resource_bindings(
        runtime_root=runtime_root,
        overlay_path=overlay_path,
        materialize_receipt_path=materialize_path,
        manifest_path=manifest_path,
        closure_receipt_path=closure_receipt_path,
    )
    payload = _signed(
        {
            'schema_version': RESOURCE_SCHEMA,
            'attempt_id': ATTEMPT_ID,
            'status': 'PASS',
            'reason_code': 'V4_RESOURCE_CLOSURE_REVALIDATED',
            'tooling_commit': tooling_commit,
            'tooling_tree': tooling_tree,
            'scientific_anchor_commit': SCIENTIFIC_ANCHOR_COMMIT,
            'scientific_anchor_tree': SCIENTIFIC_ANCHOR_TREE,
            'protocol_id': V4_PROTOCOL_ID,
            'protocol_sha256': V4_PROTOCOL_SHA256,
            'runtime_root': str(runtime_root),
            'attempt_root': str(_required_attempt_root(runtime_root)),
            'closure_root': str(closure_root.resolve()),
            'rectified_root': str(rectified_root.resolve()),
            'closure_files': {
                label: {
                    'path': str(path.resolve()),
                    'sha256': _sha256_file(path),
                }
                for label, path in paths.items()
            },
            'schedule_file_sha256': SCHEDULE_FILE_SHA256,
            'ordered_member_list_sha256': ORDERED_MEMBER_LIST_SHA256,
            'group_index_sha256': GROUP_INDEX_SHA256,
            'schedule_binding_sha256': SCHEDULE_BINDING_SHA256,
            'member_count': EXPECTED_MEMBER_COUNT,
            'group_count': EXPECTED_GROUP_COUNT,
            'member_illuminations': EXPECTED_MEMBER_ILLUMINATIONS,
            'l3_role': EXPECTED_L3_ROLE,
            'missing_count': 0,
            'duplicate_count': 0,
            'orphan_count': 0,
            'symlink_count': 0,
            'partial_count': 0,
            'science_lock': science_lock,
            'numerical_anchor_paths': numerical_paths,
            'resource_bindings': resources,
            'nvidia_smi_invocations': 0,
            'torch_probe_invocations': 0,
            'model_loads': 0,
            'model_forwards': 0,
            'scientific_result': 'NO_SCIENTIFIC_RESULT',
        },
        'resource_revalidation_sha256',
    )
    _atomic_json(output_path, payload, _validate_resource_payload)
    return dict(payload)


def revalidate_attempt03_resources(
    *,
    worktree: Path,
    runtime_root: Path,
    rectified_root: Path,
    closure_root: Path,
    overlay_path: Path,
    output_path: Path,
) -> dict[str, object]:
    runtime_root = runtime_root.resolve()
    output_path = _require_attempt_root(runtime_root, output_path)
    _require_fresh_artifact(output_path)
    tooling_commit, tooling_tree = _tooling_revision(worktree)
    try:
        return _revalidate_attempt03_resources_pass(
            worktree=worktree,
            runtime_root=runtime_root,
            rectified_root=rectified_root,
            closure_root=closure_root,
            overlay_path=overlay_path,
            output_path=output_path,
        )
    except (
        V4ExecutionError,
        V4RectifiedClosureError,
        V4ScienceLockError,
        OSError,
        ValueError,
    ) as exc:
        payload = _signed(
            {
                'schema_version': RESOURCE_SCHEMA,
                'attempt_id': ATTEMPT_ID,
                'status': 'FAIL',
                'reason_code': (
                    'V4_RESOURCE_CLOSURE_REVALIDATION_FAILED'
                ),
                'underlying_reason_code': str(exc),
                'tooling_commit': tooling_commit,
                'tooling_tree': tooling_tree,
                'scientific_anchor_commit': SCIENTIFIC_ANCHOR_COMMIT,
                'scientific_anchor_tree': SCIENTIFIC_ANCHOR_TREE,
                'protocol_id': V4_PROTOCOL_ID,
                'protocol_sha256': V4_PROTOCOL_SHA256,
                'runtime_root': str(runtime_root),
                'attempt_root': str(_required_attempt_root(runtime_root)),
                'closure_root': str(closure_root.resolve()),
                'rectified_root': str(rectified_root.resolve()),
                'nvidia_smi_invocations': 0,
                'torch_probe_invocations': 0,
                'model_loads': 0,
                'model_forwards': 0,
                'gpu_inference_seconds': 0,
                'pass_receipt_generated': False,
                'execution_authorization_generated': False,
                'scientific_result': NO_SCIENTIFIC_RESULT,
            },
            'resource_revalidation_sha256',
        )
        _atomic_json(
            output_path,
            payload,
            _validate_resource_failure_payload,
        )
        return dict(payload)


def validate_attempt03_resources(path: Path) -> dict[str, Any]:
    path = _attempt_path(path)
    payload = _load_json(path)
    _validate_resource_payload(payload)
    runtime_root = Path(str(payload.get('runtime_root')))
    attempt_root = Path(str(payload.get('attempt_root')))
    if (
        not runtime_root.is_absolute()
        or attempt_root != _required_attempt_root(runtime_root)
        or _attempt_root(path) != attempt_root
    ):
        raise V4ExecutionError('V4_ATTEMPT03_PATH_SCOPE_MISMATCH')
    _verify_resource_bindings(payload.get('resource_bindings'))
    closure_files = payload.get('closure_files')
    if not isinstance(closure_files, Mapping):
        raise V4ExecutionError('V4_ATTEMPT03_RESOURCE_RECEIPT_TAMPER')
    for label, expected in _EXPECTED_FILE_DIGESTS.items():
        item = closure_files.get(label)
        if not isinstance(item, Mapping):
            raise V4ExecutionError(
                'V4_ATTEMPT03_RESOURCE_RECEIPT_TAMPER'
            )
        source = Path(str(item.get('path')))
        if (
            not source.is_file()
            or item.get('sha256') != expected
            or _sha256_file(source) != expected
        ):
            raise V4ExecutionError(
                'V4_ATTEMPT03_RESOURCE_RECEIPT_TAMPER'
            )
    return payload


def _required_int(value: str, reason: str) -> int:
    if value.strip().lower() in {
        '',
        'n/a',
        '[not supported]',
        'not supported',
        'unknown',
    }:
        raise V4ExecutionError(reason)
    try:
        return int(float(value.strip()))
    except ValueError as exc:
        raise V4ExecutionError(reason) from exc


def _csv_rows(text: str, width: int, reason: str) -> list[list[str]]:
    rows: list[list[str]] = []
    for raw in text.splitlines():
        if not raw.strip():
            continue
        row = [item.strip() for item in raw.split(',', width - 1)]
        if len(row) != width:
            raise V4ExecutionError(reason)
        rows.append(row)
    if not rows:
        raise V4ExecutionError(reason)
    return rows


def _cuda_runtime(runner: Callable[[Sequence[str]], str]) -> str:
    banner = runner(('nvidia-smi',))
    marker = 'CUDA Version:'
    if marker not in banner:
        raise V4ExecutionError('V4_ATTEMPT03_CUDA_RUNTIME_UNPROVEN')
    value = banner.split(marker, 1)[1].split('|', 1)[0].strip()
    if not value:
        raise V4ExecutionError('V4_ATTEMPT03_CUDA_RUNTIME_UNPROVEN')
    return value


def _process_text(pid: int, name: str) -> dict[str, str]:
    proc = Path('/proc') / str(pid)
    try:
        owner = proc.stat().st_uid
        import pwd

        owner_name = pwd.getpwuid(owner).pw_name
        cwd = str((proc / 'cwd').resolve(strict=True))
        cmdline = (
            (proc / 'cmdline')
            .read_bytes()
            .replace(b'\x00', b' ')
            .decode('utf-8', errors='strict')
            .strip()
        )
    except (OSError, KeyError, UnicodeDecodeError) as exc:
        raise V4ExecutionError(
            'V4_ATTEMPT03_GPU_PROCESS_IDENTITY_UNPROVEN'
        ) from exc
    if not owner_name or not cwd or not cmdline or not name:
        raise V4ExecutionError(
            'V4_ATTEMPT03_GPU_PROCESS_IDENTITY_UNPROVEN'
        )
    return {'owner': owner_name, 'cwd': cwd, 'cmdline': cmdline}


def nvidia_smi_attempt03_inventory(
    *,
    command_runner: Callable[[Sequence[str]], str] | None = None,
    process_resolver: Callable[[int, str], Mapping[str, str]]
    | None = None,
) -> dict[str, object]:
    runner = _run_text if command_runner is None else command_runner
    resolver = _process_text if process_resolver is None else process_resolver
    query = (
        'index,uuid,pci.bus_id,name,memory.total,memory.free,memory.used,'
        'utilization.gpu,temperature.gpu,pstate,compute_mode,'
        'mig.mode.current,ecc.errors.uncorrected.volatile.total,driver_version'
    )
    try:
        rows = _csv_rows(
            runner(
                (
                    'nvidia-smi',
                    f'--query-gpu={query}',
                    '--format=csv,noheader,nounits',
                )
            ),
            14,
            'V4_ATTEMPT03_GPU_INVENTORY_UNAVAILABLE',
        )
        cuda_runtime = _cuda_runtime(runner)
        process_text = runner(
            (
                'nvidia-smi',
                '--query-compute-apps=gpu_uuid,pid,process_name,used_memory',
                '--format=csv,noheader,nounits',
            )
        )
    except V4ExecutionError:
        raise
    except Exception as exc:
        raise V4ExecutionError(
            'V4_ATTEMPT03_GPU_INVENTORY_UNAVAILABLE'
        ) from exc
    processes: dict[str, list[dict[str, object]]] = {}
    for row in _csv_rows(
        process_text,
        4,
        'V4_ATTEMPT03_GPU_PROCESS_ENUMERATION_UNPROVEN',
    ) if process_text.strip() else []:
        gpu_uuid, pid_raw, name, memory_raw = row
        pid = _required_int(
            pid_raw, 'V4_ATTEMPT03_GPU_PROCESS_IDENTITY_UNPROVEN'
        )
        identity = dict(resolver(pid, name))
        if not all(identity.get(key) for key in ('owner', 'cwd', 'cmdline')):
            raise V4ExecutionError(
                'V4_ATTEMPT03_GPU_PROCESS_IDENTITY_UNPROVEN'
            )
        processes.setdefault(gpu_uuid, []).append(
            {
                'pid': pid,
                'process_name': name,
                'used_memory_bytes': _required_int(
                    memory_raw,
                    'V4_ATTEMPT03_GPU_PROCESS_MEMORY_UNPROVEN',
                )
                * 1024
                * 1024,
                **identity,
            }
        )
    devices: list[dict[str, object]] = []
    for row in rows:
        ecc = _required_int(
            row[12], 'V4_ATTEMPT03_GPU_ECC_HEALTH_UNPROVEN'
        )
        gpu_uuid = row[1]
        devices.append(
            {
                'index': _required_int(
                    row[0], 'V4_ATTEMPT03_GPU_INDEX_UNPROVEN'
                ),
                'uuid': gpu_uuid,
                'pci_bus_id': row[2],
                'model': row[3],
                'total_memory_bytes': _required_int(
                    row[4], 'V4_ATTEMPT03_GPU_MEMORY_UNPROVEN'
                )
                * 1024
                * 1024,
                'free_memory_bytes': _required_int(
                    row[5], 'V4_ATTEMPT03_GPU_MEMORY_UNPROVEN'
                )
                * 1024
                * 1024,
                'used_memory_bytes': _required_int(
                    row[6], 'V4_ATTEMPT03_GPU_MEMORY_UNPROVEN'
                )
                * 1024
                * 1024,
                'utilization_gpu_percent': _required_int(
                    row[7], 'V4_ATTEMPT03_GPU_UTILIZATION_UNPROVEN'
                ),
                'temperature_c': _required_int(
                    row[8], 'V4_ATTEMPT03_GPU_TEMPERATURE_UNPROVEN'
                ),
                'performance_state': row[9],
                'compute_mode': row[10],
                'mig_mode': row[11],
                'ecc_uncorrected_volatile_total': ecc,
                'ecc_health': 'OK' if ecc == 0 else 'ERROR',
                'driver_version': row[13],
                'cuda_runtime': cuda_runtime,
                'compute_processes': sorted(
                    processes.get(gpu_uuid, []),
                    key=lambda value: int(value['pid']),
                ),
            }
        )
    if set(processes) - {str(device['uuid']) for device in devices}:
        raise V4ExecutionError(
            'V4_ATTEMPT03_GPU_PROCESS_DEVICE_UNPROVEN'
        )
    return {
        'schema_version': INVENTORY_SCHEMA,
        'attempt_id': ATTEMPT_ID,
        'hostname': platform.node(),
        'timestamp_utc': time.strftime(
            '%Y-%m-%dT%H:%M:%SZ', time.gmtime()
        ),
        'driver_version': rows[0][13],
        'cuda_runtime': cuda_runtime,
        'devices': devices,
    }


def _device_map(
    sample: Mapping[str, object],
) -> dict[str, Mapping[str, object]]:
    if (
        sample.get('schema_version') != INVENTORY_SCHEMA
        or sample.get('attempt_id') != ATTEMPT_ID
        or not sample.get('hostname')
        or not sample.get('timestamp_utc')
        or not isinstance(sample.get('driver_version'), str)
        or not sample.get('driver_version')
        or not isinstance(sample.get('cuda_runtime'), str)
        or not sample.get('cuda_runtime')
    ):
        raise V4ExecutionError(
            'V4_ATTEMPT03_GPU_INVENTORY_SCHEMA_REQUIRED'
        )
    devices = sample.get('devices')
    if not isinstance(devices, list) or not devices:
        raise V4ExecutionError(
            'V4_ATTEMPT03_GPU_INVENTORY_SCHEMA_REQUIRED'
        )
    result: dict[str, Mapping[str, object]] = {}
    indices: set[int] = set()
    pci_ids: set[str] = set()
    for device in devices:
        if not isinstance(device, Mapping):
            raise V4ExecutionError(
                'V4_ATTEMPT03_GPU_INVENTORY_SCHEMA_REQUIRED'
            )
        uuid = device.get('uuid')
        index = device.get('index')
        pci = device.get('pci_bus_id')
        if (
            not isinstance(uuid, str)
            or not uuid.startswith('GPU-')
            or type(index) is not int
            or index < 0
            or not isinstance(pci, str)
            or not pci
            or not isinstance(device.get('model'), str)
            or not device.get('model')
            or uuid in result
            or index in indices
            or pci in pci_ids
        ):
            raise V4ExecutionError(
                'V4_ATTEMPT03_GPU_IDENTITY_UNPROVEN'
            )
        for key in (
            'total_memory_bytes',
            'free_memory_bytes',
            'used_memory_bytes',
            'utilization_gpu_percent',
            'temperature_c',
            'ecc_uncorrected_volatile_total',
        ):
            if type(device.get(key)) is not int or int(device[key]) < 0:
                raise V4ExecutionError(
                    'V4_ATTEMPT03_GPU_INVENTORY_SCHEMA_REQUIRED'
                )
        if (
            device.get('driver_version') != sample.get('driver_version')
            or device.get('cuda_runtime') != sample.get('cuda_runtime')
            or not device.get('performance_state')
            or not device.get('compute_mode')
            or not device.get('mig_mode')
            or not isinstance(device.get('compute_processes'), list)
        ):
            raise V4ExecutionError(
                'V4_ATTEMPT03_GPU_INVENTORY_SCHEMA_REQUIRED'
            )
        processes = device.get('compute_processes')
        seen_pids: set[int] = set()
        for process in processes:
            if not isinstance(process, Mapping):
                raise V4ExecutionError(
                    'V4_ATTEMPT03_GPU_PROCESS_IDENTITY_UNPROVEN'
                )
            pid = process.get('pid')
            if (
                type(pid) is not int
                or pid <= 0
                or pid in seen_pids
                or not all(
                    isinstance(process.get(key), str)
                    and bool(process.get(key))
                    for key in (
                        'owner',
                        'cwd',
                        'cmdline',
                        'process_name',
                    )
                )
                or type(process.get('used_memory_bytes')) is not int
                or process.get('used_memory_bytes', -1) < 0
            ):
                raise V4ExecutionError(
                    'V4_ATTEMPT03_GPU_PROCESS_IDENTITY_UNPROVEN'
                )
            seen_pids.add(pid)
        result[uuid] = device
        indices.add(index)
        pci_ids.add(pci)
    return result


def _sample_timestamp(value: object) -> float:
    if not isinstance(value, str) or not value.endswith('Z'):
        raise V4ExecutionError(
            'V4_ATTEMPT03_GPU_SAMPLE_INDEPENDENCE_UNPROVEN'
        )
    try:
        return datetime.fromisoformat(value[:-1] + '+00:00').timestamp()
    except ValueError as exc:
        raise V4ExecutionError(
            'V4_ATTEMPT03_GPU_SAMPLE_INDEPENDENCE_UNPROVEN'
        ) from exc


def _candidate_reason(
    devices: Sequence[Mapping[str, object] | None],
) -> str | None:
    if len(devices) != SAMPLE_COUNT or any(item is None for item in devices):
        return 'V4_ATTEMPT03_GPU_MAPPING_DRIFT'
    rows = [item for item in devices if item is not None]
    identity_keys = (
        'uuid',
        'index',
        'pci_bus_id',
        'model',
        'driver_version',
        'cuda_runtime',
        'total_memory_bytes',
        'free_memory_bytes',
        'used_memory_bytes',
    )
    if any(
        row.get(key) != rows[0].get(key)
        for row in rows[1:]
        for key in identity_keys
    ):
        return 'V4_ATTEMPT03_GPU_MAPPING_DRIFT'
    for row in rows:
        processes = row.get('compute_processes')
        if not isinstance(processes, list):
            return 'V4_ATTEMPT03_GPU_PROCESS_ENUMERATION_UNPROVEN'
        if processes:
            return 'V4_ATTEMPT03_GPU_COMPUTE_PROCESS_PRESENT'
        if row.get('model') != AUTHORIZED_GPU_MODEL:
            return 'V4_ATTEMPT03_GPU_MODEL_NOT_AUTHORIZED'
        if str(row.get('mig_mode')).strip().lower() not in {
            'disabled',
            'off',
            '0',
        }:
            return 'V4_ATTEMPT03_GPU_MIG_ENABLED'
        if str(row.get('compute_mode')).strip().lower() != 'default':
            return 'V4_ATTEMPT03_GPU_COMPUTE_MODE_NOT_DEFAULT'
        if row.get('utilization_gpu_percent') != 0:
            return 'V4_ATTEMPT03_GPU_UNEXPLAINED_ACTIVITY'
        if int(row.get('free_memory_bytes', -1)) < MIN_FREE_MEMORY_BYTES:
            return 'V4_ATTEMPT03_GPU_FREE_MEMORY_INSUFFICIENT'
        if (
            row.get('ecc_health') != 'OK'
            or row.get('ecc_uncorrected_volatile_total') != 0
        ):
            return 'V4_ATTEMPT03_GPU_ECC_HEALTH_ERROR'
    return None


def select_attempt03_gpu(
    samples: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    if len(samples) != SAMPLE_COUNT:
        raise V4ExecutionError(
            'V4_ATTEMPT03_THREE_SAMPLES_REQUIRED'
        )
    maps = [_device_map(sample) for sample in samples]
    if len({str(sample.get('hostname')) for sample in samples}) != 1:
        raise V4ExecutionError('V4_ATTEMPT03_GPU_HOST_DRIFT')
    timestamps = [str(sample.get('timestamp_utc')) for sample in samples]
    if len(set(timestamps)) != SAMPLE_COUNT:
        raise V4ExecutionError(
            'V4_ATTEMPT03_GPU_SAMPLE_INDEPENDENCE_UNPROVEN'
        )
    sampled_at = [_sample_timestamp(value) for value in timestamps]
    if any(
        later - earlier < SAMPLE_INTERVAL_SECONDS
        for earlier, later in zip(sampled_at, sampled_at[1:])
    ):
        raise V4ExecutionError(
            'V4_ATTEMPT03_GPU_SAMPLE_INDEPENDENCE_UNPROVEN'
        )
    uuids = sorted(set().union(*(set(mapping) for mapping in maps)))
    evaluations: list[dict[str, object]] = []
    eligible: list[Mapping[str, object]] = []
    for uuid in uuids:
        rows = [mapping.get(uuid) for mapping in maps]
        reason = _candidate_reason(rows)
        evaluations.append(
            {
                'uuid': uuid,
                'eligible': reason is None,
                'reason_code': (
                    reason or 'V4_ATTEMPT03_GPU_CANDIDATE_ELIGIBLE'
                ),
                'indices': [
                    None if row is None else row.get('index') for row in rows
                ],
                'pci_bus_ids': [
                    None if row is None else row.get('pci_bus_id')
                    for row in rows
                ],
            }
        )
        if reason is None and rows[-1] is not None:
            eligible.append(rows[-1])
    eligible.sort(
        key=lambda row: (
            -int(row['total_memory_bytes']),
            -int(row['free_memory_bytes']),
            str(row['uuid']),
        )
    )
    if not eligible:
        return {
            'status': 'FAIL',
            'reason_code': 'V4_NO_ELIGIBLE_IDLE_GPU',
            'candidate_evaluations': evaluations,
            'selected_gpu': None,
        }
    return {
        'status': 'PASS',
        'reason_code': 'V4_ATTEMPT03_GPU_SELECTED',
        'candidate_evaluations': evaluations,
        'selected_gpu': dict(eligible[0]),
    }


def _validate_preflight_payload(payload: Mapping[str, Any]) -> None:
    if payload.get('schema_version') != PREFLIGHT_SCHEMA:
        raise V4ExecutionError('V4_ATTEMPT03_PREFLIGHT_SCHEMA_REQUIRED')
    if payload.get('attempt_id') != ATTEMPT_ID:
        raise V4ExecutionError('V4_ATTEMPT03_ID_MISMATCH')
    if (
        not _is_sha(payload.get('run_id'), 32)
        or not _is_sha(payload.get('tooling_commit'), 40)
        or not _is_sha(payload.get('tooling_tree'), 40)
        or payload.get('protocol_id') != V4_PROTOCOL_ID
        or payload.get('protocol_sha256') != V4_PROTOCOL_SHA256
        or not isinstance(payload.get('resource_receipt_path'), str)
        or not _is_sha(payload.get('resource_receipt_sha256'))
    ):
        raise V4ExecutionError('V4_ATTEMPT03_PREFLIGHT_TAMPER')
    if payload.get('scientific_anchor_commit') != SCIENTIFIC_ANCHOR_COMMIT:
        raise V4ExecutionError('V4_ATTEMPT03_SCIENTIFIC_ANCHOR_MISMATCH')
    if payload.get('scientific_anchor_tree') != SCIENTIFIC_ANCHOR_TREE:
        raise V4ExecutionError('V4_ATTEMPT03_SCIENTIFIC_ANCHOR_MISMATCH')
    if payload.get('sample_count') != SAMPLE_COUNT:
        raise V4ExecutionError('V4_ATTEMPT03_THREE_SAMPLES_REQUIRED')
    if payload.get('sample_interval_seconds') != SAMPLE_INTERVAL_SECONDS:
        raise V4ExecutionError(
            'V4_ATTEMPT03_SAMPLE_INTERVAL_MUST_BE_5_SECONDS'
        )
    samples = payload.get('inventory_samples')
    if not isinstance(samples, list) or len(samples) != SAMPLE_COUNT:
        raise V4ExecutionError('V4_ATTEMPT03_THREE_SAMPLES_REQUIRED')
    selection = select_attempt03_gpu(samples)
    if selection.get('selected_gpu') != payload.get('selected_gpu'):
        raise V4ExecutionError('V4_ATTEMPT03_GPU_SELECTION_TAMPER')
    if selection.get('candidate_evaluations') != payload.get(
        'candidate_evaluations'
    ):
        raise V4ExecutionError('V4_ATTEMPT03_GPU_SELECTION_TAMPER')
    status = payload.get('status')
    if status == 'PASS':
        if (
            selection.get('status') != 'PASS'
            or payload.get('reason_code') != 'V4_ATTEMPT03_PREFLIGHT_PASS'
        ):
            raise V4ExecutionError('V4_ATTEMPT03_GPU_SELECTION_TAMPER')
    elif status == 'FAIL':
        if (
            selection.get('status') != 'FAIL'
            or payload.get('reason_code') != 'V4_NO_ELIGIBLE_IDLE_GPU'
            or payload.get('selected_gpu') is not None
        ):
            raise V4ExecutionError('V4_ATTEMPT03_GPU_SELECTION_TAMPER')
    else:
        raise V4ExecutionError('V4_ATTEMPT03_PREFLIGHT_SCHEMA_REQUIRED')
    if (
        payload.get('inventory_sample_count') != SAMPLE_COUNT
        or payload.get('nvidia_smi_invocations_per_sample')
        != NVIDIA_SMI_CALLS_PER_SAMPLE
        or payload.get('nvidia_smi_invocations')
        != SAMPLE_COUNT * NVIDIA_SMI_CALLS_PER_SAMPLE
        or
        payload.get('torch_probe_invocations') != 0
        or payload.get('model_loads') != 0
        or payload.get('model_forwards') != 0
        or payload.get('gpu_inference_seconds') != 0
        or payload.get('scientific_result') != NO_SCIENTIFIC_RESULT
    ):
        raise V4ExecutionError('V4_ATTEMPT03_GPU_PROBE_SCOPE_VIOLATION')
    _validate_signature(
        payload,
        'hardware_snapshot_sha256',
        'V4_ATTEMPT03_PREFLIGHT_TAMPER',
    )


def create_attempt03_gpu_preflight(
    *,
    worktree: Path,
    resource_receipt_path: Path,
    output_path: Path,
    inventory_sampler: Callable[[], Mapping[str, object]] | None = None,
    sleeper: Callable[[float], None] = time.sleep,
    sample_interval_seconds: float = SAMPLE_INTERVAL_SECONDS,
    tooling_commit: str | None = None,
    tooling_tree: str | None = None,
) -> dict[str, object]:
    output_path = _attempt_path(output_path)
    _require_fresh_artifact(output_path)
    output_root = _attempt_root(output_path)
    if _attempt_root(resource_receipt_path) != output_root:
        raise V4ExecutionError('V4_ATTEMPT03_PATH_SCOPE_MISMATCH')
    if sample_interval_seconds != SAMPLE_INTERVAL_SECONDS:
        raise V4ExecutionError(
            'V4_ATTEMPT03_SAMPLE_INTERVAL_MUST_BE_5_SECONDS'
        )
    resources = validate_attempt03_resources(resource_receipt_path)
    runtime_root = Path(str(resources.get('runtime_root')))
    if (
        output_root != _required_attempt_root(runtime_root)
        or output_root != Path(str(resources.get('attempt_root')))
    ):
        raise V4ExecutionError('V4_ATTEMPT03_PATH_SCOPE_MISMATCH')
    if tooling_commit is None or tooling_tree is None:
        tooling_commit, tooling_tree = _tooling_revision(worktree)
    if (
        resources.get('tooling_commit') != tooling_commit
        or resources.get('tooling_tree') != tooling_tree
    ):
        raise V4ExecutionError('V4_ATTEMPT03_RESOURCE_TOOLING_MISMATCH')
    sampler = (
        nvidia_smi_attempt03_inventory
        if inventory_sampler is None
        else inventory_sampler
    )
    samples: list[dict[str, object]] = []
    for index in range(SAMPLE_COUNT):
        samples.append(dict(sampler()))
        if index + 1 < SAMPLE_COUNT:
            sleeper(SAMPLE_INTERVAL_SECONDS)
    selection = select_attempt03_gpu(samples)
    status = str(selection['status'])
    reason = (
        'V4_ATTEMPT03_PREFLIGHT_PASS'
        if status == 'PASS'
        else 'V4_NO_ELIGIBLE_IDLE_GPU'
    )
    payload = _signed(
        {
            'schema_version': PREFLIGHT_SCHEMA,
            'attempt_id': ATTEMPT_ID,
            'run_id': uuid4().hex,
            'status': status,
            'reason_code': reason,
            'tooling_commit': tooling_commit,
            'tooling_tree': tooling_tree,
            'scientific_anchor_commit': SCIENTIFIC_ANCHOR_COMMIT,
            'scientific_anchor_tree': SCIENTIFIC_ANCHOR_TREE,
            'protocol_id': V4_PROTOCOL_ID,
            'protocol_sha256': V4_PROTOCOL_SHA256,
            'resource_receipt_path': str(
                resource_receipt_path.resolve()
            ),
            'resource_receipt_sha256': _sha256_file(
                resource_receipt_path
            ),
            'sample_count': SAMPLE_COUNT,
            'sample_interval_seconds': SAMPLE_INTERVAL_SECONDS,
            'inventory_samples': samples,
            'candidate_evaluations': selection[
                'candidate_evaluations'
            ],
            'selected_gpu': selection['selected_gpu'],
            'inventory_sample_count': SAMPLE_COUNT,
            'nvidia_smi_invocations_per_sample': (
                NVIDIA_SMI_CALLS_PER_SAMPLE
            ),
            'nvidia_smi_invocations': (
                SAMPLE_COUNT * NVIDIA_SMI_CALLS_PER_SAMPLE
            ),
            'torch_probe_invocations': 0,
            'model_loads': 0,
            'model_forwards': 0,
            'gpu_inference_seconds': 0,
            'scientific_result': 'NO_SCIENTIFIC_RESULT',
        },
        'hardware_snapshot_sha256',
    )
    _atomic_json(output_path, payload, _validate_preflight_payload)
    return dict(payload)


def validate_attempt03_gpu_preflight(path: Path) -> dict[str, Any]:
    path = _attempt_path(path)
    payload = _load_json(path)
    _validate_preflight_payload(payload)
    resource_path = Path(str(payload.get('resource_receipt_path')))
    if (
        not resource_path.is_file()
        or _sha256_file(resource_path)
        != payload.get('resource_receipt_sha256')
    ):
        raise V4ExecutionError('V4_ATTEMPT03_RESOURCE_RECEIPT_TAMPER')
    resources = validate_attempt03_resources(resource_path)
    runtime_root = Path(str(resources.get('runtime_root')))
    if (
        _attempt_root(path) != _required_attempt_root(runtime_root)
        or _attempt_root(resource_path) != _attempt_root(path)
        or payload.get('tooling_commit') != resources.get('tooling_commit')
        or payload.get('tooling_tree') != resources.get('tooling_tree')
        or payload.get('protocol_id') != resources.get('protocol_id')
        or payload.get('protocol_sha256')
        != resources.get('protocol_sha256')
    ):
        raise V4ExecutionError('V4_ATTEMPT03_PREFLIGHT_TAMPER')
    return payload


def _authorized_scope() -> dict[str, object]:
    return {
        'models': list(SCIENTIFIC_MODELS),
        'model_count': 2,
        'dataset': 'DTU',
        'paired_lighting_states': [
            'L1',
            'L2',
            'L3',
            'L4',
            'L5',
            'L6',
            'L7',
        ],
        'rectified_member_count': 960,
        'synthetic_corruption': 'Koschmieder fog',
        'synthetic_severity_axis': 'beta-only',
        'scientific_unit_count': 400,
        'primary_endpoint': 'Pose',
        'supporting_evidence': ['Fusion', 'F-score'],
        'single_gpu': True,
        'models_sequential': True,
        'units_sequential': True,
        'fallback_allowed': False,
        'device_switch_allowed': False,
        'forbidden': [
            'UAVLight',
            'v4.1',
            'third model',
            'second corruption family',
            'seed expansion',
            'severity expansion',
            'scene expansion',
            'view expansion',
            'model matrix expansion',
        ],
    }


def _validate_receipt_payload(payload: Mapping[str, Any]) -> None:
    if payload.get('schema_version') != RECEIPT_SCHEMA:
        raise V4ExecutionError('V4_ATTEMPT03_RECEIPT_SCHEMA_REQUIRED')
    if (
        payload.get('attempt_id') != ATTEMPT_ID
        or payload.get('status') != 'PASS'
        or payload.get('authorization_kind') != 'V4_MVE_EXECUTION'
    ):
        raise V4ExecutionError('V4_ATTEMPT03_RECEIPT_PASS_REQUIRED')
    if (
        not _is_sha(payload.get('run_id'), 32)
        or not _is_sha(payload.get('tooling_commit'), 40)
        or not _is_sha(payload.get('tooling_tree'), 40)
        or payload.get('protocol_id') != V4_PROTOCOL_ID
        or payload.get('protocol_sha256') != V4_PROTOCOL_SHA256
        or not isinstance(payload.get('hardware_snapshot_path'), str)
        or not _is_sha(payload.get('hardware_snapshot_sha256'))
        or not isinstance(payload.get('resource_receipt_path'), str)
        or not _is_sha(payload.get('resource_receipt_sha256'))
    ):
        raise V4ExecutionError('V4_ATTEMPT03_RECEIPT_TAMPER')
    if payload.get('scientific_anchor_commit') != SCIENTIFIC_ANCHOR_COMMIT:
        raise V4ExecutionError('V4_ATTEMPT03_SCIENTIFIC_ANCHOR_MISMATCH')
    if payload.get('scientific_anchor_tree') != SCIENTIFIC_ANCHOR_TREE:
        raise V4ExecutionError('V4_ATTEMPT03_SCIENTIFIC_ANCHOR_MISMATCH')
    if payload.get('schedule_file_sha256') != SCHEDULE_FILE_SHA256:
        raise V4ExecutionError('V4_ATTEMPT03_SCHEDULE_TAMPER')
    if (
        payload.get('rectified_manifest_sha256')
        != MANIFEST_FILE_SHA256
        or payload.get('expected_set_sha256')
        != EXPECTED_SET_FILE_SHA256
        or payload.get('materialize_receipt_sha256')
        != MATERIALIZE_RECEIPT_FILE_SHA256
        or payload.get('closure_receipt_sha256')
        != CLOSURE_RECEIPT_FILE_SHA256
        or payload.get('ordered_member_list_sha256')
        != ORDERED_MEMBER_LIST_SHA256
        or payload.get('group_index_sha256') != GROUP_INDEX_SHA256
        or payload.get('schedule_binding_sha256')
        != SCHEDULE_BINDING_SHA256
    ):
        raise V4ExecutionError('V4_ATTEMPT03_RESOURCE_RECEIPT_TAMPER')
    if (
        payload.get('max_concurrent_gpus') != 1
        or payload.get('models_sequential') is not True
        or payload.get('units_sequential') is not True
        or payload.get('fallback_allowed') is not False
        or payload.get('device_switch_allowed') is not False
    ):
        raise V4ExecutionError('V4_ATTEMPT03_DEVICE_SCOPE_MISMATCH')
    if (
        not str(payload.get('selected_gpu_uuid', '')).startswith('GPU-')
        or not payload.get('selected_gpu_pci_bus_id')
        or type(payload.get('selected_gpu_index')) is not int
        or payload.get('selected_gpu_model') != AUTHORIZED_GPU_MODEL
        or type(payload.get('selected_total_memory_bytes')) is not int
        or payload.get('selected_total_memory_bytes', 0) <= 0
        or type(payload.get('selected_free_memory_bytes')) is not int
        or payload.get('selected_free_memory_bytes', 0)
        < MIN_FREE_MEMORY_BYTES
    ):
        raise V4ExecutionError('V4_ATTEMPT03_GPU_IDENTITY_UNPROVEN')
    if (
        payload.get('torch_probe_invocations') != 0
        or payload.get('model_loads') != 0
        or payload.get('model_forwards') != 0
        or payload.get('gpu_inference_seconds') != 0
        or payload.get('scientific_result') != NO_SCIENTIFIC_RESULT
    ):
        raise V4ExecutionError('V4_ATTEMPT03_RECEIPT_SCOPE_MISMATCH')
    _validate_signature(
        payload,
        'receipt_payload_sha256',
        'V4_ATTEMPT03_RECEIPT_TAMPER',
    )


def _validate_authorization_payload(payload: Mapping[str, Any]) -> None:
    if payload.get('schema_version') != AUTHORIZATION_SCHEMA:
        raise V4ExecutionError(
            'V4_ATTEMPT03_AUTHORIZATION_SCHEMA_REQUIRED'
        )
    if (
        payload.get('attempt_id') != ATTEMPT_ID
        or payload.get('status') != 'V4_MVE_EXECUTION_AUTHORIZED'
        or payload.get('scientific_result') != 'NO_SCIENTIFIC_RESULT'
    ):
        raise V4ExecutionError(
            'V4_ATTEMPT03_AUTHORIZATION_SCOPE_MISMATCH'
        )
    if (
        not _is_sha(payload.get('run_id'), 32)
        or not _is_sha(payload.get('tooling_commit'), 40)
        or not _is_sha(payload.get('tooling_tree'), 40)
        or not isinstance(payload.get('gpu_receipt_path'), str)
        or not _is_sha(payload.get('gpu_receipt_sha256'))
        or not isinstance(payload.get('resource_receipt_path'), str)
        or not _is_sha(payload.get('resource_receipt_sha256'))
        or not isinstance(payload.get('hardware_snapshot_path'), str)
        or not _is_sha(payload.get('hardware_snapshot_sha256'))
    ):
        raise V4ExecutionError('V4_ATTEMPT03_AUTHORIZATION_TAMPER')
    if payload.get('authorized_scope') != _authorized_scope():
        raise V4ExecutionError(
            'V4_ATTEMPT03_AUTHORIZATION_SCOPE_MISMATCH'
        )
    if (
        payload.get('scientific_anchor_commit')
        != SCIENTIFIC_ANCHOR_COMMIT
        or payload.get('scientific_anchor_tree')
        != SCIENTIFIC_ANCHOR_TREE
        or payload.get('protocol_id') != V4_PROTOCOL_ID
        or payload.get('protocol_sha256') != V4_PROTOCOL_SHA256
        or payload.get('schedule_file_sha256')
        != SCHEDULE_FILE_SHA256
        or payload.get('rectified_manifest_sha256')
        != MANIFEST_FILE_SHA256
    ):
        raise V4ExecutionError(
            'V4_ATTEMPT03_AUTHORIZATION_SCOPE_MISMATCH'
        )
    if payload.get('closure_digests') != _EXPECTED_CLOSURE_DIGESTS:
        raise V4ExecutionError(
            'V4_ATTEMPT03_AUTHORIZATION_SCOPE_MISMATCH'
        )
    budget = payload.get('budget')
    expected_budget = {
        'authorization_gpu_seconds': GPU_TARGET_SECONDS,
        'authorization_storage_bytes': BYTE_TARGET,
        'catastrophe_gpu_seconds': GPU_CATASTROPHE_SECONDS,
        'catastrophe_storage_bytes': BYTE_CATASTROPHE,
    }
    if budget != expected_budget:
        raise V4ExecutionError('V4_ATTEMPT03_BUDGET_SCOPE_MISMATCH')
    if payload.get('finalizer') != AUTHORIZED_FINALIZER:
        raise V4ExecutionError(
            'V4_ATTEMPT03_FINALIZER_SCOPE_MISMATCH'
        )
    runtime_root = payload.get('runtime_root')
    attempt_root = payload.get('attempt_root')
    runtime_paths = payload.get('runtime_paths')
    if (
        not isinstance(runtime_root, str)
        or not Path(runtime_root).is_absolute()
        or not isinstance(attempt_root, str)
        or Path(attempt_root) != _required_attempt_root(Path(runtime_root))
        or not isinstance(runtime_paths, Mapping)
        or set(runtime_paths)
        != {
            'run_root',
            'artifact_root',
            'gpu_ledger_path',
            'final_evidence_path',
        }
    ):
        raise V4ExecutionError('V4_ATTEMPT03_RUNTIME_PATH_ESCAPE')
    for path_value in runtime_paths.values():
        if not isinstance(path_value, str):
            raise V4ExecutionError('V4_ATTEMPT03_RUNTIME_PATH_ESCAPE')
        _under_root(Path(runtime_root), Path(path_value))
    selected = payload.get('selected_gpu')
    if (
        not isinstance(selected, Mapping)
        or not str(selected.get('uuid', '')).startswith('GPU-')
        or not selected.get('pci_bus_id')
        or type(selected.get('index')) is not int
        or selected.get('model') != AUTHORIZED_GPU_MODEL
    ):
        raise V4ExecutionError('V4_ATTEMPT03_GPU_IDENTITY_UNPROVEN')
    if (
        payload.get('execution_lock_created') is not False
        or payload.get('gpu_ledger_created') is not False
        or payload.get('dispatcher_called') is not False
        or payload.get('torch_probe_invocations') != 0
        or payload.get('model_loads') != 0
        or payload.get('model_forwards') != 0
        or payload.get('gpu_inference_seconds') != 0
    ):
        raise V4ExecutionError(
            'V4_ATTEMPT03_AUTHORIZATION_SCOPE_MISMATCH'
        )
    _validate_signature(
        payload,
        'authorization_payload_sha256',
        'V4_ATTEMPT03_AUTHORIZATION_TAMPER',
    )


def _under_root(root: Path, path: Path) -> Path:
    resolved_root = root.resolve()
    resolved = path.resolve()
    try:
        resolved.relative_to(resolved_root)
    except ValueError as exc:
        raise V4ExecutionError('V4_ATTEMPT03_RUNTIME_PATH_ESCAPE') from exc
    return resolved


def _atomic_pair(
    *,
    first_path: Path,
    first_payload: Mapping[str, Any],
    first_validator: Callable[[Mapping[str, Any]], None],
    second_path: Path,
    second_payload: Mapping[str, Any],
    second_validator: Callable[[Mapping[str, Any]], None],
) -> None:
    for path in (first_path, second_path):
        if path.exists() or path.with_name(path.name + '.partial').exists():
            raise V4ExecutionError('V4_ATTEMPT03_ARTIFACT_COLLISION')
        path.parent.mkdir(parents=True, exist_ok=True)
    first_partial = first_path.with_name(first_path.name + '.partial')
    second_partial = second_path.with_name(second_path.name + '.partial')
    replaced: list[Path] = []
    try:
        first_partial.write_bytes(_json_bytes(first_payload))
        second_partial.write_bytes(_json_bytes(second_payload))
        first_validator(_load_json(first_partial))
        second_validator(_load_json(second_partial))
        first_partial.replace(first_path)
        replaced.append(first_path)
        second_partial.replace(second_path)
        replaced.append(second_path)
    except Exception:
        for path in replaced:
            try:
                path.unlink()
            except FileNotFoundError:
                pass
        raise
    finally:
        for path in (first_partial, second_partial):
            try:
                path.unlink()
            except FileNotFoundError:
                pass


def create_attempt03_execution_authorization(
    *,
    worktree: Path,
    runtime_root: Path,
    resource_receipt_path: Path,
    hardware_snapshot_path: Path,
    receipt_path: Path,
    authorization_path: Path,
    run_root: Path,
    artifact_root: Path,
    gpu_ledger_path: Path,
    final_evidence_path: Path,
    tooling_commit: str | None = None,
    tooling_tree: str | None = None,
) -> dict[str, object]:
    runtime_root = runtime_root.resolve()
    receipt_path = _require_attempt_root(runtime_root, receipt_path)
    authorization_path = _require_attempt_root(
        runtime_root, authorization_path
    )
    resource_receipt_path = _require_attempt_root(
        runtime_root, resource_receipt_path
    )
    hardware_snapshot_path = _require_attempt_root(
        runtime_root, hardware_snapshot_path
    )
    attempt_root = _attempt_root(receipt_path)
    if _attempt_root(authorization_path) != attempt_root:
        raise V4ExecutionError('V4_ATTEMPT03_PATH_SCOPE_MISMATCH')
    if (
        _attempt_root(resource_receipt_path) != attempt_root
        or _attempt_root(hardware_snapshot_path) != attempt_root
    ):
        raise V4ExecutionError('V4_ATTEMPT03_PATH_SCOPE_MISMATCH')
    resources = validate_attempt03_resources(resource_receipt_path)
    snapshot = validate_attempt03_gpu_preflight(hardware_snapshot_path)
    if snapshot.get('status') != 'PASS':
        raise V4ExecutionError('V4_ATTEMPT03_PREFLIGHT_PASS_REQUIRED')
    if tooling_commit is None or tooling_tree is None:
        tooling_commit, tooling_tree = _tooling_revision(worktree)
    for evidence in (resources, snapshot):
        if (
            evidence.get('tooling_commit') != tooling_commit
            or evidence.get('tooling_tree') != tooling_tree
        ):
            raise V4ExecutionError(
                'V4_ATTEMPT03_AUTHORIZATION_TOOLING_MISMATCH'
            )
    selected = snapshot.get('selected_gpu')
    if not isinstance(selected, Mapping):
        raise V4ExecutionError('V4_ATTEMPT03_PREFLIGHT_PASS_REQUIRED')
    resolved_paths = {
        'run_root': str(_under_root(runtime_root, run_root)),
        'artifact_root': str(_under_root(runtime_root, artifact_root)),
        'gpu_ledger_path': str(
            _under_root(runtime_root, gpu_ledger_path)
        ),
        'final_evidence_path': str(
            _under_root(runtime_root, final_evidence_path)
        ),
    }
    if any(Path(value).exists() for value in resolved_paths.values()):
        raise V4ExecutionError('V4_ATTEMPT03_RUNTIME_ARTIFACT_COLLISION')
    receipt = _signed(
        {
            'schema_version': RECEIPT_SCHEMA,
            'attempt_id': ATTEMPT_ID,
            'run_id': snapshot['run_id'],
            'status': 'PASS',
            'authorization_kind': 'V4_MVE_EXECUTION',
            'tooling_commit': tooling_commit,
            'tooling_tree': tooling_tree,
            'scientific_anchor_commit': SCIENTIFIC_ANCHOR_COMMIT,
            'scientific_anchor_tree': SCIENTIFIC_ANCHOR_TREE,
            'protocol_id': V4_PROTOCOL_ID,
            'protocol_sha256': V4_PROTOCOL_SHA256,
            'hardware_snapshot_path': str(
                hardware_snapshot_path.resolve()
            ),
            'hardware_snapshot_sha256': _sha256_file(
                hardware_snapshot_path
            ),
            'resource_receipt_path': str(
                resource_receipt_path.resolve()
            ),
            'resource_receipt_sha256': _sha256_file(
                resource_receipt_path
            ),
            'selected_gpu_uuid': selected['uuid'],
            'selected_gpu_pci_bus_id': selected['pci_bus_id'],
            'selected_gpu_index': selected['index'],
            'selected_gpu_model': selected['model'],
            'selected_total_memory_bytes': selected[
                'total_memory_bytes'
            ],
            'selected_free_memory_bytes': selected['free_memory_bytes'],
            'schedule_file_sha256': SCHEDULE_FILE_SHA256,
            'rectified_manifest_sha256': MANIFEST_FILE_SHA256,
            'expected_set_sha256': EXPECTED_SET_FILE_SHA256,
            'materialize_receipt_sha256': (
                MATERIALIZE_RECEIPT_FILE_SHA256
            ),
            'closure_receipt_sha256': CLOSURE_RECEIPT_FILE_SHA256,
            'ordered_member_list_sha256': ORDERED_MEMBER_LIST_SHA256,
            'group_index_sha256': GROUP_INDEX_SHA256,
            'schedule_binding_sha256': SCHEDULE_BINDING_SHA256,
            'max_concurrent_gpus': 1,
            'models_sequential': True,
            'units_sequential': True,
            'fallback_allowed': False,
            'device_switch_allowed': False,
            'torch_probe_invocations': 0,
            'model_loads': 0,
            'model_forwards': 0,
            'gpu_inference_seconds': 0,
            'scientific_result': 'NO_SCIENTIFIC_RESULT',
        },
        'receipt_payload_sha256',
    )
    authorization = _signed(
        {
            'schema_version': AUTHORIZATION_SCHEMA,
            'attempt_id': ATTEMPT_ID,
            'run_id': snapshot['run_id'],
            'status': 'V4_MVE_EXECUTION_AUTHORIZED',
            'tooling_commit': tooling_commit,
            'tooling_tree': tooling_tree,
            'scientific_anchor_commit': SCIENTIFIC_ANCHOR_COMMIT,
            'scientific_anchor_tree': SCIENTIFIC_ANCHOR_TREE,
            'protocol_id': V4_PROTOCOL_ID,
            'protocol_sha256': V4_PROTOCOL_SHA256,
            'gpu_receipt_path': str(receipt_path.resolve()),
            'gpu_receipt_sha256': _sha_json(receipt),
            'hardware_snapshot_path': str(
                hardware_snapshot_path.resolve()
            ),
            'hardware_snapshot_sha256': _sha256_file(
                hardware_snapshot_path
            ),
            'resource_receipt_path': str(
                resource_receipt_path.resolve()
            ),
            'resource_receipt_sha256': _sha256_file(
                resource_receipt_path
            ),
            'schedule_file_sha256': SCHEDULE_FILE_SHA256,
            'rectified_manifest_sha256': MANIFEST_FILE_SHA256,
            'closure_digests': dict(_EXPECTED_CLOSURE_DIGESTS),
            'selected_gpu': {
                'uuid': selected['uuid'],
                'pci_bus_id': selected['pci_bus_id'],
                'index': selected['index'],
                'model': selected['model'],
            },
            'authorized_scope': _authorized_scope(),
            'budget': {
                'authorization_gpu_seconds': GPU_TARGET_SECONDS,
                'authorization_storage_bytes': BYTE_TARGET,
                'catastrophe_gpu_seconds': GPU_CATASTROPHE_SECONDS,
                'catastrophe_storage_bytes': BYTE_CATASTROPHE,
            },
            'runtime_root': str(runtime_root),
            'attempt_root': str(attempt_root),
            'runtime_paths': resolved_paths,
            'finalizer': AUTHORIZED_FINALIZER,
            'execution_lock_created': False,
            'gpu_ledger_created': False,
            'dispatcher_called': False,
            'torch_probe_invocations': 0,
            'model_loads': 0,
            'model_forwards': 0,
            'gpu_inference_seconds': 0,
            'scientific_result': 'NO_SCIENTIFIC_RESULT',
        },
        'authorization_payload_sha256',
    )
    _atomic_pair(
        first_path=receipt_path,
        first_payload=receipt,
        first_validator=_validate_receipt_payload,
        second_path=authorization_path,
        second_payload=authorization,
        second_validator=_validate_authorization_payload,
    )
    validate_attempt03_execution_authorization(authorization_path)
    return dict(authorization)


def validate_attempt03_execution_authorization(
    path: Path,
) -> dict[str, Any]:
    path = _attempt_path(path)
    payload = _load_json(path)
    _validate_authorization_payload(payload)
    runtime_root = Path(str(payload.get('runtime_root')))
    attempt_root = _required_attempt_root(runtime_root)
    if _attempt_root(path) != attempt_root:
        raise V4ExecutionError('V4_ATTEMPT03_PATH_SCOPE_MISMATCH')
    receipt_path = Path(str(payload.get('gpu_receipt_path')))
    if (
        _attempt_root(receipt_path) != attempt_root
        or
        not receipt_path.is_file()
        or _sha256_file(receipt_path) != payload.get('gpu_receipt_sha256')
    ):
        raise V4ExecutionError('V4_ATTEMPT03_RECEIPT_TAMPER')
    receipt = _load_json(receipt_path)
    _validate_receipt_payload(receipt)
    snapshot_path = Path(str(payload.get('hardware_snapshot_path')))
    if (
        _attempt_root(snapshot_path) != attempt_root
        or not snapshot_path.is_file()
        or _sha256_file(snapshot_path)
        != payload.get('hardware_snapshot_sha256')
    ):
        raise V4ExecutionError('V4_ATTEMPT03_PREFLIGHT_TAMPER')
    snapshot = validate_attempt03_gpu_preflight(snapshot_path)
    resource_path = Path(str(payload.get('resource_receipt_path')))
    if (
        _attempt_root(resource_path) != attempt_root
        or receipt.get('resource_receipt_path')
        != str(resource_path.resolve())
        or receipt.get('resource_receipt_sha256')
        != payload.get('resource_receipt_sha256')
        or
        not resource_path.is_file()
        or _sha256_file(resource_path)
        != payload.get('resource_receipt_sha256')
    ):
        raise V4ExecutionError('V4_ATTEMPT03_RESOURCE_RECEIPT_TAMPER')
    resources = validate_attempt03_resources(resource_path)
    selected = payload.get('selected_gpu')
    if not isinstance(selected, Mapping):
        raise V4ExecutionError('V4_ATTEMPT03_GPU_IDENTITY_UNPROVEN')
    if (
        receipt.get('selected_gpu_uuid') != selected.get('uuid')
        or receipt.get('selected_gpu_pci_bus_id')
        != selected.get('pci_bus_id')
        or receipt.get('selected_gpu_index') != selected.get('index')
        or receipt.get('selected_gpu_model') != selected.get('model')
        or snapshot.get('selected_gpu', {}).get('uuid')
        != selected.get('uuid')
        or snapshot.get('selected_gpu', {}).get('pci_bus_id')
        != selected.get('pci_bus_id')
        or snapshot.get('selected_gpu', {}).get('index')
        != selected.get('index')
        or snapshot.get('selected_gpu', {}).get('model')
        != selected.get('model')
    ):
        raise V4ExecutionError('V4_ATTEMPT03_GPU_IDENTITY_TAMPER')
    for evidence in (receipt, snapshot, resources):
        if (
            evidence.get('tooling_commit')
            != payload.get('tooling_commit')
            or evidence.get('tooling_tree')
            != payload.get('tooling_tree')
        ):
            raise V4ExecutionError(
                'V4_ATTEMPT03_AUTHORIZATION_TOOLING_MISMATCH'
            )
    if (
        receipt.get('run_id') != payload.get('run_id')
        or snapshot.get('run_id') != payload.get('run_id')
        or receipt.get('hardware_snapshot_path')
        != str(snapshot_path.resolve())
        or receipt.get('hardware_snapshot_sha256')
        != payload.get('hardware_snapshot_sha256')
    ):
        raise V4ExecutionError('V4_ATTEMPT03_AUTHORIZATION_TAMPER')
    return payload
