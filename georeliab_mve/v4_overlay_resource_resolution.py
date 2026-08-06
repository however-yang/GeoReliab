"""Authoritative, CPU-only resolution of frozen v4 resource archives."""

from __future__ import annotations

from collections.abc import Callable, Mapping
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import platform
import socket
import stat
import subprocess
from typing import Any
from urllib.parse import unquote, urlsplit

from . import toml_compat as tomllib


BASE_COMMIT = 'e6bc12f5f0e16d12f9115b6687f5d379ef4ecebe'
BASE_TREE = 'f133fde3a1e17d69dad927b4a483d42512e4f5f0'
OVERLAY_RELATIVE_PATH = 'configs/a100_real_mve_overlay.toml'
EXPECTED_OVERLAY_SHA256 = (
    'c65ef97684adeda6b1bba8d8c152eb559df44367bc35973f549d7e8049136011'
)

MAP_SCHEMA = 'georeliab-v4-overlay-resource-map-1.0'
CONTRACT_SCHEMA = 'georeliab-v4-overlay-resolution-contract-1.0'
IDENTITIES_SCHEMA = 'georeliab-v4-overlay-resource-identities-1.0'
RECEIPT_SCHEMA = 'georeliab-v4-overlay-resource-resolution-receipt-1.0'
RESOLUTION_RULE_ID = 'FROZEN_OVERLAY_RUNTIME_DATA_EXACT_BASENAME_V1'

RESOURCE_SPECS: dict[str, dict[str, str]] = {
    'dtu_sampleset': {
        'url_key': 'dtu_sampleset_url',
        'bytes_key': 'dtu_sampleset_bytes',
        'sha256_key': 'dtu_sampleset_sha256',
        'filename': 'SampleSet.zip',
        'environment_key': 'GEORELIAB_V4_DTU_SAMPLESET_PATH',
    },
    'dtu_points': {
        'url_key': 'dtu_points_url',
        'bytes_key': 'dtu_points_bytes',
        'sha256_key': 'dtu_points_sha256',
        'filename': 'Points.zip',
        'environment_key': 'GEORELIAB_V4_DTU_POINTS_PATH',
    },
}


class OverlayResolutionError(RuntimeError):
    """Raised whenever authoritative resource identity cannot be proven."""


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


def _signed(value: Mapping[str, Any]) -> dict[str, Any]:
    result = dict(value)
    result['payload_sha256'] = _sha_json(value)
    return result


def _validate_signed(value: Mapping[str, Any]) -> None:
    digest = value.get('payload_sha256')
    unsigned = {key: item for key, item in value.items() if key != 'payload_sha256'}
    if not _is_sha(digest) or _sha_json(unsigned) != digest:
        raise OverlayResolutionError('V4_OVERLAY_ARTIFACT_TAMPER')


def _is_sha(value: object, length: int = 64) -> bool:
    return (
        isinstance(value, str)
        and len(value) == length
        and value == value.lower()
        and all(char in '0123456789abcdef' for char in value)
    )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding='utf-8'))
    except (OSError, json.JSONDecodeError) as exc:
        raise OverlayResolutionError('V4_OVERLAY_JSON_UNREADABLE') from exc
    if not isinstance(value, dict):
        raise OverlayResolutionError('V4_OVERLAY_JSON_OBJECT_REQUIRED')
    return value


def _atomic_artifact_set(
    output_dir: Path,
    payloads: Mapping[str, Mapping[str, Any]],
    *,
    validate_staged: Callable[[Path, Path], None],
) -> None:
    if output_dir.exists():
        raise OverlayResolutionError('V4_OVERLAY_ARTIFACT_COLLISION')
    parent = output_dir.parent
    parent.mkdir(parents=True, exist_ok=True)
    partial = output_dir.with_name(output_dir.name + '.partial')
    if partial.exists():
        raise OverlayResolutionError('V4_OVERLAY_PARTIAL_ARTIFACT_PRESENT')
    partial.mkdir()
    try:
        for name, payload in payloads.items():
            path = partial / name
            path.write_bytes(_json_bytes(payload))
            _validate_signed(_load_json(path))
        validate_staged(
            partial / 'resource-resolution-receipt.json', output_dir
        )
        partial.replace(output_dir)
    except Exception:
        for child in partial.iterdir() if partial.exists() else ():
            child.unlink()
        if partial.exists():
            partial.rmdir()
        raise


def _git_overlay_bytes(worktree: Path) -> bytes:
    try:
        return subprocess.check_output(
            ['git', 'show', f'{BASE_COMMIT}:{OVERLAY_RELATIVE_PATH}'],
            cwd=worktree,
            stderr=subprocess.PIPE,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise OverlayResolutionError(
            'V4_OVERLAY_BASE_DESCRIPTOR_UNAVAILABLE'
        ) from exc


def _tooling_revision(worktree: Path) -> tuple[str, str]:
    try:
        commit = subprocess.check_output(
            ['git', 'rev-parse', 'HEAD'], cwd=worktree, text=True
        ).strip()
        tree = subprocess.check_output(
            ['git', 'rev-parse', 'HEAD^{tree}'], cwd=worktree, text=True
        ).strip()
    except (OSError, subprocess.SubprocessError) as exc:
        raise OverlayResolutionError('V4_OVERLAY_TOOLING_UNAVAILABLE') from exc
    if not _is_sha(commit, 40) or not _is_sha(tree, 40):
        raise OverlayResolutionError('V4_OVERLAY_TOOLING_UNAVAILABLE')
    return commit, tree


def _canonical_overlay_path(worktree: Path, overlay_path: Path) -> Path:
    expected = (worktree / OVERLAY_RELATIVE_PATH).resolve()
    if overlay_path.resolve() != expected:
        raise OverlayResolutionError('V4_OVERLAY_DESCRIPTOR_PATH_MISMATCH')
    return expected


def _parse_overlay(
    *, worktree: Path, overlay_path: Path, anchor_overlay_bytes: bytes | None
) -> tuple[dict[str, Any], dict[str, Any]]:
    canonical = _canonical_overlay_path(worktree, overlay_path)
    if anchor_overlay_bytes is None:
        anchor_overlay_bytes = _git_overlay_bytes(worktree)
        anchor_digest = hashlib.sha256(anchor_overlay_bytes).hexdigest()
        if anchor_digest != EXPECTED_OVERLAY_SHA256:
            raise OverlayResolutionError('V4_OVERLAY_BASE_DESCRIPTOR_TAMPER')
    else:
        anchor_digest = hashlib.sha256(anchor_overlay_bytes).hexdigest()
    try:
        observed = canonical.read_bytes()
    except OSError as exc:
        raise OverlayResolutionError('V4_OVERLAY_DESCRIPTOR_UNREADABLE') from exc
    if hashlib.sha256(observed).hexdigest() != anchor_digest:
        raise OverlayResolutionError('V4_OVERLAY_DESCRIPTOR_SHA_MISMATCH')
    try:
        overlay = tomllib.loads(observed.decode('utf-8'))
    except (UnicodeDecodeError, tomllib.TOMLDecodeError) as exc:
        raise OverlayResolutionError('V4_OVERLAY_DESCRIPTOR_UNREADABLE') from exc
    if not isinstance(overlay, dict):
        raise OverlayResolutionError('V4_OVERLAY_DESCRIPTOR_UNREADABLE')
    binding = {
        'path': str(canonical),
        'sha256': anchor_digest,
        'base_commit': BASE_COMMIT,
        'base_tree': BASE_TREE,
        'relative_path': OVERLAY_RELATIVE_PATH,
        'schema_version': MAP_SCHEMA,
    }
    return overlay, binding


def _exact_mapping(mapping: object, reason: str) -> Mapping[str, Any]:
    if not isinstance(mapping, Mapping):
        raise OverlayResolutionError(reason)
    return mapping


def _validate_case_unique_resources(resources: Mapping[str, Any]) -> None:
    folded: dict[str, str] = {}
    for key in resources:
        if not isinstance(key, str):
            raise OverlayResolutionError('V4_OVERLAY_LOGICAL_ID_INVALID')
        canonical = key.casefold()
        if canonical in folded and folded[canonical] != key:
            raise OverlayResolutionError('V4_OVERLAY_LOGICAL_ID_DUPLICATE')
        folded[canonical] = key
    required_keys = {
        item
        for spec in RESOURCE_SPECS.values()
        for item in (spec['url_key'], spec['bytes_key'], spec['sha256_key'])
    }
    for required in required_keys:
        matches = [key for key in resources if key.casefold() == required]
        if matches != [required]:
            raise OverlayResolutionError('V4_OVERLAY_LOGICAL_ID_MISMATCH')


def _url_relative_path(url: object, expected_name: str) -> str:
    if not isinstance(url, str) or not url:
        raise OverlayResolutionError('V4_OVERLAY_RESOURCE_URL_REQUIRED')
    parsed = urlsplit(url)
    pure = PurePosixPath(unquote(parsed.path))
    if (
        parsed.scheme not in {'http', 'https'}
        or pure.name != expected_name
        or '..' in pure.parts
        or pure.name in {'', '.', '..'}
    ):
        raise OverlayResolutionError('V4_OVERLAY_RESOURCE_RELATIVE_PATH_INVALID')
    return pure.name


def _frozen_archive_identity(
    frozen: Mapping[str, Any], filename: str
) -> tuple[int, str]:
    archives = _exact_mapping(
        frozen.get('archives'), 'V4_OVERLAY_MATERIALIZATION_IDENTITY_REQUIRED'
    )
    item = _exact_mapping(
        archives.get(filename),
        'V4_OVERLAY_MATERIALIZATION_IDENTITY_REQUIRED',
    )
    size, digest = item.get('bytes'), item.get('sha256')
    if type(size) is not int or size <= 0 or not _is_sha(digest):
        raise OverlayResolutionError(
            'V4_OVERLAY_MATERIALIZATION_IDENTITY_REQUIRED'
        )
    return size, str(digest)


def _symlink_chain(path: Path) -> list[dict[str, str]]:
    chain: list[dict[str, str]] = []
    current = Path(path.anchor)
    seen: set[str] = set()
    for part in path.parts[1:]:
        current = current / part
        try:
            if current.is_symlink():
                target = os.readlink(current)
                marker = str(current.absolute())
                if marker in seen:
                    raise OverlayResolutionError('V4_OVERLAY_RESOURCE_SYMLINK_LOOP')
                seen.add(marker)
                chain.append({'path': marker, 'target': target})
        except OSError as exc:
            raise OverlayResolutionError('V4_OVERLAY_RESOURCE_SYMLINK_INVALID') from exc
    return chain


def _under_root(root: Path, candidate: Path) -> tuple[Path, Path]:
    root_real = root.resolve(strict=True)
    candidate_real = candidate.resolve(strict=True)
    try:
        candidate_real.relative_to(root_real)
    except ValueError as exc:
        raise OverlayResolutionError('V4_OVERLAY_RESOURCE_ROOT_ESCAPE') from exc
    return root_real, candidate_real


def _hash_stable_file(
    path: Path, *, before_hash: Callable[[Path], None] | None = None
) -> tuple[int, str, dict[str, int]]:
    try:
        initial = path.stat()
    except OSError as exc:
        raise OverlayResolutionError('V4_OVERLAY_RESOURCE_UNREADABLE') from exc
    pre_injection_digest: str | None = None
    if before_hash is not None:
        try:
            pre_injection_digest = _sha256_file(path)
        except OSError as exc:
            raise OverlayResolutionError('V4_OVERLAY_RESOURCE_UNREADABLE') from exc
        before_hash(path)
    flags = os.O_RDONLY | getattr(os, 'O_CLOEXEC', 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise OverlayResolutionError('V4_OVERLAY_RESOURCE_UNREADABLE') from exc
    digest = hashlib.sha256()
    try:
        opened = os.fstat(descriptor)
        with os.fdopen(descriptor, 'rb', closefd=False) as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b''):
                digest.update(block)
        final = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    identity_fields = ('st_dev', 'st_ino', 'st_size', 'st_mtime_ns')
    comparison_fields = identity_fields + (
        ('st_ctime_ns',) if os.name != 'nt' else ()
    )
    if any(
        getattr(initial, field) != getattr(opened, field)
        or getattr(opened, field) != getattr(final, field)
        for field in comparison_fields
    ):
        raise OverlayResolutionError('V4_OVERLAY_RESOURCE_CHANGED_DURING_HASH')
    observed_digest = digest.hexdigest()
    if pre_injection_digest is not None and pre_injection_digest != observed_digest:
        raise OverlayResolutionError('V4_OVERLAY_RESOURCE_CHANGED_DURING_HASH')
    if not stat.S_ISREG(final.st_mode):
        raise OverlayResolutionError('V4_OVERLAY_RESOURCE_NOT_REGULAR_FILE')
    identity = {
        field: int(getattr(final, field))
        for field in identity_fields + ('st_ctime_ns',)
    }
    return int(final.st_size), observed_digest, identity


def _normalize_override(value: Path, expected: Path) -> None:
    try:
        if value.resolve(strict=True) != expected.resolve(strict=True):
            raise OverlayResolutionError('V4_OVERLAY_RESOURCE_OVERRIDE_MISMATCH')
    except OSError as exc:
        raise OverlayResolutionError('V4_OVERLAY_RESOURCE_OVERRIDE_MISMATCH') from exc


def resolve_overlay_resource_identities(
    *,
    worktree: Path,
    runtime_root: Path,
    overlay_path: Path,
    frozen_materialization_path: Path,
    resource_overrides: Mapping[str, Path] | None = None,
    environment: Mapping[str, str] | None = None,
    host_system: str | None = None,
    anchor_overlay_bytes: bytes | None = None,
    before_hash: Callable[[Path], None] | None = None,
    _posix_path_mapper: Callable[[str], Path] | None = None,
) -> dict[str, Any]:
    """Resolve only the two frozen DTU archives through runtime.data."""
    system = platform.system() if host_system is None else host_system
    if system != 'Linux':
        raise OverlayResolutionError('V4_OVERLAY_HOST_PATH_FLAVOR_MISMATCH')
    overlay, overlay_binding = _parse_overlay(
        worktree=worktree,
        overlay_path=overlay_path,
        anchor_overlay_bytes=anchor_overlay_bytes,
    )
    runtime = _exact_mapping(
        overlay.get('runtime'), 'V4_OVERLAY_RUNTIME_MAPPING_REQUIRED'
    )
    resources = _exact_mapping(
        overlay.get('resources'), 'V4_OVERLAY_RESOURCES_REQUIRED'
    )
    _validate_case_unique_resources(resources)
    declared_root_value = runtime.get('data')
    cache_root_value = runtime.get('cache')
    declared_runtime_value = runtime.get('root')
    if not all(
        isinstance(value, str) and PurePosixPath(value).is_absolute()
        for value in (declared_root_value, cache_root_value, declared_runtime_value)
    ):
        raise OverlayResolutionError('V4_OVERLAY_POSIX_ROOT_REQUIRED')
    map_path = Path if _posix_path_mapper is None else _posix_path_mapper
    mapped_runtime = map_path(str(declared_runtime_value))
    if mapped_runtime.resolve() != runtime_root.resolve():
        raise OverlayResolutionError('V4_OVERLAY_RUNTIME_ROOT_MISMATCH')
    declared_root = map_path(str(declared_root_value))
    cache_root = map_path(str(cache_root_value))
    frozen = _load_json(frozen_materialization_path)
    overrides = dict(resource_overrides or {})
    unknown_overrides = set(overrides) - set(RESOURCE_SPECS)
    if unknown_overrides:
        raise OverlayResolutionError('V4_OVERLAY_RESOURCE_OVERRIDE_UNKNOWN')
    env = os.environ if environment is None else environment
    identities: dict[str, dict[str, Any]] = {}
    for logical_id, spec in RESOURCE_SPECS.items():
        relative = _url_relative_path(resources.get(spec['url_key']), spec['filename'])
        expected_size = resources.get(spec['bytes_key'])
        expected_sha = resources.get(spec['sha256_key'])
        if type(expected_size) is not int or expected_size <= 0 or not _is_sha(expected_sha):
            raise OverlayResolutionError('V4_OVERLAY_RESOURCE_IDENTITY_REQUIRED')
        frozen_size, frozen_sha = _frozen_archive_identity(frozen, spec['filename'])
        if expected_size != frozen_size or expected_sha != frozen_sha:
            raise OverlayResolutionError(
                'V4_OVERLAY_MATERIALIZATION_IDENTITY_MISMATCH'
            )
        normalized = declared_root / relative
        if PurePosixPath(relative).parts != (relative,):
            raise OverlayResolutionError('V4_OVERLAY_RESOURCE_RELATIVE_PATH_INVALID')
        override_values: list[Path] = []
        if logical_id in overrides:
            override_values.append(Path(overrides[logical_id]))
        env_value = env.get(spec['environment_key'])
        if env_value:
            override_values.append(Path(env_value))
        for value in override_values:
            _normalize_override(value, normalized)
        chain = _symlink_chain(normalized)
        try:
            root_real, realpath = _under_root(declared_root, normalized)
        except FileNotFoundError as exc:
            raise OverlayResolutionError('V4_OVERLAY_RESOURCE_UNREADABLE') from exc
        observed_size, observed_sha, stat_identity = _hash_stable_file(
            realpath, before_hash=before_hash
        )
        if observed_size != expected_size or observed_sha != expected_sha:
            raise OverlayResolutionError('V4_OVERLAY_RESOURCE_HASH_MISMATCH')
        shadow = cache_root / relative
        try:
            shadow_exists = shadow.exists() or shadow.is_symlink()
            shadow_regular = shadow.is_file() if shadow_exists else False
            shadow_symlink = shadow.is_symlink() if shadow_exists else False
        except OSError:
            shadow_exists = shadow_regular = shadow_symlink = False
        identities[logical_id] = {
            'logical_resource_id': logical_id,
            'overlay_descriptor_path': overlay_binding['path'],
            'overlay_descriptor_sha256': overlay_binding['sha256'],
            'overlay_schema_version': MAP_SCHEMA,
            'hostname': socket.gethostname(),
            'path_flavor': 'POSIX',
            'declared_root': str(declared_root),
            'declared_relative_path': relative,
            'normalized_path': str(normalized),
            'realpath': str(realpath),
            'resolved_declared_root': str(root_real),
            'symlink_chain': chain,
            'root_containment': 'PASS',
            'regular_file': True,
            'expected_bytes': expected_size,
            'file_size': observed_size,
            'expected_sha256': expected_sha,
            'observed_sha256': observed_sha,
            'stat_identity': stat_identity,
            'resolution_rule_id': RESOLUTION_RULE_ID,
            'override_asserted': bool(override_values),
            'shadow_candidate': {
                'path': str(shadow),
                'exists': shadow_exists,
                'regular_file': shadow_regular,
                'symlink': shadow_symlink,
                'status': 'IGNORED_SHADOW',
            },
        }
    return {
        'schema_version': IDENTITIES_SCHEMA,
        'overlay': overlay_binding,
        'frozen_materialization': {
            'path': str(frozen_materialization_path.resolve()),
            'sha256': _sha256_file(frozen_materialization_path),
        },
        'resource_count': len(identities),
        'resources': identities,
        'fallback_used': False,
        'recursive_discovery_used': False,
        'shadow_candidate_used': False,
    }


def _required_output_dir(
    runtime_root: Path, output_dir: Path, tooling_commit: str
) -> Path:
    required = (
        runtime_root.resolve()
        / 'artifacts'
        / 'v4-overlay-resource-resolution'
        / tooling_commit
    )
    if output_dir.resolve() != required:
        raise OverlayResolutionError('V4_OVERLAY_OUTPUT_PATH_MISMATCH')
    return required


def resolve_overlay_resources(
    *,
    worktree: Path,
    runtime_root: Path,
    overlay_path: Path,
    frozen_materialization_path: Path,
    output_dir: Path,
    resource_overrides: Mapping[str, Path] | None = None,
    environment: Mapping[str, str] | None = None,
    host_system: str | None = None,
    anchor_overlay_bytes: bytes | None = None,
    before_hash: Callable[[Path], None] | None = None,
    tooling_commit: str | None = None,
    tooling_tree: str | None = None,
    _posix_path_mapper: Callable[[str], Path] | None = None,
) -> dict[str, Any]:
    if tooling_commit is None or tooling_tree is None:
        tooling_commit, tooling_tree = _tooling_revision(worktree)
    if not _is_sha(tooling_commit, 40) or not _is_sha(tooling_tree, 40):
        raise OverlayResolutionError('V4_OVERLAY_TOOLING_UNAVAILABLE')
    output_dir = _required_output_dir(runtime_root, output_dir, tooling_commit)
    identities = resolve_overlay_resource_identities(
        worktree=worktree,
        runtime_root=runtime_root,
        overlay_path=overlay_path,
        frozen_materialization_path=frozen_materialization_path,
        resource_overrides=resource_overrides,
        environment=environment,
        host_system=host_system,
        anchor_overlay_bytes=anchor_overlay_bytes,
        before_hash=before_hash,
        _posix_path_mapper=_posix_path_mapper,
    )
    contract = _signed(
        {
            'schema_version': CONTRACT_SCHEMA,
            'status': 'PASS',
            'authority': 'FROZEN_OVERLAY_ONLY',
            'resolution_rule_id': RESOLUTION_RULE_ID,
            'path_flavor': 'POSIX',
            'fallback_allowed': False,
            'recursive_discovery_allowed': False,
            'shadow_candidate_authoritative': False,
            'override_policy': 'EXACT_NORMALIZED_EQUALITY_ASSERTION_ONLY',
            'scientific_result': 'NO_SCIENTIFIC_RESULT',
        }
    )
    descriptor = _signed(
        {
            'schema_version': MAP_SCHEMA,
            'status': 'PASS',
            'tooling_commit': tooling_commit,
            'tooling_tree': tooling_tree,
            'overlay': identities['overlay'],
            'logical_resource_ids': sorted(RESOURCE_SPECS),
            'resources': {
                key: {
                    field: identities['resources'][key][field]
                    for field in (
                        'declared_root',
                        'declared_relative_path',
                        'expected_sha256',
                        'expected_bytes',
                    )
                }
                for key in sorted(RESOURCE_SPECS)
            },
        }
    )
    signed_identities = _signed(identities)
    names = {
        'contract': 'overlay-resolution-contract.json',
        'descriptor': 'overlay-descriptor-binding.json',
        'identities': 'resolved-resource-identities.json',
    }
    preliminary = {
        names['contract']: contract,
        names['descriptor']: descriptor,
        names['identities']: signed_identities,
    }
    links = {
        label: {
            'path': name,
            'sha256': hashlib.sha256(_json_bytes(preliminary[name])).hexdigest(),
        }
        for label, name in names.items()
    }
    receipt = _signed(
        {
            'schema_version': RECEIPT_SCHEMA,
            'status': 'PASS',
            'reason_code': 'V4_RESOURCE_PATH_RESOLUTION_VERIFIED',
            'tooling_commit': tooling_commit,
            'tooling_tree': tooling_tree,
            'output_dir': str(output_dir),
            'artifacts': links,
            'resolved_resource_count': 2,
            'fallback_used': False,
            'gpu_probe_invocations': 0,
            'torch_probe_invocations': 0,
            'model_loads': 0,
            'model_forwards': 0,
            'gpu_inference_seconds': 0,
            'scientific_result': 'NO_SCIENTIFIC_RESULT',
        }
    )
    payloads = dict(preliminary)
    payloads['resource-resolution-receipt.json'] = receipt
    _atomic_artifact_set(
        output_dir,
        payloads,
        validate_staged=lambda receipt_path, intended_output_dir: (
            _validate_overlay_resource_resolution_at(
                receipt_path,
                intended_output_dir=intended_output_dir,
                host_system=host_system,
            )
        ),
    )
    return validate_overlay_resource_resolution(
        output_dir / 'resource-resolution-receipt.json',
        host_system=host_system,
    )


def validate_overlay_resource_resolution(
    receipt_path: Path, *, host_system: str | None = None
) -> dict[str, Any]:
    return _validate_overlay_resource_resolution_at(
        receipt_path,
        intended_output_dir=receipt_path.parent,
        host_system=host_system,
    )


def _validate_overlay_resource_resolution_at(
    receipt_path: Path,
    *,
    intended_output_dir: Path,
    host_system: str | None = None,
) -> dict[str, Any]:
    if (platform.system() if host_system is None else host_system) != 'Linux':
        raise OverlayResolutionError('V4_OVERLAY_HOST_PATH_FLAVOR_MISMATCH')
    receipt = _load_json(receipt_path)
    _validate_signed(receipt)
    if (
        receipt.get('schema_version') != RECEIPT_SCHEMA
        or receipt.get('status') != 'PASS'
        or receipt.get('reason_code') != 'V4_RESOURCE_PATH_RESOLUTION_VERIFIED'
        or receipt.get('resolved_resource_count') != 2
        or receipt.get('fallback_used') is not False
        or any(
            receipt.get(field) != 0
            for field in (
                'gpu_probe_invocations',
                'torch_probe_invocations',
                'model_loads',
                'model_forwards',
                'gpu_inference_seconds',
            )
        )
        or receipt.get('scientific_result') != 'NO_SCIENTIFIC_RESULT'
    ):
        raise OverlayResolutionError('V4_OVERLAY_RECEIPT_INVALID')
    artifact_dir = receipt_path.parent.resolve()
    if receipt.get('output_dir') != str(intended_output_dir.resolve()):
        raise OverlayResolutionError('V4_OVERLAY_RECEIPT_PATH_MISMATCH')
    artifacts = _exact_mapping(
        receipt.get('artifacts'), 'V4_OVERLAY_RECEIPT_INVALID'
    )
    expected_schemas = {
        'contract': CONTRACT_SCHEMA,
        'descriptor': MAP_SCHEMA,
        'identities': IDENTITIES_SCHEMA,
    }
    loaded: dict[str, dict[str, Any]] = {}
    for label, schema in expected_schemas.items():
        link = _exact_mapping(artifacts.get(label), 'V4_OVERLAY_RECEIPT_INVALID')
        relative = link.get('path')
        if not isinstance(relative, str) or Path(relative).parts != (relative,):
            raise OverlayResolutionError('V4_OVERLAY_RECEIPT_PATH_MISMATCH')
        path = artifact_dir / relative
        if not path.is_file() or _sha256_file(path) != link.get('sha256'):
            raise OverlayResolutionError('V4_OVERLAY_ARTIFACT_TAMPER')
        payload = _load_json(path)
        _validate_signed(payload)
        if payload.get('schema_version') != schema:
            raise OverlayResolutionError('V4_OVERLAY_ARTIFACT_TAMPER')
        loaded[label] = payload
    identities = loaded['identities']
    resources = _exact_mapping(
        identities.get('resources'), 'V4_OVERLAY_ARTIFACT_TAMPER'
    )
    if set(resources) != set(RESOURCE_SPECS):
        raise OverlayResolutionError('V4_OVERLAY_ARTIFACT_TAMPER')
    for logical_id, identity_value in resources.items():
        identity = _exact_mapping(
            identity_value, 'V4_OVERLAY_ARTIFACT_TAMPER'
        )
        path = Path(str(identity.get('realpath')))
        if (
            not path.is_file()
            or path.stat().st_size != identity.get('file_size')
            or _sha256_file(path) != identity.get('observed_sha256')
            or identity.get('observed_sha256') != identity.get('expected_sha256')
            or identity.get('file_size') != identity.get('expected_bytes')
            or identity.get('logical_resource_id') != logical_id
            or identity.get('resolution_rule_id') != RESOLUTION_RULE_ID
        ):
            raise OverlayResolutionError('V4_OVERLAY_RESOURCE_TAMPER')
    return receipt


def parse_resource_overrides(values: list[str]) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for value in values:
        if '=' not in value:
            raise OverlayResolutionError('V4_OVERLAY_RESOURCE_OVERRIDE_INVALID')
        logical_id, path = value.split('=', 1)
        if logical_id in result or logical_id not in RESOURCE_SPECS or not path:
            raise OverlayResolutionError('V4_OVERLAY_RESOURCE_OVERRIDE_INVALID')
        result[logical_id] = Path(path)
    return result
